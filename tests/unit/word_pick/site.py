"""第二步甲（归类换成在固定初始词表里每轮选一个词）测试共用的现场。

测试按设计（共享说明第 3 节）写成对公开入口的端到端断言，不依赖新实现的内部类名：

- 组合根 ``build_runtime`` 装出整套行为侧（会话 / 物理两支）与语义树、换代；
- 会话的轮由 ``split_turns`` 造、交给组合根装好的会话 lane（``branches.session.fusion.consume``）；物理的交付走
  ``runtime.deliver_physical``；归约走各支的 ``reduction_runner.run_once``；落了什么看行为树与 ``read_series``；
- 模型一律用确定性替身 ``Router``：它按请求认出这是哪一种调用——"描述这一轮"（结构化输出名 ``behavior_session_turn``，
  第二步甲不动它）、选词（提示词里有【词表】一段）、旧的白天归类（``behavior_kind_classify``），其余（语义概览等）给垃圾。

选词的答复按设计的输出形状写（会话 lane ``{"pick": …, "continues": …, "new": null}``，二乙起带 ``new``）。词与记录的编号不写死成某种
写法：替身在这一次的提示词里找到那个词所在的行、那条记录所在的行，取行首的编号原样答回（全是数字就答整数）。
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from habitus.behavior.fusion.lanes.session import SessionMessage, SessionMessageRole, SessionTurn, split_turns
from habitus.config import HabitusConfig
from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.vector import VectorStoreFactory
from habitus.model_client import ModelResponse, ProviderCapabilities, ProviderFactory
from habitus.runtime import build_runtime
from tests.integration.test_runtime_assembly import FakeEmbeddingProvider, FakeRerankProvider, FakeVectorBackend
from tests.lock_store import ProcessLocalLockStore
from tests.model_helpers import prepare_chat_request

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_CONFIG = REPOSITORY_ROOT / "habitus" / "config" / "example.yaml"
CST = timezone(timedelta(hours=8))
SUBJECT = "家庭成员A"
DESCRIBE_NAME = "behavior_session_turn"
OLD_CLASSIFY_NAME = "behavior_kind_classify"
VOCABULARY_HEADER = "【词表】"
RECORDS_HEADER = "【最近的记录】"

#: 初始词表里用到的几个词（编号按源文件顺序从 1 起：``初始词表-会话.md`` / ``初始词表-物理.md``）。
SESSION_WORDS = {
    "s-k0001": "写文档",
    "s-k0002": "写周报",
    "s-k0003": "写总结",
    "s-k0004": "写方案",
    "s-k0005": "写报告",
    "s-k0006": "写论文",
}
PHYSICAL_WORDS = {"p-k0001": "睡觉", "p-k0003": "起床", "p-k0005": "穿衣"}
WORD_NAMES = {**SESSION_WORDS, **PHYSICAL_WORDS}


# ── 选词答复 ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Word:
    """答一个词：按名字在这一次的词表里找到它的行，答行首的编号。"""

    token: str

    @property
    def name(self) -> str:
        return WORD_NAMES[self.token]


@dataclass(frozen=True)
class Pick:
    """``pick`` 原样答这个值（表里没有的编号、领域标题、固定选项）。"""

    value: object
    continues: object = None


@dataclass(frozen=True)
class Continue:
    """答 ``continue``：``said`` 给了就在【最近的记录】里找含它的那一行、答行首编号；否则答 ``number``。"""

    said: str | None = None
    number: int | None = None


@dataclass(frozen=True)
class Raw:
    """原样答这段文字（不合格的 JSON）。"""

    text: str


NONE = Pick("none")
MISSING = Pick("missing")


@dataclass
class Call:
    kind: str  # describe / pick / classify / other
    name: str
    text: str
    #: 原样的各条消息（角色, 内容）与要求的输出形状。
    messages: tuple[tuple[str, str], ...] = ()
    schema: dict[str, Any] | None = None


class Router:
    """按请求的种类回放答复，记下每一次调用。``picks`` 按先后用掉，用完了用 ``pick_default``。"""

    capabilities = ProviderCapabilities()
    is_remote = False
    prepare = staticmethod(prepare_chat_request)

    def __init__(self, provider_name: str = "fake", model: str = "fake-1") -> None:
        self.provider_name = provider_name
        self.model = model
        self.calls: list[Call] = []
        self.picks: list[object] = []
        self.pick_default: object = NONE
        self.describes: list[object] = []
        self.describe_count = 0

    # ── 计数 ──
    def count(self, kind: str) -> int:
        return sum(1 for call in self.calls if call.kind == kind)

    def texts(self, kind: str) -> list[str]:
        return [call.text for call in self.calls if call.kind == kind]

    # ── 模型后端 ──
    async def complete_async(self, prepared: Any) -> ModelResponse:
        request = prepared.request
        name = request.response_format.name if request.response_format is not None else ""
        text = "\n".join(message.content or "" for message in request.messages)
        kind = (
            "describe"
            if name == DESCRIBE_NAME
            else "classify"
            if name == OLD_CLASSIFY_NAME
            else "pick"
            if VOCABULARY_HEADER in text
            else "other"
        )
        messages = tuple((str(message.role), message.content or "") for message in request.messages)
        schema = dict(request.response_format.schema) if request.response_format is not None else None
        self.calls.append(Call(kind, name, text, messages, schema))
        if kind == "describe":
            answer = self._describe()
        elif kind == "pick":
            answer = self._pick(text)
        elif kind == "classify":
            answer = _old_not_event(text)
        else:
            answer = "{}"
        if isinstance(answer, BaseException):
            raise answer
        return ModelResponse(content=answer, model=self.model, provider=self.provider_name, finish_reason="stop")

    def _describe(self) -> str | BaseException:
        self.describe_count += 1
        scripted = self.describes.pop(0) if self.describes else None
        if isinstance(scripted, BaseException):
            return scripted
        if isinstance(scripted, Raw):
            return scripted.text
        body = scripted if isinstance(scripted, dict) else description(f"办事第{self.describe_count}件")
        return json.dumps(body, ensure_ascii=False)

    def _pick(self, text: str) -> str | BaseException:
        scripted = self.picks.pop(0) if self.picks else self.pick_default
        if isinstance(scripted, BaseException):
            return scripted
        if isinstance(scripted, Raw):
            return scripted.text
        if isinstance(scripted, Word):
            body: dict[str, object] = {"pick": word_number(text, scripted.name), "continues": None}
        elif isinstance(scripted, Continue):
            number = scripted.number if scripted.said is None else record_number(text, scripted.said)
            body = {"pick": "continue", "continues": number}
        elif isinstance(scripted, Pick):
            body = {"pick": scripted.value, "continues": scripted.continues}
        else:
            raise AssertionError(f"unknown scripted pick {scripted!r}")
        if RECORDS_HEADER not in text:
            # 阶段 C 改：物理 lane 的选词提示词没有【最近的记录】，输出没有 continues（共享说明 3.2 去掉 continue）；
            # 丙二起物理 lane 也写新节点，输出多了 new（这里的替身从不写新节点）
            body = {"pick": body["pick"], "new": None}
        else:
            body["new"] = None  # 二乙：会话 lane 的输出形状多了 new（这里的替身从不写新节点）
        return json.dumps(body, ensure_ascii=False)

    def complete(self, request: Any) -> ModelResponse:  # pragma: no cover
        raise NotImplementedError

    def stream(self, request: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def stream_async(self, request: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def health_check(self) -> dict[str, object]:  # pragma: no cover
        return {}

    async def aclose(self) -> None:  # pragma: no cover
        return None


def description(name: str) -> dict[str, Any]:
    """「描述这一轮」的一份合格答复（现行输出形状）。"""

    return {
        "name": name,
        "summary": f"他让助手{name}；助手做完了。",
        "goal": f"把{name}这件事办好",
        "steps": [f"按要求{name}"],
    }


def _old_not_event(text: str) -> str:
    """旧的白天归类：每条记录都答「不是一件事」（只为"应当不变"的那几条在旧代码上走得通）。"""

    records = sorted(set(re.findall(r"\bR\d+\b", text)), key=lambda item: int(item[1:])) or ["R1"]
    items = [{"record": record, "reason": "不是一件事", "choice": "不是一件事", "proposed": None} for record in records]
    return json.dumps({"items": items}, ensure_ascii=False)


# ── 读提示词 ──────────────────────────────────────────────────────────────────


def section(text: str, header: str) -> list[str]:
    """提示词里 ``header`` 那一段的各行（到下一个空行为止）。"""

    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith(header):
            body: list[str] = []
            for item in lines[index + 1 :]:
                if not item.strip():
                    break
                body.append(item)
            return body
    raise AssertionError(f"the prompt has no {header} section")


_WORD_LINE = re.compile(
    r"^(?P<indent>\s*)(?P<id>\S+) (?P<name>[^（：\s]+)(?:（(?P<pid>\S+) (?P<pname>\S+) 的下级）)?：(?P<definition>.*)$"
)


@dataclass(frozen=True)
class VocabularyLine:
    indent: int
    id: str
    name: str
    parent_id: str | None
    parent_name: str | None
    definition: str


@dataclass(frozen=True)
class VocabularyBlock:
    headings: tuple[str, ...]
    words: tuple[VocabularyLine, ...]
    unparsed: tuple[str, ...] = field(default=())

    def levels(self) -> dict[str, int]:
        """按缩进的深浅排出层：不缩进的是第 1 层。"""

        widths = sorted({word.indent for word in self.words})
        return {word.id: widths.index(word.indent) + 1 for word in self.words}


def vocabulary_block(text: str) -> VocabularyBlock:
    headings: list[str] = []
    words: list[VocabularyLine] = []
    unparsed: list[str] = []
    for line in section(text, VOCABULARY_HEADER):
        if line.startswith("# "):
            headings.append(line[2:].strip())
            continue
        match = _WORD_LINE.match(line)
        if match is None:
            unparsed.append(line)
            continue
        words.append(
            VocabularyLine(
                len(match["indent"]),
                match["id"],
                match["name"],
                match["pid"],
                match["pname"],
                match["definition"],
            )
        )
    return VocabularyBlock(tuple(headings), tuple(words), tuple(unparsed))


def _number(token: str) -> object:
    return int(token) if token.isdigit() else token


def word_number(text: str, name: str) -> object:
    """这一次的词表里名字是 ``name`` 的那个词的编号（行首的那一段）。"""

    for word in vocabulary_block(text).words:
        if word.name == name:
            return _number(word.id)
    raise AssertionError(f"the vocabulary in the prompt has no word {name!r}")


def record_lines(text: str) -> list[str]:
    return section(text, RECORDS_HEADER)


def record_number(text: str, said: str) -> int:
    """【最近的记录】里含 ``said`` 的那一行的编号。"""

    for line in record_lines(text):
        if said in line:
            match = re.match(r"^\s*(\d+)", line)
            if match is None:
                raise AssertionError(f"record line has no number: {line!r}")
            return int(match[1])
    raise AssertionError(f"no recent record mentions {said!r}")


# ── 组合根 ────────────────────────────────────────────────────────────────────


def example_mapping(tmp_path: Path) -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    raw["storage"]["root"] = str(tmp_path / "data")
    raw["observability"]["tracing"]["enabled"] = False
    raw["models"]["chat"]["route"].update(provider="fake", adapter="fake_chat", credential_ref="")
    raw["models"]["embedding"]["route"].update(provider="fake", adapter="fake_embedding", credential_ref="")
    raw["models"]["rerank"]["route"].update(provider="fake", adapter="fake_rerank", credential_ref="")
    raw["memory"]["vector_store"]["route"].update(provider="fake", adapter="fake_vector", credential_ref="")
    raw["conversation"]["summary_vector_store"]["route"].update(
        provider="fake", adapter="fake_vector", credential_ref=""
    )
    return raw


def config(
    tmp_path: Path,
    *,
    scene: bool = True,
    behavior: dict[str, Any] | None = None,
    edit: Callable[[dict[str, Any]], None] | None = None,
) -> HabitusConfig:
    """行为侧开着（主体 ``SUBJECT``）、默认也开着语义树（有重试堆与换代）的一份配置；其余全是 example.yaml。"""

    raw = example_mapping(tmp_path)
    raw["behavior"] = {"primary_subject": SUBJECT, **(behavior or {})}
    if scene:
        raw["scene"] = {"enabled": True}
    if edit is not None:
        edit(raw)
    return HabitusConfig.from_mapping(raw)


def dependencies(router: Router) -> tuple[ProviderFactory, VectorStoreFactory]:
    providers = ProviderFactory()

    def chat(context: Any) -> Router:
        router.provider_name, router.model = context.route.provider, context.route.model
        return router

    providers.register_adapter("chat", "fake_chat", chat)
    providers.register_adapter(
        "embedding",
        "fake_embedding",
        lambda context: FakeEmbeddingProvider(context.route.provider, context.route.model, context.config.dimension),
    )
    providers.register_adapter(
        "rerank", "fake_rerank", lambda context: FakeRerankProvider(context.route.provider, context.route.model)
    )
    vectors = VectorStoreFactory()
    vectors.register_adapter(
        "fake_vector", lambda context: FakeVectorBackend(context.config.provider, context.config.collection)
    )
    return providers, vectors


def runtime_of(habitus_config: HabitusConfig, router: Router | None = None) -> tuple[Any, Router]:
    router = router if router is not None else Router()
    providers, vectors = dependencies(router)
    runtime = build_runtime(
        habitus_config, providers=providers, vector_stores=vectors, path_lock=PathLock(ProcessLocalLockStore())
    )
    runtime.initialize()
    return runtime, router


class Site:
    """一份装好的 Runtime（行为侧 + 语义树）、它的模型替身，与造会话轮的助手。"""

    def __init__(self, tmp_path: Path, *, scene: bool = True, **options: Any) -> None:
        self.runtime, self.router = runtime_of(config(tmp_path, scene=scene, **options))
        branches = self.runtime.components.behavior
        assert branches is not None
        self.branches = branches
        self.session = branches.session
        self.physical = branches.physical
        self._sequences: dict[str, int] = {}

    # ── 造轮 ──
    def turn(
        self,
        conversation: str,
        moment: datetime,
        said: str,
        *,
        reply: str = "好的，已经办好了。",
        extra: Sequence[tuple[float, SessionMessageRole, str, bool]] = (),
    ) -> SessionTurn:
        """一轮：他说 ``said``、助手答 ``reply``。``extra`` 是在两者之间插进去的（分钟偏移、角色、内容、是否插在一轮中间）。"""

        (turn,) = self.turns(conversation, [(0.0, SessionMessageRole.PROMPT, said, False), *extra], moment, reply)
        return turn

    def turns(
        self,
        conversation: str,
        messages: Sequence[tuple[float, SessionMessageRole, str, bool]],
        moment: datetime,
        reply: str | None = None,
    ) -> tuple[SessionTurn, ...]:
        start = self._sequences.get(conversation, 0)
        built: list[SessionMessage] = []
        sequence = start
        for minutes, role, content, mid_turn in messages:
            built.append(
                SessionMessage(sequence, moment + timedelta(minutes=minutes), role, content, mid_turn=mid_turn)
            )
            sequence += 1
        if reply is not None:
            last = max((item[0] for item in messages), default=0.0)
            built.append(
                SessionMessage(sequence, moment + timedelta(minutes=last + 1), SessionMessageRole.COMPLETION, reply)
            )
            sequence += 1
        self._sequences[conversation] = sequence
        return split_turns(conversation, built)

    # ── 跑 ──
    def consume(self, *turns: SessionTurn) -> tuple[Any, ...]:
        return asyncio.run(self.session.fusion.consume(turns))

    def reduce(self, branch: Any | None = None) -> Any:
        return asyncio.run((branch or self.session).reduction_runner.run_once())

    def record(self, conversation: str, moment: datetime, said: str, pick: object) -> SessionTurn:
        """一轮从会话 lane 走到行为树：选词答 ``pick``，然后归约一次。"""

        self.router.picks.append(pick)
        turn = self.turn(conversation, moment, said)
        self.consume(turn)
        self.reduce()
        return turn

    def generate(self, day: date) -> Any:
        generation = self.runtime.components.pipelines[0].generation
        assert generation is not None, "the scene must be enabled for a generation"
        return asyncio.run(generation.generation.run(day))

    def close(self) -> None:
        asyncio.run(self.runtime.close())


def at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=CST)


__all__ = [
    "CST",
    "MISSING",
    "NONE",
    "PHYSICAL_WORDS",
    "SESSION_WORDS",
    "SUBJECT",
    "Continue",
    "Pick",
    "Raw",
    "Router",
    "Site",
    "Word",
    "at",
    "config",
    "description",
    "record_lines",
    "runtime_of",
    "vocabulary_block",
]
