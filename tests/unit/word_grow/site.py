"""第二步乙（会话 lane 选词时可以顺手长一个更细的节点；节点的生命周期）测试共用的现场。

沿用第二步甲的现场（``tests.unit.word_pick.site``：组合根 ``build_runtime`` 装出行为侧两支 + 语义树与换代、模型替身
``Router``），在它上面加：

- 选词答复按共享说明 3.1 的会话输出形状 ``{"pick", "continues", "new"}``：这一次请求的输出形状里有 ``new`` 才带
  ``"new": null``（替身照请求的形状答，所以第二步甲的代码上也答得通）；脚本给了 ``new`` 的一律原样带上。物理 lane
  仍只答 ``{"pick"}``。
- 词按名字在这一次提示词的词表里找行首编号（初始词与长出来的节点同一种写法）。
- ``Hook``：答之前先在同一线程里跑一段（例如在另一个线程里整代换代），模拟"作答期间换代"。
- ``Barrier``：两次选词互相等一下再答（有界），模拟"两轮同时写"。

断言只落在公开入口与盘上：行为树记录的 ``kind_token``、换代那份常驻序列读者读出来的序列、下一次选词提示词里的词表、
词表状态文件 ``kinds.state.json``（按值宽口径找条目，不认字段名）、语义树概念存储。
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from habitus.behavior.model import BehaviorKind
from habitus.scene.concepts.store import ConceptStore
from tests.unit.word_pick.site import (
    NONE,
    RECORDS_HEADER,
    Continue,
    Pick,
    Raw,
    Router,
    Site,
    Word,
    at,
    config,
    record_number,
    runtime_of,
    vocabulary_block,
    word_number,
)

HERE = Path(__file__).resolve().parent
STATE_FILENAME = "kinds.state.json"

#: 初始会话词表 279 个词（``s-k0001``…``s-k0279``）；这条 lane 第 n 个长出来的节点编号 = 用过的最大号 + n。
INITIAL_SESSION_WORDS = 279

WRITE_DOC = "s-k0001"  # 写文档（第 1 层）
WEEKLY = "s-k0002"  # 写周报（第 2 层，写文档的下级）
SUMMARY = "s-k0003"  # 写总结（第 2 层）
LAST_DOC_CHILD = "s-k0012"  # 写合同：写文档的最后一个初始下级
QUERY = "s-k0098"  # 查数据（第 1 层，另一个领域）
NAMES = {WRITE_DOC: "写文档", WEEKLY: "写周报", SUMMARY: "写总结", LAST_DOC_CHILD: "写合同", QUERY: "查数据"}


def grown(n: int) -> str:
    """这条 lane 第 ``n`` 个长出来的节点的编号（从 1 数；删掉的也占号，编号不重用）。"""

    return f"s-k{INITIAL_SESSION_WORDS + n:04d}"


# ── 选词答复 ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Named:
    """按名字答一个词或节点：在这一次的词表里找到名字是 ``name`` 的那一行，答行首编号。"""

    name: str


@dataclass(frozen=True)
class Grow:
    """选 ``name``，并在它下面写新节点：``first`` /``second`` 是（名字, 定义）。"""

    name: str
    first: tuple[str, str] | None
    second: tuple[str, str] | None = None


@dataclass(frozen=True)
class Literal:
    """``pick`` 与 ``new`` 原样答（不合格的组合：只填 second、none 带 new）。``pick`` 是 ``Named`` 时按名字找编号。"""

    pick: object
    new: object
    continues: object = None


@dataclass(frozen=True)
class Hook:
    """答之前先跑 ``before()``，再按 ``then`` 答。"""

    before: Callable[[], None]
    then: object


def node(name: str, definition: str | None = None) -> tuple[str, str]:
    return (name, definition if definition is not None else f"{name}这一种")


def _new(first: tuple[str, str] | None, second: tuple[str, str] | None) -> object:
    if first is None and second is None:
        return None
    return {
        "first": None if first is None else {"name": first[0], "definition": first[1]},
        "second": None if second is None else {"name": second[0], "definition": second[1]},
    }


class Barrier:
    """两次选词互相等一下：先到的最多等 ``timeout`` 秒，等不到也照答（选词可能被串行处理）。"""

    def __init__(self, parties: int = 2, timeout: float = 1.0) -> None:
        self.parties, self.timeout = parties, timeout
        self.arrived = 0
        self.met = False

    async def wait(self) -> None:
        self.arrived += 1
        if self.arrived >= self.parties:
            self.met = True
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout
        while not self.met and loop.time() < deadline:
            await asyncio.sleep(0.01)


class GrowRouter(Router):
    """第二步甲的替身，会话选词按请求的输出形状带上 ``new``。"""

    def __init__(self) -> None:
        super().__init__()
        self.barrier: Barrier | None = None

    async def complete_async(self, prepared: Any) -> Any:
        request = prepared.request
        text = "\n".join(message.content or "" for message in request.messages)
        if self.barrier is not None and "【词表】" in text and RECORDS_HEADER in text:
            await self.barrier.wait()
        return await super().complete_async(prepared)

    def _asks_new(self) -> bool:
        schema = self.calls[-1].schema or {}
        properties = schema.get("properties")
        return isinstance(properties, dict) and "new" in properties

    def _pick(self, text: str) -> str | BaseException:
        scripted = self.picks.pop(0) if self.picks else self.pick_default
        while isinstance(scripted, Hook):
            scripted.before()
            scripted = scripted.then
        if isinstance(scripted, BaseException):
            return scripted
        if isinstance(scripted, Raw):
            return scripted.text
        new: object = None
        explicit = False
        if isinstance(scripted, Word):
            body: dict[str, object] = {"pick": word_number(text, scripted.name), "continues": None}
        elif isinstance(scripted, Named):
            body = {"pick": word_number(text, scripted.name), "continues": None}
        elif isinstance(scripted, Grow):
            body = {"pick": word_number(text, scripted.name), "continues": None}
            new, explicit = _new(scripted.first, scripted.second), True
        elif isinstance(scripted, Literal):
            pick = word_number(text, scripted.pick.name) if isinstance(scripted.pick, Named) else scripted.pick
            body = {"pick": pick, "continues": scripted.continues}
            new, explicit = scripted.new, True
        elif isinstance(scripted, Continue):
            number = scripted.number if scripted.said is None else record_number(text, scripted.said)
            body = {"pick": "continue", "continues": number}
        elif isinstance(scripted, Pick):
            body = {"pick": scripted.value, "continues": scripted.continues}
        else:
            raise AssertionError(f"unknown scripted pick {scripted!r}")
        if RECORDS_HEADER not in text:
            return json.dumps({"pick": body["pick"]}, ensure_ascii=False)  # 物理 lane：只有 pick
        if explicit or self._asks_new():
            body["new"] = new
        return json.dumps(body, ensure_ascii=False)


class GrowSite(Site):
    """装好的 Runtime（行为侧 + 语义树），模型替身是 ``GrowRouter``。"""

    def __init__(self, tmp_path: Path, **options: Any) -> None:
        self.runtime, self.router = runtime_of(config(tmp_path, **options), GrowRouter())
        branches = self.runtime.components.behavior
        assert branches is not None
        self.branches = branches
        self.session = branches.session
        self.physical = branches.physical
        self._sequences: dict[str, int] = {}
        self._probes = 0

    # ── 读 ──
    @property
    def state_path(self) -> Path:
        return Path(self.session.tree.root) / STATE_FILENAME

    def state(self) -> Any:
        """词表状态文件的内容（没有这个文件是 None）。"""

        path = self.state_path
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def state_bytes(self) -> bytes | None:
        path = self.state_path
        return path.read_bytes() if path.is_file() else None

    def occurrences(self) -> list[dict[str, object]]:
        tree = self.session.tree
        return [dict(tree.read(address).fields) for address in tree.list_addresses(BehaviorKind.OCCURRENCE)]

    def token_at(self, moment: datetime) -> str:
        """开始时刻是 ``moment`` 的那条记录的 ``kind_token``。"""

        found = [
            str(fields["kind_token"])
            for fields in self.occurrences()
            if datetime.fromisoformat(str(fields["started_at"])) == moment
        ]
        assert len(found) == 1, f"expected one occurrence at {moment}, found {found}"
        return found[0]

    def occurrence_path(self, moment: datetime) -> Path:
        tree = self.session.tree
        for address in tree.list_addresses(BehaviorKind.OCCURRENCE):
            fields = tree.read(address).fields
            if datetime.fromisoformat(str(fields["started_at"])) == moment:
                return Path(tree.path_for(address))
        raise AssertionError(f"no occurrence at {moment}")

    def series_tokens(self, cutoff: date) -> dict[datetime, str]:
        """换代那份常驻序列读者（组合根装的）读出来的：开始时刻 → ``kind_token``。"""

        generation = self.runtime.components.generation
        assert generation is not None
        series = generation.generation.series.read(cutoff=cutoff)
        return {record.started_at: record.kind_token for record in series.records}

    def concepts(self) -> Any:
        return ConceptStore(self.runtime.config.scene_root).read_all()

    def in_use(self) -> set[str]:
        """语义树里没停用的行为概念（= 在用的节点编号）。"""

        concepts = self.concepts()
        return {identity for identity in concepts.behaviors() if not concepts[identity].retired}

    def retired(self, identity: str) -> bool:
        concepts = self.concepts()
        assert identity in concepts.behaviors(), f"{identity} has no behaviour concept"
        return bool(concepts[identity].retired)

    def probe(self, day: date, hour: int = 22) -> Any:
        """跑一轮答 ``none`` 的会话轮（不进序列），返回这一次选词提示词里解析出的词表。"""

        self._probes += 1
        self.router.picks.append(NONE)
        before = len(self.router.texts("pick"))
        self.consume(self.turn(f"probe-{self._probes}", at(day, hour, self._probes % 60), "谢谢，辛苦了"))
        self.reduce()
        texts = self.router.texts("pick")
        assert len(texts) == before + 1
        return vocabulary_block(texts[-1])

    def last_pick_text(self) -> str:
        return self.router.texts("pick")[-1]

    def generate_in_thread(self, day: date) -> Callable[[], None]:
        """给 ``Hook`` 用：在另一个线程里整代换代、等它做完（有界）。"""

        def run() -> None:
            failure: list[BaseException] = []

            def target() -> None:
                try:
                    self.generate(day)
                except BaseException as exc:  # pragma: no cover - 失败时在主线程报出来
                    failure.append(exc)

            worker = threading.Thread(target=target)
            worker.start()
            worker.join(timeout=60)
            assert not worker.is_alive(), "the generation did not finish while the pick was waiting"
            if failure:
                raise failure[0]

        return run


def system_prompt(text_or_call: Any) -> str:
    """一次选词调用里带【词表】的那条系统消息。"""

    messages = text_or_call.messages
    (found,) = [content for role, content in messages if role == "system" and "【词表】" in content]
    return str(found)


def pick_calls(site: GrowSite) -> list[Any]:
    return [call for call in site.router.calls if call.kind == "pick"]


# ── 状态文件：按值宽口径找条目 ────────────────────────────────────────────────


def _dicts(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _dicts(item)


def flat(value: Any) -> list[Any]:
    """一个条目里的全部标量（含嵌套）。"""

    if isinstance(value, dict):
        return [scalar for item in value.values() for scalar in flat(item)]
    if isinstance(value, list):
        return [scalar for item in value for scalar in flat(item)]
    return [value]


def entries(state: Any, value: object) -> list[dict[str, Any]]:
    """状态文件里直接带着 ``value`` 这个值（字段值或标量列表里的一项）的条目。"""

    found: list[dict[str, Any]] = []
    for item in _dicts(state):
        direct = [
            scalar
            for field_value in item.values()
            for scalar in (field_value if isinstance(field_value, list) else [field_value])
            if not isinstance(scalar, (dict, list))
        ]
        if any(scalar == value and type(scalar) is type(value) for scalar in direct):
            found.append(item)
    return found


def mentions_day(entry: dict[str, Any], day: date) -> bool:
    """条目里有没有一个以 ``day`` 的 ISO 写法开头的时刻 / 日子。"""

    return any(isinstance(scalar, str) and scalar.startswith(day.isoformat()) for scalar in flat(entry))


def names(block: Any) -> list[str]:
    return [word.name for word in block.words]


def line_of(block: Any, name: str) -> Any:
    found = [word for word in block.words if word.name == name]
    assert len(found) == 1, f"expected one line named {name!r} in the vocabulary, found {len(found)}"
    return found[0]


__all__ = [
    "LAST_DOC_CHILD",
    "NAMES",
    "QUERY",
    "STATE_FILENAME",
    "SUMMARY",
    "WEEKLY",
    "WRITE_DOC",
    "Barrier",
    "Grow",
    "GrowRouter",
    "GrowSite",
    "Hook",
    "Literal",
    "Named",
    "at",
    "entries",
    "flat",
    "grown",
    "line_of",
    "mentions_day",
    "names",
    "node",
    "pick_calls",
    "system_prompt",
]
