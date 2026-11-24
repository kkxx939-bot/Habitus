"""第二步乙（会话 lane 选词时可以顺手长一个更细的节点；节点的生命周期）测试共用的现场。

沿用第二步甲的现场（``tests.unit.word_pick.site``：组合根 ``build_runtime`` 装出行为侧两支 + 语义树与换代、模型替身
``Router``），在它上面加：

- 选词答复按共享说明 3.1 的会话输出形状 ``{"pick", "continues", "new"}``：这一次请求的输出形状里有 ``new`` 才带
  ``"new": null``（替身照请求的形状答，所以第二步甲的代码上也答得通）；脚本给了 ``new`` 的一律原样带上。物理 lane
  答 ``{"pick", "new"}``（丙二起物理 lane 也写新节点）。
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
            # 物理 lane：没有 continues；丙二起也带 new
            return json.dumps({"pick": body["pick"], "new": new}, ensure_ascii=False)
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
        return Path(self.session.kind_store.root) / STATE_FILENAME

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

        generation = self.runtime.components.pipelines[0].generation
        assert generation is not None
        series = generation.generation.series.read(cutoff=cutoff)
        return {record.started_at: record.kind_token for record in series.records}

    def concepts(self) -> Any:
        return ConceptStore(self.runtime.config.scene_root / "session").read_all()

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


#: 选词提示词 v3（2026-11-16，"单独问一个问题不算一件事"）在 ``"none"`` 那一项下面加的一行，逐字。
STANDALONE_QUESTION_RULE = (
    "  只是问一个问题、要一个解释或答案，听完就完了、没有让助手替他办什么事：先看词表里有没有一个词说的正是问这一种东西"
    '（提醒他"该<那个词>了"，他这一轮问的就算做到了），有就选那个词；问的是【最近的记录】里某一条那件事的进展、结果或要求，'
    '是接着那一条，选 "continue"；两样都不是，就是没有要办的事，选 "none"。'
)
#: 同一次改动收窄的两个初始词的定义：旧行 → 新行（词表里的写法，编号、名字不变）。
V3_DEFINITIONS = (
    ("82 查资料：就一个具体问题找答案", "82 查资料：为手上的一件事找材料、数据或出处"),
    ("222 学知识：弄懂一个要学的知识点", "222 学知识：系统地学一门课或一个专题"),
)


def with_pick_v3(text: str) -> str:
    """把钉住的旧提示词（实验原文或第二步甲的提示词）换成 v3 应有的样子：只在 ``"none"`` 那一行下面插 ``STANDALONE_QUESTION_RULE``、
    换掉 ``V3_DEFINITIONS`` 两行；别处一字不动。夹具文件保持实验原文，差别写在这里。"""

    lines = text.split("\n")
    (none_at,) = [index for index, line in enumerate(lines) if line.startswith('- "none"：')]
    lines.insert(none_at + 1, STANDALONE_QUESTION_RULE)
    for old, new in V3_DEFINITIONS:
        (where,) = [index for index, line in enumerate(lines) if line == old]
        lines[where] = new
    return "\n".join(lines)


#: 选词提示词 v4（2026-11-19，裁定 106，续话改口径）："continue" 那一项原来的三行（逐字）→ 现在的四行（逐字）。
CONTINUE_RULE_V3 = (
    '- "continue"：接着【最近的记录】里的某一条，同时在 continues 填那条记录的编号。追问那件事办得怎么样、叫停或暂缓它、改口、补一句要求、'
    "让还没办完的那件接着办，都是接着那一条，不是新的一件事。",
    "  只能填【最近的记录】里给出的编号。看不出接的是哪一条、列表里也没有它接的那件事时，不要硬配：它说了要办什么事就按新的一件事选词，"
    '否则选 "none"。',
    '  拿不准是接着还是新的一件事时：这一轮说了要办什么事（哪怕带着"跟之前一样""老样子""照旧"），就是新的一件事；只有追问、叫停、改口、'
    "补一句而没说要办什么事，才是接着。",
)
CONTINUE_RULE_V4 = (
    '- "continue"：接着【最近的记录】里的某一条，同时在 continues 填那条记录的编号。先看这一轮是不是还在办"这场对话里的"某一件事：'
    "是，就接那一条（不一定是最后一条）；不是了，再往下选。",
    "  一件事看的是他要办成什么，不是这一句让助手做哪个动作。为同一件事往下做的每一步——问进展、补要求、改口、叫停、换个做法、改其中一处、"
    "再检查一遍——要办成的东西没变，都是接着那一条，说没说要办什么都算。要办成的东西变了才是新的一件事，哪怕和前面那件有关："
    '拿上一步的结果去办另一件事，是新的；同一种事换个场合再做一次（另一场对话、另一天、"跟上次一样再来一份"），也是新的。',
    CONTINUE_RULE_V3[1],
    '  拿不准是接着还是新的一件事时：可能接的是"这场对话里的"某一条，就当接着；不在这场对话里，就当新的一件事。',
)


def with_pick_v4(text: str) -> str:
    """``with_pick_v3`` 之后再把 "continue" 那一项的三行换成 v4 的四行（``CONTINUE_RULE_V3`` → ``CONTINUE_RULE_V4``）；别处一字不动。"""

    lines = with_pick_v3(text).split("\n")
    (start,) = [index for index, line in enumerate(lines) if line == CONTINUE_RULE_V3[0]]
    assert tuple(lines[start : start + 3]) == CONTINUE_RULE_V3
    lines[start : start + 3] = CONTINUE_RULE_V4
    return "\n".join(lines)


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
    "STANDALONE_QUESTION_RULE",
    "STATE_FILENAME",
    "SUMMARY",
    "V3_DEFINITIONS",
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
    "with_pick_v3",
    "with_pick_v4",
]
