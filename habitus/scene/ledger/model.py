"""账：一条假设的每一次机会——承诺（前件命中时开）与结算（后件到来、或等过了足够多次机会仍没来）。

账本**只存事实**：什么时候、因为哪条 occurrence、后件本来该在哪几次机会上来、实际落在第几次 / 等过了几次、
每次机会我们看清了没有。强度、类型、PN/PS 都是读时算的（``views/``），这里一个都不落。

- **账按钟面峰记**（2026-09-26 裁定"窗按后件的机会数量"，2026-10-01 改成"一本账一个窗口"）。后件的一个"窗口"是
  它在预测树上的一个常态发生时段（``marginal`` 曲线的一个峰，``PeakWindow``）：早餐一天一个，三杯咖啡一天三个，
  各有各的账、各有各的平时概率、不合成"今天会不会"。承诺上的快照是**这条假设自己那个窗口在锚之后的第一次落地**
  （含容差），次数方面再往后铺 ``horizon`` 个窗口；窗口的平时概率 = 落地那天曲线在窗口内的质量（曲线缺就 None，不猜）。
- **承诺写完不改**。锚 = 触发 occurrence 的 ``started_at``（多前件锚在集合里最后开始的行为概念）；开承诺那一刻
  向组合根注入的口要窗口落地（账本不读树），快照进承诺，事后树重建了也不改。
- **结算一套**：窗口里后件来了 → ``OCCURRED``（隔了几小时）；过了窗口末尾、看清了、没来 → ``ABSENT``；
  没看清 → ``CENSORED``（右删失）。不再数"第几次机会"。
- **无节律型的账不数机会**（``consequent_peak=None``，2026-09-27 裁定）：约球→打球、挂号→就诊这类后件没节律，
  承诺只有兑现（后件来了）与**释放**（``released_by`` 的概念命中，``RELEASED`` + ``releasing_uri``）两种结法，
  没有按机会数的删失——"很多行为没有机会时效，一个行为的影响在很后面"。兑现是 **FIFO** 的：同一假设的开放
  承诺按锚序，一个后件 occurrence 只兑现最早那条，不然约球两次、打球一次会读成两次都兑现了。
- **删失按机会判**：一个机会的时段没看清 → 这个机会"未观测"，不算它过了、也不算缺席（七d-2：熬夜→起晚→
  没在看，不能成"没吃早饭"的假证据）。
- 提醒是承诺之后的事，另记一份（``Intervention``），读时 join。
- 承诺快照**假设的内容指纹**：同身份的假设改了第几次机会或方向，账本能认出哪些承诺是按旧口径开的。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import Enum
from types import MappingProxyType
from typing import Protocol

from habitus.behavior.model import BehaviorAddress, BehaviorKind
from habitus.behavior.uri import BehaviorURI, BehaviorURIError, BehaviorURINodeType
from habitus.foundation.text import clean_line
from habitus.scene.concepts.model import ConceptError, concept_identity
from habitus.scene.hypotheses.model import MAX_OPPORTUNITIES, Aspect, PeakWindow
from habitus.scene.occurrences.model import ConceptHit

MAX_NOTE_CHARS = 400
#: 一份快照最多装多少个窗口落地：次数方面要铺 ``horizon`` 个；保护闸，不是设计量。
MAX_SNAPSHOT_OPPORTUNITIES = 2 * MAX_OPPORTUNITIES


class LedgerError(ValueError):
    """账本记录与自己的约束矛盾。"""


class Outcome(str, Enum):
    """结算怎么收的。

    概率：``OCCURRED``（窗口里后件来了）｜ ``ABSENT``（窗口看清了、没来）｜ ``CENSORED``（窗口没看清；右删失）
    ｜ ``RELEASED``（**只有无节律型**：``released_by`` 的概念命中，这条前提被后来的事作废了——再次挂号取代
    上一次那张号。既不是兑现也不是"没来"，读兑现率时它在分母不在分子）。
    时刻：``OBSERVED``（那次机会上来了，记时刻）｜ ``ABSENT``（那次机会已观测地过了、没来——不是"很晚"，是没有
    观测值）｜ ``CENSORED``（那次机会没看清）。
    次数：``COUNTED``（那几次机会都过完了，记计数，0 也是计数）｜ ``CENSORED``（有机会没看清）。
    """

    OCCURRED = "occurred"
    OBSERVED = "observed"
    ABSENT = "absent"
    COUNTED = "counted"
    CENSORED = "censored"
    RELEASED = "released"


#: 每个方面允许哪些结果。存储层是这条不变量唯一的守门人：``Settlement`` 自己不知道 aspect。
OUTCOMES_BY_ASPECT: Mapping[Aspect, frozenset[Outcome]] = MappingProxyType(
    {
        Aspect.PROBABILITY: frozenset({Outcome.OCCURRED, Outcome.ABSENT, Outcome.CENSORED, Outcome.RELEASED}),
        Aspect.TIMING: frozenset({Outcome.OBSERVED, Outcome.ABSENT, Outcome.CENSORED}),
        Aspect.COUNT: frozenset({Outcome.COUNTED, Outcome.CENSORED}),
    }
)


class Response(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class WindowSpan:
    """[start, end) 一段时刻；覆盖口按它答"这段看清了多少"。"""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        _aware(self.start, "span start")
        _aware(self.end, "span end")
        if self.end <= self.start:
            raise LedgerError("span end must come after its start")

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment < self.end


@dataclass(frozen=True)
class Opportunity:
    """后件的一个**钟面峰窗口**在某一天的落点：假设里写死的峰时段（两边各展容差）落在这一天上。

    ``at`` 窗口中心；``span`` 窗口（含容差）；``probability`` 那天那个周几的曲线在这一段上的质量（截到 1）——
    ``None`` = 那个周几没有曲线，**不猜**（二-4："没看过就当没看到"）：窗口还在、账照开，只是没有对照。
    由组合根注入的口从树上算，账本只核对形状。
    """

    at: datetime
    span: WindowSpan
    probability: float | None

    def __post_init__(self) -> None:
        _aware(self.at, "opportunity at")
        if not isinstance(self.span, WindowSpan):
            raise LedgerError("opportunity span must be a WindowSpan")
        if not self.span.contains(self.at):
            raise LedgerError("an opportunity's centre lies inside its span")
        if self.probability is not None:
            if isinstance(self.probability, bool) or not isinstance(self.probability, int | float) or not math.isfinite(self.probability):
                raise LedgerError("opportunity probability must be a finite number or None")
            if not 0.0 <= float(self.probability) <= 1.0:
                raise LedgerError("opportunity probability lies in [0, 1]")
            object.__setattr__(self, "probability", float(self.probability))


@dataclass(frozen=True)
class OpportunitySnapshot:
    """开承诺那一刻的对照：后件从锚起的前若干个机会，按时间升序（第 1 个就是"第 1 次机会"）；``generation`` 是树的代。

    **第 1 次机会是锚正在其中的那个峰**（只要它还没结束）：晚睡·重 07:00 命中之后，当天 06:40–09:40 的起床峰就是
    "接下来那次机会"——人晚起正是晚在这个峰上。按"峰的开始晚于锚"去铺会把它丢掉，于是当天 12:00 的起床被算到
    次日那个峰上，时刻差读成 −1210 分（真值 +230）。
    """

    generation: str
    opportunities: tuple[Opportunity, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.generation, str) or not self.generation.strip():
            raise LedgerError("snapshot generation must be non-empty text")
        items = tuple(self.opportunities)
        if not items:
            raise LedgerError("a snapshot carries at least one opportunity; with none the claim opens with control=None")
        if len(items) > MAX_SNAPSHOT_OPPORTUNITIES:
            raise LedgerError(f"a snapshot carries at most {MAX_SNAPSHOT_OPPORTUNITIES} opportunities")
        if any(not isinstance(item, Opportunity) for item in items):
            raise LedgerError("opportunities must be Opportunity values")
        for earlier, later in zip(items, items[1:], strict=False):
            if later.span.start < earlier.span.end:
                raise LedgerError("opportunities are ascending and do not overlap")
        object.__setattr__(self, "opportunities", items)

    def at(self, index: int) -> Opportunity | None:
        """第 ``index`` 次机会（从 1 数）；快照里没那么多就 None。"""

        if isinstance(index, bool) or not isinstance(index, int) or index < 1:
            raise LedgerError("opportunity index counts from 1")
        return self.opportunities[index - 1] if index <= len(self.opportunities) else None

    def index_of(self, moment: datetime) -> int | None:
        """``moment`` 落在第几个窗口（从 1 数）；不在任何窗口里 → ``None``。

        账按钟面窗口记（2026-10-01）：窗口之间的到来不归任何一本账——容差已经由机会口把窗口展宽过了
        （``slack_minutes``），这里不再第二次放宽。
        """

        for index, item in enumerate(self.opportunities, start=1):
            if item.span.contains(moment):
                return index
        return None

    @property
    def last_end(self) -> datetime:
        """最后一个机会结束的时刻：快照能说话的边界。"""

        return self.opportunities[-1].span.end

    def expected_count(self, first: int, horizon: int) -> float | None:
        """第 ``first`` 到第 ``first + horizon - 1`` 个窗口的概率之和（次数方面的对照）；快照不够长或有窗口没对照返回 None。"""

        items = self.opportunities[first - 1 : first - 1 + horizon]
        if len(items) < horizon or any(item.probability is None for item in items):
            return None
        return sum(item.probability for item in items if item.probability is not None)


@dataclass(frozen=True)
class OpportunityRequest:
    """向组合根注入的口要什么：这个后件、从这个锚往后的前 ``count`` 个**钟面窗口**的落点。

    ``window`` 是假设里写死的峰时段（后果第 k 峰）；第 1 个落点是锚之后第一个还没结束的那一天的该时段，
    往后逐日铺（次数方面要接下来几个窗口时按后件的整张峰表轮着铺，``windows`` 给全表）。
    """

    consequent: str
    anchor: datetime
    count: int
    window: PeakWindow
    windows: tuple[PeakWindow, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.window, PeakWindow):
            raise LedgerError("an opportunity request names the peak window it asks about")
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise LedgerError("count is a positive integer")


class OpportunityProvider(Protocol):
    #: 窗口两边各展多少分钟的容差（= 预测树的 ``pool_half_width`` × 槽宽；10-01 定复用它）。
    slack_minutes: int

    def opportunities(self, request: OpportunityRequest) -> OpportunitySnapshot | None: ...


@dataclass(frozen=True)
class ObservedGap:
    start: datetime
    end: datetime
    kind: str

    def __post_init__(self) -> None:
        _aware(self.start, "gap start")
        _aware(self.end, "gap end")
        if self.end <= self.start:
            raise LedgerError("gap end must come after its start")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise LedgerError("gap kind must be non-empty text")


@dataclass(frozen=True)
class Coverage:
    """一段里看清了多少：观测覆盖比例 + 空白段。由组合根注入的口给（空白在行为树上，读法在 views）。"""

    observed_fraction: float
    gaps: tuple[ObservedGap, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.observed_fraction, bool) or not isinstance(self.observed_fraction, int | float):
            raise LedgerError("observed_fraction must be a number")
        if not 0.0 <= float(self.observed_fraction) <= 1.0:
            raise LedgerError("observed_fraction lies in [0, 1]")
        object.__setattr__(self, "observed_fraction", float(self.observed_fraction))
        gaps = tuple(self.gaps)
        if any(not isinstance(gap, ObservedGap) for gap in gaps):
            raise LedgerError("gaps must be ObservedGap values")
        object.__setattr__(self, "gaps", gaps)


class CoverageProvider(Protocol):
    def coverage(self, span: WindowSpan) -> Coverage: ...


@dataclass(frozen=True)
class OpportunityPass:
    """一次已经过去、后件没来的机会：几点的、看清了没（没看清的不算它过了，也不算缺席）。"""

    at: datetime
    observed: bool

    def __post_init__(self) -> None:
        _aware(self.at, "pass at")
        if not isinstance(self.observed, bool):
            raise LedgerError("pass observed must be a boolean")


@dataclass(frozen=True)
class ClaimRef:
    """一条承诺的地址：假设身份 + 触发 occurrence 的本地日 + 它的叶名。结算与提醒都用它指回来。"""

    hypothesis_identity: str
    day: date
    leaf: str

    def __post_init__(self) -> None:
        consequent, separator, leaf = self.hypothesis_identity.partition("/")
        if not separator or not leaf or "/" in leaf:
            raise LedgerError("hypothesis identity is '<consequent>/<leaf>'")
        try:
            concept_identity(consequent)
        except ConceptError as exc:
            raise LedgerError(str(exc)) from exc
        if isinstance(self.day, datetime) or not isinstance(self.day, date):
            raise LedgerError("claim day must be a date")
        try:
            BehaviorAddress.from_identity(BehaviorKind.OCCURRENCE, self.day, self.leaf)
        except (TypeError, ValueError) as exc:
            raise LedgerError(f"claim leaf is not an occurrence leaf name: {exc}") from exc

    @property
    def consequent(self) -> str:
        return self.hypothesis_identity.partition("/")[0]

    @property
    def hypothesis_leaf(self) -> str:
        return self.hypothesis_identity.partition("/")[2]


@dataclass(frozen=True)
class Claim:
    hypothesis_identity: str
    hypothesis_fingerprint: str
    aspect: Aspect
    trigger_uri: str
    antecedent_hits: tuple[ConceptHit, ...]
    antecedent_uris: tuple[str, ...]
    situation_snapshot: tuple[str, ...]
    control: OpportunitySnapshot | None
    created_at: datetime
    #: 开承诺那条记录上**判过**的情境（在场的在 ``situation_snapshot`` 里，不在场的也在这里）。稳定性按情境分层时
    #: 只收判过它的承诺（2026-09-30 裁定八 ②）：没判过 ≠ 不在场。从命中记录的 ``situations_checked`` 抄来；
    #: 旧记录没有这一栏时为空，于是那些承诺不进任何一层。
    situations_checked: tuple[str, ...] = ()
    #: 从 ``trigger_uri`` 算一次就存着（不参与相等与 repr）：读侧问锚问得极密，每次重解析 URI 是读数的主要开销。
    _address: BehaviorAddress = field(init=False, compare=False, repr=False)
    _anchor: datetime = field(init=False, compare=False, repr=False)
    _ref: ClaimRef = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        trigger = _occurrence_uri(self.trigger_uri, "trigger_uri")
        object.__setattr__(self, "trigger_uri", str(trigger))
        # 地址、锚、ref 各算一次存起来：URI 解析不便宜，而读侧每条承诺要问锚上百次（``Account.block_of`` 逐条算块），
        # 实测一条假设 127 条承诺 = 207 万次解析、199 秒。身份不合法在这里抛（``ClaimRef`` 自己校验）。
        address = trigger.to_address()
        object.__setattr__(self, "_address", address)
        object.__setattr__(self, "_anchor", address.started_at)
        object.__setattr__(self, "_ref", ClaimRef(self.hypothesis_identity, address.occurred_on, address.identity_name))
        if not isinstance(self.hypothesis_fingerprint, str) or not self.hypothesis_fingerprint.strip():
            raise LedgerError("hypothesis_fingerprint must be non-empty text")
        object.__setattr__(self, "aspect", Aspect(self.aspect))
        hits = tuple(self.antecedent_hits)
        if not hits or any(not isinstance(hit, ConceptHit) for hit in hits):
            raise LedgerError("antecedent_hits must be a non-empty sequence of ConceptHit")
        if len({hit.identity for hit in hits}) != len(hits):
            raise LedgerError("antecedent_hits repeats a concept")
        object.__setattr__(self, "antecedent_hits", tuple(sorted(hits, key=lambda hit: hit.identity)))
        uris = tuple(str(_occurrence_uri(uri, "antecedent uri")) for uri in self.antecedent_uris)
        if str(trigger) not in uris:
            raise LedgerError("antecedent_uris must include the trigger")
        if len(set(uris)) != len(uris):
            raise LedgerError("antecedent_uris repeats an occurrence")
        object.__setattr__(self, "antecedent_uris", tuple(sorted(uris)))
        snapshot: list[str] = []
        for name in self.situation_snapshot:
            try:
                concept_identity(name)
            except ConceptError as exc:
                raise LedgerError(f"situation_snapshot: {exc}") from exc
            snapshot.append(name)
        if len({concept_identity(name) for name in snapshot}) != len(snapshot):
            raise LedgerError("situation_snapshot repeats a concept")
        object.__setattr__(self, "situation_snapshot", tuple(snapshot))
        checked: list[str] = []
        for name in self.situations_checked:
            try:
                concept_identity(name)
            except ConceptError as exc:
                raise LedgerError(f"situations_checked: {exc}") from exc
            checked.append(name)
        if len({concept_identity(name) for name in checked}) != len(checked):
            raise LedgerError("situations_checked repeats a concept")
        object.__setattr__(self, "situations_checked", tuple(checked))
        if self.control is not None:
            if not isinstance(self.control, OpportunitySnapshot):
                raise LedgerError("control must be an OpportunitySnapshot or None")
            if self.control.opportunities[0].span.end <= self.anchor:
                raise LedgerError("the first opportunity in a snapshot has not ended by the anchor")
        _aware(self.created_at, "created_at")
        object.__setattr__(self, "created_at", self.created_at.astimezone(UTC))
        # 身份叶名的尾巴是 ``--<方面>--<第几次机会>``：方面在倒数第二段。对不上说明这条承诺挂错了假设。
        head, _, _opportunity = self.hypothesis_identity.rpartition("--")
        leaf_aspect = head.rpartition("--")[2]
        if leaf_aspect != self.aspect.value:
            raise LedgerError(f"claim aspect {self.aspect.value!r} disagrees with its hypothesis identity ({leaf_aspect!r})")

    @property
    def trigger_address(self) -> BehaviorAddress:
        return self._address

    @property
    def anchor(self) -> datetime:
        """锚：触发 occurrence 的开始时刻（构造时算一次）。"""

        return self._anchor

    @property
    def consequent(self) -> str:
        return self.hypothesis_identity.partition("/")[0]

    @property
    def ref(self) -> ClaimRef:
        return self._ref


@dataclass(frozen=True)
class Settlement:
    ref: ClaimRef
    outcome: Outcome
    settled_at: datetime
    observed_at: datetime | None = None
    fulfilling_uri: str | None = None
    latency_hours: float | None = None
    opportunity_index: int | None = None
    passes: tuple[OpportunityPass, ...] = ()
    count: int | None = None
    reason: str | None = None
    #: ``RELEASED`` 专用：哪一条 occurrence 把这条前提作废了（无节律型的 ``released_by`` 命中的那条）。
    releasing_uri: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ClaimRef):
            raise LedgerError("settlement ref must be a ClaimRef")
        object.__setattr__(self, "outcome", Outcome(self.outcome))
        _aware(self.settled_at, "settled_at")
        object.__setattr__(self, "settled_at", self.settled_at.astimezone(UTC))
        if self.observed_at is not None:
            _aware(self.observed_at, "observed_at")
        if self.fulfilling_uri is not None:
            object.__setattr__(self, "fulfilling_uri", str(_occurrence_uri(self.fulfilling_uri, "fulfilling_uri")))
        if self.count is not None and (isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 0):
            raise LedgerError("count must be a non-negative integer")
        if self.latency_hours is not None and (
            isinstance(self.latency_hours, bool) or not isinstance(self.latency_hours, int | float) or not math.isfinite(self.latency_hours) or self.latency_hours < 0
        ):
            raise LedgerError("latency_hours must be a non-negative finite number")
        if self.opportunity_index is not None and (isinstance(self.opportunity_index, bool) or not isinstance(self.opportunity_index, int) or self.opportunity_index < 1):
            raise LedgerError("opportunity_index counts from 1")
        passes = tuple(self.passes)
        if any(not isinstance(item, OpportunityPass) for item in passes):
            raise LedgerError("passes must be OpportunityPass values")
        for earlier, later in zip(passes, passes[1:], strict=False):
            if later.at <= earlier.at:
                raise LedgerError("passes are in ascending order")
        object.__setattr__(self, "passes", passes)
        if self.reason is not None:
            reason = clean_line(self.reason)
            if not reason or len(reason) > MAX_NOTE_CHARS:
                raise LedgerError("reason must be a non-empty line within the length budget")
            object.__setattr__(self, "reason", reason)
        seen = (self.observed_at is not None, self.fulfilling_uri is not None)
        if self.outcome in (Outcome.OCCURRED, Outcome.OBSERVED):
            if seen != (True, True):
                raise LedgerError(f"a {self.outcome.value} settlement names when and by which occurrence")
        elif seen != (False, False):
            raise LedgerError(f"a {self.outcome.value} settlement saw no consequent")
        if (self.latency_hours is not None) != (self.outcome is Outcome.OCCURRED):
            raise LedgerError("latency_hours belongs to occurred settlements only")
        if self.opportunity_index is not None and self.outcome not in (Outcome.OCCURRED, Outcome.OBSERVED, Outcome.ABSENT):
            raise LedgerError("opportunity_index belongs to occurred, observed and absent settlements")
        if (self.count is not None) != (self.outcome is Outcome.COUNTED):
            raise LedgerError("count belongs to counted settlements only")
        if self.reason is not None and self.outcome is not Outcome.CENSORED:
            raise LedgerError("reason belongs to censored settlements only")
        if self.releasing_uri is not None:
            object.__setattr__(self, "releasing_uri", str(_occurrence_uri(self.releasing_uri, "releasing_uri")))
        if (self.releasing_uri is not None) != (self.outcome is Outcome.RELEASED):
            raise LedgerError("a released settlement names the occurrence that released it, and only it does")

    @property
    def observed_passes(self) -> int:
        """后件到来（或删失）之前已观测地过去的机会数——生存分析里的步数。"""

        return sum(1 for item in self.passes if item.observed)

    @property
    def is_right_censored(self) -> bool:
        return self.outcome is Outcome.CENSORED


@dataclass(frozen=True)
class Intervention:
    """提醒记录：指向哪条承诺、几点提醒。整个系统里唯一的干预数据；写口由组合根钉给预测层。

    他的回应另记一份（``InterventionResponse``）：提醒发出时还没有回应，add-only 下先写 ``response=None`` 再写回应就是
    同地址不同内容——"发了但他没理"这一类会根本进不了盘。
    """

    ref: ClaimRef
    reminded_at: datetime
    recorded_at: datetime
    note: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ClaimRef):
            raise LedgerError("intervention ref must be a ClaimRef")
        _aware(self.reminded_at, "reminded_at")
        _aware(self.recorded_at, "recorded_at")
        object.__setattr__(self, "recorded_at", self.recorded_at.astimezone(UTC))
        if self.note is not None:
            note = clean_line(self.note)
            if not note or len(note) > MAX_NOTE_CHARS:
                raise LedgerError("note must be a non-empty line within the length budget")
            object.__setattr__(self, "note", note)

    @property
    def leaf(self) -> str:
        return f"{self.ref.leaf}--{_stamp(self.reminded_at)}"


@dataclass(frozen=True)
class InterventionResponse:
    """他对某次提醒的回应：接受 / 拒绝，几点回的。指向提醒靠 (承诺, reminded_at)。"""

    ref: ClaimRef
    reminded_at: datetime
    responded_at: datetime
    response: Response
    note: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ClaimRef):
            raise LedgerError("response ref must be a ClaimRef")
        _aware(self.reminded_at, "reminded_at")
        _aware(self.responded_at, "responded_at")
        if self.responded_at < self.reminded_at:
            raise LedgerError("a response comes after its reminder")
        object.__setattr__(self, "response", Response(self.response))
        if self.note is not None:
            note = clean_line(self.note)
            if not note or len(note) > MAX_NOTE_CHARS:
                raise LedgerError("note must be a non-empty line within the length budget")
            object.__setattr__(self, "note", note)

    @property
    def intervention_leaf(self) -> str:
        return f"{self.ref.leaf}--{_stamp(self.reminded_at)}"

    @property
    def leaf(self) -> str:
        return f"{self.intervention_leaf}--{_stamp(self.responded_at)}"


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _aware(value: object, label: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise LedgerError(f"{label} must be a timezone-aware datetime")
    return value


def _occurrence_uri(value: object, label: str) -> BehaviorURI:
    try:
        uri = BehaviorURI.parse(value)  # type: ignore[arg-type]
    except (TypeError, BehaviorURIError) as exc:
        raise LedgerError(f"{label} is not a behaviour URI: {exc}") from exc
    if uri.node_type is not BehaviorURINodeType.DOCUMENT or uri.to_address().kind is not BehaviorKind.OCCURRENCE:
        raise LedgerError(f"{label} must identify an occurrence document")
    return uri


__all__ = [
    "MAX_SNAPSHOT_OPPORTUNITIES",
    "OUTCOMES_BY_ASPECT",
    "Claim",
    "ClaimRef",
    "Coverage",
    "CoverageProvider",
    "Intervention",
    "InterventionResponse",
    "LedgerError",
    "ObservedGap",
    "Opportunity",
    "OpportunityPass",
    "OpportunityProvider",
    "OpportunityRequest",
    "OpportunitySnapshot",
    "Outcome",
    "Response",
    "Settlement",
    "WindowSpan",
]
