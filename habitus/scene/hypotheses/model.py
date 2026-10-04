"""假设：一条待核对的关系，基准（LLM）写、账本核对。

一条假设 = {前件情境集合} → 后件 · 方面。**前因与后果都按钟面峰分**（2026-09-30 裁定一 + 二-6 后半，2026-10-01 定稿）：

- **后果的峰**（``consequent_peak``）：后件的典型一天（节律口：七天平均曲线取峰）有几个峰，就有几条假设，各自一本账——
  早上那杯、中午那杯、晚上那杯咖啡的平时概率不同、前置条件也不同，"不能把中午那顿算做早饭"（用户 09-30）。
  账只记"那个峰的时段里后件来了没来"，对照取**同一时段**的平时概率。``None`` = 后件没有节律（约球 → 打球），
  **无节律型**：不数峰、没有时效，只有兑现与释放两种结法（2026-09-27 裁定）。
- **前因的峰**（``Antecedent.peak``）：前件有节律也按峰分——下午那杯咖啡和晚上那杯是两个前因，各算对晚睡的影响。
  发生在峰外的归 ``#0`` 那一条，**不丢**（用户 10-01："样本少的时候也是需要关注前因和后果的"）。多前因集合里**每个行为元素
  都分**（"多体下要看到每个行为对后果的影响"）；情境概念没有峰。
- **峰的钟面时段写进假设**（``windows``：概念 → 它的峰表）。峰号的定义是写假设那一刻节律口给的典型一天，以后树重建
  也不改——账要稳定地指着同一个钟面时段。某一天的窗口 = 那天的同一时段两边各展 ``pool_half_width`` 槽（预测树与预测层
  共用的容差，10-01 定复用它），对照 = 那天周几曲线在展宽后那一段上的质量。
- **方向**  ↑ / ↓ —— 先验只给方向，不给大小；**类型**  使能 / 促进 / 抑制只是基准的猜测（``type_prior``），最终由账读出；
  概率方面必须给，时刻/次数方面可以不给；**强度**  不写。

次数方面另有 ``horizon``：数后件接下来几个峰窗口里的次数，对照是那几个窗口的期望之和。

锚在前件的**开始时刻**（树上没有结束时刻，``last_observed_at`` 会随观测覆盖动），多前件锚在集合里**最后开始**
的那个行为概念——所以前件集合里至少要有一个行为概念，情境概念是状态、当不了锚。

次数方面另有 ``horizon``：数后件接下来几个机会里的次数（"晚睡当天咖啡喝几杯" = 当天三个机会），对照是那几个
机会的期望之和。

身份是 (前件集合[各带峰号] · 后件 · 方面 · 后果峰号)，不含版本；集合无序，先 A 后 B 那种有序的是两条假设（链）。同身份改内容
（换方向、换窗口）会让一本账混两种量，所以有内容指纹：账本开承诺时快照它，存储覆写前核对它。没有 status 字段：退役是算法判的，
记在账或投影里，假设文件不动。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType

from habitus.foundation.ids import require_safe_path_segment
from habitus.foundation.integrity import canonical_digest
from habitus.scene.concepts.model import (
    IDENTITY_SEPARATORS,
    MINUTES_PER_DAY,
    ConceptError,
    ConceptSet,
    concept_identity,
    widen_windows,
)

MAX_NOTE_CHARS = 400
MAX_GRADE_CHARS = 40
MAX_ANTECEDENTS = 4
#: 峰号与次数方面的地平线都是小整数：一天的钟面峰不会超过这个数（保护闸）。
MAX_OPPORTUNITIES = 30
#: 身份叶名的三个分隔符。读侧（``views.relations`` 拼上一级的身份借先验）也要用它们，所以是公开的——
#: 手抄一份的话，分隔符一改借力就静默借到不存在的身份上，而 ``load_account`` 对不存在的身份返回空账、不报错。
ELEMENT_SEPARATOR = "+"
GRADE_SEPARATOR = "@"
ASPECT_SEPARATOR = "--"
#: 前件的峰号分隔符：``咖啡#3`` = 咖啡的第 3 个钟面峰；``咖啡#0`` = 峰外。
PEAK_SEPARATOR = "#"
#: 无节律型（``consequent_peak is None``）在身份里占的位：它没有峰号，但身份的段数要一致，
#: 不然解析尾巴的地方要分两种写法。
OPEN_OPPORTUNITY_TOKEN = "open"
#: 安慰剂的身份前缀。安慰剂与真假设同一个命名空间时会互相占位（评审 B-6 ③：闭环后来真想写 ``{咖啡} → 早餐``
#: 被当成"已经有账"丢掉；基准同指纹覆写会把安慰剂连账变成真假设）。前缀落在前件段最前面，
#: 身份尾巴的两段（方面、第几次机会）不动，所以按 ``--`` 解析尾巴的地方都不用改。
PLACEBO_MARK = "~"
_ELEMENT_SEPARATOR = ELEMENT_SEPARATOR
_GRADE_SEPARATOR = GRADE_SEPARATOR
_ASPECT_SEPARATOR = ASPECT_SEPARATOR


class HypothesisError(ValueError):
    """假设与自己的约束、或与概念集矛盾。"""


class Aspect(str, Enum):
    """A 影响 B 的哪个量。三者各有一套结算与强度算法，不共用公式。

    概率那本账同时给出"兑现间隔"（从锚到后件第一次到来隔了几次机会 / 几小时）：它不是第四个方面，是同一本账的
    第二个读数——无节律型假设（约球→打球）读的就是它和兑现率，不读"比平时多几成"。
    """

    PROBABILITY = "probability"
    TIMING = "timing"
    COUNT = "count"


class Direction(str, Enum):
    """概率：多 / 少；时刻：推后 / 提前；次数：增 / 减。"""

    UP = "up"
    DOWN = "down"


class TypePrior(str, Enum):
    ENABLING = "enabling"
    PROMOTING = "promoting"
    INHIBITING = "inhibiting"


class HypothesisOrigin(str, Enum):
    """这条假设是谁写的。四种来源的账**一样攒、一样读**，区别只在谁该为它负责、以及给不给人看。

    - ``BASELINE`` 基准（触点②）凭常识写的；
    - ``NEW_CONCEPT`` 新概念进来时补的；
    - ``MODERATION`` 闭环（触点③）读了"两层不一样"或常态漂移之后提的**结构假设**——
      它**从写入日起攒账、不回填**，这是"不自证"的关键：用来发现它的那批观测不算它的证据；
    - ``PLACEBO`` 算法生成的安慰剂：前件换成一条与后件**无关**的行为。它不是给人看的读数，
      而是一把尺子——安慰剂里"显著"的占比就是误报率的直接测量（七c ⑭），也是"门槛能不能定 3"的判据。
    """

    BASELINE = "baseline"
    NEW_CONCEPT = "new_concept"
    MODERATION = "moderation"
    PLACEBO = "placebo"

    @property
    def is_placebo(self) -> bool:
        return self is HypothesisOrigin.PLACEBO


@dataclass(frozen=True)
class HypothesisSource:
    origin: HypothesisOrigin
    note: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", HypothesisOrigin(self.origin))
        if self.note is not None:
            object.__setattr__(self, "note", _single_line(self.note, "hypothesis source note"))


@dataclass(frozen=True)
class PeakWindow:
    """一个概念的一个钟面峰：第几个、几点到几点（分钟，终点可以过 24:00 表示跨午夜）。写假设那一刻从节律口抄下来。"""

    ordinal: int
    start_minute: int
    end_minute: int

    def __post_init__(self) -> None:
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 1:
            raise HypothesisError("a peak window's ordinal counts from 1")
        for label in ("start_minute", "end_minute"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int):
                raise HypothesisError(f"peak window {label} must be an integer minute")
        if not 0 <= self.start_minute < MINUTES_PER_DAY or not self.start_minute < self.end_minute <= self.start_minute + MINUTES_PER_DAY:
            raise HypothesisError("a peak window starts inside one day and ends within 24 hours of its start")

    def contains(self, minute_of_day: int, *, slack_minutes: int = 0) -> bool:
        """钟面上这一刻（0–1439）落在窗口里没有，两边各留 ``slack_minutes`` 的容差；跨午夜的窗口把次日凌晨也算进去。"""

        start, end = self.start_minute - slack_minutes, self.end_minute + slack_minutes
        return start <= minute_of_day < end or start <= minute_of_day + MINUTES_PER_DAY < end

    def label(self) -> str:
        tail = "（次日）" if self.end_minute > MINUTES_PER_DAY else ""
        return f"{_clock(self.start_minute)}–{_clock(self.end_minute)}{tail}"


def _clock(minute: int) -> str:
    hour, rest = divmod(minute % MINUTES_PER_DAY, 60)
    return f"{hour:02d}:{rest:02d}"


def widened_spans(table: Sequence[PeakWindow], slack_minutes: int) -> tuple[tuple[int, int], ...]:
    """一张峰表各窗口加容差之后的 (起, 止) 分钟，不越过与邻窗的中点（见 ``concepts.model.widen_windows``）。"""

    return widen_windows([(item.start_minute, item.end_minute) for item in table], slack_minutes)


def peak_index_of(table: Sequence[PeakWindow], minute_of_day: int, *, slack_minutes: int = 0) -> int:
    """钟面上这一刻（0–1439）落在峰表的第几个窗口（含容差，窗口之间不重叠）；都不在 → 0（峰外）。"""

    for window, (start, end) in zip(table, widened_spans(table, slack_minutes), strict=True):
        if start <= minute_of_day < end or start <= minute_of_day + MINUTES_PER_DAY < end or start <= minute_of_day - MINUTES_PER_DAY < end:
            return window.ordinal
    return 0


@dataclass(frozen=True)
class Antecedent:
    """前件集合的一个元素：概念，可带档（剂量复用多体——元素带档就是另一条假设），可带峰号。

    ``peak``：这个前因发生在它典型一天的第几个钟面峰（``0`` = 峰外）；``None`` = 不按峰分（情境概念、没有节律的行为）。
    """

    concept: str
    grade: str | None = None
    peak: int | None = None

    def __post_init__(self) -> None:
        _ = self.identity  # 名字不合法在这里抛
        if self.peak is not None and (isinstance(self.peak, bool) or not isinstance(self.peak, int) or not 0 <= self.peak <= MAX_OPPORTUNITIES):
            raise HypothesisError(f"an antecedent peak is 0 (off-peak) or 1–{MAX_OPPORTUNITIES}")
        if self.grade is not None:
            grade = _single_line(self.grade, "antecedent grade")
            if len(grade) > MAX_GRADE_CHARS or any(separator in grade for separator in IDENTITY_SEPARATORS):
                raise HypothesisError(f"antecedent grade must be short and free of {IDENTITY_SEPARATORS!r}")
            try:
                require_safe_path_segment(grade, "antecedent grade")
            except ValueError as exc:
                raise HypothesisError(str(exc)) from exc

    @property
    def identity(self) -> str:
        try:
            return concept_identity(self.concept)
        except ConceptError as exc:
            raise HypothesisError(str(exc)) from exc

    @property
    def leaf_token(self) -> str:
        token = self.identity if self.peak is None else f"{self.identity}{PEAK_SEPARATOR}{self.peak}"
        return token if self.grade is None else f"{token}{_GRADE_SEPARATOR}{self.grade}"

    def label(self) -> str:
        peak = "" if self.peak is None else ("·峰外" if self.peak == 0 else f"·第{self.peak}峰")
        text = f"{self.concept}{peak}"
        return text if self.grade is None else f"{text}·{self.grade}"


@dataclass(frozen=True)
class Hypothesis:
    antecedents: tuple[Antecedent, ...]
    consequent: str
    aspect: Aspect
    direction: Direction
    note: str
    source: HypothesisSource
    created_at: datetime
    type_prior: TypePrior | None = None
    split_by: tuple[str, ...] = ()
    #: 后果的第几个钟面峰（1 起）；None = 无节律型（后件没有节律，不数峰、没有时效）。
    consequent_peak: int | None = 1
    #: 概念 → 它的钟面峰表（写假设那一刻从节律口抄下来）。后件有峰号就必须有它的峰表；前件带峰号的同理。
    windows: Mapping[str, tuple[PeakWindow, ...]] = field(default_factory=dict)
    #: 次数方面数后件接下来几个峰窗口里的次数；其他方面恒为 1。
    horizon: int = 1
    #: 无节律型的释放条件：这些行为概念之一在锚之后命中 → 前提作废（再次挂号取代上一次）。空 = 一直立着（B9）。
    #: 只有无节律型能写——节律型的承诺由机会数收口，再加一条释放路径就是两套失效逻辑。
    released_by: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        antecedents = tuple(self.antecedents)
        if not antecedents or len(antecedents) > MAX_ANTECEDENTS:
            raise HypothesisError(f"a hypothesis carries 1–{MAX_ANTECEDENTS} antecedents")
        if any(not isinstance(item, Antecedent) for item in antecedents):
            raise HypothesisError("antecedents must be Antecedent values")
        if len({item.identity for item in antecedents}) != len(antecedents):
            raise HypothesisError("an antecedent set names each concept once")
        # 集合无序：按身份排定顺序，(A,B) 与 (B,A) 是同一条。
        object.__setattr__(self, "antecedents", tuple(sorted(antecedents, key=lambda item: item.identity)))
        consequent = self.consequent_identity
        if consequent in {item.identity for item in self.antecedents}:
            raise HypothesisError("a behaviour cannot be its own antecedent (no self-loops, B13)")
        object.__setattr__(self, "aspect", Aspect(self.aspect))
        object.__setattr__(self, "direction", Direction(self.direction))
        if self.type_prior is not None:
            object.__setattr__(self, "type_prior", TypePrior(self.type_prior))
        if self.aspect is Aspect.PROBABILITY and self.type_prior is None:
            raise HypothesisError("probability hypotheses carry a type prior")
        if self.aspect is Aspect.PROBABILITY:
            expected = Direction.DOWN if self.type_prior is TypePrior.INHIBITING else Direction.UP
            if self.direction is not expected:
                raise HypothesisError("for the probability aspect the direction must agree with the type prior")
        if self.consequent_peak is not None and (
            isinstance(self.consequent_peak, bool) or not isinstance(self.consequent_peak, int) or not 1 <= self.consequent_peak <= MAX_OPPORTUNITIES
        ):
            raise HypothesisError(f"consequent_peak is the consequent's 1st–{MAX_OPPORTUNITIES}th clock peak, or None for an open-ended consequent")
        if self.consequent_peak is None and self.aspect is not Aspect.PROBABILITY:
            # 时刻与次数量的是后件**某个峰**上的到来时刻 / 接下来几个峰里的次数，都得知道是哪个峰；只有"会不会来、
            # 隔多久来"可以不限定。
            raise HypothesisError("timing and count hypotheses name the consequent peak they measure")
        windows = {concept_identity(name): tuple(items) for name, items in dict(self.windows).items()}
        for name, items in windows.items():
            if not items or any(not isinstance(item, PeakWindow) for item in items):
                raise HypothesisError(f"the peak windows of {name!r} must be a non-empty tuple of PeakWindow")
            if [item.ordinal for item in items] != list(range(1, len(items) + 1)):
                raise HypothesisError(f"the peak windows of {name!r} are numbered 1..n in clock order")
        object.__setattr__(self, "windows", MappingProxyType(windows))
        if self.consequent_peak is not None:
            table = windows.get(consequent, ())
            if self.consequent_peak > len(table):
                raise HypothesisError(f"consequent {self.consequent!r} has no peak #{self.consequent_peak} in its window table")
        for item in self.antecedents:
            if item.peak is not None and item.peak > len(windows.get(item.identity, ())):
                raise HypothesisError(f"antecedent {item.concept!r} has no peak #{item.peak} in its window table")
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or not 1 <= self.horizon <= MAX_OPPORTUNITIES:
            raise HypothesisError(f"horizon counts 1–{MAX_OPPORTUNITIES} opportunities")
        if self.aspect is not Aspect.COUNT and self.horizon != 1:
            raise HypothesisError("only count hypotheses look across several opportunities")
        released = tuple(self.released_by)
        if released and self.consequent_peak is not None:
            raise HypothesisError("released_by belongs to open-ended hypotheses (consequent_peak=None) only")
        release_identities: list[str] = []
        for name in released:
            try:
                release_identities.append(concept_identity(name))
            except ConceptError as exc:
                raise HypothesisError(str(exc)) from exc
        if len(set(release_identities)) != len(release_identities):
            raise HypothesisError("released_by names each concept once")
        if self.consequent_identity in release_identities:
            # 后件自己命中就是"兑现"；再让它同时表示"作废"，同一条记录会同时收两种结法。
            raise HypothesisError("released_by cannot name the consequent")
        object.__setattr__(self, "released_by", released)
        object.__setattr__(self, "note", _single_line(self.note, "hypothesis note"))
        if not isinstance(self.source, HypothesisSource):
            raise HypothesisError("hypothesis source must be a HypothesisSource")
        if not isinstance(self.created_at, datetime) or self.created_at.utcoffset() is None:
            raise HypothesisError("hypothesis created_at must be a timezone-aware datetime")
        object.__setattr__(self, "created_at", self.created_at.astimezone(UTC))
        split = tuple(self.split_by)
        identities: list[str] = []
        for name in split:
            try:
                identities.append(concept_identity(name))
            except ConceptError as exc:
                raise HypothesisError(str(exc)) from exc
        if len(set(identities)) != len(identities):
            raise HypothesisError("split_by names each situation concept once")
        object.__setattr__(self, "split_by", split)
        _ = self.leaf  # 叶名不合法在这里抛

    @property
    def is_open_ended(self) -> bool:
        """无节律型（不数峰、没有时效，只有兑现与释放两种结法）。"""

        return self.consequent_peak is None

    @property
    def consequent_window(self) -> PeakWindow | None:
        """后果那个峰的钟面时段；无节律型为 None。"""

        if self.consequent_peak is None:
            return None
        return self.windows[self.consequent_identity][self.consequent_peak - 1]

    def antecedent_windows(self, antecedent: Antecedent) -> tuple[PeakWindow, ...]:
        """一个带峰号的前件的全部峰表（判"发生在第几个峰 / 峰外"要看整张表）。"""

        return self.windows.get(antecedent.identity, ())

    @property
    def consequent_identity(self) -> str:
        try:
            return concept_identity(self.consequent)
        except ConceptError as exc:
            raise HypothesisError(str(exc)) from exc

    @property
    def opportunity_token(self) -> str:
        """身份里表示后果峰号的那一段：节律型是数字，无节律型是 ``open``。"""

        return OPEN_OPPORTUNITY_TOKEN if self.consequent_peak is None else str(self.consequent_peak)

    @property
    def leaf(self) -> str:
        """``<前件#峰@档+前件>--<方面>--<后果峰号>``，是这条假设在后件目录下的文件名（不含 ``.md``）。

        峰号进身份：一天多峰的后件逐峰各一条假设、各一本账；前因有峰也带峰号（2026-09-30 裁定一与二-6）。
        **安慰剂带 ``PLACEBO_MARK`` 前缀**：它是尺子不是读数，不能与真假设争同一个名字。
        """

        leaf = (
            (PLACEBO_MARK if self.source.origin.is_placebo else "")
            + _ELEMENT_SEPARATOR.join(item.leaf_token for item in self.antecedents)
            + _ASPECT_SEPARATOR
            + self.aspect.value
            + _ASPECT_SEPARATOR
            + self.opportunity_token
        )
        try:
            return require_safe_path_segment(leaf, "hypothesis leaf")
        except (TypeError, ValueError) as exc:
            raise HypothesisError(f"hypothesis identity is not a safe path segment: {exc}") from exc

    @property
    def identity(self) -> str:
        return f"{self.consequent_identity}/{self.leaf}"

    @property
    def fingerprint(self) -> str:
        """会改变"账在量什么"的内容：方向、第几次机会、地平线、释放条件、先验类型、分账建议。理由与来源不在里面。"""

        return canonical_digest(
            {
                "identity": self.identity,
                "direction": self.direction.value,
                "consequent_peak": self.consequent_peak,
                "windows": {name: [[w.ordinal, w.start_minute, w.end_minute] for w in items] for name, items in sorted(self.windows.items())},
                "horizon": self.horizon,
                "released_by": sorted(concept_identity(name) for name in self.released_by),
                "type_prior": None if self.type_prior is None else self.type_prior.value,
                "split_by": sorted(concept_identity(name) for name in self.split_by),
            }
        )[:16]

    @property
    def opportunity_label(self) -> str:
        if self.consequent_peak is None:
            release = "，直到后件到来" if not self.released_by else "，直到后件到来或 " + " / ".join(self.released_by)
            return f"无节律型：不数峰{release}"
        window = self.consequent_window
        clock = "" if window is None else f"（{window.label()}）"
        if self.aspect is Aspect.COUNT and self.horizon > 1:
            return f"后件的第 {self.consequent_peak}–{self.consequent_peak + self.horizon - 1} 个峰{clock}"
        return f"后件的第 {self.consequent_peak} 个峰{clock}"

    def label(self) -> str:
        return "{" + ", ".join(item.label() for item in self.antecedents) + "} → " + self.consequent + " · " + ASPECT_LABELS[self.aspect]

    def validate_against(self, concepts: ConceptSet) -> None:
        """对着概念集核对：后件是行为概念；前件都在、档是定义过的、至少一个行为概念、与后件不在同一条祖先链上；
        分账项是情境概念。"""

        if self.consequent_identity not in concepts or not concepts[self.consequent_identity].role.is_behavior:
            raise HypothesisError(f"consequent {self.consequent!r} must be a behaviour concept")
        anchors = 0
        for item in self.antecedents:
            if item.identity not in concepts:
                raise HypothesisError(f"antecedent {item.concept!r} is not in the concept set")
            definition = concepts[item.identity]
            if item.grade is not None and item.grade not in {grade.name for grade in definition.grades}:
                raise HypothesisError(f"concept {item.concept!r} defines no grade {item.grade!r}")
            if definition.role.is_behavior and (
                concepts.is_ancestor(item.identity, self.consequent_identity) or concepts.is_ancestor(self.consequent_identity, item.identity)
            ):
                # 「打球 → 运动」是重言：读侧沿 parent 链聚合之后它 100% 命中，会以最窄的区间排进"稳定成立"。
                raise HypothesisError(f"{item.concept!r} and {self.consequent!r} lie on one ancestor chain; the hypothesis is a tautology")
            anchors += definition.role.is_behavior
        if anchors == 0:
            raise HypothesisError("an antecedent set needs at least one behaviour concept to anchor its window (B14)")
        for name in self.split_by:
            identity = concept_identity(name)
            if identity not in concepts or not concepts[identity].role.is_situation:
                raise HypothesisError(f"split_by {name!r} must be a situation concept")
        for name in self.released_by:
            identity = concept_identity(name)
            # 释放是机械匹配的：账本按后件那一套 ``behaviour_hits``（含祖先聚合）在记录上找它，所以只能是行为概念。
            if identity not in concepts or not concepts[identity].role.is_behavior:
                raise HypothesisError(f"released_by {name!r} must be a behaviour concept")


ASPECT_LABELS = {Aspect.PROBABILITY: "概率", Aspect.TIMING: "时刻", Aspect.COUNT: "次数"}


def _single_line(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise HypothesisError(f"{label} must be non-empty text without surrounding whitespace")
    if "\n" in value or "\r" in value or any(not character.isprintable() for character in value):
        raise HypothesisError(f"{label} must be a single printable line")
    if len(value) > MAX_NOTE_CHARS:
        raise HypothesisError(f"{label} exceeds {MAX_NOTE_CHARS} characters")
    return value


__all__ = [
    "ASPECT_LABELS",
    "peak_index_of",
    "widened_spans",
    "ASPECT_SEPARATOR",
    "ELEMENT_SEPARATOR",
    "GRADE_SEPARATOR",
    "MAX_ANTECEDENTS",
    "MAX_OPPORTUNITIES",
    "OPEN_OPPORTUNITY_TOKEN",
    "PEAK_SEPARATOR",
    "PLACEBO_MARK",
    "Antecedent",
    "Aspect",
    "Direction",
    "Hypothesis",
    "HypothesisError",
    "HypothesisOrigin",
    "HypothesisSource",
    "PeakWindow",
    "TypePrior",
]
