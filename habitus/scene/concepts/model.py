"""概念定义：语义树的词汇层，一个概念一份。

行为概念跟着基础词表的类走（裁定 20），分三种：
- **基础概念**：词表里一个行为类原样对应的概念，身份 = 类编号（``s-k0004``），显示名是类名；同步词表时自动生成，不调模型。
- **细分概念**：源类 + 区别规则——提醒句相同、只是这一次的条件不同（「晚睡」= 睡觉里比近期常态晚两小时以上的）。
  区别是数值的写成 ``MechanicalRule`` 由算法判；是语义的写成一句区别判据，由模型对着声明的材料答是/否。
- **汇总概念**：几个类合起来（「写代码」= 修改代码 + 排查问题 + 验证测试），曲线是成员类曲线逐槽相加。
概念的组成只在一条 lane 内（与预测树的转移只在同 lane 配对一致）。情境概念不挂类。

- **细分概念的区别二选一**：数值的（"比近期常态晚两小时以上"）写成 ``MechanicalRule``，只由算法判、不问模型——
  这条已经是源类（就寝）的记录，"是不是入睡"不用再问；写不成数值的（"改的是前两天刚改过的同一处"）写成区别判据句，
  由模型对着声明的材料答是/否。模型从不碰数字，算法从不碰语义。区别句要写到对着材料能答是/否。
- **判据可以引用别的事件**："起床后两小时内的第一次进食"要看当天时间线，概念用 ``context=DAY`` 声明；
  引用"常态"的用 ``baseline_keys`` 声明要哪几个常态值。声明了的材料映射时没给到，那一条**不判**
  （记成未决），不让模型替我们答成 false。
- **只有行为概念能当后件**，情境概念（状态 / 对象 / 日型 / 派生）只做前件集合的元素。
- **汇总概念不参与命中**：一条记录的基础概念从它的类编号现读，算不算某个汇总概念，由读侧按成员类聚合
  （成员改了不用重算任何命中）。
- **显示名不许重**：给模型看的是名字不是编号（裁定 21-1），同一条 lane 能一起出现的概念名字不许重。
- **没有版本**：定义变了就整个重算它的命中。
- **档由算法判**：LLM 只答是不是，几档按数值规则机械定。有 ``MechanicalRule`` 的概念，档比的是规则算出的
  那个量（相对常态的偏移就按偏移分档）。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType

from habitus.behavior.model import semantic_name
from habitus.foundation.ids import canonical_path_identity, canonical_text_identity, require_safe_path_segment
from habitus.foundation.integrity import canonical_digest
from habitus.scene.concepts.situation import SituationBasis, SituationError, SituationRule

MAX_DEFINITION_CHARS = 400
MAX_NOTE_CHARS = 400
MAX_GRADE_NAME_CHARS = 40
MAX_BASELINE_KEY_CHARS = 120
MAX_BASELINE_KEYS = 8
MIN_GRADES = 2
MAX_GRADES = 3
MINUTES_PER_DAY = 24 * 60
#: 概念身份、档名会被拼进组合键（前因的集合、带档的前因），这几个分隔符不许出现在它们里面，
#: 否则两个不同的组合能拼出同一个键（实测：概念名 ``出差中+晚睡`` 与集合 ``{出差中, 晚睡}``）。
IDENTITY_SEPARATORS = ("+", "@", "--", "#")


class ConceptError(ValueError):
    """概念定义或概念集与自己的约束矛盾。"""


class ConceptLookupError(ConceptError, KeyError):
    """按名字查不到概念——同时是 ``KeyError``，让 ``Mapping.get`` 照常返回默认值。"""

    def __str__(self) -> str:  # KeyError 会给消息加引号，这里还原成普通文本
        return ValueError.__str__(self)


class ConceptRole(str, Enum):
    """行为概念可当后件；其余四种是情境概念，只当前件集合的元素。派生情境由算法从历史命中算。"""

    BEHAVIOR = "behavior"
    STATE = "state"
    OBJECT = "object"
    DAY_TYPE = "day_type"
    DERIVED = "derived"

    @property
    def is_behavior(self) -> bool:
        return self is ConceptRole.BEHAVIOR

    @property
    def is_situation(self) -> bool:
        return not self.is_behavior


class ConceptOrigin(str, Enum):
    """概念从哪来：同步词表自动生成（基础概念），或模型（触点①）写的（细分 / 汇总 / 情境）。"""

    VOCABULARY = "vocabulary"
    AUTHOR = "author"


class ConceptKind(str, Enum):
    """行为概念与词表类的关系（裁定 20）。情境概念没有这一项。"""

    BASE = "base"
    REFINEMENT = "refinement"
    GROUP = "group"


class ContextScope(str, Enum):
    """区别判据要看什么材料：只看这一条、它所在那一天的时间线，还是近几天同一个类的记录。"""

    OCCURRENCE = "occurrence"
    DAY = "day"
    RECENT = "recent"


class GradeMeasure(str, Enum):
    """规则与档能用的量。开始时刻按一天里的分钟数；时长按开始到最后所见的分钟数。

    "连续几天"这类跨日的量不在这里：它要看历史命中，属派生情境（``ConceptRole.DERIVED``），由算法另算。
    """

    START_MINUTE_OF_DAY = "start_minute_of_day"
    DURATION_MINUTES = "duration_minutes"


class BaselineStatistic(str, Enum):
    """常态值是哪种统计：常态开始时刻（``HH:MM``）或常态时长（分钟）。与 ``GradeMeasure`` 一一对应。"""

    USUAL_START = "usual_start"
    USUAL_DURATION = "usual_duration"

    @property
    def measure(self) -> GradeMeasure:
        return (
            GradeMeasure.START_MINUTE_OF_DAY if self is BaselineStatistic.USUAL_START else GradeMeasure.DURATION_MINUTES
        )

    @property
    def label(self) -> str:
        return f"常态{self.quantity}"

    @property
    def quantity(self) -> str:
        return "时刻" if self is BaselineStatistic.USUAL_START else "时长"


class BaselineWindow(str, Enum):
    """常态算的是哪一段历史。**两个窗都要**（2026-09-27 裁定，用户原话"我感觉可能两个都需要"）：

    - ``RECENT`` 近期：回答"今天这一条算不算晚睡"——判据比的就是这个（他现在的习惯）；
    - ``ALL`` 历来：与近期一比就是**漂移**（"他的就寝在往后漂"），那是 profile 的作息骨架信号；
      它不当判据，否则半年前的作息会一直压着今天的判定。

    两个窗要分成两个键，因为常态值会随每条命中记录落盘（``baseline_snapshot``，可重放的关键）：
    一个键指两种算法，历史就没法重放了。
    """

    RECENT = "recent"
    ALL = "all"

    @property
    def label(self) -> str:
        return "近期常态" if self is BaselineWindow.RECENT else "历来常态"


BASELINE_KEY_SEPARATOR = ":"
BASELINE_KEY_SEGMENTS = 3


@dataclass(frozen=True)
class BaselineKey:
    """常态键的结构：哪个行为概念、哪种统计、哪个窗，文本形式 ``<概念>:<统计>:<窗>``
    （例：``就寝:usual_start:recent``）。

    映射器拿到的常态表以这个文本为键；有了结构，算法才能机械地取值（概念 → 它的命中历史，统计 →
    中位时刻 / 中位时长，窗 → 取最近多少天还是全部），而不是猜一串自由文本指的是什么。
    """

    concept: str
    statistic: BaselineStatistic
    window: BaselineWindow = BaselineWindow.RECENT

    def __post_init__(self) -> None:
        concept_identity(self.concept)
        if BASELINE_KEY_SEPARATOR in self.concept:
            raise ConceptError(f"a baseline key's concept must not contain {BASELINE_KEY_SEPARATOR!r}")
        object.__setattr__(self, "statistic", BaselineStatistic(self.statistic))
        object.__setattr__(self, "window", BaselineWindow(self.window))

    @property
    def text(self) -> str:
        return BASELINE_KEY_SEPARATOR.join((self.concept, self.statistic.value, self.window.value))

    @property
    def label(self) -> str:
        return f"{self.concept}的{self.window.label}{self.statistic.quantity}"

    @classmethod
    def parse(cls, text: object) -> BaselineKey:
        if not isinstance(text, str):
            raise ConceptError("a baseline key is text of the form '<concept>:<statistic>:<window>'")
        parts = text.split(BASELINE_KEY_SEPARATOR)
        if len(parts) != BASELINE_KEY_SEGMENTS or not all(parts):
            raise ConceptError(f"baseline key {text!r} is not of the form '<concept>:<statistic>:<window>'")
        concept, statistic, window = parts
        try:
            return cls(concept, BaselineStatistic(statistic), BaselineWindow(window))
        except ValueError as exc:
            raise ConceptError(
                f"baseline key {text!r} names an unknown statistic or window; statistics are "
                f"{[item.value for item in BaselineStatistic]}, windows are {[item.value for item in BaselineWindow]}"
            ) from exc


def parse_baseline_value(measure: GradeMeasure, text: object) -> float | None:
    """常态值的文本形式：开始时刻写 ``HH:MM``，时长写分钟数。解不出来返回 None（材料不合格，不是 0）。"""

    if not isinstance(text, str):
        return None
    value = text.strip()
    if measure is GradeMeasure.START_MINUTE_OF_DAY:
        hour, separator, minute = value.partition(":")
        if not separator or not hour.isdigit() or not minute.isdigit() or len(minute) != 2:
            return None
        hours, minutes = int(hour), int(minute)
        if hours > 23 or minutes > 59:
            return None
        return float(hours * 60 + minutes)
    try:
        number = float(value)
    except ValueError:
        return None
    return number if number >= 0 else None


def widen_windows(spans: Sequence[tuple[int, int]], slack_minutes: int) -> tuple[tuple[int, int], ...]:
    """给一天里按钟面顺序排好、互不重叠的几段时段各加两边的容差，**不越过与邻段的中点**。

    容差是预测树的槽位容差（2026-10-01 用户定复用 ``pool_half_width``）；但两个峰之间只隔 15 分钟时，各展 45 分钟会让
    两个窗口重叠——一条 09:20 的咖啡就会同时归到 08:30 的峰和 09:30 的峰。所以每段最多展到与前后邻段的中点
    （最后一段的"后邻"是次日的第一段，第一段的"前邻"是前一天的最后一段；只有一段时邻居就是它自己隔一天）。
    返回的段可以从负分钟起、或超过 1440（落到前一天 / 次日），与 ``PeakWindow`` 跨午夜的约定一致。
    """

    if isinstance(slack_minutes, bool) or not isinstance(slack_minutes, int) or slack_minutes < 0:
        raise ConceptError("slack_minutes must be a non-negative integer")
    items = [(int(start), int(end)) for start, end in spans]
    if not items:
        return ()
    found: list[tuple[int, int]] = []
    for index, (start, end) in enumerate(items):
        previous_end = items[index - 1][1] - (MINUTES_PER_DAY if index == 0 else 0)
        next_start = items[(index + 1) % len(items)][0] + (MINUTES_PER_DAY if index == len(items) - 1 else 0)
        lower = max(start - slack_minutes, (previous_end + start) // 2)
        upper = min(end + slack_minutes, (end + next_start) // 2)
        found.append((min(lower, start), max(upper, end)))
    return tuple(found)


def circular_offset(observed: float, reference: float) -> float:
    """两个钟面时刻的差，落在 [-720, 720)：02:10 比 23:30 晚 160 分钟，不是早 1280 分钟。

    正好差半天时落在**负侧**（−720）：钟面上"晚 12 小时"与"早 12 小时"本来是同一件事，取哪边都是约定，
    这里跟着取模的结果走、不额外特判。

    整个语义树**只有这一处**环形差：判据的偏移、常态的环形中位数、漂移都用它。读侧不需要它——
    时刻读数比的是绝对时刻（``observed_at − 机会.at``），跨午夜天然就对。
    """

    return ((observed - reference + 720.0) % MINUTES_PER_DAY) - 720.0


@dataclass(frozen=True)
class ConceptSource:
    """概念从哪来；``note`` 记写它的那一版提示词与理由。"""

    origin: ConceptOrigin
    note: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", ConceptOrigin(self.origin))
        if self.note is not None:
            object.__setattr__(self, "note", _single_line(self.note, "concept source note", MAX_NOTE_CHARS))


@dataclass(frozen=True)
class MechanicalRule:
    """算法能直接判的判据：量落在 [lower, upper) 就命中。

    ``relative_to`` 给了常态键时，量 = 观测值 − 那个常态值（开始时刻按环形差），下上界就是偏移量；
    没给时量是绝对值。任一界为 None 表示那一侧不封。
    """

    measure: GradeMeasure
    lower: int | None
    upper: int | None
    relative_to: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "measure", GradeMeasure(self.measure))
        for label in ("lower", "upper"):
            value = getattr(self, label)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ConceptError(f"rule {label} must be an integer or None")
        if self.lower is None and self.upper is None:
            raise ConceptError("a rule needs at least one bound")
        if self.lower is not None and self.upper is not None and self.upper <= self.lower:
            raise ConceptError("rule bounds must be an ascending interval")
        if self.relative_to is not None:
            key = BaselineKey.parse(_single_line(self.relative_to, "rule baseline key", MAX_BASELINE_KEY_CHARS))
            if key.statistic.measure is not self.measure:
                raise ConceptError(
                    f"rule measures {self.measure.value} but its baseline key {key.text!r} is a {key.statistic.value}"
                )
            if key.window is not BaselineWindow.RECENT:
                # 2026-09-27 裁定"判据用近期"：拿历来常态当判据，等于让半年前的作息一直压着今天的判定，
                # 而"近期与历来不一样"这件事本身是漂移信号（profile 的作息骨架），不是判据。
                raise ConceptError(
                    f"a rule compares against the recent baseline; {key.text!r} names the {key.window.value} window"
                )
            object.__setattr__(self, "relative_to", key.text)
        elif self.measure is GradeMeasure.START_MINUTE_OF_DAY:
            for value in (self.lower, self.upper):
                if value is not None and not 0 <= value <= MINUTES_PER_DAY:
                    raise ConceptError("an absolute start-time rule must lie within one day")
        elif self.lower is not None and self.lower < 0:
            raise ConceptError("a duration rule cannot start below zero")

    @property
    def is_relative(self) -> bool:
        return self.relative_to is not None

    def value(self, measures: Mapping[GradeMeasure, float], baseline: Mapping[str, str]) -> float | None:
        """规则比的那个量；材料缺（量没给、常态没给或解不出）返回 None。"""

        observed = measures.get(self.measure)
        if observed is None:
            return None
        if self.relative_to is None:
            return observed
        base = parse_baseline_value(self.measure, baseline.get(self.relative_to))
        if base is None:
            return None
        if self.measure is GradeMeasure.START_MINUTE_OF_DAY:
            return circular_offset(observed, base)
        return observed - base

    def covers(self, value: float) -> bool:
        return (self.lower is None or value >= self.lower) and (self.upper is None or value < self.upper)

    def criterion(self) -> str:
        quantity = _quantity_label(self.measure, relative=self.is_relative, key=self.relative_to)
        return f"{quantity} {_interval_label(self.measure, self.lower, self.upper, relative=self.is_relative)}"


@dataclass(frozen=True)
class ConceptGrade:
    """一档 = 名字 + 量 + [下界, 上界)。

    绝对的开始时刻区间下界大于上界即跨午夜（22:00–02:00）；``relative`` 为真时界是相对常态的偏移分钟数。
    """

    name: str
    measure: GradeMeasure
    lower: int
    upper: int
    relative: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identity_token(self.name, "grade name", MAX_GRADE_NAME_CHARS))
        object.__setattr__(self, "measure", GradeMeasure(self.measure))
        for label in ("lower", "upper"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConceptError(f"grade {label} must be an integer")
        if not isinstance(self.relative, bool):
            raise ConceptError("grade relative must be a boolean")
        if self.lower == self.upper:
            raise ConceptError("grade bounds must not be equal")
        if self.relative:
            if self.upper <= self.lower:
                raise ConceptError("a relative grade must be an ascending interval")
        elif self.measure is GradeMeasure.START_MINUTE_OF_DAY:
            if not (0 <= self.lower < MINUTES_PER_DAY and 0 < self.upper <= MINUTES_PER_DAY):
                raise ConceptError("start_minute_of_day bounds must lie within one day")
        elif self.lower < 0 or self.upper <= self.lower:
            raise ConceptError("duration bounds must be a non-negative ascending interval")

    @property
    def wraps_midnight(self) -> bool:
        return not self.relative and self.measure is GradeMeasure.START_MINUTE_OF_DAY and self.lower > self.upper

    def covers(self, value: float) -> bool:
        if self.wraps_midnight:
            return value >= self.lower or value < self.upper
        return self.lower <= value < self.upper

    def segments(self) -> tuple[tuple[int, int], ...]:
        """区间拆成不跨午夜的段，用来核对两档不重叠。"""

        if self.wraps_midnight:
            return ((self.lower, MINUTES_PER_DAY), (0, self.upper))
        return ((self.lower, self.upper),)

    def criterion(self) -> str:
        """给人和模型读的一句话；从数值规则渲染出来，不另存自由文本。"""

        quantity = _quantity_label(self.measure, relative=self.relative, key=None)
        return f"{quantity} {_interval_label(self.measure, self.lower, self.upper, relative=self.relative)}"


def concept_identity(name: object) -> str:
    """概念名的规范身份：与行为树叶名同一套做法（NFC + casefold，拒隐藏名与 ``.md`` 后缀），并拒分隔符。"""

    try:
        return canonical_path_identity(
            _identity_token(semantic_name(name, "concept name"), "concept name", None), "concept name"
        )
    except (TypeError, ValueError) as exc:
        raise ConceptError(str(exc)) from exc


@dataclass(frozen=True)
class MechanicalDecision:
    """算法对一条 occurrence 的裁定：``hit`` 为 None 表示材料不够、没判。"""

    hit: bool | None
    value: float | None


@dataclass(frozen=True)
class ConceptDefinition:
    name: str
    definition: str
    role: ConceptRole
    source: ConceptSource
    created_at: datetime
    #: 行为概念与词表类的关系；情境概念为 None。
    kind: ConceptKind | None = None
    #: 基础概念：它自己那一个类；细分概念：源类；汇总概念：成员类（≥2）。情境概念为空。
    classes: tuple[str, ...] = ()
    #: 行为概念所在的 lane（组成只在一条 lane 内）；情境概念为 None。
    lane: str | None = None
    #: 基础概念的显示名（类名）；它的 ``name`` 是类编号，类改名身份不变。
    title: str | None = None
    #: 基础概念的类在词表里停用了（被并进别的类）：它不再命中，历史命中留着读。
    retired: bool = False
    grades: tuple[ConceptGrade, ...] = ()
    rule: MechanicalRule | None = None
    context: ContextScope = ContextScope.OCCURRENCE
    baseline_keys: tuple[str, ...] = ()
    #: 情境概念怎么算（只有情境概念能带）。没带的情境概念永远不会命中——「出差中」要等事实门接上
    #: 真实数据源才算得出，这是事实不是缺陷（不编关键词规则去凑）。
    situation: SituationRule | None = None

    def __post_init__(self) -> None:
        identity = concept_identity(self.name)
        object.__setattr__(
            self, "definition", _single_line(self.definition, "concept definition", MAX_DEFINITION_CHARS)
        )
        object.__setattr__(self, "role", ConceptRole(self.role))
        if not isinstance(self.source, ConceptSource):
            raise ConceptError("concept source must be a ConceptSource")
        if not isinstance(self.created_at, datetime) or self.created_at.utcoffset() is None:
            raise ConceptError("concept created_at must be a timezone-aware datetime")
        object.__setattr__(self, "created_at", self.created_at.astimezone(UTC))
        object.__setattr__(self, "context", ContextScope(self.context))
        self._check_class_link(identity)
        if self.rule is not None and not isinstance(self.rule, MechanicalRule):
            raise ConceptError("rule must be a MechanicalRule")
        if self.rule is not None and self.context is not ContextScope.OCCURRENCE:
            raise ConceptError("a mechanical rule is computed from the occurrence alone; it takes no other material")
        object.__setattr__(self, "baseline_keys", self._checked_baseline_keys())
        self._check_situation()
        object.__setattr__(self, "grades", self._checked_grades())

    def _check_class_link(self, identity: str) -> None:
        """行为概念必须挂类且三种关系各守各的形状；情境概念一样都不挂。"""

        classes = tuple(_single_line(item, "concept class", MAX_NOTE_CHARS) for item in self.classes)
        object.__setattr__(self, "classes", classes)
        if self.title is not None:
            object.__setattr__(self, "title", _single_line(self.title, "concept title", MAX_DEFINITION_CHARS))
        if not isinstance(self.retired, bool):
            raise ConceptError("retired must be a boolean")
        if self.retired and self.kind is not ConceptKind.BASE:
            raise ConceptError("only a base concept retires with its class")
        if self.role.is_situation:
            if self.kind is not None or classes or self.lane is not None or self.title is not None:
                raise ConceptError("a situation concept is not tied to vocabulary classes")
            return
        if self.kind is None:
            raise ConceptError("a behaviour concept says how it relates to the vocabulary (base / refinement / group)")
        object.__setattr__(self, "kind", ConceptKind(self.kind))
        object.__setattr__(self, "lane", _single_line(self.lane, "concept lane", MAX_NOTE_CHARS))
        if len(set(classes)) != len(classes):
            raise ConceptError("a concept names each class once")
        if self.kind is ConceptKind.BASE:
            if len(classes) != 1 or identity != concept_identity(classes[0]) or self.title is None:
                raise ConceptError("a base concept is named by its one class id and titled with the class name")
            if self.rule is not None or self.grades or self.context is not ContextScope.OCCURRENCE:
                raise ConceptError("a base concept is the class itself; it carries no rule, grade or material")
        elif self.kind is ConceptKind.REFINEMENT:
            if len(classes) != 1 or self.title is not None:
                raise ConceptError("a refinement concept names exactly one source class")
        else:
            if len(classes) < 2 or self.title is not None:
                raise ConceptError("a group concept names at least two member classes")
            if self.rule is not None or self.grades or self.context is not ContextScope.OCCURRENCE:
                raise ConceptError("a group concept is never judged; it carries no rule, grade or material")

    def _checked_baseline_keys(self) -> tuple[str, ...]:
        keys = tuple(self.baseline_keys)
        if len(keys) > MAX_BASELINE_KEYS:
            raise ConceptError(f"a concept names at most {MAX_BASELINE_KEYS} baseline keys")
        cleaned = tuple(
            BaselineKey.parse(_single_line(key, "baseline key", MAX_BASELINE_KEY_CHARS)).text for key in keys
        )
        if len(set(cleaned)) != len(cleaned):
            raise ConceptError("baseline keys must be distinct")
        if self.rule is not None and self.rule.relative_to is not None and cleaned:
            raise ConceptError("a relative rule already names its baseline key; do not repeat it in baseline_keys")
        return cleaned

    def _check_situation(self) -> None:
        if self.situation is None and self.role is ConceptRole.DERIVED:
            # 派生的定义就是"算法从历史命中算"——没有算法说明的派生概念永远不会命中，它不是派生，是空壳。
            # 其他情境（状态/对象/日型）可以先没有：「出差中」要等事实门接上数据源。
            raise ConceptError("a derived situation concept carries the rule the algorithm computes it by")
        if self.situation is None:
            return
        if not isinstance(self.situation, SituationRule):
            raise ConceptError("situation must be a SituationRule")
        if self.role.is_behavior:
            # 行为概念由词表的类与映射判，不由"那一刻外面什么样"判。
            raise ConceptError("only a situation concept carries a situation rule")
        if self.situation.concept is not None:
            try:
                concept_identity(self.situation.concept)
            except ConceptError as exc:
                raise ConceptError(f"the situation rule of {self.name!r} names an unusable concept: {exc}") from exc

    def _checked_grades(self) -> tuple[ConceptGrade, ...]:
        grades = tuple(self.grades)
        if not grades:
            return grades
        if not MIN_GRADES <= len(grades) <= MAX_GRADES:
            raise ConceptError(f"a graded concept carries {MIN_GRADES}–{MAX_GRADES} grades")
        if any(not isinstance(grade, ConceptGrade) for grade in grades):
            raise ConceptError("grades must be ConceptGrade values")
        if len({grade.measure for grade in grades}) != 1:
            raise ConceptError("all grades of one concept must use the same measure")
        if len({grade.relative for grade in grades}) != 1:
            raise ConceptError("grades of one concept are all relative or all absolute")
        if len({canonical_text_identity(grade.name, "grade name") for grade in grades}) != len(grades):
            raise ConceptError("grade names must be distinct")
        _require_disjoint(grades)
        if self.rule is not None:
            if grades[0].measure is not self.rule.measure:
                raise ConceptError("grades must measure the same quantity as the rule")
            if grades[0].relative != self.rule.is_relative:
                raise ConceptError(
                    "grades of a relative rule are relative offsets; of an absolute rule, absolute values"
                )
        elif grades[0].relative:
            raise ConceptError("relative grades need a relative rule to say what they are relative to")
        return grades

    @property
    def identity(self) -> str:
        return concept_identity(self.name)

    @property
    def label(self) -> str:
        """给人和模型看的名字：基础概念是类名，其余是自己的名字。"""

        return self.title if self.title is not None else self.name

    @property
    def watched_identity(self) -> str | None:
        """情境说明盯着的那个概念的身份（派生型才有）。"""

        if self.situation is None or self.situation.concept is None:
            return None
        return concept_identity(self.situation.concept)

    @property
    def is_mechanical(self) -> bool:
        return self.rule is not None

    @property
    def is_judged(self) -> bool:
        """映射时要不要判：只有细分概念要判（基础概念看编号，汇总概念读时聚合，情境概念算法另算）。"""

        return self.kind is ConceptKind.REFINEMENT

    @property
    def required_baseline_keys(self) -> tuple[str, ...]:
        """映射这条概念时必须给到的常态键：规则引用的 + 判据句引用的。"""

        if self.rule is not None and self.rule.relative_to is not None:
            return (self.rule.relative_to, *self.baseline_keys)
        return self.baseline_keys

    @property
    def grade_measure(self) -> GradeMeasure | None:
        return self.grades[0].measure if self.grades else None

    @property
    def fingerprint(self) -> str:
        """判据的内容指纹：会改变"什么算命中"的那些字段。来源、时间与显示名不在里面。"""

        return canonical_digest(
            {
                "identity": self.identity,
                "definition": self.definition,
                "role": self.role.value,
                "kind": None if self.kind is None else self.kind.value,
                "classes": list(self.classes),
                "lane": self.lane,
                "grades": [
                    {
                        "name": g.name,
                        "measure": g.measure.value,
                        "lower": g.lower,
                        "upper": g.upper,
                        "relative": g.relative,
                    }
                    for g in self.grades
                ],
                "rule": None
                if self.rule is None
                else {
                    "measure": self.rule.measure.value,
                    "lower": self.rule.lower,
                    "upper": self.rule.upper,
                    "relative_to": self.rule.relative_to,
                },
                "context": self.context.value,
                "baseline_keys": list(self.baseline_keys),
                "situation": None if self.situation is None else self.situation.payload(),
            }
        )[:16]

    def decide(self, measures: Mapping[GradeMeasure, float], baseline: Mapping[str, str]) -> MechanicalDecision:
        """机械判据的裁定；没有规则的概念不能在这里判（那是模型的活）。"""

        if self.rule is None:
            raise ConceptError(f"concept {self.name!r} has no mechanical rule")
        value = self.rule.value(measures, baseline)
        if value is None:
            return MechanicalDecision(hit=None, value=None)
        return MechanicalDecision(hit=self.rule.covers(value), value=value)

    def grade_for(
        self, measures: Mapping[GradeMeasure, float], baseline: Mapping[str, str] | None = None
    ) -> str | None:
        """按数值规则定档；没有档、量缺失或落在所有档之外都返回 None（命中仍算命中，只是不带档）。

        有规则的概念比的是规则算出的量（相对规则就是偏移量）。
        """

        measure = self.grade_measure
        if measure is None:
            return None
        if self.rule is not None:
            value = self.rule.value(measures, baseline or {})
        else:
            value = measures.get(measure)
        if value is None:
            return None
        for grade in self.grades:
            if grade.covers(value):
                return grade.name
        return None


class ConceptSet(Mapping[str, ConceptDefinition]):
    """一组自洽的概念：身份不重、一个类最多一个基础概念、情境盯的概念都在。键是规范身份。

    层级只有一种：汇总概念 ⊃ 它成员类的基础概念。细分概念不进层级——「晚睡」与「睡觉」是同一条记录的两种说法，
    「晚睡 → 睡觉」（当晚补觉）量的是另一条记录，不是重言。
    """

    def __init__(self, definitions: Iterable[ConceptDefinition]) -> None:
        resolved: dict[str, ConceptDefinition] = {}
        for definition in definitions:
            if not isinstance(definition, ConceptDefinition):
                raise ConceptError("a concept set holds ConceptDefinition values")
            if definition.identity in resolved:
                raise ConceptError(f"concept {definition.name!r} appears twice in the set")
            resolved[definition.identity] = definition
        base: dict[str, str] = {}
        for identity, definition in resolved.items():
            if definition.kind is ConceptKind.BASE:
                base[definition.classes[0]] = identity
            watched = definition.watched_identity
            if watched is not None and watched not in resolved:
                raise ConceptError(
                    f"the situation rule of {definition.name!r} watches {definition.situation.concept!r}, "  # type: ignore[union-attr]
                    "which is not in the set"
                )
        self._items: Mapping[str, ConceptDefinition] = MappingProxyType(resolved)
        self._base: Mapping[str, str] = MappingProxyType(base)
        for lane in {item.lane for item in resolved.values() if item.lane is not None}:
            self._require_unique_labels(lane)

    def _require_unique_labels(self, lane: str) -> None:
        """给模型看的是名字不是编号（裁定 21-1）：同一条 lane 能一起出现的概念（这条 lane 的 + 不属于任何 lane 的）名字不许重。"""

        seen: dict[str, str] = {}
        for identity, item in self._items.items():
            if self.lane_of(identity) not in (lane, None):
                continue
            key = canonical_text_identity(item.label, "concept label")
            if key in seen:
                raise ConceptError(
                    f"concepts {seen[key]!r} and {item.name!r} both show as {item.label!r} in lane {lane!r}"
                )
            seen[key] = item.name

    def __getitem__(self, key: str) -> ConceptDefinition:
        try:
            identity = concept_identity(key)
        except ConceptError as exc:
            raise ConceptLookupError(str(exc)) from exc
        try:
            return self._items[identity]
        except KeyError:
            raise ConceptLookupError(f"concept {key!r} is not in the set") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: object) -> bool:
        try:
            return concept_identity(key) in self._items
        except ConceptError:
            return False

    @property
    def fingerprint(self) -> str:
        """要判的概念（细分概念与情境概念）的指纹；映射口径带着它，定义一改，已映射的天就能被认出来要重做。

        基础概念与汇总概念不判，不进指纹：类改名、汇总改成员都不必重映射。
        """

        judged = [
            item.fingerprint
            for key, item in sorted(self._items.items())
            if item.kind is not ConceptKind.BASE and item.kind is not ConceptKind.GROUP
        ]
        return canonical_digest(judged)[:16]

    def behaviors(self) -> tuple[str, ...]:
        return tuple(identity for identity, item in self._items.items() if item.role.is_behavior)

    def situations(self) -> tuple[str, ...]:
        return tuple(identity for identity, item in self._items.items() if item.role.is_situation)

    def base_for(self, class_id: str) -> str | None:
        """一个类的基础概念的身份；这个类还没有基础概念时为 None。"""

        return self._base.get(class_id)

    def lane_of(self, name: str) -> str | None:
        """概念属于哪条 lane：行为概念是自己的 lane；派生情境跟它盯的那个概念走（「连续研发中」盯研发实现 → 会话 lane）；
        日历、对象这类情境不属于任何一条 lane（None，两边都能用）。不在集里的也给 None。"""

        seen: set[str] = set()
        current = name
        while current in self and concept_identity(current) not in seen:
            seen.add(concept_identity(current))
            item = self[current]
            if item.role.is_behavior:
                return item.lane
            if item.watched_identity is None:
                return None
            current = item.watched_identity
        return None

    def relatable_to(self, consequent: str) -> tuple[str, ...]:
        """能和这个后件连成一条关系的概念（不含它自己）：同一条 lane 的行为概念与派生情境 + 不属于任何 lane 的情境。

        关系只在一条 lane 内（裁定 20 第 1 条）：两两检验的对从这里取。
        """

        own = self[consequent].identity
        lane = self.lane_of(own)
        return tuple(
            identity for identity in sorted(self._items) if identity != own and self.lane_of(identity) in (lane, None)
        )

    def label_of(self, name: str) -> str:
        """概念 → 给人和模型看的名字；不在集里的原样返回（读侧拿它渲染，概念删了的旧记录照样印得出）。"""

        return self[name].label if name in self else name

    def class_label(self, class_id: str) -> str:
        """类编号 → 给人和模型看的类名（它的基础概念的显示名）；还没同步到的类原样给编号。"""

        base = self._base.get(class_id)
        return class_id if base is None else self._items[base].label

    def refinements_on(self, class_id: str) -> tuple[str, ...]:
        """挂在这个类上的细分概念——映射一条这个类的记录时，只判这些。"""

        return tuple(
            identity
            for identity, item in sorted(self._items.items())
            if item.kind is ConceptKind.REFINEMENT and item.classes[0] == class_id
        )

    def classes_of(self, name: str) -> tuple[str, ...]:
        """概念 → 它的曲线由哪几个类组成（桥用）；情境概念为空。"""

        return self[name].classes

    def ancestors(self, name: str) -> tuple[str, ...]:
        """基础概念的祖先 = 成员里有它那个类的汇总概念；其余概念没有祖先。"""

        item = self[name]
        if item.kind is not ConceptKind.BASE:
            return ()
        own = item.classes[0]
        return tuple(
            identity
            for identity, other in sorted(self._items.items())
            if other.kind is ConceptKind.GROUP and own in other.classes
        )

    def overlaps(self, first: str, second: str) -> bool:
        """两个行为概念会不会读同一批记录：基础 / 汇总概念的类有交集（「修改代码」与「写代码」）。

        这样的两个概念一个当前因、一个当后果是重言，两两检验里机械排除。细分概念不算：「晚睡 → 睡觉」量的是另一条记录（当晚补觉）。
        """

        a, b = self[first], self[second]
        pooled = (ConceptKind.BASE, ConceptKind.GROUP)
        return a.kind in pooled and b.kind in pooled and bool(set(a.classes) & set(b.classes))

    def with_ancestors(self, names: Iterable[str]) -> frozenset[str]:
        """读侧聚合：一组命中连同它们全部祖先的身份。"""

        closed: set[str] = set()
        for name in names:
            identity = self[name].identity
            closed.add(identity)
            closed.update(self.ancestors(identity))
        return frozenset(closed)


def _single_line(value: object, label: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ConceptError(f"{label} must be text")
    if not value.strip() or value != value.strip():
        raise ConceptError(f"{label} must be non-empty text without surrounding whitespace")
    if "\n" in value or "\r" in value or any(not character.isprintable() for character in value):
        raise ConceptError(f"{label} must be a single printable line")
    if len(value) > limit:
        raise ConceptError(f"{label} exceeds {limit} characters")
    return value


def _identity_token(value: object, label: str, limit: int | None) -> str:
    """会被拼进路径身份的名字：单行、路径安全、不含分隔符。"""

    text = _single_line(value, label, limit if limit is not None else MAX_DEFINITION_CHARS)
    try:
        require_safe_path_segment(text, label)
    except ValueError as exc:
        raise ConceptError(str(exc)) from exc
    if any(separator in text for separator in IDENTITY_SEPARATORS):
        raise ConceptError(f"{label} must not contain any of {IDENTITY_SEPARATORS!r}")
    return text


def _require_disjoint(grades: tuple[ConceptGrade, ...]) -> None:
    segments = sorted(segment for grade in grades for segment in grade.segments())
    for (_, previous_end), (next_start, _) in zip(segments, segments[1:], strict=False):
        if next_start < previous_end:
            raise ConceptError("grades of one concept must not overlap")


def _quantity_label(measure: GradeMeasure, *, relative: bool, key: str | None) -> str:
    base = "开始时刻" if measure is GradeMeasure.START_MINUTE_OF_DAY else "时长"
    if not relative:
        return base
    return f"{base}相对常态{'（' + key + '）' if key else ''}的偏移"


def _interval_label(measure: GradeMeasure, lower: int | None, upper: int | None, *, relative: bool) -> str:
    if not relative and measure is GradeMeasure.START_MINUTE_OF_DAY:
        return f"{_clock(lower)}–{_clock(upper)}"
    left = "…" if lower is None else f"{lower:+d}" if relative else str(lower)
    right = "…" if upper is None else f"{upper:+d}" if relative else str(upper)
    return f"{left}–{right} 分钟"


def _clock(minute_of_day: int | None) -> str:
    if minute_of_day is None:
        return "…"
    hour, minute = divmod(minute_of_day % MINUTES_PER_DAY, 60)
    return f"{hour:02d}:{minute:02d}"


__all__ = [
    "BASELINE_KEY_SEGMENTS",
    "BASELINE_KEY_SEPARATOR",
    "IDENTITY_SEPARATORS",
    "MAX_DEFINITION_CHARS",
    "MAX_GRADES",
    "MIN_GRADES",
    "BaselineKey",
    "BaselineStatistic",
    "BaselineWindow",
    "ConceptDefinition",
    "ConceptError",
    "ConceptGrade",
    "ConceptLookupError",
    "ConceptOrigin",
    "ConceptRole",
    "ConceptSet",
    "ConceptSource",
    "ContextScope",
    "GradeMeasure",
    "MechanicalDecision",
    "MechanicalRule",
    "SituationBasis",
    "SituationError",
    "SituationRule",
    "circular_offset",
    "widen_windows",
    "concept_identity",
    "parse_baseline_value",
]
