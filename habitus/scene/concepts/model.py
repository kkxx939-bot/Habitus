"""概念定义：语义树的词汇层，一个概念一份。

概念由基准（LLM）写、按残差升级增补；这里只定它长什么样、身份怎么算、一组概念怎样才自洽。

- **判据分两半**（2026-09-26 裁定）。数值的那一半（"晚于常态两小时以上"）写成 ``MechanicalRule``，由算法算、
  模型从不碰数字；语义的那一半（"这是入睡"）留在判据句里，由模型答是/否。带规则的概念先算数值，数值不满足
  就不问模型；满足了才问语义门——一碗 07:00 的面比常态就寝晚 7.5 小时，规则会通过，"它不是入睡"只有模型
  答得了。写不成数值的概念（"参与一场球类运动"）只有语义那一半。判据句要写到对着材料能答是/否。
- **判据可以引用别的事件**："起床后两小时内的第一次进食"要看当天时间线，概念用 ``context=DAY`` 声明；
  引用"常态"的用 ``baseline_keys`` 声明要哪几个常态值。声明了的材料映射时没给到，那一条**不判**
  （记成未决），不让模型替我们答成 false。
- **只有行为概念能当后件**，情境概念（状态 / 对象 / 日型 / 派生）只做前件集合的元素。
- **层级是树不是图**，单一上级。**祖先不参与命中**：映射只判叶子行为概念，「打球」命中了算不算「运动」
  由读侧沿 parent 链聚合（层级改了不用重算任何命中）。
- **没有版本**：定义变了就整个重算它的命中和账。
- **档由算法判**：LLM 只答是不是，几档按数值规则机械定。有 ``MechanicalRule`` 的概念，档比的是规则算出的
  那个量（相对常态的偏移就按偏移分档）。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
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
#: 概念身份、档名会被拼进假设的叶名（``<前件@档+前件>--<方面>``），这三个分隔符不许出现在它们里面，
#: 否则两条不同的假设能拼出同一个文件名（实测：概念名 ``出差中+晚睡`` 与集合 ``{出差中, 晚睡}``）。
IDENTITY_SEPARATORS = ("+", "@", "--")


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
    BASELINE = "baseline"
    RESIDUE = "residue"


class ContextScope(str, Enum):
    """判据要看多大范围的材料：只看这一条 occurrence，还是要看它所在那一天的时间线。"""

    OCCURRENCE = "occurrence"
    DAY = "day"


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
        return GradeMeasure.START_MINUTE_OF_DAY if self is BaselineStatistic.USUAL_START else GradeMeasure.DURATION_MINUTES

    @property
    def label(self) -> str:
        return f"常态{self.quantity}"

    @property
    def quantity(self) -> str:
        return "时刻" if self is BaselineStatistic.USUAL_START else "时长"


class BaselineWindow(str, Enum):
    """常态算的是哪一段历史。**两个窗都要**（2026-09-27 裁定，用户原话"我感觉可能两个都需要"）：

    - ``RECENT`` 近期：回答"今天这一条算不算晚睡"——判据比的就是这个（他现在的习惯）；
    - ``ALL`` 历来：与近期一比就是**漂移**（"他的就寝在往后漂"），那是 profile 的作息骨架信号，
      也是闭环的第二个触发源；它不当判据，否则半年前的作息会一直压着今天的判定。

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
    """概念从哪来。残差升级来的必须记下认领的是哪个 kind（``kind_token``）——残差视图靶这个把已升级的 kind 排除，
    否则升级前那些 ``hits=[]`` 的记录会让同一个 kind 每晚再报一次"可升级"（新概念不回填历史命中）。"""

    origin: ConceptOrigin
    note: str | None = None
    kind_token: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", ConceptOrigin(self.origin))
        if self.note is not None:
            object.__setattr__(self, "note", _single_line(self.note, "concept source note", MAX_NOTE_CHARS))
        if self.kind_token is not None:
            object.__setattr__(self, "kind_token", _single_line(self.kind_token, "concept source kind token", MAX_NOTE_CHARS))
        if (self.origin is ConceptOrigin.RESIDUE) != (self.kind_token is not None):
            raise ConceptError("a residue-upgraded concept names the kind it claims, and only such a concept does")


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
                raise ConceptError(f"rule measures {self.measure.value} but its baseline key {key.text!r} is a {key.statistic.value}")
            if key.window is not BaselineWindow.RECENT:
                # 2026-09-27 裁定"判据用近期"：拿历来常态当判据，等于让半年前的作息一直压着今天的判定，
                # 而"近期与历来不一样"这件事本身是漂移信号（profile 的作息骨架），不是判据。
                raise ConceptError(f"a rule compares against the recent baseline; {key.text!r} names the {key.window.value} window")
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
        return canonical_path_identity(_identity_token(semantic_name(name, "concept name"), "concept name", None), "concept name")
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
    parent: str | None = None
    grades: tuple[ConceptGrade, ...] = ()
    rule: MechanicalRule | None = None
    context: ContextScope = ContextScope.OCCURRENCE
    baseline_keys: tuple[str, ...] = ()
    #: 情境概念怎么算（只有情境概念能带）。没带的情境概念永远不会命中——「出差中」要等事实门接上
    #: 真实数据源才算得出，这是事实不是缺陷（不编关键词规则去凑）。
    situation: SituationRule | None = None

    def __post_init__(self) -> None:
        identity = concept_identity(self.name)
        object.__setattr__(self, "definition", _single_line(self.definition, "concept definition", MAX_DEFINITION_CHARS))
        object.__setattr__(self, "role", ConceptRole(self.role))
        if not isinstance(self.source, ConceptSource):
            raise ConceptError("concept source must be a ConceptSource")
        if not isinstance(self.created_at, datetime) or self.created_at.utcoffset() is None:
            raise ConceptError("concept created_at must be a timezone-aware datetime")
        object.__setattr__(self, "created_at", self.created_at.astimezone(UTC))
        if self.parent is not None and concept_identity(self.parent) == identity:
            raise ConceptError("a concept cannot be its own parent")
        object.__setattr__(self, "context", ContextScope(self.context))
        if self.rule is not None and not isinstance(self.rule, MechanicalRule):
            raise ConceptError("rule must be a MechanicalRule")
        if self.rule is not None and self.context is ContextScope.DAY:
            raise ConceptError("a mechanical rule is computed from the occurrence alone; it does not take day context")
        keys = tuple(self.baseline_keys)
        if len(keys) > MAX_BASELINE_KEYS:
            raise ConceptError(f"a concept names at most {MAX_BASELINE_KEYS} baseline keys")
        cleaned_keys = tuple(BaselineKey.parse(_single_line(key, "baseline key", MAX_BASELINE_KEY_CHARS)).text for key in keys)
        if len(set(cleaned_keys)) != len(cleaned_keys):
            raise ConceptError("baseline keys must be distinct")
        if self.rule is not None and self.rule.relative_to is not None and cleaned_keys:
            raise ConceptError("a relative rule already names its baseline key; do not repeat it in baseline_keys")
        object.__setattr__(self, "baseline_keys", cleaned_keys)
        if self.situation is None and self.role is ConceptRole.DERIVED:
            # 派生的定义就是"算法从历史命中算"——没有算法说明的派生概念永远不会命中，它不是派生，是空壳。
            # 其他情境（状态/对象/日型）可以先没有：「出差中」要等事实门接上数据源。
            raise ConceptError("a derived situation concept carries the rule the algorithm computes it by")
        if self.situation is not None:
            if not isinstance(self.situation, SituationRule):
                raise ConceptError("situation must be a SituationRule")
            if self.role.is_behavior:
                # 行为概念由映射器判（判据句 + 数值规则），不由"那一刻外面什么样"判。
                raise ConceptError("only a situation concept carries a situation rule")
            if self.situation.concept is not None:
                try:
                    concept_identity(self.situation.concept)
                except ConceptError as exc:
                    raise ConceptError(f"the situation rule of {self.name!r} names an unusable concept: {exc}") from exc
        grades = tuple(self.grades)
        if grades:
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
                    raise ConceptError("grades of a relative rule are relative offsets; of an absolute rule, absolute values")
            elif grades[0].relative:
                raise ConceptError("relative grades need a relative rule to say what they are relative to")
        object.__setattr__(self, "grades", grades)

    @property
    def identity(self) -> str:
        return concept_identity(self.name)

    @property
    def parent_identity(self) -> str | None:
        return None if self.parent is None else concept_identity(self.parent)

    @property
    def watched_identity(self) -> str | None:
        """情境说明盯着的那个概念的身份（承诺型与派生型才有）。"""

        if self.situation is None or self.situation.concept is None:
            return None
        return concept_identity(self.situation.concept)

    @property
    def is_mechanical(self) -> bool:
        return self.rule is not None

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
        """判据的内容指纹：会改变"什么算命中"的那些字段。来源与时间不在里面。"""

        return canonical_digest(
            {
                "identity": self.identity,
                "definition": self.definition,
                "role": self.role.value,
                "parent": self.parent_identity,
                "grades": [
                    {"name": g.name, "measure": g.measure.value, "lower": g.lower, "upper": g.upper, "relative": g.relative}
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

    def grade_for(self, measures: Mapping[GradeMeasure, float], baseline: Mapping[str, str] | None = None) -> str | None:
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
    """一组自洽的概念：身份不重、上级都在、不成环。键是规范身份。"""

    def __init__(self, definitions: Iterable[ConceptDefinition]) -> None:
        resolved: dict[str, ConceptDefinition] = {}
        for definition in definitions:
            if not isinstance(definition, ConceptDefinition):
                raise ConceptError("a concept set holds ConceptDefinition values")
            if definition.identity in resolved:
                raise ConceptError(f"concept {definition.name!r} appears twice in the set")
            resolved[definition.identity] = definition
        for definition in resolved.values():
            parent = definition.parent_identity
            if parent is not None and parent not in resolved:
                raise ConceptError(f"concept {definition.name!r} names an unknown parent {definition.parent!r}")
            watched = definition.watched_identity
            if watched is not None and watched not in resolved:
                raise ConceptError(
                    f"the situation rule of {definition.name!r} watches {definition.situation.concept!r}, "  # type: ignore[union-attr]
                    "which is not in the set"
                )
        self._items: Mapping[str, ConceptDefinition] = MappingProxyType(resolved)
        children: dict[str, list[str]] = {identity: [] for identity in resolved}
        for identity, definition in resolved.items():
            if definition.parent_identity is not None:
                children[definition.parent_identity].append(identity)
        self._children: Mapping[str, tuple[str, ...]] = MappingProxyType({k: tuple(v) for k, v in children.items()})
        for identity in resolved:
            self.ancestors(identity)  # 成环在这里抛

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
        """整个概念集的指纹；映射口径带着它，定义一改，已映射的天就能被认出来要重做。"""

        return canonical_digest([self._items[identity].fingerprint for identity in sorted(self._items)])[:16]

    def behaviors(self) -> tuple[str, ...]:
        return tuple(identity for identity, item in self._items.items() if item.role.is_behavior)

    def behavior_leaves(self) -> tuple[str, ...]:
        """没有子概念的行为概念——映射只判这些；有子概念的由读侧沿 parent 链聚合。"""

        return tuple(identity for identity in self.behaviors() if not self._children[identity])

    def situations(self) -> tuple[str, ...]:
        return tuple(identity for identity, item in self._items.items() if item.role.is_situation)

    def claimed_kinds(self) -> frozenset[str]:
        """已被残差升级认领的 kind：残差视图不再把它们当候选。"""

        return frozenset(item.source.kind_token for item in self._items.values() if item.source.kind_token is not None)

    def ancestors(self, name: str) -> tuple[str, ...]:
        """从直接上级到根，按序；走回自己就是环。"""

        start = self[name].identity
        chain: list[str] = []
        current = self._items[start].parent_identity
        while current is not None:
            if current == start or current in chain:
                raise ConceptError(f"concept hierarchy has a cycle through {name!r}")
            chain.append(current)
            current = self._items[current].parent_identity
        return tuple(chain)

    def children(self, name: str) -> tuple[str, ...]:
        return self._children[self[name].identity]

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return self[ancestor].identity in self.ancestors(descendant)

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
    "concept_identity",
    "parse_baseline_value",
]
