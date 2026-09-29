"""假设：一条待核对的关系，基准（LLM）写、账本核对。

一条假设 = {前件情境集合} → 后件 · 方面。四属性里**只有方向与"第几次机会"由基准写**：

- **方向**  ↑ / ↓ —— 先验只给方向，不给大小（"强/中/弱"翻成数值都是拍的）；
- **第几次机会**（``expected_at``，2026-09-26 裁定，取代"锚 + 多少小时"的窗档）：后件自己有节律（一天一次的
  早餐一天一个机会，一天三杯的咖啡一天三个），基准写的是"我猜影响落在后件的第几次机会上"——晚睡→早餐是
  第 1 次（下一个早餐机会），晚睡→就寝的补偿也是第 1 次（02:10 的锚之后第一个就寝机会就是**当晚** 23:30）；
  ``None`` = 说不准。"次日 / 后天 / 下周"离锚多少小时随锚的时辰变（同一件"当晚补偿就寝"，02:10 的锚离它
  20.5 小时、22:00 的锚离它 1.5 小时），按机会数就不会。**这个猜只用来读，不用来切数据**：账本把后件第一次
  到来之前的每个机会都记下来，落在第几次是读出来的。
- **类型**  使能 / 促进 / 抑制只是基准的猜测（``type_prior``），最终由账读出；概率方面必须给，时刻/次数方面
  可以不给（"起床推后"不是抑制也不是促进）。
- **强度**  不写。

假设由此**分两型**（2026-09-27 裁定，``expected_at`` 有没有数就是标记）：

- **节律型** ``expected_at = 1, 2, …``：后件天天有机会（树上有峰）。等过 ``censor_after`` 个已观测机会还没来
  就右删失，读数是"第 expected_at 次机会的实际率 vs 树上概率"。
- **无节律型** ``expected_at = None``（约球→打球、挂号→就诊、买菜→做饭）：后件没节律或很低频，**不数机会、
  没有时效**。承诺只有两种结法——后件来了（记隔了多久），或 ``released_by`` 里的概念命中（前提作废，再次挂号
  取代上一次）。没写 ``released_by`` 就一直立着，读数老实写"立了 300 天没兑现"。读数只有兑现率与兑现间隔，
  不读"比平时多几成"、不读类型。理由（用户原话）："很多行为他是没有机会时效的，或者说一个行为的影响在很
  后面……低频重要的事情，你永远找不到他的影响行为和被影响行为。"

锚在前件的**开始时刻**（树上没有结束时刻，``last_observed_at`` 会随观测覆盖动），多前件锚在集合里**最后开始**
的那个行为概念——所以前件集合里至少要有一个行为概念，情境概念是状态、当不了锚。

次数方面另有 ``horizon``：数后件接下来几个机会里的次数（"晚睡当天咖啡喝几杯" = 当天三个机会），对照是那几个
机会的期望之和。

身份是 (前件集合 · 后件 · 方面)，不含版本；集合无序，先 A 后 B 那种有序的是两条假设（链）。同身份改内容
（换第几次机会、换方向）会让一本账混两种量，所以有内容指纹：账本开承诺时快照它，存储覆写前核对它。没有
status 字段：退役是算法判的，记在账或投影里，假设文件不动。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from habitus.foundation.ids import require_safe_path_segment
from habitus.foundation.integrity import canonical_digest
from habitus.scene.concepts.model import IDENTITY_SEPARATORS, ConceptError, ConceptSet, concept_identity

MAX_NOTE_CHARS = 400
MAX_GRADE_CHARS = 40
MAX_ANTECEDENTS = 4
#: "第几次机会"与次数方面的地平线都是小整数：超过这个数的猜测基准写不出理由，账本也等不了那么久（保护闸）。
MAX_OPPORTUNITIES = 30
#: 身份叶名的三个分隔符。读侧（``views.relations`` 拼上一级的身份借先验）也要用它们，所以是公开的——
#: 手抄一份的话，分隔符一改借力就静默借到不存在的身份上，而 ``load_account`` 对不存在的身份返回空账、不报错。
ELEMENT_SEPARATOR = "+"
GRADE_SEPARATOR = "@"
ASPECT_SEPARATOR = "--"
#: 无节律型（``expected_at is None``）在身份里占的位：它没有"第几次机会"，但身份的段数要一致，
#: 不然解析尾巴的地方要分两种写法。
OPEN_OPPORTUNITY_TOKEN = "open"
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
    BASELINE = "baseline"
    NEW_CONCEPT = "new_concept"


@dataclass(frozen=True)
class HypothesisSource:
    origin: HypothesisOrigin
    note: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", HypothesisOrigin(self.origin))
        if self.note is not None:
            object.__setattr__(self, "note", _single_line(self.note, "hypothesis source note"))


@dataclass(frozen=True)
class Antecedent:
    """前件集合的一个元素：概念，可带档（剂量复用多体——元素带档就是另一条假设）。"""

    concept: str
    grade: str | None = None

    def __post_init__(self) -> None:
        _ = self.identity  # 名字不合法在这里抛
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
        return self.identity if self.grade is None else f"{self.identity}{_GRADE_SEPARATOR}{self.grade}"

    def label(self) -> str:
        return self.concept if self.grade is None else f"{self.concept}·{self.grade}"


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
    #: 基准猜影响落在后件的第几次机会（1 = 锚之后的下一个）；None = 无节律型，说不准、不数机会。
    expected_at: int | None = 1
    #: 次数方面数后件接下来几个机会里的次数；其他方面恒为 1。
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
        if self.expected_at is not None and (
            isinstance(self.expected_at, bool) or not isinstance(self.expected_at, int) or not 1 <= self.expected_at <= MAX_OPPORTUNITIES
        ):
            raise HypothesisError(f"expected_at is the consequent's 1st–{MAX_OPPORTUNITIES}th opportunity, or None for 'until it happens'")
        if self.expected_at is None and self.aspect is not Aspect.PROBABILITY:
            # 时刻与次数量的是后件**某一次**到来的时刻 / 接下来几次机会里的次数，都得知道是哪一次；只有"会不会来、
            # 隔多久来"可以不限定次序。
            raise HypothesisError("timing and count hypotheses name the opportunity they measure")
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or not 1 <= self.horizon <= MAX_OPPORTUNITIES:
            raise HypothesisError(f"horizon counts 1–{MAX_OPPORTUNITIES} opportunities")
        if self.aspect is not Aspect.COUNT and self.horizon != 1:
            raise HypothesisError("only count hypotheses look across several opportunities")
        released = tuple(self.released_by)
        if released and self.expected_at is not None:
            raise HypothesisError("released_by belongs to open-ended hypotheses (expected_at=None) only")
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
        """无节律型（不数机会、没有时效，只有兑现与释放两种结法）。"""

        return self.expected_at is None

    @property
    def consequent_identity(self) -> str:
        try:
            return concept_identity(self.consequent)
        except ConceptError as exc:
            raise HypothesisError(str(exc)) from exc

    @property
    def opportunity_token(self) -> str:
        """身份里表示"第几次机会"的那一段：节律型是数字，无节律型是 ``open``。"""

        return OPEN_OPPORTUNITY_TOKEN if self.expected_at is None else str(self.expected_at)

    @property
    def leaf(self) -> str:
        """``<前件@档+前件>--<方面>--<第几次机会>``，是这条假设在后件目录下的文件名（不含 ``.md``）。

        **"第几次机会"进身份**（2026-09-27 裁定）：一天多峰的后件要逐峰各挂一条假设（咖啡早上 / 中午 / 下午
        三个峰的概率是 0.50 / 0.40 / 0.30，前置条件也不一样），而三条只差 ``expected_at`` 的假设在旧格式下
        身份完全一样、只有指纹不同，存储对同身份不同指纹默认拒写 → 三条只能存一条。进了身份之后它们是三条
        不同的假设、三本不同的账，这也正是"同身份改内容会让一本账混两种量"那条纪律本来想要的。
        """

        leaf = (
            _ELEMENT_SEPARATOR.join(item.leaf_token for item in self.antecedents)
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
                "expected_at": self.expected_at,
                "horizon": self.horizon,
                "released_by": sorted(concept_identity(name) for name in self.released_by),
                "type_prior": None if self.type_prior is None else self.type_prior.value,
                "split_by": sorted(concept_identity(name) for name in self.split_by),
            }
        )[:16]

    @property
    def opportunity_label(self) -> str:
        if self.expected_at is None:
            release = "，直到后件到来" if not self.released_by else "，直到后件到来或 " + " / ".join(self.released_by)
            return f"无节律型：不数机会{release}"
        if self.aspect is Aspect.COUNT and self.horizon > 1:
            return f"后件的第 {self.expected_at}–{self.expected_at + self.horizon - 1} 次机会"
        return f"后件的第 {self.expected_at} 次机会"

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
    "ASPECT_SEPARATOR",
    "ELEMENT_SEPARATOR",
    "GRADE_SEPARATOR",
    "MAX_ANTECEDENTS",
    "MAX_OPPORTUNITIES",
    "OPEN_OPPORTUNITY_TOKEN",
    "Antecedent",
    "Aspect",
    "Direction",
    "Hypothesis",
    "HypothesisError",
    "HypothesisOrigin",
    "HypothesisSource",
    "TypePrior",
]
