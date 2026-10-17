"""行为单元（裁定 56，第 12 条为准）：一个行为一个单元，看这一刻在场的全部输入，从历史里数下一步来了多少。纯函数、只读。

**输入**是这个行为此刻的全部影响因素：每一条到了时窗的预期就是一个输入（事件前因——含汇总概念，"出差"这一层就在这里——、
钟面、几号、离上次够久），它那本账就是这个输入的权重（由结算一次次调，裁定 56 第 3 条）；压低这个行为、此刻在场的关系
（取消出差）也算进这一组。像 transformer 里一个 token 看上下文里全部 token；区别是不学注意力，直接从历史数。

**读数** = 这一组在场时，历史上下一步来了多少。历史上的"一刻"是同一个行为的几条预期在说中的范围上连成一片的那一段，
来没来看它们在预测轴上现在算数的那几笔（裁定 54、55：被提醒才做的、观测空白里的、作废的都不算知道）。样本少分层退：

1. **同一组**：在场的输入（账身份）与压低的关系完全相同的那几刻；
2. **同样多**：抬高的、压低的各有几个都一样的那几刻；
3. **最强那一个**：这一刻里账下界最高的那个输入，它那本账此刻的预测轴读数（就是原来"取一条"的读法；账上的强度、规律断了
   的打折、遗忘都在里面——它就是这个输入的权重，由结算调）。

上面两层以下一层的读数为先验收缩：先验的分量取 ``prior_strength`` 与下一层自己的有效样本里较小的那个——没见过的组合贴着
更粗的那一层，见得越多越按它自己；先验分量是配置，从回放定。读数按依据那本账的线过线（裁定 58）；依据先看事件输入里
成立、效应够大、抬升最高的那条，没有就看钟面与日子输入里账下界最高的那条——说话时报出在场的那一组。

**自检**（裁定 56 第 10 条）：每个事件输入两个数——账（从它出发预测准不准）与语义树（有它比没它多出多少）——对不对得上。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from habitus.foresight.errors import ForesightError
from habitus.foresight.ladder import Evidence, Standing
from habitus.foresight.ledger.model import AccountFact, Expectation
from habitus.foresight.opportunities.model import OpportunityTable, TreeReading
from habitus.foresight.reliability import Reliability, estimate, weight

#: 算自然发生了的预测轴结果。
NATURAL_HITS = frozenset({"准时", "早到", "晚了但做了"})
#: 看得见的落空（强度大于 0 才算：观测空白里的中性不算知道）。
_MISSES = frozenset({"过期待决", "后补"})
#: 压低的关系在签名里的前缀（与账身份区分开）。
DOWN = "↓"
#: 同一刻同一个行为的另外几个输入的递送原因：单元替它们一并决定，证据里写 ``UNIT_CITE + 依据那条预期的身份``。
UNIT = "单元已选"
UNIT_CITE = "单元:"


class Level(str, Enum):
    SAME_SET = "同一组"
    SAME_COUNT = "同样多"
    STRONGEST = "最强那一个"


@dataclass(frozen=True)
class Input:
    """单元此刻的一个输入：一条到了时窗的预期、它那本账此刻的位置、它的关系在语义树上的读数（钟面 / 日子没有）。"""

    expectation: Expectation
    standing: Standing
    reading: TreeReading | None = None
    status: str | None = None

    @property
    def account(self) -> str:
        return self.expectation.account


@dataclass(frozen=True)
class Moment:
    """历史上的一刻：哪一刻（这一段里最早形成的那条预期）、在场的那一组、下一步来没来。"""

    at: datetime
    signature: frozenset[str]
    came: bool

    @property
    def ups(self) -> int:
        return sum(1 for item in self.signature if not item.startswith(DOWN))

    @property
    def downs(self) -> int:
        return sum(1 for item in self.signature if item.startswith(DOWN))


@dataclass(frozen=True)
class UnitReading:
    """这一刻的读数：落在哪一层、那一层收缩之后的读数，以及三层的样本（同一组几刻、同样多几刻、最强那本账的有效样本；给人看）。"""

    level: Level
    reading: Reliability
    seen: tuple[int, int, int]


def signature(expectations: Iterable[Expectation]) -> frozenset[str]:
    """在场的那一组：每条预期的账身份，加上它们前因清单里此刻在场、压低这个行为的关系。"""

    found: set[str] = set()
    for item in expectations:
        found.add(item.account)
        found |= {f"{DOWN}{antecedent.relation}" for antecedent in item.antecedents if antecedent.present and not antecedent.upward}
    return frozenset(found)


def moments(
    expectations: Iterable[Expectation], prediction_of: Callable[[Expectation], AccountFact | None]
) -> tuple[Moment, ...]:
    """同一个行为的预期按说中的范围（起算 → 最晚时限）连成一片的算一刻；这一刻来没来看它们预测轴上现在算数的那几笔：
    有一条自然命中就是来了；没有命中而有看得见的落空就是没来；其余（被提醒才做的、观测空白、作废、还没结）不知道，不算。"""

    ordered = sorted(expectations, key=lambda item: (item.observed_from, item.expectation_id))
    groups: list[list[Expectation]] = []
    end: datetime | None = None
    for item in ordered:
        if groups and end is not None and item.observed_from <= end:
            groups[-1].append(item)
            end = max(end, item.deadline)
        else:
            groups.append([item])
            end = item.deadline
    found = []
    for group in groups:
        facts = [fact for item in group if (fact := prediction_of(item)) is not None]
        if any(fact.kind in NATURAL_HITS for fact in facts):
            came = True
        elif any(fact.kind in _MISSES and fact.strength > 0.0 for fact in facts):
            came = False
        else:
            continue
        found.append(Moment(min(item.formed_at for item in group), signature(group), came))
    return tuple(found)


def unit_reading(
    inputs: Sequence[Input],
    history: Sequence[Moment],
    *,
    now: datetime,
    evidence: Evidence,
    prior_strength: float,
) -> UnitReading:
    """这一组在场时，历史上下一步来了多少（三层收缩，见模块说明）。"""

    if not inputs:
        raise ForesightError("a unit reading needs at least one input")
    if isinstance(prior_strength, bool) or not isinstance(prior_strength, int | float) or prior_strength < 0:
        raise ForesightError("prior_strength must be a non-negative number")
    current = signature(item.expectation for item in inputs)
    strongest = max(inputs, key=lambda item: (item.standing.prediction.lower, item.account))
    ups = sum(1 for item in current if not item.startswith(DOWN))
    downs = len(current) - ups
    past = [moment for moment in history if moment.at < now]
    same_count = [moment for moment in past if (moment.ups, moment.downs) == (ups, downs)]
    same_set = [moment for moment in past if moment.signature == current]
    reading = strongest.standing.prediction
    level = Level.STRONGEST
    for name, layer in ((Level.SAME_COUNT, same_count), (Level.SAME_SET, same_set)):
        if not layer:
            continue  # 这一层一刻都没见过：没有读数，停在下一层
        reading = _shrunk(layer, now=now, evidence=evidence, prior=reading, strength=prior_strength)
        level = name
    return UnitReading(level, reading, (len(same_set), len(same_count), round(strongest.standing.prediction.n)))


def _shrunk(
    layer: Sequence[Moment], *, now: datetime, evidence: Evidence, prior: Reliability, strength: float
) -> Reliability:
    observed = [
        (weight(moment.at, now, strength=1.0, half_life_days=evidence.half_life_days), moment.came) for moment in layer
    ]
    share = min(strength, prior.n)
    observed += [(share * prior.mean, True), (share * (1.0 - prior.mean), False)]
    return estimate(observed, alpha=evidence.alpha)


def basis(inputs: Sequence[Input]) -> Input:
    """说话的依据：事件输入里关系成立、语义树判效应够大、抬升最高的那条；没有就取账下界最高的那条（钟面 / 日子，或抬升
    不够大的事件）——账高、抬升低的是巧合搭车，不当依据（裁定 56 第 10 条）。"""

    if not inputs:
        raise ForesightError("a unit needs at least one input to speak from")
    causes = [
        item
        for item in inputs
        if item.reading is not None
        and item.status == "established"
        and item.reading.large_enough is True
        and item.reading.lift > 0.0
    ]
    if causes:
        return max(causes, key=lambda item: (item.reading.lift if item.reading else 0.0, item.account))
    coincidental = {item.account for item in inputs if item.reading is not None and item.reading.large_enough is False}
    plain = [item for item in inputs if item.account not in coincidental] or list(inputs)
    return max(plain, key=lambda item: (item.standing.prediction.lower, item.account))


def unit_level(basis: Standing, reading: UnitReading, *, previous: int | None) -> int:
    """这一刻单元的档：依据那本账的人为设置优先（「以后都提醒我」/「以后别提了」）；依据那本账提醒组上界掉到线下的照降
    （说了他老不做，裁定 58 第 6 条）；其余按单元读数过依据那本账的线——下界过线升、上界在线下降、中间维持上一次（滞回）。"""

    if basis.setting == "以后都提醒我":
        return 1
    if basis.setting == "以后别提了":
        return 0
    if basis.reminded.n > 0 and basis.reminded.upper < basis.threshold:
        return 0
    if reading.reading.upper < basis.threshold:
        return 0
    if reading.reading.lower >= basis.threshold:
        return 1
    return 1 if previous == 1 else 0


class Check(str, Enum):
    """一个事件输入的两个数对不对得上（裁定 56 第 10 条）。"""

    BASIS = "开口依据"
    COINCIDENCE = "凑巧同现，不是原因"
    MISCOMPUTED = "关联对、预测层算得不对"
    NOT_A_CAUSE = "不是原因"
    CANNOT_RISE = "是原因、账还上不去"
    WATCHING = "还在看"


def self_check(standing: Standing, reading: TreeReading | None) -> Check:
    """账高 = 下界过线，账低 = 上界在线下；抬升大 = 语义树判效应够大且为正。"""

    high = standing.prediction.lower >= standing.threshold
    low = standing.prediction.upper < standing.threshold
    cause = reading is not None and reading.large_enough is True and reading.lift > 0.0
    if reading is None or reading.large_enough is None:
        return Check.WATCHING
    if high:
        return Check.BASIS if cause else Check.COINCIDENCE
    if low:
        return Check.MISCOMPUTED if cause else Check.NOT_A_CAUSE
    return Check.CANNOT_RISE if cause else Check.WATCHING


def tree_reading(table: OpportunityTable | None, expectation: Expectation) -> tuple[TreeReading | None, str | None]:
    """事件输入在机会表上的语义树读数与关系状态；钟面 / 日子没有。"""

    if table is None or expectation.source != "事件":
        return None, None
    relation = next((item for item in table.relations if item.key == expectation.trigger), None)
    return (None, None) if relation is None else (relation.reading, relation.status)


__all__ = [
    "DOWN",
    "NATURAL_HITS",
    "UNIT",
    "UNIT_CITE",
    "Check",
    "Input",
    "Level",
    "Moment",
    "UnitReading",
    "basis",
    "moments",
    "self_check",
    "signature",
    "tree_reading",
    "unit_level",
    "unit_reading",
]
