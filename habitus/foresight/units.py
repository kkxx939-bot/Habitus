"""行为单元（裁定 56，第 12 条为准）：一个行为一个单元，看这一刻在场的全部输入，从历史里数下一步来了多少。纯函数、只读。

**输入**是这个行为此刻的全部影响因素：每一条到了时窗的预期就是一个输入（事件前因——含汇总概念，"出差"这一层就在这里——、
日历情境（周三、工作日，裁定 59）、离上次够久），它那本账就是这个输入的权重（由结算一次次调，裁定 56 第 3 条）；压低这个行为、此刻在场的关系
（取消出差）也算进这一组。像 transformer 里一个 token 看上下文里全部 token；区别是不学注意力，直接从历史数。

**读数** = 这一组在场时，历史上下一步来了多少。历史上的"一刻"是同一个行为的几条预期在说中的范围上连成一片的那一段，
来没来看它们在预测轴上现在算数的那几笔（裁定 54：被提醒才做的、观测空白里的、作废的都不算知道），一刻的分量是定它来没来的
那一笔的强度（窗外发生、后补、规律断了的打折都在里面，与账同一把尺）。历史上每一刻在场的输入每晚按当晚的语义树重标
（裁定 55 新版第 2 条，``foresight.history``）：一条新关系长出来那一晚，它在场的那几刻自动换进新组合。样本少分层退：

1. **同一组**：在场的输入（账身份）与压低的关系完全相同的那几刻；
2. **同样多**：抬高的、压低的各有几个都一样的那几刻；
3. **最强那一个**：这一刻里账下界最高的那个输入，它那本账此刻的预测轴读数（就是原来"取一条"的读法；账上的强度、规律断了
   的打折、遗忘都在里面——它就是这个输入的权重，由结算调）。

"同一组"以"同样多"里组合不同的那几刻为先验收缩（先验的分量取有效样本上限的 ``prior_share`` 与那几刻自己的有效样本里较小的
那个；比例是配置，从回放定。每一层按这一层自己的刻数遗忘）；同一批机会不数两遍——第三层不当先验，输入个数不同的组合也不当先验（``unit_reading``）。
读数按依据那本账的线过线（裁定 58）；依据只从到了时窗的输入里选：有关系撑着的输入（事件、情境）里成立、效应够大、抬升最高的
那条，没有就看账下界最高的那条——候选关系与巧合只进读数、不当依据，单独当依据时记影子（``speakable``）；说话时报出在场的那一组。
单元的档按行为记，滞回读这个行为单元上一次的档（``unit_level``）。

**自检**（裁定 56 第 10 条）：每个事件输入两个数——账（从它出发预测准不准）与语义树（有它比没它多出多少）——对不对得上。
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol

from habitus.foresight.edges import DOWN, PastMoment, fold_sequence, ordered
from habitus.foresight.errors import ForesightError
from habitus.foresight.ladder import Evidence, Standing
from habitus.foresight.ledger.model import NATURAL_HITS, Expectation, relation_backed
from habitus.foresight.opportunities.model import OpportunityTable, TreeReading
from habitus.foresight.reliability import Reliability, estimate
from habitus.foundation.lifecycle import effective_cap

#: 同一刻同一个行为的另外几个输入的递送原因：单元替它们一并决定，证据里写 ``UNIT_CITE + 依据那条预期的身份``。
UNIT = "单元已选"
UNIT_CITE = "单元:"


class Level(str, Enum):
    SAME_SET = "同一组"
    SAME_COUNT = "同样多"
    STRONGEST = "最强那一个"


@dataclass(frozen=True)
class Input:
    """单元此刻的一个输入：一条到了时窗的预期、它那本账此刻的位置、它的关系在语义树上的读数（日子没有）。"""

    expectation: Expectation
    standing: Standing
    reading: TreeReading | None = None
    status: str | None = None

    @property
    def account(self) -> str:
        return self.expectation.account


@dataclass(frozen=True)
class UnitReading:
    """这一刻的读数：落在哪一层、那一层收缩之后的读数，以及三层的样本（同一组几刻、同样多几刻、最强那本账的有效样本；给人看）。"""

    level: Level
    reading: Reliability
    seen: tuple[int, int, int]


def signature(expectations: Iterable[Expectation], *, coincidental: Collection[str] = ()) -> frozenset[str]:
    """在场的那一组：每条预期的账身份，加上它们前因清单里此刻在场、压低这个行为的关系。``coincidental`` 是语义树已判成巧合
    （效应不够大）的关系身份：它们不进签名，不参与"同一组 / 同样多"（方案第六节第 1 条；权重照数，在边上）。"""

    found: set[str] = set()
    for item in expectations:
        if item.trigger not in coincidental:
            found.add(item.account)
        found |= {
            f"{DOWN}{antecedent.relation}"
            for antecedent in item.antecedents
            if antecedent.present and not antecedent.upward and antecedent.relation not in coincidental
        }
    return frozenset(found)


class Span(Protocol):
    """说中的范围：从什么时候起的发生算这一条说中的、最晚时限（预期与单元历史的每刻记录都有）。"""

    @property
    def expectation_id(self) -> str: ...

    @property
    def observed_from(self) -> datetime: ...

    @property
    def deadline(self) -> datetime: ...


def connected[S: Span](items: Iterable[S]) -> list[list[S]]:
    """同一个行为的预期按说中的范围（起算 → 最晚时限）连成一片的分成一组（一刻）。"""

    ordered = sorted(items, key=lambda item: (item.observed_from, item.expectation_id))
    groups: list[list[S]] = []
    end: datetime | None = None
    for item in ordered:
        if groups and end is not None and item.observed_from <= end:
            groups[-1].append(item)
            end = max(end, item.deadline)
        else:
            groups.append([item])
            end = item.deadline
    return groups


def unit_reading(
    inputs: Sequence[Input],
    history: Sequence[PastMoment],
    *,
    now: datetime,
    evidence: Evidence,
    prior_share: float,
    coincidental: Collection[str] = (),
) -> UnitReading:
    """这一组在场时，历史上下一步来了多少（三层收缩，见模块说明）。``coincidental`` 见 ``signature``。

    **不重复数同一批机会**（2026-10-17 四方评审）："同一组"以"同样多"里**组合不同的**那几刻为先验（没有就不加先验）；"同一组"
    一刻都没见过时就是"同样多"那几刻本身；连"同样多"都没见过才退到最强那本账的读数。第三层不拿来当先验：那本账的事实与上两层的
    刻是同一批，当先验会数两三遍、等于变相降线（每周 4/4 的账下界 0.60 过不了 0.67 线，叠加之后 0.71 就开了口）。也不拿输入
    个数不同的组合当先验：单个前因与几个前因一起的那几刻正是单元要分开的（各 10 刻时会被互相拉到 0.72 对 0.73）。"""

    if not inputs:
        raise ForesightError("a unit reading needs at least one input")
    if isinstance(prior_share, bool) or not isinstance(prior_share, int | float) or not 0.0 < prior_share < 1.0:
        raise ForesightError("prior_share must lie in (0, 1)")
    current = signature((item.expectation for item in inputs), coincidental=coincidental)
    strongest = max(inputs, key=lambda item: (item.standing.prediction.lower, item.account))
    ups = sum(1 for item in current if not item.startswith(DOWN))
    downs = len(current) - ups
    past = [moment for moment in history if moment.at < now]
    same_count = [moment for moment in past if (moment.ups, moment.downs) == (ups, downs)]
    same_set = [moment for moment in same_count if moment.signature == current]
    seen = (len(same_set), len(same_count), round(strongest.standing.prediction.n))
    if not same_count:
        return UnitReading(Level.STRONGEST, strongest.standing.prediction, seen)
    if not same_set:
        return UnitReading(Level.SAME_COUNT, _plain(same_count, evidence=evidence), seen)
    others = [moment for moment in same_count if moment.signature != current]
    prior = _plain(others, evidence=evidence) if others else None
    return UnitReading(Level.SAME_SET, _shrunk(same_set, evidence=evidence, prior=prior, share=prior_share), seen)


def _folded(layer: Sequence[PastMoment], *, evidence: Evidence) -> Reliability:
    """这一层的刻按这一层自己的刻数遗忘（层内单独编名次）：少见的组合不会因为别的组合天天来就被冲掉；层与层比的是读数，
    时钟不必对齐。同一时刻的几刻按在场那一组排先后，读数与输入顺序无关。只读最近 M 刻（``Evidence.window``，窗外一刻的分量
    已低于 α）：单元历史按这条线退出旧的刻（``foresight.history.retired``），删了哪一层的读数都不变。"规律断了"的打折与边
    同一个算法（``edges.fold_sequence``）。"""

    return fold_sequence(
        ordered(layer), alpha=evidence.alpha, half_life=evidence.half_life, window=evidence.window
    ).reading


def _plain(layer: Sequence[PastMoment], *, evidence: Evidence) -> Reliability:
    return _folded(layer, evidence=evidence)


def _shrunk(layer: Sequence[PastMoment], *, evidence: Evidence, prior: Reliability | None, share: float) -> Reliability:
    """这一层的刻，加上先验（不在这一层的那些刻的读数）当伪计数；先验的分量 = 有效样本上限 1 / (1 − 2^(−1/H)) 的 ``share``，
    也不超过先验自己的有效样本。分量写成固定数会在有效样本有上限之后占得太重：100/100 的组合、同层均值 0.5 时下界被拉到 0.63。"""

    own = _folded(layer, evidence=evidence)
    observed: list[tuple[float, bool]] = [(own.k, True), (own.n - own.k, False)]
    if prior is not None:
        strength = min(effective_cap(half_life=evidence.half_life) * share, prior.n)
        observed += [(strength * prior.mean, True), (strength * (1.0 - prior.mean), False)]
    return estimate(observed, alpha=evidence.alpha)


def speakable(item: Input) -> bool:
    """这个输入能不能当开口的依据（方案第六节"开口只用成立的"）：日子能；有关系撑着的输入（事件、情境）只有关系已成立、
    语义树没判它效应不够大（巧合）的才能——候选关系与巧合只进读数、不当依据。"""

    if not relation_backed(item.expectation.source):
        return True
    return item.status == "established" and not (item.reading is not None and item.reading.large_enough is False)


def basis(inputs: Sequence[Input], *, due: Iterable[str] | None = None) -> Input:
    """说话的依据：有关系撑着的输入里关系成立、语义树判效应够大、抬升最高的那条；没有就取账下界最高的那条（日子，或抬升
    不够大的事件）——账高、抬升低的是巧合搭车，不当依据（裁定 56 第 10 条）。``due`` 给了就只从这些预期（已到时窗的）里选：
    还没到时窗的只当输入，不当依据（2026-10-17 四方评审：拿次日那条当依据，重的会提前一天说、轻的会推到第二天）。"""

    if not inputs:
        raise ForesightError("a unit needs at least one input to speak from")
    allowed = None if due is None else set(due)
    pool = [item for item in inputs if allowed is None or item.expectation.expectation_id in allowed] or list(inputs)
    causes = [
        item
        for item in pool
        if item.reading is not None
        and item.status == "established"
        and item.reading.large_enough is True
        and item.reading.lift > 0.0
    ]
    if causes:
        return max(causes, key=lambda item: (item.reading.lift if item.reading else 0.0, item.account))
    coincidental = {item.account for item in pool if item.reading is not None and item.reading.large_enough is False}
    plain = [item for item in pool if item.account not in coincidental] or pool
    return max(plain, key=lambda item: (item.standing.prediction.lower, item.account))


def unit_level(
    chosen: Input,
    reading: UnitReading,
    *,
    previous: int | None,
    feedback: Reliability | None = None,
    setting: str | None = None,
) -> int:
    """这一刻单元的档（开口只看单元，裁定 61 第 1 条）：这个行为的人为设置优先（「以后都提醒我」只跳过攒把握、「以后别提了」
    恒为影子）；依据不能开口（候选关系、巧合，``speakable``）的记影子；单元的提醒反馈上界掉到线下的照降（说了他老不做，
    裁定 58 第 6 条、61 第 4 条）；其余按单元读数过依据那本账的线——下界过线升、上界在线下降、中间维持上一次（滞回；
    ``previous`` 是这个行为单元上一次的档，不是哪一本账的）。"""

    standing = chosen.standing
    if setting == "以后别提了" or not speakable(chosen):
        return 0
    if feedback is not None and feedback.n > 0 and feedback.upper < standing.threshold:
        return 0
    if setting == "以后都提醒我":
        return 1
    if reading.reading.upper < standing.threshold:
        return 0
    if reading.reading.lower >= standing.threshold:
        return 1
    return 1 if previous == 1 else 0


class Check(str, Enum):
    """一个事件输入的两个数对不对得上（裁定 56 第 10 条；可见面的说法按方案 30 第六节第 1 条）。"""

    BASIS = "开口依据"
    COINCIDENCE = "凑巧一起出现，不是原因"
    MISCOMPUTED = "关联对、预测层算得不对"
    NOT_A_CAUSE = "不是原因"
    CANNOT_RISE = "是原因、账还上不去"
    WATCHING = "还在观察"


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
    """有关系撑着的输入（事件、情境）在机会表上的语义树读数与关系状态；日子没有。"""

    if table is None or not relation_backed(expectation.source):
        return None, None
    relation = table.relation(expectation.trigger)
    return (None, None) if relation is None else (relation.reading, relation.status)


__all__ = [
    "DOWN",
    "NATURAL_HITS",
    "UNIT",
    "UNIT_CITE",
    "Check",
    "Input",
    "Level",
    "PastMoment",
    "UnitReading",
    "basis",
    "connected",
    "self_check",
    "signature",
    "speakable",
    "tree_reading",
    "unit_level",
    "unit_reading",
]
