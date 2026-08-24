"""钟面节点：计数、曝光、三层收缩，以及由它们派生的三种率。

零 IO 纯函数——输入是已归一化的行为与空白，输出是可直接发布的统计量。全部正确性都能在这里
穷举验证（见 ``TODO(PRED-TREE-001)`` 的估计纪律一节）。

三条卫生里的两条落在本模块：**曝光分母**（只数被观测到的槽，两类空白都扣）与**收缩**
（稀疏槽会出 0/1 极端值）。第三条"转移删失"在 ``edges.py``。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from prediction.config import PredictionTreeConfig
from prediction.errors import PredictionTreeError
from prediction.model import (
    MINUTES_PER_DAY,
    NodeCounts,
    NodeStatistics,
    ObservedAction,
    ObservedGap,
    SlotExposure,
    SlotKey,
)

_SECONDS_PER_DAY = 86_400.0


def decay_weight(age_days: float, half_life_days: float) -> float:
    """时间衰减：``half_life_days`` 天前的证据算半份。"""

    if half_life_days <= 0:
        raise PredictionTreeError("half_life_days must be positive")
    return 2.0 ** (-max(age_days, 0.0) / half_life_days)


@dataclass(frozen=True)
class NodeLedger:
    """一次扫描得到的全部原始账本：格子计数 + 槽位曝光 + 动作全天基线。"""

    counts: Mapping[tuple[SlotKey, str], NodeCounts]
    exposure: Mapping[SlotKey, SlotExposure]
    actions: tuple[str, ...]
    observed_days: int


def accumulate(
    actions: Sequence[ObservedAction],
    gaps: Sequence[ObservedGap],
    *,
    config: PredictionTreeConfig,
    reference: date,
) -> NodeLedger:
    """一趟扫描算出计数与曝光。

    曝光按"这一天这个槽有多少比例是真的在看"计，两类空白同等扣减——没在看和没读懂对
    "如果他做了我们能不能看见"是同一件事。
    """

    if not isinstance(reference, date):
        raise PredictionTreeError("reference must be a date")
    per_day = _group_by_day(actions, config=config)
    gap_days = group_gaps_by_day(gaps)
    days = _calendar_days(per_day.keys() | gap_days.keys(), reference)
    slots = config.slots_per_day

    counts: dict[tuple[SlotKey, str], NodeCounts] = {}
    exposure: dict[SlotKey, SlotExposure] = {}
    for day in days:
        age = float((reference - day).days)
        long_weight = decay_weight(age, config.decay_half_life_days)
        short_weight = decay_weight(age, config.recent_half_life_days)
        coverage = day_coverage(day, gap_days.get(day, ()), config.slot_minutes)
        # 那一天真的产出过 occurrence 的槽，一律按"在看"记满。occurrence 与 gap 重叠是现实
        # （两者出自不同的判断链，空白的边界本来就是近似的），不是数据自相矛盾——把它写成
        # 硬失败会让一条落在空白里的行为把整夜重建打掉。补满曝光同时是**保守**方向：
        # 分母变大、概率变小，宁可少说。
        for slots_hit in per_day.get(day, {}).values():
            for slot_index in slots_hit:
                coverage[slot_index] = 1.0
        for slot_index in range(slots):
            covered = coverage[slot_index]
            if covered <= 0.0:
                continue
            key = SlotKey(weekday=day.weekday(), slot=slot_index)
            current = exposure.get(key, SlotExposure())
            exposure[key] = current.plus(
                observed_days=long_weight * covered,
                recent_days=short_weight * covered,
            )
        for action, occurrences in per_day.get(day, {}).items():
            _accumulate_action(
                counts,
                day=day,
                action=action,
                slots_hit=occurrences,
                coverage=coverage,
                slots=slots,
                long_weight=long_weight,
                short_weight=short_weight,
            )
    return NodeLedger(
        counts=counts,
        exposure=exposure,
        actions=tuple(sorted({action for _, action in counts})),
        observed_days=len(days),
    )


def _accumulate_action(
    counts: dict[tuple[SlotKey, str], NodeCounts],
    *,
    day: date,
    action: str,
    slots_hit: Mapping[int, int],
    coverage: Sequence[float],
    slots: int,
    long_weight: float,
    short_weight: float,
) -> None:
    """把一天里某个动作的全部发生折算进四个计数。

    同日同槽封顶 1（``occurred_days``），但原始次数照留；首次那个槽单独记 ``first_days``，
    其后的每个槽记 ``earlier_days``——这两个让危险率与累积率无需跨槽累乘即可得到。

    ``earlier_days`` **必须与曝光同口径**（按覆盖比例、且只记在有覆盖的槽上）。它是危险率
    风险集 ``E − earlier`` 的减数：记在没有曝光的槽上会造出"有计数无曝光"的格子（发布时硬失败），
    记成整份权重则会把风险集压成近零的正数，让危险率炸到几百。两种都实测触发过。
    """

    weekday = day.weekday()
    first_slot = min(slots_hit)
    for slot_index, repetitions in slots_hit.items():
        # 发生本身就是"当时在看"的证据，所以计数按整份权重记；分母那边按覆盖比例记，
        # 半覆盖的槽因此可能算出大于 1 的率，发布时统一夹紧（见 derive）。
        key = (SlotKey(weekday=weekday, slot=slot_index), action)
        counts[key] = counts.get(key, NodeCounts()).plus(
            occurred_days=long_weight,
            raw_occurrences=long_weight * repetitions,
            recent_days=short_weight,
            first_days=long_weight if slot_index == first_slot else 0.0,
        )
    for slot_index in range(first_slot + 1, slots):
        covered = coverage[slot_index]
        if covered <= 0.0:
            continue
        key = (SlotKey(weekday=weekday, slot=slot_index), action)
        counts[key] = counts.get(key, NodeCounts()).plus(earlier_days=long_weight * covered)


def derive(ledger: NodeLedger, *, config: PredictionTreeConfig) -> dict[tuple[SlotKey, str], NodeStatistics]:
    """把原始账本派生成发布成品：三种率 + 两个 lift + n_eff + 趋势。

    收缩顺序**从假设最弱的借用开始**（PRED-TREE-001 修正过的技术错误）：

        (周几, 槽) ← (周几, 池化邻域) ← (跨周几, 池化邻域) ← 该动作全天平均

    时间邻域借用只假设"相邻时刻的率相近"，跨周几借用假设"各周几的率相近"——后者对周规律
    行为直接错误，放在第一步会系统性把周规律压平。
    """

    slots = config.slots_per_day
    all_day = _all_day_rates(ledger, config=config)
    statistics: dict[tuple[SlotKey, str], NodeStatistics] = {}
    for (key, action), counts in ledger.counts.items():
        exposure = ledger.exposure.get(key, SlotExposure())
        if exposure.observed_days <= 0.0:
            # 到这里只剩"我们自己的账不平"这一种可能：发生过的槽在 accumulate 里已经补满曝光，
            # 只有 earlier_days 的槽只记在有覆盖的槽上。两条都守住了还出现，就是实现错了。
            raise PredictionTreeError(
                f"node {action} at {key} has counts without exposure"
            )
        pool = pool_indexes(key.slot, config.pool_half_width, slots)
        marginal = _shrink_chain(
            ledger,
            action=action,
            key=key,
            pool=pool,
            numerator=lambda item: item.occurred_days,
            denominator=lambda exp: exp.observed_days,
            all_day_rate=all_day.marginal[action],
            config=config,
        )
        hazard = _shrink_chain(
            ledger,
            action=action,
            key=key,
            pool=pool,
            numerator=lambda item: item.first_days,
            denominator=lambda exp: exp.observed_days,
            all_day_rate=all_day.hazard[action],
            config=config,
            risk_adjustment=True,
        )
        cumulative = min(
            (counts.earlier_days + counts.first_days) / exposure.observed_days, 1.0
        )
        weekday_baseline = all_day.per_slot_cross_weekday.get((key.slot, action), 0.0)
        statistics[(key, action)] = NodeStatistics(
            marginal=_probability(marginal),
            hazard=_probability(hazard),
            cumulative=cumulative,
            lift_all_day=_ratio(_probability(marginal), all_day.marginal[action]),
            lift_weekday=_ratio(_probability(marginal), weekday_baseline),
            n_eff=exposure.observed_days,
            trend=_trend(counts, exposure),
            counts=counts,
        )
    return statistics


def all_day_marginals(ledger: NodeLedger, *, config: PredictionTreeConfig) -> dict[str, float]:
    """每个动作**不看时刻**的整体发生率。

    它是收缩链的顶层，也是读侧的兜底：树上是稀疏存储，一个动作在某个格子没有记录只说明
    "没见过"，不说明"不会发生"。没有这个基线，读侧对空格子只能给 0，而"三十天没见过"
    和"永不发生"是两回事。
    """

    return dict(_all_day_rates(ledger, config=config).marginal)


@dataclass(frozen=True)
class _Baselines:
    marginal: Mapping[str, float]
    hazard: Mapping[str, float]
    per_slot_cross_weekday: Mapping[tuple[int, str], float]


def _all_day_rates(ledger: NodeLedger, *, config: PredictionTreeConfig) -> _Baselines:
    """收缩链的顶层与 lift 的分母。

    ``per_slot_cross_weekday`` 是"同一时刻跨全部周几"的率——它既是 ``lift_周几`` 的分母
    （回答"周二比随便哪天特别多少"），也是收缩链倒数第二层的先验。
    """

    epsilon = config.laplace_epsilon
    occurred: dict[str, float] = {}
    first: dict[str, float] = {}
    earlier: dict[str, float] = {}
    per_slot_occurred: dict[tuple[int, str], float] = {}
    for (key, action), counts in ledger.counts.items():
        occurred[action] = occurred.get(action, 0.0) + counts.occurred_days
        first[action] = first.get(action, 0.0) + counts.first_days
        earlier[action] = earlier.get(action, 0.0) + counts.earlier_days
        slot_key = (key.slot, action)
        per_slot_occurred[slot_key] = per_slot_occurred.get(slot_key, 0.0) + counts.occurred_days
    total_exposure = sum(item.observed_days for item in ledger.exposure.values())
    per_slot_exposure: dict[int, float] = {}
    for key, item in ledger.exposure.items():
        per_slot_exposure[key.slot] = per_slot_exposure.get(key.slot, 0.0) + item.observed_days
    return _Baselines(
        marginal={
            action: (value + epsilon) / (total_exposure + epsilon)
            for action, value in occurred.items()
        },
        # 危险率的先验必须和收缩链每一层同口径：分母是**风险集**（还没发生的那部分曝光），
        # 不是总曝光。用总曝光会把先验系统性压低 3–4 倍（行为在一天里越早发生偏差越大），
        # 而危险率的全部用途就是"到现在还没做，这个槽会不会做"。
        hazard={
            action: (value + epsilon)
            / (max(total_exposure - earlier.get(action, 0.0), 0.0) + epsilon)
            for action, value in first.items()
        },
        per_slot_cross_weekday={
            (slot, action): (value + epsilon)
            / (per_slot_exposure.get(slot, 0.0) + epsilon)
            for (slot, action), value in per_slot_occurred.items()
        },
    )


def _shrink_chain(
    ledger: NodeLedger,
    *,
    action: str,
    key: SlotKey,
    pool: tuple[int, ...],
    numerator,
    denominator,
    all_day_rate: float,
    config: PredictionTreeConfig,
    risk_adjustment: bool = False,
) -> float:
    """三层收缩；``risk_adjustment`` 把分母换成风险集（危险率用）。"""

    def totals(*, weekday: int | None, slots: Iterable[int]) -> tuple[float, float]:
        top = 0.0
        bottom = 0.0
        weekdays = range(7) if weekday is None else (weekday,)
        for day_index in weekdays:
            for slot_index in slots:
                slot_key = SlotKey(weekday=day_index, slot=slot_index)
                counts = ledger.counts.get((slot_key, action))
                exposure = ledger.exposure.get(slot_key)
                if exposure is None:
                    continue
                available = denominator(exposure)
                if risk_adjustment and counts is not None:
                    available = max(available - counts.earlier_days, 0.0)
                bottom += available
                if counts is not None:
                    top += numerator(counts)
        return top, bottom

    cross_top, cross_bottom = totals(weekday=None, slots=pool)
    cross = _shrink(cross_top, cross_bottom, all_day_rate, config.shrink_weekday_to_all_day)

    pooled_top, pooled_bottom = totals(weekday=key.weekday, slots=pool)
    pooled = _shrink(pooled_top, pooled_bottom, cross, config.shrink_pool_to_weekday)

    slot_top, slot_bottom = totals(weekday=key.weekday, slots=(key.slot,))
    return _shrink(slot_top, slot_bottom, pooled, config.shrink_slot_to_pool)


def _probability(rate: float) -> float:
    """把一个率夹到 [0, 1]。

    需要夹是因为分子分母口径不同：计数按整份权重记（看见了就是看见了），曝光按覆盖比例记。
    半覆盖的槽里看见一次，算出来是"每槽 7.5 次"这样的率而不是概率。夹紧是诚实的答案——
    "在看的时候他确实做了"，概率就是 1；不夹会让这个数直接流进 lift、校准分箱与上层组合。
    """

    return min(max(rate, 0.0), 1.0)


def _shrink(numerator: float, denominator: float, prior_rate: float, strength: float) -> float:
    """Beta-二项伪计数收缩；``strength`` 的单位是"等效观测天数"。

    分母远大于 strength 时几乎等于原始率，分母趋零时退回先验——这正是"数据少就拉回去、
    数据多就放手"。
    """

    return (numerator + strength * prior_rate) / (denominator + strength)


def pool_indexes(slot: int, half_width: int, slots: int) -> tuple[int, ...]:
    """钟面是**环形**的：23:50 的邻域包含 00:05。"""

    if half_width <= 0:
        return (slot,)
    return tuple((slot + offset) % slots for offset in range(-half_width, half_width + 1))


def _ratio(value: float, baseline: float) -> float:
    return value / baseline if baseline > 0.0 else 0.0


def _trend(counts: NodeCounts, exposure: SlotExposure) -> float | None:
    """近期率 ÷ 长期率；两侧任一无曝光则说不出趋势，返回 None 而不是编一个数。"""

    if exposure.recent_days <= 0.0 or exposure.observed_days <= 0.0:
        return None
    long_rate = counts.occurred_days / exposure.observed_days
    if long_rate <= 0.0:
        return None
    return (counts.recent_days / exposure.recent_days) / long_rate


def _group_by_day(
    actions: Sequence[ObservedAction], *, config: PredictionTreeConfig
) -> dict[date, dict[str, dict[int, int]]]:
    """按 (天, 动作, 槽) 归并；值是该槽的发生次数（供不封顶的原始次数用）。"""

    grouped: dict[date, dict[str, dict[int, int]]] = {}
    for item in actions:
        by_action = grouped.setdefault(item.day, {})
        slots_hit = by_action.setdefault(item.action, {})
        slot_index = SlotKey.of(item.started_at, slot_minutes=config.slot_minutes).slot
        slots_hit[slot_index] = slots_hit.get(slot_index, 0) + 1
    return grouped


def group_gaps_by_day(gaps: Sequence[ObservedGap]) -> dict[date, list[ObservedGap]]:
    """按本地日归并；**跨日的空白按天切开**，否则次日那半段会被整段丢掉。"""

    grouped: dict[date, list[ObservedGap]] = {}
    for gap in gaps:
        day = gap.started_at.date()
        last = gap.ended_at.date()
        while day <= last:
            clamped = clamp_gap_to_day(gap, day)
            if clamped is not None:
                grouped.setdefault(day, []).append(clamped)
            day += timedelta(days=1)
    return grouped


def _calendar_days(recorded: Iterable[date], reference: date) -> list[date]:
    """曝光的日集合 = 从最早记录到 ``reference`` 的**每一个日历日**。

    只数"有记录的天"是错的：那等于按"这天发生了点什么"来条件化，会把所有概率整体抬高。
    没有记录的日子照样是机会（他那天就是没做），必须进分母。空白由 gap 扣减；上游覆盖信号
    未接入时其余时段一律视作在看——这是既定的退化假设。
    """

    days = sorted(recorded)
    if not days:
        return []
    earliest = days[0]
    if reference < earliest:
        raise PredictionTreeError("reference must not precede the earliest observation")
    span = (reference - earliest).days
    return [earliest + timedelta(days=offset) for offset in range(span + 1)]


def day_coverage(day: date, gaps: Sequence[ObservedGap], slot_minutes: int) -> list[float]:
    """这一天每个槽有多少比例是真的在看；上游未接覆盖信号时空白之外一律视作在看。

    接的是 ``slot_minutes`` 而不是整个 config：调用方可能拿着一棵**别的档位**建出来的树
    （树自己带槽宽正是为此），传 config 会让两个槽宽悄悄混用。
    """

    slots = MINUTES_PER_DAY // slot_minutes
    width = float(slot_minutes * 60)
    coverage = [1.0] * slots
    for gap in gaps:
        start_offset = _seconds_into_day(gap.started_at, day)
        end_offset = _seconds_into_day(gap.ended_at, day)
        if end_offset <= start_offset:
            continue
        first = max(int(start_offset // width), 0)
        last = min(int((end_offset - 1e-9) // width), slots - 1)
        for slot_index in range(first, last + 1):
            slot_start = slot_index * width
            overlap = min(end_offset, slot_start + width) - max(start_offset, slot_start)
            if overlap <= 0:
                continue
            coverage[slot_index] = max(coverage[slot_index] - overlap / width, 0.0)
    return coverage


def _seconds_into_day(moment: datetime, day: date) -> float:
    """本地时刻相对该本地日零点的秒偏移；跨日的空白在两端各自截断。"""

    start_of_day = datetime.combine(day, datetime.min.time(), tzinfo=moment.tzinfo)
    offset = (moment - start_of_day).total_seconds()
    return min(max(offset, 0.0), _SECONDS_PER_DAY)


def clamp_gap_to_day(gap: ObservedGap, day: date) -> ObservedGap | None:
    """把一段可能跨日的空白裁到某一天之内；不相交返回 None。"""

    start_of_day = datetime.combine(day, datetime.min.time(), tzinfo=gap.started_at.tzinfo)
    end_of_day = start_of_day + timedelta(days=1)
    started = max(gap.started_at, start_of_day)
    ended = min(gap.ended_at, end_of_day)
    if ended <= started:
        return None
    return ObservedGap(started_at=started, ended_at=ended)


__all__ = [
    "NodeLedger",
    "accumulate",
    "all_day_marginals",
    "clamp_gap_to_day",
    "day_coverage",
    "group_gaps_by_day",
    "pool_indexes",
    "decay_weight",
    "derive",
]
