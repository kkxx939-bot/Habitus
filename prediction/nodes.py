"""钟面节点：计数、曝光、三层收缩，以及由它们派生的三种率。

零 IO 纯函数——输入是已归一化的行为与空白，输出是可直接发布的统计量。全部正确性都能在这里
穷举验证（见 ``TODO(PRED-TREE-001)`` 的估计纪律一节）。

三条卫生里的两条落在本模块：**曝光分母**（只数被观测到的槽，两类空白都扣）与**收缩**
（稀疏槽会出 0/1 极端值）。第三条"转移删失"在 ``edges.py``。
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from prediction.config import PredictionTreeConfig
from prediction.errors import PredictionTreeError
from prediction.model import (
    MINUTES_PER_DAY,
    WEEKDAYS,
    DayCurve,
    NodeCounts,
    NodeStatistics,
    ObservedAction,
    ObservedGap,
    SlotExposure,
    SlotKey,
)

_SECONDS_PER_DAY = 86_400.0
# 空的"这一天命中的槽"；显式标注元素类型，否则 ``frozenset()`` 推成 ``frozenset[Never]``，
# 与 ``set[int]`` 求交集在 mypy 下报 operator 错。
_NO_SLOTS: frozenset[int] = frozenset()
# 密集曲线的发布精度：三条曲线各 96 个数，全精度浮点的文本表示会让发布体积翻倍。
# **按有效数字舍入，不按小数位**：收缩链在稀疏格子上会给出 4e-8 这种正确的极小值
# （池化邻域里 450 个观测日 0 次发生，率就该远低于全天基线），按小数位舍入会把它抹成
# 精确的 0.0——而 ``query.marginal_at`` 的契约明写"退回不是 0，给 0 会让任何依赖它的
# 比值直接爆掉"。实测：365 天数据上一条曲线 91/96 个值被小数位舍入抹成 0。
_PUBLISHED_DIGITS = 6


def _published(value: float) -> float:
    """把一个率舍到发布精度；有效数字口径，极小值不会被抹成 0。"""

    return float(f"{value:.{_PUBLISHED_DIGITS}g}")


def estimation_constants() -> dict[str, object]:
    """住在本模块里、但**会改变发布数字**的常量；它们随配置一起进发布指纹。

    它们没有进 ``PredictionTreeConfig`` 是因为那 14 个参数是"必须在真实数据上定档"的一组，
    而这两个是形状：档位只要量级对，精度只要细于任何下游门槛。但"不需要定档"不等于
    "改了不算另一套统计"。
    """

    return {
        "period_scale_buckets": list(_PERIOD_SCALE_BUCKETS),
        "published_digits": _PUBLISHED_DIGITS,
    }


def reconcile_gaps(
    actions: Sequence[ObservedAction], gaps: Sequence[ObservedGap]
) -> tuple[ObservedGap, ...]:
    """消解 occurrence 与空白的重叠，产出**唯一**的一份空白账。

    这里是"一条记录一个解读"的落点。以前没有这一步，同一段空白被两个消费者读成两回事：
    曝光那边看到这一槽产出过 occurrence 就把覆盖记满（"我们明明看见了"），边那边照旧把它
    当洞、把那一对转移删失（"这中间可能漏了别的"）。两种读法各自都说得通，但存储与读侧
    因此有了两份真相，每个新消费者都要重新面对同一道选择题。

    规则按 ``watched`` 二分，两个消费者此后读同一份账：

    - **在看（「没读懂」）**：它断言的正是"这段读不出行为"。这段里若真读出了一条行为的起点，
      这句断言就被证伪了——**整段作废**。作废是唯一不需要编造宽度的规则：只挖掉"那一瞬"是
      零测度、等于没挖；挖"可见跨度"要 ``last_observed_at``，而瞬时行为的跨度仍然是零。
    - **没在看（「未观测」）**：里面不可能读出行为。真出现了，那是上游把"没在看"与"看见了"
      同时写进了树——本层**不消解**，让它在 ``_accumulate_action`` 以明确的矛盾报出来。
      这类空白目前树里一条都没有（上游覆盖契约尚未接入），所以这条是前瞻护栏。

    代价说清楚：作废之后，那段时间里可能还有**没读出来**的行为，会被当成"没有"。真实数据上
    这个代价很小——DAY1 的 21 段空白全是「没读懂」、非零宽度的只有 4 段共 137.8 秒，其中被
    证伪的 1 段宽 21.9 秒。
    """

    starts = sorted(action.started_at.timestamp() for action in actions)
    kept: list[ObservedGap] = []
    for gap in gaps:
        if gap.watched:
            begin = gap.started_at.timestamp()
            index = bisect_left(starts, begin)
            if index < len(starts) and starts[index] < gap.ended_at.timestamp():
                continue
        kept.append(gap)
    return tuple(kept)


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
    # 覆盖只由空白决定，**没有任何特判**：occurrence 与空白的重叠已经在 ``reconcile_gaps``
    # 一处消解掉了（调用方是 ``builder``，它把同一份账同时交给本模块与 ``edges``）。
    # 早先这里有一条"这一槽产出过 occurrence 就把覆盖记满"的特判，它有两个副作用：A 的一次
    # 发生会替 B 抹掉同一槽里的洞；而且它掩盖了分子分母的口径不一致（见 ``_accumulate_action``）。
    coverage_by_day = _coverage_by_day(days, gap_days, slot_minutes=config.slot_minutes)
    for day in days:
        age = float((reference - day).days)
        long_weight = decay_weight(age, config.decay_half_life_days)
        short_weight = decay_weight(age, config.recent_half_life_days)
        coverage = coverage_by_day[day]
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

    **全部计数与曝光共用同一个度量** ``m = 衰减权重 × 覆盖比例``：一天对一个格子贡献多少
    证据，只有这一个答案。早先分子按整份权重记、分母按覆盖比例记，覆盖 0.78 的槽里发生一次
    会算出 1.28 的"概率"再被夹到 1.0；那时它没暴露，是因为 ``accumulate`` 另有一条"有
    occurrence 就把覆盖记满"的特判正好把分母也顶满——两个特判互相撑着。口径统一之后三种率
    天然落在 [0, 1]，``_probability`` 的夹紧退化成防御性护栏。
    """

    weekday = day.weekday()
    first_slot = min(slots_hit)
    for slot_index, repetitions in slots_hit.items():
        covered = coverage[slot_index]
        if covered <= 0.0:
            # 走到这里说明上游同时断言了"这段没在看"与"这一刻看见了这条行为"。这是**上游的
            # 矛盾数据**，不是本层该消解的重叠——「没读懂」那一类已经在 ``reconcile_gaps``
            # 里被证伪作废，能剩下的只有「未观测」。在这里报出来，比放它进去、再让发布期的
            # "有计数无曝光"以一句不知所云的内部错误炸掉要好。
            raise PredictionTreeError(
                f"{action} was read at {day} slot {slot_index} inside an unobserved gap; "
                "the upstream coverage signal contradicts the behaviour tree"
            )
        key = (SlotKey(weekday=weekday, slot=slot_index), action)
        counts[key] = counts.get(key, NodeCounts()).plus(
            occurred_days=long_weight * covered,
            raw_occurrences=long_weight * covered * repetitions,
            recent_days=short_weight * covered,
            first_days=long_weight * covered if slot_index == first_slot else 0.0,
        )
    for slot_index in range(first_slot + 1, slots):
        covered = coverage[slot_index]
        if covered <= 0.0:
            continue
        key = (SlotKey(weekday=weekday, slot=slot_index), action)
        counts[key] = counts.get(key, NodeCounts()).plus(earlier_days=long_weight * covered)


@dataclass(frozen=True)
class DerivedNodes:
    """一趟派生的全部产物：格子、曲线、``lift_周几`` 的分母。"""

    cells: Mapping[tuple[SlotKey, str], NodeStatistics]
    curves: Mapping[tuple[int, str], DayCurve]
    weekday_baselines: Mapping[str, tuple[float, ...]]


def derive_all(
    ledger: NodeLedger,
    *,
    config: PredictionTreeConfig,
    trends: Mapping[tuple[int, str], tuple[float | None, float]],
    completion: Mapping[tuple[int, str], Sequence[float]],
) -> DerivedNodes:
    """一趟算出格子与曲线。

    收缩顺序**从假设最弱的借用开始**（PRED-TREE-001 修正过的技术错误）：

        (周几, 槽) ← (周几, 池化邻域) ← (跨周几, 池化邻域) ← 该动作全天平均

    时间邻域借用只假设"相邻时刻的率相近"，跨周几借用假设"各周几的率相近"——后者对周规律
    行为直接错误，放在第一步会系统性把周规律压平。

    **率只有曲线一份**：格子上只留原始账本与这一格的机会数。率同时挂在两处时，"同一个问题
    两种答案"就永远只差一次浮点累加次序的分歧——而那正是本轮要修的毛病，不能在修的过程中
    又造一个。

    ``completion`` 是 ``completion_curves`` 算好的累积率（两级分解 ``π · F(t)``），本函数只负责
    装配、不重算：累积率与危险率描述的是同一个"当天第一次"的过程，各记一本账正是上一版
    "同一个量两种答案"的来源。
    """

    slots = config.slots_per_day
    all_day = _all_day_rates(ledger, config=config)
    grids = _grids(ledger, config=config)
    pairs = {(key.weekday, action) for key, action in ledger.counts}
    cells: dict[tuple[SlotKey, str], NodeStatistics] = {}
    curves: dict[tuple[int, str], DayCurve] = {}
    for action in sorted({action for _weekday, action in pairs}):
        marginal = _dense_chain(
            grids,
            action=action,
            config=config,
            numerator=lambda item: item.occurred_days,
            all_day_rate=all_day.marginal[action],
            risk_adjustment=False,
        )
        hazard = _dense_chain(
            grids,
            action=action,
            config=config,
            numerator=lambda item: item.first_days,
            all_day_rate=all_day.hazard[action],
            risk_adjustment=True,
        )
        for weekday in range(WEEKDAYS):
            if (weekday, action) not in pairs:
                continue
            for slot in range(slots):
                key = SlotKey(weekday=weekday, slot=slot)
                counts = ledger.counts.get((key, action))
                exposure = ledger.exposure.get(key)
                if counts is not None and (exposure is None or exposure.observed_days <= 0.0):
                    # 到这里只剩"我们自己的账不平"这一种可能：``_accumulate_action`` 对
                    # 覆盖为零的槽直接硬失败，只有 earlier_days 的槽也只记在有覆盖的槽上。
                    # 两条都守住了还出现，就是实现错了。
                    raise PredictionTreeError(f"node {action} at {key} has counts without exposure")
                if counts is None or exposure is None or counts.occurred_days <= 0.0:
                    continue
                cells[(key, action)] = NodeStatistics(
                    n_eff=exposure.observed_days, counts=counts
                )
            cumulative = completion.get((weekday, action))
            if cumulative is None or len(cumulative) != slots:
                # 两处都从同一批行为出发，键集必须一致；对不上就是装配错了，不能悄悄发一条
                # 全零的曲线——那会被读侧读成"这个周几从来不做这件事"。
                raise PredictionTreeError(
                    f"no completion curve for {action} on weekday {weekday}"
                )
            trend, trend_n_eff = trends.get((weekday, action), (None, 0.0))
            curves[(weekday, action)] = DayCurve(
                marginal=tuple(marginal[weekday]),
                hazard=tuple(hazard[weekday]),
                cumulative=tuple(cumulative),
                trend=trend,
                trend_n_eff=trend_n_eff,
            )
    weekday_baselines = {
        action: tuple(
            _published(all_day.per_slot_cross_weekday.get((slot, action), 0.0))
            for slot in range(slots)
        )
        for action in sorted({action for _weekday, action in pairs})
    }
    return DerivedNodes(cells=cells, curves=curves, weekday_baselines=weekday_baselines)


@dataclass(frozen=True)
class _Grids:
    """把账本摊成按 (周几, 槽) 索引的数组，整棵树只做一次。

    与动作无关的部分（曝光、以及它的环形池化和）在这里算完就复用。早先每个动作都重新
    构造一遍 7×96 个 ``SlotKey`` 去查字典——实测七天树上 `SlotKey.__post_init__` 被调用
    3,107,904 次，占 ``derive_all`` 的一半以上。提干后同一棵树 2.09s → 0.77s，输出逐位不变。
    """

    exposure: list[list[float]]
    observed: list[list[bool]]
    pooled_exposure: list[list[float]]
    counts: dict[str, dict[tuple[int, int], NodeCounts]]


def _grids(ledger: NodeLedger, *, config: PredictionTreeConfig) -> _Grids:
    slots = config.slots_per_day
    exposure = [[0.0] * slots for _ in range(WEEKDAYS)]
    observed = [[False] * slots for _ in range(WEEKDAYS)]
    for key, seen in ledger.exposure.items():
        exposure[key.weekday][key.slot] = seen.observed_days
        observed[key.weekday][key.slot] = True
    counts: dict[str, dict[tuple[int, int], NodeCounts]] = {}
    for (key, action), tally in ledger.counts.items():
        counts.setdefault(action, {})[(key.weekday, key.slot)] = tally
    half = config.pool_half_width
    return _Grids(
        exposure=exposure,
        observed=observed,
        pooled_exposure=[_circular_window_sums(row, half) for row in exposure],
        counts=counts,
    )


def _dense_chain(
    grids: _Grids,
    *,
    action: str,
    config: PredictionTreeConfig,
    numerator: Callable[[NodeCounts], float],
    all_day_rate: float,
    risk_adjustment: bool,
) -> list[list[float]]:
    """一次算出某个动作在**全部** (周几, 槽) 上的三层收缩估计，返回 7 × 槽数 的网格。

    逐槽重跑收缩链是 O(槽 × 池宽 × 7)（真实数据实测：朴素写法一周 10.4 秒、按此外推一年
    约 102 秒）。这里先把每个周几的分子分母摊成数组，再用**环形滑动窗**一次算出全部池化和，
    降到 O(槽)。数值口径与逐槽写法逐字一致：同一个先验构造、同一个伪计数、同样的顺序。
    """

    slots = config.slots_per_day
    cells = grids.counts.get(action, {})
    top = [[0.0] * slots for _ in range(WEEKDAYS)]
    for (weekday, slot), counts in cells.items():
        # 缺曝光记录的槽在分子分母上**一起**跳过——与逐槽写法的 ``if exposure is None: continue``
        # 同一个语义。
        if grids.observed[weekday][slot]:
            top[weekday][slot] = numerator(counts)
    if risk_adjustment:
        # 危险率的分母是风险集：曝光扣掉"当天更早已经做过"的那部分，逐槽先扣再求和。
        bottom = [list(row) for row in grids.exposure]
        for (weekday, slot), counts in cells.items():
            if grids.observed[weekday][slot]:
                bottom[weekday][slot] = max(
                    grids.exposure[weekday][slot] - counts.earlier_days, 0.0
                )
        pooled_bottom = [_circular_window_sums(row, config.pool_half_width) for row in bottom]
    else:
        bottom = grids.exposure
        pooled_bottom = grids.pooled_exposure
    pooled_top = [_circular_window_sums(row, config.pool_half_width) for row in top]
    grid = [[0.0] * slots for _ in range(WEEKDAYS)]
    for slot in range(slots):
        cross = _shrink(
            sum(pooled_top[weekday][slot] for weekday in range(WEEKDAYS)),
            sum(pooled_bottom[weekday][slot] for weekday in range(WEEKDAYS)),
            all_day_rate,
            config.shrink_weekday_to_all_day,
        )
        for weekday in range(WEEKDAYS):
            pooled = _shrink(
                pooled_top[weekday][slot],
                pooled_bottom[weekday][slot],
                cross,
                config.shrink_pool_to_weekday,
            )
            # 格子与曲线读的是**同一个已舍入的值**，所以两者永远逐位相等。
            grid[weekday][slot] = _published(
                _probability(
                    _shrink(
                        top[weekday][slot], bottom[weekday][slot], pooled, config.shrink_slot_to_pool
                    )
                )
            )
    return grid


def _circular_window_sums(values: Sequence[float], half_width: int) -> list[float]:
    """钟面是**环形**的：每个槽的邻域和含跨午夜的那一段。

    结果必须与 ``[sum(values[(i + o) % n] for o in range(-h, h + 1)) for i in range(n)]``
    逐位一致——这是本函数唯一的契约，有一条差分测试钉着它（滑动窗最容易犯的错是窗宽从
    2h+1 塌成 2h 并逐槽累积漂移，那种错误在定性测试下全绿）。

    窗口宽于整圈时朴素写法会带重复项重复计数，滑窗写法不会——两者在那里会静默分叉。
    配置层已经硬拒了这种组合（``2 × pool_half_width < slots_per_day``），所以这里直接
    断言，而不是悄悄走另一条分支。
    """

    total = len(values)
    if half_width <= 0:
        return list(values)
    width = 2 * half_width + 1
    if width > total:
        raise PredictionTreeError(
            "a circular pooling window wider than the clock face would count slots twice"
        )
    window = sum(values[(index - half_width) % total] for index in range(width))
    sums = [0.0] * total
    # 滑动的加减残差会让本该为 0 的窗口和落到 -3e-9 这种微小负值上。今天每个消费者都先加
    # 正的收缩强度再除，所以无害；但代码里没有任何东西保证这一点，夹一下是零成本的。
    sums[0] = max(window, 0.0)
    for slot in range(1, total):
        window += values[(slot + half_width) % total] - values[(slot - half_width - 1) % total]
        sums[slot] = max(window, 0.0)
    return sums


# 趋势的证据窗按行为自身周期缩放时的档位。分档的**单位是"周"**而不是"天"，因为趋势是在
# **周几通道**里算的：循环只走同一个周几的日子，一个窗口里有几次机会由通道决定（每周一次），
# 与行为周期无关——日频和周频在这条通道里机会数完全相同。随周期变的是**命中率**：双周行为
# 隔一个周一才命中一次，要攒同样多的命中就得等两倍长的窗。所以倍数是"每命中一次要等几周"。
#
# 早先按天分档（日 1 / 周 7 / 双周 14 / 月 30）是拿"不分周几"的样本量算的，用在这条通道里
# 正好过度校正一个周期倍数：实测一个每周一的习惯**停了 120 天（连续 17 次没做）**，按天分档
# 读出 0.635——"→0 = 习惯正在消失"的判据在这个值上根本不会触发，而规格写这个数存在的理由
# 正是"没有它，换药停服后会连续误报两个月"。
_DAYS_PER_WEEK = 7.0
_PERIOD_SCALE_BUCKETS: tuple[float, ...] = (1.0, 2.0, 4.0)


def period_scale(period_days: float) -> float:
    """行为的复发周期 → 证据窗的缩放倍数（单位：周几通道里的一次机会 = 一周）。

    **分档而不是用连续值**：中位间隔本身是个噪声估计——每周二健身偶尔跳一周，间隔序列就是
    7、14、7、21，p50 在 7 和 14 之间来回跳；证据窗跟着连续值抖，会自己造出趋势变化。
    """

    if period_days <= 0.0:
        return _PERIOD_SCALE_BUCKETS[0]
    weeks = period_days / _DAYS_PER_WEEK
    for bucket in _PERIOD_SCALE_BUCKETS:
        if weeks <= bucket:
            return bucket
    return _PERIOD_SCALE_BUCKETS[-1]


def completion_curves(
    actions: Sequence[ObservedAction],
    gaps: Sequence[ObservedGap],
    *,
    config: PredictionTreeConfig,
    reference: date,
) -> dict[tuple[int, str], tuple[float, ...]]:
    """累积率：**两级分解** ``CumP(t) = π · F(t)``，每个 (周几, 动作) 一条。

    - ``π``  这个周几到底做不做这件事。样本单位是**天**，所以收缩是标准的一层——先验取该动作
      跨全部周几的日发生率，伪计数复用 ``shrink_weekday_to_all_day``（同一层借用，不新增参数）。
    - ``F``  如果做，几点做：当天**首次**发生的时刻分布，环形池化之后与跨周几的同一分布混合，
      **在发生日上归一化**（总质量恒为 1）。

    为什么不是别的两种写法（2026-09-01 三户 CASAS 各 8 周、前 6 周训练后 2 周留出实测，
    完整数字见 ``TODO(PRED-RATES-002)``）：

    - **裸比值** ``(earlier+first)/E``（旧实现）：没有任何收缩，小样本下直接给 0 或 1。七天数据
      上 196,800 个值**没有一个**落在开区间；六周训练下仍有 38–53% 的曲线末槽恒为 1.0。
      留出 ECE 三户 0.0566 / 0.1163 / 0.0664，从不最好、两户最差。
    - **由危险率累乘** ``1 − Π(1−h)``：形式上最自洽（累积率本就是首次发生的分布函数），但
      收缩后的危险率每槽都有正地板，96 槽连乘会凭空堆出完成概率——七天树末槽中位 0.6723 里
      只有 0.2851 来自真的发生过的槽。留出 ECE 一胜一负，不稳。
    - **本式**：留出 ECE 0.0468 / 0.0809 / 0.0326，两胜、平均最好、从不最差；落在开区间的
      值 61–75%。地板不累积，因为 ``F`` 是归一化的分布，池化与混合只搬运质量、不新增质量。

    ``π`` 与 ``F`` **不单独发布**：树上只发这条乘出来的曲线，形状与旧实现一致，查询层零改动。
    发布之后它是否仍然是一条合法的分布函数（单调不减、落在 [0,1]），由 ``codec.decode`` 校验。
    """

    slots = config.slots_per_day
    alpha = config.shrink_weekday_to_all_day
    per_day = _group_by_day(actions, config=config)
    gap_days = group_gaps_by_day(gaps)
    days = _calendar_days(per_day.keys() | gap_days.keys(), reference)
    coverage_by_day = _coverage_by_day(days, gap_days, slot_minutes=config.slot_minutes)
    # 一天的证据量与格子那边同源：衰减权重 × 这一天平均看了多少。半天没在看的一天只算半天，
    # 分子分母同乘，所以"那天做了"仍然读成整整一次发生。
    weight = {
        day: decay_weight(float((reference - day).days), config.decay_half_life_days)
        * (sum(coverage_by_day[day]) / slots)
        for day in days
    }
    days_by_weekday: dict[int, list[date]] = {}
    for day in days:
        days_by_weekday.setdefault(day.weekday(), []).append(day)
    first_slot: dict[tuple[date, str], int] = {}
    for day, by_action in per_day.items():
        for action, slots_hit in by_action.items():
            first_slot[(day, action)] = min(slots_hit)

    total = sum(weight.values())
    curves: dict[tuple[int, str], tuple[float, ...]] = {}
    for action in sorted({action for _day, action in first_slot}):
        occurred = [day for day in days if (day, action) in first_slot]
        prior = sum(weight[day] for day in occurred) / total if total > 0.0 else 0.0
        cross = [0.0] * slots
        for day in occurred:
            cross[first_slot[(day, action)]] += weight[day]
        cross_shape = _normalized(_circular_window_sums(cross, config.pool_half_width))
        for weekday, weekday_days in days_by_weekday.items():
            own_days = [day for day in weekday_days if (day, action) in first_slot]
            if not own_days:
                continue
            seen = sum(weight[day] for day in weekday_days)
            hit = sum(weight[day] for day in own_days)
            chance = (hit + alpha * prior) / (seen + alpha)
            own = [0.0] * slots
            for day in own_days:
                own[first_slot[(day, action)]] += weight[day]
            pooled = _circular_window_sums(own, config.pool_half_width)
            mass = sum(pooled)
            shape = _normalized(
                [(pooled[slot] + alpha * cross_shape[slot]) for slot in range(slots)]
                if mass + alpha > 0.0
                else cross_shape
            )
            running = 0.0
            curve: list[float] = []
            for slot in range(slots):
                running += shape[slot]
                curve.append(_published(min(chance * running, 1.0)))
            curves[(weekday, action)] = tuple(curve)
    return curves


def _normalized(mass: Sequence[float]) -> list[float]:
    """把一列非负质量归一成分布；全零时退成均匀分布（唯一不偏向任何时刻的答案）。"""

    total = sum(mass)
    if total <= 0.0:
        return [1.0 / len(mass)] * len(mass)
    return [value / total for value in mass]


def pooled_trends(
    actions: Sequence[ObservedAction],
    gaps: Sequence[ObservedGap],
    *,
    config: PredictionTreeConfig,
    reference: date,
    periods: Mapping[str, float],
) -> dict[tuple[int, str], tuple[float | None, float]]:
    """每个 (周几, 动作) 的变化方向与它的证据量。

    两处与格子级的旧写法不同，都是必需的：

    - **在池化邻域上算，不在单格上算。** 单个 (周几, 槽) 一个月只有 4–5 次机会，在那个样本
      量上比"近期 ÷ 长期"得到的是噪声。邻域中心取该 (周几, 动作) 发生最多的那个槽——回答的
      是"这个行为在它自己的时间范围内，最近比长期多了还是少了"，早饭从 7:15 挪到 9:00 仍算
      同一件事发生了。
    - **两个证据窗按行为自身的复发周期缩放。** 固定 τ 对所有行为一视同仁：日频行为在 14 天
      近期窗里有约 14 个样本，周频只有 2 个，月频只有 0.5 个——恰恰把最该看趋势的低频行为
      压成噪声。这条原则仓库里已经用过一次（``recurrence_half_life_days`` 就是为此从钟面的
      τ 里解耦出来的），这里把它补到趋势上。

    返回 ``(趋势, 长窗里的加权发生次数)``。**不划线**：证据够不够由读侧拿伴随值判断，本层
    只在趋势没有定义时给 None。第二个返回值的单位是**这个行为自己的窗口内的加权命中次数**
    （窗长已乘过 ``period_scale``），跨行为不可直接比较——口径与代价见 ``DayCurve.trend_n_eff``
    的注释——两种情形：``periods`` 里没有这个行为（复发证据不足，量不出
    它的节奏，也就无从给它定窗），或者该周几的邻域整段没有可观测的机会。

    "没有复发证据就给 None"这一条不能省：``recurrence.derive`` 会丢掉超过
    ``recurrence_window_days`` 的间隔，于是一个季频行为在 ``periods`` 里根本不出现；若把它
    当成日频（最短的窗），实测读出 trend 2.744（"新习惯正在建立"）而它一点没变，换个基准日
    相位一变就翻到接近 0——纯噪声，正是这个字段要消灭的东西。
    """

    slots = config.slots_per_day
    per_day = _group_by_day(actions, config=config)
    gap_days = group_gaps_by_day(gaps)
    days = _calendar_days(per_day.keys() | gap_days.keys(), reference)
    coverage_by_day = _coverage_by_day(days, gap_days, slot_minutes=config.slot_minutes)
    # 按周几分组：循环只用得上七分之一的日子，逐对再走一遍全部日历日是纯空转。
    days_by_weekday: dict[int, list[date]] = {}
    for day in days:
        days_by_weekday.setdefault(day.weekday(), []).append(day)

    occurred: dict[tuple[int, str], dict[date, set[int]]] = {}
    for day in days:
        for action, slots_hit in per_day.get(day, {}).items():
            occurred.setdefault((day.weekday(), action), {})[day] = set(slots_hit)

    trends: dict[tuple[int, str], tuple[float | None, float]] = {}
    for (weekday, action), by_day in occurred.items():
        period = periods.get(action)
        if period is None:
            trends[(weekday, action)] = (None, 0.0)
            continue
        tally: dict[int, int] = {}
        for hit_slots in by_day.values():
            for slot in hit_slots:
                tally[slot] = tally.get(slot, 0) + 1
        center = max(sorted(tally), key=lambda slot: tally[slot])
        pool = set(pool_indexes(center, config.pool_half_width, slots))
        scale = period_scale(period)
        short_life = config.recent_half_life_days * scale
        long_life = config.decay_half_life_days * scale
        short_hits = short_seen = long_hits = long_seen = 0.0
        for day in days_by_weekday.get(weekday, ()):
            # 分子是"邻域里有没有发生"（0/1），分母就必须是"邻域里有没有可观测的机会"，
            # 而不是邻域覆盖的平均值——平均值口径下隐含的率能到 1/covered（实测 5.0），
            # 同一个三十周没变过的习惯会因为空洞落在哪而让趋势摆动 24%。
            covered = max(coverage_by_day[day][slot] for slot in pool)
            if covered <= 0.0:
                continue
            age = float((reference - day).days)
            hit = 1.0 if pool & by_day.get(day, _NO_SLOTS) else 0.0
            recent_weight = decay_weight(age, short_life)
            long_weight = decay_weight(age, long_life)
            short_hits += recent_weight * hit
            short_seen += recent_weight * covered
            long_hits += long_weight * hit
            long_seen += long_weight * covered
        if short_seen <= 0.0 or long_seen <= 0.0 or long_hits <= 0.0:
            # 防御性：结构上到不了这里——(周几, 动作) 这个键只在该动作**真的在那个周几发生过**
            # 时才存在，而发生过的槽在覆盖里按"在看"记满，所以邻域至少有一处可观测、至少有
            # 一次命中。留着是因为哪天覆盖口径变了，这里该给 None 而不是除零。
            trends[(weekday, action)] = (None, long_hits)
            continue
        long_rate = long_hits / long_seen
        trends[(weekday, action)] = ((short_hits / short_seen) / long_rate, long_hits)
    return trends


def _coverage_by_day(
    days: Sequence[date],
    gap_days: Mapping[date, Sequence[ObservedGap]],
    *,
    slot_minutes: int,
) -> dict[date, list[float]]:
    """每天每槽有多少比例是真的在看；**唯一**的那份口径。

    ``accumulate`` 与 ``pooled_trends`` 都从这里取，两处不再各写一遍。入参里的空白必须是
    ``reconcile_gaps`` 消解过的——本函数只做积分，不解释重叠。
    """

    return {
        day: day_coverage(day, gap_days.get(day, ()), slot_minutes) for day in days
    }


def all_day_marginals(ledger: NodeLedger, *, config: PredictionTreeConfig) -> dict[str, float]:
    """每个动作**不看时刻**的整体发生率。

    两个用途：收缩链的顶层，以及 ``lift_全天`` 的分母。它同时是读侧**最后**的兜底——只在
    "这个周几从来没做过这个动作"、连曲线都没有的时候才用得上；这个周几做过的，每一个槽都
    由密集曲线回答，不再看格子有没有。
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
    # 裁切只动边界，``watched`` 是这段空白的身份，必须原样带过去。
    return ObservedGap(started_at=started, ended_at=ended, watched=gap.watched)


__all__ = [
    "DerivedNodes",
    "estimation_constants",
    "NodeLedger",
    "accumulate",
    "derive_all",
    "period_scale",
    "pooled_trends",
    "all_day_marginals",
    "clamp_gap_to_day",
    "day_coverage",
    "group_gaps_by_day",
    "pool_indexes",
    "decay_weight",
]
