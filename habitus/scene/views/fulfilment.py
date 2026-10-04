"""无节律型的读数：兑现率与兑现间隔（2026-09-27 裁定）。2026-09-30 裁定五：这类读数**未校准**，收口规则与双向验证做完前不作数。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from habitus.scene.hypotheses.model import Hypothesis
from habitus.scene.ledger.model import Claim, Outcome
from habitus.scene.views.accounts import Account, Accumulation, settled_pairs
from habitus.scene.views.config import ViewsConfig
from habitus.scene.views.stats import Interval, jeffreys_interval, median
from habitus.scene.views.strength import Strength, usable_interval


@dataclass(frozen=True)
class FulfilmentReading:
    """无节律型（``consequent_peak is None``）的读数（用户 09-27 裁定）：只有兑现率与兑现间隔，不读 p1−p0、不读类型。

    分母 = 兑现 + 释放 + **仍立着且已经立了超过对照间隔的**（裁定原文的三项）。还没立到对照间隔的不进分母——
    "约了今晚去打球，现在下午三点"既不算兑现也不算没兑现。

    **被生命周期关掉的（``CENSORED``）只计数、不进分母**：``close_claim`` 的关闭理由是自由文本，账本分不出
    "约了三个月没去，不等了"（是一次没兑现，该进分母）与"这条假设整个退役了"（跟他去没去无关，不该进）。
    分不出就一律算会在后者上把兑现率压低，所以照裁定的三项算，这一项等生命周期那份定出"关闭有哪几种"再定。
    今天它恒为 0（``close_claim`` 没有生产调用者）。

    ``median_hours`` 是**到达者**的中位（没兑现的那些不混进来，由 ``rate`` 与 ``longest_standing_hours`` 表达）；
    ``control_hours`` = 后件本来多久真的发生一次 = 相邻机会间隔 ÷ 峰概率（见 ``expected_wait_hours``），没快照就没对照。
    """

    accumulation: Accumulation
    fulfilled: int
    released: int
    #: 被生命周期关掉的条数。**只报数、不进分母**（见上）；今天恒为 0。
    closed: int
    standing: int
    standing_total: int
    rate: float | None
    interval: Interval | None
    median_hours: float | None
    control_hours: float | None
    longest_standing_hours: float | None
    #: 读数单位（与 ``Strength.unit`` 同一个口：分层比差时查"影响大"的分界用）。
    unit: str = "fulfilment"

    @property
    def sufficient(self) -> bool:
        return self.accumulation.sufficient

    @property
    def unrefuted(self) -> bool:
        """过门槛 ∧ 兑现率算出了区间 ∧ 有对照且中位兑现间隔短于对照间隔（"比它平时自己来得快"）。"""

        return (
            self.sufficient
            and self.interval is not None
            and self.control_hours is not None
            and self.median_hours is not None
            and self.median_hours < self.control_hours
        )


#: 一层的读数：节律型给强度（p1−p0），无节律型给兑现率。两者都有 ``.interval`` 与 ``.accumulation``，
#: 所以"两层比区间"的判据共用一套。
LayerReading = Strength | FulfilmentReading


def expected_wait_hours(claims: Sequence[Claim]) -> float | None:
    """对照：后件**本来多久真的发生一次** = 相邻机会间隔 ÷ 峰概率（小时）。没有一份带两个机会的快照 → None。

    问的不是"它多久有一次机会"，而是"它自己多久来一次"（用户 09-27 裁定）：打球的峰一天一个、每个峰的概率 0.20，
    机会间隔 24 小时，真的去打球的期望等待是 24 / 0.20 = 120 小时。按间隔本身当对照的话，约球之后 52.8 小时去了
    会被判成"不如平时快"——而它平时要等五天。几何等待的直接算法，与节律型的 ``p0`` 用同一份快照数据
    （机会的 ``probability``），不新加字段、不碰预测树。
    """

    waits: list[float] = []
    for claim in claims:
        if claim.control is None or len(claim.control.opportunities) < 2:
            continue
        items = claim.control.opportunities
        gaps = [(later.at - earlier.at).total_seconds() / 3600.0 for earlier, later in zip(items, items[1:], strict=False)]
        probabilities = [item.probability for item in items if item.probability is not None]
        if not probabilities:
            continue
        waits.append(median(gaps) / median(probabilities))
    return median(waits) if waits else None


def fulfilment_of(
    account: Account,
    hypothesis: Hypothesis,
    config: ViewsConfig,
    *,
    gate: tuple[int, int] | None = None,
    now: datetime | None = None,
) -> FulfilmentReading:
    """读无节律型的账。``now`` 不给就不把仍立着的算进分母（没有"立了多久"这个量，宁可不出数也不抬高兑现率）。"""

    if not hypothesis.is_open_ended:
        raise ValueError("fulfilment_of reads open-ended hypotheses (consequent_peak=None) only")
    resolved_gate = gate or config.main_gate
    pairs = settled_pairs(account)
    fulfilled = [(claim, item) for claim, item in pairs if item.outcome is Outcome.OCCURRED]
    released = sum(1 for _c, item in pairs if item.outcome is Outcome.RELEASED)
    # 被生命周期关掉的只报数，不进分母（见类 docstring）。
    shut = [(claim, item) for claim, item in pairs if item.outcome is Outcome.CENSORED]
    control_hours = expected_wait_hours(account.claims)
    standing_claims = tuple(claim for claim in account.claims if claim.ref not in account.settlements)
    waited = [max(0.0, (now - claim.anchor).total_seconds() / 3600.0) for claim in standing_claims] if now is not None else []
    # 没有对照时把全部仍立着的都算进分母（保守：宁可把兑现率读低，不凭空抬高）。``now`` 没给就一条都不算——
    # "立了多久"这个量不存在时，猜它超没超过对照间隔只会抬高兑现率。
    threshold = control_hours if control_hours is not None else 0.0
    overdue = tuple(claim for claim, hours in zip(standing_claims, waited, strict=False) if hours > threshold)
    shut_claims = {claim.ref for claim, _s in shut}
    counted: list[Claim] = [claim for claim, _s in pairs if claim.ref not in shut_claims]
    counted.extend(overdue)
    blocks = {account.block_of(claim) for claim in counted} if counted else set()
    accumulation = Accumulation(
        count=len(counted), blocks=len(blocks), block_hours=account.block_hours, needed_count=resolved_gate[0], needed_blocks=resolved_gate[1]
    )
    latencies = [item.latency_hours for _c, item in fulfilled if item.latency_hours is not None]
    rate = len(fulfilled) / len(counted) if counted else None
    interval = None
    if counted and accumulation.sufficient:
        interval = usable_interval(jeffreys_interval(len(fulfilled), len(counted) - len(fulfilled), level=config.interval_level))
    return FulfilmentReading(
        accumulation=accumulation,
        fulfilled=len(fulfilled),
        released=released,
        closed=len(shut),
        standing=len(overdue),
        standing_total=len(standing_claims),
        rate=rate,
        interval=interval,
        median_hours=median(latencies) if latencies else None,
        control_hours=control_hours,
        longest_standing_hours=max(waited) if waited else None,
    )


__all__ = ["FulfilmentReading", "LayerReading", "expected_wait_hours", "fulfilment_of"]
