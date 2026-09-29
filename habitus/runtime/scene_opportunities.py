"""机会口：从预测树的 ``marginal`` 曲线上取峰，答"这个后件概念在锚之后的前几次机会"。

**为什么在 runtime 而不在 scene**：`scene` 不许 import `prediction`（架构测试按传递闭包钉死，
`test_dependency_boundaries` 里那句"新语义树要对照预测树的期望时，由组合根注入 callable"）。所以这座桥
住在组合根，scene 只见到 ``OpportunityProvider`` 这个协议。

**一次机会 = 一个峰**（形状稿 §1）：
- 峰 = 曲线上连续一段 ``marginal`` 高于该行为**当日均值**的槽；
- ``at`` = 峰内 ``marginal`` 最大的那个槽的中心时刻（峰的重心落在最可能的那一刻，而不是几何中点）；
- ``span`` = [峰第一个槽的开始, 峰最后一个槽的结束)；
- ``probability`` = 峰内 ``marginal`` 之和，截到 1（= 这个峰上至少开始一次的期望）；
- **第 1 次机会 = 第一个 ``span.end > anchor`` 的峰**——锚正在其中的峰也算（评审 A-2：按"峰的开始晚于锚"
  铺会把正在进行的峰丢掉，时刻差读成 −1210 分而真值是 +230）。

**概念 → 曲线**：树按 ``kind_token`` 存曲线，所以先拿 `scene.views.kinds` 数出这个概念命中过哪些 kind，
把那几条曲线**逐槽相加、截到 1**。叶子概念多半只有一个 kind，相加退化成"就用那条"；聚合概念才真的相加。
相加的依据是 B13（同一槽近似互斥）。

一天铺不出峰（这个周几从没做过 → 曲线缺失）就跳过那一天，继续往后铺；连着 ``max_days`` 天都铺不出就
认了、返回已有的（可能一个都没有 → ``None``，承诺按"要不到对照"开）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from habitus.prediction.model import MINUTES_PER_DAY, PredictionTree
from habitus.scene.ledger.model import (
    MAX_SNAPSHOT_OPPORTUNITIES,
    Opportunity,
    OpportunityRequest,
    OpportunitySnapshot,
    WindowSpan,
)

#: 往后最多翻多少天去找机会。低频后件（半年一次的就诊）在树上多半没有峰，翻再多天也没有——
#: 这是保护闸，不是设计量：翻不出来就让承诺以 ``control=None`` 开着，由无节律型那套读兑现率。
MAX_LOOKAHEAD_DAYS = 30
#: 一个峰至少要有多大概率才算得上"机会"。**没有这条下限，"高于当日均值"在稀疏曲线上会退化**：
#: 2026-09-29 探针实测，一个 45 天里 97 次的行为在 96 槽的钟面上均值只有 0.0107，于是 6 个**单槽**
#: （质量 0.033–0.046）跟着两个真峰（质量各 0.31，08:00–10:00 与 12:00–13:45）一起被当成峰——
#: 一天报出 8 个"机会"，12 个概念合计 119 个。后果是一条链：伪峰 → 逐峰各一条假设 → 一个后件打满
#: 24 本账 → 账按"第 7 次机会"分层、样本被切碎 → 永远攒不够 3 次。
#: 0.10 是**参考档**（探针数据上刚好滤掉全部单槽伪峰、留下两个真峰），真实数据上按"一天真有几次"复核。
MIN_PEAK_MASS = 0.10


@dataclass(frozen=True)
class Peak:
    """曲线上的一个峰：起止槽（含两端）、中心槽、峰内 marginal 之和。"""

    first_slot: int
    last_slot: int
    centre_slot: int
    mass: float


def merged_marginal(tree: PredictionTree, weekday: int, kinds: Sequence[str]) -> tuple[float, ...] | None:
    """这个概念在这个周几的曲线：它命中过的那几个 kind 的 ``marginal`` 逐槽相加、截到 1。

    一个 kind 都没有曲线（这个周几从没做过）→ ``None``，调用方跳过这一天。
    """

    curves = [tree.curves[(weekday, kind)] for kind in kinds if (weekday, kind) in tree.curves]
    if not curves:
        return None
    slots = len(curves[0].marginal)
    return tuple(min(1.0, sum(curve.marginal[slot] for curve in curves)) for slot in range(slots))


def peaks_of(curve: Sequence[float], *, floor: float = MIN_PEAK_MASS) -> tuple[Peak, ...]:
    """连续一段高于当日均值、**且峰内质量够得上 ``floor``** 的槽 = 一个峰。

    两道判据各管一件事：高于均值管"这一段比别处更可能"，``floor`` 管"它值得叫一次机会"。
    只有前者时稀疏曲线会把每个略高于均值的单槽都算成机会（见 ``MIN_PEAK_MASS`` 的实测）。
    均值为 0（整条曲线都是 0）时没有峰。
    """

    average = sum(curve) / len(curve) if curve else 0.0
    if average <= 0.0:
        return ()
    found: list[Peak] = []
    start: int | None = None
    for slot, value in enumerate(curve):
        if value > average:
            start = slot if start is None else start
            continue
        if start is not None:
            found.append(_peak(curve, start, slot - 1))
            start = None
    if start is not None:
        found.append(_peak(curve, start, len(curve) - 1))
    return tuple(peak for peak in found if peak.mass >= floor)


def _peak(curve: Sequence[float], first: int, last: int) -> Peak:
    centre = max(range(first, last + 1), key=lambda slot: (curve[slot], -slot))
    return Peak(first_slot=first, last_slot=last, centre_slot=centre, mass=sum(curve[first : last + 1]))


class TreeOpportunities:
    """按 ``OpportunityProvider`` 协议实现的机会口。组合根在**重建树之后**构造它（横切第 3 条）。

    ``kinds`` 是 `scene.views.kinds.kinds_by_concept` 的产物：概念身份 → 要相加的那几条曲线的键。
    概念不在里面（从没命中过任何 kind）→ 返回 ``None``，承诺以"要不到对照"开。
    """

    def __init__(
        self,
        tree: PredictionTree,
        kinds: Mapping[str, tuple[str, ...]],
        *,
        generation: str,
        timezone_of: datetime | None = None,
        max_lookahead_days: int = MAX_LOOKAHEAD_DAYS,
    ) -> None:
        if not isinstance(tree, PredictionTree):
            raise TypeError("tree must be a PredictionTree")
        if isinstance(max_lookahead_days, bool) or not isinstance(max_lookahead_days, int) or max_lookahead_days <= 0:
            raise ValueError("max_lookahead_days must be a positive integer")
        if not isinstance(generation, str) or not generation.strip():
            raise ValueError("generation must be non-empty text")
        self.tree = tree
        self.kinds = dict(kinds)
        self.generation = generation
        self.max_lookahead_days = max_lookahead_days
        self._reference = timezone_of

    def opportunities(self, request: OpportunityRequest) -> OpportunitySnapshot | None:
        kinds = self.kinds.get(request.consequent)
        if not kinds:
            return None
        wanted = min(max(1, request.count), MAX_SNAPSHOT_OPPORTUNITIES)
        anchor = request.anchor
        found: list[Opportunity] = []
        day = anchor.date()
        for _step in range(self.max_lookahead_days):
            for item in self._day_opportunities(day, kinds, anchor):
                if item.span.end > anchor:
                    found.append(item)
                    if len(found) == wanted:
                        return OpportunitySnapshot(generation=self.generation, opportunities=tuple(found))
            day += timedelta(days=1)
        return OpportunitySnapshot(generation=self.generation, opportunities=tuple(found)) if found else None

    def _day_opportunities(self, day: date, kinds: Sequence[str], anchor: datetime) -> tuple[Opportunity, ...]:
        curve = merged_marginal(self.tree, day.weekday(), kinds)
        if curve is None:
            return ()
        minutes = MINUTES_PER_DAY // len(curve)
        return tuple(
            Opportunity(
                at=self._moment(day, peak.centre_slot * minutes + minutes // 2, anchor),
                span=WindowSpan(
                    self._moment(day, peak.first_slot * minutes, anchor),
                    self._moment(day, (peak.last_slot + 1) * minutes, anchor),
                ),
                probability=min(1.0, peak.mass),
            )
            for peak in peaks_of(curve)
        )

    def _moment(self, day: date, minutes: int, anchor: datetime) -> datetime:
        """槽位 → 本地时刻。时区跟着锚走（承诺上的时间一律本地时间 + 显式偏移）。"""

        reference = self._reference if self._reference is not None else anchor
        return datetime.combine(day, datetime.min.time(), tzinfo=reference.tzinfo) + timedelta(minutes=minutes)


def generation_of(tree: PredictionTree) -> str:
    """树的代名：构建时刻（UTC）+ 配置摘要，与 ``prediction/store.py`` 的命名同向（字典序即时间序）。"""

    return f"{tree.built_at.astimezone(tz=None).strftime('%Y%m%dT%H%M%SZ')}-{tree.config_digest[:12]}"


__all__ = ["MAX_LOOKAHEAD_DAYS", "MIN_PEAK_MASS", "Peak", "TreeOpportunities", "generation_of", "merged_marginal", "peaks_of"]
