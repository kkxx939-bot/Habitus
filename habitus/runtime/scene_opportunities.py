"""机会口：答"这个后件概念的第 k 个钟面峰，从锚往后各天落在哪、那天的平时概率是多少"。

**为什么在 runtime 而不在 scene**：`scene` 不许 import `prediction`（架构测试按传递闭包钉死，
`test_dependency_boundaries` 里那句"新语义树要对照预测树的期望时，由组合根注入 callable"）。所以这座桥
住在组合根，scene 只见到 ``OpportunityProvider`` 这个协议。

**窗口写在假设里，这里只算落点与对照**（2026-09-30 裁定一 / 2026-10-01 定稿）：
- 假设带着后件的峰表（``PeakWindow``：第 k 峰 = 几点到几点，写假设那一刻从节律口抄的，树重建不改）；
- 某一天的落点 = 那天同一钟面时段**两边各展 ``slack_minutes``**（= 树的 ``pool_half_width`` × 槽宽——预测树池化、
  预测层邻域用的同一个容差，10-01 定复用，不另配）；
- ``probability`` = 那天那个周几的曲线在展宽后那一段上的质量，截到 1；那个周几**没曲线就 None，不猜**；
- **第 1 个落点 = 锚之后第一个还没结束的**——锚正在窗口里的也算（评审 A-2）；下午 3 点的锚对"早上那个峰"落到第二天。

**概念 → 曲线**：树按 ``kind_token`` 存曲线，所以先拿 `scene.views.kinds` 数出这个概念命中过哪些 kind，
把那几条曲线**逐槽相加、截到 1**。相加的依据是 B13（同一槽近似互斥）。上级概念与多 kind 的份额问题并入词表改造。

节律口（``scene_rhythms``）仍从曲线上**取峰**（``peaks_of``）来定典型一天的峰表；这里不再取峰。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from habitus.prediction.model import MINUTES_PER_DAY, PredictionTree
from habitus.scene.hypotheses.model import PeakWindow, widened_spans
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
    """曲线上的一个峰：起止槽（含两端）、中心槽、峰内 marginal 之和。

    **跨午夜的峰** ``last_slot``（可能还有 ``centre_slot``）**≥ 槽数**：就寝常态在 23:00–01:00 时，当天末尾的质量接着
    **次日周几曲线**开头的质量算一个峰，归给它**开始**的那一天，尾巴落在次日（评审 A-7 / B-8 / C-7：按日历日各取各的会
    切成两个机会，每半够不上 ``MIN_PEAK_MASS`` 的整个消失、还把后件判成无节律型；2026-09-30 二-4 定接真正的次日曲线）。
    """

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


def peaks_of(
    curve: Sequence[float],
    *,
    floor: float = MIN_PEAK_MASS,
    following: Sequence[float] | None = None,
    preceding: Sequence[float] | None = None,
) -> tuple[Peak, ...]:
    """连续一段高于当日均值、**且峰内质量够得上 ``floor``** 的槽 = 一个峰。

    两道判据各管一件事：高于均值管"这一段比别处更可能"，``floor`` 管"它值得叫一次机会"。
    只有前者时稀疏曲线会把每个略高于均值的单槽都算成机会（见 ``MIN_PEAK_MASS`` 的实测）。
    均值为 0（整条曲线都是 0）时没有峰。

    **跨午夜**（2026-09-30 二-4）：给了 ``following``（**真正的次日**那个周几的曲线）就把两天拼成一条 2×槽数 的连续轴，
    当天末尾高于均值的一段接着次日开头高于均值的一段算**一个**峰，``last_slot ≥ 槽数`` 表示尾巴在次日；
    次日开头那一段不再单独成峰（它属于前一天开始的那个峰）。没给 ``following`` = 次日没数据 → **不猜**：峰截在 24:00，
    不拿当天开头绕回来补（"没看过就当没看到"）。

    对称地，给了 ``preceding``（前一天的曲线）且它的末尾高于它的均值时，当天从 00:00 开始的那一段是**前一天那个峰的尾巴**，
    这里不再算成当天的峰；没给 ``preceding`` 就当它是当天自己的峰。
    """

    average = sum(curve) / len(curve) if curve else 0.0
    if average <= 0.0:
        return ()
    size = len(curve)
    axis = list(curve)
    if following is not None:
        if len(following) != size:
            raise ValueError("the following day's curve must have the same number of slots")
        axis += list(following)
    found: list[Peak] = []
    start: int | None = None
    for slot in range(len(axis)):
        if axis[slot] > average:
            start = slot if start is None else start
            continue
        if start is not None:
            found.append(_peak(axis, start, slot - 1))
            start = None
    if start is not None:
        found.append(_peak(axis, start, len(axis) - 1))
    tail_of_yesterday = preceding is not None and bool(preceding) and preceding[-1] > sum(preceding) / len(preceding)
    kept = []
    for peak in found:
        if peak.first_slot >= size:
            continue  # 次日自己的峰：归次日那天算，这里只取当天开始的
        if peak.first_slot == 0 and tail_of_yesterday:
            continue  # 前一天那个峰的尾巴，前一天已经把它算进去了
        if peak.mass >= floor:
            kept.append(peak)
    return tuple(kept)


def _peak(axis: Sequence[float], first: int, last: int) -> Peak:
    centre = max(range(first, last + 1), key=lambda slot: (axis[slot], -slot))
    return Peak(first_slot=first, last_slot=last, centre_slot=centre, mass=sum(axis[first : last + 1]))


class TreeOpportunities:
    """按 ``OpportunityProvider`` 协议实现的机会口。组合根在**重建树之后**构造它（横切第 3 条）。

    ``kinds`` 是 `scene.views.kinds.kinds_by_concept` 的产物：概念身份 → 要相加的那几条曲线的键。
    ``slack_slots`` = 树的 ``pool_half_width``（窗口两边各展几槽）。
    """

    def __init__(
        self,
        tree: PredictionTree,
        kinds: Mapping[str, tuple[str, ...]],
        *,
        generation: str,
        slack_slots: int = 0,
        timezone_of: datetime | None = None,
        max_lookahead_days: int = MAX_LOOKAHEAD_DAYS,
    ) -> None:
        if not isinstance(tree, PredictionTree):
            raise TypeError("tree must be a PredictionTree")
        if isinstance(max_lookahead_days, bool) or not isinstance(max_lookahead_days, int) or max_lookahead_days <= 0:
            raise ValueError("max_lookahead_days must be a positive integer")
        if isinstance(slack_slots, bool) or not isinstance(slack_slots, int) or slack_slots < 0:
            raise ValueError("slack_slots must be a non-negative integer")
        if not isinstance(generation, str) or not generation.strip():
            raise ValueError("generation must be non-empty text")
        self.tree = tree
        self.kinds = dict(kinds)
        self.generation = generation
        self.max_lookahead_days = max_lookahead_days
        self.slack_minutes = slack_slots * tree.slot_minutes
        self._reference = timezone_of

    def opportunities(self, request: OpportunityRequest) -> OpportunitySnapshot | None:
        """窗口在各天的落点，从锚之后第一个还没结束的开始，共 ``count`` 个。概念没有曲线键也照样铺（对照全 None）。"""

        kinds = self.kinds.get(request.consequent, ())
        wanted = min(max(1, request.count), MAX_SNAPSHOT_OPPORTUNITIES)
        anchor = request.anchor
        cycle = list(request.windows) if request.windows else [request.window]
        start_at = next((index for index, item in enumerate(cycle) if item == request.window), 0)
        # 容差两边各展 slack，但不越过与邻窗的中点——否则相邻两个峰的窗口会重叠，一次到来归两本账、快照也不合法。
        spans = widened_spans(cycle, self.slack_minutes)
        found: list[Opportunity] = []
        # 第 1 个落点必须是**请求的那个窗口**（这本账自己的峰）在锚之后第一次还没结束的那天；跨午夜的窗口可能从前一天开始、
        # 尾巴盖住锚，所以从前一天找起。找到之后再按峰表往后轮（次数方面要接下来的几个窗口）。
        day = anchor.date() - timedelta(days=1)
        for _offset in range(self.max_lookahead_days + 1):
            item = self._landing(day, cycle[start_at], spans[start_at], kinds, anchor)
            if item.span.end > anchor:
                found.append(item)
                break
            day += timedelta(days=1)
        if not found:
            return None
        position = start_at + 1
        while len(found) < wanted and position - start_at <= self.max_lookahead_days * len(cycle):
            if position % len(cycle) == 0:
                day += timedelta(days=1)
            index = position % len(cycle)
            found.append(self._landing(day, cycle[index], spans[index], kinds, anchor))
            position += 1
        return OpportunitySnapshot(generation=self.generation, opportunities=tuple(found))

    def _landing(self, day: date, window: PeakWindow, span: tuple[int, int], kinds: Sequence[str], anchor: datetime) -> Opportunity:
        start, end = span
        centre = (window.start_minute + window.end_minute) // 2
        return Opportunity(
            at=self._moment(day, centre, anchor),
            span=WindowSpan(self._moment(day, start, anchor), self._moment(day, end, anchor)),
            probability=self._mass(day, start, end, kinds),
        )

    def _mass(self, day: date, start_minute: int, end_minute: int, kinds: Sequence[str]) -> float | None:
        """展宽后那一段上的曲线质量；段可以跨到前一天 / 次日，各取那天周几的曲线；哪一天没曲线就 None（不猜）。"""

        if not kinds:
            return None
        slot_minutes = self.tree.slot_minutes
        total = 0.0
        minute = start_minute
        while minute < end_minute:
            offset_days, minute_of_day = divmod(minute, MINUTES_PER_DAY)
            curve = merged_marginal(self.tree, (day + timedelta(days=offset_days)).weekday(), kinds)
            if curve is None:
                return None
            total += curve[minute_of_day // slot_minutes]
            minute += slot_minutes - (minute_of_day % slot_minutes)
        return min(1.0, total)

    def _moment(self, day: date, minutes: int, anchor: datetime) -> datetime:
        """分钟 → 本地时刻（可以是负数或超过一天：落到前一天 / 次日）。时区跟着锚走（承诺上的时间一律本地时间 + 显式偏移）。"""

        reference = self._reference if self._reference is not None else anchor
        return datetime.combine(day, datetime.min.time(), tzinfo=reference.tzinfo) + timedelta(minutes=minutes)


def generation_of(tree: PredictionTree) -> str:
    """树的代名：构建时刻（UTC）+ 配置摘要，与 ``prediction/store.py`` 的命名同向（字典序即时间序）。"""

    return f"{tree.built_at.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')}-{tree.config_digest[:12]}"


__all__ = ["MAX_LOOKAHEAD_DAYS", "MIN_PEAK_MASS", "Peak", "TreeOpportunities", "generation_of", "merged_marginal", "peaks_of"]
