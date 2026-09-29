"""覆盖口的实现：一段时间里看清了多少，从行为树的空白（``BehaviorKind.GAP``）算。

账本靠它回答"这次机会看清了没"——没看清的机会既不进分母也不进分子（2026-09-27 裁定"没看到就不算"）。
**两种空白都算**：「未观测」（相机没开、人不在视野）与「没读懂」（帧在、读不出是什么行为）对这个问题
没有区别，那段里发生过什么都不知道。

``observed_fraction`` = 1 − 空白占的分钟数 ÷ 这一段的分钟数。**空白先并集再减**：两条重叠的空白
（未观测 07:00–08:00 与没读懂 07:30–08:30）按次相减会把一段算两遍，算出负的覆盖。

**起止同刻的空白是合法的，而且是多数**：行为树的归约明说"起止同刻是合法的单观测段"
（`reduction/payloads.py`），实测真实数据里 7 月 46 个空白段有 **27 个**（59%）是同刻的「没读懂」。
它的含义是"那一瞬间读不出是什么行为"，**并不声称周围那段时间没观测到**，所以它不改变覆盖比例
（零宽度的区间减不掉任何时间）。但要**数出来**（``instant_gaps``）：不然"覆盖 1.0"看上去像"全看清了"。
真正倒挂（end < start）才是上游坏了，那个照旧硬拒。

**为什么在 runtime**：它读行为树，而读行为树本身 `scene` 是允许的；放组合根是因为它与机会口、节律口
是同一组注入的口，一起在夜批开头装好、一起带缓存。一天的空白在夜批里会被问几十次（每条承诺的每个机会
一次），所以按天缓存——夜批处理的是**已封口**的历史，那一天的空白不会再变。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta

from habitus.behavior.document import BehaviorDocument
from habitus.behavior.model import BehaviorKind
from habitus.behavior.tree import BehaviorTree
from habitus.scene.ledger.model import Coverage, ObservedGap, WindowSpan


class SceneCoverageError(ValueError):
    """行为树上的空白文档缺字段或自相矛盾——不静默当成"没有空白"（那会把没看清的机会算成看清了）。"""


class TreeCoverage:
    """按 ``CoverageProvider`` 协议实现的覆盖口。

    ``lookbehind_days`` 是往前多读几天：跨午夜的空白（02:00 睡到 09:00）挂在它**开始**那天的目录下，
    只读 span 覆盖的那几天会漏掉它。
    """

    def __init__(self, tree: BehaviorTree, *, lookbehind_days: int = 1) -> None:
        if not isinstance(tree, BehaviorTree):
            raise TypeError("tree must be a BehaviorTree")
        if isinstance(lookbehind_days, bool) or not isinstance(lookbehind_days, int) or lookbehind_days < 0:
            raise ValueError("lookbehind_days must be a non-negative integer")
        self.tree = tree
        self.lookbehind_days = lookbehind_days
        self._by_day: dict[date, tuple[ObservedGap, ...]] = {}
        #: 每天有几个"起止同刻"的空白。它们不减覆盖，但要能看见。
        self.instant_gaps: dict[date, int] = {}

    def coverage(self, span: WindowSpan) -> Coverage:
        if not isinstance(span, WindowSpan):
            raise TypeError("span must be a WindowSpan")
        total = (span.end - span.start).total_seconds()
        overlapping = tuple(gap for gap in self._gaps_around(span) if gap.start < span.end and span.start < gap.end)
        if total <= 0:  # WindowSpan 已经拒了 end <= start；这一行只是让除法不必猜
            return Coverage(observed_fraction=1.0, gaps=overlapping)  # pragma: no cover
        dark = sum((end - start).total_seconds() for start, end in _merged(span, overlapping))
        return Coverage(observed_fraction=max(0.0, 1.0 - dark / total), gaps=overlapping)

    def _gaps_around(self, span: WindowSpan) -> tuple[ObservedGap, ...]:
        first = span.start.date() - timedelta(days=self.lookbehind_days)
        last = span.end.date()
        found: list[ObservedGap] = []
        day = first
        while day <= last:
            found.extend(self._day_gaps(day))
            day += timedelta(days=1)
        return tuple(found)

    def _day_gaps(self, day: date) -> tuple[ObservedGap, ...]:
        cached = self._by_day.get(day)
        if cached is None:
            found: list[ObservedGap] = []
            instants = 0
            for document in self.tree.read_day(BehaviorKind.GAP, day):
                gap = _gap_of(document)
                if gap is None:
                    instants += 1
                    continue
                found.append(gap)
            cached = tuple(found)
            self._by_day[day] = cached
            self.instant_gaps[day] = instants
        return cached

    def forget(self, day: date | None = None) -> None:
        """丢掉缓存。一天还没封口、或空白刚被重写时调用（夜批处理已封口的历史，正常不需要）。"""

        if day is None:
            self._by_day.clear()
            return
        self._by_day.pop(day, None)


def _gap_of(document: BehaviorDocument) -> ObservedGap | None:
    """一段空白；**起止同刻的返回 None**（合法，但零宽度，减不掉任何时间）。"""

    started, ended, kind = (document.fields.get(key) for key in ("started_at", "ended_at", "gap_kind"))
    if not isinstance(started, str) or not isinstance(ended, str) or not isinstance(kind, str):
        raise SceneCoverageError("a gap document carries started_at, ended_at and gap_kind")
    try:
        start, end = datetime.fromisoformat(started), datetime.fromisoformat(ended)
    except ValueError as exc:
        raise SceneCoverageError(f"gap document has unusable times: {exc}") from exc
    if end == start:
        return None
    try:
        return ObservedGap(start=start, end=end, kind=kind)
    except ValueError as exc:
        # 倒挂（end < start）是上游坏了：不静默当成没有空白，那会把没看清的机会算成看清了。
        raise SceneCoverageError(f"gap document has unusable times: {exc}") from exc


def _merged(span: WindowSpan, gaps: Sequence[ObservedGap]) -> tuple[tuple[datetime, datetime], ...]:
    """空白截到这一段之内、按起点排序、重叠的并起来。"""

    clipped = sorted((max(span.start, gap.start), min(span.end, gap.end)) for gap in gaps)
    merged: list[tuple[datetime, datetime]] = []
    for start, end in clipped:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            continue
        merged.append((start, end))
    return tuple(merged)


__all__ = ["SceneCoverageError", "TreeCoverage"]
