"""槽位邻域序列：一次发生（或此刻）前后 ±k 槽内、行为树上的全部原子行为。历史侧与此刻侧共用这一处实现。

槽的口径与预测树同一套（``views.clock``），半宽 k 就是池化用的 ``pool_half_width``。这里**不做环形**——
环形只对"同一格的统计"成立（周日 23:45 与周一 00:00 在钟面上相邻），而一次发生前后真实发生过什么走的是
时间轴，跨午夜就去前一天 / 后一天的目录里取。

历史侧的窗口两头不对称：之前那段从开始所在槽再往前 k 槽；之后那段从**最后所见**所在槽再往后 k 槽。
打球一个半小时，回来洗澡在开始后两小时，按开始 ±k 会切掉，按结束 +k 才留得住。occurrence 自带
``last_observed_at``，不加字段。

两个入口都只在 ``DayIndex.rows`` 上切片，不再读盘；比较一律换成 UTC，文档各自带自己的偏移。
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from habitus.behavior.uri import BehaviorURI
from habitus.scene.views.clock import slot_floor
from habitus.scene.views.index import DayIndexCache
from habitus.scene.views.model import FlowRow


def slot_window(at: datetime, *, slot_minutes: int, half_width: int) -> tuple[datetime, datetime]:
    """以 ``at`` 所在槽为中心、±``half_width`` 槽的墙钟区间，左闭右开；跨午夜就跨日。"""

    if isinstance(half_width, bool) or not isinstance(half_width, int) or half_width < 0:
        raise ValueError("half_width must be a non-negative integer")
    floor = slot_floor(at, slot_minutes=slot_minutes)
    width = timedelta(minutes=slot_minutes)
    return floor - half_width * width, floor + (half_width + 1) * width


def slot_neighbourhood(
    occurrence_uri: str, cache: DayIndexCache, *, slot_minutes: int, half_width: int
) -> tuple[FlowRow, ...]:
    """那次发生前后的原子行为序列，含它自己：[开始所在槽 − k 槽, 最后所见所在槽 + k 槽)。"""

    parsed = BehaviorURI.parse(occurrence_uri)
    address = parsed.to_address()
    own = cache.day(address.occurred_on).by_uri.get(str(parsed))
    if own is None:
        raise KeyError(f"occurrence is not on the behaviour tree (or is a disambiguated duplicate): {parsed}")
    start, _ = slot_window(own.at, slot_minutes=slot_minutes, half_width=half_width)
    _, end = slot_window(own.last_observed_at, slot_minutes=slot_minutes, half_width=half_width)
    return _rows_between(cache, start, end, closed_end=False)


def slot_neighbourhood_until(
    at: datetime, cache: DayIndexCache, *, slot_minutes: int, half_width: int
) -> tuple[FlowRow, ...]:
    """此刻侧：[此刻所在槽 − k 槽, 此刻] 内已经发生的原子行为。"""

    start, _ = slot_window(at, slot_minutes=slot_minutes, half_width=half_width)
    return _rows_between(cache, start, at, closed_end=True)


def _rows_between(cache: DayIndexCache, start: datetime, end: datetime, *, closed_end: bool) -> tuple[FlowRow, ...]:
    begin = start.astimezone(UTC)
    finish = end.astimezone(UTC)
    rows: list[FlowRow] = []
    # 目录按文档自己的本地日期分，偏移与 ``start`` 未必相同，两头各多看一天，缓存里切片不贵。
    for day in _days(start.date() - timedelta(days=1), end.date() + timedelta(days=1)):
        for row in cache.day(day).rows:
            instant = row.at.astimezone(UTC)
            if instant < begin:
                continue
            if instant > finish or (instant == finish and not closed_end):
                continue
            rows.append(row)
    rows.sort(key=lambda row: (row.at.astimezone(UTC), row.uri))
    return tuple(rows)


def _days(first: date, last: date) -> tuple[date, ...]:
    return tuple(first + timedelta(days=offset) for offset in range((last - first).days + 1))


__all__ = ["slot_neighbourhood", "slot_neighbourhood_until", "slot_window"]
