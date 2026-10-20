"""复现规律（裁定 59）：每个行为类自己"隔多久又来"，登记在语义树上，预测层「离上次够久」的预期从这里来。

它不过关系检验那套门槛：自己隔多久又来没有对照组可比，靠不靠得住交给账去判。内容全由 ``temporal`` 算——按天的发生、
相邻间隔（漏看的权重、被提醒结尾的截尾都在里面）；这里只把一个类的记录交给它。每晚从同一份序列重算、只在内存里传一晚，
由预测层的机会表落盘。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from habitus.scene.concepts.rhythm import RhythmRulers
from habitus.series.model import EventSeries
from habitus.temporal import Coverage, DayInterval, OccurrenceDay, coverage_of, kind_reading


@dataclass(frozen=True)
class Recurrence:
    """一个类的复现：哪条 lane、按天的发生、相邻的间隔。"""

    kind_token: str
    lane: str
    days: tuple[OccurrenceDay, ...]
    intervals: tuple[DayInterval, ...]


def coverage_by_lane(series: EventSeries, rulers: RhythmRulers) -> Mapping[str, Coverage]:
    """序列里每条 lane 的观测覆盖（复现与预测层的久别共用这一份）。"""

    lanes = sorted({record.lane for record in series.records})
    return MappingProxyType({lane: coverage_of(series, lane, slot_minutes=rulers.slot_minutes) for lane in lanes})


def recurrences_of(
    series: EventSeries, rulers: RhythmRulers, coverage: Mapping[str, Coverage] | None = None
) -> Mapping[str, Recurrence]:
    """序列里每个（已转正的）类的复现。``coverage`` 是已经算好的各 lane 覆盖（没给就现算）。"""

    lanes = coverage if coverage is not None else coverage_by_lane(series, rulers)
    kinds = sorted({record.kind_token for record in series.records if record.classified})
    found: dict[str, Recurrence] = {}
    for kind in kinds:
        records = series.records_of(kind)
        lane = records[0].lane
        reading = kind_reading(records, lanes[lane], rulers)
        found[kind] = Recurrence(kind_token=kind, lane=lane, days=reading.days, intervals=reading.intervals)
    return MappingProxyType(found)


__all__ = ["Recurrence", "coverage_by_lane", "recurrences_of"]
