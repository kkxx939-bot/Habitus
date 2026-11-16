"""复现规律（裁定 59）：每个在用节点自己"隔多久又来"，登记在语义树上，预测层「离上次够久」的预期从这里来。

它不过关系检验那套门槛：自己隔多久又来没有对照组可比，靠不靠得住交给账去判。内容全由 ``temporal`` 算——按天的发生、
相邻间隔（漏看的权重、被提醒结尾的截尾都在里面）；这里只把一个在用节点的记录交给它：它的行为概念管的编号（自己 + 全部下级，
裁定 101 第 8 条）上的记录。不在用的节点（没有概念或概念已停用）没有复现，也就没有日子预期。每晚从同一份序列重算、只在内存里传一晚，
由预测层的机会表落盘。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from habitus.scene.concepts.model import ConceptSet
from habitus.series.model import EventSeries
from habitus.temporal import Coverage, DayInterval, OccurrenceDay, Rulers, coverage_of, kind_reading


@dataclass(frozen=True)
class Recurrence:
    """一个在用节点的复现：哪条 lane、按天的发生、相邻的间隔。"""

    kind_token: str
    lane: str
    days: tuple[OccurrenceDay, ...]
    intervals: tuple[DayInterval, ...]


def coverage_by_lane(series: EventSeries, rulers: Rulers) -> Mapping[str, Coverage]:
    """序列里每条 lane 的观测覆盖（复现与预测层的久别共用这一份）。"""

    lanes = sorted({record.lane for record in series.records})
    return MappingProxyType({lane: coverage_of(series, lane, slot_minutes=rulers.slot_minutes) for lane in lanes})


def recurrences_of(
    series: EventSeries, rulers: Rulers, concepts: ConceptSet, coverage: Mapping[str, Coverage] | None = None
) -> Mapping[str, Recurrence]:
    """每个在用节点（没停用的行为概念）的复现，记录范围是它管的编号；序列里一条都没有的不登记。``coverage`` 是已经算好的
    各 lane 覆盖（没给就现算）。"""

    lanes = coverage if coverage is not None else coverage_by_lane(series, rulers)
    found: dict[str, Recurrence] = {}
    for identity in sorted(concepts.behaviors()):
        concept = concepts[identity]
        if concept.retired:
            continue
        records = series.records_of(frozenset(concept.classes))
        if not records:
            continue
        kind, lane = concept.classes[0], records[0].lane
        reading = kind_reading(records, lanes[lane], rulers)
        found[kind] = Recurrence(kind_token=kind, lane=lane, days=reading.days, intervals=reading.intervals)
    return MappingProxyType(found)


__all__ = ["Recurrence", "coverage_by_lane", "recurrences_of"]
