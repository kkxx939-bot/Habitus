"""预测层的夜批那一拍（N1，``19-预测层方案.md`` 第九节第 2 步）：排在语义树之后，用语义树这一晚建好的时间线与概念集，
读这一晚的关系表，算每条 lane 的机会表、落盘。零模型。

读的关系表是语义树**刚写的这一晚**；那一晚没有（语义树这一步没跑到）就退到盘上更早的最晚一晚，并在 ``signals`` 里说明，
表上也记着用的是哪一晚（``relation_night``）。一条 lane 失败不拖别的 lane；失败自己留观测与信号，不往上抛（夜批
不能因为机会表没算出来而不算下一晚）。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from habitus.foresight import ForesightError
from habitus.foresight.opportunities import OpportunityTable, build_table
from habitus.foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer
from habitus.runtime.foresight_causes import CauseStage
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.relations import LaneTimeline, Relation, RelationConfig
from habitus.scene.relations.store import RelationStore, RelationStoreError


@dataclass(frozen=True)
class LaneOpportunitySummary:
    relations: int
    chances: int
    hits: int
    kinds: int
    relation_night: date | None
    #: 这一晚给几次落空找到了原因（裁定 55；没接找原因那一拍是 0）。
    explained: int = 0


@dataclass(frozen=True)
class OpportunityReport:
    night: date
    lanes: Mapping[str, LaneOpportunitySummary] = field(default_factory=dict)
    signals: tuple[str, ...] = ()

    def summary(self) -> str:
        counts = "；".join(
            f"{lane} 关系 {item.relations} 条、机会 {item.chances} 次（来了 {item.hits}）、类 {item.kinds} 个、找到原因 {item.explained} 次"
            for lane, item in sorted(self.lanes.items())
        )
        return f"第 {self.night} 晚机会表：{counts or '无'}"


class ForesightNightlyRun:
    def __init__(
        self,
        store: OpportunityStore,
        relations: RelationStore,
        config: RelationConfig,
        *,
        observer: Observer | None = None,
        clock: Callable[[], datetime] | None = None,
        causes: CauseStage | None = None,
    ) -> None:
        if not isinstance(store, OpportunityStore):
            raise TypeError("store must be an OpportunityStore")
        if not isinstance(relations, RelationStore):
            raise TypeError("relations must be a RelationStore")
        if not isinstance(config, RelationConfig):
            raise TypeError("config must be a RelationConfig")
        self.store = store
        self.relations = relations
        self.config = config
        if causes is not None and not isinstance(causes, CauseStage):
            raise TypeError("causes must be a CauseStage or None")
        #: 给落空找原因（裁定 55）：要的材料（这一晚的概念集、时间线、全部关系）只在这一拍手上，所以排在这里。
        self.causes = causes
        self.observer: Observer = observer if observer is not None else NullObserver()
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    def run(self, *, night: date, concepts: ConceptSet, timelines: Mapping[str, LaneTimeline]) -> OpportunityReport:
        """第 ``night`` 晚：每条 lane 一张表。``timelines`` 是语义树这一晚用同一份序列建的时间线。"""

        started = time.monotonic()
        lanes: dict[str, LaneOpportunitySummary] = {}
        signals: list[str] = []
        for lane, line in sorted(timelines.items()):
            try:
                relation_night, found = self._relations(lane, night, signals)
                table = build_table(
                    line,
                    found,
                    concepts,
                    night=night,
                    relation_night=relation_night,
                    slot_minutes=self.config.slot_minutes,
                    window_slots=self.config.transition_window_slots,
                )
                self.store.write(table)
                explained = (
                    0
                    if self.causes is None
                    else self.causes.run(
                        line=line, concepts=concepts, relations=found, window_slots=self.config.transition_window_slots
                    )
                )
            except (ForesightError, RelationStoreError) as exc:
                signals.append(f"opportunities: {lane} 这一晚的机会表没算出来（{exc}）")
                self._observe(ObservationStatus.FAILURE, {"lane": lane, "error_type": type(exc).__name__}, started)
                continue
            lanes[lane] = _summarise(table, explained)
        # 口径：每条 lane 都写出来了才是成功；有一条没写是失败（那一条自己已经留了失败事件，这里再记一笔总账）
        self._observe(
            ObservationStatus.SUCCESS if len(lanes) == len(timelines) else ObservationStatus.FAILURE,
            {
                "lanes": len(lanes),
                "lanes_failed": len(timelines) - len(lanes),
                "relations": sum(item.relations for item in lanes.values()),
                "chances": sum(item.chances for item in lanes.values()),
                "explained": sum(item.explained for item in lanes.values()),
            },
            started,
        )
        return OpportunityReport(night=night, lanes=lanes, signals=tuple(signals))

    def _relations(self, lane: str, night: date, signals: list[str]) -> tuple[date | None, tuple[Relation, ...]]:
        """这一晚的关系表；没有就退到更早的最晚一晚。一晚都没有（语义树还没写过表）是空表。"""

        nights = self.relations.nights(lane)
        if night in nights:
            return night, self.relations.read(lane, night).relations
        earlier = [item for item in nights if item < night]
        if not earlier:
            signals.append(f"opportunities: {lane} 还没有关系表，机会表这一晚只有日历")
            return None, ()
        signals.append(f"opportunities: {lane} 第 {night} 晚没有关系表，用第 {earlier[-1]} 晚的")
        return earlier[-1], self.relations.read(lane, earlier[-1]).relations

    def _observe(
        self, status: ObservationStatus, attributes: dict[str, str | int | float | bool], started: float
    ) -> None:
        try:
            self.observer.record(
                ObservationEvent(
                    category="foresight",
                    operation="opportunities",
                    status=status,
                    duration_seconds=max(0.0, time.monotonic() - started),
                    attributes=attributes,
                )
            )
        except Exception:  # noqa: BLE001 - 观测不许影响夜批
            pass


def _summarise(table: OpportunityTable, explained: int = 0) -> LaneOpportunitySummary:
    return LaneOpportunitySummary(
        relations=len(table.relations),
        chances=sum(item.opportunities for item in table.relations),
        hits=sum(item.hits for item in table.relations),
        kinds=len(table.kinds),
        relation_night=table.relation_night,
        explained=explained,
    )


__all__ = ["ForesightNightlyRun", "LaneOpportunitySummary", "OpportunityReport"]
