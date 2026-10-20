"""预测层的夜批那一拍（N1，``19-预测层方案.md`` 第九节第 2 步）：排在语义树之后，用语义树这一晚建好的时间线与概念集，
读这一晚的关系表，算每条 lane 的机会表、落盘；接着用同一份关系表与时间线重标单元的历史（裁定 55 新版第 2 条）。零模型。

读的关系表是语义树**刚写的这一晚**；那一晚没有（语义树这一步没跑到）就退到盘上更早的最晚一晚，并在 ``signals`` 里说明，
表上也记着用的是哪一晚（``relation_night``）。一条 lane 失败不拖别的 lane；失败自己留观测与信号，不往上抛（夜批
不能因为机会表没算出来而不算下一晚）。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime

from habitus.conditions import DayTypeCalendar
from habitus.foresight import ForesightError
from habitus.foresight.opportunities import OpportunityTable, build_table
from habitus.foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer
from habitus.runtime.foresight_compaction import LedgerCompaction
from habitus.runtime.foresight_history import HistoryStage
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.recurrence import Recurrence
from habitus.scene.relations import LaneTimeline, Relation, RelationConfig
from habitus.scene.relations.store import RelationStore, RelationStoreError
from habitus.temporal import Coverage


@dataclass(frozen=True)
class LaneOpportunitySummary:
    relations: int
    chances: int
    hits: int
    kinds: int
    relation_night: date | None
    #: 这一晚单元历史重标了几刻（裁定 55 新版第 2 条；没接这一步是 0）。
    moments: int = 0
    #: 这一晚压实了几本账（26 号清单 4.8；没接这一步是 0）。
    compacted: int = 0


@dataclass(frozen=True)
class OpportunityReport:
    night: date
    lanes: Mapping[str, LaneOpportunitySummary] = field(default_factory=dict)
    signals: tuple[str, ...] = ()
    #: 这一晚没做成的步骤（"lane: 机会表 / 单元历史: 异常类型"）。机会表写成了而单元历史没写成，也要在这里看得见（26 号清单 2.5.1）。
    failed: tuple[str, ...] = ()

    def summary(self) -> str:
        counts = "；".join(
            f"{lane} 关系 {item.relations} 条、机会 {item.chances} 次（来了 {item.hits}）、类 {item.kinds} 个、单元历史 {item.moments} 刻"
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
        history: HistoryStage | None = None,
        calendar: DayTypeCalendar | None = None,
        unseen_day_types: str = "skip",
        compaction: LedgerCompaction | None = None,
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
        if history is not None and not isinstance(history, HistoryStage):
            raise TypeError("history must be a HistoryStage or None")
        #: 重标单元的历史（裁定 55 新版第 2 条）：要的材料（这一晚的关系表、时间线）只在这一拍手上，所以排在这里。
        self.history = history
        if compaction is not None and not isinstance(compaction, LedgerCompaction):
            raise TypeError("compaction must be a LedgerCompaction or None")
        if compaction is not None and history is None:
            # 压实会删掉单元历史还没记下的原始结果：没有单元历史就不许压实（L10.11）
            raise ValueError("ledger compaction requires the unit history stage")
        #: 账本压实（26 号清单 4.8）：排在单元历史之后——单元历史读的是压实之前的那一份账。
        self.compaction = compaction
        #: 读侧那一份当地日历：机会表记下这一晚用的放假数据版本（裁定 59 第 4 条）。
        self.calendar = calendar
        #: 没见过的日型怎么建预期（``config.foresight.unseen_day_types``），记在机会表上给白天那一拍、重标、结算读。
        self.unseen_day_types = unseen_day_types
        self.observer: Observer = observer if observer is not None else NullObserver()
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    def run(
        self,
        *,
        night: date,
        concepts: ConceptSet,
        timelines: Mapping[str, LaneTimeline],
        recurrences: Mapping[str, Recurrence] | None = None,
        coverage: Mapping[str, Coverage] | None = None,
    ) -> OpportunityReport:
        """第 ``night`` 晚：每条 lane 一张表。``timelines`` 是语义树这一晚用同一份序列建的时间线，``recurrences`` 是语义树
        登记的每个类的复现（机会表的日历部分从这里抄，裁定 59），``coverage`` 是各 lane 的观测覆盖（抄成每天看着的比例）。"""

        started = time.monotonic()
        lanes: dict[str, LaneOpportunitySummary] = {}
        signals: list[str] = []
        failed: list[str] = []
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
                    recurrences=recurrences,
                    calendar_version=self.calendar.version if self.calendar is not None else None,
                    unseen_day_types=self.unseen_day_types,
                    coverage=(coverage or {}).get(lane),
                    calendar=self.calendar,
                )
                self.store.write(table)
            except (ForesightError, RelationStoreError) as exc:
                signals.append(f"opportunities: {lane} 这一晚的机会表没算出来（{exc}）")
                failed.append(f"{lane}: 机会表: {type(exc).__name__}")
                self._observe(ObservationStatus.FAILURE, {"lane": lane, "error_type": type(exc).__name__}, started)
                continue
            # 单元历史与机会表分开接：表写成了、历史没写成时，表照常给白天用，但失败要报出来，不再被一并吞掉
            try:
                relabelled = 0 if self.history is None else self.history.run(night=night, table=table, line=line)
            except ForesightError as exc:
                signals.append(f"opportunities: {lane} 这一晚的单元历史没写成（{exc}），账本压实这一晚跳过")
                failed.append(f"{lane}: 单元历史: {type(exc).__name__}")
                self._observe(ObservationStatus.FAILURE, {"lane": lane, "error_type": type(exc).__name__}, started)
                # 单元历史没把结完的日子记下来：这一晚压实会删掉它还没记的原始结果，下一晚它就拿不到了（L10.11）
                lanes[lane] = _summarise(table, 0)
                continue
            compacted = self._compact(lane, night, table, signals, failed)
            lanes[lane] = replace(_summarise(table, relabelled), compacted=compacted)
        # 口径：每条 lane 都写出来了才是成功；有一条没写是失败（那一条自己已经留了失败事件，这里再记一笔总账）
        self._observe(
            ObservationStatus.SUCCESS if not failed else ObservationStatus.FAILURE,
            {
                "lanes": len(lanes),
                "lanes_failed": len(timelines) - len(lanes),
                "relations": sum(item.relations for item in lanes.values()),
                "chances": sum(item.chances for item in lanes.values()),
                "moments": sum(item.moments for item in lanes.values()),
            },
            started,
        )
        return OpportunityReport(night=night, lanes=lanes, signals=tuple(signals), failed=tuple(failed))

    def _compact(self, lane: str, night: date, table: OpportunityTable, signals: list[str], failed: list[str]) -> int:
        """这条 lane 的账本压实；压不成的那几本账各自跳过，失败照 2.5.1 的口径记进 ``failed``，别的账与这一晚照常。"""

        if self.compaction is None or self.compaction.lane != lane:
            return 0
        try:
            report = self.compaction.run(night=night, table=table)
        except ForesightError as exc:
            signals.append(f"opportunities: {lane} 这一晚的账本压实没做成（{exc}）")
            failed.append(f"{lane}: 账本压实: {type(exc).__name__}")
            return 0
        for account, error in sorted(report.failed.items()):
            signals.append(f"opportunities: {lane} 账 {account!r} 这一晚没压实（{error}）")
            failed.append(f"{lane}: 账本压实: {error}")
        return report.compacted

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


def _summarise(table: OpportunityTable, moments: int = 0) -> LaneOpportunitySummary:
    return LaneOpportunitySummary(
        relations=len(table.relations),
        chances=sum(item.opportunities for item in table.relations),
        hits=sum(item.hits for item in table.relations),
        kinds=len(table.kinds),
        relation_night=table.relation_night,
        moments=moments,
    )


__all__ = ["ForesightNightlyRun", "LaneOpportunitySummary", "OpportunityReport"]
