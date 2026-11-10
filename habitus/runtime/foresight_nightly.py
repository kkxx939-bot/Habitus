"""换代里预测层的那一段（N1，``19-预测层方案.md`` 第九节第 2 步）：排在语义树之后，用语义树这一代建好的时间线与概念集，
读这一代的关系表，算每条 lane 的机会表、落盘；接着用同一份关系表与时间线重标单元的历史（裁定 55 新版第 2 条），再压实账本。零模型。

读的关系表是语义树**刚写的这一代**；那一晚没有（关系判定没做成）就退到盘上更早的最晚一晚，并在 ``signals`` 里说明，
表上也记着用的是哪一晚（``relation_night``）。一条 lane 失败不拖别的 lane；失败自己留观测与信号，不往上抛。

两种失败分开报（实时化方案第 3 节）：机会表写不出是**必须成**的那一步没成（``failed``，换代据它判今天没有这一代）；单元历史、
压实没做成是**尽力而为**（``degraded``，表照用、压实这一次跳过）。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime

from habitus.conditions import DayTypeCalendar
from habitus.foresight import ForesightError
from habitus.foresight.opportunities import OpportunityTable, build_table
from habitus.foundation.observability import (
    NullObserver,
    ObservationStatus,
    Observer,
    record_observation,
)
from habitus.foundation.tracing import annotate, set_attributes, span
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
    #: 机会表没写成的 lane（"lane: 机会表: 异常类型"）：必须成的那一步没成。
    failed: tuple[str, ...] = ()
    #: 机会表写成了、之后尽力的那几步没做成的（"lane: 单元历史 / 账本压实: 异常类型"）：表照用，要看得见（26 号清单 2.5.1）。
    degraded: tuple[str, ...] = ()

    @property
    def history_skipped(self) -> int:
        """单元历史没做成（压实因此跳过）的 lane 数。"""

        return sum(1 for item in self.degraded if ": 单元历史: " in item)

    @property
    def compaction_failed(self) -> int:
        """账本压实没做成的条数（整条 lane 或某一本账）。"""

        return sum(1 for item in self.degraded if ": 账本压实: " in item)

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
        self.store = store
        self.relations = relations
        self.config = config
        #: 重标单元的历史（裁定 55 新版第 2 条）：要的材料（这一晚的关系表、时间线）只在这一拍手上，所以排在这里。
        self.history = history
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
        degraded: list[str] = []
        for lane, line in sorted(timelines.items()):
            summary = self._lane(
                lane,
                line,
                night,
                concepts,
                recurrences,
                coverage,
                signals=signals,
                failed=failed,
                degraded=degraded,
                started=started,
            )
            if summary is not None:
                lanes[lane] = summary
        # 口径：每条 lane 的机会表都写出来了才是成功；有一条没写是失败（那一条自己已经留了失败事件，这里再记一笔总账）
        self._observe(
            ObservationStatus.SUCCESS if not failed else ObservationStatus.FAILURE,
            {
                "lanes": len(lanes),
                "lanes_failed": len(timelines) - len(lanes),
                "relations": sum(item.relations for item in lanes.values()),
                "chances": sum(item.chances for item in lanes.values()),
                "moments": sum(item.moments for item in lanes.values()),
                # 机会表、单元历史读回时跳过的坏文件（坏的一份不让这一晚停，计入观测）
                "unreadable": len(self.store.take_unreadable())
                + (len(self.history.store.take_unreadable()) if self.history is not None else 0),
            },
            started,
        )
        return OpportunityReport(
            night=night, lanes=lanes, signals=tuple(signals), failed=tuple(failed), degraded=tuple(degraded)
        )

    def _lane(
        self,
        lane: str,
        line: LaneTimeline,
        night: date,
        concepts: ConceptSet,
        recurrences: Mapping[str, Recurrence] | None,
        coverage: Mapping[str, Coverage] | None,
        *,
        signals: list[str],
        failed: list[str],
        degraded: list[str],
        started: float,
    ) -> LaneOpportunitySummary | None:
        """一条 lane：机会表（``foresight.opportunities`` span）→ 单元历史重标（``foresight.unit_history`` span）→ 账本压实
        （``foresight.compaction`` span）。机会表没算出来是 None（这条 lane 这一晚没有表）。"""

        with span("foresight.opportunities", {"lane": lane}) as attributes:
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
                attributes["failed"] = type(exc).__name__
                return None
            set_attributes(attributes, _table_attributes(table, len(found)))
        # 单元历史与机会表分开接：表写成了、历史没写成时，表照常给白天用，但失败要报出来，不再被一并吞掉
        with span("foresight.unit_history", {"lane": lane}) as attributes:
            try:
                relabelled = self._relabel(lane, night, table, line, started)
            except ForesightError as exc:
                signals.append(f"opportunities: {lane} 这一晚的单元历史没写成（{exc}），账本压实这一晚跳过")
                degraded.append(f"{lane}: 单元历史: {type(exc).__name__}")
                self._observe(ObservationStatus.FAILURE, {"lane": lane, "error_type": type(exc).__name__}, started)
                attributes["failed"] = type(exc).__name__
                # 单元历史没把结完的日子记下来：这一晚压实会删掉它还没记的原始结果，下一晚它就拿不到了（L10.11）
                return _summarise(table, 0)
        compacted = self._compact(lane, night, table, signals, degraded)
        return replace(_summarise(table, relabelled), compacted=compacted)

    def _compact(self, lane: str, night: date, table: OpportunityTable, signals: list[str], degraded: list[str]) -> int:
        """这条 lane 的账本压实；压不成的那几本账各自跳过，失败照 2.5.1 的口径记进 ``degraded``，别的账与这一晚照常。"""

        if self.compaction is None or self.compaction.lane != lane:
            return 0
        with span("foresight.compaction", {"lane": lane}) as attributes:
            try:
                report = self.compaction.run(night=night, table=table)
            except ForesightError as exc:
                signals.append(f"opportunities: {lane} 这一晚的账本压实没做成（{exc}）")
                degraded.append(f"{lane}: 账本压实: {type(exc).__name__}")
                attributes["failed"] = type(exc).__name__
                return 0
            for account, error in sorted(report.failed.items()):
                signals.append(f"opportunities: {lane} 账 {account!r} 这一晚没压实（{error}）")
                degraded.append(f"{lane}: 账本压实: {error}")
            set_attributes(attributes, {"compacted": report.compacted, "accounts_failed": len(report.failed)})
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

    def _relabel(self, lane: str, night: date, table: OpportunityTable, line: LaneTimeline, started: float) -> int:
        """重标这一晚的单元历史并发一条观测（裁定 55 新版第 2 条；W13：词表 / 关系大改的那晚之前这些数看不见）：
        签名变了的刻数、受影响的单元数。没接单元历史时是 0；重标失败往上抛，由 ``run`` 按原口径接。"""

        if self.history is None:
            return 0
        outcome = self.history.run(night=night, table=table, line=line)
        annotate(
            {
                "relabelled": outcome.relabelled,
                "signature_changed": outcome.delta.signature_changed_moments,
                "units_affected": outcome.delta.units_affected,
            }
        )
        record_observation(
            self.observer,
            "foresight",
            "unit_history",
            ObservationStatus.SUCCESS,
            started=started,
            attributes={
                "lane": lane,
                "relabelled": outcome.relabelled,
                "signature_changed": outcome.delta.signature_changed_moments,
                "units_affected": outcome.delta.units_affected,
            },
        )
        return outcome.relabelled

    def _observe(
        self, status: ObservationStatus, attributes: dict[str, str | int | float | bool], started: float
    ) -> None:
        record_observation(self.observer, "foresight", "opportunities", status, started=started, attributes=attributes)


def _table_attributes(table: OpportunityTable, relations_read: int) -> dict[str, object]:
    """``foresight.opportunities`` 的读数：读进来的关系、表里的关系与时窗（机会）、来了几次、类几个、用的哪一晚的关系表。"""

    return {
        "relations_read": relations_read,
        "relations": len(table.relations),
        "chances": sum(item.opportunities for item in table.relations),
        "hits": sum(item.hits for item in table.relations),
        "kinds": len(table.kinds),
        "relation_night": "" if table.relation_night is None else table.relation_night.isoformat(),
    }


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
