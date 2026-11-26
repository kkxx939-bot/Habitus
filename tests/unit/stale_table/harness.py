"""预测不等换代（2026-11-16）验收测试共用的盘面：一棵真实行为树、按支一份账本 + 机会表 + 完成标记、真实的预测拍
（``ForesightTick``，接落盘水位与完成标记，接法同组合根 ``runtime/pipeline_assembly.py`` 的 ``finished=marks.finished``）、
真实的递送（``DeliveryStage``，出口是"宿主一直活着"的替身）与结算（``SettlementStage``）。

现场：BASE 起连着几天 10:00 讨论方案、10:20 修改代码、12:00 写文档；关系"讨论方案 → 修改代码（同一条链）"成立。
机会表第 G 晚按换代的口径建：序列截止到 G（含 G−1、不含 G，``read_series(cutoff=G)``），表的 night = G。
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from habitus.conditions import LocalCalendar
from habitus.config.foresight import ForesightConfig
from habitus.foresight import NoUnsealed
from habitus.foresight.ladder import Tolerance
from habitus.foresight.ledger import Expectation
from habitus.foresight.ledger.model import parse_account
from habitus.foresight.opportunities import build_table
from habitus.foundation.observability import ObservationEvent
from habitus.runtime.foresight import ForesightReadSide
from habitus.runtime.foresight_days import DayExpectationStage
from habitus.runtime.foresight_delivery import DeliveryStage
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.runtime.foresight_settlement import SettlementStage
from habitus.runtime.foresight_tick import ForesightTick, TickReport
from habitus.runtime.generation_marks import GenerationMark, GenerationMarks
from habitus.runtime.generation_watermark import TickTrack, TickWatermarkStore
from habitus.scene.concepts import ConceptSet
from habitus.scene.recurrence import recurrences_of
from habitus.scene.relations import Relation, RelationConfig, RelationKey, Segment, Status, build_timelines
from habitus.temporal import Rulers
from tests.series_helpers import read_series
from tests.unit.kind_ids import kind_id
from tests.unit.runtime.test_foresight_delivery import EVIDENCE, LOSS, Outlet, Seeded
from tests.unit.scene.concept_fixtures import base
from tests.unit.scene.fixtures import SUBJECT, Site, at, publish

ZONE = ZoneInfo("Asia/Shanghai")
SLOT = timedelta(minutes=15)
#: 现场第一天（单元历史替身的快照在 9 月初，账的事实在它之前；这里都晚于它们）。
BASE = date(2026, 10, 12)
DISCUSS, CODE, DOCS = kind_id("讨论方案"), kind_id("修改代码"), kind_id("写文档")
CONCEPTS = ConceptSet([base("讨论方案"), base("修改代码"), base("写文档")])
CONFIG = RelationConfig(slot_minutes=15, transition_window_slots=3, absence_typical_multiple=8.0, absence_min_gaps=2)
RULERS = Rulers(15, 2700, 3, 5.0, 5.0, 35)
TOLERANCE = Tolerance(line=1.0 - ForesightConfig().false_alarm_rate)
#: 给用户看的话里不许出现的内部的词（预期.md D3）。
BANNED_WORDS = ("换代", "机会表", "这一代", "定稿", "槽")


def chain(lane: str = "session") -> Relation:
    return Relation(key=RelationKey(lane, DISCUSS, CODE, Segment.CHAIN), status=Status.ESTABLISHED, upward=True)


class Recorder:
    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)

    def ticks(self) -> list[ObservationEvent]:
        return [event for event in self.events if event.operation == "tick"]


@dataclass
class Lane:
    """一支：账本、机会表、完成标记、水位、递送、观测。"""

    name: str
    root: Path
    ledger: ForesightLedgerStore
    opportunities: OpportunityStore
    marks: GenerationMarks
    seeded: Seeded
    delivery: DeliveryStage
    outlet: Outlet
    recorder: Recorder = field(default_factory=Recorder)
    unwatched: list[tuple[datetime, datetime]] = field(default_factory=list)


@dataclass
class World:
    tmp: Path
    site: Site
    lanes: dict[str, Lane]

    @property
    def session(self) -> Lane:
        return self.lanes["session"]

    # ── 现场 ──

    def mornings(self, days: list[date], *, docs: bool = True) -> None:
        for day in days:
            publish(self.site.behavior_tree, day, "讨论方案", 10, 0)
            publish(self.site.behavior_tree, day, "修改代码", 10, 20)
            if docs:
                publish(self.site.behavior_tree, day, "写文档", 12, 0)

    def read_side(self) -> ForesightReadSide:
        return ForesightReadSide(self.site.behavior_tree, SUBJECT, LocalCalendar(), ZONE)

    # ── 换代的产物 ──

    def table(self, night: date, *, relations: tuple[Relation, ...] | None = None, lane: str = "session") -> None:
        """写第 ``night`` 晚的机会表（换代读序列的口径：截止到 night，含昨天）。"""

        series = read_series(self.site.behavior_tree, cutoff=night)
        line = build_timelines(series, CONCEPTS, {}, CONFIG)["session"]
        line = dataclasses.replace(line, lane=lane) if lane != "session" else line
        table = build_table(
            line,
            (chain(lane),) if relations is None else relations,
            CONCEPTS,
            night=night,
            relation_night=night,
            slot_minutes=15,
            window_slots=3,
            recurrences=recurrences_of(series, RULERS, CONCEPTS),
        )
        self.lanes[lane].opportunities.write(table)

    def mark(self, day: date, *, lane: str = "session") -> None:
        """第 ``day`` 天这一代的完成标记（换代最后一步）。"""

        self.lanes[lane].marks.write(GenerationMark(day, datetime.combine(day, time(1, 0), tzinfo=ZONE)))

    def corrupt_mark(self, day: date, *, lane: str = "session") -> None:
        path = self.lanes[lane].marks.path(day)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"schema": "generation_mark_v1"', encoding="utf-8")

    def generation(self, day: date, *, lane: str = "session", relations: tuple[Relation, ...] | None = None) -> None:
        """今天这一代做成：写今天的表、建当天的日子预期（``days_step``）、写完成标记。"""

        self.table(day, relations=relations, lane=lane)
        self.day_expectations(day, lane=lane)
        self.mark(day, lane=lane)

    def day_expectations(self, day: date, *, lane: str = "session", moment: time = time(1, 0)) -> None:
        state = self.lanes[lane]
        asyncio.run(
            DayExpectationStage(
                state.ledger,
                state.opportunities,
                self.read_side(),
                NoUnsealed(),
                lane=lane,
                vocabulary_version=lambda: 7,
                recent=35,
                tolerance=TOLERANCE,
                clock=lambda: datetime.combine(day, moment, tzinfo=ZONE),
            ).run_once()
        )

    # ── 预测拍、结算 ──

    def tick(
        self,
        now: datetime,
        *,
        lane: str = "session",
        deliver: bool = True,
        vocabulary_of: Callable[[date], int | None] | None = None,
    ) -> TickReport:
        state = self.lanes[lane]
        run = ForesightTick(
            state.ledger,
            state.opportunities,
            self.read_side(),
            NoUnsealed(),
            lane=lane,
            vocabulary_version=lambda: 7,
            observer=state.recorder,
            clock=lambda: now,
            delivery=state.delivery if deliver else None,
            tick_seconds=SLOT.total_seconds(),
            recent=35,
            tolerance=TOLERANCE,
            track=TickTrack(
                TickWatermarkStore(state.root),
                slot=SLOT,
                tolerance_slots=ForesightConfig().unwatched_tolerance_slots,
                unwatched=lambda start, end: state.unwatched.append((start, end)),
            ),
            finished=state.marks.finished,
            vocabulary_of=vocabulary_of,
        )
        return asyncio.run(run.run_once())

    def ticks(self, day: date, start: time, end: time, *, lane: str = "session") -> list[TickReport]:
        """``start`` 到 ``end``（含）每 15 分钟一拍。"""

        reports = []
        now = datetime.combine(day, start, tzinfo=ZONE)
        while now <= datetime.combine(day, end, tzinfo=ZONE):
            reports.append(self.tick(now, lane=lane))
            now += SLOT
        return reports

    def settle(self, through: date, *, first: date, lane: str = "session") -> SettlementStage:
        state = self.lanes[lane]
        closed = tuple(first + timedelta(days=offset) for offset in range((through - first).days + 1))
        now = datetime.combine(through + timedelta(days=1), time(4), tzinfo=ZONE)
        stage = SettlementStage(
            state.ledger,
            self.read_side(),
            closed_days=lambda: closed,
            evidence=EVIDENCE,
            opportunities=state.opportunities,
            lane=lane,
            clock=lambda: now.astimezone(UTC),
            tolerance=LOSS,
        )
        stage.run_once()
        return stage

    # ── 读账 ──

    def expectations(self, day: date, *, source: str | None = None, lane: str = "session") -> list[Expectation]:
        found = self.lanes[lane].ledger.expectations_on(day)
        return [item for item in found if source is None or item.source == source]

    def earn(self, item: Expectation, *, lane: str = "session") -> None:
        """这条预期那本账已在提醒档（替身，同 ``test_foresight_delivery.Seeded.earn``）：到时窗末能开口。"""

        kind, source, trigger = parse_account(item.account)
        self.lanes[lane].seeded.earn(kind=kind, source=source, trigger=trigger, hits=8)

    def reasons(self, day: date, expectation_id: str, *, lane: str = "session") -> list[str]:
        return [
            item.reason
            for item in sorted(self.lanes[lane].ledger.deliveries_on(day), key=lambda item: item.decided_at)
            if item.expectation_id == expectation_id
        ]

    def spoken_kinds(self, day: date, *, lane: str = "session") -> list[str]:
        """这一天说出口的（进了「待取」或标了说了的）那几条预期的类，按决定先后。"""

        ledger = self.lanes[lane].ledger
        by_id = {item.expectation_id: item for item in ledger.expectations_on(day)}
        return [
            by_id[item.expectation_id].kind_token
            for item in sorted(ledger.deliveries_on(day), key=lambda item: item.decided_at)
            if (item.spoken or item.reason == "待取") and item.expectation_id in by_id
        ]


def world(tmp: Path, *, lanes: tuple[str, ...] = ("session",)) -> World:
    site = Site(tmp, now=at(BASE, 3, 0))
    built: dict[str, Lane] = {}
    for name in lanes:
        root = tmp / "foresight" / name
        ledger, opportunities = ForesightLedgerStore(root), OpportunityStore(root)
        seeded = Seeded(root, ledger)
        outlet = Outlet(alive=True)
        delivery = DeliveryStage(
            ledger,
            tolerance=LOSS,
            evidence=EVIDENCE,
            daily_cap=3,
            tick_seconds=SLOT.total_seconds(),
            outlet=outlet,
            history=seeded.store,
            lane=name,
        )
        marks = GenerationMarks(tmp / "scene" / name / "generation", artifact_root=tmp)
        built[name] = Lane(name, root, ledger, opportunities, marks, seeded, delivery, outlet)
    return World(tmp, site, built)


def mentions_day(event: ObservationEvent, night: date) -> bool:
    """观测属性里有一个值是那张表的日子（不猜字段名：按值找）。"""

    return any(value == night.isoformat() for value in event.attributes.values())


def records_table_day(item: Expectation, night: date) -> bool:
    """这条预期身上有一个字段记着那张表的日子（不猜字段名：按值找）。"""

    values = dataclasses.asdict(item).values()
    return night in values or night.isoformat() in values


def days_after(first: date, count: int) -> list[date]:
    return [first + timedelta(days=offset) for offset in range(count)]


__all__ = [
    "BANNED_WORDS",
    "BASE",
    "CODE",
    "DISCUSS",
    "DOCS",
    "SLOT",
    "ZONE",
    "World",
    "chain",
    "days_after",
    "mentions_day",
    "records_table_day",
    "world",
]
