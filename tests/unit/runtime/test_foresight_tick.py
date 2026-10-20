"""白天那一拍：读机会表、今天的日历条件、行为树今天与前几天，把此刻对得上历史的预期写进账本；同一身份只写一次；没有表时只报原因。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from habitus.conditions import LocalCalendar
from habitus.foresight import NoUnsealed
from habitus.foresight.opportunities import Opportunity, OpportunityTable, RelationOpportunities, build_table
from habitus.foundation.observability import ObservationEvent
from habitus.runtime.foresight import ForesightReadSide
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.runtime.foresight_tick import ForesightTick
from habitus.scene.concepts import ConceptSet
from habitus.scene.concepts.rhythm import RhythmRulers
from habitus.scene.recurrence import recurrences_of
from habitus.scene.relations import Relation, RelationConfig, RelationKey, Segment, Status, build_timelines
from habitus.series.reader import read_series
from tests.unit.kind_ids import kind_id
from tests.unit.scene.concept_fixtures import base
from tests.unit.scene.fixtures import DAY1, SUBJECT, Site, at, publish

TODAY = DAY1 + timedelta(days=3)  # 2026-08-18
DISCUSS, CODE = kind_id("讨论方案"), kind_id("修改代码")
CONCEPTS = ConceptSet([base("讨论方案"), base("修改代码")])
CONFIG = RelationConfig(slot_minutes=15, transition_window_slots=3)
RULERS = RhythmRulers(15, 2700, 3, 5.0, 5.0, 0.10, 35)


class Recorder:
    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)


def site_with_three_mornings(tmp_path) -> tuple[Site, list[str]]:  # type: ignore[no-untyped-def]
    """前三天每天 10:00 讨论方案、10:20 修改代码、12:00 写文档（人在场到中午，转移窗口才整段在场）。"""

    site = Site(tmp_path, now=at(TODAY, 3, 0))
    codes = []
    for offset in range(3):
        day = DAY1 + timedelta(days=offset)
        publish(site.behavior_tree, day, "讨论方案", 10, 0)
        codes.append(publish(site.behavior_tree, day, "修改代码", 10, 20))
        publish(site.behavior_tree, day, "写文档", 12, 0)
    return site, codes


def stores(tmp_path, site: Site):  # type: ignore[no-untyped-def]
    root = tmp_path / "foresight" / "session"
    ledger, opportunities = ForesightLedgerStore(root), OpportunityStore(root)
    series = read_series(site.behavior_tree, cutoff=TODAY)
    line = build_timelines(series, CONCEPTS, {}, CONFIG)["session"]
    relations = (
        Relation(key=RelationKey("session", DISCUSS, CODE, Segment.CHAIN), status=Status.ESTABLISHED, upward=True),
    )
    opportunities.write(
        build_table(
            line,
            relations,
            CONCEPTS,
            night=TODAY,
            relation_night=TODAY,
            slot_minutes=15,
            window_slots=3,
            recurrences=recurrences_of(series, RULERS),
        )
    )
    return ledger, opportunities


def tick(tmp_path, site: Site, ledger, opportunities, now: datetime, recorder: Recorder | None = None) -> ForesightTick:  # type: ignore[no-untyped-def]
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai"))
    return ForesightTick(
        ledger,
        opportunities,
        read_side,
        NoUnsealed(),
        vocabulary_version=lambda: 7,
        observer=recorder,
        clock=lambda: now,
        recent=35,
    )


def test_a_tick_records_the_expectations_once_and_the_next_tick_adds_nothing(tmp_path) -> None:
    site, codes = site_with_three_mornings(tmp_path)
    ledger, opportunities = stores(tmp_path, site)
    discussion = publish(site.behavior_tree, TODAY, "讨论方案", 10, 0)
    recorder = Recorder()
    run = tick(tmp_path, site, ledger, opportunities, at(TODAY, 10, 15), recorder)
    report = asyncio.run(run.run_once())
    assert (
        report.skipped is None and report.formed == report.recorded == 4 and report.by_source == {"事件": 1, "日子": 3}
    )
    stored = {(item.source, item.kind_token): item for item in ledger.expectations_on(TODAY)}
    event = stored[("事件", CODE)]
    assert event.kind_token == CODE and event.basis == tuple(codes)
    assert event.trigger_instance == at(TODAY, 10, 0).astimezone(UTC).isoformat()  # 前因的开始时刻
    assert event.opportunities == event.hits == 3 and event.version == "7" and event.night == TODAY
    assert event.generation == TODAY.isoformat()  # 出处记机会表那一晚（预测树删了，裁定 59）
    from habitus.runtime.foresight_implementation import IMPLEMENTATION

    assert event.implementation == IMPLEMENTATION  # 形成它的那份代码
    assert event.formed_at == at(TODAY, 10, 15) and event.window[0] >= event.formed_at
    assert event.observed_from == at(TODAY, 10, 0)  # 从前因做完那一槽起算
    assert event.antecedents[0].present and event.antecedents[0].source == discussion
    # 讨论方案今天 10:00 已经做了：它的"离上次够久"照建（裁定 49），从今天零点起算，结算会给准时；写文档的复现也登记了
    done = stored[("日子", DISCUSS)]
    assert done.observed_from == at(TODAY, 0, 0) and done.window[0] <= at(TODAY, 10, 0) < done.window[1]
    interval = stored[("日子", CODE)]
    assert interval.kind_token == CODE and interval.trigger == "interval"
    assert interval.trigger_instance == f"{codes[-1]}@{TODAY.isoformat()}"  # 上次那条 + 这一簇的第一天
    again = asyncio.run(run.run_once())
    assert again.formed == 4 and again.recorded == 0
    assert [event.operation for event in recorder.events] == ["tick", "tick"]
    assert recorder.events[0].attributes["recorded"] == 4 and recorder.events[1].attributes["recorded"] == 0


def test_without_a_table_the_tick_only_reports_why(tmp_path) -> None:
    site, _codes = site_with_three_mornings(tmp_path)
    root = tmp_path / "foresight" / "session"
    run = tick(tmp_path, site, ForesightLedgerStore(root), OpportunityStore(root), at(TODAY, 10, 15))
    assert asyncio.run(run.run_once()).skipped == "还没有机会表"


def test_the_tick_reads_todays_calendar_conditions_for_situation_expectations(tmp_path) -> None:
    """机会表里有"周二 → 修改代码"：今天（周二）的日历条件从读侧的日历来，到了往常的时刻就建一条「情境」预期。"""

    site, codes = site_with_three_mornings(tmp_path)
    root = tmp_path / "foresight" / "session"
    ledger, opportunities = ForesightLedgerStore(root), OpportunityStore(root)
    tuesdays = [TODAY - timedelta(days=7 * weeks) for weeks in (2, 1)]
    items = tuple(
        Opportunity(f"calendar:{day}", (f"calendar:{day}",), day, at(day, 0, 0), True, codes[index], at(day, 10, 20))
        for index, day in enumerate(tuesdays)
    )
    relation = RelationOpportunities(
        key="session|周二|修改代码|situation",
        antecedent="周二",
        consequent=CODE,
        antecedent_label="周二",
        consequent_label="修改代码",
        antecedent_classes=("周二",),
        consequent_classes=(CODE,),
        segment="situation",
        status="established",
        upward=True,
        items=items,
        calendar_key="日历.周几",
        calendar_values=("周二",),
    )
    opportunities.write(OpportunityTable("session", TODAY, TODAY, 15, 3, (relation,), ()))
    assert TODAY.weekday() == 1
    report = asyncio.run(tick(tmp_path, site, ledger, opportunities, at(TODAY, 10, 25)).run_once())
    assert report.by_source == {"情境": 1}
    (stored,) = ledger.expectations_on(TODAY)
    assert stored.source == "情境" and stored.trigger == relation.key and stored.trigger_instance == TODAY.isoformat()


def test_a_vocabulary_migration_in_progress_skips_the_whole_tick(tmp_path) -> None:
    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities = stores(tmp_path, site)
    publish(site.behavior_tree, TODAY, "讨论方案", 10, 0)
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai"))
    run = ForesightTick(
        ledger,
        opportunities,
        read_side,
        NoUnsealed(),
        migrating=lambda: True,
        clock=lambda: at(TODAY, 10, 15),
        recent=35,
    )
    report = asyncio.run(run.run_once())
    assert report.skipped == "词表迁移还没做完" and report.formed == 0 and ledger.expectations_on(TODAY) == ()


def test_the_tick_checks_its_collaborators(tmp_path) -> None:
    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities = stores(tmp_path, site)
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai"))
    with pytest.raises(TypeError):
        ForesightTick(object(), opportunities, read_side, NoUnsealed(), recent=35)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ForesightTick(ledger, opportunities, object(), NoUnsealed(), recent=35)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ForesightTick(ledger, opportunities, read_side, object(), recent=35)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ForesightTick(ledger, opportunities, read_side, NoUnsealed(), lane="", recent=35)
    with pytest.raises(TypeError):
        ForesightTick(ledger, opportunities, read_side, NoUnsealed(), migrating=object(), recent=35)  # type: ignore[arg-type]


def test_an_abandoned_attempt_is_seen_but_not_as_done(tmp_path) -> None:
    """开始了但放弃了的那一行：白天看得到，可不算这件事做了——时窗末不当「已发生」不说，夜里结算也不认它（一条记录一种读法，
    裁定 54；2026-10-17 四方评审：原来白天判已发生、夜里又记一笔坏）。"""

    from habitus.foresight import Moment
    from habitus.foresight.triggers.assemble import seen_rows
    from habitus.scene.views import DayIndexCache
    from habitus.series.reader import admitted

    site, _codes = site_with_three_mornings(tmp_path)
    publish(site.behavior_tree, TODAY, "修改代码", 10, 20, status="abandoned")
    cache = DayIndexCache(site.behavior_tree, subject=SUBJECT, admits=admitted, calendar=LocalCalendar())
    moment = Moment(at(TODAY, 11, 0), slot=44)
    today = [row for row in seen_rows(moment, cache, ()) if row.started_at.date() == TODAY]
    assert [(row.kind_token, row.abandoned) for row in today] == [(CODE, True)]
