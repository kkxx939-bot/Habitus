"""白天那一拍：读机会表、预测树、行为树今天与前几天，把此刻对得上历史的预期写进账本；同一身份只写一次；没有表或树时只报原因。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from habitus.foresight import NoUnsealed
from habitus.foresight.opportunities import build_table
from habitus.foundation.observability import ObservationEvent
from habitus.prediction.model import PredictionTree
from habitus.prediction.store import PredictionTreeStore
from habitus.runtime.foresight import ForesightReadSide
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.runtime.foresight_tick import ForesightTick
from habitus.scene import NominalCalendar
from habitus.scene.concepts import ConceptSet
from habitus.scene.relations import Relation, RelationConfig, RelationKey, Segment, Status, build_timelines
from habitus.series.reader import read_series
from tests.unit.kind_ids import kind_id
from tests.unit.scene.concept_fixtures import base
from tests.unit.scene.fixtures import CST, DAY1, SUBJECT, Site, at, publish

TODAY = DAY1 + timedelta(days=3)  # 2026-08-18
DISCUSS, CODE = kind_id("讨论方案"), kind_id("修改代码")
CONCEPTS = ConceptSet([base("讨论方案"), base("修改代码")])
CONFIG = RelationConfig(slot_minutes=15, transition_window_slots=3, recurrence_window_days=90)


class Recorder:
    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)


def tree() -> PredictionTree:
    return PredictionTree(
        built_at=datetime(2026, 8, 18, 3, 0, tzinfo=CST),
        reference_day=TODAY - timedelta(days=1),
        config_digest="d" * 64,
        slot_minutes=15,
        nodes={},
        curves={},
        weekday_baselines={},
        edges={},
        parallels={},
        parallel_totals={},
        recurrences={},
        exposure={},
        baselines={},
        actions=(),
        observed_days=3,
        censored_transitions=0.0,
    )


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
        build_table(line, relations, CONCEPTS, night=TODAY, relation_night=TODAY, slot_minutes=15, window_slots=3)
    )
    prediction = PredictionTreeStore(tmp_path / "prediction", retained_generations=3)
    prediction.publish(tree())
    return ledger, opportunities, prediction


def tick(
    tmp_path, site: Site, ledger, opportunities, prediction, now: datetime, recorder: Recorder | None = None
) -> ForesightTick:  # type: ignore[no-untyped-def]
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, NominalCalendar(), ZoneInfo("Asia/Shanghai"))
    return ForesightTick(
        ledger,
        opportunities,
        prediction,
        read_side,
        NoUnsealed(),
        vocabulary_version=lambda: 7,
        observer=recorder,
        clock=lambda: now,
    )


def test_a_tick_records_the_expectations_once_and_the_next_tick_adds_nothing(tmp_path) -> None:
    site, codes = site_with_three_mornings(tmp_path)
    ledger, opportunities, prediction = stores(tmp_path, site)
    discussion = publish(site.behavior_tree, TODAY, "讨论方案", 10, 0)
    recorder = Recorder()
    run = tick(tmp_path, site, ledger, opportunities, prediction, at(TODAY, 10, 15), recorder)
    report = asyncio.run(run.run_once())
    assert (
        report.skipped is None and report.formed == report.recorded == 3 and report.by_source == {"事件": 1, "日子": 2}
    )
    stored = {(item.source, item.kind_token): item for item in ledger.expectations_on(TODAY)}
    event = stored[("事件", CODE)]
    assert event.kind_token == CODE and event.basis == tuple(codes)
    assert event.trigger_instance == at(TODAY, 10, 0).astimezone(UTC).isoformat()  # 前因的开始时刻
    assert event.opportunities == event.hits == 3 and event.generation and event.version == "7" and event.night == TODAY
    from habitus.runtime.foresight_implementation import IMPLEMENTATION

    assert event.implementation == IMPLEMENTATION  # 形成它的那份代码
    assert event.formed_at == at(TODAY, 10, 15) and event.window[0] >= event.formed_at
    assert event.observed_from == at(TODAY, 10, 0)  # 从前因做完那一槽起算
    assert event.antecedents[0].present and event.antecedents[0].source == discussion
    # 讨论方案今天 10:00 已经做了：它的"离上次够久"照建（裁定 49），从今天零点起算，结算会给准时
    (done,) = [item for (source, kind), item in stored.items() if source == "日子" and kind != CODE]
    assert done.observed_from == at(TODAY, 0, 0) and done.window[0] <= at(TODAY, 10, 0) < done.window[1]
    interval = stored[("日子", CODE)]
    assert interval.kind_token == CODE and interval.trigger == "interval"
    assert interval.trigger_instance == f"{codes[-1]}@{TODAY.isoformat()}"  # 上次那条 + 这一簇的第一天
    again = asyncio.run(run.run_once())
    assert again.formed == 3 and again.recorded == 0
    assert [event.operation for event in recorder.events] == ["tick", "tick"]
    assert recorder.events[0].attributes["recorded"] == 3 and recorder.events[1].attributes["recorded"] == 0


def test_without_a_table_or_a_published_tree_the_tick_only_reports_why(tmp_path) -> None:
    site, _codes = site_with_three_mornings(tmp_path)
    root = tmp_path / "foresight" / "session"
    prediction = PredictionTreeStore(tmp_path / "prediction", retained_generations=3)
    run = tick(tmp_path, site, ForesightLedgerStore(root), OpportunityStore(root), prediction, at(TODAY, 10, 15))
    assert asyncio.run(run.run_once()).skipped == "还没有机会表"
    ledger, opportunities, _prediction = stores(tmp_path, site)
    run = tick(
        tmp_path,
        site,
        ledger,
        opportunities,
        PredictionTreeStore(tmp_path / "empty", retained_generations=3),
        at(TODAY, 10, 15),
    )
    assert asyncio.run(run.run_once()).skipped == "预测树还没发布"


def test_a_vocabulary_migration_in_progress_skips_the_whole_tick(tmp_path) -> None:
    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities, prediction = stores(tmp_path, site)
    publish(site.behavior_tree, TODAY, "讨论方案", 10, 0)
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, NominalCalendar(), ZoneInfo("Asia/Shanghai"))
    run = ForesightTick(
        ledger,
        opportunities,
        prediction,
        read_side,
        NoUnsealed(),
        migrating=lambda: True,
        clock=lambda: at(TODAY, 10, 15),
    )
    report = asyncio.run(run.run_once())
    assert report.skipped == "词表迁移还没做完" and report.formed == 0 and ledger.expectations_on(TODAY) == ()


def test_the_tick_checks_its_collaborators(tmp_path) -> None:
    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities, prediction = stores(tmp_path, site)
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, NominalCalendar(), ZoneInfo("Asia/Shanghai"))
    with pytest.raises(TypeError):
        ForesightTick(object(), opportunities, prediction, read_side, NoUnsealed())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ForesightTick(ledger, opportunities, prediction, object(), NoUnsealed())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ForesightTick(ledger, opportunities, prediction, read_side, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ForesightTick(ledger, opportunities, prediction, read_side, NoUnsealed(), lane="")
    with pytest.raises(TypeError):
        ForesightTick(ledger, opportunities, prediction, read_side, NoUnsealed(), migrating=object())  # type: ignore[arg-type]
