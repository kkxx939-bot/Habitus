"""预测层的那一拍（N1）接在换代（``Generation``）的语义树之后：用这一代的时间线与概念集，读这一代的关系表，写机会表。"""

from __future__ import annotations

from datetime import timedelta
from zoneinfo import ZoneInfo

from habitus.conditions import LocalCalendar
from habitus.foresight import Evidence, ForesightError
from habitus.foresight.history import RelabelDelta
from habitus.foundation.observability import ObservationEvent
from habitus.runtime.foresight import ForesightReadSide, ForesightRuntimeComponents, build_foresight_nightly
from habitus.runtime.foresight_history import HistoryOutcome, HistoryStage
from habitus.runtime.foresight_history_store import HistoryStore
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_nightly import ForesightNightlyRun
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.runtime.foresight_settlement import SettlementStage
from habitus.runtime.generation_result import DEGRADED, DONE, FAILED, MIGRATING, OPPORTUNITIES, SCENE, SERIES, SETTLE
from habitus.scene.concepts.rhythm import RhythmRulers
from habitus.scene.recurrence import recurrences_of
from habitus.scene.relations import build_timelines
from habitus.scene.relations.store import RelationStore
from tests.unit.runtime.test_generation import assembled_with, generate
from tests.unit.runtime.test_scene_night import NOW, SLEEP, TONIGHT, nightly, publish_tonight, site_with_three_bedtimes
from tests.unit.scene.fixtures import SUBJECT
from tests.unit.scene.relation_fixtures import CONCEPTS, CONFIG, planted


class Recorder:
    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)


def components(tmp_path, site) -> ForesightRuntimeComponents:  # type: ignore[no-untyped-def]
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai"))
    ledger = ForesightLedgerStore(tmp_path / "foresight" / "session")
    settlement = SettlementStage(
        ledger,
        read_side,
        closed_days=tuple,
        current=lambda token: (token,),
        migrating=lambda: False,
        evidence=Evidence(alpha=0.05, half_life=8.0),
    )
    history = HistoryStage(
        ledger,
        HistoryStore(tmp_path / "foresight" / "session"),
        evidence=Evidence(alpha=0.05, half_life=8.0),
        statuses=("established", "candidate"),
    )
    return ForesightRuntimeComponents(
        read_side=read_side,
        ledger=ledger,
        settlement=settlement,
        opportunities=OpportunityStore(tmp_path / "foresight" / "session"),
        history=history,
    )


def test_the_opportunity_table_is_written_after_the_scene_run_from_the_same_night(tmp_path) -> None:
    site, hits = site_with_three_bedtimes(tmp_path)
    scene, _provider = nightly(tmp_path, hits, site)
    publish_tonight(site)
    built = components(tmp_path, site)
    recorder = Recorder()
    foresight = build_foresight_nightly(
        built, relations=scene.relations, relation_config=scene.relation_config, observer=recorder, clock=lambda: NOW
    )
    run = assembled_with(tmp_path, site, scene, foresight=foresight)
    # 重标单元的历史排在同一拍、读的是这一支的账本（裁定 55 新版第 2 条）
    assert foresight.history is built.history and foresight.history.ledger is built.ledger
    result = generate(run)
    scene_step, opportunity_step = result.step(SCENE), result.step(OPPORTUNITIES)
    assert scene_step is not None and opportunity_step is not None
    scene_report, opportunities = scene_step.report, opportunity_step.report
    cutoff = result.series_cutoff
    assert cutoff is not None
    summary = opportunities["lanes"]["session"]  # type: ignore[index]
    assert summary["moments"] == 0  # 账本里还没有结过的预期
    assert built.history.store.latest_night("session") == cutoff  # 这一晚的单元历史落了盘
    assert opportunities["night"] == scene_report["night"] == cutoff.isoformat()
    # 几天的数据：关系全是样本不够，表上没有关系；日历上有就寝等几个类
    assert isinstance(summary, dict)
    assert summary["relations"] == 0 and summary["kinds"] >= 1 and summary["relation_night"] == cutoff.isoformat()
    stored = built.opportunities.read("session", cutoff)
    assert stored.relation_night == cutoff and stored.slot_minutes == scene.relation_config.slot_minutes
    assert stored.window_slots == scene.relation_config.transition_window_slots
    assert stored.calendar_version == built.read_side.calendar.version  # 记下这一晚用的放假数据版本（裁定 59 第 4 条）
    sleep = stored.kind(SLEEP)
    # 前三天的就寝只写在命中盘上、不在行为树上（夹具），日历只认行为树：只有今晚那条
    assert sleep is not None and len(sleep.days) == 1 and sleep.days[-1].day == TONIGHT and sleep.days[-1].count == 1
    assert opportunities["signals"] == [] and opportunities["failed"] == [] and opportunities["degraded"] == []
    # 单元历史重标先发一条观测（W13），再是机会表那条总账；两条都成功
    assert [event.operation for event in recorder.events] == ["unit_history", "opportunities"] and all(
        event.status.value == "success" for event in recorder.events
    )


def test_a_night_without_a_relation_table_falls_back_and_says_so(tmp_path) -> None:
    """语义树这一晚没写出关系表（或还一晚都没有）：退到更早的最晚一晚、或只有日历，都在信号里说明。"""

    relations = RelationStore(tmp_path / "scene")
    store = OpportunityStore(tmp_path / "foresight")
    run = ForesightNightlyRun(store, relations, CONFIG)
    series = planted(days=5)
    lines = build_timelines(series, CONCEPTS, {}, CONFIG)
    recurrences = recurrences_of(series, RhythmRulers(15, 2700, 3, 5.0, 5.0, 0.10, 35))
    report = run.run(night=series.cutoff, concepts=CONCEPTS, timelines=lines, recurrences=recurrences)
    assert report.lanes["session"].relation_night is None and report.lanes["session"].kinds == 5
    assert any("还没有关系表" in signal for signal in report.signals)
    assert store.read("session", series.cutoff).relation_night is None
    # 盘上有更早一晚的关系表：用它、记下来
    from habitus.scene.relations import LaneState

    earlier = series.cutoff - timedelta(days=2)
    relations.write(LaneState(lane="session", night=earlier, family=0, relations=()), ())
    later = run.run(night=series.cutoff + timedelta(days=1), concepts=CONCEPTS, timelines=lines)
    assert later.lanes["session"].relation_night == earlier
    assert any(f"用第 {earlier} 晚的" in signal for signal in later.signals)


def test_a_failing_lane_does_not_stop_the_night(tmp_path, monkeypatch) -> None:
    recorder = Recorder()
    store = OpportunityStore(tmp_path / "foresight")
    run = ForesightNightlyRun(store, RelationStore(tmp_path / "scene"), CONFIG, observer=recorder)
    series = planted(days=3)
    lines = build_timelines(series, CONCEPTS, {}, CONFIG)
    store.write(
        __import__("habitus.foresight.opportunities", fromlist=["OpportunityTable"]).OpportunityTable(
            "session", series.cutoff + timedelta(days=5), None, 15, 3, (), ()
        )
    )  # 盘上已有更晚一晚：这一晚写不进去
    report = run.run(night=series.cutoff, concepts=CONCEPTS, timelines=lines)
    assert report.lanes == {} and any("没算出来" in signal for signal in report.signals)
    # 那条 lane 自己一笔失败；总账的口径是每条 lane 都写出来了才算成功
    assert [event.status.value for event in recorder.events] == ["failure", "failure"]
    assert recorder.events[-1].attributes["lanes_failed"] == 1


def _night_with_settlement(tmp_path):  # type: ignore[no-untyped-def]
    site, hits = site_with_three_bedtimes(tmp_path)
    scene, _provider = nightly(tmp_path, hits, site)
    publish_tonight(site)
    built = components(tmp_path, site)
    foresight = build_foresight_nightly(
        built, relations=scene.relations, relation_config=scene.relation_config, clock=lambda: NOW
    )
    return built, foresight, scene, site


def test_the_settlement_runs_first_so_the_relabelled_history_sees_tonights_results(tmp_path, monkeypatch) -> None:
    """结算排在整晚的最前、重标单元历史在最后：今晚结出的那几次同一晚就进单元的历史，第二天白天的单元读得到，不晚一天。"""

    built, foresight, scene, site = _night_with_settlement(tmp_path)
    order: list[str] = []
    settle, scene_run, relabel = built.settlement.run_once, scene.run, built.history.run
    monkeypatch.setattr(built.settlement, "run_once", lambda: (order.append("结算"), settle())[1])

    async def scene_then(**kwargs):  # type: ignore[no-untyped-def]
        order.append("语义树")
        return await scene_run(**kwargs)

    monkeypatch.setattr(scene, "run", scene_then)
    monkeypatch.setattr(built.history, "run", lambda *a, **k: (order.append("重标历史"), relabel(*a, **k))[1])
    run = assembled_with(tmp_path, site, scene, foresight=foresight, settlement=built.settlement)
    result = generate(run)
    assert order == ["结算", "语义树", "重标历史"]
    settle, opportunities = result.step(SETTLE), result.step(OPPORTUNITIES)
    assert settle is not None and "settled" in settle.report and opportunities is not None
    # 结算读到的机会表那一晚早于这一代写的那一晚（必须边"结算 → 写新机会表"，E17）：这里是还一张都没有
    assert settle.report["opportunity_night"] is None and opportunities.report["night"] == str(result.series_cutoff)


def test_a_migration_left_half_done_defers_the_settlement_loudly_and_stops_the_series(tmp_path, monkeypatch) -> None:
    """词表迁移做到一半、续做迁移那一步也没做完（F1）：结算不跑（树上一半新编号一半旧编号），计入"结算留到下次"并报出
    （``degraded``，原来结算自己整拍静默跳过）；读序列那一步不成，这一代不成。"""

    built, foresight, scene, site = _night_with_settlement(tmp_path)
    calls: list[str] = []
    settle = built.settlement.run_once
    monkeypatch.setattr(built.settlement, "run_once", lambda: (calls.append("结算"), settle())[1])
    run = assembled_with(
        tmp_path, site, scene, foresight=foresight, settlement=built.settlement, vocabulary=lambda: (2, True)
    )
    result = generate(run)
    series = result.step(SERIES)
    assert series is not None and series.outcome == FAILED and series.reason == MIGRATING
    assert result.step(SCENE) is None and calls == []
    settled = result.step(SETTLE)
    assert settled is not None and settled.outcome == DEGRADED and settled.reason == MIGRATING
    assert result.fallbacks["settlement_deferred"] == 1


def test_a_failing_settlement_is_left_for_the_next_generation(tmp_path, monkeypatch) -> None:
    """结算抛了：尽力而为——报出、记"结算留到下次"（下一代补结，幂等），语义树与机会表照做（实时化方案第 3 节；原来是停下这一晚）。"""

    built, foresight, scene, site = _night_with_settlement(tmp_path)

    def broken():  # type: ignore[no-untyped-def]
        raise RuntimeError("账本坏了")

    monkeypatch.setattr(built.settlement, "run_once", broken)
    run = assembled_with(tmp_path, site, scene, foresight=foresight, settlement=built.settlement)
    result = generate(run)
    settle = result.step(SETTLE)
    assert settle is not None and settle.outcome == DEGRADED and settle.reason == "结算出错: RuntimeError"
    assert "settled" not in settle.report and result.fallbacks["settlement_deferred"] == 1
    opportunities = result.step(OPPORTUNITIES)
    assert opportunities is not None and opportunities.outcome == DONE


def test_a_failing_unit_history_is_reported_while_the_table_is_still_written(tmp_path) -> None:
    """机会表写成了、单元历史没写成：表照常给白天用，失败单独报出来，不再被一并吞掉（26 号清单 2.5.1）。"""

    class BrokenHistory(HistoryStage):
        def run(self, *, night, table, line):  # type: ignore[no-untyped-def, override]
            raise ForesightError("unit history exceeds its configured bound")

    site, _hits = site_with_three_bedtimes(tmp_path)
    built = components(tmp_path, site)
    broken = BrokenHistory(
        built.ledger,
        HistoryStore(tmp_path / "foresight" / "session"),
        evidence=Evidence(alpha=0.05, half_life=8.0),
        statuses=("established", "candidate"),
    )
    store = OpportunityStore(tmp_path / "foresight")
    run = ForesightNightlyRun(store, RelationStore(tmp_path / "scene"), CONFIG, history=broken)
    series = planted(days=3)
    report = run.run(night=series.cutoff, concepts=CONCEPTS, timelines=build_timelines(series, CONCEPTS, {}, CONFIG))

    assert "session" in report.lanes  # 表写成了
    # 机会表必须成、单元历史尽力：分开报（实时化方案第 3 节）
    assert report.failed == () and report.degraded == ("session: 单元历史: ForesightError",)
    assert report.history_skipped == 1
    assert any("单元历史没写成" in signal for signal in report.signals)


def test_a_night_whose_unit_history_failed_does_not_compact_the_ledger(tmp_path) -> None:
    """单元历史没把结完的日子记下来：这一晚压实会删掉它还没记的原始结果，下一晚它就拿不到了——这一晚跳过压实
    （L10.11）。没有单元历史就不许配压实。"""

    import pytest

    from habitus.runtime.foresight_compaction import LedgerCompaction

    class BrokenHistory(HistoryStage):
        def run(self, *, night, table, line):  # type: ignore[no-untyped-def, override]
            raise ForesightError("unit history exceeds its configured bound")

    site, _hits = site_with_three_bedtimes(tmp_path)
    built = components(tmp_path, site)
    broken = BrokenHistory(
        built.ledger,
        HistoryStore(tmp_path / "foresight" / "session"),
        evidence=Evidence(alpha=0.05, half_life=8.0),
        statuses=("established", "candidate"),
    )
    compaction = LedgerCompaction(built.ledger, evidence=Evidence(alpha=0.05, half_life=8.0), lane="session")
    calls: list[object] = []
    compaction.run = lambda **kwargs: calls.append(kwargs)  # type: ignore[method-assign]
    store = OpportunityStore(tmp_path / "foresight")
    run = ForesightNightlyRun(store, RelationStore(tmp_path / "scene"), CONFIG, history=broken, compaction=compaction)
    series = planted(days=3)

    report = run.run(night=series.cutoff, concepts=CONCEPTS, timelines=build_timelines(series, CONCEPTS, {}, CONFIG))

    assert calls == []
    assert "session" in report.lanes
    assert any("压实这一晚跳过" in signal for signal in report.signals)
    with pytest.raises(ValueError, match="requires the unit history"):
        ForesightNightlyRun(store, RelationStore(tmp_path / "scene"), CONFIG, compaction=compaction)


def test_the_ledger_is_compacted_only_after_the_unit_history_has_run(tmp_path) -> None:
    """单元历史读的是压实之前那一份账：同一晚里压实排在它后面（26 号清单 4.8 / L10.11）。"""

    from habitus.runtime.foresight_compaction import CompactionReport, LedgerCompaction

    order: list[str] = []

    class SpyHistory(HistoryStage):
        def run(self, *, night, table, line):  # type: ignore[no-untyped-def, override]
            order.append("history")
            return HistoryOutcome(0, RelabelDelta(0, 0))

    site, _hits = site_with_three_bedtimes(tmp_path)
    built = components(tmp_path, site)
    history = SpyHistory(
        built.ledger,
        HistoryStore(tmp_path / "foresight" / "session"),
        evidence=Evidence(alpha=0.05, half_life=8.0),
        statuses=("established", "candidate"),
    )
    compaction = LedgerCompaction(built.ledger, evidence=Evidence(alpha=0.05, half_life=8.0), lane="session")

    def compact(**_kwargs):  # type: ignore[no-untyped-def]
        order.append("compaction")
        return CompactionReport()

    compaction.run = compact  # type: ignore[method-assign]
    run = ForesightNightlyRun(
        OpportunityStore(tmp_path / "foresight"),
        RelationStore(tmp_path / "scene"),
        CONFIG,
        history=history,
        compaction=compaction,
    )
    series = planted(days=3)

    run.run(night=series.cutoff, concepts=CONCEPTS, timelines=build_timelines(series, CONCEPTS, {}, CONFIG))

    assert order == ["history", "compaction"]


def test_a_bad_unit_moment_record_still_lets_the_ledger_be_compacted_and_is_counted(tmp_path) -> None:
    """段 6 Q2：单元历史里一份坏记录原来让这一晚的单元历史抛、压实跟着跳过，每晚都这样；坏的那一份跳过、计入观测，
    单元历史照写，压实照常调用。"""

    from habitus.runtime.foresight_compaction import CompactionReport, LedgerCompaction

    recorder = Recorder()
    site, _hits = site_with_three_bedtimes(tmp_path)
    built = components(tmp_path, site)
    moments = built.history.store.root / "unit_moments" / "session"
    moments.mkdir(parents=True)
    bad = moments / "old.json"
    bad.write_text('{"schema_version": 999}', encoding="utf-8")
    compaction = LedgerCompaction(built.ledger, evidence=Evidence(alpha=0.05, half_life=8.0), lane="session")
    calls: list[object] = []

    def compact(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return CompactionReport()

    compaction.run = compact  # type: ignore[method-assign]
    run = ForesightNightlyRun(
        OpportunityStore(tmp_path / "foresight" / "session"),
        RelationStore(tmp_path / "scene"),
        CONFIG,
        history=built.history,
        compaction=compaction,
        observer=recorder,
    )
    series = planted(days=3)
    report = run.run(night=series.cutoff, concepts=CONCEPTS, timelines=build_timelines(series, CONCEPTS, {}, CONFIG))
    assert report.failed == () and len(calls) == 1
    assert built.history.store.latest("session") is not None
    assert recorder.events[-1].attributes["unreadable"] == 1 and bad.exists()
