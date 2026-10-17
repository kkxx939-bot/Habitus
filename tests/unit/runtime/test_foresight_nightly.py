"""预测层的夜批那一拍（N1）接在 ``Nightly.run_once`` 的语义树之后：用这一晚的时间线与概念集，读这一晚的关系表，写机会表。"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from zoneinfo import ZoneInfo

from habitus.foresight import Evidence
from habitus.foundation.observability import ObservationEvent
from habitus.prediction.store import PredictionTreeStore
from habitus.runtime.foresight import ForesightReadSide, ForesightRuntimeComponents, build_foresight_nightly
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_nightly import ForesightNightlyRun
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.runtime.foresight_settlement import SettlementStage
from habitus.runtime.nightly import Nightly
from habitus.runtime.prediction import PredictionRebuilder
from habitus.scene import NominalCalendar
from habitus.scene.concepts import ConceptSet
from habitus.scene.relations import build_timelines
from habitus.scene.relations.store import RelationStore
from tests.unit.prediction.prediction_fixtures import config
from tests.unit.runtime.test_nightly import sealed_through, steady
from tests.unit.runtime.test_scene_night import NOW, SLEEP, TONIGHT, nightly, publish_tonight, site_with_three_bedtimes
from tests.unit.scene.fixtures import SUBJECT
from tests.unit.scene.relation_fixtures import CONCEPTS, CONFIG, planted


class Recorder:
    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)


def components(tmp_path, site) -> ForesightRuntimeComponents:  # type: ignore[no-untyped-def]
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, NominalCalendar(), ZoneInfo("Asia/Shanghai"))
    ledger = ForesightLedgerStore(tmp_path / "foresight" / "session")
    settlement = SettlementStage(
        ledger,
        read_side,
        closed_days=tuple,
        current=lambda token: (token,),
        migrating=lambda: False,
        evidence=Evidence(alpha=0.05, half_life_days=60.0),
    )
    return ForesightRuntimeComponents(
        read_side=read_side,
        ledger=ledger,
        settlement=settlement,
        opportunities=OpportunityStore(tmp_path / "foresight" / "session"),
    )


def test_the_opportunity_table_is_written_after_the_scene_run_from_the_same_night(tmp_path) -> None:
    site, hits = site_with_three_bedtimes(tmp_path)
    scene, _provider = nightly(tmp_path, hits, site)
    publish_tonight(site)
    prediction = PredictionRebuilder(
        site.behavior_tree,
        PredictionTreeStore(tmp_path / "prediction", retained_generations=3),
        config=config(),
        zone=ZoneInfo("Asia/Shanghai"),
        clock=lambda: NOW,
    )
    built = components(tmp_path, site)
    recorder = Recorder()
    foresight = build_foresight_nightly(
        built, relations=scene.relations, relation_config=scene.relation_config, observer=recorder, clock=lambda: NOW
    )
    run = Nightly(
        behavior_tree=site.behavior_tree,
        prediction=prediction,
        scene=scene,
        closed_days=sealed_through(TONIGHT),
        vocabulary=steady,
        clock=lambda: NOW,
        foresight=foresight,
    )
    # 给落空找原因排在同一拍、接的是这一支的账本（裁定 55）
    assert foresight.causes is not None and foresight.causes.ledger is built.ledger and foresight.causes.lane == "session"
    outcome = asyncio.run(run.run_once())
    assert outcome.scene is not None and outcome.opportunities is not None
    assert outcome.opportunities.lanes["session"].explained == 0  # 账本里还没有落空
    assert outcome.opportunities.night == outcome.scene.night == outcome.series.cutoff
    summary = outcome.opportunities.lanes["session"]
    # 几天的数据：关系全是样本不够，表上没有关系；日历上有就寝等几个类
    assert summary.relations == 0 and summary.kinds >= 1 and summary.relation_night == outcome.series.cutoff
    stored = built.opportunities.read("session", outcome.series.cutoff)
    assert stored.relation_night == outcome.series.cutoff and stored.slot_minutes == scene.relation_config.slot_minutes
    assert stored.window_slots == scene.relation_config.transition_window_slots
    sleep = stored.kind(SLEEP)
    # 前三天的就寝只写在命中盘上、不在行为树上（夹具），日历只认行为树：只有今晚那条
    assert sleep is not None and len(sleep.days) == 1 and sleep.days[-1].day == TONIGHT and sleep.days[-1].count == 1
    assert stored.present_days == outcome.scene.materials.timelines["session"].days()  # type: ignore[union-attr]
    assert outcome.opportunities.signals == () and "机会表" in outcome.opportunities.summary()
    assert [event.operation for event in recorder.events] == ["opportunities"] and recorder.events[
        0
    ].status.value == "success"


def test_a_night_without_a_relation_table_falls_back_and_says_so(tmp_path) -> None:
    """语义树这一晚没写出关系表（或还一晚都没有）：退到更早的最晚一晚、或只有日历，都在信号里说明。"""

    relations = RelationStore(tmp_path / "scene")
    store = OpportunityStore(tmp_path / "foresight")
    run = ForesightNightlyRun(store, relations, CONFIG)
    series = planted(days=5)
    lines = build_timelines(series, CONCEPTS, {}, CONFIG)
    report = run.run(night=series.cutoff, concepts=CONCEPTS, timelines=lines)
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
            "session", series.cutoff + timedelta(days=5), None, 15, 3, (), (), ()
        )
    )  # 盘上已有更晚一晚：这一晚写不进去
    report = run.run(night=series.cutoff, concepts=CONCEPTS, timelines=lines)
    assert report.lanes == {} and any("没算出来" in signal for signal in report.signals)
    # 那条 lane 自己一笔失败；总账的口径是每条 lane 都写出来了才算成功
    assert [event.status.value for event in recorder.events] == ["failure", "failure"]
    assert recorder.events[-1].attributes["lanes_failed"] == 1


def test_nightly_refuses_a_foresight_step_of_the_wrong_type(tmp_path) -> None:
    import pytest

    site, hits = site_with_three_bedtimes(tmp_path)
    scene, _provider = nightly(tmp_path, hits, site)
    prediction = PredictionRebuilder(
        site.behavior_tree,
        PredictionTreeStore(tmp_path / "prediction", retained_generations=3),
        config=config(),
        zone=ZoneInfo("Asia/Shanghai"),
    )
    with pytest.raises(TypeError, match="ForesightNightlyRun"):
        Nightly(
            behavior_tree=site.behavior_tree,
            prediction=prediction,
            scene=scene,
            closed_days=tuple,
            vocabulary=steady,
            foresight=object(),
        )  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="RelationConfig"):
        ForesightNightlyRun(OpportunityStore(tmp_path / "f"), RelationStore(tmp_path / "scene"), object())  # type: ignore[arg-type]
    assert isinstance(ConceptSet([]), ConceptSet)
