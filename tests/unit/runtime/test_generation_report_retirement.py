"""关系行退休的观测（实时化阶段 3，共享说明 3.4）：换代报告每条 lane 一节报带着要读数的行数、退休行总数、这一代退了几条；
指标有按 lane、按种类（带着 / 退休）的行数 gauge。"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path

from habitus.infrastructure.observability import MetricRegistry
from habitus.runtime.generation_report import GenerationReports
from habitus.scene.relations import LaneState, Relation, RelationKey, Segment, Status, Transition
from habitus.scene.relations.longspan import MEASURE_VERSIONS
from habitus.scene.relations.store import RelationStore
from tests.unit.runtime.test_generation import TODAY, wired
from tests.unit.runtime.test_generation_runner import worker_for

FIRST, SECOND = date(2026, 6, 29), date(2026, 6, 30)


def _key(antecedent: str, consequent: str) -> RelationKey:
    return RelationKey("session", antecedent, consequent, Segment.CHAIN, "")


GOING, GONE, HELD, FRESH = (
    _key("s-k0001", "s-k0002"),
    _key("s-k0003", "s-k0002"),
    _key("s-k0001", "s-k0004"),
    _key("s-k0004", "s-k0002"),
)


def _state(night: date, *rows: Relation) -> LaneState:
    return LaneState(lane="session", night=night, family=0, relations=rows, measure_versions=MEASURE_VERSIONS)


def _held() -> Relation:
    return Relation(
        HELD,
        Status.ESTABLISHED,
        True,
        discovered_on=date(2026, 6, 1),
        forward_passed_on=date(2026, 6, 10),
        established_on=date(2026, 6, 10),
        maintained_through=date(2026, 6, 20),
        maintenance_needed=5,
    )


def test_the_report_counts_carried_retired_and_retired_this_generation(tmp_path: Path) -> None:
    """上一代：候选 GOING、早已退休的 GONE、成立 HELD；这一代 GOING 退休、新发现候选 FRESH。
    带着要读数的 = HELD、FRESH 两条；退休行 = GOING、GONE 两条；这一代退了 GOING 一条。"""

    store = RelationStore(tmp_path / "scene")
    store.write(
        _state(
            FIRST,
            Relation(GOING, Status.CANDIDATE, True, discovered_on=date(2026, 6, 20), forward_needed=8),
            Relation(GONE, Status.RETIRED, True, retired_on=date(2026, 6, 15)),
            _held(),
        ),
        (),
    )
    store.write(
        _state(
            SECOND,
            Relation(GOING, Status.RETIRED, True, retired_on=SECOND),
            Relation(GONE, Status.RETIRED, True, retired_on=date(2026, 6, 15)),
            _held(),
            Relation(FRESH, Status.CANDIDATE, True, discovered_on=SECOND, forward_needed=6),
        ),
        (
            Transition(SECOND, GOING, Status.CANDIDATE, Status.RETIRED, "前因久没来"),
            Transition(SECOND, FRESH, Status.TESTED, Status.CANDIDATE, "过了第 1 道"),
        ),
    )
    reports = GenerationReports(tmp_path / "scene" / "generation" / "reports", artifact_root=tmp_path, relations=store)
    section, changes = reports.relation_changes(SECOND)
    lane = section["session"]
    assert isinstance(lane, dict)
    assert (lane["carried"], lane["retired"], lane["retired_tonight"]) == (2, 2, 1)
    # 退休那一条照样作为状态变化报出来（键、前后状态、迁移原因）
    rows = {row["key"]: row for row in lane["changes"]}
    assert (rows[GOING.text]["before"], rows[GOING.text]["after"]) == ("candidate", "retired")
    assert rows[GOING.text]["reason"] == "前因久没来"
    assert any(change.key == GOING and change.after is Status.RETIRED for _lane, change in changes)


def test_a_generation_reports_relation_rows_as_a_gauge_per_lane_and_kind(tmp_path: Path) -> None:
    """一整代走一遍（真实语义树、其余步替身）：这一代写了关系表的每条 lane，都有带着 / 退休两种行数的 gauge，
    与报告那一节的数一致。"""

    run = wired(tmp_path, [])
    registry = MetricRegistry()
    worker = worker_for(run, tmp_path, observer=registry)
    result = asyncio.run(worker.look())
    assert result is not None and result.complete
    assert worker.reports is not None
    report = worker.reports.read(TODAY)
    assert report is not None
    lanes = report["relations"]
    assert isinstance(lanes, dict) and lanes, "这一代至少写了一条 lane 的关系表"
    gauges = registry.snapshot().gauges
    for lane, section in lanes.items():
        assert {"carried", "retired", "retired_tonight"} <= set(section)
        for kind in ("carried", "retired"):
            found = [
                value
                for name, value in gauges.items()
                if name.startswith("scene_relation_rows[") and f"lane={lane}" in name and f"kind={kind}" in name
            ]
            assert found == [section[kind]], (lane, kind, sorted(gauges))
