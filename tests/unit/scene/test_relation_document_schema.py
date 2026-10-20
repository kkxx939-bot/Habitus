"""关系表只认当前 schema：新字段来回不丢；任何别的版本（包括上一版）一律当不支持拒掉，不迁移（26 号清单第四节第 9 条）。"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from habitus.scene.relations import LaneState, LaneTests, build_timelines, fold
from habitus.scene.relations.document import SCHEMA, RelationRecordError, decode_night, encode_night
from tests.unit.scene.relation_fixtures import CONCEPTS, CONFIG, planted, workdays


def _last_night() -> LaneState:
    series = planted(40, linked=(0, 25))
    previous: LaneState | None = None
    for day in workdays(40)[5:]:
        tonight = series.until(day + timedelta(days=1))
        line = build_timelines(tonight, CONCEPTS, {}, CONFIG)["session"]
        previous, _moved = fold(previous, LaneTests(line, CONCEPTS, CONFIG, tonight.cutoff))
    assert previous is not None
    return previous


def test_new_fields_round_trip_in_the_current_schema() -> None:
    state = _last_night()
    carried = next(relation for relation in state.relations if relation.status.carried)
    updated = replace(
        state,
        relations=tuple(replace(r, maintenance_needed=6) if r is carried else r for r in state.relations),
        measure_versions={"return": 2},
    )
    decoded = decode_night(encode_night(updated), lane="session", night=state.night)
    assert decoded == updated
    assert decoded.measure_versions == {"return": 2}
    assert next(r for r in decoded.relations if r.key == carried.key).maintenance_needed == 6


@pytest.mark.parametrize("schema", ["habitus.scene.relations.night/v3", "habitus.scene.relations.night/v2"])
def test_any_other_schema_is_rejected(schema: str) -> None:
    text = encode_night(_last_night()).replace(SCHEMA, schema)
    with pytest.raises(RelationRecordError, match="unsupported schema"):
        decode_night(text, lane="session", night=_last_night().night)
