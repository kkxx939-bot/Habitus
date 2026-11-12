"""关系表只认当前 schema：新字段来回不丢；任何别的版本（包括上一版）一律当不支持拒掉，不迁移（26 号清单第四节第 9 条）。"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from habitus.scene.relations import LaneState, LaneTests, build_timelines, fold
from habitus.scene.relations.document import SCHEMA, RelationRecordError, decode_night, encode_night
from tests.unit.scene.relation_fixtures import CONCEPTS, CONFIG, planted, watching, workdays


def _last_night() -> LaneState:
    series = planted(40, linked=(0, 25))
    previous: LaneState | None = None
    for day in workdays(40)[5:]:
        tonight = series.until(day + timedelta(days=1))
        line = build_timelines(tonight, CONCEPTS, {}, CONFIG)["session"]
        previous, _moved = fold(previous, LaneTests(line, CONCEPTS, CONFIG, tonight.cutoff), watching=watching(tonight))
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


@pytest.fixture(scope="module")
def encoded_night() -> tuple[LaneState, str]:
    state = _last_night()
    return state, encode_night(state)


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("more_needed", "3", "relation test more_needed must be an integer or null"),
        ("more_needed", True, "relation test more_needed must be an integer or null"),
        ("more_needed", 2.5, "relation test more_needed must be an integer or null"),
        ("heterogeneity_p", "0.5", "relation test heterogeneity_p must be a number or null"),
        ("excess_share", False, "relation test excess_share must be a number or null"),
        ("same_event_share", [0.1], "relation test same_event_share must be a number or null"),
        ("leave_one_out", 1, "relation test leave_one_out must be a boolean or null"),
        ("stable", "yes", "relation test stable must be a boolean or null"),
        ("large_enough", 0, "relation test large_enough must be a boolean or null"),
    ],
)
def test_a_night_with_wrong_types_in_a_test_reading_is_rejected_when_decoded(
    encoded_night: tuple[LaneState, str], field: str, bad: object, message: str
) -> None:
    """关系表里检验读数类型错的字段在解码时就拒（W16：类型检查在边界）。"""

    import json

    state, text = encoded_night
    payload = json.loads(text)
    tested = next(item for item in payload["relations"] if "tonight" in item)
    tested["tonight"][field] = bad
    with pytest.raises(RelationRecordError, match=f"relation night is not valid: {message}"):
        decode_night(json.dumps(payload), lane="session", night=state.night)
