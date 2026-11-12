"""退休行的落盘形状（实时化阶段 3，共享说明 3.4）：退休行只写键、状态、方向与退休那一代；其余状态不写 ``retired_on``，
现有规范形一个字节不变；不升 ``SCHEMA``。"""

from __future__ import annotations

import hashlib
import json
from datetime import date

import pytest

from habitus.foundation.integrity import canonical_json
from habitus.scene.relations import LaneState, Relation, RelationKey, Segment, Status, Subset, Transition
from habitus.scene.relations.document import (
    SCHEMA,
    RelationRecordError,
    decode_night,
    decode_transition,
    encode_night,
    encode_transition,
)
from habitus.scene.relations.stats import Effect

NIGHT = date(2026, 6, 30)


def _key(antecedent: str, consequent: str, segment: Segment = Segment.CHAIN) -> RelationKey:
    return RelationKey("session", antecedent, consequent, segment, "")


#: 只用二进制下精确的小数，规范形跨平台一致。
_SUBSET = Subset(
    antecedents=4,
    blocks=3,
    effect=Effect(antecedents=4, treated_rate=0.75, control_rate=0.25),
    p_value=0.125,
    interval=(0.25, 0.5),
    relative_interval=None,
    minimum_p=0.0625,
)


def _existing_rows() -> tuple[Relation, ...]:
    """退休之前就有的四种带着的行。"""

    return (
        Relation(
            _key("s-k0001", "s-k0002"),
            Status.CANDIDATE,
            True,
            discovered_on=date(2026, 6, 20),
            forward=_SUBSET,
            forward_needed=6,
        ),
        Relation(
            _key("s-k0001", "s-k0003", Segment.REST_OF_DAY),
            Status.ESTABLISHED,
            False,
            discovered_on=date(2026, 6, 1),
            forward_passed_on=date(2026, 6, 10),
            established_on=date(2026, 6, 10),
            maintenance=_SUBSET,
            maintained_through=date(2026, 6, 25),
            misses=1,
            maintenance_needed=5,
            forward_needed=4,
        ),
        Relation(
            _key("s-k0002", "s-k0003"),
            Status.EXPIRED,
            True,
            discovered_on=date(2026, 5, 1),
            established_on=date(2026, 5, 10),
            expired_on=date(2026, 6, 28),
            maintained_through=date(2026, 6, 28),
            misses=2,
            maintenance_needed=5,
        ),
        Relation(
            _key("s-k0003", "s-k0001"),
            Status.REJECTED,
            True,
            discovered_on=date(2026, 6, 5),
            rejected_on=date(2026, 6, 15),
            prior_only=True,
            forward_needed=3,
        ),
    )


def _state(*rows: Relation) -> LaneState:
    return LaneState(
        lane="session",
        night=NIGHT,
        family=7,
        relations=rows,
        sparse={"chain": 3},
        vocabulary=2,
        measure_versions={"return": 2},
    )


def _retired() -> Relation:
    return Relation(_key("s-k0004", "s-k0001"), Status.RETIRED, True, retired_on=date(2026, 6, 29))


def _relations_of(text: str) -> list[dict[str, object]]:
    return json.loads(text)["relations"]


def test_rows_that_existed_before_retirement_keep_their_canonical_bytes() -> None:
    """c55f7da 上这份状态的规范形的 sha256：加了退休之后，没有退休行的表一个字节不变、仍是同一个 SCHEMA。"""

    text = encode_night(_state(*_existing_rows()))
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == (
        "4444a8cb896e12adc637c45b1b19a33882fdbeaa1e51db33a4b9b3a3432ee9c0"
    )
    assert SCHEMA == "habitus.scene.relations.night/v4"
    assert all("retired_on" not in item for item in _relations_of(text))


def test_a_retired_row_round_trips_and_stores_only_its_key_status_direction_and_night() -> None:
    state = _state(*_existing_rows(), _retired())
    text = encode_night(state)
    assert decode_night(text, lane="session", night=NIGHT) == state
    rows = _relations_of(text)
    (retired,) = [item for item in rows if item["status"] == Status.RETIRED.value]
    assert set(retired) == {"key", "status", "upward", "retired_on"}
    assert retired["retired_on"] == "2026-06-29"
    assert all("retired_on" not in item for item in rows if item["status"] != Status.RETIRED.value)


def _rewritten(state: LaneState, change: dict[str, object], *, status: str) -> str:
    payload = json.loads(encode_night(state))
    for item in payload["relations"]:
        if item["status"] == status:
            item.update(change)
            for name in [name for name, value in change.items() if value is None]:
                del item[name]
    return canonical_json(payload)


@pytest.mark.parametrize(
    "extra",
    [
        {"discovered_on": "2026-06-01"},
        {"tonight": {"verdict": "tested"}},
        {"forward": None, "maintenance": None, "misses": 0},
    ],
    ids=["date", "reading", "carried-fields"],
)
def test_a_retired_row_with_any_other_field_is_refused(extra: dict[str, object]) -> None:
    state = _state(_retired())
    payload = json.loads(encode_night(state))
    payload["relations"][0].update(extra)
    with pytest.raises(RelationRecordError):
        decode_night(canonical_json(payload), lane="session", night=NIGHT)


def test_retired_on_belongs_to_retired_rows_only() -> None:
    """退休行没有退休那一代、别的行带着退休那一代：都是坏记录。"""

    without = _rewritten(_state(_retired()), {"retired_on": None}, status=Status.RETIRED.value)
    with pytest.raises(RelationRecordError):
        decode_night(without, lane="session", night=NIGHT)
    stray = _rewritten(_state(*_existing_rows()), {"retired_on": "2026-06-29"}, status=Status.CANDIDATE.value)
    with pytest.raises(RelationRecordError):
        decode_night(stray, lane="session", night=NIGHT)


def test_a_retirement_transition_round_trips() -> None:
    item = Transition(
        night=NIGHT, key=_key("s-k0004", "s-k0001"), before=Status.CANDIDATE, after=Status.RETIRED, reason="前因久没来"
    )
    assert decode_transition(encode_transition(item), lane="session") == item
