"""行为侧会话 lane 留在会话源这边的回执：形状自洽、写入不可变、损坏能认出来。"""

from __future__ import annotations

import json

import pytest

from habitus.conversation.behavior_session import (
    BEHAVIOR_SESSION_OUTPUT_KIND,
    BehaviorSessionOutput,
    BehaviorSessionTurnRecord,
)
from habitus.conversation.source import ConversationSourceError
from habitus.foundation.integrity import canonical_digest
from tests.platform_helpers import long_path
from tests.unit.conversation.source_v2_helpers import NOW, source, stores

FINGERPRINT = canonical_digest("behavior-session-processor")


def turn(start: int = 0, end: int = 1, *, recorded: bool = True) -> BehaviorSessionTurnRecord:
    return BehaviorSessionTurnRecord(
        start_sequence=start, end_sequence=end, instructed_at=NOW, completed_at=NOW, recorded=recorded
    )


def receipt(source_value, *turns: BehaviorSessionTurnRecord) -> BehaviorSessionOutput:  # type: ignore[no-untyped-def]
    return BehaviorSessionOutput.create(
        source=source_value, processor_fingerprint=FINGERPRINT, turns=turns or (turn(),)
    )


def test_a_receipt_round_trips_and_its_identity_follows_from_the_source(tmp_path) -> None:
    *_others, store = stores(tmp_path)
    source_value = source()
    output = receipt(source_value, turn(0, 1), turn(2, 5, recorded=False))

    stored = store.put(source_value, output)

    assert stored == output
    assert store.read(source_value, output.output_id) == output
    assert store.list(source_value) == (output,)
    assert output.output_id == store.expected_output_id(source_value, FINGERPRINT)
    reference = store.ref(stored)
    assert (reference.output_kind, reference.output_id) == (BEHAVIOR_SESSION_OUTPUT_KIND, output.output_id)
    assert store.restore(stored) is stored
    assert BehaviorSessionOutput.from_dict(json.loads(json.dumps(output.to_dict()))) == output


def test_writing_the_same_receipt_again_is_a_no_op_but_different_content_is_refused(tmp_path) -> None:
    *_others, store = stores(tmp_path)
    source_value = source()
    output = receipt(source_value, turn(0, 1))
    store.put(source_value, output)

    assert store.put(source_value, output) == output
    with pytest.raises(ConversationSourceError, match="conflicts with different content"):
        store.put(source_value, receipt(source_value, turn(0, 1, recorded=False)))


def test_a_receipt_cannot_be_filed_under_another_source(tmp_path) -> None:
    *_others, store = stores(tmp_path)
    mine, other = source(), source(delivery_seed="delivery-b")

    with pytest.raises(ConversationSourceError, match="belongs to another source"):
        store.put(other, receipt(mine))


def test_a_tampered_receipt_is_reported_as_corrupt(tmp_path) -> None:
    *_others, store = stores(tmp_path)
    source_value = source()
    output = store.put(source_value, receipt(source_value))
    path = long_path(store._path(source_value.source_id, output.output_id))  # noqa: SLF001 - 直接改盘上的字节
    path.write_bytes(path.read_bytes().replace(b'"recorded":true', b'"recorded":false'))

    with pytest.raises(ConversationSourceError, match="corrupt"):
        store.read(source_value, output.output_id)


def test_removing_a_receipt_empties_the_listing(tmp_path) -> None:
    *_others, store = stores(tmp_path)
    source_value = source()
    output = store.put(source_value, receipt(source_value))

    assert store.remove(source_value, output.output_id)
    assert store.list(source_value) == ()
    assert store.read(source_value, output.output_id) is None


@pytest.mark.parametrize(
    "broken",
    [
        lambda: turn(3, 1),  # 序号倒了
    ],
)
def test_malformed_turn_records_are_rejected(broken) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ConversationSourceError):
        broken()


def test_a_receipt_needs_at_least_one_turn_in_order(tmp_path) -> None:
    source_value = source()
    with pytest.raises(ConversationSourceError, match="at least one turn"):
        BehaviorSessionOutput.create(source=source_value, processor_fingerprint=FINGERPRINT, turns=())
    with pytest.raises(ConversationSourceError, match="ordered and distinct"):
        receipt(source_value, turn(2, 3), turn(0, 1))


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("output_id", 1, "behavior session output_id must be lowercase SHA-256 text"),
        ("source_id", None, "behavior session source_id must be lowercase SHA-256 text"),
        ("source_payload_digest", 1, "behavior session source_payload_digest must be lowercase SHA-256 text"),
        ("processor_fingerprint", 1, "behavior session processor_fingerprint must be lowercase SHA-256 text"),
        ("output_record_digest", 1, "behavior session output_record_digest must be lowercase SHA-256 text"),
        ("start_sequence", "0", "behavior session turn start_sequence must be a non-negative integer"),
        ("end_sequence", True, "behavior session turn end_sequence must be a non-negative integer"),
        ("instructed_at", 1, "instructed_at must be a datetime"),
        ("completed_at", None, "completed_at must be a datetime"),
        ("recorded", "yes", "behavior session turn recorded must be boolean"),
    ],
)
def test_a_receipt_with_wrong_field_types_is_corrupt_on_read(tmp_path, field: str, bad: object, message: str) -> None:
    """盘上回执（含每一轮）字段类型错：读回时在解码器就被拒，Store 报 corrupt（W16：类型检查在边界）。"""

    *_others, store = stores(tmp_path)
    source_value = source()
    output = store.put(source_value, receipt(source_value))
    path = long_path(store._path(source_value.source_id, output.output_id))  # noqa: SLF001 - 直接改盘上的字节
    value = json.loads(path.read_text(encoding="utf-8"))
    (value["turns"][0] if field in value["turns"][0] else value)[field] = bad
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    with pytest.raises(ConversationSourceError, match="corrupt") as caught:
        store.read(source_value, output.output_id)
    assert isinstance(caught.value.__cause__, ConversationSourceError)
    assert message in str(caught.value.__cause__)
