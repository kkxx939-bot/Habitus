from __future__ import annotations

import json

import pytest

from conversation.projection import ConversationBehaviorProjectionStore, ConversationBehaviorProjector
from conversation.source import ConversationSourceError
from tests.unit.conversation.source_v2_helpers import NOW, source


def _store(tmp_path, *, max_bytes: int = 1_000_000, max_items: int = 100):
    return ConversationBehaviorProjectionStore(
        tmp_path,
        max_files_per_source=4,
        max_file_bytes=max_bytes,
        max_items=max_items,
    )


def test_projection_output_uses_direct_deterministic_output_path_and_round_trips(tmp_path) -> None:
    source_value = source()
    projected = ConversationBehaviorProjector(clock=lambda: NOW).project(source_value)
    assert projected is not None
    store = _store(tmp_path)
    stored = store.put(source_value, projected)
    assert stored == projected
    assert store.read(source_value, projected.output_id) == projected
    assert store.list(source_value) == (projected,)
    assert (
        tmp_path
        / "source"
        / "outputs"
        / source_value.source_id
        / "behavior_projection"
        / f"{projected.output_id}.json"
    ).is_file()


def test_projection_read_rejects_record_digest_tampering(tmp_path) -> None:
    source_value = source()
    projected = ConversationBehaviorProjector(clock=lambda: NOW).project(source_value)
    assert projected is not None
    store = _store(tmp_path)
    store.put(source_value, projected)
    path = (
        tmp_path
        / "source"
        / "outputs"
        / source_value.source_id
        / "behavior_projection"
        / f"{projected.output_id}.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    value["items"][0]["payload"]["content"] = "tampered"
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(ConversationSourceError, match="corrupt"):
        store.read(source_value, projected.output_id)


def test_projection_uses_its_own_byte_and_item_bounds(tmp_path) -> None:
    source_value = source(content="x" * 1_000)
    projected = ConversationBehaviorProjector(clock=lambda: NOW).project(source_value)
    assert projected is not None
    with pytest.raises(ConversationSourceError, match="configured file bound"):
        _store(tmp_path, max_bytes=100).put(source_value, projected)
    with pytest.raises(ValueError, match="max_items"):
        _store(tmp_path, max_items=0)
