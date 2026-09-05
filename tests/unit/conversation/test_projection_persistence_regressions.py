from __future__ import annotations

import json

import pytest

from habitus.conversation.projection import ConversationBehaviorProjectionStore, ConversationBehaviorProjector
from habitus.conversation.source import ConversationSourceError
from tests.unit.conversation.source_v2_helpers import source


def _store(tmp_path, *, max_bytes: int = 1_000_000, max_items: int = 100):
    return ConversationBehaviorProjectionStore(
        tmp_path,
        max_files_per_source=4,
        max_file_bytes=max_bytes,
        max_items=max_items,
    )


def test_projection_output_uses_direct_deterministic_output_path_and_round_trips(tmp_path) -> None:
    source_value = source()
    projected = ConversationBehaviorProjector().project(source_value)
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
    projected = ConversationBehaviorProjector().project(source_value)
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
    projected = ConversationBehaviorProjector().project(source_value)
    assert projected is not None
    with pytest.raises(ConversationSourceError, match="configured file bound"):
        _store(tmp_path, max_bytes=100).put(source_value, projected)
    with pytest.raises(ValueError, match="max_items"):
        _store(tmp_path, max_items=0)


def test_projection_is_deterministic_so_concurrent_writers_reuse_one_output(tmp_path) -> None:
    """投影必须是来源的纯函数：两个独立执行者产出逐字节相同的批次。

    若 recorded_at 取墙上时钟，两者的 output_id 相同而 output_record_digest 不同，
    后写入者会被误报为内容冲突，且重算比对必须先信任被验证文件里的时间字段。
    """

    source_value = source()
    first = ConversationBehaviorProjector().project(source_value)
    second = ConversationBehaviorProjector().project(source_value)
    assert first is not None and second is not None
    assert first == second
    assert first.recorded_at == source_value.recorded_at

    store = _store(tmp_path)
    stored = store.put(source_value, first)
    assert store.put(source_value, second) == stored
    assert store.list(source_value) == (stored,)
    # 无需从落盘文件回填任何字段即可重算并比对。
    assert ConversationBehaviorProjector().project(source_value) == stored
