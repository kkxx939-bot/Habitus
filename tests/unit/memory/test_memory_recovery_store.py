from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memory.compaction import MemoryRecoveryError, MemoryRecoveryStore
from memory.model import MemoryKind
from memory.tree import MemoryTree
from memory.uri import MemoryURI
from tests.helpers import document


def test_recovery_store_round_trips_immutable_detailed_document_and_deletes_terminally(tmp_path) -> None:
    tree = MemoryTree(tmp_path / "memory")
    item = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答结构", "content": "- 先给结论\n- 保留精确路径和日期"},
    )
    store = MemoryRecoveryStore(tree)
    saved_at = datetime(2026, 8, 3, 8, 0, tzinfo=timezone.utc)

    first = store.save(item, saved_at=saved_at)
    replay = store.save(item, saved_at=saved_at.replace(hour=9))
    assert replay == first
    assert replay.saved_at == saved_at
    assert store.restore(first.uri, created_at=item.metadata.created_at) == item
    assert store.delete(first.uri) == 1
    assert store.delete(first.uri) == 0
    with pytest.raises(MemoryRecoveryError, match="does not exist"):
        store.restore(first.uri, created_at=item.metadata.created_at)


def test_recovery_store_selects_latest_revision_within_one_content_generation(tmp_path) -> None:
    tree = MemoryTree(tmp_path / "memory")
    first = document(MemoryKind.PREFERENCE, revision=1)
    second = document(
        MemoryKind.PREFERENCE,
        revision=2,
        timestamp=first.metadata.updated_at,
    )
    uri = MemoryURI.from_address(first.address)
    assert first.metadata.created_at == second.metadata.created_at
    store = MemoryRecoveryStore(tree)
    saved_at = datetime(2026, 8, 3, 8, 0, tzinfo=timezone.utc)
    store.save(first, saved_at=saved_at)
    store.save(second, saved_at=saved_at)

    assert store.latest(uri, created_at=first.metadata.created_at).source_revision == 2
    assert store.restore(uri, created_at=first.metadata.created_at) == second
