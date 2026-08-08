"""Behavior Projection 持久反序列化必须执行完整字段校验。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from conversation import (
    CONVERSATION_BEHAVIOR_PROJECTOR_VERSION,
    ConversationBehaviorProjectionBatch,
    ConversationBehaviorProjectionItem,
    ConversationBehaviorProjectionKind,
    ConversationBehaviorProjectionStore,
    ConversationSourceError,
)
from foundation.integrity import canonical_digest, canonical_json
from pre.conversation import ConversationMessage, ConversationMessageRole

SOURCE_ID = "a" * 64
SOURCE_DIGEST = "b" * 64
OCCURRED_AT = datetime(2026, 8, 8, 1, 2, 3, tzinfo=timezone.utc)


def _projection_batch() -> ConversationBehaviorProjectionBatch:
    message = ConversationMessage(
        message_id="message-1",
        sequence=0,
        role=ConversationMessageRole.PROMPT,
        occurred_at=OCCURRED_AT,
        content="hello",
    )
    item = ConversationBehaviorProjectionItem.create(
        source_id=SOURCE_ID,
        message=message,
        projection_kind=ConversationBehaviorProjectionKind.USER_CONVERSATION_INPUT,
        payload={"content": message.content},
    )
    return ConversationBehaviorProjectionBatch.create(
        source_id=SOURCE_ID,
        source_digest=SOURCE_DIGEST,
        projector_version=CONVERSATION_BEHAVIOR_PROJECTOR_VERSION,
        items=(item,),
        created_at=OCCURRED_AT,
    )


def _with_source_message_id(
    batch: ConversationBehaviorProjectionBatch,
    source_message_id: Any,
) -> dict[str, Any]:
    value = batch.to_dict()
    item = dict(value["items"][0])
    item["source_message_id"] = source_message_id
    item_identity = {
        "schema_version": item["schema_version"],
        "source_id": item["source_id"],
        "source_message_id": item["source_message_id"],
        "source_message_digest": item["source_message_digest"],
        "occurred_at": item["occurred_at"],
        "projection_kind": item["projection_kind"],
        "payload": item["payload"],
    }
    item["projection_item_id"] = canonical_digest(item_identity)
    value["items"] = [item]

    projection_identity = {
        "schema_version": value["schema_version"],
        "source_id": value["source_id"],
        "source_digest": value["source_digest"],
        "projector_version": value["projector_version"],
        "items": value["items"],
    }
    value["projection_id"] = canonical_digest(projection_identity)
    value["content_digest"] = canonical_digest(
        {**projection_identity, "projection_id": value["projection_id"]}
    )
    return value


def test_projection_item_from_dict_rejects_empty_source_message_id() -> None:
    value = _with_source_message_id(_projection_batch(), "")["items"][0]

    with pytest.raises(ConversationSourceError, match="source_message_id"):
        ConversationBehaviorProjectionItem.from_dict(value, source_id=SOURCE_ID)


def test_projection_store_rejects_canonical_item_with_non_string_source_message_id(
    tmp_path: Path,
) -> None:
    store = ConversationBehaviorProjectionStore(tmp_path, max_file_bytes=100_000)
    value = _with_source_message_id(_projection_batch(), 7)
    projection_id = value["projection_id"]
    projection_path = tmp_path / "projections" / "behavior" / f"{projection_id}.json"
    projection_path.parent.mkdir(parents=True)
    projection_path.write_text(canonical_json(value) + "\n", encoding="utf-8")

    with pytest.raises(ConversationSourceError):
        store.read(projection_id)


def test_projection_store_preserves_valid_roundtrip(tmp_path: Path) -> None:
    store = ConversationBehaviorProjectionStore(tmp_path, max_file_bytes=100_000)
    batch = _projection_batch()

    assert store.put(batch) == batch
    assert store.read(batch.projection_id) == batch
