"""长期记忆只有四类（裁定 109）：事件与"打算做的事"在各个边界上都被明确拒绝，旧盘面启动即报错。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from habitus.integrations.http_api.schemas import RecallRequest
from habitus.memory.document.codec import MemoryDocumentIntegrityError
from habitus.memory.editor.candidate import MemoryCandidateBatch, MemoryCandidateError
from habitus.memory.model import MemoryAddress, MemoryKind
from habitus.memory.retrieval.model import MemorySearchServiceConfig
from habitus.memory.retrieval.request import MemorySearchRequestNormalizer
from habitus.memory.tree import MemoryTree, MemoryTreeIntegrityError
from tests.helpers import codec, document

_EMPTY_BATCH: dict[str, object] = {
    "profile": [],
    "preferences": [],
    "entities": [],
    "tools": [],
    "identity_proposals": [],
    "relations": [],
}


def test_memory_kinds_are_exactly_the_four_confirmed_kinds() -> None:
    assert {kind.value for kind in MemoryKind} == {"profile", "preference", "entity", "tool"}


@pytest.mark.parametrize(
    ("field", "item"),
    [
        ("events", {"page_id": 100, "event_date": "2026-11-20", "event_name": "签合同", "summary": "签了合同"}),
        ("intentions", {"page_id": 100, "intent_name": "下周交报告", "status": "open", "confirmed": True}),
    ],
)
def test_candidate_batch_with_a_removed_kind_array_is_rejected(field: str, item: dict[str, object]) -> None:
    with pytest.raises(MemoryCandidateError, match="unknown fields"):
        MemoryCandidateBatch.model_validate({**_EMPTY_BATCH, field: [item]})


def test_candidate_batch_json_schema_lists_only_the_four_kinds() -> None:
    properties = MemoryCandidateBatch.model_json_schema()["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == set(_EMPTY_BATCH)


def test_recall_request_rejects_intention_scope() -> None:
    with pytest.raises(ValidationError):
        RecallRequest.model_validate({"query": "最近在做什么", "intention_scope": "active"})


@pytest.mark.parametrize("kind", ["event", "intention"])
def test_recall_request_rejects_a_removed_kind(kind: str) -> None:
    with pytest.raises(ValidationError):
        RecallRequest.model_validate({"query": "最近在做什么", "kinds": [kind]})


@pytest.mark.parametrize("kind", ["event", "intention"])
def test_search_request_rejects_a_removed_kind(kind: str) -> None:
    normalizer = MemorySearchRequestNormalizer(MemorySearchServiceConfig())
    with pytest.raises(ValueError):
        normalizer.normalize("最近在做什么", target_uris=None, limit=None, score_threshold=None, kinds=(kind,))  # type: ignore[arg-type]


@pytest.mark.parametrize("old_directory", ["events", "intentions"])
def test_initialize_rejects_an_old_memory_root_at_startup(tmp_path: Path, old_directory: str) -> None:
    root = tmp_path / "memory"
    (root / old_directory).mkdir(parents=True)
    with pytest.raises(MemoryTreeIntegrityError, match=old_directory):
        MemoryTree(root).initialize()


def test_initialize_accepts_the_current_tree_and_creates_only_its_directories(tmp_path: Path) -> None:
    root = MemoryTree(tmp_path / "memory").initialize()
    assert sorted(path.name for path in root.iterdir()) == ["entities", "preferences", "tools"]
    assert MemoryTree(root).initialize() == root


def test_a_document_carrying_last_confirmed_at_is_rejected() -> None:
    encoded = codec().encode(document())
    marker = '\n  "fields":'
    assert marker in encoded
    old = encoded.replace(marker, '\n  "last_confirmed_at": null,' + marker, 1)
    with pytest.raises(MemoryDocumentIntegrityError, match="invalid shape"):
        codec().decode(old, expected_address=MemoryAddress.preference("回答风格"))


def test_encoded_metadata_has_no_confirmation_key() -> None:
    encoded = codec().encode(document())
    metadata = json.loads(encoded.split("<!-- HABITUS_MEMORY_FIELDS\n", 1)[1].rsplit("\n-->", 1)[0])
    assert "last_confirmed_at" not in metadata
