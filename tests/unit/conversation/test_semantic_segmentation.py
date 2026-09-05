"""Conversation 语义边界评分与确定性降级测试。"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from habitus.memory.conversation import ConversationSemanticBoundaryScorer
from habitus.pre.conversation import ConversationBatch
from tests.helpers import closed_turn


class _Vector:
    def __init__(self, *values: float) -> None:
        self.values = values


class _Embedder:
    def __init__(self, vectors: tuple[_Vector, ...] = (), error: Exception | None = None) -> None:
        self.vectors = vectors
        self.error = error
        self.inputs: tuple[str, ...] = ()

    async def embed_documents(self, texts: Sequence[str]) -> tuple[_Vector, ...]:
        self.inputs = tuple(texts)
        if self.error is not None:
            raise self.error
        return self.vectors


def test_boundary_scorer_uses_adjacent_cosine_distance_and_binds_live_digest() -> None:
    live = ConversationBatch("conversation-1", closed_turn())
    embedder = _Embedder((_Vector(1.0, 0.0), _Vector(0.0, 1.0)))
    scorer = ConversationSemanticBoundaryScorer(
        embedder,
        embedding_fingerprint="embedding-v1",
        max_unit_chars=1_000,
    )

    hints = asyncio.run(scorer.score(live))

    assert hints is not None
    assert hints.source_digest == live.digest
    assert hints.embedding_fingerprint == "embedding-v1"
    assert hints.fallback_reason is None
    assert len(embedder.inputs) == 2
    assert hints.distance_after(0) == 1.0


def test_boundary_scorer_failure_returns_structural_fallback_instead_of_blocking() -> None:
    live = ConversationBatch("conversation-1", closed_turn())
    scorer = ConversationSemanticBoundaryScorer(
        _Embedder(error=RuntimeError("provider unavailable")),
        embedding_fingerprint="embedding-v1",
        max_unit_chars=1_000,
    )

    hints = asyncio.run(scorer.score(live))

    assert hints is not None
    assert hints.boundaries == ()
    assert hints.embedding_fingerprint is None
    assert "provider unavailable" in (hints.fallback_reason or "")
