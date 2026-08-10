"""样本的监督窗口、系谱、来源与质量元数据。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from prediction.projection._contract import PROJECTION_VERSION, PROJECTOR_DIGEST


def _supervision(
    *,
    label_status: str,
    started_at: datetime | None,
    closed_at: datetime | None,
    censoring_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "label_status": label_status,
        "window_started_at": started_at,
        "window_closed_at": closed_at,
        "censored": censoring_reason is not None,
        "censoring_reason": censoring_reason,
    }


def _lineage(
    *,
    root_uri: str,
    event_uri: str | None = None,
    episode_uri: str | None = None,
    outcome_uri: str | None = None,
    occurrence_group_id: str,
    consequence_group_id: str | None = None,
) -> dict[str, Any]:
    return {
        "behavior_root_uri": root_uri,
        "event_uri": event_uri,
        "episode_uri": episode_uri,
        "outcome_uri": outcome_uri,
        "occurrence_group_id": occurrence_group_id,
        "consequence_group_id": consequence_group_id,
    }


def _provenance(bindings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "source_bindings": tuple(bindings),
        "projection_version": PROJECTION_VERSION,
        "projector_digest": PROJECTOR_DIGEST,
    }


def _quality(
    *,
    confidence: float,
    conflicts: int,
    inferred_ratio: float,
    evidence_coverage: float | None = None,
) -> dict[str, Any]:
    return {
        "source_confidence": confidence,
        "evidence_coverage": evidence_coverage,
        "context_completeness": None,
        "conflict_count": conflicts,
        "inferred_fact_ratio": inferred_ratio,
    }
