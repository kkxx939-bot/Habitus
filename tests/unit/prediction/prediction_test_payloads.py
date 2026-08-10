"""Prediction Tree 测试共享的严格样本工厂。"""

from __future__ import annotations

from datetime import date, datetime, timezone
from hashlib import sha256
from typing import Any

from foundation.integrity import canonical_digest
from prediction.identity import derive_sample_identity
from prediction.model import PredictionKind


def transition_payload(identity_token: str = "fixture-default") -> dict[str, Any]:
    cutoff = datetime(2026, 8, 8, 10, 30, tzinfo=timezone.utc)
    projection_version = "behavior-prediction-v2"
    fixture_suffix = sha256(identity_token.encode("utf-8")).hexdigest()[:16]
    event_uri = (
        f"behavior://behaviors/events/2026/08/08/主人回家-{fixture_suffix}"
        "--20260808T103000000000%2B0000.md"
    )
    target_ref = f"{event_uri}#action:act_0001"
    identity_material = {
        "container_uri": event_uri,
        "anchor_type": "action",
        "prefix_length": 0,
        "target_ref": target_ref,
    }
    identity = derive_sample_identity(PredictionKind.TRANSITION, identity_material, projection_version, {})
    return {
        "sample_date": date(2026, 8, 8),
        "logical_sample_id": identity.logical_sample_id,
        "materialization_id": identity.materialization_id,
        "prediction_scope": {
            "participants": ["主人"],
            "target_level": "action",
            "target_domain": "device_control",
            "prediction_mode": "next_step",
        },
        "anchor": {
            "anchor_type": "action",
            "container_ref": "behavior-container:" + canonical_digest({"uri": event_uri}),
            "prefix_length": 0,
            "previous_step_ref": None,
            "decision_basis": "container_started",
            "cutoff_at": cutoff,
            "precision": "exact",
            "lower_bound_at": None,
            "upper_bound_at": None,
        },
        "input": {
            "observation_frame": {
                "observed_at": cutoff,
                "available_at": cutoff,
                "observer": "camera:living-room",
                "subjects": ["主人"],
                "facts": [
                    {
                        "fact_id": "fact_0001",
                        "category": "environment",
                        "semantics": "室内温度为 29 摄氏度",
                        "subject_ref": "客厅",
                        "attribute": "temperature",
                        "value": 29,
                        "unit": "celsius",
                        "observed_at": cutoff,
                        "available_at": cutoff,
                        "valid_from": cutoff,
                        "valid_to": None,
                        "knowledge_state": "observed",
                        "confidence": 0.98,
                        "evidence_refs": ["sensor:temperature"],
                    }
                ],
                "active_goals": ["完成回家后的生活安排"],
                "constraints": [],
                "coverage": {
                    "available_modalities": ["vision", "temperature"],
                    "missing_modalities": ["audio"],
                    "blind_intervals": [],
                    "coverage_score": 0.9,
                },
            },
            "behavior_history": empty_history(),
            "decision_space": {
                "known_available": [],
                "known_unavailable": [],
                "prohibited": [],
                "unknown": ["counterfactual_action_space"],
            },
        },
        "label": {
            "target_kind": "action",
            "source_ref": target_ref,
            "actor": "主人",
            "behavior_type": "device_control",
            "semantics": "打开空调",
            "target_refs": ["空调"],
            "parameters": {"mode": "cool"},
            "started_at": None,
            "delay_seconds": None,
            "relations": [],
            "terminal": None,
        },
        "supervision": {
            "label_status": "observed",
            "window_started_at": cutoff,
            "window_closed_at": datetime(2026, 8, 8, 10, 32, tzinfo=timezone.utc),
            "censored": False,
            "censoring_reason": None,
        },
        "lineage": {
            "behavior_root_uri": event_uri,
            "event_uri": event_uri,
            "episode_uri": None,
            "outcome_uri": None,
            "occurrence_group_id": event_uri,
            "consequence_group_id": None,
        },
        "identity_material": identity_material,
        "materialization_context": {},
        "provenance": {
            "source_bindings": [
                {
                    "uri": event_uri,
                    "revision": 1,
                    "digest": "b" * 64,
                    "member_type": "action",
                    "member_id": "act_0001",
                }
            ],
            "projection_version": projection_version,
            "projector_digest": "c" * 64,
        },
        "quality": {
            "source_confidence": 0.96,
            "evidence_coverage": 0.9,
            "context_completeness": 0.8,
            "conflict_count": 0,
            "inferred_fact_ratio": 0.0,
        },
    }


def empty_history() -> dict[str, list[Any]]:
    return {
        "completed_events": [],
        "completed_actions": [],
        "completed_phases": [],
        "active_behaviors": [],
        "parallel_behaviors": [],
        "interruptions": [],
        "resumptions": [],
    }


__all__ = ["empty_history", "transition_payload"]
