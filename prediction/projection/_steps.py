"""把 Behavior 成员投影为模型可见的行为步骤。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from prediction.projection._behavior_source import BehaviorSourceRecord
from prediction.projection._refs import (
    _event_targets,
    _opaque_step_ref,
    _single,
)


def _projected_action(
    event_uri: str,
    action: Mapping[str, Any],
    *,
    expose_source: bool = True,
) -> dict[str, Any]:
    return {
        "step_kind": "action",
        "step_ref": f"action:{action['action_id']}",
        "source_uri": event_uri if expose_source else None,
        "local_id": action["action_id"],
        "sequence": action["sequence"],
        "semantics": action["semantics"],
        "actor": action["actor"],
        "behavior_type": action["action_type"],
        "target_refs": action["target_refs"],
        "status": action["status"],
        "started_at": action["started_at"],
        "ended_at": action["ended_at"],
        "available_at": action["available_at"],
    }


def _active_projected_action(
    event_uri: str,
    action: Mapping[str, Any],
    *,
    expose_source: bool = True,
) -> dict[str, Any]:
    """投影 Action 开始边界可见的身份，不复制结束状态。"""

    return {
        "step_kind": "action",
        "step_ref": f"action:{action['action_id']}",
        "source_uri": event_uri if expose_source else None,
        "local_id": action["action_id"],
        "sequence": action["sequence"],
        "semantics": action["semantics"],
        "actor": action["actor"],
        "behavior_type": action["action_type"],
        "target_refs": action["target_refs"],
        "status": "active",
        "started_at": action["started_at"],
        "ended_at": None,
        "available_at": action["available_at"],
    }


def _projected_event(event: BehaviorSourceRecord, *, expose_source: bool = True) -> dict[str, Any]:
    fields = event.fields
    return {
        "step_kind": "event",
        "step_ref": _opaque_step_ref("event", event.uri),
        "source_uri": event.uri if expose_source else None,
        "local_id": None,
        "sequence": 1,
        "semantics": fields["semantic_summary"],
        "actor": _single(fields["participants"]),
        "behavior_type": "event",
        "target_refs": _event_targets(fields),
        "status": fields["status"],
        "started_at": fields["started_at"],
        "ended_at": fields["ended_at"],
        "available_at": fields["ended_at"] or fields["onset_available_at"],
    }


def _active_projected_event(event: BehaviorSourceRecord, *, expose_source: bool = True) -> dict[str, Any]:
    """Event 开始时不暴露最终摘要、最终状态或未来 Action 目标。"""

    fields = event.fields
    return {
        "step_kind": "event",
        "step_ref": _opaque_step_ref("event", event.uri),
        "source_uri": event.uri if expose_source else None,
        "local_id": None,
        "sequence": 1,
        "semantics": fields["onset_semantics"],
        "actor": _single(fields["participants"]),
        "behavior_type": "event",
        "target_refs": (),
        "status": "active",
        "started_at": fields["started_at"],
        "ended_at": None,
        "available_at": fields["onset_available_at"],
    }


def _projected_phase(
    episode_uri: str,
    phase: Mapping[str, Any],
    *,
    expose_source: bool = True,
) -> dict[str, Any]:
    return {
        "step_kind": "phase",
        "step_ref": f"phase:{phase['phase_id']}",
        "source_uri": episode_uri if expose_source else None,
        "local_id": phase["phase_id"],
        "sequence": phase["sequence"],
        "semantics": phase["semantics"],
        "actor": None,
        "behavior_type": "phase",
        "target_refs": phase["event_uris"],
        "status": phase["status"],
        "started_at": phase["started_at"],
        "ended_at": phase["ended_at"],
        "available_at": phase["ended_at"] or phase["started_at"],
    }


def _active_projected_phase(
    episode_uri: str,
    phase: Mapping[str, Any],
    *,
    expose_source: bool = True,
) -> dict[str, Any]:
    """Phase 开始时只暴露阶段存在，不泄露其最终事件集合。"""

    return {
        "step_kind": "phase",
        "step_ref": f"phase:{phase['phase_id']}",
        "source_uri": episode_uri if expose_source else None,
        "local_id": phase["phase_id"],
        "sequence": phase["sequence"],
        "semantics": "阶段进行中",
        "actor": None,
        "behavior_type": "phase",
        "target_refs": (),
        "status": "active",
        "started_at": phase["started_at"],
        "ended_at": None,
        "available_at": phase["started_at"],
    }

