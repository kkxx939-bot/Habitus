"""投影期间共用的纯引用与统计工具。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from foundation.integrity import canonical_digest
from prediction.projection._behavior_source import BehaviorSourceRecord


def _inferred_ratio(actions: Sequence[Mapping[str, Any]]) -> float:
    if not actions:
        return 0.0
    inferred = sum(action["knowledge_state"] == "inferred" for action in actions)
    return inferred / len(actions)


def _events_inferred_ratio(events: Sequence[BehaviorSourceRecord]) -> float:
    actions = tuple(action for event in events for action in event.fields["actions"])
    return _inferred_ratio(actions)


def _action_subjects(actions: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return _stable_unique(action["actor"] for action in actions)


def _stable_subjects(*groups: Sequence[str]) -> tuple[str, ...]:
    return _stable_unique(subject for group in groups for subject in group)


def _event_subjects(events: Sequence[BehaviorSourceRecord]) -> tuple[str, ...]:
    return _stable_unique(subject for event in events for subject in event.fields["participants"])


def _event_targets(fields: Mapping[str, Any]) -> tuple[str, ...]:
    return _stable_unique(target for action in fields["actions"] for target in action["target_refs"])


def _stable_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _single(values: Sequence[str]) -> str | None:
    return values[0] if len(values) == 1 else None


def _member_ref(uri: str, member_type: str, member_id: str) -> str:
    return f"{uri}#{member_type}:{member_id}"


def _opaque_container_ref(uri: str) -> str:
    return f"behavior-container:{canonical_digest({'uri': uri})}"


def _opaque_step_ref(step_kind: str, uri: str) -> str:
    return f"{step_kind}:{canonical_digest({'uri': uri})}"
