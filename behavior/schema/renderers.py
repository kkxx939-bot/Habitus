"""把已经校验完成的领域字段确定性渲染为人类可读的 L2 正文。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from behavior.model import BehaviorKind


def render_markdown(kind: BehaviorKind, payload: Mapping[str, Any]) -> str:
    """同一份字段永远渲染出同一段正文；正文是字段的确定性函数。"""

    if kind is BehaviorKind.EVENT:
        return _render_event(payload)
    if kind is BehaviorKind.OUTCOME:
        return _render_outcome(payload)
    return _render_episode(payload)


def _render_event(payload: Mapping[str, Any]) -> str:
    lines = [
        f"# {payload['event_name']}",
        "",
        f"**Date:** {payload['event_date'].isoformat()}",
        f"**Time:** {_time_range(payload['started_at'], payload['ended_at'])}",
        f"**Onset available:** {payload['onset_available_at'].isoformat()}",
        f"**Status:** {payload['status']}",
        "",
        "## Onset",
        str(payload["onset_semantics"]),
        "",
        "## Summary",
        str(payload["semantic_summary"]),
    ]
    _append_bullets(lines, "Facts", payload["semantic_facts"])
    _append_optional(lines, "Trigger", payload["trigger"])
    _append_optional(lines, "Goal", payload["goal"])
    _append_bullets(lines, "Constraints", payload["constraints"])
    _append_bullets(lines, "Participants", payload["participants"])
    lines.extend(["", "## Actions"])
    for action in payload["actions"]:
        lines.append(
            f"{action['sequence']}. **{action['actor']} · {action['action_type']}** — "
            f"{action['semantics']} ({action['status']})"
        )
    _append_optional(lines, "Closure", payload["closure_reason"])
    _append_bullets(lines, "Conflicts", payload["conflicts"])
    return "\n".join(lines).rstrip() + "\n"


def _render_outcome(payload: Mapping[str, Any]) -> str:
    lines = [
        f"# {payload['event_name']} — Outcomes",
        "",
        f"**Event:** {payload['event_uri']}",
        "",
        "## Results",
    ]
    for outcome in payload["outcomes"]:
        target = outcome["target_type"]
        if outcome["target_action_id"] is not None:
            target = f"{target}:{outcome['target_action_id']}"
        lines.append(
            f"- **{outcome['occurred_at'].isoformat()} · {outcome['outcome_type']} · {target}** — "
            f"{outcome['semantics']} ({outcome['valence']})"
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_episode(payload: Mapping[str, Any]) -> str:
    lines = [
        f"# {payload['episode_name']}",
        "",
        f"**Time:** {_time_range(payload['started_at'], payload['ended_at'])}",
        f"**Status:** {payload['status']}",
        "",
        "## Summary",
        str(payload["semantic_summary"]),
    ]
    _append_optional(lines, "Goal", payload["goal"])
    _append_bullets(lines, "Participants", payload["participants"])
    lines.extend(["", "## Event Timeline"])
    lines.extend(f"{index}. {uri}" for index, uri in enumerate(payload["ordered_event_uris"], start=1))
    if payload["phases"]:
        lines.extend(["", "## Phases"])
        for phase in payload["phases"]:
            lines.append(
                f"{phase['sequence']}. **{phase['semantics']}** "
                f"({phase['status']}, {_time_range(phase['started_at'], phase['ended_at'])}, "
                f"confidence={phase['confidence']:.3f})"
            )
            lines.extend(f"   - {uri}" for uri in phase["event_uris"])
    if payload["unphased_events"]:
        lines.extend(["", "## Unphased Events"])
        for event in payload["unphased_events"]:
            lines.append(f"- {event['event_uri']} [{event['role']}] — {event['reason']}")
    if payload["transitions"]:
        lines.extend(["", "## Transitions"])
        for transition in payload["transitions"]:
            lines.append(f"- {transition['from_event_uri']} --{transition['relation']}--> {transition['to_event_uri']}")
    _append_bullets(lines, "Key Turning Points", payload["key_turning_points"])
    _append_bullets(lines, "Outcomes", payload["outcome_uris"])
    _append_optional(lines, "Closure", payload["closure_reason"])
    return "\n".join(lines).rstrip() + "\n"


def _append_optional(lines: list[str], title: str, value: Any) -> None:
    if value is not None:
        lines.extend(["", f"## {title}", str(value)])


def _append_bullets(lines: list[str], title: str, values: Sequence[Any]) -> None:
    if values:
        lines.extend(["", f"## {title}"])
        lines.extend(f"- {value}" for value in values)


def _time_range(started_at: datetime, ended_at: datetime | None) -> str:
    end = "unknown" if ended_at is None else ended_at.isoformat()
    return f"{started_at.isoformat()} — {end}"


__all__ = ["render_markdown"]
