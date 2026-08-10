"""按预测切点把 Behavior 成员划分为已完成与进行中。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from prediction.projection._anchors import (
    _anchor_observed_at,
)
from prediction.projection._behavior_source import BehaviorSourceRecord
from prediction.projection._steps import (
    _projected_event,
)


def _episode_bindings(
    episode: BehaviorSourceRecord,
    events: Sequence[BehaviorSourceRecord],
) -> tuple[dict[str, Any], ...]:
    return (
        episode.binding(member_type="episode"),
        *(event.binding(member_type="event") for event in events),
    )


def _partition_unphased(
    unphased_events: Sequence[Mapping[str, Any]],
    event_positions: Mapping[str, int],
    boundary: int,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    prior = tuple(item for item in unphased_events if event_positions[item["event_uri"]] <= boundary)
    future = tuple(item for item in unphased_events if event_positions[item["event_uri"]] > boundary)
    return prior, future


def _resumption_steps(
    transitions: Sequence[Mapping[str, Any]],
    events_by_uri: Mapping[str, BehaviorSourceRecord],
    boundary: int,
    event_positions: Mapping[str, int],
    *,
    future: bool,
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for edge in transitions:
        if edge["relation"] != "resumes":
            continue
        position = event_positions[edge["to_event_uri"]]
        if (position > boundary) is future:
            result.append(_projected_event(events_by_uri[edge["to_event_uri"]]))
    return tuple(result)


def _partition_actions(
    actions: Sequence[Mapping[str, Any]],
    anchor: Mapping[str, Any],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    """只把 cutoff 前已结束的 Action 声明为完成，并隐藏 active 的未来结局。"""

    cutoff = _anchor_observed_at(anchor)
    if cutoff is None:
        return (), ()
    completed: list[Mapping[str, Any]] = []
    active: list[Mapping[str, Any]] = []
    for action in actions:
        started_at = action["started_at"]
        ended_at = action["ended_at"]
        available_at = action["available_at"]
        if available_at > cutoff:
            continue
        if ended_at is not None and ended_at <= cutoff:
            completed.append(action)
        elif started_at is None or started_at <= cutoff:
            active.append(action)
    return tuple(completed), tuple(active)


def _partition_events(
    events: Sequence[BehaviorSourceRecord],
    anchor: Mapping[str, Any],
) -> tuple[tuple[BehaviorSourceRecord, ...], tuple[BehaviorSourceRecord, ...]]:
    """按真实时间把 Event 划分为截止时刻已完成或仍在进行。"""

    cutoff = _anchor_observed_at(anchor)
    if cutoff is None:
        return (), ()
    completed: list[BehaviorSourceRecord] = []
    active: list[BehaviorSourceRecord] = []
    for event in events:
        available_at = event.fields["onset_available_at"]
        if available_at > cutoff:
            continue
        ended_at = event.fields["ended_at"]
        if ended_at is not None and ended_at <= cutoff:
            completed.append(event)
        elif event.fields["started_at"] <= cutoff:
            active.append(event)
    return tuple(completed), tuple(active)

