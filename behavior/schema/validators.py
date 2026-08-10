"""三类行为文档的跨字段业务不变量；字段本身的类型校验由 fields 完成。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from behavior.model import BehaviorAddress, BehaviorKind, semantic_name
from behavior.schema.model import BehaviorSchemaError
from behavior.schema.vocabulary import EPISODE_STATUSES, EVENT_STATUSES
from behavior.uri import BehaviorURI


def validate_payload(kind: BehaviorKind, payload: dict[str, Any]) -> None:
    """按文档类型执行跨字段校验；Outcome 会就地规范化 URI 与结果顺序。"""

    if kind is BehaviorKind.EVENT:
        _validate_event(payload)
    elif kind is BehaviorKind.OUTCOME:
        _validate_outcome(payload)
    else:
        _validate_episode(payload)


def _document_uris(
    values: Sequence[str],
    expected_kind: BehaviorKind,
    label: str,
) -> tuple[BehaviorURI, ...]:
    uris: list[BehaviorURI] = []
    for value in values:
        try:
            uri = BehaviorURI.parse(value)
            if uri.to_address().kind is not expected_kind:
                raise BehaviorSchemaError(f"{label} URI identifies the wrong document type")
        except ValueError as exc:
            if isinstance(exc, BehaviorSchemaError):
                raise
            raise BehaviorSchemaError(
                f"{label} document URI must identify an L2 {expected_kind.value} document"
            ) from exc
        uris.append(uri)
    return tuple(uris)


def _validate_event(payload: dict[str, Any]) -> None:
    semantic_name(payload["event_name"], "event name")
    if payload["event_date"] != payload["started_at"].date():
        raise BehaviorSchemaError("event_date must match the local started_at date")
    if payload["ended_at"] is not None and payload["ended_at"] < payload["started_at"]:
        raise BehaviorSchemaError("event ended_at cannot precede started_at")
    if payload["onset_available_at"] < payload["started_at"]:
        raise BehaviorSchemaError("event onset_available_at cannot precede started_at")
    if payload["status"] not in EVENT_STATUSES:
        raise BehaviorSchemaError(f"event status must be one of {sorted(EVENT_STATUSES)}")
    if not payload["actions"]:
        raise BehaviorSchemaError("event must contain at least one observed action")
    sequences = tuple(action["sequence"] for action in payload["actions"])
    if sequences != tuple(range(1, len(sequences) + 1)):
        raise BehaviorSchemaError("event action sequences must be contiguous and start at one")
    action_ids = tuple(action["action_id"] for action in payload["actions"])
    if len(action_ids) != len(set(action_ids)):
        raise BehaviorSchemaError("event action IDs must be unique")
    _validate_action_times(payload)


def _validate_action_times(payload: dict[str, Any]) -> None:
    """Action 必须落在 Event 时间窗内，且顺序不得与已知时间戳矛盾。"""

    event_start = payload["started_at"]
    event_end = payload["ended_at"]
    for action in payload["actions"]:
        for timestamp in (action["started_at"], action["ended_at"]):
            if timestamp is None:
                continue
            if timestamp < event_start or (event_end is not None and timestamp > event_end):
                raise BehaviorSchemaError("Action time must remain inside its Event time window")
        if action["available_at"] < event_start:
            raise BehaviorSchemaError("Action available_at cannot precede its Event")
    for index, earlier in enumerate(payload["actions"]):
        earlier_start = earlier["started_at"]
        if earlier_start is None:
            continue
        for later in payload["actions"][index + 1 :]:
            later_end = later["ended_at"]
            if later_end is not None and later_end <= earlier_start:
                raise BehaviorSchemaError("Action order contradicts known non-overlapping timestamps")


def _validate_outcome(payload: dict[str, Any]) -> None:
    semantic_name(payload["event_name"], "event name")
    try:
        actual_event = BehaviorURI.parse(payload["event_uri"])
        actual_address = actual_event.to_address()
        if actual_address.kind is not BehaviorKind.EVENT:
            raise BehaviorSchemaError("outcome event_uri must identify an Event document")
    except ValueError as exc:
        raise BehaviorSchemaError("outcome event_uri is not a valid Event URI") from exc
    expected_event = BehaviorURI.from_address(
        BehaviorAddress.event(
            payload["event_date"],
            payload["event_name"],
            actual_address.started_at,
        )
    )
    if actual_event != expected_event:
        raise BehaviorSchemaError("outcome address must mirror its target Event URI")
    payload["event_uri"] = str(actual_event)
    if not payload["outcomes"]:
        raise BehaviorSchemaError("outcome document must contain at least one result")
    identifiers = tuple(item["outcome_id"] for item in payload["outcomes"])
    if len(identifiers) != len(set(identifiers)):
        raise BehaviorSchemaError("outcome IDs must be unique")
    payload["outcomes"] = tuple(
        sorted(
            payload["outcomes"],
            key=lambda item: (item["occurred_at"], item["outcome_id"]),
        )
    )


def _validate_episode(payload: dict[str, Any]) -> None:
    semantic_name(payload["episode_name"], "episode name")
    if payload["episode_date"] != payload["started_at"].date():
        raise BehaviorSchemaError("episode_date must match the local started_at date")
    if payload["ended_at"] < payload["started_at"]:
        raise BehaviorSchemaError("episode ended_at cannot precede started_at")
    if payload["status"] not in EPISODE_STATUSES:
        raise BehaviorSchemaError(f"episode status must be one of {sorted(EPISODE_STATUSES)}")
    events = _document_uris(payload["ordered_event_uris"], BehaviorKind.EVENT, "Episode Event")
    if len(events) < 2:
        raise BehaviorSchemaError("episode must reference at least two Events")
    _validate_episode_outcome_snapshots(payload)

    event_positions = {str(uri): index for index, uri in enumerate(events)}
    assigned_events = _validate_episode_phases(payload, event_positions)
    _validate_episode_unphased_events(payload, event_positions, assigned_events)
    _validate_episode_transitions(payload, event_positions)


def _validate_episode_outcome_snapshots(payload: dict[str, Any]) -> None:
    """系统冻结的 Outcome 快照必须与 outcome_uris 一一对应且顺序一致。"""

    outcomes = _document_uris(payload["outcome_uris"], BehaviorKind.OUTCOME, "Episode Outcome")
    snapshot_outcomes = _document_uris(
        tuple(snapshot["uri"] for snapshot in payload["outcome_snapshots"]),
        BehaviorKind.OUTCOME,
        "Episode Outcome snapshot",
    )
    if tuple(str(uri) for uri in snapshot_outcomes) != tuple(str(uri) for uri in outcomes):
        raise BehaviorSchemaError("episode outcome snapshots must match outcome_uris in order")


def _validate_episode_phases(
    payload: dict[str, Any],
    event_positions: dict[str, int],
) -> set[str]:
    """Phase 必须落在 Episode 时间窗内，顺序跟随 ordered_event_uris 且互不交叉。"""

    if not payload["phases"]:
        raise BehaviorSchemaError("episode must contain at least one semantic Phase")
    phase_sequences = tuple(phase["sequence"] for phase in payload["phases"])
    if phase_sequences != tuple(range(1, len(phase_sequences) + 1)):
        raise BehaviorSchemaError("episode phase sequences must be contiguous and start at one")
    phase_ids = tuple(phase["phase_id"] for phase in payload["phases"])
    if len(phase_ids) != len(set(phase_ids)):
        raise BehaviorSchemaError("episode phase IDs must be unique")
    assigned_events: set[str] = set()
    previous_phase_position = -1
    for phase in payload["phases"]:
        if phase["started_at"] < payload["started_at"]:
            raise BehaviorSchemaError("Episode Phase starts outside the Episode time window")
        if phase["ended_at"] is not None:
            if phase["ended_at"] < phase["started_at"]:
                raise BehaviorSchemaError("Episode Phase ended_at cannot precede started_at")
            if phase["ended_at"] > payload["ended_at"]:
                raise BehaviorSchemaError("Episode Phase ends outside the Episode time window")
        phase_events = phase["event_uris"]
        if any(uri not in event_positions for uri in phase_events):
            raise BehaviorSchemaError("episode phase references an Event outside ordered_event_uris")
        if any(uri in assigned_events for uri in phase_events):
            raise BehaviorSchemaError("one Event cannot belong to multiple Episode phases")
        positions = tuple(event_positions[uri] for uri in phase_events)
        if positions != tuple(sorted(positions)):
            raise BehaviorSchemaError("episode phase Event order must follow ordered_event_uris")
        if positions and positions[0] <= previous_phase_position:
            raise BehaviorSchemaError("Episode Phase order must follow ordered_event_uris without crossing")
        if positions:
            previous_phase_position = positions[-1]
        assigned_events.update(phase_events)
    return assigned_events


def _validate_episode_unphased_events(
    payload: dict[str, Any],
    event_positions: dict[str, int],
    assigned_events: set[str],
) -> None:
    """每个被引用 Event 必须恰好被分类为进入 Phase 或明确未进入 Phase。"""

    unphased = tuple(item["event_uri"] for item in payload["unphased_events"])
    _document_uris(unphased, BehaviorKind.EVENT, "Episode unphased Event")
    unphased_positions = tuple(event_positions.get(uri, -1) for uri in unphased)
    if any(position < 0 for position in unphased_positions):
        raise BehaviorSchemaError("episode unphased Event lies outside ordered_event_uris")
    if unphased_positions != tuple(sorted(unphased_positions)):
        raise BehaviorSchemaError("episode unphased Events must follow ordered_event_uris")
    unphased_set = set(unphased)
    if assigned_events & unphased_set:
        raise BehaviorSchemaError("one Event cannot be both phased and unphased")
    if assigned_events | unphased_set != set(event_positions):
        raise BehaviorSchemaError("every Episode Event must be classified as phased or unphased")


def _validate_episode_transitions(
    payload: dict[str, Any],
    event_positions: dict[str, int],
) -> None:
    """转移只能指向 ordered_event_uris 中更靠后的 Event，且不得重复。"""

    identities: set[tuple[str, str, str]] = set()
    for transition in payload["transitions"]:
        source = transition["from_event_uri"]
        target = transition["to_event_uri"]
        if source not in event_positions or target not in event_positions:
            raise BehaviorSchemaError("episode transition references an Event outside ordered_event_uris")
        if event_positions[source] >= event_positions[target]:
            raise BehaviorSchemaError("episode transition must follow the real Event order")
        identity = (source, target, transition["relation"])
        if identity in identities:
            raise BehaviorSchemaError("episode transitions must be unique")
        identities.add(identity)


__all__ = ["validate_payload"]
