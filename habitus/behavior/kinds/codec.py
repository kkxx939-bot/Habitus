"""词表记录 ↔ JSON 的编解码；只管形状，不碰文件。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from habitus.behavior.kinds.changes import (
    AddClass,
    Branch,
    ChangeReason,
    Move,
    Operation,
    RetireClass,
    ReviseClass,
    VersionRecord,
)
from habitus.behavior.kinds.ids import ClassId, KindIdError, Lane
from habitus.behavior.kinds.model import BehaviorClass, BehaviorKindError, ClassOrigin, ClassStatus, Exclusion
from habitus.behavior.kinds.pending import PendingEntry
from habitus.behavior.kinds.schedule import JobState


class BehaviorKindCodecError(ValueError):
    pass


def class_payload(item: BehaviorClass) -> dict[str, Any]:
    return {
        "id": str(item.id),
        "name": item.name,
        "criterion": item.criterion,
        "reminder": item.reminder,
        "excludes": [
            {"text": exclusion.text, "goes_to": None if exclusion.goes_to is None else str(exclusion.goes_to)}
            for exclusion in item.excludes
        ],
        "examples": list(item.examples),
        "status": item.status.value,
        "origin": item.origin.value,
    }


def class_from(raw: object) -> BehaviorClass:
    data = _object(raw, {"id", "name", "criterion", "reminder", "excludes", "examples", "status", "origin"}, "class")
    try:
        return BehaviorClass(
            id=ClassId.parse(data["id"]),
            name=data["name"],
            criterion=data["criterion"],
            reminder=data["reminder"],
            excludes=tuple(_exclusion(item) for item in _list(data["excludes"], "excludes")),
            examples=tuple(_list(data["examples"], "examples")),
            status=ClassStatus(data["status"]),
            origin=ClassOrigin(data["origin"]),
        )
    except (BehaviorKindError, KindIdError, ValueError, TypeError) as exc:
        raise BehaviorKindCodecError(f"behavior class is invalid: {exc}") from exc


def record_payload(record: VersionRecord) -> dict[str, Any]:
    return {
        "version": record.version,
        "at": record.at.isoformat(timespec="microseconds"),
        "reason": record.reason.value,
        "note": record.note,
        "operations": [_operation_payload(operation) for operation in record.operations],
        "moves": [[move.occurrence, move.source, move.target] for move in record.moves],
    }


def record_from(raw: object) -> VersionRecord:
    data = _object(raw, {"version", "at", "reason", "note", "operations", "moves"}, "version record")
    try:
        moves = []
        for item in _list(data["moves"], "moves"):
            if not isinstance(item, list) or len(item) != 3:
                raise BehaviorKindCodecError("move must be [occurrence, source, target]")
            moves.append(Move(*item))
        return VersionRecord(
            version=data["version"],
            at=datetime.fromisoformat(data["at"]),
            reason=ChangeReason(data["reason"]),
            operations=tuple(_operation_from(item) for item in _list(data["operations"], "operations")),
            moves=tuple(moves),
            note=data["note"] if isinstance(data["note"], str) else "",
        )
    except (BehaviorKindError, KindIdError, ValueError, TypeError) as exc:
        raise BehaviorKindCodecError(f"version record is invalid: {exc}") from exc


def pending_payload(entry: PendingEntry) -> dict[str, Any]:
    return {
        "occurrence": entry.occurrence,
        "lane": entry.lane.value,
        "day": entry.day.isoformat(),
        "proposed": entry.proposed,
        "content": entry.content,
        "rechecked": entry.rechecked,
    }


def pending_from(raw: object) -> PendingEntry:
    data = _object(raw, {"occurrence", "lane", "day", "proposed", "content", "rechecked"}, "pending entry")
    try:
        return PendingEntry(
            occurrence=data["occurrence"],
            lane=Lane(data["lane"]),
            day=date.fromisoformat(data["day"]),
            proposed=data["proposed"],
            content=data["content"],
            rechecked=data["rechecked"],
        )
    except (BehaviorKindError, ValueError, TypeError) as exc:
        raise BehaviorKindCodecError(f"pending entry is invalid: {exc}") from exc


def job_state_payload(state: JobState) -> dict[str, Any]:
    return {
        "nightly_at": _iso(state.nightly_at),
        "revision_at": _iso(state.revision_at),
        "merge_history": [[period, sorted([list(pair) for pair in pairs])] for period, pairs in state.merge_history],
    }


def job_state_from(raw: object) -> JobState:
    data = _object(raw, {"nightly_at", "revision_at", "merge_history"}, "job state")
    try:
        history = []
        for entry in _list(data["merge_history"], "merge_history"):
            if not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], str):
                raise BehaviorKindCodecError("merge history entry must be [period, pairs]")
            pairs = set()
            for pair in _list(entry[1], "merge pairs"):
                if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(item, str) for item in pair):
                    raise BehaviorKindCodecError("merge pair must be [source, target]")
                pairs.add((pair[0], pair[1]))
            history.append((entry[0], frozenset(pairs)))
        return JobState(_parse_time(data["nightly_at"]), _parse_time(data["revision_at"]), tuple(history))
    except (BehaviorKindError, ValueError, TypeError) as exc:
        raise BehaviorKindCodecError(f"job state is invalid: {exc}") from exc


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat(timespec="microseconds")


def _parse_time(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise BehaviorKindCodecError("time must be ISO text or null")
    return datetime.fromisoformat(value)


def _operation_payload(operation: Operation) -> dict[str, Any]:
    if isinstance(operation, AddClass):
        return {"op": "add", "class": class_payload(operation.item)}
    if isinstance(operation, ReviseClass):
        return {"op": "revise", "class": class_payload(operation.item)}
    if isinstance(operation, RetireClass):
        return {"op": "retire", "id": str(operation.class_id)}
    return {"op": "branch", "source": str(operation.source), "target": str(operation.target)}


def _operation_from(raw: object) -> Operation:
    if not isinstance(raw, dict) or "op" not in raw:
        raise BehaviorKindCodecError("operation must be an object with `op`")
    kind = raw["op"]
    if kind == "add":
        return AddClass(class_from(_object(raw, {"op", "class"}, "add")["class"]))
    if kind == "revise":
        return ReviseClass(class_from(_object(raw, {"op", "class"}, "revise")["class"]))
    if kind == "retire":
        return RetireClass(ClassId.parse(_object(raw, {"op", "id"}, "retire")["id"]))
    if kind == "branch":
        data = _object(raw, {"op", "source", "target"}, "branch")
        return Branch(ClassId.parse(data["source"]), ClassId.parse(data["target"]))
    raise BehaviorKindCodecError(f"unknown operation: {kind!r}")


def _exclusion(raw: object) -> Exclusion:
    data = _object(raw, {"text", "goes_to"}, "exclusion")
    target = data["goes_to"]
    return Exclusion(text=data["text"], goes_to=None if target is None else ClassId.parse(target))


def _object(raw: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != keys:
        raise BehaviorKindCodecError(f"{label} shape is invalid")
    return raw


def _list(raw: object, label: str) -> list[Any]:
    if not isinstance(raw, list):
        raise BehaviorKindCodecError(f"{label} must be a list")
    return raw


__all__ = [
    "BehaviorKindCodecError",
    "class_from",
    "class_payload",
    "pending_from",
    "pending_payload",
    "record_from",
    "record_payload",
    "job_state_from",
    "job_state_payload",
]
