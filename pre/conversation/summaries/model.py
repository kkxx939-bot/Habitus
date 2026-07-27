"""Conversation 归档片段的宽语义历史过程摘要 Schema。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from foundation.ids import require_safe_path_segment
from foundation.integrity import canonical_digest, canonicalize
from pre.conversation.messages.model import (
    ConversationSegment,
    conversation_datetime,
    require_sha256,
)


class ConversationSummarySchemaError(ValueError):
    """Conversation 片段摘要不满足来源或内容契约。"""


_SUMMARY_RANGE_ID = re.compile(r"^[0-9]{12}-[0-9]{12}$")
_MAX_SUMMARY_SEQUENCE = 999_999_999_999


@dataclass(frozen=True)
class ConversationSummaryContent:
    """LLM 只负责生成的历史过程语义，不包含可信来源字段。"""

    overview: str
    chronology: tuple[str, ...]
    corrections: tuple[str, ...]
    ending_state: str
    open_threads: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "overview", _required_text(self.overview, "overview"))
        object.__setattr__(
            self,
            "chronology",
            _text_sequence(self.chronology, "chronology", required=True),
        )
        object.__setattr__(
            self,
            "corrections",
            _text_sequence(self.corrections, "corrections", required=False),
        )
        object.__setattr__(self, "ending_state", _required_text(self.ending_state, "ending_state"))
        object.__setattr__(
            self,
            "open_threads",
            _text_sequence(self.open_threads, "open_threads", required=False),
        )
        for name, maximum in (
            ("overview", 16_000),
            ("ending_state", 4_000),
        ):
            if len(getattr(self, name)) > maximum:
                raise ConversationSummarySchemaError(f"{name} exceeds its character bound")
        for name in ("chronology", "corrections", "open_threads"):
            values = getattr(self, name)
            if len(values) > 128 or any(len(item) > 2_000 for item in values):
                raise ConversationSummarySchemaError(f"{name} exceeds its bounded list shape")

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(
            {
                "overview": self.overview,
                "chronology": self.chronology,
                "corrections": self.corrections,
                "ending_state": self.ending_state,
                "open_threads": self.open_threads,
            }
        )

    @classmethod
    def model_json_schema(cls) -> dict[str, object]:
        item = {"type": "string", "minLength": 1, "maxLength": 2_000}
        return {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "overview",
                "chronology",
                "corrections",
                "ending_state",
                "open_threads",
            ],
            "properties": {
                "overview": {"type": "string", "minLength": 1, "maxLength": 16_000},
                "chronology": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 128,
                    "items": item,
                },
                "corrections": {
                    "type": "array",
                    "maxItems": 128,
                    "items": item,
                },
                "ending_state": {"type": "string", "minLength": 1, "maxLength": 4_000},
                "open_threads": {
                    "type": "array",
                    "maxItems": 128,
                    "items": item,
                },
            },
        }

    @classmethod
    def model_validate(cls, value: object) -> ConversationSummaryContent:
        if not isinstance(value, Mapping):
            raise ConversationSummarySchemaError("conversation summary content must be an object")
        expected = {
            "overview",
            "chronology",
            "corrections",
            "ending_state",
            "open_threads",
        }
        if set(value) != expected:
            raise ConversationSummarySchemaError(
                "conversation summary content must contain exactly the declared fields"
            )
        return cls(
            overview=value["overview"],
            chronology=value["chronology"],
            corrections=value["corrections"],
            ending_state=value["ending_state"],
            open_threads=value["open_threads"],
        )


def _safe_identifier(value: object, label: str) -> str:
    try:
        identifier = require_safe_path_segment(value, label)
    except ValueError as exc:
        raise ConversationSummarySchemaError(str(exc)) from exc
    if identifier != identifier.strip() or any(ord(character) < 32 for character in identifier):
        raise ConversationSummarySchemaError(f"{label} contains unsafe characters")
    return identifier


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversationSummarySchemaError(f"{label} must be non-empty text")
    return value


def _text_sequence(value: object, label: str, *, required: bool) -> tuple[str, ...]:
    if not isinstance(value, tuple | list):
        raise ConversationSummarySchemaError(f"{label} must be a list of text items")
    resolved = tuple(_required_text(item, f"{label} item") for item in value)
    if required and not resolved:
        raise ConversationSummarySchemaError(f"{label} must contain at least one item")
    return resolved


def _summary_datetime(value: object, label: str) -> datetime:
    if not isinstance(value, datetime | str):
        raise ConversationSummarySchemaError(f"{label} must be a datetime or ISO-8601 string")
    try:
        return conversation_datetime(value, label)
    except ValueError as exc:
        raise ConversationSummarySchemaError(str(exc)) from exc


@dataclass(frozen=True)
class ConversationSegmentSummary:
    """描述一个不可变归档片段中完整历史过程的派生语义。"""

    conversation_id: str
    segment_id: str
    source_message_digest: str
    start_sequence: int
    end_sequence: int
    started_at: datetime
    ended_at: datetime
    generated_at: datetime
    overview: str
    chronology: tuple[str, ...]
    corrections: tuple[str, ...]
    ending_state: str
    open_threads: tuple[str, ...]

    SCHEMA_VERSION = "conversation_segment_summary_v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "conversation_id",
            _safe_identifier(self.conversation_id, "conversation_id"),
        )
        object.__setattr__(self, "segment_id", _safe_identifier(self.segment_id, "segment_id"))
        try:
            source_digest = require_sha256(self.source_message_digest, "source_message_digest")
        except ValueError as exc:
            raise ConversationSummarySchemaError(str(exc)) from exc
        object.__setattr__(self, "source_message_digest", source_digest)

        for label, value in (
            ("start_sequence", self.start_sequence),
            ("end_sequence", self.end_sequence),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConversationSummarySchemaError(f"{label} must be a non-negative integer")
        if self.start_sequence > self.end_sequence:
            raise ConversationSummarySchemaError("summary sequence range is invalid")

        started_at = _summary_datetime(self.started_at, "summary started_at")
        ended_at = _summary_datetime(self.ended_at, "summary ended_at")
        generated_at = _summary_datetime(self.generated_at, "summary generated_at")
        if started_at > ended_at or generated_at < ended_at:
            raise ConversationSummarySchemaError("summary time range is invalid")
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "ended_at", ended_at)
        object.__setattr__(self, "generated_at", generated_at)

        object.__setattr__(self, "overview", _required_text(self.overview, "overview"))
        object.__setattr__(
            self,
            "chronology",
            _text_sequence(self.chronology, "chronology", required=True),
        )
        object.__setattr__(
            self,
            "corrections",
            _text_sequence(self.corrections, "corrections", required=False),
        )
        object.__setattr__(self, "ending_state", _required_text(self.ending_state, "ending_state"))
        object.__setattr__(
            self,
            "open_threads",
            _text_sequence(self.open_threads, "open_threads", required=False),
        )

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())

    def require_matches_source(self, segment: ConversationSegment) -> None:
        """确认摘要只绑定其声明的不可变消息片段。"""

        if not isinstance(segment, ConversationSegment):
            raise TypeError("segment must be a ConversationSegment")
        expected = (
            segment.conversation_id,
            segment.segment_id,
            segment.digest,
            segment.start_sequence,
            segment.end_sequence,
            segment.started_at,
            segment.ended_at,
        )
        actual = (
            self.conversation_id,
            self.segment_id,
            self.source_message_digest,
            self.start_sequence,
            self.end_sequence,
            self.started_at,
            self.ended_at,
        )
        if actual != expected:
            raise ConversationSummarySchemaError("conversation summary does not match its source segment")

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema_version": self.SCHEMA_VERSION,
                "conversation_id": self.conversation_id,
                "segment_id": self.segment_id,
                "source_message_digest": self.source_message_digest,
                "start_sequence": self.start_sequence,
                "end_sequence": self.end_sequence,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "generated_at": self.generated_at,
                "overview": self.overview,
                "chronology": self.chronology,
                "corrections": self.corrections,
                "ending_state": self.ending_state,
                "open_threads": self.open_threads,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConversationSegmentSummary:
        allowed = {
            "schema_version",
            "conversation_id",
            "segment_id",
            "source_message_digest",
            "start_sequence",
            "end_sequence",
            "started_at",
            "ended_at",
            "generated_at",
            "overview",
            "chronology",
            "corrections",
            "ending_state",
            "open_threads",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ConversationSummarySchemaError(
                f"conversation summary contains unknown fields: {sorted(unknown)}"
            )
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ConversationSummarySchemaError("unsupported conversation summary schema")
        required = allowed - {"schema_version"}
        missing = required - set(payload)
        if missing:
            raise ConversationSummarySchemaError(f"conversation summary is missing fields: {sorted(missing)}")
        return cls(
            conversation_id=payload["conversation_id"],
            segment_id=payload["segment_id"],
            source_message_digest=payload["source_message_digest"],
            start_sequence=payload["start_sequence"],
            end_sequence=payload["end_sequence"],
            started_at=_summary_datetime(payload["started_at"], "summary started_at"),
            ended_at=_summary_datetime(payload["ended_at"], "summary ended_at"),
            generated_at=_summary_datetime(payload["generated_at"], "summary generated_at"),
            overview=payload["overview"],
            chronology=payload["chronology"],
            corrections=payload["corrections"],
            ending_state=payload["ending_state"],
            open_threads=payload["open_threads"],
        )


class ConversationSummarySourceKind(str, Enum):
    """范围摘要允许绑定的两种不可变来源。"""

    SEGMENT = "segment_summary"
    RANGE = "range_summary"


class ConversationRangeSummaryStage(str, Enum):
    """两次且仅两次自动压缩所对应的显式阶段。"""

    RANGE = "range"
    ARCHIVE = "archive"

    @property
    def source_kind(self) -> ConversationSummarySourceKind:
        if self is ConversationRangeSummaryStage.RANGE:
            return ConversationSummarySourceKind.SEGMENT
        return ConversationSummarySourceKind.RANGE


@dataclass(frozen=True)
class ConversationSummarySourceRef:
    """范围摘要对一个不可变来源的确定性绑定。"""

    kind: ConversationSummarySourceKind
    summary_id: str
    digest: str
    start_sequence: int
    end_sequence: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ConversationSummarySourceKind):
            raise ConversationSummarySchemaError("summary source kind must be declared")
        object.__setattr__(self, "summary_id", _safe_identifier(self.summary_id, "summary source id"))
        try:
            digest = require_sha256(self.digest, "summary source digest")
        except ValueError as exc:
            raise ConversationSummarySchemaError(str(exc)) from exc
        object.__setattr__(self, "digest", digest)
        _require_summary_range(self.start_sequence, self.end_sequence)
        expected_id = _summary_range_id(self.start_sequence, self.end_sequence)
        if self.summary_id != expected_id:
            raise ConversationSummarySchemaError("summary source id does not match its sequence range")

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(
            {
                "kind": self.kind.value,
                "summary_id": self.summary_id,
                "digest": self.digest,
                "start_sequence": self.start_sequence,
                "end_sequence": self.end_sequence,
            }
        )

    @classmethod
    def from_summary(
        cls,
        source: ConversationSegmentSummary | ConversationRangeSummary,
    ) -> ConversationSummarySourceRef:
        """从已经通过 Schema 校验的不可变摘要构造可信来源绑定。"""

        return _summary_source_ref(source)

    @classmethod
    def from_dict(cls, payload: object) -> ConversationSummarySourceRef:
        if not isinstance(payload, Mapping):
            raise ConversationSummarySchemaError("summary source reference must be an object")
        expected = {
            "kind",
            "summary_id",
            "digest",
            "start_sequence",
            "end_sequence",
        }
        if set(payload) != expected:
            raise ConversationSummarySchemaError("summary source reference fields do not match the schema")
        try:
            kind = ConversationSummarySourceKind(payload["kind"])
        except (TypeError, ValueError) as exc:
            raise ConversationSummarySchemaError("summary source kind is unsupported") from exc
        return cls(
            kind=kind,
            summary_id=payload["summary_id"],
            digest=payload["digest"],
            start_sequence=payload["start_sequence"],
            end_sequence=payload["end_sequence"],
        )


@dataclass(frozen=True)
class ConversationRangeSummary:
    """覆盖一段连续消息范围、且永不原地改写的长期过程摘要。"""

    conversation_id: str
    range_id: str
    stage: ConversationRangeSummaryStage
    source_refs: tuple[ConversationSummarySourceRef, ...]
    start_sequence: int
    end_sequence: int
    started_at: datetime
    ended_at: datetime
    generated_at: datetime
    overview: str
    chronology: tuple[str, ...]
    corrections: tuple[str, ...]
    ending_state: str
    open_threads: tuple[str, ...]

    SCHEMA_VERSION = "conversation_range_summary_v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "conversation_id",
            _safe_identifier(self.conversation_id, "conversation_id"),
        )
        object.__setattr__(self, "range_id", _safe_identifier(self.range_id, "range_id"))
        if not isinstance(self.stage, ConversationRangeSummaryStage):
            raise ConversationSummarySchemaError("range summary stage must be declared")
        if not isinstance(self.source_refs, tuple) or not 2 <= len(self.source_refs) <= 1_000:
            raise ConversationSummarySchemaError("range summary must bind between 2 and 1000 sources")
        if any(not isinstance(item, ConversationSummarySourceRef) for item in self.source_refs):
            raise ConversationSummarySchemaError("range summary sources must be source references")
        if any(item.kind is not self.stage.source_kind for item in self.source_refs):
            raise ConversationSummarySchemaError("range summary source kind does not match its stage")
        for previous, current in zip(self.source_refs, self.source_refs[1:], strict=False):
            if current.start_sequence != previous.end_sequence + 1:
                raise ConversationSummarySchemaError("range summary sources must be contiguous and ordered")

        _require_summary_range(self.start_sequence, self.end_sequence)
        expected_range_id = _summary_range_id(self.start_sequence, self.end_sequence)
        if self.range_id != expected_range_id:
            raise ConversationSummarySchemaError("range_id does not match its sequence range")
        if (
            self.source_refs[0].start_sequence != self.start_sequence
            or self.source_refs[-1].end_sequence != self.end_sequence
        ):
            raise ConversationSummarySchemaError("range summary coverage does not match its sources")

        started_at = _summary_datetime(self.started_at, "range summary started_at")
        ended_at = _summary_datetime(self.ended_at, "range summary ended_at")
        generated_at = _summary_datetime(self.generated_at, "range summary generated_at")
        if started_at > ended_at or generated_at < ended_at:
            raise ConversationSummarySchemaError("range summary time range is invalid")
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "ended_at", ended_at)
        object.__setattr__(self, "generated_at", generated_at)

        content = ConversationSummaryContent(
            overview=self.overview,
            chronology=self.chronology,
            corrections=self.corrections,
            ending_state=self.ending_state,
            open_threads=self.open_threads,
        )
        object.__setattr__(self, "overview", content.overview)
        object.__setattr__(self, "chronology", content.chronology)
        object.__setattr__(self, "corrections", content.corrections)
        object.__setattr__(self, "ending_state", content.ending_state)
        object.__setattr__(self, "open_threads", content.open_threads)

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())

    def require_matches_sources(
        self,
        sources: tuple[ConversationSegmentSummary | ConversationRangeSummary, ...],
    ) -> None:
        """确认父摘要完整绑定传入的同会话、同阶段连续来源。"""

        if not isinstance(sources, tuple) or len(sources) != len(self.source_refs):
            raise ConversationSummarySchemaError("range summary source count does not match")
        expected_type = (
            ConversationSegmentSummary
            if self.stage is ConversationRangeSummaryStage.RANGE
            else ConversationRangeSummary
        )
        if any(not isinstance(source, expected_type) for source in sources):
            raise ConversationSummarySchemaError("range summary source type does not match its stage")
        if self.stage is ConversationRangeSummaryStage.ARCHIVE and any(
            not isinstance(source, ConversationRangeSummary)
            or source.stage is not ConversationRangeSummaryStage.RANGE
            for source in sources
        ):
            raise ConversationSummarySchemaError("archive summary sources must be range-stage summaries")
        if any(source.conversation_id != self.conversation_id for source in sources):
            raise ConversationSummarySchemaError("range summary sources belong to another conversation")
        references = tuple(_summary_source_ref(source) for source in sources)
        if references != self.source_refs:
            raise ConversationSummarySchemaError("range summary source bindings do not match")
        if self.started_at != sources[0].started_at or self.ended_at != sources[-1].ended_at:
            raise ConversationSummarySchemaError("range summary time coverage does not match its sources")

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema_version": self.SCHEMA_VERSION,
                "conversation_id": self.conversation_id,
                "range_id": self.range_id,
                "stage": self.stage.value,
                "source_refs": tuple(item.to_dict() for item in self.source_refs),
                "start_sequence": self.start_sequence,
                "end_sequence": self.end_sequence,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "generated_at": self.generated_at,
                "overview": self.overview,
                "chronology": self.chronology,
                "corrections": self.corrections,
                "ending_state": self.ending_state,
                "open_threads": self.open_threads,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConversationRangeSummary:
        expected = {
            "schema_version",
            "conversation_id",
            "range_id",
            "stage",
            "source_refs",
            "start_sequence",
            "end_sequence",
            "started_at",
            "ended_at",
            "generated_at",
            "overview",
            "chronology",
            "corrections",
            "ending_state",
            "open_threads",
        }
        if set(payload) != expected:
            raise ConversationSummarySchemaError("range summary fields do not match the schema")
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ConversationSummarySchemaError("unsupported range summary schema")
        try:
            stage = ConversationRangeSummaryStage(payload["stage"])
        except (TypeError, ValueError) as exc:
            raise ConversationSummarySchemaError("range summary stage is unsupported") from exc
        source_refs = payload["source_refs"]
        if not isinstance(source_refs, list | tuple):
            raise ConversationSummarySchemaError("range summary source_refs must be a list")
        return cls(
            conversation_id=payload["conversation_id"],
            range_id=payload["range_id"],
            stage=stage,
            source_refs=tuple(ConversationSummarySourceRef.from_dict(item) for item in source_refs),
            start_sequence=payload["start_sequence"],
            end_sequence=payload["end_sequence"],
            started_at=_summary_datetime(payload["started_at"], "range summary started_at"),
            ended_at=_summary_datetime(payload["ended_at"], "range summary ended_at"),
            generated_at=_summary_datetime(payload["generated_at"], "range summary generated_at"),
            overview=payload["overview"],
            chronology=payload["chronology"],
            corrections=payload["corrections"],
            ending_state=payload["ending_state"],
            open_threads=payload["open_threads"],
        )


def _require_summary_range(start_sequence: int, end_sequence: int) -> None:
    for label, value in (("start_sequence", start_sequence), ("end_sequence", end_sequence)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_SUMMARY_SEQUENCE:
            raise ConversationSummarySchemaError(f"{label} must be a bounded non-negative integer")
    if start_sequence > end_sequence:
        raise ConversationSummarySchemaError("summary sequence range is invalid")


def _summary_range_id(start_sequence: int, end_sequence: int) -> str:
    value = f"{start_sequence:012d}-{end_sequence:012d}"
    if _SUMMARY_RANGE_ID.fullmatch(value) is None:
        raise ConversationSummarySchemaError("summary sequence range cannot form a valid identity")
    return value


def _summary_source_ref(
    source: ConversationSegmentSummary | ConversationRangeSummary,
) -> ConversationSummarySourceRef:
    if isinstance(source, ConversationSegmentSummary):
        kind = ConversationSummarySourceKind.SEGMENT
        summary_id = source.segment_id
    elif isinstance(source, ConversationRangeSummary):
        if source.stage is not ConversationRangeSummaryStage.RANGE:
            raise ConversationSummarySchemaError("archive range summaries cannot be compressed again")
        kind = ConversationSummarySourceKind.RANGE
        summary_id = source.range_id
    else:
        raise TypeError("range summary source has an unsupported type")
    return ConversationSummarySourceRef(
        kind=kind,
        summary_id=summary_id,
        digest=source.digest,
        start_sequence=source.start_sequence,
        end_sequence=source.end_sequence,
    )


__all__ = [
    "ConversationRangeSummary",
    "ConversationRangeSummaryStage",
    "ConversationSegmentSummary",
    "ConversationSummaryContent",
    "ConversationSummarySchemaError",
    "ConversationSummarySourceKind",
    "ConversationSummarySourceRef",
]
