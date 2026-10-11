"""行为侧会话 lane 处理完一份会话源之后留下的回执。

会话源的交付契约要求每个消费者为每份源留下一份耐久输出（或一个"跳过"）。会话 lane 的产物本身在行为侧
（判断记录，之后落到行为树），所以这里只留一份很小的回执：这份源里拆出了哪几轮、各轮的时刻、哪几轮出了记录。
不存对话内容。回执的每个字段都只由这份源决定，同一份源重做得到同一份回执。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from habitus.conversation.source.model import (
    ConversationSourceEnvelope,
    ConversationSourceError,
    require_record,
    require_sha256,
    source_timestamp,
)
from habitus.conversation.source.receipt import ConversationSourceConsumer, conversation_consumer_output_id
from habitus.foundation.integrity import canonical_digest, canonicalize

BEHAVIOR_SESSION_OUTPUT_KIND = "behavior_session_receipt"
BEHAVIOR_SESSION_OUTPUT_SCHEMA_VERSION = "conversation_behavior_session_receipt_v1"


@dataclass(frozen=True)
class BehaviorSessionTurnRecord:
    """一轮在回执里的一行。``recorded`` 为假表示这一轮没出记录（助手没有答复也没有调用）。"""

    start_sequence: int
    end_sequence: int
    instructed_at: datetime
    completed_at: datetime
    recorded: bool

    def __post_init__(self) -> None:
        for label, value in (("start_sequence", self.start_sequence), ("end_sequence", self.end_sequence)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConversationSourceError(f"behavior session turn {label} must be a non-negative integer")
        if self.end_sequence < self.start_sequence:
            raise ConversationSourceError("behavior session turn sequences are reversed")
        object.__setattr__(self, "instructed_at", source_timestamp(self.instructed_at, "instructed_at"))
        object.__setattr__(self, "completed_at", source_timestamp(self.completed_at, "completed_at"))
        if self.completed_at < self.instructed_at:
            raise ConversationSourceError("behavior session turn completes before it was instructed")
        if not isinstance(self.recorded, bool):
            raise ConversationSourceError("behavior session turn recorded must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_sequence": self.start_sequence,
            "end_sequence": self.end_sequence,
            "instructed_at": self.instructed_at.isoformat(timespec="microseconds"),
            "completed_at": self.completed_at.isoformat(timespec="microseconds"),
            "recorded": self.recorded,
        }

    @classmethod
    def from_dict(cls, value: object) -> BehaviorSessionTurnRecord:
        record = require_record(
            value,
            expected={"start_sequence", "end_sequence", "instructed_at", "completed_at", "recorded"},
            label="behavior session turn",
        )
        return cls(
            start_sequence=record["start_sequence"],
            end_sequence=record["end_sequence"],
            instructed_at=record["instructed_at"],
            completed_at=record["completed_at"],
            recorded=record["recorded"],
        )


@dataclass(frozen=True)
class BehaviorSessionOutput:
    """一份会话源对应的会话 lane 回执。"""

    output_id: str
    source_id: str
    source_payload_digest: str
    processor_fingerprint: str
    turns: tuple[BehaviorSessionTurnRecord, ...]
    output_record_digest: str

    consumer = ConversationSourceConsumer.BEHAVIOR_SESSION

    def __post_init__(self) -> None:
        for label, value in (
            ("output_id", self.output_id),
            ("source_id", self.source_id),
            ("source_payload_digest", self.source_payload_digest),
            ("processor_fingerprint", self.processor_fingerprint),
            ("output_record_digest", self.output_record_digest),
        ):
            require_sha256(value, f"behavior session {label}")
        if not isinstance(self.turns, tuple) or not self.turns:
            raise ConversationSourceError("behavior session receipt must list at least one turn")
        if any(not isinstance(item, BehaviorSessionTurnRecord) for item in self.turns):
            raise TypeError("behavior session turns must be BehaviorSessionTurnRecord values")
        starts = [item.start_sequence for item in self.turns]
        if starts != sorted(set(starts)):
            raise ConversationSourceError("behavior session turns must be ordered and distinct")
        expected = conversation_consumer_output_id(
            source_id=self.source_id,
            source_payload_digest=self.source_payload_digest,
            consumer=self.consumer,
            processor_fingerprint=self.processor_fingerprint,
            output_schema_version=BEHAVIOR_SESSION_OUTPUT_SCHEMA_VERSION,
        )
        if self.output_id != expected:
            raise ConversationSourceError("behavior session output_id does not match output identity")
        if self.output_record_digest != canonical_digest(self._record_without_digest()):
            raise ConversationSourceError("behavior session output_record_digest does not match output record")

    @classmethod
    def create(
        cls,
        *,
        source: ConversationSourceEnvelope,
        processor_fingerprint: str,
        turns: tuple[BehaviorSessionTurnRecord, ...],
    ) -> BehaviorSessionOutput:
        output_id = conversation_consumer_output_id(
            source_id=source.source_id,
            source_payload_digest=source.source_payload_digest,
            consumer=ConversationSourceConsumer.BEHAVIOR_SESSION,
            processor_fingerprint=processor_fingerprint,
            output_schema_version=BEHAVIOR_SESSION_OUTPUT_SCHEMA_VERSION,
        )
        record = _record(output_id, source.source_id, source.source_payload_digest, processor_fingerprint, turns)
        return cls(
            output_id=output_id,
            source_id=source.source_id,
            source_payload_digest=source.source_payload_digest,
            processor_fingerprint=processor_fingerprint,
            turns=turns,
            output_record_digest=canonical_digest(record),
        )

    def _record_without_digest(self) -> dict[str, Any]:
        return _record(
            self.output_id, self.source_id, self.source_payload_digest, self.processor_fingerprint, self.turns
        )

    def to_dict(self) -> dict[str, Any]:
        return canonicalize({**self._record_without_digest(), "output_record_digest": self.output_record_digest})

    @classmethod
    def from_dict(cls, value: object) -> BehaviorSessionOutput:
        record = require_record(
            value,
            expected={
                "schema_version",
                "output_id",
                "source_id",
                "source_payload_digest",
                "consumer",
                "processor_fingerprint",
                "turns",
                "output_record_digest",
            },
            label="behavior session receipt",
            schema_version=BEHAVIOR_SESSION_OUTPUT_SCHEMA_VERSION,
        )
        if record["consumer"] != ConversationSourceConsumer.BEHAVIOR_SESSION.value:
            raise ConversationSourceError("behavior session receipt has the wrong consumer")
        raw_turns = record["turns"]
        if not isinstance(raw_turns, list):
            raise ConversationSourceError("behavior session receipt turns must be a list")
        return cls(
            output_id=record["output_id"],
            source_id=record["source_id"],
            source_payload_digest=record["source_payload_digest"],
            processor_fingerprint=record["processor_fingerprint"],
            turns=tuple(BehaviorSessionTurnRecord.from_dict(item) for item in raw_turns),
            output_record_digest=record["output_record_digest"],
        )


def _record(
    output_id: str,
    source_id: str,
    source_payload_digest: str,
    processor_fingerprint: str,
    turns: tuple[BehaviorSessionTurnRecord, ...],
) -> dict[str, Any]:
    return canonicalize(
        {
            "schema_version": BEHAVIOR_SESSION_OUTPUT_SCHEMA_VERSION,
            "output_id": output_id,
            "source_id": source_id,
            "source_payload_digest": source_payload_digest,
            "consumer": ConversationSourceConsumer.BEHAVIOR_SESSION.value,
            "processor_fingerprint": processor_fingerprint,
            "turns": [item.to_dict() for item in turns],
        }
    )


__all__ = [
    "BEHAVIOR_SESSION_OUTPUT_KIND",
    "BEHAVIOR_SESSION_OUTPUT_SCHEMA_VERSION",
    "BehaviorSessionOutput",
    "BehaviorSessionTurnRecord",
]
