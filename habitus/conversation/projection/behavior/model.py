"""Behavior Projection 的不可变条目、批次与三层身份。

投影输出使用三个各司其职的 SHA-256 身份，读代码时必须先分清它们：

- ``projection_item_id`` 是**单条信号**的身份，覆盖来源消息身份、来源消息摘要、
  发生时间、投影类型和 payload。同一条来源消息重放必得同一编号。
- ``projection_id`` 是**本批内容**的身份，覆盖来源、Projector 版本和全部条目。
  它回答"算出来的这批东西是不是同一份"。
- ``output_id`` 是**幂等键**，由来源身份、来源载荷摘要、Consumer、
  ``processor_fingerprint`` 和 Output Schema 版本派生，**刻意不含条目内容**。
  它回答"这个来源在这个处理器版本下该不该再算一次"：同一来源同一处理器
  只会存在一个输出文件，重算撞车时按内容摘要比对后直接复用既有文件。

因为 ``output_id`` 不含条目，映射规则或 payload 语义变化必须体现在
``processor_fingerprint`` 或 Projector 版本上，否则新语义会复用旧身份。
Projector 侧用同一张映射表同时驱动分派和指纹来保证这一点。

## Outbox 允许混版，下游必须按 ``projector_version`` 分治

一个 (Source, Consumer) 永远只有一个 Output 文件：交付层在 Outcome 已存在时直接
短路返回，Inspector 也把"同一来源出现多个 Output"判为损坏。因此提升 Projector
版本**不会**重算已完成的来源，也没有重投影入口——重投影必然产生第二个文件，直接
撞上该损坏判定。版本提升的语义只能是"从此刻起新来源用新语义，旧来源保留旧语义"。

于是 Outbox 会长期同时存在多个 ``projector_version`` 的批次，这是被接受的设计而
不是缺陷。**消费方必须按 ``projector_version`` 分治，不得把不同版本的批次混在
一次聚合里**：混版会让"某个字段有没有值"这类差异来自投影版本而不是真实行为，
下游一旦据此学习就是纯粹的伪信号。预测侧已经执行同一纪律，见
``prediction/learning/learner.py`` 中"一轮学习只能用一个 projection version"。

每条 Outcome 同时记录了它实际采用的 ``processor_fingerprint``（升级窗口里可能是
旧版本的），因此"这批数据到底用哪个版本算出来的"始终可审计。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from habitus.conversation.source.model import (
    ConversationSourceEnvelope,
    ConversationSourceError,
    require_record,
    require_sha256,
)
from habitus.conversation.source.receipt import (
    ConversationSourceConsumer,
    conversation_consumer_output_id,
)
from habitus.foundation.integrity import canonical_digest, canonicalize, immutable_snapshot
from habitus.pre.conversation import ConversationMessage
from habitus.pre.conversation.messages.model import conversation_datetime

BEHAVIOR_PROJECTION_OUTPUT_SCHEMA_VERSION = "conversation_behavior_projection_output_v2"
BEHAVIOR_PROJECTION_OUTPUT_KIND = "conversation_behavior_projection"
_PROJECTION_ID_SCHEMA = "conversation_behavior_projection_identity_v1"
_ITEM_SCHEMA = "conversation_behavior_projection_item_v1"


class ConversationBehaviorProjectionKind(str, Enum):
    USER_CONVERSATION_INPUT = "user_conversation_input"
    AGENT_TOOL_CALL = "agent_tool_call"
    TOOL_EXECUTION_RESULT = "tool_execution_result"


@dataclass(frozen=True)
class ConversationBehaviorProjectionItem:
    # TODO(CONV-PROJECTION-001) 缺口 B：这里缺 sequence、批次缺 conversation_id /
    # protocol / after_turn，跨批次因此既无法归属会话也无法定序；完整说明见
    # projector 模块 docstring。
    projection_item_id: str
    source_id: str
    source_message_id: str
    source_message_digest: str
    occurred_at: datetime
    projection_kind: ConversationBehaviorProjectionKind
    payload: Any

    def __post_init__(self) -> None:
        require_sha256(self.projection_item_id, "projection_item_id")
        require_sha256(self.source_id, "projection item source_id")
        if not isinstance(self.source_message_id, str) or not self.source_message_id:
            raise ConversationSourceError("source_message_id must be non-empty text")
        require_sha256(self.source_message_digest, "source_message_digest")
        object.__setattr__(self, "occurred_at", conversation_datetime(self.occurred_at, "occurred_at"))
        try:
            object.__setattr__(self, "projection_kind", ConversationBehaviorProjectionKind(self.projection_kind))
        except ValueError as exc:
            raise ConversationSourceError("projection_kind is invalid") from exc
        object.__setattr__(self, "payload", immutable_snapshot(canonicalize(self.payload)))
        # 身份自校验对解码路径是必需的；create() 路径重算一次是刻意的代价，
        # 换取"任何来路的条目都不可能带着错误编号存在"这一不变量。
        if self.projection_item_id != canonical_digest(self._identity_payload()):
            raise ConversationSourceError("projection_item_id does not match item content")

    @classmethod
    def create(
        cls,
        *,
        source_id: str,
        message: ConversationMessage,
        projection_kind: ConversationBehaviorProjectionKind,
        payload: Any,
    ) -> ConversationBehaviorProjectionItem:
        require_sha256(source_id, "source_id")
        source_message_digest = canonical_digest(message.to_dict())
        identity = {
            "schema_version": _ITEM_SCHEMA,
            "source_id": source_id,
            "source_message_id": message.message_id,
            "source_message_digest": source_message_digest,
            "occurred_at": message.occurred_at,
            "projection_kind": projection_kind.value,
            "payload": payload,
        }
        return cls(
            projection_item_id=canonical_digest(identity),
            source_id=source_id,
            source_message_id=message.message_id,
            source_message_digest=source_message_digest,
            occurred_at=message.occurred_at,
            projection_kind=projection_kind,
            payload=payload,
        )

    def _identity_payload(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema_version": _ITEM_SCHEMA,
                "source_id": self.source_id,
                "source_message_id": self.source_message_id,
                "source_message_digest": self.source_message_digest,
                "occurred_at": self.occurred_at,
                "projection_kind": self.projection_kind.value,
                "payload": canonicalize(self.payload),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema_version": _ITEM_SCHEMA,
                "projection_item_id": self.projection_item_id,
                "source_id": self.source_id,
                "source_message_id": self.source_message_id,
                "source_message_digest": self.source_message_digest,
                "occurred_at": self.occurred_at,
                "projection_kind": self.projection_kind.value,
                "payload": canonicalize(self.payload),
            }
        )

    @classmethod
    def from_dict(cls, value: object, *, source_id: str) -> ConversationBehaviorProjectionItem:
        record = require_record(
            value,
            expected={
                "schema_version",
                "projection_item_id",
                "source_id",
                "source_message_id",
                "source_message_digest",
                "occurred_at",
                "projection_kind",
                "payload",
            },
            label="projection item",
            schema_version=_ITEM_SCHEMA,
        )
        try:
            kind = ConversationBehaviorProjectionKind(record["projection_kind"])
        except (TypeError, ValueError) as exc:
            raise ConversationSourceError("projection item kind is invalid") from exc
        item = cls(
            projection_item_id=record["projection_item_id"],
            source_id=record["source_id"],
            source_message_id=record["source_message_id"],
            source_message_digest=record["source_message_digest"],
            occurred_at=conversation_datetime(record["occurred_at"], "occurred_at"),
            projection_kind=kind,
            payload=record["payload"],
        )
        if item.source_id != source_id:
            raise ConversationSourceError("projection item belongs to another source")
        return item


@dataclass(frozen=True)
class ConversationBehaviorProjectionBatch:
    output_id: str
    projection_id: str
    source_id: str
    source_payload_digest: str
    consumer: ConversationSourceConsumer
    processor_fingerprint: str
    projector_version: str
    items: tuple[ConversationBehaviorProjectionItem, ...]
    recorded_at: datetime
    output_record_digest: str

    def __post_init__(self) -> None:
        for label, value in (
            ("projection output_id", self.output_id),
            ("projection_id", self.projection_id),
            ("projection source_id", self.source_id),
            ("projection source_payload_digest", self.source_payload_digest),
            ("projection processor_fingerprint", self.processor_fingerprint),
            ("projection output_record_digest", self.output_record_digest),
        ):
            require_sha256(value, label)
        object.__setattr__(self, "consumer", ConversationSourceConsumer(self.consumer))
        if self.consumer is not ConversationSourceConsumer.BEHAVIOR_PROJECTION:
            raise ConversationSourceError("projection output has the wrong consumer")
        if not isinstance(self.projector_version, str) or not self.projector_version:
            raise ConversationSourceError("projector_version must be non-empty text")
        if not isinstance(self.items, tuple) or not self.items:
            raise ConversationSourceError("projection output must contain items")
        if any(not isinstance(item, ConversationBehaviorProjectionItem) for item in self.items):
            raise TypeError("projection items must be ConversationBehaviorProjectionItem values")
        if any(item.source_id != self.source_id for item in self.items):
            raise ConversationSourceError("projection output contains an item from another source")
        if len({item.projection_item_id for item in self.items}) != len(self.items):
            raise ConversationSourceError("projection item IDs must be unique")
        object.__setattr__(self, "recorded_at", conversation_datetime(self.recorded_at, "recorded_at"))
        if self.output_id != conversation_consumer_output_id(
            source_id=self.source_id,
            source_payload_digest=self.source_payload_digest,
            consumer=self.consumer,
            processor_fingerprint=self.processor_fingerprint,
            output_schema_version=BEHAVIOR_PROJECTION_OUTPUT_SCHEMA_VERSION,
        ):
            raise ConversationSourceError("projection output_id does not match output identity")
        if self.projection_id != canonical_digest(self._projection_identity_payload()):
            raise ConversationSourceError("projection_id does not match projection content")
        if self.output_record_digest != canonical_digest(self._record_without_digest()):
            raise ConversationSourceError("projection output_record_digest does not match output record")

    @classmethod
    def create(
        cls,
        *,
        source: ConversationSourceEnvelope,
        processor_fingerprint: str,
        projector_version: str,
        items: tuple[ConversationBehaviorProjectionItem, ...],
        recorded_at: datetime,
    ) -> ConversationBehaviorProjectionBatch:
        output_id = conversation_consumer_output_id(
            source_id=source.source_id,
            source_payload_digest=source.source_payload_digest,
            consumer=ConversationSourceConsumer.BEHAVIOR_PROJECTION,
            processor_fingerprint=processor_fingerprint,
            output_schema_version=BEHAVIOR_PROJECTION_OUTPUT_SCHEMA_VERSION,
        )
        identity = canonicalize(
            {
                "schema_version": _PROJECTION_ID_SCHEMA,
                "source_id": source.source_id,
                "source_payload_digest": source.source_payload_digest,
                "projector_version": projector_version,
                "items": [item.to_dict() for item in items],
            }
        )
        projection_id = canonical_digest(identity)
        record = canonicalize(
            {
                "schema_version": BEHAVIOR_PROJECTION_OUTPUT_SCHEMA_VERSION,
                "output_id": output_id,
                "projection_id": projection_id,
                "source_id": source.source_id,
                "source_payload_digest": source.source_payload_digest,
                "consumer": ConversationSourceConsumer.BEHAVIOR_PROJECTION.value,
                "processor_fingerprint": processor_fingerprint,
                "projector_version": projector_version,
                "items": [item.to_dict() for item in items],
                "recorded_at": conversation_datetime(recorded_at, "recorded_at"),
            }
        )
        return cls(
            output_id=output_id,
            projection_id=projection_id,
            source_id=source.source_id,
            source_payload_digest=source.source_payload_digest,
            consumer=ConversationSourceConsumer.BEHAVIOR_PROJECTION,
            processor_fingerprint=processor_fingerprint,
            projector_version=projector_version,
            items=items,
            recorded_at=recorded_at,
            output_record_digest=canonical_digest(record),
        )

    def _projection_identity_payload(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema_version": _PROJECTION_ID_SCHEMA,
                "source_id": self.source_id,
                "source_payload_digest": self.source_payload_digest,
                "projector_version": self.projector_version,
                "items": [item.to_dict() for item in self.items],
            }
        )

    def _record_without_digest(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema_version": BEHAVIOR_PROJECTION_OUTPUT_SCHEMA_VERSION,
                "output_id": self.output_id,
                "projection_id": self.projection_id,
                "source_id": self.source_id,
                "source_payload_digest": self.source_payload_digest,
                "consumer": self.consumer.value,
                "processor_fingerprint": self.processor_fingerprint,
                "projector_version": self.projector_version,
                "items": [item.to_dict() for item in self.items],
                "recorded_at": self.recorded_at,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return canonicalize({**self._record_without_digest(), "output_record_digest": self.output_record_digest})

    @classmethod
    def from_dict(cls, value: object) -> ConversationBehaviorProjectionBatch:
        record = require_record(
            value,
            expected={
                "schema_version",
                "output_id",
                "projection_id",
                "source_id",
                "source_payload_digest",
                "consumer",
                "processor_fingerprint",
                "projector_version",
                "items",
                "recorded_at",
                "output_record_digest",
            },
            label="projection output",
            schema_version=BEHAVIOR_PROJECTION_OUTPUT_SCHEMA_VERSION,
        )
        raw_items = record["items"]
        if not isinstance(raw_items, list):
            raise ConversationSourceError("projection output items must be a list")
        try:
            consumer = ConversationSourceConsumer(record["consumer"])
        except (TypeError, ValueError) as exc:
            raise ConversationSourceError("projection output consumer is invalid") from exc
        return cls(
            output_id=record["output_id"],
            projection_id=record["projection_id"],
            source_id=record["source_id"],
            source_payload_digest=record["source_payload_digest"],
            consumer=consumer,
            processor_fingerprint=record["processor_fingerprint"],
            projector_version=record["projector_version"],
            items=tuple(
                ConversationBehaviorProjectionItem.from_dict(item, source_id=record["source_id"])
                for item in raw_items
            ),
            recorded_at=conversation_datetime(record["recorded_at"], "recorded_at"),
            output_record_digest=record["output_record_digest"],
        )


__all__ = [
    "BEHAVIOR_PROJECTION_OUTPUT_KIND",
    "BEHAVIOR_PROJECTION_OUTPUT_SCHEMA_VERSION",
    "ConversationBehaviorProjectionBatch",
    "ConversationBehaviorProjectionItem",
    "ConversationBehaviorProjectionKind",
]
