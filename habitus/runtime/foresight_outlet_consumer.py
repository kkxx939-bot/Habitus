"""会话源 → 出口 → 会话 lane（方案 32 第 4.2 条）：同一份会话先过提醒出口（在转写里认出 agent 的那次提问、写账本「说了」与固定选项
的回话），再进会话 lane——这样他作答的那一轮进桥时，这条提醒已经是"开着的"。

是一座桥，住在组合根：会话 lane 消费者（``runtime/session_lane.py``）照旧不认识预测层，出口也不认识会话源——这里把会话源信封里的
消息翻成出口要的形状、再把信封原样交给里面那个消费者。出口坏了不许拦住融合：认不出、写不进只记观测，这一份照常进 lane。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from habitus.conversation.source import (
    ConversationConsumerExecutionLease,
    ConversationConsumerRunResult,
    ConversationSourceEnvelope,
)
from habitus.foresight import ForesightError
from habitus.foundation.integrity import canonical_json
from habitus.foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer
from habitus.pre.conversation import ConversationAdapterRegistry
from habitus.pre.conversation.messages import ConversationMessage, ConversationToolResultStatus
from habitus.runtime.foresight_outlet import HostMessage, ReminderOutlet
from habitus.runtime.session_lane import BehaviorSessionConsumer


class OutletObservingConsumer:
    """包在会话 lane 消费者外面的一层：先 ``outlet.observe``，再交给里面那个。交付契约认的几个字段原样转手。"""

    def __init__(
        self,
        inner: BehaviorSessionConsumer,
        outlet: ReminderOutlet,
        *,
        observer: Observer | None = None,
        host_of: Callable[[str], str | None] | None = None,
    ) -> None:
        if not isinstance(inner, BehaviorSessionConsumer):
            raise TypeError("inner must be a BehaviorSessionConsumer")
        if not isinstance(outlet, ReminderOutlet):
            raise TypeError("outlet must be a ReminderOutlet")
        self.inner = inner
        self.outlet = outlet
        #: 会话协议 → 宿主名（出口按宿主挑提问工具）；由协议适配器声明，组合根把注册表的查法传进来
        self.host_of = host_of if host_of is not None else ConversationAdapterRegistry.with_builtins().host
        self.observer: Observer = observer if observer is not None else NullObserver()
        self.consumer = inner.consumer
        self.ordered_within_conversation = inner.ordered_within_conversation
        self.skips_released_sources = inner.skips_released_sources
        self.processor_fingerprint = inner.processor_fingerprint
        self.output_store = inner.output_store
        self.store = inner.store

    async def execute(
        self, envelope: ConversationSourceEnvelope, lease: ConversationConsumerExecutionLease
    ) -> ConversationConsumerRunResult:
        await self._observe(envelope)
        return await self.inner.execute(envelope, lease)

    async def abandon(
        self, envelope: ConversationSourceEnvelope, lease: ConversationConsumerExecutionLease, *, reason: str
    ) -> ConversationConsumerRunResult:
        """重试账放弃了这份源：提问若在转写里照样认（说出口是事实，与 lane 读不读得懂无关），再让里面那个放弃。"""

        await self._observe(envelope)
        return await self.inner.abandon(envelope, lease, reason=reason)

    async def _observe(self, envelope: ConversationSourceEnvelope) -> None:
        host = self.host_of(envelope.protocol)
        if host is None:
            return  # 通用 SDK 协议的会话不是从哪个宿主来取过提醒的，里面也没有我们认的提问工具
        messages = tuple(host_message(item) for item in envelope.batch.messages)
        try:
            await asyncio.to_thread(self.outlet.observe, envelope.conversation_id, messages, host=host)
        except (ForesightError, OSError) as exc:
            try:
                self.observer.record(
                    ObservationEvent(
                        category="foresight",
                        operation="outlet_observe",
                        status=ObservationStatus.FAILURE,
                        duration_seconds=0.0,
                        attributes={"error": type(exc).__name__},
                    )
                )
            except Exception:  # noqa: BLE001 - 观测不许影响融合
                pass


def host_message(message: ConversationMessage) -> HostMessage:
    """会话源的一条消息 → 出口读的形状（正文不是文字的按规范 JSON 给，工具参数本来就是 JSON）。"""

    return HostMessage(
        sequence=message.sequence,
        occurred_at=message.occurred_at,
        role=message.role.value,
        content=message.content if isinstance(message.content, str) else canonical_json(message.content),
        tool_name=message.tool_name,
        tool_call_id=message.tool_call_id,
        failed=message.tool_status is ConversationToolResultStatus.ERROR,
    )


__all__ = ["OutletObservingConsumer", "host_message"]
