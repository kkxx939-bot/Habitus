"""组合根各段共用的那几样：锁、观测、结构化模型客户端与宿主适配器。只在 ``build_runtime`` 组装期间传递，不进 Runtime。"""

from __future__ import annotations

from dataclasses import dataclass

from habitus.foundation.observability import Observer
from habitus.infrastructure.observability import ManagedObservability
from habitus.infrastructure.store.contracts import PathLock
from habitus.model_client import StructuredChatClient
from habitus.pre.conversation import ConversationAdapterRegistry


@dataclass(frozen=True)
class AssemblyShared:
    path_lock: PathLock
    observer: Observer
    managed_observability: ManagedObservability
    structured_chat: StructuredChatClient
    #: 会话协议的适配器也声明各宿主的提问工具与宿主名：出口、会话 lane、出口那层消费者都从这一份取
    adapters: ConversationAdapterRegistry


__all__ = ["AssemblyShared"]
