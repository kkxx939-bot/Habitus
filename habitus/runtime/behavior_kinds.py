"""一支行为的词表部件：词表状态与选词器（``runtime.behavior`` 装配一支时调；分出来是因为那个文件到了大小上限）。

词表的门槛从唯一的 Config 边界进来；模型只经 ``KindModelCaller`` 一个触点。词表状态的写入整段持行为根锁库上的租约，时长与行为树
写锁同一个配置（``behavior.write_lock_ttl_seconds``）。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from habitus.behavior.kinds.calls import KindModelCaller
from habitus.behavior.kinds.ids import Lane
from habitus.behavior.kinds.pick.picker import WordPicker
from habitus.behavior.kinds.store import BehaviorKindStore
from habitus.behavior.kinds.words import initial_words
from habitus.infrastructure.store.contracts.lock import LockStore
from habitus.model_client import StructuredChatClient

if TYPE_CHECKING:
    from habitus.runtime.behavior import BehaviorDomainConfigs


def vocabulary_parts(
    chat: StructuredChatClient, lane: Lane, root: Path, domain: BehaviorDomainConfigs, lock_store: LockStore
) -> tuple[BehaviorKindStore, WordPicker]:
    """这一支的词表状态（``root`` 是这一支在树外的根 ``behavior/<lane>``）与选词器。选词器答复结构不合格时带着错误重问 ``pick_retries`` 次（另起一个结构化
    客户端，只换重问次数）；两支的选词器都拿这一支的词表状态（每一条现读、写新节点）。"""

    kinds = domain.kinds
    store = BehaviorKindStore(
        root, lane=lane, lock_store=lock_store, lock_ttl_seconds=domain.write.lock_ttl_seconds, config=kinds
    )
    picking = StructuredChatClient(
        chat.client, allow_json_repair=chat.allow_json_repair, validation_retries=kinds.pick_retries
    )
    picker = WordPicker(KindModelCaller(picking, config=kinds), initial_words(lane), state=store)
    return store, picker


__all__ = ["vocabulary_parts"]
