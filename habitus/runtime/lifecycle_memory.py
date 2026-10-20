"""memory 侧挂在 LifecycleWorker 上的两个参与者：会话链维护、L2 记忆冷热维护。

两样的逻辑一行不改（memory 的生命周期本轮待定），只是从 Worker 里搬出来、包成参与者：
- 会话链：按耐久游标有界轮转全部 Conversation，逐个 ``maintain_once``，单会话失败隔离；游标只以日期与
  conversation_id 写入 ``workflow/lifecycle/state.json``，写入受本轮全局 lease 的 fence 保护。
- L2：调一次 ``MemoryLifecycleManager.maintain``。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from habitus.config import ConversationLifecycleConfig
from habitus.foundation.integrity import canonical_json
from habitus.infrastructure.lifecycle import LifecycleContext, ParticipantReport
from habitus.infrastructure.store.contracts import LockStore
from habitus.infrastructure.store.filesystem import atomic_replace_bytes, read_regular_bytes
from habitus.memory.compaction import MemoryLifecycleMaintenanceResult, MemoryLifecycleManager
from habitus.memory.conversation import ConversationAddress, ConversationMessageJournal
from habitus.memory.workflow import ConversationLifecycleManager
from habitus.runtime.lifecycle import LifecycleWorkerStateError

_LIFECYCLE_CURSOR_SCHEMA = "conversation_lifecycle_cursor_v1"
_MESSAGE_LIMIT = 1_000


def conversation_lease_key(journal: ConversationMessageJournal) -> str:
    """全局维护 lease 的键：由 Conversation 根派生，同一根只有一个维护者。"""

    root_digest = hashlib.sha256(str(journal.layout.root).encode("utf-8")).hexdigest()[:24]
    return f"runtime:lifecycle:{root_digest}"


@dataclass(frozen=True)
class LifecycleMaintenanceFailure:
    """单个 Conversation 维护失败的有界可观察信息。"""

    address: ConversationAddress
    error_type: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.address, ConversationAddress):
            raise TypeError("address must be ConversationAddress")
        for name in ("error_type", "message"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TypeError(f"{name} must be non-empty text")
        if len(self.message) > _MESSAGE_LIMIT:
            raise ValueError("maintenance failure message exceeds its bound")


@dataclass(frozen=True)
class ConversationChainResult:
    """一轮会话链维护的选取、成功和失败。"""

    selected_addresses: tuple[ConversationAddress, ...]
    maintained_addresses: tuple[ConversationAddress, ...]
    failures: tuple[LifecycleMaintenanceFailure, ...]

    def __post_init__(self) -> None:
        for name in ("selected_addresses", "maintained_addresses"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(value, ConversationAddress) for value in values):
                raise TypeError(f"{name} must contain ConversationAddress values")
        if not isinstance(self.failures, tuple) or any(
            not isinstance(value, LifecycleMaintenanceFailure) for value in self.failures
        ):
            raise TypeError("failures must contain LifecycleMaintenanceFailure values")
        selected = set(self.selected_addresses)
        if any(value not in selected for value in self.maintained_addresses):
            raise ValueError("maintained addresses must come from the selected batch")
        if any(value.address not in selected for value in self.failures):
            raise ValueError("failed addresses must come from the selected batch")
        if set(self.maintained_addresses) & {value.address for value in self.failures}:
            raise ValueError("one address cannot both succeed and fail in one cycle")


class LifecycleCursorStore:
    """在 Workflow 根内保存跨进程轮转所需的最小耐久游标。"""

    def __init__(self, manager: ConversationLifecycleManager) -> None:
        self.root = manager.jobs.root
        self.path = self.root / "lifecycle" / "state.json"
        self.max_bytes = manager.jobs.config.max_file_bytes

    def read(self) -> tuple[date, str] | None:
        try:
            encoded = read_regular_bytes(self.path, artifact_root=self.root, max_bytes=self.max_bytes)
        except FileNotFoundError:
            return None
        try:
            raw = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise LifecycleWorkerStateError("lifecycle cursor state is invalid JSON") from exc
        expected = {"schema", "started_on", "conversation_id"}
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise LifecycleWorkerStateError("lifecycle cursor state has an invalid shape")
        if raw["schema"] != _LIFECYCLE_CURSOR_SCHEMA:
            raise LifecycleWorkerStateError("lifecycle cursor state has an unsupported schema")
        started_on_value = raw["started_on"]
        conversation_id = raw["conversation_id"]
        if not isinstance(started_on_value, str) or not isinstance(conversation_id, str):
            raise LifecycleWorkerStateError("lifecycle cursor state contains invalid field types")
        try:
            address = ConversationAddress(
                conversation_id=conversation_id, started_on=date.fromisoformat(started_on_value)
            )
        except ValueError as exc:
            raise LifecycleWorkerStateError("lifecycle cursor state is invalid") from exc
        if encoded != self._encode(address):
            raise LifecycleWorkerStateError("lifecycle cursor state is not canonically encoded")
        return address.started_on, address.conversation_id

    def write(self, address: ConversationAddress) -> None:
        if not isinstance(address, ConversationAddress):
            raise TypeError("lifecycle cursor address must be ConversationAddress")
        encoded = self._encode(address)
        if len(encoded) > self.max_bytes:
            raise LifecycleWorkerStateError("lifecycle cursor state exceeds its configured bound")
        atomic_replace_bytes(self.path, encoded, artifact_root=self.root)

    @staticmethod
    def _encode(address: ConversationAddress) -> bytes:
        payload = {
            "schema": _LIFECYCLE_CURSOR_SCHEMA,
            "started_on": address.started_on.isoformat(),
            "conversation_id": address.conversation_id,
        }
        return (canonical_json(payload) + "\n").encode("utf-8")


class ConversationChainParticipant:
    """按耐久游标有界轮转全部 Conversation，逐个调用统一的 Conversation 生命周期管理器。"""

    name = "conversation-chain"

    def __init__(
        self,
        manager: ConversationLifecycleManager,
        config: ConversationLifecycleConfig,
        *,
        lock_store: LockStore,
        lease_ttl_seconds: int,
    ) -> None:
        if not isinstance(manager, ConversationLifecycleManager):
            raise TypeError("manager must be ConversationLifecycleManager")
        if not isinstance(config, ConversationLifecycleConfig):
            raise TypeError("config must be ConversationLifecycleConfig")
        self.manager = manager
        self.journal: ConversationMessageJournal = manager.journal
        self.config = config
        self.lock_store = lock_store
        self.lease_ttl_seconds = lease_ttl_seconds
        self.cursor_store = LifecycleCursorStore(manager)
        self.last_result: ConversationChainResult | None = None

    async def maintain(self, ctx: LifecycleContext) -> ParticipantReport:
        addresses = await asyncio.to_thread(self.journal.list_addresses)
        cursor_key = await asyncio.to_thread(self.cursor_store.read)
        selected = self._select_batch(addresses, cursor_key, limit=self.config.max_conversations_per_cycle)
        maintained: list[ConversationAddress] = []
        failures: list[LifecycleMaintenanceFailure] = []
        for address in selected:
            await asyncio.to_thread(ctx.assert_owned)
            try:
                await self.manager.maintain_once(address)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failures.append(_failure(address, exc))
            else:
                maintained.append(address)
            finally:
                await asyncio.to_thread(self._write_cursor_fenced, ctx, address)
        await asyncio.to_thread(ctx.assert_owned)
        result = ConversationChainResult(selected, tuple(maintained), tuple(failures))
        self.last_result = result
        return ParticipantReport(
            released=len(maintained),
            signals=tuple(f"{item.address.conversation_id}: {item.error_type}" for item in failures),
        )

    def _write_cursor_fenced(self, ctx: LifecycleContext, address: ConversationAddress) -> None:
        """只允许仍持有本轮全局 lease 的维护者推进耐久游标。"""

        with self.lock_store.fenced((ctx.fence_token(),), ttl_seconds=self.lease_ttl_seconds):
            self.cursor_store.write(address)

    @staticmethod
    def _select_batch(
        addresses: tuple[ConversationAddress, ...],
        cursor_key: tuple[date, str] | None,
        *,
        limit: int,
    ) -> tuple[ConversationAddress, ...]:
        if not addresses:
            return ()
        keys = tuple((address.started_on, address.conversation_id) for address in addresses)
        start = 0 if cursor_key is None else bisect_right(keys, cursor_key)
        ordered = (*addresses[start:], *addresses[:start])
        return tuple(ordered[: min(limit, len(addresses))])


class MemoryL2Participant:
    """L2 记忆的冷热、压缩与退休维护。"""

    name = "memory-l2"

    def __init__(self, manager: MemoryLifecycleManager) -> None:
        if not isinstance(manager, MemoryLifecycleManager):
            raise TypeError("manager must be MemoryLifecycleManager")
        self.manager = manager
        self.last_result: MemoryLifecycleMaintenanceResult | None = None

    async def maintain(self, ctx: LifecycleContext) -> ParticipantReport:
        await asyncio.to_thread(ctx.assert_owned)
        result = await self.manager.maintain()
        await asyncio.to_thread(ctx.assert_owned)
        self.last_result = result
        return ParticipantReport(
            released=len(result.compacted) + len(result.retired),
            signals=tuple(f"{item.uri}: {item.error_type}" for item in result.failures),
        )


def _failure(address: ConversationAddress, error: Exception) -> LifecycleMaintenanceFailure:
    message = str(error).strip() or error.__class__.__name__
    return LifecycleMaintenanceFailure(
        address=address, error_type=error.__class__.__name__, message=message[:_MESSAGE_LIMIT]
    )


__all__ = [
    "ConversationChainParticipant",
    "ConversationChainResult",
    "LifecycleCursorStore",
    "LifecycleMaintenanceFailure",
    "MemoryL2Participant",
    "conversation_lease_key",
]
