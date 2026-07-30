"""Conversation live 消息追加与不可变 history 封存。"""

from __future__ import annotations

import json
import math
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any

from foundation.integrity import canonical_json
from infrastructure.store.contracts.path_lock import PathLock
from infrastructure.store.filesystem.durable_io import (
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_replace_bytes,
    atomic_temporary_destination,
    durable_unlink,
    read_regular_bytes,
)
from memory.conversation.layout import ConversationAddress, ConversationLayout
from pre.conversation.messages.model import (
    ConversationBatch,
    ConversationMessage,
    ConversationMessageRole,
    ConversationMessageSchemaError,
    ConversationSegment,
)


class ConversationJournalError(ValueError):
    """Conversation live 或 history 数据不完整、不合法或无法安全处理。"""


class ConversationWriteConflictError(ConversationJournalError):
    """已有会话事实与本次追加或封存请求冲突。"""


@dataclass(frozen=True)
class ConversationJournalConfig:
    """Conversation 文件容量、history 枚举与路径锁等待边界。"""

    max_file_bytes: int = 64 * 1024 * 1024
    max_history_files: int = 10_000
    max_conversation_tree_entries: int = 100_000
    lock_ttl_seconds: int = 30
    lock_wait_timeout_seconds: float = 5.0
    lock_retry_delay_seconds: float = 0.01

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("max_file_bytes", self.max_file_bytes, 256 * 1024 * 1024),
            ("max_history_files", self.max_history_files, 1_000_000),
            (
                "max_conversation_tree_entries",
                self.max_conversation_tree_entries,
                1_000_000,
            ),
            ("lock_ttl_seconds", self.lock_ttl_seconds, 3_600),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be between 1 and {maximum}")
        for float_name, float_value, float_maximum in (
            ("lock_wait_timeout_seconds", self.lock_wait_timeout_seconds, 60.0),
            ("lock_retry_delay_seconds", self.lock_retry_delay_seconds, 1.0),
        ):
            if (
                isinstance(float_value, bool)
                or not isinstance(float_value, int | float)
                or not math.isfinite(float(float_value))
                or not 0 < float(float_value) <= float_maximum
            ):
                raise ValueError(f"{float_name} must be greater than zero and at most {float_maximum:g}")


class ConversationAppendStatus(str, Enum):
    CREATED = "created"
    EXTENDED = "extended"
    UNCHANGED = "unchanged"


class ConversationSealStatus(str, Enum):
    CREATED = "created"
    UNCHANGED = "unchanged"


_JOURNAL_STATE_SCHEMA = "conversation_journal_state_v1"


@dataclass(frozen=True)
class ConversationJournalState:
    """在 History 物理删除后仍保持全局消息序号与释放边界的耐久游标。"""

    conversation_id: str
    archived_through_sequence: int | None
    released_through_sequence: int | None
    latest_segment_id: str | None
    latest_segment_digest: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.conversation_id, str) or not self.conversation_id:
            raise ConversationJournalError("journal state conversation_id must be non-empty text")
        for name in ("archived_through_sequence", "released_through_sequence"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ConversationJournalError(f"journal state {name} must be null or non-negative")
        if self.archived_through_sequence is None:
            if any(
                value is not None
                for value in (
                    self.released_through_sequence,
                    self.latest_segment_id,
                    self.latest_segment_digest,
                )
            ):
                raise ConversationJournalError("empty journal state cannot contain archive metadata")
            return
        if (
            self.released_through_sequence is not None
            and self.released_through_sequence > self.archived_through_sequence
        ):
            raise ConversationJournalError("released history cannot exceed the archive high-watermark")
        if not isinstance(self.latest_segment_id, str) or not self.latest_segment_id:
            raise ConversationJournalError("journal state requires latest_segment_id")
        _start, end = ConversationLayout.segment_range(self.latest_segment_id)
        if end != self.archived_through_sequence:
            raise ConversationJournalError("latest segment does not end at the archive high-watermark")
        if not _is_sha256(self.latest_segment_digest):
            raise ConversationJournalError("journal state latest_segment_digest must be lowercase SHA-256")

    @property
    def next_sequence(self) -> int:
        return 0 if self.archived_through_sequence is None else self.archived_through_sequence + 1

    @property
    def released_through(self) -> int:
        return -1 if self.released_through_sequence is None else self.released_through_sequence

    @classmethod
    def empty(cls, conversation_id: str) -> ConversationJournalState:
        return cls(conversation_id, None, None, None, None)


@dataclass(frozen=True)
class ConversationAppendResult:
    status: ConversationAppendStatus
    appended_count: int
    live: ConversationBatch | None
    next_sequence: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.appended_count, bool)
            or not isinstance(self.appended_count, int)
            or self.appended_count < 0
        ):
            raise ValueError("appended_count must be a non-negative integer")
        if (
            isinstance(self.next_sequence, bool)
            or not isinstance(self.next_sequence, int)
            or self.next_sequence < 0
        ):
            raise ValueError("next_sequence must be a non-negative integer")


@dataclass(frozen=True)
class ConversationSealResult:
    status: ConversationSealStatus
    segment: ConversationSegment
    live: ConversationBatch | None


@dataclass(frozen=True)
class _HistoryReference:
    segment_id: str
    start_sequence: int
    end_sequence: int
    path: Path


@dataclass(frozen=True)
class _ConversationSealPlan:
    """在发布前已经由 Conversation 租约保护并完整校验的封存计划。"""

    segment: ConversationSegment
    retained_messages: tuple[ConversationMessage, ...]
    encoded_history: bytes
    encoded_live: bytes


class ConversationMessageJournal:
    """按 Conversation 粒度串行化 live/history 文件操作。"""

    def __init__(
        self,
        root: str | Path,
        path_lock: PathLock,
        *,
        config: ConversationJournalConfig | None = None,
    ) -> None:
        if not isinstance(path_lock, PathLock):
            raise TypeError("path_lock must be a PathLock")
        if config is not None and not isinstance(config, ConversationJournalConfig):
            raise TypeError("config must be ConversationJournalConfig")
        self.layout = ConversationLayout(root)
        self.path_lock = path_lock
        self.config = config or ConversationJournalConfig()

    def list_addresses(self) -> tuple[ConversationAddress, ...]:
        """严格解析并有界枚举消息树中的全部 Conversation 地址。"""

        root = self.layout.messages_root()
        if root.is_symlink():
            raise ConversationJournalError("conversation messages root cannot be a symbolic link")
        if not root.exists():
            return ()
        if not root.is_dir():
            raise ConversationJournalError("conversation messages root must be a directory")

        entry_count = 0

        def directories(parent: Path, label: str) -> tuple[Path, ...]:
            nonlocal entry_count
            children: list[Path] = []
            for child in parent.iterdir():
                entry_count += 1
                if entry_count > self.config.max_conversation_tree_entries:
                    raise ConversationJournalError("conversation directory tree exceeds its enumeration bound")
                if child.is_symlink() or not child.is_dir():
                    raise ConversationJournalError(f"conversation {label} directory contains an unsupported entry")
                if child.name.startswith("."):
                    raise ConversationJournalError(f"conversation {label} directory contains a hidden entry")
                children.append(child)
            return tuple(sorted(children, key=lambda item: item.name))

        addresses: list[ConversationAddress] = []
        for year_directory in directories(root, "year"):
            year = self._calendar_component(year_directory.name, digits=4, label="year")
            for month_directory in directories(year_directory, "month"):
                month = self._calendar_component(
                    month_directory.name,
                    digits=2,
                    label="month",
                )
                for day_directory in directories(month_directory, "day"):
                    day = self._calendar_component(day_directory.name, digits=2, label="day")
                    try:
                        started_on = date(year, month, day)
                    except ValueError as exc:
                        raise ConversationJournalError(
                            "conversation directory contains an invalid calendar date"
                        ) from exc
                    for conversation_directory in directories(day_directory, "conversation"):
                        try:
                            addresses.append(
                                ConversationAddress(
                                    conversation_directory.name,
                                    started_on,
                                )
                            )
                        except ValueError as exc:
                            raise ConversationJournalError(
                                "conversation directory contains an invalid conversation_id"
                            ) from exc
        addresses.sort(key=lambda item: (item.started_on, item.conversation_id))
        return tuple(addresses)

    def append(
        self,
        address: ConversationAddress,
        batch: ConversationBatch,
    ) -> ConversationAppendResult:
        """幂等追加一个连续消息批次到 live.jsonl。"""

        self._require_batch(address, batch)
        with self.path_lock.acquire(
            self.layout.lock_key(address),
            ttl_seconds=self.config.lock_ttl_seconds,
            wait_timeout_seconds=self.config.lock_wait_timeout_seconds,
            retry_delay_seconds=self.config.lock_retry_delay_seconds,
        ) as guard:
            with guard.fenced():
                live_messages = list(self._read_live_messages(address))
                references = self._history_references(address)
                state = self._load_state(address, references)
                latest = self._latest_retained_segment(address, state, references)
                live_messages = self._recover_archived_prefix(address, latest, live_messages)
                self._require_tail_continuity(state, live_messages)
                expected_next = self._expected_next_sequence(state, live_messages)
                known = self._known_for_append(
                    address,
                    batch,
                    state,
                    references,
                    latest,
                    live_messages,
                    expected_next=expected_next,
                )

                known_by_sequence = {message.sequence: message for message in known}
                known_ids = {message.message_id: message.sequence for message in known}
                known_tool_calls = {
                    message.tool_call_id: (message.sequence, message.tool_name)
                    for message in known
                    if message.role is ConversationMessageRole.TOOL_CALL
                }
                known_tool_results = {
                    message.tool_call_id: message.sequence
                    for message in known
                    if message.role is ConversationMessageRole.TOOL_RESULT
                }
                unseen: list[ConversationMessage] = []
                for message in batch.messages:
                    if message.sequence < expected_next:
                        existing = known_by_sequence.get(message.sequence)
                        if existing is None:
                            raise ConversationWriteConflictError(
                                "append replay predates the retained conversation tail"
                            )
                        if existing.to_dict() != message.to_dict():
                            raise ConversationWriteConflictError(
                                "append replay conflicts with an existing message sequence"
                            )
                        continue
                    required_sequence = expected_next + len(unseen)
                    if message.sequence != required_sequence:
                        raise ConversationWriteConflictError("append would create a gap in the global message sequence")
                    existing_sequence = known_ids.get(message.message_id)
                    if existing_sequence is not None and existing_sequence != message.sequence:
                        raise ConversationWriteConflictError("message_id is already bound to another sequence")
                    if message.role is ConversationMessageRole.TOOL_CALL:
                        existing_call = known_tool_calls.get(message.tool_call_id)
                        if existing_call is not None and existing_call[0] != message.sequence:
                            raise ConversationWriteConflictError("tool_call_id is already bound to another sequence")
                        known_tool_calls[message.tool_call_id] = (
                            message.sequence,
                            message.tool_name,
                        )
                    elif message.role is ConversationMessageRole.TOOL_RESULT:
                        call = known_tool_calls.get(message.tool_call_id)
                        if call is None:
                            raise ConversationWriteConflictError("tool_result does not follow a known tool_call")
                        if call[1] != message.tool_name:
                            raise ConversationWriteConflictError("tool_result tool_name differs from its tool_call")
                        existing_result_sequence = known_tool_results.get(message.tool_call_id)
                        if existing_result_sequence is not None:
                            raise ConversationWriteConflictError("tool_call already has a terminal tool_result")
                        known_tool_results[message.tool_call_id] = message.sequence
                    unseen.append(message)

                if not unseen:
                    return ConversationAppendResult(
                        status=ConversationAppendStatus.UNCHANGED,
                        appended_count=0,
                        live=self._batch_or_none(address, live_messages),
                        next_sequence=expected_next,
                    )

                updated_live = tuple([*live_messages, *unseen])
                encoded = self._encode_messages(updated_live)
                self._require_write_bound(encoded)
                atomic_replace_bytes(
                    self.layout.live_path(address),
                    encoded,
                    artifact_root=self.layout.root,
                )
                status = ConversationAppendStatus.CREATED if not known else ConversationAppendStatus.EXTENDED
                return ConversationAppendResult(
                    status=status,
                    appended_count=len(unseen),
                    live=ConversationBatch(address.conversation_id, updated_live),
                    next_sequence=expected_next + len(unseen),
                )

    def read_live(self, address: ConversationAddress) -> ConversationBatch | None:
        """读取一致的 live 快照，并完成可证明安全的中断封存恢复。"""

        with self.path_lock.acquire(
            self.layout.lock_key(address),
            ttl_seconds=self.config.lock_ttl_seconds,
            wait_timeout_seconds=self.config.lock_wait_timeout_seconds,
            retry_delay_seconds=self.config.lock_retry_delay_seconds,
        ) as guard:
            with guard.fenced():
                live_messages = list(self._read_live_messages(address))
                references = self._history_references(address)
                state = self._load_state(address, references)
                latest = self._latest_retained_segment(address, state, references)
                live_messages = self._recover_archived_prefix(address, latest, live_messages)
                self._require_tail_continuity(state, live_messages)
                return self._batch_or_none(address, live_messages)

    def next_sequence(self, address: ConversationAddress) -> int:
        """在 Conversation 租约内返回下一条消息的全局序号。"""

        with self.path_lock.acquire(
            self.layout.lock_key(address),
            ttl_seconds=self.config.lock_ttl_seconds,
            wait_timeout_seconds=self.config.lock_wait_timeout_seconds,
            retry_delay_seconds=self.config.lock_retry_delay_seconds,
        ) as guard:
            with guard.fenced():
                live_messages = list(self._read_live_messages(address))
                references = self._history_references(address)
                state = self._load_state(address, references)
                latest = self._latest_retained_segment(address, state, references)
                live_messages = self._recover_archived_prefix(address, latest, live_messages)
                self._require_tail_continuity(state, live_messages)
                return self._expected_next_sequence(state, live_messages)

    def seal(
        self,
        address: ConversationAddress,
        *,
        through_sequence: int,
        before_history_publish: Callable[[ConversationSegment], None] | None = None,
    ) -> ConversationSealResult:
        """先准备下游 outbox，再耐久创建 history 并移除 live 前缀。"""

        if isinstance(through_sequence, bool) or not isinstance(through_sequence, int):
            raise TypeError("through_sequence must be an integer")
        if through_sequence < 0:
            raise ValueError("through_sequence must be non-negative")
        if before_history_publish is not None and not callable(before_history_publish):
            raise TypeError("before_history_publish must be callable or None")
        with self.path_lock.acquire(
            self.layout.lock_key(address),
            ttl_seconds=self.config.lock_ttl_seconds,
            wait_timeout_seconds=self.config.lock_wait_timeout_seconds,
            retry_delay_seconds=self.config.lock_retry_delay_seconds,
        ) as guard:
            with guard.fenced():
                prepared = self._prepare_seal(address, through_sequence)
                if isinstance(prepared, ConversationSealResult):
                    return prepared
                if before_history_publish is None:
                    return self._publish_seal(address, prepared)

            # 保留 Conversation 租约，但退出 SQLite fencing 事务，避免下游 outbox 锁自嵌套。
            before_history_publish(prepared.segment)
            guard.checkpoint()
            with guard.fenced():
                current = self._prepare_seal(address, through_sequence)
                if isinstance(current, ConversationSealResult):
                    if not self._same_segment(current.segment, prepared.segment):
                        raise ConversationWriteConflictError(
                            "conversation seal source changed while preparing its outbox"
                        )
                    return current
                if not self._same_segment(current.segment, prepared.segment):
                    raise ConversationWriteConflictError("conversation seal source changed while preparing its outbox")
                return self._publish_seal(address, current)

    def _prepare_seal(
        self,
        address: ConversationAddress,
        through_sequence: int,
    ) -> _ConversationSealPlan | ConversationSealResult:
        """在 fencing 区内读取完整状态并形成不含下游副作用的封存计划。"""

        live_messages = list(self._read_live_messages(address))
        references = self._history_references(address)
        state = self._load_state(address, references)
        latest = self._latest_retained_segment(address, state, references)
        live_messages = self._recover_archived_prefix(address, latest, live_messages)
        self._require_tail_continuity(state, live_messages)

        if not live_messages or through_sequence < live_messages[0].sequence:
            if latest is not None and latest.end_sequence == through_sequence:
                return ConversationSealResult(
                    status=ConversationSealStatus.UNCHANGED,
                    segment=latest,
                    live=self._batch_or_none(address, live_messages),
                )
            raise ConversationWriteConflictError("seal boundary does not select an unarchived live prefix")
        if through_sequence > live_messages[-1].sequence:
            raise ConversationWriteConflictError("seal boundary exceeds the live message range")

        split_index = through_sequence - live_messages[0].sequence + 1
        archived_messages = tuple(live_messages[:split_index])
        retained_messages = tuple(live_messages[split_index:])
        if not archived_messages or archived_messages[-1].sequence != through_sequence:
            raise ConversationWriteConflictError("seal boundary is not present in live messages")
        if archived_messages[0].sequence != state.next_sequence:
            raise ConversationWriteConflictError("sealed segment would not continue the archive high-watermark")

        segment_id = self.layout.segment_id(
            archived_messages[0].sequence,
            archived_messages[-1].sequence,
        )
        segment = ConversationSegment(
            conversation_id=address.conversation_id,
            segment_id=segment_id,
            messages=archived_messages,
        )
        encoded_history = self._encode_messages(segment.messages)
        encoded_live = self._encode_messages(retained_messages)
        self._require_write_bound(encoded_history)
        self._require_write_bound(encoded_live)
        return _ConversationSealPlan(
            segment=segment,
            retained_messages=retained_messages,
            encoded_history=encoded_history,
            encoded_live=encoded_live,
        )

    def _publish_seal(
        self,
        address: ConversationAddress,
        plan: _ConversationSealPlan,
    ) -> ConversationSealResult:
        """在第二次状态校验通过后耐久发布 history 并裁剪 live。"""

        try:
            created = atomic_create_bytes(
                self.layout.history_path(address, plan.segment.segment_id),
                plan.encoded_history,
                artifact_root=self.layout.root,
            )
        except ImmutableArtifactConflictError as exc:
            raise ConversationWriteConflictError("history segment identity conflicts with different bytes") from exc
        atomic_replace_bytes(
            self.layout.live_path(address),
            plan.encoded_live,
            artifact_root=self.layout.root,
        )
        current_state = self._load_state(address, self._history_references(address))
        if current_state.archived_through_sequence != plan.segment.end_sequence:
            raise ConversationJournalError("published history did not advance the archive high-watermark")
        return ConversationSealResult(
            status=(ConversationSealStatus.CREATED if created else ConversationSealStatus.UNCHANGED),
            segment=plan.segment,
            live=self._batch_or_none(address, plan.retained_messages),
        )

    @staticmethod
    def _same_segment(left: ConversationSegment, right: ConversationSegment) -> bool:
        return left.segment_id == right.segment_id and left.digest == right.digest

    def read_segment(
        self,
        address: ConversationAddress,
        segment_id: str,
    ) -> ConversationSegment:
        """读取一个已经封存且不可变的 history 片段。"""

        start_sequence, end_sequence = self.layout.segment_range(segment_id)
        path = self.layout.history_path(address, segment_id)
        messages = self._read_messages(path, missing_ok=False)
        segment = ConversationSegment(address.conversation_id, segment_id, messages)
        if segment.start_sequence != start_sequence or segment.end_sequence != end_sequence:
            raise ConversationJournalError("history filename range does not match its messages")
        return segment

    def read_state(self, address: ConversationAddress) -> ConversationJournalState:
        """读取并在必要时从耐久 History 修复 Conversation 高水位。"""

        with self.path_lock.acquire(
            self.layout.lock_key(address),
            ttl_seconds=self.config.lock_ttl_seconds,
            wait_timeout_seconds=self.config.lock_wait_timeout_seconds,
            retry_delay_seconds=self.config.lock_retry_delay_seconds,
        ) as guard:
            with guard.fenced():
                return self._load_state(address, self._history_references(address))

    def list_history(self, address: ConversationAddress) -> tuple[ConversationSegment, ...]:
        """按序读取仍未被生命周期游标逻辑释放的 History Segment。"""

        with self.path_lock.acquire(
            self.layout.lock_key(address),
            ttl_seconds=self.config.lock_ttl_seconds,
            wait_timeout_seconds=self.config.lock_wait_timeout_seconds,
            retry_delay_seconds=self.config.lock_retry_delay_seconds,
        ) as guard:
            with guard.fenced():
                references = self._history_references(address)
                state = self._load_state(address, references)
                active = tuple(reference for reference in references if reference.end_sequence > state.released_through)
                return tuple(self.read_segment(address, reference.segment_id) for reference in active)

    def release_history_prefix(
        self,
        address: ConversationAddress,
        segments: tuple[ConversationSegment, ...],
    ) -> tuple[str, ...]:
        """先耐久推进释放游标，再幂等删除已经验证的最旧连续原文前缀。"""

        if not isinstance(segments, tuple) or any(not isinstance(segment, ConversationSegment) for segment in segments):
            raise TypeError("segments must be a tuple of ConversationSegment values")
        if any(segment.conversation_id != address.conversation_id for segment in segments):
            raise ValueError("history release segments belong to another conversation")
        with self.path_lock.acquire(
            self.layout.lock_key(address),
            ttl_seconds=self.config.lock_ttl_seconds,
            wait_timeout_seconds=self.config.lock_wait_timeout_seconds,
            retry_delay_seconds=self.config.lock_retry_delay_seconds,
        ) as guard:
            with guard.fenced():
                references = self._history_references(address)
                state = self._load_state(address, references)
                active = tuple(reference for reference in references if reference.end_sequence > state.released_through)
                expected = active[: len(segments)]
                if tuple(item.segment_id for item in expected) != tuple(item.segment_id for item in segments):
                    raise ConversationWriteConflictError(
                        "history release must select the oldest unreleased contiguous prefix"
                    )
                for reference, expected_segment in zip(expected, segments, strict=True):
                    current = self.read_segment(address, reference.segment_id)
                    if current.digest != expected_segment.digest:
                        raise ConversationWriteConflictError("history segment changed before lifecycle release")
                if segments:
                    final = segments[-1]
                    updated = ConversationJournalState(
                        conversation_id=state.conversation_id,
                        archived_through_sequence=state.archived_through_sequence,
                        released_through_sequence=final.end_sequence,
                        latest_segment_id=state.latest_segment_id,
                        latest_segment_digest=state.latest_segment_digest,
                    )
                    self._write_state(address, updated)
                for reference in expected:
                    durable_unlink(reference.path, artifact_root=self.layout.root)
                return tuple(reference.segment_id for reference in expected)

    def purge_released_history(
        self,
        address: ConversationAddress,
        *,
        max_items: int,
    ) -> tuple[str, ...]:
        """恢复上次中断的物理删除；游标已经释放的文件不再对外可见。"""

        if isinstance(max_items, bool) or not isinstance(max_items, int) or max_items <= 0:
            raise ValueError("max_items must be a positive integer")
        with self.path_lock.acquire(
            self.layout.lock_key(address),
            ttl_seconds=self.config.lock_ttl_seconds,
            wait_timeout_seconds=self.config.lock_wait_timeout_seconds,
            retry_delay_seconds=self.config.lock_retry_delay_seconds,
        ) as guard:
            with guard.fenced():
                references = self._history_references(address)
                state = self._load_state(address, references)
                stale = tuple(
                    reference for reference in references if reference.end_sequence <= state.released_through
                )[:max_items]
                deleted: list[str] = []
                for reference in stale:
                    if durable_unlink(reference.path, artifact_root=self.layout.root):
                        deleted.append(reference.segment_id)
                return tuple(deleted)

    def _read_live_messages(self, address: ConversationAddress) -> tuple[ConversationMessage, ...]:
        return self._read_messages(self.layout.live_path(address), missing_ok=True)

    def _read_messages(
        self,
        path: Path,
        *,
        missing_ok: bool,
    ) -> tuple[ConversationMessage, ...]:
        try:
            encoded = read_regular_bytes(
                path,
                artifact_root=self.layout.root,
                max_bytes=self.config.max_file_bytes,
            )
        except FileNotFoundError:
            if missing_ok:
                return ()
            raise
        try:
            source = encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConversationJournalError("conversation JSONL is not valid UTF-8") from exc
        if not source:
            return ()
        messages: list[ConversationMessage] = []
        for line_number, line in enumerate(source.splitlines(), start=1):
            if not line:
                raise ConversationJournalError(f"conversation JSONL contains an empty line at {line_number}")
            try:
                raw: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConversationJournalError(f"conversation JSONL is invalid at line {line_number}") from exc
            if not isinstance(raw, dict):
                raise ConversationJournalError(f"conversation JSONL line {line_number} must be an object")
            try:
                messages.append(ConversationMessage.from_dict(raw))
            except ConversationMessageSchemaError as exc:
                raise ConversationJournalError(f"conversation JSONL message is invalid at line {line_number}") from exc
        try:
            return ConversationBatch("read-validation", tuple(messages)).messages
        except ConversationMessageSchemaError as exc:
            raise ConversationJournalError("conversation JSONL messages are not contiguous") from exc

    def _load_state(
        self,
        address: ConversationAddress,
        references: tuple[_HistoryReference, ...],
    ) -> ConversationJournalState:
        """严格读取游标，并只从新增的连续 History 修复中断封存。"""

        state = self._read_state_file(address) or ConversationJournalState.empty(address.conversation_id)
        if state.conversation_id != address.conversation_id:
            raise ConversationJournalError("journal state belongs to another conversation")
        archived = -1 if state.archived_through_sequence is None else state.archived_through_sequence
        for reference in references:
            if reference.start_sequence <= archived < reference.end_sequence:
                raise ConversationJournalError("history segment crosses the archive high-watermark")
            if reference.start_sequence <= state.released_through < reference.end_sequence:
                raise ConversationJournalError("history segment crosses the release high-watermark")

        newer = tuple(reference for reference in references if reference.start_sequence > archived)
        if newer:
            expected_start = archived + 1
            for reference in newer:
                if reference.start_sequence != expected_start:
                    raise ConversationJournalError("new history does not continue the archive high-watermark")
                expected_start = reference.end_sequence + 1
            latest = self.read_segment(address, newer[-1].segment_id)
            state = ConversationJournalState(
                conversation_id=address.conversation_id,
                archived_through_sequence=latest.end_sequence,
                released_through_sequence=state.released_through_sequence,
                latest_segment_id=latest.segment_id,
                latest_segment_digest=latest.digest,
            )
            self._write_state(address, state)

        archived = -1 if state.archived_through_sequence is None else state.archived_through_sequence
        active = tuple(
            reference for reference in references if state.released_through < reference.end_sequence <= archived
        )
        if archived > state.released_through:
            if not active or active[0].start_sequence != state.released_through + 1:
                raise ConversationJournalError("retained history does not start after the release high-watermark")
            for previous, current in zip(active, active[1:], strict=False):
                if current.start_sequence != previous.end_sequence + 1:
                    raise ConversationJournalError("retained history contains a sequence gap")
            if active[-1].end_sequence != archived:
                raise ConversationJournalError("retained history does not reach the archive high-watermark")
            if active[-1].segment_id != state.latest_segment_id:
                raise ConversationJournalError("retained history tail differs from journal state")
            latest = self.read_segment(address, active[-1].segment_id)
            if latest.digest != state.latest_segment_digest:
                raise ConversationJournalError("retained history tail digest differs from journal state")
        return state

    def _read_state_file(self, address: ConversationAddress) -> ConversationJournalState | None:
        try:
            encoded = read_regular_bytes(
                self.layout.state_path(address),
                artifact_root=self.layout.root,
                max_bytes=self.config.max_file_bytes,
            )
        except FileNotFoundError:
            return None
        try:
            raw = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ConversationJournalError("conversation journal state is invalid JSON") from exc
        expected_fields = {
            "schema",
            "conversation_id",
            "archived_through_sequence",
            "released_through_sequence",
            "latest_segment_id",
            "latest_segment_digest",
        }
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            raise ConversationJournalError("conversation journal state has an invalid shape")
        if raw["schema"] != _JOURNAL_STATE_SCHEMA:
            raise ConversationJournalError("conversation journal state has an unsupported schema")
        state = ConversationJournalState(
            conversation_id=raw["conversation_id"],
            archived_through_sequence=raw["archived_through_sequence"],
            released_through_sequence=raw["released_through_sequence"],
            latest_segment_id=raw["latest_segment_id"],
            latest_segment_digest=raw["latest_segment_digest"],
        )
        if encoded != self._encode_state(state):
            raise ConversationJournalError("conversation journal state is not canonically encoded")
        return state

    def _write_state(self, address: ConversationAddress, state: ConversationJournalState) -> None:
        if state.conversation_id != address.conversation_id:
            raise ValueError("journal state belongs to another conversation")
        encoded = self._encode_state(state)
        self._require_write_bound(encoded)
        atomic_replace_bytes(
            self.layout.state_path(address),
            encoded,
            artifact_root=self.layout.root,
        )

    @staticmethod
    def _encode_state(state: ConversationJournalState) -> bytes:
        return (
            canonical_json(
                {
                    "schema": _JOURNAL_STATE_SCHEMA,
                    "conversation_id": state.conversation_id,
                    "archived_through_sequence": state.archived_through_sequence,
                    "released_through_sequence": state.released_through_sequence,
                    "latest_segment_id": state.latest_segment_id,
                    "latest_segment_digest": state.latest_segment_digest,
                }
            )
            + "\n"
        ).encode("utf-8")

    def _latest_retained_segment(
        self,
        address: ConversationAddress,
        state: ConversationJournalState,
        references: tuple[_HistoryReference, ...],
    ) -> ConversationSegment | None:
        if state.archived_through_sequence is None or state.archived_through_sequence <= state.released_through:
            return None
        latest = next(
            (reference for reference in reversed(references) if reference.segment_id == state.latest_segment_id),
            None,
        )
        if latest is None:
            raise ConversationJournalError("journal state points to missing retained history")
        return self.read_segment(address, latest.segment_id)

    def _history_references(self, address: ConversationAddress) -> tuple[_HistoryReference, ...]:
        directory = self.layout.history_directory(address)
        if directory.is_symlink():
            raise ConversationJournalError("history directory cannot be a symbolic link")
        if not directory.exists():
            return ()
        if not directory.is_dir():
            raise ConversationJournalError("history path is not a directory")
        references: list[_HistoryReference] = []
        for entry_count, child in enumerate(directory.iterdir(), start=1):
            if entry_count > self.config.max_history_files:
                raise ConversationJournalError("history entry count exceeds its enumeration bound")
            if child.is_symlink():
                raise ConversationJournalError("history cannot contain symbolic links")
            if child.name.startswith("."):
                if child.is_file():
                    temporary_destination = atomic_temporary_destination(child.name)
                    if temporary_destination is not None and temporary_destination.endswith(".jsonl"):
                        try:
                            self.layout.segment_range(Path(temporary_destination).stem)
                        except ValueError:
                            pass
                        else:
                            continue
                raise ConversationJournalError("history contains an unsupported hidden entry")
            try:
                metadata = child.stat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(metadata.st_mode) or child.suffix != ".jsonl":
                raise ConversationJournalError("history may contain only segment JSONL files")
            segment_id = child.stem
            start_sequence, end_sequence = self.layout.segment_range(segment_id)
            references.append(_HistoryReference(segment_id, start_sequence, end_sequence, child))
        references.sort(key=lambda item: (item.start_sequence, item.end_sequence))
        for previous, current in zip(references, references[1:], strict=False):
            if current.start_sequence <= previous.end_sequence:
                raise ConversationJournalError("history segment ranges overlap")
        return tuple(references)

    @staticmethod
    def _calendar_component(value: str, *, digits: int, label: str) -> int:
        if len(value) != digits or not value.isascii() or not value.isdigit():
            raise ConversationJournalError(f"conversation {label} directory must contain exactly {digits} digits")
        return int(value)

    def _recover_archived_prefix(
        self,
        address: ConversationAddress,
        latest: ConversationSegment | None,
        live_messages: list[ConversationMessage],
    ) -> list[ConversationMessage]:
        if latest is None or not live_messages:
            return live_messages
        if live_messages[0].sequence > latest.end_sequence:
            return live_messages
        if live_messages[0].sequence < latest.start_sequence:
            raise ConversationJournalError("live messages precede the latest history segment")

        archived_by_sequence = {message.sequence: message for message in latest.messages}
        overlap_count = 0
        for message in live_messages:
            if message.sequence > latest.end_sequence:
                break
            archived = archived_by_sequence.get(message.sequence)
            if archived is None or archived.to_dict() != message.to_dict():
                raise ConversationWriteConflictError("live/history overlap contains different message bytes")
            overlap_count += 1
        if overlap_count == 0:
            return live_messages
        retained = live_messages[overlap_count:]
        encoded = self._encode_messages(retained)
        atomic_replace_bytes(
            self.layout.live_path(address),
            encoded,
            artifact_root=self.layout.root,
        )
        return retained

    @staticmethod
    def _known_tail(
        latest: ConversationSegment | None,
        live_messages: list[ConversationMessage],
    ) -> tuple[ConversationMessage, ...]:
        archived = latest.messages if latest is not None else ()
        return tuple([*archived, *live_messages])

    def _known_for_append(
        self,
        address: ConversationAddress,
        batch: ConversationBatch,
        state: ConversationJournalState,
        references: tuple[_HistoryReference, ...],
        latest: ConversationSegment | None,
        live_messages: list[ConversationMessage],
        *,
        expected_next: int,
    ) -> tuple[ConversationMessage, ...]:
        """为一次可能跨多个新 Segment 的重放恢复仍可验证的事实范围。"""

        if batch.start_sequence < expected_next and batch.start_sequence <= state.released_through:
            raise ConversationWriteConflictError(
                "append replay predates released conversation history"
            )
        known_by_sequence = {
            item.sequence: item for item in self._known_tail(latest, live_messages)
        }
        if batch.start_sequence < expected_next:
            for reference in references:
                if (
                    reference.end_sequence < batch.start_sequence
                    or reference.start_sequence >= expected_next
                ):
                    continue
                segment = self.read_segment(address, reference.segment_id)
                for item in segment.messages:
                    if batch.start_sequence <= item.sequence < expected_next:
                        known_by_sequence[item.sequence] = item
        return tuple(known_by_sequence[key] for key in sorted(known_by_sequence))

    @staticmethod
    def _require_tail_continuity(
        state: ConversationJournalState,
        live_messages: list[ConversationMessage],
    ) -> None:
        if not live_messages:
            return
        expected_start = state.next_sequence
        if live_messages[0].sequence != expected_start:
            raise ConversationJournalError("live messages do not continue the archive high-watermark")

    @staticmethod
    def _expected_next_sequence(
        state: ConversationJournalState,
        live_messages: list[ConversationMessage],
    ) -> int:
        if live_messages:
            return live_messages[-1].sequence + 1
        return state.next_sequence

    @staticmethod
    def _encode_messages(messages: Sequence[ConversationMessage]) -> bytes:
        return "".join(canonical_json(message.to_dict()) + "\n" for message in messages).encode("utf-8")

    def _require_write_bound(self, encoded: bytes) -> None:
        if len(encoded) > self.config.max_file_bytes:
            raise ConversationJournalError("conversation JSONL exceeds its hard safety bound")

    @staticmethod
    def _require_batch(address: ConversationAddress, batch: ConversationBatch) -> None:
        if not isinstance(address, ConversationAddress):
            raise TypeError("address must be a ConversationAddress")
        if not isinstance(batch, ConversationBatch):
            raise TypeError("batch must be a ConversationBatch")
        if address.conversation_id != batch.conversation_id:
            raise ConversationWriteConflictError("conversation address does not match the appended batch")

    @staticmethod
    def _batch_or_none(
        address: ConversationAddress,
        messages: tuple[ConversationMessage, ...] | list[ConversationMessage],
    ) -> ConversationBatch | None:
        resolved = tuple(messages)
        return ConversationBatch(address.conversation_id, resolved) if resolved else None


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


__all__ = [
    "ConversationAppendResult",
    "ConversationAppendStatus",
    "ConversationJournalConfig",
    "ConversationJournalError",
    "ConversationJournalState",
    "ConversationMessageJournal",
    "ConversationSealResult",
    "ConversationSealStatus",
    "ConversationWriteConflictError",
]
