"""记忆变更回执的幂等文件存储。"""

from __future__ import annotations

import json
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from foundation.ids import same_path_identity
from foundation.integrity import canonical_json
from infrastructure.store.filesystem import (
    atomic_create_bytes,
    atomic_replace_bytes,
    atomic_temporary_destination,
    durable_unlink,
    read_regular_bytes,
)
from memory.conversation import ConversationAddress
from memory.document import MemoryDocumentCodec
from memory.editor.engine import MemoryEditorPlan
from memory.editor.transaction_log import MemoryTransactionJournalRecord
from memory.workflow.receipt.model import (
    MemoryChangeReceipt,
    MemoryChangeReceiptError,
    MemoryChangeReceiptState,
    MemoryChangeSource,
)
from memory.workflow.receipt.projector import MemoryChangeReceiptProjector


@dataclass(frozen=True)
class MemoryChangeReceiptStoreConfig:
    """耐久记忆变更回执的单文件与目录枚举容量。"""

    max_file_bytes: int = 4 * 1024 * 1024
    max_files: int = 100_000

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_file_bytes, bool)
            or not isinstance(self.max_file_bytes, int)
            or not 4_096 <= self.max_file_bytes <= 64 * 1024 * 1024
        ):
            raise ValueError("max_file_bytes must be between 4096 and 67108864")
        if (
            isinstance(self.max_files, bool)
            or not isinstance(self.max_files, int)
            or not 1 <= self.max_files <= 10_000_000
        ):
            raise ValueError("max_files must be between 1 and 10000000")


class MemoryChangeReceiptStore:
    """在 L2 树外保存可恢复的准备态和最终态变更回执。"""

    def __init__(
        self,
        root: str | Path,
        codec: MemoryDocumentCodec,
        *,
        config: MemoryChangeReceiptStoreConfig | None = None,
    ) -> None:
        requested = Path(root).expanduser().absolute()
        if requested.is_symlink():
            raise MemoryChangeReceiptError("memory change receipt root cannot be a symbolic link")
        if not isinstance(codec, MemoryDocumentCodec):
            raise TypeError("codec must be MemoryDocumentCodec")
        if config is not None and not isinstance(config, MemoryChangeReceiptStoreConfig):
            raise TypeError("config must be MemoryChangeReceiptStoreConfig")
        self.root = requested.resolve(strict=False)
        self.receipts_root = self.root / "receipts"
        self.codec = codec
        self.projector = MemoryChangeReceiptProjector(codec)
        self.config = config or MemoryChangeReceiptStoreConfig()

    def prepare(
        self,
        source: MemoryChangeSource,
        plan: MemoryEditorPlan,
        *,
        timestamp: datetime,
    ) -> MemoryChangeReceipt:
        """在发布 L2 前只创建一次完整的语义变更意图。"""

        if not isinstance(source, MemoryChangeSource):
            raise TypeError("source must be MemoryChangeSource")
        if not isinstance(plan, MemoryEditorPlan):
            raise TypeError("plan must be MemoryEditorPlan")
        prepared = self.projector.prepare(source, plan, timestamp=timestamp)
        existing = self.try_read(source)
        if existing is not None:
            if existing.source != source:
                raise MemoryChangeReceiptError("existing change receipt is bound to another source")
            if not self.projector.same_change_intent(existing, prepared):
                raise MemoryChangeReceiptError("prepared change receipt conflicts with a new semantic plan")
            return existing
        try:
            self._create(prepared)
        except Exception:
            raced = self.try_read(source)
            if raced is None or not self.projector.same_change_intent(raced, prepared):
                raise
            return raced
        return self.read(source)

    def finalize(
        self,
        source: MemoryChangeSource,
        journal: MemoryTransactionJournalRecord,
    ) -> MemoryChangeReceipt:
        """只接受 COMMITTED 事务，并以实际 before/after 文档完成回执。"""

        if not isinstance(source, MemoryChangeSource):
            raise TypeError("source must be MemoryChangeSource")
        if not isinstance(journal, MemoryTransactionJournalRecord):
            raise TypeError("journal must be MemoryTransactionJournalRecord")
        current = self.read(source)
        committed = self.projector.finalize(current, source, journal)
        if current.state is MemoryChangeReceiptState.COMMITTED:
            if current != committed:
                raise MemoryChangeReceiptError("committed change receipt conflicts with its transaction journal")
            return current
        self._replace(committed)
        return self.read(source)

    def read(self, source: MemoryChangeSource) -> MemoryChangeReceipt:
        if not isinstance(source, MemoryChangeSource):
            raise TypeError("source must be MemoryChangeSource")
        receipt = self._read_path(self._path(source))
        if not receipt.source.matches_lookup(source):
            raise MemoryChangeReceiptError("memory change receipt source does not match its path")
        return receipt

    def try_read(self, source: MemoryChangeSource) -> MemoryChangeReceipt | None:
        try:
            return self.read(source)
        except MemoryChangeReceiptError as exc:
            if isinstance(exc.__cause__, FileNotFoundError):
                return None
            raise

    def discard_prepared(self, source: MemoryChangeSource) -> bool:
        """事务确认未提交或已回滚后，允许删除对应准备态。"""

        current = self.try_read(source)
        if current is None:
            return False
        if current.state is not MemoryChangeReceiptState.PREPARED:
            raise MemoryChangeReceiptError("committed change receipt cannot be discarded")
        return durable_unlink(self._path(source), artifact_root=self.root)

    def list_for_conversation(
        self,
        address: ConversationAddress,
    ) -> tuple[MemoryChangeReceipt, ...]:
        """按全局记忆序号返回一个 Conversation 的全部耐久回执。"""

        if not isinstance(address, ConversationAddress):
            raise TypeError("address must be a ConversationAddress")
        receipts = tuple(
            receipt
            for receipt in self._read_all()
            if same_path_identity(
                receipt.source.conversation_id,
                address.conversation_id,
                "conversation_id",
            )
            and receipt.source.started_on == address.started_on
        )
        return tuple(sorted(receipts, key=lambda item: item.source.memory_sequence))

    def discard_committed(self, receipt: MemoryChangeReceipt) -> bool:
        """耐久删除一条已经满足外部保留期和来源安全门槛的终态回执。"""

        if not isinstance(receipt, MemoryChangeReceipt):
            raise TypeError("receipt must be a MemoryChangeReceipt")
        if receipt.state is not MemoryChangeReceiptState.COMMITTED:
            raise MemoryChangeReceiptError("only a COMMITTED receipt can be discarded")
        current = self.try_read(receipt.source)
        if current is None:
            return False
        if current != receipt:
            raise MemoryChangeReceiptError("memory change receipt changed before lifecycle cleanup")
        return durable_unlink(self._path(receipt.source), artifact_root=self.root)

    def _create(self, receipt: MemoryChangeReceipt) -> None:
        encoded = self._encode(receipt)
        atomic_create_bytes(self._path(receipt.source), encoded, artifact_root=self.root)

    def _replace(self, receipt: MemoryChangeReceipt) -> None:
        encoded = self._encode(receipt)
        atomic_replace_bytes(self._path(receipt.source), encoded, artifact_root=self.root)

    def _read_all(self) -> tuple[MemoryChangeReceipt, ...]:
        if not self.receipts_root.exists():
            return ()
        if self.receipts_root.is_symlink() or not self.receipts_root.is_dir():
            raise MemoryChangeReceiptError("memory change receipt root is not a safe directory")
        receipts: list[MemoryChangeReceipt] = []
        seen_entries = 0
        for child in self.receipts_root.iterdir():
            seen_entries += 1
            if seen_entries > self.config.max_files:
                raise MemoryChangeReceiptError("memory change receipt entry count exceeds its safety bound")
            try:
                metadata = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise MemoryChangeReceiptError("failed to inspect memory change receipt entry") from exc
            if not stat.S_ISREG(metadata.st_mode):
                raise MemoryChangeReceiptError("memory change receipt root contains an unsupported entry")
            temporary_destination = atomic_temporary_destination(child.name)
            if temporary_destination is not None:
                destination = Path(temporary_destination)
                if destination.suffix == ".json" and MemoryChangeSource._hex(destination.stem, 64):
                    continue
            if child.suffix != ".json" or not MemoryChangeSource._hex(child.stem, 64):
                raise MemoryChangeReceiptError("memory change receipt root contains an unsupported entry")
            receipt = self._read_path(child)
            if child.stem != receipt.source.receipt_id:
                raise MemoryChangeReceiptError("memory change receipt filename does not match its source")
            receipts.append(receipt)
        return tuple(receipts)

    def _read_path(self, path: Path) -> MemoryChangeReceipt:
        try:
            encoded = read_regular_bytes(
                path,
                artifact_root=self.root,
                max_bytes=self.config.max_file_bytes,
            )
            value = json.loads(encoded)
            if not isinstance(value, Mapping):
                raise MemoryChangeReceiptError("memory change receipt must contain an object")
            receipt = MemoryChangeReceipt.from_dict(value)
            if encoded != self._encode(receipt):
                raise MemoryChangeReceiptError("memory change receipt is not canonically encoded")
            return receipt
        except Exception as exc:
            if isinstance(exc, MemoryChangeReceiptError):
                raise
            raise MemoryChangeReceiptError("failed to read a valid memory change receipt") from exc

    def _encode(self, receipt: MemoryChangeReceipt) -> bytes:
        encoded = (canonical_json(receipt.to_dict()) + "\n").encode("utf-8")
        if len(encoded) > self.config.max_file_bytes:
            raise MemoryChangeReceiptError("memory change receipt exceeds its file bound")
        return encoded

    def _path(self, source: MemoryChangeSource) -> Path:
        return self.receipts_root / f"{source.receipt_id}.json"


__all__ = ["MemoryChangeReceiptStore", "MemoryChangeReceiptStoreConfig"]
