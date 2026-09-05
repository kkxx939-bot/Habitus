"""Conversation 外部投递的持久幂等回执。"""

from __future__ import annotations

import json
import re
import stat
from dataclasses import dataclass, replace
from enum import Enum

from habitus.foundation.integrity import canonical_json
from habitus.infrastructure.store.filesystem import DurablePathIntegrityError, list_real_directory
from habitus.infrastructure.store.filesystem.durable_io import (
    atomic_create_bytes,
    atomic_replace_bytes,
    durable_unlink,
    read_regular_bytes,
)
from habitus.memory.conversation.layout import ConversationAddress, ConversationLayout

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCHEMA = "conversation_ingress_receipt_v1"


class ConversationIngressError(ValueError):
    """投递身份或回执文件不满足持久幂等契约。"""


class ConversationIngressState(str, Enum):
    PREPARED = "prepared"
    COMMITTED = "committed"


@dataclass(frozen=True)
class ConversationIngressRequest:
    delivery_id: str
    request_digest: str

    def __post_init__(self) -> None:
        for name, value in (("delivery_id", self.delivery_id), ("request_digest", self.request_digest)):
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ConversationIngressError(f"{name} must be lowercase SHA-256 text")


@dataclass(frozen=True)
class ConversationIngressReceipt:
    delivery_id: str
    request_digest: str
    batch_digest: str
    start_sequence: int
    end_sequence: int
    state: ConversationIngressState
    next_sequence: int | None = None

    def __post_init__(self) -> None:
        for name in ("delivery_id", "request_digest", "batch_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ConversationIngressError(f"receipt {name} must be lowercase SHA-256 text")
        for name in ("start_sequence", "end_sequence"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConversationIngressError(f"receipt {name} must be non-negative")
        if self.end_sequence < self.start_sequence:
            raise ConversationIngressError("receipt sequence range is invalid")
        if not isinstance(self.state, ConversationIngressState):
            raise ConversationIngressError("receipt state is invalid")
        if self.state is ConversationIngressState.PREPARED:
            if self.next_sequence is not None:
                raise ConversationIngressError("prepared receipt cannot contain next_sequence")
        elif self.next_sequence != self.end_sequence + 1:
            raise ConversationIngressError("committed receipt next_sequence is inconsistent")

    def committed(self) -> ConversationIngressReceipt:
        return replace(self, state=ConversationIngressState.COMMITTED, next_sequence=self.end_sequence + 1)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA,
            "delivery_id": self.delivery_id,
            "request_digest": self.request_digest,
            "batch_digest": self.batch_digest,
            "start_sequence": self.start_sequence,
            "end_sequence": self.end_sequence,
            "state": self.state.value,
            "next_sequence": self.next_sequence,
        }

    @classmethod
    def from_dict(cls, value: object) -> ConversationIngressReceipt:
        if not isinstance(value, dict):
            raise ConversationIngressError("ingress receipt must be an object")
        allowed = {
            "schema_version",
            "delivery_id",
            "request_digest",
            "batch_digest",
            "start_sequence",
            "end_sequence",
            "state",
            "next_sequence",
        }
        if set(value) != allowed or value.get("schema_version") != _SCHEMA:
            raise ConversationIngressError("ingress receipt schema is invalid")
        try:
            state = ConversationIngressState(value["state"])
        except (TypeError, ValueError) as exc:
            raise ConversationIngressError("ingress receipt state is invalid") from exc
        return cls(
            delivery_id=value["delivery_id"],
            request_digest=value["request_digest"],
            batch_digest=value["batch_digest"],
            start_sequence=value["start_sequence"],
            end_sequence=value["end_sequence"],
            state=state,
            next_sequence=value["next_sequence"],
        )


class ConversationIngressReceiptStore:
    """只负责回执文件的有界、安全读写，不决定 Conversation 追加语义。"""

    def __init__(self, layout: ConversationLayout, *, max_files: int, max_file_bytes: int) -> None:
        if not isinstance(layout, ConversationLayout):
            raise TypeError("layout must be ConversationLayout")
        for name, value in (("max_files", max_files), ("max_file_bytes", max_file_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.layout = layout
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes

    def read(self, address: ConversationAddress, delivery_id: str) -> ConversationIngressReceipt | None:
        path = self.layout.ingress_path(address, delivery_id)
        try:
            encoded = read_regular_bytes(
                path,
                artifact_root=self.layout.root,
                max_bytes=self.max_file_bytes,
            )
        except FileNotFoundError:
            return None
        try:
            return ConversationIngressReceipt.from_dict(json.loads(encoded.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError, ConversationIngressError) as exc:
            raise ConversationIngressError("ingress receipt is corrupt") from exc

    def list(self, address: ConversationAddress) -> tuple[ConversationIngressReceipt, ...]:
        try:
            entries = list_real_directory(
                self.layout.ingress_directory(address),
                artifact_root=self.layout.root,
                max_entries=self.max_files,
            )
        except DurablePathIntegrityError as exc:
            raise ConversationIngressError("ingress receipt directory is invalid") from exc
        receipts: list[ConversationIngressReceipt] = []
        for entry in entries:
            if not stat.S_ISREG(entry.mode) or re.fullmatch(r"[0-9a-f]{64}\.json", entry.name) is None:
                raise ConversationIngressError("ingress receipt directory contains an unsupported entry")
            receipt = self.read(address, entry.name[:-5])
            if receipt is None:
                raise ConversationIngressError("ingress receipt disappeared during enumeration")
            receipts.append(receipt)
        return tuple(receipts)

    def prepare(
        self,
        address: ConversationAddress,
        request: ConversationIngressRequest,
        *,
        batch_digest: str,
        start_sequence: int,
        end_sequence: int,
    ) -> ConversationIngressReceipt:
        current = self.read(address, request.delivery_id)
        if current is not None:
            return current
        if len(self.list(address)) >= self.max_files:
            raise ConversationIngressError("ingress receipt directory reached its configured capacity")
        receipt = ConversationIngressReceipt(
            delivery_id=request.delivery_id,
            request_digest=request.request_digest,
            batch_digest=batch_digest,
            start_sequence=start_sequence,
            end_sequence=end_sequence,
            state=ConversationIngressState.PREPARED,
        )
        encoded = self._encode(receipt)
        atomic_create_bytes(
            self.layout.ingress_path(address, request.delivery_id),
            encoded,
            artifact_root=self.layout.root,
        )
        stored = self.read(address, request.delivery_id)
        if stored is None:
            raise ConversationIngressError("prepared ingress receipt was not persisted")
        return stored

    def commit(self, address: ConversationAddress, receipt: ConversationIngressReceipt) -> ConversationIngressReceipt:
        committed = receipt.committed()
        atomic_replace_bytes(
            self.layout.ingress_path(address, receipt.delivery_id),
            self._encode(committed),
            artifact_root=self.layout.root,
        )
        return committed

    def discard_prepared(self, address: ConversationAddress, receipt: ConversationIngressReceipt) -> None:
        if receipt.state is not ConversationIngressState.PREPARED:
            raise ConversationIngressError("only a prepared ingress receipt can be discarded")
        durable_unlink(
            self.layout.ingress_path(address, receipt.delivery_id),
            artifact_root=self.layout.root,
        )

    def _encode(self, receipt: ConversationIngressReceipt) -> bytes:
        encoded = (canonical_json(receipt.to_dict()) + "\n").encode("utf-8")
        if len(encoded) > self.max_file_bytes:
            raise ConversationIngressError("ingress receipt exceeds its configured byte limit")
        return encoded


__all__ = [
    "ConversationIngressError",
    "ConversationIngressReceipt",
    "ConversationIngressReceiptStore",
    "ConversationIngressRequest",
    "ConversationIngressState",
]
