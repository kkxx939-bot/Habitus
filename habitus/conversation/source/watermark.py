"""地址水位：这个地址的源已经按顺序释放到哪个 end_sequence。

释放严格按地址内的顺序（更早的没放，后面的不放），所以水位就是一个只增不减的整数，没有零散项。
先写水位、再删信封：删到一半崩溃，读侧看到"水位已盖住、信封没了"就知道那份源已经放过了。
"""

from __future__ import annotations

import json
from pathlib import Path

from habitus.conversation.source.address import SourceAddress
from habitus.conversation.source.model import ConversationSourceError, encode_durable_record
from habitus.infrastructure.store.filesystem import atomic_replace_bytes, read_regular_bytes

_SCHEMA = "conversation_source_watermark_v1"
_MAX_WATERMARK_BYTES = 4 * 1024


class AddressWatermarks:
    def __init__(self, conversation_root: str | Path) -> None:
        self.root = Path(conversation_root).expanduser().resolve(strict=False)
        self.watermark_root = self.root / "source" / "watermarks"

    def released_through(self, address: SourceAddress) -> int | None:
        """已释放到的 end_sequence；一份都没放过时为 None。"""

        try:
            encoded = read_regular_bytes(self._path(address), artifact_root=self.root, max_bytes=_MAX_WATERMARK_BYTES)
        except FileNotFoundError:
            return None
        try:
            payload = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConversationSourceError("source watermark is corrupt") from exc
        if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
            raise ConversationSourceError("source watermark has an unexpected schema")
        if SourceAddress.from_dict(payload["address"]) != address:
            raise ConversationSourceError("source watermark belongs to another address")
        value = payload["released_through"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConversationSourceError("source watermark value is invalid")
        return value

    def covers(self, address: SourceAddress, end_sequence: int) -> bool:
        released = self.released_through(address)
        return released is not None and end_sequence <= released

    def advance(self, address: SourceAddress, end_sequence: int) -> None:
        """推进到 ``end_sequence``；不许后退。调用方负责在 fence 内调用。"""

        current = self.released_through(address)
        if current is not None and end_sequence < current:
            raise ConversationSourceError("source watermark cannot move backwards")
        if current == end_sequence:
            return
        record = {"schema": _SCHEMA, "address": address.to_dict(), "released_through": end_sequence}
        encoded = encode_durable_record(record, max_bytes=_MAX_WATERMARK_BYTES, label="source watermark")
        atomic_replace_bytes(self._path(address), encoded, artifact_root=self.root)

    def _path(self, address: SourceAddress) -> Path:
        return self.watermark_root / f"{address.digest}.json"


__all__ = ["AddressWatermarks"]
