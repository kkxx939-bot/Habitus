"""L2 记忆文档的严格规范编解码器。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date
from typing import Any, Protocol

from habitus.infrastructure.tree import envelope
from habitus.infrastructure.tree.metadata import DocumentMetadata
from habitus.memory.document.link import MemoryStoredLink, parse_stored_links
from habitus.memory.document.model import MemoryDocument
from habitus.memory.model import MemoryAddress, MemoryKind

_METADATA_KEYS = frozenset({envelope.TYPE_KEY, "revision", "created_at", "updated_at", "fields", "links", "backlinks"})


class MemoryDocumentIntegrityError(ValueError):
    """L2 文档缺少结构字段，或正文、路径与字段不一致。"""


class _MemorySchemaRegistry(Protocol):
    def validate(self, kind: MemoryKind | str, payload: Mapping[str, Any]) -> dict[str, Any]: ...

    def address_for(
        self,
        kind: MemoryKind | str,
        payload: Mapping[str, Any],
    ) -> MemoryAddress: ...

    def render_markdown(
        self,
        kind: MemoryKind | str,
        payload: Mapping[str, Any],
    ) -> str: ...


class MemoryDocumentCodec:
    """使用同一份 Schema 构造、序列化并验证 L2 文档。"""

    def __init__(self, registry: _MemorySchemaRegistry) -> None:
        self.registry = registry

    def build(
        self,
        kind: MemoryKind | str,
        payload: Mapping[str, Any],
        *,
        metadata: DocumentMetadata,
        links: tuple[MemoryStoredLink, ...] = (),
        backlinks: tuple[MemoryStoredLink, ...] = (),
    ) -> MemoryDocument:
        """从结构字段生成地址和可读正文，不接受调用者提供的路径或正文。"""

        normalized_kind = MemoryKind(kind)
        normalized = self.registry.validate(normalized_kind, payload)
        address = self.registry.address_for(normalized_kind, normalized)
        markdown_body = self.registry.render_markdown(normalized_kind, normalized)
        try:
            envelope.require_plain_body(markdown_body)
        except envelope.TreeEnvelopeError as exc:
            raise MemoryDocumentIntegrityError("memory Markdown body contains the reserved metadata marker") from exc
        return MemoryDocument(
            kind=normalized_kind,
            address=address,
            metadata=metadata,
            fields=normalized,
            markdown_body=markdown_body,
            links=links,
            backlinks=backlinks,
        )

    def encode(self, document: MemoryDocument) -> str:
        """重新校验文档后输出唯一规范的 Markdown 物理格式。"""

        canonical = self.build(
            document.kind,
            document.fields,
            metadata=document.metadata,
            links=document.links,
            backlinks=document.backlinks,
        )
        if canonical.address != document.address:
            raise MemoryDocumentIntegrityError("memory document address is not canonical")
        if canonical.markdown_body != document.markdown_body:
            raise MemoryDocumentIntegrityError("memory document body is not canonical")
        fields = {name: self._json_value(value) for name, value in canonical.fields.items()}
        metadata = {
            envelope.TYPE_KEY: canonical.kind.value,
            "revision": canonical.metadata.revision,
            "created_at": envelope.format_timestamp(canonical.metadata.created_at),
            "updated_at": envelope.format_timestamp(canonical.metadata.updated_at),
            "fields": fields,
            "links": [link.to_dict() for link in canonical.links],
            "backlinks": [link.to_dict() for link in canonical.backlinks],
        }
        return envelope.encode(canonical.markdown_body, metadata)

    def decode(self, raw: str, *, expected_address: MemoryAddress) -> MemoryDocument:
        """解析并验证唯一末尾注释、Schema、真实地址和正文规范性。"""

        try:
            markdown_body, metadata = envelope.decode(raw, keys=_METADATA_KEYS)
        except envelope.TreeEnvelopeError as exc:
            raise MemoryDocumentIntegrityError(f"memory document envelope is invalid: {exc}") from exc
        raw_kind = metadata[envelope.TYPE_KEY]
        raw_fields = metadata["fields"]
        raw_links = metadata["links"]
        raw_backlinks = metadata["backlinks"]
        try:
            links = parse_stored_links(raw_links, label="memory document links")
            backlinks = parse_stored_links(
                raw_backlinks,
                label="memory document backlinks",
            )
            system_metadata = DocumentMetadata(
                revision=envelope.parse_revision(metadata["revision"]),
                created_at=envelope.parse_timestamp(metadata["created_at"], "created_at"),
                updated_at=envelope.parse_timestamp(metadata["updated_at"], "updated_at"),
            )
            document = self.build(
                MemoryKind(raw_kind),
                raw_fields,
                metadata=system_metadata,
                links=links,
                backlinks=backlinks,
            )
        except (TypeError, ValueError) as exc:
            raise MemoryDocumentIntegrityError("memory document fields do not satisfy their Schema") from exc
        if document.address != expected_address:
            raise MemoryDocumentIntegrityError("memory document fields do not match the physical tree address")
        if document.markdown_body != markdown_body:
            raise MemoryDocumentIntegrityError("memory document body does not match its structured fields")
        if self.encode(document) != raw:
            raise MemoryDocumentIntegrityError("memory document is not canonically serialized")
        return document

    @classmethod
    def _json_value(cls, value: Any) -> str | int | float | bool:
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, bool | str | int):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        raise MemoryDocumentIntegrityError("memory document contains a field that cannot be serialized")


__all__ = ["MemoryDocumentCodec", "MemoryDocumentIntegrityError"]
