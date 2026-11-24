"""行为语义 L2 Markdown 的严格规范编解码器。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from habitus.behavior.document.model import BehaviorDocument
from habitus.behavior.model import BehaviorAddress, BehaviorKind
from habitus.behavior.schema import BehaviorSchemaRegistry
from habitus.foundation.integrity import canonicalize
from habitus.infrastructure.tree import envelope
from habitus.infrastructure.tree.metadata import DocumentMetadata

_METADATA_KEYS = frozenset({envelope.TYPE_KEY, "revision", "created_at", "updated_at", "fields"})


class BehaviorDocumentIntegrityError(ValueError):
    """行为 L2 的正文、结构字段和物理地址不一致。"""


class BehaviorDocumentCodec:
    """使用同一个 Schema 构造地址、正文和末尾结构字段。

    Schema 注册表不是可替换的策略：它强制 path_template 必须逐字等于代码里的
    规范路径，本身就是行为树的领域单例不变量，因此这里直接依赖具体类型。
    """

    def __init__(self, registry: BehaviorSchemaRegistry) -> None:
        self.registry = registry

    def build(
        self,
        kind: BehaviorKind | str,
        payload: Mapping[str, Any],
        *,
        metadata: DocumentMetadata,
    ) -> BehaviorDocument:
        normalized_kind = BehaviorKind(kind)
        materialized = self.registry.materialize(normalized_kind, payload)
        try:
            envelope.require_plain_body(materialized.markdown_body)
        except envelope.TreeEnvelopeError as exc:
            raise BehaviorDocumentIntegrityError(
                "behavior Markdown body contains the reserved metadata marker"
            ) from exc
        return BehaviorDocument(
            kind=normalized_kind,
            address=materialized.address,
            metadata=metadata,
            fields=materialized.storage_fields,
            markdown_body=materialized.markdown_body,
        )

    def encode(self, document: BehaviorDocument) -> str:
        canonical = self.build(document.kind, document.fields, metadata=document.metadata)
        if canonical.address != document.address:
            raise BehaviorDocumentIntegrityError("behavior document address is not canonical")
        if canonical.markdown_body != document.markdown_body:
            raise BehaviorDocumentIntegrityError("behavior document body is not canonical")
        metadata = {
            envelope.TYPE_KEY: canonical.kind.value,
            "revision": canonical.metadata.revision,
            "created_at": envelope.format_timestamp(canonical.metadata.created_at),
            "updated_at": envelope.format_timestamp(canonical.metadata.updated_at),
            "fields": canonicalize(canonical.fields),
        }
        return envelope.encode(canonical.markdown_body, metadata)

    def decode(self, raw: str, *, expected_address: BehaviorAddress) -> BehaviorDocument:
        if not isinstance(raw, str):
            raise TypeError("raw behavior document must be a string")
        if not isinstance(expected_address, BehaviorAddress):
            raise TypeError("expected_address must be a BehaviorAddress")
        try:
            markdown_body, metadata = envelope.decode(raw, keys=_METADATA_KEYS)
        except envelope.TreeEnvelopeError as exc:
            raise BehaviorDocumentIntegrityError(f"behavior document envelope is invalid: {exc}") from exc
        raw_kind = metadata[envelope.TYPE_KEY]
        raw_fields = metadata["fields"]
        try:
            kind = BehaviorKind(raw_kind)
            document = self.build(
                kind,
                raw_fields,
                metadata=DocumentMetadata(
                    revision=envelope.parse_revision(metadata["revision"]),
                    created_at=envelope.parse_timestamp(metadata["created_at"], "created_at"),
                    updated_at=envelope.parse_timestamp(metadata["updated_at"], "updated_at"),
                ),
            )
        except (TypeError, ValueError) as exc:
            raise BehaviorDocumentIntegrityError("behavior document fields do not satisfy their Schema") from exc
        if document.address != expected_address:
            raise BehaviorDocumentIntegrityError("behavior document fields do not match the physical tree address")
        if document.markdown_body != markdown_body:
            raise BehaviorDocumentIntegrityError("behavior document body does not match its structured fields")
        if self.encode(document) != raw:
            raise BehaviorDocumentIntegrityError("behavior document is not canonically serialized")
        return document


__all__ = ["BehaviorDocumentCodec", "BehaviorDocumentIntegrityError"]
