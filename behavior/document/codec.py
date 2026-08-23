"""行为语义 L2 Markdown 的严格规范编解码器。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any, NoReturn

from behavior.document.link import BehaviorStoredLink, parse_stored_links
from behavior.document.model import BehaviorDocument, BehaviorDocumentMetadata
from behavior.model import BehaviorAddress, BehaviorKind
from behavior.schema import BehaviorSchemaRegistry
from foundation.integrity import canonicalize

_MARKER = "\n<!-- HABITUS_BEHAVIOR_FIELDS\n"
_FOOTER = "\n-->\n"
_METADATA_KEYS = {
    "behavior_type",
    "revision",
    "created_at",
    "updated_at",
    "fields",
    "links",
}


class BehaviorDocumentIntegrityError(ValueError):
    """行为 L2 的正文、结构字段和物理地址不一致。"""


class BehaviorDocumentCodec:
    """使用同一个 Schema 构造地址、正文和末尾结构字段。

    Schema 注册表不是可替换的策略：它强制 path_template 必须逐字等于代码里的
    规范路径，本身就是行为树的领域单例不变量，因此这里直接依赖具体类型。
    """

    def __init__(self, registry: BehaviorSchemaRegistry) -> None:
        if not isinstance(registry, BehaviorSchemaRegistry):
            raise TypeError("registry must be a BehaviorSchemaRegistry")
        self.registry = registry

    def build(
        self,
        kind: BehaviorKind | str,
        payload: Mapping[str, Any],
        *,
        metadata: BehaviorDocumentMetadata,
        links: tuple[BehaviorStoredLink, ...] = (),
    ) -> BehaviorDocument:
        if not isinstance(metadata, BehaviorDocumentMetadata):
            raise TypeError("metadata must be BehaviorDocumentMetadata")
        normalized_kind = BehaviorKind(kind)
        materialized = self.registry.materialize(normalized_kind, payload)
        if _MARKER in materialized.markdown_body:
            raise BehaviorDocumentIntegrityError("behavior Markdown body contains the reserved metadata marker")
        return BehaviorDocument(
            kind=normalized_kind,
            address=materialized.address,
            metadata=metadata,
            fields=materialized.storage_fields,
            markdown_body=materialized.markdown_body,
            links=links,
        )

    def encode(self, document: BehaviorDocument) -> str:
        if not isinstance(document, BehaviorDocument):
            raise TypeError("document must be a BehaviorDocument")
        canonical = self.build(
            document.kind,
            document.fields,
            metadata=document.metadata,
            links=document.links,
        )
        if canonical.address != document.address:
            raise BehaviorDocumentIntegrityError("behavior document address is not canonical")
        if canonical.markdown_body != document.markdown_body:
            raise BehaviorDocumentIntegrityError("behavior document body is not canonical")
        metadata = {
            "behavior_type": canonical.kind.value,
            "revision": canonical.metadata.revision,
            "created_at": self._timestamp(canonical.metadata.created_at),
            "updated_at": self._timestamp(canonical.metadata.updated_at),
            "fields": canonicalize(canonical.fields),
            "links": [link.to_dict() for link in canonical.links],
        }
        # 结构字段保存在 HTML 注释里，正文中任意 `--` 都会提前闭合该注释；
        # 统一转义成 JSON 的 -，decode 时由 json.loads 原样还原，往返仍然逐字节一致。
        metadata_json = json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).replace("--", "\\u002d\\u002d")
        return f"{canonical.markdown_body}{_MARKER}{metadata_json}{_FOOTER}"

    def decode(self, raw: str, *, expected_address: BehaviorAddress) -> BehaviorDocument:
        if not isinstance(raw, str):
            raise TypeError("raw behavior document must be a string")
        if not isinstance(expected_address, BehaviorAddress):
            raise TypeError("expected_address must be a BehaviorAddress")
        if raw.count(_MARKER) != 1 or not raw.endswith(_FOOTER):
            raise BehaviorDocumentIntegrityError(
                "behavior document must contain one terminal HABITUS_BEHAVIOR_FIELDS comment"
            )
        markdown_body, _separator, metadata_with_footer = raw.partition(_MARKER)
        metadata_source = metadata_with_footer[: -len(_FOOTER)]
        try:
            metadata = json.loads(
                metadata_source,
                object_pairs_hook=self._unique_object,
                parse_constant=self._reject_json_constant,
            )
        except (json.JSONDecodeError, BehaviorDocumentIntegrityError) as exc:
            raise BehaviorDocumentIntegrityError("behavior document metadata is not strict JSON") from exc
        if not isinstance(metadata, dict) or set(metadata) != _METADATA_KEYS:
            raise BehaviorDocumentIntegrityError("behavior document metadata has an invalid shape")
        raw_kind = metadata["behavior_type"]
        raw_fields = metadata["fields"]
        if not isinstance(raw_kind, str):
            raise BehaviorDocumentIntegrityError("behavior document type must be a string")
        if not isinstance(raw_fields, dict) or any(not isinstance(name, str) for name in raw_fields):
            raise BehaviorDocumentIntegrityError("behavior document fields must be an object")
        try:
            document = self.build(
                BehaviorKind(raw_kind),
                raw_fields,
                metadata=BehaviorDocumentMetadata(
                    revision=metadata["revision"],
                    created_at=self._parse_timestamp(metadata["created_at"], "created_at"),
                    updated_at=self._parse_timestamp(metadata["updated_at"], "updated_at"),
                ),
                links=parse_stored_links(metadata["links"], label="behavior document links"),
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

    @staticmethod
    def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BehaviorDocumentIntegrityError("behavior document metadata contains a duplicate JSON key")
            result[key] = value
        return result

    @staticmethod
    def _reject_json_constant(value: str) -> NoReturn:
        raise BehaviorDocumentIntegrityError(f"behavior document metadata contains an invalid JSON constant: {value}")

    @staticmethod
    def _timestamp(value: datetime) -> str:
        return value.isoformat(timespec="microseconds").replace("+00:00", "Z")

    @staticmethod
    def _parse_timestamp(value: object, field_name: str) -> datetime:
        if not isinstance(value, str):
            raise BehaviorDocumentIntegrityError(f"behavior document {field_name} must be a timestamp string")
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise BehaviorDocumentIntegrityError(
                f"behavior document {field_name} is not a valid ISO timestamp"
            ) from exc


__all__ = ["BehaviorDocumentCodec", "BehaviorDocumentIntegrityError"]
