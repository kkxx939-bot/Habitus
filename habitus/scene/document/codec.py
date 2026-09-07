"""情景 L2 Markdown 的严格规范编解码器（沿行为树同一格式：正文 + 末尾 JSON 结构块）。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any, NoReturn

from habitus.foundation.integrity import canonicalize
from habitus.scene.document.link import SceneStoredLink, parse_stored_links
from habitus.scene.document.model import SceneDocument, SceneDocumentMetadata
from habitus.scene.model import SceneAddress
from habitus.scene.schema import SceneSchemaRegistry

_MARKER = "\n<!-- HABITUS_SCENE_FIELDS\n"
_FOOTER = "\n-->\n"
_METADATA_KEYS = {"scene_type", "created_at", "fields", "links"}
_SCENE_TYPE = "scene"


class SceneDocumentIntegrityError(ValueError):
    """情景 L2 的正文、结构字段和物理地址不一致。"""


class SceneDocumentCodec:
    """使用同一个 Schema 构造地址、正文和末尾结构字段。"""

    def __init__(self, registry: SceneSchemaRegistry) -> None:
        if not isinstance(registry, SceneSchemaRegistry):
            raise TypeError("registry must be a SceneSchemaRegistry")
        self.registry = registry

    def build(
        self,
        payload: Mapping[str, Any],
        *,
        metadata: SceneDocumentMetadata,
        links: tuple[SceneStoredLink, ...] = (),
    ) -> SceneDocument:
        if not isinstance(metadata, SceneDocumentMetadata):
            raise TypeError("metadata must be SceneDocumentMetadata")
        materialized = self.registry.materialize(payload)
        if _MARKER in materialized.markdown_body:
            raise SceneDocumentIntegrityError("scene Markdown body contains the reserved metadata marker")
        return SceneDocument(
            address=materialized.address,
            metadata=metadata,
            fields=materialized.storage_fields,
            markdown_body=materialized.markdown_body,
            links=links,
        )

    def encode(self, document: SceneDocument) -> str:
        if not isinstance(document, SceneDocument):
            raise TypeError("document must be a SceneDocument")
        canonical = self.build(document.fields, metadata=document.metadata, links=document.links)
        if canonical.address != document.address:
            raise SceneDocumentIntegrityError("scene document address is not canonical")
        if canonical.markdown_body != document.markdown_body:
            raise SceneDocumentIntegrityError("scene document body is not canonical")
        metadata = {
            "scene_type": _SCENE_TYPE,
            "created_at": canonical.metadata.created_at.isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "fields": canonicalize(canonical.fields),
            "links": [link.to_dict() for link in canonical.links],
        }
        # 结构字段住在 HTML 注释里，正文中任意 ``--`` 都会提前闭合该注释；统一转义成 JSON 的 -。
        metadata_json = json.dumps(
            metadata, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        ).replace("--", "\\u002d\\u002d")
        return f"{canonical.markdown_body}{_MARKER}{metadata_json}{_FOOTER}"

    def decode(self, raw: str, *, expected_address: SceneAddress) -> SceneDocument:
        if not isinstance(raw, str):
            raise TypeError("raw scene document must be a string")
        if not isinstance(expected_address, SceneAddress):
            raise TypeError("expected_address must be a SceneAddress")
        if raw.count(_MARKER) != 1 or not raw.endswith(_FOOTER):
            raise SceneDocumentIntegrityError("scene document must contain one terminal HABITUS_SCENE_FIELDS comment")
        markdown_body, _separator, metadata_with_footer = raw.partition(_MARKER)
        metadata_source = metadata_with_footer[: -len(_FOOTER)]
        try:
            metadata = json.loads(
                metadata_source, object_pairs_hook=self._unique_object, parse_constant=self._reject_json_constant
            )
        except (json.JSONDecodeError, RecursionError, SceneDocumentIntegrityError) as exc:
            raise SceneDocumentIntegrityError("scene document metadata is not strict JSON") from exc
        if not isinstance(metadata, dict) or set(metadata) != _METADATA_KEYS:
            raise SceneDocumentIntegrityError("scene document metadata has an invalid shape")
        if metadata["scene_type"] != _SCENE_TYPE:
            raise SceneDocumentIntegrityError("scene document type is not a scene")
        raw_fields = metadata["fields"]
        if not isinstance(raw_fields, dict) or any(not isinstance(name, str) for name in raw_fields):
            raise SceneDocumentIntegrityError("scene document fields must be an object")
        try:
            document = self.build(
                raw_fields,
                metadata=SceneDocumentMetadata(created_at=self._parse_timestamp(metadata["created_at"])),
                links=parse_stored_links(metadata["links"], label="scene document links"),
            )
        except (TypeError, ValueError) as exc:
            raise SceneDocumentIntegrityError("scene document fields do not satisfy their Schema") from exc
        if document.address != expected_address:
            raise SceneDocumentIntegrityError("scene document fields do not match the physical tree address")
        if document.markdown_body != markdown_body:
            raise SceneDocumentIntegrityError("scene document body does not match its structured fields")
        if self.encode(document) != raw:
            raise SceneDocumentIntegrityError("scene document is not canonically serialized")
        return document

    @staticmethod
    def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SceneDocumentIntegrityError("scene document metadata contains a duplicate JSON key")
            result[key] = value
        return result

    @staticmethod
    def _reject_json_constant(value: str) -> NoReturn:
        raise SceneDocumentIntegrityError(f"scene document metadata contains an invalid JSON constant: {value}")

    @staticmethod
    def _parse_timestamp(value: object) -> datetime:
        if not isinstance(value, str):
            raise SceneDocumentIntegrityError("scene document created_at must be a timestamp string")
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SceneDocumentIntegrityError("scene document created_at is not a valid ISO timestamp") from exc


__all__ = ["SceneDocumentCodec", "SceneDocumentIntegrityError"]
