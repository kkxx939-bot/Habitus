"""预测样本 Markdown 正文与结构字段的严格编解码器。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, NoReturn

from foundation.integrity import canonicalize
from prediction.document.model import PredictionDocument
from prediction.model import PredictionAddress, PredictionKind
from prediction.schema import PredictionSchemaRegistry

_MARKER = "\n<!-- M2BOS_PREDICTION_FIELDS\n"
_FOOTER = "\n-->\n"
_METADATA_KEYS = {"prediction_type", "fields"}


class PredictionDocumentIntegrityError(ValueError):
    """预测样本正文、结构字段或物理地址不一致。"""


class PredictionDocumentCodec:
    """使用同一个 Schema 构造地址、正文和末尾结构字段。

    Schema 注册表不是可替换的策略：它强制 path_template 必须逐字等于代码里的
    规范路径，本身就是预测树的领域单例不变量，因此这里直接依赖具体类型。
    """

    def __init__(self, registry: PredictionSchemaRegistry) -> None:
        if not isinstance(registry, PredictionSchemaRegistry):
            raise TypeError("registry must be a PredictionSchemaRegistry")
        self.registry = registry

    def build(self, kind: PredictionKind | str, payload: Mapping[str, Any]) -> PredictionDocument:
        normalized_kind = PredictionKind(kind)
        materialized = self.registry.materialize(normalized_kind, payload)
        if _MARKER in materialized.markdown_body:
            raise PredictionDocumentIntegrityError("prediction Markdown body contains the reserved metadata marker")
        return PredictionDocument(
            kind=normalized_kind,
            address=materialized.address,
            fields=materialized.storage_fields,
            markdown_body=materialized.markdown_body,
        )

    def encode(self, document: PredictionDocument) -> str:
        if not isinstance(document, PredictionDocument):
            raise TypeError("document must be a PredictionDocument")
        canonical = self.build(document.kind, document.fields)
        if canonical.address != document.address:
            raise PredictionDocumentIntegrityError("prediction document address is not canonical")
        if canonical.markdown_body != document.markdown_body:
            raise PredictionDocumentIntegrityError("prediction document body is not canonical")
        metadata_json = json.dumps(
            {
                "prediction_type": canonical.kind.value,
                "fields": canonicalize(canonical.fields),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).replace("--", "\\u002d\\u002d")
        return f"{canonical.markdown_body}{_MARKER}{metadata_json}{_FOOTER}"

    def decode(self, raw: str, *, expected_address: PredictionAddress) -> PredictionDocument:
        if not isinstance(raw, str):
            raise TypeError("raw prediction document must be a string")
        if not isinstance(expected_address, PredictionAddress):
            raise TypeError("expected_address must be a PredictionAddress")
        if raw.count(_MARKER) != 1 or not raw.endswith(_FOOTER):
            raise PredictionDocumentIntegrityError(
                "prediction document must contain one terminal M2BOS_PREDICTION_FIELDS comment"
            )
        markdown_body, _separator, metadata_with_footer = raw.partition(_MARKER)
        metadata_source = metadata_with_footer[: -len(_FOOTER)]
        try:
            metadata = json.loads(
                metadata_source,
                object_pairs_hook=self._unique_object,
                parse_constant=self._reject_json_constant,
            )
        except (json.JSONDecodeError, PredictionDocumentIntegrityError) as exc:
            raise PredictionDocumentIntegrityError("prediction document metadata is not strict JSON") from exc
        if not isinstance(metadata, dict) or set(metadata) != _METADATA_KEYS:
            raise PredictionDocumentIntegrityError("prediction document metadata has an invalid shape")
        raw_kind = metadata["prediction_type"]
        raw_fields = metadata["fields"]
        if not isinstance(raw_kind, str) or not isinstance(raw_fields, dict):
            raise PredictionDocumentIntegrityError("prediction document metadata types are invalid")
        try:
            document = self.build(PredictionKind(raw_kind), raw_fields)
        except (TypeError, ValueError) as exc:
            raise PredictionDocumentIntegrityError("prediction document fields do not satisfy their Schema") from exc
        if document.address != expected_address:
            raise PredictionDocumentIntegrityError("prediction document fields do not match the physical tree address")
        if document.markdown_body != markdown_body:
            raise PredictionDocumentIntegrityError("prediction document body does not match its structured fields")
        if self.encode(document) != raw:
            raise PredictionDocumentIntegrityError("prediction document is not canonically serialized")
        return document

    @staticmethod
    def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PredictionDocumentIntegrityError("prediction document metadata contains a duplicate JSON key")
            result[key] = value
        return result

    @staticmethod
    def _reject_json_constant(value: str) -> NoReturn:
        raise PredictionDocumentIntegrityError(f"prediction document contains invalid JSON constant: {value}")


__all__ = ["PredictionDocumentCodec", "PredictionDocumentIntegrityError"]
