"""预测样本身份和规范文档的领域工厂。"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

from prediction.document import PredictionDocument, PredictionDocumentCodec
from prediction.identity import derive_sample_identity
from prediction.model import PredictionKind


class PredictionSampleFactory:
    """集中控制样本 SHA-256 身份，投影适配器只声明语义身份材料。"""

    def __init__(self, document_codec: PredictionDocumentCodec) -> None:
        if not isinstance(document_codec, PredictionDocumentCodec):
            raise TypeError("document_codec must be a PredictionDocumentCodec")
        self.document_codec = document_codec

    def build(
        self,
        kind: PredictionKind,
        *,
        sample_date: date,
        identity: Mapping[str, Any],
        projection_version: str,
        materialization_context: Mapping[str, Any] | None = None,
        fields: Mapping[str, Any],
    ) -> PredictionDocument:
        """由稳定身份材料生成地址字段，并交给唯一 Schema 物化文档。"""

        normalized_kind = PredictionKind(kind)
        if isinstance(sample_date, datetime) or not isinstance(sample_date, date):
            raise TypeError("sample_date must be a date without a time")
        if not isinstance(identity, Mapping) or not identity:
            raise ValueError("prediction sample identity must be a non-empty mapping")
        if not isinstance(fields, Mapping):
            raise TypeError("prediction sample fields must be a mapping")
        reserved = {
            "sample_date",
            "logical_sample_id",
            "materialization_id",
            "identity_material",
            "materialization_context",
        } & set(fields)
        if reserved:
            raise ValueError("prediction sample address fields are owned by the factory")
        context = {} if materialization_context is None else dict(materialization_context)
        sample_identity = derive_sample_identity(normalized_kind, identity, projection_version, context)
        return self.document_codec.build(
            normalized_kind,
            {
                "sample_date": sample_date,
                "logical_sample_id": sample_identity.logical_sample_id,
                "materialization_id": sample_identity.materialization_id,
                "identity_material": dict(identity),
                "materialization_context": context,
                **fields,
            },
        )


__all__ = ["PredictionSampleFactory"]
