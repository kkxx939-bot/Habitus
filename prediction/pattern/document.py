"""StatePattern 与 BranchPattern 的规范 Markdown 文档。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NoReturn

from foundation.integrity import canonicalize, immutable_snapshot
from prediction.model import PredictionPatternAddress, PredictionPatternKind
from prediction.pattern.schema import PredictionPatternSchemaRegistry

_MARKER = "\n<!-- M2BOS_PREDICTION_PATTERN_FIELDS\n"
_FOOTER = "\n-->\n"


class PredictionPatternDocumentIntegrityError(ValueError):
    """模式图正文、字段或物理地址不一致。"""


@dataclass(frozen=True)
class PredictionPatternDocument:
    kind: PredictionPatternKind
    address: PredictionPatternAddress
    fields: Mapping[str, Any]
    markdown_body: str

    def __post_init__(self) -> None:
        kind = PredictionPatternKind(self.kind)
        object.__setattr__(self, "kind", kind)
        if not isinstance(self.address, PredictionPatternAddress) or self.address.kind is not kind:
            raise ValueError("prediction pattern address does not match its kind")
        if not isinstance(self.fields, Mapping):
            raise TypeError("prediction pattern fields must be a mapping")
        object.__setattr__(self, "fields", immutable_snapshot(self.fields))
        if not isinstance(self.markdown_body, str) or not self.markdown_body.strip() or not self.markdown_body.endswith("\n"):
            raise ValueError("prediction pattern Markdown must be non-empty and end with a newline")


class PredictionPatternDocumentCodec:
    def __init__(self, registry: PredictionPatternSchemaRegistry | None = None) -> None:
        self.registry = registry or PredictionPatternSchemaRegistry.load_default()

    def build(self, kind: PredictionPatternKind | str, payload: Mapping[str, Any]) -> PredictionPatternDocument:
        normalized_kind = PredictionPatternKind(kind)
        fields, address, body = self.registry.materialize(normalized_kind, payload)
        return PredictionPatternDocument(normalized_kind, address, fields, body)

    def encode(self, document: PredictionPatternDocument) -> str:
        if not isinstance(document, PredictionPatternDocument):
            raise TypeError("document must be a PredictionPatternDocument")
        canonical = self.build(document.kind, document.fields)
        if canonical != document:
            raise PredictionPatternDocumentIntegrityError("prediction pattern document is not canonical")
        metadata = json.dumps(
            {"pattern_type": document.kind.value, "fields": canonicalize(document.fields)},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).replace("--", "\\u002d\\u002d")
        return f"{document.markdown_body}{_MARKER}{metadata}{_FOOTER}"

    def decode(self, raw: str, *, expected_address: PredictionPatternAddress) -> PredictionPatternDocument:
        if not isinstance(raw, str) or raw.count(_MARKER) != 1 or not raw.endswith(_FOOTER):
            raise PredictionPatternDocumentIntegrityError("prediction pattern has invalid terminal metadata")
        body, _separator, source = raw.partition(_MARKER)
        try:
            metadata = json.loads(
                source[: -len(_FOOTER)],
                object_pairs_hook=self._unique_object,
                parse_constant=self._reject_constant,
            )
        except (json.JSONDecodeError, PredictionPatternDocumentIntegrityError) as exc:
            raise PredictionPatternDocumentIntegrityError("prediction pattern metadata is not strict JSON") from exc
        if not isinstance(metadata, dict) or set(metadata) != {"pattern_type", "fields"}:
            raise PredictionPatternDocumentIntegrityError("prediction pattern metadata has an invalid shape")
        try:
            document = self.build(PredictionPatternKind(metadata["pattern_type"]), metadata["fields"])
        except (TypeError, ValueError, KeyError) as exc:
            raise PredictionPatternDocumentIntegrityError("prediction pattern fields are invalid") from exc
        if document.address != expected_address or document.markdown_body != body or self.encode(document) != raw:
            raise PredictionPatternDocumentIntegrityError("prediction pattern does not match its address or body")
        return document

    @staticmethod
    def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PredictionPatternDocumentIntegrityError("prediction pattern contains duplicate JSON keys")
            result[key] = value
        return result

    @staticmethod
    def _reject_constant(value: str) -> NoReturn:
        raise PredictionPatternDocumentIntegrityError(f"prediction pattern contains invalid JSON constant: {value}")


__all__ = [
    "PredictionPatternDocument",
    "PredictionPatternDocumentCodec",
    "PredictionPatternDocumentIntegrityError",
]
