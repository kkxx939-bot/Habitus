"""预测样本的声明式 Schema 模型。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

from prediction.model import PredictionAddress, PredictionKind

_FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_CANONICAL_PATHS = {
    kind: (
        f"samples/{kind.branch_name}/{{sample_date:%Y}}/{{sample_date:%m}}/"
        "{sample_date:%d}/{shard}/{materialization_id}.md"
    )
    for kind in PredictionKind
}


class PredictionSchemaError(ValueError):
    """Schema 声明或预测样本不满足严格合同。"""


class PredictionFieldType(str, Enum):
    DATE = "date"
    SAMPLE_ID = "sample_id"
    IDENTITY_MATERIAL = "identity_material"
    SCOPE = "scope"
    ANCHOR = "anchor"
    INPUT = "input"
    TREATMENT = "treatment"
    TRANSITION_LABEL = "transition_label"
    TRAJECTORY_LABEL = "trajectory_label"
    CONSEQUENCE_LABEL = "consequence_label"
    SUPERVISION = "supervision"
    LINEAGE = "lineage"
    PROVENANCE = "provenance"
    QUALITY = "quality"


class PredictionFieldRole(str, Enum):
    ADDRESS = "address"
    CONTENT = "content"
    SYSTEM = "system"


class PredictionOperationMode(str, Enum):
    ADD_ONLY = "add_only"


@dataclass(frozen=True)
class PredictionFieldSchema:
    name: str
    field_type: PredictionFieldType
    role: PredictionFieldRole
    required: bool
    description: str

    def __post_init__(self) -> None:
        if _FIELD_NAME.fullmatch(self.name) is None:
            raise PredictionSchemaError("prediction schema field name must use lowercase snake_case")
        object.__setattr__(self, "field_type", PredictionFieldType(self.field_type))
        object.__setattr__(self, "role", PredictionFieldRole(self.role))
        if not isinstance(self.required, bool):
            raise PredictionSchemaError("prediction schema field required must be boolean")
        if not isinstance(self.description, str) or not self.description.strip():
            raise PredictionSchemaError("prediction schema field description must be non-empty")
        if self.role in {PredictionFieldRole.ADDRESS, PredictionFieldRole.SYSTEM} and not self.required:
            raise PredictionSchemaError("prediction address and system fields must be required")


@dataclass(frozen=True)
class PredictionTypeSchema:
    kind: PredictionKind
    description: str
    path_template: str
    operation_mode: PredictionOperationMode
    fields: tuple[PredictionFieldSchema, ...]

    def __post_init__(self) -> None:
        kind = PredictionKind(self.kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "operation_mode", PredictionOperationMode(self.operation_mode))
        object.__setattr__(self, "fields", tuple(self.fields))
        if not isinstance(self.description, str) or not self.description.strip():
            raise PredictionSchemaError("prediction type description must be non-empty")
        if self.path_template != _CANONICAL_PATHS[kind]:
            raise PredictionSchemaError("prediction schema path does not match the confirmed tree")
        path = PurePosixPath(self.path_template)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".md":
            raise PredictionSchemaError("prediction schema path template is unsafe")
        names = tuple(field.name for field in self.fields)
        if not names or len(names) != len(set(names)):
            raise PredictionSchemaError("prediction schema fields must be non-empty and unique")
        address_names = tuple(field.name for field in self.fields if field.role is PredictionFieldRole.ADDRESS)
        if address_names != ("sample_date", "logical_sample_id", "materialization_id"):
            raise PredictionSchemaError(
                "prediction schema address fields must be sample_date, logical_sample_id and materialization_id"
            )
        system_names = tuple(field.name for field in self.fields if field.role is PredictionFieldRole.SYSTEM)
        if system_names != ("identity_material", "materialization_context", "provenance"):
            raise PredictionSchemaError(
                "prediction schema system fields must be identity_material, materialization_context and provenance"
            )

    @property
    def field_map(self) -> dict[str, PredictionFieldSchema]:
        return {field.name: field for field in self.fields}


@dataclass(frozen=True)
class PredictionSchemaMaterialization:
    address: PredictionAddress
    storage_fields: Mapping[str, Any]
    markdown_body: str


__all__ = [
    "PredictionFieldRole",
    "PredictionFieldSchema",
    "PredictionFieldType",
    "PredictionOperationMode",
    "PredictionSchemaError",
    "PredictionSchemaMaterialization",
    "PredictionTypeSchema",
]
