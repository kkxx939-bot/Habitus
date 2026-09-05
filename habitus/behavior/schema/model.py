"""行为语义树的声明式 Schema 模型。

字段按**面**声明角色（``TODO(BHV-TREE-REBUILD-001)`` 的双面设计）：``address`` 构成身份，
``numeric`` 是数字面（时间预测树夜批读，机器类型白名单、全部必填），``semantic`` 是语义面
（语义关联读，允许可空与自由文本），``system`` 是溯源（不渲染进正文）。分组的唯一权威在
schema 声明——物理 JSON 块保持扁平，数据不重复承载分组信息。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

from habitus.behavior.model import BehaviorAddress, BehaviorKind

_FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_CANONICAL_PATHS = {
    BehaviorKind.OCCURRENCE: (
        "occurrences/{occurred_on:%Y}/{occurred_on:%m}/{occurred_on:%d}/"
        "{name}--{started_at:%Y%m%dT%H%M%S%f%z}.md"
    ),
    BehaviorKind.GAP: (
        "gaps/{occurred_on:%Y}/{occurred_on:%m}/{occurred_on:%d}/"
        "{gap_kind}--{started_at:%Y%m%dT%H%M%S%f%z}.md"
    ),
}
_EXPECTED_ADDRESS_NAMES = {
    BehaviorKind.OCCURRENCE: ("occurred_on", "name", "started_at"),
    BehaviorKind.GAP: ("occurred_on", "gap_kind", "started_at"),
}


class BehaviorSchemaError(ValueError):
    """Schema 声明或结构化行为字段不满足已确认合同。"""


class BehaviorFieldType(str, Enum):
    STRING = "string"
    OPTIONAL_STRING = "optional_string"
    DATE = "date"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    STRING_LIST = "string_list"
    SHA256_LIST = "sha256_list"
    SHA256 = "sha256"
    OPTIONAL_SHA256 = "optional_sha256"
    OCCURRENCE_STATUS = "occurrence_status"
    STATUS_BASIS = "status_basis"
    GAP_KIND = "gap_kind"
    BASIS_LIST = "basis_list"


class BehaviorFieldRole(str, Enum):
    ADDRESS = "address"
    NUMERIC = "numeric"
    SEMANTIC = "semantic"
    SYSTEM = "system"


# 数字面只准机器类型：无可空、无自由复合结构——它是给夜批逐字段扫的。
_NUMERIC_FIELD_TYPES = frozenset(
    {
        BehaviorFieldType.STRING,
        BehaviorFieldType.DATETIME,
        BehaviorFieldType.BOOLEAN,
        BehaviorFieldType.OCCURRENCE_STATUS,
        BehaviorFieldType.STATUS_BASIS,
    }
)


class BehaviorOperationMode(str, Enum):
    """整棵树纯 add-only；追加/修改模式随 Outcome 通道一并退役。"""

    ADD_ONLY = "add_only"


@dataclass(frozen=True)
class BehaviorFieldSchema:
    name: str
    field_type: BehaviorFieldType
    role: BehaviorFieldRole
    required: bool
    description: str

    def __post_init__(self) -> None:
        if not _FIELD_NAME.fullmatch(self.name):
            raise BehaviorSchemaError("behavior schema field name must use lowercase snake_case")
        object.__setattr__(self, "field_type", BehaviorFieldType(self.field_type))
        object.__setattr__(self, "role", BehaviorFieldRole(self.role))
        if not isinstance(self.required, bool):
            raise BehaviorSchemaError("behavior schema field required must be boolean")
        if not isinstance(self.description, str) or not self.description.strip():
            raise BehaviorSchemaError("behavior schema field description must be non-empty")
        if self.role in {
            BehaviorFieldRole.ADDRESS,
            BehaviorFieldRole.NUMERIC,
            BehaviorFieldRole.SYSTEM,
        } and not self.required:
            raise BehaviorSchemaError(
                "behavior address, numeric and system fields must be required"
            )
        if (
            self.role is BehaviorFieldRole.NUMERIC
            and self.field_type not in _NUMERIC_FIELD_TYPES
        ):
            raise BehaviorSchemaError(
                f"behavior numeric field {self.name} must use a machine-typed field type"
            )


@dataclass(frozen=True)
class BehaviorTypeSchema:
    kind: BehaviorKind
    description: str
    path_template: str
    operation_mode: BehaviorOperationMode
    fields: tuple[BehaviorFieldSchema, ...]

    def __post_init__(self) -> None:
        kind = BehaviorKind(self.kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "operation_mode", BehaviorOperationMode(self.operation_mode))
        object.__setattr__(self, "fields", tuple(self.fields))
        if not isinstance(self.description, str) or not self.description.strip():
            raise BehaviorSchemaError("behavior type description must be non-empty")
        if self.path_template != _CANONICAL_PATHS[kind]:
            raise BehaviorSchemaError(
                f"{kind.value} schema path does not match the confirmed behavior tree"
            )
        path = PurePosixPath(self.path_template)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".md":
            raise BehaviorSchemaError("behavior schema path template is unsafe")
        names = tuple(field.name for field in self.fields)
        if not names or len(names) != len(set(names)):
            raise BehaviorSchemaError("behavior schema fields must be non-empty and unique")
        address_names = tuple(
            field.name for field in self.fields if field.role is BehaviorFieldRole.ADDRESS
        )
        if address_names != _EXPECTED_ADDRESS_NAMES[kind]:
            raise BehaviorSchemaError("behavior schema address fields do not match its path")

    @property
    def field_map(self) -> dict[str, BehaviorFieldSchema]:
        return {field.name: field for field in self.fields}

    def fields_of(self, role: BehaviorFieldRole) -> tuple[BehaviorFieldSchema, ...]:
        """按面取字段——消费者据此选择自己的读集，加字段不改读取代码。"""

        resolved = BehaviorFieldRole(role)
        return tuple(field for field in self.fields if field.role is resolved)


@dataclass(frozen=True)
class BehaviorSchemaMaterialization:
    """Schema 为 L2 Codec 一次性生成的地址、持久字段与可读正文。"""

    address: BehaviorAddress
    storage_fields: Mapping[str, Any]
    markdown_body: str


__all__ = [
    "BehaviorFieldRole",
    "BehaviorFieldSchema",
    "BehaviorFieldType",
    "BehaviorOperationMode",
    "BehaviorSchemaError",
    "BehaviorSchemaMaterialization",
    "BehaviorTypeSchema",
]
