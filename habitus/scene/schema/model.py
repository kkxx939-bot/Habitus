"""情景树的声明式 Schema 模型。

字段按角色分面：``address`` 构成身份，``semantic`` 是给判决 LLM 与投影读的内容，``system`` 是
溯源（不渲染进正文）。只有一种文档类型（情景），但仍走 YAML 声明——字段的唯一权威在声明里，
渲染守卫保证每个非 system 字段都出现在正文中。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

from habitus.scene.model import SceneAddress

_FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
CANONICAL_PATH = "scenes/{occurred_on:%Y}/{occurred_on:%m}/{occurred_on:%d}/{label}--{started_at:%Y%m%dT%H%M%S%f%z}.md"
EXPECTED_ADDRESS_NAMES = ("occurred_on", "label", "started_at")


class SceneSchemaError(ValueError):
    """Schema 声明或结构化情景字段不满足已确认合同。"""


class SceneFieldType(str, Enum):
    STRING = "string"
    DATE = "date"
    DATETIME = "datetime"
    STRING_LIST = "string_list"
    MEMBER_LIST = "member_list"
    PENDING_EFFECT_LIST = "pending_effect_list"
    SHA256 = "sha256"


class SceneFieldRole(str, Enum):
    ADDRESS = "address"
    SEMANTIC = "semantic"
    SYSTEM = "system"


@dataclass(frozen=True)
class SceneFieldSchema:
    name: str
    field_type: SceneFieldType
    role: SceneFieldRole
    required: bool
    description: str

    def __post_init__(self) -> None:
        if not _FIELD_NAME.fullmatch(self.name):
            raise SceneSchemaError("scene schema field name must use lowercase snake_case")
        object.__setattr__(self, "field_type", SceneFieldType(self.field_type))
        object.__setattr__(self, "role", SceneFieldRole(self.role))
        if not isinstance(self.required, bool):
            raise SceneSchemaError("scene schema field required must be boolean")
        if not isinstance(self.description, str) or not self.description.strip():
            raise SceneSchemaError("scene schema field description must be non-empty")
        if not self.required:
            # 全部字段必填：可空语义用空数组表达（"少说"），不用缺席表达——缺席分不清"模型认为没有"
            # 与"模型忘了写"。
            raise SceneSchemaError("scene schema fields must all be required")


@dataclass(frozen=True)
class SceneTypeSchema:
    description: str
    path_template: str
    fields: tuple[SceneFieldSchema, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        if not isinstance(self.description, str) or not self.description.strip():
            raise SceneSchemaError("scene type description must be non-empty")
        if self.path_template != CANONICAL_PATH:
            raise SceneSchemaError("scene schema path does not match the confirmed scene tree")
        path = PurePosixPath(self.path_template)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".md":
            raise SceneSchemaError("scene schema path template is unsafe")
        names = tuple(field.name for field in self.fields)
        if not names or len(names) != len(set(names)):
            raise SceneSchemaError("scene schema fields must be non-empty and unique")
        address_names = tuple(field.name for field in self.fields if field.role is SceneFieldRole.ADDRESS)
        if address_names != EXPECTED_ADDRESS_NAMES:
            raise SceneSchemaError("scene schema address fields do not match its path")

    @property
    def field_map(self) -> dict[str, SceneFieldSchema]:
        return {field.name: field for field in self.fields}

    def fields_of(self, role: SceneFieldRole) -> tuple[SceneFieldSchema, ...]:
        resolved = SceneFieldRole(role)
        return tuple(field for field in self.fields if field.role is resolved)


@dataclass(frozen=True)
class SceneSchemaMaterialization:
    """Schema 为 L2 Codec 一次性生成的地址、持久字段与可读正文。"""

    address: SceneAddress
    storage_fields: Mapping[str, Any]
    markdown_body: str


__all__ = [
    "CANONICAL_PATH",
    "EXPECTED_ADDRESS_NAMES",
    "SceneFieldRole",
    "SceneFieldSchema",
    "SceneFieldType",
    "SceneSchemaError",
    "SceneSchemaMaterialization",
    "SceneTypeSchema",
]
