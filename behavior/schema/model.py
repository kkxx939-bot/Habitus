"""行为语义树的声明式 Schema 模型。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

from behavior.model import BehaviorAddress, BehaviorKind

_FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_CANONICAL_PATHS = {
    BehaviorKind.EVENT: (
        "behaviors/events/{event_date:%Y}/{event_date:%m}/{event_date:%d}/"
        "{event_name}--{started_at:%Y%m%dT%H%M%S%f%z}.md"
    ),
    BehaviorKind.OUTCOME: (
        "behaviors/outcomes/{event_date:%Y}/{event_date:%m}/{event_date:%d}/"
        "{event_name}--{event_started_at:%Y%m%dT%H%M%S%f%z}.md"
    ),
    BehaviorKind.EPISODE: (
        "episodes/{episode_date:%Y}/{episode_date:%m}/{episode_date:%d}/"
        "{episode_name}--{started_at:%Y%m%dT%H%M%S%f%z}.md"
    ),
}


class BehaviorSchemaError(ValueError):
    """Schema 声明或结构化行为字段不满足已确认合同。"""


class BehaviorFieldType(str, Enum):
    STRING = "string"
    OPTIONAL_STRING = "optional_string"
    DATE = "date"
    DATETIME = "datetime"
    OPTIONAL_DATETIME = "optional_datetime"
    NUMBER = "number"
    STRING_LIST = "string_list"
    ACTION_LIST = "action_list"
    OUTCOME_LIST = "outcome_list"
    URI_LIST = "uri_list"
    PHASE_LIST = "phase_list"
    UNPHASED_EVENT_LIST = "unphased_event_list"
    OUTCOME_SNAPSHOT_LIST = "outcome_snapshot_list"
    TRANSITION_LIST = "transition_list"


class BehaviorFieldRole(str, Enum):
    ADDRESS = "address"
    CONTENT = "content"
    SYSTEM = "system"


class BehaviorMergeStrategy(str, Enum):
    IMMUTABLE = "immutable"
    APPEND = "append"


class BehaviorOperationMode(str, Enum):
    ADD_ONLY = "add_only"
    APPEND_ONLY = "append_only"


@dataclass(frozen=True)
class BehaviorFieldSchema:
    name: str
    field_type: BehaviorFieldType
    role: BehaviorFieldRole
    required: bool
    merge_strategy: BehaviorMergeStrategy
    description: str

    def __post_init__(self) -> None:
        if not _FIELD_NAME.fullmatch(self.name):
            raise BehaviorSchemaError("behavior schema field name must use lowercase snake_case")
        object.__setattr__(self, "field_type", BehaviorFieldType(self.field_type))
        object.__setattr__(self, "role", BehaviorFieldRole(self.role))
        object.__setattr__(self, "merge_strategy", BehaviorMergeStrategy(self.merge_strategy))
        if not isinstance(self.required, bool):
            raise BehaviorSchemaError("behavior schema field required must be boolean")
        if not isinstance(self.description, str) or not self.description.strip():
            raise BehaviorSchemaError("behavior schema field description must be non-empty")
        if self.role in {BehaviorFieldRole.ADDRESS, BehaviorFieldRole.SYSTEM} and (
            not self.required or self.merge_strategy is not BehaviorMergeStrategy.IMMUTABLE
        ):
            raise BehaviorSchemaError("behavior address and system fields must be required and immutable")


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
            raise BehaviorSchemaError(f"{kind.value} schema path does not match the confirmed behavior tree")
        path = PurePosixPath(self.path_template)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".md":
            raise BehaviorSchemaError("behavior schema path template is unsafe")
        names = tuple(field.name for field in self.fields)
        if not names or len(names) != len(set(names)):
            raise BehaviorSchemaError("behavior schema fields must be non-empty and unique")
        if not any(field.role is BehaviorFieldRole.CONTENT for field in self.fields):
            raise BehaviorSchemaError("behavior schema must declare content fields")
        address_names = tuple(field.name for field in self.fields if field.role is BehaviorFieldRole.ADDRESS)
        expected_address_names = {
            BehaviorKind.EVENT: ("event_date", "event_name", "started_at"),
            BehaviorKind.OUTCOME: ("event_date", "event_name", "event_uri"),
            BehaviorKind.EPISODE: ("episode_date", "episode_name", "started_at"),
        }[kind]
        if address_names != expected_address_names:
            raise BehaviorSchemaError("behavior schema address fields do not match its path")
        if self.operation_mode is BehaviorOperationMode.ADD_ONLY and any(
            field.merge_strategy is BehaviorMergeStrategy.APPEND for field in self.fields
        ):
            raise BehaviorSchemaError("add-only behavior schema cannot contain append fields")

    @property
    def field_map(self) -> dict[str, BehaviorFieldSchema]:
        return {field.name: field for field in self.fields}


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
    "BehaviorMergeStrategy",
    "BehaviorOperationMode",
    "BehaviorSchemaError",
    "BehaviorSchemaMaterialization",
    "BehaviorTypeSchema",
]
