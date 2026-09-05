"""单一 YAML 内的多厂商秘密凭据注册表。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from habitus.config.loader import ConfigError, strict_object

_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_FIELD = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_MAX_SECRET_CHARS = 16_384
_EMPTY_CREDENTIAL: Mapping[str, str] = MappingProxyType({})


@dataclass(frozen=True, repr=False)
class CredentialRegistry:
    """保存多个具名凭据；秘密值不会进入对象 repr。"""

    entries: Mapping[str, Mapping[str, str]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.entries, Mapping):
            raise TypeError("credentials must be an object")
        normalized: dict[str, Mapping[str, str]] = {}
        for raw_name, raw_fields in self.entries.items():
            name = _credential_name(raw_name)
            if name in normalized:
                raise ValueError("credentials contain duplicate normalized names")
            normalized[name] = MappingProxyType(_credential_fields(raw_fields, name=name))
        object.__setattr__(self, "entries", MappingProxyType(normalized))

    @classmethod
    def from_mapping(cls, value: object) -> CredentialRegistry:
        data = strict_object(value, path="config.credentials")
        return cls(entries=data)  # type: ignore[arg-type]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.entries))

    @property
    def contains_secret_values(self) -> bool:
        return any(value for fields in self.entries.values() for value in fields.values())

    def resolve(self, reference: str) -> Mapping[str, str]:
        """按规范引用返回只读秘密字段；空引用表示不需要凭据。"""

        name = _optional_reference(reference)
        if not name:
            return _EMPTY_CREDENTIAL
        credential = self.entries.get(name)
        if credential is None:
            raise ConfigError(f"credential reference does not exist: {name}")
        return credential

    def require_fields(self, reference: str, fields: set[str], *, path: str) -> None:
        """只校验引用和字段结构，不要求示例模板中的秘密已经填写。"""

        name = _optional_reference(reference)
        if not name:
            return
        credential = self.resolve(name)
        missing = sorted(fields - set(credential))
        if missing:
            raise ConfigError(f"'{path}' credential '{name}' is missing fields: {missing}")

    def __repr__(self) -> str:
        return f"CredentialRegistry(names={self.names!r})"


def _credential_name(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("credential names must be strings")
    normalized = value.strip().lower()
    if _NAME.fullmatch(normalized) is None:
        raise ValueError("credential names must be normalized lowercase identifiers")
    return normalized


def _optional_reference(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("credential reference must be a string")
    normalized = value.strip().lower()
    if normalized and _NAME.fullmatch(normalized) is None:
        raise ValueError("credential reference must be a normalized lowercase identifier")
    return normalized


def _credential_fields(value: object, *, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"credential '{name}' must be an object")
    result: dict[str, str] = {}
    for raw_field, raw_secret in value.items():
        if not isinstance(raw_field, str):
            raise TypeError(f"credential '{name}' field names must be strings")
        field_name = raw_field.strip().lower()
        if _FIELD.fullmatch(field_name) is None:
            raise ValueError(f"credential '{name}' contains an invalid field name")
        if field_name in result:
            raise ValueError(f"credential '{name}' contains duplicate normalized fields")
        if not isinstance(raw_secret, str):
            raise TypeError(f"credential '{name}.{field_name}' must be a string")
        if raw_secret != raw_secret.strip():
            raise ValueError(f"credential '{name}.{field_name}' cannot contain surrounding whitespace")
        if any(character in raw_secret for character in "\x00\r\n"):
            raise ValueError(f"credential '{name}.{field_name}' cannot contain control characters")
        if len(raw_secret) > _MAX_SECRET_CHARS:
            raise ValueError(f"credential '{name}.{field_name}' exceeds the size limit")
        result[field_name] = raw_secret
    if not result:
        raise ValueError(f"credential '{name}' must contain at least one field")
    return result


__all__ = ["CredentialRegistry"]
