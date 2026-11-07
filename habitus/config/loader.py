"""严格读取 Habitus 的单一 YAML 配置文件。"""

from __future__ import annotations

import dataclasses
import difflib
import math
import re
import stat
import types
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TypeVar, Union, get_args, get_origin, get_type_hints

import yaml
from yaml.nodes import MappingNode

from habitus.foundation.validation import is_int, is_number
from habitus.infrastructure.store.filesystem import is_path_alias

_MAX_CONFIG_BYTES = 1024 * 1024
_ConfigValue = TypeVar("_ConfigValue")


class ConfigError(ValueError):
    """配置文件、配置对象或跨领域约束不合法。"""


class _StrictSafeLoader(yaml.SafeLoader):
    """在 SafeLoader 基础上拒绝重复映射键。"""


def load_config_object(path: str | Path) -> dict[str, object]:
    """有界读取 UTF-8 YAML 对象，不展开文本中的环境变量。"""

    try:
        requested = Path(path).expanduser().absolute()
    except TypeError as exc:
        raise ConfigError("config path must be a filesystem path") from exc
    if requested.suffix.casefold() != ".yaml":
        raise ConfigError("config file must use the .yaml suffix")
    if is_path_alias(requested):
        raise ConfigError("config file cannot be a symbolic link")
    try:
        metadata = requested.stat()
    except FileNotFoundError as exc:
        raise ConfigError(f"config file does not exist: {requested}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ConfigError("config path must identify a regular file")
    if metadata.st_size > _MAX_CONFIG_BYTES:
        raise ConfigError("config file exceeds the one-megabyte limit")
    try:
        payload = requested.read_bytes()
    except OSError as exc:
        raise ConfigError("failed to read config file") from exc
    if len(payload) > _MAX_CONFIG_BYTES:
        raise ConfigError("config file exceeds the one-megabyte limit")
    try:
        decoded = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ConfigError("config file must be UTF-8") from exc
    try:
        parsed = yaml.load(decoded, Loader=_StrictSafeLoader)
    except ConfigError:
        raise
    except yaml.MarkedYAMLError as exc:
        mark = exc.problem_mark
        location = ""
        if mark is not None:
            location = f" at line {mark.line + 1}, column {mark.column + 1}"
        raise ConfigError(f"config file contains invalid YAML{location}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError("config file contains invalid YAML") from exc
    _reject_non_finite_numbers(parsed, path="config")
    return strict_object(parsed, path="config")


def strict_object(value: object, *, path: str) -> dict[str, object]:
    """返回字符串键对象，拒绝 list、null 和其他宽松形式。"""

    if not isinstance(value, Mapping):
        raise ConfigError(f"'{path}' must be an object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ConfigError(f"'{path}' must contain non-empty string keys")
        result[key] = item
    return result


def strict_fields(
    value: object,
    *,
    path: str,
    allowed: set[str],
) -> dict[str, object]:
    """拒绝未声明字段，并为常见拼写错误给出确定性建议。"""

    result = strict_object(value, path=path)
    unknown = sorted(set(result) - allowed)
    if not unknown:
        return result
    messages: list[str] = []
    for field_name in unknown:
        matches = difflib.get_close_matches(field_name, sorted(allowed), n=1, cutoff=0.6)
        message = f"unknown config field '{path}.{field_name}'"
        if matches:
            message += f"; did you mean '{path}.{matches[0]}'?"
        messages.append(message)
    raise ConfigError("\n".join(messages))


def group_fields(
    model_type: type[object],
    value: object,
    path: str,
) -> dict[str, object]:
    """以 dataclass 声明字段作为唯一允许集。"""

    declared = getattr(model_type, "__dataclass_fields__", None)
    if not isinstance(declared, Mapping):
        raise TypeError("config groups must be dataclass types")
    return strict_fields(value, path=path, allowed=set(declared))


def construct_config(
    model_type: type[_ConfigValue],
    value: object,
    path: str,
) -> _ConfigValue:
    """严格校验字段后构造一个强类型配置值对象。"""

    data = group_fields(model_type, value, path)
    require_field_types(model_type, data, path=path)
    try:
        return model_type(**data)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid '{path}': {exc}") from exc


_SCALAR_CHECKS: dict[object, tuple[Callable[[object], bool], str, str]] = {
    bool: (lambda value: isinstance(value, bool), "a boolean", "booleans"),
    int: (is_int, "an integer", "integers"),
    float: (is_number, "a number", "numbers"),
    str: (lambda value: isinstance(value, str), "text", "texts"),
}


def require_field_types(model_type: type[object], data: Mapping[str, object], *, path: str) -> None:
    """YAML 读进来的标量字段按 dataclass 注解查类型，构造值对象之前就拒（W16：类型检查在边界）。

    只查注解（去掉 ``None`` 后）恰为 ``bool``/``int``/``float``/``str`` 或其 ``tuple[..., ...]`` 的字段；多成员联合、
    嵌套配置、枚举、``Literal``、映射由各自的解码或取值规则负责。``int`` 不收 ``bool``，``float`` 收整数不收 ``bool``。
    """

    hints = get_type_hints(model_type)
    for name, value in data.items():
        accepted = get_args(hints[name]) if get_origin(hints[name]) in (Union, types.UnionType) else (hints[name],)
        members = tuple(member for member in accepted if member is not type(None))
        if (value is None and len(members) < len(accepted)) or len(members) != 1:
            continue
        expected = members[0]
        if get_origin(expected) is tuple and get_args(expected)[1:] == (Ellipsis,):
            item_check = _SCALAR_CHECKS.get(get_args(expected)[0])
            if item_check is not None and not (isinstance(value, list | tuple) and all(map(item_check[0], value))):
                raise ConfigError(f"{path}.{name} must be a list of {item_check[2]}")
        elif expected in _SCALAR_CHECKS and not _SCALAR_CHECKS[expected][0](value):
            raise ConfigError(f"{path}.{name} must be {_SCALAR_CHECKS[expected][1]}")


def require_override_types(group: object, *, path: str, prefix: str = "") -> None:
    """覆盖项（``int | None`` / ``float | None``，``None`` = 用领域默认值）只查类型：值域归领域配置，见 ``construct_overrides``。

    整数还是数按字段的真实类型判（``typing.get_type_hints``），不看注解写成什么字符串。
    """

    hints = get_type_hints(type(group))
    for item in dataclasses.fields(group):  # type: ignore[arg-type]
        value = getattr(group, item.name)
        if not item.name.startswith(prefix) or value is None:
            continue
        accepted = get_args(hints[item.name]) or (hints[item.name],)
        if int in accepted and float not in accepted:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{path}.{item.name} must be an integer")
        elif isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{path}.{item.name} must be a number")


def construct_overrides(
    factory: Callable[..., _ConfigValue],
    overrides: Mapping[str, object],
    *,
    path: str,
    prefix: str = "",
) -> _ConfigValue:
    """用配置里写了的覆盖项构造一份领域配置；值域只由领域配置校验，拒了就包成指得出配置键的 ``ConfigError``。

    领域的报错只认它自己的字段名（``batch_size must ...``），这里把报错里出现的覆盖项还原成配置键
    （``config.behavior.kinds_batch_size``）；一个都认不出（例如几项之间互相矛盾、报错没点名写了的那项）时指到这一组。
    """

    try:
        return factory(**overrides)
    except (TypeError, ValueError) as exc:
        message = str(exc)
        named = [name for name in overrides if re.search(rf"\b{re.escape(name)}\b", message)]
        keys = ", ".join(f"'{path}.{prefix}{name}'" for name in named) or f"'{path}'"
        raise ConfigError(f"invalid {keys}: {message}") from exc


def required_field(data: Mapping[str, object], name: str, *, path: str) -> object:
    """读取必填字段，不使用假值判断篡改合法值。"""

    if name not in data:
        raise ConfigError(f"missing required config field '{path}.{name}'")
    return data[name]


def _construct_unique_mapping(
    loader: _StrictSafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    """在展平 YAML merge key 后仍拒绝任何重复键。"""

    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicated = key in result
        except TypeError as exc:
            raise ConfigError("config mapping keys must be scalar values") from exc
        if duplicated:
            raise ConfigError(f"config contains a duplicate YAML field: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


def _reject_non_finite_numbers(
    value: object,
    *,
    path: str,
    active: set[int] | None = None,
) -> None:
    """拒绝 YAML `.nan` 和无穷值，并防止递归 alias 无限遍历。"""

    if isinstance(value, float) and not math.isfinite(value):
        raise ConfigError(f"'{path}' cannot contain a non-finite number")
    if not isinstance(value, Mapping | Sequence) or isinstance(value, str | bytes):
        return
    marker = id(value)
    visited = set() if active is None else active
    if marker in visited:
        raise ConfigError(f"'{path}' cannot contain a recursive YAML alias")
    visited.add(marker)
    try:
        if isinstance(value, Mapping):
            for key, item in value.items():
                _reject_non_finite_numbers(item, path=f"{path}.{key}", active=visited)
        else:
            for index, item in enumerate(value):
                _reject_non_finite_numbers(item, path=f"{path}[{index}]", active=visited)
    finally:
        visited.remove(marker)


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


__all__ = [
    "ConfigError",
    "construct_config",
    "construct_overrides",
    "group_fields",
    "load_config_object",
    "require_override_types",
    "required_field",
    "strict_fields",
    "strict_object",
]
