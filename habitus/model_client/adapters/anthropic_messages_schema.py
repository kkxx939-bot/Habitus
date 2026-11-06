"""把结构化输出与严格工具的 JSON Schema 换成 Anthropic Messages 能编译的子集。

Messages 协议的 ``output_config.format`` 与 ``strict`` 工具只接受 JSON Schema 的一个子集：对象必须
``additionalProperties: false``，不支持数值、字符串长度与多数数组约束，``minItems`` 只认 0 与 1，字符串 ``format``
只认固定几种，不支持递归和外部 ``$ref``；带了不支持的关键字服务端直接回 400。这里只做协议转换：

- 不支持的约束从发出去的 Schema 里拿掉，原样写进该节点的 ``description``（``{minLength: 1}``），让模型仍看得到；
- 没写 ``additionalProperties`` 的对象补 ``false``；``oneOf`` 改写成 ``anyOf``；带 ``$ref`` 的节点只留 ``$ref``；
- 写了 ``additionalProperties: true`` 或一个子 Schema（字典字段）、没声明 ``properties`` 的开放对象、带
  ``patternProperties`` / ``propertyNames`` 的对象、递归引用、外部或找不到的引用，协议表达不了，补成 ``false``
  会让模型只能交空对象、静默改了语义，所以直接报配置错误。

完整约束仍由结构化输出运行层（``model_client/structured.py``）按原 Schema 在客户端校验，这里不承担校验职责。
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from habitus.model_client.contracts import ModelConfigurationError

_SUPPORTED_FORMATS = frozenset(
    {"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"}
)
# 原样保留的关键字（子 Schema 另行递归）
_KEPT = frozenset({"type", "required", "enum", "const", "default", "description", "title"})
# 只是元数据、对生成没有意义的关键字，直接去掉
_DROPPED = frozenset({"$schema", "$id", "$comment"})
_DEFINITION_KEYS = ("$defs", "definitions")


def messages_output_schema(schema: Mapping[str, object]) -> dict[str, object]:
    """返回转换后的新 Schema，不改动入参。"""

    _reject_unexpressible_references(schema)
    return _convert(schema, path="$")


def _convert(node: object, *, path: str) -> dict[str, object]:
    if not isinstance(node, Mapping):
        raise ModelConfigurationError(f"structured output schema node {path} must be an object schema")
    if "$ref" in node:
        return {"$ref": node["$ref"], **_definitions(node, path)}
    result: dict[str, object] = {}
    moved: dict[str, object] = {}
    for key, value in node.items():
        if key in _KEPT:
            result[key] = value
        elif key == "properties":
            result[key] = _properties(value, path)
        elif key in _DEFINITION_KEYS:
            result.update(_definitions({key: value}, path))
        elif key == "items" and isinstance(value, Mapping):
            result[key] = _convert(value, path=f"{path}.items")
        elif key in {"anyOf", "allOf", "oneOf"}:
            result["allOf" if key == "allOf" else "anyOf"] = _variants(value, f"{path}.{key}")
        elif key == "format" and value in _SUPPORTED_FORMATS:
            result[key] = value
        elif key == "minItems" and value in (0, 1) and not isinstance(value, bool):
            result[key] = value
        elif key == "additionalProperties":
            if value is not False:
                raise ModelConfigurationError(
                    f"structured output schema {path}: additionalProperties other than false (open or typed maps) "
                    "cannot be expressed by the anthropic_messages protocol"
                )
            result[key] = False
        elif key not in _DROPPED:
            moved[key] = value
    if _is_object(node):
        if "additionalProperties" not in node and (
            "properties" not in node or "patternProperties" in node or "propertyNames" in node
        ):
            raise ModelConfigurationError(
                f"structured output schema {path}: an open object (no declared properties, or patternProperties / "
                "propertyNames) cannot be expressed by the anthropic_messages protocol"
            )
        result.setdefault("additionalProperties", False)
    if moved:
        result["description"] = _describe(result.get("description"), moved)
    return result


def _properties(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ModelConfigurationError(f"structured output schema {path}.properties must be an object")
    return {str(name): _convert(child, path=f"{path}.properties.{name}") for name, child in value.items()}


def _definitions(node: Mapping[str, object], path: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in _DEFINITION_KEYS:
        if key in node:
            value = node[key]
            if not isinstance(value, Mapping):
                raise ModelConfigurationError(f"structured output schema {path}.{key} must be an object")
            result[key] = {str(name): _convert(child, path=f"{path}.{key}.{name}") for name, child in value.items()}
    return result


def _variants(value: object, path: str) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise ModelConfigurationError(f"structured output schema {path} must be a non-empty array")
    return [_convert(item, path=f"{path}[{index}]") for index, item in enumerate(value)]


def _is_object(node: Mapping[str, object]) -> bool:
    declared = node.get("type")
    if declared == "object" or (isinstance(declared, list) and "object" in declared):
        return True
    return "properties" in node


def _describe(existing: object, moved: Mapping[str, object]) -> str:
    rendered = ", ".join(f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in moved.items())
    note = "{" + rendered + "}"
    return f"{existing}\n\n{note}" if isinstance(existing, str) and existing else note


def _reject_unexpressible_references(schema: Mapping[str, object]) -> None:
    """外部引用、找不到的引用和递归引用协议都编不了；在发请求前报清楚是哪一个。"""

    definitions: dict[str, object] = {}
    for key in _DEFINITION_KEYS:
        value = schema.get(key)
        if isinstance(value, Mapping):
            definitions.update({f"#/{key}/{name}": child for name, child in value.items()})
    graph = {target: _references(child) for target, child in definitions.items()}
    for reference in _references(schema) | {item for refs in graph.values() for item in refs}:
        if reference == "#":
            raise ModelConfigurationError("structured output schema is recursive through the root ($ref '#')")
        if not reference.startswith("#/"):
            raise ModelConfigurationError(f"structured output schema uses an external $ref: {reference}")
        if reference not in definitions:
            raise ModelConfigurationError(f"structured output schema uses an unknown $ref: {reference}")
    for start in graph:
        if _reaches(graph, start, start):
            raise ModelConfigurationError(f"structured output schema is recursive through {start}")


def _references(node: object) -> set[str]:
    found: set[str] = set()
    if isinstance(node, Mapping):
        reference = node.get("$ref")
        if isinstance(reference, str):
            found.add(reference)
        for key, value in node.items():
            if key not in _DEFINITION_KEYS:
                found |= _references(value)
    elif isinstance(node, list):
        for item in node:
            found |= _references(item)
    return found


def _reaches(graph: Mapping[str, set[str]], start: str, target: str) -> bool:
    pending = list(graph.get(start, ()))
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current == target:
            return True
        if current in seen:
            continue
        seen.add(current)
        pending.extend(graph.get(current, ()))
    return False


__all__ = ["messages_output_schema"]
