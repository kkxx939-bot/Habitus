"""分层提取和修复 JSON 语法，不臆造语义字段。"""

from __future__ import annotations

import ast
import importlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal

JSONParseMode = Literal[
    "strict",
    "code_fence",
    "extracted",
    "trailing_comma_repair",
    "json_repair",
    "python_literal_repair",
]

_CODE_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


class DuplicateJSONObjectKeyError(ValueError):
    """结构化输出中的同一对象重复声明了字段。"""


@dataclass(frozen=True)
class ParsedJSON:
    """解析后的值及可审计的语法修复说明。"""

    value: object
    mode: JSONParseMode

    @property
    def repaired(self) -> bool:
        return self.mode != "strict"


def parse_json_response(source: str, *, allow_repair: bool = True) -> ParsedJSON:
    """解析模型返回的 JSON，只修复语法并拒绝非 JSON 的 Python 值。"""

    if not isinstance(source, str) or not source.strip():
        raise ValueError("structured model response must be non-empty text")
    text = source.strip()
    parsed = _loads(text)
    if parsed is not _MISSING:
        return ParsedJSON(parsed, "strict")

    candidates: list[tuple[str, JSONParseMode]] = []
    fence = _CODE_FENCE.search(text)
    if fence:
        candidates.append((fence.group(1).strip(), "code_fence"))
    extracted = _extract_balanced_json(text)
    if extracted and all(extracted != candidate for candidate, _mode in candidates):
        candidates.append((extracted, "extracted"))
    for candidate, mode in candidates:
        parsed = _loads(candidate)
        if parsed is not _MISSING:
            return ParsedJSON(parsed, mode)

    if not allow_repair:
        raise ValueError("model response is not valid JSON")

    repair_sources = [candidate for candidate, _mode in candidates]
    if not repair_sources:
        raise ValueError("model response contains no JSON candidate to repair")
    for candidate in repair_sources:
        without_trailing_commas = _remove_trailing_commas(candidate)
        if without_trailing_commas != candidate:
            parsed = _loads(without_trailing_commas)
            if parsed is not _MISSING:
                return ParsedJSON(parsed, "trailing_comma_repair")

    dependency_candidates: list[str] = []
    for candidate in repair_sources:
        _reject_duplicate_python_literal_keys(candidate)
        try:
            value = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            dependency_candidates.append(candidate)
            continue
        if _is_safe_python_repair_value(value) and _preserves_container_root(candidate, value):
            return ParsedJSON(value, "python_literal_repair")

    for candidate in dependency_candidates:
        repaired = _repair_with_optional_dependency(candidate)
        if repaired is not _MISSING and _preserves_container_root(candidate, repaired):
            return ParsedJSON(repaired, "json_repair")
    raise ValueError("model response could not be repaired as JSON")


def _remove_trailing_commas(source: str) -> str:
    result: list[str] = []
    in_string = False
    escaped = False
    for index, character in enumerate(source):
        if in_string:
            result.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            result.append(character)
            continue
        if character == ",":
            lookahead = index + 1
            while lookahead < len(source) and source[lookahead].isspace():
                lookahead += 1
            if lookahead < len(source) and source[lookahead] in "}]":
                continue
        result.append(character)
    return "".join(result)


class _Missing:
    pass


_MISSING = _Missing()


def _loads(source: str) -> object | _Missing:
    try:
        value = json.loads(
            source,
            parse_constant=_reject_non_finite,
            object_pairs_hook=_unique_object,
        )
    except DuplicateJSONObjectKeyError:
        raise
    except (json.JSONDecodeError, ValueError):
        return _MISSING
    return value if _is_json_value(value) else _MISSING


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONObjectKeyError(f"duplicate JSON object key is not allowed: {key}")
        result[key] = value
    return result


def _reject_duplicate_python_literal_keys(source: str) -> None:
    """在 ``literal_eval`` 折叠字典前按 AST 的真实字符串键逐层查重。"""

    try:
        parsed = ast.parse(source, mode="eval")
    except SyntaxError:
        return
    for node in ast.walk(parsed):
        if not isinstance(node, ast.Dict):
            continue
        keys: set[str] = set()
        for key_node in node.keys:
            if key_node is None:
                continue
            try:
                key = ast.literal_eval(key_node)
            except (ValueError, TypeError):
                continue
            if not isinstance(key, str):
                continue
            if key in keys:
                raise DuplicateJSONObjectKeyError(f"duplicate JSON object key is not allowed: {key}")
            keys.add(key)


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _repair_with_optional_dependency(source: str) -> object | _Missing:
    try:
        module = importlib.import_module("json_repair")
    except ImportError:
        return _MISSING
    try:
        repair_source = _prepare_dependency_repair_source(source)
        wrapped = module.loads(
            f"[{repair_source}]",
            skip_json_loads=True,
            strict=True,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return _MISSING
    if not isinstance(wrapped, list) or len(wrapped) != 1:
        return _MISSING
    value = wrapped[0]
    return value if _is_json_value(value) else _MISSING


def _prepare_dependency_repair_source(source: str) -> str:
    """移除字符串外注释，并拒绝会改变值类型的 Python 专属容器语法。"""

    repaired: list[str] = []
    syntax: list[str] = []
    quote: str | None = None
    escaped = False
    line_comment = False
    block_comment = False
    index = 0
    while index < len(source):
        character = source[index]
        next_character = source[index + 1] if index + 1 < len(source) else ""
        if line_comment:
            if character in "\r\n":
                line_comment = False
                repaired.append(character)
                syntax.append(character)
            else:
                repaired.append(" ")
                syntax.append(" ")
            index += 1
            continue
        if block_comment:
            if character == "*" and next_character == "/":
                repaired.extend((" ", " "))
                syntax.extend((" ", " "))
                block_comment = False
                index += 2
                continue
            replacement = character if character in "\r\n" else " "
            repaired.append(replacement)
            syntax.append(replacement)
            index += 1
            continue
        if quote is not None:
            repaired.append(character)
            syntax.append(" ")
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            index += 1
            continue
        if character in {'"', "'"}:
            quote = character
            repaired.append(character)
            syntax.append(" ")
            index += 1
            continue
        if character == "/" and next_character == "/":
            repaired.extend((" ", " "))
            syntax.extend((" ", " "))
            line_comment = True
            index += 2
            continue
        if character == "/" and next_character == "*":
            repaired.extend((" ", " "))
            syntax.extend((" ", " "))
            block_comment = True
            index += 2
            continue
        if character == "#":
            repaired.append(" ")
            syntax.append(" ")
            line_comment = True
            index += 1
            continue
        repaired.append(character)
        syntax.append(character)
        index += 1

    if quote is not None or block_comment:
        raise ValueError("JSON repair candidate contains an unterminated string or comment")
    syntax_text = "".join(syntax)
    if "(" in syntax_text or ")" in syntax_text:
        raise ValueError("JSON repair candidate contains non-JSON container syntax")
    if re.search(r"(?<![\w])[+-]?\d[\w.+-]*\s*:", syntax_text):
        raise ValueError("JSON repair candidate contains an ambiguous unquoted object key")
    return "".join(repaired)


def _extract_balanced_json(source: str) -> str | None:
    start = next((index for index, character in enumerate(source) if character in "[{"), None)
    if start is None:
        return None
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    for index in range(start, len(source)):
        character = source[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
            continue
        if character in "[{":
            stack.append(character)
            continue
        if character in "]}":
            if not stack:
                return None
            opening = stack.pop()
            if (opening, character) not in {("[", "]"), ("{", "}")}:
                return None
            if not stack:
                return source[start : index + 1]
    return None


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, str | bool | int | float):
        return not isinstance(value, float) or math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


def _is_safe_python_repair_value(value: Any) -> bool:
    """只接受能由单引号等语法差异产生、不会改写 JSON 语义的 Python 字面量。"""

    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, str | int | float):
        return not isinstance(value, float) or math.isfinite(value)
    if isinstance(value, list):
        return all(_is_safe_python_repair_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_safe_python_repair_value(item) for key, item in value.items())
    return False


def _preserves_container_root(source: str, value: object) -> bool:
    """修复前后保持显式对象或数组的根类型，拒绝 set/tuple 被改造成数组。"""

    stripped = source.lstrip()
    if stripped.startswith("{"):
        return isinstance(value, dict)
    if stripped.startswith("["):
        return isinstance(value, list)
    return True


__all__ = [
    "DuplicateJSONObjectKeyError",
    "JSONParseMode",
    "ParsedJSON",
    "parse_json_response",
]
