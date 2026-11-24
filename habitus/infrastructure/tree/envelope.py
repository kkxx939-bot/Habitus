"""树上 L2 文件的外壳：正文 + 文件末尾唯一一个 ``HABITUS_FIELDS`` 注释（严格 JSON）。

记忆卡片、事件与空白、习惯与提醒记录用同一个注释名和同一个类型键 ``type``；注释里还有哪些键由各领域给（卡片多 ``links``、``backlinks``）。
这里只做切分与严格解析：唯一注释、拒重复键、拒非法常量、``--`` 转义、时刻与版本的读法。字段对不对、正文是不是由字段渲染出来、
重新编码是否逐字节相等，由各领域的编解码器在这之上判。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any, NoReturn

from habitus.foundation.validation import is_int

MARKER = "\n<!-- HABITUS_FIELDS\n"
FOOTER = "\n-->\n"
#: 注释顶层的类型键：值是文档的种类（``profile``、``preference``、``entity``、``tool``、``occurrence``、``gap``、``habit``、
#: ``reminder``）。
TYPE_KEY = "type"


class TreeEnvelopeError(ValueError):
    """文件外壳不合格：注释缺失或重复、注释不是严格 JSON、键集不对。"""


def require_plain_body(body: str) -> None:
    """正文里不许出现注释的开头标记，否则读回时切不出唯一注释。"""

    if MARKER in body:
        raise TreeEnvelopeError("Markdown body contains the reserved metadata marker")


def encode(body: str, metadata: Mapping[str, Any]) -> str:
    """正文 + 注释。JSON 按键排序、缩进 2；HTML 注释不能含 ``--``，只在 JSON 文本里转义成 ``\\u002d\\u002d``。"""

    payload = json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    return body + MARKER + payload.replace("--", "\\u002d\\u002d") + FOOTER


def decode(raw: str, *, keys: frozenset[str]) -> tuple[str, dict[str, Any]]:
    """切出正文与注释并严格解析；注释顶层键必须恰好是 ``keys``，类型键必须是字符串、``fields`` 必须是对象。"""

    if raw.count(MARKER) != 1 or not raw.endswith(FOOTER):
        raise TreeEnvelopeError("document must contain one terminal HABITUS_FIELDS comment")
    body, _separator, rest = raw.partition(MARKER)
    try:
        metadata = json.loads(rest[: -len(FOOTER)], object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (json.JSONDecodeError, TreeEnvelopeError) as exc:
        raise TreeEnvelopeError("document metadata is not strict JSON") from exc
    if not isinstance(metadata, dict) or set(metadata) != keys:
        raise TreeEnvelopeError("document metadata has an invalid shape")
    if not isinstance(metadata[TYPE_KEY], str):
        raise TreeEnvelopeError("document type must be a string")
    fields = metadata["fields"]
    if not isinstance(fields, dict) or any(not isinstance(name, str) for name in fields):
        raise TreeEnvelopeError("document fields must be an object")
    return body, metadata


def format_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def parse_timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise TreeEnvelopeError(f"document {field_name} must be a timestamp string")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TreeEnvelopeError(f"document {field_name} is not a valid ISO timestamp") from exc


def parse_revision(value: object) -> int:
    if not is_int(value):
        raise ValueError("document revision must be a positive integer")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TreeEnvelopeError("document metadata contains a duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise TreeEnvelopeError(f"document metadata contains an invalid JSON constant: {value}")


__all__ = [
    "FOOTER",
    "MARKER",
    "TYPE_KEY",
    "TreeEnvelopeError",
    "decode",
    "encode",
    "format_timestamp",
    "parse_revision",
    "parse_timestamp",
    "require_plain_body",
]
