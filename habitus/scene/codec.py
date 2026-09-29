"""语义树各支共用的落盘编码：Markdown 正文 + 终端 HTML 注释里的规范 JSON。

每支只有一种形状，所以这里不走 schema 注册表——正文由各支自己渲染，这里只管"正文与规范 JSON 怎么
拼成一个文件、怎么拆回来"。编码是规范化的：解码时把 JSON 段按规范形式重新序列化比对，一个字节不同
就是损坏，不做"尽力解析"。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from habitus.foundation.integrity import CanonicalSerializationError, canonical_json

_FOOTER = "\n-->\n"


class SceneRecordError(ValueError):
    """语义树记录的编码与我们自己的产物矛盾。"""


def encode_record(body: str, marker: str, payload: Mapping[str, Any]) -> str:
    """``body`` 必须以换行结尾；``marker`` 是这一支的注释头名（如 ``HABITUS_CONCEPT``）。"""

    if not isinstance(body, str) or not body.strip() or not body.endswith("\n"):
        raise SceneRecordError("record body must be non-empty text ending with a newline")
    _require_marker(marker)
    try:
        encoded = canonical_json(payload)
    except CanonicalSerializationError as exc:
        raise SceneRecordError("record payload is not canonically serializable") from exc
    return f"{body}\n<!-- {marker}\n{encoded}{_FOOTER}"


def decode_record(text: str, marker: str) -> tuple[str, dict[str, Any]]:
    """拆回 (正文, 规范 JSON 对象)。JSON 段必须已是规范形式，否则报损坏。"""

    _require_marker(marker)
    if not isinstance(text, str):
        raise SceneRecordError("record text must be a string")
    head = f"\n<!-- {marker}\n"
    index = text.rfind(head)
    if index < 0 or not text.endswith(_FOOTER):
        raise SceneRecordError(f"record is missing its {marker} footer")
    body = text[:index]
    if not body.strip() or not body.endswith("\n"):
        raise SceneRecordError("record body must be non-empty text ending with a newline")
    encoded = text[index + len(head) : -len(_FOOTER)]
    try:
        parsed = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise SceneRecordError("record footer is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise SceneRecordError("record footer must be a JSON object")
    try:
        canonical = canonical_json(parsed)
    except CanonicalSerializationError as exc:
        # ``json.loads`` 接受 NaN / Infinity 字面量，规范形式不接受：这是损坏，不是别的层的错。
        raise SceneRecordError("record footer holds a non-canonical value") from exc
    if canonical != encoded:
        raise SceneRecordError("record footer is not in canonical form")
    return body, parsed


def _require_marker(marker: object) -> None:
    if not isinstance(marker, str) or not marker or not marker.isascii() or not marker.replace("_", "").isalnum():
        raise SceneRecordError("record marker must be an ASCII identifier")


__all__ = ["SceneRecordError", "decode_record", "encode_record"]
