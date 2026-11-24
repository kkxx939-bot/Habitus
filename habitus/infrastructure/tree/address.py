"""``habitus://`` 地址的公共部分：协议名、路径段的百分号编解码、地址与路径段互转。

地址与树根下的相对路径一一对应：``habitus://<段>/<段>/...``，每一段是一个目录名或文件名。哪些路径是合法节点由各领域
自己判（记忆认四类卡片，行为认事件与空白），这里只管协议和编码，不认识任何领域类型。
"""

from __future__ import annotations

from urllib.parse import unquote

#: 一棵 habitus 树只有这一个协议名（全仓只在这里定义）。
SCHEME = "habitus"
PREFIX = f"{SCHEME}://"

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_UNRESERVED_ASCII = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")


class TreeAddressError(ValueError):
    """字符串不是规范的 ``habitus://`` 地址。"""


def split_uri(uri: str) -> tuple[str, ...]:
    """把 ``habitus://a/b`` 拆成已解码的路径段；根是空元组。不判节点是否合法。"""

    if not isinstance(uri, str):
        raise TypeError("habitus URI must be a string")
    if not uri or uri != uri.strip():
        raise TreeAddressError("habitus URI must be non-empty without surrounding whitespace")
    if not uri.startswith(PREFIX):
        raise TreeAddressError(
            f"habitus URI must start with '{PREFIX}' (addresses written by an older storage layout are not migrated; "
            "start from an empty storage root)"
        )
    raw_path = uri[len(PREFIX) :]
    if raw_path.endswith("/"):
        raise TreeAddressError("habitus URI must not contain a trailing slash")
    raw_segments = tuple(raw_path.split("/")) if raw_path else ()
    if any(not segment for segment in raw_segments):
        raise TreeAddressError("habitus URI contains an empty path segment")
    return tuple(decode_segment(segment) for segment in raw_segments)


def format_uri(segments: tuple[str, ...]) -> str:
    """已解码的路径段拼成唯一的规范地址：Unicode 保持原文，保留字符写大写百分号编码。"""

    return PREFIX + "/".join(encode_segment(segment) for segment in segments)


def decode_segment(value: str) -> str:
    index = 0
    while index < len(value):
        if value[index] != "%":
            index += 1
            continue
        if index + 2 >= len(value) or value[index + 1] not in _HEX_DIGITS or value[index + 2] not in _HEX_DIGITS:
            raise TreeAddressError("habitus URI contains malformed percent encoding")
        index += 3
    try:
        decoded = unquote(value, encoding="utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise TreeAddressError("habitus URI contains invalid UTF-8 percent encoding") from exc
    if not decoded:
        raise TreeAddressError("habitus URI contains an empty decoded segment")
    return decoded


def encode_segment(value: str) -> str:
    if not value:
        raise TreeAddressError("habitus URI segments must be non-empty strings")
    encoded: list[str] = []
    for character in value:
        if character in _UNRESERVED_ASCII or ord(character) >= 128:
            encoded.append(character)
        else:
            encoded.extend(f"%{byte:02X}" for byte in character.encode("utf-8"))
    return "".join(encoded)


__all__ = ["PREFIX", "SCHEME", "TreeAddressError", "decode_segment", "encode_segment", "format_uri", "split_uri"]
