"""核心工具里的标识。"""

from __future__ import annotations

import unicodedata

_WINDOWS_RESERVED_STEMS = {
    "aux",
    "clock$",
    "con",
    "conin$",
    "conout$",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
    *(f"com{index}" for index in "¹²³"),
    *(f"lpt{index}" for index in "¹²³"),
}


def require_safe_path_segment(value: object, field_name: str) -> str:
    """返回无法逃逸目标父目录的安全标识。"""

    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or value.endswith((" ", "."))
        or any(character in '<>:"|?*' or ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{field_name} must be one safe non-empty path segment")
    portable_stem = unicodedata.normalize("NFC", value).casefold().split(".", 1)[0]
    if portable_stem in _WINDOWS_RESERVED_STEMS:
        raise ValueError(f"{field_name} must be one safe non-empty path segment")
    return value


def canonical_path_identity(value: object, field_name: str) -> str:
    """把会落到文件系统的逻辑身份归一为跨平台唯一形式。"""

    source = require_safe_path_segment(value, field_name)
    normalized = unicodedata.normalize("NFC", source).casefold()
    normalized = unicodedata.normalize("NFC", normalized)
    return require_safe_path_segment(normalized, field_name)


def same_path_identity(left: object, right: object, field_name: str) -> bool:
    return canonical_path_identity(left, field_name) == canonical_path_identity(
        right,
        field_name,
    )


__all__ = [
    "canonical_path_identity",
    "require_safe_path_segment",
    "same_path_identity",
]
