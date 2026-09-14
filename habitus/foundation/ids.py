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


def canonical_text_identity(value: object, field_name: str) -> str:
    """把一句**自由文本**归一成可比较的身份：NFC + casefold + 折叠空白。

    与 ``canonical_path_identity`` 的区别是**用途**，不是强弱：那一个是给会变成目录名的东西用的，
    所以要拒 ``/ \\ : < > " | ? *``、拒以空格或句点结尾、拒 Windows 保留名。一句人话里出现这些
    再正常不过（"冰箱里有菜/水果"、"明天 9:00 要复诊"、"灯泡买好了."），拿那一个去校验，等于让
    模型写出一句普通的话就把流程卡死。

    这个函数对任何非空文本都**有返回值**，不抛——它的调用方是"两句话是不是同一句"这种比较，
    不是"这个名字能不能当目录"。空白折叠是必要的：同一句话被重新渲染时多一个空格不该变成另一条。
    """

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be text")
    folded = unicodedata.normalize("NFC", " ".join(value.split()).casefold())
    if not folded:
        raise ValueError(f"{field_name} must contain something other than whitespace")
    return folded


def same_path_identity(left: object, right: object, field_name: str) -> bool:
    return canonical_path_identity(left, field_name) == canonical_path_identity(
        right,
        field_name,
    )


__all__ = [
    "canonical_path_identity",
    "canonical_text_identity",
    "require_safe_path_segment",
    "same_path_identity",
]
