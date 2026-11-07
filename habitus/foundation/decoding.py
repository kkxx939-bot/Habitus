"""落盘读回字段的公共解码：在构造值对象之前确认类型，报错文字由调用方给。

调用方传入的文字照抄被构造值对象自身的检查，读回坏数据和直接构造坏值对象报同一句话。
"""

from __future__ import annotations

from habitus.foundation.validation import is_int


def int_field(value: object, message: str) -> int:
    """整数（不是 ``bool``）；否则抛 ``ValueError(message)``。"""

    if not is_int(value):
        raise ValueError(message)
    return value


def text_field(value: object, message: str) -> str:
    """字符串；否则抛 ``ValueError(message)``。"""

    if not isinstance(value, str):
        raise ValueError(message)
    return value


def optional_text_field(value: object, message: str) -> str | None:
    """``None`` 或字符串；否则抛 ``ValueError(message)``。"""

    if value is not None and not isinstance(value, str):
        raise ValueError(message)
    return value


__all__ = ["int_field", "optional_text_field", "text_field"]
