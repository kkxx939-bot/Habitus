"""模型输出的文本预清洗。

装配层要先把一行洗干净，再判断它有没有内容——模型输出里夹一个零宽字符不能让整次调用失败。
洗不出内容的那一项由调用方丢弃并留信号。

与 ``schema/fields.py::line`` 的关系是**单向**的：清洗后的产物必定能通过 ``line()``，但两者
不是同一个变换（``line()`` 只去首尾空白、保留内部连续空白、遇控制字符直接抛）。
"""

from __future__ import annotations


def clean_line(value: object) -> str:
    """去不可打印字符、折叠空白、去首尾；非文本得空串。"""

    if not isinstance(value, str):
        return ""
    printable = "".join(character if character.isprintable() else " " for character in value)
    return " ".join(printable.split())


__all__ = ["clean_line"]
