"""边界解码用的公共类型判定。

只给系统边界用：从磁盘、HTTP、模型输出、外部服务、配置读进来的原始值，在构造值对象之前用这里的判定
确认类型。内部调用之间的类型由 mypy 与 pyright 静态保证，不再做运行时类型检查（W16）。

每个判定只回答"是不是这种类型"，不抛错：调用方保留自己的异常类型和报错文字，调用方式是
``if not is_int(value) or value < 0: raise ValueError(...)``。架构闸把这些判定与 ``isinstance`` 同等计数，
内部模块里写它们一样会让基线变大。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TypeGuard


def is_int(value: object) -> TypeGuard[int]:
    """整数，但不是 ``bool``（``bool`` 是 ``int`` 的子类，JSON 的 ``true`` 不能当 1 用）。"""

    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value: object) -> TypeGuard[int | float]:
    """整数或浮点数，但不是 ``bool``。"""

    return isinstance(value, int | float) and not isinstance(value, bool)


def is_nonblank_str(value: object) -> TypeGuard[str]:
    """去掉首尾空白后仍有内容的字符串。"""

    return isinstance(value, str) and bool(value.strip())


def is_aware_datetime(value: object) -> TypeGuard[datetime]:
    """带时区偏移的 ``datetime``（``utcoffset()`` 不是 ``None``）。"""

    return isinstance(value, datetime) and value.utcoffset() is not None


def is_str_list(value: object) -> TypeGuard[list[str] | tuple[str, ...]]:
    """每一项都是字符串的 ``list`` 或 ``tuple``（JSON 数组读进来是 ``list``）。"""

    return isinstance(value, list | tuple) and all(isinstance(item, str) for item in value)


def is_date_only(value: object) -> TypeGuard[date]:
    """纯日期 ``date``，不是 ``datetime``（``datetime`` 是 ``date`` 的子类）。"""

    return isinstance(value, date) and not isinstance(value, datetime)


TYPE_PREDICATES = ("is_int", "is_number", "is_nonblank_str", "is_str_list", "is_aware_datetime", "is_date_only")
"""架构闸按这组名字把调用当作类型检查计数，新增判定要同步加进来。"""

__all__ = [
    "TYPE_PREDICATES",
    "is_aware_datetime",
    "is_date_only",
    "is_int",
    "is_nonblank_str",
    "is_number",
    "is_str_list",
]
