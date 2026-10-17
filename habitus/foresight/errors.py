"""预测层的错误：本层把所有失败归一成一个类型，不让下层的异常直接漏给调用方。"""

from __future__ import annotations


class ForesightError(Exception):
    """预测层里的任何失败：账本记录不合法、读不出来、依赖的派生树读不到。"""


__all__ = ["ForesightError"]
