"""预测层的错误：本层把所有失败归一成一个类型，不让下层的异常直接漏给调用方。"""

from __future__ import annotations


class ForesightError(Exception):
    """装配证据包时的任何失败。"""


__all__ = ["ForesightError"]
