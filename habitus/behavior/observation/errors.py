"""行为观测的异常谱系。

容量错误必须是观测错误的子类：调用方按 ``BehaviorObservationError`` 兜底时，不应该因为
"这条超长"而漏网，否则把配置调小就能让既有耐久数据读不出来，并连带整个枚举失败。
"""

from __future__ import annotations


class BehaviorObservationError(ValueError):
    """行为观测的身份、内容或耐久文件不满足契约。"""


class BehaviorObservationLimitError(BehaviorObservationError):
    """行为观测交付超过显式容量边界。"""


__all__ = [
    "BehaviorObservationError",
    "BehaviorObservationLimitError",
]
