"""行为观测清洗层的异常谱系。

容量错误必须是观测错误的子类：调用方按 ``BehaviorObservationError`` 兜底时，不应该因为
"这条超长"而漏网。曾经把两者放在两条独立继承链上，结果是把配置调小就能让既有耐久数据
读不出来，并连带整个枚举失败。
"""

from __future__ import annotations


class BehaviorObservationError(ValueError):
    """行为观测的身份、内容或耐久文件不满足契约。"""


class BehaviorObservationLimitError(BehaviorObservationError):
    """行为观测交付超过显式容量边界。"""


class BehaviorObservationProtocolError(BehaviorObservationError):
    """上游载荷不满足其声明协议的结构约束。"""


__all__ = [
    "BehaviorObservationError",
    "BehaviorObservationLimitError",
    "BehaviorObservationProtocolError",
]
