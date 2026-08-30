"""行为融合层的异常谱系。

容量错误是融合错误的子类：调用方按 ``BehaviorFusionError`` 兜底时，不应该因为"这批产物
太大"而漏网。观测层曾经把两者放在两条独立继承链上，代价是调小配置就能让既有数据读不出来。
"""

from __future__ import annotations


class BehaviorFusionError(ValueError):
    """融合判断的结构、引用或不变量不满足契约。"""


class BehaviorFusionLimitError(BehaviorFusionError):
    """融合产物超过显式容量边界。"""


class BehaviorFusionTruncatedError(BehaviorFusionError):
    """模型输出被长度截断，且重试仍截断——这是段容量与输出预算的错配，不是可重试的环境故障。"""


__all__ = ["BehaviorFusionError", "BehaviorFusionLimitError", "BehaviorFusionTruncatedError"]
