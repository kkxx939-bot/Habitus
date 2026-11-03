"""行为融合层的异常谱系。

容量错误是融合错误的子类：调用方按 ``BehaviorFusionError`` 兜底时，不应该因为"这批产物
太大"而漏网。观测层曾经把两者放在两条独立继承链上，代价是调小配置就能让既有数据读不出来。
"""

from __future__ import annotations

from habitus.foundation.lifecycle import DeterministicFailure


class BehaviorFusionError(ValueError):
    """融合判断的结构、引用或不变量不满足契约。"""


class BehaviorFusionLimitError(DeterministicFailure, BehaviorFusionError):
    """融合产物超过显式容量边界。同样的输入再做一遍也超，所以是确定性失败（26 号清单 2.4.4）。"""


class BehaviorFusionRefusedError(DeterministicFailure, BehaviorFusionError):
    """模型对这一份材料必然拒绝（输入超长、触发内容安全）：同一份材料再问几遍也一样，所以是确定性失败。"""


__all__ = [
    "BehaviorFusionError",
    "BehaviorFusionLimitError",
    "BehaviorFusionRefusedError",
]
