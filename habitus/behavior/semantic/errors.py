"""L0/L1 刷新的失败分类：超限是确定性的（同样的输入再试也超），其余按可重试处理。"""

from __future__ import annotations

from habitus.foundation.lifecycle import DeterministicFailure


class BehaviorSemanticRefreshError(RuntimeError):
    """生成结果无法安全发布。"""


class BehaviorOverviewEmptyError(BehaviorSemanticRefreshError):
    """模型返回了空文本：可重试。"""


class BehaviorOverviewHeadMissingError(BehaviorSemanticRefreshError):
    """概览没有开头一段，派生不出 L0：可重试（模型输出的问题）。"""


class BehaviorOverviewLimitError(DeterministicFailure, ValueError):
    """条数、提示词、概览超出配置上限：同样的目录再试也超，不重试（26 号清单 2.4.1）。"""


__all__ = [
    "BehaviorOverviewEmptyError",
    "BehaviorOverviewHeadMissingError",
    "BehaviorOverviewLimitError",
    "BehaviorSemanticRefreshError",
]
