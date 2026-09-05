"""时间预测树的错误类型。"""

from __future__ import annotations


class PredictionTreeError(ValueError):
    """输入或不变量被破坏；本层拒绝带病产出。"""


class PredictionTreeStoreError(RuntimeError):
    """预测树无法安全读写或发布。"""


__all__ = ["PredictionTreeError", "PredictionTreeStoreError"]
