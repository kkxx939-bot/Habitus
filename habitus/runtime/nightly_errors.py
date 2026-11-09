"""夜线 Job 账的错误（单独一个模块：读它的预测层白天那一拍不必依赖整个 Job 账，避免环形导入）。"""

from __future__ import annotations


class NightJobsError(ValueError):
    """Job 账坏了或读不了（格式不对、越界、文件系统出错）：明确报出，不当成"没跑过"或"没跑完"。"""


__all__ = ["NightJobsError"]
