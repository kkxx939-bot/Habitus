"""夜里待处理的堆（裁定 67；裁定 94 第 3、4 条）：白天出的错误先落「错误」，重来所需的那一份留在这里，当晚线最前面的前置处理
统一看一次（能重试的重试一次），看完就删。

``model``：一项的形状与进堆的口（``RetryBacklog``）；``store``：落盘、按天收；``stage``：每晚的编排，对每种项调注入的处理器。
处理器跟着它试的东西住：会话轮的在会话 lane（``fusion/lanes/session/lane.py:TurnHandler``），链的在归约
（``reduction/failed.py:ChainHandler``）——本包不依赖它们，它们依赖本包，behavior 子包之间不成环。组合根在夜批最前面装配、调用。
"""

from habitus.behavior.retry.model import (
    Admission,
    NoRetryBacklog,
    RetryBacklog,
    RetryCause,
    RetryError,
    RetryItem,
    RetryKind,
    retry_id,
)
from habitus.behavior.retry.stage import (
    Attempt,
    Reduction,
    RetryBranch,
    RetryCheckpoint,
    RetryHandler,
    RetryReport,
    RetryStage,
)
from habitus.behavior.retry.store import RetryBacklogStore, RetryBatch

__all__ = [
    "Admission",
    "Attempt",
    "NoRetryBacklog",
    "RetryBacklog",
    "RetryBacklogStore",
    "RetryBatch",
    "Reduction",
    "RetryBranch",
    "RetryCheckpoint",
    "RetryHandler",
    "RetryReport",
    "RetryStage",
    "RetryCause",
    "RetryError",
    "RetryItem",
    "RetryKind",
    "retry_id",
]
