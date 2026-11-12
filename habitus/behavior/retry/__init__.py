"""重试堆（裁定 67；裁定 94 第 3、4 条）：白天出的错误先落「错误」，重来所需的那一份留在这里；模型队列就近试属于还没封口
的日子的可重试项，换代里的重试收尾（结算之前）把那一天剩下的丢弃、撤回、计数。

``model``：一项的形状与进堆的口（``RetryBacklog``）；``store``：落盘、按天收；``stage``：就近试与重试收尾的编排，对每种项调
注入的处理器。处理器跟着它试的东西住：会话轮的在会话 lane（``fusion/lanes/session/lane.py:TurnHandler``），链的在归约
（``reduction/failed.py:ChainHandler``）——本包不依赖它们，它们依赖本包，behavior 子包之间不成环。组合根装配。
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
    "RetryHandler",
    "RetryReport",
    "RetryStage",
    "RetryCause",
    "RetryError",
    "RetryItem",
    "RetryKind",
    "retry_id",
]
