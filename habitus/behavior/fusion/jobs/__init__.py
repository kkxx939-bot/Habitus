"""融合作业的耐久队列。

严格串行、两阶段：``QUEUED → RUNNING → STAGED → COMMITTED``。``STAGED`` 是"模型已经调用过、判断已经耐久
暂存"的检查点，越过它之后任何重放都不再碰模型——因为融合不是纯函数，重跑一次就是另一批判断，
落盘之后就成了两套互不相认的记录。
"""

from habitus.behavior.fusion.jobs.model import (
    FUSION_JOB_ERROR_MAX_CHARS,
    FUSION_JOB_SCHEMA,
    BehaviorFusionJob,
    BehaviorFusionJobBlockedError,
    BehaviorFusionJobConfig,
    BehaviorFusionJobError,
    BehaviorFusionJobLease,
    BehaviorFusionJobLeaseLostError,
    BehaviorFusionJobNotReadyError,
    BehaviorFusionJobStatus,
    BehaviorFusionQueueSnapshot,
    StagedFusion,
)
from habitus.behavior.fusion.jobs.store import BehaviorFusionJobStore

__all__ = [
    "FUSION_JOB_ERROR_MAX_CHARS",
    "FUSION_JOB_SCHEMA",
    "BehaviorFusionJob",
    "BehaviorFusionJobBlockedError",
    "BehaviorFusionJobConfig",
    "BehaviorFusionJobError",
    "BehaviorFusionJobLease",
    "BehaviorFusionJobLeaseLostError",
    "BehaviorFusionJobNotReadyError",
    "BehaviorFusionJobStatus",
    "BehaviorFusionJobStore",
    "BehaviorFusionQueueSnapshot",
    "StagedFusion",
]
