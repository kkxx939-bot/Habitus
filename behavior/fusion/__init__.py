"""观测片段到行为判断的语义融合层。

本层是行为链上唯一调用模型的一环：观测清洗是无模型的确定性归一，判断的落盘是确定性的身份与
事务，中间这一步负责"看完这段观测，我判断这里发生了什么"。

## 产物是判断，不是事实

融合不产出"一件已经发生完的事"，而是**我们在某个时刻对一段观测做出的判断**。同一段行为在不同
时刻可以有不同判断，它们都为真，只是知道的时间不同；后来的判断不修改先前的判断，只新增一条指向
它。三层递减由两个字段决定——``behavior`` 与 ``goal`` 的空与非空——而不是三种并列的结构。

## 本层不碰行为树

判断到行为树文档（事件、结果、长期事件）的归约属于**另一层**：它需要事后才有的信息（开空调的
结果要等室温变化，长期事件要看完整段），节奏与融合相反。把它塞进融合，融合就永远完不了。
"""

from behavior.fusion.assembly import assemble_judgement_batch
from behavior.fusion.config import BehaviorFusionConfig
from behavior.fusion.derivation import (
    FUSION_IMPLEMENTATION_VERSION,
    FUSION_VERSION,
    DurableFact,
    DurableJudgement,
    derive_judgements,
    judgement_payload,
    without_unresolvable_relations,
)
from behavior.fusion.enqueue import (
    DEFAULT_QUIET_PERIOD_SECONDS,
    BehaviorFusionEnqueuer,
    BehaviorFusionEnqueueResult,
)
from behavior.fusion.errors import BehaviorFusionError, BehaviorFusionLimitError
from behavior.fusion.jobs import (
    BehaviorFusionJob,
    BehaviorFusionJobBlockedError,
    BehaviorFusionJobConfig,
    BehaviorFusionJobError,
    BehaviorFusionJobLease,
    BehaviorFusionJobLeaseLostError,
    BehaviorFusionJobNotReadyError,
    BehaviorFusionJobStatus,
    BehaviorFusionJobStore,
    BehaviorFusionQueueSnapshot,
    StagedFusion,
)
from behavior.fusion.judgement import (
    BehaviorClaim,
    BehaviorFact,
    BehaviorJudgement,
    BehaviorJudgementBatch,
    JudgementLink,
    JudgementRelation,
    JudgementStatus,
    JudgementStatusBasis,
)
from behavior.fusion.prompt import (
    FUSION_PROMPT_VERSION,
    FUSION_SYSTEM_PROMPT,
    render_context_judgements,
    render_fragments,
)
from behavior.fusion.receipt import (
    RECEIPT_SCHEMA_VERSION,
    BehaviorFusionReceipt,
    build_fusion_receipt,
    receipt_identity,
    segment_identity,
)
from behavior.fusion.receipt_store import BehaviorFusionReceiptStore
from behavior.fusion.result import BehaviorFusionResult
from behavior.fusion.runner import BehaviorFusionRunner, BehaviorFusionRunResult
from behavior.fusion.schema import JUDGEMENT_FUSION_JSON_SCHEMA, fusion_json_schema
from behavior.fusion.segmentation import BehaviorFusionSegment, segment_observations
from behavior.fusion.service import BehaviorJudgementFuser
from behavior.fusion.store import JUDGEMENT_SCHEMA_VERSION, BehaviorJudgementStore
from behavior.fusion.validation import unreadable_ratio, validate_judgement_batch

__all__ = [
    "DEFAULT_QUIET_PERIOD_SECONDS",
    "JUDGEMENT_FUSION_JSON_SCHEMA",
    "FUSION_IMPLEMENTATION_VERSION",
    "FUSION_PROMPT_VERSION",
    "FUSION_SYSTEM_PROMPT",
    "FUSION_VERSION",
    "JUDGEMENT_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "BehaviorClaim",
    "BehaviorJudgementFuser",
    "BehaviorFact",
    "BehaviorFusionConfig",
    "BehaviorFusionEnqueueResult",
    "BehaviorFusionEnqueuer",
    "BehaviorFusionError",
    "BehaviorFusionJob",
    "BehaviorFusionJobBlockedError",
    "BehaviorFusionJobConfig",
    "BehaviorFusionJobError",
    "BehaviorFusionJobLease",
    "BehaviorFusionJobLeaseLostError",
    "BehaviorFusionJobNotReadyError",
    "BehaviorFusionJobStatus",
    "BehaviorFusionJobStore",
    "BehaviorFusionLimitError",
    "BehaviorFusionQueueSnapshot",
    "BehaviorFusionReceipt",
    "BehaviorFusionReceiptStore",
    "BehaviorFusionResult",
    "BehaviorFusionRunResult",
    "BehaviorFusionRunner",
    "BehaviorFusionSegment",
    "BehaviorJudgement",
    "BehaviorJudgementBatch",
    "BehaviorJudgementStore",
    "DurableFact",
    "DurableJudgement",
    "JudgementLink",
    "JudgementRelation",
    "JudgementStatus",
    "JudgementStatusBasis",
    "StagedFusion",
    "assemble_judgement_batch",
    "build_fusion_receipt",
    "derive_judgements",
    "fusion_json_schema",
    "judgement_payload",
    "without_unresolvable_relations",
    "receipt_identity",
    "render_context_judgements",
    "render_fragments",
    "segment_identity",
    "segment_observations",
    "unreadable_ratio",
    "validate_judgement_batch",
]
