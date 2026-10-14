"""事件融合：把上游送来的一个单位写成一条"这里发生了什么"的判断。

方案见桌面 ``14-事件融合方案.md``。融合按两条 lane 各做各的（``lanes/session``、``lanes/physical``）：
数据从哪来、怎么读、要不要调模型互不相同；一个单位只出一条判断，不接续、不合并，带语义的归类只在词表一处。

两条 lane 共用的只有本包顶层这几样：

- ``judgement``：判断的形状（做了什么、目标、概要、步骤；没读懂的那一层 behavior 为空）；
- ``derivation``：把判断绑定成耐久记录——身份与时间由系统出，不由模型出；
- ``store`` / ``receipt`` / ``receipt_store``：判断与回执的落盘；
- ``coverage``：哪些观测已经被判过（重做时认得出来）。

## 产物是判断，不是事实

融合产出的是**我们在某个时刻对一段观测做出的判断**，不是"一件已经发生完的事"。判断到行为树文档
（行为与观测空白）的归约是另一层（``behavior/reduction``）：封口、词表归类、落树都在那里。

为 Reachy 逐帧观测写的那条线（入队、切段、作业、逐帧提示词、逐帧归属、跨窗续写）已随物理 lane 改用
上游送最终结果而删除（裁定 40）。``judgement`` 里的续接、修正、并行关系与归约里的组链是它留下的，
两条 lane 都不产出这些关系；删它们是归约的改动（TODO(BHV-SESSION-LANE-001)）。
"""

from habitus.behavior.fusion.config import BehaviorFusionConfig
from habitus.behavior.fusion.coverage import BehaviorCoverageIndex
from habitus.behavior.fusion.derivation import (
    FUSION_IMPLEMENTATION_VERSION,
    FUSION_VERSION,
    DurableFact,
    DurableJudgement,
    derive_judgements,
    judgement_payload,
    without_unresolvable_relations,
)
from habitus.behavior.fusion.errors import BehaviorFusionError, BehaviorFusionLimitError
from habitus.behavior.fusion.judgement import (
    BehaviorClaim,
    BehaviorFact,
    BehaviorJudgement,
    BehaviorJudgementBatch,
    JudgementLink,
    JudgementRelation,
    JudgementStatus,
    JudgementStatusBasis,
)
from habitus.behavior.fusion.receipt import (
    RECEIPT_SCHEMA_VERSION,
    BehaviorFusionReceipt,
    build_fusion_receipt,
    receipt_identity,
    segment_identity,
)
from habitus.behavior.fusion.receipt_store import BehaviorFusionReceiptStore
from habitus.behavior.fusion.store import JUDGEMENT_SCHEMA_VERSION, BehaviorJudgementStore

__all__ = [
    "FUSION_IMPLEMENTATION_VERSION",
    "FUSION_VERSION",
    "JUDGEMENT_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "BehaviorClaim",
    "BehaviorCoverageIndex",
    "BehaviorFact",
    "BehaviorFusionConfig",
    "BehaviorFusionError",
    "BehaviorFusionLimitError",
    "BehaviorFusionReceipt",
    "BehaviorFusionReceiptStore",
    "BehaviorJudgement",
    "BehaviorJudgementBatch",
    "BehaviorJudgementStore",
    "DurableFact",
    "DurableJudgement",
    "JudgementLink",
    "JudgementRelation",
    "JudgementStatus",
    "JudgementStatusBasis",
    "build_fusion_receipt",
    "derive_judgements",
    "judgement_payload",
    "receipt_identity",
    "segment_identity",
    "without_unresolvable_relations",
]
