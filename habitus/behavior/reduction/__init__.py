"""归约写入层：闭合的判断链 → 行为树文档。

行为树唯一的生产入口（观测 → 融合 → **归约** → 落盘），零发明：语义逐字来自判断，唯一的
LLM 触点是 kinds 归一。见 ``TODO(BHV-TREE-REBUILD-001)`` 与各模块 docstring。
"""

from habitus.behavior.reduction.chains import BehaviorChain, ChainAssembly, assemble_chains
from habitus.behavior.reduction.errors import BehaviorReductionBusyError, BehaviorReductionError
from habitus.behavior.reduction.ledger import BehaviorReductionEntry, BehaviorReductionLedger
from habitus.behavior.reduction.payloads import (
    REDUCTION_VERSION,
    UNREADABLE_GAP_KIND,
    chain_address,
    gap_payload,
    occurrence_payload,
)
from habitus.behavior.reduction.record import ReducibleJudgement, parse_judgement_record
from habitus.behavior.reduction.runner import (
    DEFAULT_SWEEP_LOCK_TTL_SECONDS,
    BehaviorKindMergeReport,
    BehaviorReductionReport,
    BehaviorReductionRunner,
)
from habitus.behavior.reduction.sealing import seal_horizon, sealed_chain_indexes, sealed_gaps

__all__ = [
    "DEFAULT_SWEEP_LOCK_TTL_SECONDS",
    "BehaviorKindMergeReport",
    "REDUCTION_VERSION",
    "UNREADABLE_GAP_KIND",
    "BehaviorChain",
    "BehaviorReductionEntry",
    "BehaviorReductionBusyError",
    "BehaviorReductionError",
    "BehaviorReductionLedger",
    "BehaviorReductionReport",
    "BehaviorReductionRunner",
    "ChainAssembly",
    "ReducibleJudgement",
    "assemble_chains",
    "chain_address",
    "gap_payload",
    "occurrence_payload",
    "parse_judgement_record",
    "seal_horizon",
    "sealed_chain_indexes",
    "sealed_gaps",
]
