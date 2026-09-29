"""④ ``scene/ledger/`` 账：算法写。前件命中开承诺（带后件机会快照与假设指纹），后件到来或等够机会数结算；提醒另记。

账本只存事实；强度、类型、PN/PS 在 ``views/`` 读时算。后件的机会与观测覆盖都由组合根注入的口给——本支不读预测树、
不读 ``views``。
"""

from habitus.scene.ledger.codec import LedgerCodecError, LedgerSchemaError
from habitus.scene.ledger.model import (
    OUTCOMES_BY_ASPECT,
    Claim,
    ClaimRef,
    Coverage,
    CoverageProvider,
    Intervention,
    InterventionResponse,
    LedgerError,
    ObservedGap,
    Opportunity,
    OpportunityPass,
    OpportunityProvider,
    OpportunityRequest,
    OpportunitySnapshot,
    Outcome,
    Response,
    Settlement,
    WindowSpan,
)
from habitus.scene.ledger.opening import (
    LedgerConfig,
    OpeningReport,
    behaviour_hits,
    matches_antecedent,
    open_claims_for_day,
)
from habitus.scene.ledger.settlement import SettlementReport, close_claim, settle_claim, settle_due, settle_due_all
from habitus.scene.ledger.store import LEDGER_SEGMENT, LedgerStore, LedgerStoreError

__all__ = [
    "LEDGER_SEGMENT",
    "OUTCOMES_BY_ASPECT",
    "Claim",
    "ClaimRef",
    "Coverage",
    "CoverageProvider",
    "Intervention",
    "InterventionResponse",
    "LedgerConfig",
    "LedgerCodecError",
    "LedgerError",
    "LedgerSchemaError",
    "LedgerStore",
    "LedgerStoreError",
    "ObservedGap",
    "OpeningReport",
    "Opportunity",
    "OpportunityPass",
    "OpportunityProvider",
    "OpportunityRequest",
    "OpportunitySnapshot",
    "Outcome",
    "Response",
    "Settlement",
    "SettlementReport",
    "WindowSpan",
    "behaviour_hits",
    "close_claim",
    "matches_antecedent",
    "open_claims_for_day",
    "settle_claim",
    "settle_due",
    "settle_due_all",
]
