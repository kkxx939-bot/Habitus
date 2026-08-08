"""单 Evidence Claim Normalization 的应用编排。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from behavior.claim.ledger import BehaviorClaimLedger, ClaimPage
from behavior.claim.model import BehaviorClaim, BehaviorClaimLedgerEntry
from behavior.claim.planner import (
    ClaimNormalizationPlan,
    ClaimNormalizationPlanner,
    NormalizationLane,
)
from behavior.claim.receipt import (
    NO_CORE_NORMALIZER_FINGERPRINT,
    ClaimNormalizationReceipt,
)
from behavior.claim.route_executor import (
    ClaimProcessingDegradation,
    NormalizationRouteExecutor,
)
from behavior.errors import ClaimNormalizationConflictError, ClaimNormalizationError
from behavior.evidence.ledger import BehaviorEvidenceLedger
from behavior.evidence.record import BehaviorEvidenceRecord


class ClaimNormalizationResultStatus(str, Enum):
    COMPLETE = "COMPLETE"
    CORE_COMMITTED_ENHANCEMENT_PENDING = "CORE_COMMITTED_ENHANCEMENT_PENDING"
    CORE_COMMITTED_WITH_DEGRADATION = "CORE_COMMITTED_WITH_DEGRADATION"


@dataclass(frozen=True, slots=True)
class ClaimNormalizationResult:
    evidence_record_id: str
    status: ClaimNormalizationResultStatus
    core_receipt: ClaimNormalizationReceipt
    enhancement_receipts: tuple[ClaimNormalizationReceipt, ...]
    degradations: tuple[ClaimProcessingDegradation, ...]


class ClaimNormalizationService:
    def __init__(
        self,
        evidence_ledger: BehaviorEvidenceLedger,
        claim_ledger: BehaviorClaimLedger,
        planner: ClaimNormalizationPlanner,
        route_executor: NormalizationRouteExecutor,
    ) -> None:
        self.evidence_ledger = evidence_ledger
        self.claim_ledger = claim_ledger
        self.planner = planner
        self.route_executor = route_executor

    async def normalize(self, evidence_record_id: str) -> ClaimNormalizationResult:
        record, plan = self._context(evidence_record_id)
        core = await self._core(record, plan)
        for route in plan.enhancement_routes:
            await self.route_executor.execute(record, plan, route, retry=False)
        return self._result(record, plan, core)

    async def retry_enhancement(
        self,
        evidence_record_id: str,
        normalizer_name: str,
    ) -> ClaimNormalizationResult:
        record, plan = self._context(evidence_record_id)
        route = next(
            (item for item in plan.enhancement_routes if item.normalizer_name == normalizer_name),
            None,
        )
        if route is None:
            raise ClaimNormalizationError("requested Enhancement route is not planned")
        core = self.claim_ledger.read_receipt(self._core_identity(record, plan))
        if core is None:
            raise ClaimNormalizationConflictError(
                "Core must be completed before retrying an Enhancement"
            )
        await self.route_executor.execute(record, plan, route, retry=True)
        return self._result(record, plan, core)

    def read_claim(self, claim_id: str) -> BehaviorClaim | None:
        return self.claim_ledger.read_claim(claim_id)

    def list_claims_for_evidence(
        self,
        evidence_record_id: str,
        limit: int,
        cursor: str | None = None,
    ) -> ClaimPage:
        return self.claim_ledger.list_for_evidence(evidence_record_id, limit, cursor)

    def list_claims_after_sequence(
        self,
        sequence: int,
        limit: int,
    ) -> tuple[BehaviorClaimLedgerEntry, ...]:
        return self.claim_ledger.list_after_sequence(sequence, limit)

    def _context(
        self,
        evidence_record_id: str,
    ) -> tuple[BehaviorEvidenceRecord, ClaimNormalizationPlan]:
        record = self.evidence_ledger.read(evidence_record_id)
        if record is None:
            raise ClaimNormalizationError("Evidence record does not exist")
        return record, self.planner.plan(record)

    async def _core(
        self,
        record: BehaviorEvidenceRecord,
        plan: ClaimNormalizationPlan,
    ) -> ClaimNormalizationReceipt:
        if not plan.core_routes:
            return await self.route_executor.publish_no_core(record, plan)
        result = await self.route_executor.execute(
            record,
            plan,
            plan.core_routes[0],
            retry=False,
        )
        if result.receipt is None:
            raise ClaimNormalizationError("Deterministic Core did not publish a Receipt")
        return result.receipt

    def _result(
        self,
        record: BehaviorEvidenceRecord,
        plan: ClaimNormalizationPlan,
        core: ClaimNormalizationReceipt,
    ) -> ClaimNormalizationResult:
        receipts: list[ClaimNormalizationReceipt] = []
        degradations: list[ClaimProcessingDegradation] = []
        for route in plan.enhancement_routes:
            identity = self.route_executor.identity(record, plan, route).value
            receipt = self.claim_ledger.read_receipt(identity)
            if receipt is not None:
                receipts.append(receipt)
                continue
            attempt = self.claim_ledger.read_latest_attempt(identity)
            degradations.append(
                ClaimProcessingDegradation(
                    route.normalizer_name,
                    "ENHANCEMENT_PENDING" if attempt is None else attempt.error_code or "NORMALIZATION_FAILED",
                    True if attempt is None else attempt.retryable,
                )
            )
        status = self._status(tuple(degradations))
        return ClaimNormalizationResult(
            record.evidence_record_id,
            status,
            core,
            tuple(receipts),
            tuple(degradations),
        )

    def _core_identity(
        self,
        record: BehaviorEvidenceRecord,
        plan: ClaimNormalizationPlan,
    ) -> str:
        fingerprint = (
            NO_CORE_NORMALIZER_FINGERPRINT
            if not plan.core_routes
            else plan.core_routes[0].normalizer_fingerprint
        )
        return self.route_executor.publication_factory.identity(
            record,
            plan,
            lane=NormalizationLane.CORE,
            normalizer_fingerprint=fingerprint,
        ).value

    @staticmethod
    def _status(
        degradations: tuple[ClaimProcessingDegradation, ...],
    ) -> ClaimNormalizationResultStatus:
        if not degradations:
            return ClaimNormalizationResultStatus.COMPLETE
        if all(item.retryable for item in degradations):
            return ClaimNormalizationResultStatus.CORE_COMMITTED_ENHANCEMENT_PENDING
        return ClaimNormalizationResultStatus.CORE_COMMITTED_WITH_DEGRADATION
