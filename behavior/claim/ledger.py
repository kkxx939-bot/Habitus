"""不可变 Claim Ledger、Attempt 与 Receipt 的领域协议。"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from behavior.claim.model import BehaviorClaim, BehaviorClaimLedgerEntry
from behavior.claim.publication import ClaimPublication
from behavior.claim.receipt import ClaimNormalizationAttempt, ClaimNormalizationReceipt

ClaimPage = tuple[tuple[BehaviorClaimLedgerEntry, ...], str | None]


@runtime_checkable
class BehaviorClaimLedger(Protocol):
    def publish(
        self,
        publication: ClaimPublication,
    ) -> tuple[ClaimNormalizationReceipt, bool]: ...

    def publish_failed_attempt(
        self,
        attempt: ClaimNormalizationAttempt,
    ) -> ClaimNormalizationAttempt: ...

    def read_claim(self, claim_id: str) -> BehaviorClaim | None: ...

    def read_claim_entry(self, claim_id: str) -> BehaviorClaimLedgerEntry | None: ...

    def list_after_sequence(self, sequence: int, limit: int) -> tuple[BehaviorClaimLedgerEntry, ...]: ...

    def list_for_evidence(
        self,
        evidence_record_id: str,
        limit: int,
        cursor: str | None = None,
    ) -> ClaimPage: ...

    def list_by_event_time(
        self,
        start: datetime,
        end: datetime,
        limit: int,
        cursor: str | None = None,
    ) -> ClaimPage: ...

    def list_by_semantic_fingerprint(
        self,
        fingerprint: str,
        limit: int,
        cursor: str | None = None,
    ) -> ClaimPage: ...

    def read_attempt(self, attempt_id: str) -> ClaimNormalizationAttempt | None: ...

    def read_latest_attempt(self, processing_identity: str) -> ClaimNormalizationAttempt | None: ...

    def read_receipt(self, processing_identity: str) -> ClaimNormalizationReceipt | None: ...
