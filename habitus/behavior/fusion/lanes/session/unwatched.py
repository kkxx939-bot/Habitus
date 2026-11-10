"""会话一支的「未观测」：系统没在跑的那一段（关机、合盖、进程没起来）写成一条空判断（实时化方案第 6 节）。

预测拍发现离上一拍超过容差，就把 ``[上一拍, 这一拍)`` 交到这里：一条 behavior 为空、依据写着"没在看"（``unobserved``）的
判断，两条凭据（这一段的开头与结尾），回执与覆盖照常落——与物理一支写没在看的时段同一种写法（``fusion/writing.py``），
之后由会话一支的归约落成「未观测」空白（``reduction.payloads``）。这里不碰行为树，也不拿 sweep 锁。

身份只由这一段的起止派生（判定时刻取结尾），同一段重写幂等；凭据已被回执覆盖的不再写。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime, tzinfo

from habitus.behavior.fusion.coverage import BehaviorCoverageIndex
from habitus.behavior.fusion.judgement import BehaviorClaim, BehaviorJudgement, JudgementStatusBasis
from habitus.behavior.fusion.lanes.session.protocol import SESSION_PROTOCOL
from habitus.behavior.fusion.lanes.session.recorder import session_observer
from habitus.behavior.fusion.receipt_store import BehaviorFusionReceiptStore
from habitus.behavior.fusion.store import BehaviorJudgementStore
from habitus.behavior.fusion.writing import FusionVersions, publish_judgement
from habitus.behavior.observation import (
    BehaviorObservation,
    BehaviorObservationBatch,
    BehaviorObservationEnvelope,
    BehaviorObservationStore,
)

#: 这条写法不调模型、没有提示词：版本只标"这条记录是哪一版规则写的"。
UNWATCHED_FUSION_VERSION = "behavior_session_unwatched_v1"
UNWATCHED_RULE_VERSION = "behavior_session_unwatched_rule_v1"
#: 凭据的来源身份：不是哪个会话，是系统自己（会话号不会带冒号前缀的这一段）。
UNWATCHED_OBSERVER = session_observer("system:unwatched")


class SessionUnwatchedRecorder:
    """一段没在看的时间 → 两条凭据 + 一条空判断（依据 ``unobserved``）+ 一份回执。"""

    def __init__(
        self,
        *,
        observations: BehaviorObservationStore,
        judgements: BehaviorJudgementStore,
        receipts: BehaviorFusionReceiptStore,
        coverage: BehaviorCoverageIndex,
        primary_subject: str,
        zone: tzinfo,
        clock: Callable[[], datetime],
    ) -> None:
        if not primary_subject:
            raise ValueError("primary_subject must be non-empty text")
        self.observations = observations
        self.judgements = judgements
        self.receipts = receipts
        self.coverage = coverage
        self.primary_subject = primary_subject
        self.zone = zone
        self.clock = clock

    def record(self, started_at: datetime, ended_at: datetime) -> str | None:
        """写 ``[started_at, ended_at)`` 这一段；返回判断编号，已经写完过的返回 None。"""

        if started_at.utcoffset() is None or ended_at.utcoffset() is None:
            raise ValueError("an unwatched span needs timezone-aware bounds")
        if not started_at < ended_at:
            raise ValueError("an unwatched span must end after it starts")
        key = f"unwatched#{_instant(started_at)}#{_instant(ended_at)}"
        evidence = self._evidence(key, started_at, ended_at)
        if {item.observation_id for item in evidence} <= self.coverage.covered_observation_ids(self.clock()):
            return None
        envelope = BehaviorObservationEnvelope.create(
            observer_id=UNWATCHED_OBSERVER,
            protocol=SESSION_PROTOCOL,
            batch=BehaviorObservationBatch(observer_id=UNWATCHED_OBSERVER, observations=evidence),
            delivery_id=hashlib.sha256(key.encode("utf-8")).hexdigest(),
            recorded_at=max(self.clock(), ended_at),
            config=self.observations.config,
        )
        judgement = BehaviorJudgement(
            judgement_no=1,
            covers=(1, 2),
            subjects=(),
            claim=BehaviorClaim(behavior=None, goal=None, summary=None, basis=()),
            status=None,
            status_basis=JudgementStatusBasis.UNOBSERVED,
        )
        return publish_judgement(
            observations=self.observations,
            judgements=self.judgements,
            receipts=self.receipts,
            coverage=self.coverage,
            envelope=envelope,
            judgement=judgement,
            versions=FusionVersions(UNWATCHED_FUSION_VERSION, UNWATCHED_RULE_VERSION),
            # 判定时刻取这一段的结尾：重写时身份不变
            judged_at=ended_at,
            validation_attempts=1,
            primary_subject=self.primary_subject,
        )

    def _evidence(
        self, key: str, started_at: datetime, ended_at: datetime
    ) -> tuple[BehaviorObservation, BehaviorObservation]:
        """两条凭据：这一段的开头与结尾。两条都在结尾那一刻才可用（那时才知道这一段没在看）。"""

        available = ended_at.astimezone(self.zone)

        def mark(occurred_at: datetime, edge: str) -> BehaviorObservation:
            return BehaviorObservation.create(
                observer_id=UNWATCHED_OBSERVER,
                occurred_at=occurred_at.astimezone(self.zone),
                available_at=available,
                modality="session",
                semantics=f"系统没在看的时段{edge}",
                participants=[self.primary_subject],
                knowledge_state="observed",
                confidence=1.0,
                evidence_refs=[key],
                config=self.observations.config,
            )

        return mark(started_at, "开始"), mark(ended_at, "结束")


def _instant(value: datetime) -> str:
    """一个瞬间的规范写法（UTC），不随调用方用哪个偏移写而变。"""

    return value.astimezone(UTC).isoformat(timespec="microseconds")


__all__ = ["UNWATCHED_FUSION_VERSION", "UNWATCHED_OBSERVER", "UNWATCHED_RULE_VERSION", "SessionUnwatchedRecorder"]
