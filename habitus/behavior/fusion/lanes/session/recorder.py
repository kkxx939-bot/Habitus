"""把一轮的描述写成判断记录（确定性，不调模型）。

一轮一条判断，不带任何续接关系（裁定 32）：落到行为树上也是一轮一条。走与逐帧融合相同的派生、回执、落盘，
之后由同一个归约封口落树。

每一轮留两条凭据观测——人开口的时刻一条、助手做完的时刻一条——只存时刻和指向会话源的引用，不存内容（裁定 33）。
判断的开始与最后所见由这两条派生：开始是人开口的时刻，最后所见是助手做完的时刻（裁定 31）。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, tzinfo

from habitus.behavior.fusion.coverage import BehaviorCoverageIndex
from habitus.behavior.fusion.derivation import derive_judgements, judgement_payload
from habitus.behavior.fusion.errors import BehaviorFusionError
from habitus.behavior.fusion.judgement import (
    BehaviorClaim,
    BehaviorFact,
    BehaviorJudgement,
    BehaviorJudgementBatch,
    JudgementStatus,
    JudgementStatusBasis,
)
from habitus.behavior.fusion.lanes.session.model import SessionTurn
from habitus.behavior.fusion.lanes.session.prompt import SESSION_PROMPT_VERSION, TurnDescription
from habitus.behavior.fusion.lanes.session.protocol import SESSION_PROTOCOL
from habitus.behavior.fusion.receipt import build_fusion_receipt
from habitus.behavior.fusion.receipt_store import BehaviorFusionReceiptStore
from habitus.behavior.fusion.store import BehaviorJudgementStore
from habitus.behavior.observation import (
    BehaviorObservation,
    BehaviorObservationBatch,
    BehaviorObservationEnvelope,
    BehaviorObservationStore,
)

SESSION_FUSION_VERSION = f"behavior_session_fusion_v1+{SESSION_PROMPT_VERSION}"
_OBSERVER_PREFIX = "session/"


def session_observer(conversation_id: str) -> str:
    """凭据的来源身份：哪个会话。会话身份就带在这里（裁定 37）。"""

    return f"{_OBSERVER_PREFIX}{conversation_id}"


@dataclass(frozen=True)
class InterruptedRecord:
    """上一次写到一半的那条判断：判断已经落盘，回执还没有。"""

    judgement_id: str
    judged_at: datetime
    description: TurnDescription


class SessionTurnRecorder:
    """一轮 → 两条凭据观测 + 一条判断 + 一份回执。同一轮重做不重复写。"""

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
        if not isinstance(observations, BehaviorObservationStore):
            raise TypeError("observations must be BehaviorObservationStore")
        if not isinstance(judgements, BehaviorJudgementStore):
            raise TypeError("judgements must be BehaviorJudgementStore")
        if not isinstance(receipts, BehaviorFusionReceiptStore):
            raise TypeError("receipts must be BehaviorFusionReceiptStore")
        if not isinstance(coverage, BehaviorCoverageIndex):
            raise TypeError("coverage must be BehaviorCoverageIndex")
        if not isinstance(primary_subject, str) or not primary_subject:
            raise ValueError("primary_subject must be non-empty text")
        if not isinstance(zone, tzinfo):
            raise TypeError("zone must be a tzinfo")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.observations = observations
        self.judgements = judgements
        self.receipts = receipts
        self.coverage = coverage
        self.primary_subject = primary_subject
        self.zone = zone
        self.clock = clock

    def covered(self) -> frozenset[str]:
        """已经被回执覆盖的观测。读一次要扫全部覆盖记录，一批轮只读一次，不要每轮读。"""

        return self.coverage.covered_observation_ids(self.clock())

    def is_recorded(self, turn: SessionTurn, covered: frozenset[str] | None = None) -> bool:
        """这一轮是不是已经写完了（崩溃后重做同一份会话源时用）。

        凭据观测的身份只由这一轮的内容派生，所以重做时算出来的是同一组；它们已被某份回执覆盖，就是写完了。
        """

        evidence = {item.observation_id for item in self._evidence(turn)}
        return evidence <= (self.covered() if covered is None else covered)

    def interrupted(self, turn: SessionTurn) -> InterruptedRecord | None:
        """这一轮上一次是不是写到一半：判断落盘了，回执没落。

        判断的身份里有模型写的描述，重做时再问一遍模型，措辞一变就是另一条判断——所以回执没落的那条
        必须认回来接着写完，不能重问。凭据是最先落的；凭据都不在，就是从没写过，不用去翻判断。
        """

        evidence = self._evidence(turn)
        if self.observations.read(self._envelope(turn, evidence).source_id) is None:
            return None
        owned = [item.observation_id for item in evidence]
        written = [
            item
            for item in self.judgements.list()
            if item["observation_ids"] == owned and item["fusion_version"] == SESSION_FUSION_VERSION
        ]
        if not written:
            return None
        first = min(written, key=lambda item: (item["judged_at"], item["judgement_id"]))
        return InterruptedRecord(
            judgement_id=first["judgement_id"],
            judged_at=datetime.fromisoformat(first["judged_at"]),
            description=TurnDescription(
                name=first["behavior"],
                summary=first["summary"],
                goal=first["goal"],
                steps=tuple(fact["semantics"] for fact in first["basis"]),
            ),
        )

    def resume(self, turn: SessionTurn, interrupted: InterruptedRecord) -> str:
        """把写到一半的那一轮接着写完：照落盘的那条判断补上回执，不产生第二条判断。"""

        judgement_id = self.record(
            turn, interrupted.description, validation_attempts=1, judged_at=interrupted.judged_at
        )
        if judgement_id != interrupted.judgement_id:
            raise BehaviorFusionError("an interrupted session judgement could not be resumed under its identity")
        return judgement_id

    def record(
        self,
        turn: SessionTurn,
        description: TurnDescription,
        *,
        validation_attempts: int,
        judged_at: datetime | None = None,
    ) -> str:
        """写这一轮的凭据、判断与回执，返回判断的编号。``judged_at`` 只在接着写完上一次时给。"""

        opened, closed = self._evidence(turn)
        envelope = self._envelope(turn, (opened, closed))
        # 先落凭据再落判断：归约落树时要读凭据；顺序反过来的话，中途崩溃会留下一条读不到凭据的判断。
        self.observations.put(envelope)
        judgement = BehaviorJudgement(
            judgement_no=1,
            covers=(1, 2),
            subjects=(self.primary_subject,),
            claim=BehaviorClaim(
                behavior=description.name,
                goal=description.goal,
                summary=description.summary,
                basis=tuple(BehaviorFact(semantics=step, fragment_nos=(2,)) for step in description.steps),
            ),
            # 这一轮已经做完才送到这里；"在不在进行"由下游按时间读，融合不判。
            status=JudgementStatus.ONGOING,
            status_basis=JudgementStatusBasis.OBSERVED,
            relations=(),
        )
        judged_at = self.clock() if judged_at is None else judged_at
        derived = derive_judgements(
            BehaviorJudgementBatch((judgement,)),
            (opened, closed),
            source_refs=(envelope.source_id,),
            judged_at=judged_at,
        )
        # 身份已按内容派生；这里只把版本标签改成会话 lane 自己的，免得记录上写着逐帧融合的提示词版本。
        derived = tuple(
            replace(item, fusion_version=SESSION_FUSION_VERSION, prompt_version=SESSION_PROMPT_VERSION)
            for item in derived
        )
        receipt = build_fusion_receipt(
            derived,
            (opened, closed),
            source_refs=(envelope.source_id,),
            prompt_version=SESSION_PROMPT_VERSION,
            validation_attempts=validation_attempts,
            primary_subject=self.primary_subject,
            judged_at=judged_at,
        )
        for item in derived:
            self.judgements.put_payload(judgement_payload(item))
        self.coverage.record(self.receipts.put(receipt))
        return derived[0].judgement_id

    def _evidence(self, turn: SessionTurn) -> tuple[BehaviorObservation, BehaviorObservation]:
        """这一轮的两条凭据：人开口、助手做完。两条都在这一轮结束时才可用。"""

        available = turn.completed_at.astimezone(self.zone)

        def mark(occurred_at: datetime, what: str) -> BehaviorObservation:
            return BehaviorObservation.create(
                observer_id=session_observer(turn.conversation_id),
                occurred_at=occurred_at.astimezone(self.zone),
                available_at=available,
                modality="session",
                semantics=f"会话的一轮（第 {turn.start_sequence}–{turn.end_sequence} 条）{what}",
                participants=[self.primary_subject],
                knowledge_state="observed",
                confidence=1.0,
                evidence_refs=[turn.key],
                config=self.observations.config,
            )

        return mark(turn.instructed_at, "开始"), mark(turn.completed_at, "结束")

    def _envelope(
        self, turn: SessionTurn, evidence: tuple[BehaviorObservation, BehaviorObservation]
    ) -> BehaviorObservationEnvelope:
        observer = session_observer(turn.conversation_id)
        return BehaviorObservationEnvelope.create(
            observer_id=observer,
            protocol=SESSION_PROTOCOL,
            batch=BehaviorObservationBatch(observer_id=observer, observations=evidence),
            delivery_id=hashlib.sha256(turn.key.encode("utf-8")).hexdigest(),
            recorded_at=max(self.clock(), turn.completed_at),
            config=self.observations.config,
        )


__all__ = ["SESSION_FUSION_VERSION", "InterruptedRecord", "SessionTurnRecorder", "session_observer"]
