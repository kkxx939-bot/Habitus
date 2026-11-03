"""把上游的一段行为、一段没在看的时间写成判断记录（确定性，不调模型）。

一段行为一条判断，不带任何续接关系；一段没在看的时间一条 behavior 为空的判断，归约把它落成「未观测」的空白。
落盘序列与"写到一半"的认法与会话 lane 共用（``fusion/writing.py``），之后由同一个归约封口落树。

每个单位留两条凭据观测（开始、结束），判断的开始与最后所见由它们派生；时刻保留上游给的本地偏移，不折到本机时区
（人在外地时，行为算在它发生地的那一天、那个钟点）。

收过的单位记在一份小账上（``ledger``）：同一段再送一遍认得出来、不重复写；同一个编号换了内容（上游改判）不收。
写的顺序是 凭据 → 判断 → 回执 → 覆盖记录 → 小账：小账最后落，它在就说明这个单位完整写完了。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from habitus.behavior.fusion.coverage import BehaviorCoverageIndex
from habitus.behavior.fusion.judgement import (
    BehaviorClaim,
    BehaviorJudgement,
    JudgementStatus,
    JudgementStatusBasis,
)
from habitus.behavior.fusion.lanes.physical.config import PhysicalLaneConfig
from habitus.behavior.fusion.lanes.physical.ledger import AcceptedSpan, PhysicalSegmentLedger, SpanKind
from habitus.behavior.fusion.lanes.physical.model import CoverageState, PhysicalSegment
from habitus.behavior.fusion.lanes.physical.protocol import PHYSICAL_PROTOCOL
from habitus.behavior.fusion.receipt_store import BehaviorFusionReceiptStore
from habitus.behavior.fusion.store import BehaviorJudgementStore
from habitus.behavior.fusion.writing import FusionVersions, publish_judgement, written_judgements
from habitus.behavior.model import address_name, behavior_name
from habitus.behavior.observation import (
    BehaviorObservation,
    BehaviorObservationBatch,
    BehaviorObservationEnvelope,
    BehaviorObservationStore,
)
from habitus.foundation.integrity import canonical_digest

# 这条 lane 不调模型，没有提示词；版本只标"这条记录是哪一版转换规则写的"。
PHYSICAL_FUSION_VERSION = "behavior_physical_fusion_v1"
PHYSICAL_RULE_VERSION = "behavior_physical_segment_rule_v1"
_OBSERVER_PREFIX = "physical/"
_GAP_REASONS = {
    CoverageState.NOT_WORN: "没戴",
    CoverageState.CHARGING: "充电",
    CoverageState.NO_DATA: "无数据",
}


def physical_observer(device: str) -> str:
    """凭据的来源身份：哪台设备。"""

    return f"{_OBSERVER_PREFIX}{device}"


def gap_unit(started_at: datetime, ended_at: datetime) -> str:
    """一段没在看的时间在小账上的编号：只由起止的那两个瞬间定（同一瞬间不管上游用哪个偏移写，都是同一个）。"""

    return f"gap#{_instant(started_at)}#{_instant(ended_at)}"


class UnitDisposition(str, Enum):
    """一段的去向。"""

    RECORDED = "recorded"  # 这次写了判断
    ALREADY_RECORDED = "already_recorded"  # 之前写过（同一段再送来、或上次写到一半这次接着写完）
    REVISION_REJECTED = "revision_rejected"  # 同一个编号换了内容：上游改判（或编号被重复使用），不收
    UNUSABLE_NAME = "unusable_name"  # 行为的说法当不了记录的名字，这一段不收


@dataclass(frozen=True)
class CoveredUnits:
    """已经被回执覆盖的观测与交付位置（一份交付读一次）。"""

    observations: frozenset[str]
    envelopes: frozenset[str]


@dataclass(frozen=True)
class _Unit:
    """要写的一个单位：一段行为或一段没在看的时间。"""

    device: str
    key: str  # 在小账上的编号
    what: str  # 写进凭据的一句说明
    started_at: datetime
    ended_at: datetime
    claim: BehaviorClaim
    content_digest: str  # 上游给的内容的摘要：重发与改判靠它分
    kind: SpanKind


class PhysicalSegmentRecorder:
    """一段 → 两条凭据观测 + 一条判断 + 一份回执 + 小账上的一笔。"""

    def __init__(
        self,
        *,
        observations: BehaviorObservationStore,
        judgements: BehaviorJudgementStore,
        receipts: BehaviorFusionReceiptStore,
        coverage: BehaviorCoverageIndex,
        ledger: PhysicalSegmentLedger,
        primary_subject: str,
        clock: Callable[[], datetime],
        config: PhysicalLaneConfig | None = None,
    ) -> None:
        if not isinstance(observations, BehaviorObservationStore):
            raise TypeError("observations must be BehaviorObservationStore")
        if not isinstance(judgements, BehaviorJudgementStore):
            raise TypeError("judgements must be BehaviorJudgementStore")
        if not isinstance(receipts, BehaviorFusionReceiptStore):
            raise TypeError("receipts must be BehaviorFusionReceiptStore")
        if not isinstance(coverage, BehaviorCoverageIndex):
            raise TypeError("coverage must be BehaviorCoverageIndex")
        if not isinstance(ledger, PhysicalSegmentLedger):
            raise TypeError("ledger must be PhysicalSegmentLedger")
        if not isinstance(primary_subject, str) or not primary_subject:
            raise ValueError("primary_subject must be non-empty text")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.observations = observations
        self.judgements = judgements
        self.receipts = receipts
        self.coverage = coverage
        self.ledger = ledger
        self.primary_subject = primary_subject
        self.clock = clock
        self.config = config or PhysicalLaneConfig()

    def covered(self) -> CoveredUnits:
        """已经被回执覆盖的观测与交付位置。读一次要扫全部覆盖记录，一份交付只读一次。"""

        return CoveredUnits(
            observations=self.coverage.covered_observation_ids(self.clock()),
            envelopes=self.coverage.covered_envelope_ids(),
        )

    def record_segment(self, device: str, segment: PhysicalSegment, covered: CoveredUnits) -> UnitDisposition:
        """写一段行为。名字用上游给的说法，概要按固定句式拼，目标与步骤为空。"""

        name = address_name(segment.behavior)  # 名字会拿去当路径上的一段：路径符号换成空格
        try:
            behavior_name(name)
        except (TypeError, ValueError):
            return UnitDisposition.UNUSABLE_NAME
        return self._record(
            _Unit(
                device=device,
                key=f"segment#{segment.segment_id}",
                what=f"行为段 {segment.segment_id}（{name}）",
                started_at=segment.started_at,
                ended_at=segment.ended_at,
                claim=BehaviorClaim(behavior=name, goal=None, summary=self._summary(name, segment), basis=()),
                content_digest=canonical_digest(
                    {
                        "behavior": segment.behavior,
                        "started_at": _instant(segment.started_at),
                        "ended_at": _instant(segment.ended_at),
                        "facts": list(segment.facts),
                    }
                ),
                kind=SpanKind.SEGMENT,
            ),
            covered,
        )

    def record_gap(
        self, device: str, started_at: datetime, ended_at: datetime, state: CoverageState, covered: CoveredUnits
    ) -> UnitDisposition:
        """写一段没在看的时间：一条 behavior 为空、依据写着"没在看"的判断，归约把它落成「未观测」的空白。"""

        if state is CoverageState.OBSERVING:
            raise ValueError("an observed span is not a gap")
        return self._record(
            _Unit(
                device=device,
                key=gap_unit(started_at, ended_at),
                what=f"没在看的时段（{_GAP_REASONS[state]}）",
                started_at=started_at,
                ended_at=ended_at,
                claim=BehaviorClaim(behavior=None, goal=None, summary=None, basis=()),
                # 原因（没戴 / 充电 / 无数据）不进摘要：同一段没在看的时间换个原因再报，不是两段。lane 对收过的空白
                # 不再调到这里（只把没收过的那几截交下来），这份摘要只在直接调 recorder 时起作用：防御。
                content_digest=canonical_digest({"unobserved": [_instant(started_at), _instant(ended_at)]}),
                kind=SpanKind.UNOBSERVED,
            ),
            covered,
        )

    def _record(self, unit: _Unit, covered: CoveredUnits) -> UnitDisposition:
        accepted = self.ledger.accepted(unit.device, unit.key, unit.started_at)
        if accepted is not None:
            # 收过的单位：内容一样就是重发；不一样就是上游送进来之后又改了，行为树只增不改，不收。
            return (
                UnitDisposition.ALREADY_RECORDED
                if accepted == unit.content_digest
                else UnitDisposition.REVISION_REJECTED
            )
        opened, closed = self._evidence(unit)
        envelope = self._envelope(unit, (opened, closed))
        span = AcceptedSpan(unit.kind, unit.started_at, unit.ended_at)
        if {opened.observation_id, closed.observation_id} <= covered.observations:
            # 回执落了、小账没落就崩了：补上小账。
            self.ledger.accept(unit.device, unit.key, unit.content_digest, span)
            return UnitDisposition.ALREADY_RECORDED
        if envelope.source_id in covered.envelopes:
            # 这个位置上的单位其实写完了（回执都落了，只差小账），这次送来的内容不一样：是改判，不收。
            # 按交付位置认，凭据被归约释放之后也认得出。
            return UnitDisposition.REVISION_REJECTED
        existing = self.observations.read(envelope.source_id)
        if existing is not None and existing.payload_digest != envelope.payload_digest:
            # 上一次只写到一半、这次内容变了：从没写完过，算不上改判。把那半截清掉，按这次的重新写——
            # 留着它，那两条没被覆盖的凭据会把归约的封口永远卡在它那一刻。
            self._discard_half_written(
                existing.source_id, [item.observation_id for item in existing.batch.observations]
            )
            existing = None
        # 凭据是最先落的：它已经在、却没被回执覆盖，就是上次写到一半；判断若已落，照它的时刻与版本接着写完
        # （判断的身份里有判断时刻，换一个时刻就是另一条判断）。
        resumed = None if existing is None else self._first_written([opened.observation_id, closed.observation_id])
        self._write_judgement(
            unit,
            envelope,
            judged_at=self.clock() if resumed is None else datetime.fromisoformat(resumed["judged_at"]),
            versions=FusionVersions(PHYSICAL_FUSION_VERSION, PHYSICAL_RULE_VERSION)
            if resumed is None
            else FusionVersions.of(resumed),
        )
        self.ledger.accept(unit.device, unit.key, unit.content_digest, span)
        return UnitDisposition.RECORDED if resumed is None else UnitDisposition.ALREADY_RECORDED

    def _first_written(self, observation_ids: list[str]) -> dict[str, Any] | None:
        written = written_judgements(self.judgements, observation_ids)
        return min(written, key=lambda item: (item["judged_at"], item["judgement_id"])) if written else None

    def _discard_half_written(self, source_id: str, observation_ids: list[str]) -> None:
        for item in written_judgements(self.judgements, observation_ids):
            self.judgements.discard(item["judgement_id"])
        self.observations.discard(source_id)

    def _write_judgement(
        self,
        unit: _Unit,
        envelope: BehaviorObservationEnvelope,
        *,
        judged_at: datetime,
        versions: FusionVersions,
    ) -> None:
        readable = unit.claim.is_readable
        judgement = BehaviorJudgement(
            judgement_no=1,
            covers=(1, 2),
            subjects=(self.primary_subject,) if readable else (),
            claim=unit.claim,
            # 上游只送已经结束的段（进行中的不进来、不落树），所以落下来的每一条都是做完了的。
            status=JudgementStatus.COMPLETED if readable else None,
            status_basis=JudgementStatusBasis.OBSERVED if readable else JudgementStatusBasis.UNOBSERVED,
        )
        publish_judgement(
            observations=self.observations,
            judgements=self.judgements,
            receipts=self.receipts,
            coverage=self.coverage,
            envelope=envelope,
            judgement=judgement,
            versions=versions,
            judged_at=judged_at,
            validation_attempts=1,
            primary_subject=self.primary_subject,
        )

    def _summary(self, name: str, segment: PhysicalSegment) -> str:
        started, ended = segment.started_at, segment.ended_at.astimezone(segment.started_at.tzinfo)
        until = f"{ended:%H:%M}" if ended.date() == started.date() else f"{ended:%m-%d %H:%M}"
        text = f"{name}，{started:%H:%M}–{until}"
        if segment.facts:
            text += "；" + "，".join(segment.facts)
        return text[: self.config.max_summary_chars]

    def _evidence(self, unit: _Unit) -> tuple[BehaviorObservation, BehaviorObservation]:
        """一个单位的两条凭据：开始、结束。两条都在结束时才可用（上游只送已经结束的）。"""

        def mark(occurred_at: datetime, edge: str) -> BehaviorObservation:
            return BehaviorObservation.create(
                observer_id=physical_observer(unit.device),
                occurred_at=occurred_at,
                available_at=unit.ended_at,
                modality="physical",
                semantics=f"{unit.what}{edge}",
                participants=[self.primary_subject],
                knowledge_state="observed",
                confidence=1.0,
                evidence_refs=[f"{unit.device}#{unit.key}"],
                config=self.observations.config,
            )

        return mark(unit.started_at, "开始"), mark(unit.ended_at, "结束")

    def _envelope(
        self, unit: _Unit, evidence: tuple[BehaviorObservation, BehaviorObservation]
    ) -> BehaviorObservationEnvelope:
        observer = physical_observer(unit.device)
        return BehaviorObservationEnvelope.create(
            observer_id=observer,
            protocol=PHYSICAL_PROTOCOL,
            batch=BehaviorObservationBatch(observer_id=observer, observations=evidence),
            # 交付身份只由"哪台设备的哪个单位"定：同一段不管随哪份交付送来，都落在同一个位置上。
            delivery_id=hashlib.sha256(f"{unit.device}#{unit.key}".encode()).hexdigest(),
            recorded_at=max(self.clock(), unit.ended_at),
            config=self.observations.config,
        )


def _instant(value: datetime) -> str:
    """一个瞬间的规范写法（UTC），不随上游用哪个偏移写而变。"""

    return value.astimezone(UTC).isoformat(timespec="microseconds")


__all__ = [
    "PHYSICAL_FUSION_VERSION",
    "PHYSICAL_RULE_VERSION",
    "CoveredUnits",
    "PhysicalSegmentRecorder",
    "UnitDisposition",
    "gap_unit",
    "physical_observer",
]
