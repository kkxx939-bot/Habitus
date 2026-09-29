"""intentions/：还立着的前提 ＝ 账本里没结算的承诺。

约球命中 → 「约球 → 打球」开一条承诺 → 没结算 ＝ "约好了打球，还没去"。预测层的"此刻还立着的前提"从这里读；
兑现是机械的"后件概念命中"，不再靠 LLM 认一句自由文本。每条带 ``ClaimRef``，预测层要提醒时按它写提醒记录。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from habitus.scene.concepts.model import concept_identity
from habitus.scene.hypotheses.model import Hypothesis
from habitus.scene.ledger.model import Claim, ClaimRef
from habitus.scene.ledger.store import LedgerStore
from habitus.scene.occurrences.model import ConceptHit


@dataclass(frozen=True)
class StandingIntention:
    ref: ClaimRef
    antecedents: tuple[ConceptHit, ...]
    consequent: str
    since: datetime
    waited_hours: float
    trigger_uri: str
    #: 到此刻已经结束的机会数（不看覆盖——覆盖只有结算时才问）；没有快照就 None。
    opportunities_passed: int | None
    #: 下一次机会几点；快照里没有更多机会就 None。
    next_opportunity_at: datetime | None

    def label(self) -> str:
        items = "、".join(hit.concept if hit.grade is None else f"{hit.concept}·{hit.grade}" for hit in self.antecedents)
        passed = "" if self.opportunities_passed is None else f"，已过 {self.opportunities_passed} 次机会"
        return f"{items} → 等 {self.consequent}，已等 {self.waited_hours:.1f} 小时{passed}"


def standing_intentions(ledger: LedgerStore, hypotheses: Mapping[str, Hypothesis], *, now: datetime) -> tuple[StandingIntention, ...]:
    """全部未结算的承诺，按立起来的时刻升序。``hypotheses`` 按身份索引，账本里有而假设集里没有的跳过。"""

    found: list[StandingIntention] = []
    for identity, hypothesis in hypotheses.items():
        for claim in ledger.open_claims(identity):
            found.append(_intention(claim, hypothesis, now))
    return tuple(sorted(found, key=lambda item: (item.since, item.trigger_uri)))


def _intention(claim: Claim, hypothesis: Hypothesis, now: datetime) -> StandingIntention:
    passed = None
    following = None
    if claim.control is not None:
        passed = sum(1 for item in claim.control.opportunities if item.span.end <= now)
        following = next((item.at for item in claim.control.opportunities if item.span.start > now), None)
    return StandingIntention(
        ref=claim.ref,
        antecedents=claim.antecedent_hits,
        consequent=hypothesis.consequent,
        since=claim.anchor,
        waited_hours=max(0.0, (now - claim.anchor).total_seconds() / 3600.0),
        trigger_uri=claim.trigger_uri,
        opportunities_passed=passed,
        next_opportunity_at=following,
    )


def open_consequents_at(ledger: LedgerStore, hypotheses: Mapping[str, Hypothesis], *, moment: datetime) -> frozenset[str]:
    """那一刻还立着的承诺的**后件概念身份**——情境里「开放承诺」那一族（"约了球还没打"）读它。

    判据是"锚点已经到了、而结算还没到"：结算的决定性时刻取 ``observed_at``（后件到来/前提作废那一刻），
    没有就退到 ``settled_at``（夜批写下它那一刻）。这样**重放历史时答的是历史**——直接用
    ``open_claims`` 会答"现在还开着的"，把一条上周就结掉的承诺算成当时不在，或者反过来。
    """

    if not isinstance(moment, datetime) or moment.utcoffset() is None:
        raise TypeError("moment must be a timezone-aware datetime")
    found: set[str] = set()
    for identity, hypothesis in hypotheses.items():
        closed = {
            (settlement.ref.day, settlement.ref.leaf): (settlement.observed_at or settlement.settled_at)
            for settlement in ledger.settlements_for(identity)
        }
        for claim in ledger.claims_for(identity):
            if claim.anchor > moment:
                continue
            settled_at = closed.get((claim.ref.day, claim.ref.leaf))
            if settled_at is not None and settled_at <= moment:
                continue
            found.add(concept_identity(hypothesis.consequent))
    return frozenset(found)


__all__ = ["StandingIntention", "open_consequents_at", "standing_intentions"]
