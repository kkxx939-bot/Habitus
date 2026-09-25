"""从一拍判断里抽承诺。

只抽**带时窗的「会」**：账本要的是可核对的话。「不会」没有时点，「说不准」是弃权，说不出时窗的「会」
也核对不了——三者都不进账（说不准之后多久发生了这类读数，等真要调判断层时再从判断记录里算）。

去重按时窗**有交集**：判断者每槽重判，同一次即将发生会在连续几槽反复出现，一次命中不能记成十条；
而判断者改口说另一段（晚上那次）时窗不相交，那是另一条承诺。

情形从引用的卡的关联记录取，按规范身份去重；卡没关联时为空，账记在 (行为, "") 上。
纯函数：不读时钟、不落盘。复用的一拍由调用方跳过——那不是新的话。
"""

from __future__ import annotations

from collections.abc import Iterable

from habitus.foresight.assemble import CandidateEvidence, EvidencePack
from habitus.foresight.judge.model import CandidateVerdict, Judgement
from habitus.foresight.ledger.model import Claim, Conditions
from habitus.foundation.ids import canonical_text_identity
from habitus.foundation.integrity import canonical_digest


def claims_from(
    pack: EvidencePack,
    judgement: Judgement,
    *,
    open_claims: Iterable[Claim] = (),
    conditions: Conditions = (),
    condition_keys: tuple[str, ...] = (),
    facts_version: str = "none",
) -> tuple[Claim, ...]:
    """这一拍新产生的承诺。

    ``open_claims`` 是那天已落盘、还没结算的承诺，用来去重。``conditions`` / ``condition_keys`` /
    ``facts_version`` 是事实门在那一刻给的答案、问过的键、以及答话者的口径（见 ``scene.facts``），
    原样记进每条承诺——结算之后它们就是 loss 的证据。
    """

    by_kind = {item.kind_token: item for item in pack.expanded}
    standing = [claim for claim in open_claims if claim.day == pack.moment.day]
    claims: list[Claim] = []
    for verdict in judgement.verdicts:
        if verdict.verdict != "会" or verdict.window is None or not verdict.basis:
            continue
        evidence = by_kind.get(verdict.kind_token)
        if evidence is None:
            continue
        if any(
            claim.kind_token == verdict.kind_token and claim.overlaps(verdict.window)
            for claim in (*standing, *claims)
        ):
            continue
        claims.append(_claim(pack, judgement, verdict, evidence, conditions, condition_keys, facts_version))
    return tuple(claims)


def _claim(
    pack: EvidencePack,
    judgement: Judgement,
    verdict: CandidateVerdict,
    evidence: CandidateEvidence,
    conditions: Conditions,
    condition_keys: tuple[str, ...],
    facts_version: str,
) -> Claim:
    assert verdict.window is not None
    cards = {card.uri: card for card in evidence.background.cards}
    situations: dict[str, str] = {}
    for uri in verdict.basis:
        card = cards.get(uri)
        text = None if card is None or card.gloss is None else card.gloss.situation
        if text:
            situations.setdefault(canonical_text_identity(text, "situation text"), text)
    day = pack.moment.day
    identity = canonical_digest(
        {
            "kind": verdict.kind_token,
            "day": day.isoformat(),
            "slot": pack.moment.slot,
            "window": list(verdict.window),
            "generation": pack.generation,
            "judged_at": judgement.judged_at.isoformat(),
            "basis": list(verdict.basis),
            # 条件进身份：同一刻同一句话配上不同的条件是两条不同的记录，撞同一个身份会被存储判成账本损坏。
            "conditions": [list(pair) for pair in conditions],
            "facts_version": facts_version,
        }
    )[:32]
    return Claim(
        claim_id=identity,
        kind_token=verdict.kind_token,
        day=day,
        slot=pack.moment.slot,
        slot_minutes=pack.slot_minutes,
        window=verdict.window,
        judged_at=judgement.judged_at,
        generation=pack.generation,
        judge_version=judgement.judge_version,
        basis=verdict.basis,
        situations=tuple(situations.values()),
        numbers=evidence.numbers,
        conditions=conditions,
        condition_keys=condition_keys,
        facts_version=facts_version,
    )


__all__ = ["claims_from"]
