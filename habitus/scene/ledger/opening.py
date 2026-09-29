"""开承诺：一天的概念命中里，哪些让哪条假设的前件集合凑齐了。

- **前件命中 ＝ 集合里全部元素命中**：行为元素查 ``occurrences/hits``（含读时沿 parent 链聚合的祖先，档只对叶子
  比），情境元素查触发那条的 ``situation_hits``。
- **锚 ＝ 集合里最后开始的行为概念的 ``started_at``**（2026-09-26 裁定）。多行为集合里其余元素要在锚之前
  ``gathering_hours`` 内出现过——这个时长是待定值，在重放上定。
- **去重**：同一假设、两条触发等的是后件的**同一次机会**（前一条承诺的第 1 次机会还没开始，新触发就来了）→ 一条；
  同一触发同一假设只开一次。没有对照快照的承诺只按触发去重。
- **对照**：开的那一刻向组合根注入的口要后件在锚之后的前 ``snapshot_opportunities`` 个机会，快照进承诺；要不到就
  ``control=None`` 照样开（读时算不出强度，但机会本身是事实）。
- **幂等**：承诺已在盘上就跳过，不比内容——事后树重建了对照也不改。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta

from habitus.scene.concepts.model import ConceptSet
from habitus.scene.hypotheses.model import Antecedent, Hypothesis
from habitus.scene.ledger.model import (
    MAX_SNAPSHOT_OPPORTUNITIES,
    Claim,
    LedgerError,
    OpportunityProvider,
    OpportunityRequest,
)
from habitus.scene.ledger.store import LedgerStore
from habitus.scene.occurrences.model import ConceptHit, ConceptHits
from habitus.scene.occurrences.store import ConceptHitStore


@dataclass(frozen=True)
class LedgerConfig:
    """全部是待定值，在重放上定，这里只是默认。

    - ``gathering_hours``  多行为集合的凑齐时长；
    - ``snapshot_opportunities``  开承诺时向对照口要后件的前几个机会（要够结算数到 ``censor_after`` 次，再加次数方面的地平线）；
    - ``censor_after``  等过这么多个**已观测**机会后件还没来 → 右删失；
    - ``opportunity_coverage``  一个机会的 span 覆盖比例低于它就算没看清。
    """

    gathering_hours: float = 24.0
    snapshot_opportunities: int = 16
    censor_after: int = 10
    opportunity_coverage: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.gathering_hours, bool) or not isinstance(self.gathering_hours, int | float) or self.gathering_hours <= 0:
            raise ValueError("gathering_hours must be a positive number")
        for label in ("snapshot_opportunities", "censor_after"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if self.snapshot_opportunities > MAX_SNAPSHOT_OPPORTUNITIES:
            raise ValueError(f"snapshot_opportunities is at most {MAX_SNAPSHOT_OPPORTUNITIES}")
        if self.censor_after > self.snapshot_opportunities:
            raise ValueError("censor_after cannot exceed snapshot_opportunities: the settler could never count that many passes")
        if (
            isinstance(self.opportunity_coverage, bool)
            or not isinstance(self.opportunity_coverage, int | float)
            or not 0.0 < self.opportunity_coverage <= 1.0
        ):
            raise ValueError("opportunity_coverage lies in (0, 1]")


@dataclass(frozen=True)
class OpeningReport:
    day: date
    opened: int
    already_open: int
    overlapping: int
    without_control: int
    #: 前件或后件引用了已不存在的概念、这一晚开不了账的假设身份。改了概念名而忘了改假设时，这条假设会
    #: 永久停账，外部只表现为读数里一个 ``n=0``——所以要在这里报出来。
    unmappable: tuple[str, ...] = ()
    #: 机会口给了账本收不下的快照，那些承诺按"没有对照"开了。机会口自己的 bug，要能看见。
    unusable_snapshots: tuple[str, ...] = ()


def behaviour_hits(record: ConceptHits, concepts: ConceptSet) -> Mapping[str, str | None]:
    """记录的行为命中连同祖先（读时聚合）：叶子带档，祖先不带档。概念集里已不存在的名字跳过。"""

    table: dict[str, str | None] = {}
    for identity, grade in record.graded_hits.items():
        if identity not in concepts:
            continue
        table[identity] = grade
        for ancestor in concepts.ancestors(identity):
            table.setdefault(ancestor, None)
    return table


def matches_antecedent(record: ConceptHits, antecedent: Antecedent, concepts: ConceptSet) -> bool:
    """这条记录是否命中一个前件元素：行为元素看命中（含祖先），情境元素看情境栏；带档的要档相同。"""

    identity = antecedent.identity
    if identity not in concepts:
        return False
    if concepts[identity].role.is_behavior:
        table = behaviour_hits(record, concepts)
    else:
        table = dict(record.graded_situations)
    if identity not in table:
        return False
    return antecedent.grade is None or table[identity] == antecedent.grade


def open_claims_for_day(
    day: date,
    *,
    hypotheses: Sequence[Hypothesis],
    concepts: ConceptSet,
    hits: ConceptHitStore,
    ledger: LedgerStore,
    opportunities: OpportunityProvider,
    now: datetime,
    config: LedgerConfig | None = None,
) -> OpeningReport:
    """把这一天的命中过一遍全部假设，凑齐了前件的开承诺。夜批里排在"映射封口日"之后、"结算"之前。"""

    resolved = config or LedgerConfig()
    records = hits.read_day(day)
    if not records:
        return OpeningReport(day=day, opened=0, already_open=0, overlapping=0, without_control=0)
    lookback_start = min(record.started_at for record in records) - timedelta(hours=resolved.gathering_hours)
    earlier = hits.read_window(lookback_start, min(record.started_at for record in records))
    opened = already = overlapping = without_control = 0
    unmappable: list[str] = []
    unusable: list[str] = []
    existing_claims: dict[str, list[Claim]] = {}
    for hypothesis in hypotheses:
        behaviours = tuple(item for item in hypothesis.antecedents if item.identity in concepts and concepts[item.identity].role.is_behavior)
        situations = tuple(item for item in hypothesis.antecedents if item.identity in concepts and concepts[item.identity].role.is_situation)
        if len(behaviours) + len(situations) != len(hypothesis.antecedents) or hypothesis.consequent_identity not in concepts:
            # 前件或后件引用了已不存在的概念。后件这一半尤其要拦：结算侧的 ``behaviour_hits`` 会跳过不在
            # 集里的身份，后件永远匹配不上，窗末全部记成落空——凭空造负样本，而且 add-only 撤不掉。
            unmappable.append(hypothesis.identity)
            continue
        for record in records:
            gathered = _gather(record, behaviours, situations, earlier + records, concepts, resolved.gathering_hours)
            if gathered is None:
                continue
            claim_hits, uris = gathered
            if hypothesis.identity not in existing_claims:
                # 惰性：只有真的凑齐了才读这条假设的账（一夜几百条假设、多半一条都不开；每条都读整本账是线性涨的）。
                # 只装**还开着**的：已经兑现的承诺不再"等"任何机会，拿它去挡新触发会让第二次约球永远开不了账（评审 B6）。
                # 去重只看**当期指纹**的承诺：改过第几次机会/方向之后，旧口径的承诺不该把新口径的机会挡掉。
                existing_claims[hypothesis.identity] = [
                    claim for claim in ledger.open_claims(hypothesis.identity) if claim.hypothesis_fingerprint == hypothesis.fingerprint
                ]
            candidate = Claim(
                hypothesis_identity=hypothesis.identity,
                hypothesis_fingerprint=hypothesis.fingerprint,
                aspect=hypothesis.aspect,
                trigger_uri=record.occurrence_uri,
                antecedent_hits=claim_hits,
                antecedent_uris=uris,
                situation_snapshot=tuple(hit.concept for hit in record.situation_hits),
                control=None,
                created_at=now,
            )
            if ledger.claim_exists(candidate.ref):
                already += 1
                continue
            if any(_awaits_same_opportunity(other, record.started_at) for other in existing_claims[hypothesis.identity]):
                overlapping += 1
                continue
            # 要够结算数得到的个数：``censor_after`` 个删失步，以及时刻/次数量的那一次（第 expected_at + horizon − 1 个）。
            # 要少了，那条假设的时刻/次数账永远结不了（评审 B5）。
            wanted = max(resolved.snapshot_opportunities, (hypothesis.expected_at or 1) + hypothesis.horizon - 1)
            snapshot = opportunities.opportunities(
                OpportunityRequest(consequent=hypothesis.consequent_identity, anchor=record.started_at, count=wanted)
            )
            try:
                claim = replace(candidate, control=snapshot)
            except LedgerError as exc:
                # 机会口给了一份账本收不下的快照（第一个机会在锚之前结束、机会重叠……）。当"要不到对照"处理：
                # 机会本身是事实，照样开；一条坏快照不该把整晚所有假设的承诺都掀掉（十 ④、评审 A-11）。
                unusable.append(f"{hypothesis.identity}: {exc}")
                claim = candidate
            if claim.control is None:
                without_control += 1
            ledger.write_claim(claim)
            existing_claims[hypothesis.identity].append(claim)
            opened += 1
    return OpeningReport(
        day=day,
        opened=opened,
        already_open=already,
        overlapping=overlapping,
        without_control=without_control,
        unmappable=tuple(sorted(unmappable)),
        unusable_snapshots=tuple(sorted(unusable)),
    )


def _awaits_same_opportunity(existing: Claim, anchor: datetime) -> bool:
    """已有承诺的第 1 次机会还没开始、新触发就来了 → 两条等的是后件的同一次机会，不再开。没有快照的承诺不参与。"""

    if existing.control is None:
        return False
    return existing.anchor <= anchor < existing.control.opportunities[0].span.start


def _gather(
    trigger: ConceptHits,
    behaviours: Sequence[Antecedent],
    situations: Sequence[Antecedent],
    pool: Sequence[ConceptHits],
    concepts: ConceptSet,
    gathering_hours: float,
) -> tuple[tuple[ConceptHit, ...], tuple[str, ...]] | None:
    """以 ``trigger`` 为锚，前件集合凑齐了吗？凑齐返回 (每个元素怎么命中的, 出了力的 occurrence)。

    触发必须自己命中至少一个行为元素；其余行为元素要在锚之前 ``gathering_hours`` 内的别的记录上命中
    （取最近的一条）；情境元素全在触发这条的情境栏里。
    """

    if not any(matches_antecedent(trigger, item, concepts) for item in behaviours):
        return None
    for item in situations:
        if not matches_antecedent(trigger, item, concepts):
            return None
    horizon = trigger.started_at - timedelta(hours=gathering_hours)
    matched: list[ConceptHit] = []
    uris = {trigger.occurrence_uri}
    for item in behaviours:
        if matches_antecedent(trigger, item, concepts):
            matched.append(ConceptHit(item.concept, _grade_of(trigger, item, concepts)))
            continue
        earlier = [
            record
            for record in pool
            if horizon <= record.started_at < trigger.started_at and record.occurrence_uri != trigger.occurrence_uri and matches_antecedent(record, item, concepts)
        ]
        if not earlier:
            return None
        latest = max(earlier, key=lambda record: (record.started_at.astimezone(UTC), record.occurrence_uri))
        matched.append(ConceptHit(item.concept, _grade_of(latest, item, concepts)))
        uris.add(latest.occurrence_uri)
    for item in situations:
        matched.append(ConceptHit(item.concept, trigger.graded_situations.get(item.identity)))
    return tuple(matched), tuple(sorted(uris))


def _grade_of(record: ConceptHits, antecedent: Antecedent, concepts: ConceptSet) -> str | None:
    """记录上这个元素实际带的档（祖先没有档）。"""

    if concepts[antecedent.identity].role.is_behavior:
        return behaviour_hits(record, concepts).get(antecedent.identity)
    return record.graded_situations.get(antecedent.identity)


__all__ = ["LedgerConfig", "OpeningReport", "behaviour_hits", "matches_antecedent", "open_claims_for_day"]
