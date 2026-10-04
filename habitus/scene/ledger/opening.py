"""开承诺：一天的概念命中里，哪些让哪条假设的前件集合凑齐了。

- **前件命中 ＝ 集合里全部元素命中**：行为元素查 ``occurrences/hits``（含读时沿 parent 链聚合的祖先，档只对叶子
  比），情境元素查触发那条的 ``situation_hits``。
- **锚 ＝ 集合里最后开始的行为概念的 ``started_at``**（2026-09-26 裁定）。多行为集合里其余元素要在锚之前
  ``gathering_hours`` 内出现过——这个时长是待定值，在重放上定。
- **去重只有一道：同一触发同一假设只开一次**（幂等）。**每一次前因都独立开账**（2026-09-30 二-6，用户原话："不管一天有
  多少相同的行为，要算都是分开算的，对某个行为的影响无论是前因还是后果都是要分开算的"）——下午两杯咖啡各开一条，
  周一约球周二又约是两条前提。旧写法"等同一次机会就不开"被去掉了：它把第二次约球吞掉（评审 A-6）、结算后重跑又
  不幂等（评审 B-4），而且要往回看几天的账。几条承诺等同一个后果时段时，**读侧**按块算样本数（R3-05），不在这里合。
- **不回填**（2026-09-30 裁定四）：闭环与安慰剂写的假设，只给**写入时刻之后**的触发开账——用来发现它的那批观测
  不算它的证据；同一夜重跑两遍也不会多开。基准凭常识写、没看过账，可以回看历史。
- **前因按峰归账**（2026-10-01）：前件带峰号的，触发（或凑齐它的那条记录）的开始时刻要落在那个峰的钟面窗口里
  （两边各展机会口的容差）；``#0`` 要落在它全部窗口之外。不落在这个峰号上的，不给这条假设开，给同前因另一个峰号的那条开。
- **对照**：开的那一刻向组合根注入的口要后果峰窗口在锚之后各天的落点（概率一个；次数方面要接下来几个），快照进承诺；
  那天没曲线对照为空，窗口照样在，账照样开。无节律型（没有窗口）不要对照。
- **幂等**：承诺已在盘上就跳过，不比内容——事后树重建了对照也不改。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta

from habitus.scene.concepts.model import ConceptSet
from habitus.scene.hypotheses.model import Antecedent, Hypothesis, HypothesisOrigin, peak_index_of
from habitus.scene.ledger.model import (
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
    - ``opportunity_coverage``  一个窗口的覆盖比例低于它就算没看清。

    "等过几个机会就删失"那个数（``censor_after``）2026-10-01 随账改成按钟面窗口记而删掉：每本账只等自己那一个窗口。
    """

    gathering_hours: float = 24.0
    opportunity_coverage: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.gathering_hours, bool) or not isinstance(self.gathering_hours, int | float) or self.gathering_hours <= 0:
            raise ValueError("gathering_hours must be a positive number")
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
    #: 恒为 0，留着是为了不改调用方的形状。"等同一次机会就不开"那道去重 2026-09-30 按二-6 去掉了。
    overlapping: int
    without_control: int
    #: 前件或后件引用了已不存在的概念、这一晚开不了账的假设身份。改了概念名而忘了改假设时，这条假设会
    #: 永久停账，外部只表现为读数里一个 ``n=0``——所以要在这里报出来。
    unmappable: tuple[str, ...] = ()
    #: 机会口给了账本收不下的快照，那些承诺按"没有对照"开了。机会口自己的 bug，要能看见。
    unusable_snapshots: tuple[str, ...] = ()
    #: 触发早于假设写入时刻、按"不回填"没开的条数（闭环与安慰剂）。
    backfill_refused: int = 0


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
    opened = already = without_control = refused = 0
    unmappable: list[str] = []
    unusable: list[str] = []
    for hypothesis in hypotheses:
        behaviours = tuple(item for item in hypothesis.antecedents if item.identity in concepts and concepts[item.identity].role.is_behavior)
        situations = tuple(item for item in hypothesis.antecedents if item.identity in concepts and concepts[item.identity].role.is_situation)
        if len(behaviours) + len(situations) != len(hypothesis.antecedents) or hypothesis.consequent_identity not in concepts:
            # 前件或后件引用了已不存在的概念。后件这一半尤其要拦：结算侧的 ``behaviour_hits`` 会跳过不在
            # 集里的身份，后件永远匹配不上，窗末全部记成落空——凭空造负样本，而且 add-only 撤不掉。
            unmappable.append(hypothesis.identity)
            continue
        for record in records:
            gathered = _gather(record, behaviours, situations, earlier + records, concepts, resolved.gathering_hours, hypothesis, opportunities.slack_minutes)
            if gathered is None:
                continue
            if not backfills(hypothesis) and record.started_at < hypothesis.created_at:
                refused += 1
                continue
            claim_hits, uris = gathered
            candidate = Claim(
                hypothesis_identity=hypothesis.identity,
                hypothesis_fingerprint=hypothesis.fingerprint,
                aspect=hypothesis.aspect,
                trigger_uri=record.occurrence_uri,
                antecedent_hits=claim_hits,
                antecedent_uris=uris,
                situation_snapshot=tuple(hit.concept for hit in record.situation_hits),
                situations_checked=record.situations_checked,
                control=None,
                created_at=now,
            )
            if ledger.claim_exists(candidate.ref):
                already += 1
                continue
            window = hypothesis.consequent_window
            try:
                snapshot = None
                if window is not None:
                    # 概率与时刻只看自己那个窗口；次数方面接下来 horizon 个窗口（按后件的整张峰表轮着铺）。
                    snapshot = opportunities.opportunities(
                        OpportunityRequest(
                            consequent=hypothesis.consequent_identity,
                            anchor=record.started_at,
                            count=hypothesis.horizon,
                            window=window,
                            windows=hypothesis.windows[hypothesis.consequent_identity],
                        )
                    )
                claim = replace(candidate, control=snapshot)
            except LedgerError as exc:
                # 机会口给了一份账本收不下的快照（第一个机会在锚之前结束、机会重叠……）——不管它是在机会口里就抛、
                # 还是装进承诺时才抛。当"要不到对照"处理：机会本身是事实，照样开；一条坏快照不该把整晚所有假设的
                # 承诺都掀掉（十 ④、评审 A-11；第三批冒烟 10-01 实测塌过一夜）。
                unusable.append(f"{hypothesis.identity}: {exc}")
                claim = candidate
            if claim.control is None:
                without_control += 1
            ledger.write_claim(claim)
            opened += 1
    return OpeningReport(
        day=day,
        opened=opened,
        already_open=already,
        overlapping=0,
        without_control=without_control,
        unmappable=tuple(sorted(unmappable)),
        unusable_snapshots=tuple(sorted(unusable)),
        backfill_refused=refused,
    )


def backfills(hypothesis: Hypothesis) -> bool:
    """这条假设能不能给写入时刻之前的触发开账。基准凭常识写、没看过账 → 能；闭环与安慰剂读过账 → 不能（裁定四 (b)）。"""

    return hypothesis.source.origin not in (HypothesisOrigin.MODERATION, HypothesisOrigin.PLACEBO)


def peak_of(hypothesis: Hypothesis, antecedent: Antecedent, moment: datetime, *, slack_minutes: int) -> int:
    """这条记录落在这个前件的第几个峰（0 = 全部窗口之外）。窗口两边各展容差、不越过与邻峰的中点；跨午夜的窗口把次日凌晨也算进去。"""

    return peak_index_of(hypothesis.antecedent_windows(antecedent), moment.hour * 60 + moment.minute, slack_minutes=slack_minutes)


def _matches_peak(hypothesis: Hypothesis, antecedent: Antecedent, record: ConceptHits, slack_minutes: int) -> bool:
    return antecedent.peak is None or peak_of(hypothesis, antecedent, record.started_at, slack_minutes=slack_minutes) == antecedent.peak


def _gather(
    trigger: ConceptHits,
    behaviours: Sequence[Antecedent],
    situations: Sequence[Antecedent],
    pool: Sequence[ConceptHits],
    concepts: ConceptSet,
    gathering_hours: float,
    hypothesis: Hypothesis,
    slack_minutes: int,
) -> tuple[tuple[ConceptHit, ...], tuple[str, ...]] | None:
    """以 ``trigger`` 为锚，前件集合凑齐了吗？凑齐返回 (每个元素怎么命中的, 出了力的 occurrence)。

    触发必须自己命中至少一个行为元素（**且落在那个元素的峰号上**）；其余行为元素要在锚之前 ``gathering_hours`` 内的
    别的记录上命中（取最近的一条，同样要落在它的峰号上）；情境元素全在触发这条的情境栏里。
    """

    def hits(record: ConceptHits, item: Antecedent) -> bool:
        return matches_antecedent(record, item, concepts) and _matches_peak(hypothesis, item, record, slack_minutes)

    if not any(hits(trigger, item) for item in behaviours):
        return None
    for item in situations:
        if not matches_antecedent(trigger, item, concepts):
            return None
    horizon = trigger.started_at - timedelta(hours=gathering_hours)
    matched: list[ConceptHit] = []
    uris = {trigger.occurrence_uri}
    for item in behaviours:
        if hits(trigger, item):
            matched.append(ConceptHit(item.concept, _grade_of(trigger, item, concepts)))
            continue
        earlier = [
            record
            for record in pool
            if horizon <= record.started_at < trigger.started_at and record.occurrence_uri != trigger.occurrence_uri and hits(record, item)
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


__all__ = ["LedgerConfig", "OpeningReport", "backfills", "behaviour_hits", "matches_antecedent", "open_claims_for_day", "peak_of"]
