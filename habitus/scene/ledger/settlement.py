"""结算：后件到底来没来、落在第几次机会。按方面分三套，不共用逻辑（用户裁定）。

- **概率 · 节律型**（``expected_at`` 有数）  后件一来就结（``OCCURRED``：落在第几次机会、隔了几小时、之前每个机会
  过没过 / 看清了没）；没来就数已经过去且看清了的机会，够 ``censor_after`` 个 → 右删失（``CENSORED``）。没有"落空"
  ——"到现在没来"和"永远不来"是两件事，前者是一条带步数的删失记录。
- **概率 · 无节律型**（``expected_at is None``，2026-09-27 裁定）  约球→打球、挂号→就诊：后件没节律，**不数机会、
  没有时效**。只有两种结法——后件来了（``OCCURRED``，记 ``latency_hours``），或 ``released_by`` 里的概念在锚之后命中
  （``RELEASED`` + ``releasing_uri``，前提被作废：再次挂号取代上一次那张号）。没写 ``released_by`` 就一直开着，
  ``intentions`` 里一直在。兑现是 **FIFO** 的：同一假设的开放承诺按锚序，一个后件 occurrence 只兑现最早那条
  （``consumed``）——约球两次、打球一次，不能读成两次都去了（评审 A-3）。
- **时刻**  只看第 ``expected_at`` 次机会：那次机会上来了记时刻（``OBSERVED``）；那次机会已观测地过了没来 → ``ABSENT``
  （不是"很晚"，是没有观测值）；那次机会没看清 → ``CENSORED``。
- **次数**  第 ``expected_at`` 到第 ``expected_at + horizon − 1`` 次机会都过完 → 记计数（0 也是计数）；其中有没看清的 → ``CENSORED``。

后件命中的口径与开承诺同一套：``behaviour_hits`` 的行为那一半（含祖先聚合）。后件必须**开始于锚之后**且不是触发那条自己；
不拿 ``last_observed_at`` 做"不重叠"的判据——那会把观测覆盖重新接进窗边界（09-26 裁定）。一次到来归哪次机会由
``OpportunitySnapshot.index_of`` 定：已开始的机会数，两峰之间的到来归前一峰。

**一个机会看清了没**：向注入的覆盖口要它 span 的覆盖比例，低于 ``opportunity_coverage`` 就是没看清；**或者**那段里有
记录把后件记成了「未决」（映射器判不了：要的材料没给、模型没答成、或判据引用的事件落在观测空白里）——两种都
不算它过了、也不算缺席（七d-2；用户 09-27"没看到就不进分母更不会进入分子"；评审 A-4）。没有对照快照的承诺（树上没这个后件的数）只能在后件到来时结，等不到就交生命周期关。

**``now`` 与 ``until`` 是两件事**：``now`` 只是落在记录上的 ``settled_at``；"时间走到哪了"一律看 ``until`` ——
它是**命中记录可信到哪一刻**（= 最后一个已映射日的日界，组合根从 ``hits.days_done()`` 取）。映射按封口日走，
``now`` 用墙钟必然越过还没映射的日子：跨午夜的就寝落在次日目录，次日还没映射时它在账上就成了"没来"，
而 add-only 撤不掉（评审 B2 实测）。缺省 ``until = now``，调用方确信两者同步时才这么用。
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta

from habitus.scene.concepts.model import ConceptError, ConceptSet, concept_identity
from habitus.scene.hypotheses.model import Aspect, Hypothesis
from habitus.scene.ledger.model import (
    Claim,
    CoverageProvider,
    Opportunity,
    OpportunityPass,
    OpportunitySnapshot,
    Outcome,
    Settlement,
)
from habitus.scene.ledger.opening import LedgerConfig, behaviour_hits
from habitus.scene.ledger.store import LedgerStore
from habitus.scene.occurrences.model import ConceptHits
from habitus.scene.occurrences.store import ConceptHitStore


@dataclass(frozen=True)
class SettlementReport:
    hypothesis_identity: str
    settled: int
    pending: int


def mapped_until(hits: ConceptHitStore, now: datetime) -> datetime | None:
    """命中记录可信到哪一刻：最后一个已映射日的**次日零点**（时区取自 ``now``），封顶到 ``now``。

    一天都没映射过 → ``None`` = 什么都不可信，一条都不结。夜批可以自己算好传 ``until``；这里是缺省口径，
    免得调用方忘了这条前置条件（评审 B2：``now`` 用墙钟会越过还没映射的日子，跨午夜的后件就被判成"没来"）。
    """

    days = hits.days_done()
    if not days:
        return None
    boundary = datetime.combine(max(days) + timedelta(days=1), time(0, 0), tzinfo=now.tzinfo)
    return min(boundary, now)


def consequent_records(claim: Claim, records: Sequence[ConceptHits], concepts: ConceptSet, *, until: datetime) -> tuple[ConceptHits, ...]:
    """锚之后、``until`` 之前、不是触发自己、命中后件（含祖先聚合）的记录，按时刻升序。"""

    consequent = claim.consequent
    found = [
        record
        for record in records
        if record.occurrence_uri != claim.trigger_uri
        and record.started_at > claim.anchor
        and record.started_at < until
        and consequent in behaviour_hits(record, concepts)
    ]
    return tuple(sorted(found, key=lambda record: (record.started_at.astimezone(UTC), record.occurrence_uri)))


def releasing_record(claim: Claim, hypothesis: Hypothesis, records: Sequence[ConceptHits], concepts: ConceptSet, *, until: datetime) -> ConceptHits | None:
    """锚之后、``until`` 之前第一条命中 ``released_by`` 的记录（无节律型的作废条件）；口径与后件同一套 ``behaviour_hits``。"""

    wanted = {concept_identity(name) for name in hypothesis.released_by}
    if not wanted:
        return None
    for record in sorted(records, key=lambda item: (item.started_at.astimezone(UTC), item.occurrence_uri)):
        if record.occurrence_uri == claim.trigger_uri or not claim.anchor < record.started_at < until:
            continue
        if wanted & set(behaviour_hits(record, concepts)):
            return record
    return None


def unjudged_moments(claim: Claim, records: Sequence[ConceptHits]) -> tuple[datetime, ...]:
    """这本账的后件被映射器记成「未决」的那些时刻（评审 A-4）。

    「未决」是"判不了"，不是"没命中"：要当天时间线没给、要的常态没给、模型这一次没答成、或者判据引用的事件落在
    观测空白里（09-27 的三态 unknown）。落在某个机会 span 里的一条未决 = **那次机会没看清**——不算它过了、
    也不算缺席（用户 09-27"没看到就不进分母更不会进入分子"）。不这么判的话，映射侧好不容易不答 false 了，
    账本这边照样把那次机会算成"后件没来"。
    """

    consequent = claim.consequent
    return tuple(record.started_at for record in records if any(_identity(name) == consequent for name in record.unresolved))


def _identity(name: str) -> str:
    """未决里存的是概念名，比的是身份；名字已经不合法（概念被改名/删掉）就不可能是这本账的后件。"""

    try:
        return concept_identity(name)
    except ConceptError:
        return ""


def observed(
    opportunity: Opportunity, coverage: CoverageProvider, config: LedgerConfig, *, unjudged: Collection[datetime] = ()
) -> bool:
    """这个机会看清了没：span 的覆盖比例不低于阈值，**且**这段里没有把后件记成「未决」的记录（A-4）。"""

    if any(opportunity.span.contains(moment) for moment in unjudged):
        return False
    return coverage.coverage(opportunity.span).observed_fraction >= config.opportunity_coverage


def passes_until(
    snapshot: OpportunitySnapshot,
    *,
    before_index: int | None,
    until: datetime,
    coverage: CoverageProvider,
    config: LedgerConfig,
    unjudged: Collection[datetime] = (),
) -> tuple[OpportunityPass, ...]:
    """已经过去、后件没来的机会：``before_index`` 给了就是第 1 到第 before_index−1 次；没给就是到 ``until`` 已经结束的那些。"""

    if before_index is not None:
        items = snapshot.opportunities[: before_index - 1]
    else:
        items = tuple(item for item in snapshot.opportunities if item.span.end <= until)
    return tuple(OpportunityPass(at=item.at, observed=observed(item, coverage, config, unjudged=unjudged)) for item in items)


def settle_probability(
    claim: Claim, records: Sequence[ConceptHits], concepts: ConceptSet, *, now: datetime, until: datetime, coverage: CoverageProvider, config: LedgerConfig
) -> Settlement | None:
    """结果不随"哪一晚跑的结算"变：后件在第 k 次机会才来、而它之前已观测地过了 ``censor_after`` 次 → 仍是右删失
    （夜批要是每晚都跑，早在第 censor_after 次就删失了；隔了三周一次性补跑不能把它改写成"来了"）。

    后件晚于快照最后一个机会才来（``index_of`` 给不出名次）→ 也记右删失、带全部机会：快照说不出它落在第几次，
    写一个饱和的假名次会进 KM 的步数。
    """

    seen = consequent_records(claim, records, concepts, until=until)
    snapshot = claim.control
    unjudged = unjudged_moments(claim, records)
    if seen:
        first = seen[0]
        latency = (first.started_at - claim.anchor).total_seconds() / 3600.0
        index = None if snapshot is None else snapshot.index_of(first.started_at)
        beyond_snapshot = snapshot is not None and index is None
        passes = (
            ()
            if snapshot is None
            else passes_until(snapshot, before_index=None if beyond_snapshot else index, until=until, coverage=coverage, config=config, unjudged=unjudged)
        )
        censored_at = _censor_point(passes, config.censor_after)
        if censored_at is not None:
            return Settlement(ref=claim.ref, outcome=Outcome.CENSORED, settled_at=now, passes=passes[:censored_at])
        if beyond_snapshot:
            return Settlement(
                ref=claim.ref, outcome=Outcome.CENSORED, settled_at=now, passes=passes, reason="后件晚于快照最后一个机会才到来；这一批机会里它没来"
            )
        return Settlement(
            ref=claim.ref,
            outcome=Outcome.OCCURRED,
            settled_at=now,
            observed_at=first.started_at,
            fulfilling_uri=first.occurrence_uri,
            latency_hours=latency,
            opportunity_index=index,
            passes=passes,
        )
    if snapshot is None:
        return None
    passes = passes_until(snapshot, before_index=None, until=until, coverage=coverage, config=config, unjudged=unjudged)
    censored_at = _censor_point(passes, config.censor_after)
    if censored_at is not None:
        return Settlement(ref=claim.ref, outcome=Outcome.CENSORED, settled_at=now, passes=passes[:censored_at])
    if until >= snapshot.last_end:
        # 快照铺到头了、后件一次没来：账不能永远挂着，收成右删失（步数 = 看清了的机会数）。
        return Settlement(ref=claim.ref, outcome=Outcome.CENSORED, settled_at=now, passes=passes, reason="快照里的机会全过完了，后件一次没来")
    return None


def settle_open_ended(
    claim: Claim,
    hypothesis: Hypothesis,
    records: Sequence[ConceptHits],
    concepts: ConceptSet,
    *,
    now: datetime,
    until: datetime,
    coverage: CoverageProvider,
    config: LedgerConfig,
    consumed: Collection[str] = (),
) -> Settlement | None:
    """无节律型的结算：兑现 / 释放 / 一直开着。**不按机会数删失**——"很多行为没有机会时效"（用户 09-27）。

    ``consumed`` 是这条假设已经被别的承诺用掉的后件 occurrence（FIFO）：调用方按锚序结，结掉一条就把它兑现的那条
    occurrence 放进来。同一条 occurrence 兑现两条承诺 = 约球两次、打球一次读成两次都去了。

    释放与兑现同时在场时**兑现优先**（那件事确实做了）；释放只在它**早于**兑现时收口。
    """

    snapshot = claim.control
    unjudged = unjudged_moments(claim, records)
    seen = [record for record in consequent_records(claim, records, concepts, until=until) if record.occurrence_uri not in consumed]
    first = seen[0] if seen else None
    release = releasing_record(claim, hypothesis, records, concepts, until=until)
    if release is not None and (first is None or release.started_at < first.started_at):
        passes = (
            () if snapshot is None else passes_until(snapshot, before_index=None, until=release.started_at, coverage=coverage, config=config, unjudged=unjudged)
        )
        return Settlement(ref=claim.ref, outcome=Outcome.RELEASED, settled_at=now, passes=passes, releasing_uri=release.occurrence_uri)
    if first is None:
        return None
    index = None if snapshot is None else snapshot.index_of(first.started_at)
    passes = () if snapshot is None else passes_until(snapshot, before_index=index, until=until, coverage=coverage, config=config, unjudged=unjudged)
    return Settlement(
        ref=claim.ref,
        outcome=Outcome.OCCURRED,
        settled_at=now,
        observed_at=first.started_at,
        fulfilling_uri=first.occurrence_uri,
        latency_hours=(first.started_at - claim.anchor).total_seconds() / 3600.0,
        opportunity_index=index,
        passes=passes,
    )


def _censor_point(passes: Sequence[OpportunityPass], censor_after: int) -> int | None:
    """第几个 pass 之后满了 ``censor_after`` 个已观测的机会（切片终点）；没满返回 None。"""

    observed_count = 0
    for position, item in enumerate(passes, start=1):
        if item.observed:
            observed_count += 1
            if observed_count >= censor_after:
                return position
    return None


def settle_timing(
    claim: Claim, hypothesis: Hypothesis, records: Sequence[ConceptHits], concepts: ConceptSet, *, now: datetime, until: datetime, coverage: CoverageProvider, config: LedgerConfig
) -> Settlement | None:
    snapshot = claim.control
    if snapshot is None or hypothesis.expected_at is None:
        return None
    target = snapshot.at(hypothesis.expected_at)
    if target is None:
        # 快照没铺到第 expected_at 次机会（开承诺时要的个数不够）。快照事后不改，等下去也等不出来：
        # 收成删失并说明原因，别把承诺永久挂在"还没到时候"里。
        return Settlement(ref=claim.ref, outcome=Outcome.CENSORED, settled_at=now, reason=f"快照只有 {len(snapshot.opportunities)} 个机会，量不到第 {hypothesis.expected_at} 次")
    unjudged = unjudged_moments(claim, records)
    for record in consequent_records(claim, records, concepts, until=until):
        if snapshot.index_of(record.started_at) == hypothesis.expected_at:
            return Settlement(
                ref=claim.ref,
                outcome=Outcome.OBSERVED,
                settled_at=now,
                observed_at=record.started_at,
                fulfilling_uri=record.occurrence_uri,
                opportunity_index=hypothesis.expected_at,
            )
    if not _opportunity_over(snapshot, hypothesis.expected_at, until):
        return None
    if not observed(target, coverage, config, unjudged=unjudged):
        return Settlement(ref=claim.ref, outcome=Outcome.CENSORED, settled_at=now)
    return Settlement(ref=claim.ref, outcome=Outcome.ABSENT, settled_at=now, opportunity_index=hypothesis.expected_at)


def settle_count(
    claim: Claim, hypothesis: Hypothesis, records: Sequence[ConceptHits], concepts: ConceptSet, *, now: datetime, until: datetime, coverage: CoverageProvider, config: LedgerConfig
) -> Settlement | None:
    snapshot = claim.control
    if snapshot is None or hypothesis.expected_at is None:
        return None
    first, last = hypothesis.expected_at, hypothesis.expected_at + hypothesis.horizon - 1
    if snapshot.at(last) is None:
        return Settlement(ref=claim.ref, outcome=Outcome.CENSORED, settled_at=now, reason=f"快照只有 {len(snapshot.opportunities)} 个机会，数不到第 {last} 次")
    if not _opportunity_over(snapshot, last, until):
        return None
    window = snapshot.opportunities[first - 1 : last]
    unjudged = unjudged_moments(claim, records)
    if any(not observed(item, coverage, config, unjudged=unjudged) for item in window):
        return Settlement(ref=claim.ref, outcome=Outcome.CENSORED, settled_at=now)
    count = 0
    for record in consequent_records(claim, records, concepts, until=until):
        index = snapshot.index_of(record.started_at)
        if index is not None and first <= index <= last:
            count += 1
    return Settlement(ref=claim.ref, outcome=Outcome.COUNTED, settled_at=now, count=count)


def _opportunity_over(snapshot: OpportunitySnapshot, index: int, now: datetime) -> bool:
    """第 ``index`` 次机会算不算过完了：下一次机会已经开始（之后的到来归下一次），或没有下一次而它自己已结束。"""

    following = snapshot.at(index + 1)
    if following is not None:
        return now >= following.span.start
    current = snapshot.at(index)
    return current is not None and now >= current.span.end


def settle_claim(
    claim: Claim,
    hypothesis: Hypothesis,
    records: Sequence[ConceptHits],
    concepts: ConceptSet,
    *,
    now: datetime,
    coverage: CoverageProvider,
    config: LedgerConfig,
    until: datetime | None = None,
    consumed: Collection[str] = (),
) -> Settlement | None:
    """按方面分派（概率再按两型分派）；还不到时候返回 None。``until`` 缺省 = ``now``（见模块 docstring）。

    ``consumed`` 只对无节律型有意义（FIFO，见 ``settle_open_ended``）。
    """

    if claim.hypothesis_identity != hypothesis.identity:
        raise ValueError("claim and hypothesis disagree about which relation this is")
    horizon = now if until is None else until
    if claim.aspect is Aspect.PROBABILITY:
        if hypothesis.is_open_ended:
            return settle_open_ended(claim, hypothesis, records, concepts, now=now, until=horizon, coverage=coverage, config=config, consumed=consumed)
        return settle_probability(claim, records, concepts, now=now, until=horizon, coverage=coverage, config=config)
    if claim.aspect is Aspect.TIMING:
        return settle_timing(claim, hypothesis, records, concepts, now=now, until=horizon, coverage=coverage, config=config)
    return settle_count(claim, hypothesis, records, concepts, now=now, until=horizon, coverage=coverage, config=config)


def close_claim(claim: Claim, *, reason: str, now: datetime, coverage: CoverageProvider, config: LedgerConfig | None = None) -> Settlement:
    """生命周期把一条还没结的承诺关掉：右删失，带到现在为止过去的机会。关闭规则本身属生命周期那份，这里只留写口。"""

    resolved = config or LedgerConfig()
    passes = () if claim.control is None else passes_until(claim.control, before_index=None, until=now, coverage=coverage, config=resolved)
    if claim.aspect is not Aspect.PROBABILITY:
        passes = ()
    return Settlement(ref=claim.ref, outcome=Outcome.CENSORED, settled_at=now, passes=passes, reason=reason)


def settle_due(
    hypothesis: Hypothesis,
    *,
    concepts: ConceptSet,
    hits: ConceptHitStore,
    ledger: LedgerStore,
    coverage: CoverageProvider,
    now: datetime,
    until: datetime | None = None,
    config: LedgerConfig | None = None,
) -> SettlementReport:
    """把这条假设所有到时候的承诺结掉。夜批里排在"开承诺"之后、"算 views"之前。

    ``until`` = 命中记录可信到哪一刻（最后一个已映射日的日界）；不给就是 ``now``，那要调用方自己保证映射没落后。
    """

    resolved = config or LedgerConfig()
    horizon = mapped_until(hits, now) if until is None else until
    open_claims = ledger.open_claims(hypothesis.identity)
    if not open_claims:
        return SettlementReport(hypothesis_identity=hypothesis.identity, settled=0, pending=0)
    if horizon is None:
        return SettlementReport(hypothesis_identity=hypothesis.identity, settled=0, pending=len(open_claims))
    if hypothesis.consequent_identity not in concepts:
        # 后件概念被删掉了：``behaviour_hits`` 会跳过不在集里的身份，于是后件永远匹配不上，
        # 等够次数全部记成删失——凭空造出来的记录，而且 add-only 撤不掉。一条都不结。
        return SettlementReport(hypothesis_identity=hypothesis.identity, settled=0, pending=len(open_claims))
    earliest = min(claim.anchor for claim in open_claims)
    records = hits.read_window(earliest, horizon) if horizon > earliest else ()
    settled = _settle_each(open_claims, hypothesis, records, concepts, now=now, until=horizon, coverage=coverage, config=resolved, ledger=ledger)
    return SettlementReport(hypothesis_identity=hypothesis.identity, settled=settled, pending=len(open_claims) - settled)


def _settle_each(
    open_claims: Sequence[Claim],
    hypothesis: Hypothesis,
    records: Sequence[ConceptHits],
    concepts: ConceptSet,
    *,
    now: datetime,
    until: datetime,
    coverage: CoverageProvider,
    config: LedgerConfig,
    ledger: LedgerStore,
) -> int:
    """按锚序结一条假设的开放承诺，返回结掉几条。

    无节律型要 **FIFO**：``consumed`` 先装盘上已经用掉的后件 occurrence（前几晚结的那些），再随这一轮结掉的往里加。
    不从盘上装的话，第一晚 A 被 X 兑现之后，下一晚锚更晚的 B 会被同一条 X 再兑现一次（评审 A-3）。节律型
    各自量自己的第 k 次机会，共用一条后件记录是对的，不进这条路。
    """

    consumed: set[str] = set()
    if hypothesis.is_open_ended:
        consumed = {item.fulfilling_uri for item in ledger.settlements_for(hypothesis.identity) if item.fulfilling_uri is not None}
    settled = 0
    for claim in sorted(open_claims, key=lambda item: (item.anchor, item.trigger_uri)):
        settlement = settle_claim(
            claim, hypothesis, records, concepts, now=now, until=until, coverage=coverage, config=config, consumed=consumed
        )
        if settlement is None:
            continue
        ledger.write_settlement(settlement)
        settled += 1
        if settlement.fulfilling_uri is not None:
            consumed.add(settlement.fulfilling_uri)
    return settled


def settle_due_all(
    hypotheses: Sequence[Hypothesis],
    *,
    concepts: ConceptSet,
    hits: ConceptHitStore,
    ledger: LedgerStore,
    coverage: CoverageProvider,
    now: datetime,
    until: datetime | None = None,
    config: LedgerConfig | None = None,
) -> tuple[SettlementReport, ...]:
    """一晚把全部假设结一遍：命中记录只读一次（从最早的未结锚到 ``until``），再按假设分发。

    ``settle_due`` 每条假设各读一遍记录库，320 条假设实测 203 秒；这里读盘与结算解耦。
    """

    resolved = config or LedgerConfig()
    horizon = mapped_until(hits, now) if until is None else until
    pending = {hypothesis.identity: ledger.open_claims(hypothesis.identity) for hypothesis in hypotheses}
    if horizon is None:
        return tuple(
            SettlementReport(hypothesis_identity=item.identity, settled=0, pending=len(pending[item.identity])) for item in hypotheses
        )
    anchors = [claim.anchor for claims in pending.values() for claim in claims]
    earliest = min(anchors) if anchors else None
    records: tuple[ConceptHits, ...] = hits.read_window(earliest, horizon) if earliest is not None and horizon > earliest else ()
    reports: list[SettlementReport] = []
    for hypothesis in hypotheses:
        open_claims = pending[hypothesis.identity]
        if not open_claims or hypothesis.consequent_identity not in concepts:
            reports.append(SettlementReport(hypothesis_identity=hypothesis.identity, settled=0, pending=len(open_claims)))
            continue
        settled = _settle_each(open_claims, hypothesis, records, concepts, now=now, until=horizon, coverage=coverage, config=resolved, ledger=ledger)
        reports.append(SettlementReport(hypothesis_identity=hypothesis.identity, settled=settled, pending=len(open_claims) - settled))
    return tuple(reports)


__all__ = [
    "SettlementReport",
    "close_claim",
    "consequent_records",
    "mapped_until",
    "observed",
    "passes_until",
    "releasing_record",
    "settle_claim",
    "settle_count",
    "settle_due",
    "settle_due_all",
    "settle_open_ended",
    "settle_probability",
    "settle_timing",
    "unjudged_moments",
]
