"""闭环的组合根那一半：把算法扫出来的"不一样"备成材料，交触点③，把它提的假设写下去。

**两个触发源**（2026-09-27 裁定）：

1. **调节**：稳定性扫出某条关系按某个情境分层之后两层**分得开**（``StabilityReport.moderations``）；
2. **常态漂移**：某个概念的近期常态与历来常态明显不同（``BaselineSnapshot.drifting``）。

两者都只是"不一样"这个事实。算法说不出为什么，所以交给模型读两边的叙事。

**这一层的职责是"备材料"，而且要小心不把数字带过去**（B8）：`moderations` 与 `drifts` 上都挂着强度、
区间、分钟数，这里**只取名字与方向**，叙事从命中记录里现取。给模型看见数字，它下一轮写的假设就会去
迎合数字，先验与读数互相自证。

**叙事是 occurrence 的一行一条**（时刻 + 概念名 + kind），不是原始文本：它要让模型看出"这几天和那几天
差在哪"，而不是把行为树整个搬过去。**取的是那条关系的承诺周围**（锚到第一个机会结束）——旧写法取"那个情境
在场的所有日子里最早 20 条"，与要解释的关系无关，模型拿到的材料里可能根本没有那两个行为（评审 C-10）。

**问过的不再问**（``HypothesisStore.closure_asked``）："不一样"是从读数里现扫的，账没变它每晚都在；
不记就每晚重问同三条、第 4 条起永远轮不到（评审 A-5 / B-7）。

**不回填**：新假设的 ``created_at`` 是**夜批的时刻**（``now`` 一路传到触点③），账本开账时按它拦（裁定四）。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from habitus.scene.concepts.model import ConceptSet, concept_identity
from habitus.scene.concepts.rhythm import Rhythm
from habitus.scene.hypotheses.author import NO_RHYTHMS
from habitus.scene.hypotheses.closure import ClosureAuthor, ClosureProposal, DriftFact, SplitFact
from habitus.scene.hypotheses.model import Hypothesis
from habitus.scene.hypotheses.store import HypothesisStore
from habitus.scene.ledger.store import LedgerStore
from habitus.scene.occurrences.baselines import BaselineSnapshot
from habitus.scene.occurrences.model import ConceptHits
from habitus.scene.occurrences.store import ConceptHitStore
from habitus.scene.views.relations import RelationReading

#: 一夜最多问几次触点③。闭环每晚都会扫出一批"不一样"，而每条新假设都要再攒几周——不设闸的话
#: 假设集会自己膨胀。**待定值**。
MAX_CLOSURE_ROUNDS = 3
#: 每边取几天的叙事（漂移用：近期 N 天对更早的 N 天，两段不重叠）。
NARRATIVE_DAYS = 3
#: 一条承诺的叙事往后取多久（没有对照快照时的兜底；有快照取到第一个机会结束）。
CLAIM_NARRATIVE_HOURS = 24


def fact_key(fact: SplitFact | DriftFact) -> str:
    """一件"不一样"的稳定键，用来记"问过了"。"""

    if isinstance(fact, SplitFact):
        return "split|" + fact.consequent + "|" + "+".join(sorted(fact.antecedents)) + "|" + fact.situation
    return f"drift|{fact.concept}|{fact.quantity}|{fact.direction}"


@dataclass(frozen=True)
class ClosureReport:
    """一夜闭环的账：问了几次、写下几条、每次的解释与留痕。"""

    asked: int = 0
    written: int = 0
    explanations: tuple[str, ...] = ()
    signals: tuple[str, ...] = ()

    def summary(self) -> str:
        return f"闭环：问了 {self.asked} 次，写下 {self.written} 条新假设"


def split_facts(
    readings: Sequence[RelationReading],
    hypotheses: Mapping[str, Hypothesis],
    narratives: Callable[[Hypothesis, str], tuple[tuple[str, ...], tuple[str, ...]]],
) -> tuple[SplitFact, ...]:
    """把"分得开的那些调节"备成材料。``narratives(假设, 情境)`` 给这条关系两边的叙事（见 ``relation_narratives``）。

    只取**名字**：哪条关系、按哪个情境分的。强度与区间留在读数里，不往这边带（B8）。
    """

    found: list[SplitFact] = []
    for reading in readings:
        hypothesis = hypotheses.get(reading.hypothesis_identity)
        if hypothesis is None or hypothesis.source.origin.is_placebo:
            continue  # 安慰剂分得开只说明读法在冒泡，不该拿去问"为什么"
        for moderation in reading.stability.moderations:
            with_situation, without_situation = narratives(hypothesis, moderation.situation)
            if not with_situation or not without_situation:
                continue  # 有一边没有叙事就问不出东西
            found.append(
                SplitFact(
                    consequent=hypothesis.consequent,
                    antecedents=tuple(item.concept for item in hypothesis.antecedents),
                    situation=moderation.situation,
                    with_situation=with_situation,
                    without_situation=without_situation,
                )
            )
    return tuple(found)


def drift_facts(baselines: BaselineSnapshot, narratives: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]]) -> tuple[DriftFact, ...]:
    """把"在漂的那些常态"备成材料。只取方向（往后/往前、变长/变短），**不取分钟数**。"""

    found: list[DriftFact] = []
    for drift in baselines.drifting:
        recent, earlier = narratives.get(drift.concept, ((), ()))
        if not recent or not earlier:
            continue
        later = drift.minutes > 0
        quantity = drift.statistic.quantity
        direction = ("后" if later else "前") if quantity == "时刻" else ("长" if later else "短")
        found.append(DriftFact(concept=drift.concept, quantity=quantity, direction=direction, recent=recent, earlier=earlier))
    return tuple(found)


def relation_narratives(
    ledger: LedgerStore,
    hits: ConceptHitStore,
    hypothesis: Hypothesis,
    situation: str,
    concepts: ConceptSet,
    *,
    limit: int = 20,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """这条关系按某个情境分成两边的叙事：(在场, 不在场)。

    每条承诺取锚到第一个机会结束（没有快照就 24 小时）之间的记录——前件、后件、中间发生的事都在里面，
    模型看的正是"这次前件发生之后到后件该来之前，两边各发生了什么"。
    """

    wanted = concept_identity(situation)
    sides: dict[bool, list[ConceptHits]] = {True: [], False: []}
    for claim in ledger.claims_for(hypothesis.identity):
        present = wanted in {concept_identity(name) for name in claim.situation_snapshot}
        end = claim.control.opportunities[0].span.end if claim.control is not None else claim.anchor + timedelta(hours=CLAIM_NARRATIVE_HOURS)
        sides[present].extend(hits.read_window(claim.anchor, end))
    return narrative_of(sides[True], concepts, limit=limit), narrative_of(sides[False], concepts, limit=limit)


def narrative_of(records: Sequence[ConceptHits], concepts: ConceptSet, *, limit: int = 20) -> tuple[str, ...]:
    """一批命中记录 → 给模型读的一行一条（时刻 · 命中的概念 · 上游的 kind）。

    带上 kind 是因为概念名可能很抽象（「实施软件代码改动」），而 kind 保留了那一次的具体说法。
    """

    lines = []
    unique = {record.occurrence_uri: record for record in records}
    for record in sorted(unique.values(), key=lambda item: item.started_at)[:limit]:
        names = "、".join(concepts[hit.identity].name for hit in record.hits if hit.identity in concepts)
        lines.append(f"{record.started_at.strftime('%m-%d %H:%M')} {names or '（没命中任何概念）'}（{record.kind_token}）")
    return tuple(lines)


def day_narratives(hits: ConceptHitStore, days: Sequence[date], concepts: ConceptSet, *, limit: int = 20) -> tuple[str, ...]:
    records = [record for day in days for record in hits.read_day(day)]
    return narrative_of(records, concepts, limit=limit)


async def run_closure(
    author: ClosureAuthor,
    facts: Sequence[SplitFact | DriftFact],
    concepts: ConceptSet,
    store: HypothesisStore,
    *,
    day: date,
    now: datetime,
    known: Sequence[Hypothesis] = (),
    rhythms: Mapping[str, Rhythm] = NO_RHYTHMS,
    max_rounds: int = MAX_CLOSURE_ROUNDS,
) -> ClosureReport:
    """逐条问、逐条写。**写进去就开始攒账，不回填**——用来发现它的那批观测不算它的证据。

    问过的事实（``store.closure_asked``）跳过，没问过的按给定顺序取前 ``max_rounds`` 条；问完就记下，
    模型一条都没提也记——再问一遍答案不会变。
    """

    asked = written = 0
    explanations: list[str] = []
    signals: list[str] = []
    seen = list(known)
    already = store.closure_asked()
    fresh = [fact for fact in facts if fact_key(fact) not in already]
    if len(fresh) < len(facts):
        signals.append(f"closure: {len(facts) - len(fresh)} 件不一样问过了，跳过")
    for fact in fresh[:max_rounds]:
        proposal: ClosureProposal = await author.propose(fact, concepts, known=seen, rhythms=rhythms, now=now)
        store.record_closure_asked(fact_key(fact), day)
        asked += 1
        signals.extend(proposal.signals)
        if proposal.explanation:
            explanations.append(proposal.explanation)
        for hypothesis in proposal.hypotheses:
            try:
                store.write(hypothesis, concepts)
            except Exception as exc:  # noqa: BLE001 - 写不进去要报出来，不静默吞掉一条新假设
                signals.append(f"closure: 写不进 {hypothesis.identity}：{type(exc).__name__}: {exc}")
                continue
            seen.append(hypothesis)
            written += 1
    return ClosureReport(asked=asked, written=written, explanations=tuple(explanations), signals=tuple(signals))


__all__ = [
    "CLAIM_NARRATIVE_HOURS",
    "MAX_CLOSURE_ROUNDS",
    "NARRATIVE_DAYS",
    "ClosureReport",
    "day_narratives",
    "drift_facts",
    "fact_key",
    "narrative_of",
    "relation_narratives",
    "run_closure",
    "split_facts",
]
