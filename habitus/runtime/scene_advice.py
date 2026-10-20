"""关系检验的两样模型建议在组合根的接线：先验 → 权重，候选调节条件 → 提议簿（语义树新方案 ``13`` ②）。

- **先验**：每条 lane、每个后果问一次（三遍乱序取中位），按输入摘要 + 触点版本缓存；词表与概念不变就不重问。
  权重只给不带条件的关系（短跨度与长跨度都给，按前因、后果这一对）；情境概念当前因的、带条件的按 1。
- **候选条件**：只问够样本的前因（与调节检验同一个门槛），只在概念集变了（或前因刚够样本）时问；收下的与拒掉的都记进提议簿。
  第 N 晚提出的只在 N 之后的数据上检验；回放更早的一晚时，比它晚提出的不拿来用（不然就是拿未来的提议去检验过去）。
- ``cache_only``：回放与回测只读缓存，缓存里没有的不问模型、按没有先验 / 没有新提议处理，并报出来。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

from habitus.conditions import DAY_TYPE_KEY
from habitus.foundation.integrity import canonical_digest
from habitus.scene.advice.conditions import CONDITION_PROMPT_VERSION, ConditionProposer, Occasion
from habitus.scene.advice.conditions import question_for as condition_question
from habitus.scene.advice.model import PriorLevel, Proposal, ProposalBook, tidy_book
from habitus.scene.advice.prior import PriorAdvisor
from habitus.scene.advice.prior import question_for as prior_question
from habitus.scene.advice.store import AdviceStore
from habitus.scene.concepts.calendar_situations import calendar_key, is_calendar
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.relations import LaneTests, LaneTimeline, RelationKey, comparable
from habitus.scene.relations.conditions import Condition, parse
from habitus.scene.relations.spans import Chain

#: 给候选条件那个触点看几次 A（按时间均匀取）。
OCCASIONS = 6
#: 每次 A 带它当天之前的几件事。
BEFORE = 3
WEEKDAYS = "一二三四五六日"


class RelationAdvice:
    def __init__(
        self,
        *,
        store: AdviceStore,
        prior: PriorAdvisor | None,
        conditions: ConditionProposer | None,
        cache_only: bool = False,
    ) -> None:
        self.store = store
        self.prior = prior
        self.conditions = conditions
        self.cache_only = cache_only

    async def weights(self, tests: LaneTests, signals: list[str]) -> dict[RelationKey, float]:
        """这条 lane 的先验权重（键是不带条件的关系）。"""

        if self.prior is None:
            return {}
        concepts, lane = tests.concepts, tests.line.lane
        names = [name for name in concepts.behaviors() if concepts.lane_of(name) == lane]
        levels: dict[tuple[str, str], PriorLevel] = {}
        current: set[str] = set()
        missing = 0
        for consequent in names:
            antecedents = [name for name in names if comparable(concepts, name, consequent)]
            if not antecedents:
                continue
            # 按（后果, 单个前因）缓存（第四轮评审 E9）：以前缓存键是整份前因清单，加一个概念整条 lane 的后果全部重问三遍；
            # 现在只问这个后果还没答过的前因（前因改了名字或判据，只重问它）
            key = canonical_digest(
                {"lane": lane, "consequent": _pair_part(concepts, consequent), "advisor": self.prior.version}
            )
            current.add(key)
            cached = dict(self.store.prior(key) or {})
            ids = {antecedent: canonical_digest(_pair_part(concepts, antecedent))[:16] for antecedent in antecedents}
            unasked = [antecedent for antecedent in antecedents if ids[antecedent] not in cached]
            if unasked and not self.cache_only:
                answer = await self.prior.rate(prior_question(concepts, lane, consequent, unasked))
                signals.extend(answer.signals)
                if answer.levels:
                    cached.update({ids[antecedent]: level for antecedent, level in answer.levels.items()})
                    self.store.write_prior(
                        key,
                        lane=lane,
                        consequent=consequent,
                        levels=cached,
                        version=answer.version,
                        models=answer.models,
                    )
            known = {antecedent: cached[ids[antecedent]] for antecedent in antecedents if ids[antecedent] in cached}
            if len(known) < len(antecedents):
                missing += 1
            levels.update(((antecedent, consequent), level) for antecedent, level in known.items())
        if missing:
            signals.append(f"prior: {lane} 有 {missing} 个后果缺先验（缺的前因权重按 1）")
        if not self.cache_only:
            # 清这条 lane 的孤儿先验：后果改名、改判据、换了先验触点的版本或模型，旧键没人再读（26 号清单 2.5.5）
            for orphan in set(self.store.prior_keys(lane)) - current:
                self.store.discard_prior(orphan)
        return {
            key: levels[(key.antecedent, key.consequent)].weight
            for key in tests.keys()
            if not key.condition and (key.antecedent, key.consequent) in levels
        }

    async def proposals(
        self, tests: LaneTests, night: date, signals: list[str]
    ) -> dict[str, tuple[tuple[Condition, date], ...]]:
        """前因 → （收下的条件, 提出的那一晚）。先补问该问的，再按"提出得不晚于今晚"取。"""

        lane, concepts = tests.line.lane, tests.concepts
        book = self.store.book(lane)
        if self.conditions is not None and not self.cache_only:
            book = await self._ask(tests, book, night, signals)
        if not self.cache_only:
            tidied = tidy_book(
                book,
                alive=lambda antecedent: antecedent in concepts,
                compiles=_compiles,
                max_refused_per_antecedent=MAX_REFUSED_PER_ANTECEDENT,
            )
            if tidied != book:
                signals.append(
                    f"advice: {lane} 提议簿清掉 {len(book.proposals) - len(tidied.proposals)} 条（前因已删、条件读不了、重复的拒因）"
                )
                self.store.write_book(tidied)
                book = tidied
        usable: dict[str, list[tuple[Condition, date]]] = {}
        for item in book.proposals:
            if item.condition is not None and item.proposed_on <= night and item.antecedent in concepts:
                try:
                    condition = parse(item.condition)
                except ValueError:
                    # 旧提议簿里的条件落不成现在的模板（日历字段 v2 起拿掉了）：跳过并报出来，不让整晚抛（2026-10-18 四方评审）
                    signals.append(f"advice: {lane} 提议簿里的条件「{item.condition}」现在读不了，跳过")
                    continue
                usable.setdefault(item.antecedent, []).append((condition, item.proposed_on))
        return {antecedent: tuple(found) for antecedent, found in usable.items()}

    async def _ask(self, tests: LaneTests, book: ProposalBook, night: date, signals: list[str]) -> ProposalBook:
        assert self.conditions is not None
        concepts, lane = tests.concepts, tests.line.lane
        # 只看这条 lane 的概念（与不属于任何 lane 的情境）：另一条 lane 改了概念不该连带这条重问（裁定 21-2；E9）。日历情境不算：
        # 它们不进问题（系统机械检验），长出一个周几集合不该让两条 lane 全部前因重问一遍；提示词版本算进去：改了提示词要重问
        # （2026-10-18 四方评审）。
        fingerprint = canonical_digest(
            {
                "version": CONDITION_PROMPT_VERSION,
                "concepts": sorted(
                    concepts[name].fingerprint
                    for name in concepts
                    if concepts.lane_of(name) in (lane, None) and not is_calendar(concepts, name)
                ),
            }
        )
        proposals = list(book.proposals)
        asked = dict(book.asked)
        for antecedent in sorted(name for name in concepts.behaviors() if concepts.lane_of(name) == lane):
            chains = tests.chains(antecedent)
            if (
                len(chains) < tests.config.thresholds.condition_ask_min_antecedents
                or asked.get(antecedent) == fingerprint
            ):
                continue
            question = condition_question(
                concepts,
                lane,
                antecedent,
                occasions(tests, concepts, chains),
                accepted=sorted(book.known(antecedent)),
                refused=sorted(_refused_text(item) for item in book.refused(antecedent)),
            )
            answer = await self.conditions.propose(question, concepts, night=night)
            signals.extend(answer.signals)
            if not answer.answered:
                continue  # 没答成：下一晚再问
            known = book.known(antecedent)
            proposals.extend(item for item in answer.proposals if item.condition is None or item.condition not in known)
            asked[antecedent] = fingerprint
        updated = ProposalBook(lane=lane, proposals=tuple(proposals), asked=asked)
        if updated != book:
            self.store.write_book(updated)
        return updated


#: 每个前因保留的已拒提议条数：它们全量进提示词，给模型看"这些说过不行"，只留最近的（保护闸）。
MAX_REFUSED_PER_ANTECEDENT = 20


def _compiles(condition: str) -> bool:
    try:
        parse(condition)
    except ValueError:
        return False
    return True


def _day_type(line: LaneTimeline, concepts: ConceptSet, day: date) -> str:
    """那天的日型（日历情境机械给的，裁定 59）；答不出就不写。"""

    held = [
        concepts[identity].name
        for identity in sorted(line.calendar)
        if (key := calendar_key(concepts, identity)) is not None
        and key[0] == DAY_TYPE_KEY
        and line.situations.get(identity, {}).get(day)
    ]
    return f"，{'、'.join(held)}" if held else ""


def occasions(tests: LaneTests, concepts: ConceptSet, chains: tuple[Chain, ...]) -> tuple[Occasion, ...]:
    """按时间均匀挑几次 A；每次给它自己与当天之前的几件事（同 lane）。不给之后发生的任何事。"""

    line = tests.line
    if not chains:
        return ()
    step = max(1, len(chains) // OCCASIONS)
    picked = list(chains[::step])[:OCCASIONS]
    by_day: dict[date, list[str]] = {}
    for uri, record in sorted(line.records.items(), key=lambda item: (item[1].started_at, item[0])):
        by_day.setdefault(record.day, []).append(uri)
    found = []
    for chain in picked:
        record = line.records[chain.anchor.uri]
        before = [
            line.records[uri] for uri in by_day.get(record.day, []) if line.records[uri].started_at < record.started_at
        ][-BEFORE:]
        found.append(
            Occasion(
                moment=f"{record.started_at:%Y-%m-%d %H:%M}（周{WEEKDAYS[record.started_at.weekday()]}{_day_type(line, concepts, record.day)}）",
                summary=record.summary,
                goal=record.goal,
                place=record.place,
                subjects=record.subjects,
                before=tuple(
                    f"{item.started_at:%H:%M} {concepts.class_label(item.kind_token) if item.classified else '待定'}：{item.summary}"
                    for item in before
                ),
            )
        )
    return tuple(found)


def _refused_text(item: Proposal) -> str:
    raw: Mapping[str, object] = item.raw
    parts = [f"{key}={value}" for key, value in raw.items() if value]
    return f"{'，'.join(parts)}（{item.refused}）"


def _pair_part(concepts: ConceptSet, name: str) -> list[str]:
    """先验缓存键里一个概念的那一份：给模型看的名字与判据（改名或改判据就是另一个问题）。"""

    return [concepts.label_of(name), concepts[name].definition]


__all__ = ["RelationAdvice", "occasions"]
