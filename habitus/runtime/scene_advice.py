"""关系检验的两样模型建议在组合根的接线：先验 → 权重，候选调节条件 → 提议簿（语义树新方案 ``13`` ②）。

- **先验**：混合问法（实时化阶段 2）——一对（前因, 后果）只归一题：后果比前因晚出现的归"这个后果"那一题（旧问法，
  固定后果、列前因），否则归"这个前因"那一题（新问法，固定前因、列后果）；每题三遍乱序取中位，两份缓存分开存
  （``runtime.scene_priors``）。一个新概念出现只多两题。模型队列对着"概念集 + 将要生成的基础概念 + 待生效的概念"问
  （``ask_priors``），答案直接写缓存；换代折叠时只读（``weights``）。
  权重只给不带条件的关系（短跨度与长跨度都给，按前因、后果这一对）；情境概念当前因的、带条件的按 1。
- **候选条件**：只问够样本的前因（与调节检验同一个门槛），只在概念集变了（或前因刚够样本）时问；收下的与拒掉的都记进提议簿。
  第 N 晚提出的只在 N 之后的数据上检验；回放更早的一晚时，比它晚提出的不拿来用（不然就是拿未来的提议去检验过去）。
- ``cache_only``：只读缓存，缓存里没有的不问模型、按没有先验 / 没有新提议处理，并报出来。换代折叠关系时用的就是这一份
  （``cached``）：补问归模型队列（``RelationStage.ask``），换代里没有模型调用。
- ``owed`` / ``priors_owed``：这条 lane 还欠几件事（要问的候选条件、提议簿要整理；要问的先验题、孤儿先验要清）——模型队列
  据它们从盘上认欠账，取法与补问时同一套。
"""

from __future__ import annotations

from collections.abc import Container, Mapping
from datetime import date, tzinfo

from habitus.conditions import DAY_TYPE_KEY, WEEKDAY_NAMES
from habitus.foundation.integrity import canonical_digest
from habitus.runtime.scene_priors import LanePriors
from habitus.scene.advice.conditions import CONDITION_PROMPT_VERSION, ConditionProposer, Occasion
from habitus.scene.advice.conditions import question_for as condition_question
from habitus.scene.advice.model import Proposal, ProposalBook, tidy_book
from habitus.scene.advice.prior import PriorAdvisor
from habitus.scene.advice.store import AdviceStore, AdviceStoreError
from habitus.scene.concepts.calendar_situations import calendar_key, is_calendar
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.relations import LaneTests, LaneTimeline, RelationKey
from habitus.scene.relations.conditions import Condition, parse
from habitus.scene.relations.spans import Chain

#: 给候选条件那个触点看几次 A（按时间均匀取）。
OCCASIONS = 6
#: 每次 A 带它当天之前的几件事。
BEFORE = 3


class RelationAdvice:
    def __init__(
        self,
        *,
        store: AdviceStore,
        prior: PriorAdvisor | None,
        conditions: ConditionProposer | None,
        cache_only: bool = False,
        zone: tzinfo | None = None,
    ) -> None:
        self.store = store
        self.prior = prior
        self.conditions = conditions
        self.cache_only = cache_only
        #: 概念"哪一天出现"按这个时区的当地日期算（先验归题用）；没给按 UTC。
        self.zone = zone

    def cached(self) -> RelationAdvice:
        """只读缓存的那一份（同一个存储、同样的触点版本）：不问模型、不整理提议簿、不清孤儿先验。"""

        return RelationAdvice(
            store=self.store, prior=self.prior, conditions=self.conditions, cache_only=True, zone=self.zone
        )

    def priors(self, concepts: ConceptSet, lane: str, settled: Container[str] | None = None) -> LanePriors:
        """这条 lane 的先验（两份缓存 + 归题）。``settled``：``concepts`` 里已经进了概念集的身份（不给就是全部）。"""

        assert self.prior is not None
        return LanePriors(self.store, self.prior, concepts, lane, settled=settled, zone=self.zone)

    def owed(self, tests: LaneTests) -> int:
        """这条 lane 的候选条件还欠几件事：要问的前因数 + 提议簿要整理（1）。只读缓存的那一份不欠（它不问、不整理）。"""

        if self.cache_only:
            return 0
        concepts, lane = tests.concepts, tests.line.lane
        book = self.store.book(lane)
        owed = int(_tidied(book, concepts) != book)
        if self.conditions is not None:
            owed += len(self._due_antecedents(tests, book, _condition_fingerprint(concepts, lane)))
        return owed

    def priors_owed(self, concepts: ConceptSet, lane: str, settled: Container[str] | None = None) -> int:
        """这条 lane 的先验还欠几件事：两种问法各要问的题数 + 要清的孤儿缓存数。"""

        if self.cache_only or self.prior is None:
            return 0
        priors = self.priors(concepts, lane, settled)
        old, new = priors.orphans()
        return priors.questions().count + len(old) + len(new)

    async def ask_priors(
        self, concepts: ConceptSet, lane: str, signals: list[str], settled: Container[str] | None = None
    ) -> None:
        """补问这条 lane 的先验（模型队列调，``concepts`` 是"概念集 + 同步词表将要生成的基础概念 + 待生效的概念"）、清孤儿缓存。
        答案直接写缓存：只有换代读它，概念进概念集的那一代先验已经备好。"""

        if self.cache_only or self.prior is None:
            return
        priors = self.priors(concepts, lane, settled)
        await priors.ask(signals)
        priors.discard_orphans()

    async def weights(self, tests: LaneTests, signals: list[str]) -> dict[RelationKey, float]:
        """这条 lane 的先验权重（键是不带条件的关系）。不是只读缓存的那一份（离线一次做完）先把该问的问了。"""

        if self.prior is None:
            return {}
        concepts, lane = tests.concepts, tests.line.lane
        if not self.cache_only:
            await self.ask_priors(concepts, lane, signals)
        priors = self.priors(concepts, lane)
        levels = priors.levels()
        missing = priors.missing()
        if missing:
            signals.append(f"prior: {lane} 有 {missing} 个后果缺先验（缺的前因权重按 1）")
        return {
            key: levels[(key.antecedent, key.consequent)].weight
            for key in tests.keys()
            if not key.condition and (key.antecedent, key.consequent) in levels
        }

    async def proposals(
        self, tests: LaneTests, night: date, signals: list[str], *, proposed_on: date | None = None
    ) -> dict[str, tuple[tuple[Condition, date], ...]]:
        """前因 → （收下的条件, 提出的那一晚）。先补问该问的，再按"提出得不晚于今晚"取。``proposed_on`` 是这一回问到的提议
        记成哪一天提出的（不给就是 ``night``；模型队列在今天这一代之后问，记成下一次换代的日子——今天这一代已经折叠完了）。"""

        lane, concepts = tests.line.lane, tests.concepts
        book = self.store.book(lane)
        try:
            if self.conditions is not None and not self.cache_only:
                book = await self._ask(tests, book, proposed_on or night, signals)
            if not self.cache_only:
                tidied = _tidied(book, concepts)
                if tidied != book:
                    signals.append(
                        f"advice: {lane} 提议簿清掉 {len(book.proposals) - len(tidied.proposals)} 条（前因已删、条件读不了、重复的拒因）"
                    )
                    self.store.write_book(tidied)
                    book = tidied
        except AdviceStoreError as exc:
            # 提议簿写不出（超了单份上限等）：这一晚只用盘上已有的那份，其余照常；这一晚问到的没记下，下一晚会再问
            signals.append(f"advice: {lane} 提议簿写不出（{exc}），这一晚只用盘上已有的提议")
            book = self.store.book(lane)
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
        fingerprint = _condition_fingerprint(concepts, lane)
        proposals = list(book.proposals)
        asked = dict(book.asked)
        for antecedent in self._due_antecedents(tests, book, fingerprint):
            chains = tests.chains(antecedent)
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

    @staticmethod
    def _due_antecedents(tests: LaneTests, book: ProposalBook, fingerprint: str) -> list[str]:
        """这条 lane 要问候选条件的前因：够样本（与调节检验同一个门槛），且提议簿记的概念集指纹与现在的不同。"""

        concepts, lane = tests.concepts, tests.line.lane
        return [
            antecedent
            for antecedent in sorted(name for name in concepts.behaviors() if concepts.lane_of(name) == lane)
            if len(tests.chains(antecedent)) >= tests.config.thresholds.condition_ask_min_antecedents
            and book.asked.get(antecedent) != fingerprint
        ]


#: 每个前因保留的已拒提议条数：它们全量进提示词，给模型看"这些说过不行"，只留最近的（保护闸）。
MAX_REFUSED_PER_ANTECEDENT = 20


def _condition_fingerprint(concepts: ConceptSet, lane: str) -> str:
    """候选条件"问过没有"认的概念集指纹。只看这条 lane 的概念（与不属于任何 lane 的情境）：另一条 lane 改了概念不该连带这条
    重问（裁定 21-2；E9）。日历情境不算：它们不进问题（系统机械检验），长出一个周几集合不该让两条 lane 全部前因重问一遍；
    提示词版本算进去：改了提示词要重问（2026-10-18 四方评审）。"""

    return canonical_digest(
        {
            "version": CONDITION_PROMPT_VERSION,
            "concepts": sorted(
                concepts[name].fingerprint
                for name in concepts
                if concepts.lane_of(name) in (lane, None) and not is_calendar(concepts, name)
            ),
        }
    )


def _tidied(book: ProposalBook, concepts: ConceptSet) -> ProposalBook:
    return tidy_book(
        book,
        alive=lambda antecedent: antecedent in concepts,
        compiles=_compiles,
        max_refused_per_antecedent=MAX_REFUSED_PER_ANTECEDENT,
    )


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
                moment=f"{record.started_at:%Y-%m-%d %H:%M}（{WEEKDAY_NAMES[record.started_at.weekday()]}{_day_type(line, concepts, record.day)}）",
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


__all__ = ["RelationAdvice", "occasions"]
