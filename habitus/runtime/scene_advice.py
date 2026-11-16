"""关系检验的模型建议在组合根的接线：先验 → 权重（语义树新方案 ``13`` ②）。

- **先验**：混合问法（实时化阶段 2）——一对（前因, 后果）只归一题：后果比前因晚出现的归"这个后果"那一题（旧问法，
  固定后果、列前因），否则归"这个前因"那一题（新问法，固定前因、列后果）；每题三遍乱序取中位，两份缓存分开存
  （``runtime.scene_priors``）。一个新概念出现只多两题。模型队列对着"概念集 + 同步词表将要写的概念"问
  （``ask_priors``），答案直接写缓存；换代折叠时只读（``weights``）。
  权重只给不带条件的关系（短跨度与长跨度都给，按前因、后果这一对）；情境概念当前因的、带条件的按 1。
- ``cache_only``：只读缓存，缓存里没有的不问模型、按没有先验处理，并报出来。换代折叠关系时用的就是这一份
  （``cached``）：补问归模型队列，换代里没有模型调用。
- ``priors_owed``：这条 lane 还欠几件事（要问的先验题、孤儿先验要清）——模型队列据它从盘上认欠账，取法与补问时同一套。
- 原来还有"候选调节条件"（问模型提条件、提议簿）：加不进机械列举之外的任何东西，2026-11-15 删了；盘上的旧提议簿读先验之前
  明确报错（``AdviceStore.refuse_proposal_books``）。
"""

from __future__ import annotations

from collections.abc import Container
from datetime import tzinfo

from habitus.runtime.scene_priors import LanePriors
from habitus.scene.advice.prior import PriorAdvisor
from habitus.scene.advice.store import AdviceStore
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.relations import LaneTests, RelationKey


class RelationAdvice:
    def __init__(
        self,
        *,
        store: AdviceStore,
        prior: PriorAdvisor | None,
        cache_only: bool = False,
        zone: tzinfo | None = None,
    ) -> None:
        self.store = store
        self.prior = prior
        self.cache_only = cache_only
        #: 概念"哪一天出现"按这个时区的当地日期算（先验归题用）；没给按 UTC。
        self.zone = zone

    def cached(self) -> RelationAdvice:
        """只读缓存的那一份（同一个存储、同样的触点版本）：不问模型、不清孤儿先验。"""

        return RelationAdvice(store=self.store, prior=self.prior, cache_only=True, zone=self.zone)

    def priors(self, concepts: ConceptSet, lane: str, settled: Container[str] | None = None) -> LanePriors:
        """这条 lane 的先验（两份缓存 + 归题）。``settled``：``concepts`` 里已经进了概念集的身份（不给就是全部）。"""

        assert self.prior is not None
        self.store.refuse_proposal_books()
        return LanePriors(self.store, self.prior, concepts, lane, settled=settled, zone=self.zone)

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
        """补问这条 lane 的先验（模型队列调，``concepts`` 是"概念集 + 同步词表将要写的概念"）、清孤儿缓存。
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


__all__ = ["RelationAdvice"]
