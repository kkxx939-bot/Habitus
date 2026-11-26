"""模型队列里先验的补问（``runtime.model_queue`` 的 advice 一类）。

今天这一代完成之后才做，一条 lane 一项：先验对着"概念集 + 同步词表将要写的概念"问（``_prospect``），答案直接写缓存——
只有换代读它，新类的行为概念进概念集的那一代先验已经备好。一对归哪一题只看两个概念出现的先后（``scene.advice.pairing``）。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import date

from habitus.foundation.integrity import canonical_digest
from habitus.runtime.model_queue_state import ADVICE, DONE, UNFINISHED, Item, ModelQueueState, Round
from habitus.runtime.scene_advice import RelationAdvice
from habitus.runtime.scene_night import SceneNightlyRun
from habitus.scene.advice.pairing import Questions
from habitus.scene.concepts.model import ConceptSet
from habitus.series import EventSeries


class AdviceWork:
    def __init__(self, *, state: ModelQueueState, scene: SceneNightlyRun, vocabulary: Callable[[], int]) -> None:
        self.state = state
        self.scene = scene
        self.vocabulary = vocabulary
        #: 先验"没有要补问的"时的输入摘要（只在进程里）：输入没变就不重算。
        self._priors_clear: str | None = None

    async def items(self, current: Round, night: date, series: EventSeries) -> list[Item]:
        """先验的补问，一条 lane 一项：对着"概念集 + 同步词表将要写的概念"问（新类的行为概念进概念集的那一代先验已经备好）。"""

        if self.scene.advice is None:
            self.state.record(ADVICE, 0, current.now)
            return []
        priors = await self._priors(current, night, series)
        self.state.record(ADVICE, len(priors), current.now)
        return priors

    async def _priors(self, current: Round, night: date, series: EventSeries) -> list[Item]:
        advice = self.scene.advice
        assert advice is not None
        prospect, settled = await self._prospect(series, current)
        signature = canonical_digest(
            {
                "concepts": sorted(prospect[name].fingerprint for name in prospect),
                "settled": sorted(settled),
                "advisor": None if advice.prior is None else [advice.prior.version, advice.prior.version_by_antecedent],
            }
        )
        if signature == self._priors_clear:
            return []
        lanes = sorted({lane for name in prospect.behaviors() if (lane := prospect.lane_of(name)) is not None})
        owed: dict[str, int] = {}
        asked = {"by_consequent": 0, "by_antecedent": 0}
        for lane in lanes:
            count = await asyncio.to_thread(advice.priors_owed, prospect, lane, settled)
            if count:
                owed[lane] = count
                questions = await asyncio.to_thread(_questions, advice, prospect, lane, settled)
                asked["by_consequent"] += len(questions.by_consequent)
                asked["by_antecedent"] += len(questions.by_antecedent)
        self.state.pending.update({f"priors_{key}": value for key, value in asked.items()})
        self._priors_clear = None if owed else signature

        def item(lane: str) -> Item:
            async def run(detail: dict[str, object], signals: list[str]) -> str:
                await advice.ask_priors(prospect, lane, signals, settled)
                left = await asyncio.to_thread(advice.priors_owed, prospect, lane, settled)
                detail.update({"kind": "priors", "questions": owed[lane], "questions_left": left})
                return UNFINISHED if left else DONE

            return Item(ADVICE, f"priors:{lane}:{night.isoformat()}", run, night)

        return [item(lane) for lane in owed]

    async def _prospect(self, series: EventSeries, current: Round) -> tuple[ConceptSet, frozenset[str]]:
        """下一次换代之后概念集会是什么样（现在的概念集 + 同步词表将要写的：新类的行为概念、改名改判据、停用、新的日历情境），
        以及其中现在已经在概念集里的那些身份。概念按身份排好，与整支读出来的先后相同。纯读盘、纯计算。"""

        plan = await self.scene.syncing.plan(series, current.now)
        settled = await asyncio.to_thread(self.scene.concepts.read_all)
        merged = dict(settled.items())
        merged.update((item.identity, item) for item in plan.write)
        return ConceptSet(merged[identity] for identity in sorted(merged)), frozenset(settled)


def _questions(advice: RelationAdvice, concepts: ConceptSet, lane: str, settled: frozenset[str]) -> Questions:
    return advice.priors(concepts, lane, settled).questions()


__all__ = ["AdviceWork"]
