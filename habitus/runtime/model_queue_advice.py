"""模型队列里候选条件与先验的补问（``runtime.model_queue`` 的 advice 一类；实时化阶段 2「生效时点」「先验混合问法」）。

今天这一代完成之后才做，一条 lane 各一项，先验在前：

- **先验**对着"概念集 + 同步词表将要生成的基础概念 + 待生效的概念"问（``SceneNightlyRun.prospect``），答案直接写缓存——只有
  换代读它，概念进概念集的那一代先验已经备好。一对归哪一题只看两个概念出现的先后（``scene.advice.pairing``）。
- **候选条件**对着今天这一代的概念集与时间线问，提出日记成下一次换代的日子，直接写提议簿（只有换代读，写进去就等于待生效）。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import date, timedelta

from habitus.foundation.integrity import canonical_digest
from habitus.runtime.model_queue_state import ADVICE, DONE, UNFINISHED, Item, ModelQueueState, Round
from habitus.runtime.scene_advice import RelationAdvice
from habitus.runtime.scene_night import SceneNightlyRun
from habitus.scene.advice.pairing import Questions
from habitus.scene.concepts.model import ConceptSet
from habitus.series import EventSeries


class AdviceWork:
    def __init__(
        self, *, state: ModelQueueState, scene: SceneNightlyRun, vocabulary: Callable[[], tuple[int, bool]]
    ) -> None:
        self.state = state
        self.scene = scene
        self.vocabulary = vocabulary
        #: 候选条件 / 先验"没有要补问的"时的输入摘要（只在进程里）：输入没变就不重算。
        self._advice_clear: str | None = None
        self._priors_clear: str | None = None

    async def items(self, current: Round, night: date, series: EventSeries, concepts: ConceptSet) -> list[Item]:
        """先验、候选条件的补问，一条 lane 各一项，先验在前。先验对着"概念集 + 同步词表将要生成的基础概念 + 待生效的概念"问
        （概念进概念集的那一代先验已经备好）；候选条件对着今天这一代的概念集与时间线问，提出日记成下一次换代的日子。"""

        scene = self.scene
        if scene.advice is None:
            self.state.record(ADVICE, 0, current.now)
            return []
        priors = await self._priors(current, night, series)
        conditions = await self._conditions(night, series, concepts)
        self.state.record(ADVICE, len(priors) + len(conditions), current.now)
        return [*priors, *conditions]

    async def _priors(self, current: Round, night: date, series: EventSeries) -> list[Item]:
        advice = self.scene.advice
        assert advice is not None
        prospect, settled = await self.scene.prospect(series, current.now)
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

    async def _conditions(self, night: date, series: EventSeries, concepts: ConceptSet) -> list[Item]:
        scene = self.scene
        stage = scene.relation_stage
        signature = await asyncio.to_thread(self._advice_signature, night, series, concepts)
        if signature == self._advice_clear:
            return []
        lines = await stage.timelines(series, concepts)
        owed = {
            lane: count
            for lane, line in sorted(lines.items())
            if (count := await asyncio.to_thread(stage.owed, lane, line, series, concepts))
        }
        self._advice_clear = None if owed else signature

        def item(lane: str) -> Item:
            async def run(detail: dict[str, object], signals: list[str]) -> str:
                line = lines[lane]
                # 提出日 = 下一次换代的日子：今天这一代已经折叠完了，提议簿只有换代读，写进去就等于待生效
                await stage.ask_conditions(lane, line, series, concepts, signals, proposed_on=night + timedelta(days=1))
                left = await asyncio.to_thread(stage.owed, lane, line, series, concepts)
                detail.update({"kind": "conditions", "questions": owed[lane], "questions_left": left})
                return UNFINISHED if left else DONE

            return Item(ADVICE, f"conditions:{lane}:{night.isoformat()}", run, night)

        return [item(lane) for lane in owed]

    def _advice_signature(self, night: date, series: EventSeries, concepts: ConceptSet) -> str:
        """候选条件的补问取决于的输入的摘要：这一晚、序列、概念集、盘上盖了章的命中、各 lane 最近一晚的关系表。"""

        hits = self.scene.hits
        relations = self.scene.relations
        return canonical_digest(
            {
                "night": night.isoformat(),
                "records": [len(series.records), len(series.gaps)],
                "vocabulary": self.vocabulary()[0],
                "concepts": sorted(concepts[name].fingerprint for name in concepts),
                "hits": [
                    [day.isoformat(), marker.mapper, marker.records, marker.completed_at.isoformat()]
                    for day in sorted(hits.days_done())
                    if (marker := hits.read_marker(day)) is not None
                ],
                "relations": {
                    lane: [item.isoformat() for item in relations.nights(lane)[-1:]] for lane in relations.lanes()
                },
            }
        )


def _questions(advice: RelationAdvice, concepts: ConceptSet, lane: str, settled: frozenset[str]) -> Questions:
    return advice.priors(concepts, lane, settled).questions()


__all__ = ["AdviceWork"]
