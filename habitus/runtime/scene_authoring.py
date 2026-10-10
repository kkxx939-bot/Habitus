"""夜批 B2：触点①写概念（细分 / 汇总 / 情境）在什么时候问、拿什么问。

**什么时候问**（裁定 23：「两种都要，类清单变了就问，再加每周一次，放在每周的周一开始」）：

- 一条 lane 的类清单（在用的类的编号、类名、判据）与上次给这条 lane 写概念时不一样了，当晚问——第一晚、词表每晚新增出新类、
  定期拆改改了判据或拆合了类；
- 每周从周一那一晚开始（哪一天可配），这一周还没写过的 lane 问一遍——词表没变，数据攒多了也可能有新的细分值得写；
  周一没答成，之后几晚接着补，直到这一周写成。

**拿什么问**：每个在用的类给类名、判据与几条**真例子**（序列里这个类的记录，按时间均匀挑，名字 + 概要）——模型只看类名会凭联想
写区别；再给已有的概念（不要重写、不要重名）与这些类平常的作息（节律）。

写成的概念直接落盘；没答成的这一回不写，类清单指纹也不记，下一晚再问。新写的细分概念之后由 B3 的回填补映射历史。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, timedelta

from habitus.foundation.integrity import canonical_digest
from habitus.scene.concepts.author import MAX_CONCEPTS_PER_BATCH, ClassBrief, ConceptAuthor
from habitus.scene.concepts.catalog import CatalogClass
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.concepts.rhythm import Rhythm
from habitus.scene.concepts.store import Authored, ConceptStore
from habitus.series import EventSeries

#: 每个类给模型看几条真例子。
EXAMPLES_PER_CLASS = 5


def lane_digests(classes: Sequence[CatalogClass]) -> dict[str, str]:
    """每条 lane 在用的类清单指纹（编号、类名、判据）。"""

    lanes: dict[str, list[list[str]]] = {}
    for item in classes:
        if item.active:
            lanes.setdefault(item.lane, []).append([item.id, item.name, item.criterion])
    return {lane: canonical_digest(sorted(rows)) for lane, rows in sorted(lanes.items())}


def briefs_for(classes: Sequence[CatalogClass], lane: str, series: EventSeries) -> tuple[ClassBrief, ...]:
    by_class: dict[str, list[str]] = {}
    for record in series.records:
        if record.classified and record.lane == lane:
            by_class.setdefault(record.kind_token, []).append(f"{record.name}：{record.summary}")
    found = []
    for item in classes:
        if not item.active or item.lane != lane:
            continue
        examples = by_class.get(item.id, [])
        step = max(1, len(examples) // EXAMPLES_PER_CLASS)
        found.append(
            ClassBrief(item.id, item.name, item.criterion, item.lane, tuple(examples[::step][:EXAMPLES_PER_CLASS]))
        )
    return tuple(found)


class ConceptAuthoring:
    def __init__(self, author: ConceptAuthor, *, weekly_on: int = 0) -> None:
        """``weekly_on``：每周从哪一天那一晚开始问（0 = 周一，与 ``date.weekday()`` 同）。"""

        if not isinstance(author, ConceptAuthor):
            raise TypeError("author must be a ConceptAuthor")
        if isinstance(weekly_on, bool) or not isinstance(weekly_on, int) or not 0 <= weekly_on <= 6:
            raise ValueError("weekly_on must be a weekday number 0–6")
        self.author = author
        self.weekly_on = weekly_on

    def week_start(self, night: date) -> date:
        """这一晚所在的"一周"从哪一晚开始：最近一个（含今晚）``weekly_on`` 那天。"""

        return night - timedelta(days=(night.weekday() - self.weekly_on) % 7)

    def due(self, lane: str, digest: str, night: date, authored: Mapping[str, Authored]) -> str | None:
        """这条 lane 今晚该不该问、为什么（不该问是 None）。"""

        mark = authored.get(lane)
        if mark is None:
            return "第一次"
        if mark.digest != digest:
            return "类清单变了"
        if mark.night < self.week_start(night):
            return "这一周还没写过"
        return None

    async def run(
        self,
        store: ConceptStore,
        classes: Sequence[CatalogClass],
        series: EventSeries,
        rhythms: Mapping[str, Rhythm],
        signals: list[str],
    ) -> ConceptSet:
        """该问的 lane 各问一遍（一批最多 ``MAX_CONCEPTS_PER_BATCH`` 个类），写成的落盘，返回最新的概念集。"""

        night = series.cutoff
        concepts = store.read_all()
        authored = store.authored()
        for lane, digest in lane_digests(classes).items():
            reason = self.due(lane, digest, night, authored)
            if reason is None:
                continue
            signals.append(f"concepts: {lane} 写概念（{reason}）")
            briefs = briefs_for(classes, lane, series)
            complete = True
            for start in range(0, len(briefs), MAX_CONCEPTS_PER_BATCH):
                proposal = await self.author.propose(briefs[start : start + MAX_CONCEPTS_PER_BATCH], concepts, rhythms)
                signals.extend(proposal.signals)
                if not proposal.answered:
                    complete = False
                    continue
                for definition in proposal.definitions:
                    store.write(definition)
                    signals.append(f"concepts: {lane} 新概念「{definition.label}」")
                concepts = store.read_all()
            if complete:
                authored[lane] = Authored(digest=digest, night=night)
                store.write_authored(authored)
        return concepts


__all__ = ["EXAMPLES_PER_CLASS", "ConceptAuthoring", "briefs_for", "lane_digests"]
