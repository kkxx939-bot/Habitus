"""语义树的生命周期参与者（26 号清单 6.14）。语义树自己的旧东西在写的那一刻就收住了：关系表只留最晚一晚 + 备用几代
（``RelationStore``，5.8）、迁移日志按份数轮转、冻住与休眠的行随折叠清掉（5.6、5.7）；命中是事实、按做完的日子留着（常态要
全部历史）。这里每轮只报几个量，并回答"语义树还要读词表的哪一版之后的变更"（词表变更日志截断的下限，``lifecycle_behavior``）。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from habitus.infrastructure.lifecycle import CapacityReading, LifecycleContext, ParticipantReport
from habitus.scene.concepts.store import ConceptStore
from habitus.scene.relations.store import RelationStore


def scene_vocabulary_versions(relations: RelationStore, concepts: ConceptStore) -> Callable[[], tuple[int, ...]]:
    """语义树还要读 ``changes_since(版本)`` 的那些版本：概念集的同步进度，与各 lane 留着的每一晚关系状态记的词表版本
    （最晚那一晚读不了时要接在更早一晚之后折叠，那一晚记的版本也还要读）。读不了的那一晚跳过：折叠本来就不接它，
    让它卡住会让行为侧参与者整轮失败、物理一支的过期也跟着停（L10.7）。"""

    def versions() -> tuple[int, ...]:
        found = [concepts.synced_version()]
        for lane in relations.lanes():
            for night in relations.nights(lane):
                version = relations.vocabulary_of(lane, night)
                if version is not None:
                    found.append(version)
        return tuple(found)

    return versions


class SceneParticipant:
    name = "scene"

    def __init__(self, relations: RelationStore) -> None:
        if not isinstance(relations, RelationStore):
            raise TypeError("relations must be a RelationStore")
        self.relations = relations

    async def maintain(self, ctx: LifecycleContext) -> ParticipantReport:
        return await asyncio.to_thread(self._maintain_blocking, ctx)

    def _maintain_blocking(self, ctx: LifecycleContext) -> ParticipantReport:
        ctx.assert_owned()
        readings = tuple(
            CapacityReading(
                f"relation_nights_{lane}", len(self.relations.nights(lane)), 1 + self.relations.fallback_generations
            )
            for lane in self.relations.lanes()
        )
        return ParticipantReport(readings=readings)


__all__ = ["SceneParticipant", "scene_vocabulary_versions"]
