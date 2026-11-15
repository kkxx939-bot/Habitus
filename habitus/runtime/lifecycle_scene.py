"""语义树的生命周期参与者（26 号清单 6.14）。语义树自己的旧东西在写的那一刻就收住了：关系表只留最晚一晚 + 备用几代
（``RelationStore``，5.8）、迁移日志按份数轮转、冻住与休眠的行随折叠清掉（5.6、5.7）；命中是事实、按封好的日子留着（关系
检验每代全量重算要全部历史）。这里每轮只报几个量。
"""

from __future__ import annotations

import asyncio

from habitus.infrastructure.lifecycle import CapacityReading, LifecycleContext, ParticipantReport
from habitus.scene.relations.store import RelationStore


class SceneParticipant:
    name = "scene"

    def __init__(self, relations: RelationStore) -> None:
        self.relations = relations

    async def maintain(self, ctx: LifecycleContext) -> ParticipantReport:
        return await asyncio.to_thread(self._maintain_blocking, ctx)

    def _maintain_blocking(self, ctx: LifecycleContext) -> ParticipantReport:
        """关系表写一晚就收一次（``RelationStore._prune``），留满保留代数是常态、不是快满了：报的是超出保留代数几晚，上限 1——
        常态 0；``_prune`` 没删掉、多出一晚就到 1/1，告警线（只能在 (0, 1) 里配）怎么配都报得出来。"""

        ctx.assert_owned()
        kept = 1 + self.relations.fallback_generations
        readings = tuple(
            CapacityReading(
                f"relation_nights_{lane}_over_retention", max(0, len(self.relations.nights(lane)) - kept), 1
            )
            for lane in self.relations.lanes()
        )
        return ParticipantReport(readings=readings)


__all__ = ["SceneParticipant"]
