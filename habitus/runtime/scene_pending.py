"""待生效的概念（实时化阶段 2「生效时点」）：写概念的结果不直接进概念集，下一次换代并入。

- ``prospect``：下一次换代之后概念集会是什么样（概念集 + 同步词表将要写的 + 待生效的概念）。写概念与补问先验对着它做——
  概念进概念集的那一代，先验已经备好。纯读盘、纯计算。
- ``author``：写概念，写成的存成待生效。
- ``merge_pending``：换代的语义树一步在同步词表之后把它们并进概念集，这时才定创建时刻。

规则（哪些并得进去）在领域包 ``scene.concepts.sync.prospective``；这里取材料、调用、落盘。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from habitus.scene.concepts.model import ConceptSet
from habitus.scene.concepts.sync import prospective
from habitus.scene.recurrence import coverage_by_lane
from habitus.series import EventSeries
from habitus.temporal import Coverage

if TYPE_CHECKING:
    from habitus.runtime.scene_night import SceneNightlyRun


async def prospect(scene: SceneNightlyRun, series: EventSeries, now: datetime) -> tuple[ConceptSet, frozenset[str]]:
    """下一次换代之后概念集会是什么样（``scene.concepts.sync.prospective``：概念集 + 同步词表将要写的 + 待生效的概念），
    以及其中现在已经在概念集里的那些身份。写概念与补问先验对着它做。纯读盘、纯计算。"""

    plan = await scene.syncing.plan(series, now)
    settled = await asyncio.to_thread(scene.concepts.read_all)
    pending = await asyncio.to_thread(scene.concepts.pending)
    found, _refused = prospective(settled, plan.write, pending)
    return found, frozenset(settled)


async def author(
    scene: SceneNightlyRun, series: EventSeries, signals: list[str], coverage: Mapping[str, Coverage] | None = None
) -> int:
    """B2 写概念：该问的 lane 各问一遍，写成的存成待生效（下一次换代并入），返回写了几个。没接概念作者时不写。材料：
    这一份序列与据它算的节律；"已有的概念"是下一次换代之后会有的样子（``prospect``）。"""

    if scene.authoring is None:
        return 0
    plan = await scene.syncing.plan(series, datetime.now(UTC))

    def existing() -> ConceptSet:
        found, _refused = prospective(scene.concepts.read_all(), plan.write, scene.concepts.pending())
        return found

    concepts = await asyncio.to_thread(existing)
    if coverage is None:
        coverage = await asyncio.to_thread(coverage_by_lane, series, scene.rulers)
    rhythms = await asyncio.to_thread(scene._rhythms, series, concepts, coverage)
    return await scene.authoring.run(
        scene.concepts, scene.catalog.classes(), series, rhythms, signals, existing=existing
    )


def merge_pending(scene: SceneNightlyRun, now: datetime, signals: list[str]) -> int:
    """把待生效的概念并进概念集（换代的语义树一步，在同步词表之后）：这时才定创建时刻。与概念集撞了身份、或并进去
    不自洽的那份不并、撤掉、报出来。返回并入了几个。纯写盘，不调模型。"""

    pending = scene.concepts.pending()
    if not pending:
        return 0
    _merged, refused = prospective(scene.concepts.read_all(), (), pending)
    rejected = {item.identity: reason for item, reason in refused}
    merged = 0
    for definition in pending:
        if definition.identity in rejected:
            signals.append(
                f"concepts: 待生效的「{definition.label}」并不进概念集（{rejected[definition.identity]}），撤掉"
            )
        else:
            scene.concepts.write(replace(definition, created_at=now))
            merged += 1
        scene.concepts.discard_pending(definition.identity)
    return merged


__all__ = ["author", "merge_pending", "prospect"]
