"""行为侧、语义树、预测层与生命周期骨架自己的参与者（26 号清单 6.14），从 ``build_runtime`` 搬出来按配置组装。"""

from __future__ import annotations

from habitus.config import HabitusConfig
from habitus.infrastructure.lifecycle import LifecycleParticipant, RetryLedger
from habitus.runtime.behavior import BehaviorBranches
from habitus.runtime.evidence import evidence_of
from habitus.runtime.lifecycle_behavior import BehaviorParticipant
from habitus.runtime.lifecycle_foresight import ForesightParticipant
from habitus.runtime.lifecycle_housekeeping import LifecycleHousekeepingParticipant
from habitus.runtime.lifecycle_scene import SceneParticipant
from habitus.runtime.pipeline import ScenePipeline


def downstream_participants(
    config: HabitusConfig,
    pipelines: tuple[ScenePipeline, ...],
    behavior: BehaviorBranches | None,
    *,
    source_retries: RetryLedger,
) -> tuple[LifecycleParticipant, ...]:
    """语义树、预测层一支一个；行为侧一个；最后是重试账与每晚重试的堆的统一回收（L10.10；裁定 67）。"""

    participants: list[LifecycleParticipant] = [
        SceneParticipant(item.scene.relations, lane=item.lane) for item in pipelines
    ]
    if behavior is not None:
        participants.append(BehaviorParticipant(behavior, backfill_days=config.lifecycle.physical_max_backfill_days))
    for item in pipelines:
        if item.foresight is not None:
            participants.append(
                ForesightParticipant(
                    item.foresight.ledger,
                    item.foresight.history.store,
                    lane=item.foresight.lane,
                    feedback_kept=config.lifecycle.feedback_kept,
                    window=evidence_of(config).window,
                )
            )
    branches = () if behavior is None else tuple(behavior)
    participants.append(
        LifecycleHousekeepingParticipant(
            (
                source_retries,
                *(item.queue.retries for item in pipelines),
            ),
            keep_days=config.lifecycle.retry.given_up_days,
            backlogs=tuple(branch.retry_backlog for branch in branches if branch.retry_backlog is not None),
        )
    )
    return tuple(participants)


__all__ = ["downstream_participants"]
