"""一支的下游管线（裁定 42：两条 lane 各一整套，不共用任何东西；丙二 2026-11-16）：这一支的行为部件、语义树、换代的那一眼、
模型队列、预测层，与预测 worker 每槽替它做的那一份（预测拍、预测根租约）。

代码一套，按支各装一份、各用各的根（``<scene_root>/<lane>``、``<foresight_root>/<lane>``）。两支只共用两个常驻 worker
（每一槽、每一轮依次做两支，会话在前）、当地日历与提醒出口（出口只有宿主里的会话一个，账按支记）。
"""

from __future__ import annotations

from dataclasses import dataclass

from habitus.runtime.behavior import BehaviorRuntimeComponents
from habitus.runtime.foresight import ForesightRuntimeComponents
from habitus.runtime.foresight_tick import ForesightTick
from habitus.runtime.generation_lease import Lease
from habitus.runtime.generation_runner import GenerationRunner
from habitus.runtime.model_queue import ModelQueue
from habitus.runtime.scene_night import SceneNightlyRun


@dataclass(frozen=True)
class ScenePipeline:
    behavior: BehaviorRuntimeComponents
    scene: SceneNightlyRun
    #: 这一支换代的那一眼（预测 worker 每槽在这一支的预测根租约之内先调它）。
    generation: GenerationRunner
    queue: ModelQueue
    #: 这一支的预测层；没开预测层是 None。
    foresight: ForesightRuntimeComponents | None
    #: 这一支的预测拍（没开预测层是 None）。
    tick: ForesightTick | None
    #: 这一支的预测根租约（有预测拍时在预测拍手里，没有时那一槽自己持）。
    lease: Lease

    def __post_init__(self) -> None:
        """实例同一性：这一支的换代、队列、预测层读的都是**这一支**的行为树、定稿日、完成标记与写锁。接错了进程照常、健康全绿，
        一处都不报错。"""

        generation = self.generation.generation
        tree = self.behavior.tree
        if generation.behavior_tree is not tree or self.scene.behavior_tree is not tree:
            raise ValueError(f"the {self.lane} generation must read the {self.lane} behaviour tree")
        if generation.scene is not self.scene or self.queue.scene is not self.scene:
            raise ValueError(f"the {self.lane} generation and model queue must work on the {self.lane} scene")
        if self.queue.lane != self.lane:
            raise ValueError(f"the {self.lane} branch must drive its own model queue")
        # 换代问"模型队列是不是在整体退避"读的必须是这一支队列自己的状态（裁定 96：是就不判关系、沿用上一代）
        if generation.models_down != self.queue.state.down:
            raise ValueError("the generation must read its own model queue's backoff state")
        # 语义树写锁两边必须是同一把：各持各的话，换代读语义树时队列照写
        if self.queue.scene_lock is not generation.scene_lock:
            raise ValueError("the model queue and the generation must share one scene write lock")
        if self.tick is not None and (self.foresight is None or self.tick.lane != self.lane):
            raise ValueError(f"the {self.lane} foresight tick needs the {self.lane} foresight components")
        foresight = self.foresight
        if foresight is None:
            return
        if foresight.lane != self.lane:
            raise ValueError(f"the {self.lane} branch must carry its own foresight components")
        foresight.assert_attached_to(behavior=self.behavior)
        # 接线漏了的话，进程照常启动、health 全绿、结算一代都不会跑。结算是换代里的一步
        if generation.settlement is not foresight.settlement:
            raise ValueError("the settlement stage must be a step of the generation")
        # 机会表那一步漏接：白天永远停在"还没有机会表"（2026-10-18 四方评审）
        if generation.foresight is None or generation.foresight.store is not foresight.opportunities:
            raise ValueError("the generation must write the opportunity table the foresight layer reads")
        # 结算认"那天有没有这一代"读的必须是这一支换代写的那一份完成标记
        if foresight.settlement.finished != generation.marks.finished:
            raise ValueError("the settlement must read the generation's own completion marks")

    @property
    def lane(self) -> str:
        return self.behavior.lane.value


__all__ = ["ScenePipeline"]
