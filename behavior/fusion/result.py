"""一次融合的可审计结果。

单独成模块而不是留在 ``service`` 里：确定性派生与融合回执都要消费这个形状，而 ``service`` 是
本层唯一 import ``ModelClient`` 的模块。若结果类型留在那里，回执就会**传递性地**依赖模型客户端
——那正好背离"模型只出现在融合一环"的约束，也让回执在没有模型依赖的场景下不可用。
"""

from __future__ import annotations

from dataclasses import dataclass

from behavior.fusion.judgement import BehaviorJudgementBatch
from behavior.fusion.segmentation import BehaviorFusionSegment


@dataclass(frozen=True)
class BehaviorFusionResult:
    """一次融合的完整可审计结果。"""

    segment: BehaviorFusionSegment
    batch: BehaviorJudgementBatch
    prompt_version: str
    validation_attempts: int


__all__ = ["BehaviorFusionResult"]
