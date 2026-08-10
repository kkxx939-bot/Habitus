"""分支概率的等渗校准映射。

Pattern 分支的 ``conditional_probability`` 是训练分布上的估计，执行判决
需要的是"此刻这个预测为真的概率"。等渗回归（PAV，pool adjacent
violators）从回测的 (预测概率, 是否命中) 对拟合一个单调非降的校准映射；
映射本身是版本化的确定性值对象，进入 predictor 身份，同输入必同输出。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from foundation.integrity import canonical_digest
from prediction.learning.config import PredictionLearningError

_CALIBRATION_CONTRACT = "prediction-probability-calibration-v1"


@dataclass(frozen=True)
class PredictionProbabilityCalibration:
    """单调分段线性的概率校准映射；空映射表示恒等（未校准）。"""

    points: tuple[tuple[float, float], ...] = ()
    sample_count: int = 0

    def __post_init__(self) -> None:
        points = tuple((_unit(x, "calibration x"), _unit(y, "calibration y")) for x, y in self.points)
        for (left_x, left_y), (right_x, right_y) in zip(points, points[1:], strict=False):
            if right_x <= left_x:
                raise PredictionLearningError("calibration points must have strictly increasing inputs")
            if right_y < left_y:
                raise PredictionLearningError("calibration points must be monotonically non-decreasing")
        object.__setattr__(self, "points", points)
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 0
        ):
            raise PredictionLearningError("calibration sample_count must be a non-negative integer")

    @property
    def version(self) -> str:
        """校准映射内容的稳定版本摘要，进入 predictor 身份。"""

        return canonical_digest(
            {
                "contract": _CALIBRATION_CONTRACT,
                "points": [list(point) for point in self.points],
                "sample_count": self.sample_count,
            }
        )

    def apply(self, probability: float) -> float:
        """把一个原始概率映射为校准概率；端点外取端点值，空映射恒等。"""

        value = _unit(probability, "probability")
        if not self.points:
            return value
        if value <= self.points[0][0]:
            return self.points[0][1]
        if value >= self.points[-1][0]:
            return self.points[-1][1]
        for (left_x, left_y), (right_x, right_y) in zip(self.points, self.points[1:], strict=False):
            if left_x <= value <= right_x:
                share = (value - left_x) / (right_x - left_x)
                return left_y + share * (right_y - left_y)
        return value

    @classmethod
    def from_outcomes(
        cls,
        outcomes: Sequence[tuple[float, bool]],
    ) -> PredictionProbabilityCalibration:
        """用 PAV 从 (预测概率, 是否命中) 对拟合单调校准映射。

        相邻违反单调性的块被合并为加权均值，结果是经验准确率对预测概率的
        最优单调拟合；同概率的块合并为一个支点。空输入返回恒等映射。
        """

        pairs = sorted(
            (
                (_unit(probability, "outcome probability"), 1.0 if hit else 0.0)
                for probability, hit in outcomes
            ),
            key=lambda item: item[0],
        )
        if not pairs:
            return cls()
        blocks: list[list[float]] = []
        for probability, hit in pairs:
            blocks.append([probability, hit, 1.0])
            while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] > blocks[-1][1] / blocks[-1][2]:
                last = blocks.pop()
                blocks[-1][0] += last[0]
                blocks[-1][1] += last[1]
                blocks[-1][2] += last[2]
        merged: dict[float, list[float]] = {}
        for total_probability, total_hit, weight in blocks:
            x = total_probability / weight
            bucket = merged.setdefault(x, [0.0, 0.0])
            bucket[0] += total_hit
            bucket[1] += weight
        points = tuple(
            (x, merged[x][0] / merged[x][1]) for x in sorted(merged)
        )
        return cls(points=points, sample_count=len(pairs))


def _unit(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictionLearningError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0 <= normalized <= 1:
        raise PredictionLearningError(f"{label} must be between zero and one")
    return normalized


__all__ = ["PredictionProbabilityCalibration"]
