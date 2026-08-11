"""预测判决链的门槛与调整配置。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class PredictionPredictorError(ValueError):
    """预测判决的输入、配置或图状态违反合同。"""


@dataclass(frozen=True)
class PredictionDecisionConfig:
    """执行判决的代价门槛与顾问调整强度。

    执行错误行为的代价远高于漏预测，因此门槛全部偏保守：top-1 校准概率、
    对 top-2 的边际、状态经验数和负向结果风险必须同时通过才允许执行。
    ``advisor_boost`` 控制在线顾问权重在候选之间重分配概率质量的最大
    幅度，权重只搬移质量，不创造质量。

    TODO(PRED-TUNING-001): 当前全部默认门槛与增益是保守初值。真实样本
    可用后，执行门槛按各 domain 的代价比 p* = C_wrong/(C_wrong+U_correct)
    结合回测的覆盖率–精确率曲线定档（每个 domain 一个配置实例），
    advisor_boost 与建议档门槛同样以回测增量定值。
    """

    min_coverage_score: float = 0.5
    execute_probability_threshold: float = 0.6
    execute_margin: float = 0.15
    min_execute_support: int = 3
    max_negative_outcome_probability: float = 0.2
    advisor_boost: float = 0.5
    max_candidates: int = 3
    temporal_bucket_hours: int = 3
    temporal_utc_offset_minutes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_coverage_score", _unit(self.min_coverage_score, "min_coverage_score"))
        object.__setattr__(
            self,
            "execute_probability_threshold",
            _unit(self.execute_probability_threshold, "execute_probability_threshold"),
        )
        object.__setattr__(self, "execute_margin", _unit(self.execute_margin, "execute_margin"))
        object.__setattr__(
            self,
            "min_execute_support",
            _positive_int(self.min_execute_support, "min_execute_support"),
        )
        object.__setattr__(
            self,
            "max_negative_outcome_probability",
            _unit(self.max_negative_outcome_probability, "max_negative_outcome_probability"),
        )
        advisor_boost = _finite(self.advisor_boost, "advisor_boost")
        if advisor_boost < 0:
            raise PredictionPredictorError("advisor_boost must be at least zero")
        object.__setattr__(self, "advisor_boost", advisor_boost)
        object.__setattr__(self, "max_candidates", _positive_int(self.max_candidates, "max_candidates"))
        hours = _positive_int(self.temporal_bucket_hours, "temporal_bucket_hours")
        if hours >= 24 or 24 % hours != 0:
            raise PredictionPredictorError("temporal_bucket_hours must divide a 24-hour day")
        object.__setattr__(self, "temporal_bucket_hours", hours)
        offset = self.temporal_utc_offset_minutes
        if isinstance(offset, bool) or not isinstance(offset, int) or abs(offset) > 14 * 60:
            raise PredictionPredictorError(
                "temporal_utc_offset_minutes must be an integer within fourteen hours"
            )

    def identity_material(self) -> dict[str, Any]:
        """返回参与 predictor 身份摘要的完整配置快照。"""

        return {
            "min_coverage_score": self.min_coverage_score,
            "execute_probability_threshold": self.execute_probability_threshold,
            "execute_margin": self.execute_margin,
            "min_execute_support": self.min_execute_support,
            "max_negative_outcome_probability": self.max_negative_outcome_probability,
            "advisor_boost": self.advisor_boost,
            "max_candidates": self.max_candidates,
            "temporal_bucket_hours": self.temporal_bucket_hours,
            "temporal_utc_offset_minutes": self.temporal_utc_offset_minutes,
        }


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictionPredictorError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise PredictionPredictorError(f"{label} must be finite")
    return normalized


def _unit(value: object, label: str) -> float:
    normalized = _finite(value, label)
    if not 0 <= normalized <= 1:
        raise PredictionPredictorError(f"{label} must be between zero and one")
    return normalized


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PredictionPredictorError(f"{label} must be a positive integer")
    return value


__all__ = ["PredictionDecisionConfig", "PredictionPredictorError"]
