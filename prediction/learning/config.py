"""Sample → Pattern 学习聚合的超参配置。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class PredictionLearningError(ValueError):
    """学习聚合的输入、配置或样本内容违反学习合同。"""


@dataclass(frozen=True)
class PredictionLearningConfig:
    """聚合估计的准确率超参；全部进入 generation 身份材料。

    ``decay_tau_days`` 控制旧样本的指数遗忘速度；``group_weight_cap`` 限制同一
    occurrence_group 对单个 (State, 行为) 的总权重，消除前缀展开与 Event/Episode
    双重投影造成的重复计数；``discount`` 与 ``strength`` 是 Pitman–Yor 插值的
    折扣与强度，共同决定未见行为的留白概率和上下文状态向根状态收缩的力度；
    ``min_context_support`` 是上下文状态物化的最小原始样本数；
    ``confidence_prior`` 是置信度收缩的等效先验样本量；
    ``max_sample_refs`` 限制单个 Pattern 携带的证据引用数量。

    TODO(PRED-TUNING-001): 当前全部默认值是保守拍定的初值，没有数据依据。
    behavior 数据链路打通、真实样本可用后，必须用 ``entropy_report`` 与
    ``backtest``（log-loss / escape_rate / ECE / 覆盖率–精确率曲线）逐项定档
    替换，任何调整不涨指标不合入。
    """

    decay_tau_days: float = 90.0
    group_weight_cap: float = 3.0
    discount: float = 0.5
    strength: float = 1.0
    min_context_support: int = 2
    confidence_prior: float = 5.0
    max_sample_refs: int = 64
    temporal_bucket_hours: int = 3
    temporal_kernel_concentration: float = 8.0
    temporal_utc_offset_minutes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "decay_tau_days", _positive_finite(self.decay_tau_days, "decay_tau_days"))
        object.__setattr__(self, "group_weight_cap", _positive_finite(self.group_weight_cap, "group_weight_cap"))
        discount = _finite(self.discount, "discount")
        if not 0 <= discount < 1:
            raise PredictionLearningError("discount must be at least zero and below one")
        object.__setattr__(self, "discount", discount)
        object.__setattr__(self, "strength", _positive_finite(self.strength, "strength"))
        object.__setattr__(self, "min_context_support", _positive_int(self.min_context_support, "min_context_support"))
        object.__setattr__(self, "confidence_prior", _positive_finite(self.confidence_prior, "confidence_prior"))
        object.__setattr__(self, "max_sample_refs", _positive_int(self.max_sample_refs, "max_sample_refs"))
        hours = _positive_int(self.temporal_bucket_hours, "temporal_bucket_hours")
        if hours >= 24 or 24 % hours != 0:
            raise PredictionLearningError("temporal_bucket_hours must divide a 24-hour day")
        object.__setattr__(self, "temporal_bucket_hours", hours)
        object.__setattr__(
            self,
            "temporal_kernel_concentration",
            _positive_finite(self.temporal_kernel_concentration, "temporal_kernel_concentration"),
        )
        offset = self.temporal_utc_offset_minutes
        if isinstance(offset, bool) or not isinstance(offset, int) or abs(offset) > 14 * 60:
            raise PredictionLearningError(
                "temporal_utc_offset_minutes must be an integer within fourteen hours"
            )

    def identity_material(self) -> dict[str, Any]:
        """返回参与 generation 身份摘要的完整超参快照。"""

        return {
            "decay_tau_days": self.decay_tau_days,
            "group_weight_cap": self.group_weight_cap,
            "discount": self.discount,
            "strength": self.strength,
            "min_context_support": self.min_context_support,
            "confidence_prior": self.confidence_prior,
            "max_sample_refs": self.max_sample_refs,
            "temporal_bucket_hours": self.temporal_bucket_hours,
            "temporal_kernel_concentration": self.temporal_kernel_concentration,
            "temporal_utc_offset_minutes": self.temporal_utc_offset_minutes,
        }


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictionLearningError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise PredictionLearningError(f"{label} must be finite")
    return normalized


def _positive_finite(value: object, label: str) -> float:
    normalized = _finite(value, label)
    if normalized <= 0:
        raise PredictionLearningError(f"{label} must be positive")
    return normalized


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PredictionLearningError(f"{label} must be a positive integer")
    return value


__all__ = ["PredictionLearningConfig", "PredictionLearningError"]
