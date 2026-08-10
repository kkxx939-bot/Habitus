"""由组合根从长期记忆检索后注入的低权重预测提示。"""

from __future__ import annotations

from dataclasses import dataclass

from prediction.predictor.config import PredictionPredictorError

_SIGNAL_KINDS = frozenset({"intention", "preference", "profile"})
_SIGNAL_STANCES = frozenset({"support", "oppose"})


@dataclass(frozen=True)
class PredictionMemorySignal:
    """一条来自长期记忆的行为倾向提示。

    prediction 不反向依赖 memory 领域：这里只接受调用方（Runtime 组合根）
    已经检索、筛选好的语义提示，并要求携带来源 URI，供 PredictionRun 的
    来源绑定审计"这次预测参考了哪条记忆"。信号只影响候选之间的概率
    重分配，权重高低由判决配置控制，不构成执行授权。
    ``stance`` 区分支持与反对：否定式偏好（"不要/避免做某事"）必须以
    oppose 进入，绝不能因为文本命中而被当成加强。
    """

    source_uri: str
    kind: str
    semantics: str
    target_refs: tuple[str, ...] = ()
    weight: float = 1.0
    stance: str = "support"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_uri, str)
            or "://" not in self.source_uri
            or self.source_uri != self.source_uri.strip()
            or not self.source_uri.strip()
        ):
            raise PredictionPredictorError("memory signal source_uri must be an absolute URI")
        if self.kind not in _SIGNAL_KINDS:
            raise PredictionPredictorError(f"memory signal kind must be one of {sorted(_SIGNAL_KINDS)}")
        if not isinstance(self.semantics, str) or not self.semantics.strip():
            raise PredictionPredictorError("memory signal semantics must be non-empty text")
        object.__setattr__(self, "semantics", self.semantics.strip())
        refs = tuple(self.target_refs)
        if any(not isinstance(item, str) or not item.strip() for item in refs):
            raise PredictionPredictorError("memory signal target_refs must be non-empty text")
        object.__setattr__(self, "target_refs", refs)
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)):
            raise PredictionPredictorError("memory signal weight must be numeric")
        weight = float(self.weight)
        if not 0 < weight <= 1:
            raise PredictionPredictorError("memory signal weight must be within (0, 1]")
        object.__setattr__(self, "weight", weight)
        if self.stance not in _SIGNAL_STANCES:
            raise PredictionPredictorError(f"memory signal stance must be one of {sorted(_SIGNAL_STANCES)}")


__all__ = ["PredictionMemorySignal"]
