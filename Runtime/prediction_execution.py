"""预测之后的三档行动决策：沉默 / 主动建议 / 直接执行（组合根侧）。

确认与风险逻辑在预测算法后面：预测器只产出候选与门槛评估，本层把
结果映射为产品动作。默认档是**主动建议**——向用户提议并等确认，
用户本人就是建议的裁决者；只有经长期观察被验证稳定的行为规律
（``qualifier`` 判定）才有资格进入**直接执行**档，且执行前必须通过
LLM 约束检查（fail-closed：查不了就降级为建议，而不是硬执行）。
合规必须被建立而非未被否定：从未被送审的候选不允许直接执行。
没有任何人工维护的行为风险表——直接执行资格靠数据挣来。

TODO(PRED-RUNTIME-002): 晋升判据依赖建议确认史（用户对建议的确认/
拒绝记录）；该存储与回流链路待 prediction 接入 Runtime 主链时设计。
落地前 ``qualifier`` 缺省为 None，即一切执行意图都降级为建议档。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from foundation.integrity import canonical_digest
from ModelClient import ModelClientError
from prediction import PredictionContext, PredictionRun, PredictionRunSourceBinding
from prediction.predictor import PatternGraphPredictor
from Runtime.prediction_bridge import MAX_MEMORY_CONTEXT_ENTRIES
from Runtime.prediction_llm import (
    PatternGraphLLMAdvisor,
    PredictionConstraintChecker,
    predict_with_advice,
)

_TIERS = frozenset({"silent", "suggest", "execute"})
_CONSTRAINT_STATUSES = frozenset(
    {
        "not_applicable",
        "bypassed_no_checker",
        "bypassed_no_memory_context",
        "passed",
        "violated",
        "unavailable",
    }
)


@dataclass(frozen=True)
class PredictionTierConfig:
    """建议档的门槛；执行档沿用预测器配置的四道门。

    建议线低于执行线：原本"有信号但不够确定"的沉默地带成为建议，
    产品从灰色地带持续拿到价值，而建错的代价只是用户拒绝一次——
    这个拒绝本身还是高价值的监督证据。
    """

    suggest_probability: float = 0.35
    suggest_margin: float = 0.05

    def __post_init__(self) -> None:
        for name in ("suggest_probability", "suggest_margin"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= float(value) <= 1
            ):
                raise ValueError(f"{name} must be between zero and one")
            object.__setattr__(self, name, float(value))

    def identity_material(self) -> dict[str, float]:
        return {
            "suggest_probability": self.suggest_probability,
            "suggest_margin": self.suggest_margin,
        }


@dataclass(frozen=True)
class PredictionActionPlan:
    """一次预测最终映射到的产品动作、理由与约束检查的审计绑定。

    ``constraint_status`` 如实记录执行档到达约束检查时发生了什么：
    通过、违反、不可用，或两种显式旁路（无检查器 / 无记忆上下文）。
    ``checker_digest`` 与 ``verdicts_digest`` 把三档结论钉到具体的
    检查器路由与本次判定内容上，旁路与通过在记录上可区分。
    """

    run: PredictionRun
    tier: str
    reasons: tuple[str, ...]
    constraint_status: str = "not_applicable"
    checker_digest: str | None = None
    verdicts_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run, PredictionRun):
            raise TypeError("action plan run must be PredictionRun")
        if self.tier not in _TIERS:
            raise ValueError(f"action tier must be one of {sorted(_TIERS)}")
        reasons = tuple(self.reasons)
        if any(not isinstance(item, str) or not item.strip() for item in reasons):
            raise ValueError("action plan reasons must be non-empty text")
        if self.tier == "execute" and reasons:
            raise ValueError("an execute plan cannot carry blocking reasons")
        if self.tier != "execute" and not reasons:
            raise ValueError("a non-execute plan must name its reasons")
        object.__setattr__(self, "reasons", reasons)
        if self.constraint_status not in _CONSTRAINT_STATUSES:
            raise ValueError(
                f"constraint status must be one of {sorted(_CONSTRAINT_STATUSES)}"
            )
        for name in ("checker_digest", "verdicts_digest"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"action plan {name} must be non-empty text or None")


async def resolve_action(
    predictor: PatternGraphPredictor,
    advisor: PatternGraphLLMAdvisor,
    context: PredictionContext,
    *,
    predicted_at: datetime,
    horizon_seconds: float,
    source_bindings: Sequence[PredictionRunSourceBinding],
    memory_context: Sequence[str] = (),
    memory_provenance: str | None = None,
    checker: PredictionConstraintChecker | None = None,
    qualifier: Callable[[PredictionRun], bool] | None = None,
    config: PredictionTierConfig | None = None,
) -> PredictionActionPlan:
    """完整决策链：预测（含顾问）→ 三档映射 → 执行前约束检查。

    - 四道门未全过：过建议线则建议（理由 = 被拦的门），否则沉默；
    - 四道门全过但未通过稳定性资格（``qualifier``）：降级为建议；
    - 资格通过：批量约束检查全部候选。放行条件是 top-1 **属于已审
      且 allowed 的集合**——合规必须被建立，未送审不等于合规。top-1
      违反 → 排除全部违反候选重判；重判 top-1 若未被审过，追加一次
      有界补审，仍不合规或补审失败则降级为建议；检查不可用 →
      fail-closed 降级为建议。``checker`` 为 None 表示该 domain 显式
      关闭约束检查（记录为旁路）。
    """

    resolved = config or PredictionTierConfig()
    if not isinstance(resolved, PredictionTierConfig):
        raise TypeError("config must be PredictionTierConfig")
    entries = _validated_context_entries(memory_context)
    decision = await predict_with_advice(
        predictor,
        advisor,
        context,
        predicted_at=predicted_at,
        horizon_seconds=horizon_seconds,
        source_bindings=source_bindings,
        memory_context=entries,
        memory_provenance=memory_provenance,
    )
    run = decision.run
    if not decision.execute:
        tier = "suggest" if _passes_suggest_gates(run, resolved) else "silent"
        return PredictionActionPlan(run=run, tier=tier, reasons=decision.blocked_gates)
    if qualifier is None or not qualifier(run):
        return PredictionActionPlan(run=run, tier="suggest", reasons=("not_yet_stable_routine",))
    if checker is None:
        return PredictionActionPlan(
            run=run, tier="execute", reasons=(), constraint_status="bypassed_no_checker"
        )
    if not entries:
        return PredictionActionPlan(
            run=run,
            tier="execute",
            reasons=(),
            constraint_status="bypassed_no_memory_context",
            checker_digest=checker.checker_digest,
        )
    try:
        verdicts = await checker.review(run, memory_context=entries)
    except ModelClientError:
        return PredictionActionPlan(
            run=run,
            tier="suggest",
            reasons=("constraint_check_unavailable",),
            constraint_status="unavailable",
            checker_digest=checker.checker_digest,
        )
    allowed = {key for key, verdict in verdicts.items() if verdict["allowed"]}
    blocked = set(verdicts) - allowed
    top_key = str(run.candidates[0].payload["branch_key"])
    if top_key in allowed:
        return PredictionActionPlan(
            run=run,
            tier="execute",
            reasons=(),
            constraint_status="passed",
            checker_digest=checker.checker_digest,
            verdicts_digest=_verdicts_digest(verdicts),
        )
    retry = predictor.predict(
        context,
        predicted_at=predicted_at,
        horizon_seconds=horizon_seconds,
        source_bindings=source_bindings,
        memory_provenance=memory_provenance,
        excluded_branch_keys=tuple(sorted(blocked)),
    )
    if retry.execute and retry.run.candidates and qualifier(retry.run):
        retry_top = str(retry.run.candidates[0].payload["branch_key"])
        if retry_top in allowed:
            return PredictionActionPlan(
                run=retry.run,
                tier="execute",
                reasons=(),
                constraint_status="passed",
                checker_digest=checker.checker_digest,
                verdicts_digest=_verdicts_digest(verdicts),
            )
        if retry_top not in verdicts:
            try:
                retry_verdicts = await checker.review(retry.run, memory_context=entries)
            except ModelClientError:
                return PredictionActionPlan(
                    run=retry.run,
                    tier="suggest",
                    reasons=("constraint_check_unavailable",),
                    constraint_status="unavailable",
                    checker_digest=checker.checker_digest,
                )
            merged = {**verdicts, **retry_verdicts}
            if retry_verdicts.get(retry_top, {}).get("allowed") is True:
                return PredictionActionPlan(
                    run=retry.run,
                    tier="execute",
                    reasons=(),
                    constraint_status="passed",
                    checker_digest=checker.checker_digest,
                    verdicts_digest=_verdicts_digest(merged),
                )
            verdicts = merged
            blocked = {key for key, verdict in verdicts.items() if not verdict["allowed"]}
    reasons = ("memory_constraint_violation",) + tuple(
        f"constraint:{verdicts[key]['reason']}" for key in sorted(blocked)
    )
    return PredictionActionPlan(
        run=run,
        tier="suggest",
        reasons=reasons,
        constraint_status="violated",
        checker_digest=checker.checker_digest,
        verdicts_digest=_verdicts_digest(verdicts),
    )


def _validated_context_entries(memory_context: Sequence[str]) -> tuple[str, ...]:
    """入口即校验记忆上下文，调用方契约错误在四道门之前显式暴露。"""

    if isinstance(memory_context, (str, bytes)) or not isinstance(memory_context, Sequence):
        raise TypeError("memory_context must be a sequence of text entries")
    entries: list[str] = []
    for entry in memory_context:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError("memory_context entries must be non-empty text")
        entries.append(entry.strip())
    if len(entries) > MAX_MEMORY_CONTEXT_ENTRIES:
        raise ValueError("memory_context exceeds the bounded entry count")
    return tuple(entries)


def _verdicts_digest(verdicts: Mapping[str, Mapping[str, Any]]) -> str:
    return canonical_digest(
        {
            "contract": "prediction-constraint-verdicts-v1",
            "verdicts": {key: dict(verdicts[key]) for key in sorted(verdicts)},
        }
    )


def _passes_suggest_gates(run: PredictionRun, config: PredictionTierConfig) -> bool:
    if not run.candidates:
        return False
    top = run.candidates[0].probability
    runner_up = run.candidates[1].probability if len(run.candidates) > 1 else 0.0
    return top >= config.suggest_probability and top - runner_up >= config.suggest_margin


__all__ = [
    "PredictionActionPlan",
    "PredictionTierConfig",
    "resolve_action",
]
