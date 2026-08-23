"""消费激活 Pattern 图并产出可审计判决的确定性预测器。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from foundation.integrity import canonical_digest, canonical_json
from prediction.context import PredictionContext
from prediction.learning.calibration import PredictionProbabilityCalibration
from prediction.learning.keys import (
    logical_state_key,
    previous_step_key,
    sequence_state_identity,
    target_kind_for_level,
    temporal_bucket,
    temporal_state_identity,
)
from prediction.learning.vocabulary import PredictionBehaviorVocabulary
from prediction.model import PredictionKind, PredictionTargetLevel
from prediction.pattern.document import PredictionPatternDocument
from prediction.pattern.generation import PredictionPatternGenerationStoreError
from prediction.pattern.graph import PredictionPatternGraph, PredictionStateExpansion
from prediction.predictor.config import PredictionDecisionConfig, PredictionPredictorError
from prediction.run import (
    PredictionAbstentionReason,
    PredictionCandidate,
    PredictionRun,
    PredictionRunPatternBinding,
    PredictionRunSourceBinding,
)
from prediction.uri import PredictionURI

_PREDICTOR_ID = "pattern-graph-predictor"
_PREDICTOR_VERSION = "v1"
_PREDICTOR_CONTRACT = "prediction-pattern-graph-predictor-v1"
_SUPPORTED_LEVELS = frozenset({"action", "event"})

# TODO(PRED-ALGO-003): 分层规律性度量（见 ``TODO(PRED-REGULARITY-001)``）落地后，预测端还缺三块。
# - 一、时间维。现在 ``PredictionCandidate.expected_delay_seconds`` 恒为 ``None``，即完全不预测
#   "什么时候"，而对提醒场景时机是一半的价值。**已不需要从头设计**：PRED-REGULARITY-001 定稿的
#   概率层是离散时间危险率 h_B(t)，时刻分布由 P(槽 t 才发生)=h(t)·Π(1−h(s)) 逐槽累乘免费得到，
#   中位数即预计时刻。本条只剩接线：把该分布接进候选的 expected_delay_seconds 与时窗输出。
#   仍要记住"时长"（做多久）与"时刻"（何时开始）之分：公开结果显示前者误差极大（约 35 分钟
#   量级）、后者可预测性高得多，输出的是后者。
# - 二、查询层。给定此刻构造各条件值、按该行为自己的解释链查对应分布、按对数线性组合，得到候选与
#   概率。这一块更接近接口而非新算法——条件分布在算度量时已经估出来了。
# - 三、判决层重定。``_blocked_gates`` 的四道门当前是拍的初值：概率、边际、支持度三道应当用
#   覆盖率-精确率曲线反推定档，不再拍；**负向结果那道门建议直接删除**，它读 Outcome 的 ``valence``，
#   而上游观测契约只接受有主体的行为观测、不送环境反馈，该字段没有生产者，这道门实际上从不触发。
# - 另需补上反馈闭环：``PredictionRun`` 目前只记录预测了什么，没有任何字段承载"后来实际发生了
#   什么、用户接受还是拒绝"。而"默认主动提醒 + 用户确认"这一产品形态本身就在持续产生高质量标签，
#   现在全部丢弃。等渗校准若改为按真实接受率拟合，校准的才是执行门真正需要的那个概率。结算还必须
#   带"此前有没有提醒"标记：被提醒后的发生不得进入自然危险率的估计（干预混淆，见
#   PRED-REGULARITY-001 概率算法一节），此标记要在提醒功能上线前就位，事后无法把两种数据分开。
# - 影响大小：大，但可以分批。二、三两块依赖 PRED-REGULARITY-001 的产物；一和反馈闭环相互独立，
#   可以先做。
# - 时机：PRED-REGULARITY-001 在真实数据上验证通过之后。


@dataclass(frozen=True)
class PredictionDecision:
    """一次预测的冻结记录加执行判决；候选本身不承担执行授权。"""

    run: PredictionRun
    execute: bool
    blocked_gates: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.run, PredictionRun):
            raise TypeError("decision run must be PredictionRun")
        if not isinstance(self.execute, bool):
            raise TypeError("decision execute flag must be a boolean")
        gates = tuple(self.blocked_gates)
        if any(not isinstance(item, str) or not item.strip() for item in gates):
            raise ValueError("decision blocked gates must be non-empty text")
        if self.execute and gates:
            raise ValueError("an executable decision cannot carry blocked gates")
        if not self.execute and not gates:
            raise ValueError("a blocked decision must name its gates")
        object.__setattr__(self, "blocked_gates", gates)


class PatternGraphPredictor:
    """基于每逻辑 State 激活代的确定性 next-step 预测器。

    状态匹配不做检索：预测器用与学习器完全相同的词表和上下文键派生
    logical_state_key，先精确命中上下文状态，缺失时回退同层根状态。
    记忆不再以预折算信号进入本层——它作为原文上下文交给组合根的两个
    LLM 位置（不确定时的顾问、执行前的约束检查）在情境里裁量；顾问
    权重只在既有候选间做质量守恒重排，不发明行为、不创造概率。执行
    判决从代价不对称出发，概率、边际、经验数和负向结果风险四道门
    全部通过才允许执行 top-1，任何弃答或拦截都带受控原因。
    """

    def __init__(
        self,
        graph: PredictionPatternGraph,
        *,
        config: PredictionDecisionConfig | None = None,
        vocabulary: PredictionBehaviorVocabulary | None = None,
        calibration: PredictionProbabilityCalibration | None = None,
        advisor_digest: str | None = None,
    ) -> None:
        if not isinstance(graph, PredictionPatternGraph):
            raise TypeError("graph must be PredictionPatternGraph")
        if config is not None and not isinstance(config, PredictionDecisionConfig):
            raise TypeError("config must be PredictionDecisionConfig")
        if vocabulary is not None and not isinstance(vocabulary, PredictionBehaviorVocabulary):
            raise TypeError("vocabulary must be PredictionBehaviorVocabulary")
        if calibration is not None and not isinstance(calibration, PredictionProbabilityCalibration):
            raise TypeError("calibration must be PredictionProbabilityCalibration")
        if advisor_digest is not None and (
            not isinstance(advisor_digest, str) or not advisor_digest.strip()
        ):
            raise TypeError("advisor_digest must be non-empty text or None")
        self.graph = graph
        self.config = config or PredictionDecisionConfig()
        self.vocabulary = vocabulary or PredictionBehaviorVocabulary()
        self.calibration = calibration
        self.advisor_digest = advisor_digest
        self.predictor_digest = canonical_digest(
            {
                "contract": _PREDICTOR_CONTRACT,
                "config": self.config.identity_material(),
                "vocabulary_version": self.vocabulary.version,
                "calibration_version": None if calibration is None else calibration.version,
                "advisor_digest": advisor_digest,
            }
        )

    def predict(
        self,
        context: PredictionContext,
        *,
        predicted_at: datetime,
        horizon_seconds: float,
        source_bindings: Sequence[PredictionRunSourceBinding],
        advisor_weights: Mapping[str, float] | None = None,
        memory_provenance: str | None = None,
        excluded_branch_keys: Sequence[str] = (),
    ) -> PredictionDecision:
        """对一个运行时上下文做一次完整预测并冻结为可审计判决。

        ``advisor_weights`` 是在线语义顾问对候选分支（按 branch_key）的
        偏好权重：只在已有分支之间做有界重分配，不发明行为；带权重的
        预测要求构造时声明 ``advisor_digest``，保证判决可审计。
        """

        if not isinstance(context, PredictionContext):
            raise PredictionPredictorError("context must be PredictionContext")
        if advisor_weights is not None:
            if self.advisor_digest is None:
                raise PredictionPredictorError("advisor weights require a declared advisor_digest")
            if not isinstance(advisor_weights, Mapping) or any(
                not isinstance(key, str)
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= float(value) <= 1
                for key, value in advisor_weights.items()
            ):
                raise PredictionPredictorError(
                    "advisor weights must map branch keys to values within [0, 1]"
                )
        if context.kind is not PredictionKind.TRANSITION:
            raise PredictionPredictorError("the baseline predictor only serves Transition contexts")
        excluded = frozenset(str(key) for key in excluded_branch_keys)
        level = str(context.anchor["anchor_type"])
        if level not in _SUPPORTED_LEVELS:
            raise PredictionPredictorError("transition anchors must be at action or event level")
        coverage = float(context.input["observation_frame"]["coverage"]["coverage_score"])
        if coverage < self.config.min_coverage_score:
            return self._abstain(
                context,
                PredictionAbstentionReason.INSUFFICIENT_CONTEXT,
                predicted_at=predicted_at,
                horizon_seconds=horizon_seconds,
                source_bindings=source_bindings,
                memory_provenance=memory_provenance,
            )
        history = context.input["behavior_history"]
        steps = history["completed_actions"] if level == "action" else history["completed_events"]
        context_key = previous_step_key(steps, self.vocabulary)
        domain = self.vocabulary.canonical_token(
            context.prediction_scope["target_domain"], "prediction_scope.target_domain"
        )
        matched = self._expand_most_specific(level, domain, context_key, predicted_at)
        if matched is None:
            return self._abstain(
                context,
                PredictionAbstentionReason.NO_MATCHING_STATE,
                predicted_at=predicted_at,
                horizon_seconds=horizon_seconds,
                source_bindings=source_bindings,
                memory_provenance=memory_provenance,
            )
        pairs, pattern_bindings = self._with_inherited_candidates(level, domain, matched)
        target_level = str(context.prediction_scope["target_level"])
        target_kind = target_kind_for_level(target_level)
        pairs = tuple(
            (branch, probability)
            for branch, probability in pairs
            if branch.fields["target_kind"] == target_kind
            and (branch.address.branch_key or "") not in excluded
        )
        if not pairs:
            return self._abstain(
                context,
                PredictionAbstentionReason.LOW_CONFIDENCE,
                predicted_at=predicted_at,
                horizon_seconds=horizon_seconds,
                source_bindings=source_bindings,
                pattern_bindings=pattern_bindings,
                memory_provenance=memory_provenance,
                excluded_branch_keys=tuple(sorted(excluded)),
            )
        branches = tuple(branch for branch, _probability in pairs)
        raw_probabilities = tuple(probability for _branch, probability in pairs)
        calibrated = self._calibrated(raw_probabilities)
        adjusted = self._advisor_adjusted(branches, calibrated, advisor_weights)
        ordered = sorted(
            zip(branches, adjusted, raw_probabilities, strict=True),
            key=lambda item: (-item[1], -item[2], item[0].address.branch_key or ""),
        )[: self.config.max_candidates]
        candidates = tuple(
            PredictionCandidate(
                rank=index + 1,
                target_level=PredictionTargetLevel(target_level),
                semantics=branch.fields["behavior"]["semantics"],
                probability=probability,
                expected_delay_seconds=None,
                payload={
                    "target_kind": branch.fields["target_kind"],
                    "behavior": branch.fields["behavior"],
                    "branch_key": branch.address.branch_key,
                },
                evidence_refs=(str(PredictionURI.from_pattern_address(branch.address)),),
            )
            for index, (branch, probability, _raw) in enumerate(ordered)
        )
        run = PredictionRun.create(
            predicted_at=predicted_at,
            horizon_seconds=horizon_seconds,
            context=context,
            predictor_id=_PREDICTOR_ID,
            predictor_version=_PREDICTOR_VERSION,
            predictor_digest=self.predictor_digest,
            source_bindings=source_bindings,
            candidates=candidates,
            pattern_bindings=pattern_bindings,
            memory_provenance=memory_provenance,
            advisor_adjustment=advisor_weights,
            excluded_branch_keys=tuple(sorted(excluded)),
        )
        gates = self._blocked_gates(candidates, ordered[0][0])
        return PredictionDecision(run=run, execute=not gates, blocked_gates=gates)

    def _expand_most_specific(
        self,
        level: str,
        domain: str,
        context_key: dict[str, object] | None,
        predicted_at: datetime,
    ) -> tuple[PredictionStateExpansion, PredictionRunPatternBinding] | None:
        """按特异度降序匹配：序列上下文 → 当前时段 → 同域根状态。

        时段状态承接无前序触发的惯例（如"早上第一件事"）：没有已完成
        步骤可作上下文时，当前时刻本身就是最强的可用条件。
        """

        candidates: list[str] = []
        if context_key is not None:
            candidates.append(logical_state_key(sequence_state_identity(level, domain, context_key)))
        bucket = temporal_bucket(
            predicted_at,
            bucket_hours=self.config.temporal_bucket_hours,
            utc_offset_minutes=self.config.temporal_utc_offset_minutes,
        )
        candidates.append(logical_state_key(temporal_state_identity(level, domain, bucket)))
        candidates.append(logical_state_key(sequence_state_identity(level, domain, None)))
        for key in candidates:
            matched = self._expand_state(key)
            if matched is not None:
                return matched
        return None

    def _expand_state(
        self,
        key: str,
    ) -> tuple[PredictionStateExpansion, PredictionRunPatternBinding] | None:
        try:
            record = self.graph.generation_store.read_active_state(key)
        except PredictionPatternGenerationStoreError as exc:
            if "no active generation" in str(exc):
                return None
            raise PredictionPredictorError(
                "pattern generation store failed integrity checks during prediction"
            ) from exc
        identity = record.manifest.learning_identity
        if identity is not None:
            expected = {
                "temporal_bucket_hours": self.config.temporal_bucket_hours,
                "temporal_utc_offset_minutes": self.config.temporal_utc_offset_minutes,
                "vocabulary_version": self.vocabulary.version,
            }
            if dict(identity) != expected:
                raise PredictionPredictorError(
                    "predictor configuration does not match the learned generation"
                )
        expansion = self.graph.expand(key)
        if expansion.pattern_generation != record.manifest.generation_id:
            raise PredictionPredictorError("active pattern generation changed during prediction")
        binding = PredictionRunPatternBinding(
            key,
            record.manifest.generation_id,
            record.manifest.manifest_digest,
        )
        return expansion, binding

    def _with_inherited_candidates(
        self,
        level: str,
        domain: str,
        matched: tuple[PredictionStateExpansion, PredictionRunPatternBinding],
    ) -> tuple[
        tuple[tuple[PredictionPatternDocument, float], ...],
        tuple[PredictionRunPatternBinding, ...],
    ]:
        """并联同域根状态，为稀疏状态合成父层独有分支的继承候选。

        上下文/时段状态只物化本地观测过的分支；其留白里 escape×parent
        的部分本应属于父层已观测分支。escape 可从已发布概率精确还原：
        Σ(matched) = (1 − escape) + escape·S，其中 S 是父层分布落在
        matched 分支身份上的质量，于是 escape = (1 − Σmatched)/(1 − S)。
        合成后总概率仍严格小于一（父层自身留白 × escape 保持未分配），
        边际门因此面对真实的第二名，而不是被截断后的假高边际。

        TODO(PRED-STORE-001): 匹配态与根状态各自独立激活,重学习发布
        窗口或发布中途崩溃后两者可能出自不同学习批次,escape 还原随之
        失真。批次原子化与同批断言的完整方案见 ``pattern/publication.py``
        模块 docstring,随 Runtime 主链接入一并实现。
        """

        expansion, binding = matched
        pairs = [
            (branch, float(branch.fields["conditional_probability"]))
            for branch in expansion.branches
        ]
        bindings = [binding]
        root_key = logical_state_key(sequence_state_identity(level, domain, None))
        if binding.logical_state_key != root_key:
            root = self._expand_state(root_key)
            if root is not None:
                root_expansion, root_binding = root
                matched_identities = {
                    canonical_json(branch.fields["identity_material"]["branch_identity"]["target"])
                    for branch, _probability in pairs
                }
                root_pairs = [
                    (
                        canonical_json(
                            branch.fields["identity_material"]["branch_identity"]["target"]
                        ),
                        branch,
                        float(branch.fields["conditional_probability"]),
                    )
                    for branch in root_expansion.branches
                ]
                total = sum(probability for _branch, probability in pairs)
                overlap = sum(
                    probability
                    for identity, _branch, probability in root_pairs
                    if identity in matched_identities
                )
                if overlap < 1.0 - 1e-9:
                    escape = min(1.0, max(0.0, (1.0 - total) / (1.0 - overlap)))
                    inherited = [
                        (branch, escape * probability)
                        for identity, branch, probability in root_pairs
                        if identity not in matched_identities and escape * probability > 0
                    ]
                    if inherited:
                        pairs.extend(inherited)
                        bindings.append(root_binding)
        return tuple(pairs), tuple(bindings)

    def _advisor_adjusted(
        self,
        branches: Sequence[PredictionPatternDocument],
        probabilities: tuple[float, ...],
        advisor_weights: Mapping[str, float] | None,
    ) -> tuple[float, ...]:
        """在线顾问偏好的质量守恒重分配；与记忆信号同一套缩放纪律。

        每个分支的调整幅度按其经验置信度缩放——数据充分的分支顾问搬
        不动；总质量守恒，顾问只在既有候选之间搬概率，不创造概率。
        """

        if advisor_weights is None or not advisor_weights or self.config.advisor_boost == 0:
            return probabilities
        total = sum(probabilities)
        if total <= 0:
            return probabilities
        scores: list[float] = []
        for branch, probability in zip(branches, probabilities, strict=True):
            weight = float(advisor_weights.get(branch.address.branch_key or "", 0.0))
            reach = self.config.advisor_boost * (1.0 - float(branch.fields["confidence"]))
            scores.append(probability * (1.0 + reach * weight))
        scale = total / sum(scores)
        return tuple(score * scale for score in scores)

    def _calibrated(self, probabilities: tuple[float, ...]) -> tuple[float, ...]:
        """在任何重分配之前应用等渗校准，保证拟合域与应用域同源。

        映射单调但非严格：PAV 平段会把不同输入坍缩为同值，排序因此以
        校准前概率作次级破平键，平段内仍按证据强度定序。校准逐分支
        独立进行，极端映射可能让总质量越界；越界时整体缩放回到一以内，
        保证 PredictionRun 的概率闭合契约。
        """

        if self.calibration is None:
            return probabilities
        calibrated = tuple(self.calibration.apply(probability) for probability in probabilities)
        total = sum(calibrated)
        if total > 1.0:
            calibrated = tuple(probability / total for probability in calibrated)
        return calibrated

    def _blocked_gates(
        self,
        candidates: tuple[PredictionCandidate, ...],
        top_branch: PredictionPatternDocument,
    ) -> tuple[str, ...]:
        gates: list[str] = []
        top_probability = candidates[0].probability
        runner_up = candidates[1].probability if len(candidates) > 1 else 0.0
        if top_probability < self.config.execute_probability_threshold:
            gates.append("probability_below_threshold")
        if top_probability - runner_up < self.config.execute_margin:
            gates.append("margin_below_threshold")
        if top_branch.fields["support_count"] < self.config.min_execute_support:
            gates.append("support_below_minimum")
        negative = sum(
            outcome["probability"]
            for outcome in top_branch.fields["outcomes"]
            if outcome["valence"] == "negative"
        )
        if negative > self.config.max_negative_outcome_probability:
            gates.append("negative_outcome_risk")
        return tuple(gates)

    def _abstain(
        self,
        context: PredictionContext,
        reason: PredictionAbstentionReason,
        *,
        predicted_at: datetime,
        horizon_seconds: float,
        source_bindings: Sequence[PredictionRunSourceBinding],
        pattern_bindings: Sequence[PredictionRunPatternBinding] = (),
        memory_provenance: str | None = None,
        excluded_branch_keys: tuple[str, ...] = (),
    ) -> PredictionDecision:
        run = PredictionRun.create(
            predicted_at=predicted_at,
            horizon_seconds=horizon_seconds,
            context=context,
            predictor_id=_PREDICTOR_ID,
            predictor_version=_PREDICTOR_VERSION,
            predictor_digest=self.predictor_digest,
            source_bindings=source_bindings,
            abstention_reason=reason,
            pattern_bindings=pattern_bindings,
            memory_provenance=memory_provenance,
            excluded_branch_keys=excluded_branch_keys,
        )
        return PredictionDecision(
            run=run,
            execute=False,
            blocked_gates=(f"abstained:{reason.value}",),
        )


__all__ = ["PatternGraphPredictor", "PredictionDecision"]
