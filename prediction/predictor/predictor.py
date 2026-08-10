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
    temporal_bucket,
    temporal_state_identity,
)
from prediction.learning.vocabulary import PredictionBehaviorVocabulary
from prediction.model import PredictionKind, PredictionTargetLevel
from prediction.pattern.document import PredictionPatternDocument
from prediction.pattern.generation import PredictionPatternGenerationStoreError
from prediction.pattern.graph import PredictionPatternGraph, PredictionStateExpansion
from prediction.predictor.config import PredictionDecisionConfig, PredictionPredictorError
from prediction.predictor.signals import PredictionMemorySignal
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
    记忆信号只在候选分支之间做质量守恒的概率重分配，不凭空创造概率；
    执行判决从代价不对称出发，概率、边际、经验数和负向结果风险四道
    门槛全部通过才允许执行 top-1，任何弃答或拦截都带受控原因。
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
        memory_signals: Sequence[PredictionMemorySignal] = (),
        advisor_weights: Mapping[str, float] | None = None,
        signal_provenance: str | None = None,
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
        signals = tuple(memory_signals)
        if any(not isinstance(item, PredictionMemorySignal) for item in signals):
            raise PredictionPredictorError("memory signals must be PredictionMemorySignal values")
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
                signal_provenance=signal_provenance,
            )
        pairs, pattern_bindings = self._with_inherited_candidates(level, domain, matched)
        target_level = str(context.prediction_scope["target_level"])
        pairs = tuple(
            (branch, probability)
            for branch, probability in pairs
            if branch.fields["target_kind"] == target_level
        )
        if not pairs:
            return self._abstain(
                context,
                PredictionAbstentionReason.LOW_CONFIDENCE,
                predicted_at=predicted_at,
                horizon_seconds=horizon_seconds,
                source_bindings=source_bindings,
                pattern_bindings=pattern_bindings,
                signal_provenance=signal_provenance,
            )
        branches = tuple(branch for branch, _probability in pairs)
        probabilities = tuple(probability for _branch, probability in pairs)
        adjusted = self._memory_adjusted(branches, probabilities, signals)
        adjusted = self._advisor_adjusted(branches, adjusted, advisor_weights)
        adjusted = self._calibrated(adjusted)
        ordered = sorted(
            zip(branches, adjusted, strict=True),
            key=lambda item: (-item[1], item[0].address.branch_key or ""),
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
            for index, (branch, probability) in enumerate(ordered)
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
            signal_provenance=signal_provenance,
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

    def _memory_adjusted(
        self,
        branches: Sequence[PredictionPatternDocument],
        probabilities: tuple[float, ...],
        signals: Sequence[PredictionMemorySignal],
    ) -> tuple[float, ...]:
        """记忆信号的门控概率重分配；记忆是补充证据，不是常开输入。

        行为数据已经高度确定的惯例（top-1 概率与经验置信度同时过线）
        直接旁路记忆；旁路检查与执行门槛使用同一量纲——有校准时先校准
        再比较，避免"旁路认为确定、执行门认为不确定"的矛盾。参与时
        每个分支的调整幅度按其经验置信度缩放——
        数据充分的分支难以被记忆搬动，数据稀疏的分支才由记忆补充；
        支持信号乘性加强，反对信号对称压制，最后按原总质量归一，
        记忆只搬移概率，不创造概率。
        """

        if not signals or self.config.memory_boost == 0:
            return probabilities
        total = sum(probabilities)
        if total <= 0:
            return probabilities
        top_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        top_probability = probabilities[top_index]
        if self.calibration is not None:
            top_probability = self.calibration.apply(top_probability)
        if (
            top_probability >= self.config.routine_bypass_probability
            and float(branches[top_index].fields["confidence"])
            >= self.config.routine_bypass_confidence
        ):
            return probabilities
        scores: list[float] = []
        for branch, probability in zip(branches, probabilities, strict=True):
            support, oppose = self._signal_match(branch, signals)
            reach = self.config.memory_boost * (1.0 - float(branch.fields["confidence"]))
            multiplier = (1.0 + reach * support) / (1.0 + reach * oppose)
            scores.append(probability * multiplier)
        scale = total / sum(scores)
        return tuple(score * scale for score in scores)

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
        """把等渗校准应用到最终分支概率；映射单调，排序不被打乱。

        校准逐分支独立进行，极端映射可能让总质量越界；越界时整体缩放
        回到一以内，保证 PredictionRun 的概率闭合契约。
        """

        if self.calibration is None:
            return probabilities
        calibrated = tuple(self.calibration.apply(probability) for probability in probabilities)
        total = sum(calibrated)
        if total > 1.0:
            calibrated = tuple(probability / total for probability in calibrated)
        return calibrated

    def _signal_match(
        self,
        branch: PredictionPatternDocument,
        signals: Sequence[PredictionMemorySignal],
    ) -> tuple[float, float]:
        """返回该分支命中的最强支持权重与最强反对权重。"""

        behavior = branch.fields["behavior"]
        branch_semantics = self.vocabulary.canonical_token(behavior["semantics"], "branch semantics")
        branch_refs = {
            self.vocabulary.canonical_token(item, "branch target_ref")
            for item in behavior["target_refs"]
        }
        support = 0.0
        oppose = 0.0
        for signal in signals:
            signal_semantics = self.vocabulary.canonical_token(signal.semantics, "signal semantics")
            signal_refs = {
                self.vocabulary.canonical_token(item, "signal target_ref")
                for item in signal.target_refs
            }
            semantics_hit = _contains_either(signal_semantics, branch_semantics)
            refs_hit = bool(branch_refs & signal_refs) or any(
                _contains_either(signal_semantics, ref) for ref in branch_refs
            )
            if not (semantics_hit or refs_hit):
                continue
            if signal.stance == "oppose":
                oppose = max(oppose, signal.weight)
            else:
                support = max(support, signal.weight)
        return support, oppose

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
        signal_provenance: str | None = None,
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
            signal_provenance=signal_provenance,
        )
        return PredictionDecision(
            run=run,
            execute=False,
            blocked_gates=(f"abstained:{reason.value}",),
        )


def _contains_either(left: str, right: str) -> bool:
    """规范化文本的确定性包含匹配。

    记忆信号是自由中文（如"回家后记得倒一杯水"），与行为词表的分支语义
    （"倒一杯水"）几乎不会逐字相等；较短一侧完整出现在较长一侧即视为命中。
    单字符不参与包含匹配，避免虚假命中。
    """

    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    return len(shorter) >= 2 and shorter in longer


__all__ = ["PatternGraphPredictor", "PredictionDecision"]
