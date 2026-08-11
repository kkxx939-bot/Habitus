"""预测树学习结果的确定性回测评估。

评估是所有准确率增量的度量基础：时间切分回测给出 top-k 命中率、log-loss、
校准误差与覆盖率–精确率曲线；熵上限报告给出数据本身的可预测性天花板，
先知道"最好能到多少"再谈指标。全部计算是样本集的纯函数，同输入必同输出。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from foundation.integrity import canonical_json
from prediction.document import PredictionDocument
from prediction.learning.calibration import PredictionProbabilityCalibration
from prediction.learning.config import PredictionLearningError
from prediction.learning.keys import (
    label_branch_identity,
    logical_state_key,
    previous_step_key,
    sequence_state_identity,
    target_kind_for_level,
    temporal_bucket,
    temporal_state_identity,
)
from prediction.learning.learner import PredictionPatternLearner
from prediction.model import PredictionKind, PredictionPatternKind
from prediction.pattern.document import PredictionPatternDocument

_OBSERVED_LABEL_STATUSES = frozenset({"observed", "terminal"})
_DEFAULT_THRESHOLD_GRID = tuple(round(0.05 * index, 2) for index in range(1, 20))


@dataclass(frozen=True)
class PredictionEvaluationConfig:
    """回测评估的切分比例、指标粒度与数值边界。"""

    train_fraction: float = 0.8
    top_k: tuple[int, ...] = (1, 2, 3)
    threshold_grid: tuple[float, ...] = _DEFAULT_THRESHOLD_GRID
    calibration_bins: int = 10
    log_probability_floor: float = 1e-6

    def __post_init__(self) -> None:
        fraction = _finite(self.train_fraction, "train_fraction")
        if not 0 < fraction < 1:
            raise PredictionLearningError("train_fraction must be strictly between zero and one")
        object.__setattr__(self, "train_fraction", fraction)
        ranks = tuple(self.top_k)
        if (
            not ranks
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in ranks)
            or tuple(sorted(set(ranks))) != ranks
        ):
            raise PredictionLearningError("top_k must be strictly increasing positive integers")
        object.__setattr__(self, "top_k", ranks)
        grid = tuple(_finite(item, "threshold_grid item") for item in self.threshold_grid)
        if not grid or any(not 0 < item < 1 for item in grid) or tuple(sorted(set(grid))) != grid:
            raise PredictionLearningError("threshold_grid must be strictly increasing within (0, 1)")
        object.__setattr__(self, "threshold_grid", grid)
        if (
            isinstance(self.calibration_bins, bool)
            or not isinstance(self.calibration_bins, int)
            or self.calibration_bins < 2
        ):
            raise PredictionLearningError("calibration_bins must be an integer of at least two")
        floor = _finite(self.log_probability_floor, "log_probability_floor")
        if not 0 < floor < 1:
            raise PredictionLearningError("log_probability_floor must be strictly between zero and one")
        object.__setattr__(self, "log_probability_floor", floor)


@dataclass(frozen=True)
class PredictionEntropyCeiling:
    """一个 (层级, 域) 分组的经验条件熵与理想 top-1 上限。"""

    state_level: str
    target_domain: str
    labeled_count: int
    root_entropy_bits: float
    context_entropy_bits: float
    bayes_top1_root: float
    bayes_top1_context: float


@dataclass(frozen=True)
class PredictionBacktestReport:
    """一次时间切分回测的完整可复现结果。"""

    train_count: int
    holdout_count: int
    censored_count: int
    unmatched_count: int
    labeled_count: int
    escape_rate: float | None
    top_k_hit_rates: tuple[tuple[int, float], ...]
    log_loss: float | None
    expected_calibration_error: float | None
    coverage_precision: tuple[tuple[float, float, float | None], ...]


@dataclass(frozen=True)
class _SampleView:
    """一条样本在评估中需要的最小事实。"""

    level: str
    domain: str
    target_kind: str
    context: dict[str, Any] | None
    label_key: str | None
    cutoff: datetime
    uri_order: str


def temporal_split(
    samples: Sequence[PredictionDocument],
    *,
    train_fraction: float = 0.8,
) -> tuple[tuple[PredictionDocument, ...], tuple[PredictionDocument, ...]]:
    """按预测切点时间把样本切成训练段与留出段；绝不随机切分。

    随机切分会把未来的惯例漂移泄漏进训练段，得到虚高的指标。
    """

    documents = _documents(samples)
    if len(documents) < 2:
        raise PredictionLearningError("temporal split requires at least two samples")
    fraction = _finite(train_fraction, "train_fraction")
    if not 0 < fraction < 1:
        raise PredictionLearningError("train_fraction must be strictly between zero and one")
    ordered = sorted(
        documents,
        key=lambda item: (
            str(item.fields["anchor"]["cutoff_at"]),
            str(item.fields["materialization_id"]),
        ),
    )
    boundary = min(len(ordered) - 1, max(1, int(len(ordered) * fraction)))
    return tuple(ordered[:boundary]), tuple(ordered[boundary:])


def entropy_report(
    samples: Sequence[PredictionDocument],
    *,
    learner: PredictionPatternLearner | None = None,
) -> tuple[PredictionEntropyCeiling, ...]:
    """计算各 (层级, 域) 分组的经验条件熵与理想 top-1 准确率上限。

    上限只由数据决定，与估计器无关：根粒度回答"完全不看上下文最好能多准"，
    上下文粒度回答"看了上一步之后最好能多准"。哪一层不再降熵，
    哪一层的状态就不值得物化。
    """

    resolved = learner or PredictionPatternLearner()
    views = [
        view
        for view in (_sample_view(document, resolved) for document in _documents(samples))
        if view.label_key is not None
    ]
    groups: dict[tuple[str, str], list[_SampleView]] = {}
    for view in views:
        groups.setdefault((view.level, view.domain), []).append(view)
    report: list[PredictionEntropyCeiling] = []
    for (level, domain), members in sorted(groups.items()):
        root_labels = [view.label_key for view in members if view.label_key is not None]
        by_context: dict[str, list[str]] = {}
        for view in members:
            context_key = "" if view.context is None else canonical_json(view.context)
            assert view.label_key is not None
            by_context.setdefault(context_key, []).append(view.label_key)
        context_entropy = 0.0
        context_bayes = 0.0
        for labels in by_context.values():
            share = len(labels) / len(members)
            context_entropy += share * _entropy_bits(labels)
            context_bayes += share * _bayes_top1(labels)
        report.append(
            PredictionEntropyCeiling(
                state_level=level,
                target_domain=domain,
                labeled_count=len(members),
                root_entropy_bits=_entropy_bits(root_labels),
                context_entropy_bits=context_entropy,
                bayes_top1_root=_bayes_top1(root_labels),
                bayes_top1_context=context_bayes,
            )
        )
    return tuple(report)


def fit_probability_calibration(
    samples: Sequence[PredictionDocument],
    *,
    learned_at: datetime,
    learner: PredictionPatternLearner | None = None,
    config: PredictionEvaluationConfig | None = None,
) -> PredictionProbabilityCalibration:
    """在时间切分的留出段上拟合分支概率的等渗校准映射。

    三段协议防泄漏：只取时间上前 ``train_fraction`` 的训练窗口，再在
    窗口内部按同比切成"内层训练段 / 校准段"——学习在内层训练段、校准
    对采自校准段。最终留出段（后 1−train_fraction）校准从未见过，
    ``backtest(calibration=...)`` 在其上报告的 ECE 与覆盖率曲线才是
    真实的外推校准质量；在拟合数据上自评的等渗 ECE 必然接近零，
    会把执行门槛定得系统性过松。校准对来自校准段每条有标签样本所
    匹配状态的全部分支：(分支概率, 该分支是否就是真实下一步)。
    """

    resolved_config = config or PredictionEvaluationConfig()
    if not isinstance(resolved_config, PredictionEvaluationConfig):
        raise PredictionLearningError("config must be PredictionEvaluationConfig")
    resolved_learner = learner or PredictionPatternLearner()
    if not isinstance(resolved_learner, PredictionPatternLearner):
        raise PredictionLearningError("learner must be PredictionPatternLearner")
    window, _unseen = temporal_split(samples, train_fraction=resolved_config.train_fraction)
    train, holdout = temporal_split(window, train_fraction=resolved_config.train_fraction)
    patterns = resolved_learner.learn(train, learned_at=learned_at)
    distributions = _distribution_index(patterns)
    outcomes: list[tuple[float, bool]] = []
    for document in holdout:
        supervision = document.fields["supervision"]
        if bool(supervision["censored"]) or str(supervision["label_status"]) not in _OBSERVED_LABEL_STATUSES:
            continue
        view = _sample_view(document, resolved_learner)
        distribution = _matched_distribution(distributions, view, resolved_learner)
        if distribution:
            distribution = _level_filtered(distribution, view.target_kind)
        if not distribution:
            continue
        outcomes.extend(
            (probability, key == view.label_key) for key, probability in sorted(distribution.items())
        )
    return PredictionProbabilityCalibration.from_outcomes(outcomes)


def backtest(
    samples: Sequence[PredictionDocument],
    *,
    learned_at: datetime,
    learner: PredictionPatternLearner | None = None,
    config: PredictionEvaluationConfig | None = None,
    calibration: PredictionProbabilityCalibration | None = None,
) -> PredictionBacktestReport:
    """时间切分回测：早段训练、晚段评估，输出可复现的完整指标。

    评估用与学习器完全相同的键派生在留出段上匹配状态（先上下文后根），
    因此度量的是整条"学习 + 匹配"链路，而不只是估计器本身。逃逸样本
    （真实标签不在匹配分布内）按下限概率记损——把整块留白记给单个
    未见行为会让 log-loss 失去 proper 性，空分布模型反而得零损失；
    零分支的空分布视同未匹配，不参与指标。
    """

    resolved_config = config or PredictionEvaluationConfig()
    if not isinstance(resolved_config, PredictionEvaluationConfig):
        raise PredictionLearningError("config must be PredictionEvaluationConfig")
    resolved_learner = learner or PredictionPatternLearner()
    if not isinstance(resolved_learner, PredictionPatternLearner):
        raise PredictionLearningError("learner must be PredictionPatternLearner")
    if calibration is not None and not isinstance(calibration, PredictionProbabilityCalibration):
        raise PredictionLearningError("calibration must be PredictionProbabilityCalibration")
    train, holdout = temporal_split(samples, train_fraction=resolved_config.train_fraction)
    patterns = resolved_learner.learn(train, learned_at=learned_at)
    distributions = _distribution_index(patterns)
    censored_count = 0
    unmatched_count = 0
    escape_count = 0
    log_loss_sum = 0.0
    hit_counts = dict.fromkeys(resolved_config.top_k, 0)
    top1_outcomes: list[tuple[float, bool]] = []
    labeled_count = 0
    for document in holdout:
        supervision = document.fields["supervision"]
        if bool(supervision["censored"]) or str(supervision["label_status"]) not in _OBSERVED_LABEL_STATUSES:
            censored_count += 1
            continue
        view = _sample_view(document, resolved_learner)
        distribution = _matched_distribution(distributions, view, resolved_learner)
        if distribution:
            distribution = _level_filtered(distribution, view.target_kind)
        if not distribution:
            unmatched_count += 1
            continue
        assert view.label_key is not None
        labeled_count += 1
        ordered = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
        true_probability = distribution.get(view.label_key)
        if true_probability is None:
            escape_count += 1
            loss_probability = resolved_config.log_probability_floor
        else:
            loss_probability = max(true_probability, resolved_config.log_probability_floor)
        log_loss_sum += -math.log(loss_probability)
        for rank in resolved_config.top_k:
            if any(key == view.label_key for key, _probability in ordered[:rank]):
                hit_counts[rank] += 1
        if ordered:
            top1_key, top1_probability = ordered[0]
            if calibration is not None:
                top1_probability = calibration.apply(top1_probability)
            top1_outcomes.append((top1_probability, top1_key == view.label_key))
    return PredictionBacktestReport(
        train_count=len(train),
        holdout_count=len(holdout),
        censored_count=censored_count,
        unmatched_count=unmatched_count,
        labeled_count=labeled_count,
        escape_rate=None if labeled_count == 0 else escape_count / labeled_count,
        top_k_hit_rates=tuple(
            (rank, 0.0 if labeled_count == 0 else hit_counts[rank] / labeled_count)
            for rank in resolved_config.top_k
        ),
        log_loss=None if labeled_count == 0 else log_loss_sum / labeled_count,
        expected_calibration_error=_expected_calibration_error(
            top1_outcomes, resolved_config.calibration_bins
        ),
        coverage_precision=_coverage_precision(top1_outcomes, resolved_config.threshold_grid),
    )


def _documents(samples: Sequence[PredictionDocument]) -> tuple[PredictionDocument, ...]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise PredictionLearningError("samples must be a sequence of PredictionDocument values")
    documents: list[PredictionDocument] = []
    for document in samples:
        if not isinstance(document, PredictionDocument):
            raise PredictionLearningError("samples must contain PredictionDocument values")
        if document.kind is not PredictionKind.TRANSITION:
            raise PredictionLearningError("evaluation only covers TransitionSamples")
        documents.append(document)
    return tuple(documents)


def _sample_view(document: PredictionDocument, learner: PredictionPatternLearner) -> _SampleView:
    fields = document.fields
    level = str(fields["anchor"]["anchor_type"])
    domain = learner.vocabulary.canonical_token(
        fields["prediction_scope"]["target_domain"], "prediction_scope.target_domain"
    )
    history = fields["input"]["behavior_history"]
    steps = history["completed_actions"] if level == "action" else history["completed_events"]
    context = previous_step_key(steps, learner.vocabulary)
    supervision = fields["supervision"]
    labeled = (
        not bool(supervision["censored"])
        and str(supervision["label_status"]) in _OBSERVED_LABEL_STATUSES
    )
    label_key = (
        canonical_json(label_branch_identity(fields["label"], learner.vocabulary)) if labeled else None
    )
    target_level = str(fields["prediction_scope"]["target_level"])
    return _SampleView(
        level=level,
        domain=domain,
        target_kind=target_kind_for_level(target_level),
        context=context,
        label_key=label_key,
        cutoff=datetime.fromisoformat(str(fields["anchor"]["cutoff_at"]).replace("Z", "+00:00")),
        uri_order=str(fields["materialization_id"]),
    )


def _distribution_index(
    patterns: Sequence[PredictionPatternDocument],
) -> dict[str, dict[str, float]]:
    logical_by_state_key: dict[str, str] = {}
    distributions: dict[str, dict[str, float]] = {}
    for document in patterns:
        if document.kind is PredictionPatternKind.STATE:
            logical_by_state_key[str(document.fields["state_key"])] = str(
                document.fields["logical_state_key"]
            )
            distributions.setdefault(str(document.fields["logical_state_key"]), {})
    for document in patterns:
        if document.kind is not PredictionPatternKind.BRANCH:
            continue
        logical_key = logical_by_state_key[str(document.fields["state_key"])]
        target = document.fields["identity_material"]["branch_identity"]["target"]
        distributions[logical_key][canonical_json(target)] = float(
            document.fields["conditional_probability"]
        )
    return distributions


def _matched_distribution(
    distributions: Mapping[str, Mapping[str, float]],
    view: _SampleView,
    learner: PredictionPatternLearner,
) -> Mapping[str, float] | None:
    """镜像预测器的匹配链：序列上下文 → 样本时刻的时段 → 同域根状态。"""

    root = distributions.get(
        logical_state_key(sequence_state_identity(view.level, view.domain, None))
    )
    matched: Mapping[str, float] | None = None
    if view.context is not None:
        matched = distributions.get(
            logical_state_key(sequence_state_identity(view.level, view.domain, view.context))
        )
    if matched is None:
        bucket = temporal_bucket(
            view.cutoff,
            bucket_hours=learner.config.temporal_bucket_hours,
            utc_offset_minutes=learner.config.temporal_utc_offset_minutes,
        )
        matched = distributions.get(
            logical_state_key(temporal_state_identity(view.level, view.domain, bucket))
        )
    if matched is None:
        return root
    return _with_inherited_mass(matched, root)


def _level_filtered(
    distribution: Mapping[str, float],
    target_kind: str,
) -> dict[str, float]:
    """镜像线上判决的 target_level 过滤：只对样本目标层级的分支记分。

    混层分布里的 termination 分支线上永远不会成为 next_step 候选，
    让它参与 top-k/log-loss 会把指标测偏到非服务路径上。
    """

    return {
        key: probability
        for key, probability in distribution.items()
        if json.loads(key)["target_kind"] == target_kind
    }


def _with_inherited_mass(
    matched: Mapping[str, float],
    root: Mapping[str, float] | None,
) -> Mapping[str, float]:
    """镜像预测器的继承合成：把稀疏状态留白中 escape×parent 的部分还给父层分支。"""

    if not root:
        return matched
    total = sum(matched.values())
    overlap = sum(probability for key, probability in root.items() if key in matched)
    if overlap >= 1.0 - 1e-9:
        return matched
    escape = min(1.0, max(0.0, (1.0 - total) / (1.0 - overlap)))
    merged = dict(matched)
    for key, probability in root.items():
        if key not in merged and escape * probability > 0:
            merged[key] = escape * probability
    return merged


def _entropy_bits(labels: Sequence[str]) -> float:
    total = len(labels)
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    entropy = 0.0
    for count in counts.values():
        share = count / total
        entropy -= share * math.log2(share)
    return entropy


def _bayes_top1(labels: Sequence[str]) -> float:
    total = len(labels)
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return max(counts.values()) / total


def _expected_calibration_error(
    outcomes: Sequence[tuple[float, bool]],
    bins: int,
) -> float | None:
    if not outcomes:
        return None
    grouped: dict[int, list[tuple[float, bool]]] = {}
    for probability, hit in outcomes:
        index = min(int(probability * bins), bins - 1)
        grouped.setdefault(index, []).append((probability, hit))
    error = 0.0
    for members in grouped.values():
        confidence = sum(probability for probability, _hit in members) / len(members)
        accuracy = sum(1 for _probability, hit in members if hit) / len(members)
        error += (len(members) / len(outcomes)) * abs(accuracy - confidence)
    return error


def _coverage_precision(
    outcomes: Sequence[tuple[float, bool]],
    grid: Sequence[float],
) -> tuple[tuple[float, float, float | None], ...]:
    points: list[tuple[float, float, float | None]] = []
    for threshold in grid:
        selected = [hit for probability, hit in outcomes if probability >= threshold]
        if not outcomes:
            points.append((threshold, 0.0, None))
            continue
        coverage = len(selected) / len(outcomes)
        precision = None if not selected else sum(selected) / len(selected)
        points.append((threshold, coverage, precision))
    return tuple(points)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictionLearningError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise PredictionLearningError(f"{label} must be finite")
    return normalized


__all__ = [
    "PredictionBacktestReport",
    "PredictionEntropyCeiling",
    "PredictionEvaluationConfig",
    "backtest",
    "entropy_report",
    "fit_probability_calibration",
    "temporal_split",
]
