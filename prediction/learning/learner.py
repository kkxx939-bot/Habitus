"""Sample → Pattern 的确定性学习聚合器。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from foundation.integrity import canonical_digest, canonical_json, canonicalize
from prediction.document import PredictionDocument
from prediction.learning.config import PredictionLearningConfig, PredictionLearningError
from prediction.learning.keys import (
    label_branch_identity,
    logical_state_key,
    previous_step_key,
    sequence_state_identity,
    temporal_state_identity,
    treatment_branch_identity,
)
from prediction.learning.prior import PredictionBehaviorPrior
from prediction.learning.vocabulary import PredictionBehaviorVocabulary
from prediction.model import PredictionKind
from prediction.pattern.document import PredictionPatternDocument
from prediction.pattern.factory import PredictionPatternFactory
from prediction.uri import PredictionURI

_LEVELS = frozenset({"action", "event"})
_OBSERVED_LABEL_STATUSES = frozenset({"observed", "terminal"})
_ROOT_CONTEXT_KEY = ""
_LEVEL_NAMES = {"action": "动作", "event": "事件"}


@dataclass(frozen=True)
class _Observation:
    """一条 TransitionSample 中参与聚合的最小事实。"""

    uri: str
    level: str
    domain: str
    context: dict[str, Any] | None
    branch: dict[str, Any] | None
    raw_semantics: str | None
    raw_actor: str | None
    weight: float
    cutoff: datetime
    group_id: str


class _BranchAccumulator:
    """一个 (State, 行为) 组合的加权计数器。"""

    def __init__(self, identity: Mapping[str, Any]) -> None:
        self.identity: dict[str, Any] = dict(identity)
        self.group_weights: dict[str, float] = {}
        self.raw_count = 0
        self.references: list[tuple[datetime, str]] = []
        self.semantics_weights: dict[str, float] = {}
        self.actor_weights: dict[str, float] = {}

    def add(self, observation: _Observation, *, scaled_weight: float, record: bool) -> None:
        if record:
            self.raw_count += 1
            self.references.append((observation.cutoff, observation.uri))
        self.group_weights[observation.group_id] = (
            self.group_weights.get(observation.group_id, 0.0) + scaled_weight
        )
        if observation.raw_semantics is not None:
            self.semantics_weights[observation.raw_semantics] = (
                self.semantics_weights.get(observation.raw_semantics, 0.0) + scaled_weight
            )
        if observation.raw_actor is not None:
            self.actor_weights[observation.raw_actor] = (
                self.actor_weights.get(observation.raw_actor, 0.0) + scaled_weight
            )


class _StateAccumulator:
    """一个抽象格状态的曝光、分支与删失计数器。

    时间族状态用核平滑把一条样本按分数权重摊到相邻时段；分数贡献只进
    加权计数，原始支持度、证据引用与统计窗口只在主时段记账一次。
    """

    def __init__(
        self,
        level: str,
        domain: str,
        context: dict[str, Any] | None,
        bucket: str | None = None,
    ) -> None:
        self.level = level
        self.domain = domain
        self.context = context
        self.bucket = bucket
        self.branches: dict[str, _BranchAccumulator] = {}
        self.censored_group_weights: dict[str, float] = {}
        self.raw_count = 0
        self.references: list[tuple[datetime, str]] = []
        self.groups: set[str] = set()
        self.first_observed_at: datetime | None = None
        self.last_observed_at: datetime | None = None

    def add(self, observation: _Observation, *, weight_scale: float = 1.0, record: bool = True) -> None:
        scaled_weight = observation.weight * weight_scale
        if record:
            self.raw_count += 1
            self.references.append((observation.cutoff, observation.uri))
            self.groups.add(observation.group_id)
            if self.first_observed_at is None or observation.cutoff < self.first_observed_at:
                self.first_observed_at = observation.cutoff
            if self.last_observed_at is None or observation.cutoff > self.last_observed_at:
                self.last_observed_at = observation.cutoff
        if observation.branch is None:
            self.censored_group_weights[observation.group_id] = (
                self.censored_group_weights.get(observation.group_id, 0.0) + scaled_weight
            )
            return
        key = canonical_json(observation.branch)
        accumulator = self.branches.get(key)
        if accumulator is None:
            accumulator = _BranchAccumulator(observation.branch)
            self.branches[key] = accumulator
        accumulator.add(observation, scaled_weight=scaled_weight, record=record)


class PredictionPatternLearner:
    """把 TransitionSample 集合聚合为一代 State/Branch Pattern 的纯函数学习器。

    估计采用两层 Pitman–Yor 插值回退：上下文状态（上一步行为）的分布向同层
    根状态收缩，折扣项产生的未分配概率质量显式表示"下一步是未见行为"的留白，
    因此每个 State 的出边概率和严格小于一。计数经过质量加权（样本
    ``source_confidence``）、同 occurrence_group 封顶和时间指数衰减修正；
    删失样本只进入状态曝光量，不为任何具体分支作证。全部超参、词表版本、
    ``learned_at`` 与样本身份共同构成 generation 身份，同输入必同输出；
    本层不落盘、不加锁，物化与激活交给 Pattern 发布器。
    """

    def __init__(
        self,
        *,
        config: PredictionLearningConfig | None = None,
        vocabulary: PredictionBehaviorVocabulary | None = None,
        factory: PredictionPatternFactory | None = None,
        prior: PredictionBehaviorPrior | None = None,
    ) -> None:
        if config is not None and not isinstance(config, PredictionLearningConfig):
            raise TypeError("config must be PredictionLearningConfig")
        if vocabulary is not None and not isinstance(vocabulary, PredictionBehaviorVocabulary):
            raise TypeError("vocabulary must be PredictionBehaviorVocabulary")
        if factory is not None and not isinstance(factory, PredictionPatternFactory):
            raise TypeError("factory must be PredictionPatternFactory")
        if prior is not None and not isinstance(prior, PredictionBehaviorPrior):
            raise TypeError("prior must be PredictionBehaviorPrior")
        self.config = config or PredictionLearningConfig()
        self.vocabulary = vocabulary or PredictionBehaviorVocabulary()
        self.factory = factory or PredictionPatternFactory()
        self.prior = prior

    def match_identity(self) -> dict[str, Any]:
        """预测端构造期握手所需的匹配关键参数快照。

        发布时经 ``learning_identity`` 随 manifest 落盘；预测器配置或词表
        与这份快照不一致时必须显式失败——失配的时段参数会让预测器精确
        命中语义错误的时段状态，失配的词表会让全部键静默 miss 到根。
        """

        return {
            "temporal_bucket_hours": self.config.temporal_bucket_hours,
            "temporal_utc_offset_minutes": self.config.temporal_utc_offset_minutes,
            "vocabulary_version": self.vocabulary.version,
        }

    def learn(
        self,
        samples: Sequence[PredictionDocument],
        *,
        learned_at: datetime,
        consequences: Sequence[PredictionDocument] = (),
    ) -> tuple[PredictionPatternDocument, ...]:
        """从不可变 TransitionSample 聚合出可直接发布的一批 Pattern 文档。

        ``consequences`` 是可选的 ConsequenceSample 集合：按 (域, 行为键)
        聚合出结果类别分布，挂到对应分支的 ``outcomes``，供执行判决的
        风险门消费。同一 Outcome 的多次修订先收敛到最新有效物化再计数。
        """

        if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
            raise PredictionLearningError("samples must be a sequence of PredictionDocument values")
        if (
            not isinstance(learned_at, datetime)
            or learned_at.tzinfo is None
            or learned_at.utcoffset() is None
        ):
            raise PredictionLearningError("learned_at must be a timezone-aware datetime")
        observations: list[_Observation] = []
        versions: set[str] = set()
        materialization_ids: set[str] = set()
        seen: dict[str, PredictionDocument] = {}
        for document in samples:
            if not isinstance(document, PredictionDocument):
                raise PredictionLearningError("samples must contain PredictionDocument values")
            if document.kind is not PredictionKind.TRANSITION:
                raise PredictionLearningError("the baseline learner only aggregates TransitionSamples")
            uri = str(PredictionURI.from_address(document.address))
            existing = seen.get(uri)
            if existing is not None:
                if existing != document:
                    raise PredictionLearningError("one learning round binds one sample URI to different content")
                continue
            seen[uri] = document
            versions.add(str(document.fields["provenance"]["projection_version"]))
            materialization_ids.add(str(document.fields["materialization_id"]))
            observations.append(self._observation(uri, document, learned_at))
        if not observations:
            raise PredictionLearningError("pattern learning requires at least one TransitionSample")
        effective_consequences = self._effective_consequences(consequences, versions)
        if len(versions) != 1:
            raise PredictionLearningError("one learning round must use one projection version")
        projection_version = next(iter(versions))
        outcome_index = self._consequence_outcomes(effective_consequences, learned_at)
        generation_id = canonical_digest(
            {
                "contract": "prediction-pattern-learning-v1",
                "projection_version": projection_version,
                "config": self.config.identity_material(),
                "vocabulary_version": self.vocabulary.version,
                "learned_at": canonicalize(learned_at),
                "sample_materialization_ids": sorted(materialization_ids),
                "consequence_materialization_ids": sorted(
                    str(document.fields["materialization_id"]) for document in effective_consequences
                ),
                "prior_version": None if self.prior is None else self.prior.version,
            }
        )
        states: dict[tuple[str, str, str], _StateAccumulator] = {}
        for observation in observations:
            self._accumulate(
                states,
                (observation.level, observation.domain, _ROOT_CONTEXT_KEY),
                observation,
                context=None,
            )
            if observation.context is not None:
                self._accumulate(
                    states,
                    (observation.level, observation.domain, canonical_json(observation.context)),
                    observation,
                    context=observation.context,
                )
            for bucket, fraction, primary in self._temporal_fractions(observation.cutoff):
                self._accumulate(
                    states,
                    (observation.level, observation.domain, f"hour:{bucket}"),
                    observation,
                    context=None,
                    bucket=bucket,
                    weight_scale=fraction,
                    record=primary,
                )
        documents: list[PredictionPatternDocument] = []
        base_by_group: dict[tuple[str, str], dict[str, float]] = {}
        for group in sorted({key[:2] for key in states}):
            root_accumulator = states[(*group, _ROOT_CONTEXT_KEY)]
            base_by_group[group] = self._distribution(
                root_accumulator,
                parent=None,
                pseudo=self._pseudo_counts(root_accumulator),
            )
        for key in sorted(states):
            level, domain, context_key = key
            accumulator = states[key]
            if context_key == _ROOT_CONTEXT_KEY:
                probabilities = base_by_group[(level, domain)]
            else:
                if accumulator.raw_count < self.config.min_context_support:
                    continue
                probabilities = self._distribution(
                    accumulator,
                    parent=base_by_group[(level, domain)],
                    pseudo=self._pseudo_counts(accumulator),
                )
            documents.extend(
                self._materialize_state(
                    accumulator, probabilities, generation_id, projection_version, outcome_index
                )
            )
        return tuple(documents)

    def _effective_consequences(
        self,
        consequences: Sequence[PredictionDocument],
        versions: set[str],
    ) -> tuple[PredictionDocument, ...]:
        """按逻辑样本收敛 Outcome 修订：只保留最新有效物化参与计数。"""

        if isinstance(consequences, (str, bytes)) or not isinstance(consequences, Sequence):
            raise PredictionLearningError("consequences must be a sequence of PredictionDocument values")
        latest: dict[str, PredictionDocument] = {}
        for document in consequences:
            if not isinstance(document, PredictionDocument):
                raise PredictionLearningError("consequences must contain PredictionDocument values")
            if document.kind is not PredictionKind.CONSEQUENCE:
                raise PredictionLearningError("consequences must contain ConsequenceSamples")
            versions.add(str(document.fields["provenance"]["projection_version"]))
            logical_id = str(document.fields["logical_sample_id"])
            current = latest.get(logical_id)
            if current is None or self._revision_order(document) > self._revision_order(current):
                latest[logical_id] = document
        return tuple(latest[key] for key in sorted(latest))

    @staticmethod
    def _revision_order(document: PredictionDocument) -> tuple[int, str]:
        return (
            int(document.fields["materialization_context"]["outcome_revision"]),
            str(document.fields["materialization_id"]),
        )

    def _consequence_outcomes(
        self,
        consequences: Sequence[PredictionDocument],
        learned_at: datetime,
    ) -> dict[tuple[str, str], tuple[dict[str, Any], ...]]:
        """按 (域, 行为键) 聚合结果类别分布，供分支 ``outcomes`` 消费。

        类别 = outcome_type × valence；计数经质量加权、consequence_group
        封顶与时间衰减，折扣产生的留白表示"还有未见过的结果"。延迟取
        中位数——结果延迟长尾重，均值会被单个迟到结果拖爆。
        """

        buckets: dict[tuple[str, str], dict[tuple[str, str], dict[str, Any]]] = {}
        for document in consequences:
            fields = document.fields
            supervision = fields["supervision"]
            if bool(supervision["censored"]):
                continue
            domain = self.vocabulary.canonical_token(
                fields["prediction_scope"]["target_domain"], "prediction_scope.target_domain"
            )
            branch_key = canonical_json(
                treatment_branch_identity(fields["treatment"], self.vocabulary)
            )
            cutoff = _parse_datetime(fields["anchor"]["cutoff_at"], "anchor.cutoff_at")
            age_seconds = max(0.0, (learned_at - cutoff).total_seconds())
            weight = float(fields["quality"]["source_confidence"]) * math.exp(
                -age_seconds / (self.config.decay_tau_days * 86400.0)
            )
            outcome = fields["label"]["outcome"]
            category = (str(outcome["outcome_type"]), str(outcome["valence"]))
            group_id = str(fields["lineage"]["consequence_group_id"])
            bucket = buckets.setdefault((domain, branch_key), {})
            entry = bucket.setdefault(
                category,
                {"group_weights": {}, "delays": [], "semantics_weights": {}},
            )
            entry["group_weights"][group_id] = entry["group_weights"].get(group_id, 0.0) + weight
            delay = outcome["delay_seconds"]
            if delay is not None:
                entry["delays"].append(float(delay))
            semantics = str(outcome["semantics"])
            entry["semantics_weights"][semantics] = (
                entry["semantics_weights"].get(semantics, 0.0) + weight
            )
        index: dict[tuple[str, str], tuple[dict[str, Any], ...]] = {}
        for key, categories in buckets.items():
            counts = {
                category: self._capped_weight(entry["group_weights"])
                for category, entry in categories.items()
            }
            denominator = self.config.strength + sum(counts.values())
            entries: list[dict[str, Any]] = []
            for category in sorted(categories):
                entry = categories[category]
                probability = max(0.0, counts[category] - self.config.discount) / denominator
                if probability <= 0:
                    continue
                outcome_type, valence = category
                delays = sorted(entry["delays"])
                entries.append(
                    {
                        "semantics": _weighted_mode(
                            entry["semantics_weights"], f"{outcome_type}({valence})"
                        ),
                        "probability": probability,
                        "average_delay_seconds": _median(delays),
                        "valence": valence,
                    }
                )
            entries.sort(key=lambda item: (-item["probability"], item["semantics"]))
            index[key] = tuple(entries)
        return index

    @staticmethod
    def _accumulate(
        states: dict[tuple[str, str, str], _StateAccumulator],
        key: tuple[str, str, str],
        observation: _Observation,
        *,
        context: dict[str, Any] | None,
        bucket: str | None = None,
        weight_scale: float = 1.0,
        record: bool = True,
    ) -> None:
        accumulator = states.get(key)
        if accumulator is None:
            accumulator = _StateAccumulator(observation.level, observation.domain, context, bucket)
            states[key] = accumulator
        accumulator.add(observation, weight_scale=weight_scale, record=record)

    def _temporal_fractions(self, cutoff: datetime) -> tuple[tuple[str, float, bool], ...]:
        """按 von Mises 核把一个时刻的权重摊到各时段桶；圆周上无边界断裂。

        23:50 与 00:10 是邻居，直方分桶会人为切断；核平滑让每条样本按
        与桶中心的圆周距离分数计入相邻桶，主桶（时刻所在桶）负责记账。
        """

        hours = self.config.temporal_bucket_hours
        offset = timezone(timedelta(minutes=self.config.temporal_utc_offset_minutes))
        local = cutoff.astimezone(offset)
        theta = 2 * math.pi * (local.hour + local.minute / 60 + local.second / 3600) / 24
        concentration = self.config.temporal_kernel_concentration
        bucket_count = 24 // hours
        raw = [
            math.exp(
                concentration
                * (math.cos(theta - 2 * math.pi * ((index + 0.5) * hours) / 24) - 1.0)
            )
            for index in range(bucket_count)
        ]
        total = sum(raw)
        primary_index = local.hour // hours
        return tuple(
            (
                f"{index * hours:02d}-{(index + 1) * hours:02d}",
                raw[index] / total,
                index == primary_index,
            )
            for index in range(bucket_count)
        )

    def _observation(
        self,
        uri: str,
        document: PredictionDocument,
        learned_at: datetime,
    ) -> _Observation:
        fields = document.fields
        anchor = fields["anchor"]
        level = str(anchor["anchor_type"])
        if level not in _LEVELS:
            raise PredictionLearningError("transition anchors must be at action or event level")
        cutoff = _parse_datetime(anchor["cutoff_at"], "anchor.cutoff_at")
        age_seconds = max(0.0, (learned_at - cutoff).total_seconds())
        weight = float(fields["quality"]["source_confidence"]) * math.exp(
            -age_seconds / (self.config.decay_tau_days * 86400.0)
        )
        supervision = fields["supervision"]
        labeled = (
            not bool(supervision["censored"])
            and str(supervision["label_status"]) in _OBSERVED_LABEL_STATUSES
        )
        branch: dict[str, Any] | None = None
        raw_semantics: str | None = None
        raw_actor: str | None = None
        if labeled:
            label = fields["label"]
            raw_semantics = label["semantics"]
            raw_actor = label["actor"]
            branch = label_branch_identity(label, self.vocabulary)
        history = fields["input"]["behavior_history"]
        steps = history["completed_actions"] if level == "action" else history["completed_events"]
        context = previous_step_key(steps, self.vocabulary)
        return _Observation(
            uri=uri,
            level=level,
            domain=self.vocabulary.canonical_token(
                fields["prediction_scope"]["target_domain"], "prediction_scope.target_domain"
            ),
            context=context,
            branch=branch,
            raw_semantics=raw_semantics,
            raw_actor=raw_actor,
            weight=weight,
            cutoff=cutoff,
            group_id=str(fields["lineage"]["occurrence_group_id"]),
        )

    def _state_identity(self, accumulator: _StateAccumulator) -> dict[str, Any]:
        if accumulator.bucket is not None:
            return temporal_state_identity(accumulator.level, accumulator.domain, accumulator.bucket)
        return sequence_state_identity(accumulator.level, accumulator.domain, accumulator.context)

    def _pseudo_counts(self, accumulator: _StateAccumulator) -> Mapping[str, float]:
        if self.prior is None:
            return {}
        return self.prior.pseudo_counts(logical_state_key(self._state_identity(accumulator)))

    def _distribution(
        self,
        accumulator: _StateAccumulator,
        *,
        parent: Mapping[str, float] | None,
        pseudo: Mapping[str, float],
    ) -> dict[str, float]:
        """Pitman–Yor 插值：局部折扣计数加上按 escape 系数继承的父层概率。

        escape 携带的质量 = 强度 + 各分支实际被折扣走的量（min(count, d)，
        而不是理想 CRP 的 d×T）+ 删失与未见伪计数曝光；这保证在质量加权、
        时间衰减、组封顶产生的分数计数下（单分支计数可小于折扣 d），
        Σ local + escape ≡ 1 恒成立，于是每状态出边概率和严格小于一，
        任何单分支概率不越界。``pseudo`` 是语义先验的伪计数：观测到的
        分支直接加进计数，未观测分支的伪计数只进曝光量——它们没有证据
        不能物化，其先验质量以留白形式存在，等真实观测出现后立即转化。
        """

        discount = self.config.discount
        strength = self.config.strength
        counts = {
            key: self._capped_weight(branch.group_weights) + pseudo.get(key, 0.0)
            for key, branch in accumulator.branches.items()
        }
        unseen_pseudo = sum(
            value for key, value in pseudo.items() if key not in accumulator.branches
        )
        censored = self._capped_weight(accumulator.censored_group_weights)
        exposure = sum(counts.values()) + censored + unseen_pseudo
        denominator = strength + exposure
        discounted = sum(min(count, discount) for count in counts.values())
        escape = (strength + discounted + censored + unseen_pseudo) / denominator
        result: dict[str, float] = {}
        for key, count in counts.items():
            local = max(0.0, count - discount) / denominator
            inherited = escape * parent.get(key, 0.0) if parent is not None else 0.0
            result[key] = local + inherited
        return result

    def _capped_weight(self, group_weights: Mapping[str, float]) -> float:
        cap = self.config.group_weight_cap
        return sum(min(cap, weight) for weight in group_weights.values())

    def _materialize_state(
        self,
        accumulator: _StateAccumulator,
        probabilities: Mapping[str, float],
        generation_id: str,
        projection_version: str,
        outcome_index: Mapping[tuple[str, str], tuple[dict[str, Any], ...]],
    ) -> list[PredictionPatternDocument]:
        assert accumulator.first_observed_at is not None
        assert accumulator.last_observed_at is not None
        predicates: list[dict[str, Any]] = [
            {"field": "anchor_type", "operator": "eq", "value": accumulator.level},
            {"field": "target_domain", "operator": "eq", "value": accumulator.domain},
        ]
        identity = self._state_identity(accumulator)
        if accumulator.bucket is not None:
            predicates.append(
                {"field": "hour_bucket", "operator": "eq", "value": accumulator.bucket}
            )
        elif accumulator.context is not None:
            predicates.append(
                {"field": "previous_step_key", "operator": "eq", "value": accumulator.context}
            )
        state = self.factory.build_state(
            pattern_generation=generation_id,
            identity=identity,
            fields={
                "semantic_summary": self._state_summary(accumulator),
                "state_level": accumulator.level,
                "predicates": predicates,
                "active_goals": (),
                "constraints": (),
                "support_count": accumulator.raw_count,
                "sample_refs": self._bounded_refs(accumulator.references),
                "statistics": {
                    "first_observed_at": accumulator.first_observed_at,
                    "last_observed_at": accumulator.last_observed_at,
                    "source_count": len(accumulator.groups),
                    "confidence": self._shrunk_confidence(accumulator.raw_count),
                },
                "projection_version": projection_version,
            },
        )
        documents = [state]
        for key in sorted(accumulator.branches):
            branch = accumulator.branches[key]
            if branch.raw_count == 0:
                continue
            identity = branch.identity
            if identity["target_kind"] == "termination":
                behavior_type = "termination"
                target_refs: list[str] = []
                fallback_semantics = f"终止（{identity['status']}）"
            else:
                behavior_type = str(identity["behavior_type"])
                target_refs = list(identity["target_refs"])
                fallback_semantics = str(identity["semantics"])
            documents.append(
                self.factory.build_branch(
                    state=state,
                    identity={"target": identity},
                    fields={
                        "target_kind": str(identity["target_kind"]),
                        "behavior": {
                            "actor_role": _weighted_mode(branch.actor_weights, "unspecified"),
                            "behavior_type": behavior_type,
                            "semantics": _weighted_mode(branch.semantics_weights, fallback_semantics),
                            "target_refs": target_refs,
                            "parameters": {},
                        },
                        "conditions": predicates,
                        "support_count": branch.raw_count,
                        "conditional_probability": probabilities[key],
                        "confidence": self._shrunk_confidence(branch.raw_count),
                        "sample_refs": self._bounded_refs(branch.references),
                        "outcomes": outcome_index.get((accumulator.domain, key), ()),
                        "projection_version": projection_version,
                    },
                )
            )
        return documents

    def _state_summary(self, accumulator: _StateAccumulator) -> str:
        level_name = _LEVEL_NAMES[accumulator.level]
        if accumulator.bucket is not None:
            return f"{accumulator.domain} 域{level_name}层 {accumulator.bucket} 时段的下一步分布"
        if accumulator.context is None:
            return f"{accumulator.domain} 域{level_name}层任意上下文的下一步基线分布"
        return (
            f"{accumulator.domain} 域{level_name}层上一步"
            f"「{accumulator.context['semantics']}」之后的下一步分布"
        )

    def _bounded_refs(self, references: Sequence[tuple[datetime, str]]) -> tuple[str, ...]:
        ordered = sorted(references, key=lambda item: item[1])
        ordered.sort(key=lambda item: item[0], reverse=True)
        return tuple(item[1] for item in ordered[: self.config.max_sample_refs])

    def _shrunk_confidence(self, raw_count: int) -> float:
        return raw_count / (raw_count + self.config.confidence_prior)


def _weighted_mode(weights: Mapping[str, float], fallback: str) -> str:
    if not weights:
        return fallback
    return min(weights.items(), key=lambda item: (-item[1], item[0]))[0]


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def _parse_datetime(value: object, label: str) -> datetime:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise PredictionLearningError(f"{label} must be an ISO timestamp") from exc
    if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PredictionLearningError(f"{label} must be timezone-aware")
    return parsed


__all__ = ["PredictionPatternLearner"]
