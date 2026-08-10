"""行为词表别名归并的清单提取与统计守门。

同义表达分裂聚合键会把有效样本量除以同义词数，归并因此是准确率的
第一杠杆；但把两个"同名不同义"的行为词错误合并，会直接污染条件分布，
比不合并更糟。本模块提供确定性的两半：从样本提取行为词清单（供离线
LLM 或人工产出合并提议），再用后继分布的 Jensen–Shannon 散度守门每条
提议——同分布才可合并，异分布宁可保留分裂。LLM 只出现在提议生产侧
（组合根离线批处理），本模块与学习聚合保持零 LLM、可重放。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from foundation.integrity import canonical_json
from prediction.document import PredictionDocument
from prediction.learning.config import PredictionLearningError
from prediction.learning.keys import label_branch_identity, previous_step_key
from prediction.learning.vocabulary import PredictionBehaviorVocabulary
from prediction.model import PredictionKind

_OBSERVED_LABEL_STATUSES = frozenset({"observed", "terminal"})


@dataclass(frozen=True)
class PredictionTokenUsage:
    """一个规范行为词在样本集中的用量与后继分布。"""

    token: str
    label_count: int
    context_count: int
    next_distribution: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class PredictionVocabularyMergeConfig:
    """合并守门的支持度门槛与散度上限。"""

    min_context_support: int = 3
    max_divergence_bits: float = 0.3

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_context_support, bool)
            or not isinstance(self.min_context_support, int)
            or self.min_context_support < 1
        ):
            raise PredictionLearningError("min_context_support must be a positive integer")
        if (
            isinstance(self.max_divergence_bits, bool)
            or not isinstance(self.max_divergence_bits, (int, float))
            or not 0 < float(self.max_divergence_bits) <= 1
        ):
            raise PredictionLearningError("max_divergence_bits must be within (0, 1]")
        object.__setattr__(self, "max_divergence_bits", float(self.max_divergence_bits))


@dataclass(frozen=True)
class PredictionVocabularyMergeReport:
    """一次合并校验的完整结果：可用词表加逐条接受/拒绝记录。"""

    vocabulary: PredictionBehaviorVocabulary
    accepted: tuple[tuple[str, str], ...]
    rejected: tuple[tuple[str, str, str], ...]


def behavior_token_inventory(
    samples: Sequence[PredictionDocument],
    *,
    vocabulary: PredictionBehaviorVocabulary | None = None,
) -> tuple[PredictionTokenUsage, ...]:
    """提取样本中的规范行为词清单，供离线归并提议（LLM 或人工）使用。

    每个词带标签用量、上下文用量与作为上下文时的后继分布；后继分布
    同时是合并守门的判据来源。终止标签没有可归并的行为词，不入清单。
    """

    base = vocabulary or PredictionBehaviorVocabulary()
    if not isinstance(base, PredictionBehaviorVocabulary):
        raise PredictionLearningError("vocabulary must be PredictionBehaviorVocabulary")
    label_counts: dict[str, int] = {}
    context_counts: dict[str, int] = {}
    successors: dict[str, dict[str, int]] = {}
    for document in _documents(samples):
        fields = document.fields
        supervision = fields["supervision"]
        labeled = (
            not bool(supervision["censored"])
            and str(supervision["label_status"]) in _OBSERVED_LABEL_STATUSES
        )
        label_key: str | None = None
        if labeled:
            identity = label_branch_identity(fields["label"], base)
            label_key = canonical_json(identity)
            if identity["target_kind"] != "termination":
                token = str(identity["semantics"])
                label_counts[token] = label_counts.get(token, 0) + 1
        level = str(fields["anchor"]["anchor_type"])
        history = fields["input"]["behavior_history"]
        steps = history["completed_actions"] if level == "action" else history["completed_events"]
        context = previous_step_key(steps, base)
        if context is not None:
            token = str(context["semantics"])
            context_counts[token] = context_counts.get(token, 0) + 1
            if label_key is not None:
                bucket = successors.setdefault(token, {})
                bucket[label_key] = bucket.get(label_key, 0) + 1
    tokens = sorted(set(label_counts) | set(context_counts))
    return tuple(
        PredictionTokenUsage(
            token=token,
            label_count=label_counts.get(token, 0),
            context_count=context_counts.get(token, 0),
            next_distribution=tuple(sorted(successors.get(token, {}).items())),
        )
        for token in tokens
    )


def behavior_branch_catalog(
    samples: Sequence[PredictionDocument],
    *,
    vocabulary: PredictionBehaviorVocabulary | None = None,
) -> dict[str, dict[str, Any]]:
    """按展示语义列出样本中出现过的完整分支身份，供先验蒸馏提示使用。

    LLM 只能对"已知的行为"给份额，不能发明行为；同名语义冲突时保留
    排序最先的身份，保证目录确定性。终止标签不参与先验。
    """

    base = vocabulary or PredictionBehaviorVocabulary()
    catalog: dict[str, dict[str, Any]] = {}
    for document in _documents(samples):
        fields = document.fields
        supervision = fields["supervision"]
        if bool(supervision["censored"]) or str(supervision["label_status"]) not in _OBSERVED_LABEL_STATUSES:
            continue
        identity = label_branch_identity(fields["label"], base)
        if identity["target_kind"] == "termination":
            continue
        token = str(identity["semantics"])
        if token not in catalog or canonical_json(identity) < canonical_json(catalog[token]):
            catalog[token] = identity
    return {token: catalog[token] for token in sorted(catalog)}


def validate_merge_proposals(
    samples: Sequence[PredictionDocument],
    proposals: Mapping[str, str],
    *,
    config: PredictionVocabularyMergeConfig | None = None,
) -> PredictionVocabularyMergeReport:
    """用后继分布守门合并提议，产出可直接用于学习器的版本化词表。

    两个词都以足够支持度作为上下文出现时，其后继分布的 JS 散度超过
    上限即拒绝该条提议——数据证明它们引出不同的下一步，就不是同义词。
    守门在"应用全部提议后的合并键空间"上计算：后继标签先经候选词表
    归一（配套提议的同义后继不被误计为散度），被检验的上下文词本身
    保持原词区分。数据不足以反驳时接受语义提议（提议方的判断成立）。
    结构性问题（别名链、冲突）不属于可拒绝项，直接失败。
    """

    resolved = config or PredictionVocabularyMergeConfig()
    if not isinstance(resolved, PredictionVocabularyMergeConfig):
        raise PredictionLearningError("config must be PredictionVocabularyMergeConfig")
    if not isinstance(proposals, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in proposals.items()
    ):
        raise PredictionLearningError("proposals must map source tokens to canonical tokens")
    candidate = PredictionBehaviorVocabulary(aliases=dict(proposals))
    successors, context_counts = _raw_context_successors(samples, candidate)
    accepted: dict[str, str] = {}
    accepted_pairs: list[tuple[str, str]] = []
    rejected: list[tuple[str, str, str]] = []
    for source in sorted(proposals):
        target = proposals[source]
        if (
            context_counts.get(source, 0) >= resolved.min_context_support
            and context_counts.get(target, 0) >= resolved.min_context_support
        ):
            divergence = _jensen_shannon_bits(
                successors.get(source, {}),
                successors.get(target, {}),
            )
            if divergence > resolved.max_divergence_bits:
                rejected.append(
                    (
                        source,
                        target,
                        f"successor distributions diverge by {divergence:.3f} bits",
                    )
                )
                continue
        accepted[source] = target
        accepted_pairs.append((source, target))
    return PredictionVocabularyMergeReport(
        vocabulary=PredictionBehaviorVocabulary(aliases=accepted),
        accepted=tuple(accepted_pairs),
        rejected=tuple(rejected),
    )


def _raw_context_successors(
    samples: Sequence[PredictionDocument],
    candidate: PredictionBehaviorVocabulary,
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """上下文词保持原词、后继标签经候选词表归一的后继计数。"""

    base = PredictionBehaviorVocabulary()
    successors: dict[str, dict[str, int]] = {}
    context_counts: dict[str, int] = {}
    for document in _documents(samples):
        fields = document.fields
        level = str(fields["anchor"]["anchor_type"])
        history = fields["input"]["behavior_history"]
        steps = history["completed_actions"] if level == "action" else history["completed_events"]
        context = previous_step_key(steps, base)
        if context is None:
            continue
        token = str(context["semantics"])
        context_counts[token] = context_counts.get(token, 0) + 1
        supervision = fields["supervision"]
        if bool(supervision["censored"]) or str(supervision["label_status"]) not in _OBSERVED_LABEL_STATUSES:
            continue
        label_key = canonical_json(label_branch_identity(fields["label"], candidate))
        bucket = successors.setdefault(token, {})
        bucket[label_key] = bucket.get(label_key, 0) + 1
    return successors, context_counts


def _jensen_shannon_bits(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if left_total == 0 or right_total == 0:
        return 0.0
    divergence = 0.0
    for key in set(left) | set(right):
        p = left.get(key, 0) / left_total
        q = right.get(key, 0) / right_total
        mixture = (p + q) / 2
        if p > 0:
            divergence += 0.5 * p * math.log2(p / mixture)
        if q > 0:
            divergence += 0.5 * q * math.log2(q / mixture)
    return divergence


def _documents(samples: Sequence[PredictionDocument]) -> tuple[PredictionDocument, ...]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise PredictionLearningError("samples must be a sequence of PredictionDocument values")
    documents: list[PredictionDocument] = []
    for document in samples:
        if not isinstance(document, PredictionDocument):
            raise PredictionLearningError("samples must contain PredictionDocument values")
        if document.kind is not PredictionKind.TRANSITION:
            raise PredictionLearningError("vocabulary merging only covers TransitionSamples")
        documents.append(document)
    return tuple(documents)


__all__ = [
    "PredictionTokenUsage",
    "PredictionVocabularyMergeConfig",
    "PredictionVocabularyMergeReport",
    "behavior_branch_catalog",
    "behavior_token_inventory",
    "validate_merge_proposals",
]
