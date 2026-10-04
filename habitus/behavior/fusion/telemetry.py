"""融合各步骤的可观测属性：从各步自己的产物算计数与时效，不读存储、不碰模型。

融合是行为管线里唯一由模型做语义判断的一层，所以这里要能看出模型交回来的东西被确定性装配
改动了多少（降级、截断、被剪掉的关系），以及证据从可用到成为判断等了多久。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from habitus.behavior.fusion.judgement import JudgementRelation
from habitus.behavior.telemetry import Attributes, count_notes_by_leading_token, seconds_between

if TYPE_CHECKING:
    from habitus.behavior.fusion.derivation import DurableJudgement
    from habitus.behavior.fusion.enqueue import BehaviorFusionEnqueueResult
    from habitus.behavior.fusion.receipt import BehaviorFusionReceipt
    from habitus.behavior.fusion.result import BehaviorFusionResult
    from habitus.behavior.observation import BehaviorObservation


def enqueue_attributes(result: BehaviorFusionEnqueueResult) -> Attributes:
    return {
        "enqueued": result.count,
        "segments": result.segments,
        "withheld_segments": result.withheld_segments,
        "withheld_observations": result.withheld_observations,
        "covered_observations": result.covered_observations,
        "envelopes": result.envelopes,
    }


def context_attributes(context: Sequence[Mapping[str, Any]]) -> Attributes:
    """带给模型的先前判断：跨窗口指回只能指向这里出现过的条目。"""

    return {
        "context_judgements": len(context),
        "context_open_judgements": sum(1 for item in context if item.get("status") != "completed"),
    }


def judgement_attributes(result: BehaviorFusionResult, *, truncated: bool) -> Attributes:
    """模型交回的判断经装配之后的样子；降级说明按类别计数。"""

    batch = result.batch
    relations = Counter(link.kind for item in batch.judgements for link in item.relations)
    attributes: Attributes = {
        "fragments": len(result.segment.fragments),
        "segment_span_seconds": round(result.segment.span_seconds, 3),
        "truncated": truncated,
        "validation_attempts": result.validation_attempts,
        "judgements": len(batch.judgements),
        "readable_judgements": len(batch.readable),
        "unreadable_fragments": batch.unreadable_fragment_count,
        "unowned_fragments": len(batch.unowned_fragment_nos),
        "cross_window_relations": sum(
            1 for item in batch.judgements for link in item.relations if link.is_cross_window
        ),
        "degradations": len(batch.degradations),
    }
    for kind in JudgementRelation:
        attributes[f"relations_{kind.value}"] = relations.get(kind, 0)
    attributes.update(count_notes_by_leading_token(batch.degradations, prefix="degradation_"))
    return attributes


def staging_attributes(
    derived: Sequence[DurableJudgement],
    in_scope: Sequence[DurableJudgement],
    staged_judgements: Sequence[Mapping[str, Any]],
    fragments: Sequence[BehaviorObservation],
    *,
    judged_at: datetime,
) -> Attributes:
    """主体分流与时效。

    ``fusion_lag`` 是证据齐备到判断成立；``evidence_wait`` 从本段最早可用的观测算起，包含静默期
    与排队——串行队列被队首卡住时，先涨的是它。
    """

    kept_relations = sum(len(item["relations"]) for item in staged_judgements)
    attributes: Attributes = {
        "judgements_in_scope": len(in_scope),
        "judgements_out_of_scope": len(derived) - len(in_scope),
        "relations_pruned": sum(len(item.relations) for item in in_scope) - kept_relations,
        "evidence_wait_seconds_max": seconds_between(judged_at, min(item.available_at for item in fragments)),
    }
    if derived:
        attributes["fusion_lag_seconds_max"] = round(max(max(0.0, item.fusion_lag_seconds) for item in derived), 3)
    return attributes


def job_attributes(receipt: BehaviorFusionReceipt, *, fused: bool, attempt: int) -> Attributes:
    """一次作业的终态；重放与首次执行都从回执读，口径一致。"""

    return {
        "job": receipt.receipt_id[:12],
        "fused": fused,
        "attempt": attempt,
        "observations": len(receipt.observation_ids),
        "source_deliveries": len(receipt.source_refs),
        "judgements_persisted": len(receipt.judgement_ids),
        "unreadable_observations": len(receipt.unreadable_observation_ids),
        "out_of_scope_observations": len(receipt.out_of_scope_observation_ids),
        "unowned_observations": len(receipt.unowned_observation_ids),
        "validation_attempts": receipt.validation_attempts,
    }


__all__ = [
    "context_attributes",
    "enqueue_attributes",
    "job_attributes",
    "judgement_attributes",
    "staging_attributes",
]
