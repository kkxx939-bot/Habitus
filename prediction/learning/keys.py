"""学习期与预测期共享的确定性聚合键派生。

学习器和预测器必须用同一版词表、同一套键派生规则；否则线上上下文
永远命不中线下学出来的状态。这里是序列上下文键与状态身份的唯一出处。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from foundation.integrity import canonical_digest
from prediction.learning.vocabulary import PredictionBehaviorVocabulary

STATE_FAMILY = "sequence-context"
TEMPORAL_FAMILY = "temporal"
_LOGICAL_STATE_CONTRACT = "prediction-logical-state-pattern-v1"


def select_last_step(steps: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """选出历史里的最后一步；这里是"刚发生的行为"的唯一裁定规则。

    检索查询与状态匹配必须描述同一步，否则并列步会让两侧各选各的。
    """

    if not steps:
        return None
    return max(steps, key=_step_order)


def previous_step_key(
    steps: Sequence[Mapping[str, Any]],
    vocabulary: PredictionBehaviorVocabulary,
) -> dict[str, Any] | None:
    """从已完成历史步骤派生上一步的规范上下文键；无历史时返回 None。"""

    last = select_last_step(steps)
    if last is None:
        return None
    base_type = last["behavior_type"] if last["behavior_type"] is not None else last["semantics"]
    return {
        "behavior_type": vocabulary.canonical_token(base_type, "history behavior_type"),
        "semantics": vocabulary.canonical_token(last["semantics"], "history semantics"),
        "target_refs": sorted(
            {vocabulary.canonical_token(item, "history target_ref") for item in last["target_refs"]}
        ),
    }


def sequence_state_identity(
    level: str,
    domain: str,
    context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """构造序列族状态的逻辑身份材料；context 为 None 表示同域同层根状态。

    ``domain`` 来自样本的 ``prediction_scope.target_domain``：coding agent
    的开发动作序列与 VLM/ASR 感知的人类日常行为共用同一套算法，但必须
    各自成态，否则两个场景的计数会互相污染对方的回退基线。
    """

    return {
        "family": STATE_FAMILY,
        "state_level": level,
        "target_domain": domain,
        "context": None if context is None else dict(context),
    }


def logical_state_key(identity: Mapping[str, Any]) -> str:
    """按 PredictionPatternFactory 的同一契约派生 logical_state_key。"""

    return canonical_digest({"contract": _LOGICAL_STATE_CONTRACT, "identity": identity})


def temporal_bucket(moment: datetime, *, bucket_hours: int, utc_offset_minutes: int) -> str:
    """把一个时刻映射到本地时段桶标签，如 ``06-09``。

    样本时间以 UTC 存储，行为的昼夜语义在本地时间上；偏移由配置给出并
    进入 generation 身份，学习期与预测期必须使用同一组时段参数。
    """

    if not isinstance(moment, datetime) or moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("temporal bucket moment must be a timezone-aware datetime")
    local = moment.astimezone(timezone(timedelta(minutes=utc_offset_minutes)))
    start = (local.hour // bucket_hours) * bucket_hours
    return f"{start:02d}-{start + bucket_hours:02d}"


def temporal_state_identity(level: str, domain: str, bucket: str) -> dict[str, Any]:
    """构造时间族状态的逻辑身份材料。"""

    return {
        "family": TEMPORAL_FAMILY,
        "state_level": level,
        "target_domain": domain,
        "hour_bucket": bucket,
    }


def target_kind_for_level(target_level: str) -> str:
    """预测目标层级到分支 target_kind 的唯一映射(terminal→termination)。

    学习器物化、预测器过滤与回测记分必须用同一映射,否则 terminal 层
    查询会因键域不匹配恒空过滤、以 LOW_CONFIDENCE 掩盖缺陷。
    """

    return "termination" if target_level == "terminal" else target_level


def label_branch_identity(
    label: Mapping[str, Any],
    vocabulary: PredictionBehaviorVocabulary,
) -> dict[str, Any]:
    """从 TransitionSample 标签派生分支的规范身份键。

    学习器聚合与回测评估必须用同一个键，否则评估会把命中判成未命中。
    """

    if label["target_kind"] == "terminal":
        return {
            "target_kind": "termination",
            "status": vocabulary.canonical_token(label["terminal"]["status"], "label terminal status"),
        }
    return {
        "target_kind": str(label["target_kind"]),
        "behavior_type": vocabulary.canonical_token(label["behavior_type"], "label behavior_type"),
        "semantics": vocabulary.canonical_token(label["semantics"], "label semantics"),
        "target_refs": sorted(
            {vocabulary.canonical_token(item, "label target_ref") for item in label["target_refs"]}
        ),
    }


def treatment_branch_identity(
    treatment: Mapping[str, Any],
    vocabulary: PredictionBehaviorVocabulary,
) -> dict[str, Any]:
    """从 ConsequenceSample 的 treatment 派生它作证的分支身份键。

    结果分布要挂到"做了什么行为"上，treatment 与 Transition 标签必须
    归一到同一个键，否则后果证据永远对不上分支。
    """

    base_type = (
        treatment["behavior_type"] if treatment["behavior_type"] is not None else treatment["semantics"]
    )
    return {
        "target_kind": str(treatment["step_kind"]),
        "behavior_type": vocabulary.canonical_token(base_type, "treatment behavior_type"),
        "semantics": vocabulary.canonical_token(treatment["semantics"], "treatment semantics"),
        "target_refs": sorted(
            {vocabulary.canonical_token(item, "treatment target_ref") for item in treatment["target_refs"]}
        ),
    }


def _step_order(step: Mapping[str, Any]) -> tuple[int, str, str]:
    sequence = step["sequence"]
    available_at = step["available_at"]
    return (
        sequence if isinstance(sequence, int) and not isinstance(sequence, bool) else 0,
        available_at if isinstance(available_at, str) else "",
        str(step["step_ref"]),
    )


__all__ = [
    "STATE_FAMILY",
    "TEMPORAL_FAMILY",
    "label_branch_identity",
    "logical_state_key",
    "previous_step_key",
    "select_last_step",
    "sequence_state_identity",
    "target_kind_for_level",
    "temporal_bucket",
    "temporal_state_identity",
    "treatment_branch_identity",
]
