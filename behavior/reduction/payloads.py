"""把闭合的链与没读懂段物化成行为树 payload。

纯函数、零发明：每个字段的取值规则都是 ``TODO(BHV-TREE-REBUILD-001)`` 的既定裁定——身份看
链头、结局看链尾、过程全保留；时间在这里做**唯一一次**换算（本地+偏移进树，树内与读侧零换算）。
产出为 JSON 可序列化的字典（时间全部是 ISO 字符串）：同一份 payload 既直接过 schema 校验，也
能逐字节写进 staged 检查点——崩溃重试逐字节重放的前提。

已定但值得点名的两条取数规则：

- ``goal`` 与 summary 同一条纪律：**链内非空值按序去重拼接**（用户裁定：原封不动保留，语义
  关联层靠它做判断，不许替模型二选一或置空丢失）。各段一致时去重后就是那一个；前段判不出、
  后段判出时自然只剩后段的；前后段声明了不同目标时两个都在（"清理餐桌；准备待客"）——分歧
  本身就是语义。goal 全空则 basis 必为空（融合层不变量），三层递减在树上保持。
- basis 每步的时间从观测物化：起止取覆盖观测 ``occurred_at`` 的最小/最大（按各自的本地偏移
  还原），``available_at`` 取覆盖观测的最大值、用末观测的偏移表达——判断存储会释放，这些时间
  必须在此自足。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from behavior.model import BehaviorAddress
from behavior.observation import BehaviorObservation
from behavior.reduction.chains import BehaviorChain
from behavior.reduction.errors import BehaviorReductionError
from behavior.reduction.record import ReducibleJudgement

REDUCTION_VERSION = "behavior_reduction_v1"

UNREADABLE_GAP_KIND = "没读懂"


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="microseconds")


def chain_address(chain: BehaviorChain, name: str) -> BehaviorAddress:
    """链在树上的地址；撞车消歧时以后缀名重算。"""

    started_at = chain.head.started_at
    return BehaviorAddress.occurrence(started_at.date(), name, started_at)


def occurrence_payload(
    chain: BehaviorChain,
    *,
    name: str,
    original_name: str | None,
    kind_token: str,
    observations: Mapping[str, BehaviorObservation],
) -> dict[str, Any]:
    """一条闭合链的 occurrence payload。``name`` 已含可能的消歧后缀。"""

    head, tail = chain.head, chain.tail
    if head.behavior is None or tail.status is None or tail.status_basis is None:
        raise BehaviorReductionError("an occurrence chain must be built from readable judgements")
    summaries = [member.summary for member in chain.view]
    if any(item is None for item in summaries):
        raise BehaviorReductionError("an occurrence chain must be built from readable judgements")
    last_observed = max((item.last_observed_at for item in chain.view), key=_as_instant)
    # "何时可知"存储为 UTC；进树换算成链头行为时刻的本地偏移——occurrence 上只有一种时间约定。
    onset = head.evidence_ready_at.astimezone(head.started_at.tzinfo)
    consumed = chain.consumed
    basis = [
        _basis_step(semantics, observation_ids, observations)
        for member in chain.view
        for semantics, observation_ids in member.basis
    ]
    distinct_goals = {item.goal: None for item in chain.view if item.goal is not None}
    goal = "；".join(distinct_goals) if distinct_goals else None
    return {
        "occurred_on": head.started_at.date().isoformat(),
        "name": name,
        "started_at": _iso(head.started_at),
        "kind_token": kind_token,
        "status": tail.status,
        "status_basis": tail.status_basis,
        "last_observed_at": _iso(last_observed),
        "onset_available_at": _iso(onset),
        # 干预账本（预测执行层）尚未存在——契约缺口：接入后此处换成"开始前窗口内是否提醒过"
        # 的账本查询，且必须在 stage 前定格（死规则⑤）。
        "reminded": False,
        "goal": goal,
        "summary": "；".join(item for item in summaries if item is not None),
        "subjects": _merged_subjects(chain),
        # 上游契约缺口：place 等观测侧提供后由此填入。
        "place": None,
        "original_name": original_name,
        "basis": basis,
        "judgement_ids": [item.judgement_id for item in consumed],
        "observation_ids": sorted({oid for item in consumed for oid in item.observation_ids}),
        "source_refs": sorted({ref for item in consumed for ref in item.source_refs}),
        "fusion_version": head.fusion_version,
        "reduction_version": REDUCTION_VERSION,
    }


def gap_payload(record: ReducibleJudgement) -> dict[str, Any]:
    """一段没读懂的观测空白；字段逐字来自那条空判断，起止同刻是合法的单观测段。"""

    if record.is_readable:
        raise BehaviorReductionError("a gap payload must be built from an unreadable judgement")
    return {
        "occurred_on": record.started_at.date().isoformat(),
        "gap_kind": UNREADABLE_GAP_KIND,
        "started_at": _iso(record.started_at),
        "ended_at": _iso(record.last_observed_at),
        "judgement_ids": [record.judgement_id],
        "observation_ids": sorted(set(record.observation_ids)),
        "reduction_version": REDUCTION_VERSION,
    }


def _merged_subjects(chain: BehaviorChain) -> list[str]:
    """链内主体的并集，按首次出现的顺序——共同行动的人都保住，不重复。"""

    seen: dict[str, None] = {}
    for member in chain.view:
        for subject in member.subjects:
            seen.setdefault(subject, None)
    return list(seen)


def _basis_step(
    semantics: str,
    observation_ids: tuple[str, ...],
    observations: Mapping[str, BehaviorObservation],
) -> dict[str, Any]:
    resolved: list[BehaviorObservation] = []
    for observation_id in observation_ids:
        observation = observations.get(observation_id)
        if observation is None:
            raise BehaviorReductionError(
                f"basis step references an observation that is no longer stored: {observation_id}"
            )
        resolved.append(observation)
    first = min(resolved, key=lambda item: (item.occurred_at, item.observation_id))
    last = max(resolved, key=lambda item: (item.occurred_at, item.observation_id))
    available = max(item.available_at for item in resolved)
    last_offset = timezone(timedelta(minutes=last.utc_offset_minutes))
    return {
        "semantics": semantics,
        "observation_ids": list(observation_ids),
        "started_at": _iso(first.local_occurred_at),
        "ended_at": _iso(last.local_occurred_at),
        "available_at": _iso(available.astimezone(last_offset)),
    }


def _as_instant(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


__all__ = [
    "REDUCTION_VERSION",
    "UNREADABLE_GAP_KIND",
    "chain_address",
    "gap_payload",
    "occurrence_payload",
]
