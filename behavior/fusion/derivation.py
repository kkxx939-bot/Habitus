"""把判断绑定成耐久记录。

模型只出语义，系统出身份与时间——最容易被编造的字段因此没有编造的机会。

## 两个时间，用途不同

    judged_at          这次融合实际返回的时刻。**不可派生，必须存**。
                       用途：判断之间定序（谁修正谁）、系统真正能提醒的最早时刻。

    evidence_ready_at  证据齐备的时刻 = 所覆盖观测 ``available_at`` 的 **max**。
                       用途：**预测的可见性截断**。用它而不用 judged_at，是因为它只取决于
                       观测本身——用 judged_at 的话，融合队列堵三天，训练数据就以为我们三天后
                       才知道周一的事，那是把运维噪音喂给模型。

两者之差就是融合滞后，是个天然的运维指标。

## 行为时间保留本地偏移

``started_at`` / ``last_observed_at`` 取所覆盖观测的 ``local_occurred_at``，**带本地偏移**。
不能折 UTC：人的一天是本地日历日，东八区凌晨的行为折成 UTC 会掉到前一天，归约层按"天"做的
任何聚合都会错。``foundation.integrity.canonicalize`` 会折 UTC，所以这些字段的序列化必须单独
处理（见 ``store.py``）。

注意 ``last_observed_at`` 是"**最后一条支持它的观测**发生在什么时候"，不是"行为在什么时候
结束"——后者我们经常不知道，那正是 ``status_basis=observation_lost`` 要表达的事。

## 身份不含关系

``judgement_id`` 由"它说了什么、依据哪些观测、什么时候"派生，**不含 relation**。因为并行是
互指的：甲指乙、乙指甲，若关系进身份就成了循环依赖，谁也算不出来。关系是判断之间的连接，
不属于一条判断自身的身份。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from behavior.fusion.errors import BehaviorFusionError
from behavior.fusion.judgement import (
    BehaviorJudgement,
    BehaviorJudgementBatch,
    JudgementLink,
    JudgementStatus,
    JudgementStatusBasis,
)
from behavior.fusion.prompt import FUSION_PROMPT_VERSION
from behavior.observation import BehaviorObservation
from foundation.integrity import canonical_digest

FUSION_IMPLEMENTATION_VERSION = "behavior_judgement_fusion_v1"

# 产物的语义同时取决于派生实现与提示词，任一变化都会改变输出语义，因此版本必须同时覆盖两者；
# 只记实现版本会让提示词改动在数据上不可分辨。
FUSION_VERSION = f"{FUSION_IMPLEMENTATION_VERSION}+{FUSION_PROMPT_VERSION}"

_IDENTITY_SCHEMA = "behavior_judgement_identity_v1"


@dataclass(frozen=True)
class DurableFact:
    """一条行为事实，片段编号已换成观测的内容身份。"""

    semantics: str
    observation_ids: tuple[str, ...]


@dataclass(frozen=True)
class DurableJudgement:
    """一条可落盘的行为判断。"""

    judgement_id: str
    judgement_no: int
    judged_at: datetime
    evidence_ready_at: datetime
    started_at: datetime
    last_observed_at: datetime
    observation_ids: tuple[str, ...]
    source_refs: tuple[str, ...]
    subjects: tuple[str, ...]
    behavior: str | None
    goal: str | None
    summary: str | None
    basis: tuple[DurableFact, ...]
    status: JudgementStatus | None
    status_basis: JudgementStatusBasis | None
    relations: tuple[tuple[str, str], ...]
    fusion_version: str
    prompt_version: str

    @property
    def is_readable(self) -> bool:
        return self.behavior is not None

    @property
    def fusion_lag_seconds(self) -> float:
        """判断比证据晚了多久；运维指标，不参与任何语义。"""

        return (self.judged_at - self.evidence_ready_at).total_seconds()


def derive_judgements(
    batch: BehaviorJudgementBatch,
    fragments: Sequence[BehaviorObservation],
    *,
    source_refs: Sequence[str],
    judged_at: datetime,
    context_ids: Sequence[str] = (),
) -> tuple[DurableJudgement, ...]:
    """把一批判断绑定成耐久记录。

    调用前必须先执行 ``validate_judgement_batch``：这里假定引用完整、顺序正确，只做确定性绑定，
    不重复语义校验。``source_refs`` 是本次融合读取的来源身份，无法从片段自身推出，属于系统在模型
    输出之外绑定的事实。
    """

    if not isinstance(batch, BehaviorJudgementBatch):
        raise TypeError("batch must be BehaviorJudgementBatch")
    if not isinstance(judged_at, datetime) or judged_at.utcoffset() is None:
        raise BehaviorFusionError("judged_at must be a timezone-aware datetime")
    by_no = _fragment_index(fragments)
    resolved_sources = _refs(source_refs)
    if not resolved_sources:
        raise BehaviorFusionError("source_refs must not be empty")

    # 两遍：先算身份（身份不含关系，所以没有循环），再把关系指向具体的身份。
    identities = {
        item.judgement_no: _identity(item, by_no, judged_at=judged_at) for item in batch.judgements
    }
    context = tuple(context_ids)
    return tuple(
        _derive(
            item,
            by_no,
            source_refs=resolved_sources,
            judged_at=judged_at,
            judgement_id=identities[item.judgement_no],
            identities=identities,
            context=context,
        )
        for item in batch.judgements
    )


def _identity(
    judgement: BehaviorJudgement,
    by_no: Mapping[int, BehaviorObservation],
    *,
    judged_at: datetime,
) -> str:
    claim = judgement.claim
    return canonical_digest(
        {
            "schema_version": _IDENTITY_SCHEMA,
            "judged_at": judged_at,
            "fusion_version": FUSION_VERSION,
            "observation_ids": [by_no[no].observation_id for no in judgement.covers],
            "subjects": list(judgement.subjects),
            "behavior": claim.behavior,
            "goal": claim.goal,
            "summary": claim.summary,
            "basis": [
                {
                    "semantics": fact.semantics,
                    "observation_ids": [by_no[no].observation_id for no in fact.fragment_nos],
                }
                for fact in claim.basis
            ],
            "status": judgement.status,
            "status_basis": judgement.status_basis,
        }
    )


def _derive(
    judgement: BehaviorJudgement,
    by_no: Mapping[int, BehaviorObservation],
    *,
    source_refs: tuple[str, ...],
    judged_at: datetime,
    judgement_id: str,
    identities: Mapping[int, str],
    context: tuple[str, ...],
) -> DurableJudgement:
    owned = [by_no[no] for no in judgement.covers]
    claim = judgement.claim
    return DurableJudgement(
        judgement_id=judgement_id,
        judgement_no=judgement.judgement_no,
        judged_at=judged_at,
        evidence_ready_at=_evidence_ready_at(owned),
        started_at=_earliest_local(owned),
        last_observed_at=_latest_local(owned),
        observation_ids=tuple(item.observation_id for item in owned),
        source_refs=source_refs,
        subjects=judgement.subjects,
        behavior=claim.behavior,
        goal=claim.goal,
        summary=claim.summary,
        basis=tuple(
            DurableFact(
                semantics=fact.semantics,
                observation_ids=tuple(by_no[no].observation_id for no in fact.fragment_nos),
            )
            for fact in claim.basis
        ),
        status=judgement.status,
        status_basis=judgement.status_basis,
        relations=tuple(
            (link.kind.value, _target_identity(link, identities, context))
            for link in judgement.relations
        ),
        fusion_version=FUSION_VERSION,
        prompt_version=FUSION_PROMPT_VERSION,
    )


def _target_identity(
    link: JudgementLink, identities: Mapping[int, str], context: tuple[str, ...]
) -> str:
    """把关系的临时目标换成耐久身份；本批与上下文两个来源在这里合流。"""

    if link.context_no is not None:
        if link.context_no > len(context):
            raise BehaviorFusionError(
                f"relation references context judgement C{link.context_no}, "
                f"but only {len(context)} were supplied"
            )
        return context[link.context_no - 1]
    assert link.target_no is not None
    return identities[link.target_no]


def _fragment_index(
    fragments: Sequence[BehaviorObservation],
) -> Mapping[int, BehaviorObservation]:
    if isinstance(fragments, (str, bytes)) or not isinstance(fragments, Sequence) or not fragments:
        raise BehaviorFusionError("derivation requires a non-empty fragment sequence")
    if any(not isinstance(item, BehaviorObservation) for item in fragments):
        raise TypeError("fragments must contain BehaviorObservation values")
    return {index: fragment for index, fragment in enumerate(fragments, start=1)}


def _evidence_ready_at(observations: Sequence[BehaviorObservation]) -> datetime:
    """取 max：一条判断要等它全部证据到齐才成立。取 min 会把还不存在的信息标成已可用。"""

    return max(item.available_at for item in observations)


def _earliest_local(observations: Sequence[BehaviorObservation]) -> datetime:
    """最早发生时刻，按其自身的本地偏移还原——人的一天是本地日历日，不能折 UTC。"""

    return min(observations, key=lambda item: item.occurred_at).local_occurred_at


def _latest_local(observations: Sequence[BehaviorObservation]) -> datetime:
    """最后一条支持观测发生的时刻；这不等于行为结束的时刻。"""

    return max(observations, key=lambda item: item.occurred_at).local_occurred_at


def _refs(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise BehaviorFusionError("source_refs must be a sequence of text")
    resolved: list[str] = []
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise BehaviorFusionError("source_refs entries must be non-empty text")
        normalized = item.strip()
        if normalized not in resolved:
            resolved.append(normalized)
    return tuple(resolved)


def judgement_payload(judgement: DurableJudgement) -> dict[str, Any]:
    """耐久记录的字典形式；时间保留本地偏移，不走 canonicalize。"""

    return {
        "judgement_id": judgement.judgement_id,
        "judged_at": _instant(judgement.judged_at),
        "evidence_ready_at": _instant(judgement.evidence_ready_at),
        "started_at": _local(judgement.started_at),
        "last_observed_at": _local(judgement.last_observed_at),
        "observation_ids": list(judgement.observation_ids),
        "source_refs": list(judgement.source_refs),
        "subjects": list(judgement.subjects),
        "behavior": judgement.behavior,
        "goal": judgement.goal,
        "summary": judgement.summary,
        "basis": [
            {"semantics": fact.semantics, "observation_ids": list(fact.observation_ids)}
            for fact in judgement.basis
        ],
        "status": None if judgement.status is None else judgement.status.value,
        "status_basis": None if judgement.status_basis is None else judgement.status_basis.value,
        "relations": [{"kind": kind, "target_id": target} for kind, target in judgement.relations],
        "fusion_version": judgement.fusion_version,
        "prompt_version": judgement.prompt_version,
    }


def _instant(value: datetime) -> str:
    """一个时刻，归一到 UTC——"我们什么时候知道的"没有本地日历含义。"""

    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _local(value: datetime) -> str:
    """行为时间，**保留本地偏移**；折 UTC 会改掉本地日历日。"""

    return value.isoformat(timespec="microseconds")


__all__ = [
    "FUSION_IMPLEMENTATION_VERSION",
    "FUSION_VERSION",
    "DurableFact",
    "DurableJudgement",
    "derive_judgements",
    "judgement_payload",
]
