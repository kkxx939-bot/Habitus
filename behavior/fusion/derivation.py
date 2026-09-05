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
不能折 UTC：行为的时间事实是它**实际发生的本地时刻**——树按本地日期分目录、时间预测树按本地
钟面计数，东八区凌晨的行为折成 UTC 会掉到前一天、错一个钟面槽位。``foundation.integrity.canonicalize`` 会折 UTC，所以这些字段的序列化必须单独
处理（见 ``store.py``）。

注意 ``last_observed_at`` 是"**最后一条支持它的观测**发生在什么时候"，不是"行为在什么时候
结束"——后者我们经常不知道，那正是 ``status_basis=observation_lost`` 要表达的事。

## 身份不含关系

``judgement_id`` 由"它说了什么、依据哪些观测、什么时候"派生，**不含 relation**。因为并行是
互指的：甲指乙、乙指甲，若关系进身份就成了循环依赖，谁也算不出来。关系是判断之间的连接，
不属于一条判断自身的身份。

也**不含 source_refs**：它是"这些观测由哪几次交付送来的"，属于溯源而不是内容。同一批观测被重复
投递两次，得到的应当是同一条判断，而不是两条。代价是身份的输入是 payload 的真子集——理论上
同内容、同 ``judged_at``、不同来源的两条会算出同一个身份却有不同的字节，写第二份时被存储的撞车
检查硬失败。实测构造不出正常路径：同一段观测的 ``source_refs`` 由切段一次定死，而不同批次必然
有不同的 ``judged_at``、身份本来就不同。所以这里保持现状，但把前提写下来——真要出现，说明有人
绕过了排队层用同一个时间戳重放了不同来源的同一段观测。
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
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
from behavior.fusion.schema import JUDGEMENT_FUSION_JSON_SCHEMA
from behavior.observation import BehaviorObservation
from foundation.integrity import canonical_digest

# v2：同一主体同刻开始的同名行为只接受一条判断（判重守卫）——之前的数据里可能存在这类重复。
# v3：behavior=空（没读懂）的判断也进判断存储（此前只留回执信号）——归约层要把它物化成 gap
#     节点，v2 及更早的存储里没有这批记录。
FUSION_IMPLEMENTATION_VERSION = "behavior_judgement_fusion_v4"

# 产物的语义取决于**送给模型的全部东西**加上派生实现，任一变化都会改变输出语义，所以版本必须
# 同时覆盖三者：实现、提示词、以及 **schema**。
#
# schema 不是形状声明而已——字段描述里承载着硬契约（"goal 非 null 时 basis 必须至少写一条"、
# interrupted 与 abandoned 的区别），实测改这些描述会实打实地改变模型行为。而它一度不在版本里：
# 改完描述、模型行为变了，版本却可以一字不动，两批语义不同的判断在数据上就分辨不出来。
#
# 用摘要而不是手写一个版本号，是为了**不可能忘记**：改了描述版本自动就变。代价是每次改 schema
# 都会让排队中的作业身份漂移，而那条路径已经由 ``BehaviorFusionJobStore.retarget`` 在认领之前
# 无损纠正。
_SCHEMA_FINGERPRINT = canonical_digest(JUDGEMENT_FUSION_JSON_SCHEMA)[:12]
FUSION_VERSION = f"{FUSION_IMPLEMENTATION_VERSION}+{FUSION_PROMPT_VERSION}+schema{_SCHEMA_FINGERPRINT}"

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

    **前提：``fragments`` 里不能有重复的观测身份。** 帧号在这里被换成观测身份，而
    ``assembly._require_distinguishable`` 是按帧号判两条判断相不相同的；映射一旦不是单射，两条
    内容相同、帧号不同的判断就会算出同一个 ``judgement_id`` 并在内容寻址的存储里静默合并。
    这个前提由结构保证而不是由检查保证：``segment_observations`` 与 ``BehaviorFusionRunner._fragments``
    都用 ``observation_id`` 作字典键收集片段，重复根本构造不出来。写在这里是为了将来新增调用方
    时知道它存在——不写成运行时检查，是因为那等于为一个不存在的调用方付代价。
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


def without_unresolvable_relations(
    judgement: DurableJudgement, visible_ids: Collection[str]
) -> DurableJudgement:
    """剪掉指向"不会落盘的判断"的关系。

    个人版只跟踪主体一个人，旁人的判断不落盘。而并行关系是**相互**的：模型按提示词只在一边
    声明，装配层会自动补上另一边——于是主体那条判断上会出现一条指向旁人判断的关系，随后旁人
    那条被分流掉。留着它，不可变存储里就永久躺着一个指向不存在记录的 ``target_id``；旁人在做
    什么本来就不在范围内，所以剪掉不损失任何我们要的东西。

    ``judgement_id`` 是**不含 relations** 的内容摘要（并行相互指向会让身份循环依赖），所以剪枝
    不会改变身份，也就不会与已经落盘的同一条判断冲突。
    """

    visible = set(visible_ids)
    kept = tuple(link for link in judgement.relations if link[1] in visible)
    if len(kept) == len(judgement.relations):
        return judgement
    return replace(judgement, relations=kept)


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
    """最早发生时刻，按其自身的本地偏移还原——树按实际发生的本地时刻定日期，不能折 UTC。"""

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


def persistable_judgements(
    judgements: Sequence[DurableJudgement], primary_subject: str
) -> tuple[DurableJudgement, ...]:
    """哪些判断进判断存储：主体的行为 + 没读懂的观测段。

    旁人的可读判断分流（会淹没判断存储，下游对它们毫无用处，观测身份已留在回执）；
    behavior=空的判断不属于任何人——它是"主体的观测轨道上这段没读懂"，归约层要把它物化成
    gap 节点，必须落盘。回执与落盘共用本函数：两处各算一遍会让 StagedFusion 的
    "staged 判断 == 回执 judgement_ids" 校验静默失效。
    """

    return tuple(
        item for item in judgements if not item.is_readable or primary_subject in item.subjects
    )


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

    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


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
    "persistable_judgements",
]
