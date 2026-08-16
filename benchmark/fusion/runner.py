"""对一批用例做真实模型调用，产出可核对的原始记录。

这里只跑和记录，不下结论——聚合与判定在 ``report.py``。分开是为了让同一批 ``records.jsonl``
可以反复重新聚合（改了判定口径不必重跑模型）。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from behavior.fusion import (
    BehaviorFusionConfig,
    BehaviorFusionSegment,
    assemble_judgement_batch,
    derive_judgements,
    fusion_json_schema,
    validate_judgement_batch,
)
from behavior.fusion.prompt import (
    FUSION_PROMPT_VERSION,
    FUSION_SYSTEM_PROMPT,
    render_context_judgements,
    render_fragments,
    render_primary_subject,
)
from behavior.observation import BehaviorObservation, BehaviorObservationConfig
from benchmark.fusion.dataset import FusionCase, FusionFragment
from ModelClient import ChatMessage, ChatRequest, StructuredChatClient

_OBSERVATION_CONFIG = BehaviorObservationConfig()
_EPOCH = datetime(2026, 1, 1, 9, 0, tzinfo=timezone(timedelta(hours=8)))


@dataclass
class FusionSegmentRun:
    """一段观测跑完之后的原始记录。"""

    index: int
    judgements: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    unreadable_fragments: list[int] = field(default_factory=list)
    out_of_scope_fragments: list[int] = field(default_factory=list)
    model_calls: int = 0
    failed: str | None = None


@dataclass
class FusionCaseRun:
    """一个用例跑完之后的原始记录。"""

    case_id: str
    category: str
    intent: str
    attempt: int
    segments: list[FusionSegmentRun] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "intent": self.intent,
            "attempt": self.attempt,
            "prompt_version": FUSION_PROMPT_VERSION,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "segments": [
                {
                    "index": item.index,
                    "judgements": item.judgements,
                    "relations": item.relations,
                    "unreadable_fragments": item.unreadable_fragments,
                    "out_of_scope_fragments": item.out_of_scope_fragments,
                    "model_calls": item.model_calls,
                    "failed": item.failed,
                }
                for item in self.segments
            ],
        }


def _observation(fragment: FusionFragment) -> BehaviorObservation:
    at = _EPOCH + timedelta(seconds=fragment.offset_seconds)
    return BehaviorObservation.create(
        observer_id="benchmark/fusion",
        occurred_at=at,
        available_at=at + timedelta(milliseconds=900),
        modality=fragment.modality,
        semantics=fragment.semantics,
        participants=list(fragment.participants),
        knowledge_state=fragment.knowledge_state,
        confidence=0.9,
        evidence_refs=[f"case:{fragment.offset_seconds}"],
        config=_OBSERVATION_CONFIG,
    )


async def run_case(
    case: FusionCase,
    client: StructuredChatClient,
    *,
    attempt: int,
    primary_subject: str,
    config: BehaviorFusionConfig | None = None,
) -> FusionCaseRun:
    """按顺序融合一个用例的各段；后一段拿前一段已落盘的判断做上下文。"""

    resolved = config or BehaviorFusionConfig()
    run = FusionCaseRun(case.case_id, case.category, case.intent, attempt)
    started = time.perf_counter()
    context: list[dict[str, Any]] = []
    context_ids: list[str] = []
    # 判断身份 → 它出自第几段。关系可能指向本段或先前任意一段，用例的期望是按段号写的。
    segment_of: dict[str, int] = {}

    for index, fragments in enumerate(case.segments, start=1):
        observations = tuple(_observation(item) for item in fragments)
        segment = BehaviorFusionSegment(observations, ("benchmark:fusion",))
        record = FusionSegmentRun(index=index)
        content = "\n\n".join(
            (
                "【本次跟踪的主体】",
                render_primary_subject(primary_subject),
                "【先前的判断】",
                render_context_judgements(context, segment.fragments[0].occurred_at),
                "【本次片段】",
                render_fragments(segment.fragments),
            )
        )
        request = ChatRequest(
            messages=(
                ChatMessage(role="system", content=FUSION_SYSTEM_PROMPT),
                ChatMessage(role="user", content=content),
            )
        )
        # 显式绑定：闭包若捕获循环变量，下一段的值会回头改掉这一段的校验口径。
        assemble = _assembler(
            len(observations), tuple(item["status"] for item in context), resolved
        )
        try:
            response = await client.complete_json_async(
                request,
                schema=fusion_json_schema(len(observations)),
                name="behavior_judgement_batch",
                validator=assemble,
            )
        except Exception as exc:  # 模型持续违约也是一种结果，记录下来而不是中断整批
            record.failed = f"{type(exc).__name__}: {exc}"
            run.segments.append(record)
            break
        batch = response.value
        record.model_calls = response.validation_attempts
        try:
            validate_judgement_batch(batch, segment.fragments)
        except Exception as exc:  # pragma: no cover - 装配已保证，纵深防御
            record.failed = f"{type(exc).__name__}: {exc}"
            run.segments.append(record)
            break

        judged_at = segment.fragments[-1].available_at + timedelta(minutes=1)
        derived = derive_judgements(
            batch,
            segment.fragments,
            source_refs=("benchmark:fusion",),
            judged_at=judged_at,
            context_ids=tuple(context_ids),
        )
        by_id = {item.judgement_id: item for item in derived}
        for item in derived:
            segment_of[item.judgement_id] = index
        for item in derived:
            record.judgements.append(
                {
                    "judgement_id": item.judgement_id,
                    "subjects": list(item.subjects),
                    "behavior": item.behavior,
                    "goal": item.goal,
                    "summary": item.summary,
                    "status": None if item.status is None else item.status.value,
                    "status_basis": None if item.status_basis is None else item.status_basis.value,
                    "basis": [fact.semantics for fact in item.basis],
                    "covers": len(item.observation_ids),
                }
            )
            positions = {
                observation.observation_id: order
                for order, observation in enumerate(segment.fragments, start=1)
            }
            if item.behavior is None:
                record.unreadable_fragments.extend(
                    positions[observation_id] for observation_id in item.observation_ids
                )
            elif primary_subject not in item.subjects:
                record.out_of_scope_fragments.extend(
                    positions[observation_id] for observation_id in item.observation_ids
                )
            for kind, target in item.relations:
                record.relations.append(
                    {
                        "kind": kind,
                        "from": item.behavior,
                        "to_segment": segment_of.get(target),
                        "to_behavior": _behavior_of(target, by_id, context),
                    }
                )
        # 一帧若同时被某条可读判断认领，它已经被读懂了，不该计入——与生产侧
        # ``receipt.py`` / ``BehaviorJudgementBatch.unreadable_fragment_count`` 必须同口径，
        # 否则评测是在拿一个上线后根本不会被算出来的量做判定。
        readable = {
            positions[observation_id]
            for item in derived
            if item.behavior is not None
            for observation_id in item.observation_ids
        }
        record.unreadable_fragments = sorted(set(record.unreadable_fragments) - readable)
        # 被主体的判断覆盖的帧就不算"落在主体之外"——旁人帧若被吸收进去，这里会空，正是要抓的。
        tracked = {
            positions[observation_id]
            for item in derived
            if primary_subject in item.subjects
            for observation_id in item.observation_ids
        }
        record.out_of_scope_fragments = sorted(set(record.out_of_scope_fragments) - tracked)
        run.segments.append(record)

        context_ids.extend(item.judgement_id for item in derived)
        context.extend(
            {
                "judgement_id": item.judgement_id,
                "started_at": item.started_at.isoformat(),
                "behavior": item.behavior,
                "goal": item.goal,
                "status": None if item.status is None else item.status.value,
                "status_basis": None if item.status_basis is None else item.status_basis.value,
            }
            for item in derived
        )

    run.elapsed_seconds = time.perf_counter() - started
    return run


def _assembler(
    fragment_count: int, context_states: tuple[str | None, ...], config: BehaviorFusionConfig
) -> Callable[[object], Any]:
    def assemble(parsed: object) -> Any:
        return assemble_judgement_batch(
            parsed,
            fragment_count=fragment_count,
            context_states=context_states,
            config=config,
        )

    return assemble


def _behavior_of(target: str, by_id: dict[str, Any], context: Sequence[dict[str, Any]]) -> str | None:
    if target in by_id:
        return by_id[target].behavior
    for item in context:
        if item["judgement_id"] == target:
            return item["behavior"]
    return None


__all__ = ["FusionCaseRun", "FusionSegmentRun", "run_case"]
