"""归约层测试共用的判断记录与观测构造。

记录形状与 ``behavior.fusion.derivation.judgement_payload`` 的线格式一致（时间：行为时刻带
本地偏移、成立时刻归 UTC ``Z``）——归约的输入契约本身就是被测对象之一。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from behavior.observation import BehaviorObservation, BehaviorObservationConfig
from foundation.integrity import canonical_digest

CST = timezone(timedelta(hours=8))
BASE = datetime(2026, 8, 16, 19, 30, 0, tzinfo=CST)
SUBJECT = "家庭成员A"
OBSERVER = "home-a/hall"
OBSERVATION_CONFIG = BehaviorObservationConfig()
FUSION_VERSION_TEXT = "behavior_judgement_fusion_v3+prompt_v15+schema000000000000"


def at(seconds: int) -> datetime:
    return BASE + timedelta(seconds=seconds)


def _instant(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _local(value: datetime) -> str:
    return value.isoformat(timespec="microseconds")


def observation(offset: int, semantics: str, *, participants: list[str] | None = None) -> BehaviorObservation:
    moment = at(offset)
    return BehaviorObservation.create(
        observer_id=OBSERVER,
        occurred_at=moment,
        available_at=moment + timedelta(seconds=2),
        modality="vision",
        semantics=semantics,
        participants=participants or [SUBJECT],
        knowledge_state="observed",
        confidence=0.9,
        evidence_refs=[f"cam:{offset}"],
        config=OBSERVATION_CONFIG,
    )


def judgement_record(
    seed: str,
    *,
    behavior: str | None,
    started_at: datetime,
    last_observed_at: datetime,
    evidence_ready_at: datetime,
    observation_ids: tuple[str, ...],
    source_refs: tuple[str, ...],
    goal: str | None = None,
    summary: str | None = None,
    subjects: tuple[str, ...] | None = None,
    basis: tuple[tuple[str, tuple[str, ...]], ...] = (),
    status: str | None = "completed",
    status_basis: str | None = "observed",
    relations: tuple[tuple[str, str], ...] = (),
) -> dict[str, Any]:
    unreadable = behavior is None
    return {
        "judgement_id": canonical_digest({"seed": seed}),
        "judged_at": _instant(evidence_ready_at + timedelta(seconds=5)),
        "evidence_ready_at": _instant(evidence_ready_at),
        "started_at": _local(started_at),
        "last_observed_at": _local(last_observed_at),
        "observation_ids": list(observation_ids),
        "source_refs": list(source_refs),
        "subjects": [] if unreadable else list(subjects or (SUBJECT,)),
        "behavior": behavior,
        "goal": None if unreadable else goal,
        "summary": None if unreadable else (summary or f"{behavior}的摘要"),
        "basis": [
            {"semantics": semantics, "observation_ids": list(ids)} for semantics, ids in basis
        ],
        "status": None if unreadable else status,
        "status_basis": None if unreadable else status_basis,
        "relations": [{"kind": kind, "target_id": target} for kind, target in relations],
        "fusion_version": FUSION_VERSION_TEXT,
        "prompt_version": "behavior_judgement_prompt_v15",
    }


def record_id(seed: str) -> str:
    return canonical_digest({"seed": seed})


__all__ = [
    "BASE",
    "CST",
    "FUSION_VERSION_TEXT",
    "OBSERVATION_CONFIG",
    "OBSERVER",
    "SUBJECT",
    "at",
    "judgement_record",
    "observation",
    "record_id",
]
