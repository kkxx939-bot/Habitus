"""判断存储记录的归约侧只读视图。

存储的线格式是字典（见 ``behavior.fusion.derivation.judgement_payload``）；归约把它解析成带
真正时间对象的不可变值，之后的全部机械规则只面对这个视图。解析即校验：字段缺失或时间格式不对
说明有人绕过融合层写了存储，立即失败优于带病归约。

时间的两种约定在这里显式分开（与融合层同一纪律）："何时成立/何时可知"是 UTC 时刻
（``evidence_ready_at``），"行为何时发生"是带本地偏移的时刻（``started_at`` /
``last_observed_at``）——后者解析后原样保留偏移，归约在物化 payload 时才做唯一一次换算。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from habitus.behavior.reduction.errors import BehaviorReductionError


def _text(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise BehaviorReductionError(f"judgement record field {key} must be non-empty text")
    return value


def _optional_text(record: Mapping[str, Any], key: str) -> str | None:
    value = record.get(key)
    if value is not None and not isinstance(value, str):
        raise BehaviorReductionError(f"judgement record field {key} must be text or null")
    return value


def _texts(record: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = record.get(key)
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise BehaviorReductionError(f"judgement record field {key} must be an array")
    resolved = tuple(value)
    if any(not isinstance(item, str) or not item for item in resolved):
        raise BehaviorReductionError(f"judgement record field {key} must contain non-empty text")
    return resolved


def _instant(record: Mapping[str, Any], key: str) -> datetime:
    """UTC 时刻（"我们什么时候知道的"）；存储写成 ``...Z``。"""

    raw = _text(record, key)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BehaviorReductionError(f"judgement record field {key} is not a timestamp") from exc
    if parsed.utcoffset() is None:
        raise BehaviorReductionError(f"judgement record field {key} must be timezone-aware")
    return parsed


def _local(record: Mapping[str, Any], key: str) -> datetime:
    """行为时刻；存储保留本地偏移，这里原样还原、不折 UTC。

    零偏移与树的裁定同口径在门口现形（+00:00 = 上游折 UTC 的事故信号）——放行的话会在
    stage/publish 最深处才被树硬拒，把点故障放大成整轮失败。
    """

    raw = _text(record, key)
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise BehaviorReductionError(f"judgement record field {key} is not a timestamp") from exc
    offset = parsed.utcoffset()
    if offset is None or not offset:
        raise BehaviorReductionError(
            f"judgement record field {key} must carry a non-zero local offset"
        )
    return parsed


@dataclass(frozen=True)
class ReducibleJudgement:
    """一条待归约的判断；字段语义与耐久记录一致，只是类型换成了可计算的值。"""

    judgement_id: str
    evidence_ready_at: datetime
    started_at: datetime
    last_observed_at: datetime
    observation_ids: tuple[str, ...]
    source_refs: tuple[str, ...]
    subjects: tuple[str, ...]
    behavior: str | None
    goal: str | None
    summary: str | None
    basis: tuple[tuple[str, tuple[str, ...]], ...]
    status: str | None
    status_basis: str | None
    relations: tuple[tuple[str, str], ...]
    fusion_version: str

    @property
    def is_readable(self) -> bool:
        return self.behavior is not None

    @property
    def order_key(self) -> tuple[datetime, str]:
        """链内排序键：行为时刻优先、身份定序——与融合上下文的排序纪律一致。"""

        return (self.started_at, self.judgement_id)


def parse_judgement_record(record: Mapping[str, Any]) -> ReducibleJudgement:
    if not isinstance(record, Mapping):
        raise BehaviorReductionError("judgement record must be a mapping")
    basis_raw = record.get("basis")
    if isinstance(basis_raw, str | bytes) or not isinstance(basis_raw, Sequence):
        raise BehaviorReductionError("judgement record field basis must be an array")
    basis: list[tuple[str, tuple[str, ...]]] = []
    for index, item in enumerate(basis_raw, start=1):
        if not isinstance(item, Mapping):
            raise BehaviorReductionError(f"judgement record basis[{index}] must be a mapping")
        basis.append((_text(item, "semantics"), _texts(item, "observation_ids")))
    relations_raw = record.get("relations")
    if isinstance(relations_raw, str | bytes) or not isinstance(relations_raw, Sequence):
        raise BehaviorReductionError("judgement record field relations must be an array")
    relations: list[tuple[str, str]] = []
    for index, item in enumerate(relations_raw, start=1):
        if not isinstance(item, Mapping):
            raise BehaviorReductionError(f"judgement record relations[{index}] must be a mapping")
        relations.append((_text(item, "kind"), _text(item, "target_id")))
    subjects_raw = record.get("subjects")
    if isinstance(subjects_raw, str | bytes) or not isinstance(subjects_raw, Sequence):
        raise BehaviorReductionError("judgement record field subjects must be an array")
    subjects = tuple(subjects_raw)
    if any(not isinstance(item, str) or not item for item in subjects):
        raise BehaviorReductionError("judgement record subjects must contain non-empty text")
    resolved = ReducibleJudgement(
        judgement_id=_text(record, "judgement_id"),
        evidence_ready_at=_instant(record, "evidence_ready_at"),
        started_at=_local(record, "started_at"),
        last_observed_at=_local(record, "last_observed_at"),
        observation_ids=_texts(record, "observation_ids"),
        source_refs=_texts(record, "source_refs"),
        subjects=subjects,
        behavior=_optional_text(record, "behavior"),
        goal=_optional_text(record, "goal"),
        summary=_optional_text(record, "summary"),
        basis=tuple(basis),
        status=_optional_text(record, "status"),
        status_basis=_optional_text(record, "status_basis"),
        relations=tuple(relations),
        fusion_version=_text(record, "fusion_version"),
    )
    # 融合层不变量在门口复核：可读判断必带摘要/状态/主体。缺了说明存储被绕写，
    # 快失败优于让 "None" 之类的字面值一路走进树的正文。
    if resolved.is_readable and (
        resolved.summary is None
        or resolved.status is None
        or resolved.status_basis is None
        or not resolved.subjects
    ):
        raise BehaviorReductionError(
            "a readable judgement record must carry summary, status and subjects"
        )
    return resolved


__all__ = ["ReducibleJudgement", "parse_judgement_record"]
