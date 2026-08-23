"""重建后行为树测试共用的规范载荷。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

CST = timezone(timedelta(hours=8))
DAY = date(2026, 8, 16)


def _sha(prefix: str) -> str:
    return (prefix * 64)[:64]


OBS_A = _sha("a")
OBS_B = _sha("b")
OBS_C = _sha("c")
JUDGEMENT_1 = _sha("d")
SOURCE_1 = _sha("e")


def local(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2026, 8, 16, hour, minute, second, tzinfo=CST)


def occurrence_payload(**overrides: Any) -> dict[str, Any]:
    """一条"洗手"occurrence 的完整规范载荷；按需覆盖单项。"""

    payload: dict[str, Any] = {
        "occurred_on": DAY,
        "name": "洗了手",
        "started_at": local(19, 30, 18),
        "kind_token": "洗手",
        "status": "completed",
        "status_basis": "observed",
        "last_observed_at": local(19, 31, 30),
        "onset_available_at": local(19, 30, 20),
        "reminded": False,
        "goal": "清洁双手",
        "summary": "回家后到水池边洗手",
        "subjects": ("家庭成员A",),
        "place": "厨房",
        "original_name": None,
        "basis": (
            {
                "semantics": "打开水龙头打肥皂搓手",
                "observation_ids": (OBS_A, OBS_B),
                "started_at": local(19, 30, 18),
                "ended_at": local(19, 31, 0),
                "available_at": local(19, 30, 20),
            },
            {
                "semantics": "冲水关龙头擦干",
                "observation_ids": (OBS_C,),
                "started_at": local(19, 31, 0),
                "ended_at": local(19, 31, 30),
                "available_at": local(19, 31, 2),
            },
        ),
        "judgement_ids": (JUDGEMENT_1,),
        "observation_ids": (OBS_A, OBS_B, OBS_C),
        "source_refs": (SOURCE_1,),
        "fusion_version": "behavior_judgement_fusion_v1+prompt_v15+schema0000",
        "reduction_version": "behavior_reduction_v1",
    }
    payload.update(overrides)
    return payload


def action_segment_payload(**overrides: Any) -> dict[str, Any]:
    """无目标动作段：goal 空、basis 空，照常进树。"""

    payload = occurrence_payload(
        name="起身走开",
        kind_token="起身走开",
        goal=None,
        basis=(),
        summary="放下筷子起身走向画面外",
        status="completed",
        status_basis="observed",
        place=None,
    )
    payload.update(overrides)
    return payload


def gap_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "occurred_on": DAY,
        "gap_kind": "没读懂",
        "started_at": local(20, 10),
        "ended_at": local(20, 40),
        "judgement_ids": (JUDGEMENT_1,),
        "observation_ids": (OBS_A,),
        "reduction_version": "behavior_reduction_v1",
    }
    payload.update(overrides)
    return payload
