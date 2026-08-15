"""构造事件融合的线格式（judgements / frames），供各融合测试共用。

线格式的形状本身就是被测对象之一，所以这里刻意保持"全字段显式"：不适用的字段填 null 或空数组，
而不是靠默认值补齐——那会让测试掩盖掉真实调用里模型必须填全的要求。
"""

from __future__ import annotations

from typing import Any

SUBJECT = "家庭成员A"


def judgement(
    judgement_no: int,
    *,
    behavior: str | None = None,
    goal: str | None = None,
    summary: str | None = None,
    subjects: list[str] | None = None,
    basis: list[str] | None = None,
    status: str | None = "completed",
    status_basis: str | None = "observed",
    relations: list[tuple[str, int]] | None = None,
    context_relations: list[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    """一条判断；``basis`` 按顺序给出事实语义，编号自动从 1 排。"""

    facts = basis or []
    unreadable = behavior is None
    return {
        "judgement_no": judgement_no,
        "subjects": [] if unreadable else list(subjects or [SUBJECT]),
        "behavior": behavior,
        "goal": goal,
        "summary": None if unreadable else (summary or f"{behavior}的摘要"),
        "basis": [
            {"basis_no": index, "semantics": item} for index, item in enumerate(facts, start=1)
        ],
        "status": None if unreadable else status,
        "status_basis": None if unreadable else status_basis,
        "relations": [
            *(
                {"kind": kind, "target": target, "context_target": None}
                for kind, target in (relations or [])
            ),
            *(
                {"kind": kind, "target": None, "context_target": target}
                for kind, target in (context_relations or [])
            ),
        ],
    }


def unreadable(judgement_no: int) -> dict[str, Any]:
    """一条"这段观测没读懂"的判断。"""

    return judgement(judgement_no, behavior=None, status=None, status_basis=None)


def frames(assignment: list[list[tuple[int, int | None]]]) -> list[dict[str, Any]]:
    """按位置给出每一帧的归属；每项是 ``(judgement_no, basis_no)`` 的列表。"""

    return [
        {
            "no": index,
            "assignments": [
                {"judgement_no": judgement_no, "basis_no": basis_no}
                for judgement_no, basis_no in items
            ],
        }
        for index, items in enumerate(assignment, start=1)
    ]


def wire(
    judgements: list[dict[str, Any]],
    assignment: list[list[tuple[int, int | None]]],
) -> dict[str, Any]:
    return {"judgements": judgements, "frames": frames(assignment)}


def single_event_wire(
    fragment_count: int = 3,
    *,
    behavior: str = "洗手",
    goal: str | None = "清洁双手",
    status: str = "completed",
) -> dict[str, Any]:
    """一条判断覆盖全部片段的最小线格式。"""

    basis = ["走到水池边打开水龙头"] if goal is not None else []
    return wire(
        [judgement(1, behavior=behavior, goal=goal, basis=basis, status=status)],
        [[(1, 1 if goal is not None else None)]] * fragment_count,
    )


__all__ = ["SUBJECT", "frames", "judgement", "single_event_wire", "unreadable", "wire"]
