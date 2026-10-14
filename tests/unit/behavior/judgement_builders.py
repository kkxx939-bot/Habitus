"""造判断的小工具：按"一条判断 + 每一帧归谁"的写法描述一批判断，翻成 ``BehaviorJudgementBatch``。

判断的派生、落盘、回执这些共用部分的测试用它造输入；它只是测试里的翻译，不是产品逻辑。
"""

from __future__ import annotations

from typing import Any

from habitus.behavior.fusion.judgement import (
    BehaviorClaim,
    BehaviorFact,
    BehaviorJudgement,
    BehaviorJudgementBatch,
    JudgementLink,
    JudgementRelation,
    JudgementStatus,
    JudgementStatusBasis,
)

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
        "basis": [{"basis_no": index, "semantics": item} for index, item in enumerate(facts, start=1)],
        "status": None if unreadable else status,
        "status_basis": None if unreadable else status_basis,
        "relations": [
            *({"kind": kind, "target": target, "context_target": None} for kind, target in (relations or [])),
            *({"kind": kind, "target": None, "context_target": target} for kind, target in (context_relations or [])),
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
            "assignments": [{"judgement_no": judgement_no, "basis_no": basis_no} for judgement_no, basis_no in items],
        }
        for index, items in enumerate(assignment, start=1)
    ]


def wire(
    judgements: list[dict[str, Any]],
    assignment: list[list[tuple[int, int | None]]],
) -> dict[str, Any]:
    return {"judgements": judgements, "frames": frames(assignment)}


def batch_from(raw: dict[str, Any], *, fragment_count: int) -> BehaviorJudgementBatch:
    """把 ``wire`` 的写法翻成一批判断：每条判断覆盖哪些帧、每条步骤落在哪些帧，由 ``frames`` 的归属反推。"""

    covers: dict[int, list[int]] = {}
    steps: dict[tuple[int, int], list[int]] = {}
    for frame in raw["frames"]:
        for item in frame["assignments"]:
            covers.setdefault(item["judgement_no"], []).append(frame["no"])
            if item["basis_no"] is not None:
                steps.setdefault((item["judgement_no"], item["basis_no"]), []).append(frame["no"])
    owned = {no for numbers in covers.values() for no in numbers}
    judgements = []
    for item in raw["judgements"]:
        number = item["judgement_no"]
        basis = tuple(
            BehaviorFact(semantics=fact["semantics"], fragment_nos=tuple(steps[(number, fact["basis_no"])]))
            for fact in item["basis"]
            if (number, fact["basis_no"]) in steps
        )
        judgements.append(
            BehaviorJudgement(
                judgement_no=number,
                covers=tuple(covers[number]),
                subjects=tuple(item["subjects"]),
                claim=BehaviorClaim(behavior=item["behavior"], goal=item["goal"], summary=item["summary"], basis=basis),
                status=None if item["status"] is None else JudgementStatus(item["status"]),
                status_basis=None if item["status_basis"] is None else JudgementStatusBasis(item["status_basis"]),
                relations=tuple(
                    JudgementLink(JudgementRelation(link["kind"]), link["target"], link["context_target"])
                    for link in item["relations"]
                ),
            )
        )
    judgements.sort(key=lambda entry: entry.earliest_fragment_no)
    unowned = tuple(no for no in range(1, fragment_count + 1) if no not in owned)
    return BehaviorJudgementBatch(tuple(judgements), unowned_fragment_nos=unowned)


__all__ = ["SUBJECT", "batch_from", "frames", "judgement", "unreadable", "wire"]
