"""递送那一拍问模型的那一半：把真要开口的几条装成小包问一次，按模型排的先后给名额，用模型的话（没有就用模板）。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from habitus.foresight.judge import (
    Judge,
    Judgement,
    Package,
    UnitInput,
    Verdict,
    build_package,
    plain_text,
    sources_of,
)
from habitus.foresight.ledger import Delivery, Expectation
from habitus.runtime.foresight_units import Unit

if TYPE_CHECKING:  # 只给类型检查看：``foresight_delivery`` 在运行时导入本模块
    from habitus.runtime.foresight_delivery import JudgeMaterials


async def ask(
    judge: Judge,
    asked: tuple[Expectation, ...],
    levels: Mapping[str, int],
    materials: JudgeMaterials,
    now: datetime,
    inputs: Mapping[str, Sequence[UnitInput]],
) -> Judgement:
    return await judge.judge(package(asked, materials, now, levels, inputs))


def package(
    items: tuple[Expectation, ...],
    materials: JudgeMaterials,
    now: datetime,
    levels: Mapping[str, int],
    inputs: Mapping[str, Sequence[UnitInput]] | None = None,
) -> Package:
    rows = materials.rows_for(sources_of(items, materials.table, units=inputs))
    return build_package(
        items, materials.table, rows, now=now, levels=levels, day_note=materials.day_note, units=inputs
    )


def ordered_closing(
    closing: tuple[tuple[Expectation, Delivery], ...], judgement: Judgement | None
) -> tuple[tuple[Expectation, Delivery], ...]:
    """模型排的先后在前（名额先给它们），其余按时窗。"""

    if judgement is None:
        return closing
    by_id = {item.expectation_id: (item, pending) for item, pending in closing}
    first = [by_id[identity] for identity in judgement.order if identity in by_id]
    rest = [pair for pair in closing if pair[0].expectation_id not in judgement.order]
    return tuple(first + rest)


def text_for(
    item: Expectation,
    verdict: Verdict | None,
    materials: JudgeMaterials | None,
    now: datetime,
    inputs: Mapping[str, Sequence[UnitInput]],
) -> str:
    if verdict is not None and verdict.text:
        return verdict.text
    table = materials.table if materials is not None else None
    rows = materials.rows_for(sources_of((item,), table, units=inputs)) if materials is not None else {}
    built = build_package((item,), table, rows, now=now, levels={}, units=inputs)
    return plain_text(built.candidates[0])


def unit_inputs(unit: Unit) -> tuple[UnitInput, ...]:
    """单元的全部输入给小包：每条预期与它那本账预测轴上（遗忘之后的）说中几次、几次机会。"""

    return tuple(
        UnitInput(item.expectation, item.standing.prediction.k, item.standing.prediction.n) for item in unit.inputs
    )


__all__ = ["ask", "ordered_closing", "package", "text_for", "unit_inputs"]
