"""递送那一拍开口前要核的事实：做没做、「没读懂」、久别、前因还成不成立。只取材料，判法在 ``foresight.delivery`` 与
``foresight.ledger``。"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from habitus.foresight import Evidence, Standing
from habitus.foresight.absence import returned_after_absence
from habitus.foresight.delivery import Circumstances, quiet_after_absence
from habitus.foresight.hierarchy import FLAT, KindHierarchy
from habitus.foresight.ledger import AntecedentRow, Expectation, antecedent_holds_when_delivering
from habitus.runtime.foresight_ledger import ForesightLedgerStore

if TYPE_CHECKING:  # 只给类型检查看：``foresight_delivery`` 在运行时导入本模块
    from habitus.runtime.foresight_delivery import JudgeMaterials


def circumstances(
    item: Expectation, materials: JudgeMaterials | None, ledger: ForesightLedgerStore, evidence: Evidence
) -> Circumstances:
    """这一条的材料，问到才读：做没做、「没读懂」从这一拍的材料读，久别从账上读，前因从机会表与这一拍的行读。"""

    table = materials.table if materials is not None else None
    return Circumstances(
        active=lambda: kinds_of(materials).active(item.kind_token),
        happened=lambda: materials is not None and materials.happened(item),
        absent=lambda: returned_after_absence(
            ledger.facts_of(item.account),
            item,
            table,
            ledger.compaction_of(item.account),
            unsettled_of(ledger, item.account),
            half_life=evidence.half_life,
        ),
        unobserved=lambda: materials is not None and materials.unobserved(item),
        antecedent_holds=lambda: antecedent_holds(item, materials),
    )


def returned(
    item: Expectation,
    standing: Standing,
    materials: JudgeMaterials | None,
    ledger: ForesightLedgerStore,
    evidence: Evidence,
) -> bool:
    return quiet_after_absence(standing, circumstances(item, materials, ledger, evidence).absent)


def unsettled_of(ledger: ForesightLedgerStore, account: str) -> tuple[Expectation, ...]:
    """这本账已形成、还没结算的预期（只在还要看的日子里；结完的日子全结过）：久别把它们也算机会。"""

    return tuple(
        item for day in ledger.open_days() for item in ledger.unsettled_expectations(day) if item.account == account
    )


def antecedent_holds(item: Expectation, materials: JudgeMaterials | None) -> bool:
    """开口之前再核一遍前因（规则在 ``ledger.settle.antecedent_holds_when_delivering``，找法与类的映射和结算同一个）：
    行从这一拍读到的那些里取。"""

    if materials is None:
        return True
    table = materials.table

    def rows_on(day: date) -> tuple[AntecedentRow, ...]:
        return tuple(
            AntecedentRow(row.started_at, row.ended_at, row.kind_token)
            for row in materials.seen
            if row.started_at.astimezone(item.formed_at.tzinfo).date() == day
        )

    return antecedent_holds_when_delivering(
        item,
        table.relation(item.trigger) if table is not None else None,
        rows_on=rows_on,
    )


def kinds_of(materials: JudgeMaterials | None) -> KindHierarchy:
    return materials.kinds if materials is not None else FLAT


__all__ = ["antecedent_holds", "circumstances", "kinds_of", "returned", "unsettled_of"]
