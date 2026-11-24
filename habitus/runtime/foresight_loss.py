"""loss 尺子在组合根的那一半（裁定 114）：每次结算之后从账上现算，不存累计值——事后改判（后补、重结、作废）下一次自动反映。

取材料与单元历史记刻同一个读法（``runtime.foresight_history.HistoryStage._record``）：结完（``is_final``）的日子里每条预期，
连同账上此刻算数的那一笔事实、结算认出的那次发生，记成 ``history.record_of`` 的一份；递送整天取来（读数在递送上）。
扫多长按账本完整留着的那一段（诊断够得着的 ``DIAGNOSIS_REACH_DAYS`` 天：更早的事实可能已经压实掉，读成"不知道"会混进去），
不加配置。算分在 ``foresight.loss``（纯函数），这里只读账、数漏报。尺子只读、只是量：算不出来不让结算失败，记下异常类型。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from habitus.foresight.clock import current_facts
from habitus.foresight.diagnosis import DIAGNOSIS_REACH_DAYS
from habitus.foresight.history import MomentRecord, record_of
from habitus.foresight.ledger import AccountFact, Delivery, Expectation
from habitus.foresight.loss import Loss, measure
from habitus.runtime.foresight_ledger import ForesightLedgerStore


@dataclass(frozen=True)
class LedgerLoss:
    """这一支账上此刻的 loss：扫了几天结完的日子、这几天里的漏报条数（尺子看不到的那部分，放在旁边）、算出来的分；
    算不出来时 ``loss`` 为 None、``failed`` 是异常类型名。"""

    days: int
    omissions: int
    loss: Loss | None
    failed: str | None = None


def measure_loss(ledger: ForesightLedgerStore, *, now: datetime) -> LedgerLoss:
    """``now`` 那天及之前 ``DIAGNOSIS_REACH_DAYS`` 天里结完的日子上的 loss（时钟回退时账上更晚的日子不算）。"""

    try:
        return _measure(ledger, now)
    except Exception as exc:  # noqa: BLE001 - 尺子只读、只是量，坏了不影响结算（裁定 114）
        return LedgerLoss(0, 0, None, failed=type(exc).__name__)


def _measure(ledger: ForesightLedgerStore, now: datetime) -> LedgerLoss:
    since = now.date() - timedelta(days=DIAGNOSIS_REACH_DAYS)
    records: list[MomentRecord] = []
    expectations: dict[str, Expectation] = {}
    deliveries: list[Delivery] = []
    facts: dict[str, tuple[AccountFact, ...]] = {}
    # 尺子读到的坏文件不进结算观测那一份共享清单（账本只有预期与递送能另给清单；结算、事实、漏报照旧进共享的那一份）
    skipped: list[str] = []
    days = omissions = 0
    for day in ledger.days_with_expectations(since=since):
        if day > now.date() or not ledger.is_final(day):
            continue
        days += 1
        omissions += len(ledger.omissions_on(day))
        deliveries.extend(ledger.deliveries_on(day, unreadable=skipped))
        found_on = {item.expectation_id: item.occurrence_uri for item in ledger.settlements_on(day)}
        for item in ledger.expectations_on(day, unreadable=skipped):
            if item.account not in facts:
                facts[item.account] = current_facts(ledger.facts_of(item.account))
            fact = next((found for found in facts[item.account] if found.expectation_id == item.expectation_id), None)
            records.append(record_of(item, fact, occurrence_uri=found_on.get(item.expectation_id)))
            expectations[item.expectation_id] = item
    return LedgerLoss(days, omissions, measure(records, expectations, deliveries))


__all__ = ["LedgerLoss", "measure_loss"]
