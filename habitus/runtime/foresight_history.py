"""单元历史在组合根的那一半（裁定 55 新版第 2 条，方案 19 第九节 6e；26 号清单 4.9）：每刻一份记录落盘，每晚按当晚的关系表与
时间线把全部记录重标一遍在场的输入，一条 lane 一晚一份快照；白天的单元只读最晚一晚的快照，不再逐拍扫账本。落盘的目录与字节在
``runtime.foresight_history_store``。

- **记**：账本里结完（``is_final``）、还没记过的日子，每条预期记一份；结完之后又被重结撤回的日子从"记过的"里拿掉，再结完时
  整天重记（同一条预期的记录整份替换）。不再每晚回读"回看多少天"的日目录：账本压实会删掉旧事实，单元历史的原料自己存着。
- **重标**：每晚读全部记录，按当晚的语义树重标在场的输入；一刻的分量是原始强度，"规律断了"的打折读的时候对正在读的那串刻算
  （``edges.fold_sequence``），不存打折后的。
- **退出**：一刻在它所属的每一层里，排在它之后的都已有 M 刻（单元每层只读最近 M 刻），并且早于诊断够得着的最早一天，
  就把它的记录删掉（``foresight.history.retired``）——删了哪一层的读数都不变。

算法在 ``foresight.history``（纯函数），这里只管读账本、把记录交给落盘那一层。
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, time, timedelta

from habitus.foresight import Evidence
from habitus.foresight.clock import current_facts
from habitus.foresight.history import (
    MomentRecord,
    UnitHistory,
    build_history,
    record_of,
    retired,
)
from habitus.foresight.ledger import AccountFact
from habitus.foresight.opportunities import OpportunityTable
from habitus.runtime.foresight_history_store import HistoryStore
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_settlement import DIAGNOSIS_REACH_DAYS
from habitus.scene.relations import LaneTimeline


class HistoryStage:
    """``statuses`` 与建预期同一组关系状态（成立；``shadow_uses_unestablished`` 开着时加候选）。"""

    def __init__(
        self, ledger: ForesightLedgerStore, store: HistoryStore, *, evidence: Evidence, statuses: Iterable[str]
    ) -> None:
        if not isinstance(ledger, ForesightLedgerStore):
            raise TypeError("ledger must be a ForesightLedgerStore")
        if not isinstance(store, HistoryStore):
            raise TypeError("store must be a HistoryStore")
        if not isinstance(evidence, Evidence):
            raise TypeError("evidence must be an Evidence")
        self.ledger = ledger
        self.store = store
        self.evidence = evidence
        self.statuses = tuple(statuses)

    def run(self, *, night: date, table: OpportunityTable, line: LaneTimeline) -> int:
        """第 ``night`` 晚这条 lane 的单元历史；返回重标了几刻。只读这一晚之前形成、这一晚之前写下的东西。"""

        lane = table.lane
        records = self._record(lane, night)
        if not records:
            self.store.write(UnitHistory(lane, night, {}))
            return 0
        now = datetime.combine(night, time(0, 0), tzinfo=records[0].formed_at.tzinfo)
        history, labelled = build_history(lane, night, records, table=table, line=line, statuses=self.statuses)
        self.store.write(history)
        before = now - timedelta(days=DIAGNOSIS_REACH_DAYS)
        gone = set().union(*(retired(found, window=self.evidence.window, before=before) for found in labelled.values()))
        self.store.discard_records(lane, gone)
        return sum(len(found) for found in history.moments.values())

    def _record(self, lane: str, night: date) -> tuple[MomentRecord, ...]:
        """把结完、还没记过的日子记下来；被重结撤回的日子从"记过的"里拿掉。返回这条 lane 的全部记录。"""

        days = self.ledger.days_with_expectations()
        # 账本放掉的日子（26 号清单 6.11）从"记过的"里一起拿掉：它们的每刻记录已经在了，这份集合不跟着历史一直长
        recorded = set(self.store.recorded_days(lane)) & set(days)
        known_by = None
        facts: dict[str, tuple[AccountFact, ...]] = {}
        for day in days:
            if day >= night:
                continue
            if not self.ledger.is_final(day):
                recorded.discard(day)
                continue
            if day in recorded:
                continue
            for item in self.ledger.expectations_on(day):
                if known_by is None:
                    known_by = datetime.combine(night, time(0, 0), tzinfo=item.formed_at.tzinfo)
                if item.account not in facts:
                    visible = (fact for fact in self.ledger.facts_of(item.account) if fact.precedence <= known_by)
                    facts[item.account] = current_facts(visible)
                fact = next(
                    (found for found in facts[item.account] if found.expectation_id == item.expectation_id), None
                )
                self.store.put_record(lane, record_of(item, fact))
            recorded.add(day)
        self.store.set_recorded_days(lane, recorded)
        return self.store.records(lane)


__all__ = ["HistoryStage"]
