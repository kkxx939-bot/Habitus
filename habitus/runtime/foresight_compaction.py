"""账本按账精确压实的编排（26 号清单 4.8；裁定 61 第 7 条）：夜里在单元历史落盘之后，逐本账把窗外的旧事实文件删掉，只留
全历史摘要（偏差、机会间隔）。

策略（压哪些、压成什么）在 ``foresight.compaction``，字节在账本门面（``runtime.foresight_ledger``）；这里只把两者接起来，
不碰文件系统。每本账先写累计值、再删文件：中途崩溃时读的一方忽略残留（``ForesightLedgerStore.facts_of``），下一晚的
``discard_compacted`` 把残留删掉。级的变化只留诊断够得着的那一段，外加它之前最近的一条。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from habitus.foresight import Evidence, ForesightError, ranked_account
from habitus.foresight.compaction import compact, compactable
from habitus.foresight.opportunities import OpportunityTable
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_settlement import DIAGNOSIS_REACH_DAYS


@dataclass(frozen=True)
class CompactionReport:
    accounts: int = 0
    compacted: int = 0
    discarded: int = 0
    trimmed: int = 0
    #: 压不成的账（账身份 → 异常类型名）：这一本跳过，别的照压。
    failed: dict[str, str] = field(default_factory=dict)


class LedgerCompaction:
    def __init__(
        self,
        ledger: ForesightLedgerStore,
        *,
        evidence: Evidence,
        lane: str = "session",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(ledger, ForesightLedgerStore):
            raise TypeError("ledger must be a ForesightLedgerStore")
        if not isinstance(evidence, Evidence):
            raise TypeError("evidence must be an Evidence")
        self.ledger = ledger
        self.evidence = evidence
        #: 这本账本是哪条 lane 的（机会表按 lane 来，只用自己那条的覆盖算机会间隔）。
        self.lane = lane
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    def run(self, *, night: date, table: OpportunityTable | None) -> CompactionReport:
        now = self._clock()
        open_days = self.ledger.open_days()
        open_day = min(open_days) if open_days else None
        reach = night - timedelta(days=DIAGNOSIS_REACH_DAYS)
        keep_levels_from = datetime.combine(reach, datetime.min.time(), tzinfo=now.tzinfo)
        compacted = discarded = trimmed = 0
        failed: dict[str, str] = {}
        accounts = self.ledger.accounts()
        for account in accounts:
            try:
                compacted += self._account(account, now=now, open_day=open_day, reach=reach, table=table)
                discarded += self.ledger.discard_compacted(account)
                trimmed += self.ledger.trim_levels(account, before=keep_levels_from)
            except ForesightError as exc:
                failed[account] = type(exc).__name__
        return CompactionReport(len(accounts), compacted, discarded, trimmed, failed)

    def _account(
        self, account: str, *, now: datetime, open_day: date | None, reach: date, table: OpportunityTable | None
    ) -> int:
        base = self.ledger.compaction_of(account)
        view = ranked_account(self.ledger.facts_of(account), visible_at=now, base=base)
        plan = compactable(view, open_day=open_day, diagnosis_since=reach, window=self.evidence.window)
        if plan.boundary is None:
            return 0
        own = table if table is not None and table.lane == self.lane else None
        self.ledger.write_compaction(compact(view, plan, table=own))
        return 1


__all__ = ["CompactionReport", "LedgerCompaction"]
