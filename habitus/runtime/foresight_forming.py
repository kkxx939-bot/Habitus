"""建预期、落账的那一段，两处共用（裁定 94 第 1 条）：白天那一拍（``runtime.foresight_tick``，事件与情境）与夜里那条线末尾的
当天日子预期（``runtime.foresight_days``）。

取材料：此刻是主体本地时刻（``read_side.zone``），今天的日历条件从 ``read_side.calendar`` 取；行为树经 ``scene.views`` 只读今天
与前 7 天、还没封口的判断经 ``UnsealedReader``。装配交给 ``foresight.form_expectations``（``Batch`` 定建哪几种触发）。写账本：
同一个身份已经在账上的不再写（身份不含槽号、时刻与日子，账本按身份索引，跨天重算得到同一条）；人说过「以后别提了」的行为不再建
（裁定 50、61）。进程不留内存状态：重启后重扫得到的都是已有的身份。

两处共用的闸也在这里一份：词表迁移做到一半时不建（``held_by_migration``）、表是不是今天这一晚的（``tonight_table``，裁定 94
第 1 条），以及观测怎么记（``observe_foresight``）。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from datetime import time as clock_time
from typing import Protocol, runtime_checkable
from zoneinfo import ZoneInfo

from habitus.foresight import Batch, Moment, Seen, UnsealedReader, UnsealedRow, form_expectations
from habitus.foresight.ladder import Tolerance, silenced
from habitus.foresight.opportunities import OpportunityTable
from habitus.foresight.triggers.assemble import LOOKBACK_DAYS, seen_rows
from habitus.foundation.observability import ObservationEvent, ObservationStatus, Observer
from habitus.runtime.foresight_implementation import IMPLEMENTATION
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.runtime.foresight_settlement import BehaviorReadSide
from habitus.runtime.nightly_errors import NightJobsError
from habitus.scene.views import DayIndexCache
from habitus.series.reader import admitted
from habitus.temporal.slots import slot_index


@runtime_checkable
class FormingReadSide(BehaviorReadSide, Protocol):
    """建预期要读的那棵行为树、主体、日历（与结算同一侧）外加时区（预测层的装配器 ``ForesightReadSide`` 就是这个形状）。"""

    @property
    def zone(self) -> ZoneInfo: ...


@dataclass(frozen=True)
class Recent:
    """这一刻读到的最近几天：此刻、行为树的按天索引、还没封口的判断、两者合起来按开始时刻排好的行。"""

    now: datetime
    cache: DayIndexCache
    unsealed: tuple[UnsealedRow, ...]
    seen: tuple[Seen, ...]


@dataclass(frozen=True)
class Formed:
    #: 此刻对得上历史的预期有几条、其中新写进账本的几条、按触发种类分。
    formed: int = 0
    recorded: int = 0
    by_source: dict[str, int] = field(default_factory=dict)


class ExpectationForming:
    def __init__(
        self,
        ledger: ForesightLedgerStore,
        read_side: FormingReadSide,
        unsealed: UnsealedReader,
        *,
        recent: int,
        tolerance: Tolerance,
        shadow_unestablished: bool = False,
        vocabulary_version: Callable[[], int] | None = None,
    ) -> None:
        if recent < 1:
            raise ValueError("recent must be a positive integer")
        self.ledger = ledger
        self.read_side = read_side
        self.unsealed = unsealed
        #: 证据的窗口 M：「离上次够久」的时窗与预期的依据只看最近 M 次（``Evidence.window``）。
        self.recent = recent
        #: 账的那条线 θ：「离上次够久」的时窗往后接几天由它定（裁定 89 第 1 条，与递送、结算同一条线）。
        self.tolerance = tolerance
        self.shadow_unestablished = bool(shadow_unestablished)
        self.vocabulary_version = vocabulary_version

    async def read(self, now: datetime) -> Recent:
        """此刻往回看的几天。还没封口的判断读到与行为树同一个回看窗（今天与前 7 天）：过了零点、归约又落后时，昨晚还没封口的
        「没读懂」与可读的前因也在里面（``rows`` 本来就把全部未封口判断读进来，按窗过滤不省开销）。"""

        cache = DayIndexCache(
            self.read_side.behavior_tree,
            subject=self.read_side.subject,
            admits=admitted,
            calendar=self.read_side.calendar,
        )
        since = datetime.combine(now.date() - timedelta(days=LOOKBACK_DAYS), clock_time(0), tzinfo=now.tzinfo)
        await self.unsealed.prepare(until=now)
        unsealed = self.unsealed.rows(since=since, until=now)
        return Recent(now, cache, unsealed, seen_rows(Moment(at=now, slot=0), cache, unsealed))

    def form(
        self,
        recent: Recent,
        table: OpportunityTable,
        *,
        batch: Batch,
        looked_since: datetime | None = None,
        known_instances: Callable[[str], tuple[str, ...]] | None = None,
    ) -> Formed:
        """``batch`` 那几种触发此刻对得上历史的预期，新的写进账本。"""

        now = recent.now
        expectations = form_expectations(
            moment=Moment(
                at=now,
                slot=slot_index(now, slot_minutes=table.slot_minutes),
                calendar=dict(self.read_side.calendar.for_expectations(now.date())),
                looked_since=looked_since,
            ),
            table=table,
            rows=recent.seen,
            batch=batch,
            shadow_unestablished=self.shadow_unestablished,
            generation=table.night.isoformat(),
            version=str(self.vocabulary_version()) if self.vocabulary_version is not None else "",
            implementation=IMPLEMENTATION,
            recent=self.recent,
            line=self.tolerance.line,
            calendar_of=lambda day: dict(self.read_side.calendar.for_expectations(day)),
            known_instances=known_instances,
        )
        recorded = 0
        by_source: dict[str, int] = {}
        silent: dict[str, bool] = {}
        for item in expectations:
            by_source[item.source] = by_source.get(item.source, 0) + 1
            if self.ledger.has_expectation(item.expectation_id):
                continue
            if item.kind_token not in silent:
                silent[item.kind_token] = silenced(self.ledger.setting_of(item.kind_token))
            if not silent[item.kind_token] and self.ledger.record_expectation(item):
                recorded += 1
        return Formed(len(expectations), recorded, by_source)


MIGRATING = "词表迁移还没做完"
NO_TABLE = "还没有机会表"
STALE_TABLE = "今天这一晚的机会表还没有"
LINE_UNFINISHED = "今天这一晚的线还没跑完"
LEDGER_UNREADABLE = "夜线的 Job 账读不了"


def held_by_migration(migrating: Callable[[], bool] | None) -> str | None:
    """词表迁移做到一半时不建（归类一半新编号一半旧编号）：返回没建的原因，不挡是 None。"""

    return MIGRATING if migrating is not None and migrating() else None


def tonight_table(
    store: OpportunityStore,
    lane: str,
    today: date,
    *,
    line_finished: Callable[[date], bool | None] | None = None,
) -> tuple[OpportunityTable | None, str | None]:
    """盘上最晚那张表与"能不能用它建预期"：没有表 → (None, 原因)；表不是今天这一晚的，或给了 ``line_finished``（白天那一拍）
    而今天这一晚的线还没整条跑完（没跑到、表写出后才停）或夜线的 Job 账读不了 → (那张表, 原因)——递送照用它，建预期等今晚的线跑完（裁定 94 第 1 条）；
    否则 → (表, None)。线上的日子预期那一段自己就在线里，不给 ``line_finished``。"""

    table = store.latest(lane)
    if table is None:
        return None, NO_TABLE
    if table.night != today:
        return table, STALE_TABLE
    if line_finished is None:
        return table, None
    try:
        finished = line_finished(today)
    except NightJobsError:
        # 账坏了：不当跑完、也不当没跑完，不建预期并报出原因（递送照用表；健康面另报读不了）
        return table, LEDGER_UNREADABLE
    return (table, None) if finished is True else (table, LINE_UNFINISHED)


def observe_foresight(
    observer: Observer,
    operation: str,
    status: ObservationStatus,
    attributes: dict[str, str | int | float | bool],
    started: float,
) -> None:
    """记一笔预测层的观测；观测本身出错不许影响白天那一拍与夜里那条线。"""

    try:
        observer.record(
            ObservationEvent(
                category="foresight",
                operation=operation,
                status=status,
                duration_seconds=max(0.0, time.monotonic() - started),
                attributes=attributes,
            )
        )
    except Exception:  # noqa: BLE001 - 观测不许影响这一拍
        pass


__all__ = [
    "LINE_UNFINISHED",
    "MIGRATING",
    "NO_TABLE",
    "STALE_TABLE",
    "ExpectationForming",
    "Formed",
    "FormingReadSide",
    "Recent",
    "held_by_migration",
    "observe_foresight",
    "tonight_table",
]
