"""建预期、落账的那一段，两处共用（裁定 94 第 1 条）：预测拍（``runtime.foresight_tick``，事件与情境）与换代末尾的
当天日子预期（``runtime.foresight_days``）。

取材料：此刻是主体本地时刻（``read_side.zone``），今天的日历条件从 ``read_side.calendar`` 取；行为树经 ``scene.views`` 只读今天
与前 7 天、还没封口的判断经 ``UnsealedReader``。装配交给 ``foresight.form_expectations``（``Batch`` 定建哪几种触发）。写账本：
同一个身份已经在账上的不再写（身份不含槽号、时刻与日子，账本按身份索引，跨天重算得到同一条）；人说过「以后别提了」的行为不再建
（裁定 50、61）。进程不留内存状态：重启后重扫得到的都是已有的身份。

预测拍用哪张表也在这里（``usable_tables``，用户 2026-11-16「预测不要被换代卡住」）：建预期用盘上最晚一张那一代做完了的表，
多旧都用；递送照用盘上最晚那张。旧表下的日子预期只建表那一晚之后一次都没再出现过的类（``days_unchanged``）。观测怎么记
（``observe_foresight``）两处共用。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from datetime import time as clock_time
from typing import Protocol, runtime_checkable
from zoneinfo import ZoneInfo

from habitus.behavior.kinds.ids import Lane, pending_token
from habitus.foresight import Batch, Moment, Seen, UnsealedReader, UnsealedRow, form_expectations
from habitus.foresight.hierarchy import FLAT, KindHierarchy
from habitus.foresight.ladder import Tolerance, silenced
from habitus.foresight.opportunities import OpportunityTable
from habitus.foresight.triggers.assemble import LOOKBACK_DAYS, seen_rows
from habitus.foundation.observability import ObservationEvent, ObservationStatus, Observer
from habitus.runtime.foresight_implementation import IMPLEMENTATION
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.runtime.foresight_settlement import BehaviorReadSide
from habitus.runtime.foresight_tree_rows import day_cache, hierarchy_of
from habitus.runtime.generation_marks import GenerationMarkError
from habitus.scene.views import DayIndexCache
from habitus.temporal.slots import slot_index


@runtime_checkable
class FormingReadSide(BehaviorReadSide, Protocol):
    """建预期要读的那棵行为树、主体、日历（与结算同一侧）外加时区（预测层的装配器 ``ForesightReadSide`` 就是这个形状）。"""

    @property
    def zone(self) -> ZoneInfo: ...


@dataclass(frozen=True)
class Recent:
    """这一刻读到的最近几天：此刻、行为树的按天索引、还没封口的判断、两者合起来按开始时刻排好的行，以及此刻的上下级（读的时候
    取一份，这一拍建预期与递送都用这一份，裁定 101 第 8 条）。"""

    now: datetime
    cache: DayIndexCache
    unsealed: tuple[UnsealedRow, ...]
    seen: tuple[Seen, ...]
    kinds: KindHierarchy = FLAT


@dataclass(frozen=True)
class Formed:
    #: 此刻对得上历史的预期有几条、其中新写进账本的几条、按触发种类分（对得上的、新写进的各一份）。
    formed: int = 0
    recorded: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    recorded_by_source: dict[str, int] = field(default_factory=dict)


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

        kinds = hierarchy_of(self.read_side)  # 这一拍开头取一份，之后长出来的候选这一拍不认（C27）
        cache = day_cache(self.read_side)
        since = datetime.combine(now.date() - timedelta(days=LOOKBACK_DAYS), clock_time(0), tzinfo=now.tzinfo)
        await self.unsealed.prepare(until=now)
        unsealed = self.unsealed.rows(since=since, until=now)
        return Recent(now, cache, unsealed, seen_rows(Moment(at=now, slot=0), cache, unsealed), kinds)

    def snapshot(self, now: datetime) -> Recent:
        """此刻往回看的几天，不调模型（宿主来取提醒那一刻用，``runtime.foresight_claim_check``）：与 ``read`` 同一个读法，
        物理一支还没选上词的判断不归类。不改写任何共用状态，可以在预测拍之外的线程里调。"""

        kinds, cache = hierarchy_of(self.read_side), day_cache(self.read_side)
        since = datetime.combine(now.date() - timedelta(days=LOOKBACK_DAYS), clock_time(0), tzinfo=now.tzinfo)
        unsealed = self.unsealed.rows(since=since, until=now)
        return Recent(now, cache, unsealed, seen_rows(Moment(at=now, slot=0), cache, unsealed), kinds)

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
                kinds=recent.kinds,
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
        recorded_by_source: dict[str, int] = {}
        silent: dict[str, bool] = {}
        for item in expectations:
            by_source[item.source] = by_source.get(item.source, 0) + 1
            if self.ledger.has_expectation(item.expectation_id):
                continue
            if item.kind_token not in silent:
                silent[item.kind_token] = silenced(self.ledger.setting_of(item.kind_token))
            if silent[item.kind_token]:
                continue
            standing = self.ledger.level_of(item.account)
            if self.ledger.record_expectation(replace(item, level=standing.level if standing is not None else 0)):
                recorded += 1
                recorded_by_source[item.source] = recorded_by_source.get(item.source, 0) + 1
        return Formed(len(expectations), recorded, by_source, recorded_by_source)

    def days_unchanged(self, recent: Recent, table: OpportunityTable) -> OpportunityTable | None:
        """旧表下能建日子预期的那几类（用户 2026-11-16）：日子预期的身份与时窗全从表里"上次是哪天、哪条"算，旧表与今天的表
        对这一类读到的历史不同，就会算出不同的身份、把做过的当成久没做（说错）。所以只留历史没变的类：

        - 表那一晚到昨天，行为树与还没封口的判断里一次都没出现过（连上级一起算）；
        - 表那一晚之前的 ``LOOKBACK_DAYS`` 天里，此刻树上出现的日子与表里记的相同——表写成之后才落进这几天的记录（就近重试写回、
          物理一支的映射）会让它对不上（阶段 C 问题 1）。

        这几天里有还没定下是哪一类的（「待定」、还没选上词）就认不出，返回 None、这一拍不建日子预期。词表换算口变了（删词后旧记录
        改读成上级）由调用方比词表版本挡（``ForesightTick``）。"""

        now = recent.now
        today = now.date()
        since = table.night - timedelta(days=LOOKBACK_DAYS)
        after: list[str] = []
        before: dict[str, set[date]] = {}
        pending = {pending_token(lane) for lane in Lane}
        undecided = False

        def saw(day: date, tokens: list[str]) -> None:
            nonlocal undecided
            undecided = undecided or not pending.isdisjoint(tokens)
            if day >= table.night:
                after.extend(tokens)
            else:
                for kind in recent.kinds.done(tokens):
                    before.setdefault(kind, set()).add(day)

        day = since
        while day < today:
            saw(day, [row.kind_token for row in recent.cache.day(day).rows])
            day += timedelta(days=1)
        for row in self.unsealed.rows(
            since=datetime.combine(since, clock_time(0), tzinfo=now.tzinfo),
            until=datetime.combine(today, clock_time(0), tzinfo=now.tzinfo),
        ):
            if row.name is not None and row.kind_token is None:
                return None
            if row.kind_token is not None:
                saw(row.started_at.astimezone(now.tzinfo).date(), [row.kind_token])
        if undecided:
            return None
        seen = recent.kinds.done(after)
        return replace(
            table,
            kinds=tuple(
                kind
                for kind in table.kinds
                if kind.kind_token not in seen
                and {item.day for item in kind.days if item.day >= since} == before.get(kind.kind_token, set())
            ),
        )


NO_TABLE = "还没有机会表"
NO_FINISHED_TABLE = "还没有做完的那一代的机会表"
MARK_UNREADABLE = "完成标记读不了"
AHEAD_TABLE = "最晚那张机会表在今天之后"


@dataclass(frozen=True)
class Tables:
    """这一拍用的表（用户 2026-11-16「预测不要被换代卡住」）：``latest`` 是盘上最晚那张，递送照用它；``forming`` 是建预期用的
    那张——盘上最晚一张读得出来、那一代做完了（有完成标记）的，多旧都用；没有时是 None，``skipped`` 说为什么。"""

    latest: OpportunityTable | None
    forming: OpportunityTable | None
    skipped: str | None = None


def usable_tables(
    store: OpportunityStore,
    lane: str,
    today: date,
    *,
    finished: Callable[[date], bool | None] | None = None,
) -> Tables:
    """递送用的表与建预期用的表（``Tables``）。建预期只认那一代做完了的：今天的表写出了、日子预期或完成标记没成（下一眼重做
    整代会重写它）不用，往前找上一张；某一晚的完成标记坏了也不用那一张、往前找（坏标记健康面另报）。最晚那张在今天之后（往西
    改了时区，裁定 96 第 11 条）不建。没给 ``finished``（没接换代）就用最晚那张。"""

    latest = store.latest(lane)
    if latest is None:
        return Tables(None, None, NO_TABLE)
    if latest.night > today:
        return Tables(latest, None, AHEAD_TABLE)
    if finished is None:
        return Tables(latest, latest)
    unreadable = False

    def done(night: date) -> bool:
        nonlocal unreadable
        try:
            return finished(night) is True
        except GenerationMarkError:
            unreadable = True  # 标记坏了：不当做完、也不当没做，这一张不用
            return False

    newest = latest.night
    forming = latest if done(newest) else store.latest(lane, accept=lambda night: night < newest and done(night))
    if forming is None:
        return Tables(latest, None, MARK_UNREADABLE if unreadable else NO_FINISHED_TABLE)
    return Tables(latest, forming)


def observe_foresight(
    observer: Observer,
    operation: str,
    status: ObservationStatus,
    attributes: dict[str, str | int | float | bool],
    started: float,
) -> None:
    """记一笔预测层的观测；观测本身出错不许影响预测拍与换代。"""

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
    "AHEAD_TABLE",
    "MARK_UNREADABLE",
    "NO_FINISHED_TABLE",
    "NO_TABLE",
    "ExpectationForming",
    "Formed",
    "FormingReadSide",
    "Recent",
    "Tables",
    "observe_foresight",
    "usable_tables",
]
