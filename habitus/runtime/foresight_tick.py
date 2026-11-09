"""白天那一拍（方案第四节），三步：读材料 → 此刻对得上历史的事件、情境预期建好、落账（不管说不说）→ 到了时窗的按账的级决定
说不说（``runtime.foresight_delivery``，没接时只建不递）。

白天那一拍只走事实（裁定 94 第 1 条）：结算、回话、账的读数，事件与情境在夜里定下的关系与日历上实例化；**不建日子预期**——
当天的日子预期一天只建一次，是夜里那条线的最后一步（``runtime.foresight_days``），白天不按"日期变了"自己再建一份。读材料与落账
的那一段与夜里共用（``runtime.foresight_forming``）。

**今晚的线跑完之前不建任何预期**（情境、事件也一样，裁定 94 第 1 条：只有整条线跑完，这一晚才算跑过）：今天这一晚的线还没
跑完（零点到夜里那条线跑完之间、线被拖住或停下的那一天——含表写出后才停的），这一拍只递已经建好的、不建新的；停掉的那天白天也
不建，与结算"停掉那天按看不到"读同一个判断（夜线 Job 账的 ``NightJobs.finished``）。**线跑完后的下一拍补建**：这一拍从今天零点起看
（``looked_since`` = 零点），零点到这一拍之间关掉的时窗也算赶上了，照建、照结（裁定 49 第 1 条）；之后照常按一拍的宽度看。
"这一晚的表已经补建过"只记在这一拍的对象上（进程重启后的第一拍再从零点看一遍：同一身份只建一次，多看不多建）。
补建抓不回来的：「同一条链」那一段的事件预期只认此刻离前因做完不超过转移窗口的链（``triggers.events``），零点后做完、到补建
那一拍已经过了转移窗口的前因，那一段的预期补不回来（这条链此刻落在「当天剩下的」那一段：那一段另有成立的关系才建，是另一条
预期）。「当天剩下的」「次日」「第 2–7 天」与情境（日历）的都按天或按今天的链判，补建时抓得回来。

词表迁移做到一半时整拍不做（归类一半新编号一半旧编号，结算那一拍同一口径）。没有今天这一晚的机会表时不建预期、只递已经建好的；
建不出来（派生树读不到、装配失败）留观测、抛出，让调用方决定下一拍。账本里读不出来的文件条数进观测。
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from datetime import time as clock_time

from habitus.behavior.schema.vocabulary import UNSEEN_GAPS
from habitus.behavior.uri import BehaviorURI
from habitus.conditions import day_note
from habitus.foresight import Batch, Seen, UnsealedReader, UnsealedRow
from habitus.foresight.delivery import done_since
from habitus.foresight.ladder import Tolerance
from habitus.foresight.ledger import Expectation
from habitus.foresight.opportunities import OpportunityTable
from habitus.foresight.triggers.assemble import LOOKBACK_DAYS
from habitus.foundation.intervals import standing_pieces
from habitus.foundation.observability import NullObserver, ObservationStatus, Observer
from habitus.foundation.tracing import TraceSink, open_trace, set_attributes, span
from habitus.runtime.foresight_delivery import DeliveryReport, DeliveryStage, JudgeMaterials
from habitus.runtime.foresight_forming import (
    ExpectationForming,
    Formed,
    FormingReadSide,
    Recent,
    held_by_migration,
    observe_foresight,
    tonight_table,
)
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.scene.views import DayIndexCache

#: 递送只为「没读懂」与「错误」不说（都是在看却看不到；「未观测」照常说，裁定 68 第 2 条、94 第 3 条）。
_UNREADABLE = UNSEEN_GAPS


@dataclass(frozen=True)
class TickReport:
    at: datetime
    #: 此刻对得上历史的预期有几条、其中新写进账本的几条、按触发种类分。
    formed: int
    recorded: int
    by_source: dict[str, int] = field(default_factory=dict)
    #: 这一拍为什么没建（建了是 None）。
    skipped: str | None = None
    #: 到了时窗的预期决定了几条、其中说出口的几条（没接递送时都是 0）。
    decided: int = 0
    spoken: int = 0
    #: 递送那一拍的明细（没接递送是 None）。
    delivery: DeliveryReport | None = None
    #: 账本里读不出来、被跳过的文件数。
    unreadable: int = 0


@dataclass(frozen=True)
class _Materials:
    """这一拍读到的东西：最近几天、机会表（没有时为 None，原因在 ``skipped``）。"""

    recent: Recent
    table: OpportunityTable | None
    skipped: str | None

    @property
    def now(self) -> datetime:
        return self.recent.now


class ForesightTick:
    def __init__(
        self,
        ledger: ForesightLedgerStore,
        opportunities: OpportunityStore,
        read_side: FormingReadSide,
        unsealed: UnsealedReader,
        *,
        lane: str = "session",
        shadow_unestablished: bool = False,
        vocabulary_version: Callable[[], int] | None = None,
        migrating: Callable[[], bool] | None = None,
        observer: Observer | None = None,
        clock: Callable[[], datetime] | None = None,
        delivery: DeliveryStage | None = None,
        tick_seconds: float | None = None,
        recent: int,
        tolerance: Tolerance,
        reread: Callable[[datetime, datetime], Awaitable[bool]] | None = None,
        line_finished: Callable[[date], bool | None] | None = None,
        trace: Callable[[date], TraceSink] | None = None,
    ) -> None:
        self.forming = ExpectationForming(
            ledger,
            read_side,
            unsealed,
            recent=recent,
            tolerance=tolerance,
            shadow_unestablished=shadow_unestablished,
            vocabulary_version=vocabulary_version,
        )
        if not lane:
            raise ValueError("lane must be non-empty text")
        self.ledger = ledger
        self.opportunities = opportunities
        self.read_side = read_side
        self.lane = lane
        self.migrating = migrating
        if delivery is not None and delivery.ledger is not ledger:
            raise TypeError("delivery must be a DeliveryStage on the same ledger")
        self.delivery = delivery
        if tick_seconds is not None and tick_seconds <= 0:
            raise ValueError("tick_seconds must be a positive number or None")
        #: 一拍多长：此刻往前这么久关的时窗这一拍还算赶上（拍点晚了一点不丢没做的那次机会）。
        self.tick = timedelta(seconds=float(tick_seconds)) if tick_seconds is not None else None
        self.observer: Observer = observer if observer is not None else NullObserver()
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)
        #: 时窗末当场重读那段「没读懂」「错误」的口（会话一支的重试堆，裁定 68 第 1 条）；没接就是 None。
        self.reread = reread
        #: 夜里那条线自己记的"这一晚跑完了没有"（与结算认停掉的那天同一个判断，夜线 Job 账的 ``NightJobs.finished``）：今天这一晚的线跑完了
        #: 才建预期（裁定 94 第 1 条）；没接就只看表是不是今天这一晚的。
        self.line_finished = line_finished
        #: 用今天这一晚的表补建过的那一天（表出来后第一拍从零点起看，之后按一拍看）。
        self._caught_up: date | None = None
        #: 这一拍的 span 落哪儿（``runtime.nightly_trace.TraceFiles.sink``，按当地日子一天一份）；没接就不记 span。
        self.trace = trace

    async def run_once(self) -> TickReport:
        started = time.monotonic()
        now = self._clock().astimezone(self.read_side.zone)
        traced = open_trace(self.trace(now.date())) if self.trace is not None else nullcontext()
        with traced, span("foresight.tick", {"at": now.isoformat(), "lane": self.lane}) as attributes:
            try:
                report = await self._run()
            except Exception as exc:
                observe_foresight(
                    self.observer, "tick", ObservationStatus.FAILURE, {"error_type": type(exc).__name__}, started
                )
                raise
            # "今晚的表"闸放没放行、为什么（裁定 94 第 1 条），以及建了几条、按种类、递了几条
            set_attributes(attributes, {"gate_open": report.skipped is None, **_attributes(report)})
        observe_foresight(self.observer, "tick", ObservationStatus.SUCCESS, _attributes(report), started)
        return report

    async def _run(self) -> TickReport:
        now = self._clock().astimezone(self.read_side.zone)
        held = held_by_migration(self.migrating)
        if held is not None:
            return TickReport(now, 0, 0, skipped=held)
        materials = await self._read(now)
        formed = self._form(materials)
        delivered = await self._deliver(materials)
        return TickReport(
            now,
            formed.formed,
            formed.recorded,
            formed.by_source,
            materials.skipped,
            delivered.decided if delivered is not None else 0,
            delivered.spoken if delivered is not None else 0,
            delivered,
            len(self.ledger.take_unreadable())
            + len(self.opportunities.take_unreadable())
            + (delivered.unreadable if delivered is not None else 0),
        )

    # ── 第一步：读 ──────────────────────────────────────────────────────────

    async def _read(self, now: datetime) -> _Materials:
        recent = await self.forming.read(now)
        # 今晚的表还没出来：递送照常用盘上最晚那张，建预期等表出来（裁定 94 第 1 条）
        table, skipped = tonight_table(self.opportunities, self.lane, now.date(), line_finished=self.line_finished)
        return _Materials(recent, table, skipped)

    # ── 第二步：建预期 ──────────────────────────────────────────────────────

    def _form(self, materials: _Materials) -> Formed:
        if materials.table is None or materials.skipped is not None:
            return Formed()
        now = materials.now
        formed = self.forming.form(
            materials.recent,
            materials.table,
            batch=Batch.MOMENT,
            looked_since=self._looked_since(now),
            known_instances=self._known_instances(now),
        )
        self._caught_up = now.date()
        return formed

    def _looked_since(self, now: datetime) -> datetime | None:
        """这一拍从哪一刻起看：今天这一晚的表出来后的第一拍从零点起（补建，裁定 94 第 1 条），之后按一拍的宽度。"""

        if self._caught_up != now.date():
            return datetime.combine(now.date(), clock_time(0), tzinfo=now.tzinfo)
        return now - self.tick if self.tick is not None else None

    def _known_instances(self, now: datetime) -> Callable[[str], tuple[str, ...]]:
        """账 → 这本账上已经建过的事件预期的触发实例（还要看的、回看窗里的日子；问到才读，一拍只读一遍）：前因的开始被更正后，
        同一次机会沿用原来的实例（``form_expectations`` 的 ``known_instances``）。"""

        found: dict[str, list[str]] | None = None
        since = now.date() - timedelta(days=LOOKBACK_DAYS)

        def of(account: str) -> tuple[str, ...]:
            nonlocal found
            if found is None:
                found = {}
                for day in self.ledger.open_days():
                    if day < since:
                        continue
                    for item in self.ledger.expectations_on(day):
                        if item.source == "事件":
                            found.setdefault(item.account, []).append(item.trigger_instance)
            return tuple(found.get(account, ()))

        return of

    # ── 第三步：递送 ────────────────────────────────────────────────────────

    async def _deliver(self, materials: _Materials) -> DeliveryReport | None:
        """到了时窗的决定说不说。没有新预期也要跑：昨天建的、今天才到时窗的在等。"""

        if self.delivery is None:
            return None
        now, seen = materials.now, materials.recent.seen
        current = self.delivery.current
        return await self.delivery.run_once(
            now,
            JudgeMaterials(
                materials.table,
                lambda uris: self._rows_for(uris, materials.recent.cache, materials.recent.unsealed, seen, now),
                lambda item: done_since(item, seen, now=now, current=current),
                lambda day: day_note(self.read_side.calendar, day),
                lambda item: _unobserved(item, materials.recent, now),
                None if self.reread is None else self._reread_for(now),
                seen,
            ),
        )

    def _reread_for(self, now: datetime) -> Callable[[Expectation], Awaitable[bool]]:
        reread = self.reread
        assert reread is not None

        async def run(item: Expectation) -> bool:
            return await reread(item.observed_from, now)

        return run

    @staticmethod
    def _rows_for(
        uris: Iterable[str],
        cache: DayIndexCache,
        unsealed: Sequence[UnsealedRow],
        seen: Sequence[Seen],
        now: datetime,
    ) -> dict[str, Seen]:
        """按出处取行：最近几天的在 ``seen`` 里已经填全（按出处，也按开始时刻——事件触发的实例是前因的开始时刻）；
        更早的已封口记录按 URI 里的日子到行为树取。"""

        found: dict[str, Seen] = {}
        recent = {row.source: row for row in seen}
        recent.update({row.started_at.astimezone(UTC).isoformat(): row for row in seen})
        for uri in uris:
            if uri in recent:
                found[uri] = recent[uri]
                continue
            try:
                day = BehaviorURI.parse(uri).to_address().occurred_on
            except (TypeError, ValueError):
                # 不是行为树 URI 的出处（``unsealed:<开始时刻>``）只在最近几天的 ``seen`` 里有，过了就取不到行
                continue
            sealed = cache.day(day).by_uri.get(uri)
            if sealed is not None:
                found[uri] = Seen(
                    sealed.at,
                    sealed.last_observed_at,
                    sealed.kind_token,
                    uri,
                    sealed.name,
                    sealed.summary,
                    place=sealed.place,
                    subjects=sealed.subjects,
                )
        return found


def _unobserved(item: Expectation, recent: Recent, now: datetime) -> bool:
    """说中的范围（起算 → 此刻）里有没有「没读懂」或「错误」（都是看不到；「未观测」照常说，裁定 68 第 2 条、94 第 3 条）：已落树的（被行为作废的
    已去掉，与结算同一个读法），加上还没封口的「没读懂」判断；还没封口的可读行为从那段里开头的，也把它作废——白天当场
    重读补回来、还没归约的那一条就是这样（裁定 68 第 1 条）。作废用的是空白的那一个读法（``foundation.intervals.standing_pieces``：
    按天切开，每天那段由那一天的行为判）。"""

    start, end = item.observed_from.astimezone(UTC), now.astimezone(UTC)
    stretches = [
        (gap.started_at, gap.ended_at) for gap in recent.cache.gaps_within(item.observed_from, now, kinds=_UNREADABLE)
    ]
    stretches += [
        (row.started_at, row.last_observed_at)
        for row in recent.unsealed
        if row.name is None and row.started_at.astimezone(UTC) < end and row.last_observed_at.astimezone(UTC) > start
    ]
    readable: dict[date, list[datetime]] = {}
    for row in recent.unsealed:
        if row.name is not None:
            readable.setdefault(row.started_at.date(), []).append(row.started_at)
    for moments in readable.values():
        moments.sort(key=lambda moment: moment.astimezone(UTC))
    return any(
        begin.astimezone(UTC) < end and finish.astimezone(UTC) > start
        for opened, closed in stretches
        for begin, finish in standing_pieces(opened, closed, voidable=True, starts_on=lambda day: readable.get(day, ()))
    )


def _attributes(report: TickReport) -> dict[str, str | int | float | bool]:
    """观测事件带的读数：建了几条、递了几条、模型问了几个调了几次、两遍不一致几次、名额与不可达各几条、坏文件几份。"""

    found: dict[str, str | int | float | bool] = {
        "formed": report.formed,
        "recorded": report.recorded,
        "decided": report.decided,
        "spoken": report.spoken,
        "skipped": report.skipped or "",
        "unreadable": report.unreadable,
        **{f"source_{key}": value for key, value in report.by_source.items()},
    }
    if report.delivery is not None:
        found["judged"] = report.delivery.judged
        found["level_changes"] = report.delivery.level_changes
        found["judge_calls"] = report.delivery.calls
        found["judge_inconsistent"] = sum(1 for signal in report.delivery.signals if "inconsistent" in signal)
        found["judge_failed"] = report.delivery.judge_failed
        found.update({f"reason_{key}": value for key, value in report.delivery.by_reason.items()})
    return found


__all__ = ["ForesightTick", "TickReport"]
