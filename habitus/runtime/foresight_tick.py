"""预测拍（方案第四节；实时化方案第 5、6 节），三步：读材料 → 此刻对得上历史的事件、情境预期建好、落账（不管说不说）→ 到了
时窗的按账的级决定说不说（``runtime.foresight_delivery``，没接时只建不递）。

预测拍只走事实（裁定 94 第 1 条）：结算、回话、账的读数，事件与情境在换代定下的关系与日历上实例化；**不建日子预期**——
当天的日子预期一天只建一次，是换代的最后几步之一（``runtime.foresight_days``），这里不按"日期变了"自己再建一份。读材料与落账
的那一段与换代共用（``runtime.foresight_forming``）。

**今天这一代的表写出之前不建任何预期**（白天那道闸，实时化方案第 3 节）：盘上最晚那张表不是今天这一代的，这一拍只递已经建好的、
不建新的（拿缺昨天的表会算出错的身份）。

**往回看的起点是落盘的水位**（``runtime.generation_watermark``，实时化缺陷 1；取代原来只在内存里的 ``now - 一拍``）：今天这一代
第一次建预期的那一拍从今天零点起看（补建，裁定 94 第 2 条、裁定 49：零点到这一拍之间关掉的时窗也算赶上了，做了的和没做的都建），
之后从上一次建预期看到的那一刻（``formed_through``）起看——中间合盖、关机几个小时，醒来那一拍照样把那几个小时里关掉的时窗
两边都建上。补建抓不回来的：「同一条链」那一段的事件预期只认此刻离前因做完不超过转移窗口的链（``triggers.events``），零点后
做完、到补建那一拍已经过了转移窗口的前因，那一段的预期补不回来（这条链此刻落在「当天剩下的」那一段：那一段另有成立的关系才建，
是另一条预期）。「当天剩下的」「次日」「第 2–7 天」与情境（日历）的都按天或按今天的链判，补建时抓得回来。

**没在看的时段**（实时化方案第 6 节）：每拍开头读水位，离系统确知在看的那一刻（``ticked_at`` 与最后一段空白的结尾里晚的那个）
超过容差（``unwatched_tolerance_slots`` 槽），就把这一段交会话一支写成一条「未观测」空判断（由归约落树，这里不碰行为树）；写不成
这一拍报出、水位不推（下一拍再写，不会漏）。第一次启动、盘上没有水位时不写（不知道之前在不在跑）。

整拍持预测根租约（与换代写账本的那几步互斥，两个进程挂同一个根也不重复建同一身份）。词表迁移做到一半时整拍不做（归类一半新编号
一半旧编号，结算那一拍同一口径），水位照推（那一拍系统在看）。建不出来（派生树读不到、装配失败）留观测、抛出，让调用方决定
下一拍。账本里读不出来的文件条数进观测。
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from contextlib import AbstractAsyncContextManager, nullcontext
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
from habitus.runtime.generation_lease import Lease, LeaseBusyError
from habitus.runtime.generation_watermark import TickTrack, TickWatermark
from habitus.scene.views import DayIndexCache

#: 递送只为「没读懂」与「错误」不说（都是在看却看不到；「未观测」照常说，裁定 68 第 2 条、94 第 3 条）。
_UNREADABLE = UNSEEN_GAPS
#: 预测根租约被另一个持有者（换代写账本、另一个进程的预测拍）拿着、等到时限也没拿到：这一拍不做，水位不推。
LEDGER_BUSY = "预测根租约被占"


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
    #: 新写进账本的、按触发种类分；这一拍是不是补建（往回看的起点早于此刻一拍以上）。
    recorded_by_source: dict[str, int] = field(default_factory=dict)
    catch_up: bool = False
    #: 这一拍开头写成「未观测」的那一段（没有是 None）。
    unwatched: tuple[datetime, datetime] | None = None
    #: 这一拍离上一拍（落盘的系统确知在看到的那一刻）隔了几拍：正常 1，上一拍拖过了槽边界、被挤掉一拍是 2；不知道（第一次
    #: 启动、没接水位）是 None。被挤掉的拍在 ``lateness_seconds`` 里看不到（常驻循环不补跑拖过的边界）。
    ticks_since_previous: int | None = None


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
        trace: Callable[[date], TraceSink] | None = None,
        track: TickTrack | None = None,
        lease: Lease | None = None,
        finished: Callable[[date], bool | None] | None = None,
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
        #: 这一拍的 span 落哪儿（``runtime.generation_trace.TraceFiles.sink``，按当地日子一天一份）；没接就不记 span。
        self.trace = trace
        #: 落盘的水位与没在看的时段（``TickTrack``）；没接就按一拍的宽度往回看、不写没在看的时段。预测根租约整拍持有。
        self.track, self.lease = track, lease
        #: 今天这一代完成了没有（完成标记）：白天那道闸要它为真才建预期（F5）；没接就只看表是不是今天的。
        self.finished = finished

    async def run_once(self, *, scheduled: datetime | None = None) -> TickReport:
        """跑一拍。``scheduled`` 是这一拍该开始的槽边界（常驻 worker 给）：span 上记离它晚了多少。"""

        started = time.monotonic()
        now = self._clock().astimezone(self.read_side.zone)
        traced = open_trace(self.trace(now.date())) if self.trace is not None else nullcontext()
        with traced, span("foresight.tick", {"at": now.isoformat(), "lane": self.lane}) as attributes:
            if scheduled is not None:
                # 这一拍离它该开始的槽边界晚了多少（实时化方案第 10 节：预测拍不被卡）
                attributes["lateness_seconds"] = max(0.0, (now - scheduled).total_seconds())
            try:
                async with self._held():
                    report = await self._run(now)
            except LeaseBusyError:
                report = TickReport(now, 0, 0, skipped=LEDGER_BUSY)
            except Exception as exc:
                observe_foresight(
                    self.observer, "tick", ObservationStatus.FAILURE, {"error_type": type(exc).__name__}, started
                )
                raise
            # 白天那道闸放没放行、为什么，以及建了几条、按种类、递了几条、这一拍用了多久
            readings = {
                **_attributes(report),
                **{key: attributes[key] for key in ("lateness_seconds",) if key in attributes},
            }
            set_attributes(
                attributes, {"gate_open": report.skipped is None, "tick_seconds": time.monotonic() - started}
            )
            set_attributes(attributes, readings)
        observe_foresight(self.observer, "tick", ObservationStatus.SUCCESS, readings, started)
        return report

    def _held(self) -> AbstractAsyncContextManager[object]:
        return self.lease.held() if self.lease is not None else nullcontext()

    async def _run(self, now: datetime) -> TickReport:
        mark, gap = await self.track.begin(now) if self.track is not None else (TickWatermark(), None)
        previous = gap[0] if gap is not None else mark.watched_until  # 刚写了「未观测」：``begin`` 已把水位推到此刻
        ticks = None if previous is None or self.track is None else round((now - previous) / self.track.slot)
        held = held_by_migration(self.migrating)
        if held is not None:
            await self._advance(mark.ticked(now))
            return TickReport(now, 0, 0, skipped=held, unwatched=gap, ticks_since_previous=ticks)
        materials = await self._read(now)
        formed, looked_since = self._form(materials, mark)
        if looked_since is not None:
            mark = mark.formed(now, now.date())
            await self._advance(mark)
        delivered = await self._deliver(materials, None if gap is not None else previous)
        await self._advance(mark.ticked(now))
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
            recorded_by_source=formed.recorded_by_source,
            catch_up=looked_since is not None and self.tick is not None and looked_since < now - self.tick,
            unwatched=gap,
            ticks_since_previous=ticks,
        )

    async def _advance(self, mark: TickWatermark) -> None:
        if self.track is not None:
            await self.track.write(mark)

    # ── 第一步：读 ──────────────────────────────────────────────────────────

    async def _read(self, now: datetime) -> _Materials:
        recent = await self.forming.read(now)
        # 今天这一代的表还没出来：递送照常用盘上最晚那张，建预期等表出来（白天那道闸）
        table, skipped = tonight_table(self.opportunities, self.lane, now.date(), finished=self.finished)
        return _Materials(recent, table, skipped)

    # ── 第二步：建预期 ──────────────────────────────────────────────────────

    def _form(self, materials: _Materials, mark: TickWatermark) -> tuple[Formed, datetime | None]:
        """建预期；返回建的结果与这一拍从哪一刻起看（没建是 None）。"""

        if materials.table is None or materials.skipped is not None:
            return Formed(), None
        now = materials.now
        looked_since = self._looked_since(now, mark, materials.table.night)
        formed = self.forming.form(
            materials.recent,
            materials.table,
            batch=Batch.MOMENT,
            looked_since=looked_since,
            known_instances=self._known_instances(now),
        )
        return formed, looked_since

    def _looked_since(self, now: datetime, mark: TickWatermark, generation: date) -> datetime:
        """这一拍从哪一刻起看（``TickTrack.looked_since``）；没接水位（不知道上一次看到哪儿）就每拍都从零点起看——同一身份只建一次，
        多看不多建。"""

        if self.track is not None:
            return self.track.looked_since(now, mark, generation)
        return datetime.combine(now.date(), clock_time(0), now.tzinfo)

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

    async def _deliver(self, materials: _Materials, previous: datetime | None) -> DeliveryReport | None:
        """到了时窗的决定说不说。没有新预期也要跑：昨天建的、今天才到时窗的在等。``previous`` 见 ``delivery_since``。"""

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
                None if self.reread is None else _reread_for(self.reread, now),
                seen,
                delivery_since(
                    now, previous, None if self.track is None else self.track.slot * self.track.tolerance_slots
                ),
            ),
        )

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


def delivery_since(now: datetime, previous: datetime | None, reach: timedelta | None) -> datetime | None:
    """递送这一拍的区间从哪一刻起（UTC）：max(上一拍落盘的系统确知在看到的那一刻, 此刻 − ``reach``)，``reach`` = 「未观测」
    容差槽数 × 一拍（``foresight.unwatched_tolerance_slots``，默认 2；用户 11-10「跟着容差走」）。

    正常一拍隔一槽，起点就是上一拍；上一拍被模型拖过了槽边界、后面几拍没跑时，这一拍把被挤掉的那几拍的区间一起看了——在那几拍
    里开、又在那几拍里关的时窗照现有判法决定（过了最晚时限的不说、只记账，结算照常）。最多往回多看「容差 − 1」拍：在看 / 没在看
    用同一条容差线划分（空当 ≤ 容差算在看，都补得到；超过就写「未观测」，不补）。不知道上一拍（第一次启动、没接水位）、或这一拍
    开头刚写了「未观测」（tick 交 None）时是 None：递送从此刻 − 1 拍起，没在看的时段不补说（21 号提示词第 4 节第 7 条）。"""

    if previous is None or reach is None:
        return None
    return max(previous.astimezone(UTC), now.astimezone(UTC) - reach)


def _reread_for(
    reread: Callable[[datetime, datetime], Awaitable[bool]], now: datetime
) -> Callable[[Expectation], Awaitable[bool]]:
    async def run(item: Expectation) -> bool:
        return await reread(item.observed_from, now)

    return run


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
    """观测事件带的读数：建了几条（按种类，补建的单列）、递了几条（合盖后补说的单列）、没在看的那一段、模型问了几个调了几次、
    两遍不一致几次、名额与不可达各几条、坏文件几份。"""

    gap = report.unwatched
    found: dict[str, str | int | float | bool] = {
        "formed": report.formed,
        "recorded": report.recorded,
        "decided": report.decided,
        "spoken": report.spoken,
        "skipped": report.skipped or "",
        "unreadable": report.unreadable,
        "catch_up": report.catch_up,
        # 合盖、关机之后醒来的这一拍说出口的（这一拍开头写了一段「未观测」）
        "spoken_after_gap": report.spoken if gap is not None else 0,
        "unwatched_spans": 0 if gap is None else 1,
        "unwatched_minutes": 0.0 if gap is None else round((gap[1] - gap[0]).total_seconds() / 60, 3),
        # 离上一拍隔了几拍（被挤掉一拍是 2；不知道是 -1）
        "ticks_since_previous": -1 if report.ticks_since_previous is None else report.ticks_since_previous,
        **{f"source_{key}": value for key, value in report.by_source.items()},
        **{f"recorded_{key}": value for key, value in report.recorded_by_source.items()},
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
