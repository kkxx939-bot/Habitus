"""白天那一拍（方案第四节），三步：读材料 → 此刻对得上历史的预期全部建好、落账（不管说不说）→ 到了时窗的按账的级决定说不说
（``runtime.foresight_delivery``，没接时只建不递）。

组合根只做几件只有它知道的事：此刻是主体的本地时刻（``read_side.zone``）；读派生树——预测树取正在服务的那一代、机会表取这条
lane 盘上最晚一晚、行为树经 ``scene.views`` 只读今天与前 7 天、还没封口的判断经 ``UnsealedReader``；写账本——同一个身份已经在账上
的不再写（身份不含槽号、时刻与日子，账本按身份索引，跨天重算得到同一条）；人说过「以后别提了」的账不再建（裁定 50）。
进程不留内存状态：重启后重扫得到的都是已有的身份。

词表迁移做到一半时整拍不做（归类一半新编号一半旧编号，结算那一拍同一口径）。没有机会表或预测树还没发布时不建预期、只递昨天建的；
建不出来（派生树读不到、装配失败）留观测、抛出，让调用方决定下一拍。账本里读不出来的文件条数进观测。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from datetime import time as clock_time
from typing import Protocol, runtime_checkable
from zoneinfo import ZoneInfo

from habitus.behavior.uri import BehaviorURI
from habitus.foresight import ForesightError, Moment, Seen, UnsealedReader, UnsealedRow, form_expectations
from habitus.foresight.opportunities import OpportunityTable
from habitus.foresight.triggers.assemble import seen_rows
from habitus.foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer
from habitus.prediction.model import PredictionTree, SlotKey
from habitus.prediction.store import PredictionTreeStore, PredictionTreeStoreError
from habitus.runtime.foresight_delivery import DeliveryReport, DeliveryStage, JudgeMaterials
from habitus.runtime.foresight_implementation import IMPLEMENTATION
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.runtime.foresight_settlement import BehaviorReadSide
from habitus.scene.views import DayIndexCache
from habitus.series.reader import admitted


@runtime_checkable
class TickReadSide(BehaviorReadSide, Protocol):
    """这一拍要读的那棵行为树、主体、日历（与结算同一侧）外加时区（预测层的装配器 ``ForesightReadSide`` 就是这个形状）。"""

    @property
    def zone(self) -> ZoneInfo: ...


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
    """这一拍读到的东西：此刻、最近几天的行、机会表、预测树（后两样没有时为 None，原因在 ``skipped``）。"""

    now: datetime
    cache: DayIndexCache
    unsealed: tuple[UnsealedRow, ...]
    seen: tuple[Seen, ...]
    table: OpportunityTable | None
    tree: PredictionTree | None
    generation: str
    skipped: str | None


class ForesightTick:
    def __init__(
        self,
        ledger: ForesightLedgerStore,
        opportunities: OpportunityStore,
        prediction: PredictionTreeStore,
        read_side: TickReadSide,
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
    ) -> None:
        if not isinstance(ledger, ForesightLedgerStore):
            raise TypeError("ledger must be a ForesightLedgerStore")
        if not isinstance(opportunities, OpportunityStore):
            raise TypeError("opportunities must be an OpportunityStore")
        if not isinstance(prediction, PredictionTreeStore):
            raise TypeError("prediction must be a PredictionTreeStore")
        if not isinstance(read_side, TickReadSide):
            raise TypeError("read_side must expose the behaviour tree, subject, calendar and zone")
        if not hasattr(unsealed, "prepare") or not hasattr(unsealed, "rows"):
            raise TypeError("unsealed must be an UnsealedReader")
        if not isinstance(lane, str) or not lane:
            raise ValueError("lane must be non-empty text")
        if migrating is not None and not callable(migrating):
            raise TypeError("migrating must be callable or None")
        self.ledger = ledger
        self.opportunities = opportunities
        self.prediction = prediction
        self.read_side = read_side
        self.unsealed = unsealed
        self.lane = lane
        self.shadow_unestablished = bool(shadow_unestablished)
        self.vocabulary_version = vocabulary_version
        self.migrating = migrating
        if delivery is not None and (not isinstance(delivery, DeliveryStage) or delivery.ledger is not ledger):
            raise TypeError("delivery must be a DeliveryStage on the same ledger")
        self.delivery = delivery
        if tick_seconds is not None and (
            isinstance(tick_seconds, bool) or not isinstance(tick_seconds, int | float) or tick_seconds <= 0
        ):
            raise ValueError("tick_seconds must be a positive number or None")
        #: 一拍多长：此刻往前这么久关的时窗这一拍还算赶上（拍点晚了一点不丢没做的那次机会）。
        self.tick = timedelta(seconds=float(tick_seconds)) if tick_seconds is not None else None
        self.observer: Observer = observer if observer is not None else NullObserver()
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    async def run_once(self) -> TickReport:
        started = time.monotonic()
        try:
            report = await self._run()
        except Exception as exc:
            self._observe(ObservationStatus.FAILURE, {"error_type": type(exc).__name__}, started)
            raise
        self._observe(ObservationStatus.SUCCESS, _attributes(report), started)
        return report

    async def _run(self) -> TickReport:
        now = self._clock().astimezone(self.read_side.zone)
        if self.migrating is not None and self.migrating():
            return TickReport(now, 0, 0, skipped="词表迁移还没做完")
        materials = await self._read(now)
        formed, recorded, by_source = self._form(materials)
        delivered = await self._deliver(materials)
        return TickReport(
            now,
            formed,
            recorded,
            by_source,
            materials.skipped,
            delivered.decided if delivered is not None else 0,
            delivered.spoken if delivered is not None else 0,
            delivered,
            len(self.ledger.take_unreadable()) + (delivered.unreadable if delivered is not None else 0),
        )

    # ── 第一步：读 ──────────────────────────────────────────────────────────

    async def _read(self, now: datetime) -> _Materials:
        cache = DayIndexCache(
            self.read_side.behavior_tree,
            subject=self.read_side.subject,
            admits=admitted,
            calendar=self.read_side.calendar,
        )
        day_start = datetime.combine(now.date(), clock_time(0), tzinfo=now.tzinfo)
        await self.unsealed.prepare(until=now)
        unsealed = self.unsealed.rows(since=day_start, until=now)
        seen = seen_rows(
            Moment(at=now, slot=0, weekday=now.weekday(), day_note=cache.day(now.date()).day_note), cache, unsealed
        )
        table = self.opportunities.latest(self.lane)
        if table is None:
            return _Materials(now, cache, unsealed, seen, None, None, "", "还没有机会表")
        published = self.prediction.active()
        if published is None:
            return _Materials(now, cache, unsealed, seen, table, None, "", "预测树还没发布")
        try:
            tree = self.prediction.load_generation(published.generation, expected_digest=published.digest)
        except PredictionTreeStoreError as exc:
            raise ForesightError(f"prediction tree cannot be read: {exc}") from exc
        return _Materials(now, cache, unsealed, seen, table, tree, published.generation, None)

    # ── 第二步：建预期 ──────────────────────────────────────────────────────

    def _form(self, materials: _Materials) -> tuple[int, int, dict[str, int]]:
        if materials.table is None or materials.tree is None:
            return 0, 0, {}
        now = materials.now
        moment = Moment(
            at=now,
            slot=SlotKey.of(now, slot_minutes=materials.tree.slot_minutes).slot,
            weekday=now.weekday(),
            day_note=materials.cache.day(now.date()).day_note,
            looked_since=now - self.tick if self.tick is not None else None,
        )
        expectations = form_expectations(
            moment=moment,
            tree=materials.tree,
            table=materials.table,
            rows=materials.seen,
            shadow_unestablished=self.shadow_unestablished,
            generation=materials.generation,
            version=str(self.vocabulary_version()) if self.vocabulary_version is not None else "",
            implementation=IMPLEMENTATION,
        )
        recorded = 0
        by_source: dict[str, int] = {}
        silenced: dict[str, bool] = {}
        for item in expectations:
            by_source[item.source] = by_source.get(item.source, 0) + 1
            if self.ledger.has_expectation(item.expectation_id):
                continue
            if item.account not in silenced:
                setting = self.ledger.setting_of(item.account)
                silenced[item.account] = setting is not None and setting.setting == "以后别提了"
            if not silenced[item.account] and self.ledger.record_expectation(item):
                recorded += 1
        return len(expectations), recorded, by_source

    # ── 第三步：递送 ────────────────────────────────────────────────────────

    async def _deliver(self, materials: _Materials) -> DeliveryReport | None:
        """到了时窗的决定说不说。没有新预期也要跑：昨天建的、今天才到时窗的在等。"""

        if self.delivery is None:
            return None
        now, seen = materials.now, materials.seen
        return await self.delivery.run_once(
            now,
            JudgeMaterials(
                materials.table,
                lambda uris: self._rows_for(uris, materials.cache, materials.unsealed, seen, now),
                lambda item: any(
                    row.kind_token == item.kind_token and item.observed_from < row.started_at <= now for row in seen
                ),
                self.read_side.calendar.describe,
                seen,
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
                continue
            sealed = cache.day(day).by_uri.get(uri)
            if sealed is not None:
                found[uri] = Seen(
                    sealed.at, sealed.last_observed_at, sealed.kind_token, uri, sealed.name, sealed.summary
                )
        return found

    def _observe(
        self, status: ObservationStatus, attributes: dict[str, str | int | float | bool], started: float
    ) -> None:
        try:
            self.observer.record(
                ObservationEvent(
                    category="foresight",
                    operation="tick",
                    status=status,
                    duration_seconds=max(0.0, time.monotonic() - started),
                    attributes=attributes,
                )
            )
        except Exception:  # noqa: BLE001 - 观测不许影响这一拍
            pass


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
        found["severities_judged"] = report.delivery.severities
        found["level_changes"] = report.delivery.level_changes
        found["judge_calls"] = report.delivery.calls
        found["judge_inconsistent"] = sum(1 for signal in report.delivery.signals if "inconsistent" in signal)
        found["judge_failed"] = report.delivery.judge_failed
        found.update({f"reason_{key}": value for key, value in report.delivery.by_reason.items()})
    return found


__all__ = ["ForesightTick", "TickReadSide", "TickReport"]
