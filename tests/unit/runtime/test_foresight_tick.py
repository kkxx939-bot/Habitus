"""白天那一拍：读机会表、今天的日历条件、行为树今天与前几天，把此刻对得上历史的预期写进账本；同一身份只写一次；没有表时只报原因。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from habitus.conditions import LocalCalendar
from habitus.config.foresight import ForesightConfig
from habitus.foresight import NoUnsealed
from habitus.foresight.ladder import Tolerance
from habitus.foresight.opportunities import Opportunity, OpportunityTable, RelationOpportunities, build_table
from habitus.foundation.observability import ObservationEvent
from habitus.runtime.foresight import ForesightReadSide
from habitus.runtime.foresight_days import DayExpectationStage
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.runtime.foresight_tick import ForesightTick
from habitus.scene.concepts import ConceptSet
from habitus.scene.concepts.rhythm import RhythmRulers
from habitus.scene.recurrence import recurrences_of
from habitus.scene.relations import Relation, RelationConfig, RelationKey, Segment, Status, build_timelines
from habitus.series.reader import read_series
from tests.unit.kind_ids import kind_id
from tests.unit.scene.concept_fixtures import base
from tests.unit.scene.fixtures import DAY1, SUBJECT, Site, at, publish

TODAY = DAY1 + timedelta(days=3)  # 2026-08-18
DISCUSS, CODE = kind_id("讨论方案"), kind_id("修改代码")
CONCEPTS = ConceptSet([base("讨论方案"), base("修改代码")])
CONFIG = RelationConfig(
    slot_minutes=15, transition_window_slots=3, absence_typical_multiple=8.0, absence_min_gaps=2
)
RULERS = RhythmRulers(15, 2700, 3, 5.0, 5.0, 0.10, 35)
TOLERANCE = Tolerance(line=1.0 - ForesightConfig().false_alarm_rate)  # 产品默认的线


class Recorder:
    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)


def site_with_three_mornings(tmp_path) -> tuple[Site, list[str]]:  # type: ignore[no-untyped-def]
    """前三天每天 10:00 讨论方案、10:20 修改代码、12:00 写文档（人在场到中午，转移窗口才整段在场）。"""

    site = Site(tmp_path, now=at(TODAY, 3, 0))
    codes = []
    for offset in range(3):
        day = DAY1 + timedelta(days=offset)
        publish(site.behavior_tree, day, "讨论方案", 10, 0)
        codes.append(publish(site.behavior_tree, day, "修改代码", 10, 20))
        publish(site.behavior_tree, day, "写文档", 12, 0)
    return site, codes


def stores(tmp_path, site: Site, night=TODAY):  # type: ignore[no-untyped-def]
    root = tmp_path / "foresight" / "session"
    ledger, opportunities = ForesightLedgerStore(root), OpportunityStore(root)
    series = read_series(site.behavior_tree, cutoff=TODAY)
    line = build_timelines(series, CONCEPTS, {}, CONFIG)["session"]
    relations = (
        Relation(key=RelationKey("session", DISCUSS, CODE, Segment.CHAIN), status=Status.ESTABLISHED, upward=True),
    )
    opportunities.write(
        build_table(
            line,
            relations,
            CONCEPTS,
            night=night,
            relation_night=night,
            slot_minutes=15,
            window_slots=3,
            recurrences=recurrences_of(series, RULERS),
        )
    )
    return ledger, opportunities


def tick(tmp_path, site: Site, ledger, opportunities, now: datetime, recorder: Recorder | None = None) -> ForesightTick:  # type: ignore[no-untyped-def]
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai"))
    return ForesightTick(
        ledger,
        opportunities,
        read_side,
        NoUnsealed(),
        vocabulary_version=lambda: 7,
        observer=recorder,
        clock=lambda: now,
        recent=35,
        tolerance=TOLERANCE,
    )


def days(site: Site, ledger, opportunities, now: datetime, recorder: Recorder | None = None) -> DayExpectationStage:  # type: ignore[no-untyped-def]
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai"))
    return DayExpectationStage(
        ledger,
        opportunities,
        read_side,
        NoUnsealed(),
        vocabulary_version=lambda: 7,
        observer=recorder,
        clock=lambda: now,
        recent=35,
        tolerance=TOLERANCE,
    )


def test_a_tick_records_the_expectations_once_and_the_next_tick_adds_nothing(tmp_path) -> None:
    """白天那一拍只建事件与情境；当天的日子预期由夜里那条线的最后一步建（裁定 94 第 1 条），两边都按身份只建一次。"""

    site, codes = site_with_three_mornings(tmp_path)
    ledger, opportunities = stores(tmp_path, site)
    discussion = publish(site.behavior_tree, TODAY, "讨论方案", 10, 0)
    recorder = Recorder()
    run = tick(tmp_path, site, ledger, opportunities, at(TODAY, 10, 15), recorder)
    report = asyncio.run(run.run_once())
    assert report.skipped is None and report.formed == report.recorded == 1 and report.by_source == {"事件": 1}
    nightly = days(site, ledger, opportunities, at(TODAY, 10, 15), recorder)
    built = asyncio.run(nightly.run_once())
    assert built.skipped is None and built.formed.formed == built.formed.recorded == 3
    assert built.formed.by_source == {"日子": 3}
    stored = {(item.source, item.kind_token): item for item in ledger.expectations_on(TODAY)}
    event = stored[("事件", CODE)]
    assert event.kind_token == CODE and event.basis == tuple(codes)
    assert event.trigger_instance == at(TODAY, 10, 0).astimezone(UTC).isoformat()  # 前因的开始时刻
    assert event.opportunities == event.hits == 3 and event.version == "7" and event.night == TODAY
    assert event.generation == TODAY.isoformat()  # 出处记机会表那一晚（预测树删了，裁定 59）
    from habitus.runtime.foresight_implementation import IMPLEMENTATION

    assert event.implementation == IMPLEMENTATION  # 形成它的那份代码
    assert event.formed_at == at(TODAY, 10, 15) and event.window[0] >= event.formed_at
    assert event.observed_from == at(TODAY, 10, 0)  # 从前因做完那一槽起算
    assert event.antecedents[0].present and event.antecedents[0].source == discussion
    # 讨论方案今天 10:00 已经做了：它的"离上次够久"照建（裁定 49），从今天零点起算，结算会给准时；写文档的复现也登记了
    done = stored[("日子", DISCUSS)]
    assert done.observed_from == at(TODAY, 0, 0) and done.window[0] <= at(TODAY, 10, 0) < done.window[1]
    assert done.version == "7" and done.implementation == IMPLEMENTATION
    interval = stored[("日子", CODE)]
    assert interval.kind_token == CODE and interval.trigger == "interval"
    assert interval.trigger_instance == f"{codes[-1]}@{TODAY.isoformat()}"  # 上次那条 + 这一簇的第一天
    again = asyncio.run(run.run_once())
    assert again.formed == 1 and again.recorded == 0
    assert asyncio.run(nightly.run_once()).formed.recorded == 0
    assert [event.operation for event in recorder.events] == ["tick", "days", "tick", "days"]
    assert recorder.events[0].attributes["recorded"] == 1 and recorder.events[1].attributes["recorded"] == 3
    assert recorder.events[2].attributes["recorded"] == 0 and recorder.events[3].attributes["recorded"] == 0


def test_the_day_expectations_wait_for_tonights_table(tmp_path) -> None:
    """裁定 94 第 1 条：盘上最晚那张表不是今天这一晚的（零点到夜里那条线跑完之间，表只到前天）就不建日子预期——拿缺昨天的
    表建出来的时窗缺昨天，同一个身份只建一次，夜里算对的那条就进不了账（D1）。词表迁移做到一半也不建。"""

    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities = stores(tmp_path, site)  # 表是 TODAY 那一晚的
    tomorrow = TODAY + timedelta(days=1)
    stale = asyncio.run(days(site, ledger, opportunities, at(tomorrow, 0, 15)).run_once())
    assert stale.skipped == "今天这一代的机会表还没有" and ledger.expectations_on(tomorrow) == ()
    root = tmp_path / "empty"
    none = days(site, ForesightLedgerStore(root), OpportunityStore(root), at(TODAY, 1, 15))
    assert asyncio.run(none.run_once()).skipped == "还没有机会表"
    migrating = days(site, ledger, opportunities, at(TODAY, 1, 15))
    migrating.migrating = lambda: True
    assert asyncio.run(migrating.run_once()).skipped == "词表迁移还没做完" and ledger.expectations_on(TODAY) == ()


def test_the_daytime_gate_only_asks_whether_the_table_is_todays_generation(tmp_path) -> None:
    """白天那道闸（实时化方案第 3 节）：只看盘上最晚那张表是不是今天这一代的——是就建，不再等"今晚的线整条跑完"（夜线的
    Job 账删了）。表不是今天的只递不建（见 ``test_the_day_expectations_wait_for_tonights_table``）。"""

    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities = stores(tmp_path, site)  # 表是 TODAY 这一代的
    publish(site.behavior_tree, TODAY, "讨论方案", 10, 0)
    run = tick(tmp_path, site, ledger, opportunities, at(TODAY, 10, 15))
    report = asyncio.run(run.run_once())
    assert report.skipped is None and report.recorded == 1


def test_each_tick_is_a_span_with_the_gate_what_it_formed_and_how_late_it_started(tmp_path) -> None:
    """trace 细度：每拍一个 ``foresight.tick`` span，带时刻、lane、白天那道闸放没放行及原因、建了几条、递了几条、离它该开始的
    槽边界晚了多少、一拍用时（实时化方案第 10 节 E12）；落换代同一份本地 trace（按当地日子一天一份）。"""

    from habitus.runtime.generation_trace import TraceFiles

    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities = stores(tmp_path, site)
    traces = TraceFiles(tmp_path / "traces", artifact_root=tmp_path)
    run = tick(tmp_path, site, ledger, opportunities, at(TODAY, 10, 15) + timedelta(seconds=7))
    run.trace = traces.sink
    asyncio.run(run.run_once(scheduled=at(TODAY, 10, 15)))
    (span,) = traces.read(TODAY)
    attributes = span["attributes"]
    assert span["name"] == "foresight.tick" and span["parent_id"] is None
    assert attributes["lane"] == "session" and attributes["gate_open"] is True
    assert attributes["skipped"] == "" and attributes["recorded"] == 0 and "spoken" in attributes
    assert attributes["lateness_seconds"] == 7.0 and "tick_seconds" in attributes


def test_without_a_table_the_tick_only_reports_why(tmp_path) -> None:
    site, _codes = site_with_three_mornings(tmp_path)
    root = tmp_path / "foresight" / "session"
    run = tick(tmp_path, site, ForesightLedgerStore(root), OpportunityStore(root), at(TODAY, 10, 15))
    assert asyncio.run(run.run_once()).skipped == "还没有机会表"


def test_the_tick_reads_todays_calendar_conditions_for_situation_expectations(tmp_path) -> None:
    """机会表里有"周二 → 修改代码"：今天（周二）的日历条件从读侧的日历来，到了往常的时刻就建一条「情境」预期。"""

    site, codes = site_with_three_mornings(tmp_path)
    root = tmp_path / "foresight" / "session"
    ledger, opportunities = ForesightLedgerStore(root), OpportunityStore(root)
    tuesdays = [TODAY - timedelta(days=7 * weeks) for weeks in (2, 1)]
    items = tuple(
        Opportunity(f"calendar:{day}", (f"calendar:{day}",), day, at(day, 0, 0), True, codes[index], at(day, 10, 20))
        for index, day in enumerate(tuesdays)
    )
    relation = RelationOpportunities(
        key="session|周二|修改代码|situation",
        antecedent="周二",
        consequent=CODE,
        antecedent_label="周二",
        consequent_label="修改代码",
        antecedent_classes=("周二",),
        consequent_classes=(CODE,),
        segment="situation",
        status="established",
        upward=True,
        items=items,
        calendar_key="日历.周几",
        calendar_values=("周二",),
    )
    opportunities.write(OpportunityTable("session", TODAY, TODAY, 15, 3, (relation,), ()))
    assert TODAY.weekday() == 1
    report = asyncio.run(tick(tmp_path, site, ledger, opportunities, at(TODAY, 10, 25)).run_once())
    assert report.by_source == {"情境": 1}
    (stored,) = ledger.expectations_on(TODAY)
    assert stored.source == "情境" and stored.trigger == relation.key and stored.trigger_instance == TODAY.isoformat()


def test_a_vocabulary_migration_in_progress_skips_the_whole_tick(tmp_path) -> None:
    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities = stores(tmp_path, site)
    publish(site.behavior_tree, TODAY, "讨论方案", 10, 0)
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai"))
    run = ForesightTick(
        ledger,
        opportunities,
        read_side,
        NoUnsealed(),
        migrating=lambda: True,
        clock=lambda: at(TODAY, 10, 15),
        recent=35,
        tolerance=TOLERANCE,
    )
    report = asyncio.run(run.run_once())
    assert report.skipped == "词表迁移还没做完" and report.formed == 0 and ledger.expectations_on(TODAY) == ()


def test_the_tick_checks_its_collaborators(tmp_path) -> None:
    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities = stores(tmp_path, site)
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai"))
    with pytest.raises(ValueError):
        ForesightTick(ledger, opportunities, read_side, NoUnsealed(), lane="", recent=35, tolerance=TOLERANCE)


def test_an_abandoned_attempt_is_seen_but_not_as_done(tmp_path) -> None:
    """开始了但放弃了的那一行：白天看得到，可不算这件事做了——时窗末不当「已发生」不说，夜里结算也不认它（一条记录一种读法，
    裁定 54；2026-10-17 四方评审：原来白天判已发生、夜里又记一笔坏）。"""

    from habitus.foresight import Moment
    from habitus.foresight.triggers.assemble import seen_rows
    from habitus.scene.views import DayIndexCache
    from habitus.series.reader import admitted

    site, _codes = site_with_three_mornings(tmp_path)
    publish(site.behavior_tree, TODAY, "修改代码", 10, 20, status="abandoned")
    cache = DayIndexCache(site.behavior_tree, subject=SUBJECT, admits=admitted, calendar=LocalCalendar())
    moment = Moment(at(TODAY, 11, 0), slot=44)
    today = [row for row in seen_rows(moment, cache, ()) if row.started_at.date() == TODAY]
    assert [(row.kind_token, row.abandoned) for row in today] == [(CODE, True)]


def test_an_unreadable_stretch_is_found_on_the_tree_or_among_the_unsealed_judgements() -> None:
    """递送"说不准"的判据（裁定 67 第 14 条、68 第 1、2 条）：树上的「没读懂」（只认它，与结算同一个作废判法），或还没封口
    的「没读懂」判断；还没封口的可读行为从那段里开头的（白天当场重读补回来、还没归约的）把它作废。"""

    from types import SimpleNamespace

    from habitus.foresight import UnsealedRow
    from habitus.runtime.foresight_tick import _unobserved

    start = datetime(2026, 10, 21, 9, 0, tzinfo=UTC)
    item = SimpleNamespace(observed_from=start)
    now = start + timedelta(hours=1)
    stretch = SimpleNamespace(started_at=start + timedelta(minutes=10), ended_at=start + timedelta(minutes=12))

    class Cache:
        def __init__(self, gaps) -> None:  # type: ignore[no-untyped-def]
            self.gaps = gaps

        def gaps_within(self, begin, end, *, kinds):  # type: ignore[no-untyped-def]
            # 「未观测」不让它不说；「错误」同为看不到（裁定 94 第 3 条）
            assert (begin, end, kinds) == (start, now, frozenset({"没读懂", "错误"}))
            return self.gaps

    unreadable = UnsealedRow(None, None, start + timedelta(minutes=10), start + timedelta(minutes=12), None)
    readable = UnsealedRow("写周报", "s-k0002", start + timedelta(minutes=10), start + timedelta(minutes=12), None)
    earlier = UnsealedRow(None, None, start - timedelta(hours=2), start - timedelta(hours=1), None)
    assert _unobserved(item, SimpleNamespace(cache=Cache((stretch,)), unsealed=()), now)  # type: ignore[arg-type]
    assert _unobserved(item, SimpleNamespace(cache=Cache(()), unsealed=(unreadable,)), now)  # type: ignore[arg-type]
    assert not _unobserved(item, SimpleNamespace(cache=Cache(()), unsealed=(earlier,)), now)  # type: ignore[arg-type]
    # 补回来、还没归约：树上那段与未封口的「没读懂」都被它作废
    assert not _unobserved(item, SimpleNamespace(cache=Cache((stretch,)), unsealed=(readable,)), now)  # type: ignore[arg-type]
    assert not _unobserved(item, SimpleNamespace(cache=Cache(()), unsealed=(unreadable, readable)), now)  # type: ignore[arg-type]


def test_an_unsealed_unreadable_stretch_across_midnight_is_voided_day_by_day() -> None:
    """还没封口的「没读懂」23:50 → 次日 00:20，次日 00:05 当场重读出一条：与树上空白同一个读法，按天切开、每天那段由那一天的
    行为判——次日那段作废，当天 23:50–24:00 那段还站得住，说中的范围盖到它就不说。"""

    from types import SimpleNamespace
    from zoneinfo import ZoneInfo

    from habitus.foresight import UnsealedRow
    from habitus.runtime.foresight_tick import _unobserved

    zone = ZoneInfo("Asia/Shanghai")
    start = datetime(2026, 10, 21, 23, 40, tzinfo=zone)
    now = start + timedelta(hours=1)

    class Cache:
        def gaps_within(self, begin, end, *, kinds):  # type: ignore[no-untyped-def]
            return ()

    unreadable = UnsealedRow(None, None, start + timedelta(minutes=10), start + timedelta(minutes=40), None)
    readable = UnsealedRow("统计行数", "s-k0002", start + timedelta(minutes=25), start + timedelta(minutes=26), None)
    materials = SimpleNamespace(cache=Cache(), unsealed=(unreadable, readable))
    assert _unobserved(SimpleNamespace(observed_from=start), materials, now)  # type: ignore[arg-type]
    # 说中的范围从次日 00:00 起算：只剩被作废的次日那段
    after_midnight = SimpleNamespace(observed_from=datetime(2026, 10, 22, 0, 0, tzinfo=zone))
    assert not _unobserved(after_midnight, materials, now)  # type: ignore[arg-type]


def test_after_midnight_the_tick_still_reads_last_nights_unsealed_judgements(tmp_path) -> None:
    """段 6 Q11：白天那一拍读未封口判断的范围与行为树一样（今天与前 7 天）。原来只从当天零点读：过了零点、归约又落后时，
    昨晚 23:10 起还没封口的「没读懂」与可读的前因就从这一拍里丢了——递送照样开口、事件触发照样找不到前因。"""

    from types import SimpleNamespace

    from habitus.foresight import UnsealedRow
    from habitus.runtime.foresight_tick import _unobserved

    class Judgements:
        """按真实读口的过滤：与 ``[since, until]`` 有交集的。"""

        def __init__(self, rows: tuple[UnsealedRow, ...]) -> None:
            self._rows = rows

        async def prepare(self, *, until: datetime) -> None:
            return None

        def rows(self, *, since: datetime, until: datetime) -> tuple[UnsealedRow, ...]:
            return tuple(row for row in self._rows if row.last_observed_at >= since and row.started_at <= until)

    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities = stores(tmp_path, site)
    night = at(TODAY, 23, 0)
    blind = UnsealedRow(None, None, night + timedelta(minutes=10), night + timedelta(minutes=40), None)
    discussion = UnsealedRow("讨论方案", DISCUSS, night, night + timedelta(minutes=45), None)
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai"))
    after = night + timedelta(hours=1, minutes=10)  # 次日 00:10
    run = ForesightTick(
        ledger,
        opportunities,
        read_side,
        Judgements((blind, discussion)),
        clock=lambda: after,
        recent=35,
        tolerance=TOLERANCE,
    )
    materials = asyncio.run(run._read(after.astimezone(read_side.zone)))
    assert set(materials.recent.unsealed) == {blind, discussion}
    assert any(row.started_at == night and row.kind_token == DISCUSS for row in materials.recent.seen)
    item = SimpleNamespace(observed_from=night)
    assert _unobserved(item, materials.recent, after)  # type: ignore[arg-type]


def test_the_tick_hands_formation_the_event_instances_already_on_the_ledger(tmp_path) -> None:
    """段 6 Q5：建预期时拿得到这本账上已经建过的事件实例（还要看的、回看窗里的日子），前因开始被更正后同一次机会沿用它。"""

    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities = stores(tmp_path, site)
    publish(site.behavior_tree, TODAY, "讨论方案", 10, 0)
    run = tick(tmp_path, site, ledger, opportunities, at(TODAY, 10, 15))
    asyncio.run(run.run_once())
    event = next(item for item in ledger.expectations_on(TODAY) if item.source == "事件")
    known = run._known_instances(at(TODAY, 10, 30))
    assert known(event.account) == (event.trigger_instance,) and known("nobody") == ()
    assert run._known_instances(at(TODAY + timedelta(days=9), 10, 0))(event.account) == ()  # 回看窗以外的日子不读


# --- 段 6 Q18：存活变异补的断言 ----------------------------------------------------------------------------------


def test_a_behaviour_he_said_never_again_for_forms_no_expectations(tmp_path) -> None:
    """人说过「以后别提了」的行为白天那一拍与夜里的日子预期都不再建（裁定 50、61；M77：去掉这一道也全绿）；别的行为照建。"""

    from habitus.foresight.ledger import UnitSetting

    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities = stores(tmp_path, site)
    publish(site.behavior_tree, TODAY, "讨论方案", 10, 0)
    ledger.record_setting(UnitSetting(CODE, at(TODAY, 9, 0), "以后别提了"))
    asyncio.run(tick(tmp_path, site, ledger, opportunities, at(TODAY, 10, 15)).run_once())
    asyncio.run(days(site, ledger, opportunities, at(TODAY, 10, 15)).run_once())
    kinds = {item.kind_token for item in ledger.expectations_on(TODAY)}
    assert CODE not in kinds and DISCUSS in kinds


def test_the_tick_looks_back_over_its_own_interval_when_forming(tmp_path) -> None:
    """建预期时按上一拍看的时刻判退场（``looked_since`` = 此刻 − 一拍；M79：不接这一口）：讨论方案 10:00，10:40 那一拍按此刻
    已过了改代码的时窗，按上一拍（10:25）还没过——没做的这次机会照建，不然账上只丢没做的。今天这一晚的表已经补建过
    （09:00 那一拍），10:40 这一拍按一拍的宽度看。"""

    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities = stores(tmp_path, site)
    publish(site.behavior_tree, TODAY, "讨论方案", 10, 0)
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai"))
    clock = [at(TODAY, 9, 0)]
    run = ForesightTick(
        ledger,
        opportunities,
        read_side,
        NoUnsealed(),
        clock=lambda: clock[0],
        recent=35,
        tolerance=TOLERANCE,
        tick_seconds=900.0,
    )
    asyncio.run(run.run_once())
    clock[0] = at(TODAY, 10, 40)
    asyncio.run(run.run_once())
    assert [item.kind_token for item in ledger.expectations_on(TODAY) if item.source == "事件"] == [CODE]


# ── 裁定 94 第 1 条：今晚的表出来之前白天那一拍不建预期，表出来后下一拍补建 ─────────────────────────────


def _minute_tick(site: Site, ledger, opportunities, clock: list[datetime]) -> ForesightTick:  # type: ignore[no-untyped-def]
    from habitus.runtime.generation_watermark import TickTrack, TickWatermarkStore

    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai"))
    return ForesightTick(
        ledger,
        opportunities,
        read_side,
        NoUnsealed(),
        clock=lambda: clock[0],
        recent=35,
        tolerance=TOLERANCE,
        tick_seconds=60.0,
        track=TickTrack(TickWatermarkStore(ledger.root), slot=timedelta(minutes=1), tolerance_slots=2),
    )


def test_before_tonights_table_the_tick_builds_no_expectation(tmp_path) -> None:
    """盘上最晚那张表是昨晚的（夜里那条线还没跑完）：事件、情境预期一条也不建，只报原因。"""

    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities = stores(tmp_path, site, night=TODAY - timedelta(days=1))
    publish(site.behavior_tree, TODAY, "讨论方案", 10, 0)
    report = asyncio.run(tick(tmp_path, site, ledger, opportunities, at(TODAY, 10, 15)).run_once())
    assert report.skipped == "今天这一代的机会表还没有" and report.recorded == 0
    assert ledger.expectations_on(TODAY) == ()


def test_the_first_tick_after_tonights_table_catches_up_from_midnight(tmp_path) -> None:
    """表出来后的第一拍从零点起看：讨论方案 10:00，改代码的时窗在 10:49 之前就关了——表出来后第一拍（10:50）照样建（补建）。
    之后的拍从落盘的水位（上一次建预期看到的那一刻）起看：09:00 看过一拍、10:50 才有下一拍（中间合盖），10:00–10:49 那段照样
    看到、照样建（实时化缺陷 1：原来按 ``now - 一拍`` 只看 10:49 之后，这一条就丢了——没做的那一侧不建，账往上偏）。
    补建抓不回来的（已写进说明）：前因做完到第一拍已过转移窗口，「同一条链」那一段的预期补不回来。"""

    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities = stores(tmp_path, site)
    publish(site.behavior_tree, TODAY, "讨论方案", 10, 0)
    clock = [at(TODAY, 10, 50)]
    asyncio.run(_minute_tick(site, ledger, opportunities, clock).run_once())
    assert [item.kind_token for item in ledger.expectations_on(TODAY) if item.source == "事件"] == [CODE]

    root = tmp_path / "caught-up"
    other_ledger, other_tables = stores(root, site)
    caught_up = _minute_tick(site, other_ledger, other_tables, [at(TODAY, 9, 0)])
    asyncio.run(caught_up.run_once())  # 09:00 已用今天这一代的表看过一拍
    caught_up._clock = lambda: at(TODAY, 10, 50)  # type: ignore[method-assign]
    asyncio.run(caught_up.run_once())
    assert [item.kind_token for item in other_ledger.expectations_on(TODAY) if item.source == "事件"] == [CODE]

    late_root = tmp_path / "late"
    late_ledger, late_tables = stores(late_root, site)
    asyncio.run(_minute_tick(site, late_ledger, late_tables, [at(TODAY, 11, 30)]).run_once())
    assert [item for item in late_ledger.expectations_on(TODAY) if item.source == "事件"] == []


def test_a_gap_between_ticks_is_written_as_unwatched_and_the_watermark_drives_the_look_back(tmp_path) -> None:
    """实时化方案第 5、6 节（缺陷 1，E7、E15）：每拍跑完把水位落盘；下一拍离上一拍超过容差（2 槽），中间那段交会话一支写成
    「未观测」，落盘它的结尾；第一次启动（没有水位）不写。往回看的起点：今天这一代第一次建预期从零点起，之后从上一次建预期看到的
    那一刻（落盘的 ``formed_through``）——合盖 10:00–13:45 醒来那一拍从 10:00 起看，不是 ``now - 一拍``。"""

    from habitus.runtime.generation_watermark import TickTrack, TickWatermarkStore

    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities = stores(tmp_path, site)
    written: list[tuple[datetime, datetime]] = []
    store = TickWatermarkStore(tmp_path / "foresight" / "session")
    track = TickTrack(
        store, slot=timedelta(minutes=15), tolerance_slots=2, unwatched=lambda start, end: written.append((start, end))
    )
    clock = [at(TODAY, 10, 0)]
    run = tick(tmp_path, site, ledger, opportunities, clock[0])
    run.tick = timedelta(minutes=15)
    run._clock = lambda: clock[0]  # type: ignore[method-assign]
    run.track = track
    looked: list[datetime | None] = []
    form = run.forming.form

    def spy(recent, table, **kwargs):  # type: ignore[no-untyped-def]
        looked.append(kwargs["looked_since"])
        return form(recent, table, **kwargs)

    run.forming.form = spy  # type: ignore[method-assign]
    first = asyncio.run(run.run_once())
    assert first.unwatched is None and written == []  # 第一次启动：不知道之前在不在跑
    assert looked == [at(TODAY, 0, 0)]  # 今天这一代第一次建预期：从零点起看
    assert store.read().ticked_at == at(TODAY, 10, 0) and store.read().formed_through == at(TODAY, 10, 0)
    clock[0] = at(TODAY, 10, 15)
    assert asyncio.run(run.run_once()).unwatched is None and looked[-1] == at(TODAY, 10, 0)
    clock[0] = at(TODAY, 13, 45)  # 合盖三个半小时
    late = asyncio.run(run.run_once())
    assert written == [(at(TODAY, 10, 15), at(TODAY, 13, 45))] and late.unwatched == written[0]
    assert looked[-1] == at(TODAY, 10, 15)  # 从上一次建预期看到的那一刻起看
    assert store.read().unwatched_until == at(TODAY, 13, 45)

    def broken(_start, _end) -> None:  # type: ignore[no-untyped-def]
        raise OSError("judgement store")

    track.unwatched = broken
    clock[0] = at(TODAY, 18, 0)
    with pytest.raises(OSError):
        asyncio.run(run.run_once())
    assert store.read().ticked_at == at(TODAY, 13, 45)  # 写不成：水位不推，下一拍再写


def test_the_daytime_gate_waits_for_todays_generation_to_be_complete(tmp_path) -> None:
    """F5：表是今天的、可今天这一代还没完成（机会表写出了、日子预期或完成标记没成；下一眼重做整代会重写这张表）——预测拍不建
    预期，只报原因；完成标记读不了也不建。今天的完成标记在了才建。"""

    from habitus.runtime.foresight_forming import MARK_UNREADABLE, UNFINISHED
    from habitus.runtime.generation_marks import GenerationMarkError

    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities = stores(tmp_path, site)  # 表是 TODAY 这一代的
    publish(site.behavior_tree, TODAY, "讨论方案", 10, 0)
    run = tick(tmp_path, site, ledger, opportunities, at(TODAY, 10, 15))
    done: list[bool | None] = [None]
    run.finished = lambda _day: done[0]
    assert asyncio.run(run.run_once()).skipped == UNFINISHED and ledger.expectations_on(TODAY) == ()
    done[0] = False
    assert asyncio.run(run.run_once()).skipped == UNFINISHED

    def broken(_day):  # type: ignore[no-untyped-def]
        raise GenerationMarkError("generation mark is corrupt")

    run.finished = broken
    assert asyncio.run(run.run_once()).skipped == MARK_UNREADABLE
    done[0] = True
    run.finished = lambda _day: done[0]
    assert asyncio.run(run.run_once()).recorded == 1


def test_a_tick_watermark_with_a_repeated_key_is_refused(tmp_path) -> None:
    """F7：水位文件里同一个键写了两次——报错，不让后一个悄悄盖掉前一个。"""

    from habitus.runtime.generation_watermark import TickWatermark, TickWatermarkError, TickWatermarkStore

    store = TickWatermarkStore(tmp_path)
    store.write(TickWatermark(ticked_at=at(TODAY, 10, 0)))
    text = store.path.read_text(encoding="utf-8")
    store.path.write_text(text.replace("{", '{\n  "ticked_at": null,', 1), encoding="utf-8")
    with pytest.raises(TickWatermarkError, match="repeats"):
        store.read()


# ── 第二轮修复 G2：递送补被挤掉的那一拍，读数"隔了几拍" ─────────────────────────────


def test_the_tick_hands_delivery_the_last_watched_moment_unless_it_just_wrote_unwatched(tmp_path) -> None:
    """递送的起点 = max(上一拍落盘的系统确知在看到的那一刻, 此刻 − 2 拍)（``JudgeMaterials.since``）。第一次启动（盘上没有
    水位）、这一拍开头刚写了「未观测」时是 None（递送从此刻 − 1 拍起，没在看的时段不补说——``begin`` 写完空白已把水位推到此刻，
    不能拿它当起点）。span 读数 ``ticks_since_previous`` 记这一拍离上一拍隔了几拍（被挤掉一拍是 2）。"""

    from habitus.runtime.foresight_delivery import DeliveryReport
    from habitus.runtime.generation_watermark import TickTrack, TickWatermarkStore

    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities = stores(tmp_path, site)
    clock = [at(TODAY, 10, 0)]
    run = tick(tmp_path, site, ledger, opportunities, clock[0])
    run.tick = timedelta(minutes=15)
    run._clock = lambda: clock[0]  # type: ignore[method-assign]
    run.track = TickTrack(
        TickWatermarkStore(tmp_path / "foresight" / "session"), slot=timedelta(minutes=15), tolerance_slots=2
    )
    handed: list[datetime | None] = []

    class Delivery:
        current = staticmethod(lambda token: (token,))

        tick = timedelta(minutes=15)

        async def run_once(self, now, materials):  # type: ignore[no-untyped-def]
            handed.append(materials.since)
            return DeliveryReport(now, 0, 0)

    run.delivery = Delivery()  # type: ignore[assignment]
    ticks = []
    for hour, minute in ((10, 0), (10, 15), (10, 45), (14, 0)):
        clock[0] = at(TODAY, hour, minute)
        ticks.append(asyncio.run(run.run_once()).ticks_since_previous)
    assert handed == [None, at(TODAY, 10, 0), at(TODAY, 10, 15), None]  # 10:45：max(10:15, 10:45 − 30 分)
    assert ticks == [None, 1, 2, 13]


def test_the_look_back_follows_the_unwatched_tolerance(tmp_path) -> None:
    """用户 11-10「跟着容差走」：容差 3 槽时，挤掉两拍（空当 3 槽，不写「未观测」）的这一拍交给递送的起点就是上一拍；
    空当 4 槽写了「未观测」，交 None。"""

    from habitus.runtime.foresight_delivery import DeliveryReport
    from habitus.runtime.generation_watermark import TickTrack, TickWatermarkStore

    site, _codes = site_with_three_mornings(tmp_path)
    ledger, opportunities = stores(tmp_path, site)
    clock = [at(TODAY, 10, 0)]
    run = tick(tmp_path, site, ledger, opportunities, clock[0])
    run.tick = timedelta(minutes=15)
    run._clock = lambda: clock[0]  # type: ignore[method-assign]
    run.track = TickTrack(
        TickWatermarkStore(tmp_path / "foresight" / "session"), slot=timedelta(minutes=15), tolerance_slots=3
    )
    handed: list[datetime | None] = []

    class Delivery:
        current = staticmethod(lambda token: (token,))
        tick = timedelta(minutes=15)

        async def run_once(self, now, materials):  # type: ignore[no-untyped-def]
            handed.append(materials.since)
            return DeliveryReport(now, 0, 0)

    run.delivery = Delivery()  # type: ignore[assignment]
    for hour, minute in ((10, 0), (10, 45), (11, 45)):
        clock[0] = at(TODAY, hour, minute)
        asyncio.run(run.run_once())
    assert handed == [None, at(TODAY, 10, 0), None]
