"""单元历史在组合根的那一半（26 号清单 4.9；裁定 61）：结完的日子每条预期记一份原始结果；每晚按当晚的关系表重标、一条 lane
一晚一份快照；只看这一晚之前写下的；重结撤回的日子再结完时重记；分量是原始强度、"规律断了"的打折读的时候算；退出的刻删了
不改变任何一层的读数。"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from habitus.foresight import Evidence, ForesightError
from habitus.foresight.edges import PastMoment, edge_reading
from habitus.foresight.history import Moment, UnitHistory
from habitus.foresight.ledger import AccountFact, Expectation, Settlement
from habitus.foresight.opportunities import OpportunityTable
from habitus.foresight.triggers import timeline_of
from habitus.runtime.foresight_history import HistoryStage
from habitus.runtime.foresight_history_store import HistoryStore
from habitus.runtime.foresight_ledger import ForesightLedgerStore

CST = timezone(timedelta(hours=8))
EVIDENCE = Evidence(alpha=0.05, half_life=8.0)
NIGHT = date(2026, 9, 30)
NIGHT_AT = datetime(2026, 9, 30, tzinfo=CST)


def gym(day: date) -> Expectation:
    start = datetime(day.year, day.month, day.day, tzinfo=CST)
    return Expectation(
        expectation_id=f"gym-{day.isoformat()}",
        kind_token="s-k0002",
        day=day,
        formed_at=start,
        source="日子",
        trigger="weekday=1",
        trigger_instance=day.isoformat(),
        window=(start + timedelta(hours=19), start + timedelta(hours=20)),
        deadline=start + timedelta(hours=23, minutes=59),
        basis=(),
        opportunities=20,
        hits=18,
        observed_from=start,
    )


def stage(tmp_path) -> tuple[HistoryStage, ForesightLedgerStore, HistoryStore]:  # type: ignore[no-untyped-def]
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    store = HistoryStore(tmp_path / "ledger")
    return HistoryStage(ledger, store, evidence=EVIDENCE, statuses=("established",)), ledger, store


def record(ledger: ForesightLedgerStore, item: Expectation, *, came: bool, written: datetime | None = None) -> None:
    ledger.record_expectation(item)
    ledger.append_fact(
        AccountFact(
            item.account,
            item.formed_at,
            came,
            1.0,
            "准时" if came else "过期待决",
            item.expectation_id,
            written_at=written,
        )
    )
    ledger.mark_final(item.day, at=item.deadline)


def nothing() -> tuple[OpportunityTable, object]:
    table = OpportunityTable("session", NIGHT, NIGHT, 15, 3, (), ())
    return table, timeline_of((), lane="session", slot_minutes=15, horizon=NIGHT)


def test_the_night_records_every_finished_day_once_and_writes_one_history(tmp_path) -> None:
    run, ledger, store = stage(tmp_path)
    for weeks, came in ((1, True), (2, False), (3, True)):
        record(ledger, gym(NIGHT - timedelta(weeks=weeks)), came=came)
    record(ledger, gym(NIGHT - timedelta(days=400)), came=True)  # 不再按"回看多少天"截：退出由名次定
    open_item = gym(NIGHT - timedelta(days=2))
    ledger.record_expectation(open_item)  # 还没结完：不记
    table, line = nothing()
    assert run.run(night=NIGHT, table=table, line=line).relabelled == 4  # type: ignore[arg-type]
    stored = store.latest("session")
    assert stored is not None and stored.night == NIGHT
    assert [moment.came for moment in stored.of("s-k0002")] == [True, True, False, True]
    assert len(store.records("session")) == 4 and open_item.day not in store.recorded_days("session")


def test_the_history_records_a_day_with_every_fact_written_by_the_moment_it_records(tmp_path) -> None:
    """实时化缺陷 3：单元历史记某天时按"记录这一刻"取账上已写下的事实——结算在第 N 天零点之后才写下的后补（并同一拍把那天标结完），
    同一代重标时就带上，不按过期待决记下（原来只看那一晚零点之前写下的，那天之后再也不重记）。"""

    run, ledger, store = stage(tmp_path)
    item = gym(NIGHT - timedelta(days=7))
    record(ledger, item, came=False)
    ledger.append_fact(  # 补做写在这一晚零点之后（换代里结算刚写下）
        AccountFact(
            item.account,
            item.formed_at,
            False,
            0.0,
            "后补",
            item.expectation_id,
            written_at=datetime(2026, 10, 2, tzinfo=CST),
        )
    )
    table, line = nothing()
    run.run(night=NIGHT, table=table, line=line)  # type: ignore[arg-type]
    (moment,) = store.records("session")
    assert moment.expectation_id == item.expectation_id and moment.strength == pytest.approx(0.0)


def test_a_day_reopened_by_a_resettle_is_recorded_again_when_it_finishes(tmp_path) -> None:
    run, ledger, store = stage(tmp_path)
    item = gym(NIGHT - timedelta(days=14))
    record(ledger, item, came=False)
    table, line = nothing()
    run.run(night=NIGHT, table=table, line=line)  # type: ignore[arg-type]
    ledger.reopen(item.day)
    ledger.append_fact(
        AccountFact(item.account, item.formed_at, True, 1.0, "准时", item.expectation_id, written_at=NIGHT_AT)
    )
    later = NIGHT + timedelta(days=1)
    run.run(night=later, table=OpportunityTable("session", later, later, 15, 3, (), ()), line=line)  # type: ignore[arg-type]
    assert item.day not in store.recorded_days("session")  # 撤回了：从"记过的"里拿掉
    ledger.mark_final(item.day, at=NIGHT_AT)
    again = later + timedelta(days=1)
    run.run(night=again, table=OpportunityTable("session", again, again, 15, 3, (), ()), line=line)  # type: ignore[arg-type]
    (moment,) = store.latest("session").of("s-k0002")  # type: ignore[union-attr]
    assert moment.came and item.day in store.recorded_days("session")


def test_the_moments_keep_their_raw_strength_and_the_break_is_applied_when_read(tmp_path) -> None:
    """记录与快照存原始强度；规律断了时断点之前的那几刻打折在读边的时候算（``edges.fold_sequence``）——断点一变，重读跟着变。"""

    run, ledger, store = stage(tmp_path)
    days = [NIGHT - timedelta(days=60 - index) for index in range(40)]
    for day in days:
        record(ledger, gym(day), came=True)
    table, line = nothing()
    run.run(night=NIGHT, table=table, line=line)  # type: ignore[arg-type]
    assert all(moment.weight == 1.0 for moment in store.latest("session").of("s-k0002"))  # type: ignore[union-attr]
    for day in (NIGHT - timedelta(days=offset) for offset in (19, 18, 17, 16, 15)):
        record(ledger, gym(day), came=False)  # 一直全中的账连漏五次：规律断了
    later = NIGHT + timedelta(days=1)
    run.run(night=later, table=OpportunityTable("session", later, later, 15, 3, (), ()), line=line)  # type: ignore[arg-type]
    found = store.latest("session").of("s-k0002")  # type: ignore[union-attr]
    assert all(moment.weight == 1.0 for moment in found)
    account = gym(days[0]).account
    folded = edge_reading(found, account, alpha=EVIDENCE.alpha, half_life=EVIDENCE.half_life, window=EVIDENCE.window)
    assert folded.break_at == gym(NIGHT - timedelta(days=19)).formed_at and folded.discount < 1.0


def test_retired_moments_leave_every_layer_reading_unchanged(tmp_path) -> None:
    from habitus.foresight import Tolerance, climb
    from habitus.foresight.units import Input, unit_reading

    run, ledger, store = stage(tmp_path)
    for index in range(80):  # 80 周，每周一刻
        record(ledger, gym(NIGHT - timedelta(weeks=80 - index)), came=index % 5 != 0)
    table, line = nothing()
    run.run(night=NIGHT, table=table, line=line)  # type: ignore[arg-type]
    full = store.latest("session")
    assert full is not None and len(full.of("s-k0002")) == 80
    remaining = store.records("session")
    assert len(remaining) == EVIDENCE.window  # 同一层只留最近 M 刻
    later = NIGHT + timedelta(days=1)
    run.run(night=later, table=OpportunityTable("session", later, later, 15, 3, (), ()), line=line)  # type: ignore[arg-type]
    trimmed = store.latest("session")
    assert trimmed is not None and len(trimmed.of("s-k0002")) == EVIDENCE.window
    today = gym(NIGHT + timedelta(days=7))
    now = NIGHT_AT + timedelta(days=7)
    loss = Tolerance(line=2 / 3)
    standing = climb(full.of("s-k0002"), today.account, now=now, tolerance=loss, evidence=EVIDENCE)
    assert (
        standing.prediction
        == climb(trimmed.of("s-k0002"), today.account, now=now, tolerance=loss, evidence=EVIDENCE).prediction
    )
    item = Input(today, standing)
    before = unit_reading((item,), full.of("s-k0002"), now=now, evidence=EVIDENCE, prior_share=1 / 3)
    after = unit_reading((item,), trimmed.of("s-k0002"), now=now, evidence=EVIDENCE, prior_share=1 / 3)
    assert after.reading == before.reading


def test_the_store_keeps_the_latest_nights_and_refuses_to_rewrite_the_past(tmp_path) -> None:
    store = HistoryStore(tmp_path / "ledger", retained_nights=2)
    for offset in range(3):
        store.write(
            UnitHistory(
                "session",
                NIGHT + timedelta(days=offset),
                {"s-k0002": (Moment(PastMoment(datetime(2026, 9, 1, tzinfo=CST), frozenset({"a"}), True), ("a",)),)},
            )
        )
    assert store.latest_night("session") == NIGHT + timedelta(days=2)
    with pytest.raises(ForesightError, match="does not exist"):
        store.read("session", NIGHT)  # 只留最近两晚
    with pytest.raises(ForesightError, match="not rewritten"):
        store.write(UnitHistory("session", NIGHT, {}))


@pytest.mark.parametrize("content", ['{"schema_version": 999}', "{", "[]"])
def test_a_bad_moment_record_is_skipped_and_counted_and_the_snapshot_is_still_written(tmp_path, content) -> None:
    """段 6 Q2：单元历史一份坏 / 旧 schema 的每刻记录不让这一晚抛（原来每晚抛、快照停在出错前、压实跟着停）——跳过、计入观测，
    快照照写；"记过的日子"读不出来就当没记过，结完的日子整天重记。"""

    run, ledger, store = stage(tmp_path)
    for weeks in (1, 2):
        record(ledger, gym(NIGHT - timedelta(weeks=weeks)), came=True)
    table, line = nothing()
    run.run(night=NIGHT - timedelta(days=1), table=table, line=line)  # type: ignore[arg-type]
    bad = store.root / "unit_moments" / "session" / "gym-old.json"
    bad.write_text(content, encoding="utf-8")
    recorded = store.root / "unit_moments" / "session" / "_recorded.json"
    recorded.write_text("{", encoding="utf-8")
    assert run.run(night=NIGHT, table=table, line=line).relabelled == 2  # type: ignore[arg-type]
    stored = store.latest("session")
    assert stored is not None and stored.night == NIGHT and len(stored.of("s-k0002")) == 2
    assert set(store.take_unreadable()) == {str(bad), str(recorded)}
    assert len(store.recorded_days("session")) == 2  # 重记之后整份重写成读得出的


def test_a_corrupt_latest_snapshot_falls_back_to_the_night_before(tmp_path) -> None:
    store = HistoryStore(tmp_path / "ledger")
    store.write(UnitHistory("session", NIGHT, {}))
    latest = store.write(UnitHistory("session", NIGHT + timedelta(days=1), {}))
    latest.write_text("{", encoding="utf-8")
    found = store.latest("session")
    assert found is not None and found.night == NIGHT and store.take_unreadable() == (str(latest),)


def test_an_occurrence_two_overlapping_day_expectations_settled_on_is_one_moment(tmp_path) -> None:
    """裁定 81：结算给每条预期各认一次发生（账不变）；记录带上结算认出的那次发生，01-06 没做、01-07 做了时 01-06 与 01-07
    两条都认 01-07 那一次，单元历史只算给起算最早的 01-06 那一刻。"""

    run, ledger, store = stage(tmp_path)
    found = "behavior://occurrences/2026/09/07/news.md"
    for day in (date(2026, 9, 6), date(2026, 9, 7)):
        start = datetime(day.year, day.month, day.day, tzinfo=CST)
        item = replace(
            gym(day), trigger="interval", deadline=start + timedelta(days=2), window=(start, start + timedelta(hours=9))
        )
        record(ledger, item, came=True)
        ledger.record_settlement(Settlement(item.expectation_id, item.kind_token, day, NIGHT_AT, "准时", found))
    table, line = nothing()
    assert run.run(night=NIGHT, table=table, line=line).relabelled == 1  # type: ignore[arg-type]
    (moment,) = store.latest("session").labelled("s-k0002")  # type: ignore[union-attr]
    assert moment.expectation_ids == ("gym-2026-09-06",) and moment.past.came
    assert {item.occurrence_uri for item in store.records("session")} == {found}


def test_the_unit_reader_sees_a_rewritten_snapshot_of_the_same_night(tmp_path) -> None:
    """实时化缺陷 7（E14）：同一晚的单元历史快照被重写（重跑那一代），读侧的缓存按文件此刻的样子失效，读到新内容。"""

    from habitus.runtime.foresight_units import UnitPlanner

    NIGHT_AT = datetime.combine(NIGHT - timedelta(days=1), datetime.min.time(), tzinfo=CST)

    _run, _ledger, store = stage(tmp_path)
    planner = UnitPlanner(store, evidence=EVIDENCE, prior_share=1 / 3)
    first = UnitHistory("session", NIGHT, {})
    store.write(first)
    assert planner.history_of(None) == first
    rewritten = UnitHistory("session", NIGHT, {"s-k0009": (Moment(PastMoment(NIGHT_AT, frozenset({"a"}), True), ("x",)),)})
    store.write(rewritten)
    assert planner.history_of(None) == rewritten
