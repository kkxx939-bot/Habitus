"""结算这一拍对着真实的行为树：只结最晚时限那天已定稿的；过期待决之后补做了记后补；定稿日的漏报查一次；
结算与裁决落账时往这本账上写事实（裁定 48、50、51）。"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from habitus.foresight import Evidence
from habitus.foresight.ledger import Delivery, Expectation
from habitus.foundation.observability import ObservationEvent
from habitus.runtime.foresight import ForesightReadSide
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_settlement import SettlementStage
from habitus.scene import NominalCalendar
from tests.unit.kind_ids import kind_id
from tests.unit.scene.fixtures import SUBJECT, Site, at, publish, publish_gap

DAY = date(2026, 12, 10)
FLIGHT, HOTEL, CARD = kind_id("订机票"), kind_id("订酒店"), kind_id("还信用卡")


class Recorder:
    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)


def expectation(
    identity: str, kind: str, formed: datetime, *, window_minutes: tuple[int, int], deadline_hours: int
) -> Expectation:
    return Expectation(
        expectation_id=identity,
        kind_token=kind,
        day=formed.date(),
        formed_at=formed,
        source="事件",
        trigger="session|flight|hotel|chain",
        trigger_instance="behavior://flight.md",
        window=(formed + timedelta(minutes=window_minutes[0]), formed + timedelta(minutes=window_minutes[1])),
        deadline=formed + timedelta(hours=deadline_hours),
        basis=(),
        opportunities=13,
        hits=10,
        observed_from=formed,
    )


def stage(tmp_path, closed: tuple[date, ...], *, observer: Recorder | None = None, migrating: bool = False):  # type: ignore[no-untyped-def]
    site = Site(tmp_path, now=at(DAY, 23, 0))
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, NominalCalendar(), ZoneInfo("Asia/Shanghai"))
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    settlement = SettlementStage(
        ledger,
        read_side,
        closed_days=lambda: closed,
        current=lambda token: (token,),
        migrating=lambda: migrating,
        evidence=Evidence(alpha=0.05, half_life_days=60.0),
        observer=observer,
        clock=lambda: at(DAY + timedelta(days=1), 4, 0).astimezone(UTC),
    )
    return site, ledger, settlement


def test_only_expectations_whose_deadline_day_is_closed_are_settled(tmp_path) -> None:
    site, ledger, settlement = stage(tmp_path, closed=(DAY,))
    formed = at(DAY, 14, 7)
    hotel = expectation("hotel", HOTEL, formed, window_minutes=(4, 51), deadline_hours=3)
    ledger.record_expectation(hotel)
    # 次日的预期：最晚时限落在还没定稿的明天，这一拍不结
    tomorrow = expectation("card", CARD, formed, window_minutes=(60 * 20, 60 * 21), deadline_hours=30)
    ledger.record_expectation(tomorrow)
    publish(site.behavior_tree, DAY, "订机票", 14, 7, kind="订机票")
    uri = publish(site.behavior_tree, DAY, "订酒店", 14, 28, kind="订酒店")
    report = settlement.run_once()
    # 还信用卡那条的最晚时限在明天，明天没定稿：这一天仍算有未结算的
    assert report.settled == 1 and report.outcomes == {"准时": 1} and report.pending_days == (DAY,)
    (settled,) = ledger.settlements_on(DAY)
    assert (settled.expectation_id, settled.occurrence_uri) == ("hotel", uri)
    assert [item.expectation_id for item in ledger.unsettled_expectations(DAY)] == ["card"]
    # 定稿日查漏报：订机票发生时没有在时的预期（那天也没有它的预期）；订酒店被预期盖着
    (omission,) = ledger.omissions_on(DAY)
    assert (omission.kind_token, omission.had_expectation) == (FLIGHT, False)
    assert ledger.scanned(DAY)
    # 账上记了一笔预测轴的好（窗内，强度 1）
    (fact,) = ledger.facts_of(hotel.account)
    assert (fact.axis, fact.good, fact.kind, fact.strength, fact.offset_minutes) == ("预测", True, "准时", 1.0, 0)
    # 第二拍：没有新东西，幂等
    again = settlement.run_once()
    assert again.settled == 0 and again.omissions == 0 and len(ledger.facts_of(hotel.account)) == 1


def test_a_reminded_occurrence_is_settled_as_after_the_reminder(tmp_path) -> None:
    site, ledger, settlement = stage(tmp_path, closed=(DAY,))
    formed = at(DAY, 14, 7)
    ledger.record_expectation(expectation("hotel", HOTEL, formed, window_minutes=(4, 51), deadline_hours=3))
    ledger.record_delivery(DAY, Delivery("hotel", at(DAY, 14, 15), 1, True, "说了", 1.0, "要订酒店吗"))
    publish(site.behavior_tree, DAY, "订酒店", 14, 30, kind="订酒店")
    settlement.run_once()
    (settled,) = ledger.settlements_on(DAY)
    assert (settled.outcome, settled.offset_minutes) == ("提醒后发生", 15)
    # 预测轴是中性（提醒的功劳不算这个习惯本身，裁定 58 第 5 条）；提醒组记一笔单位权重的好，说完 15 分钟就做了
    facts = {item.axis: item for item in ledger.facts_of(expectation("hotel", HOTEL, formed, window_minutes=(4, 51), deadline_hours=3).account)}
    assert facts["预测"].good and facts["预测"].strength == 0.0
    assert facts["提醒组"].good and facts["提醒组"].strength == 1.0 and facts["提醒组"].offset_minutes == 15


def test_overdue_pending_is_ruled_caught_up_once_a_later_closed_day_shows_the_occurrence(tmp_path) -> None:
    site, ledger, settlement = stage(tmp_path, closed=(DAY, DAY + timedelta(days=1)))
    formed = at(DAY, 14, 7)
    ledger.record_expectation(expectation("hotel", HOTEL, formed, window_minutes=(4, 51), deadline_hours=3))
    publish(site.behavior_tree, DAY + timedelta(days=1), "订酒店", 9, 0, kind="订酒店")
    report = settlement.run_once()
    assert report.outcomes == {"过期待决": 1} and report.rulings == 1
    (ruling,) = ledger.rulings_on(DAY)
    assert ruling.ruling == "后补" and ruling.delay_minutes == 16 * 60 - 7
    # 过期待决的坏被后补那一笔覆盖（没有机会表的日历：只有方向，强度 1）
    account = expectation("hotel", HOTEL, formed, window_minutes=(4, 51), deadline_hours=3).account
    kinds = sorted(item.kind for item in ledger.facts_of(account))
    assert kinds == ["后补", "过期待决"]
    # 后补留在那次机会的时刻（遗忘与连错的先后按它），写下的时刻是裁的那一晚（谁覆盖谁按它）
    caught = next(item for item in ledger.facts_of(account) if item.kind == "后补")
    assert caught.at == formed and caught.written_at == ruling.ruled_at
    # 补做的那条不是漏报：它被裁决认领了？——不，漏报按"发生时有没有在时的预期"判，次日 9 点没有在时的预期，如实记漏报
    assert [item.kind_token for item in ledger.omissions_on(DAY + timedelta(days=1))] == [HOTEL]


def test_a_catch_up_is_only_before_the_next_chance_of_the_same_account(tmp_path) -> None:
    """次日 08:00 同一本账又有了一条预期：09:00 的发生是这本账下一次机会的正常发生，不是补做；不裁、留着过期待决。
    别的账（另一种触发）的预期不挡：每个机会都建之后，同一本账的下一条就是这条触发自己的下一次机会（裁定 49）。"""

    site, ledger, settlement = stage(tmp_path, closed=(DAY, DAY + timedelta(days=1)))
    ledger.record_expectation(expectation("hotel", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3))
    morning = at(DAY + timedelta(days=1), 8, 0)
    ledger.record_expectation(
        replace(
            expectation("hotel-next", HOTEL, morning, window_minutes=(60, 120), deadline_hours=5),
            trigger_instance="behavior://flight-2.md",
        )
    )
    publish(site.behavior_tree, DAY + timedelta(days=1), "订酒店", 9, 0, kind="订酒店")
    report = settlement.run_once()
    assert report.outcomes == {"过期待决": 1, "准时": 1} and report.rulings == 0
    assert ledger.rulings_on(DAY) == ()


def test_another_accounts_next_chance_does_not_block_a_catch_up(tmp_path) -> None:
    site, ledger, settlement = stage(tmp_path, closed=(DAY, DAY + timedelta(days=1)))
    ledger.record_expectation(expectation("hotel", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3))
    morning = at(DAY + timedelta(days=1), 8, 0)
    ledger.record_expectation(
        replace(
            expectation("hotel-next", HOTEL, morning, window_minutes=(60, 120), deadline_hours=5),
            source="日子",
            trigger="interval",
        )
    )
    publish(site.behavior_tree, DAY + timedelta(days=1), "订酒店", 9, 0, kind="订酒店")
    report = settlement.run_once()
    assert report.rulings == 1 and ledger.rulings_on(DAY)[0].ruling == "后补"


def test_migration_in_progress_settles_nothing_and_reports_pending_days(tmp_path) -> None:
    _site, ledger, settlement = stage(tmp_path, closed=(DAY,), migrating=True)
    ledger.record_expectation(expectation("hotel", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3))
    report = settlement.run_once()
    assert report.settled == 0 and report.pending_days == (DAY,)
    assert ledger.settlements_on(DAY) == ()


def test_a_corrupt_ledger_file_is_skipped_and_counted_not_fatal(tmp_path) -> None:
    recorder = Recorder()
    _site, ledger, settlement = stage(tmp_path, closed=(DAY,), observer=recorder)
    ledger.record_expectation(expectation("hotel", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3))
    (tmp_path / "ledger" / "expectations" / "2026" / "12" / "10" / "hotel.json").write_text("{", encoding="utf-8")
    report = settlement.run_once()
    assert report.settled == 0
    (event,) = recorder.events
    assert event.status.value == "success" and event.attributes["unreadable"] == 1


def test_a_crash_between_the_facts_and_the_settlement_is_healed_on_the_next_night(tmp_path, monkeypatch) -> None:
    site, ledger, settlement = stage(tmp_path, closed=(DAY,))
    hotel = expectation("hotel", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3)
    ledger.record_expectation(hotel)
    publish(site.behavior_tree, DAY, "订酒店", 14, 28, kind="订酒店")

    def crash(*_args: object) -> bool:
        raise OSError("disk went away")

    monkeypatch.setattr(ledger, "append_fact", crash)
    try:
        settlement.run_once()
    except OSError:
        pass
    monkeypatch.undo()
    # 事实没写成，结算也就没写：下一晚照结一遍，事实补上
    assert ledger.settlements_on(DAY) == ()
    settlement.run_once()
    assert [item.outcome for item in ledger.settlements_on(DAY)] == ["准时"]
    (fact,) = ledger.facts_of(hotel.account)
    assert fact.kind == "准时" and fact.at == hotel.formed_at


def test_a_day_with_everything_settled_and_nothing_left_to_rule_is_final_and_not_read_again(tmp_path) -> None:
    site, ledger, settlement = stage(tmp_path, closed=(DAY,))
    ledger.record_expectation(expectation("hotel", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3))
    publish(site.behavior_tree, DAY, "订酒店", 14, 28, kind="订酒店")
    settlement.run_once()
    assert ledger.is_final(DAY) and ledger.open_days() == ()
    # 过期待决、又没有间隔可比（补做的强度恒为 1，裁不出不同的结果）：也算结完
    other = DAY + timedelta(days=3)
    _site, ledger2, settlement2 = stage(tmp_path / "b", closed=(other,))
    ledger2.record_expectation(expectation("card", CARD, at(other, 9, 0), window_minutes=(0, 30), deadline_hours=3))
    settlement2.run_once()
    assert [item.outcome for item in ledger2.settlements_on(other)] == ["过期待决"] and ledger2.is_final(other)


def test_a_catch_up_is_weighed_by_how_long_since_the_last_occurrence_not_since_the_deadline() -> None:
    """周二周四跑步：周四漏了，周五补跑是半个坏；拖到下周二（离上次隔了 7 天，比哪一次正常间隔都长）就是整笔坏。"""

    from habitus.foresight.ledger import Ruling
    from habitus.foresight.opportunities import KindDay, KindDays, OpportunityTable
    from habitus.runtime.foresight_settlement import SettlementStage

    tz = ZoneInfo("Asia/Shanghai")
    runs = [date(2026, 9, 1) + timedelta(days=d) for d in range(0, 56) if (date(2026, 9, 1) + timedelta(days=d)).weekday() in (1, 3)]
    history = tuple(KindDay(day, f"run-{day}", datetime.combine(day, datetime.min.time(), tz).replace(hour=19), 1, 1) for day in runs[:-1])
    table = OpportunityTable("session", runs[-1], runs[-1], 15, 3, tuple(runs), (), (KindDays("s-run", "跑步", history),))
    thursday = runs[-1]
    formed = datetime.combine(thursday, datetime.min.time(), tz).replace(hour=19)
    item = replace(
        expectation("run", "s-run", formed, window_minutes=(0, 30), deadline_hours=5),
        source="钟面",
        trigger="weekday=3",
        observed_from=formed.replace(hour=0),
    )
    friday = Ruling("run", formed + timedelta(days=2), "后补", "run-fri", 14 * 60 + 60)
    tuesday = Ruling("run", formed + timedelta(days=6), "后补", "run-tue", (4 * 24 + 14) * 60)
    assert SettlementStage._catch_up_fact(item, friday, table).strength == 0.5
    assert SettlementStage._catch_up_fact(item, tuesday, table).strength == 1.0


def test_an_event_expectation_whose_antecedent_is_now_another_class_is_voided_not_counted(tmp_path) -> None:
    """事件触发在前因没封口时建；封口后那条前因归成了别的类：这次机会不存在，账上只记「作废」（权重 0），不裁后补（裁定 54）。"""

    from habitus.foresight.opportunities import OpportunityTable, RelationOpportunities
    from habitus.runtime.foresight_opportunities import OpportunityStore

    site = Site(tmp_path, now=at(DAY, 23, 0))
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, NominalCalendar(), ZoneInfo("Asia/Shanghai"))
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    store = OpportunityStore(tmp_path / "ledger")
    relation = RelationOpportunities(
        key="session|flight|hotel|chain",
        antecedent="flight",
        consequent="hotel",
        antecedent_label="订机票",
        consequent_label="订酒店",
        antecedent_classes=(FLIGHT,),
        consequent_classes=(HOTEL,),
        segment="chain",
        status="established",
        upward=True,
        items=(),
    )
    store.write(OpportunityTable("session", DAY, DAY, 15, 3, (DAY,), (relation,), ()))
    settlement = SettlementStage(
        ledger,
        read_side,
        closed_days=lambda: (DAY, DAY + timedelta(days=1)),
        current=lambda token: (token,),
        migrating=lambda: False,
        evidence=Evidence(alpha=0.05, half_life_days=60.0),
        opportunities=store,
        clock=lambda: at(DAY + timedelta(days=2), 4, 0).astimezone(UTC),
    )
    started = at(DAY, 14, 0)
    publish(site.behavior_tree, DAY, "问机票价格", 14, 0, kind="还信用卡")  # 那条前因封口后成了别的类
    hotel = replace(
        expectation("hotel", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3),
        trigger_instance=started.astimezone(UTC).isoformat(),
    )
    ledger.record_expectation(hotel)
    publish(site.behavior_tree, DAY + timedelta(days=1), "订酒店", 9, 0, kind="订酒店")  # 次日才订：本来会裁成后补
    report = settlement.run_once()
    (fact,) = ledger.facts_of(hotel.account)
    assert (fact.kind, fact.strength) == ("作废", 0.0)
    assert report.rulings == 0 and ledger.rulings_on(DAY) == ()


def test_a_miss_under_an_observation_gap_is_neutral_not_bad(tmp_path) -> None:
    """说中的范围里有一段没观测到：他可能做了、我们没看到。照结过期待决，但账上那一笔权重 0（裁定 54：他其实做了）。"""

    site, ledger, settlement = stage(tmp_path, closed=(DAY,))
    hotel = expectation("hotel", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3)
    ledger.record_expectation(hotel)
    publish_gap(site.behavior_tree, DAY, (14, 30), (15, 30))
    settlement.run_once()
    assert [item.outcome for item in ledger.settlements_on(DAY)] == ["过期待决"]
    (fact,) = ledger.facts_of(hotel.account)
    assert (fact.kind, fact.strength) == ("过期待决", 0.0)
    # 没有空白的另一条照常是一笔坏
    _site2, ledger2, settlement2 = stage(tmp_path / "b", closed=(DAY,))
    ledger2.record_expectation(hotel)
    settlement2.run_once()
    (plain,) = ledger2.facts_of(hotel.account)
    assert (plain.kind, plain.strength) == ("过期待决", 1.0)


def test_the_nightly_diagnosis_reads_each_miss_from_the_ledger_and_the_tree(tmp_path) -> None:
    """说出口了却没做的一条：账上什么解释都没有 → 说不清；窗里有观测空白的另一条 → 他可能做了、我们没看到。只读，不改账。"""

    from habitus.foresight import Tolerance
    from habitus.foresight.diagnosis import UNEXPLAINED, UNSEEN

    site, ledger, settlement = stage(tmp_path, closed=(DAY,))
    settlement.tolerance = Tolerance(floor=0.2, ceiling=0.9, unjudged=0.5)
    said = expectation("said", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3)
    ledger.record_expectation(said)
    ledger.record_delivery(DAY, Delivery("said", at(DAY, 14, 15), 1, True, "说了", 1.0, "要订酒店吗"))
    unseen = replace(
        expectation("unseen", CARD, at(DAY, 18, 0), window_minutes=(0, 30), deadline_hours=2),
        trigger="session|x|card|chain",
    )
    ledger.record_expectation(unseen)
    ledger.record_delivery(DAY, Delivery("unseen", at(DAY, 18, 5), 1, True, "说了", 1.0, "还卡了吗"))
    publish_gap(site.behavior_tree, DAY, (18, 10), (19, 0))
    settlement.run_once()
    before = {account: ledger.facts_of(account) for account in (said.account, unseen.account)}
    found = settlement.diagnose(now=at(DAY + timedelta(days=1), 4, 0))
    causes = {miss.expectation_id: cause for miss, cause in found.misses}
    assert causes == {"said": UNEXPLAINED, "unseen": UNSEEN}
    assert found.counts(spoken_only=True)[UNEXPLAINED] == 1
    assert {account: ledger.facts_of(account) for account in before} == before  # 只读
    # 校准表：中性的那条不进；说错的那条算一次没中
    (row,) = found.calibration
    assert row.expectations == 1 and row.claimed == pytest.approx(10 / 13)


def test_resettle_rewrites_only_the_facts_the_current_code_would_write_differently(tmp_path) -> None:
    """修了 bug 之后重结：账上那一笔和新代码算出的不一样就追加改判（后写的覆盖先写的）；一样的不动；后补这类之后的改判不盖。"""

    from habitus.foresight import current_facts as live
    from habitus.foresight.ledger import AccountFact as Fact

    site, ledger, settlement = stage(tmp_path, closed=(DAY,))
    hotel = expectation("hotel", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3)
    ledger.record_expectation(hotel)
    publish(site.behavior_tree, DAY, "订酒店", 14, 28, kind="订酒店")
    settlement.run_once()
    assert settlement.resettle([DAY]) == 0  # 代码没变：一条不改
    # 假装旧代码当时写错了：把它记成了过期待决
    wrong = Fact(hotel.account, hotel.formed_at, "预测", False, 1.0, "过期待决", "hotel", written_at=at(DAY, 23, 0))
    ledger.append_fact(wrong)
    assert [f.kind for f in live(ledger.facts_of(hotel.account))] == ["过期待决"]
    assert settlement.resettle([DAY]) == 1
    assert [f.kind for f in live(ledger.facts_of(hotel.account))] == ["准时"]
    # 账上最新的是后补（之后的改判）：重结不盖它
    caught = Fact(hotel.account, hotel.formed_at, "预测", False, 0.4, "后补", "hotel", written_at=at(DAY + timedelta(days=3), 4, 0))
    ledger.append_fact(caught)
    assert settlement.resettle([DAY]) == 0


# ── 审计回归（2026-10-17）：重结、崩溃重来、诊断按当时、前因找行、别的账的提醒 ─────────────────────────


def _resettle_stage(tmp_path, closed, *, now):  # type: ignore[no-untyped-def]
    from habitus.foresight import Tolerance

    site = Site(tmp_path, now=at(DAY, 23, 0))
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, NominalCalendar(), ZoneInfo("Asia/Shanghai"))
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    settlement = SettlementStage(
        ledger,
        read_side,
        closed_days=lambda: closed,
        current=lambda token: (token,),
        migrating=lambda: False,
        evidence=Evidence(alpha=0.05, half_life_days=60.0),
        clock=lambda: now,
        tolerance=Tolerance(floor=0.2, ceiling=0.9, unjudged=0.5),
    )
    return site, ledger, settlement


def test_resettle_with_nothing_changed_changes_nothing_because_history_only_looks_backwards(tmp_path) -> None:
    """四天订酒店，偏差各不相同：每一笔的强度只拿它之前那几次比。之前拿全账（含之后的）比，重结时历史变了，没改代码也改出两笔。"""

    days = tuple(DAY + timedelta(days=d) for d in range(4))
    site, ledger, settlement = _resettle_stage(tmp_path, days, now=at(days[-1] + timedelta(days=1), 4, 0))
    for index, (day, minute) in enumerate(zip(days, (40, 20, 0, 50), strict=True)):
        formed = at(day, 14, 0)
        ledger.record_expectation(
            replace(expectation(f"h{index}", HOTEL, formed, window_minutes=(15, 25), deadline_hours=3), day=day)
        )
        publish(site.behavior_tree, day, "订酒店", 14 if minute else 15, minute or 10, kind="订酒店")
    settlement.run_once()
    assert settlement.resettle(days) == 0


def test_resettle_keeps_the_opportunity_time_retracts_an_axis_that_no_longer_applies_and_diagnosis_follows(tmp_path) -> None:
    """旧代码没看见那段观测空白：说了没做，记了一笔坏和一笔落空。修好后重结：预测轴改成中性，提醒组那笔落空用强度 0 盖掉；
    改判的机会时刻不变（不重置遗忘），写下的时刻是此刻；诊断读改判后的，归到"没看到"。"""

    from habitus.foresight import current_facts as live
    from habitus.foresight.diagnosis import UNSEEN

    now = at(DAY + timedelta(days=2), 4, 0)
    site, ledger, settlement = _resettle_stage(tmp_path, (DAY,), now=now)
    hotel = expectation("hotel", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3)
    ledger.record_expectation(hotel)
    ledger.record_delivery(DAY, Delivery("hotel", at(DAY, 14, 15), 1, True, "说了", 1.0, "要订酒店吗"))
    settlement.run_once()
    assert sorted((f.axis, f.strength > 0) for f in live(ledger.facts_of(hotel.account))) == [("提醒组", True), ("预测", True)]
    publish_gap(site.behavior_tree, DAY, (14, 30), (15, 30))
    assert settlement.resettle([DAY]) == 1
    current = {f.axis: f for f in live(ledger.facts_of(hotel.account))}
    assert (current["预测"].kind, current["预测"].strength) == ("过期待决", 0.0)
    assert current["提醒组"].strength == 0.0
    assert all(f.at == hotel.formed_at and f.written_at == now for f in current.values())
    (miss,) = settlement.diagnose(now=now).misses
    assert miss[1] == UNSEEN
    assert settlement.resettle([DAY]) == 0  # 再来一遍：已经一样了


def test_resettle_back_to_overdue_reopens_a_final_day_so_a_catch_up_can_still_be_ruled(tmp_path) -> None:
    """旧代码把一次没做记成了准时、那天标了结完。重结改回过期待决：那天重新打开，之后照常裁后补；诊断与校准跟着改判走。"""

    from habitus.foresight import current_facts as live
    from habitus.foresight.ledger import AccountFact as Fact
    from habitus.foresight.ledger import Settlement

    later = DAY + timedelta(days=1)
    site, ledger, settlement = _resettle_stage(tmp_path, (DAY, later), now=at(DAY + timedelta(days=2), 4, 0))
    hotel = expectation("hotel", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3)
    ledger.record_expectation(hotel)
    ledger.record_settlement(Settlement("hotel", HOTEL, DAY, at(DAY, 23, 0), "准时", "behavior://ghost.md"))
    ledger.append_fact(Fact(hotel.account, hotel.formed_at, "预测", True, 1.0, "准时", "hotel"))
    ledger.mark_final(DAY, at=at(DAY, 23, 0))
    assert settlement.resettle([DAY]) == 1
    assert not ledger.is_final(DAY)
    publish(site.behavior_tree, later, "订酒店", 9, 0, kind="订酒店")
    report = settlement.run_once()
    assert report.rulings == 1
    assert [f.kind for f in live(ledger.facts_of(hotel.account))] == ["后补"]
    (miss,) = settlement.diagnose(now=at(DAY + timedelta(days=2), 4, 0)).misses
    assert miss[0].expectation_id == "hotel"


def test_a_crash_after_the_facts_does_not_wedge_the_night_when_the_history_has_grown(tmp_path, monkeypatch) -> None:
    """事实写下、结算没写就断了；第二晚之前账上多了一笔更早的机会，重算出的落空强度不同：沿用已写下的原判，不撞同一身份。"""

    from habitus.foresight.ledger import AccountFact as Fact

    site, ledger, settlement = _resettle_stage(tmp_path, (DAY,), now=at(DAY + timedelta(days=1), 4, 0))
    hotel = expectation("hotel", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3)
    ledger.record_expectation(hotel)
    ledger.record_delivery(DAY, Delivery("hotel", at(DAY, 14, 15), 1, True, "说了", 1.0, "要订酒店吗"))

    def crash(*_args: object) -> bool:
        raise OSError("disk went away")

    monkeypatch.setattr(ledger, "record_settlement", crash)
    with pytest.raises(OSError):
        settlement.run_once()
    monkeypatch.undo()
    ledger.append_fact(Fact(hotel.account, hotel.formed_at - timedelta(days=7), "预测", True, 1.0, "准时", "older"))
    settlement.run_once()
    assert [item.outcome for item in ledger.settlements_on(DAY)] == ["过期待决"]


def test_diagnosis_judges_a_miss_by_the_account_as_it_stood_then_not_now(tmp_path) -> None:
    """落空那一刻这本账下界还在线上：那是预算内的一份。之后连错一串把账打下去了，回头看那一次仍是预算内，不是说不清。"""

    from habitus.foresight.ledger import AccountFact as Fact
    from habitus.foresight.ledger import Settlement

    now = at(DAY + timedelta(days=12), 4, 0)
    _site, ledger, settlement = _resettle_stage(tmp_path, (DAY,), now=now)
    hotel = expectation("hotel", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3)
    ledger.record_expectation(hotel)
    ledger.record_settlement(Settlement("hotel", HOTEL, DAY, at(DAY, 23, 0), "过期待决"))
    for index in range(40):
        ledger.append_fact(Fact(hotel.account, hotel.formed_at - timedelta(days=40 - index), "预测", True, 1.0, "准时", f"old{index}"))
    ledger.append_fact(Fact(hotel.account, hotel.formed_at, "预测", False, 1.0, "过期待决", "hotel"))
    for index in range(10):
        ledger.append_fact(Fact(hotel.account, hotel.formed_at + timedelta(days=1 + index), "预测", False, 1.0, "过期待决", f"new{index}"))
    (miss,) = settlement.diagnose(now=now).misses
    assert miss[0].within_budget


def test_a_quick_catch_up_is_not_diagnosed_as_unseen(tmp_path) -> None:
    """补得比他最短的正常间隔还快，后补的强度是 0——那是树上看见了的补做，不是"他可能做了、我们没看到"。"""

    from habitus.foresight.diagnosis import UNSEEN
    from habitus.foresight.ledger import AccountFact as Fact
    from habitus.foresight.ledger import Settlement

    now = at(DAY + timedelta(days=2), 4, 0)
    _site, ledger, settlement = _resettle_stage(tmp_path, (DAY,), now=now)
    hotel = expectation("hotel", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3)
    ledger.record_expectation(hotel)
    ledger.record_settlement(Settlement("hotel", HOTEL, DAY, at(DAY, 23, 0), "过期待决"))
    ledger.append_fact(Fact(hotel.account, hotel.formed_at, "预测", False, 1.0, "过期待决", "hotel"))
    ledger.append_fact(
        Fact(hotel.account, hotel.formed_at, "预测", False, 0.0, "后补", "hotel", written_at=at(DAY + timedelta(days=1), 4, 0))
    )
    (miss,) = settlement.diagnose(now=now).misses
    assert miss[1] != UNSEEN


def test_an_antecedent_whose_start_was_corrected_still_holds(tmp_path) -> None:
    """事件触发记的是前因没封口时的开始时刻（14:00）；封口时更正成 13:58 开始、到 14:08。还是那一行、还是那一类：不作废。"""

    from habitus.foresight.opportunities import OpportunityTable, RelationOpportunities
    from habitus.runtime.foresight_opportunities import OpportunityStore

    site = Site(tmp_path, now=at(DAY, 23, 0))
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, NominalCalendar(), ZoneInfo("Asia/Shanghai"))
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    store = OpportunityStore(tmp_path / "ledger")
    relation = RelationOpportunities(
        key="session|flight|hotel|chain",
        antecedent="flight",
        consequent="hotel",
        antecedent_label="订机票",
        consequent_label="订酒店",
        antecedent_classes=(FLIGHT,),
        consequent_classes=(HOTEL,),
        segment="chain",
        status="established",
        upward=True,
        items=(),
    )
    store.write(OpportunityTable("session", DAY, DAY, 15, 3, (DAY,), (relation,), ()))
    settlement = SettlementStage(
        ledger,
        read_side,
        closed_days=lambda: (DAY,),
        current=lambda token: (token,),
        migrating=lambda: False,
        evidence=Evidence(alpha=0.05, half_life_days=60.0),
        opportunities=store,
        clock=lambda: at(DAY + timedelta(days=1), 4, 0).astimezone(UTC),
    )
    publish(site.behavior_tree, DAY, "订机票", 13, 58, kind="订机票")
    hotel = replace(
        expectation("hotel", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3),
        trigger_instance=at(DAY, 14, 0).astimezone(UTC).isoformat(),
    )
    ledger.record_expectation(hotel)
    publish(site.behavior_tree, DAY, "订酒店", 14, 30, kind="订酒店")
    settlement.run_once()
    (fact,) = ledger.facts_of(hotel.account)
    assert fact.kind == "准时"


def test_an_occurrence_drawn_out_by_another_accounts_reminder_is_no_natural_hit_here(tmp_path) -> None:
    """同一类的钟面账开了口、他随后做了（树上盖了 reminded、发起者 system）；这条事件触发的预期自己没说过，窗里也盖着这次：
    它不是这本账的自然命中，记一笔强度 0 的「提醒后发生」，不进校准。"""

    site, ledger, settlement = _resettle_stage(tmp_path, (DAY,), now=at(DAY + timedelta(days=1), 4, 0))
    hotel = expectation("hotel", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3)
    ledger.record_expectation(hotel)
    publish(site.behavior_tree, DAY, "订酒店", 14, 28, kind="订酒店", reminded=True, prompted_by="system")
    settlement.run_once()
    (fact,) = ledger.facts_of(hotel.account)
    assert (fact.kind, fact.good, fact.strength) == ("提醒后发生", True, 0.0)
    assert settlement.diagnose(now=at(DAY + timedelta(days=1), 4, 0)).calibration == ()


def test_the_cause_stage_explains_each_visible_miss_once_without_moving_the_reading(tmp_path) -> None:
    """裁定 55：结算结出的落空，同一晚机会表那一拍对照答案找原因，作为同值改判挂上去——好坏、强度不变，账的读数不动；
    一天的都找过了标结，第二晚不再读；最晚时限还不在这一晚时间线里的等下一晚；中性的（观测空白）不找。"""

    from habitus.foresight import Tolerance, climb
    from habitus.foresight import current_facts as live
    from habitus.foresight.ledger import AccountFact as Fact
    from habitus.foresight.ledger import Settlement
    from habitus.runtime.foresight_causes import CauseStage
    from tests.unit.foresight.test_causes import CONCEPTS as CAUSE_CONCEPTS
    from tests.unit.foresight.test_causes import FAMILY, TRAVEL, hotel_after_flight, line_of

    ledger = ForesightLedgerStore(tmp_path / "ledger")
    monday = date(2026, 6, 1)
    line, rows = line_of((monday, 9, 0, "看邮件"), (monday, 10, 0, "订机票"), (monday, 10, 30, "订家人机票"))
    miss = hotel_after_flight(at(monday, 10, 0))
    gap = replace(hotel_after_flight(at(monday, 13, 0)), expectation_id="gap")
    for item in (miss, gap):
        ledger.record_expectation(item)
        ledger.record_settlement(Settlement(item.expectation_id, item.kind_token, monday, at(monday, 23, 0), "过期待决"))
    ledger.append_fact(Fact(miss.account, miss.formed_at, "预测", False, 1.0, "过期待决", "hotel"))
    ledger.append_fact(Fact(gap.account, gap.formed_at, "预测", False, 0.0, "过期待决", "gap"))
    stage = CauseStage(
        ledger, lane="session", current=lambda token: (token,), calendar=NominalCalendar(), clock=lambda: at(monday + timedelta(days=1), 4, 0)
    )
    tolerance, evidence = Tolerance(floor=0.2, ceiling=0.9, unjudged=0.5), Evidence(alpha=0.05, half_life_days=60.0)
    before = climb(ledger.facts_of(miss.account), tolerance=tolerance, evidence=evidence, now=at(monday, 23, 0))
    assert stage.run(line=line, concepts=CAUSE_CONCEPTS, relations=(), window_slots=3) == 1
    current = {fact.expectation_id: fact for fact in live(ledger.facts_of(miss.account))}
    cause = current["hotel"].cause
    assert cause is not None and (cause.kind, cause.classes, cause.relation) == ("同一件事", (FAMILY,), TRAVEL)
    assert cause.uris == (rows["订家人机票2"].uri,)
    assert (current["hotel"].kind, current["hotel"].good, current["hotel"].strength) == ("过期待决", False, 1.0)
    assert current["gap"].cause is None  # 中性的不找
    after = climb(ledger.facts_of(miss.account), tolerance=tolerance, evidence=evidence, now=at(monday, 23, 0))
    assert after.prediction == before.prediction
    assert ledger.explained(monday)
    assert stage.run(line=line, concepts=CAUSE_CONCEPTS, relations=(), window_slots=3) == 0
    # 另一条 lane 的时间线不碰这本账；最晚时限还不在时间线里的（明天的预期、今晚的线只到今天）等下一晚
    tuesday = monday + timedelta(days=1)
    late = replace(hotel_after_flight(at(tuesday, 10, 0)), expectation_id="late")
    ledger.record_expectation(late)
    ledger.record_settlement(Settlement("late", late.kind_token, tuesday, at(tuesday, 23, 0), "过期待决"))
    ledger.append_fact(Fact(late.account, late.formed_at, "预测", False, 1.0, "过期待决", "late"))
    assert stage.run(line=line, concepts=CAUSE_CONCEPTS, relations=(), window_slots=3) == 0
    assert not ledger.explained(tuesday)


def test_diagnosis_names_the_found_cause_and_counts_whether_holding_back_was_right(tmp_path) -> None:
    """上次落空找到的原因进诊断（排在规律断了之前）；因为它今天又在而没说的那一次，他果然没做 → 调整对了一次。"""

    from habitus.foresight.ledger import AccountFact as Fact
    from habitus.foresight.ledger import Cause, Settlement

    now = at(DAY + timedelta(days=3), 4, 0)
    _site, ledger, settlement = _resettle_stage(tmp_path, (DAY,), now=now)
    first = expectation("first", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3)
    second = replace(expectation("second", HOTEL, at(DAY + timedelta(days=1), 14, 7), window_minutes=(4, 51), deadline_hours=3), day=DAY + timedelta(days=1))
    for item in (first, second):
        ledger.record_expectation(item)
        ledger.record_settlement(Settlement(item.expectation_id, HOTEL, item.day, at(item.day, 23, 0), "过期待决"))
        ledger.append_fact(Fact(item.account, item.formed_at, "预测", False, 1.0, "过期待决", item.expectation_id))
    ledger.append_fact(
        Fact(first.account, first.formed_at, "预测", False, 1.0, "过期待决", "first", written_at=at(DAY + timedelta(days=1), 4, 0), cause=Cause("线索没了", relation="r"))
    )
    ledger.record_delivery(second.day, Delivery("second", at(second.day, 15, 0), 1, False, "上次的原因今天又在", 0.0, evidence=("原因:first",)))
    found = settlement.diagnose(now=now)
    causes = {miss.expectation_id: cause for miss, cause in found.misses}
    assert causes["first"] == "线索没了"
    assert (found.adjustments["线索没了"].held, found.adjustments["线索没了"].right) == (1, 1)


def test_the_substitution_table_reads_what_he_did_next_from_the_tree(tmp_path) -> None:
    """结过的预期各取"时窗开始之后接着做的第一件别的事"（行为树上读）：没订酒店那天接着还了信用卡，订了的那天订完也接着还了。"""

    later = DAY + timedelta(days=1)
    site, ledger, settlement = _resettle_stage(tmp_path, (DAY, later), now=at(later + timedelta(days=1), 4, 0))
    missed = expectation("missed", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3)
    done = replace(expectation("done", HOTEL, at(later, 14, 7), window_minutes=(4, 51), deadline_hours=3), day=later)
    for item in (missed, done):
        ledger.record_expectation(item)
    publish(site.behavior_tree, DAY, "还信用卡", 14, 20, kind="还信用卡")
    publish(site.behavior_tree, later, "订酒店", 14, 20, kind="订酒店")
    publish(site.behavior_tree, later, "还信用卡", 15, 0, kind="还信用卡")
    settlement.run_once()
    (row,) = settlement.diagnose(now=at(later + timedelta(days=1), 4, 0)).substitutions
    assert (row.next_kind, row.after_misses, row.misses, row.after_hits, row.hits) == (CARD, 1, 1, 1, 1)
    assert not row.flagged  # 一次对一次，不下结论


def test_held_out_and_reminded_expectations_land_on_their_own_unit_weight_groups(tmp_path) -> None:
    """裁定 58 第 6、7 条：说了没做 → 提醒组一笔单位权重的坏（不再取此刻的准确率当强度）；随机扣下没说之后他自己做了 →
    预测轴照常一笔好、扣发组一笔好；扣下之后没做 → 扣发组一笔坏。提醒的效果 = 两组完成率之差。"""

    from habitus.foresight import current_facts as live

    later = DAY + timedelta(days=1)
    site, ledger, settlement = _resettle_stage(tmp_path, (DAY, later), now=at(later + timedelta(days=1), 4, 0))
    spoken = expectation("spoken", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3)
    held_done = replace(expectation("held-done", CARD, at(DAY, 9, 0), window_minutes=(0, 30), deadline_hours=3), trigger="session|x|card|chain")
    held_missed = replace(
        expectation("held-missed", CARD, at(later, 9, 0), window_minutes=(0, 30), deadline_hours=3),
        trigger="session|x|card|chain",
        day=later,
    )
    for item in (spoken, held_done, held_missed):
        ledger.record_expectation(item)
    ledger.record_delivery(DAY, Delivery("spoken", at(DAY, 14, 15), 1, True, "说了", 1.0, "要订酒店吗"))
    ledger.record_delivery(DAY, Delivery("held-done", at(DAY, 9, 5), 1, False, "随机扣发", 0.9))
    ledger.record_delivery(later, Delivery("held-missed", at(later, 9, 5), 1, False, "随机扣发", 0.9))
    publish(site.behavior_tree, DAY, "还信用卡", 9, 20, kind="还信用卡")
    settlement.run_once()
    facts = {(f.expectation_id, f.axis): f for f in live(ledger.facts_of(spoken.account))}
    assert (facts[("spoken", "提醒组")].good, facts[("spoken", "提醒组")].strength, facts[("spoken", "提醒组")].kind) == (False, 1.0, "落空")
    card = {(f.expectation_id, f.axis): f for f in live(ledger.facts_of(held_done.account))}
    assert card[("held-done", "预测")].good and card[("held-done", "预测")].strength > 0
    assert (card[("held-done", "扣发组")].good, card[("held-done", "扣发组")].kind) == (True, "扣发后发生")
    assert (card[("held-missed", "扣发组")].good, card[("held-missed", "扣发组")].kind) == (False, "扣发后落空")
    assert ("held-done", "提醒组") not in card and ("spoken", "扣发组") not in facts


def test_a_unit_moment_is_one_miss_and_one_cause_search_not_one_per_input(tmp_path) -> None:
    """同一刻两个输入都落空：依据那条记一次落空、找一次原因；「单元已选」引它的那条照常结算（它那本账的权重照调），但诊断不再
    把它算成第二次落空、找原因也不在它身上再找一遍。"""

    from habitus.foresight import current_facts as live
    from habitus.foresight.ledger import AccountFact as Fact
    from habitus.foresight.ledger import Settlement
    from habitus.runtime.foresight_causes import CauseStage
    from tests.unit.foresight.test_causes import CONCEPTS as CAUSE_CONCEPTS
    from tests.unit.foresight.test_causes import hotel_after_flight, line_of

    monday = date(2026, 6, 1)
    now = at(monday + timedelta(days=1), 4, 0)
    _site, ledger, settlement = _resettle_stage(tmp_path, (monday,), now=now)
    line, _rows = line_of((monday, 9, 0, "看邮件"), (monday, 10, 0, "订机票"), (monday, 10, 30, "订家人机票"))
    basis = hotel_after_flight(at(monday, 10, 0))
    member = replace(basis, expectation_id="member", source="钟面", trigger="weekday=0", trigger_instance=monday.isoformat())
    for item in (basis, member):
        ledger.record_expectation(item)
        ledger.record_settlement(Settlement(item.expectation_id, item.kind_token, monday, at(monday, 23, 0), "过期待决"))
        ledger.append_fact(Fact(item.account, item.formed_at, "预测", False, 1.0, "过期待决", item.expectation_id))
    ledger.record_delivery(monday, Delivery("hotel", at(monday, 10, 15), 0, False, "影子", 0.0))
    ledger.record_delivery(monday, Delivery("member", at(monday, 10, 15), 0, False, "单元已选", 0.0, evidence=("单元:hotel",)))
    stage = CauseStage(ledger, lane="session", current=lambda token: (token,), calendar=NominalCalendar(), clock=lambda: now)
    assert stage.run(line=line, concepts=CAUSE_CONCEPTS, relations=(), window_slots=3) == 1
    assert next(f for f in live(ledger.facts_of(basis.account)) if f.expectation_id == "hotel").cause is not None
    assert next(f for f in live(ledger.facts_of(member.account)) if f.expectation_id == "member").cause is None
    misses = [miss.expectation_id for miss, _cause in settlement.diagnose(now=now).misses]
    assert misses == ["hotel"]
