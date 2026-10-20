"""结算这一拍对着真实的行为树：只结最晚时限那天已定稿的；过期待决之后补做了记后补；定稿日的漏报查一次；
结算与裁决落账时往这本账上写事实（裁定 48、50、51）；说出口的那条给单元写提醒反馈（裁定 61）。"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from habitus.conditions import LocalCalendar
from habitus.foresight import Evidence, Tolerance
from habitus.foresight.edges import PastMoment
from habitus.foresight.ledger import Delivery, Expectation, Response
from habitus.foundation.observability import ObservationEvent
from habitus.runtime.foresight import ForesightReadSide
from habitus.runtime.foresight_history_store import HistoryStore
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_settlement import SettlementStage
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
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai"))
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    settlement = SettlementStage(
        ledger,
        read_side,
        closed_days=lambda: closed,
        current=lambda token: (token,),
        migrating=lambda: migrating,
        evidence=Evidence(alpha=0.05, half_life=8.0),
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
    from habitus.foundation.lifecycle import covers

    assert covers(ledger.scan_watermark(start=DAY), DAY)
    # 账上记了一笔好（窗内，强度 1）；没说出口：单元上没有提醒反馈
    (fact,) = ledger.facts_of(hotel.account)
    assert (fact.good, fact.kind, fact.strength, fact.offset_minutes) == (True, "准时", 1.0, 0)
    assert ledger.feedback_of(HOTEL) == ()
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
    # 账上是中性（提醒的功劳不算这个习惯本身，裁定 58 第 5 条）；单元上记一笔提醒反馈的好（裁定 61）
    (fact,) = ledger.facts_of(expectation("hotel", HOTEL, formed, window_minutes=(4, 51), deadline_hours=3).account)
    assert fact.good and fact.strength == 0.0 and fact.offset_minutes == 15
    (feedback,) = ledger.feedback_of(HOTEL)
    assert feedback.good and feedback.expectation_id == "hotel" and feedback.at == formed and feedback.polarity is None


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

    from habitus.foresight.facts import catch_up_fact
    from habitus.foresight.ledger import Ruling
    from habitus.foresight.opportunities import KindDays, OpportunityTable
    from habitus.temporal import DayInterval, OccurrenceDay

    tz = ZoneInfo("Asia/Shanghai")
    runs = [
        date(2026, 9, 1) + timedelta(days=d)
        for d in range(0, 56)
        if (date(2026, 9, 1) + timedelta(days=d)).weekday() in (1, 3)
    ]
    starts = [datetime.combine(day, datetime.min.time(), tz).replace(hour=19) for day in runs[:-1]]
    history = tuple(OccurrenceDay(at.date(), f"run-{at.date()}", at, 1, 1, f"run-{at.date()}", at) for at in starts)
    intervals = tuple(
        DayInterval(a, b, (b.day - a.day).days, 1.0, False) for a, b in zip(history, history[1:], strict=False)
    )
    table = OpportunityTable("session", runs[-1], runs[-1], 15, 3, (), (KindDays("s-run", "跑步", history, intervals),))
    thursday = runs[-1]
    formed = datetime.combine(thursday, datetime.min.time(), tz).replace(hour=19)
    item = replace(
        expectation("run", "s-run", formed, window_minutes=(0, 30), deadline_hours=5),
        source="日子",
        trigger="weekday=3",
        observed_from=formed.replace(hour=0),
    )
    friday = Ruling("run", formed + timedelta(days=2), "后补", "run-fri", 14 * 60 + 60)
    tuesday = Ruling("run", formed + timedelta(days=6), "后补", "run-tue", (4 * 24 + 14) * 60)
    assert catch_up_fact(item, friday, table).strength == 0.5
    assert catch_up_fact(item, tuesday, table).strength == 1.0
    # 只拿这次机会之前的间隔比：今晚的日历里多了这次补做与之后天天跑的几天，补做的强度不变（回放可重复，2026-10-17 四方评审）
    later_starts = [
        datetime.combine(thursday + timedelta(days=d), datetime.min.time(), tz).replace(hour=19) for d in range(1, 8)
    ]
    everything = history + tuple(
        OccurrenceDay(at.date(), f"run-{at.date()}", at, 1, 1, f"run-{at.date()}", at) for at in later_starts
    )
    all_intervals = tuple(
        DayInterval(a, b, (b.day - a.day).days, 1.0, False) for a, b in zip(everything, everything[1:], strict=False)
    )
    tonight = replace(table, kinds=(KindDays("s-run", "跑步", everything, all_intervals),))
    assert catch_up_fact(item, friday, tonight).strength == 0.5


def test_an_event_expectation_whose_antecedent_is_now_another_class_is_voided_not_counted(tmp_path) -> None:
    """事件触发在前因没封口时建；封口后那条前因归成了别的类：这次机会不存在，账上只记「作废」（权重 0），不裁后补（裁定 54）。"""

    from habitus.foresight.opportunities import OpportunityTable, RelationOpportunities
    from habitus.runtime.foresight_opportunities import OpportunityStore

    site = Site(tmp_path, now=at(DAY, 23, 0))
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai"))
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
    store.write(OpportunityTable("session", DAY, DAY, 15, 3, (relation,), ()))
    settlement = SettlementStage(
        ledger,
        read_side,
        closed_days=lambda: (DAY, DAY + timedelta(days=1)),
        current=lambda token: (token,),
        migrating=lambda: False,
        evidence=Evidence(alpha=0.05, half_life=8.0),
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


@pytest.mark.parametrize(("holiday", "kind"), [(False, "过期待决"), (True, "作废")])
def test_a_situation_expectation_is_rechecked_against_the_latest_calendar(tmp_path, holiday: bool, kind: str) -> None:
    """「情境」预期（工作日 → 订酒店）按名义周末建在周四；后来放假数据补上、那天其实放假：前因不成立，这次机会不存在，记「作废」
    （25 号清单 4.7）。数据没变的那份照常结过期待决。"""

    from habitus.conditions import HolidayTable, HolidayYear
    from habitus.foresight.opportunities import OpportunityTable, RelationOpportunities
    from habitus.runtime.foresight_opportunities import OpportunityStore

    year = HolidayYear("CN", 2026, "测试", "1", frozenset({5, 6}), {DAY: "测试假"} if holiday else {}, frozenset())
    site = Site(tmp_path, now=at(DAY, 23, 0))
    read_side = ForesightReadSide(
        site.behavior_tree, SUBJECT, LocalCalendar(HolidayTable({2026: year})), ZoneInfo("Asia/Shanghai")
    )
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    store = OpportunityStore(tmp_path / "ledger")
    relation = RelationOpportunities(
        key="session|工作日|hotel|situation",
        antecedent="工作日",
        consequent="hotel",
        antecedent_label="工作日",
        consequent_label="订酒店",
        antecedent_classes=("工作日",),
        consequent_classes=(HOTEL,),
        segment="situation",
        status="established",
        upward=True,
        items=(),
        calendar_key="日历.日型",
        calendar_values=("工作日",),
    )
    store.write(OpportunityTable("session", DAY, DAY, 15, 3, (relation,), ()))
    settlement = SettlementStage(
        ledger,
        read_side,
        closed_days=lambda: (DAY, DAY + timedelta(days=1)),
        current=lambda token: (token,),
        migrating=lambda: False,
        evidence=Evidence(alpha=0.05, half_life=8.0),
        opportunities=store,
        tolerance=Tolerance(line=2 / 3),
        clock=lambda: at(DAY + timedelta(days=2), 4, 0).astimezone(UTC),
    )
    formed = at(DAY, 9, 0)
    hotel = replace(
        expectation("hotel", HOTEL, formed, window_minutes=(0, 60), deadline_hours=15),
        source="情境",
        trigger=relation.key,
        trigger_instance=DAY.isoformat(),
        observed_from=at(DAY, 0, 0),
    )
    ledger.record_expectation(hotel)
    settlement.run_once()
    (fact,) = ledger.facts_of(hotel.account)
    assert fact.kind == kind
    if not holiday:  # 之后那一晚日历关系不在表上了：诊断说"关系没了"（情境和事件一样有关系撑着）
        store.write(OpportunityTable("session", DAY + timedelta(days=1), DAY + timedelta(days=1), 15, 3, (), ()))
        (miss,) = settlement.diagnose(now=at(DAY + timedelta(days=2), 4, 0)).misses
        assert miss[0].relation_gone


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
    settlement.tolerance = Tolerance(line=2 / 3)
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
    wrong = Fact(hotel.account, hotel.formed_at, False, 1.0, "过期待决", "hotel", written_at=at(DAY, 23, 0))
    ledger.append_fact(wrong)
    assert [f.kind for f in live(ledger.facts_of(hotel.account))] == ["过期待决"]
    assert settlement.resettle([DAY]) == 1
    assert [f.kind for f in live(ledger.facts_of(hotel.account))] == ["准时"]
    # 账上最新的是后补（之后的改判）：重结不盖它
    caught = Fact(
        hotel.account, hotel.formed_at, False, 0.4, "后补", "hotel", written_at=at(DAY + timedelta(days=3), 4, 0)
    )
    ledger.append_fact(caught)
    assert settlement.resettle([DAY]) == 0


# ── 审计回归（2026-10-17）：重结、崩溃重来、诊断按当时、前因找行、别的账的提醒 ─────────────────────────


def _resettle_stage(tmp_path, closed, *, now):  # type: ignore[no-untyped-def]
    from habitus.foresight import Tolerance

    site = Site(tmp_path, now=at(DAY, 23, 0))
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai"))
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    settlement = SettlementStage(
        ledger,
        read_side,
        closed_days=lambda: closed,
        current=lambda token: (token,),
        migrating=lambda: False,
        evidence=Evidence(alpha=0.05, half_life=8.0),
        clock=lambda: now,
        tolerance=Tolerance(line=2 / 3),
        history=HistoryStore(tmp_path / "ledger"),
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


def test_resettle_keeps_the_opportunity_time_and_diagnosis_follows(tmp_path) -> None:
    """旧代码没看见那段观测空白：说了没做，记了一笔坏。修好后重结：改成中性；改判的机会时刻不变（不重置遗忘），写下的时刻
    是此刻；诊断读改判后的，归到"没看到"。单元的提醒反馈不随重结改（它记的是说了之后做没做）。"""

    from habitus.foresight import current_facts as live
    from habitus.foresight.diagnosis import UNSEEN

    now = at(DAY + timedelta(days=2), 4, 0)
    site, ledger, settlement = _resettle_stage(tmp_path, (DAY,), now=now)
    hotel = expectation("hotel", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3)
    ledger.record_expectation(hotel)
    ledger.record_delivery(DAY, Delivery("hotel", at(DAY, 14, 15), 1, True, "说了", 1.0, "要订酒店吗"))
    settlement.run_once()
    (before,) = live(ledger.facts_of(hotel.account))
    assert before.strength > 0 and [item.good for item in ledger.feedback_of(HOTEL)] == [False]
    publish_gap(site.behavior_tree, DAY, (14, 30), (15, 30))
    assert settlement.resettle([DAY]) == 1
    (current,) = live(ledger.facts_of(hotel.account))
    assert (current.kind, current.strength) == ("过期待决", 0.0)
    assert current.at == hotel.formed_at and current.written_at == now
    assert [item.good for item in ledger.feedback_of(HOTEL)] == [False]
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
    ledger.append_fact(Fact(hotel.account, hotel.formed_at, True, 1.0, "准时", "hotel"))
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
    ledger.append_fact(Fact(hotel.account, hotel.formed_at - timedelta(days=7), True, 1.0, "准时", "older"))
    settlement.run_once()
    assert [item.outcome for item in ledger.settlements_on(DAY)] == ["过期待决"]


def test_diagnosis_judges_a_miss_by_the_edge_as_it_stood_then_not_now(tmp_path) -> None:
    """落空那一刻这条边下界还在线上：那是预算内的一份。之后连错一串把边打下去了，回头看那一次仍是预算内，不是说不清。
    边从单元历史（夜里重标的快照）读，截到落空那一刻（裁定 61）。"""

    from habitus.foresight.edges import PastMoment
    from habitus.foresight.history import Moment, UnitHistory
    from habitus.foresight.ledger import AccountFact as Fact
    from habitus.foresight.ledger import Settlement

    now = at(DAY + timedelta(days=12), 4, 0)
    _site, ledger, settlement = _resettle_stage(tmp_path, (DAY,), now=now)
    hotel = expectation("hotel", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3)
    ledger.record_expectation(hotel)
    ledger.record_settlement(Settlement("hotel", HOTEL, DAY, at(DAY, 23, 0), "过期待决"))
    ledger.append_fact(Fact(hotel.account, hotel.formed_at, False, 1.0, "过期待决", "hotel"))
    edge = frozenset({hotel.account})
    moments = [PastMoment(hotel.formed_at - timedelta(days=40 - index), edge, True) for index in range(40)]
    moments.append(PastMoment(hotel.formed_at, edge, False))
    moments += [PastMoment(hotel.formed_at + timedelta(days=1 + index), edge, False) for index in range(10)]
    labelled = tuple(Moment(item, (f"m{index}",)) for index, item in enumerate(moments))
    settlement.history.write(UnitHistory("session", now.date(), {HOTEL: labelled}))  # type: ignore[union-attr]
    (miss,) = settlement.diagnose(now=now).misses
    assert miss[0].within_budget
    # 没有单元历史时边没有读数：不算预算内，别的关照常判
    _site2, _ledger2, bare = _resettle_stage(tmp_path / "b", (DAY,), now=now)
    assert bare.history is not None and bare.history.latest("session") is None


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
    ledger.append_fact(Fact(hotel.account, hotel.formed_at, False, 1.0, "过期待决", "hotel"))
    ledger.append_fact(
        Fact(hotel.account, hotel.formed_at, False, 0.0, "后补", "hotel", written_at=at(DAY + timedelta(days=1), 4, 0))
    )
    (miss,) = settlement.diagnose(now=now).misses
    assert miss[1] != UNSEEN


def test_an_antecedent_whose_start_was_corrected_still_holds(tmp_path) -> None:
    """事件触发记的是前因没封口时的开始时刻（14:00）；封口时更正成 13:58 开始、到 14:08。还是那一行、还是那一类：不作废。"""

    from habitus.foresight.opportunities import OpportunityTable, RelationOpportunities
    from habitus.runtime.foresight_opportunities import OpportunityStore

    site = Site(tmp_path, now=at(DAY, 23, 0))
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai"))
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
    store.write(OpportunityTable("session", DAY, DAY, 15, 3, (relation,), ()))
    settlement = SettlementStage(
        ledger,
        read_side,
        closed_days=lambda: (DAY,),
        current=lambda token: (token,),
        migrating=lambda: False,
        evidence=Evidence(alpha=0.05, half_life=8.0),
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
    """同一类的日子账开了口、他随后做了（树上盖了 reminded、发起者 system）；这条事件触发的预期自己没说过，窗里也盖着这次：
    它不是这本账的自然命中，记一笔强度 0 的「提醒后发生」，不进校准。"""

    site, ledger, settlement = _resettle_stage(tmp_path, (DAY,), now=at(DAY + timedelta(days=1), 4, 0))
    hotel = expectation("hotel", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3)
    ledger.record_expectation(hotel)
    publish(site.behavior_tree, DAY, "订酒店", 14, 28, kind="订酒店", reminded=True, prompted_by="system")
    settlement.run_once()
    (fact,) = ledger.facts_of(hotel.account)
    assert (fact.kind, fact.good, fact.strength) == ("提醒后发生", True, 0.0)
    assert settlement.diagnose(now=at(DAY + timedelta(days=1), 4, 0)).calibration == ()


def test_a_spoken_miss_lands_one_unit_weight_bad_on_the_units_feedback(tmp_path) -> None:
    """裁定 58 第 6 条、61 第 4 条：说了没做 → 单元上一笔单位权重的坏；账上只有那一笔"做"的坏。回了中又没做的不写、回了负的
    记坏并带着方向；单元替它一并决定的那条按依据的说出时刻结算（做了记「提醒后发生」），反馈只由依据写一笔。"""

    from habitus.foresight import current_facts as live
    from habitus.foresight.ledger import Response

    site, ledger, settlement = _resettle_stage(tmp_path, (DAY,), now=at(DAY + timedelta(days=1), 4, 0))
    spoken = expectation("spoken", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3)
    ledger.record_expectation(spoken)
    ledger.record_delivery(DAY, Delivery("spoken", at(DAY, 14, 15), 1, True, "说了", 1.0, "要订酒店吗"))
    settlement.run_once()
    (fact,) = live(ledger.facts_of(spoken.account))
    assert (fact.good, fact.strength, fact.kind) == (False, 1.0, "过期待决")
    (feedback,) = ledger.feedback_of(HOTEL)
    assert (feedback.good, feedback.expectation_id, feedback.polarity) == (False, "spoken", None)
    # 回了中、没做：不写；回了负：坏、带方向
    for identity, polarity, expected in (("later", "中", None), ("no", "负", False)):
        root = tmp_path / identity
        _site, ledger2, settlement2 = _resettle_stage(root, (DAY,), now=at(DAY + timedelta(days=1), 4, 0))
        item = replace(spoken, expectation_id=identity, trigger_instance=f"behavior://{identity}.md")
        ledger2.record_expectation(item)
        ledger2.record_delivery(DAY, Delivery(identity, at(DAY, 14, 15), 1, True, "说了", 1.0, "要订酒店吗"))
        ledger2.record_response(DAY, Response(identity, at(DAY, 14, 40), polarity, "conv#1-3"))
        settlement2.run_once()
        found = ledger2.feedback_of(HOTEL)
        assert ([] if expected is None else [(expected, polarity)]) == [(f.good, f.polarity) for f in found]
    # 单元替它一并决定的那条：按依据的说出时刻结算（说完才做的是「提醒后发生」，强度 0），反馈只有依据那一笔
    root = tmp_path / "member"
    site3, ledger3, settlement3 = _resettle_stage(root, (DAY,), now=at(DAY + timedelta(days=1), 4, 0))
    basis = expectation("basis", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3)
    member = replace(basis, expectation_id="member", source="日子", trigger="interval", trigger_instance="x@2026-12-10")
    ledger3.record_expectation(basis)
    ledger3.record_expectation(member)
    ledger3.record_delivery(
        DAY, Delivery("member", at(DAY, 14, 10), 1, False, "单元已选", 0.0, evidence=("单元:basis",))
    )
    ledger3.record_delivery(DAY, Delivery("basis", at(DAY, 14, 15), 1, True, "说了", 1.0, "要订酒店吗"))
    publish(site3.behavior_tree, DAY, "订酒店", 14, 30, kind="订酒店")
    settlement3.run_once()
    (member_fact,) = ledger3.facts_of(member.account)
    assert (member_fact.kind, member_fact.strength) == ("提醒后发生", 0.0)
    assert [f.expectation_id for f in ledger3.feedback_of(HOTEL)] == ["basis"]


def test_a_unit_moment_is_one_miss_not_one_per_input(tmp_path) -> None:
    """同一刻两个输入都落空：依据那条在诊断里记一次落空；「单元已选」引它的那条照常结算（它那本账的权重照调），但诊断不再
    把它算成第二次落空。"""

    from habitus.foresight.ledger import AccountFact as Fact
    from habitus.foresight.ledger import Settlement

    monday = date(2026, 6, 1)
    now = at(monday + timedelta(days=1), 4, 0)
    _site, ledger, settlement = _resettle_stage(tmp_path, (monday,), now=now)
    basis = expectation("hotel", HOTEL, at(monday, 10, 0), window_minutes=(4, 51), deadline_hours=3)
    member = replace(
        basis, expectation_id="member", source="日子", trigger="weekday=0", trigger_instance=monday.isoformat()
    )
    for item in (basis, member):
        ledger.record_expectation(item)
        ledger.record_settlement(
            Settlement(item.expectation_id, item.kind_token, monday, at(monday, 23, 0), "过期待决")
        )
        ledger.append_fact(Fact(item.account, item.formed_at, False, 1.0, "过期待决", item.expectation_id))
    ledger.record_delivery(monday, Delivery("hotel", at(monday, 10, 15), 0, False, "影子", 0.0))
    ledger.record_delivery(
        monday, Delivery("member", at(monday, 10, 15), 0, False, "单元已选", 0.0, evidence=("单元:hotel",))
    )
    misses = [miss.expectation_id for miss, _cause in settlement.diagnose(now=now).misses]
    assert misses == ["hotel"]


def test_a_crash_after_the_fact_and_before_the_feedback_fills_in_only_the_feedback(tmp_path, monkeypatch) -> None:
    """说了没做：账上一笔、单元上一笔。账上那笔写了、单元那笔还没写就断——下一晚只补单元那笔，不重算已写的那一笔。"""

    site, ledger, settlement = stage(tmp_path, closed=(DAY,))
    hotel = expectation("hotel", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3)
    ledger.record_expectation(hotel)
    ledger.record_delivery(DAY, Delivery("hotel", at(DAY, 14, 15), 1, True, "说了", 1.0, "要订酒店吗"))

    def crash(*_args: object) -> bool:
        raise OSError("disk went away")

    monkeypatch.setattr(ledger, "record_feedback", crash)
    with pytest.raises(OSError):
        settlement.run_once()
    monkeypatch.undo()
    assert [fact.kind for fact in ledger.facts_of(hotel.account)] == ["过期待决"] and ledger.settlements_on(DAY) == ()
    settlement.run_once()
    assert [fact.kind for fact in ledger.facts_of(hotel.account)] == ["过期待决"]
    assert [item.good for item in ledger.feedback_of(HOTEL)] == [False]
    assert [item.outcome for item in ledger.settlements_on(DAY)] == ["过期待决"]


@pytest.mark.parametrize(("formed", "gap"), [((14, 7), ((14, 30), (15, 30))), ((9, 0), ((8, 50), (12, 0)))])
def test_a_spoken_miss_under_an_observation_gap_writes_no_feedback(tmp_path, formed, gap) -> None:  # type: ignore[no-untyped-def]
    """说了、窗里有观测空白：他可能做了、我们没看到（裁定 54）——账上中性，单元的提醒反馈也不记坏（不知道他做没做就不是误提醒）。"""

    site, ledger, settlement = stage(tmp_path, closed=(DAY,))
    hotel = expectation("hotel", HOTEL, at(DAY, *formed), window_minutes=(4, 51), deadline_hours=3)
    ledger.record_expectation(hotel)
    ledger.record_delivery(
        DAY, Delivery("hotel", at(DAY, formed[0], formed[1] + 8), 1, True, "说了", 1.0, "要订酒店吗")
    )
    publish_gap(site.behavior_tree, DAY, *gap)
    settlement.run_once()
    (fact,) = ledger.facts_of(hotel.account)
    assert (fact.kind, fact.strength) == ("过期待决", 0.0)
    assert ledger.feedback_of(HOTEL) == ()


def test_a_spoken_expectation_whose_chance_is_void_writes_no_feedback(tmp_path) -> None:
    """说了、封口后前因成了别的类：这次机会不存在（作废），单元的提醒反馈不记坏。"""

    from habitus.foresight.opportunities import OpportunityTable, RelationOpportunities
    from habitus.runtime.foresight_opportunities import OpportunityStore

    site = Site(tmp_path, now=at(DAY, 23, 0))
    read_side = ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai"))
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
    store.write(OpportunityTable("session", DAY, DAY, 15, 3, (relation,), ()))
    settlement = SettlementStage(
        ledger,
        read_side,
        closed_days=lambda: (DAY, DAY + timedelta(days=1)),
        current=lambda token: (token,),
        migrating=lambda: False,
        evidence=Evidence(alpha=0.05, half_life=8.0),
        opportunities=store,
        clock=lambda: at(DAY + timedelta(days=2), 4, 0).astimezone(UTC),
    )
    publish(site.behavior_tree, DAY, "问机票价格", 14, 0, kind="还信用卡")
    hotel = replace(
        expectation("hotel", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3),
        trigger_instance=at(DAY, 14, 0).astimezone(UTC).isoformat(),
    )
    ledger.record_expectation(hotel)
    ledger.record_delivery(DAY, Delivery("hotel", at(DAY, 14, 15), 1, True, "说了", 1.0, "要订酒店吗"))
    settlement.run_once()
    (fact,) = ledger.facts_of(hotel.account)
    assert (fact.kind, fact.strength) == ("作废", 0.0)
    assert ledger.feedback_of(HOTEL) == ()


@pytest.mark.parametrize(("done_at", "expected"), [((10, 40), "提醒后发生"), ((13, 30), "提醒后发生")])
def test_a_unit_member_on_the_next_day_settles_by_the_basis_spoken_time_from_the_basis_day(
    tmp_path, done_at, expected
) -> None:  # type: ignore[no-untyped-def]
    """单元跨天：依据是次日段的事件预期（挂在 D 下），成员是 D+1 的日子预期（挂在 D+1 下）、「单元已选」引依据。
    依据的「说了」在 D 的递送里——成员按同一个说出时刻结算（方案 30 第二节），不能因为不在自己这一天下就当没说。"""

    later = DAY + timedelta(days=1)
    site, ledger, settlement = _resettle_stage(tmp_path, (DAY, later), now=at(later + timedelta(days=1), 4, 0))
    basis = expectation("basis", HOTEL, at(DAY, 20, 0), window_minutes=(14 * 60, 15 * 60), deadline_hours=27)
    member = replace(
        expectation("member", HOTEL, at(later, 9, 0), window_minutes=(60, 120), deadline_hours=14),
        source="日子",
        trigger="interval",
        trigger_instance=f"x@{later.isoformat()}",
    )
    ledger.record_expectation(basis)
    ledger.record_expectation(member)
    spoken_at = at(later, 10, 30)
    ledger.record_delivery(basis.day, Delivery("basis", spoken_at, 1, True, "说了", 1.0, "要订酒店吗"))
    ledger.record_delivery(
        member.day, Delivery("member", spoken_at, 1, False, "单元已选", 0.0, evidence=("单元:basis",))
    )
    publish(site.behavior_tree, later, "订酒店", *done_at, kind="订酒店")
    settlement.run_once()
    (basis_fact,) = ledger.facts_of(basis.account)
    (member_fact,) = ledger.facts_of(member.account)
    assert (basis_fact.kind, basis_fact.strength) == (expected, 0.0)
    assert (member_fact.kind, member_fact.strength) == (expected, 0.0)
    assert [item.expectation_id for item in ledger.feedback_of(HOTEL)] == ["basis"]


@pytest.mark.parametrize(
    ("done", "late", "outcome"),
    [
        (False, Response("hotel", at(DAY, 14, 40), "负", "conv#9-9"), "过期待决"),
        (True, Response("hotel", at(DAY, 14, 20), "中", "conv#9-9"), "提醒后发生"),
        (False, None, "过期待决"),
    ],
)
def test_a_feedback_already_written_is_not_rewritten_when_a_late_response_would_change_it(
    tmp_path, monkeypatch, done, late, outcome
) -> None:  # type: ignore[no-untyped-def]
    """写事实 → 写反馈 → 写结算；反馈写了、结算没写就断，两晚之间来了一条回应：第二晚算出的反馈内容不同（方向变了），
    账本只增不改——已经写过的那笔不再写，整晚照常结完，不卡死。"""

    site, ledger, settlement = _resettle_stage(tmp_path, (DAY,), now=at(DAY + timedelta(days=1), 4, 0))
    hotel = expectation("hotel", HOTEL, at(DAY, 14, 7), window_minutes=(4, 51), deadline_hours=3)
    ledger.record_expectation(hotel)
    ledger.record_delivery(DAY, Delivery("hotel", at(DAY, 14, 15), 1, True, "说了", 1.0, "要订酒店吗"))
    if done:
        publish(site.behavior_tree, DAY, "订酒店", 14, 30, kind="订酒店")

    def crash(*_args: object) -> bool:
        raise OSError("disk went away")

    monkeypatch.setattr(ledger, "record_settlement", crash)
    with pytest.raises(OSError):
        settlement.run_once()
    monkeypatch.undo()
    (first,) = ledger.feedback_of(HOTEL)
    assert (first.good, first.polarity) == (done, None)
    if late is not None:
        ledger.record_response(DAY, late)
    settlement.run_once()
    assert [item.outcome for item in ledger.settlements_on(DAY)] == [outcome]
    (again,) = ledger.feedback_of(HOTEL)
    assert again == first


def test_the_diagnosis_reads_the_unit_history_snapshot_once_not_once_per_miss(tmp_path, monkeypatch) -> None:
    from habitus.foresight.history import Moment, UnitHistory
    from habitus.foresight.ledger import AccountFact as Fact
    from habitus.foresight.ledger import Settlement

    days, per_day = 4, 3
    start = DAY - timedelta(days=days)
    closed = tuple(start + timedelta(days=offset) for offset in range(days))
    now = at(DAY, 4, 0)
    _site, ledger, settlement = _resettle_stage(tmp_path, closed, now=now)
    for offset in range(days):
        day = start + timedelta(days=offset)
        for number in range(per_day):
            item = replace(
                expectation(
                    f"h{offset}-{number}", HOTEL, at(day, 9 + number, 0), window_minutes=(4, 51), deadline_hours=3
                ),
                trigger_instance=f"behavior://flight-{offset}-{number}.md",
            )
            ledger.record_expectation(item)
            ledger.record_settlement(Settlement(item.expectation_id, HOTEL, day, at(day, 23, 0), "过期待决"))
            ledger.append_fact(Fact(item.account, item.formed_at, False, 1.0, "过期待决", item.expectation_id))
    edge = frozenset({expectation("x", HOTEL, at(DAY, 9, 0), window_minutes=(4, 51), deadline_hours=3).account})
    moments = tuple(
        Moment(PastMoment(at(start - timedelta(days=40 - index), 9, 0), edge, True), (f"m{index}",))
        for index in range(40)
    )
    assert settlement.history is not None
    settlement.history.write(UnitHistory("session", now.date(), {HOTEL: moments}))
    calls = {"n": 0}
    original = settlement.history.latest

    def counted(lane: str):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return original(lane)

    monkeypatch.setattr(settlement.history, "latest", counted)
    assert len(settlement.diagnose(now=now).misses) == days * per_day
    assert calls["n"] == 1
