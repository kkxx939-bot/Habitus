"""行为单元（裁定 56 第 12 条）：一个行为看这一刻在场的全部输入，从历史里数下一步来了多少；分层退、依据、档、自检。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from habitus.foresight import Evidence, Tolerance, climb
from habitus.foresight.edges import edge_reading
from habitus.foresight.errors import ForesightError
from habitus.foresight.feedback import feedback_reading
from habitus.foresight.history import labelled_moments, record_of
from habitus.foresight.ledger import AccountFact, Antecedent, Expectation, UnitFeedback, account_key
from habitus.foresight.opportunities import TreeReading
from habitus.foresight.units import (
    Check,
    Input,
    Level,
    PastMoment,
    basis,
    connected,
    joinable,
    meets,
    self_check,
    signature,
    unit_level,
    unit_reading,
)

CST = timezone(timedelta(hours=8))
NOW = datetime(2026, 9, 1, 21, 0, tzinfo=CST)
EVIDENCE = Evidence(alpha=0.05, half_life=8.0)
LOSS = Tolerance(line=0.67)
LUGGAGE = "s-k0010"


def expect(
    identity: str,
    *,
    source: str = "事件",
    trigger: str = "session|flight|luggage|days_2_7",
    formed: datetime = NOW - timedelta(hours=1),
    hours: int = 6,
    antecedents: tuple[Antecedent, ...] = (),
) -> Expectation:
    return Expectation(
        expectation_id=identity,
        kind_token=LUGGAGE,
        day=formed.date(),
        formed_at=formed,
        source=source,
        trigger=trigger,
        trigger_instance=formed.isoformat(),
        window=(formed, formed + timedelta(hours=1)),
        deadline=formed + timedelta(hours=hours),
        basis=(),
        opportunities=10,
        hits=9,
        observed_from=formed,
        antecedents=antecedents,
    )


def account_standing(hits: int, misses: int, *, account: str = "a") -> object:
    """一条边的位置：单元历史里签名含它的那几刻（先中后漏）。"""

    key = account_key(LUGGAGE, "事件", account)
    moments = [PastMoment(NOW - timedelta(days=30 - i), frozenset({key}), i < hits) for i in range(hits + misses)]
    return climb(moments, key, now=NOW, tolerance=LOSS, evidence=EVIDENCE)


def entry(identity: str, hits: int, misses: int, **kwargs: object) -> Input:
    expectation = expect(identity, trigger=f"session|{identity}|luggage|days_2_7", **kwargs)  # type: ignore[arg-type]
    standing = account_standing(hits, misses, account=f"session|{identity}|luggage|days_2_7")
    return Input(expectation, standing)  # type: ignore[arg-type]


def test_the_signature_is_the_present_inputs_and_the_present_suppressors() -> None:
    cancel = Antecedent("session|cancel|luggage|next_day", True, False, "behavior://cancel.md")
    absent = Antecedent("session|other|luggage|next_day", False, False)
    raising = Antecedent("session|hotel|luggage|days_2_7", True, True, "behavior://hotel.md")
    first = expect("x", trigger="session|flight|luggage|days_2_7", antecedents=(cancel, absent, raising))
    second = expect("y", source="日子", trigger="weekday=3")
    assert signature((first, second)) == frozenset({first.account, second.account, "↓session|cancel|luggage|next_day"})


def test_history_moments_are_the_overlapping_expectations_of_one_behavior_and_unknowns_are_dropped() -> None:
    """说中的范围连成一片的几条预期是同一刻；有一条自然命中就是来了，只有看得见的落空才是没来，其余（被提醒才做、观测空白）不算。"""

    day = NOW - timedelta(days=10)
    a = expect("a", formed=day, trigger="session|flight|luggage|days_2_7")
    b = expect("b", formed=day + timedelta(hours=2), source="日子", trigger="weekday=3")
    c = expect("c", formed=day + timedelta(days=3))
    d = expect("d", formed=day + timedelta(days=5))
    facts = {
        "a": AccountFact(a.account, a.formed_at, False, 1.0, "过期待决", "a"),
        "b": AccountFact(b.account, b.formed_at, True, 0.4, "晚了但做了", "b"),
        "c": AccountFact(c.account, c.formed_at, False, 1.0, "过期待决", "c"),
        "d": AccountFact(d.account, d.formed_at, True, 0.0, "提醒后发生", "d"),
    }

    def moments(items, fact_of):  # type: ignore[no-untyped-def]
        records = [record_of(item, fact_of(item)) for item in items]
        found = labelled_moments(records, present=lambda group: frozenset(r.account for r in group))
        return [moment.past for moment in found]

    found = moments((d, c, b, a), lambda item: facts.get(item.expectation_id))
    assert [(moment.signature, moment.came) for moment in found] == [
        (frozenset({a.account, b.account}), True),  # 同一刻：日子那条做了（强度低也是做了）
        (frozenset({c.account}), False),
    ]
    neutral = AccountFact(c.account, c.formed_at, False, 0.0, "过期待决", "c")  # 观测空白里的中性
    assert [m.came for m in moments((c,), lambda _item: neutral)] == []


def test_more_causes_present_together_read_higher_than_one_alone() -> None:
    """用户 10-17：出差、订机票、订酒店都命中时，收拾行李该比只有出差时高得多——从历史数：单个在场 6/10，两个一起在场 9/10。"""

    flight, hotel = entry("flight", 6, 4), entry("hotel", 6, 4)
    alone = frozenset({flight.account})
    together = frozenset({flight.account, hotel.account})
    history = [PastMoment(NOW - timedelta(days=40 - i), alone, i < 6) for i in range(10)]
    history += [PastMoment(NOW - timedelta(days=25 - i), together, i < 9) for i in range(10)]
    one = unit_reading((flight,), history, now=NOW, evidence=EVIDENCE, prior_share=1 / 3)
    both = unit_reading((flight, hotel), history, now=NOW, evidence=EVIDENCE, prior_share=1 / 3)
    assert one.level is Level.SAME_SET and both.level is Level.SAME_SET
    assert both.reading.mean > one.reading.mean + 0.15 and both.reading.lower > one.reading.lower


def test_a_suppressor_present_reads_lower_and_an_unseen_combination_leans_on_the_coarser_layer() -> None:
    """取消出差在场（压低）的那几刻下一步多半没来：同一组读数低；没见过的组合退到"同样多"、再退到最强那本账。"""

    flight = entry("flight", 9, 1)
    cancel = Antecedent("session|cancel|luggage|next_day", True, False, "behavior://cancel.md")
    cancelled = replace(flight, expectation=replace(flight.expectation, antecedents=(cancel,)))
    plain = frozenset({flight.account})
    suppressed = frozenset({flight.account, "↓session|cancel|luggage|next_day"})
    history = [PastMoment(NOW - timedelta(days=30 - i), plain, True) for i in range(8)]
    history += [PastMoment(NOW - timedelta(days=20 - i), suppressed, i < 1) for i in range(6)]
    normal = unit_reading((flight,), history, now=NOW, evidence=EVIDENCE, prior_share=1 / 3)
    low = unit_reading((cancelled,), history, now=NOW, evidence=EVIDENCE, prior_share=1 / 3)
    assert low.reading.mean < normal.reading.mean - 0.3
    # 没见过的组合：退到同样多；一刻都没有：就是最强那本账自己的读数
    hotel = entry("hotel", 5, 5)
    unseen = unit_reading((flight, hotel), history, now=NOW, evidence=EVIDENCE, prior_share=1 / 3)
    assert unseen.level is Level.STRONGEST and unseen.reading == flight.standing.prediction
    assert unseen.seen == (0, 0, round(flight.standing.prediction.n))
    # 先验的分量不超过下一层自己的有效样本：一本只见过一次的账，带不出 8 次那么硬的先验
    thin = entry("thin", 1, 0)
    one_moment = [PastMoment(NOW - timedelta(days=1), frozenset({thin.account}), False)]
    shrunk = unit_reading((thin,), one_moment, now=NOW, evidence=EVIDENCE, prior_share=1 / 3)
    assert shrunk.reading.n < 3.0
    with pytest.raises(ForesightError):
        unit_reading((), history, now=NOW, evidence=EVIDENCE, prior_share=1 / 3)


def test_a_combination_always_followed_clears_the_line_even_when_its_layer_averages_a_coin_flip() -> None:
    """26 号清单 4.4：按机会数遗忘后有效样本上限约 12（H = 8），先验分量写死 8 会占到四成——一个组合 100 次里 100 次都来了，
    同一层别的组合均值 0.5，下界只有 0.628、永远开不了口。先验分量改成有效样本上限的 1/3 之后下界 0.696，过 0.67 的线。
    每一层按自己的刻数遗忘：别的组合再多再密，也冲不掉这个组合自己的那些刻。"""

    flight, hotel = entry("flight", 9, 1), entry("hotel", 9, 1)
    mine = frozenset({flight.account})
    other = frozenset({hotel.account})
    history = [PastMoment(NOW - timedelta(days=300 - i), mine, True) for i in range(100)]
    history += [PastMoment(NOW - timedelta(hours=100 - i), other, i % 2 == 0) for i in range(100)]
    reading = unit_reading((flight,), history, now=NOW, evidence=EVIDENCE, prior_share=1 / 3)
    assert reading.level is Level.SAME_SET
    # 每层只读最近 M = 35 刻（有效样本从 12.05 降到约 11.47），同层均值按遗忘约 0.48（最近那一刻没来）：下界 0.678，仍过线；
    # 不截窗、均值正好一半时是方案里算的 0.696
    assert LOSS.threshold < reading.reading.lower < 0.70


def test_the_basis_is_the_established_cause_with_the_largest_lift_and_never_a_coincidence() -> None:
    """依据：事件输入里成立、效应够大、抬升最高的那条；巧合（效应不够大）不当依据，哪怕它的账更高；都没有就取账下界最高的。"""

    trip = replace(entry("trip", 9, 1), reading=TreeReading(1.0, 0.19, True), status="established")
    flight = replace(entry("flight", 9, 1), reading=TreeReading(0.8, 0.0, True), status="established")
    weekly = replace(entry("weekly", 30, 1), reading=TreeReading(0.92, 0.67, False), status="established")
    clock = Input(expect("clock", source="日子", trigger="weekday=3"), account_standing(40, 2, account="clock"))  # type: ignore[arg-type]
    assert (
        basis((weekly, trip, flight, clock)).account == trip.account
    )  # 出差进日历抬升 0.81 > 订机票 0.80 > 巧合（不算）
    assert basis((weekly, flight, clock)).account == flight.account
    assert basis((weekly, clock)).account == clock.account  # 只有巧合与日子：取日子，巧合不当依据
    candidate = replace(flight, status="candidate")
    assert basis((candidate, clock)).account == clock.account  # 候选关系不当依据
    with pytest.raises(ForesightError):
        basis(())


def test_the_unit_level_follows_settings_then_the_feedback_then_the_reading_with_hysteresis() -> None:
    """开口只看单元（裁定 61）：这个行为的设置、单元的提醒反馈、单元读数过依据那本账的线，滞回读单元上一次的档。"""

    flight = replace(entry("flight", 12, 0), status="established")
    history = [PastMoment(NOW - timedelta(days=20 - i), frozenset({flight.account}), True) for i in range(10)]
    strong = unit_reading((flight,), history, now=NOW, evidence=EVIDENCE, prior_share=1 / 3)
    assert unit_level(flight, strong, previous=None) == 1
    assert unit_level(flight, strong, previous=1, setting="以后别提了") == 0
    ignored = feedback_reading(
        [UnitFeedback(LUGGAGE, NOW - timedelta(days=5 - i), False, f"r{i}") for i in range(5)],
        alpha=EVIDENCE.alpha,
        half_life=EVIDENCE.half_life,
        window=EVIDENCE.window,
    )
    assert unit_level(flight, strong, previous=1, feedback=ignored) == 0  # 说了他老不做：照降
    middling = replace(strong, reading=replace(strong.reading, lower=0.5, upper=0.8))
    assert unit_level(flight, middling, previous=1) == 1 and unit_level(flight, middling, previous=0) == 0
    weak = replace(strong, reading=replace(strong.reading, lower=0.1, upper=0.3))
    assert unit_level(flight, weak, previous=None, setting="以后都提醒我") == 1
    # 开口只用成立的（方案第六节）：候选关系、巧合当依据时记影子，读数再高也一样
    assert unit_level(replace(flight, status="candidate"), strong, previous=1) == 0
    from habitus.foresight.opportunities import TreeReading

    coincidence = replace(flight, reading=TreeReading(0.92, 0.67, False))
    assert unit_level(coincidence, strong, previous=1) == 0
    clock = Input(expect("clock", source="日子", trigger="weekday=3"), flight.standing)
    assert unit_level(clock, strong, previous=None) == 1  # 日子输入能当依据
    # 情境输入和事件一样要关系成立才能当依据（裁定 59）：候选的日历关系只进读数
    situation = replace(
        flight, expectation=replace(flight.expectation, source="情境", trigger="session|周三|s-k0010|situation")
    )
    assert unit_level(situation, strong, previous=None) == 1
    assert unit_level(replace(situation, status="candidate"), strong, previous=1) == 0


def test_the_self_check_names_what_the_two_readings_say_together() -> None:
    """裁定 56 第 10 条：账高抬升大 = 依据；账高抬升小 = 凑巧同现；账低抬升大 = 预测层算得不对；账低抬升小 = 不是原因。"""

    high, low = account_standing(30, 1), account_standing(1, 30)
    middle = account_standing(3, 1)
    cause, coincidence = TreeReading(1.0, 0.1, True), TreeReading(0.92, 0.67, False)
    assert self_check(high, cause) is Check.BASIS  # type: ignore[arg-type]
    assert self_check(high, coincidence) is Check.COINCIDENCE  # type: ignore[arg-type]
    assert self_check(low, cause) is Check.MISCOMPUTED  # type: ignore[arg-type]
    assert self_check(low, coincidence) is Check.NOT_A_CAUSE  # type: ignore[arg-type]
    assert self_check(middle, cause) is Check.CANNOT_RISE  # type: ignore[arg-type]
    assert self_check(high, None) is Check.WATCHING  # type: ignore[arg-type]


def test_one_input_with_its_own_moments_reads_no_higher_than_its_account_no_double_counting() -> None:
    """2026-10-17 四方评审：单元原来以最强那本账为先验，那本账的事实与"同一组"的刻是同一批，数了两三遍——每周一次、4/4 全中的账
    自己下界 0.60 过不了 0.67，进单元后 0.71 就开了口。现在先验只用这一层以外、输入同样多的那几刻：没有就不加先验。"""

    account = account_key(LUGGAGE, "事件", "session|weekly|luggage|days_2_7")
    days = [NOW - timedelta(days=7 * week + 1) for week in range(4)]
    history = [PastMoment(day, frozenset({account}), True) for day in days]
    standing = climb(history, account, now=NOW, tolerance=LOSS, evidence=EVIDENCE)
    item = Input(expect("weekly", trigger="session|weekly|luggage|days_2_7"), standing)  # type: ignore[arg-type]
    reading = unit_reading((item,), history, now=NOW, evidence=EVIDENCE, prior_share=1 / 3)
    assert standing.prediction.lower < LOSS.threshold
    assert reading.level is Level.SAME_SET and reading.reading.lower == pytest.approx(
        standing.prediction.lower, abs=1e-6
    )
    assert unit_level(replace(item, status="established"), reading, previous=None) == 0


def test_the_basis_is_chosen_only_among_inputs_already_due() -> None:
    """同一刻还没到时窗的那条只当输入、不当依据（2026-10-17 四方评审：次日那条当依据，重的会提前一天说、轻的会推到第二天）。"""

    today = replace(entry("today", 6, 4), reading=TreeReading(0.8, 0.2, True), status="established")
    tomorrow = replace(entry("tomorrow", 9, 1), reading=TreeReading(1.0, 0.1, True), status="established")
    assert basis((today, tomorrow)).expectation.expectation_id == "tomorrow"  # 不限：抬升最高的
    assert basis((today, tomorrow), due=("today",)).expectation.expectation_id == "today"


def test_coincidental_inputs_stay_out_of_the_signature_and_the_counts() -> None:
    """语义树判成巧合（效应不够大）的输入不进"同一组"的签名、不计入"同样多"（方案第六节第 1 条）；权重照数，不在这里。"""

    cancel = Antecedent("session|cancel|luggage|next_day", True, False, "behavior://cancel.md")
    first = expect("x", trigger="session|flight|luggage|days_2_7", antecedents=(cancel,))
    second = expect("y", trigger="session|photo|luggage|days_2_7")
    full = signature((first, second))
    assert full == frozenset({first.account, second.account, "↓session|cancel|luggage|next_day"})
    assert signature((first, second), coincidental={"session|photo|luggage|days_2_7"}) == frozenset(
        {first.account, "↓session|cancel|luggage|next_day"}
    )
    assert signature((first, second), coincidental={"session|cancel|luggage|next_day"}) == frozenset(
        {first.account, second.account}
    )
    inputs = (entry("flight", 9, 1), entry("photo", 9, 1))
    history = (PastMoment(NOW - timedelta(days=7), frozenset({inputs[0].account}), True, 1.0),)
    with_both = unit_reading(inputs, history, now=NOW, evidence=EVIDENCE, prior_share=1 / 3)
    without = unit_reading(
        inputs, history, now=NOW, evidence=EVIDENCE, prior_share=1 / 3, coincidental={"session|photo|luggage|days_2_7"}
    )
    assert with_both.level is Level.STRONGEST and without.level is Level.SAME_SET


def test_meets_needs_a_real_overlap_while_connected_also_joins_ranges_that_only_touch() -> None:
    first = expect("a", formed=NOW - timedelta(hours=6), hours=3)  # 15:00 → 18:00
    touching = expect("b", formed=NOW - timedelta(hours=3), hours=3)  # 18:00 → 21:00
    assert meets(first, NOW - timedelta(hours=4), NOW)
    assert not meets(first, NOW - timedelta(hours=3), NOW)  # 只碰到端点
    assert not meets(touching, NOW - timedelta(hours=6), NOW - timedelta(hours=3))
    assert [[item.expectation_id for item in group] for group in connected([first, touching])] == [["a", "b"]]


def day_expect(identity: str, day: int, *, source: str = "日子", trigger: str = "interval") -> Expectation:
    """以当天为单位的预期：起算当天零点、最晚时限次日零点（``triggers.days`` / ``triggers.situations``）。"""

    midnight = datetime(2026, 1, day, tzinfo=CST)
    return replace(
        expect(identity, source=source, trigger=trigger, formed=midnight + timedelta(hours=4, minutes=15)),
        window=(midnight + timedelta(hours=8), midnight + timedelta(hours=9)),
        observed_from=midnight,
        deadline=midnight + timedelta(days=1),
    )


def test_day_unit_expectations_on_adjacent_days_are_separate_moments() -> None:
    """每天都做的事：今天那条的最晚时限正是明天那条的起算，只碰到端点不连成一刻（日子、情境都是）；同一天的照连。"""

    days = [day_expect(f"d{day}", day) for day in (4, 5, 6)]
    assert [[item.expectation_id for item in group] for group in connected(days)] == [["d4"], ["d5"], ["d6"]]
    workdays = [
        day_expect(f"w{day}", day, source="情境", trigger="session|weekday|luggage|situation") for day in (5, 6)
    ]
    assert [[item.expectation_id for item in group] for group in connected(workdays)] == [["w5"], ["w6"]]
    same_day = [day_expect("d5", 5), workdays[0]]
    assert [[item.expectation_id for item in group] for group in connected(same_day)] == [["d5", "w5"]]


def test_an_event_range_that_only_touches_a_day_range_still_joins_it() -> None:
    """只有两头都是以当天为单位的预期才不连：事件那条（前因做完起算的当天剩下的）碰到明天零点，与明天那条照连。"""

    today = datetime(2026, 1, 5, tzinfo=CST)
    rest_of_day = replace(
        expect("evening", formed=today + timedelta(hours=20), hours=4), observed_from=today + timedelta(hours=20)
    )
    tomorrow = day_expect("d6", 6)
    assert [[item.expectation_id for item in group] for group in connected([rest_of_day, tomorrow])] == [
        ["evening", "d6"]
    ]


def test_twenty_daily_hits_are_twenty_moments_and_reach_the_line() -> None:
    """单元历史：20 天每天准时命中是 20 刻，读数下界过 0.67 线（并成一刻时有效样本约 1，过不了线）。"""

    items = [day_expect(f"d{day:02d}", day) for day in range(4, 24)]
    records = [
        record_of(item, AccountFact(item.account, item.formed_at, True, 1.0, "准时", item.expectation_id))
        for item in items
    ]
    moments = labelled_moments(records, present=lambda group: frozenset({group[0].account}))
    assert len(moments) == 20
    reading = edge_reading(
        [moment.past for moment in moments], items[0].account, alpha=0.05, half_life=8.0, window=40
    ).reading
    assert reading.lower > 0.67


def two_day_expect(identity: str, day: int) -> Expectation:
    """他偶尔隔一天做时的「离上次够久」：起算当天零点、最晚时限后天零点（覆盖今天和明天，``triggers.days``）。"""

    item = day_expect(identity, day)
    return replace(item, deadline=item.observed_from + timedelta(days=2))


def hit(item: Expectation, occurrence: str):  # type: ignore[no-untyped-def]
    fact = AccountFact(item.account, item.formed_at, True, 1.0, "准时", item.expectation_id)
    return record_of(item, fact, occurrence_uri=occurrence)


def test_overlapping_day_unit_expectations_from_different_days_are_separate_moments() -> None:
    """裁定 81 第 1 条：起算不在同一天的两条以当天为单位的预期，重叠也不连；同一天的照连。"""

    days = [two_day_expect(f"d{day}", day) for day in (6, 7, 8)]
    assert [[item.expectation_id for item in group] for group in connected(days)] == [["d6"], ["d7"], ["d8"]]
    same_day = [
        two_day_expect("d7", 7),
        day_expect("w7", 7, source="情境", trigger="session|weekday|luggage|situation"),
    ]
    assert [[item.expectation_id for item in group] for group in connected(same_day)] == [["d7", "w7"]]


def test_an_occurrence_two_day_expectations_both_saw_counts_once_for_the_earlier_one() -> None:
    """裁定 81 第 2 条：01-06 没做、01-07 做了——01-06 那条（到 01-08）与 01-07 那条都认 01-07 那一次，只算给起算最早的
    01-06 那一刻；01-07 那条自己的发生账上看不到，不算。01-09 做了的那一次同样只算一次。"""

    d6, d7, d8, d9 = (two_day_expect(f"d{day}", day) for day in (6, 7, 8, 9))
    records = [hit(d6, "o7"), hit(d7, "o7"), hit(d8, "o9"), hit(d9, "o9")]
    moments = labelled_moments(records, present=lambda group: frozenset({group[0].account}))
    assert [(moment.expectation_ids, moment.past.came) for moment in moments] == [(("d6",), True), (("d8",), True)]


def test_twenty_days_each_done_under_two_day_ranges_are_twenty_moments_and_reach_the_line() -> None:
    """每天都做、每条预期覆盖今天和明天：每条认的是自己那天的发生，20 刻、下界过 0.67 线（重叠即同一刻时并成 1 刻）。"""

    items = [two_day_expect(f"d{day:02d}", day) for day in range(4, 24)]
    records = [hit(item, f"o{item.day.isoformat()}") for item in items]
    moments = labelled_moments(records, present=lambda group: frozenset({group[0].account}))
    assert len(moments) == 20 and all(moment.past.came for moment in moments)
    reading = edge_reading(
        [moment.past for moment in moments], items[0].account, alpha=0.05, half_life=8.0, window=40
    ).reading
    assert reading.lower > 0.67


def test_expectations_joined_into_one_moment_still_share_their_occurrence() -> None:
    """裁定 81 第 3 条：同一天的日子与情境、与事件那条连成的一刻照旧是一刻、一次命中；去重只在不相连的刻之间。"""

    d7 = two_day_expect("d7", 7)
    w7 = day_expect("w7", 7, source="情境", trigger="session|weekday|luggage|situation")
    evening = replace(
        expect("evening", formed=datetime(2026, 1, 7, 20, tzinfo=CST), hours=4),
        observed_from=datetime(2026, 1, 7, 20, tzinfo=CST),
    )
    moments = labelled_moments(
        [hit(d7, "o7"), hit(w7, "o7"), hit(evening, "o7")], present=lambda group: frozenset({group[0].account})
    )
    assert [(moment.expectation_ids, moment.past.came) for moment in moments] == [(("d7", "evening", "w7"), True)]


# ── 裁定 90：并单元也照"以天为单位、起算不在同一天的不是同一刻" ─────────────────


def _two_days(identity: str, day: int) -> Expectation:
    """1、2 天混着的「离上次够久」：时窗从当天 08:00 到次日，最晚时限次日过完。"""

    item = day_expect(identity, day)
    return replace(
        item, window=(item.window[0], item.window[1] + timedelta(days=1)), deadline=item.deadline + timedelta(days=1)
    )


def test_day_unit_expectations_from_different_days_are_never_one_moment() -> None:
    """``joinable``：两条都以当天为单位、起算不在同一天的不是同一刻（重叠也不是）；同一天的、有一条是事件的照算同一刻。"""

    yesterday, today = _two_days("y", 5), _two_days("t", 6)
    assert meets(today, yesterday.observed_from, yesterday.deadline)  # 范围真的相交
    assert not joinable(yesterday, today) and not joinable(today, yesterday)
    assert joinable(
        day_expect("d6", 6), day_expect("w6", 6, source="情境", trigger="session|weekday|luggage|situation")
    )
    evening = replace(expect("evening", formed=NOW, hours=4), observed_from=NOW)
    assert joinable(yesterday, evening) and joinable(evening, today)


def test_a_waiting_day_unit_expectation_of_another_day_is_not_pulled_into_the_unit() -> None:
    """白天拢单元（``runtime.foresight_units._groups``）：昨天起算、今天时窗开了的那条与今天起算、还没到时窗的那条范围相交，
    但不是同一刻，不拉进同一个单元（不然后者记「单元已选」、真漏了也不提醒）；还没到时窗的事件那条照拉。"""

    from habitus.runtime.foresight_units import _groups

    opened = _two_days("y", 5)
    later = replace(
        _two_days("t", 6), window=(opened.window[1] + timedelta(hours=1), opened.window[1] + timedelta(hours=2))
    )
    event = replace(
        expect("e", formed=opened.window[1], hours=2),
        observed_from=opened.window[1] - timedelta(hours=1),
    )
    ((group, due),) = _groups((opened,), (later, event))
    assert [item.expectation_id for item in group] == ["y", "e"] and due == ("y",)
