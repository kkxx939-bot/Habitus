"""行为单元（裁定 56 第 12 条）：一个行为看这一刻在场的全部输入，从历史里数下一步来了多少；分层退、依据、档、自检。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from habitus.foresight import Evidence, Tolerance, climb
from habitus.foresight.errors import ForesightError
from habitus.foresight.ledger import AccountFact, Antecedent, Expectation, account_key
from habitus.foresight.opportunities import TreeReading
from habitus.foresight.units import (
    Check,
    Input,
    Level,
    Moment,
    basis,
    moments,
    self_check,
    signature,
    unit_level,
    unit_reading,
)

CST = timezone(timedelta(hours=8))
NOW = datetime(2026, 9, 1, 21, 0, tzinfo=CST)
EVIDENCE = Evidence(alpha=0.05, half_life_days=60.0)
LOSS = Tolerance(floor=0.2, ceiling=0.9, unjudged=0.5)
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
    facts = [
        AccountFact(account_key(LUGGAGE, "事件", account), NOW - timedelta(days=30 - i), "预测", i < hits, 1.0, "准时" if i < hits else "过期待决", f"{account}{i}")
        for i in range(hits + misses)
    ]
    return climb(facts, tolerance=LOSS, evidence=EVIDENCE, now=NOW)


def entry(identity: str, hits: int, misses: int, **kwargs: object) -> Input:
    expectation = expect(identity, trigger=f"session|{identity}|luggage|days_2_7", **kwargs)  # type: ignore[arg-type]
    standing = account_standing(hits, misses, account=f"session|{identity}|luggage|days_2_7")
    return Input(expectation, standing)  # type: ignore[arg-type]


def test_the_signature_is_the_present_inputs_and_the_present_suppressors() -> None:
    cancel = Antecedent("session|cancel|luggage|next_day", True, False, "behavior://cancel.md")
    absent = Antecedent("session|other|luggage|next_day", False, False)
    raising = Antecedent("session|hotel|luggage|days_2_7", True, True, "behavior://hotel.md")
    first = expect("x", trigger="session|flight|luggage|days_2_7", antecedents=(cancel, absent, raising))
    second = expect("y", source="钟面", trigger="weekday=3")
    assert signature((first, second)) == frozenset({first.account, second.account, "↓session|cancel|luggage|next_day"})


def test_history_moments_are_the_overlapping_expectations_of_one_behavior_and_unknowns_are_dropped() -> None:
    """说中的范围连成一片的几条预期是同一刻；有一条自然命中就是来了，只有看得见的落空才是没来，其余（被提醒才做、观测空白）不算。"""

    day = NOW - timedelta(days=10)
    a = expect("a", formed=day, trigger="session|flight|luggage|days_2_7")
    b = expect("b", formed=day + timedelta(hours=2), source="钟面", trigger="weekday=3")
    c = expect("c", formed=day + timedelta(days=3))
    d = expect("d", formed=day + timedelta(days=5))
    facts = {
        "a": AccountFact(a.account, a.formed_at, "预测", False, 1.0, "过期待决", "a"),
        "b": AccountFact(b.account, b.formed_at, "预测", True, 0.4, "晚了但做了", "b"),
        "c": AccountFact(c.account, c.formed_at, "预测", False, 1.0, "过期待决", "c"),
        "d": AccountFact(d.account, d.formed_at, "预测", True, 0.0, "提醒后发生", "d"),
    }
    found = moments((d, c, b, a), lambda item: facts.get(item.expectation_id))
    assert [(moment.signature, moment.came) for moment in found] == [
        (frozenset({a.account, b.account}), True),  # 同一刻：钟面那条做了（强度低也是做了）
        (frozenset({c.account}), False),
    ]
    neutral = AccountFact(c.account, c.formed_at, "预测", False, 0.0, "过期待决", "c")  # 观测空白里的中性
    assert [m.came for m in moments((c,), lambda _item: neutral)] == []


def test_more_causes_present_together_read_higher_than_one_alone() -> None:
    """用户 10-17：出差、订机票、订酒店都命中时，收拾行李该比只有出差时高得多——从历史数：单个在场 6/10，两个一起在场 9/10。"""

    flight, hotel = entry("flight", 6, 4), entry("hotel", 6, 4)
    alone = frozenset({flight.account})
    together = frozenset({flight.account, hotel.account})
    history = [Moment(NOW - timedelta(days=40 - i), alone, i < 6) for i in range(10)]
    history += [Moment(NOW - timedelta(days=25 - i), together, i < 9) for i in range(10)]
    one = unit_reading((flight,), history, now=NOW, evidence=EVIDENCE, prior_strength=8.0)
    both = unit_reading((flight, hotel), history, now=NOW, evidence=EVIDENCE, prior_strength=8.0)
    assert one.level is Level.SAME_SET and both.level is Level.SAME_SET
    assert both.reading.mean > one.reading.mean + 0.15 and both.reading.lower > one.reading.lower


def test_a_suppressor_present_reads_lower_and_an_unseen_combination_leans_on_the_coarser_layer() -> None:
    """取消出差在场（压低）的那几刻下一步多半没来：同一组读数低；没见过的组合退到"同样多"、再退到最强那本账。"""

    flight = entry("flight", 9, 1)
    cancel = Antecedent("session|cancel|luggage|next_day", True, False, "behavior://cancel.md")
    cancelled = replace(flight, expectation=replace(flight.expectation, antecedents=(cancel,)))
    plain = frozenset({flight.account})
    suppressed = frozenset({flight.account, "↓session|cancel|luggage|next_day"})
    history = [Moment(NOW - timedelta(days=30 - i), plain, True) for i in range(8)]
    history += [Moment(NOW - timedelta(days=20 - i), suppressed, i < 1) for i in range(6)]
    normal = unit_reading((flight,), history, now=NOW, evidence=EVIDENCE, prior_strength=8.0)
    low = unit_reading((cancelled,), history, now=NOW, evidence=EVIDENCE, prior_strength=8.0)
    assert low.reading.mean < normal.reading.mean - 0.3
    # 没见过的组合：退到同样多；一刻都没有：就是最强那本账自己的读数
    hotel = entry("hotel", 5, 5)
    unseen = unit_reading((flight, hotel), history, now=NOW, evidence=EVIDENCE, prior_strength=8.0)
    assert unseen.level is Level.STRONGEST and unseen.reading == flight.standing.prediction
    assert unseen.seen == (0, 0, round(flight.standing.prediction.n))
    # 先验的分量不超过下一层自己的有效样本：一本只见过一次的账，带不出 8 次那么硬的先验
    thin = entry("thin", 1, 0)
    one_moment = [Moment(NOW - timedelta(days=1), frozenset({thin.account}), False)]
    shrunk = unit_reading((thin,), one_moment, now=NOW, evidence=EVIDENCE, prior_strength=8.0)
    assert shrunk.reading.n < 3.0
    with pytest.raises(ForesightError):
        unit_reading((), history, now=NOW, evidence=EVIDENCE, prior_strength=8.0)


def test_the_basis_is_the_established_cause_with_the_largest_lift_and_never_a_coincidence() -> None:
    """依据：事件输入里成立、效应够大、抬升最高的那条；巧合（效应不够大）不当依据，哪怕它的账更高；都没有就取账下界最高的。"""

    trip = replace(entry("trip", 9, 1), reading=TreeReading(1.0, 0.19, True), status="established")
    flight = replace(entry("flight", 9, 1), reading=TreeReading(0.8, 0.0, True), status="established")
    weekly = replace(entry("weekly", 30, 1), reading=TreeReading(0.92, 0.67, False), status="established")
    clock = Input(expect("clock", source="钟面", trigger="weekday=3"), account_standing(40, 2, account="clock"))  # type: ignore[arg-type]
    assert basis((weekly, trip, flight, clock)).account == trip.account  # 出差进日历抬升 0.81 > 订机票 0.80 > 巧合（不算）
    assert basis((weekly, flight, clock)).account == flight.account
    assert basis((weekly, clock)).account == clock.account  # 只有巧合与钟面：取钟面，巧合不当依据
    candidate = replace(flight, status="candidate")
    assert basis((candidate, clock)).account == clock.account  # 候选关系不当依据
    with pytest.raises(ForesightError):
        basis(())


def test_the_unit_level_follows_settings_then_the_reminded_group_then_the_reading_with_hysteresis() -> None:
    flight = entry("flight", 12, 0)
    history = [Moment(NOW - timedelta(days=20 - i), frozenset({flight.account}), True) for i in range(10)]
    strong = unit_reading((flight,), history, now=NOW, evidence=EVIDENCE, prior_strength=8.0)
    assert unit_level(flight.standing, strong, previous=None) == 1
    never = replace(flight.standing, setting="以后别提了")
    assert unit_level(never, strong, previous=1) == 0
    ignored = replace(
        flight.standing,
        reminded=climb(
            [AccountFact(flight.account, NOW - timedelta(days=5 - i), "提醒组", False, 1.0, "落空", f"r{i}") for i in range(5)],
            tolerance=LOSS,
            evidence=EVIDENCE,
            now=NOW,
        ).reminded,
    )
    assert unit_level(ignored, strong, previous=1) == 0  # 说了他老不做：照降
    middling = replace(strong, reading=replace(strong.reading, lower=0.5, upper=0.8))
    assert unit_level(flight.standing, middling, previous=1) == 1 and unit_level(flight.standing, middling, previous=0) == 0
    always = replace(flight.standing, setting="以后都提醒我")
    weak = replace(strong, reading=replace(strong.reading, lower=0.1, upper=0.3))
    assert unit_level(always, weak, previous=None) == 1


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
