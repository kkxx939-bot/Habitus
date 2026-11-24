"""loss 尺子的算分（裁定 114）：按一刻计分、一次发生只算给一刻；"不知道"不计分；没有读数的是"没量到"；取决定时实际用的那次读数；
说出口与没说出口分开；对数损失有界；分量照单元历史的口径。"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from habitus.foresight.history import MomentRecord
from habitus.foresight.ledger import Delivery, Expectation, Readings, decode_delivery, encode_delivery
from habitus.foresight.loss import LOG_LOSS_FLOOR, NO_HISTORY, SPOKEN, UNSPOKEN, measure, score
from habitus.foresight.reliability import Reliability, reading
from habitus.foresight.units import UNIT

CST = timezone(timedelta(hours=8))
DAY0 = datetime(2026, 10, 1, 9, 0, tzinfo=CST)
KIND = "s-k0002"


def expectation(identity: str, formed: datetime, *, source: str = "事件", hits: int = 3, opportunities: int = 4):
    return Expectation(
        expectation_id=identity,
        kind_token=KIND,
        day=formed.date(),
        formed_at=formed,
        source=source,
        trigger=f"t-{identity}",
        trigger_instance=f"occ:{identity}.md",
        window=(formed + timedelta(minutes=10), formed + timedelta(minutes=40)),
        deadline=formed + timedelta(hours=2),
        basis=(),
        opportunities=opportunities,
        hits=hits,
        observed_from=formed,
    )


def record(item: Expectation, came: bool | None, strength: float = 1.0, occurrence: str | None = None) -> MomentRecord:
    return MomentRecord(
        item.expectation_id,
        item.kind_token,
        item.account,
        item.formed_at,
        item.observed_from,
        item.deadline,
        came,
        strength if came is not None else 0.0,
        occurrence if came is True else None,
    )


def readings(unit: float, account: float = 0.5) -> Readings:
    def of(mean: float) -> Reliability:
        return Reliability(k=mean, n=1.0, mean=mean, lower=mean / 2, upper=min(1.0, mean * 1.5))

    return Readings(of(unit), "同一组", of(account))


def delivery(identity: str, at: datetime, *, unit: float | None, reason: str = "等时窗末", account: float = 0.5):
    spoken = reason == "说了"
    return Delivery(
        identity,
        at,
        1,
        spoken,
        reason,
        1.0 if spoken else 0.0,
        "话" if spoken else None,
        loss_readings=None if unit is None else readings(unit, account),
    )


def test_one_occurrence_said_by_three_connected_expectations_is_scored_as_one_moment() -> None:
    a, b, c = (expectation(name, DAY0 + timedelta(minutes=offset)) for name, offset in (("a", 0), ("b", 30), ("c", 60)))
    found = measure(
        [record(item, True, occurrence="occ:x.md") for item in (a, b, c)],
        {item.expectation_id: item for item in (a, b, c)},
        [delivery(item.expectation_id, DAY0 + timedelta(minutes=15), unit=0.8) for item in (a, b, c)],
    )
    assert found.expectations == 3 and found.moments == 1 and found.unknown_moments == 0
    assert found.total is not None and found.total.moments == 1
    assert found.total.unit.log_loss == pytest.approx(-math.log(0.8))


def test_a_moment_whose_outcome_is_unknown_is_not_scored() -> None:
    known, unknown = expectation("known", DAY0), expectation("unknown", DAY0 + timedelta(days=1))
    found = measure(
        [record(known, False), record(unknown, None)],
        {"known": known, "unknown": unknown},
        [delivery("known", DAY0, unit=0.7), delivery("unknown", DAY0 + timedelta(days=1), unit=0.7)],
    )
    assert found.moments == 1 and found.unknown_moments == 1
    assert found.total is not None and found.total.moments == 1
    assert [item.expectation_ids for item in found.scored] == [("known",)]


def test_an_expectation_never_seen_in_an_open_window_has_no_reading_and_is_counted_apart() -> None:
    seen, missed = expectation("seen", DAY0), expectation("missed", DAY0 + timedelta(days=1))
    found = measure(
        [record(seen, True), record(missed, True)],
        {"seen": seen, "missed": missed},
        [delivery("seen", DAY0, unit=0.9), delivery("missed", DAY0 + timedelta(days=1), unit=None, reason=UNIT)],
    )
    assert found.unmeasured_expectations == 1 and found.unmeasured_moments == 1 and found.moments == 2
    assert found.total is not None and found.total.moments == 1


def test_the_reading_used_is_the_latest_decision_and_a_basis_wins_a_tie_with_its_members() -> None:
    basis, member = expectation("basis", DAY0, source="日子"), expectation("member", DAY0 + timedelta(minutes=5))
    opened = DAY0 + timedelta(minutes=15)
    deliveries = [
        delivery("member", opened, unit=0.2, reason=UNIT, account=0.1),
        delivery("basis", opened, unit=0.3, account=0.4),
    ]
    first = measure([record(basis, True), record(member, True)], {"basis": basis, "member": member}, deliveries)
    (moment,) = first.scored
    assert (moment.unit, moment.account, moment.source) == (0.3, 0.4, "日子")  # 同一拍并列：依据优先
    closed = [*deliveries, delivery("basis", DAY0 + timedelta(minutes=41), unit=0.6, reason="待取", account=0.7)]
    later = measure([record(basis, True), record(member, True)], {"basis": basis, "member": member}, closed)
    assert (later.scored[0].unit, later.scored[0].account) == (0.6, 0.7)  # 时窗末重算的那一次


def test_spoken_moments_are_grouped_apart_including_members_whose_basis_spoke() -> None:
    quiet, basis, member = (
        expectation("quiet", DAY0),
        expectation("basis", DAY0 + timedelta(days=1)),
        expectation("member", DAY0 + timedelta(days=1, minutes=5)),
    )
    day1 = DAY0 + timedelta(days=1)
    found = measure(
        [record(quiet, True), record(basis, False), record(member, False)],
        {item.expectation_id: item for item in (quiet, basis, member)},
        [
            delivery("quiet", DAY0, unit=0.5),
            Delivery("member", day1, 1, False, UNIT, 0.0, evidence=("单元:basis",), loss_readings=readings(0.4)),
            delivery("basis", day1 + timedelta(minutes=1), unit=0.4),
            delivery("basis", day1 + timedelta(minutes=50), unit=None, reason="说了"),
        ],
    )
    assert set(found.by_spoken) == {SPOKEN, UNSPOKEN}
    assert found.by_spoken[SPOKEN].moments == 1 and found.by_spoken[UNSPOKEN].moments == 1


def test_the_log_loss_is_bounded_when_a_certain_probability_turns_out_wrong() -> None:
    assert score(1.0, False, 1.0)[0] == pytest.approx(-math.log(LOG_LOSS_FLOOR))
    assert score(0.0, True, 1.0)[0] == pytest.approx(-math.log(LOG_LOSS_FLOOR))
    assert score(1.0, False, 1.0)[1] == 1.0
    always = expectation("always", DAY0, hits=20, opportunities=20)
    fresh = expectation("fresh", DAY0 + timedelta(days=1), hits=0, opportunities=0)
    found = measure(
        [record(always, False), record(fresh, True)],
        {"always": always, "fresh": fresh},
        [delivery("always", DAY0, unit=0.9), delivery("fresh", DAY0 + timedelta(days=1), unit=0.5)],
    )
    raw = {item.expectation_ids[0]: item.raw for item in found.scored}
    assert raw == {"always": 1.0, "fresh": NO_HISTORY}
    assert found.total is not None and math.isfinite(found.total.raw.log_loss)
    assert found.total.raw.log_loss == pytest.approx((-math.log(LOG_LOSS_FLOOR) - math.log(0.5)) / 2)


def test_scores_are_weighted_by_the_moment_strength() -> None:
    hit, late = expectation("hit", DAY0), expectation("late", DAY0 + timedelta(days=1))
    found = measure(
        [record(hit, True, 1.0), record(late, False, 0.25)],
        {"hit": hit, "late": late},
        [delivery("hit", DAY0, unit=0.8), delivery("late", DAY0 + timedelta(days=1), unit=0.8)],
    )
    assert found.total is not None and found.total.weight == pytest.approx(1.25)
    expected = (-math.log(0.8) - 0.25 * math.log(0.2)) / 1.25
    assert found.total.unit.log_loss == pytest.approx(expected)
    assert found.total.unit.brier == pytest.approx((0.2**2 + 0.25 * 0.8**2) / 1.25)


def test_readings_round_trip_through_the_codec_and_an_old_record_without_them_reads_as_unmeasured() -> None:
    stored = Delivery(
        "e1",
        DAY0,
        1,
        False,
        "等时窗末",
        0.0,
        loss_readings=Readings(reading(2.5, 3.75, alpha=0.05), "同样多", reading(1.0, 4.0, alpha=0.05)),
    )
    assert decode_delivery(encode_delivery(stored)) == stored
    old = encode_delivery(Delivery("e1", DAY0, 1, False, "等时窗末", 0.0))
    old.pop("loss_readings")
    assert decode_delivery(old).loss_readings is None
