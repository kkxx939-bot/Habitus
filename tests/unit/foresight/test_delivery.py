"""递送原因（裁定 48–51、54、58、64、67、68）：时窗开 / 时窗末 / 等宿主各按顺序过原因，前面的成立了后面的材料不读；
「以后都提醒我」优先于久别；递送判做了、归约判提醒过各自的判法。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from habitus.foresight import Standing
from habitus.foresight.delivery import (
    PENDING,
    Circumstances,
    done_since,
    on_awaiting,
    on_closing,
    on_opening,
    quiet_after_absence,
    reminded_before,
)
from habitus.foresight.ledger import AWAITING_HOST, HOST_NEVER_CLAIMED, RETURNED, UNOBSERVED, Delivery, Expectation
from habitus.foresight.model import Seen
from habitus.foresight.reliability import Reliability

CST = timezone(timedelta(hours=8))
FORMED = datetime(2026, 12, 10, 14, 0, tzinfo=CST)
NOW = FORMED + timedelta(hours=1)
KIND = "s-k0002"


def expectation(**overrides: object) -> Expectation:
    values: dict[str, object] = dict(
        expectation_id="hotel-1210",
        kind_token=KIND,
        day=FORMED.date(),
        formed_at=FORMED,
        source="事件",
        trigger="session|s-k0001|s-k0002|chain",
        trigger_instance="behavior://flight.md",
        window=(FORMED + timedelta(minutes=4), FORMED + timedelta(minutes=51)),
        deadline=FORMED + timedelta(hours=3),
        basis=(),
        opportunities=13,
        hits=10,
        observed_from=FORMED,
    )
    values.update(overrides)
    return Expectation(**values)  # type: ignore[arg-type]


def standing(level: int, *, setting: str | None = None) -> Standing:
    return Standing(level=level, threshold=0.67, prediction=Reliability(9.0, 10.0, 0.9, 0.7, 0.97), setting=setting)


ITEM = expectation()
PENDING_ONE = Delivery(ITEM.expectation_id, FORMED + timedelta(minutes=4), 1, False, PENDING, 0.0)


def facts(
    *, happened: bool = False, absent: bool = False, unobserved: bool = False, holds: bool = True
) -> Circumstances:
    return Circumstances(
        happened=lambda: happened,
        absent=lambda: absent,
        unobserved=lambda: unobserved,
        antecedent_holds=lambda: holds,
    )


def must_not_ask() -> bool:
    raise AssertionError("this material must not be read once an earlier reason holds")


def must_not_ask_text() -> str:
    raise AssertionError("the sentence must not be written when nothing will be said")


def closing(**overrides: object) -> Delivery:
    values: dict[str, object] = dict(
        now=NOW,
        facts=facts(),
        kept=None,
        evidence=(),
        spoken_kinds=frozenset(),
        alive=True,
        spoken_today=0,
        daily_cap=3,
        text=lambda: "该订酒店了",
    )
    values.update(overrides)
    return on_closing(ITEM, PENDING_ONE, standing(1), **values)  # type: ignore[arg-type]


def test_opening_records_happened_then_shadow_then_returned_then_waits_for_the_window_end() -> None:
    assert on_opening(ITEM, standing(1), now=NOW, facts=facts(happened=True)).reason == "已发生"
    shadow = on_opening(ITEM, standing(0), now=NOW, facts=Circumstances(happened=lambda: False, absent=must_not_ask))
    assert (shadow.reason, shadow.level) == ("影子", 0)
    assert on_opening(ITEM, standing(1), now=NOW, facts=facts(absent=True)).reason == RETURNED
    waiting = on_opening(ITEM, standing(1), now=NOW, facts=facts())
    assert (waiting.reason, waiting.level, waiting.spoken) == (PENDING, 1, False)


def test_always_remind_me_overrides_a_return_after_absence_and_absence_is_only_read_when_needed() -> None:
    assert quiet_after_absence(standing(1, setting="以后都提醒我"), must_not_ask) is False
    assert quiet_after_absence(standing(1), lambda: True) is True
    assert on_opening(ITEM, standing(1, setting="以后都提醒我"), now=NOW, facts=facts(absent=True)).reason == PENDING


def test_closing_goes_through_the_reasons_in_order_before_the_ladder_decides() -> None:
    happened = closing(facts=Circumstances(happened=lambda: True, absent=must_not_ask, unobserved=must_not_ask))
    assert (happened.reason, happened.level) == ("已发生", PENDING_ONE.level)
    assert closing(facts=facts(unobserved=True)).reason == UNOBSERVED
    assert (
        on_closing(
            ITEM,
            PENDING_ONE,
            standing(0),
            now=NOW,
            facts=facts(),
            kept=None,
            evidence=(),
            spoken_kinds=frozenset(),
            alive=True,
            spoken_today=0,
            daily_cap=3,
            text=must_not_ask_text,
        ).reason
        == "影子"
    )
    assert closing(facts=facts(absent=True)).reason == RETURNED
    broken = closing(facts=facts(holds=False), evidence=("behavior://a.md",))
    assert (broken.reason, broken.evidence) == ("前因不成立", ("behavior://a.md",))
    assert closing(kept=False).reason == "模型判不像"
    assert closing(spoken_kinds=frozenset({KIND})).reason == "同类已说"
    assert closing(alive=False, text=must_not_ask_text).reason == "不可达"
    assert closing(spoken_today=3, daily_cap=3).reason == "名额已满"
    said = closing(kept=True)
    assert (said.reason, said.text, said.spoken) == (AWAITING_HOST, "该订酒店了", False)


def test_awaiting_closes_on_doing_it_or_on_the_deadline_and_otherwise_waits() -> None:
    assert on_awaiting(ITEM, PENDING_ONE, now=NOW, happened=True).reason == "已发生"  # type: ignore[union-attr]
    late = on_awaiting(ITEM, PENDING_ONE, now=ITEM.deadline + timedelta(minutes=1), happened=False)
    assert late is not None and late.reason == HOST_NEVER_CLAIMED
    assert on_awaiting(ITEM, PENDING_ONE, now=ITEM.deadline, happened=False) is None


def seen(at: datetime, *, kind: str = KIND, abandoned: bool = False) -> Seen:
    return Seen(at, at + timedelta(minutes=5), kind, f"behavior://{at:%H%M}.md", abandoned=abandoned)


def unchanged(token: str) -> tuple[str, ...]:
    return (token,)


def test_done_since_counts_the_same_kind_started_after_observed_from_up_to_now_and_not_abandoned() -> None:
    assert done_since(ITEM, [seen(FORMED + timedelta(minutes=10))], now=NOW, current=unchanged)
    assert not done_since(ITEM, [seen(FORMED)], now=NOW, current=unchanged)  # 起算那一刻不算
    assert not done_since(ITEM, [seen(NOW + timedelta(minutes=1))], now=NOW, current=unchanged)
    assert not done_since(ITEM, [seen(FORMED + timedelta(minutes=10), abandoned=True)], now=NOW, current=unchanged)
    assert not done_since(ITEM, [seen(FORMED + timedelta(minutes=10), kind="s-k0099")], now=NOW, current=unchanged)


def test_done_since_reads_like_the_settlement_the_mapped_kind_and_nothing_after_the_deadline() -> None:
    """段 6 Q4 / Q15：递送判做了与结算（``first_occurrence``）同一个口径——词表把后果拆成了别的编号，做了就是做了（原来只认
    原编号：他做完了还被问一遍）；最晚时限之后才做的不算这一回做了（原来照算「已发生」，结算却结成过期待决）。"""

    from habitus.foresight.ledger import first_occurrence
    from habitus.scene.views import FlowRow

    split = lambda token: ("s-k0099",) if token == KIND else (token,)  # noqa: E731
    later = ITEM.deadline + timedelta(minutes=10)
    for at, kind, current in (
        (FORMED + timedelta(minutes=10), "s-k0099", split),
        (FORMED + timedelta(minutes=10), "s-k0099", unchanged),
        (later, KIND, unchanged),
        (ITEM.deadline, KIND, unchanged),
    ):
        row = FlowRow(uri="u", name="u", kind_token=kind, at=at, last_observed_at=at, summary="", day_count=1)
        settled = first_occurrence(ITEM, [row], current=current) is not None
        assert done_since(ITEM, [seen(at, kind=kind)], now=later, current=current) is settled
    assert done_since(ITEM, [seen(FORMED + timedelta(minutes=10), kind="s-k0099")], now=NOW, current=split)
    assert not done_since(ITEM, [seen(later)], now=later, current=unchanged)


def test_reminded_before_needs_the_decision_first_the_record_in_range_and_the_current_kind() -> None:
    started = FORMED + timedelta(minutes=30)
    decided = FORMED + timedelta(minutes=10)
    unchanged = lambda token: (token,)  # noqa: E731
    assert reminded_before(ITEM, decided, kind_token=KIND, started_at=started, current=unchanged)
    assert not reminded_before(ITEM, started, kind_token=KIND, started_at=started, current=unchanged)
    assert not reminded_before(
        ITEM, decided, kind_token=KIND, started_at=ITEM.deadline + timedelta(minutes=1), current=unchanged
    )
    assert not reminded_before(ITEM, decided, kind_token="s-k0099", started_at=started, current=unchanged)
    assert reminded_before(ITEM, decided, kind_token="s-k0099", started_at=started, current=lambda _t: ("s-k0099",))
