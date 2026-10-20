"""开着的提醒与回话的桥：从账本里还没结完的日子取说出口的提醒，回应按轮写回；写不进去只记观测。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from habitus.behavior.fusion.lanes.session import ReminderResponse
from habitus.foresight.ledger import Delivery, Expectation
from habitus.foundation.observability import ObservationEvent
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_responses import ReminderResponses

CST = timezone(timedelta(hours=8))
DAY0 = datetime(2026, 10, 1, 9, 0, tzinfo=CST)


class Recorder:
    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)


def expectation(identity: str, formed: datetime, *, deadline_hours: int = 3) -> Expectation:
    return Expectation(
        expectation_id=identity,
        kind_token="s-k0002",
        day=formed.date(),
        formed_at=formed,
        source="日子",
        trigger="rel",
        trigger_instance=f"behavior://{identity}.md",
        window=(formed + timedelta(minutes=10), formed + timedelta(minutes=40)),
        deadline=formed + timedelta(hours=deadline_hours),
        basis=(),
        opportunities=3,
        hits=3,
        observed_from=formed,
    )


def test_the_bridge_lists_open_reminders_and_writes_responses_by_turn(tmp_path) -> None:
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    recorder = Recorder()
    bridge = ReminderResponses(ledger, limit=5, observer=recorder)
    ledger.record_expectation(expectation("hotel", DAY0))
    ledger.record_expectation(expectation("card", DAY0, deadline_hours=1))
    ledger.record_delivery(
        DAY0.date(), Delivery("hotel", DAY0 + timedelta(minutes=41), 1, True, "说了", 1.0, "订酒店吗")
    )
    ledger.record_delivery(DAY0.date(), Delivery("card", DAY0 + timedelta(minutes=41), 1, True, "说了", 1.0, "还卡吗"))
    at = DAY0 + timedelta(hours=2)
    found = bridge.open(at)
    assert [item.reminder_id for item in found] == ["hotel"]  # 还信用卡那条最晚时限已过
    assert found[0].text == "订酒店吗" and found[0].deadline == DAY0 + timedelta(hours=3)
    bridge.respond("conv#1-3", ReminderResponse("hotel", "中"), at=at)
    assert bridge.open(at + timedelta(minutes=1))[0].reminder_id == "hotel"  # 中不关
    bridge.respond("conv#4-6", ReminderResponse("hotel", "负"), at=at + timedelta(minutes=5))
    assert bridge.open(at + timedelta(minutes=10)) == ()  # 负关
    stored = ledger.responses_on(DAY0.date())
    assert [(item.polarity, item.turn) for item in stored] == [("中", "conv#1-3"), ("负", "conv#4-6")]
    # 同一轮重放同一个方向：幂等；不同方向：账本只增不改，这一次不写、只记观测
    bridge.respond("conv#4-6", ReminderResponse("hotel", "负"), at=at + timedelta(minutes=5))
    bridge.respond("conv#4-6", ReminderResponse("hotel", "正"), at=at + timedelta(minutes=5))
    assert len(ledger.responses_on(DAY0.date())) == 2
    bridge.respond("conv#7-8", ReminderResponse("nope", "正"), at=at)
    assert [event.status.value for event in recorder.events] == ["success", "success", "success", "failure", "failure"]
    with pytest.raises(ValueError):
        ReminderResponses(ledger, limit=0)
    with pytest.raises(TypeError):
        ReminderResponses(object(), limit=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("formed", "said", "asked"),
    [
        ((0, 30), (1, 0), (1, 30)),  # 本地 01:30 = 前一 UTC 日 17:30Z
        ((7, 30), (7, 45), (7, 50)),  # 本地 07:50 = 前一 UTC 日 23:50Z
    ],
)
def test_a_reminder_is_open_whichever_zone_the_turn_is_stamped_in(tmp_path, formed, said, asked) -> None:  # type: ignore[no-untyped-def]
    """预期的日子是主体本地日，会话的时刻一律 UTC 写法：本地凌晨说出口的提醒，UTC 日期还是前一天，不能因此漏掉。"""

    from datetime import UTC

    day = datetime(2026, 10, 2, tzinfo=CST)
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    bridge = ReminderResponses(ledger, limit=5)
    item = expectation("hotel", day.replace(hour=formed[0], minute=formed[1]), deadline_hours=6)
    ledger.record_expectation(item)
    ledger.record_delivery(
        item.day, Delivery("hotel", day.replace(hour=said[0], minute=said[1]), 1, True, "说了", 1.0, "订酒店吗")
    )
    local = day.replace(hour=asked[0], minute=asked[1])
    assert [found.reminder_id for found in bridge.open(local)] == ["hotel"]
    assert [found.reminder_id for found in bridge.open(local.astimezone(UTC))] == ["hotel"]


def test_the_bridge_reads_only_days_with_something_spoken(tmp_path, monkeypatch) -> None:
    """送达没接上时每一天都没说出口：只读递送目录就知道，不去读预期与回话。"""

    ledger = ForesightLedgerStore(tmp_path / "ledger")
    bridge = ReminderResponses(ledger, limit=5)
    for offset in range(5):
        formed = DAY0 + timedelta(days=offset)
        ledger.record_expectation(expectation(f"e{offset}", formed, deadline_hours=24 * 7))
        ledger.record_delivery(
            formed.date(), Delivery(f"e{offset}", formed + timedelta(minutes=41), 0, False, "影子", 0.0)
        )
    ledger.record_delivery(DAY0.date(), Delivery("e0", DAY0 + timedelta(minutes=42), 1, True, "说了", 1.0, "订酒店吗"))
    read: list[object] = []
    original = ledger.expectations_on
    monkeypatch.setattr(ledger, "expectations_on", lambda day: (read.append(day), original(day))[1])
    assert [item.reminder_id for item in bridge.open(DAY0 + timedelta(days=5))] == ["e0"]
    assert read == [DAY0.date()]


def test_a_response_that_cannot_be_written_is_observed_not_raised(tmp_path, monkeypatch) -> None:
    """索引坏了、盘写不进（不是账本自己的错误）：这一轮的记录已经落定，回应丢了只记观测、按"没回应"结算。"""

    import errno

    from habitus.infrastructure.store.filesystem.durable_io import atomic_file

    ledger = ForesightLedgerStore(tmp_path / "ledger")
    recorder = Recorder()
    bridge = ReminderResponses(ledger, limit=5, observer=recorder)
    ledger.record_expectation(expectation("hotel", DAY0))
    ledger.record_delivery(
        DAY0.date(), Delivery("hotel", DAY0 + timedelta(minutes=41), 1, True, "说了", 1.0, "订酒店吗")
    )
    at = DAY0 + timedelta(hours=1)

    def no_space(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(atomic_file.os, "link", no_space)
    bridge.respond("conv#1", ReminderResponse("hotel", "正"), at=at)
    monkeypatch.undo()
    index = tmp_path / "ledger" / "index" / "expectations" / "hotel.json"
    assert index.exists()
    index.write_text("{not json", encoding="utf-8")
    bridge.respond("conv#2", ReminderResponse("hotel", "负"), at=at)
    assert ledger.responses_on(DAY0.date()) == ()
    assert [event.status.value for event in recorder.events] == ["failure", "failure"]
    assert recorder.events[0].attributes["error"] == "OSError"


def test_the_lanes_polarities_are_the_ledgers_polarities() -> None:
    """会话 lane 不认识预测层，方向的三个字各写一份：这里钉住两份相同，漂了桥会把回应静默丢掉。"""

    from habitus.behavior.fusion.lanes.session import reminders
    from habitus.foresight.ledger import model

    assert tuple(reminders.POLARITIES) == tuple(model.POLARITIES)
