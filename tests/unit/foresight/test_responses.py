"""开着的提醒（方案 30 第五节）：说出口、没到最晚时限、没判出正 / 负的；按截止由近到远、最多 N 条；中不关、正 / 负关。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from habitus.foresight.errors import ForesightError
from habitus.foresight.ledger import Delivery, Expectation, Response
from habitus.foresight.responses import closes_after, open_reminders

CST = timezone(timedelta(hours=8))
NOW = datetime(2026, 10, 1, 10, 0, tzinfo=CST)


def expectation(identity: str, *, deadline_hours: int) -> Expectation:
    formed = NOW - timedelta(hours=1)
    return Expectation(
        expectation_id=identity,
        kind_token=f"s-{identity}",
        day=formed.date(),
        formed_at=formed,
        source="日子",
        trigger="rel",
        trigger_instance=f"behavior://{identity}.md",
        window=(formed, formed + timedelta(minutes=30)),
        deadline=formed + timedelta(hours=deadline_hours),
        basis=(),
        opportunities=3,
        hits=3,
        observed_from=formed,
    )


def said(identity: str, minutes_ago: int, text: str = "要吗") -> Delivery:
    return Delivery(identity, NOW - timedelta(minutes=minutes_ago), 1, True, "说了", 1.0, text)


def test_open_reminders_are_spoken_unexpired_and_not_yet_closed_sorted_by_deadline_and_capped() -> None:
    items = [
        expectation("late", deadline_hours=9),
        expectation("soon", deadline_hours=2),
        expectation("expired", deadline_hours=1),  # 最晚时限已过（形成 1 小时前 + 1 小时 = 此刻）
        expectation("quiet", deadline_hours=5),  # 没说出口
        expectation("closed", deadline_hours=5),
        expectation("waiting", deadline_hours=4),  # 回了中：还开着
    ]
    deliveries = [
        said("late", 50),
        said("soon", 40, "订酒店吗"),
        said("expired", 55),
        Delivery("quiet", NOW - timedelta(minutes=30), 0, False, "影子", 0.0),
        said("closed", 45),
        said("waiting", 35),
        said("late", 10),  # 同一条说了两次：取第一次
    ]
    responses = [
        Response("closed", NOW - timedelta(minutes=20), "负", "conv#1-2"),
        Response("waiting", NOW - timedelta(minutes=15), "中", "conv#3-4"),
    ]
    found = open_reminders(items, deliveries, responses, at=NOW, limit=5)
    assert [item.expectation_id for item in found] == ["soon", "waiting", "late"]
    assert found[0].text == "订酒店吗" and found[0].kind_token == "s-soon"
    assert found[2].said_at == NOW - timedelta(minutes=50)
    assert [item.expectation_id for item in open_reminders(items, deliveries, responses, at=NOW, limit=2)] == [
        "soon",
        "waiting",
    ]
    # 判出负之后它不再开着；判出中的照样开着；时刻之后才说的、之后才判的都不算
    later = Response("waiting", NOW + timedelta(minutes=5), "负", "conv#5-6")
    assert "waiting" not in {
        i.expectation_id
        for i in open_reminders(items, deliveries, [*responses, later], at=NOW + timedelta(minutes=10), limit=5)
    }
    assert "waiting" in {
        i.expectation_id for i in open_reminders(items, deliveries, [*responses, later], at=NOW, limit=5)
    }
    assert closes_after(responses[:1]) and not closes_after(responses[1:])
    with pytest.raises(ForesightError):
        open_reminders(items, deliveries, responses, at=NOW, limit=0)
    with pytest.raises(ForesightError):
        open_reminders(items, deliveries, responses, at=NOW.replace(tzinfo=None), limit=5)
