"""语义树上的复现规律：每个类按天的发生与相邻间隔，由 temporal 算（裁定 59）。"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

from habitus.scene.concepts.rhythm import RhythmRulers
from habitus.scene.recurrence import recurrences_of
from tests.unit.temporal.fixtures import at, record, series

RULERS = RhythmRulers(15, 2700, 3, 5.0, 5.0, 0.10, 35)
START = date(2026, 3, 4)  # 周三


def test_each_classified_kind_gets_its_days_and_intervals() -> None:
    weekly = [record(at(START + timedelta(days=7 * week), 17), "s-k0001") for week in range(5)]
    pending = replace(record(at(START, 9), "s-待定"), classified=False)
    found = recurrences_of(series([*weekly, pending], cutoff=START + timedelta(days=30)), RULERS)
    assert set(found) == {"s-k0001"}
    meeting = found["s-k0001"]
    assert len(meeting.days) == 5 and [item.days for item in meeting.intervals] == [7, 7, 7, 7]
    assert all(item.seen == 1.0 for item in meeting.intervals) and meeting.lane == "session"
