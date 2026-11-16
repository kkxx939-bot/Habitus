"""预测不等换代（2026-11-16，``预期.md`` D3、C5、C6、C7、C8 的健康面那一半）：走组合根装出来的 Runtime 与真实
``RuntimeHealthService``（``runtime.health()``）。

新的那一项叫什么、状态是什么不猜：在"旧表"盘面下找一项——名字带支（``_session``）、detail 里有那个天数（不是日期的一部分）、
detail 里没有内部的词——并且在"今天这一代做成"的盘面下同名那一项已经消失或变了。健康面用真实的当地今天（``datetime.now``）。
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from habitus.foresight.opportunities import OpportunityTable
from habitus.runtime import RuntimeHealthCheck
from habitus.runtime.generation_marks import GenerationMark
from habitus.runtime.pipeline import ScenePipeline
from tests.integration.test_runtime_assembly import _fully_enabled_runtime
from tests.unit.stale_table.harness import BANNED_WORDS


class Disk:
    """组合根装出来的两支：往各支的机会表与完成标记里写东西，读健康面。"""

    def __init__(self, tmp: Path) -> None:
        self.runtime = _fully_enabled_runtime(tmp)
        zone = self.runtime.config.locale.zone()
        self.zone = zone
        self.today = datetime.now(UTC).astimezone(zone).date()

    def pipeline(self, lane: str) -> ScenePipeline:
        found = self.runtime.components.pipeline(lane)
        assert found is not None and found.foresight is not None
        return found

    def generation(self, day: date, *, lane: str = "session", marked: bool = True) -> None:
        pipeline = self.pipeline(lane)
        foresight = pipeline.foresight
        assert foresight is not None
        foresight.opportunities.write(OpportunityTable(lane, day, day, 15, 3, (), ()))
        if marked:
            moment = datetime.combine(day, datetime.min.time(), tzinfo=self.zone) + timedelta(hours=1)
            pipeline.generation.generation.marks.write(GenerationMark(day, moment))

    def corrupt_mark(self, day: date, *, lane: str = "session") -> None:
        path = self.pipeline(lane).generation.generation.marks.path(day)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"schema": "generation_mark_v1"', encoding="utf-8")

    def closed_through(self, day: date) -> None:
        def closed() -> tuple[date, ...]:
            return (day,)

        for lane in ("session", "physical"):
            self.pipeline(lane).generation.generation.closed_days = closed

    def checks(self) -> list[RuntimeHealthCheck]:
        return list(asyncio.run(self.runtime.health()).checks)


def _days_in(detail: str, days: int) -> bool:
    return re.search(rf"(?<![\d:./\-]){days}(?![\d:./\-])", detail) is not None


def stale_items(checks: list[RuntimeHealthCheck], days: int, lane: str) -> list[RuntimeHealthCheck]:
    return [
        check
        for check in checks
        if check.name.endswith(f"_{lane}")
        and _days_in(check.detail, days)
        and not any(word in check.detail for word in BANNED_WORDS)
    ]


def _stale_disk(tmp: Path, days: int) -> Disk:
    """会话一支：最近一张完成的表是 ``days`` 天前的、今天这一代还没成；物理一支今天这一代已经做成。"""

    disk = Disk(tmp)
    disk.closed_through(disk.today - timedelta(days=1))
    disk.generation(disk.today - timedelta(days=days), lane="session")
    disk.generation(disk.today, lane="physical")
    return disk


def test_d3_the_health_view_says_in_plain_words_how_many_days_old_the_habits_behind_the_reminders_are(
    tmp_path: Path,
) -> None:
    """D3：多一种状态——现在的提醒是按几天前整理好的习惯给的，按支报；那句话里没有"换代""机会表""这一代""定稿""槽"。"""

    disk = _stale_disk(tmp_path, 3)
    checks = disk.checks()
    found = stale_items(checks, 3, "session")
    assert found, f"会话一支没有一项说出 3 天：{[(c.name, c.detail) for c in checks]}"
    assert not stale_items(checks, 3, "physical")  # 物理一支今天这一代做成了：它那一支不报 3 天


def test_c6_c7_two_weeks_old_is_said_as_fourteen_days_and_goes_away_after_todays_generation(tmp_path: Path) -> None:
    """C6：最近一张完成的表是 14 天前的，健康面那句话说的是 14 天；C7：今天这一代做成后那一项消失（或不再报这个天数）。"""

    disk = _stale_disk(tmp_path, 14)
    found = stale_items(disk.checks(), 14, "session")
    assert found, "会话一支没有一项说出 14 天"
    names = {check.name for check in found}
    disk.generation(disk.today, lane="session")
    after = {check.name: check for check in disk.checks()}
    for name in names:
        assert name not in after or not _days_in(after[name].detail, 14), f"做成以后 {name} 还在报 14 天"


def test_c8_the_number_of_days_grows_while_the_generation_stays_stuck(tmp_path: Path) -> None:
    """C8：换代连着几天做不成，健康面报的天数跟着长（每多一天重造一次盘面：表停在同一晚、今天往后挪 ⇔ 表的日子往前挪）。
    差 3 天起才报（用户 2026-11-16「不应该是差两天 应该要差3天才可以」）：差 1、2 天是每天早上换代做成之前的常态。"""

    said: list[int] = []
    for days in (1, 2, 3, 5, 10):
        disk = _stale_disk(tmp_path / f"d{days}", days)
        if stale_items(disk.checks(), days, "session"):
            said.append(days)
    assert said == [3, 5, 10]


def test_c5_a_corrupt_todays_mark_is_still_reported_as_unreadable(tmp_path: Path) -> None:
    """C5（健康面那一半，改动前后都该过）：今天的完成标记坏了，健康面照常报标记读不了。"""

    disk = Disk(tmp_path)
    disk.generation(disk.today - timedelta(days=1))
    disk.generation(disk.today, marked=False)
    disk.corrupt_mark(disk.today)
    check = next(item for item in disk.checks() if item.name == "generation_marks_session")
    assert check.status.value == "degraded" and "unreadable" in check.detail
