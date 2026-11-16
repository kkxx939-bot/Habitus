"""预测不等换代（2026-11-16，``预期.md`` 第 1 节）：应当不变的。

- S1：今天这一代在今天第一拍之前就做成了——单元级见 ``test_stale_table_forming.test_control_*``，整体比对靠丙一效果跑数 Haiku 那一遍
  的重放（阶段 A 报告）；
- S2：换代本身不变——靠现有 ``tests/unit/runtime/test_generation*.py``（这里不重写）与两份重放里每天换代的结局；
- S3：递送用盘上最晚那张表；S4：一张表都没有不建、原因不变；S5：见 ``test_stale_table_forming.test_c14_s5_*``。
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path

from habitus.runtime.foresight_delivery import DeliveryReport, JudgeMaterials
from habitus.runtime.foresight_forming import NO_TABLE
from tests.unit.scene.fixtures import publish
from tests.unit.stale_table.harness import BASE, ZONE, days_after, world

TODAY = BASE + timedelta(days=4)
YESTERDAY = TODAY - timedelta(days=1)


def clock(hour: int, minute: int) -> datetime:
    return datetime.combine(TODAY, time(hour, minute), tzinfo=ZONE)


def test_s3_delivery_reads_the_latest_table_on_disk(tmp_path: Path) -> None:
    """S3：递送那一步拿到的是盘上最晚那张表——今天的表写出了、还没完成标记时也是它（建预期用哪张与递送无关）。"""

    w = world(tmp_path)
    w.mornings(days_after(BASE, 4))
    w.table(YESTERDAY)
    w.mark(YESTERDAY)
    w.table(TODAY)  # 今天的表写出了，没有完成标记
    seen: list[object] = []
    stage = w.session.delivery
    original = stage.run_once

    async def spy(now: datetime, materials: JudgeMaterials | None = None) -> DeliveryReport:
        seen.append(None if materials is None or materials.table is None else materials.table.night)
        return await original(now, materials)

    stage.run_once = spy  # type: ignore[method-assign]
    publish(w.site.behavior_tree, TODAY, "讨论方案", 10, 0)
    w.tick(clock(10, 15))
    assert seen == [TODAY]


def test_s4_without_any_table_nothing_is_formed_and_the_reason_is_unchanged(tmp_path: Path) -> None:
    """S4：一张表都没有（全新安装、第一代还没做）：不建预期，原因照旧是「还没有机会表」。"""

    w = world(tmp_path)
    w.mornings(days_after(BASE, 4))
    publish(w.site.behavior_tree, TODAY, "讨论方案", 10, 0)
    report = w.tick(clock(10, 15))
    assert report.skipped == NO_TABLE and report.recorded == 0
    assert w.expectations(TODAY) == []


def test_s4_a_table_without_any_completed_generation_is_not_used_to_form(tmp_path: Path) -> None:
    """S4 的边（共享说明 4："没有完成标记的今天的表不用，用上一张完成了的"）：盘上只有今天这张没完成的表、没有任何完成了的那一代，
    不建预期（改动前后都不建；原因的写法不约束）。"""

    w = world(tmp_path)
    w.mornings(days_after(BASE, 4))
    w.table(TODAY)
    publish(w.site.behavior_tree, TODAY, "讨论方案", 10, 0)
    report = w.tick(clock(10, 15))
    assert report.skipped is not None and report.recorded == 0
    assert w.expectations(TODAY) == []
