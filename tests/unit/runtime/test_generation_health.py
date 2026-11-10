"""换代的健康面（``runtime.health._generation_state_checks``；实时化方案第 3、10 节，用户 11-10 失败重试三条）：

- ``generation``：必须成的那一步没成（今天没有这一代），带下一次重试的时刻；锁被占（还没准备好）不报；
- ``generation_fallbacks``：最近写成的那一代走了退路，逐步列出；
- ``generation_errors``：重试收尾之后仍保持「错误·看不到」的；
- ``generation_stuck``：定稿日不连续、连着配置的眼数卡在同一天；
- ``generation_marks``：最新的完成标记读不了。

都走真实的 ``GenerationWorker.look()``（各步是 ``test_generation.wired`` 的替身）。原来夜线健康面有 3 条测试，换代之后没有接上。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from habitus.behavior.reduction import BehaviorReductionBusyError
from habitus.behavior.retry import RetryReport
from habitus.runtime.foresight_days import DayReport
from habitus.runtime.foresight_forming import Formed
from habitus.runtime.health import RuntimeHealthCheck, RuntimeHealthStatus, _generation_state_checks
from tests.unit.runtime.test_generation import NOW, TODAY, StubRetry, wired
from tests.unit.runtime.test_generation_worker import worker_for


def checks_of(worker) -> dict[str, RuntimeHealthCheck]:  # type: ignore[no-untyped-def]
    found = {item.name: item for item in _generation_state_checks(worker)}
    assert all(item.critical is False for item in found.values())  # 都不挡白天的递送
    return found


def test_a_healthy_generation_reports_nothing(tmp_path) -> None:
    worker = worker_for(wired(tmp_path, []), tmp_path)
    result = asyncio.run(worker.look())
    assert result is not None and result.complete
    assert checks_of(worker) == {}


def test_a_failed_must_step_is_reported_with_the_day_the_step_and_the_next_retry(tmp_path) -> None:
    order: list[str] = []
    clock = [NOW]
    run = wired(tmp_path, order, clock=lambda: clock[0])

    class Stale:
        async def run_once(self):  # type: ignore[no-untyped-def]
            return DayReport(datetime.now().astimezone(), Formed(), "今天这一代的机会表还没有")

    run.days = Stale()  # type: ignore[assignment]
    worker = worker_for(run, tmp_path)
    worker._clock = lambda: clock[0]  # type: ignore[method-assign]
    asyncio.run(worker.look())
    check = checks_of(worker)["generation"]
    assert check.status is RuntimeHealthStatus.DEGRADED
    assert f"{TODAY} not written: days" in check.detail
    assert f"next retry at {(NOW + timedelta(seconds=300)).isoformat()}" in check.detail
    clock[0] = NOW + timedelta(seconds=301)
    asyncio.run(worker.look())  # 第二次也没成：下一次重试的时刻跟着退避往后挪
    assert f"next retry at {(clock[0] + timedelta(seconds=600)).isoformat()}" in checks_of(worker)["generation"].detail


def test_a_generation_held_by_a_busy_lock_is_not_reported_as_a_failure(tmp_path) -> None:
    run = wired(tmp_path, [])

    async def locked(_night) -> None:  # type: ignore[no-untyped-def]
        raise BehaviorReductionBusyError("sweep lock held")

    run.vocabulary_jobs = (locked,)
    worker = worker_for(run, tmp_path)
    held = asyncio.run(worker.look())
    assert held is not None and held.held is not None
    assert "generation" not in checks_of(worker)


def test_fallbacks_and_errors_kept_after_the_retry_are_reported_from_the_latest_mark(tmp_path) -> None:
    order: list[str] = []
    run = wired(tmp_path, order)
    run.retry = StubRetry(RetryReport(taken=3, deterministic=1, refused=1), order=order)  # type: ignore[assignment]

    def broken() -> None:
        raise OSError("holidays")

    run.refresh_calendar = broken
    worker = worker_for(run, tmp_path)
    result = asyncio.run(worker.look())
    assert result is not None and result.complete
    found = checks_of(worker)
    assert set(found) == {"generation_fallbacks", "generation_errors"}
    assert found["generation_fallbacks"].detail.startswith(f"{TODAY} ")
    assert "calendar: 重读放假数据出错: OSError" in found["generation_fallbacks"].detail
    assert found["generation_errors"].detail == "错误·看不到：处理不了 1，内容安全 1，格式不合 0"


def test_a_stuck_closed_day_is_reported_once_it_has_lasted_the_configured_looks(tmp_path) -> None:
    yesterday = TODAY - timedelta(days=1)
    run = wired(tmp_path, [], closed_days=lambda: (yesterday - timedelta(days=2), yesterday))
    worker = worker_for(run, tmp_path, stuck_ticks=3)
    for _ in range(2):
        asyncio.run(worker.look())
        assert "generation_stuck" not in checks_of(worker)
    asyncio.run(worker.look())
    check = checks_of(worker)["generation_stuck"]
    assert check.status is RuntimeHealthStatus.DEGRADED and str(yesterday - timedelta(days=1)) in check.detail


def test_an_unreadable_latest_mark_is_reported(tmp_path) -> None:
    run = wired(tmp_path, [])
    worker = worker_for(run, tmp_path)
    run.marks.path(TODAY).parent.mkdir(parents=True, exist_ok=True)
    run.marks.path(TODAY).write_text('{"schema": "generation_mark_v1"', encoding="utf-8")
    check = checks_of(worker)["generation_marks"]
    assert check.status is RuntimeHealthStatus.DEGRADED and check.detail == "unreadable:GenerationMarkError"
