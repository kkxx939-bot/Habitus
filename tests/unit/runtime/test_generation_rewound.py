"""换代的第二轮修复（06-第二轮修复清单 G4、G6）：真实 ``GenerationWorker.look()`` 与 ``_generation_state_checks``。

- G4：上一次做的那一代没写成，之后今天这一代已有完成标记——健康面不再挂着那次没写成，``generation_worker`` 的 ``last_error`` 同理；
- G6：往西改时区、当地日期回到已换代的日子之前——不换代、不算失败、不退避（关系检验 0 次），健康面报日期回退，日期追上后自己消掉。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from habitus.runtime.foresight_days import DayReport
from habitus.runtime.foresight_forming import Formed
from habitus.runtime.generation_marks import GenerationMark
from habitus.runtime.health import _generation_state_checks
from tests.unit.runtime.test_generation import NOW, TODAY, wired
from tests.unit.runtime.test_generation_runner import worker_for

LONDON = ZoneInfo("Europe/London")


def names(worker) -> set[str]:  # type: ignore[no-untyped-def]
    return {item.name for item in _generation_state_checks(worker)}


def test_a_failure_is_no_longer_reported_once_todays_generation_has_its_mark(tmp_path) -> None:
    run = wired(tmp_path, [])

    class Stale:
        async def run_once(self):  # type: ignore[no-untyped-def]
            return DayReport(datetime.now().astimezone(), Formed(), "今天这一代的机会表还没有")

    run.days = Stale()  # type: ignore[assignment]
    worker = worker_for(run, tmp_path)
    failed = asyncio.run(worker.look())
    assert failed is not None and not failed.complete
    assert "generation" in names(worker) and worker.last_error is not None
    run.marks.write(GenerationMark(day=TODAY, finished_at=NOW))  # 今天这一代已完成（另一个进程写成的）
    assert asyncio.run(worker.look()) is None
    assert "generation" not in names(worker) and worker.last_error is None


def test_a_local_date_before_the_latest_generation_waits_without_failing_and_is_reported(tmp_path) -> None:
    """04-28 这一代（这里是 TODAY）写成后改成 Europe/London，当地回到前一天：不换代、不进退避、关系检验 0 次；健康面报日期
    回退（带今天与最晚标记的日子）；当地日期追上（TODAY 零点起）不再报。"""

    order: list[str] = []
    clock = [NOW]
    run = wired(tmp_path, order, clock=lambda: clock[0])
    worker = worker_for(run, tmp_path)
    worker._clock = lambda: clock[0]  # type: ignore[method-assign]
    assert asyncio.run(worker.look()).complete  # type: ignore[union-attr]
    order.clear()
    run.zone = LONDON  # 改时区重启：+08 的 03:00 是 London 前一天 20:00
    yesterday = TODAY - timedelta(days=1)
    assert run.today() == yesterday
    for _ in range(3):
        assert asyncio.run(worker.look()) is None
        clock[0] += timedelta(minutes=5)
    assert order == [] and worker.retry_at is None and worker.last_result is not None and worker.last_result.complete
    check = {item.name: item for item in _generation_state_checks(worker)}["generation_rewound"]
    assert str(yesterday) in check.detail and str(TODAY) in check.detail and check.critical is False
    clock[0] = datetime.combine(TODAY, datetime.min.time(), tzinfo=LONDON) + timedelta(minutes=5)
    assert asyncio.run(worker.look()) is None and order == []  # 当地 TODAY：这一代已经做完
    assert "generation_rewound" not in names(worker)
