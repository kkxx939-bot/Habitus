"""换代多进程测试用的一个进程（``test_generation_multiprocess.py`` 起它）。

用法：``python generation_mp_worker.py <root> <mode> <worker_id> <looks> <scene_seconds>``；开跑前等 ``<root>/go`` 出现（几个进程
同时起跑），每个事件追加一行 JSON 到 ``<root>/events.log``。

- ``gen``：生产的预测 worker 的一槽（``ForesightWorker.slot()``：持预测根租约 → ``GenerationRunner`` 看一眼；这里不接预测拍）
  + ``SQLiteLockStore``（``<root>/locks.sqlite3``）；阶段 2 第二小步起没有换代租约，"同一天只换一次"靠预测根租约加完成标记。各步是 ``test_generation.wired`` 的
  替身（自己的树在 ``<root>/<worker_id>/``），完成标记共用 ``<root>/generation``；语义树那一步睡 ``scene_seconds`` 秒，让几个进程的
  那一眼重叠、也给"做到一半被杀"留出时间。看 ``looks`` 眼，眼与眼之间隔 0.3 秒。
- ``tick``：生产的 ``ForesightTick``，共用 ``<root>`` 下的行为树、机会表与账本，预测根租约用真实 ``Lease`` + 同一个 SQLite 锁库；
  账本写一条预期前睡 0.5 秒，拉宽两个进程同时写的窗口。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

TTL_SECONDS = 2.0
HEARTBEAT_SECONDS = 0.5


def main(root: Path, mode: str, worker_id: str, looks: int, scene_seconds: float) -> None:
    def log(**row: object) -> None:
        row.update(pid=os.getpid(), wid=worker_id, t=time.time())
        with open(root / "events.log", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def wait_go() -> None:
        while not (root / "go").exists():
            time.sleep(0.01)

    async def gen() -> None:
        from zoneinfo import ZoneInfo

        from habitus.infrastructure.store.sqlite import SQLiteLockStore
        from habitus.runtime.foresight_worker import ForesightWorker
        from habitus.runtime.generation_lease import Lease
        from habitus.runtime.generation_marks import GenerationMarks
        from habitus.runtime.generation_runner import GenerationRunner
        from tests.unit.runtime.test_generation import wired

        order: list[str] = []
        run = wired(root / worker_id, order)
        run.marks = GenerationMarks(root / "generation", artifact_root=root)
        scene_run = run.scene.generate  # 实时化阶段 2（A1）：换代调的是语义树不调模型的那一步（``generate``）

        async def slow_scene(**kwargs):  # type: ignore[no-untyped-def]
            log(ev="scene_start")
            await asyncio.sleep(scene_seconds)
            return await scene_run(**kwargs)

        run.scene.generate = slow_scene  # type: ignore[method-assign]
        runner = GenerationRunner(run, worker_id=worker_id)
        worker = ForesightWorker(
            None,
            generation=runner,
            lease=Lease(
                SQLiteLockStore(root / "locks.sqlite3"),
                "foresight:root:mp",
                ttl_seconds=TTL_SECONDS,
                heartbeat_seconds=HEARTBEAT_SECONDS,
                poll_seconds=0.05,
            ),
            interval_seconds=900.0,
            shutdown_timeout_seconds=5.0,
            zone=ZoneInfo("Asia/Shanghai"),
        )
        wait_go()
        log(ev="go")
        for _ in range(looks):
            before = runner.last_result
            report = await worker.slot()
            result = runner.last_result if runner.last_result is not before else None
            log(
                ev="look",
                got=None if result is None else {"day": result.day, "complete": result.complete},
                skipped=report.skipped,
                error=None if runner.last_error is None else type(runner.last_error).__name__,
            )
            await asyncio.sleep(0.3)

    async def tick() -> None:
        from zoneinfo import ZoneInfo

        from habitus.conditions import LocalCalendar
        from habitus.foresight import NoUnsealed
        from habitus.infrastructure.store.sqlite import SQLiteLockStore
        from habitus.runtime.foresight import ForesightReadSide
        from habitus.runtime.foresight_ledger import ForesightLedgerStore
        from habitus.runtime.foresight_opportunities import OpportunityStore
        from habitus.runtime.foresight_tick import ForesightTick
        from habitus.runtime.generation_lease import Lease
        from tests.unit.runtime.test_foresight_tick import TODAY, TOLERANCE
        from tests.unit.scene.fixtures import SUBJECT, Site, at

        site = Site(root, now=at(TODAY, 3, 0))
        ledger_root = root / "foresight" / "session"
        ledger, opportunities = ForesightLedgerStore(ledger_root), OpportunityStore(ledger_root)
        original = ledger.record_expectation

        def record(item):  # type: ignore[no-untyped-def]
            log(ev="record_start", id=item.expectation_id)
            time.sleep(0.5)
            wrote = original(item)
            log(ev="record_end", id=item.expectation_id, wrote=wrote)
            return wrote

        ledger.record_expectation = record  # type: ignore[method-assign]
        lease = Lease(
            SQLiteLockStore(root / "locks.sqlite3"),
            "foresight:root:mp",
            ttl_seconds=10.0,
            heartbeat_seconds=1.0,
            poll_seconds=0.05,
        )
        run = ForesightTick(
            ledger,
            opportunities,
            ForesightReadSide(site.behavior_tree, SUBJECT, LocalCalendar(), ZoneInfo("Asia/Shanghai")),
            NoUnsealed(),
            clock=lambda: at(TODAY, 10, 15),
            recent=35,
            tolerance=TOLERANCE,
            tick_seconds=900.0,
            lease=lease,
        )
        wait_go()
        log(ev="go")
        report = await run.run_once()
        log(ev="tick", recorded=report.recorded, formed=report.formed, skipped=report.skipped)

    asyncio.run({"gen": gen, "tick": tick}[mode]())
    log(ev="exit")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]))
