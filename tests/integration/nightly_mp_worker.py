"""多进程测试用的一个夜线执行器进程（生产的 ``NightlyWorker`` + ``SQLiteLockStore`` + ``NightJobs``，各段换成定时长的替身）。

用法：``python nightly_mp_worker.py <root> <worker_id> <looks> [<段>=<秒> ...]``。每段开始 / 结束、每一眼的结果、异常都追加写进
``<root>/events.log``（一行一条 JSON）。验证轮 #5 的对抗脚本收进仓库（``test_nightly_multiprocess.py`` 起它）。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

from habitus.infrastructure.store.sqlite import SQLiteLockStore
from habitus.runtime.nightly import DONE, StepResult
from habitus.runtime.nightly_jobs import NightJobs
from habitus.runtime.nightly_settings import NightlySettings
from habitus.runtime.nightly_worker import NightlyWorker

NIGHT = date(2026, 4, 28)
TTL_SECONDS = 2.0
HEARTBEAT_SECONDS = 0.5


def main(root: Path, worker_id: str, looks: int, slow: dict[str, float]) -> None:
    def log(**row: object) -> None:
        row.update(pid=os.getpid(), wid=worker_id, t=time.time())
        with open(root / "events.log", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    class Line:
        def tonight(self) -> date:
            return NIGHT

        def ready(self) -> date:
            return NIGHT

        def clean_leftovers(self) -> int:
            return 0

        async def run_step(self, step: str, night: date, **_options: object) -> StepResult:
            log(ev="start", step=step)
            await asyncio.sleep(slow.get(step, 0.1))
            log(ev="end", step=step)
            return StepResult(step, DONE, None, {"by": worker_id})

    async def run() -> None:
        worker = NightlyWorker(
            Line(),  # type: ignore[arg-type]
            jobs=NightJobs(root / "jobs", artifact_root=root, kept_nights=366),
            lock_store=SQLiteLockStore(root / "locks.sqlite3"),
            lease_key="scene:nightly:mp",
            shutdown_timeout_seconds=5.0,
            settings=NightlySettings(
                lease_ttl_seconds=TTL_SECONDS, heartbeat_seconds=HEARTBEAT_SECONDS, poll_seconds=0.2
            ),
            worker_id=worker_id,
        )
        log(ev="boot")
        for _ in range(looks):
            try:
                ledger = await worker.look()
                got = None if ledger is None else {job.step: [job.state, job.claim_generation] for job in ledger.jobs}
                log(ev="look", got=got)
                if ledger is not None and ledger.complete:
                    break
            except BaseException as exc:  # noqa: BLE001 - 测试进程：把异常记进事件日志
                log(ev="error", error=f"{type(exc).__name__}: {exc}"[:300])
            await asyncio.sleep(0.2)
        log(ev="exit")

    asyncio.run(run())


if __name__ == "__main__":
    main(
        Path(sys.argv[1]),
        sys.argv[2],
        int(sys.argv[3]),
        {key: float(value) for key, value in (item.split("=") for item in sys.argv[4:])},
    )
