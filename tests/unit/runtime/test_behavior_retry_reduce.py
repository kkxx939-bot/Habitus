"""重试收尾里每一支那一次归约的三分法（``runtime.behavior._reduce``，裁定 94 第 9 条；用户 11-10 失败重试三条）：

- sweep 锁被另一进程拿着 → ``Reduction.BUSY``：还没准备好，换代这一眼到此为止（``held``）、不写完成标记、不算失败；
- 归类时模型整个用不了（报告 ``unavailable``、触点抛的 ``ModelUnreachable``、模型客户端原样抛的"整个用不了"）
  → ``Reduction.UNREACHABLE``：这一代跳过结算、计入"结算留到下次"，其余照做、写完成标记；
- 别的异常是 bug：原样抛出，换代报出（重试那一步 ``degraded``）、同样跳过结算。

前一半直接测 ``_reduce``（原 ``test_nightly_line.py`` 里那条删掉后没有接上的守护）；后一半把它接进真实的 ``RetryStage``、
再放进换代，看三种结局各自落到哪一步。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from habitus.behavior.kinds.ids import Lane
from habitus.behavior.reduction import BehaviorReductionBusyError, BehaviorReductionReport
from habitus.behavior.retry import (
    Attempt,
    Reduction,
    RetryBacklogStore,
    RetryBranch,
    RetryCause,
    RetryItem,
    RetryKind,
    RetryStage,
    retry_id,
)
from habitus.model_client import ModelQuotaError, ModelUnreachable
from habitus.runtime.behavior import _reduce, retry_stage_of
from habitus.runtime.generation_result import (
    DEGRADED,
    FALLBACK_SETTLEMENT,
    HELD,
    LOCK_BUSY,
    RETRY,
    RETRY_UNFINISHED,
    SETTLE,
    SKIPPED,
)
from tests.unit.runtime.test_generation import TODAY, generate, outcomes, wired


def _branch(run_once, **extra):  # type: ignore[no-untyped-def]
    return SimpleNamespace(lane=Lane.SESSION, reduction_runner=SimpleNamespace(run_once=run_once), **extra)


def reporting(unavailable: int):  # type: ignore[no-untyped-def]
    async def run_once() -> BehaviorReductionReport:
        return BehaviorReductionReport(0, 0, 0, 0, (), unavailable=unavailable)

    return run_once


def raising(exc: BaseException):  # type: ignore[no-untyped-def]
    async def run_once() -> BehaviorReductionReport:
        raise exc

    return run_once


def test_the_retry_reduction_is_done_busy_unreachable_or_raises_a_bug() -> None:
    assert asyncio.run(_reduce(_branch(reporting(0)))()) is Reduction.DONE  # type: ignore[arg-type]
    assert asyncio.run(_reduce(_branch(reporting(2)))()) is Reduction.UNREACHABLE  # type: ignore[arg-type]
    busy = raising(BehaviorReductionBusyError("sweep lock held"))
    assert asyncio.run(_reduce(_branch(busy))()) is Reduction.BUSY  # type: ignore[arg-type]
    for down in (ModelUnreachable("down"), ModelQuotaError("quota")):
        assert asyncio.run(_reduce(_branch(raising(down)))()) is Reduction.UNREACHABLE  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        asyncio.run(_reduce(_branch(raising(KeyError("verdicts"))))())  # type: ignore[arg-type]


def test_the_retry_stage_of_the_runtime_wires_each_branch_to_its_own_three_way_reduction(tmp_path: Path) -> None:
    """组合根接的就是 ``_reduce``：锁被占交回 ``BUSY``，不是被当成 bug 抛出去。"""

    branch = _branch(
        raising(BehaviorReductionBusyError("sweep lock held")),
        retry_backlog=RetryBacklogStore(tmp_path, max_items=10),
        fusion=object(),
    )
    branch.reduction_runner.failed_chains = object()
    stage = retry_stage_of([branch])  # type: ignore[list-item]
    assert stage is not None
    (wired_branch,) = stage.branches
    assert asyncio.run(wired_branch.reduce()) is Reduction.BUSY


class _Chains:
    """链处理器的替身：放回就算写回了，归约之后问它补回没有都说没有。"""

    async def attempt(self, item: RetryItem) -> Attempt:
        return Attempt.RESTORED

    def recovered(self, item: RetryItem) -> bool:
        return False

    def abandon(self, item: RetryItem) -> None:
        pass


def _stage_with_one_chain(tmp_path: Path, run_once) -> RetryStage:  # type: ignore[no-untyped-def]
    backlog = RetryBacklogStore(tmp_path / "backlog", max_items=10)
    yesterday = TODAY - timedelta(days=1)
    backlog.put(
        RetryItem(
            retry_id(RetryKind.CHAIN, "链一"),
            RetryKind.CHAIN,
            RetryCause.UNAVAILABLE,
            yesterday,
            datetime(2026, 8, 17, 12, 0, tzinfo=UTC),
            {"records": [{"name": "链一"}]},
        )
    )
    return RetryStage([RetryBranch("session", backlog, {RetryKind.CHAIN: _Chains()}, _reduce(_branch(run_once)))])  # type: ignore[arg-type]


def test_a_busy_retry_reduction_holds_the_generation_and_writes_no_mark(tmp_path: Path) -> None:
    order: list[str] = []
    run = wired(tmp_path, order)
    run.retry = _stage_with_one_chain(tmp_path, raising(BehaviorReductionBusyError("sweep lock held")))
    result = generate(run)
    assert result.held == RETRY and result.held_reason == LOCK_BUSY and not result.complete
    assert outcomes(result)[RETRY] == HELD and SETTLE not in outcomes(result)
    assert "结算" not in order and run.marks.finished(TODAY) is None
    assert result.fallbacks[FALLBACK_SETTLEMENT] == 0


@pytest.mark.parametrize(
    "run_once",
    [reporting(1), raising(ModelUnreachable("down")), raising(KeyError("verdicts"))],
    ids=["reported-unavailable", "model-unreachable", "bug"],
)
def test_an_unreachable_or_buggy_retry_reduction_skips_the_settlement_but_writes_the_generation(
    tmp_path: Path, run_once
) -> None:  # type: ignore[no-untyped-def]
    order: list[str] = []
    run = wired(tmp_path, order)
    run.retry = _stage_with_one_chain(tmp_path, run_once)
    result = generate(run)
    assert result.complete and result.held is None
    retry, settle = result.step(RETRY), result.step(SETTLE)
    assert retry is not None and retry.outcome == DEGRADED
    assert settle is not None and settle.outcome == SKIPPED and settle.reason == RETRY_UNFINISHED
    assert "结算" not in order and result.fallbacks[FALLBACK_SETTLEMENT] == 1
    assert run.marks.finished(TODAY) is True
