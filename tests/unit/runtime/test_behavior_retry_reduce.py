"""重试堆每一支那一次归约的三分法（``runtime.behavior._reduce``，裁定 94 第 9 条；用户 11-10 失败重试三条）：

- sweep 锁被另一进程拿着 → ``Reduction.BUSY``：还没准备好，这一支记成在等，下一次直接归约；
- 归类时模型整个用不了（报告 ``unavailable``、触点抛的 ``ModelUnreachable``、模型客户端原样抛的"整个用不了"）
  → ``Reduction.UNREACHABLE``：没补回的项留着，报告标 ``unreachable``（模型队列据它整体退避）；
- 别的异常是 bug：原样抛出。

前一半直接测 ``_reduce``；后一半把它接进真实的 ``RetryStage``。实时化阶段 2（A1）起那一次归约发生在模型队列的就近试里
（``RetryStage.run_once``），不在换代里——换代的重试收尾（``close``）不试、不跑归约，所以后一半从"放进换代看落到哪一步"改成
两段各看各的：就近试的三种结局，与换代里的重试收尾不跑归约、照常结算。
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
from habitus.model_client import ModelQuotaError, ModelTransportError, ModelUnreachable
from habitus.runtime.behavior import _reduce, retry_stage_of
from habitus.runtime.generation_result import (
    DEGRADED,
    DONE,
    FALLBACK_SETTLEMENT,
    RETRY,
    RETRY_REDUCING,
    RETRY_UNFINISHED,
    SETTLE,
    SKIPPED,
)
from tests.unit.runtime.test_generation import TODAY, generate, wired


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
    # 归约自己吞掉的"模型用不了"（报告里记着）与抛出来的同一个结局：模型队列按一次模型调用失败算
    assert asyncio.run(_reduce(_branch(reporting(2)))()) is Reduction.UNREACHABLE  # type: ignore[arg-type]
    busy = raising(BehaviorReductionBusyError("sweep lock held"))
    assert asyncio.run(_reduce(_branch(busy))()) is Reduction.BUSY  # type: ignore[arg-type]
    for down in (ModelUnreachable("down"), ModelQuotaError("quota"), ModelTransportError("read timed out")):
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


def test_a_busy_retry_reduction_keeps_the_branch_waiting_and_the_item_in_the_backlog(tmp_path: Path) -> None:
    """就近试里归约的锁被占：这一支在等、项留着（原来是换代 ``held``、不写完成标记——归约不在换代里了）。"""

    stage = _stage_with_one_chain(tmp_path, raising(BehaviorReductionBusyError("sweep lock held")))
    report = asyncio.run(stage.run_once(through=TODAY))
    assert report.waiting == ("session",) and not report.ready and not report.unreachable
    assert stage.branches[0].backlog.count() == 1


@pytest.mark.parametrize(
    ("run_once", "unreachable"),
    [(reporting(1), True), (raising(ModelUnreachable("down")), True), (raising(KeyError("verdicts")), False)],
    ids=["reported-unavailable", "model-unreachable", "bug"],
)
def test_an_unreachable_retry_reduction_keeps_the_item_and_a_bug_is_raised(
    tmp_path: Path,
    run_once,  # type: ignore[no-untyped-def]
    unreachable: bool,
) -> None:
    """就近试里归约碰到模型整个用不了：报告标 ``unreachable``、项留着（那一次不算试过）；bug 原样抛、写回的项留着等下一次。"""

    stage = _stage_with_one_chain(tmp_path, run_once)
    if unreachable:
        report = asyncio.run(stage.run_once(through=TODAY))
        assert report.unreachable and report.reduction_unreachable and report.unavailable == 1
    else:
        with pytest.raises(KeyError):
            asyncio.run(stage.run_once(through=TODAY))
    assert stage.branches[0].backlog.count() == 1


def test_the_generations_retry_closing_runs_no_reduction_and_settles(tmp_path: Path) -> None:
    """A1：换代里的重试收尾只丢弃、撤回、计数——不跑归约（归约里有模型调用），所以不会 ``held``、不会因模型调不通跳过结算：
    昨天还在堆里的那一项丢弃，结算照做，这一代写成。"""

    order: list[str] = []
    run = wired(tmp_path, order)
    stage = _stage_with_one_chain(tmp_path, raising(AssertionError("the closing must not run a reduction")))
    run.retry = stage  # type: ignore[assignment]
    result = generate(run)
    assert result.complete and result.held is None and run.marks.finished(TODAY) is True
    retry, settle = result.step(RETRY), result.step(SETTLE)
    assert retry is not None and retry.outcome == DONE and retry.report["retry"]["unavailable"] == 1  # type: ignore[index]
    assert settle is not None and settle.outcome == DONE and "结算" in order
    assert result.fallbacks[FALLBACK_SETTLEMENT] == 0 and stage.branches[0].backlog.count() == 0


def test_the_generation_defers_the_settlement_while_a_written_retry_is_still_to_be_reduced(tmp_path: Path) -> None:
    """修复轮 G6（裁定 68 第 3 条）：就近重试写回了昨天的判断、归约的锁被占着——换代的重试收尾不动那一项，这一代的结算
    留到下次（``settlement_deferred``），其余各步照做、照写完成标记：归约里要调模型，换代不等它。"""

    order: list[str] = []
    run = wired(tmp_path, order)

    async def run_once() -> BehaviorReductionReport:
        raise BehaviorReductionBusyError("sweep lock held")

    stage = _stage_with_one_chain(tmp_path, run_once)
    run.retry = stage  # type: ignore[assignment]
    assert not asyncio.run(stage.run_once(through=TODAY)).ready  # 队列就近试：写回了，归约没做成
    result = generate(run)
    assert result.complete and result.held is None and run.marks.finished(TODAY) is True
    retry, settle = result.step(RETRY), result.step(SETTLE)
    assert retry is not None and retry.outcome == DEGRADED and retry.reason == RETRY_REDUCING
    assert settle is not None and settle.outcome == SKIPPED and settle.reason == RETRY_UNFINISHED
    assert "结算" not in order and result.fallbacks[FALLBACK_SETTLEMENT] == 1
    assert stage.branches[0].backlog.count() == 1 and stage.branches[0].backlog.through() is None  # 重试收尾没动
