"""Runtime 生命周期的编排：初始化各存储、按顺序起停各 Worker、关闭资源。

状态机（CREATED/READY/RUNNING/CLOSING/CLOSED 的转换与门）、共同存储根的创建和启动/初始化的观测留在 Runtime 门面；
这里只回答"起停哪些、什么顺序、哪些失败不阻断记忆主链"。
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

from habitus.config import HabitusConfig
from habitus.conversation import ConversationSourceConsumer, ConversationSourceRecoveryResult
from habitus.foundation.observability import ObservationStatus
from habitus.infrastructure.observability import MetricRegistry
from habitus.runtime.components import RuntimeComponents
from habitus.runtime.runtime_jobs import Observe


class RuntimeLifecycle:
    """``observe`` 是 Runtime 的观测出口；本类不持有也不改变 Runtime 状态。"""

    def __init__(self, config: HabitusConfig, components: RuntimeComponents, *, observe: Observe) -> None:
        self.config = config
        self.components = components
        self._observe = observe

    def initialize_stores(self) -> tuple[Path, tuple[str, ...]]:
        """初始化树与各存储；无任务时恢复孤儿事务，有任务时留给租约持有者恢复。返回（记忆根, 恢复的事务）。"""

        self.components.infrastructure.initialize()
        memory_root = self.components.memory.tree.initialize()
        self.components.workflow.jobs.initialize()
        self.components.memory.lifecycle.initialize()
        self.components.conversation.summary_use.initialize()
        self.components.workflow.lifecycle.retirement_store.initialize()
        if self.components.behavior is not None:
            # 行为树静态目录；幂等，重复初始化零写入。
            # 此处失败**允许**阻断整个 Runtime（对"行为失败不阻断"哲学的边界取舍）：
            # 同一 storage root 下建不了目录，几乎必然意味着记忆侧同样不可用；
            # 静默跳过反而会让 RUNNING 状态掩盖一个坏掉的存储根。
            for branch in self.components.behavior:
                branch.tree.initialize()
        oldest_job = self.components.workflow.jobs.oldest_uncommitted()
        recovered = self.components.workflow.runner.transaction_recovery.recover_pending() if oldest_job is None else ()
        return memory_root, recovered

    async def start(self, started: float) -> None:
        """确认两套向量索引 ready，恢复交付，依次启动 Job Worker、生命周期维护 Worker 与可选的后台循环。"""

        await self.components.memory.vector_index.ensure_ready()
        await self.components.conversation.summary_vector_index.ensure_ready()
        # 恢复失败不阻断启动（裁定 60 第 6 条），只降级出事的那个地址：有序的 Memory 对同地址后面的源报"等前序"或
        # "前序已坏"，别的地址照常；可重试的由周期恢复按退避再试，放弃了的等人工修复。失败都计入 gauge。
        # 整轮恢复本身出错（读不了账）同样不挡启动：记一条失败事件，交给周期恢复再来。
        try:
            source_recoveries = await self.components.conversation.source_recovery.recover_pending()
        except Exception as exc:  # noqa: BLE001 - 恢复失败不阻断启动
            self._observe(
                "runtime", "source_recovery", ObservationStatus.FAILURE, started, {"error_type": type(exc).__name__}
            )
        else:
            publish_source_delivery_gauges(self.components.infrastructure.observability, source_recoveries)
            self._observe_degraded_source_addresses(source_recoveries, started)
        await self.components.workflow.worker.start()
        try:
            await self.components.workflow.lifecycle_worker.start()
        except BaseException:
            try:
                await self.components.workflow.lifecycle_worker.stop()
            finally:
                await self.components.workflow.worker.stop()
            raise
        # 行为管线（可选启用）：启动失败不阻断 Runtime——沿既有"Behavior 失败不阻断"
        # 哲学，失败经观测记录，行为侧下次启动自愈（存储都是耐久幂等的）。
        if self.components.behavior is not None:
            # 两支各起各的：一支起不来不连累另一支。
            for branch in self.components.behavior:
                await self._start_optional(
                    "behavior_start", branch.reduction_worker.start, started, {"lane": branch.lane.value}
                )
        # 整晚那一趟同一哲学：它只读行为树、只写派生的存储，起不来不影响任何在线路径。
        nightly = self.components.nightly
        if nightly is not None:
            await self._start_optional("nightly_start", nightly.start, started)
        # 预测层每槽一拍，只读两棵派生树与判断存储、只调模型，同一哲学。
        foresight = self.components.foresight
        if foresight is not None and foresight.worker is not None:
            await self._start_optional("foresight_start", foresight.worker.start, started)

    async def _start_optional(
        self,
        operation: str,
        start: Callable[[], Awaitable[None]],
        started: float,
        attributes: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        """起一个可选的后台循环（行为侧、夜批、预测层）：失败只记一条观测，不阻断记忆主链。"""

        try:
            await start()
        except Exception as exc:  # noqa: BLE001 - 行为侧、夜批、预测层失败不阻断记忆主链
            self._observe(
                "runtime",
                operation,
                ObservationStatus.FAILURE,
                started,
                {"error_type": type(exc).__name__, **(attributes or {})},
            )

    async def stop(self) -> None:
        """停止认领并有界排空 Worker（Runtime 保留，可再次启动）。"""

        await self.stop_background_workers()
        try:
            await self.components.workflow.lifecycle_worker.stop()
        finally:
            try:
                await self.components.conversation.source_coordinator.wait_for_idle(
                    self.config.conversation.source.shutdown_timeout_seconds
                )
            finally:
                await self.components.workflow.worker.stop()

    async def stop_for_close(self) -> None:
        """关闭前停全部 Worker；Source Consumer 的有界排空由 Runtime 的 close 自己做（超时要显式失败）。"""

        # 行为 Worker 与主链 Worker 对称停机：``async with`` 的退出走的是 close 不是
        # stop——漏掉这里会让融合/归约循环带着已关闭的模型客户端继续跑（评审实测泄漏）。
        await self.stop_background_workers()
        try:
            await self.components.workflow.lifecycle_worker.stop()
        finally:
            await self.components.workflow.worker.stop()

    async def stop_background_workers(self) -> None:
        """停行为侧两个循环与预测夜批；失败不阻断记忆主链的停机（与启动侧同一哲学）。

        名字不叫 behavior：启动侧是分开的两块，停机侧合成一个函数是为了保证顺序（预测先停，
        它读的行为树才停），但既然管着两个域，就不该顶着一个域的名字。
        """

        # 预测层先停（它读机会表与行为树），然后整晚那一趟，最后行为侧。
        foresight = self.components.foresight
        if foresight is not None and foresight.worker is not None:
            try:
                await foresight.worker.stop()
            except Exception as exc:  # noqa: BLE001 - 预测层停机失败不阻断主链
                self._observe(
                    "runtime",
                    "foresight_stop",
                    ObservationStatus.FAILURE,
                    time.monotonic(),
                    {"error_type": type(exc).__name__},
                )
        if self.components.nightly is not None:
            try:
                await self.components.nightly.stop()
            except Exception as exc:  # noqa: BLE001 - 夜批停机失败不阻断主链
                self._observe(
                    "runtime",
                    "nightly_stop",
                    ObservationStatus.FAILURE,
                    time.monotonic(),
                    {"error_type": type(exc).__name__},
                )
        behavior = self.components.behavior
        if behavior is None:
            return
        for branch in behavior:
            try:
                await branch.reduction_worker.stop()
            except Exception as exc:  # noqa: BLE001 - 行为侧停机失败不阻断主链，也不连累另一支
                self._observe(
                    "runtime",
                    "behavior_stop",
                    ObservationStatus.FAILURE,
                    time.monotonic(),
                    {"error_type": type(exc).__name__, "lane": branch.lane.value},
                )

    async def close_resources(self) -> None:
        try:
            await self.components.conversation.summary_vector_index.close()
        finally:
            try:
                await self.components.memory.vector_index.close()
            finally:
                try:
                    await self.components.models.aclose()
                finally:
                    manager = self.components.infrastructure.managed_observability
                    if manager is not None:
                        manager.close()

    def _observe_degraded_source_addresses(
        self,
        results: Sequence[ConversationSourceRecoveryResult],
        started: float,
    ) -> None:
        """启动恢复里 Memory 没恢复成的源落在几个地址上：这些地址降级运行，记一条 DEGRADED 事件。"""

        degraded = {
            (result.envelope.conversation_id, result.envelope.started_on)
            for result in results
            if result.consumer is ConversationSourceConsumer.MEMORY and result.error is not None
        }
        if degraded:
            self._observe(
                "runtime",
                "source_recovery",
                ObservationStatus.DEGRADED,
                started,
                {"degraded_addresses": len(degraded)},
            )


def publish_source_delivery_gauges(
    registry: MetricRegistry,
    results: Sequence[ConversationSourceRecoveryResult],
) -> None:
    """把本轮恢复后仍未完成的 Source Consumer 数量发布为 gauge。

    交付失败只在发生的那一刻产生一条事件，而损坏状态不会自愈：事件一旦
    被漏看，这条来源就再也不会发出任何信号。gauge 表达的是"现在还卡着
    几条"，每次恢复都会刷新，因此漏看一次不等于永远看不到。
    """

    for consumer in ConversationSourceConsumer:
        stuck = sum(1 for result in results if result.consumer is consumer and result.error is not None)
        registry.set_gauge(
            "conversation_source_delivery_stuck",
            stuck,
            labels={"consumer": consumer.value},
        )


__all__ = ["RuntimeLifecycle", "publish_source_delivery_gauges"]
