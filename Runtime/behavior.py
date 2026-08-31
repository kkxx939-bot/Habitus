"""行为管线在组合根的组装与两个常驻 Worker。

行为侧是可选启用的：``config.behavior.primary_subject`` 留空（上游感知未接入）时组合根完全
跳过本模块，Runtime 与既有行为零差异。启用后接成的整条链：

    观测投递（Runtime 正门）→ 观测存储 → 入队扫描 → 融合作业（严格串行，LLM）
    → 判断/回执存储 → 归约 sweep（封口 → staged → 行为树 + kinds + L0/L1）

模型路由与 memory 共用同一个 structured chat client（用户裁定）；上下文窗口的数值由这里从
``behavior/fusion/config.py`` 的唯一出处解析后，**同一份**喂给融合与归约两处，并由
``BehaviorRuntimeComponents.__post_init__`` 结构性钉死（BHV-FUSION-003）。锁沿组合根现状共享
（SQLite 锁库；取舍与观测盲区记录在 ``behavior/reduction/runner.py`` 的 sweep 锁 docstring）。

与 MemoryWorker 的形态分叉（刻意，勿"补齐"）：行为作业的失败结算、重试退避与 FAILED 封锁全部
耐久在融合作业存储里，归约则是无队列的定拍 sweep——两个 Worker 因此不复刻 MemoryWorker 的
状态机与 BLOCKED 对外语义，只保留循环存活、租约心跳与可观测事件。
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections import Counter
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime

from behavior.fusion import (
    BehaviorFusionEnqueuer,
    BehaviorFusionJobStore,
    BehaviorFusionReceiptStore,
    BehaviorFusionRunner,
    BehaviorJudgementFuser,
    BehaviorJudgementStore,
)
from behavior.fusion.config import (
    FUSION_CONTEXT_LIMIT,
    FUSION_CONTEXT_LOOKBACK_SECONDS,
    BehaviorFusionConfig,
)
from behavior.fusion.coverage import BehaviorCoverageIndex
from behavior.fusion.enqueue import DEFAULT_QUIET_PERIOD_SECONDS
from behavior.kinds.config import BehaviorKindConfig
from behavior.kinds.rebuild import BehaviorKindRebuildReport
from behavior.kinds.resolver import BehaviorKindResolver
from behavior.kinds.store import BehaviorKindStore
from behavior.kinds.vectors import BehaviorKindVectorStore
from behavior.observation import BehaviorObservationEnvelope, BehaviorObservationStore
from behavior.reduction import (
    DEFAULT_SWEEP_LOCK_TTL_SECONDS,
    BehaviorKindMergeReport,
    BehaviorReductionBusyError,
    BehaviorReductionLedger,
    BehaviorReductionRunner,
)
from behavior.semantic import BehaviorSemanticRefresher, LLMBehaviorOverviewGenerator
from behavior.tree import BehaviorTree
from Config import HabitusConfig
from foundation.observability import ObservationEvent, ObservationStatus, Observer
from infrastructure.store.contracts.lock import LockStore
from infrastructure.store.contracts.path_lock import PathLock
from ModelClient import StructuredChatClient
from ModelClient.embedding import Embedder
from Runtime.resident import ResidentWorker


@dataclass(frozen=True)
class BehaviorRuntimeComponents:
    """行为管线的全部已组装部件。

    ``__post_init__`` 按仓库既有纪律做**实例同一性**校验：融合/归约/Worker 引用的存储必须
    就是组件字段那一份，两个 runner 的封口窗口必须是同一份数值——这些是"自己产物自洽"类
    约束，违反只可能是接线错误。
    """

    observations: BehaviorObservationStore
    judgements: BehaviorJudgementStore
    receipts: BehaviorFusionReceiptStore
    jobs: BehaviorFusionJobStore
    enqueuer: BehaviorFusionEnqueuer
    fusion_runner: BehaviorFusionRunner
    tree: BehaviorTree
    kind_store: BehaviorKindStore
    reduction_runner: BehaviorReductionRunner
    fusion_worker: BehaviorFusionWorker
    reduction_worker: BehaviorReductionWorker

    def __post_init__(self) -> None:
        expected = (
            ("observations", self.observations, BehaviorObservationStore),
            ("judgements", self.judgements, BehaviorJudgementStore),
            ("receipts", self.receipts, BehaviorFusionReceiptStore),
            ("jobs", self.jobs, BehaviorFusionJobStore),
            ("enqueuer", self.enqueuer, BehaviorFusionEnqueuer),
            ("fusion_runner", self.fusion_runner, BehaviorFusionRunner),
            ("tree", self.tree, BehaviorTree),
            ("kind_store", self.kind_store, BehaviorKindStore),
            ("reduction_runner", self.reduction_runner, BehaviorReductionRunner),
            ("fusion_worker", self.fusion_worker, BehaviorFusionWorker),
            ("reduction_worker", self.reduction_worker, BehaviorReductionWorker),
        )
        for name, value, expected_type in expected:
            if not isinstance(value, expected_type):
                raise TypeError(f"behavior components {name} must be {expected_type.__name__}")
        shared = (
            (self.enqueuer.observations, self.observations, "enqueuer observations"),
            (self.enqueuer.jobs, self.jobs, "enqueuer jobs"),
            (self.enqueuer.receipts, self.receipts, "enqueuer receipts"),
            (self.fusion_runner.jobs, self.jobs, "fusion runner jobs"),
            (self.fusion_runner.observations, self.observations, "fusion runner observations"),
            (self.fusion_runner.judgements, self.judgements, "fusion runner judgements"),
            (self.fusion_runner.receipts, self.receipts, "fusion runner receipts"),
            (self.reduction_runner.judgements, self.judgements, "reduction judgements"),
            (self.reduction_runner.observations, self.observations, "reduction observations"),
            (self.reduction_runner.receipts, self.receipts, "reduction receipts"),
            (self.reduction_runner.tree, self.tree, "reduction tree"),
            (self.reduction_runner.kind_store, self.kind_store, "reduction kind store"),
            (self.enqueuer.coverage, self.fusion_runner.coverage, "coverage index (enqueue/fusion)"),
            (self.reduction_runner.coverage, self.fusion_runner.coverage, "coverage index (reduction/fusion)"),
            (self.fusion_worker.runner, self.fusion_runner, "fusion worker runner"),
            (self.fusion_worker.enqueuer, self.enqueuer, "fusion worker enqueuer"),
            (self.reduction_worker.runner, self.reduction_runner, "reduction worker runner"),
        )
        for actual, wanted, label in shared:
            if actual is not wanted:
                raise ValueError(f"behavior components must share one {label} instance")
        if self.kind_store.path != self.tree.root / self.kind_store.path.name:
            raise ValueError("behavior kind registry must live at the tree root")
        vectors = self.reduction_runner.kind_vectors
        if vectors is not None and vectors.path != self.tree.root / vectors.path.name:
            raise ValueError("behavior kind vectors must live at the tree root")
        # BHV-FUSION-003 的结构性保障："融合还能续"与"归约已封口"必须是同一个窗口。
        if (
            self.fusion_runner.context_lookback_seconds
            != self.reduction_runner.context_lookback_seconds
        ):
            raise ValueError(
                "fusion and reduction must share one context lookback window"
            )


class BehaviorFusionWorker(ResidentWorker):
    """驱动融合队列的常驻循环：扫描入队 → 认领 → 心跳护租约 → 执行。

    单条作业的失败由融合 runner 在租约内结算（重试/退避/FAILED 封锁都耐久在作业存储里）；
    这里保证三件事：循环本身不死（异常记观测事件 + last_error、退避一拍）、慢模型调用期间
    租约持续续期（claim/execute 分离正是融合层为此设计的接口）、同步全量扫描不冻结事件循环
    （下沉 ``asyncio.to_thread``；扫描成本随存储增长的根治在 BHV-LIFECYCLE-001 的时间分区）。
    """

    _task_name = "habitus-behavior-fusion"
    _observation_category = "behavior"

    def __init__(
        self,
        enqueuer: BehaviorFusionEnqueuer,
        runner: BehaviorFusionRunner,
        *,
        poll_interval_seconds: float,
        shutdown_timeout_seconds: float,
        observer: Observer | None = None,
    ) -> None:
        super().__init__(
            shutdown_timeout_seconds=shutdown_timeout_seconds, observer=observer
        )
        self.enqueuer = enqueuer
        self.runner = runner
        self.poll_interval_seconds = float(poll_interval_seconds)
        # 进程唯一的 worker 身份：租约接管日志要能区分持有者（对齐 MemoryWorker）。
        self.worker_id = f"habitus-behavior-fusion-{os.getpid()}-{uuid.uuid4().hex[:8]}"

    async def run_once(self) -> bool:
        """扫描入队并执行至多一项作业；返回是否有作业被执行（测试与手动驱动用）。"""

        if self.running:
            raise RuntimeError("manual run_once cannot race the resident worker loop")
        return await self._tick()

    async def _tick(self) -> bool:
        await asyncio.to_thread(self.enqueuer.enqueue_ready)
        lease = await asyncio.to_thread(self.runner.claim, self.worker_id)
        if lease is None:
            return False
        await self._execute_with_heartbeat(lease)
        return True

    async def _execute_with_heartbeat(self, lease) -> None:
        """执行作业并在模型调用期间独立续约——租约 TTL(300s) 小于慢模型调用的可能时长，
        不续约的下场是评审推演过的：租约被接管、每轮白烧一次模型调用直至队首 FAILED。"""

        interval = max(self.runner.jobs.config.lease_ttl_seconds / 3.0, 1.0)
        stop_beat = asyncio.Event()

        async def beat() -> None:
            while not stop_beat.is_set():
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop_beat.wait(), timeout=interval)
                if stop_beat.is_set():
                    return
                await asyncio.to_thread(self.runner.jobs.renew, lease)

        heartbeat = asyncio.create_task(beat(), name=f"{self._task_name}-heartbeat")
        try:
            result = await self.runner.execute(lease)
        finally:
            stop_beat.set()
            heartbeat.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await heartbeat
        if result.degradations:
            # 装配层的降级（去重/剔名/丢边）是信号不是语义：按类计数进可观测面板，
            # 让"模型记账疏漏的频率"随真实数据可见（BHV-REALDATA-001 的量化依据）。
            counts = Counter(note.split(" ", 1)[0] for note in result.degradations)
            self._observe(
                "fusion_degradations",
                ObservationStatus.SUCCESS,
                {"job": result.job.job_id[:12], **{kind: count for kind, count in counts.items()}},
            )
        if result.fused:
            # 无归属占比："允许模型不产出"的出口用得多不多——压制产出的唯一量化告警。
            receipt = result.receipt
            self._observe(
                "fusion_unowned",
                ObservationStatus.SUCCESS,
                {
                    "job": result.job.job_id[:12],
                    "unowned": len(receipt.unowned_observation_ids),
                    "observations": len(receipt.observation_ids),
                    "ratio": round(receipt.unowned_ratio, 3),
                },
            )

    async def _run_loop(self) -> None:
        while not self._stop_requested.is_set():
            started = time.monotonic()
            try:
                worked = await self._tick()
            except Exception as exc:  # noqa: BLE001 - 常驻循环必须活过基础设施抖动
                self.last_error = exc
                self._observe(
                    "fusion_tick",
                    ObservationStatus.FAILURE,
                    {"error_type": type(exc).__name__},
                    started=started,
                )
                worked = False
            else:
                self._succeeded()
            if worked:
                continue
            await self._wait(self.poll_interval_seconds)


class BehaviorReductionWorker(ResidentWorker):
    """归约 sweep 的常驻节拍（用户裁定 5 分钟一轮）；正确性全在 runner，这里只管节奏。

    sweep 内部的同步 IO 会短暂占用事件循环——量级受 BHV-LIFECYCLE-001 的全量扫描欠账支配，
    节拍是 5 分钟一次，随时间分区改造一并根治。
    """

    _task_name = "habitus-behavior-reduction"
    _observation_category = "behavior"

    def __init__(
        self,
        runner: BehaviorReductionRunner,
        *,
        interval_seconds: float,
        shutdown_timeout_seconds: float,
        observer: Observer | None = None,
    ) -> None:
        super().__init__(
            shutdown_timeout_seconds=shutdown_timeout_seconds, observer=observer
        )
        self.runner = runner
        self.interval_seconds = float(interval_seconds)

    async def _run_loop(self) -> None:
        while not self._stop_requested.is_set():
            started = time.monotonic()
            try:
                await self.runner.run_once()
            except BehaviorReductionBusyError:
                # sweep 锁被另一持有者占用：多实例场景的正常让路，跳过本拍即可。
                self._observe(
                    "reduction_sweep", ObservationStatus.SUCCESS, {"skipped": "lock_busy"}
                )
            except Exception as exc:  # noqa: BLE001 - 常驻循环必须活过基础设施抖动
                self.last_error = exc
                self._observe(
                    "reduction_sweep",
                    ObservationStatus.FAILURE,
                    {"error_type": type(exc).__name__},
                    started=started,
                )
            else:
                self._succeeded()
            await self._wait(self.interval_seconds)


def build_behavior_components(
    config: HabitusConfig,
    *,
    structured_chat: StructuredChatClient,
    lock_store: LockStore,
    path_lock: PathLock,
    observer: Observer | None = None,
    clock: Callable[[], datetime] | None = None,
    embedder: Embedder | None = None,
) -> BehaviorRuntimeComponents | None:
    """组装行为管线；``primary_subject`` 未配置时返回 None（行为侧未启用）。

    无存储写入、无模型请求——与 ``build_runtime`` 同一纪律。全部存储传**同一个行为根**
    （各存储按自己的契约派生子路径——传子目录会造成目录二次嵌套与锁键漂移，评审实测过）。
    """

    behavior_config = config.behavior
    if not behavior_config.enabled:
        return None
    root = config.behavior_root
    context_limit = (
        behavior_config.context_limit
        if behavior_config.context_limit is not None
        else FUSION_CONTEXT_LIMIT
    )
    context_lookback = (
        float(behavior_config.context_lookback_seconds)
        if behavior_config.context_lookback_seconds is not None
        else FUSION_CONTEXT_LOOKBACK_SECONDS
    )
    # 自洽校验：封口视界短于入队静默期时，链总在续篇有机会被融合之前就机械封口——
    # 每个跨交付的行为必然裂成互不相认的 occurrence。这是我们自己两个参数的矛盾，硬拒。
    if context_lookback <= DEFAULT_QUIET_PERIOD_SECONDS:
        raise ValueError(
            "behavior.context_lookback_seconds must exceed the fusion quiet period "
            f"({DEFAULT_QUIET_PERIOD_SECONDS:g}s); a shorter window seals every chain "
            "before its continuation can even be fused"
        )

    observations = BehaviorObservationStore(root)
    judgements = BehaviorJudgementStore(root)
    receipts = BehaviorFusionReceiptStore(root)
    # 一份覆盖索引三处共用：融合写、入队读、归约读——三者若各自实例化会按不同窗口口径回答
    # "处理过没有"。
    coverage = BehaviorCoverageIndex(root, window_days=behavior_config.coverage_window_days)
    jobs = BehaviorFusionJobStore(root, path_lock, clock=clock)
    # 段容量随配置进来，切段与融合两处必须是同一个数（切出 60 条的段、融合却按 512 校验就是分叉）。
    fusion_config = BehaviorFusionConfig(
        max_fragments_per_segment=behavior_config.max_fragments_per_segment
    )
    enqueuer = BehaviorFusionEnqueuer(
        observations, jobs, receipts, clock=clock, coverage=coverage, config=fusion_config
    )
    fusion_runner = BehaviorFusionRunner(
        jobs,
        observations,
        BehaviorJudgementFuser(structured_chat, config=fusion_config),
        judgements,
        receipts,
        primary_subject=behavior_config.primary_subject,
        context_limit=context_limit,
        context_lookback_seconds=context_lookback,
        coverage=coverage,
    )

    tree = BehaviorTree(root / "tree")
    # 词表参数从唯一的 Config 边界进来；embedder 只做候选召回（BHV-KINDS-002），没有它就退字面重合。
    kind_config = BehaviorKindConfig(**behavior_config.kinds_overrides())
    kind_store = BehaviorKindStore(tree.root, config=kind_config)
    # 旁册的身份键只从配置取（provider/model/dimension 一处出处）；换任一项旁册作废重算——它是派生物。
    embedding = config.models.embedding
    kind_vectors = (
        BehaviorKindVectorStore(
            tree.root,
            model=f"{embedding.route.provider}/{embedding.route.model}",
            dimension=embedding.dimension,
        )
        if embedder is not None
        else None
    )
    reduction_runner = BehaviorReductionRunner(
        judgements=judgements,
        observations=observations,
        receipts=receipts,
        tree=tree,
        lock_store=lock_store,
        kind_store=kind_store,
        kind_resolver=BehaviorKindResolver(structured_chat, config=kind_config, embedder=embedder),
        kind_vectors=kind_vectors,
        ledger=BehaviorReductionLedger(root / "reduction"),
        semantic_refresher=BehaviorSemanticRefresher(
            tree, LLMBehaviorOverviewGenerator(structured_chat)
        ),
        clock=clock,
        context_lookback_seconds=context_lookback,
        coverage=coverage,
        sweep_lock_ttl_seconds=(
            behavior_config.reduction_sweep_lock_ttl_seconds
            if behavior_config.reduction_sweep_lock_ttl_seconds is not None
            else DEFAULT_SWEEP_LOCK_TTL_SECONDS
        ),
    )
    return BehaviorRuntimeComponents(
        observations=observations,
        judgements=judgements,
        receipts=receipts,
        jobs=jobs,
        enqueuer=enqueuer,
        fusion_runner=fusion_runner,
        tree=tree,
        kind_store=kind_store,
        reduction_runner=reduction_runner,
        fusion_worker=BehaviorFusionWorker(
            enqueuer,
            fusion_runner,
            poll_interval_seconds=behavior_config.fusion_poll_interval_seconds,
            shutdown_timeout_seconds=behavior_config.worker_shutdown_timeout_seconds,
            observer=observer,
        ),
        reduction_worker=BehaviorReductionWorker(
            reduction_runner,
            interval_seconds=behavior_config.reduction_sweep_interval_seconds,
            shutdown_timeout_seconds=behavior_config.worker_shutdown_timeout_seconds,
            observer=observer,
        ),
    )


def deliver_observations(
    components: BehaviorRuntimeComponents, envelope: BehaviorObservationEnvelope
) -> str:
    """观测投递的正门：入库、唤醒融合循环；返回交付身份（幂等——同身份同内容重复投递无害，
    同身份异内容 fail-closed）。"""

    stored = components.observations.put(envelope)
    components.fusion_worker.wake()
    return stored.source_id


async def merge_behavior_kinds(
    components: BehaviorRuntimeComponents,
    source: str,
    target: str,
    *,
    observer: Observer | None = None,
) -> BehaviorKindMergeReport:
    """词表合并的正门（BHV-KINDS-002 方案⑤的执行动作）：``source`` 并入 ``target``，树上旧 token 重打。

    判定由离线整理交模型做，这里只执行已定的合并。持 sweep 锁，归约 Worker 在此期间让路
    （lock_busy 跳拍）；重跑幂等。
    """

    started = time.monotonic()
    report = await components.reduction_runner.merge_kinds(source, target)
    _record(
        observer,
        "behavior_kind_merge",
        {"source": source, "target": target, "restamped": report.restamped, "days": len(report.days)},
        started,
    )
    return report


async def rebuild_behavior_kinds(
    components: BehaviorRuntimeComponents, *, observer: Observer | None = None
) -> BehaviorKindRebuildReport:
    """词表重建的正门：按树补齐 + 账重算 + 向量补算，零模型调用；v1 词表迁移与账自愈走这里。"""

    started = time.monotonic()
    report = await components.reduction_runner.rebuild_kinds()
    _record(
        observer,
        "behavior_kind_rebuild",
        {"occurrences": report.occurrences, "kinds": report.kinds, "signals": len(report.signals)},
        started,
    )
    return report


def _record(
    observer: Observer | None,
    operation: str,
    attributes: dict[str, str | int | float | bool],
    started: float,
) -> None:
    """可观测事件与 Worker 同一形态（``ResidentWorker._observe``）；观测失败不影响业务。"""

    if observer is None:
        return
    try:
        observer.record(
            ObservationEvent(
                category="behavior",
                operation=operation,
                status=ObservationStatus.SUCCESS,
                duration_seconds=max(0.0, time.monotonic() - started),
                attributes=attributes,
            )
        )
    except Exception:  # noqa: BLE001 - 观测失败不许影响运维动作
        pass


__all__ = [
    "BehaviorFusionWorker",
    "BehaviorReductionWorker",
    "BehaviorRuntimeComponents",
    "build_behavior_components",
    "deliver_observations",
    "merge_behavior_kinds",
    "rebuild_behavior_kinds",
]
