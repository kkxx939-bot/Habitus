"""行为管线在组合根的组装与常驻的归约 Worker。

行为侧是可选启用的：``config.behavior.primary_subject`` 留空时组合根完全跳过本模块，Runtime 与既有
行为零差异。启用后按 lane 各接一整条、互不相通的链（两条 lane 是行为树的两个分支，各在 ``behavior/<lane>/`` 下）：

    会话那一支：会话源的交付 → ``runtime/session_lane.py`` 的桥 → 一轮一条判断（LLM 描述）
    物理那一支：上游的一份交付（Runtime 正门）→ 一段一条判断（不调模型）
    每一支各自：判断 / 回执存储 → 归约 sweep（封口 → 词表归类 → 行为树 + L0/L1 → 定稿日记账）
    → 整晚那一趟（会话那一支；见 runtime/nightly.py）

两支不共用任何存储、词表、归约或 Worker：一支上的记录、空白、词表的类，另一支看不到（裁定 35）。

模型路由与 memory 共用同一个 structured chat client（用户裁定）。锁沿组合根现状共享
（SQLite 锁库；取舍与观测盲区记录在 ``behavior/reduction/runner.py`` 的 sweep 锁 docstring）。
归约是无队列的定拍 sweep，Worker 只保留循环存活与可观测事件。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from datetime import UTC, datetime, tzinfo

from habitus.behavior.fusion import BehaviorFusionReceiptStore, BehaviorJudgementStore
from habitus.behavior.fusion.config import FUSION_CONTEXT_LOOKBACK_SECONDS, BehaviorFusionConfig
from habitus.behavior.fusion.coverage import BehaviorCoverageIndex
from habitus.behavior.fusion.lanes.physical import (
    PhysicalLane,
    PhysicalSegmentLedger,
    PhysicalSegmentRecorder,
    UnitOutcome,
    parse_delivery,
)
from habitus.behavior.fusion.lanes.session import (
    SessionLane,
    SessionLaneConfig,
    SessionTurnDescriber,
    SessionTurnRecorder,
)
from habitus.behavior.kinds.calls import KindModelCaller
from habitus.behavior.kinds.classify import DaytimeClassifier
from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.kinds.ids import Lane
from habitus.behavior.kinds.nightly import NightlyGrower, PendingRecheck
from habitus.behavior.kinds.revision import AnchorGate, Reviser
from habitus.behavior.kinds.store import BehaviorKindStore
from habitus.behavior.observation import BehaviorObservationStore
from habitus.behavior.observation.config import BehaviorObservationConfig
from habitus.behavior.reduction import (
    DEFAULT_SWEEP_LOCK_TTL_SECONDS,
    BehaviorReductionBusyError,
    BehaviorReductionLedger,
    BehaviorReductionRunner,
    VocabularyJobReport,
)
from habitus.behavior.reduction.kinds_jobs import VocabularyJobs
from habitus.behavior.reduction.kinds_migration import KindMigrator
from habitus.behavior.reduction.kinds_step import KindStamping
from habitus.behavior.reduction.quarantine import ChainQuarantine
from habitus.behavior.reduction.refresh import RefreshRetries
from habitus.behavior.semantic import BehaviorSemanticRefresher, LLMBehaviorOverviewGenerator
from habitus.behavior.tree import BehaviorTree
from habitus.behavior.tree.config import BehaviorTreeConfig
from habitus.behavior.tree.dirty_days import DirtyDayLog
from habitus.config import HabitusConfig
from habitus.foundation.observability import (
    NullObserver,
    ObservationEvent,
    ObservationStatus,
    Observer,
    SpanController,
    observe_operation,
)
from habitus.infrastructure.lifecycle import Quarantine, RetryLedger
from habitus.infrastructure.store.contracts.lock import LockStore
from habitus.infrastructure.store.contracts.path_lock import PathLock
from habitus.model_client import StructuredChatClient
from habitus.runtime.lifecycle_retry import retry_ledger
from habitus.runtime.resident import ResidentWorker


@dataclass(frozen=True)
class BehaviorRuntimeComponents:
    """一条 lane 这一支的全部已组装部件：它自己的存储、融合、归约、行为树与词表。

    ``fusion`` 是这条 lane 的融合前端（会话 lane 或物理 lane），与 ``lane`` 对得上。
    ``__post_init__`` 按仓库既有纪律做**实例同一性**校验：融合、归约、Worker 引用的存储必须
    就是组件字段那一份——这些是"自己产物自洽"类约束，违反只可能是接线错误。
    """

    lane: Lane

    observations: BehaviorObservationStore
    judgements: BehaviorJudgementStore
    receipts: BehaviorFusionReceiptStore
    coverage: BehaviorCoverageIndex
    tree: BehaviorTree
    kind_store: BehaviorKindStore
    reduction_runner: BehaviorReductionRunner
    fusion: SessionLane | PhysicalLane
    reduction_worker: BehaviorReductionWorker

    def __post_init__(self) -> None:
        object.__setattr__(self, "lane", Lane(self.lane))
        front_end = SessionLane if self.lane is Lane.SESSION else PhysicalLane
        expected = (
            ("observations", self.observations, BehaviorObservationStore),
            ("judgements", self.judgements, BehaviorJudgementStore),
            ("receipts", self.receipts, BehaviorFusionReceiptStore),
            ("coverage", self.coverage, BehaviorCoverageIndex),
            ("tree", self.tree, BehaviorTree),
            ("kind_store", self.kind_store, BehaviorKindStore),
            ("reduction_runner", self.reduction_runner, BehaviorReductionRunner),
            ("fusion", self.fusion, front_end),
            ("reduction_worker", self.reduction_worker, BehaviorReductionWorker),
        )
        for name, value, expected_type in expected:
            if not isinstance(value, expected_type):
                raise TypeError(f"behavior components {name} must be {expected_type.__name__}")
        shared = (
            (self.reduction_runner.judgements, self.judgements, "reduction judgements"),
            (self.reduction_runner.observations, self.observations, "reduction observations"),
            (self.reduction_runner.receipts, self.receipts, "reduction receipts"),
            (self.reduction_runner.tree, self.tree, "reduction tree"),
            (self.reduction_runner.kinds.store, self.kind_store, "reduction kind store"),
            (self.reduction_runner.coverage, self.coverage, "coverage index (reduction)"),
            # 融合写、归约读的是同一套判断、回执、覆盖索引。
            (self.fusion.recorder.observations, self.observations, "fusion observations"),
            (self.fusion.recorder.judgements, self.judgements, "fusion judgements"),
            (self.fusion.recorder.receipts, self.receipts, "fusion receipts"),
            (self.fusion.recorder.coverage, self.coverage, "coverage index (fusion)"),
            (self.reduction_worker.runner, self.reduction_runner, "reduction worker runner"),
        )
        for actual, wanted, label in shared:
            if actual is not wanted:
                raise ValueError(f"behavior components must share one {label} instance")
        if self.kind_store.root != self.tree.root:
            raise ValueError("behavior vocabulary files must live at the tree root")
        if self.reduction_runner.kinds.lane is not self.lane:
            raise ValueError("a branch classifies into its own lane's vocabulary")
        # 一支的全部存储在同一个根下（树在它的 ``tree/`` 里）：哪一样指到别处，都是接到了另一支上。
        root = self.observations.root
        homes = (
            ("judgements", self.judgements.root),
            ("receipts", self.receipts.root),
            ("coverage", self.coverage.root),
            ("reduction ledger", self.reduction_runner.ledger.root.parent),
            ("tree", self.tree.root.parent),
        )
        for label, home in homes:
            if home != root:
                raise ValueError(f"behavior branch {self.lane.value} keeps its {label} outside its own root")
        ledger = getattr(self.fusion.recorder, "ledger", None)
        if ledger is not None and ledger.root != root:
            raise ValueError(f"behavior branch {self.lane.value} keeps its segment ledger outside its own root")

    @property
    def physical_ledger(self) -> PhysicalSegmentLedger | None:
        """物理一支的小账（生命周期按补发跨度整日释放，26 号清单 6.4）；会话一支没有。"""

        return self.fusion.recorder.ledger if isinstance(self.fusion, PhysicalLane) else None

    @property
    def quarantines(self) -> tuple[Quarantine, ...]:
        """这一支的隔离区（归约的链、会话 lane 的轮）：放弃的项按统一寿命清、各报容量（L10.10）。"""

        found = [self.reduction_runner.chain_quarantine.quarantine]
        if isinstance(self.fusion, SessionLane):
            found.append(self.fusion.recorder.quarantine)
        return tuple(found)

    @property
    def retry_ledgers(self) -> tuple[RetryLedger, ...]:
        refresh = self.reduction_runner.refresh_retries
        return () if refresh is None else (refresh.ledger,)


@dataclass(frozen=True)
class BehaviorBranches:
    """行为侧的两支：会话一支、物理一支，各是一套完整的 ``BehaviorRuntimeComponents``，不共用任何东西。"""

    session: BehaviorRuntimeComponents
    physical: BehaviorRuntimeComponents

    def __post_init__(self) -> None:
        for name, lane in (("session", Lane.SESSION), ("physical", Lane.PHYSICAL)):
            branch = getattr(self, name)
            if not isinstance(branch, BehaviorRuntimeComponents) or branch.lane is not lane:
                raise TypeError(f"behavior branch {name} must be the {lane.value} lane's components")
        apart = (
            ("observations", self.session.observations.root, self.physical.observations.root),
            ("tree", self.session.tree.root, self.physical.tree.root),
        )
        for label, left, right in apart:
            if left == right:
                raise ValueError(f"the two behavior branches must not share one {label} root")

    def __iter__(self) -> Iterator[BehaviorRuntimeComponents]:
        return iter((self.session, self.physical))

    def of(self, lane: Lane | str) -> BehaviorRuntimeComponents:
        return self.session if Lane(lane) is Lane.SESSION else self.physical

    @property
    def session_lane(self) -> SessionLane:
        """会话那一支的融合前端（接成会话源的消费者用）。"""

        lane = self.session.fusion
        if not isinstance(lane, SessionLane):  # 由 ``BehaviorRuntimeComponents`` 的构造保证；这里只是给类型检查看
            raise TypeError("the session branch must carry the session lane")
        return lane

    @property
    def physical_lane(self) -> PhysicalLane:
        """物理那一支的融合前端（Runtime 的交付正门用）。"""

        lane = self.physical.fusion
        if not isinstance(lane, PhysicalLane):  # 同上
            raise TypeError("the physical branch must carry the physical lane")
        return lane


class BehaviorReductionWorker(ResidentWorker):
    """归约 sweep 的常驻节拍（用户裁定 5 分钟一轮）；正确性全在 runner，这里只管节奏。

    sweep 内部的同步 IO 会短暂占用事件循环；原料随消费释放、读法按天缓存之后（26 号清单 L2、L6）量级不再随历史涨，
    节拍是 5 分钟一次。
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
        span_controller: SpanController | None = None,
    ) -> None:
        super().__init__(shutdown_timeout_seconds=shutdown_timeout_seconds, observer=observer)
        self.runner = runner
        self.interval_seconds = float(interval_seconds)
        self.span_controller = span_controller

    async def _run_loop(self) -> None:
        while not self._stop_requested.is_set():
            started = time.monotonic()
            try:
                # 一轮 sweep 一条 trace：成链、封口、定类型（含模型调用）、发布、摘要刷新都挂在下面。
                with _span(self.span_controller, "reduction_sweep"):
                    await self.runner.run_once()
            except BehaviorReductionBusyError:
                # sweep 锁被另一持有者占用：多实例场景的正常让路，跳过本拍即可。
                self._observe("reduction_sweep", ObservationStatus.SUCCESS, {"skipped": "lock_busy"})
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


class _LaneObserver:
    """给一支发出的观测事件都标上 lane。"""

    def __init__(self, observer: Observer, lane: Lane) -> None:
        self._observer = observer
        self._lane = lane.value

    def record(self, event: ObservationEvent) -> None:
        self._observer.record(replace(event, attributes={**event.attributes, "lane": self._lane}))


def build_behavior_components(
    config: HabitusConfig,
    *,
    structured_chat: StructuredChatClient,
    lock_store: LockStore,
    path_lock: PathLock,
    observer: Observer | None = None,
    span_controller: SpanController | None = None,
    clock: Callable[[], datetime] | None = None,
    source_settled: Callable[[str], bool] | None = None,
) -> BehaviorBranches | None:
    """组装行为管线的两支；``primary_subject`` 未配置时返回 None（行为侧未启用）。``source_settled`` 答"这份会话源的
    Behavior 到终态了没有"（会话 lane 的覆盖记录据它释放，26 号清单 6.1）；behavior 不 import conversation，由这里注入。

    无存储写入、无模型请求——与 ``build_runtime`` 同一纪律。
    """

    if not config.behavior.enabled:
        return None

    def branch(lane: Lane) -> BehaviorRuntimeComponents:
        return _build_branch(
            config,
            lane,
            structured_chat=structured_chat,
            lock_store=lock_store,
            observer=observer,
            span_controller=span_controller,
            clock=clock,
            source_settled=source_settled,
        )

    return BehaviorBranches(session=branch(Lane.SESSION), physical=branch(Lane.PHYSICAL))


def _build_branch(
    config: HabitusConfig,
    lane: Lane,
    *,
    structured_chat: StructuredChatClient,
    lock_store: LockStore,
    observer: Observer | None,
    span_controller: SpanController | None,
    clock: Callable[[], datetime] | None,
    source_settled: Callable[[str], bool] | None = None,
) -> BehaviorRuntimeComponents:
    """一条 lane 的那一支。全部存储传**这一支的根**（各存储按自己的契约派生子路径——传子目录会造成
    目录二次嵌套与锁键漂移，评审实测过）；归约的 sweep 锁按根派生，两支互不相抢。"""

    behavior_config = config.behavior
    root = config.behavior_root / lane.value
    # 这一支发出的每条观测事件都带上是哪一支：两支的事件、指标同名，不带就分不出、还会互相盖掉。
    observer = None if observer is None else _LaneObserver(observer, lane)
    context_lookback = (
        float(behavior_config.context_lookback_seconds)
        if behavior_config.context_lookback_seconds is not None
        else FUSION_CONTEXT_LOOKBACK_SECONDS
    )

    # 存储的容量闸从 Config 边界进来（26 号清单 8.2）；没写的是各领域配置的默认值
    observations = BehaviorObservationStore(
        root, config=BehaviorObservationConfig(**behavior_config.overrides_for("observation_"))
    )
    fusion_config = BehaviorFusionConfig(**behavior_config.overrides_for("fusion_"))
    judgements = BehaviorJudgementStore(root, config=fusion_config)
    receipts = BehaviorFusionReceiptStore(root, config=fusion_config)
    # 一份覆盖索引两处共用：融合写、归约读——各自实例化会按不同窗口口径回答"处理过没有"。
    coverage = BehaviorCoverageIndex(
        root,
        window_days=behavior_config.coverage_window_days,
        settled=source_settled if lane is Lane.SESSION else None,
    )

    tree = BehaviorTree(root / "tree", tree_config=BehaviorTreeConfig(**behavior_config.overrides_for("tree_")))
    # 词表的门槛从唯一的 Config 边界进来；模型只经 KindModelCaller 一个触点。
    kind_config = BehaviorKindConfig(**behavior_config.kinds_overrides())
    kind_store = BehaviorKindStore(tree.root, config=kind_config)
    kinds, vocabulary_jobs = _vocabulary(tree, lock_store, kind_store, structured_chat, config.locale.zone(), lane)
    reduction_runner = BehaviorReductionRunner(
        judgements=judgements,
        observations=observations,
        receipts=receipts,
        tree=tree,
        lock_store=lock_store,
        kinds=kinds,
        vocabulary_jobs=vocabulary_jobs,
        ledger=BehaviorReductionLedger(root / "reduction"),
        semantic_refresher=BehaviorSemanticRefresher(tree, LLMBehaviorOverviewGenerator(structured_chat)),
        refresh_retries=RefreshRetries(
            retry_ledger(root, owner=f"behavior-semantic-{lane.value}", config=config.lifecycle.retry, clock=clock)
        ),
        # 脏日日志只有语义树夜批读（只接会话一支）：别的支不记
        dirty_days=DirtyDayLog(root / "reduction") if lane is Lane.SESSION and config.scene.enabled else None,
        chain_quarantine=ChainQuarantine(
            judgements, Quarantine(root, owner=f"behavior-reduction-{lane.value}", clock=clock or _utc_now)
        ),
        clock=clock,
        context_lookback_seconds=context_lookback,
        coverage=coverage,
        sweep_lock_ttl_seconds=(
            behavior_config.reduction_sweep_lock_ttl_seconds
            if behavior_config.reduction_sweep_lock_ttl_seconds is not None
            else DEFAULT_SWEEP_LOCK_TTL_SECONDS
        ),
        observer=observer,
    )
    zone = config.locale.zone()
    lane_clock = clock if clock is not None else (lambda: datetime.now(UTC))
    fusion: SessionLane | PhysicalLane
    if lane is Lane.SESSION:
        fusion = SessionLane(
            SessionTurnDescriber(
                structured_chat, zone=zone, config=SessionLaneConfig(**behavior_config.session_overrides())
            ),
            SessionTurnRecorder(
                observations=observations,
                judgements=judgements,
                receipts=receipts,
                coverage=coverage,
                primary_subject=behavior_config.primary_subject,
                zone=zone,
                clock=lane_clock,
                quarantine=Quarantine(root, owner="behavior-session-lane", clock=lane_clock),
            ),
        )
    else:
        fusion = PhysicalLane(
            PhysicalSegmentRecorder(
                observations=observations,
                judgements=judgements,
                receipts=receipts,
                coverage=coverage,
                ledger=PhysicalSegmentLedger(root),
                primary_subject=behavior_config.primary_subject,
                clock=lane_clock,
            ),
            max_backfill_days=config.lifecycle.physical_max_backfill_days,
        )
    return BehaviorRuntimeComponents(
        lane=lane,
        observations=observations,
        judgements=judgements,
        receipts=receipts,
        coverage=coverage,
        tree=tree,
        kind_store=kind_store,
        reduction_runner=reduction_runner,
        fusion=fusion,
        reduction_worker=BehaviorReductionWorker(
            reduction_runner,
            interval_seconds=behavior_config.reduction_sweep_interval_seconds,
            shutdown_timeout_seconds=behavior_config.worker_shutdown_timeout_seconds,
            observer=observer,
            span_controller=span_controller,
        ),
    )


def deliver_physical(
    branches: BehaviorBranches,
    payload: object,
    *,
    observer: Observer | None = None,
) -> tuple[UnitOutcome, ...]:
    """物理 lane 的正门：上游的一份交付进来，校验、一段一条写成判断；返回每一段的去向。

    不合约定的整份拒收（``PhysicalDeliveryError``）。幂等：同一段再送一遍不重复写，上游改判不收。
    全是同步的文件读写，调用方放到线程里做。落树由归约的定拍 sweep 接手。
    """

    with observe_operation(observer or NullObserver(), "behavior", "physical_delivery") as attributes:
        lane = branches.physical_lane
        delivery = parse_delivery(payload, config=lane.recorder.config)
        outcomes = lane.consume(delivery)
        attributes.update(
            {
                "segments": len(delivery.segments),
                "gaps": len(delivery.gaps),
                **{
                    f"units_{name}": sum(1 for item in outcomes if item.disposition.value == name)
                    for name in sorted({item.disposition.value for item in outcomes})
                },
            }
        )
    return outcomes


def _span(controller: SpanController | None, operation: str) -> AbstractContextManager[None]:
    """行为 Worker 的根 span；未配置追踪时是空上下文。作业身份等属性由 runner 的事件带上同一个 span。"""

    if controller is None:
        return nullcontext()
    return controller.start_span("behavior", operation)


def _vocabulary(
    tree: BehaviorTree,
    lock_store: LockStore,
    store: BehaviorKindStore,
    structured_chat: StructuredChatClient,
    zone: tzinfo,
    lane: Lane,
) -> tuple[KindStamping, VocabularyJobs]:
    """词表的两块：归约里的白天归类，与三件整理活（定期拆改的放行只靠用户自己的数据，裁定 17）。"""

    caller = KindModelCaller(structured_chat, config=store.config)
    classifier = DaytimeClassifier(caller)
    stamping = KindStamping(store, classifier, lane=lane)
    jobs = VocabularyJobs(
        tree=tree,
        stamping=stamping,
        migrator=KindMigrator(tree, lock_store, store),
        grower=NightlyGrower(caller, PendingRecheck(classifier)),
        reviser=Reviser(caller, AnchorGate(classifier)),
        zone=zone,
    )
    return stamping, jobs


async def run_vocabulary_job(
    components: BehaviorRuntimeComponents,
    job: str,
    *,
    observer: Observer | None = None,
) -> VocabularyJobReport:
    """词表整理活的运维正门：``grow``（立刻每晚新增）、``revise``（立刻定期拆改）。

    持 sweep 锁，归约 Worker 在此期间让路（lock_busy 跳拍）。
    """

    runner = components.reduction_runner
    jobs = {
        "grow": runner.grow_vocabulary,
        "revise": runner.revise_vocabulary,
    }
    if job not in jobs:
        raise ValueError(f"unknown vocabulary job: {job!r}")
    started = time.monotonic()
    report = await jobs[job]()
    _record(
        observer,
        f"behavior_vocabulary_{job}",
        {"days": len(report.days), "model_calls": report.model_calls, "signals": len(report.signals)},
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


def _utc_now() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "BehaviorReductionWorker",
    "BehaviorBranches",
    "BehaviorRuntimeComponents",
    "UnitOutcome",
    "build_behavior_components",
    "deliver_physical",
    "run_vocabulary_job",
]
