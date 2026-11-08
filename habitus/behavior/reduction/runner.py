"""归约写入层的编排：扫描 → 封口 → 定格（stage）→ 确定性落盘 → 消费账本。

抄融合 ``StagedFusion`` 的检查点形状（死规则⑤）：白天归类、时间戳这些会随时间漂移的输入全部
发生在 stage 之前；stage 之后只有确定性落盘——检查点里存的就是将要落盘的最终 payload（时间已是
ISO 字符串）与文档时间戳，崩溃重试逐字节重放，落盘器的同字节幂等保证不撞车。

运行节奏是运维决定（用户裁定 5 分钟一轮 sweep，由组合根/调度器调用 ``run_once``，本层不自带
定时器）；同样的存储内容，无论哪一轮扫到，产出的文档逐字节相同。

树的单一写入口：新文档只由本 runner 经 ``BehaviorDocumentWriter`` 发布；改已有文档的只有词表迁移
（``kinds_migration``，同一个 writer、只重打 ``kind_token``）——观测 → 融合 → 归约，没有旁路。

本类只留 sweep 编排骨架与落盘、释放、过期、语义刷新；检查点文件（``checkpoint``）、定稿日账（``day_ledger``）、
待刷新集合（``refresh_pending``）、词表整理活的编排（``vocabulary``）与物化（``stager``）各是独立协作者。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from habitus.behavior.fusion.config import FUSION_CONTEXT_LOOKBACK_SECONDS
from habitus.behavior.fusion.coverage import BehaviorCoverageIndex
from habitus.behavior.fusion.receipt_store import BehaviorFusionReceiptStore
from habitus.behavior.fusion.store import BehaviorJudgementStore
from habitus.behavior.kinds.calls import CallFailure, call_failure
from habitus.behavior.model import BehaviorKind
from habitus.behavior.observation import BehaviorObservationStore
from habitus.behavior.reduction.checkpoint import CheckpointStore, document_days
from habitus.behavior.reduction.day_ledger import DayClosureLedger
from habitus.behavior.reduction.errors import BehaviorReductionBusyError, BehaviorReductionError
from habitus.behavior.reduction.failed import FailedChains
from habitus.behavior.reduction.kinds_jobs import VocabularyJobs
from habitus.behavior.reduction.kinds_step import KindStamping, KindStamps
from habitus.behavior.reduction.ledger import BehaviorReductionEntry, BehaviorReductionLedger
from habitus.behavior.reduction.pending import PendingJudgements, pending_judgements
from habitus.behavior.reduction.record import ReducibleJudgement
from habitus.behavior.reduction.refresh import RefreshRetries, RefreshVerdict
from habitus.behavior.reduction.refresh_pending import RefreshPendingLog
from habitus.behavior.reduction.sealing import arrival_horizon, seal_horizon, sealed_gaps, sealed_indexes
from habitus.behavior.reduction.stager import Failed, ReductionStager
from habitus.behavior.reduction.telemetry import (
    assembly_attributes,
    kind_attributes,
    publish_attributes,
    refresh_attributes,
    seal_attributes,
    stage_attributes,
)
from habitus.behavior.reduction.vocabulary import VocabularyJobReport, VocabularyRunner
from habitus.behavior.retry import RetryCause
from habitus.behavior.semantic.model import BehaviorSemanticRefreshResult
from habitus.behavior.semantic.refresher import BehaviorSemanticRefresher
from habitus.behavior.telemetry import OBSERVATION_CATEGORY
from habitus.behavior.tree import BehaviorTree
from habitus.behavior.tree.dirty_days import DirtyDayLog
from habitus.behavior.tree.writer import BehaviorDocumentWriter, BehaviorWriteConfig
from habitus.foundation.observability import NullObserver, Observer, observe_operation
from habitus.infrastructure.store.contracts.lock import LockStore
from habitus.infrastructure.store.contracts.path_lock import LeaseGuard, PathLock

# sweep 锁的默认租约时长；实际值由 ``Config.behavior.reduction_sweep_lock_ttl_seconds`` 注入。
# 周尺度回填（万级小文件写入会把 fseventsd 顶到 240% CPU，单篇 publish 可能被文件系统卡住数分钟）
# 需要更长的租约——那不该靠改代码或猴子补丁，所以它是运维参数而不是常量。
DEFAULT_SWEEP_LOCK_TTL_SECONDS = 600


@dataclass(frozen=True)
class BehaviorReductionReport:
    """一轮 sweep 的可观测结果；数字与信号，不含语义。"""

    replayed_documents: int
    published_occurrences: int
    published_gaps: int
    chains_pending: int
    dropped_edges: tuple[str, ...]
    # 词表侧的事件与信号（新建/命中/过期/降级），与归约本身的丢边、隔离分开列——面板数"异常"时
    # 不把"新建了一个 kind"这种正常事件数进去。
    kind_signals: tuple[str, ...] = ()
    #: 这一轮因为模型整个用不了、换成「错误」放一边的链（夜里那一支的归约据它停线，裁定 94 第 8 条）。
    unavailable: int = 0
    #: 这一轮拦住的我们自己的 bug（归类抛了模型之外的异常）：常驻节拍把它挂到健康面，当场报出。
    bugs: tuple[str, ...] = ()

    @property
    def published_documents(self) -> int:
        return self.published_occurrences + self.published_gaps


@dataclass(frozen=True)
class PinnedFrontier:
    """封口视界被某份观测钉住：哪份交付（``source_id``）、从哪一刻起（它的观测可用的那一刻）。"""

    source_id: str
    since: datetime


@dataclass(frozen=True)
class _Sealed:
    """一轮封口的结果：定稿视界、落树的链与空白段、仍未封口的日子。"""

    horizon: datetime
    ready_indexes: tuple[int, ...]
    ready_gaps: tuple[ReducibleJudgement, ...]
    unsealed_days: frozenset[date]


def _never_reminded(kind_token: str, started_at: datetime) -> bool:
    """没接预测层时的查法：从没提醒过。"""

    return False


class BehaviorReductionRunner:
    """归约写入层；同一 behavior-root 的写入方必须共享 LockStore。"""

    def __init__(
        self,
        *,
        judgements: BehaviorJudgementStore,
        observations: BehaviorObservationStore,
        receipts: BehaviorFusionReceiptStore,
        tree: BehaviorTree,
        lock_store: LockStore,
        kinds: KindStamping,
        vocabulary_jobs: VocabularyJobs,
        ledger: BehaviorReductionLedger,
        semantic_refresher: BehaviorSemanticRefresher | None = None,
        clock: Callable[[], datetime] | None = None,
        context_lookback_seconds: float = FUSION_CONTEXT_LOOKBACK_SECONDS,
        seal_on_arrival: bool = False,
        write_config: BehaviorWriteConfig | None = None,
        coverage: BehaviorCoverageIndex | None = None,
        sweep_lock_ttl_seconds: int = DEFAULT_SWEEP_LOCK_TTL_SECONDS,
        observer: Observer | None = None,
        reminded: Callable[[str, datetime], bool] | None = None,
        refresh_retries: RefreshRetries | None = None,
        dirty_days: DirtyDayLog | None = None,
        failed_chains: FailedChains,
    ) -> None:
        """``context_lookback_seconds`` 必须与融合 runner 实际使用的值一致（同一配置源）——
        融合"还能续"与归约"已封口"是同一个窗口的两面，各配一个数会静默分叉。``seal_on_arrival``：这一支的记录一送到就封口
        （会话 lane，裁定 31），落树不等这个窗口；定稿日仍看它（``sealing``）。

        ``reminded(kind_token, started_at)``：这一类在这一刻开始之前有没有被系统提醒过——预测层的账本回答，
        由组合根注入（本层对预测层零知识）；没接时恒为假。它在物化 payload 时定格（stage 之前），查不出来就整轮失败，
        不许静默记成没提醒（被提醒的发生混进自然率事后分不开）。"""

        if semantic_refresher is not None and refresh_retries is None:
            raise TypeError("semantic refresh needs a RefreshRetries ledger (26 号清单 2.4.1)")
        if context_lookback_seconds <= 0:
            raise ValueError("context_lookback_seconds must be a positive number")
        self.judgements = judgements
        self.observations = observations
        self.receipts = receipts
        self.tree = tree
        self.lock_store = lock_store
        if not 60 <= sweep_lock_ttl_seconds <= 86_400:
            raise ValueError("sweep_lock_ttl_seconds must be an integer between 60 and 86400")
        self.sweep_lock_ttl_seconds = sweep_lock_ttl_seconds
        self.kinds = kinds
        self.vocabulary_jobs = vocabulary_jobs
        self.ledger = ledger
        self.semantic_refresher = semantic_refresher
        self.refresh_retries = refresh_retries
        #: 改树改到的日子记一笔增量，按天缓存的读侧据此只重读变过的日子（26 号清单 6.7）。没有读侧的支（物理一支、
        #: 没开语义树的会话一支）不给，不记——没人截，记了只会一直长（L10.3）。
        self.dirty_days = dirty_days
        if failed_chains.judgements is not judgements or failed_chains.ledger is not ledger:
            raise ValueError("failed chains must set aside judgements from the runner's own store and ledger")
        self.failed_chains = failed_chains
        self.clock = clock or (lambda: datetime.now(UTC))
        self.context_lookback_seconds = float(context_lookback_seconds)
        self.seal_on_arrival = seal_on_arrival
        #: 发布文档时写锁的租约（``BehaviorDocumentWriter``）；没给取领域默认值。
        self.write_config = write_config
        self.coverage = coverage or BehaviorCoverageIndex(observations.root)
        self._path_lock = PathLock(lock_store)
        digest = hashlib.sha256(str(tree.root).encode("utf-8")).hexdigest()[:24]
        self._sweep_lock_key = f"behavior-reduction:{digest}"
        self.observer: Observer = observer or NullObserver()
        self.reminded: Callable[[str, datetime], bool] = reminded if reminded is not None else _never_reminded
        #: 最近一轮认出的"封口视界被哪几份观测钉住"（按钉住的时刻排，没有是空），健康面读它（裁定 94 第 8 条）。
        self.pinned: tuple[PinnedFrontier, ...] = ()
        self._build_collaborators()

    def _build_collaborators(self) -> None:
        """检查点、待刷新集合、定稿日账都落在消费账本的根下；词表编排与本轮共用 sweep 锁与语义刷新。"""

        root = self.ledger.root
        self.checkpoints = CheckpointStore(root)
        self.refresh_pending = RefreshPendingLog(root, dirty_days=self.dirty_days, refresh_retries=self.refresh_retries)
        self.day_ledger = DayClosureLedger(root, tree=self.tree, now=self._now)
        self.stager = ReductionStager(tree=self.tree, observations=self.observations, checkpoints=self.checkpoints)
        self.vocabulary = VocabularyRunner(
            jobs=self.vocabulary_jobs,
            checkpoints=self.checkpoints,
            refresh_pending=self.refresh_pending,
            hold_lock=self._hold_sweep_lock,
            refresh_semantics=self._refresh_semantics,
            now=self._now,
        )

    def attach_reminder_lookup(self, lookup: Callable[[str, datetime], bool]) -> None:
        """组合根在预测层装好之后把账本的查法接进来（归约在预测层之前装配，接不进构造函数）。"""

        self.reminded = lookup

    async def run_once(self) -> BehaviorReductionReport:
        """执行一轮归约；先重放遗留检查点，再归约新封口的链与空白段。

        全程持一把 behavior-root 级的**租约**：单写入方是机械保证不是运维假设——两个归约
        进程并跑会各自 stage 出时间戳不同的同址文档，互相把对方的检查点变成永久撞车。锁被
        占用时本轮直接让路（``BehaviorReductionBusyError``），下一轮 sweep 自然重试。

        刻意**不**把整轮包进 fenced：SQLite 锁实现的 fenced 会跨整个临界区持一个未提交写事务
        ——里面的 writer 文档锁 acquire 撞它即死锁，且共享锁库被占住整轮（评审实测的接线级
        阻塞）。互斥由租约承担、阶段间 ``guard.checkpoint()`` 续约；围栏只围检查点文件的两次
        小写；树文档的写入由 writer 自己的逐文档 fenced 保护。
        """

        with ExitStack() as held:
            guard = self._hold_sweep_lock(held)
            with observe_operation(self.observer, OBSERVATION_CATEGORY, "reduction_sweep") as attributes:
                report = await self._run_locked(guard)
                attributes.update(
                    replayed_documents=report.replayed_documents,
                    published_occurrences=report.published_occurrences,
                    published_gaps=report.published_gaps,
                    chains_pending=report.chains_pending,
                    notes=len(report.dropped_edges),
                    kind_signals=len(report.kind_signals),
                )
                return report

    async def _run_locked(self, guard: LeaseGuard) -> BehaviorReductionReport:
        self._kind_signals: list[str] = []
        self._sweep_signals: list[str] = []
        self._bugs: list[str] = []
        with observe_operation(self.observer, OBSERVATION_CATEGORY, "reduction_replay") as attributes:
            replayed, replayed_days = self.checkpoints.replay(
                lambda checkpoint: self._publish_checkpoint(checkpoint, guard)
            )
            attributes.update(replayed_documents=replayed, refresh_days=len(replayed_days))
        if replayed_days or self.checkpoints.exists():
            # 重放已全部落树：先把那些日子记进待刷新集合，再立刻清检查点——之后的词表整理活可能重打这些条目，
            # 检查点若还留着，下一轮重放就会与树上内容冲突而永久卡住。
            self.refresh_pending.touch(replayed_days, guard)
            self.checkpoints.clear(guard)
        now = self._now()
        # 以下每一段在周尺度都可能跑很久且**不写任何东西**（另一条线实测：stage 前段静默 16 分钟即把
        # 600s 租约耗死，续约点再多也救不了已过期的租约）。因此每段边界都续，长循环内部按条数续。
        guard.checkpoint()
        # 没做完的词表迁移排在检查点重放之后、本轮归类之前续做：本轮新封口的链就按改过的清单归类；改过的日子并进本轮刷新。
        # 词表的每晚新增与定期拆改不在这里，在夜里那条线上、结构那一段之前（裁定 94 第 9 条，``nightly_vocabulary``）。
        with observe_operation(self.observer, OBSERVATION_CATEGORY, "reduction_vocabulary") as attributes:
            jobs = self.vocabulary.resume_migration(guard)
            self.refresh_pending.touch(jobs.days, guard)
            self._kind_signals.extend(jobs.signals)
            attributes.update(changed_days=len(jobs.days), model_calls=jobs.model_calls)
        replayed_days = replayed_days | jobs.days
        guard.checkpoint()
        with observe_operation(self.observer, OBSERVATION_CATEGORY, "reduction_expire") as attributes:
            attributes.update(self._expire(now, guard))
        guard.checkpoint()
        # 消费账本这一轮只读一次、在过期之后读：扫描与物化共用（之前各读一遍，账本一长就是三遍全量读，26 号清单 6.3）。
        # 这之后到物化之前没有任何一步写账本。
        entries = self.ledger.load()
        # 上一轮没刷成的日子并进来（词表整理活改过 kind_token 的日子也在这里）
        replayed_days = replayed_days | self.refresh_pending.days()
        # "哪些判断还没归约"只有一种答法（``pending``），此刻场景的未封口读口也用它。
        with observe_operation(self.observer, OBSERVATION_CATEGORY, "reduction_assemble") as attributes:
            pending = pending_judgements(self.judgements, self.ledger, checkpoint=guard.checkpoint, entries=entries)
            attributes.update(assembly_attributes(pending))
        quarantined = list(pending.quarantined)
        for raw in pending.rejected:
            guard.checkpoint()
            self.failed_chains.drop_record(raw)
        guard.checkpoint()
        sealed = self._seal(pending, now, guard)
        guard.checkpoint()
        if not sealed.ready_indexes and not sealed.ready_gaps:
            refresh_notes = await self._finish_sweep(
                replayed_days, guard, unsealed_days=sealed.unsealed_days, horizon=sealed.horizon
            )
            return BehaviorReductionReport(
                replayed_documents=replayed,
                published_occurrences=0,
                published_gaps=0,
                chains_pending=len(pending.occurrences),
                dropped_edges=(*quarantined, *refresh_notes, *self._sweep_signals),
                kind_signals=tuple(self._kind_signals),
                bugs=tuple(self._bugs),
            )

        with observe_operation(self.observer, OBSERVATION_CATEGORY, "reduction_stage") as attributes:
            # 归类只做一次：缩批只是少发布几条，复用同一份结果（不重复调模型、结论也不会变）。
            stamps = await self._classify([pending.occurrences[index] for index in sealed.ready_indexes], guard=guard)
            batch = await self.stager.stage_bounded(
                pending,
                (sealed.ready_indexes, sealed.ready_gaps),
                now,
                stamps,
                entries=entries,
                refresh_days=replayed_days,
                reminded=self.reminded,
                signals=self._sweep_signals,
                guard=guard,
            )
            attributes.update(
                stage_attributes(
                    batch.documents,
                    batch.dropped,
                    ready_chains=len(sealed.ready_indexes),
                    ready_gaps=len(sealed.ready_gaps),
                    shrink_rounds=batch.shrink_rounds,
                    checkpoint_bytes=len(batch.encoded),
                )
            )
        self.checkpoints.write(batch.encoded, guard)
        self._publish_checkpoint(batch.checkpoint, guard)
        guard.checkpoint()
        self._isolate_failed_chains(
            batch.failed, guard, consumed=frozenset(i for entry in entries for i in entry.judgement_ids)
        )
        documents = batch.documents
        self.refresh_pending.touch(document_days(documents), guard)
        # 这一轮没落树的链所在的日子也不能定稿：缩批留下的要下一轮才发布；隔离的换成了「没读懂」判断，下一轮才归约成
        # 空白——先定稿，派生层就把那一段读成"没做"（L10.5）
        held_back = _unsealed_days(pending, batch.active_indexes, batch.active_gaps) | {
            record.started_at.date() for record, _reason, _cause in batch.failed
        }
        refresh_notes = await self._finish_sweep(
            replayed_days | document_days(documents),
            guard,
            unsealed_days=sealed.unsealed_days | held_back,
            horizon=sealed.horizon,
        )
        published_occurrences = sum(1 for item in documents if item["kind"] == BehaviorKind.OCCURRENCE.value)
        return BehaviorReductionReport(
            replayed_documents=replayed,
            published_occurrences=published_occurrences,
            published_gaps=sum(1 for item in documents if item["kind"] == BehaviorKind.GAP.value),
            # skip/defer/缩批留下的链仍未消费，一并计入 pending——报告不许把它们说成已处理。
            chains_pending=len(pending.occurrences) - published_occurrences,
            dropped_edges=(*quarantined, *batch.dropped, *refresh_notes, *self._sweep_signals),
            kind_signals=tuple(self._kind_signals),
            unavailable=sum(1 for _record, _reason, cause in batch.failed if cause is RetryCause.UNAVAILABLE),
            bugs=tuple(self._bugs),
        )

    def _seal(self, pending: PendingJudgements, now: datetime, guard: LeaseGuard) -> _Sealed:
        """封口：认出钉住视界的交付、报出读不出来的交付，定这一轮落树的链与空白段。"""

        with observe_operation(self.observer, OBSERVATION_CATEGORY, "reduction_seal") as attributes:
            frontier_cutoff = self._frontier_cutoff(guard)
            self._sweep_signals.extend(
                f"seal horizon pinned by observation delivery {pinned.source_id[:12]} since {pinned.since.isoformat()}"
                for pinned in self.pinned
            )
            # 读不出来的交付被存储放进了隔离区（数据坏了，裁定 94 错误分类）：不钉住视界、不挡定稿，在这里报出
            self._sweep_signals.extend(
                f"observation delivery {source_id[:12]} unreadable; isolated"
                for source_id in self.observations.take_isolated()
            )
            horizon = seal_horizon(
                now=now,
                frontier_cutoff=frontier_cutoff,
                lookback_seconds=self.context_lookback_seconds,
            )
            # 落树的视界：一送到就封口的那一支不等 lookback；定稿日一律看 ``horizon``。
            landing = arrival_horizon(now=now, frontier_cutoff=frontier_cutoff) if self.seal_on_arrival else horizon
            ready_indexes = sealed_indexes(pending.occurrences, landing)
            ready_gaps = sealed_gaps(pending.gaps, landing)
            unsealed_days = _unsealed_days(pending, ready_indexes, ready_gaps)
            attributes.update(
                seal_attributes(
                    pending,
                    ready_indexes,
                    ready_gaps,
                    now=now,
                    horizon=horizon,
                    frontier_cutoff=frontier_cutoff,
                )
            )
            attributes["unsealed_days"] = len(unsealed_days)
        return _Sealed(horizon, ready_indexes, ready_gaps, unsealed_days)

    async def nightly_vocabulary(self, night: date) -> VocabularyJobReport:
        """夜里那条线在结构那一段之前调它（裁定 94 第 8、9 条）：这一晚的每晚新增，与这一晚所在拆改周期里还没做的定期拆改。
        锁被占抛 ``BehaviorReductionBusyError``（线据此下一眼再来）；模型整个用不了、代码出错原样抛给线（停下这一晚）。"""

        return await self.vocabulary.nightly(night)

    async def grow_vocabulary(self) -> VocabularyJobReport:
        """立刻跑一次每晚新增（运维与实验用；常规路径是夜里那条线的 ``nightly_vocabulary``）。"""

        return await self.vocabulary.grow()

    async def revise_vocabulary(self) -> VocabularyJobReport:
        """立刻跑一次定期拆改（运维与实验用；常规路径是夜里那条线的 ``nightly_vocabulary``）。"""

        return await self.vocabulary.revise()

    def _hold_sweep_lock(self, held: ExitStack) -> LeaseGuard:
        """取 sweep 租约、挂到 ``held`` 上（随它退出释放）。``PathLock.acquire`` 是上下文管理器，真正取锁发生在进入时，
        所以进入本身包在 try 里：只有**这里**的超时是"锁被占"（``BehaviorReductionBusyError``，调用方让路）；持锁之后正文里
        的 TimeoutError（续约失败、文档锁竞争）是真故障，不许被归因成让路。"""

        try:
            return held.enter_context(
                self._path_lock.acquire(self._sweep_lock_key, ttl_seconds=self.sweep_lock_ttl_seconds)
            )
        except TimeoutError as exc:
            raise BehaviorReductionBusyError(str(exc)) from exc

    async def _finish_sweep(
        self,
        days: set[date],
        guard: LeaseGuard,
        *,
        unsealed_days: frozenset[date] = frozenset(),
        horizon: datetime | None = None,
    ) -> tuple[str, ...]:
        """一轮的收尾，两条返回路径共用，顺序即崩溃安全的依据：

        受影响日先耐久进待刷新集合 → 清检查点（发布完成即清，不再被刷新失败扣住——那会让 merge
        之后的重放撞出永久冲突）→ 逐日刷新（失败留在集合里下轮重试）→ 定稿日记账 → 释放无人引用的交付。
        """

        # 现有的调用方交来的日子都已在各自那一步落进集合（重放、词表整理活、本轮文档日）：这一次写是防御，
        # 保证清检查点之前它们一定耐久，不靠调用方各自记得。
        self.refresh_pending.write(self.refresh_pending.days() | days, guard)
        self.checkpoints.clear(guard)
        notes = await self._refresh_semantics(days, guard)
        with observe_operation(self.observer, OBSERVATION_CATEGORY, "day_close") as attributes:
            closed, still_open = self.day_ledger.close(days, guard, unsealed_days=unsealed_days, horizon=horizon)
            attributes.update(closed_days=closed, open_days=still_open, unsealed_days=len(unsealed_days))
        with observe_operation(self.observer, OBSERVATION_CATEGORY, "raw_release") as attributes:
            released, retained = self._release_unreferenced(guard)
            attributes.update(envelopes_released=released, envelopes_retained=retained)
        return notes

    # ── 定稿日：派生层（语义树、预测层的结算）只处理已定稿的历史 ────────────────────────────

    def closed_days(self) -> tuple[date, ...]:
        """已定稿的本地日历日（升序）：那一天的链都已落树、且封口视界已过那天的本地结束。"""

        return self.day_ledger.closed_days()

    async def _classify(self, records: Sequence[ReducibleJudgement], *, guard: LeaseGuard) -> KindStamps:
        """白天归类——写入层唯一的 LLM 触点。只定"判断 → kind_token"，不改词表（白天不建类）。

        结果随检查点定格：崩溃重放走检查点里的 token，不再问模型。
        """

        with observe_operation(self.observer, OBSERVATION_CATEGORY, "kind_classify") as attributes:
            try:
                stamps = await self.kinds.classify(records, checkpoint=guard.checkpoint)
            except Exception as exc:  # noqa: BLE001 - 归类失败不拖住那天：这批链换成「错误」放一边（裁定 94 第 3 条）
                guard.checkpoint()  # 租约丢了（另一进程接手）照常整轮失败，不当成归类失败
                # 按错误分类分开（裁定 94 第 8 条）：模型整个用不了的放一边等当晚（夜里那一支的归约据报告停线），
                # 别的是我们自己的 bug，当场报出（``bugs``，常驻节拍据它挂到健康面）、不白试
                failure = call_failure(exc)
                if failure is CallFailure.BUG:
                    self._bugs.append(f"kind_classify {type(exc).__name__}: {exc}")
                self._kind_signals.append(
                    f"kind_classify_failed {failure.value} {type(exc).__name__}: {len(records)} chains set aside"
                )
                attributes.update(failed=type(exc).__name__)
                return KindStamps(
                    {}, {}, len(records), 0, 0, 0, (), set_aside={record.chain_digest: failure for record in records}
                )
            self._kind_signals.extend(stamps.signals)
            attributes.update(
                kind_attributes(
                    requests=stamps.requests,
                    classified=stamps.classified,
                    pending=len(stamps.pending),
                    not_events=stamps.not_events,
                    model_calls=stamps.model_calls,
                    signals=stamps.signals,
                )
            )
            return stamps

    def _record_pending(self, documents: Sequence[Any]) -> int:
        """发布后把待定条目放进待定池（按 occurrence 地址幂等；池满挤掉最老的旧条目、留信号，不抛错）。"""

        admission = self.kinds.record_published(documents)
        self._kind_signals.extend(admission.signals())
        return admission.added

    # ── 落盘：stage 之后只有确定性动作 ────────────────────────────────────────────────

    def _isolate_failed_chains(self, failed: Sequence[Failed], guard: LeaseGuard, *, consumed: frozenset[str]) -> None:
        """stage 里出了错误的判断进夜里待处理的堆，换成一条「错误」判断，下一轮归约成空白（``failed.py``）；当晚重试又失败的
        直接撤下。"""

        for record, reason, cause in failed:
            guard.checkpoint()
            with guard.fenced():
                replacement, admission = self.failed_chains.fail_chain(record, consumed=consumed, cause=cause)
            if replacement is None:
                self._sweep_signals.append(f"chain {record.chain_digest} failed again on retry ({reason}); dropped")
                continue
            kept = admission is not None and admission.kept
            self._sweep_signals.append(
                f"chain {record.chain_digest} {'set aside for retry' if kept else 'not kept for retry'} ({reason}); "
                f"error {replacement[:12]}" + ("" if kept or admission is None else f"; backlog {admission.value}")
            )

    async def _refresh_semantics(self, days: set[date], guard: LeaseGuard) -> tuple[str, ...]:
        """刷新受影响目录的 L0/L1（仍在 sweep 锁内——刷新器的单写入方前提）；**逐日隔离**。

        摘要是可重建派生物：某一天刷新失败只影响那一天。失败按重试账处理（``refresh.py``）：可重试的退避后再试，
        退避期内留在待刷新集合里；超限这类确定性失败第一次就放弃、移出集合、留信号，不再每轮调一次模型。
        """

        if not days:
            return ()
        if self.semantic_refresher is None or self.refresh_retries is None:
            self.refresh_pending.write(self.refresh_pending.days() - days, guard)
            return ()
        with observe_operation(self.observer, OBSERVATION_CATEGORY, "semantic_refresh") as attributes:
            notes: list[str] = []
            results: list[BehaviorSemanticRefreshResult] = []
            remaining = self.refresh_pending.days() | days
            given_up = 0
            for day in sorted(days):
                verdict = self.refresh_retries.verdict(day)
                if verdict is RefreshVerdict.GIVEN_UP:
                    remaining.discard(day)
                    continue
                if verdict is RefreshVerdict.WAIT:
                    continue
                guard.checkpoint()
                try:
                    results.extend(await self.semantic_refresher.refresh_days((day,)))
                except Exception as exc:  # noqa: BLE001 - 派生物刷新失败一律降级为信号
                    decision = self.refresh_retries.failed(day, exc)
                    if decision.give_up:
                        remaining.discard(day)
                        given_up += 1
                        notes.append(f"semantic refresh given up for [{day.isoformat()}]: {exc}")
                    else:
                        notes.append(f"semantic refresh failed for [{day.isoformat()}]: {exc}")
                    continue
                self.refresh_retries.succeeded(day)
                remaining.discard(day)
            self.refresh_pending.write(remaining, guard)
            attributes.update(refresh_attributes(results, days=len(days), failed_days=len(notes)))
            attributes["pending_days"] = len(remaining)
            attributes["given_up_days"] = given_up
            return tuple(notes)

    def _publish_checkpoint(self, checkpoint: Mapping[str, Any], guard: LeaseGuard) -> None:
        """把检查点逐字落盘：同字节幂等，故本函数可任意次重入；逐段续约防租约过期。"""

        with observe_operation(self.observer, OBSERVATION_CATEGORY, "reduction_publish") as attributes:
            documents = self._publish_documents(checkpoint, guard)
            pending_added = self._record_pending(documents)
            attributes.update(
                publish_attributes(
                    [item for item in documents if isinstance(item, Mapping)],
                    published_at=self._now(),
                    kind_pending=pending_added,
                )
            )
            # 树写完、账本写完，原料才释放（顺序是崩溃安全的依据：重放路径重新走到这里再补删）。
            attributes["judgements_released"] = self._release(documents, guard)

    def _publish_documents(self, checkpoint: Mapping[str, Any], guard: LeaseGuard) -> list[Any]:
        """逐篇落盘、记账；返回文档，供调用方记待定、计数与释放。"""

        staged_at_raw = checkpoint.get("staged_at")
        if not isinstance(staged_at_raw, str):
            raise BehaviorReductionError("reduction checkpoint is missing staged_at")
        staged_at = datetime.fromisoformat(staged_at_raw)
        if staged_at.utcoffset() is None:
            raise BehaviorReductionError("reduction checkpoint staged_at must be timezone-aware")
        documents = checkpoint.get("documents")
        if not isinstance(documents, list):
            raise BehaviorReductionError("reduction checkpoint is missing documents")
        writer = BehaviorDocumentWriter(self.tree, self.lock_store, clock=lambda: staged_at, config=self.write_config)
        for index, item in enumerate(documents):
            # 每 10 篇续一次：一次续约远比一篇文档的写入便宜，而周尺度下单篇 publish 会被文件系统
            # 拖到秒级（万级小文件 + fseventsd 放大），间隔太疏时一段慢写就能把租约耗光。
            if index % 10 == 0:
                guard.checkpoint()
            if not isinstance(item, Mapping):
                raise BehaviorReductionError("reduction checkpoint document must be a mapping")
            entry = BehaviorReductionEntry.from_mapping(item["ledger"])
            if item["kind"] != "ledger-only":
                writer.publish(BehaviorKind(item["kind"]), item["payload"])
            self.ledger.append(entry)
        return documents

    # ── 释放：原料被消费后即删，真正的数据只在树上 ─────────────────────────────────────

    def _release(self, documents: list[Any], guard: LeaseGuard) -> int:
        """删掉本批已发布链消费的判断（判断在发布后零读者）；交付的释放见 ``_release_unreferenced``。"""

        judgement_ids: set[str] = set()
        for item in documents:
            if not isinstance(item, Mapping):
                continue
            ledger = item.get("ledger")
            if isinstance(ledger, Mapping):
                judgement_ids.update(str(value) for value in ledger.get("judgement_ids", ()))
        for index, judgement_id in enumerate(sorted(judgement_ids)):
            if index % 200 == 0:
                guard.checkpoint()
            self.judgements.discard(judgement_id)
        return len(judgement_ids)

    def _release_unreferenced(self, guard: LeaseGuard) -> tuple[int, int]:
        """每轮释放"全部观测已被回执覆盖、且不再被任何存储中判断引用"的交付——不靠从判断反查交付。

        从判断反查收不到两类交付：内容重复投递的、崩溃重放时判断已删的；它们会一直留在存储里。被隔离判断
        引用的交付仍然保留（可见、等人处置），封口前沿按覆盖记录把它们排除在外。
        """

        guard.checkpoint()
        covered = self.coverage.covered_observation_ids(self._now())
        guard.checkpoint()
        still_referenced: set[str] = set()
        for position, raw in enumerate(self.judgements.list()):
            if position % 500 == 0:
                guard.checkpoint()
            still_referenced.update(str(value) for value in raw.get("observation_ids", ()))
        released = retained = 0
        for index, envelope in enumerate(self.observations.list()):
            if index % 200 == 0:
                guard.checkpoint()
            ids = {observation.observation_id for observation in envelope.batch.observations}
            if ids <= covered and not (ids & still_referenced):
                self.observations.discard(envelope.source_id)
                released += 1
            else:
                retained += 1
        return released, retained

    def _expire(self, now: datetime, guard: LeaseGuard | None = None) -> dict[str, int]:
        """回执、覆盖索引、消费账本按同一个窗口整块过期；窗口 = 上游最大补发跨度。"""

        before = now - timedelta(days=self.coverage.window_days)
        # 覆盖记录只在其观测已释放后才过期：交付还在存储里，它的"已融合"就还得答得出来。
        stored_ids: set[str] = set()
        for position, envelope in enumerate(self.observations.list()):
            if guard is not None and position % 200 == 0:
                guard.checkpoint()
            stored_ids.update(item.observation_id for item in envelope.batch.observations)
        stored = frozenset(stored_ids)
        return {
            "coverage_expired": self.coverage.expire(now, retain=stored),
            "receipts_expired": self.receipts.expire(before),
            "ledger_entries_expired": self.ledger.expire(before),
            "stored_observations": len(stored),
        }

    # ── 机械读取 ─────────────────────────────────────────────────────────────────────

    def _now(self) -> datetime:
        value = self.clock()
        if value.utcoffset() is None:
            raise BehaviorReductionError("reduction clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    def _frontier_cutoff(self, guard: LeaseGuard | None = None) -> datetime | None:
        """还没写成判断的观测把封口视界往回拉到哪里；一切观测都已融合完成时为 None。

        判据：**尚未被任何融合回执覆盖的观测**的最早 ``available_at``。两条 lane 都是先落凭据（观测）、再落判断、
        最后落回执与覆盖记录，凭据已在而回执未落的（正在判、做到一半崩了）都没有覆盖记录，一网打尽。
        每轮列一遍观测存储与覆盖记录：两者都随释放保持小量（观测发布即释放，覆盖按释放条件删，26 号清单 L2、6.1），
        解码按文件缓存（6.2）。释放的顺序保证"已释放的观测都已有覆盖记录"，否则旧观测会被误判为未融合、把封口视界
        永远拖在过去。

        顺手认出"封口视界被哪几份观测钉住"（``pinned``，裁定 94 第 8 条）：一份交付里没被覆盖的观测，比它晚的观测已有被回执覆盖
        的——正在判的那一轮之后不会有更晚的已覆盖观测，所以不是"还在判"，而是它的回执不会来了。几份各卡在不同时刻的逐份都报；
        比所有已覆盖观测都晚、还没覆盖的（还在判）不报。不加时限、不隔离（两条 lane 逐份都落回执由融合保证），只报出是哪份、
        从哪一刻起钉住，交健康面等人看。
        """

        covered = self.coverage.covered_observation_ids(self._now())
        #: 每份交付里最早一条没被覆盖的观测
        waiting: dict[str, datetime] = {}
        latest_covered: datetime | None = None
        for position, envelope in enumerate(self.observations.list()):
            if guard is not None and position % 200 == 0:
                guard.checkpoint()
            for observation in envelope.batch.observations:
                at = observation.available_at
                if observation.observation_id in covered:
                    latest_covered = at if latest_covered is None or at > latest_covered else latest_covered
                    continue
                earliest = waiting.get(envelope.source_id)
                if earliest is None or at < earliest:
                    waiting[envelope.source_id] = at
        self.pinned = tuple(
            sorted(
                (
                    PinnedFrontier(source_id, since)
                    for source_id, since in waiting.items()
                    if latest_covered is not None and latest_covered > since
                ),
                key=lambda pinned: (pinned.since, pinned.source_id),
            )
        )
        return min(waiting.values()) if waiting else None


def _unsealed_days(
    pending: PendingJudgements, ready_indexes: Sequence[int], ready_gaps: Sequence[ReducibleJudgement]
) -> frozenset[date]:
    """仍有未封口的判断（行为或空白段）落在哪些本地日历日：派生层不该归组还会长的日子。"""

    ready = set(ready_indexes)
    ready_gap_ids = {item.judgement_id for item in ready_gaps}
    days = {record.started_at.date() for index, record in enumerate(pending.occurrences) if index not in ready}
    days |= {item.started_at.date() for item in pending.gaps if item.judgement_id not in ready_gap_ids}
    return frozenset(days)


__all__ = ["BehaviorReductionReport", "BehaviorReductionRunner", "PinnedFrontier"]
