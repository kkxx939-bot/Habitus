"""归约写入层的编排：扫描 → 组链 → 封口 → 定格（stage）→ 确定性落盘 → 消费账本。

抄融合 ``StagedFusion`` 的检查点形状（死规则⑤）：kind 归一、时间戳这些会随时间漂移的输入全部
发生在 stage 之前；stage 之后只有确定性落盘——检查点里存的就是将要落盘的最终 payload（时间已是
ISO 字符串）与文档时间戳，崩溃重试逐字节重放，落盘器的同字节幂等保证不撞车。

运行节奏是运维决定（用户裁定 5 分钟一轮 sweep，由组合根/调度器调用 ``run_once``，本层不自带
定时器）；同样的存储内容，无论哪一轮扫到，产出的文档逐字节相同。

树的单一写入口：本 runner 是 ``BehaviorDocumentWriter`` 的唯一调用方——观测 → 融合 → 归约，
没有旁路。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from behavior.document import BehaviorDocumentMetadata
from behavior.document.config import BehaviorDocumentLimitError
from behavior.document.link import BehaviorLinkType, BehaviorStoredLink
from behavior.editor.writer import BehaviorDocumentWriter
from behavior.fusion.config import FUSION_CONTEXT_LOOKBACK_SECONDS
from behavior.fusion.coverage import BehaviorCoverageIndex
from behavior.fusion.receipt_store import BehaviorFusionReceiptStore
from behavior.fusion.store import BehaviorJudgementStore
from behavior.kinds.model import BehaviorKindRegistry
from behavior.kinds.rebuild import BehaviorKindRebuildReport, rebuild_registry
from behavior.kinds.resolver import BehaviorKindRequest, BehaviorKindResolver
from behavior.kinds.store import BehaviorKindStore
from behavior.kinds.vectors import (
    BehaviorKindVectorError,
    BehaviorKindVectorIndex,
    BehaviorKindVectorStore,
)
from behavior.model import BehaviorAddress, BehaviorKind
from behavior.observation import BehaviorObservation, BehaviorObservationStore
from behavior.reduction.chains import ChainAssembly, assemble_chains
from behavior.reduction.errors import BehaviorReductionBusyError, BehaviorReductionError
from behavior.reduction.ledger import BehaviorReductionEntry, BehaviorReductionLedger
from behavior.reduction.payloads import (
    REDUCTION_VERSION,
    UNREADABLE_GAP_KIND,
    chain_address,
    gap_payload,
    occurrence_payload,
)
from behavior.reduction.record import ReducibleJudgement, parse_judgement_record
from behavior.reduction.sealing import (
    closed_under_links,
    seal_horizon,
    sealed_chain_indexes,
    sealed_gaps,
)
from behavior.schema.model import BehaviorSchemaError
from behavior.semantic.refresher import BehaviorSemanticRefresher
from behavior.tree import BehaviorTree, BehaviorTreeIntegrityError
from behavior.uri import BehaviorURI
from foundation.integrity import canonical_digest
from infrastructure.store.contracts.lock import LockStore
from infrastructure.store.contracts.path_lock import LeaseGuard, PathLock
from infrastructure.store.filesystem import atomic_replace_bytes, read_regular_bytes

_MAX_CHECKPOINT_BYTES = 67_108_864
_CHECKPOINT_NAME = "staged.json"
# 待刷新的日目录集合：与检查点分开耐久。刷新失败不再扣住检查点（那会让一次 LLM 抖动后的 merge
# 撞出永久冲突），而是把日子记在这里，每轮 sweep 逐日重试、成功一天清一天。
_REFRESH_PENDING_NAME = "refresh_pending.json"
_MAX_REFRESH_PENDING_BYTES = 1_048_576
_SWEEP_LOCK_TTL_SECONDS = 600


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

    @property
    def published_documents(self) -> int:
        return self.published_occurrences + self.published_gaps


@dataclass(frozen=True)
class BehaviorKindMergeReport:
    """一次词表合并 + 树上重打的可观测结果。"""

    source: str
    target: str
    restamped: int
    days: tuple[date, ...]
    signals: tuple[str, ...]


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
        kind_store: BehaviorKindStore,
        kind_resolver: BehaviorKindResolver,
        ledger: BehaviorReductionLedger,
        kind_vectors: BehaviorKindVectorStore | None = None,
        semantic_refresher: BehaviorSemanticRefresher | None = None,
        clock: Callable[[], datetime] | None = None,
        context_lookback_seconds: float = FUSION_CONTEXT_LOOKBACK_SECONDS,
        coverage: BehaviorCoverageIndex | None = None,
    ) -> None:
        """``context_lookback_seconds`` 必须与融合 runner 实际使用的值一致（同一配置源）——
        融合"还能续"与归约"已封口"是同一个窗口的两面，各配一个数会静默分叉。"""

        if not isinstance(judgements, BehaviorJudgementStore):
            raise TypeError("judgements must be BehaviorJudgementStore")
        if not isinstance(observations, BehaviorObservationStore):
            raise TypeError("observations must be BehaviorObservationStore")
        if not isinstance(receipts, BehaviorFusionReceiptStore):
            raise TypeError("receipts must be BehaviorFusionReceiptStore")
        if not isinstance(tree, BehaviorTree):
            raise TypeError("tree must be BehaviorTree")
        if not isinstance(kind_store, BehaviorKindStore):
            raise TypeError("kind_store must be BehaviorKindStore")
        if not isinstance(kind_resolver, BehaviorKindResolver):
            raise TypeError("kind_resolver must be BehaviorKindResolver")
        if kind_vectors is not None and not isinstance(kind_vectors, BehaviorKindVectorStore):
            raise TypeError("kind_vectors must be BehaviorKindVectorStore or None")
        if not isinstance(ledger, BehaviorReductionLedger):
            raise TypeError("ledger must be BehaviorReductionLedger")
        if semantic_refresher is not None and not isinstance(
            semantic_refresher, BehaviorSemanticRefresher
        ):
            raise TypeError("semantic_refresher must be BehaviorSemanticRefresher or None")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        if (
            isinstance(context_lookback_seconds, bool)
            or not isinstance(context_lookback_seconds, int | float)
            or context_lookback_seconds <= 0
        ):
            raise ValueError("context_lookback_seconds must be a positive number")
        self.judgements = judgements
        self.observations = observations
        self.receipts = receipts
        self.tree = tree
        self.lock_store = lock_store
        self.kind_store = kind_store
        self.kind_resolver = kind_resolver
        self.kind_vectors = kind_vectors
        self.ledger = ledger
        self.semantic_refresher = semantic_refresher
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.context_lookback_seconds = float(context_lookback_seconds)
        if coverage is not None and not isinstance(coverage, BehaviorCoverageIndex):
            raise TypeError("coverage must be BehaviorCoverageIndex")
        self.coverage = coverage or BehaviorCoverageIndex(observations.root)
        self._checkpoint_path = ledger.root / _CHECKPOINT_NAME
        self._path_lock = PathLock(lock_store)
        digest = hashlib.sha256(str(tree.root).encode("utf-8")).hexdigest()[:24]
        self._sweep_lock_key = f"behavior-reduction:{digest}"

    async def run_once(self) -> BehaviorReductionReport:
        """执行一轮归约；先重放遗留检查点，再归约新封口的链与空白段。

        全程持一把 behavior-root 级的**租约**：单写入方是机械保证不是运维假设——两个归约
        进程并跑会各自 stage 出时间戳不同的同址文档，互相把对方的检查点变成永久撞车。锁被
        占用时本轮直接失败（TimeoutError），下一轮 sweep 自然重试。

        刻意**不**把整轮包进 fenced：SQLite 锁实现的 fenced 会跨整个临界区持一个未提交写事务
        ——里面的 writer 文档锁 acquire 撞它即死锁，且共享锁库被占住整轮（评审实测的接线级
        阻塞）。互斥由租约承担、阶段间 ``guard.checkpoint()`` 续约；围栏只围检查点文件的两次
        小写；树文档的写入由 writer 自己的逐文档 fenced 保护。
        """

        try:
            acquired = self._path_lock.acquire(
                self._sweep_lock_key, ttl_seconds=_SWEEP_LOCK_TTL_SECONDS
            )
        except TimeoutError as exc:
            # 只有**这里**的超时是"锁被占"；正文里的 TimeoutError（续约失败、文档锁竞争）
            # 是真故障，不许被归因成让路。
            raise BehaviorReductionBusyError(str(exc)) from exc
        with acquired as guard:
            return await self._run_locked(guard)

    async def _run_locked(self, guard: LeaseGuard) -> BehaviorReductionReport:
        self._kind_signals: list[str] = []
        self._sweep_signals: list[str] = []
        replayed, replayed_days = self._replay_checkpoint(guard)
        now = self._now()
        self._expire(now)
        # 上一轮没刷成的日子并进来（merge/rebuild 留下的也在这里）
        replayed_days = replayed_days | self._pending_refresh_days()
        consumed = self.ledger.consumed_judgement_ids()
        records = []
        quarantined: list[str] = []
        for raw in self.judgements.list():
            if raw.get("judgement_id") in consumed:
                continue
            # 坏记录单条隔离：一条污染不许瘫痪整轮归约（判断存储无删除，整轮失败=永久停摆）。
            # 隔离的记录不被消费，每轮都会再次报出——持续可见，等人处置。
            try:
                records.append(parse_judgement_record(raw))
            except BehaviorReductionError as exc:
                quarantined.append(f"judgement {raw.get('judgement_id')} quarantined: {exc}")
        assembly = assemble_chains(tuple(records))
        horizon = seal_horizon(
            now=now,
            frontier_cutoff=self._frontier_cutoff(),
            lookback_seconds=self.context_lookback_seconds,
        )
        ready_indexes = sealed_chain_indexes(assembly, horizon)
        ready_gaps = sealed_gaps(assembly.gaps, horizon)
        guard.checkpoint()
        if not ready_indexes and not ready_gaps:
            refresh_notes = await self._finish_sweep(replayed_days, guard)
            return BehaviorReductionReport(
                replayed_documents=replayed,
                published_occurrences=0,
                published_gaps=0,
                chains_pending=len(assembly.chains),
                dropped_edges=(*assembly.dropped_edges, *quarantined, *refresh_notes, *self._sweep_signals),
                kind_signals=tuple(self._kind_signals),
            )

        # 检查点字节上界的确定性缩批：超限时按封口顺序留前一半重来（留下的下一轮自然处理），
        # 缩批后必须重新做批内引用闭合——否则宿主/目标被砍掉的链会带着悬空依赖落盘。
        active_indexes, active_gaps = ready_indexes, ready_gaps
        while True:
            documents, dropped = await self._stage(
                assembly, active_indexes, active_gaps, now, guard=guard
            )
            checkpoint = {
                "reduction_version": REDUCTION_VERSION,
                "staged_at": now.isoformat(timespec="microseconds"),
                "refresh_days": sorted(
                    day.isoformat()
                    for day in (replayed_days | _document_days(documents))
                ),
                "documents": documents,
            }
            encoded = json.dumps(checkpoint, ensure_ascii=False, sort_keys=True).encode("utf-8")
            # 写入与重放共用同一上界：写得进读不回的检查点 = 永久卡死（对齐融合层
            # "作业放得下而回执放不下"的既有教训）。超限发生在 stage 前，缩批可自愈。
            if len(encoded) <= _MAX_CHECKPOINT_BYTES:
                break
            total = len(active_indexes) + len(active_gaps)
            if total <= 1:
                raise BehaviorReductionError(
                    "a single reduction document exceeds the checkpoint byte bound"
                )
            dropped_count = total - max(total // 2, 1)
            if len(active_gaps) >= dropped_count:
                active_gaps = active_gaps[: len(active_gaps) - dropped_count]
            else:
                keep_chains = len(active_indexes) - (dropped_count - len(active_gaps))
                active_gaps = ()
                active_indexes = closed_under_links(
                    assembly, set(active_indexes[: max(keep_chains, 0)])
                )
        with guard.fenced():
            atomic_replace_bytes(self._checkpoint_path, encoded, artifact_root=self.ledger.root)
        self._publish_checkpoint(checkpoint, guard)
        guard.checkpoint()
        refresh_notes = await self._finish_sweep(replayed_days | _document_days(documents), guard)
        published_occurrences = sum(
            1 for item in documents if item["kind"] == BehaviorKind.OCCURRENCE.value
        )
        return BehaviorReductionReport(
            replayed_documents=replayed,
            published_occurrences=published_occurrences,
            published_gaps=sum(1 for item in documents if item["kind"] == BehaviorKind.GAP.value),
            # skip/defer/缩批留下的链仍未消费，一并计入 pending——报告不许把它们说成已处理。
            chains_pending=len(assembly.chains) - published_occurrences,
            dropped_edges=(*assembly.dropped_edges, *quarantined, *dropped, *refresh_notes, *self._sweep_signals),
            kind_signals=tuple(self._kind_signals),
        )

    async def merge_kinds(self, source: str, target: str) -> BehaviorKindMergeReport:
        """把词表里的 ``source`` 并入 ``target``，并把树上 ``source`` 的 occurrence 重打为 ``target``。

        这是方案⑤"合并道"的落地动作（判定由离线整理交模型做，这里只执行已定的合并）：持 sweep 锁
        （与归约互斥，词表与树在同一把锁下改）；先词表 ``merged`` 落盘，再全量扫树逐条 restamp
        ——两步都幂等（重跑时词表里已无 source、树上已无旧 token），中途崩溃重跑即可补完。
        受影响的日目录概览随后刷新（正文含类型，digest 必变）。预测树不用改，下次夜批读树即得。
        """

        try:
            acquired = self._path_lock.acquire(self._sweep_lock_key, ttl_seconds=_SWEEP_LOCK_TTL_SECONDS)
        except TimeoutError as exc:
            raise BehaviorReductionBusyError(str(exc)) from exc
        with acquired as guard:
            self._require_no_checkpoint("merge_kinds")
            now = self._now()
            signals: list[str] = []
            snapshot = self.kind_store.read()
            registry = snapshot.registry
            # 按"source 是否仍是 token"判幂等：并过之后它只是 target 的别名，token_for 会命中但不该再并。
            if source in registry.tokens and target not in registry.tokens:
                raise BehaviorReductionError(f"merge target is not a registered kind: {target!r}")
            if source in registry.tokens:
                registry = registry.merged(source, target)
                self.kind_store.replace(
                    registry, expected_revision=snapshot.revision, timestamp=now,
                    hits_applied_checkpoint=snapshot.hits_applied_checkpoint,
                )
                signals.append(f"kind_merged {source!r} -> {target!r}")
                vectors, dirty = self._read_kind_vectors(signals)
                if vectors is not None:
                    self._persist_kind_vectors(vectors.retain(registry.names_in_use()), vectors, dirty, signals)
            writer = BehaviorDocumentWriter(self.tree, self.lock_store, clock=lambda: now)
            restamped = 0
            days: set[date] = set()
            for index, document in enumerate(self.tree.iter_documents(BehaviorKind.OCCURRENCE)):
                if index % 200 == 0:
                    guard.checkpoint()
                if document.fields.get("kind_token") != source:
                    continue
                writer.restamp_kind_token(document.address, target)
                restamped += 1
                days.add(document.address.occurred_on)
            self._write_pending_refresh_days(self._pending_refresh_days() | days, guard)
            notes = await self._refresh_semantics(days, guard)
            return BehaviorKindMergeReport(
                source=source, target=target, restamped=restamped, days=tuple(sorted(days)),
                signals=(*signals, *notes),
            )

    async def rebuild_kinds(self) -> BehaviorKindRebuildReport:
        """按树重建词表（补齐 + 账重算 + 向量补算，零模型调用）；持 sweep 锁，与归约互斥。"""

        try:
            acquired = self._path_lock.acquire(self._sweep_lock_key, ttl_seconds=_SWEEP_LOCK_TTL_SECONDS)
        except TimeoutError as exc:
            raise BehaviorReductionBusyError(str(exc)) from exc
        with acquired:
            self._require_no_checkpoint("rebuild_kinds")
            return await rebuild_registry(
                self.tree,
                self.kind_store,
                now=self._now(),
                vectors=self.kind_vectors,
                embedder=self.kind_resolver.embedder,
            )

    async def _finish_sweep(self, days: set[date], guard: LeaseGuard) -> tuple[str, ...]:
        """一轮的收尾，两条返回路径共用，顺序即崩溃安全的依据：

        受影响日先耐久进待刷新集合 → 清检查点（发布完成即清，不再被刷新失败扣住——那会让 merge
        之后的重放撞出永久冲突）→ 逐日刷新（失败留在集合里下轮重试）→ 释放无人引用的交付。
        """

        self._write_pending_refresh_days(self._pending_refresh_days() | days, guard)
        self._clear_checkpoint(guard)
        notes = await self._refresh_semantics(days, guard)
        self._release_unreferenced(guard)
        return notes

    def _clear_checkpoint(self, guard: LeaseGuard) -> None:
        with guard.fenced():
            self._checkpoint_path.unlink(missing_ok=True)

    def _require_no_checkpoint(self, operation: str) -> None:
        """运维动作不许压在悬挂的检查点上：先让 sweep 把它重放干净（否则重打 token 后重放撞冲突）。"""

        if not self._checkpoint_path.exists():
            return
        try:
            encoded = read_regular_bytes(
                self._checkpoint_path, artifact_root=self.ledger.root, max_bytes=_MAX_CHECKPOINT_BYTES
            )
            checkpoint = json.loads(encoded.decode("utf-8"))
            # 判据与 ``_publish_checkpoint`` 的必需字段对齐：两者缺一都不可重放，否则"sweep 报缺
            # staged_at、运维动作却让你先跑 sweep"就是死锁（审计 NEW-6）。
            replayable = (
                isinstance(checkpoint, Mapping)
                and isinstance(checkpoint.get("documents"), list)
                and _is_instant(checkpoint.get("staged_at"))
            )
        except Exception:  # noqa: BLE001 - 只为给出准确的运维指引
            replayable = False
        if replayable:
            raise BehaviorReductionBusyError(
                f"{operation} refused: a staged reduction checkpoint is pending; run a sweep first"
            )
        raise BehaviorReductionError(
            f"{operation} refused: the staged reduction checkpoint at {self._checkpoint_path} is not "
            "replayable (corrupt); inspect and delete it manually before retrying"
        )

    @property
    def _refresh_pending_path(self) -> Path:
        return self.ledger.root / _REFRESH_PENDING_NAME

    def _pending_refresh_days(self) -> set[date]:
        try:
            encoded = read_regular_bytes(
                self._refresh_pending_path, artifact_root=self.ledger.root, max_bytes=_MAX_REFRESH_PENDING_BYTES
            )
        except FileNotFoundError:
            return set()
        try:
            raw = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BehaviorReductionError("reduction refresh_pending file is not decodable") from exc
        if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
            raise BehaviorReductionError("reduction refresh_pending file shape is invalid")
        try:
            return {date.fromisoformat(item) for item in raw}
        except ValueError as exc:
            raise BehaviorReductionError("reduction refresh_pending file holds a non-ISO date") from exc

    def _write_pending_refresh_days(self, days: set[date], guard: LeaseGuard) -> None:
        with guard.fenced():
            if not days:
                self._refresh_pending_path.unlink(missing_ok=True)
                return
            encoded = json.dumps(sorted(day.isoformat() for day in days)).encode("utf-8")
            atomic_replace_bytes(self._refresh_pending_path, encoded, artifact_root=self.ledger.root)

    # ── stage：全部易漂移输入在此定格 ─────────────────────────────────────────────────

    async def _stage(
        self,
        assembly: ChainAssembly,
        ready_indexes: tuple[int, ...],
        ready_gaps: tuple[ReducibleJudgement, ...],
        now: datetime,
        *,
        guard: LeaseGuard | None = None,
    ) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
        """物化本轮全部文档；返回值即检查点内容（JSON 可序列化、逐字节确定）。"""

        staged_at = now.isoformat(timespec="microseconds")
        dropped: list[str] = []
        ready_chains = [(index, assembly.chains[index]) for index in ready_indexes]

        entries = self.ledger.load()
        ledger_uris = frozenset(entry.uri for entry in entries)
        occupied = set(ledger_uris)
        # 跨批链接只能指向 occurrence：gap 节点不可作关系目标（没读懂段无法当并行/因果的
        # 一方），而 writer 只查目标存在不查类型——不在这里过滤，指向 gap 的链接会在检查点
        # 之后才被 writer 硬拒，整条归约永久卡死。
        uri_by_judgement = {
            judgement_id: entry.uri
            for entry in entries
            if entry.kind == BehaviorKind.OCCURRENCE.value
            for judgement_id in entry.judgement_ids
        }
        consumed_gap_targets = {
            judgement_id
            for entry in entries
            if entry.kind == BehaviorKind.GAP.value
            for judgement_id in entry.judgement_ids
        }
        batch_gap_ids = {item.judgement_id for item in assembly.gaps}

        # kinds 归一先于命名（只用链头名字，不依赖地址）——写入层唯一的 LLM 触点。
        tokens = await self._resolve_kind_tokens(
            (chain.head for _, chain in ready_chains), now, guard=guard
        )
        sources = {ref for _, chain in ready_chains for item in chain.consumed for ref in item.source_refs}
        observation_index = self._observation_index(sources) if sources else {}

        # 撞车消歧（死规则②）：按 (链头 evidence_ready_at, 链头身份) 定序，后序加确定性序号后缀，
        # 原始名照存语义面。**先物化再占位**：payload 物化失败（坏名字、basis 引用已丢观测）的链只
        # 跳过、不占地址——否则它占了基名，幸存者却带着 -2/original_name 落盘，被预测源当"已知重复"
        # 整条丢掉。占用集合 = 账本地址 ∪ 树上已存在的地址（账本按窗口过期，树不会）∪ 本批。
        naming: dict[int, tuple[str, str | None, str]] = {}
        payloads: dict[int, dict[str, Any]] = {}
        skipped_chains: set[int] = set()
        for position, (index, chain) in enumerate(
            sorted(
                ready_chains,
                key=lambda pair: (pair[1].head.evidence_ready_at, pair[1].head.judgement_id),
            )
        ):
            if guard is not None and position % 50 == 0:
                # 周尺度实测：9,270 条链的命名+物化是一个单阶段长循环，只在阶段边界续约会踩满
                # 600s 租约 TTL（另一条线的冻结周重放实测）。长循环内按条数周期性续约。
                guard.checkpoint()
            base = str(chain.head.behavior)
            token = tokens.get(base)
            if token is None:
                skipped_chains.add(index)
                dropped.append(f"chain {chain.chain_digest} skipped: behavior name is not addressable")
                continue
            try:
                occurrence_payload(
                    chain,
                    name=base,
                    original_name=None,
                    kind_token=token,
                    observations=observation_index,
                )
                candidate, ordinal = base, 1
                while self._address_taken(chain_address(chain, candidate), occupied):
                    ordinal += 1
                    candidate = f"{base}-{ordinal}"
                uri = str(BehaviorURI.from_address(chain_address(chain, candidate)))
                payloads[index] = occurrence_payload(
                    chain,
                    name=candidate,
                    original_name=base if ordinal > 1 else None,
                    kind_token=token,
                    observations=observation_index,
                )
            except BehaviorTreeIntegrityError:
                raise  # 树本身不可安全检视（别名冲突、符号链接）：整轮失败、可见告警，不当坏名字吞
            except (TypeError, ValueError, BehaviorReductionError) as exc:
                skipped_chains.add(index)
                dropped.append(f"chain {chain.chain_digest} skipped: {exc}")
                continue
            occupied.add(uri)
            naming[index] = (candidate, base if ordinal > 1 else None, uri)

        documents: list[dict[str, Any]] = []

        # gap 先落（无链接）。gap 地址的叶名是枚举类型、无法加后缀：批内**同一微秒同偏移**
        # 起点的空白机械合并成
        # 一个节点（溯源并集、终点取最大）；撞上账本里既有空白时，本段记账指向既有节点——
        # 空白事实已在树上有代表，不重复、不丢账、不卡队列。
        merged_gaps: dict[str, list[ReducibleJudgement]] = {}
        for record in ready_gaps:
            address = BehaviorAddress.gap(
                record.started_at.date(), UNREADABLE_GAP_KIND, record.started_at
            )
            merged_gaps.setdefault(str(BehaviorURI.from_address(address)), []).append(record)
        for uri in sorted(merged_gaps):
            group = merged_gaps[uri]
            judgement_ids = sorted(item.judgement_id for item in group)
            entry = {
                "chain_digest": canonical_digest({"judgement_ids": judgement_ids}),
                "kind": BehaviorKind.GAP.value,
                "uri": uri,
                "judgement_ids": judgement_ids,
                "staged_at": staged_at,
                "reduction_version": REDUCTION_VERSION,
            }
            if uri in ledger_uris or self.tree.exists(BehaviorURI.parse(uri).to_address()):
                dropped.append(
                    f"gap at {uri} already published; consuming {judgement_ids} by reference"
                )
                documents.append({"kind": "ledger-only", "payload": None, "links": [], "ledger": entry})
                continue
            payload = self._merged_gap_payload(group, chain_digest=str(entry["chain_digest"]))
            documents.append(
                {"kind": BehaviorKind.GAP.value, "payload": payload, "links": [], "ledger": entry}
            )

        # occurrence 按 order_key 升序落盘：批内前向边只指向更小的键，目标必然先落。
        for position, (index, chain) in enumerate(
            sorted(ready_chains, key=lambda pair: pair[1].order_key)
        ):
            if guard is not None and position % 50 == 0:
                guard.checkpoint()
            if index in skipped_chains:
                continue
            name, original_name, uri = naming[index]
            payload = payloads[index]
            links: list[list[str]] = []
            deferred_by_skip = False
            for kind, target_index in assembly.cross_links_of(index):
                target = naming.get(target_index)
                # 目标不在批内（缩批砍掉）、命名失败、或命名成功后在 payload 物化时被跳过——
                # 三种情况都推迟本链，不带着悬空引用落盘（评审实测过：悬空链接进检查点 =
                # 重放永败、整条归约卡死）。目标 order_key 更小、必先于本链处理，故此时可知。
                if target is None or target_index in skipped_chains:
                    deferred_by_skip = True
                    dropped.append(
                        f"chain {chain.chain_digest} deferred: its link target was skipped"
                    )
                    break
                links.append([kind, target[2]])
            if deferred_by_skip:
                skipped_chains.add(index)
                continue
            for kind, target_id in chain.semantic_edges:
                if target_id in assembly.chain_of or target_id in batch_gap_ids:
                    continue  # 批内边已规范化处理；指向本批没读懂段的边已在组装时作废并记录
                target_uri = uri_by_judgement.get(target_id)
                if target_uri is None:
                    if target_id in consumed_gap_targets:
                        reason = "target is a published gap node, not linkable"
                    elif target_id in assembly.quarantined_ids:
                        # 目标链被隔离（时间倒挂/成环）：隔离是等人处置的滞留态，无限推迟好链
                        # 不可取——丢边但把理由说准，处置后如需补关系由读侧/语义层自行判断。
                        reason = "target belongs to a quarantined chain"
                    else:
                        reason = "target is neither reducible nor consumed"
                    dropped.append(
                        f"{kind} from chain {chain.chain_digest} dropped: {reason} ({target_id})"
                    )
                elif [kind, target_uri] not in links:
                    links.append([kind, target_uri])
            entry = {
                "chain_digest": chain.chain_digest,
                "kind": BehaviorKind.OCCURRENCE.value,
                "uri": uri,
                "judgement_ids": [item.judgement_id for item in chain.consumed],
                "staged_at": staged_at,
                "reduction_version": REDUCTION_VERSION,
            }
            documents.append(
                {
                    "kind": BehaviorKind.OCCURRENCE.value,
                    "payload": payload,
                    "links": links,
                    "ledger": entry,
                }
            )
        documents = self._publishable(documents, now, dropped, guard)
        return documents, tuple(dropped)

    async def _resolve_kind_tokens(
        self,
        heads: Iterable[ReducibleJudgement],
        now: datetime,
        *,
        guard: LeaseGuard | None,
    ) -> dict[str, str]:
        """kinds 归一——写入层唯一的 LLM 触点；按批 CAS 落盘并续租。只定"名字 → token"，不记账。

        每条链头给出名字与证据（判断的 summary）。已知名字走确定性快路径；未知名字由 resolver 按批
        判定（embedding 候选 + 模型判"是不是同一件事"），每批落一次词表——崩溃重试时先前已记入词表
        的名字走快路径，"先落词表、再落检查点"的顺序保证重试不漂移。命中账在发布时记
        （``_record_kind_hits``），与树上的 occurrence 一一对应。
        """

        signals = self._kind_signals
        requests: list[BehaviorKindRequest] = []
        for head in heads:
            try:
                requests.append(BehaviorKindRequest(name=str(head.behavior), evidence=head.summary))
            except (TypeError, ValueError):
                # 绕过融合守卫写入的坏名字：不进归一，命名阶段按链隔离并留信号（tokens 里没有它）。
                continue
        snapshot = self.kind_store.read()
        registry, revision = snapshot.registry, snapshot.revision
        vectors, vectors_dirty = self._read_kind_vectors(signals)
        vectors_before = vectors
        tokens: dict[str, str] = {}
        async for batch in self.kind_resolver.resolve_batches(requests, registry, vectors=vectors):
            tokens.update(batch.tokens)
            signals.extend(batch.signals)
            for name in batch.created:
                signals.append(f"kind_created {name!r} label={batch.registry.label_of(name)!r}")
            registry, vectors = batch.registry, batch.vectors
            registry, revision = self._persist_kinds(registry, revision, now)
            if guard is not None:
                guard.checkpoint()
        self._persist_kind_vectors(vectors, vectors_before, vectors_dirty, signals)
        return tokens

    def _record_kind_hits(self, hits: Sequence[tuple[str, date]], checkpoint_id: str, now: datetime) -> None:
        """发布时记命中账；幂等键是检查点内容摘要：词表记着"已应用到哪个检查点"，重放同一检查点跳过。

        账的 owner 按 ``token_for`` 找（树上的 token 可能已被 merge 成别名）；找不到就补登记（留复核记号）；
        词表满了只留信号。随后按数据时钟删到期条目：数据时钟 = 本批最新行为日，且钳在墙钟当日之内
        （一条 2099 的坏时间戳不能把词表删空）；本批命中过的 token 不参与过期。删除只动词表与向量旁册，
        树上 occurrence 不动。
        """

        if not hits:
            return
        signals = self._kind_signals
        config = self.kind_resolver.config
        snapshot = self.kind_store.read()
        if snapshot.hits_applied_checkpoint == checkpoint_id:
            signals.append(f"kind_hits_skipped checkpoint {checkpoint_id[:12]} already applied")
            return
        registry, revision = snapshot.registry, snapshot.revision
        hit_now: set[str] = set()
        for token, day in hits:
            owner = registry.token_for(token)
            if owner is None:
                if registry.kind_count >= config.max_kinds:
                    signals.append(f"kind_registry_full {token!r} hit not recorded (missing at publish time)")
                    continue
                # 归一之后、发布之前词表被重建或该 token 已过期：树上的 token 是真相，补登记。
                registry = registry.with_new_kind(token, review_reason="reregistered")
                signals.append(f"kind_reregistered {token!r} (missing at publish time)")
                owner = token
            registry = registry.with_hit(owner, day)
            hit_now.add(owner)
        clock = min(max(day for _, day in hits), now.astimezone(timezone.utc).date())
        expired = tuple(
            token
            for token in registry.expired(on=clock, base_days=config.base_days, gap_multiplier=config.gap_multiplier)
            if token not in hit_now
        )
        for token in expired:
            entry = registry.entry_of(token)
            signals.append(
                f"kind_expired {token!r} last hit {entry.last_hit_day} "
                f"(hit_days={entry.hit_days_total}, max_gap={entry.max_gap_days})"
            )
            registry = registry.without(token)
        registry, revision = self._persist_kinds(registry, revision, now, hits_applied_checkpoint=checkpoint_id)
        if expired and self.kind_vectors is not None:
            vectors, dirty = self._read_kind_vectors(signals)
            assert vectors is not None
            # 旁册按名字键、条目间可能同名：按"谁还在用"收，不按 (token, label) 直删。
            self._persist_kind_vectors(vectors.retain(registry.names_in_use()), vectors, dirty, signals)

    def _persist_kinds(
        self,
        registry: BehaviorKindRegistry,
        revision: int,
        now: datetime,
        *,
        hits_applied_checkpoint: str | None = None,
    ) -> tuple[BehaviorKindRegistry, int]:
        """词表变了（或要更新已应用检查点标记）就 CAS 落盘；返回落盘后的 (registry, revision)。

        不传 ``hits_applied_checkpoint`` 时保留旧标记——归一路径落盘不能把发布路径的幂等键抹掉。
        """

        current = self.kind_store.read()
        marker = current.hits_applied_checkpoint if hits_applied_checkpoint is None else hits_applied_checkpoint
        if current.registry == registry and marker == current.hits_applied_checkpoint:
            return current.registry, current.revision
        written = self.kind_store.replace(
            registry, expected_revision=revision, timestamp=now, hits_applied_checkpoint=marker
        )
        return written.registry, written.revision

    def _read_kind_vectors(self, signals: list[str]) -> tuple[BehaviorKindVectorIndex | None, bool]:
        """向量旁册是派生物：读不了就按空索引走并标脏（本轮结束必重写），留信号，不阻塞归约。"""

        if self.kind_vectors is None:
            return None, False
        try:
            return self.kind_vectors.read(), False
        except BehaviorKindVectorError as exc:
            signals.append(f"kind_vectors_unreadable {exc}; rebuilding from empty")
            return self.kind_vectors.empty(), True

    def _persist_kind_vectors(
        self,
        vectors: BehaviorKindVectorIndex | None,
        before: BehaviorKindVectorIndex | None,
        dirty: bool,
        signals: list[str],
    ) -> None:
        if vectors is None or self.kind_vectors is None or not (dirty or vectors is not before):
            return
        try:
            self.kind_vectors.replace(vectors)
        except BehaviorKindVectorError as exc:
            signals.append(f"kind_vectors_not_persisted {exc}")

    def _address_taken(self, address: BehaviorAddress, occupied: set[str]) -> bool:
        """地址被账本、本批或树上既有文档占用。树是最终真相：账本按窗口过期，树不会。"""

        return str(BehaviorURI.from_address(address)) in occupied or self.tree.exists(address)

    @staticmethod
    def _merged_gap_payload(group: list[ReducibleJudgement], *, chain_digest: str) -> dict[str, Any]:
        """同地址空白段的机械合并：起点同刻（合并前提）、终点取最大；溯源只留账本条目身份。"""

        first = min(group, key=lambda item: item.judgement_id)
        payload = gap_payload(first, chain_digest=chain_digest)
        if len(group) > 1:
            ended = max((item.last_observed_at for item in group), key=_as_instant)
            payload["ended_at"] = ended.isoformat(timespec="microseconds")
        return payload

    # ── 落盘：stage 之后只有确定性动作 ────────────────────────────────────────────────

    def _replay_checkpoint(self, guard: LeaseGuard) -> tuple[int, set[date]]:
        """重放遗留检查点；清除在 ``_run_locked`` 发布完成后进行（语义刷新另有待刷新集合耐久）。"""

        try:
            encoded = read_regular_bytes(
                self._checkpoint_path,
                artifact_root=self.ledger.root,
                max_bytes=_MAX_CHECKPOINT_BYTES,
            )
        except FileNotFoundError:
            return 0, set()
        try:
            checkpoint = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BehaviorReductionError("reduction checkpoint is not decodable") from exc
        self._publish_checkpoint(checkpoint, guard)
        documents = checkpoint.get("documents")
        if not isinstance(documents, list):
            return 0, set()
        days = _document_days(documents)
        raw_days = checkpoint.get("refresh_days")
        if isinstance(raw_days, list):
            days |= {date.fromisoformat(str(item)) for item in raw_days}
        replayed = sum(
            1
            for item in documents
            if isinstance(item, Mapping) and item.get("kind") != "ledger-only"
        )
        return replayed, days

    async def _refresh_semantics(self, days: set[date], guard: LeaseGuard) -> tuple[str, ...]:
        """刷新受影响目录的 L0/L1（仍在 sweep 锁内——刷新器的单写入方前提）；**逐日隔离**。

        摘要是可重建派生物：某一天刷新失败只影响那一天——留在待刷新集合里下轮重试并留信号，
        其余日子照刷、照清。一个超限的日目录不再让整批（含月/年上卷）连坐、也不再让集合只增不减。
        """

        if not days:
            return ()
        if self.semantic_refresher is None:
            self._write_pending_refresh_days(self._pending_refresh_days() - days, guard)
            return ()
        notes: list[str] = []
        remaining = self._pending_refresh_days() | days
        for day in sorted(days):
            guard.checkpoint()
            try:
                await self.semantic_refresher.refresh_days((day,))
            except Exception as exc:  # noqa: BLE001 - 派生物刷新失败一律降级为信号
                notes.append(f"semantic refresh failed for [{day.isoformat()}]: {exc}")
                continue
            remaining.discard(day)
        self._write_pending_refresh_days(remaining, guard)
        return tuple(notes)

    def _publish_checkpoint(self, checkpoint: Mapping[str, Any], guard: LeaseGuard) -> None:
        """把检查点逐字落盘：同字节幂等，故本函数可任意次重入；逐段续约防租约过期。"""

        staged_at_raw = checkpoint.get("staged_at")
        if not isinstance(staged_at_raw, str):
            raise BehaviorReductionError("reduction checkpoint is missing staged_at")
        staged_at = datetime.fromisoformat(staged_at_raw)
        if staged_at.utcoffset() is None:
            raise BehaviorReductionError("reduction checkpoint staged_at must be timezone-aware")
        documents = checkpoint.get("documents")
        if not isinstance(documents, list):
            raise BehaviorReductionError("reduction checkpoint is missing documents")
        writer = BehaviorDocumentWriter(self.tree, self.lock_store, clock=lambda: staged_at)
        hits: list[tuple[str, date]] = []
        for index, item in enumerate(documents):
            if index % 50 == 0:
                guard.checkpoint()
            if not isinstance(item, Mapping):
                raise BehaviorReductionError("reduction checkpoint document must be a mapping")
            entry = BehaviorReductionEntry.from_mapping(item["ledger"])
            if item["kind"] != "ledger-only":
                kind = BehaviorKind(item["kind"])
                links = tuple((str(link[0]), str(link[1])) for link in item.get("links", ()))
                writer.publish(kind, item["payload"], links=links)
                if kind is BehaviorKind.OCCURRENCE and item["payload"].get("original_name") is None:
                    # 撞车消歧记录（original_name 非空）= 已知重复：预测树、语义层、命中账一律不计。
                    payload = item["payload"]
                    hits.append((str(payload["kind_token"]), date.fromisoformat(str(payload["occurred_on"]))))
            self.ledger.append(entry)
        # 命中账与树上的 occurrence 一一对应；幂等键是检查点**内容**摘要：同一检查点只记一次。
        self._record_kind_hits(hits, _checkpoint_identity(documents), staged_at)
        # 树写完、账本写完，原料才释放（顺序是崩溃安全的依据：重放路径重新走到这里再补删）。
        self._release(documents, guard)

    # ── 释放：原料被消费后即删，真正的数据只在树上 ─────────────────────────────────────

    def _release(self, documents: list[Any], guard: LeaseGuard) -> None:
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

    def _release_unreferenced(self, guard: LeaseGuard) -> None:
        """每轮释放"全部观测已被回执覆盖、且不再被任何存储中判断引用"的交付——不靠判断反查。

        判断反查（旧实现）收不到三类交付：全段无归属/旁人、内容重复投递、崩溃重放时判断已删的；
        它们留在存储里，覆盖记录按窗口过期后就被当成"未融合"重新入队重跑模型。被隔离判断引用的
        交付仍然保留（可见、等人处置），但入队与封口前沿都按覆盖窗口把它们排除在外。
        """

        covered = self.coverage.covered_observation_ids(self._now())
        still_referenced: set[str] = set()
        for raw in self.judgements.list():
            still_referenced.update(str(value) for value in raw.get("observation_ids", ()))
        for index, envelope in enumerate(self.observations.list()):
            if index % 200 == 0:
                guard.checkpoint()
            ids = {observation.observation_id for observation in envelope.batch.observations}
            if ids <= covered and not (ids & still_referenced):
                self.observations.discard(envelope.source_id)

    def _expire(self, now: datetime) -> None:
        """回执、覆盖索引、消费账本按同一个窗口整块过期；窗口 = 上游最大补发跨度。"""

        before = now - timedelta(days=self.coverage.window_days)
        # 覆盖记录只在其观测已释放后才过期：交付还在存储里，它的"已融合"就还得答得出来。
        stored = frozenset(
            observation.observation_id
            for envelope in self.observations.list()
            for observation in envelope.batch.observations
        )
        self.coverage.expire(now, retain=stored)
        self.receipts.expire(before)
        self.ledger.expire(before)

    # ── 机械读取 ─────────────────────────────────────────────────────────────────────

    def _now(self) -> datetime:
        value = self.clock()
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise BehaviorReductionError("reduction clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)

    def _frontier_cutoff(self) -> datetime | None:
        """未来融合段的 cutoff 下界；一切观测都已融合完成时为 None。

        判据：**尚未被任何融合回执覆盖的观测**的最早 ``available_at``。排队中、已入队未提交、
        还没被入队扫描捡走的观测全都没有回执，一网打尽——不能只看最老未提交作业：补发场景下
        段的 cutoff 不随队列序单调，已交付未入队的观测更是不在队列里。代价是每轮全量扫描
        观测与回执（BHV-LIFECYCLE-001 的已知欠账）；注意两个存储将来做保留期释放时，必须保证
        "已释放的观测都已有回执覆盖"，否则旧观测会被误判为未融合、把封口视界永远拖在过去。
        """

        covered = self.coverage.covered_observation_ids(self._now())
        pending: datetime | None = None
        for envelope in self.observations.list():
            for observation in envelope.batch.observations:
                if observation.observation_id in covered:
                    continue
                if pending is None or observation.available_at < pending:
                    pending = observation.available_at
        return pending

    def _publishable(
        self,
        documents: list[dict[str, Any]],
        now: datetime,
        dropped: list[str],
        guard: LeaseGuard | None = None,
    ) -> list[dict[str, Any]]:
        """stage 末端的干跑校验：落盘期只允许**已验证会成功**的确定性动作。

        任何会在 publish 期抛错的文档（schema 校验、链接形状），若放进检查点就会变成
        "检查点已写、重放永败"的永久卡死；在这里按文档剔除并留信号，其余照常落。链接目标
        被剔除的文档一并推迟——不带悬空引用落盘。
        """

        metadata = BehaviorDocumentMetadata.initial(now)
        valid: list[dict[str, Any]] = []
        valid_uris: set[str] = set()
        failed_uris: set[str] = set()
        for position, item in enumerate(documents):
            if guard is not None and position % 50 == 0:
                guard.checkpoint()
            if item["kind"] == "ledger-only":
                valid.append(item)
                continue
            uri = str(item["ledger"]["uri"])
            if any(str(link[1]) in failed_uris for link in item["links"]):
                failed_uris.add(uri)
                dropped.append(f"document {uri} deferred: its link target failed staging")
                continue
            try:
                stored_links = tuple(
                    BehaviorStoredLink(
                        from_uri=BehaviorURI.parse(uri),
                        to_uri=BehaviorURI.parse(str(link[1])),
                        link_type=BehaviorLinkType(str(link[0])),
                    )
                    for link in item["links"]
                )
                # 目标存在性双保险：writer 在 publish 期硬拒不存在的目标，而那时检查点已落盘
                # =永久卡死。目标要么是本批已通过干跑的文档、要么已在树上——都不是就在这里拦。
                for link in stored_links:
                    if str(link.to_uri) in valid_uris:
                        continue
                    if not self.tree.exists(link.to_uri.to_address()):
                        raise BehaviorReductionError(
                            f"link target does not exist anywhere: {link.to_uri}"
                        )
                try:
                    self.tree.document_codec.build(
                        BehaviorKind(item["kind"]),
                        item["payload"],
                        metadata=metadata,
                        links=stored_links,
                    )
                except BehaviorDocumentLimitError:
                    # 最后一道保险：去掉死引用之后仍超限（要两千步以上的链）——截断 basis 中段、
                    # 留信号继续发布，不让一个文档打掉整轮。goal/summary/起止/状态/links 不动，
                    # 它们才是下游真正读的东西；丢的只是步骤明细的中段，首尾与总步数保留。
                    truncated = _truncate_basis(item["payload"])
                    if truncated is None:
                        raise
                    dropped.append(
                        f"document {uri} truncated: basis {len(item['payload']['basis'])} steps "
                        f"-> {len(truncated['basis'])} (document exceeded its byte bound)"
                    )
                    item["payload"] = truncated
                    self.tree.document_codec.build(
                        BehaviorKind(item["kind"]),
                        truncated,
                        metadata=metadata,
                        links=stored_links,
                    )
            except (BehaviorReductionError, BehaviorSchemaError, BehaviorDocumentLimitError, TypeError, ValueError) as exc:
                failed_uris.add(uri)
                dropped.append(f"document {uri} skipped: fails validation before stage ({exc})")
                continue
            valid.append(item)
            valid_uris.add(uri)
        return valid

    def _observation_index(self, source_refs: Iterable[str]) -> dict[str, BehaviorObservation]:
        index: dict[str, BehaviorObservation] = {}
        for source_ref in sorted(set(source_refs)):
            envelope = self.observations.read(source_ref)
            if envelope is None:
                # 交付已不在存储里（运维删除、隔离窗口）：不整轮失败。带 basis 的链在物化时会因
                # "步骤引用的观测已不存在"按链跳过；**没有 basis 的链照常发布**（判断本体自带
                # summary/goal，不依赖观测），所以这里留信号是唯一的可见痕迹（审计 NEW-8）。
                self._sweep_signals.append(f"source_delivery_missing {source_ref}")
                continue
            for observation in envelope.batch.observations:
                index.setdefault(observation.observation_id, observation)
        return index


def _as_instant(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


def _is_instant(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return datetime.fromisoformat(value).utcoffset() is not None
    except ValueError:
        return False


def _checkpoint_identity(documents: list[Any]) -> str:
    """检查点的内容身份：本批全部落盘文档的链身份摘要（与 staged_at 无关，冻结时钟下也不撞）。

    occurrence 与 gap 分列——只摘 occurrence 会让两个不同的 gap-only 批算出同一个空摘要，眼下靠
    ``_record_kind_hits`` 的 ``if not hits: return`` 兜住，但那是隐性依赖（审计 NEW-7）。
    """

    def digests(kind: BehaviorKind) -> list[str]:
        return sorted(
            str(item["ledger"]["chain_digest"])
            for item in documents
            if isinstance(item, Mapping) and item.get("kind") == kind.value
        )

    return canonical_digest(
        {"occurrences": digests(BehaviorKind.OCCURRENCE), "gaps": digests(BehaviorKind.GAP)}
    )


_TRUNCATED_HEAD = 20
_TRUNCATED_TAIL = 5


def _truncate_basis(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """保留 basis 首尾各若干步；已经不可再截时返回 None。"""

    basis = list(payload.get("basis", ()))
    if len(basis) <= _TRUNCATED_HEAD + _TRUNCATED_TAIL:
        return None
    trimmed = dict(payload)
    trimmed["basis"] = basis[:_TRUNCATED_HEAD] + basis[-_TRUNCATED_TAIL:]
    return trimmed


def _document_days(documents: list) -> set[date]:
    """检查点文档触及的本地日历日；ledger-only 条目不改树、不触发刷新。"""

    days: set[date] = set()
    for item in documents:
        if not isinstance(item, Mapping) or item.get("kind") == "ledger-only":
            continue
        payload = item.get("payload")
        if isinstance(payload, Mapping) and isinstance(payload.get("occurred_on"), str):
            days.add(date.fromisoformat(payload["occurred_on"]))
    return days


__all__ = ["BehaviorReductionReport", "BehaviorReductionRunner"]
