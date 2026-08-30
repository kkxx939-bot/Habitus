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
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from behavior.document import BehaviorDocumentMetadata
from behavior.document.config import BehaviorDocumentLimitError
from behavior.document.link import BehaviorLinkType, BehaviorStoredLink
from behavior.editor.writer import BehaviorDocumentWriter
from behavior.fusion.config import FUSION_CONTEXT_LOOKBACK_SECONDS
from behavior.fusion.coverage import BehaviorCoverageIndex
from behavior.fusion.receipt_store import BehaviorFusionReceiptStore
from behavior.fusion.store import BehaviorJudgementStore
from behavior.kinds.model import BehaviorKindLimitError
from behavior.kinds.resolver import BehaviorKindResolver
from behavior.kinds.store import BehaviorKindStore
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
from behavior.tree import BehaviorTree
from behavior.uri import BehaviorURI
from foundation.integrity import canonical_digest
from infrastructure.store.contracts.lock import LockStore
from infrastructure.store.contracts.path_lock import LeaseGuard, PathLock
from infrastructure.store.filesystem import atomic_replace_bytes, read_regular_bytes

_MAX_CHECKPOINT_BYTES = 67_108_864
_CHECKPOINT_NAME = "staged.json"
_SWEEP_LOCK_TTL_SECONDS = 600


@dataclass(frozen=True)
class BehaviorReductionReport:
    """一轮 sweep 的可观测结果；数字与信号，不含语义。"""

    replayed_documents: int
    published_occurrences: int
    published_gaps: int
    chains_pending: int
    dropped_edges: tuple[str, ...]

    @property
    def published_documents(self) -> int:
        return self.published_occurrences + self.published_gaps


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
        replayed, replayed_days = self._replay_checkpoint(guard)
        now = self._now()
        self._expire(now)
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
            refresh_notes = await self._refresh_semantics(replayed_days)
            if not refresh_notes:
                self._clear_checkpoint(guard)
            return BehaviorReductionReport(
                replayed_documents=replayed,
                published_occurrences=0,
                published_gaps=0,
                chains_pending=len(assembly.chains),
                dropped_edges=(*assembly.dropped_edges, *quarantined, *refresh_notes),
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
        refresh_notes = await self._refresh_semantics(
            replayed_days | _document_days(documents)
        )
        if not refresh_notes:
            # 检查点在语义刷新成功后才清：publish 后、刷新前崩溃时，重放路径会连带补刷——
            # "digest 比对自动补齐"的自愈承诺靠这一步成立（受影响日集合随检查点耐久）。
            self._clear_checkpoint(guard)
        published_occurrences = sum(
            1 for item in documents if item["kind"] == BehaviorKind.OCCURRENCE.value
        )
        return BehaviorReductionReport(
            replayed_documents=replayed,
            published_occurrences=published_occurrences,
            published_gaps=sum(1 for item in documents if item["kind"] == BehaviorKind.GAP.value),
            # skip/defer/缩批留下的链仍未消费，一并计入 pending——报告不许把它们说成已处理。
            chains_pending=len(assembly.chains) - published_occurrences,
            dropped_edges=(*assembly.dropped_edges, *quarantined, *dropped, *refresh_notes),
        )

    def _clear_checkpoint(self, guard: LeaseGuard) -> None:
        with guard.fenced():
            self._checkpoint_path.unlink(missing_ok=True)

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

        # 撞车消歧（死规则②）：按 (链头 evidence_ready_at, 链头身份) 定序，后序加确定性序号
        # 后缀，原始名照存语义面。占用集合含账本里的全部既有地址——跨批撞车同样消歧。
        # 地址构造失败（如绕过融合守卫写入的坏名字）按链隔离：跳过本链、留信号、不吞整轮。
        naming: dict[int, tuple[str, str | None, str]] = {}
        skipped_chains: set[int] = set()
        for index, chain in sorted(
            ready_chains,
            key=lambda pair: (pair[1].head.evidence_ready_at, pair[1].head.judgement_id),
        ):
            base = str(chain.head.behavior)
            try:
                candidate, ordinal = base, 1
                while str(BehaviorURI.from_address(chain_address(chain, candidate))) in occupied:
                    ordinal += 1
                    candidate = f"{base}-{ordinal}"
                uri = str(BehaviorURI.from_address(chain_address(chain, candidate)))
            except (TypeError, ValueError) as exc:
                skipped_chains.add(index)
                dropped.append(f"chain {chain.chain_digest} skipped: {exc}")
                continue
            occupied.add(uri)
            naming[index] = (candidate, base if ordinal > 1 else None, uri)

        # kinds 归一只对命名幸存的链做（写入层唯一的 LLM 触点）——坏名字的链已隔离，
        # 不许它在 resolver 处把整轮吞掉。
        tokens = await self._resolve_kind_tokens(
            (str(assembly.chains[index].head.behavior) for index in naming),
            now,
            guard=guard,
            dropped=dropped,
        )
        sources = {
            ref
            for index in naming
            for item in assembly.chains[index].consumed
            for ref in item.source_refs
        }
        observation_index = self._observation_index(sources) if sources else {}

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
            if uri in ledger_uris:
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
        for index, chain in sorted(ready_chains, key=lambda pair: pair[1].order_key):
            if index in skipped_chains:
                continue
            name, original_name, uri = naming[index]
            try:
                payload = occurrence_payload(
                    chain,
                    name=name,
                    original_name=original_name,
                    kind_token=tokens[str(chain.head.behavior)],
                    observations=observation_index,
                )
            except BehaviorReductionError as exc:
                skipped_chains.add(index)
                dropped.append(f"chain {chain.chain_digest} skipped: {exc}")
                continue
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
        documents = self._publishable(documents, now, dropped)
        return documents, tuple(dropped)

    async def _resolve_kind_tokens(
        self,
        names: Iterable[str],
        now: datetime,
        *,
        guard: LeaseGuard | None = None,
        dropped: list[str] | None = None,
    ) -> dict[str, str]:
        """kinds 归一——写入层唯一的 LLM 触点；词表按批 CAS 落盘（瞬态重试在 resolver 内）。

        崩溃重试时先前已记入词表的名字走确定性快路径（``token_for``），所以"先落词表、再落
        检查点"的顺序保证重试不漂移。七天真实数据实测：一次 sweep 近三千个不同名字≈两小时
        串行调用，词表若只在循环结束后落盘一次，中途一次断连就全部作废（BHV-REALDATA-001）——
        所以每 ``_KIND_PERSIST_EVERY`` 个名字落一次、并顺手续 sweep 租约。词表撞顶不再让整轮
        失败：超限的名字暂以原始名作 token 并留信号（词表在微动作粒度下不收敛是折叠问题的症状）。
        """

        snapshot = self.kind_store.read()
        registry = snapshot.registry
        revision = snapshot.revision
        tokens: dict[str, str] = {}
        since_persist = 0
        for name in sorted(set(names)):
            resolution = None
            try:
                # 瞬态错误的有界重试在 resolver 内（它是允许接触模型客户端的一层）
                resolution = await self.kind_resolver.resolve(name, registry)
            except BehaviorKindLimitError:
                resolution = None
            if resolution is None:
                # 词表容量撞顶：超限名字暂以原始名作 token，语义面不丢、统计可事后重打
                # （occurrence 永远保留原始名）。留信号，不让一轮归约整个失败。
                tokens[name] = name
                if dropped is not None:
                    dropped.append(f"kind registry full: {name!r} kept as its own token")
                continue
            registry = resolution.registry
            tokens[name] = resolution.token
            since_persist += 1
            if since_persist >= _KIND_PERSIST_EVERY and registry is not snapshot.registry:
                self.kind_store.replace(registry, expected_revision=revision, timestamp=now)
                snapshot = self.kind_store.read()
                registry = snapshot.registry
                revision = snapshot.revision
                since_persist = 0
                if guard is not None:
                    guard.checkpoint()
        if registry is not snapshot.registry:
            self.kind_store.replace(registry, expected_revision=revision, timestamp=now)
        return tokens

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
        """重放遗留检查点；**不**在这里清除——检查点要活到语义刷新成功之后（自愈依据）。"""

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

    async def _refresh_semantics(self, days: set[date]) -> tuple[str, ...]:
        """落盘后刷新受影响目录的 L0/L1（仍在 sweep 锁内——刷新器的单写入方前提）。

        摘要是可重建派生物：刷新失败降级成信号、不阻塞归约主流程，下一轮 sweep 的 digest
        比对会自动补齐。
        """

        if self.semantic_refresher is None or not days:
            return ()
        try:
            await self.semantic_refresher.refresh_days(days)
        except Exception as exc:  # noqa: BLE001 - 派生物刷新失败一律降级为信号
            listed = ", ".join(sorted(day.isoformat() for day in days))
            return (f"semantic refresh failed for [{listed}]: {exc}",)
        return ()

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
        for index, item in enumerate(documents):
            if index % 50 == 0:
                guard.checkpoint()
            if not isinstance(item, Mapping):
                raise BehaviorReductionError("reduction checkpoint document must be a mapping")
            if item["kind"] != "ledger-only":
                kind = BehaviorKind(item["kind"])
                links = tuple((str(link[0]), str(link[1])) for link in item.get("links", ()))
                writer.publish(kind, item["payload"], links=links)
            self.ledger.append(BehaviorReductionEntry.from_mapping(item["ledger"]))
        # 树写完、账本写完，原料才释放（顺序是崩溃安全的依据：重放路径重新走到这里再补删）。
        self._release(documents, guard)

    # ── 释放：原料被消费后即删，真正的数据只在树上 ─────────────────────────────────────

    def _release(self, documents: list[Any], guard: LeaseGuard) -> None:
        """删掉本批已发布链消费的判断，以及不再被任何判断引用、且全部观测已融合的交付。

        判断在发布后零读者（融合上下文只回看未封口的窗口，归约只读未消费的）；观测最后一次被读
        是 stage 物化 basis。删除幂等，重放路径可任意次进入。被隔离（quarantined）的判断仍引用
        着它的交付，那份交付就留着——可见、不丢，等人处置。
        """

        judgement_ids: set[str] = set()
        source_refs: set[str] = set()
        for item in documents:
            if not isinstance(item, Mapping):
                continue
            ledger = item.get("ledger")
            if isinstance(ledger, Mapping):
                judgement_ids.update(str(value) for value in ledger.get("judgement_ids", ()))
            payload = item.get("payload")
            if isinstance(payload, Mapping):
                source_refs.update(str(value) for value in payload.get("source_refs", ()))
        for index, judgement_id in enumerate(sorted(judgement_ids)):
            if index % 200 == 0:
                guard.checkpoint()
            record = self.judgements.read(judgement_id)
            if record is not None:
                source_refs.update(str(value) for value in record.get("source_refs", ()))
            self.judgements.discard(judgement_id)
        if not source_refs:
            return
        # 释放交付的条件：它的每条观测都有覆盖记录（已融合），且没有任何仍在存储里的判断引用它。
        still_referenced: set[str] = set()
        for raw in self.judgements.list():
            still_referenced.update(str(value) for value in raw.get("observation_ids", ()))
        covered = self.coverage.covered_observation_ids(self._now())
        for index, source_ref in enumerate(sorted(source_refs)):
            if index % 200 == 0:
                guard.checkpoint()
            envelope = self.observations.read(source_ref)
            if envelope is None:
                continue
            ids = {observation.observation_id for observation in envelope.batch.observations}
            if ids <= covered and not (ids & still_referenced):
                self.observations.discard(source_ref)

    def _expire(self, now: datetime) -> None:
        """回执、覆盖索引、消费账本按同一个窗口整块过期；窗口 = 上游最大补发跨度。"""

        before = now - timedelta(days=self.coverage.window_days)
        self.coverage.expire(now)
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
        self, documents: list[dict[str, Any]], now: datetime, dropped: list[str]
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
        for item in documents:
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
                raise BehaviorReductionError(
                    f"reduction requires a source delivery that is not stored: {source_ref}"
                )
            for observation in envelope.batch.observations:
                index.setdefault(observation.observation_id, observation)
        return index


def _as_instant(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


_KIND_PERSIST_EVERY = 25
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
