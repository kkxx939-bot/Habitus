"""关联的夜批编排：按格子线性推进，一个候选的一天一次调用。

顺序与旧归组**相反**：归约把一天定稿 → 预测树重建 → 关联。因为待办来自树上的出处日。

编排层不认识 ``PredictionTree``：待办与前因排序依据由 ``scene.backlog`` 折成事实传进来，
所以"语义层不重算数字"由类型保证。

耐久性的三道：整轮一把租约、模型调用期间打点续租；检查点在调用成功之后、落盘之前写；完成标记
在全部记录回读通过之后才写。任何时刻被杀，要么这一天没有完成标记（下轮重做，而检查点让它连那次
调用也省掉），要么已完成（差集里不再出现）。
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable, Sequence
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol

from habitus.behavior.tree import BehaviorTree
from habitus.foundation.integrity import canonical_digest
from habitus.infrastructure.store.contracts.lock import LockStore
from habitus.infrastructure.store.contracts.path_lock import LeaseGuard, PathLock
from habitus.scene.association.input import MAX_CAUSE_ROWS, MAX_PENDING_ROWS, build_association_input
from habitus.scene.association.materialize import materialize
from habitus.scene.association.model import AssociationAssembly, AssociationInput
from habitus.scene.association.premises import Premise, PremiseTable
from habitus.scene.association.progress import AssociationProgress, Checkpoint, TaskKey
from habitus.scene.association.service import AssociationLimitError, Associator
from habitus.scene.backlog import AssociationLedger, AssociationTask, CauseFacts
from habitus.scene.calendar import DayTypeCalendar, NominalCalendar
from habitus.scene.regularity.document import AssociationDocument
from habitus.scene.regularity.overview import Overview
from habitus.scene.regularity.store import RegularityTree

_KEEPALIVE_INTERVAL_SECONDS = 60.0
_LOCK_TTL_SECONDS = 600


class AssociationBusyError(RuntimeError):
    """另一轮关联正在跑。"""


class AssociationInputBuilder(Protocol):
    def __call__(
        self,
        kind_token: str,
        day: date,
        *,
        behavior_tree: BehaviorTree,
        regularity_tree: RegularityTree,
        premises: PremiseTable,
        causes: CauseFacts,
        calendar: DayTypeCalendar,
        max_cause_rows: int = ...,
        max_pending_rows: int = ...,
    ) -> AssociationInput | None: ...


@dataclass(frozen=True)
class AssociationRefreshConfig:
    max_attempts_per_input: int = 3
    max_model_calls_per_run: int = 4

    def __post_init__(self) -> None:
        for name in ("max_attempts_per_input", "max_model_calls_per_run"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class AssociationRefreshReport:
    """一轮的结果。``open`` 是"答上了一部分、这一天没做完"——它既不是成功也不是失败，
    必须单独一档：报成成功会让它每轮重来、每轮白烧一次调用（编排层自己把 ``unanswered``
    抹平就是这个后果）。"""

    associated: tuple[str, ...] = ()
    open: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    deferred: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    blocked: tuple[str, ...] = ()
    signals: tuple[str, ...] = ()
    model_calls: int = 0


class AssociationRefresher:
    def __init__(
        self,
        *,
        behavior_tree: BehaviorTree,
        regularity_tree: RegularityTree,
        associator: Associator,
        progress_root: str | Path,
        lock_store: LockStore,
        calendar: DayTypeCalendar | None = None,
        config: AssociationRefreshConfig | None = None,
        clock: Callable[[], datetime] | None = None,
        input_builder: AssociationInputBuilder = build_association_input,
        max_cause_rows: int = MAX_CAUSE_ROWS,
        max_pending_rows: int = MAX_PENDING_ROWS,
    ) -> None:
        if not isinstance(behavior_tree, BehaviorTree):
            raise TypeError("behavior_tree must be a BehaviorTree")
        if not isinstance(regularity_tree, RegularityTree):
            raise TypeError("regularity_tree must be a RegularityTree")
        if not callable(getattr(associator, "associate", None)) or not isinstance(
            getattr(associator, "version", None), str
        ):
            raise TypeError("associator must implement associate(payload) and version")
        if any(not callable(getattr(lock_store, name, None)) for name in ("acquire", "renew", "fenced", "release")):
            raise TypeError("lock_store must implement the LockStore contract")
        resolved = config or AssociationRefreshConfig()
        if not isinstance(resolved, AssociationRefreshConfig):
            raise TypeError("config must be AssociationRefreshConfig")
        if not callable(input_builder):
            raise TypeError("input_builder must be callable")
        self.behavior_tree = behavior_tree
        self.regularity_tree = regularity_tree
        self.associator = associator
        self.calendar = calendar or NominalCalendar()
        self.config = resolved
        self.clock = clock or (lambda: datetime.now(UTC))
        self.input_builder = input_builder
        for name, value in (("max_cause_rows", max_cause_rows), ("max_pending_rows", max_pending_rows)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.max_cause_rows = max_cause_rows
        self.max_pending_rows = max_pending_rows
        self.root = Path(progress_root).expanduser().resolve()
        tree_root = Path(regularity_tree.root).resolve()
        if self.root == tree_root or tree_root in self.root.parents or self.root in tree_root.parents:
            raise ValueError("progress_root must not overlap the regularity tree root")
        self.progress = AssociationProgress(self.root, version=associator.version)
        self._completed_days = _CompletedDays(regularity_tree, associator)
        self._path_lock = PathLock(lock_store)
        self._lock_key = f"association:{hashlib.sha256(str(regularity_tree.root).encode('utf-8')).hexdigest()[:24]}"

    @property
    def associated_days(self) -> AssociationLedger:
        """给 ``backlog`` 与预测层用的"已关联完成"事实源，**带上当前版本**。

        直接把 ``RegularityTree`` 递过去会漏掉版本这一维：换了提示词版本要全量重做，而不带版本
        的完成标记仍然算"做完了"。让调用方自己记得传版本是靠自觉，这里由类型给出。

        **是同一个实例**，不是每次新建：组合根要拿它做实例同一性校验，而绑定方法每取一次都是
        一个新对象（``a.f is a.f`` 恒为假），每次新建就让那条校验永远不成立。
        """

        return self._completed_days

    def reset(self, kind_token: str | None = None) -> tuple[str, ...]:
        """**重放的正门**：把这些候选退回"没关联过"，下一轮夜批会整批重做。

        换提示词版本之后按版本作废的那条只解决了一半：完成标记按版本比对，所以待办会重新出现；
        但 L1 只增不减，旧版本归纳出来的情形会与新的混在一起累积，L0 的排名也跟着被带偏。所以
        重放必须是一个**显式操作**，不能指望夜批顺带做——它要清三样：完成标记、L1/L0、失败记账
        与检查点。记录本身不删，重做会按地址覆写，多出来的那些由 ``retain_only`` 清。

        不给候选就清全部。返回清了哪几个候选。
        """

        kinds = self.regularity_tree.list_kinds() if kind_token is None else (kind_token,)
        cleared: list[str] = []
        for kind in kinds:
            days = self.regularity_tree.discard_completion(kind)
            self.regularity_tree.discard_layers(kind)
            for day in days:
                self.progress.clear_checkpoint(TaskKey(kind, day))
            self.progress.unblock(kind)
            cleared.append(kind)
        return tuple(sorted(cleared))

    async def refresh(
        self, tasks: Sequence[AssociationTask], *, causes: CauseFacts, force: bool = False
    ) -> AssociationRefreshReport:
        """按给定顺序推进。顺序由 ``backlog`` 定：(日期, 槽位, 候选)，候选内部必须升序。"""

        ordered = tuple(tasks)
        if any(not isinstance(task, AssociationTask) for task in ordered):
            raise TypeError("tasks must contain AssociationTask values")
        # 候选内部必须时间**不降序**：规律级是增量叠加的，乱序会让情境的演化顺序错乱，而且
        # ``_previous_day`` 取的"上一次"会指到未来。重放脚本与修复工具都够得着这条，所以由入口
        # 拦，不靠自觉。重复递同一天是允许的（调用方拿了份过期快照），由"已关联"那道闸兜住。
        seen: dict[str, date] = {}
        for task in ordered:
            previous = seen.get(task.kind_token)
            if previous is not None and task.day < previous:
                raise ValueError("a candidate's tasks must be handed over in ascending day order")
            seen[task.kind_token] = task.day
        if not isinstance(causes, CauseFacts):
            raise TypeError("causes must be CauseFacts")
        with ExitStack() as stack:
            try:
                guard = stack.enter_context(self._path_lock.acquire(self._lock_key, ttl_seconds=_LOCK_TTL_SECONDS))
            except TimeoutError as exc:
                raise AssociationBusyError(str(exc)) from exc
            return await self._run(ordered, guard, causes=causes, force=force)

    async def _run(
        self, tasks: tuple[AssociationTask, ...], guard: LeaseGuard, *, causes: CauseFacts, force: bool
    ) -> AssociationRefreshReport:
        if not tasks:
            return AssociationRefreshReport()
        # 全树扫描是整轮开头最重的一次同步 IO，下沉到线程——它和后面每件任务的读盘都会冻住
        # 事件循环，而夜批期间记忆与融合的 worker 还在跑。
        premises, scan_signals = await asyncio.to_thread(PremiseTable.scan, self.regularity_tree)
        outcomes: dict[str, list[str]] = {
            name: [] for name in ("associated", "open", "skipped", "deferred", "failed", "blocked")
        }
        signals: list[str] = list(scan_signals)
        budget = _Budget(self.config.max_model_calls_per_run)
        for task in tasks:
            guard.checkpoint()
            # 键只管身份（规范化、casefold），读树与装输入要用**人写法**的 token：行为树的 occurrence 上写的
            # 是人写法，拿规范身份去比一条都对不上，带大写字母的候选（vLLM、Tagent）会每晚被判成"那天没发生"。
            key = TaskKey(task.kind_token, task.day)
            try:
                outcome, notes = await self._one(
                    key, task.kind_token, guard, premises=premises, causes=causes, budget=budget, force=force
                )
            except Exception as exc:  # noqa: BLE001 - 一件失败不许拖垮整轮
                outcome, notes = self._on_failure(key, task.kind_token, exc, premises=premises, causes=causes)
            signals.extend(f"[{key.identity}] {note}" for note in notes)
            outcomes[outcome].append(key.identity)
        return AssociationRefreshReport(
            associated=tuple(outcomes["associated"]),
            open=tuple(outcomes["open"]),
            skipped=tuple(outcomes["skipped"]),
            deferred=tuple(outcomes["deferred"]),
            failed=tuple(outcomes["failed"]),
            blocked=tuple(outcomes["blocked"]),
            signals=tuple(signals),
            model_calls=budget.used,
        )

    async def _one(
        self,
        key: TaskKey,
        kind_token: str,
        guard: LeaseGuard,
        *,
        premises: PremiseTable,
        causes: CauseFacts,
        budget: _Budget,
        force: bool,
    ) -> tuple[str, tuple[str, ...]]:
        done = await asyncio.to_thread(
            self.regularity_tree.days_for, key.kind_token, version=self.associator.version
        )
        if not force and key.day in done:
            # 同一轮里被递两次、或调用方拿了一份过期的已完成快照——都不该再烧一次调用。
            # 顺手清掉可能残留的检查点：写完成标记之后、清检查点之前崩过的话，那一天此后
            # 不再进待办，检查点就永远没人清得掉了（单个上限 16 MiB）。
            self.progress.clear_checkpoint(key)
            return "skipped", ("already associated",)
        payload = await asyncio.to_thread(self._input, kind_token, key.day, premises=premises, causes=causes)
        if payload is None:
            return "skipped", ("this candidate did not occur on that day",)
        digest = self._source_digest(payload)
        record = self.progress.failures().get(key)
        if record is not None and record.blocked and record.source_digest == digest and not force:
            return "blocked", (f"blocked for this input: {record.error}",)
        staged = self.progress.load_checkpoint(key, digest)
        notes: list[str] = []
        if staged is None:
            if not budget.take():
                return "deferred", ("model call budget exhausted for this run",)
            assembly = await self._associate(payload, guard)
            staged, materialized = self._materialize(payload, assembly)
            notes.extend(assembly.signals)
            notes.extend(materialized)
            self.progress.write_checkpoint(key, digest, self.associator.version, staged)
        else:
            notes.append("checkpoint replayed without a model call")
        guard.checkpoint()
        notes.extend(await asyncio.to_thread(self._publish, key, staged, premises))
        if staged.unanswered:
            # 答不全是**同一输入下确定性重现**的：模型说不出那几次的上下文，下次多半还是说不出。
            # 走同一条计次封锁，否则这一天既不 done 也不 blocked，每轮重来、每轮白烧一次调用，
            # 还把那个候选的配额永久占死。检查点留着，让重来那次连调用也省掉。
            record = self.progress.record_failure(
                key,
                digest,
                f"unanswered targets: {', '.join(f'#{no}' for no in staged.unanswered)}",
                deterministic=False,
                max_attempts=self.config.max_attempts_per_input,
            )
            notes.append(f"day left open ({record.attempts} attempts)")
            return ("blocked" if record.blocked else "open"), tuple(notes)
        self.progress.clear_checkpoint(key)
        self.progress.clear_failure(key)
        return "associated", tuple(notes)

    def _publish(self, key: TaskKey, staged: Checkpoint, premises: PremiseTable) -> tuple[str, ...]:
        """先写记录，再更新 L1 / L0，最后才写完成标记——标记是"这一天做完了"的唯一凭据。"""

        notes: list[str] = []
        if not staged.records:
            return ("no record could be built for this day",)
        # 重做一天：先清掉不属于本次的记录。``write`` 按地址覆写，本轮比上轮少时多出来的那些会
        # 让完成标记的条数核对永远过不去，而它们的 ``left`` 还会被投影当成有效前提读出来。
        orphans = self.regularity_tree.retain_only(
            key.kind_token, key.day, frozenset(document.address.identity_name for document in staged.records)
        )
        if orphans:
            notes.append(f"records_discarded: {', '.join(orphans)} are not part of this round")
        overview = Overview.read(self.regularity_tree, key.kind_token)
        for document in staged.records:
            self.regularity_tree.write(document)
            # 由来取这个候选**第一次**被关联时的那句上下文。"只写一次"由 ``with_origin`` 独自
            # 守着，这里不再另设判据——两处各判一遍，其中一处就一定会先漂。
            overview = overview.with_origin(document.context)
            if document.situation is not None:
                overview = overview.with_occurrence(day=key.day, text=document.situation)
            for producer_uri, text in document.consumed:
                if not premises.consume(producer_uri, text):
                    notes.append(f"premise_unknown: consumed «{text}» has no record of being produced")
            for text, waits_for in document.left:
                premises.add(_new_premise(document, text, waits_for, key))
        overview.write(self.regularity_tree)
        if not staged.unanswered:
            self.regularity_tree.complete_day(
                key.kind_token,
                key.day,
                records=len(staged.records),
                completed_at=self.clock(),
                version=self.associator.version,
            )
        return tuple(notes)

    def _materialize(self, payload: AssociationInput, assembly: AssociationAssembly) -> tuple[Checkpoint, list[str]]:
        overview = Overview.read(self.regularity_tree, payload.kind_token)
        records: list[AssociationDocument] = []
        unanswered = list(assembly.unanswered)
        notes: list[str] = []
        for draft in assembly.drafts:
            document, drops = materialize(payload, draft, overview, created_at=self.clock())
            notes.extend(drops)
            if document is None:
                unanswered.append(draft.occurrence_no)
            else:
                records.append(document)
        return Checkpoint(records=tuple(records), unanswered=tuple(sorted(unanswered))), notes

    def _on_failure(
        self, key: TaskKey, kind_token: str, exc: Exception, *, premises: PremiseTable, causes: CauseFacts
    ) -> tuple[str, tuple[str, ...]]:
        try:
            payload = self._input(kind_token, key.day, premises=premises, causes=causes)
            digest = "" if payload is None else self._source_digest(payload)
        except Exception:  # noqa: BLE001 - 记账本身不许再抛
            digest = ""
        deterministic = isinstance(exc, AssociationLimitError)
        record = self.progress.record_failure(
            key,
            digest,
            f"{type(exc).__name__}: {exc}",
            deterministic=deterministic,
            max_attempts=self.config.max_attempts_per_input,
        )
        notes = [f"association failed: {exc}"]
        if not record.blocked:
            return "failed", tuple(notes)
        # 封锁之后这件事不会再被生成成任务，检查点留着就没人清得掉了（单个上限 16 MiB）。
        self.progress.clear_checkpoint(key)
        reason = (
            "deterministic failure" if deterministic else f"{record.attempts} consecutive failures on the same input"
        )
        notes.append(f"blocked: {reason}")
        return "blocked", tuple(notes)

    def _input(
        self, kind_token: str, day: date, *, premises: PremiseTable, causes: CauseFacts
    ) -> AssociationInput | None:
        return self.input_builder(
            kind_token,
            day,
            behavior_tree=self.behavior_tree,
            regularity_tree=self.regularity_tree,
            premises=premises,
            causes=causes,
            calendar=self.calendar,
            max_cause_rows=self.max_cause_rows,
            max_pending_rows=self.max_pending_rows,
        )

    def _source_digest(self, payload: AssociationInput) -> str:
        """指纹只覆盖**稳定部分**。

        情境来自这个候选的历史，每成功关联一次就多一天覆盖，所以把整份输入摘要进去的话，同一件
        任务在两轮之间指纹必变，"装配成功、发布失败、下次零调用重放"这条保护永远不生效。
        """

        facts = payload.facts
        return canonical_digest(
            {
                "association_version": self.associator.version,
                "kind_token": payload.kind_token,
                "day": payload.day.isoformat(),
                "targets": list(payload.targets),
                # summary / goal / 日型 / 观测空白都直接渲染进提示词：漏掉它们，补一个观测空白
                # 之后重放会拿一份"那天一直在看"的答复，而模型再也没机会看见那个洞。
                "rows": [
                    [row.uri, row.started_at.isoformat(), row.summary, row.goal or ""]
                    for row in payload.occurrences
                ],
                "causes": [
                    [row.uri, row.started_at.isoformat(), row.summary] for row in payload.causes
                ],
                "day_note": facts.day_note or "",
                "gaps": [[start.isoformat(), end.isoformat()] for start, end in facts.observed_gaps],
            }
        )

    async def _associate(self, payload: AssociationInput, guard: LeaseGuard) -> AssociationAssembly:
        stop = asyncio.Event()

        async def beat() -> None:
            while not stop.is_set():
                with suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=_KEEPALIVE_INTERVAL_SECONDS)
                if stop.is_set():
                    return
                guard.checkpoint()

        heartbeat = asyncio.create_task(beat(), name="habitus-association-keepalive")
        try:
            return await self.associator.associate(payload)
        finally:
            stop.set()
            heartbeat.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await heartbeat


def _new_premise(document: AssociationDocument, text: str, waits_for: str, key: TaskKey) -> Premise:
    return Premise(
        producer_uri=document.occurrence_uri,
        text=text,
        waits_for=waits_for,
        created_on=key.day,
        created_at=document.address.started_at,
    )


class _CompletedDays:
    """``AssociatedDays`` 的生产实现：按当前关联版本回答"这个候选哪几天做完了"。

    ``tree`` 是公开的：组合根要核对"这个事实源读的是**这个** Runtime 的那棵树"，比对象身份比
    比绑定方法可靠。
    """

    def __init__(self, tree: RegularityTree, associator: Associator) -> None:
        self.tree = tree
        self._associator = associator

    @property
    def version(self) -> str:
        """回答"已关联"用的那把尺子。读记录的人按同一个版本读，两边才对得上。"""

        return self._associator.version

    def days_for(self, kind_token: str) -> frozenset[date]:
        # 版本**每次现问**，不在构造时定死：这个对象要长期活着（组合根拿它做实例同一性校验），
        # 而换提示词版本正是它必须跟着变的那一刻。
        return self.tree.days_for(kind_token, version=self._associator.version)


class _Budget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def take(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


__all__ = [
    "AssociationBusyError",
    "AssociationInputBuilder",
    "AssociationRefreshConfig",
    "AssociationRefreshReport",
    "AssociationRefresher",
]
