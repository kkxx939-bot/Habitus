"""情景刷新器：定稿日 → 归组一次 → 情景文档 → 发布一代（编排层）。

各层按固定先后顺序处理已经定稿的历史：行为侧先把一天定稿（那天的链都已落树、封口视界已过
那天的本地结束，见归约 runner 的 ``closed_days``），本层再对那天归组**一次**，预测树再读两者。
定稿之后到达的补发照常进行为树，但不改变"已定稿"这个事实，本层也不因此重算——历史不会变，
变的只是我们看到了多少；重算只作为运维正门的主动动作存在（改了提示词版本、或指定某几天强制）。

分工：``progress.py`` 管耐久进度（待刷新集合、失败记账、检查点），``materialize.py`` 管草稿→文档，
``input.py`` 管归组输入的装配（可注入替换），本文件只做编排：

- **单写入方**：本层持自己的租约（与归约 sweep 无关），模型调用期间按拍续约。
- **检查点在干跑之后**：模型答复装配成功、且已确认能编成文档，才按 source_digest 落检查点；
  发布失败下次零调用重放；发布成功后清掉。
- **失败预算**：同一输入连续失败 ``max_attempts_per_input`` 次即封锁（留信号、出集合），输入
  变了自动解封；超限（行为数/提示词）这类确定性失败直接封锁。
- **调用预算**：一次刷新最多 ``max_model_calls_per_run`` 次模型调用，其余留待刷新集合下次。
- **逐日隔离**：一天失败只影响那一天。
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable, Iterable
from contextlib import ExitStack, suppress
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

from habitus.behavior.tree import BehaviorTree
from habitus.foundation.integrity import canonical_digest
from habitus.infrastructure.store.contracts.lock import LockStore
from habitus.infrastructure.store.contracts.path_lock import LeaseGuard, PathLock
from habitus.scene.document import SceneDocument
from habitus.scene.grouping.model import GroupingAssembly, SceneGroupingInput
from habitus.scene.grouping.service import SceneGrouper, SceneGroupingLimitError
from habitus.scene.model import SceneLinkType, SceneRole
from habitus.scene.refresh.input import build_grouping_input
from habitus.scene.refresh.materialize import materialize_documents
from habitus.scene.refresh.progress import RefreshProgress, SceneRefreshError
from habitus.scene.tree import SceneTree

_KEEPALIVE_INTERVAL_SECONDS = 60.0
_LOCK_TTL_SECONDS = 600


class SceneRefreshBusyError(RuntimeError):
    """另一个刷新（夜批或正门）正持有情景刷新租约。"""


class GroupingInputBuilder(Protocol):
    """归组输入的装配契约：给一天与两棵树，装出模型要看的东西（参照怎么选在这里决定）。"""

    def __call__(
        self,
        day: date,
        *,
        subject: str,
        behavior_tree: BehaviorTree,
        scene_tree: SceneTree,
        lookback_days: int,
        pending_expiry_days: int,
    ) -> SceneGroupingInput | None: ...


@dataclass(frozen=True)
class SceneRefreshConfig:
    lookback_days: int = 7
    pending_expiry_days: int = 90
    # 同一输入连续失败几次后封锁（输入变了自动解封）；一次刷新最多几次模型调用（其余留集合）。
    max_attempts_per_input: int = 3
    max_model_calls_per_run: int = 4

    def __post_init__(self) -> None:
        for name in ("lookback_days", "pending_expiry_days", "max_attempts_per_input", "max_model_calls_per_run"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class SceneRefreshReport:
    published: tuple[date, ...]
    unchanged: tuple[date, ...]
    deferred: tuple[date, ...]
    failed: tuple[date, ...]
    blocked: tuple[date, ...]
    signals: tuple[str, ...]
    model_calls: int = 0


class SceneRefresher:
    def __init__(
        self,
        *,
        behavior_tree: BehaviorTree,
        scene_tree: SceneTree,
        grouper: SceneGrouper,
        subject: str,
        progress_root: str | Path,
        lock_store: LockStore,
        config: SceneRefreshConfig | None = None,
        clock: Callable[[], datetime] | None = None,
        input_builder: GroupingInputBuilder = build_grouping_input,
    ) -> None:
        if not isinstance(behavior_tree, BehaviorTree):
            raise TypeError("behavior_tree must be a BehaviorTree")
        if not isinstance(scene_tree, SceneTree):
            raise TypeError("scene_tree must be a SceneTree")
        if not callable(getattr(grouper, "group", None)) or not isinstance(getattr(grouper, "version", None), str):
            raise TypeError("grouper must implement group(payload) and version")
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError("subject must be non-empty text")
        if any(not callable(getattr(lock_store, name, None)) for name in ("acquire", "renew", "fenced", "release")):
            raise TypeError("lock_store must implement the LockStore contract")
        resolved = config or SceneRefreshConfig()
        if not isinstance(resolved, SceneRefreshConfig):
            raise TypeError("config must be SceneRefreshConfig")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        if not callable(input_builder):
            raise TypeError("input_builder must be callable")
        self.behavior_tree = behavior_tree
        self.scene_tree = scene_tree
        self.grouper = grouper
        self.subject = subject
        self.config = resolved
        self.clock = clock or (lambda: datetime.now(UTC))
        self.input_builder = input_builder
        # 刷新器的耐久文件与树分家：树根下只有情景；这里的东西是可重建的进度，必须住在树外。
        self.root = Path(progress_root).expanduser().absolute()
        if self.root == scene_tree.root or scene_tree.root in self.root.parents or self.root in scene_tree.root.parents:
            raise ValueError("progress_root must not overlap the scene tree root")
        self.progress = RefreshProgress(self.root)
        self._path_lock = PathLock(lock_store)
        self._lock_key = f"scene-refresh:{hashlib.sha256(str(scene_tree.root).encode('utf-8')).hexdigest()[:24]}"

    # ── 对外 ────────────────────────────────────────────────────────────────

    async def refresh_days(self, days: Iterable[date], *, force: bool = False) -> SceneRefreshReport:
        """对已定稿的日子各归组一次。已有一代的日子跳过（``force`` 为真时重算）；调用方负责只交定稿日。"""

        requested = set(days)
        if any(isinstance(day, datetime) or not isinstance(day, date) for day in requested):
            raise TypeError("days must contain date values")
        with ExitStack() as stack:
            try:
                guard = stack.enter_context(self._path_lock.acquire(self._lock_key, ttl_seconds=_LOCK_TTL_SECONDS))
            except TimeoutError as exc:
                raise SceneRefreshBusyError(str(exc)) from exc
            return await self._refresh_locked(requested, guard, force=force)

    def pending_days(self) -> tuple[date, ...]:
        return tuple(sorted(self.progress.pending_days()))

    def blocked_days(self) -> dict[date, str]:
        return {day: record.error for day, record in self.progress.failures().items() if record.blocked}

    def stale_days(self, closed_days: Iterable[date]) -> tuple[date, ...]:
        """已定稿、但情景树没有当前版本一代的日子（夜批与回填的工作集）。"""

        version = self.grouper.version
        stale = []
        for day in sorted(set(closed_days)):
            state = self.scene_tree.day_state(day)
            if state is None or state.scene_version != version:
                stale.append(day)
        return tuple(stale)

    # ── 编排 ────────────────────────────────────────────────────────────────

    async def _refresh_locked(self, requested: set[date], guard: LeaseGuard, *, force: bool) -> SceneRefreshReport:
        pending = self.progress.pending_days() | requested
        if not pending:
            return SceneRefreshReport((), (), (), (), (), ())
        self.progress.write_pending_days(pending)
        outcomes: dict[str, list[date]] = {"published": [], "unchanged": [], "deferred": [], "failed": [], "blocked": []}
        signals: list[str] = []
        budget = _Budget(self.config.max_model_calls_per_run)
        for day in sorted(pending):
            guard.checkpoint()
            try:
                outcome, notes = await self._refresh_day(day, guard, budget=budget, force=force and day in requested)
            except Exception as exc:  # noqa: BLE001 - 逐日隔离：失败留集合下次重试
                outcome, notes = self._on_failure(day, exc)
            signals.extend(f"[{day.isoformat()}] {note}" for note in notes)
            outcomes[outcome].append(day)
            if outcome not in ("deferred", "failed"):
                pending.discard(day)
                self.progress.write_pending_days(pending)
        return SceneRefreshReport(
            tuple(outcomes["published"]),
            tuple(outcomes["unchanged"]),
            tuple(outcomes["deferred"]),
            tuple(outcomes["failed"]),
            tuple(outcomes["blocked"]),
            tuple(signals),
            budget.used,
        )

    async def _refresh_day(self, day: date, guard: LeaseGuard, *, budget: _Budget, force: bool) -> tuple[str, tuple[str, ...]]:
        version = self.grouper.version
        state = self.scene_tree.day_state(day)
        if state is not None and not force:
            self._settle(day)
            return "unchanged", ()
        payload = self._input(day)
        digest = self._source_digest(payload, version)
        if state is not None and state.source_digest == digest and state.scene_version == version:
            self._settle(day)
            return "unchanged", ("forced refresh skipped: same input and version",)
        record = self.progress.failures().get(day)
        if record is not None and record.blocked and record.source_digest == digest:
            return "blocked", (f"blocked for this input: {record.error}",)
        notes: list[str] = []
        documents: tuple[SceneDocument, ...] = ()
        if payload is not None:
            staged = self.progress.load_checkpoint(day, digest)
            if staged is None:
                if not budget.take():
                    return "deferred", ("model call budget exhausted for this run",)
                assembly = await self._group_with_keepalive(payload, guard)
            else:
                assembly = staged
                notes.append("checkpoint replayed without a model call")
            documents, degraded = materialize_documents(
                self.scene_tree, day, payload, assembly, source_digest=digest, scene_version=version, created_at=self.clock()
            )
            if staged is None:
                self.progress.write_checkpoint(day, digest, version, assembly)
            notes.extend(assembly.signals)
            notes.extend(degraded)
            notes.append(f"scenes={len(documents)} unassigned={len(assembly.unassigned)}")
        guard.checkpoint()
        self.scene_tree.publish_day(day, documents, published_at=self.clock(), source_digest=digest, scene_version=version)
        self._settle(day)
        return "published", tuple(notes)

    def _on_failure(self, day: date, exc: Exception) -> tuple[str, tuple[str, ...]]:
        """记一次失败；达到预算或属确定性失败即封锁。指纹算不出来时按空指纹记账。"""

        try:
            digest = self._source_digest(self._input(day), self.grouper.version)
        except Exception:  # noqa: BLE001 - 指纹本身算不出来也要能记账
            digest = ""
        deterministic = isinstance(exc, SceneGroupingLimitError)
        record = self.progress.record_failure(
            day,
            digest,
            f"{type(exc).__name__}: {exc}",
            deterministic=deterministic,
            max_attempts=self.config.max_attempts_per_input,
        )
        notes = [f"scene refresh failed: {exc}"]
        if not record.blocked:
            return "failed", tuple(notes)
        reason = "deterministic failure" if deterministic else f"{record.attempts} consecutive failures on the same input"
        notes.append(f"blocked: {reason}")
        return "blocked", tuple(notes)

    def _settle(self, day: date) -> None:
        self.progress.clear_checkpoint(day)
        self.progress.clear_failure(day)

    def _input(self, day: date) -> SceneGroupingInput | None:
        return self.input_builder(
            day,
            subject=self.subject,
            behavior_tree=self.behavior_tree,
            scene_tree=self.scene_tree,
            lookback_days=self.config.lookback_days,
            pending_expiry_days=self.config.pending_expiry_days,
        )

    async def _group_with_keepalive(self, payload: SceneGroupingInput, guard: LeaseGuard) -> GroupingAssembly:
        """模型调用期间按固定间隔续约：一次归组实测 2–8 分钟，与租约同量级。"""

        stop = asyncio.Event()

        async def beat() -> None:
            while not stop.is_set():
                with suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=_KEEPALIVE_INTERVAL_SECONDS)
                if stop.is_set():
                    return
                guard.checkpoint()

        heartbeat = asyncio.create_task(beat(), name="habitus-scene-refresh-keepalive")
        try:
            return await self.grouper.group(payload)
        finally:
            stop.set()
            heartbeat.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await heartbeat

    @staticmethod
    def _source_digest(payload: SceneGroupingInput | None, version: str) -> str:
        """对输入本身（URI、完整时间戳、参照）取指纹，不对渲染后的文本——渲染会抹掉身份。
        它是溯源与失败记账的键，不是重算的触发器。"""

        if payload is None:
            return canonical_digest({"scene_version": version, "empty_day": True})
        return canonical_digest({"scene_version": version, "input": _jsonable(asdict(payload))})


class _Budget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def take(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat(timespec="microseconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, SceneRole | SceneLinkType):
        return value.value
    return value


__all__ = [
    "GroupingInputBuilder",
    "SceneRefreshBusyError",
    "SceneRefreshConfig",
    "SceneRefreshError",
    "SceneRefreshReport",
    "SceneRefresher",
]
