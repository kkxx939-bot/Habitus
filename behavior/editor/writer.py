"""使用租约、CAS 和读回校验发布行为语义文档。"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

from behavior.document import BehaviorDocument, BehaviorDocumentMetadata
from behavior.model import BehaviorDirectory, BehaviorKind
from behavior.schema import BehaviorOperationMode
from behavior.snapshot import BehaviorSnapshotReader
from behavior.tree import BehaviorTree, BehaviorTreeConflictError
from behavior.uri import BehaviorURI
from infrastructure.store.contracts.lock import LockStore
from infrastructure.store.contracts.path_lock import LeaseGuard, PathLock

_PHASE_SYSTEM_FIELDS = frozenset({"started_at", "ended_at"})


class BehaviorPublishConflictError(RuntimeError):
    """add-only 行为地址已经存在。"""


class BehaviorCASConflictError(RuntimeError):
    """Outcome 追加使用了过期 revision。"""


class BehaviorReadBackError(RuntimeError):
    """耐久写入后的完整读回结果与预期不一致。"""


@dataclass(frozen=True)
class BehaviorWriteConfig:
    lock_ttl_seconds: int = 30

    def __post_init__(self) -> None:
        if (
            isinstance(self.lock_ttl_seconds, bool)
            or not isinstance(self.lock_ttl_seconds, int)
            or not 1 <= self.lock_ttl_seconds <= 3_600
        ):
            raise ValueError("lock_ttl_seconds must be between 1 and 3600")


class _LeaseHeartbeat:
    """在锁内的长时间引用校验期间续租，避免租约先于校验结束过期。

    Episode 会在同一个 fenced 区内读回全部被引用的 Event 和 Outcome，
    每次读回都要逐级枚举目录，总耗时可能超过 ``lock_ttl_seconds``。
    """

    def __init__(self, guards: Sequence[LeaseGuard], *, ttl_seconds: int) -> None:
        self._guards = tuple(guards)
        self._interval = max(1.0, ttl_seconds / 3)
        self._last_renewed = monotonic()

    def beat(self) -> None:
        """到达续租间隔时统一续租；未到间隔不产生任何锁存储调用。"""

        if not self._guards:
            return
        now = monotonic()
        if now - self._last_renewed < self._interval:
            return
        for guard in self._guards:
            guard.checkpoint()
        self._last_renewed = now


@dataclass(frozen=True)
class BehaviorDocumentLockKeyspace:
    root: Path
    _prefix: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise TypeError("root must be a Path")
        normalized = self.root.expanduser().absolute().resolve(strict=False)
        object.__setattr__(self, "root", normalized)
        digest = hashlib.sha256(str(normalized).encode("utf-8")).hexdigest()[:24]
        object.__setattr__(self, "_prefix", f"behavior-document:{digest}")

    def key(self, uri: BehaviorURI | str) -> str:
        parsed = BehaviorURI.parse(uri)
        digest = hashlib.sha256(str(parsed).encode("utf-8")).hexdigest()
        return f"{self._prefix}:{digest}"


class BehaviorDocumentWriter:
    """唯一公开写策略：Event/Episode add-only，Outcome append-only。"""

    def __init__(
        self,
        tree: BehaviorTree,
        lock_store: LockStore,
        *,
        clock: Callable[[], datetime] | None = None,
        config: BehaviorWriteConfig | None = None,
    ) -> None:
        if not isinstance(tree, BehaviorTree):
            raise TypeError("tree must be a BehaviorTree")
        required = ("acquire", "renew", "fenced", "release")
        if any(not callable(getattr(lock_store, name, None)) for name in required):
            raise TypeError("lock_store must implement the LockStore contract")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        if config is not None and not isinstance(config, BehaviorWriteConfig):
            raise TypeError("config must be BehaviorWriteConfig")
        self.tree = tree
        self.registry = tree.registry
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.config = config or BehaviorWriteConfig()
        self._path_lock = PathLock(lock_store)
        self._keyspace = BehaviorDocumentLockKeyspace(tree.root)
        self._snapshot_reader = BehaviorSnapshotReader(tree)

    def publish(
        self,
        kind: BehaviorKind | str,
        payload: Mapping[str, Any],
    ) -> BehaviorDocument:
        """发布一个新 L2；所有已存在身份都作为冲突处理。"""

        normalized_kind = BehaviorKind(kind)
        schema = self.registry.get(normalized_kind)
        if schema.operation_mode not in {
            BehaviorOperationMode.ADD_ONLY,
            BehaviorOperationMode.APPEND_ONLY,
        }:
            raise ValueError("unsupported behavior operation mode")
        source_payload, event_fields_by_uri = self._with_system_fields(normalized_kind, payload)
        timestamp = self._timestamp()
        document = self.tree.document_codec.build(
            normalized_kind,
            source_payload,
            metadata=BehaviorDocumentMetadata.initial(timestamp),
        )
        uri = BehaviorURI.from_address(document.address)
        directory_uri = BehaviorURI.from_directory(BehaviorDirectory.for_address(document.address))
        with self._fenced_uris(uri, directory_uri, *self._reference_uris(document)) as heartbeat:
            self._validate_references(document, known_event_fields=event_fields_by_uri, heartbeat=heartbeat)
            if self.tree.exists(document.address):
                raise BehaviorPublishConflictError(f"behavior document already exists: {uri}")
            try:
                # TODO(BHV-COMMIT-002): Behavior 统一事务完成后由 PREPARED 日志消除模糊提交。
                self.tree.create(document)
            except BehaviorTreeConflictError as exc:
                raise BehaviorPublishConflictError(f"behavior document already exists: {uri}") from exc
            self._require_read_back(document)
        return document

    def append_outcomes(
        self,
        uri: BehaviorURI | str,
        outcomes: Sequence[Mapping[str, Any]],
        *,
        expected_revision: int,
    ) -> BehaviorDocument:
        """只向已有 Outcome 文档尾部追加新结果，并执行 revision CAS。"""

        parsed = BehaviorURI.parse(uri)
        address = parsed.to_address()
        if address.kind is not BehaviorKind.OUTCOME:
            raise ValueError("append_outcomes requires an Outcome document URI")
        if isinstance(outcomes, str) or not isinstance(outcomes, Sequence) or not outcomes:
            raise ValueError("outcomes must be a non-empty sequence of mappings")
        if any(not isinstance(item, Mapping) for item in outcomes):
            raise TypeError("outcomes must contain mappings")
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision <= 0:
            raise ValueError("expected_revision must be a positive integer")
        with self._fenced_uris(parsed):
            current = self.tree.read(address)
            if current.metadata.revision != expected_revision:
                raise BehaviorCASConflictError(
                    f"behavior Outcome revision changed: expected {expected_revision}, "
                    f"found {current.metadata.revision}"
                )
            payload = dict(current.fields)
            payload["outcomes"] = [*current.fields["outcomes"], *(dict(item) for item in outcomes)]
            updated = self.tree.document_codec.build(
                BehaviorKind.OUTCOME,
                payload,
                metadata=current.metadata.next_revision(self._timestamp()),
                links=current.links,
                backlinks=current.backlinks,
            )
            self._validate_references(updated)
            # TODO(BHV-COMMIT-002): Behavior 统一事务完成后由 PREPARED 日志消除模糊提交。
            self.tree.write(updated)
            self._require_read_back(updated)
            return updated

    def _validate_references(
        self,
        document: BehaviorDocument,
        *,
        known_event_fields: Mapping[str, dict[str, Any]] | None = None,
        heartbeat: _LeaseHeartbeat | None = None,
    ) -> None:
        """按文档类型校验其指向的既有 L2；Event 不引用任何其他文档。"""

        if document.kind is BehaviorKind.OUTCOME:
            self._validate_outcome_references(document)
        elif document.kind is BehaviorKind.EPISODE:
            self._validate_episode_references(
                document,
                known_event_fields=known_event_fields,
                heartbeat=heartbeat,
            )

    def _validate_outcome_references(self, document: BehaviorDocument) -> None:
        """每条结果的目标 Action 必须存在，且不得早于其目标 Event 或 Action。"""

        event_uri = BehaviorURI.parse(document.fields["event_uri"])
        event_document = self.tree.read(event_uri.to_address())
        target_event_fields = self.registry.validate(BehaviorKind.EVENT, event_document.fields)
        outcome_fields = self.registry.validate(BehaviorKind.OUTCOME, document.fields)
        actions = {action["action_id"]: action for action in target_event_fields["actions"]}
        for outcome in outcome_fields["outcomes"]:
            target_action_id = outcome["target_action_id"]
            if target_action_id is not None and target_action_id not in actions:
                raise ValueError(f"Outcome targets an Action absent from its Event: {target_action_id}")
            lower_bound = target_event_fields["started_at"]
            if target_action_id is not None and actions[target_action_id]["started_at"] is not None:
                lower_bound = actions[target_action_id]["started_at"]
            if outcome["occurred_at"] < lower_bound:
                raise ValueError("Outcome occurred before its target Event or Action")

    def _validate_episode_references(
        self,
        document: BehaviorDocument,
        *,
        known_event_fields: Mapping[str, dict[str, Any]] | None,
        heartbeat: _LeaseHeartbeat | None,
    ) -> None:
        """被引用 Event 必须落在 Episode 时间窗内，被引用 Outcome 必须仍等于其冻结快照。"""

        episode_fields = self.registry.validate(BehaviorKind.EPISODE, document.fields)
        event_fields_by_uri = self._episode_event_fields(
            episode_fields,
            known_event_fields=known_event_fields,
            heartbeat=heartbeat,
        )
        expected_phases = self._derive_phase_times(episode_fields["phases"], event_fields_by_uri)
        if tuple(episode_fields["phases"]) != expected_phases:
            raise ValueError("Episode Phase times do not match their referenced Events")
        self._require_frozen_outcomes(episode_fields, heartbeat=heartbeat)

    def _episode_event_fields(
        self,
        episode_fields: Mapping[str, Any],
        *,
        known_event_fields: Mapping[str, dict[str, Any]] | None,
        heartbeat: _LeaseHeartbeat | None,
    ) -> dict[str, dict[str, Any]]:
        """读回全部被引用 Event，校验时间窗并确认顺序不与真实时间矛盾。"""

        ordered: list[dict[str, Any]] = []
        by_uri: dict[str, dict[str, Any]] = {}
        for uri_text in episode_fields["ordered_event_uris"]:
            if heartbeat is not None:
                heartbeat.beat()
            fields = self._event_fields(uri_text, known_event_fields)
            if fields["started_at"] < episode_fields["started_at"] or fields["started_at"] > episode_fields["ended_at"]:
                raise ValueError("Episode Event starts outside the Episode time window")
            if fields["ended_at"] is not None and fields["ended_at"] > episode_fields["ended_at"]:
                raise ValueError("Episode Event ends outside the Episode time window")
            ordered.append(fields)
            by_uri[uri_text] = fields
        for index, earlier in enumerate(ordered):
            for later in ordered[index + 1 :]:
                if later["ended_at"] is not None and later["ended_at"] <= earlier["started_at"]:
                    raise ValueError("Episode ordered_event_uris contradict the real Event order")
        return by_uri

    def _require_frozen_outcomes(
        self,
        episode_fields: Mapping[str, Any],
        *,
        heartbeat: _LeaseHeartbeat | None,
    ) -> None:
        """被引用 Outcome 的 revision 与摘要必须仍等于创建时冻结的快照。"""

        event_uris = set(episode_fields["ordered_event_uris"])
        snapshots_by_uri = {snapshot["uri"]: snapshot for snapshot in episode_fields["outcome_snapshots"]}
        for uri_text in episode_fields["outcome_uris"]:
            if heartbeat is not None:
                heartbeat.beat()
            snapshot = self._snapshot_reader.read(uri_text)
            expected_snapshot = snapshots_by_uri[uri_text]
            if (
                not snapshot.exists
                or snapshot.revision != expected_snapshot["revision"]
                or snapshot.source_digest != expected_snapshot["digest"]
            ):
                raise BehaviorCASConflictError("Episode Outcome changed before its frozen snapshot was committed")
            outcome_document = snapshot.value
            assert isinstance(outcome_document, BehaviorDocument)
            if outcome_document.kind is not BehaviorKind.OUTCOME:
                raise ValueError("Episode outcome reference does not resolve to an Outcome")
            fields = self.registry.validate(BehaviorKind.OUTCOME, outcome_document.fields)
            if fields["event_uri"] not in event_uris:
                raise ValueError("Episode Outcome belongs to an Event outside the Episode")

    def _event_fields(
        self,
        uri_text: str,
        known_event_fields: Mapping[str, dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """复用系统字段派生阶段已经读回的 Event；Event 是 add-only 且全部字段不可变。"""

        cached = None if known_event_fields is None else known_event_fields.get(uri_text)
        if cached is not None:
            return cached
        referenced = self.tree.read(BehaviorURI.parse(uri_text).to_address())
        if referenced.kind is not BehaviorKind.EVENT:
            raise ValueError("Episode event reference does not resolve to an Event")
        return self.registry.validate(BehaviorKind.EVENT, referenced.fields)

    def _with_system_fields(
        self,
        kind: BehaviorKind,
        payload: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], dict[str, dict[str, Any]]]:
        """只由系统为 Episode 派生 Phase 时间并绑定 Outcome 版本证据。

        同时返回派生期间读回的 Event 字段，供锁内引用校验复用，避免同一批 Event 读两遍。
        """

        if kind is not BehaviorKind.EPISODE:
            return payload, {}
        if "outcome_snapshots" in payload:
            raise ValueError("outcome_snapshots is system-owned and must not be supplied by the caller")
        phases = self._caller_phases(payload)
        event_fields_by_uri = self._read_ordered_events(payload)
        enriched = dict(payload)
        enriched["phases"] = self._derive_phase_times(phases, event_fields_by_uri)
        enriched["outcome_snapshots"] = self._freeze_outcome_snapshots(payload)
        return enriched, event_fields_by_uri

    @staticmethod
    def _caller_phases(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        """取出调用方声明的 Phase，并拒绝任何由调用方填写的系统时间字段。"""

        phase_values = payload.get("phases")
        if isinstance(phase_values, (str, bytes)) or not isinstance(phase_values, Sequence):
            raise ValueError("Episode phases must be an array")
        phases: list[Mapping[str, Any]] = []
        for phase in phase_values:
            if not isinstance(phase, Mapping):
                raise TypeError("Episode phases must contain mappings")
            if _PHASE_SYSTEM_FIELDS & set(phase):
                raise ValueError("Episode Phase time fields are system-owned and must not be supplied by the caller")
            phases.append(phase)
        return tuple(phases)

    def _read_ordered_events(self, payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """读回全部被引用 Event，供 Phase 时间派生和锁内引用校验共用。"""

        event_values = payload.get("ordered_event_uris")
        if isinstance(event_values, (str, bytes)) or not isinstance(event_values, Sequence):
            raise ValueError("Episode ordered_event_uris must be an array")
        event_fields_by_uri: dict[str, dict[str, Any]] = {}
        for value in event_values:
            uri = BehaviorURI.parse(value)
            if uri.to_address().kind is not BehaviorKind.EVENT:
                raise ValueError("Episode ordered_event_uris must identify Event documents")
            document = self.tree.read(uri.to_address())
            if document.kind is not BehaviorKind.EVENT:
                raise ValueError("Episode event reference does not resolve to an Event")
            event_fields_by_uri[str(uri)] = self.registry.validate(BehaviorKind.EVENT, document.fields)
        return event_fields_by_uri

    def _freeze_outcome_snapshots(self, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        """把创建时看到的 Outcome revision 与规范摘要冻结为版本证据。"""

        outcome_values = payload.get("outcome_uris")
        if isinstance(outcome_values, str) or not isinstance(outcome_values, Sequence):
            raise ValueError("Episode outcome_uris must be an array")
        outcome_uris = tuple(BehaviorURI.parse(value) for value in outcome_values)
        batch = self._snapshot_reader.read_many(outcome_uris)
        snapshots_by_uri = {snapshot.identity: snapshot for snapshot in batch.snapshots}
        frozen: list[dict[str, Any]] = []
        for uri in outcome_uris:
            snapshot = snapshots_by_uri[str(uri)]
            if not snapshot.exists:
                raise FileNotFoundError(f"Episode Outcome does not exist: {uri}")
            if uri.to_address().kind is not BehaviorKind.OUTCOME:
                raise ValueError("Episode outcome_uris must identify Outcome documents")
            assert snapshot.revision is not None and snapshot.source_digest is not None
            frozen.append(
                {
                    "uri": str(uri),
                    "revision": snapshot.revision,
                    "digest": snapshot.source_digest,
                }
            )
        return frozen

    @staticmethod
    def _derive_phase_times(
        phases: Sequence[Mapping[str, Any]],
        event_fields_by_uri: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        """从 Phase 引用的真实 Event 派生时间，不补造缺失的结束时间。"""

        derived: list[dict[str, Any]] = []
        for phase in phases:
            event_uris = phase.get("event_uris")
            if isinstance(event_uris, (str, bytes)) or not isinstance(event_uris, Sequence) or not event_uris:
                raise ValueError("Episode Phase event_uris must be a non-empty array")
            try:
                event_fields = tuple(event_fields_by_uri[str(BehaviorURI.parse(uri))] for uri in event_uris)
            except KeyError as exc:
                raise ValueError("Episode Phase references an Event outside ordered_event_uris") from exc
            started_at = min(fields["started_at"] for fields in event_fields)
            known_ends = tuple(fields["ended_at"] for fields in event_fields)
            ended_at = None if any(value is None for value in known_ends) else max(known_ends)
            enriched = dict(phase)
            enriched["started_at"] = started_at
            enriched["ended_at"] = ended_at
            derived.append(enriched)
        return tuple(derived)

    @staticmethod
    def _reference_uris(document: BehaviorDocument) -> tuple[BehaviorURI, ...]:
        """返回发布期间必须与目标文档共同 fencing 的现有 L2。"""

        if document.kind is BehaviorKind.OUTCOME:
            return (BehaviorURI.parse(document.fields["event_uri"]),)
        if document.kind is BehaviorKind.EPISODE:
            values = (
                *document.fields["ordered_event_uris"],
                *document.fields["outcome_uris"],
            )
            return tuple(BehaviorURI.parse(value) for value in values)
        return ()

    @contextmanager
    def _fenced_uris(self, *uris: BehaviorURI) -> Iterator[_LeaseHeartbeat]:
        # TODO(BHV-RUNTIME-001): Runtime 组合根必须为同一 behavior-root 共享一个 LockStore。
        keys = tuple(sorted({self._keyspace.key(uri) for uri in uris}))
        with ExitStack() as stack:
            guards = tuple(
                stack.enter_context(
                    self._path_lock.acquire(
                        key,
                        ttl_seconds=self.config.lock_ttl_seconds,
                    )
                )
                for key in keys
            )
            with self._path_lock.fenced(guards):
                yield _LeaseHeartbeat(guards, ttl_seconds=self.config.lock_ttl_seconds)

    def _require_read_back(self, expected: BehaviorDocument) -> None:
        actual = self.tree.read(expected.address)
        if actual != expected:
            raise BehaviorReadBackError(
                f"behavior document read-back mismatch: {BehaviorURI.from_address(expected.address)}"
            )

    def _timestamp(self) -> datetime:
        value = self.clock()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("behavior writer clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)


__all__ = [
    "BehaviorCASConflictError",
    "BehaviorDocumentWriter",
    "BehaviorPublishConflictError",
    "BehaviorReadBackError",
    "BehaviorWriteConfig",
]
