"""使用租约、CAS 和读回校验发布行为语义文档。"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from behavior.document import BehaviorDocument, BehaviorDocumentMetadata
from behavior.model import BehaviorDirectory, BehaviorKind
from behavior.schema import BehaviorOperationMode, BehaviorSchemaRegistry
from behavior.snapshot import BehaviorSnapshotReader
from behavior.tree import BehaviorTree, BehaviorTreeConflictError
from behavior.uri import BehaviorURI
from infrastructure.store.contracts.lock import LockStore
from infrastructure.store.contracts.path_lock import PathLock


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
        registry: BehaviorSchemaRegistry | None = None,
        clock: Callable[[], datetime] | None = None,
        config: BehaviorWriteConfig | None = None,
    ) -> None:
        if not isinstance(tree, BehaviorTree):
            raise TypeError("tree must be a BehaviorTree")
        required = ("acquire", "renew", "fenced", "release")
        if any(not callable(getattr(lock_store, name, None)) for name in required):
            raise TypeError("lock_store must implement the LockStore contract")
        tree_registry = getattr(tree.document_codec, "registry", None)
        if not isinstance(tree_registry, BehaviorSchemaRegistry):
            raise TypeError("tree document codec must use a BehaviorSchemaRegistry")
        if registry is not None and not isinstance(registry, BehaviorSchemaRegistry):
            raise TypeError("registry must be a BehaviorSchemaRegistry")
        if registry is not None and registry is not tree_registry:
            raise ValueError("writer and tree must share one BehaviorSchemaRegistry instance")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        if config is not None and not isinstance(config, BehaviorWriteConfig):
            raise TypeError("config must be BehaviorWriteConfig")
        self.tree = tree
        self.registry = tree_registry
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
        source_payload = self._with_system_fields(normalized_kind, payload)
        timestamp = self._timestamp()
        document = self.tree.document_codec.build(
            normalized_kind,
            source_payload,
            metadata=BehaviorDocumentMetadata.initial(timestamp),
        )
        uri = BehaviorURI.from_address(document.address)
        directory_uri = BehaviorURI.from_directory(BehaviorDirectory.for_address(document.address))
        with self._fenced_uris(uri, directory_uri, *self._reference_uris(document)):
            self._validate_references(document)
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

    def _validate_references(self, document: BehaviorDocument) -> None:
        if document.kind is BehaviorKind.OUTCOME:
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
            return
        if document.kind is BehaviorKind.EPISODE:
            episode_fields = self.registry.validate(BehaviorKind.EPISODE, document.fields)
            episode_event_fields: list[dict[str, Any]] = []
            for uri_text in episode_fields["ordered_event_uris"]:
                referenced = self.tree.read(BehaviorURI.parse(uri_text).to_address())
                if referenced.kind is not BehaviorKind.EVENT:
                    raise ValueError("Episode event reference does not resolve to an Event")
                fields = self.registry.validate(BehaviorKind.EVENT, referenced.fields)
                if (
                    fields["started_at"] < episode_fields["started_at"]
                    or fields["started_at"] > episode_fields["ended_at"]
                ):
                    raise ValueError("Episode Event starts outside the Episode time window")
                if fields["ended_at"] is not None and fields["ended_at"] > episode_fields["ended_at"]:
                    raise ValueError("Episode Event ends outside the Episode time window")
                episode_event_fields.append(fields)
            for index, earlier in enumerate(episode_event_fields):
                for later in episode_event_fields[index + 1 :]:
                    if later["ended_at"] is not None and later["ended_at"] <= earlier["started_at"]:
                        raise ValueError("Episode ordered_event_uris contradict the real Event order")

            event_uris = set(episode_fields["ordered_event_uris"])
            snapshots_by_uri = {snapshot["uri"]: snapshot for snapshot in episode_fields["outcome_snapshots"]}
            for uri_text in episode_fields["outcome_uris"]:
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

    def _with_system_fields(
        self,
        kind: BehaviorKind,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """只由系统为 Episode 绑定创建时看到的 Outcome 版本证据。"""

        if kind is not BehaviorKind.EPISODE:
            return payload
        if "outcome_snapshots" in payload:
            raise ValueError("outcome_snapshots is system-owned and must not be supplied by the caller")
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
        enriched = dict(payload)
        enriched["outcome_snapshots"] = frozen
        return enriched

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
    def _fenced_uris(self, *uris: BehaviorURI) -> Iterator[None]:
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
                yield

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
