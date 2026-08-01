"""在任意远程向量 Backend 之上实现统一的可恢复发布语义。"""

from __future__ import annotations

import asyncio
import hashlib
import threading
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace

from infrastructure.store.contracts import LeaseGuard, PathLock
from infrastructure.vector.contracts import RawVectorBackend
from infrastructure.vector.model import (
    VectorPublicationSnapshot,
    VectorStoreBusyError,
    VectorStoreConflictError,
    VectorStoreFilter,
    VectorStoreIntegrityError,
    VectorStoreMatch,
    VectorStoreRecord,
    VectorStoreState,
    VectorStoreUnsupportedTopologyError,
)
from ModelClient import EmbeddingVector

_PUBLICATION_FORMAT = "m2bos_vector_publication_v1"
_STATE_METADATA = "state"
_CLAIM_METADATA = "claim"
_SHARED_LOCKS_GUARD = threading.Lock()
_SHARED_LOCKS: dict[str, threading.Lock] = {}


class PublishedVectorStore:
    """集中实现全量、增量、代次、检查点和失败重放，厂商只实现物理操作。"""

    def __init__(
        self,
        backend: RawVectorBackend,
        *,
        path_lock: PathLock,
        publication_lock_key: str | None = None,
    ) -> None:
        _validate_backend(backend)
        if not isinstance(path_lock, PathLock):
            raise TypeError("path_lock must be PathLock")
        self.backend = backend
        self._initialize_lock = asyncio.Lock()
        self._path_lock = path_lock
        self._publication_lock_key = publication_lock_key or _default_publication_lock_key(backend)
        if (
            not isinstance(self._publication_lock_key, str)
            or not self._publication_lock_key
            or self._publication_lock_key != self._publication_lock_key.strip()
        ):
            raise ValueError("publication_lock_key must be normalized non-empty text")
        if (
            backend.requires_cross_process_publication_fencing
            and getattr(path_lock.lock_store, "coordination_scope", None) != "host"
        ):
            raise VectorStoreUnsupportedTopologyError(
                "remote vector publication requires a host-scoped publication PathLock"
            )
        coordination_domain = getattr(path_lock.lock_store, "coordination_domain", None)
        if not isinstance(coordination_domain, str) or not coordination_domain:
            coordination_domain = f"object:{id(path_lock.lock_store)}"
        self._coordination_domain = hashlib.sha256(coordination_domain.encode("utf-8")).hexdigest()
        self._initialized = False

    @property
    def adapter_name(self) -> str:
        return self.backend.adapter_name

    @property
    def provider_name(self) -> str:
        return self.backend.provider_name

    @property
    def collection(self) -> str:
        return self.backend.collection

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._initialize_lock:
            if self._initialized:
                return
            await self.backend.initialize()
            publication = await self._load_publication()
            _validate_publication(publication, operation="initialize")
            if publication.state is not None:
                await self.backend.ensure_schema(
                    publication.state.dimension,
                    published_dimension=publication.state.dimension,
                )
            self._initialized = True

    async def state(self) -> VectorStoreState | None:
        await self.initialize()
        publication = await self._load_publication()
        _validate_publication(publication, operation="read")
        if publication.state is None:
            return None
        return replace(publication.state, ready=not publication.building)

    async def read(self, identities: tuple[str, ...]) -> tuple[VectorStoreRecord, ...]:
        normalized = _identities(identities)
        if not normalized:
            return ()
        await self.initialize()
        for _attempt in range(8):
            publication = await self._load_publication()
            _validate_publication(publication, operation="read")
            if publication.building:
                raise VectorStoreBusyError("vector index is being rebuilt")
            records = () if publication.state is None else tuple(await self.backend.read(normalized))
            _validate_read_result(records, requested=normalized)
            if await self._publication_is_unchanged(publication, operation="read"):
                return records
        raise VectorStoreConflictError("vector publication changed continuously during read")

    async def replace_all(
        self,
        records: tuple[VectorStoreRecord, ...],
        *,
        schema_version: str,
        embedding_fingerprint: str,
        dimension: int,
        checkpoint: int,
        expected_generation: int | None,
    ) -> VectorStoreState:
        normalized = _records(records, dimension=dimension, maximum=self.backend.max_records)
        _state_inputs(schema_version, embedding_fingerprint, dimension, checkpoint)
        await self.initialize()
        async with self._mutation_guard():
            publication = await self._load_publication()
            current = _current_state(publication, operation="full replacement")
            _assert_generation(current, expected_generation)
            await self.backend.ensure_schema(
                dimension,
                published_dimension=None if current is None else current.dimension,
            )
            await self.backend.validate_records(normalized, replacing=True)
            await self._write_claim(building=True, dimension=dimension)
            await self.backend.delete_all()
            await self.backend.upsert(normalized)
            await self.backend.wait_visible(normalized, (), complete=True)
            next_state = VectorStoreState(
                schema_version=schema_version,
                embedding_fingerprint=embedding_fingerprint,
                dimension=dimension,
                checkpoint=checkpoint,
                generation=1 if current is None else current.generation + 1,
                record_count=len(normalized),
            )
            await self._write_state(next_state)
            await self._write_claim(building=False, dimension=dimension)
            return next_state

    async def apply(
        self,
        upserts: tuple[VectorStoreRecord, ...],
        deletes: tuple[str, ...],
        *,
        checkpoint: int,
        expected_generation: int,
        expected_checkpoint: int,
    ) -> VectorStoreState:
        normalized_upserts = _records(upserts, maximum=self.backend.max_records)
        normalized_deletes = _identities(deletes)
        if {item.identity for item in normalized_upserts} & set(normalized_deletes):
            raise ValueError("one vector identity cannot be upserted and deleted together")
        _non_negative_int(checkpoint, "checkpoint")
        _positive_int(expected_generation, "expected_generation")
        _non_negative_int(expected_checkpoint, "expected_checkpoint")
        if checkpoint < expected_checkpoint:
            raise ValueError("vector checkpoint cannot move backwards")
        await self.initialize()
        async with self._mutation_guard():
            publication = await self._load_publication()
            current = _current_state(publication, operation="incremental apply")
            if current is None:
                raise VectorStoreConflictError("vector collection has not published an index")
            if current.generation != expected_generation or current.checkpoint != expected_checkpoint:
                raise VectorStoreConflictError("vector state changed before incremental apply")
            if publication.building:
                raise VectorStoreConflictError(
                    "vector backend has an unfinished publication and requires a full rebuild"
                )
            if any(item.vector.dimension != current.dimension for item in normalized_upserts):
                raise ValueError("incremental vector dimensions do not match the published collection")
            await self.backend.validate_records(normalized_upserts, replacing=False)
            changed = tuple(item.identity for item in normalized_upserts) + normalized_deletes
            existing = {item.identity for item in await self.read(changed)}
            result_count = (
                current.record_count
                + sum(item.identity not in existing for item in normalized_upserts)
                - sum(identity in existing for identity in normalized_deletes)
            )
            if result_count < 0 or result_count > self.backend.max_records:
                raise ValueError("incremental vector write exceeds the configured record capacity")
            await self._write_claim(building=True, dimension=current.dimension)
            await self.backend.upsert(normalized_upserts)
            await self.backend.delete(normalized_deletes)
            await self.backend.wait_visible(normalized_upserts, normalized_deletes, complete=False)
            await self._assert_state_unchanged(current)
            next_state = VectorStoreState(
                schema_version=current.schema_version,
                embedding_fingerprint=current.embedding_fingerprint,
                dimension=current.dimension,
                checkpoint=checkpoint,
                generation=current.generation + 1,
                record_count=result_count,
            )
            await self._write_state(next_state)
            await self._write_claim(building=False, dimension=current.dimension)
            return next_state

    async def search(
        self,
        query_vector: EmbeddingVector,
        *,
        filters: VectorStoreFilter,
        limit: int,
    ) -> tuple[VectorStoreMatch, ...]:
        if not isinstance(query_vector, EmbeddingVector):
            raise TypeError("query_vector must be EmbeddingVector")
        _search_inputs(filters, limit, self.backend.max_search_hits)
        await self.initialize()
        for _attempt in range(8):
            publication = await self._load_publication()
            _validate_publication(publication, operation="search")
            if publication.building:
                raise VectorStoreBusyError("vector index is being rebuilt")
            state = publication.state
            if state is None:
                matches: tuple[VectorStoreMatch, ...] = ()
            else:
                if query_vector.dimension != state.dimension:
                    raise VectorStoreIntegrityError("query vector dimension does not match the published index")
                matches = tuple(await self.backend.search(query_vector, filters=filters, limit=limit))
                if any(not isinstance(item, VectorStoreMatch) for item in matches):
                    raise VectorStoreIntegrityError("vector backend returned an invalid search result")
            if await self._publication_is_unchanged(publication, operation="search"):
                return matches
        raise VectorStoreConflictError("vector publication changed continuously during search")

    async def scan(
        self,
        *,
        filters: VectorStoreFilter,
        limit: int,
    ) -> tuple[VectorStoreRecord, ...]:
        _search_inputs(filters, limit, self.backend.max_records)
        await self.initialize()
        for _attempt in range(8):
            publication = await self._load_publication()
            _validate_publication(publication, operation="scan")
            if publication.building:
                raise VectorStoreBusyError("vector index is being rebuilt")
            records = () if publication.state is None else tuple(await self.backend.scan(filters=filters, limit=limit))
            _validate_scan_result(records, maximum=limit)
            if await self._publication_is_unchanged(publication, operation="scan"):
                return records
        raise VectorStoreConflictError("vector publication changed continuously during scan")

    async def close(self) -> None:
        await self.backend.close()

    @asynccontextmanager
    async def _mutation_guard(self) -> AsyncIterator[None]:
        async with _shared_mutation_guard(self._publication_lock_key):
            lease_context = self._path_lock.acquire(
                self._publication_lock_key,
                ttl_seconds=30,
                wait_timeout_seconds=30.0,
                retry_delay_seconds=0.02,
            )
            lease_acquisition = asyncio.create_task(asyncio.to_thread(lease_context.__enter__))
            try:
                guard = await asyncio.shield(lease_acquisition)
            except asyncio.CancelledError as cancelled:
                try:
                    await lease_acquisition
                except BaseException as acquisition_error:
                    raise cancelled from acquisition_error
                await asyncio.to_thread(
                    lease_context.__exit__,
                    type(cancelled),
                    cancelled,
                    cancelled.__traceback__,
                )
                raise
            entered = threading.Event()
            release = threading.Event()
            fence_errors: list[BaseException] = []
            fence_task = asyncio.create_task(
                asyncio.to_thread(
                    _hold_publication_fence,
                    guard,
                    entered,
                    release,
                    fence_errors,
                )
            )
            body_error: BaseException | None = None
            try:
                await asyncio.to_thread(entered.wait)
                if fence_errors:
                    raise fence_errors[0]
                yield
            except BaseException as exc:
                body_error = exc
                raise
            finally:
                release.set()
                fence_error: BaseException | None = None
                try:
                    await fence_task
                except BaseException as exc:
                    fence_error = exc
                exit_arguments = (
                    (None, None, None)
                    if body_error is None
                    else (type(body_error), body_error, body_error.__traceback__)
                )
                try:
                    await asyncio.to_thread(lease_context.__exit__, *exit_arguments)
                except BaseException:
                    if body_error is None:
                        raise
                if fence_error is not None and body_error is None:
                    raise fence_error

    async def _load_publication(self) -> VectorPublicationSnapshot:
        values = await self.backend.read_metadata((_STATE_METADATA, _CLAIM_METADATA))
        if not isinstance(values, Mapping) or any(
            not isinstance(name, str) or not isinstance(value, Mapping) for name, value in values.items()
        ):
            raise VectorStoreIntegrityError("vector backend returned invalid publication metadata")
        unknown = set(values) - {_STATE_METADATA, _CLAIM_METADATA}
        if unknown:
            raise VectorStoreIntegrityError("vector backend returned unrequested publication metadata")
        state_value = values.get(_STATE_METADATA)
        claim_value = values.get(_CLAIM_METADATA)
        building = False
        coordination_domain = None
        if claim_value is not None:
            building, coordination_domain = _claim_from_metadata(claim_value)
        publication = VectorPublicationSnapshot(
            state=None if state_value is None else _state_from_metadata(state_value),
            claim_exists=claim_value is not None,
            building=building,
            coordination_domain=coordination_domain,
        )
        return publication

    async def _write_claim(self, *, building: bool, dimension: int) -> None:
        if not isinstance(building, bool):
            raise TypeError("vector publication building state must be boolean")
        await self.backend.write_metadata(
            {
                _CLAIM_METADATA: {
                    "format": _PUBLICATION_FORMAT,
                    "building": building,
                    "coordination_domain": self._coordination_domain,
                }
            },
            dimension=dimension,
        )

    async def _write_state(self, state: VectorStoreState) -> None:
        await self.backend.write_metadata(
            {
                _STATE_METADATA: {
                    "format": _PUBLICATION_FORMAT,
                    "schema_version": state.schema_version,
                    "embedding_fingerprint": state.embedding_fingerprint,
                    "dimension": state.dimension,
                    "checkpoint": state.checkpoint,
                    "generation": state.generation,
                    "record_count": state.record_count,
                }
            },
            dimension=state.dimension,
        )

    async def _assert_state_unchanged(self, expected: VectorStoreState) -> None:
        publication = await self._load_publication()
        if publication.state != expected:
            raise VectorStoreConflictError("vector state changed during a replayable mutation")

    async def _publication_is_unchanged(
        self,
        expected: VectorPublicationSnapshot,
        *,
        operation: str,
    ) -> bool:
        current = await self._load_publication()
        _validate_publication(current, operation=operation)
        return current == expected


def _validate_backend(backend: object) -> None:
    for name in (
        "initialize",
        "read_metadata",
        "write_metadata",
        "ensure_schema",
        "read",
        "delete_all",
        "upsert",
        "delete",
        "validate_records",
        "wait_visible",
        "search",
        "scan",
        "close",
    ):
        if not callable(getattr(backend, name, None)):
            raise TypeError(f"backend must implement RawVectorBackend.{name}")
    for name in ("adapter_name", "provider_name", "collection"):
        if not isinstance(getattr(backend, name, None), str) or not getattr(backend, name):
            raise TypeError(f"backend {name} must be non-empty text")
    for name in ("max_records", "max_search_hits"):
        value = getattr(backend, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise TypeError(f"backend {name} must be a positive integer")
    if not isinstance(
        getattr(backend, "requires_cross_process_publication_fencing", None),
        bool,
    ):
        raise TypeError("backend requires_cross_process_publication_fencing must be boolean")


def _shared_mutation_lock(key: str) -> threading.Lock:
    with _SHARED_LOCKS_GUARD:
        return _SHARED_LOCKS.setdefault(key, threading.Lock())


@asynccontextmanager
async def _shared_mutation_guard(key: str) -> AsyncIterator[None]:
    lock = _shared_mutation_lock(key)
    acquisition = asyncio.create_task(asyncio.to_thread(lock.acquire))
    try:
        acquired = await asyncio.shield(acquisition)
    except asyncio.CancelledError:
        acquired = await acquisition
        if acquired:
            lock.release()
        raise
    if not acquired:  # pragma: no cover - threading.Lock 的阻塞 acquire 契约。
        raise RuntimeError("vector publication process lock was not acquired")
    try:
        yield
    finally:
        lock.release()


def _default_publication_lock_key(backend: RawVectorBackend) -> str:
    identity = "\0".join(
        (
            str(backend.adapter_name),
            str(backend.provider_name),
            str(backend.collection),
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"vector-publication:{digest}"


def _hold_publication_fence(
    guard: LeaseGuard,
    entered: threading.Event,
    release: threading.Event,
    errors: list[BaseException],
) -> None:
    try:
        with guard.fenced():
            entered.set()
            release.wait()
    except BaseException as exc:
        errors.append(exc)
        entered.set()


def _state_from_metadata(value: Mapping[str, object]) -> VectorStoreState:
    expected = {
        "format",
        "schema_version",
        "embedding_fingerprint",
        "dimension",
        "checkpoint",
        "generation",
        "record_count",
    }
    if set(value) != expected or value.get("format") != _PUBLICATION_FORMAT:
        raise VectorStoreIntegrityError("vector publication state metadata has an invalid schema")
    try:
        return VectorStoreState(
            schema_version=_metadata_text(value, "schema_version"),
            embedding_fingerprint=_metadata_text(value, "embedding_fingerprint"),
            dimension=_metadata_int(value, "dimension"),
            checkpoint=_metadata_int(value, "checkpoint"),
            generation=_metadata_int(value, "generation"),
            record_count=_metadata_int(value, "record_count"),
        )
    except (TypeError, ValueError) as exc:
        raise VectorStoreIntegrityError("vector publication state metadata is invalid") from exc


def _claim_from_metadata(value: Mapping[str, object]) -> tuple[bool, str | None]:
    if (
        set(value)
        not in (
            {"format", "building"},
            {"format", "building", "coordination_domain"},
        )
        or value.get("format") != _PUBLICATION_FORMAT
    ):
        raise VectorStoreIntegrityError("vector publication claim metadata has an invalid schema")
    building = value.get("building")
    if not isinstance(building, bool):
        raise VectorStoreIntegrityError("vector publication claim building value is invalid")
    coordination_domain = value.get("coordination_domain")
    if coordination_domain is not None and (
        not isinstance(coordination_domain, str)
        or not coordination_domain
        or coordination_domain != coordination_domain.strip()
    ):
        raise VectorStoreIntegrityError("vector publication claim coordination domain is invalid")
    return building, coordination_domain


def _metadata_text(value: Mapping[str, object], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str):
        raise TypeError(f"vector publication {name} must be text")
    return item


def _metadata_int(value: Mapping[str, object], name: str) -> int:
    item = value.get(name)
    if isinstance(item, bool) or not isinstance(item, int):
        raise TypeError(f"vector publication {name} must be an integer")
    return item


def _validate_publication(publication: object, *, operation: str) -> None:
    if not isinstance(publication, VectorPublicationSnapshot):
        raise VectorStoreIntegrityError("vector backend returned an invalid publication snapshot")
    if publication.state is None:
        if publication.claim_exists and not publication.building:
            raise VectorStoreIntegrityError(f"vector ownership claim is ready without state during {operation}")
        return
    if not publication.claim_exists:
        raise VectorStoreIntegrityError(f"published vector state lost its ownership claim during {operation}")


def _current_state(
    publication: VectorPublicationSnapshot,
    *,
    operation: str,
) -> VectorStoreState | None:
    _validate_publication(publication, operation=operation)
    return publication.state


def _records(
    records: Sequence[VectorStoreRecord],
    *,
    dimension: int | None = None,
    maximum: int,
) -> tuple[VectorStoreRecord, ...]:
    if isinstance(records, str | bytes) or not isinstance(records, Sequence):
        raise TypeError("vector records must be a sequence")
    normalized = tuple(records)
    if len(normalized) > maximum:
        raise ValueError("vector record batch exceeds configured capacity")
    identities: set[str] = set()
    expected_dimension = dimension
    for record in normalized:
        if not isinstance(record, VectorStoreRecord):
            raise TypeError("vector record batch contains an invalid item")
        if record.identity in identities:
            raise ValueError("vector record batch contains duplicate identities")
        identities.add(record.identity)
        expected_dimension = record.vector.dimension if expected_dimension is None else expected_dimension
        if record.vector.dimension != expected_dimension:
            raise ValueError("vector record dimensions are inconsistent")
    return tuple(sorted(normalized, key=lambda item: item.identity))


def _identities(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError("vector identities must be a sequence")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError("vector identity must be normalized non-empty text")
        if value in seen:
            raise ValueError("vector identities must be unique")
        seen.add(value)
        result.append(value)
    return tuple(result)


def _validate_read_result(
    records: Sequence[VectorStoreRecord],
    *,
    requested: tuple[str, ...],
) -> None:
    if any(not isinstance(record, VectorStoreRecord) for record in records):
        raise VectorStoreIntegrityError("vector backend returned an invalid record")
    identities = tuple(record.identity for record in records)
    if len(identities) != len(set(identities)) or not set(identities) <= set(requested):
        raise VectorStoreIntegrityError("vector backend returned duplicate or unrequested records")
    positions = {identity: index for index, identity in enumerate(requested)}
    if tuple(sorted(identities, key=positions.__getitem__)) != identities:
        raise VectorStoreIntegrityError("vector backend did not preserve requested identity order")


def _validate_scan_result(records: Sequence[VectorStoreRecord], *, maximum: int) -> None:
    if len(records) > maximum or any(not isinstance(record, VectorStoreRecord) for record in records):
        raise VectorStoreIntegrityError("vector backend returned an invalid scan result")
    identities = tuple(record.identity for record in records)
    if len(identities) != len(set(identities)):
        raise VectorStoreIntegrityError("vector backend returned duplicate scan identities")


def _state_inputs(schema_version: str, fingerprint: str, dimension: int, checkpoint: int) -> None:
    for name, value in (("schema_version", schema_version), ("embedding_fingerprint", fingerprint)):
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"vector {name} must be normalized non-empty text")
    _positive_int(dimension, "dimension")
    _non_negative_int(checkpoint, "checkpoint")


def _assert_generation(state: VectorStoreState | None, expected: int | None) -> None:
    actual = None if state is None else state.generation
    if actual != expected:
        raise VectorStoreConflictError("vector generation changed before full replacement")


def _positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"vector {name} must be a positive integer")


def _non_negative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"vector {name} must be a non-negative integer")


def _search_inputs(filters: VectorStoreFilter, limit: int, maximum: int) -> None:
    if not isinstance(filters, VectorStoreFilter):
        raise TypeError("filters must be VectorStoreFilter")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= maximum:
        raise ValueError("vector search limit is outside its configured bound")


__all__ = ["PublishedVectorStore"]
