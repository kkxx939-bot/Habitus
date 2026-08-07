from __future__ import annotations

import asyncio
import hashlib
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from behavior.claim import ClaimNormalizerKind, NormalizerFingerprint
from behavior.config import BehaviorConfig
from behavior.evidence import (
    BehaviorAdapterCapability,
    BehaviorModality,
    BehaviorOriginKind,
    BehaviorRecordKind,
    BehaviorRole,
    BehaviorSemanticInput,
    BehaviorSourceDescriptor,
    BehaviorSourceTrust,
    BehaviorTimeMode,
    ProducerFingerprint,
    ProducerImplementationKind,
    SourceEventRef,
    StreamRef,
)
from behavior.persistence import BehaviorDatabase, SQLiteBehaviorClaimLedger, SQLiteBehaviorEvidenceLedger
from infrastructure.store.contracts import PathLock
from infrastructure.store.sqlite import SQLiteLockStore, SQLiteLockStoreConfig

BASE_TIME = datetime(2026, 8, 6, 1, 2, 3, tzinfo=timezone.utc)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass
class FakeClock:
    value: datetime = BASE_TIME

    def now(self) -> datetime:
        return self.value


class FakeAdapter:
    def __init__(
        self,
        result: BehaviorSemanticInput | object,
        *,
        name: str = "fake_adapter",
        trust: BehaviorSourceTrust = BehaviorSourceTrust.MODEL_INFERRED,
        time_mode: BehaviorTimeMode = BehaviorTimeMode.BACKFILL,
        origins: tuple[BehaviorOriginKind, ...] = (BehaviorOriginKind.DIRECT_PERCEPTION,),
        kinds: tuple[BehaviorRecordKind, ...] = (BehaviorRecordKind.ACTIVITY_SEGMENT,),
        modalities: tuple[BehaviorModality, ...] = (BehaviorModality.VISION,),
        role_pairs: tuple[tuple[BehaviorRole, BehaviorRole | None], ...] = (
            (BehaviorRole.USER, BehaviorRole.USER),
        ),
        maximum_batch_size: int = 32,
        fingerprint: ProducerFingerprint | None = None,
    ) -> None:
        self.name = name
        self.result = result
        self.fingerprint = fingerprint or ProducerFingerprint(
            producer_name=name,
            producer_version="1",
            pipeline_version="1",
            output_schema_version="1",
            implementation_kind=ProducerImplementationKind.ADAPTER,
        )
        self.capabilities = BehaviorAdapterCapability(
            source_trust=trust,
            time_mode=time_mode,
            allowed_origin_kinds=origins,
            allowed_record_kinds=kinds,
            allowed_modalities=modalities,
            allowed_role_pairs=role_pairs,
            maximum_batch_size=maximum_batch_size,
        )

    async def adapt(self, payload: object):
        del payload
        return self.result


def source_descriptor(
    *,
    event: str = "event-1",
    stream: str = "stream-1",
    generation: int = 0,
    sequence: int = 1,
    item_index: int = 0,
    origin: BehaviorOriginKind = BehaviorOriginKind.DIRECT_PERCEPTION,
    content_digest: str | None = None,
) -> BehaviorSourceDescriptor:
    return BehaviorSourceDescriptor(
        source_event_ref=SourceEventRef("test", event),
        stream_ref=StreamRef("test", stream, generation),
        source_sequence=sequence,
        source_item_index=item_index,
        origin_kind=origin,
        source_ref=None,
        source_content_digest=content_digest or digest(event),
        parent_source_event_refs=(),
        correlation_refs=(),
        causal_refs=(),
        projection_ref=None,
    )


class SQLiteProcessingLock:
    def __init__(self, tmp_path) -> None:
        store = SQLiteLockStore(
            tmp_path / "processing-locks.sqlite3",
            config=SQLiteLockStoreConfig(),
            initialize=True,
        )
        self.path_lock = PathLock(store)

    @asynccontextmanager
    async def acquire(self, processing_identity: str):
        context = self.path_lock.acquire(
            "behavior-processing:" + processing_identity,
            ttl_seconds=30,
            wait_timeout_seconds=5.0,
        )
        guard = await asyncio.to_thread(context.__enter__)
        try:
            yield guard
        finally:
            await asyncio.to_thread(context.__exit__, None, None, None)


class FakeModelNormalizer:
    kind = ClaimNormalizerKind.MODEL

    def __init__(self, proposals=(), *, error: Exception | None = None, name: str = "fake_model") -> None:
        self.name = name
        self.proposals = proposals
        self.error = error
        self.calls = 0
        self.fingerprint = NormalizerFingerprint(
            normalizer_name=name,
            normalizer_version="1",
            pipeline_version="1",
            output_schema_version="1",
            kind=self.kind,
            model_provider="test",
            model_name="test-model",
            prompt_version="1",
        )

    async def normalize(self, record):
        del record
        self.calls += 1
        await asyncio.sleep(0)
        if self.error is not None:
            raise self.error
        return tuple(self.proposals)


@pytest.fixture
def behavior_config() -> BehaviorConfig:
    return BehaviorConfig()


@pytest.fixture
def ledgers(tmp_path, behavior_config):
    database = BehaviorDatabase(tmp_path / "behavior", config=behavior_config, initialize=True)
    return (
        database,
        SQLiteBehaviorEvidenceLedger(database),
        SQLiteBehaviorClaimLedger(database),
    )
