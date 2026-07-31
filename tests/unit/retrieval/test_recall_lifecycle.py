from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from memory.model import MemoryKind
from memory.retrieval import (
    MemoryRecallCandidate,
    MemoryRecallLifecycle,
    MemoryRecallLifecycleConfig,
    MemoryRecallLifecycleError,
    MemoryRecallState,
    MemoryRecallTarget,
    MemoryTemperature,
    SQLiteMemoryRecallLifecycleStore,
    lifecycle_adjusted_score,
    memory_hotness,
    memory_temperature,
)
from memory.uri import MemoryURI
from tests.helpers import document

NOW = datetime(2026, 7, 31, 8, 0, tzinfo=timezone.utc)


def candidate(
    kind: MemoryKind = MemoryKind.PREFERENCE,
    *,
    semantic_score: float = 0.9,
    updated_at: datetime = NOW,
    revision: int = 1,
    suffix: str = "a",
) -> MemoryRecallCandidate:
    fields = {
        MemoryKind.PREFERENCE: {"topic": f"回答风格-{suffix}", "content": "- 偏好简洁回答"},
        MemoryKind.ENTITY: {"category": "项目", "name": f"m2bOS-{suffix}", "summary": "记忆系统。"},
    }.get(kind)
    item = document(kind, fields=fields, revision=revision, timestamp=updated_at)
    return MemoryRecallCandidate(
        MemoryRecallTarget(
            MemoryURI.from_address(item.address),
            item.metadata.revision,
            item.metadata.created_at,
        ),
        item.kind,
        item.metadata.updated_at,
        semantic_score,
    )


def lifecycle(tmp_path: Path, config: MemoryRecallLifecycleConfig | None = None) -> MemoryRecallLifecycle:
    resolved = config or MemoryRecallLifecycleConfig()
    return MemoryRecallLifecycle(
        SQLiteMemoryRecallLifecycleStore(tmp_path / "recall.sqlite3", config=resolved),
        config=resolved,
    )


def test_new_memory_is_warm_and_first_successful_support_makes_it_hot(tmp_path: Path) -> None:
    instance = lifecycle(tmp_path)
    item = candidate()

    initial = instance.rank((item,), now=NOW)[0]
    assert initial.state == MemoryRecallState.initial(
        item.uri,
        item.target.document_revision,
        item.target.document_created_at,
    )
    assert initial.hotness == pytest.approx(0.5)
    assert initial.temperature is MemoryTemperature.WARM

    state = instance.record_success((item.target,), recalled_at=NOW)[0]
    used = instance.rank((item,), now=NOW)[0]
    assert state.successful_recall_count == 1
    assert used.hotness == pytest.approx(2 / 3)
    assert used.temperature is MemoryTemperature.HOT


def test_unused_memory_dynamically_degrades_from_hot_to_warm_and_cold(tmp_path: Path) -> None:
    config = MemoryRecallLifecycleConfig(preference_half_life_days=10)
    instance = lifecycle(tmp_path, config)
    item = candidate()
    instance.record_success((item.target,), recalled_at=NOW)

    assert instance.rank((item,), now=NOW)[0].temperature is MemoryTemperature.HOT
    assert instance.rank((item,), now=NOW + timedelta(days=10))[0].temperature is MemoryTemperature.WARM
    assert instance.rank((item,), now=NOW + timedelta(days=20))[0].temperature is MemoryTemperature.COLD
    reheated = instance.record_success((item.target,), recalled_at=NOW + timedelta(days=20))[0]
    assert reheated.successful_recall_count == 2
    assert instance.rank((item,), now=NOW + timedelta(days=20))[0].temperature is MemoryTemperature.HOT


def test_memory_kinds_use_distinct_half_lives_without_changing_document_timestamps() -> None:
    config = MemoryRecallLifecycleConfig()
    profile = candidate(MemoryKind.PROFILE, updated_at=NOW - timedelta(days=60))
    event = candidate(MemoryKind.EVENT, updated_at=NOW - timedelta(days=60))

    profile_heat = memory_hotness(
        MemoryRecallState.initial(profile.uri, 1, profile.target.document_created_at),
        updated_at=profile.updated_at,
        now=NOW,
        half_life_days=config.half_life_days(profile.kind),
    )
    event_heat = memory_hotness(
        MemoryRecallState.initial(event.uri, 1, event.target.document_created_at),
        updated_at=event.updated_at,
        now=NOW,
        half_life_days=config.half_life_days(event.kind),
    )
    assert profile_heat == pytest.approx(0.5 * 2 ** (-60 / 180))
    assert event_heat == pytest.approx(0.5 * 2 ** (-60 / 14))
    assert profile_heat > event_heat


@pytest.mark.parametrize(
    ("hotness", "expected"),
    [(0.199, MemoryTemperature.COLD), (0.2, MemoryTemperature.WARM), (0.6, MemoryTemperature.WARM), (0.601, MemoryTemperature.HOT)],
)
def test_temperature_boundaries_are_explicit(hotness: float, expected: MemoryTemperature) -> None:
    assert memory_temperature(hotness, cold_threshold=0.2, hot_threshold=0.6) is expected


def test_lifecycle_score_only_degrades_positive_and_negative_semantic_scores() -> None:
    assert lifecycle_adjusted_score(0.8, 0.5, alpha=0.2) == pytest.approx(0.72)
    assert lifecycle_adjusted_score(-0.8, 0.5, alpha=0.2) == pytest.approx(-0.8 / 0.9)
    assert lifecycle_adjusted_score(0.8, 1.0, alpha=0.2) == pytest.approx(0.8)


def test_lifecycle_can_reorder_close_matches_but_not_override_a_large_semantic_gap(tmp_path: Path) -> None:
    config = MemoryRecallLifecycleConfig(preference_half_life_days=10)
    instance = lifecycle(tmp_path, config)
    cold = candidate(semantic_score=0.9, updated_at=NOW - timedelta(days=20), suffix="cold")
    hot = candidate(semantic_score=0.8, suffix="hot")
    instance.record_success((hot.target,), recalled_at=NOW)

    close = instance.rank((cold, hot), now=NOW)
    assert tuple(item.candidate.uri for item in close) == (hot.uri, cold.uri)
    strong_cold = replace(cold, semantic_score=0.95)
    weaker_hot = replace(hot, semantic_score=0.7)
    separated = instance.rank((strong_cold, weaker_hot), now=NOW)
    assert tuple(item.candidate.uri for item in separated) == (cold.uri, hot.uri)


def test_document_revision_does_not_inherit_previous_content_heat(tmp_path: Path) -> None:
    instance = lifecycle(tmp_path)
    old = candidate(revision=1)
    instance.record_success((old.target,), recalled_at=NOW)
    revised = replace(
        old,
        target=MemoryRecallTarget(old.uri, 2, old.target.document_created_at),
        updated_at=NOW + timedelta(seconds=1),
    )

    ranking = instance.rank((revised,), now=NOW + timedelta(seconds=1))[0]
    assert ranking.state.successful_recall_count == 0
    assert ranking.temperature is MemoryTemperature.WARM
    reset = instance.record_success((revised.target,), recalled_at=NOW + timedelta(seconds=1))[0]
    assert reset.document_revision == 2
    assert reset.successful_recall_count == 1


def test_late_old_revision_cannot_overwrite_newer_revision_state(tmp_path: Path) -> None:
    instance = lifecycle(tmp_path)
    old = candidate(revision=1)
    new = replace(
        old,
        target=MemoryRecallTarget(old.uri, 2, old.target.document_created_at),
        updated_at=NOW + timedelta(seconds=1),
    )
    instance.record_success((old.target,), recalled_at=NOW)
    current = instance.record_success((new.target,), recalled_at=NOW + timedelta(seconds=2))[0]
    late = instance.record_success((old.target,), recalled_at=NOW + timedelta(seconds=1))[0]

    assert current.document_revision == 2
    assert late == current
    assert instance.store.read_many((old.uri,))[0] == current


def test_recreated_uri_uses_new_content_generation_and_does_not_inherit_heat(tmp_path: Path) -> None:
    instance = lifecycle(tmp_path)
    old = candidate()
    for index in range(5):
        instance.record_success((old.target,), recalled_at=NOW + timedelta(seconds=index))
    recreated = replace(
        old,
        target=MemoryRecallTarget(old.uri, 1, NOW + timedelta(days=1)),
        updated_at=NOW + timedelta(days=1),
    )

    ranking = instance.rank((recreated,), now=NOW + timedelta(days=1))[0]
    assert ranking.state.successful_recall_count == 0
    assert ranking.hotness == pytest.approx(0.5)
    assert ranking.temperature is MemoryTemperature.WARM
    replaced = instance.record_success((recreated.target,), recalled_at=NOW + timedelta(days=1))[0]
    assert replaced.document_created_at == recreated.target.document_created_at
    assert replaced.successful_recall_count == 1


def test_forget_removes_retired_state_idempotently(tmp_path: Path) -> None:
    instance = lifecycle(tmp_path)
    item = candidate()
    instance.record_success((item.target,), recalled_at=NOW)
    assert instance.forget((item.uri,)) == 1
    assert instance.store.read_many((item.uri,)) == ()
    assert instance.forget((item.uri,)) == 0


def test_store_is_lazy_when_requested_and_persists_atomic_success_counts(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "recall.sqlite3"
    config = MemoryRecallLifecycleConfig()
    store = SQLiteMemoryRecallLifecycleStore(path, config=config, initialize=False)
    item = candidate()
    assert not path.exists()
    assert not store.initialized

    store.initialize()
    assert path.is_file()
    first = store.record_success((item.target,), recalled_at=NOW)[0]
    second = store.record_success((item.target,), recalled_at=NOW - timedelta(days=1))[0]
    reopened = SQLiteMemoryRecallLifecycleStore(path, config=config)
    persisted = reopened.read_many((item.uri,))[0]
    assert first.successful_recall_count == 1
    assert second.successful_recall_count == 2
    assert second.last_successful_recall_at == NOW
    assert persisted == second


def test_store_serializes_concurrent_success_updates_without_lost_counts(tmp_path: Path) -> None:
    config = MemoryRecallLifecycleConfig(sqlite_timeout_seconds=10)
    store = SQLiteMemoryRecallLifecycleStore(tmp_path / "recall.sqlite3", config=config)
    item = candidate()

    with ThreadPoolExecutor(max_workers=8) as executor:
        states = tuple(executor.map(lambda _: store.record_success((item.target,), recalled_at=NOW)[0], range(32)))

    persisted = store.read_many((item.uri,))[0]
    assert persisted.successful_recall_count == 32
    assert persisted.version == 32
    assert {state.version for state in states} == set(range(1, 33))


def test_lowered_count_cap_never_makes_a_successful_recall_count_go_backwards(tmp_path: Path) -> None:
    path = tmp_path / "recall.sqlite3"
    high = MemoryRecallLifecycleConfig(max_successful_recall_count=10)
    item = candidate()
    store = SQLiteMemoryRecallLifecycleStore(path, config=high)
    for index in range(6):
        store.record_success((item.target,), recalled_at=NOW + timedelta(seconds=index))

    lowered = SQLiteMemoryRecallLifecycleStore(
        path,
        config=MemoryRecallLifecycleConfig(max_successful_recall_count=5),
    )
    state = lowered.record_success((item.target,), recalled_at=NOW + timedelta(seconds=10))[0]
    assert state.successful_recall_count == 6


def test_store_rejects_conflicting_revisions_oversized_batches_and_bad_schema(tmp_path: Path) -> None:
    config = MemoryRecallLifecycleConfig(max_batch_size=1)
    store = SQLiteMemoryRecallLifecycleStore(tmp_path / "recall.sqlite3", config=config)
    first = candidate(suffix="one")
    second = candidate(suffix="two")
    with pytest.raises(ValueError, match="max_batch_size"):
        store.read_many((first.uri, second.uri))
    conflict_store = SQLiteMemoryRecallLifecycleStore(
        tmp_path / "conflict.sqlite3",
        config=MemoryRecallLifecycleConfig(max_batch_size=2),
    )
    with pytest.raises(ValueError, match="conflict"):
        conflict_store.record_success(
            (first.target, MemoryRecallTarget(first.uri, 2, first.target.document_created_at)),
            recalled_at=NOW,
        )

    invalid_path = tmp_path / "invalid.sqlite3"
    with closing(sqlite3.connect(invalid_path)) as connection:
        connection.execute("CREATE TABLE memory_recall_lifecycle (uri TEXT PRIMARY KEY)")
    with pytest.raises(MemoryRecallLifecycleError, match="initialize"):
        SQLiteMemoryRecallLifecycleStore(invalid_path)


def test_disabled_lifecycle_neither_reads_store_nor_changes_scores() -> None:
    class UnusedStore:
        def initialize(self) -> None:
            raise AssertionError("disabled lifecycle initialized its store")

        def read_many(self, uris):
            raise AssertionError("disabled lifecycle read its store")

        def record_success(self, targets, *, recalled_at):
            raise AssertionError("disabled lifecycle wrote its store")

        def delete_many(self, uris):
            raise AssertionError("disabled lifecycle deleted from its store")

    instance = MemoryRecallLifecycle(UnusedStore(), config=MemoryRecallLifecycleConfig(enabled=False))
    item = candidate()
    ranking = instance.rank((item,), now=NOW)[0]
    assert ranking.final_score == item.semantic_score
    assert ranking.hotness is None
    assert ranking.temperature is None
    assert instance.record_success((item.target,), recalled_at=NOW) == ()
    assert instance.forget((item.uri,)) == 0


@pytest.mark.parametrize(
    "config",
    [
        MemoryRecallLifecycleConfig(),
        MemoryRecallLifecycleConfig(enabled=False, ranking_alpha=0),
        MemoryRecallLifecycleConfig(ranking_alpha=0.95),
    ],
)
def test_valid_lifecycle_configs_are_immutable(config: MemoryRecallLifecycleConfig) -> None:
    with pytest.raises((AttributeError, TypeError)):
        config.enabled = False  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"ranking_alpha": 0.951},
        {"ranking_alpha": 0},
        {"cold_threshold": 0},
        {"cold_threshold": 0.6, "hot_threshold": 0.6},
        {"cold_threshold": 0.51, "hot_threshold": 0.6},
        {"cold_threshold": 0.2, "hot_threshold": 0.49},
        {"cold_threshold": 0.2, "hot_threshold": 2 / 3},
        {"event_half_life_days": 0},
        {"sqlite_timeout_seconds": 61},
        {"max_batch_size": True},
        {"enabled": 1},
    ],
)
def test_lifecycle_config_rejects_invalid_policy(changes: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        MemoryRecallLifecycleConfig(**changes)  # type: ignore[arg-type]
