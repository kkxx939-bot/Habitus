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
        MemoryKind.ENTITY: {"category": "项目", "name": f"Habitus-{suffix}", "summary": "记忆系统。"},
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


def test_new_memory_is_warm_and_only_explicit_actual_use_makes_it_hot(tmp_path: Path) -> None:
    instance = lifecycle(tmp_path)
    item = candidate()

    initial = instance.rank((item,), now=NOW)[0]
    assert initial.hotness == pytest.approx(0.5)
    assert initial.temperature is MemoryTemperature.WARM
    assert instance.store.read_many((item.uri,)) == ()

    state = instance.record_use((item.target,), used_at=NOW)[0]
    used = instance.rank((item,), now=NOW)[0]
    assert state.useful_recall_count == 1
    assert used.hotness == pytest.approx(2 / 3)
    assert used.temperature is MemoryTemperature.HOT


def test_memory_dynamically_degrades_hot_warm_cold1_cold2_and_reheats_on_use(tmp_path: Path) -> None:
    config = MemoryRecallLifecycleConfig(preference_half_life_days=10)
    instance = lifecycle(tmp_path, config)
    item = candidate()
    instance.record_use((item.target,), used_at=NOW)

    assert instance.rank((item,), now=NOW)[0].temperature is MemoryTemperature.HOT
    assert instance.rank((item,), now=NOW + timedelta(days=10))[0].temperature is MemoryTemperature.WARM
    assert instance.rank((item,), now=NOW + timedelta(days=20))[0].temperature is MemoryTemperature.COLD_1
    assert instance.rank((item,), now=NOW + timedelta(days=40))[0].temperature is MemoryTemperature.COLD_2
    reheated = instance.record_use((item.target,), used_at=NOW + timedelta(days=40))[0]
    assert reheated.useful_recall_count == 2
    assert instance.rank((item,), now=NOW + timedelta(days=40))[0].temperature is MemoryTemperature.HOT


def test_memory_kinds_use_distinct_half_lives_without_fixed_hot_expiry() -> None:
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
    [
        (0.079, MemoryTemperature.COLD_2),
        (0.08, MemoryTemperature.COLD_1),
        (0.199, MemoryTemperature.COLD_1),
        (0.2, MemoryTemperature.WARM),
        (0.6, MemoryTemperature.WARM),
        (0.601, MemoryTemperature.HOT),
    ],
)
def test_temperature_boundaries_are_explicit(hotness: float, expected: MemoryTemperature) -> None:
    assert memory_temperature(
        hotness,
        cold2_threshold=0.08,
        cold1_threshold=0.2,
        hot_threshold=0.6,
    ) is expected


def test_cold2_probe_must_be_inside_semantic_topk_and_never_expands_k(tmp_path: Path) -> None:
    config = MemoryRecallLifecycleConfig(
        preference_half_life_days=1,
        cold2_probe_interval_days=30,
        cold2_probe_limit=2,
        max_cold2_probes_per_search=1,
    )
    instance = lifecycle(tmp_path, config)
    warm = candidate(MemoryKind.PROFILE, semantic_score=0.8, suffix="warm")
    cold2 = candidate(semantic_score=0.95, updated_at=NOW - timedelta(days=10), suffix="cold2")

    first = instance.select((cold2, warm), limit=1, now=NOW)
    assert tuple(item.candidate.uri for item in first) == (cold2.uri,)
    assert instance.store.read_many((cold2.uri,)) == ()

    stronger_warm = replace(warm, semantic_score=0.99)
    outside = instance.select((cold2, stronger_warm), limit=1, now=NOW)
    assert tuple(item.candidate.uri for item in outside) == (warm.uri,)

    bounded = instance.select((cold2, warm), limit=2, now=NOW)
    assert len(bounded) == 2
    assert {item.candidate.uri for item in bounded} == {cold2.uri, warm.uri}


def test_compacted_cold2_retires_after_inactivity_and_explicit_grace_without_probe_quota(
    tmp_path: Path,
) -> None:
    config = MemoryRecallLifecycleConfig(
        preference_half_life_days=1,
        preference_retire_days=5,
        cold2_probe_interval_days=1,
        cold2_probe_limit=2,
        retire_candidate_grace_days=1,
    )
    instance = lifecycle(tmp_path, config)
    item = candidate(updated_at=NOW - timedelta(days=20))
    compacted = instance.mark_compacted(
        item.target,
        lifecycle_activity_at=item.updated_at,
        compacted_at=NOW - timedelta(days=10),
        expected_version=0,
    )
    assert compacted.compacted_at == NOW - timedelta(days=10)
    ranking = instance.rank((item,), now=NOW)[0]
    assert instance.retirement_eligible(ranking, now=NOW)
    candidate_state = instance.mark_retire_candidate(
        item.target,
        marked_at=NOW,
        expected_version=compacted.version,
    )
    assert not instance.retirement_grace_elapsed(candidate_state, now=NOW)
    assert instance.retirement_grace_elapsed(candidate_state, now=NOW + timedelta(days=1))


def test_new_business_revision_exits_old_compaction_and_keeps_use_history(tmp_path: Path) -> None:
    instance = lifecycle(tmp_path)
    old = candidate(revision=1)
    instance.record_use((old.target,), used_at=NOW)
    instance.mark_compacted(
        old.target,
        lifecycle_activity_at=NOW,
        compacted_at=NOW,
        expected_version=1,
    )
    revised = replace(
        old,
        target=MemoryRecallTarget(old.uri, 2, old.target.document_created_at),
        updated_at=NOW + timedelta(days=1),
    )

    ranking = instance.rank((revised,), now=NOW + timedelta(days=1))[0]
    assert ranking.state.useful_recall_count == 1
    assert ranking.state.compacted_at is None
    assert ranking.state.lifecycle_activity_at == revised.updated_at
    persisted = instance.record_use((revised.target,), used_at=NOW + timedelta(days=1))[0]
    assert persisted.document_revision == 2
    assert persisted.useful_recall_count == 2
    assert persisted.compacted_at is None


def test_probe_on_new_revision_cannot_resurrect_old_compaction_or_retirement_state(
    tmp_path: Path,
) -> None:
    instance = lifecycle(tmp_path)
    old = candidate(revision=1)
    compacted = instance.mark_compacted(
        old.target,
        lifecycle_activity_at=NOW,
        compacted_at=NOW,
        expected_version=0,
    )
    candidate_state = instance.mark_retire_candidate(
        old.target,
        marked_at=NOW + timedelta(days=1),
        expected_version=compacted.version,
    )
    instance.mark_retired(
        old.target,
        retired_at=NOW + timedelta(days=2),
        expected_version=candidate_state.version,
    )
    revised = MemoryRecallTarget(old.uri, 2, old.target.document_created_at)

    state = instance.store.record_probe((revised,), probed_at=NOW + timedelta(days=3))[0]

    assert state.document_revision == 2
    assert state.compacted_at is None
    assert state.retire_candidate_at is None
    assert state.retired_at is None


def test_recreated_uri_does_not_inherit_previous_generation_heat(tmp_path: Path) -> None:
    instance = lifecycle(tmp_path)
    old = candidate()
    instance.record_use((old.target,), used_at=NOW)
    recreated = replace(
        old,
        target=MemoryRecallTarget(old.uri, 1, NOW + timedelta(days=1)),
        updated_at=NOW + timedelta(days=1),
    )
    ranking = instance.rank((recreated,), now=NOW + timedelta(days=1))[0]
    assert ranking.state.useful_recall_count == 0
    assert ranking.temperature is MemoryTemperature.WARM


def test_store_serializes_concurrent_actual_use_updates_without_lost_counts(tmp_path: Path) -> None:
    config = MemoryRecallLifecycleConfig(sqlite_timeout_seconds=10)
    store = SQLiteMemoryRecallLifecycleStore(tmp_path / "recall.sqlite3", config=config)
    item = candidate()
    with ThreadPoolExecutor(max_workers=8) as executor:
        states = tuple(executor.map(lambda _: store.record_use((item.target,), used_at=NOW)[0], range(32)))
    persisted = store.read_many((item.uri,))[0]
    assert persisted.useful_recall_count == 32
    assert persisted.version == 32
    assert {state.version for state in states} == set(range(1, 33))


def test_lowered_count_cap_never_makes_actual_use_count_go_backwards(tmp_path: Path) -> None:
    path = tmp_path / "recall.sqlite3"
    high = MemoryRecallLifecycleConfig(max_useful_recall_count=10)
    item = candidate()
    store = SQLiteMemoryRecallLifecycleStore(path, config=high)
    for index in range(6):
        store.record_use((item.target,), used_at=NOW + timedelta(seconds=index))
    lowered = SQLiteMemoryRecallLifecycleStore(
        path,
        config=MemoryRecallLifecycleConfig(max_useful_recall_count=5),
    )
    state = lowered.record_use((item.target,), used_at=NOW + timedelta(seconds=10))[0]
    assert state.useful_recall_count == 6


def test_store_rejects_conflicting_targets_oversized_batches_and_old_schema(tmp_path: Path) -> None:
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
        conflict_store.record_use(
            (first.target, MemoryRecallTarget(first.uri, 2, first.target.document_created_at)),
            used_at=NOW,
        )
    invalid_path = tmp_path / "invalid.sqlite3"
    with closing(sqlite3.connect(invalid_path)) as connection:
        connection.execute("CREATE TABLE memory_recall_lifecycle (uri TEXT PRIMARY KEY)")
    with pytest.raises(MemoryRecallLifecycleError, match="initialize"):
        SQLiteMemoryRecallLifecycleStore(invalid_path)


def test_lifecycle_score_only_degrades_positive_and_negative_semantic_scores() -> None:
    assert lifecycle_adjusted_score(0.8, 0.5, alpha=0.2) == pytest.approx(0.72)
    assert lifecycle_adjusted_score(-0.8, 0.5, alpha=0.2) == pytest.approx(-0.8 / 0.9)
    assert lifecycle_adjusted_score(0.8, 1.0, alpha=0.2) == pytest.approx(0.8)


def test_disabled_lifecycle_neither_reads_nor_writes_store() -> None:
    class UnusedStore:
        def __getattr__(self, _name):
            def fail(*_args, **_kwargs):
                raise AssertionError("disabled lifecycle touched its store")

            return fail

    instance = MemoryRecallLifecycle(UnusedStore(), config=MemoryRecallLifecycleConfig(enabled=False))
    item = candidate()
    ranking = instance.rank((item,), now=NOW)[0]
    assert ranking.final_score == item.semantic_score
    assert ranking.temperature is None
    assert instance.record_use((item.target,), used_at=NOW) == ()
    assert instance.forget((item.uri,)) == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"ranking_alpha": 0.951},
        {"ranking_alpha": 0},
        {"cold2_threshold": 0},
        {"cold2_threshold": 0.2, "cold1_threshold": 0.2},
        {"cold1_threshold": 0.61, "hot_threshold": 0.6},
        {"hot_threshold": 2 / 3},
        {"event_half_life_days": 0},
        {"profile_retire_days": 0},
        {"cold2_probe_limit": 0},
        {"retire_candidate_grace_days": 0},
        {"max_cold2_probes_per_search": 2},
        {"sqlite_timeout_seconds": 61},
        {"max_batch_size": True},
        {"enabled": 1},
    ],
)
def test_lifecycle_config_rejects_invalid_policy(changes: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        MemoryRecallLifecycleConfig(**changes)  # type: ignore[arg-type]
