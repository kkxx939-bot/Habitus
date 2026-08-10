"""Prediction Sample 与可分支 Pattern Graph 的双层合同测试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from threading import Barrier

import pytest
from prediction_test_payloads import transition_payload

from prediction import (
    PredictionDocumentCodec,
    PredictionKind,
    PredictionPatternFactory,
    PredictionPatternGenerationPublisher,
    PredictionPatternGenerationState,
    PredictionPatternGenerationStore,
    PredictionPatternGenerationStoreError,
    PredictionPatternGraph,
    PredictionPatternKind,
    PredictionPatternSchemaError,
    PredictionPublisher,
    PredictionSchemaError,
    PredictionSchemaRegistry,
    PredictionTree,
    PredictionTreeIntegrityError,
    PredictionURI,
    derive_sample_identity,
)

UTC = timezone.utc
PATTERN_GENERATION = "1" * 64


def _sample(tree: PredictionTree):
    document = PredictionDocumentCodec(PredictionSchemaRegistry.load_default()).build(
        PredictionKind.TRANSITION,
        transition_payload("pattern-evidence"),
    )
    PredictionPublisher(tree).publish((document,))
    return document


def _state(
    factory: PredictionPatternFactory,
    sample_uri: str,
    *,
    summary: str,
    identity: str,
    generation: str = PATTERN_GENERATION,
):
    return factory.build_state(
        pattern_generation=generation,
        identity={"state": identity},
        fields={
            "semantic_summary": summary,
            "state_level": "event",
            "predicates": [{"field": "temperature", "operator": "gte", "value": 28}],
            "active_goals": ["改善舒适度"],
            "constraints": [],
            "support_count": 12,
            "sample_refs": [sample_uri],
            "statistics": {
                "first_observed_at": datetime(2026, 8, 1, tzinfo=UTC),
                "last_observed_at": datetime(2026, 8, 8, tzinfo=UTC),
                "source_count": 12,
                "confidence": 0.9,
            },
            "projection_version": "behavior-prediction-v2",
        },
    )


def _branch(
    factory: PredictionPatternFactory,
    sample_uri: str,
    *,
    state,
    behavior: str,
    probability: float,
):
    return factory.build_branch(
        state=state,
        identity={"behavior": behavior},
        fields={
            "target_kind": "action",
            "behavior": {
                "actor_role": "owner",
                "behavior_type": "device_control",
                "semantics": behavior,
                "target_refs": ["空调"],
                "parameters": {},
            },
            "conditions": [{"field": "temperature", "operator": "gte", "value": 28}],
            "support_count": 8,
            "conditional_probability": probability,
            "confidence": 0.88,
            "sample_refs": [sample_uri],
            "outcomes": [
                {
                    "semantics": "状态发生变化",
                    "probability": 0.8,
                    "average_delay_seconds": 300,
                    "valence": "positive",
                }
            ],
            "projection_version": "behavior-prediction-v2",
        },
    )


def test_pattern_graph_retrieves_state_content_and_expands_multiple_branches(tmp_path) -> None:
    tree = PredictionTree(tmp_path / "prediction")
    evidence = _sample(tree)
    sample_uri = str(PredictionURI.from_address(evidence.address))
    factory = PredictionPatternFactory()
    hot = _state(factory, sample_uri, summary="主人回家且室温较高", identity="hot-home")
    comfortable = _state(factory, sample_uri, summary="室温已经舒适", identity="comfortable")
    open_ac = _branch(
        factory,
        sample_uri,
        state=hot,
        behavior="打开空调",
        probability=0.67,
    )
    wash_hands = _branch(
        factory,
        sample_uri,
        state=hot,
        behavior="直接去洗手",
        probability=0.25,
    )
    PredictionPatternGenerationPublisher(tree).publish(
        (hot, comfortable, wash_hands, open_ac),
        created_at=datetime(2026, 8, 9, tzinfo=UTC),
    )

    graph = PredictionPatternGraph(tree)
    matched = graph.search_states(
        semantic_terms=("回家",),
        observed_values={"temperature": 29},
        active_goals=("改善舒适度",),
    )
    expansion = graph.expand(hot.fields["logical_state_key"])

    assert matched[0] == hot
    assert [item.fields["behavior"]["semantics"] for item in expansion.branches] == ["打开空调", "直接去洗手"]
    assert str(PredictionURI.from_pattern_address(hot.address)).startswith("prediction://patterns/states/")
    assert tree.list_pattern_addresses(PredictionPatternKind.BRANCH) == (
        open_ac.address,
        wash_hands.address,
    ) or tree.list_pattern_addresses(PredictionPatternKind.BRANCH) == (
        wash_hands.address,
        open_ac.address,
    )
    first_state_page = tree.list_pattern_addresses(PredictionPatternKind.STATE, limit=1)
    second_state_page = tree.list_pattern_addresses(
        PredictionPatternKind.STATE,
        limit=1,
        after=first_state_page[0],
    )
    assert set((*first_state_page, *second_state_page)) == {hot.address, comfortable.address}
    first_branch_page = tree.list_branch_addresses(hot.address.state_key, limit=1)
    second_branch_page = tree.list_branch_addresses(
        hot.address.state_key,
        limit=1,
        after=first_branch_page[0],
    )
    assert set((*first_branch_page, *second_branch_page)) == {open_ac.address, wash_hands.address}


def test_pattern_schema_rejects_tampered_identity_and_invalid_probability(tmp_path) -> None:
    tree = PredictionTree(tmp_path / "prediction")
    evidence = _sample(tree)
    sample_uri = str(PredictionURI.from_address(evidence.address))
    factory = PredictionPatternFactory()
    state = _state(factory, sample_uri, summary="主人回家", identity="home")

    tampered = dict(state.fields)
    tampered["state_key"] = "f" * 64
    with pytest.raises(PredictionPatternSchemaError, match="identity_material"):
        factory.codec.build(PredictionPatternKind.STATE, tampered)

    with pytest.raises(PredictionPatternSchemaError, match="between zero and one"):
        _branch(
            factory,
            sample_uri,
            state=state,
            behavior="打开空调",
            probability=1.1,
        )


def test_pattern_persistence_requires_existing_same_projection_samples(tmp_path) -> None:
    tree = PredictionTree(tmp_path / "prediction")
    factory = PredictionPatternFactory()
    missing_uri = (
        "prediction://samples/transitions/2026/08/08/ff/"
        f"{'f' * 64}.md"
    )
    state = _state(factory, missing_uri, summary="主人回家", identity="missing-evidence")

    with pytest.raises(PredictionTreeIntegrityError, match="missing PredictionSample"):
        PredictionPatternGenerationPublisher(tree).publish((state,))


def test_branch_requires_its_parent_state_and_cannot_cross_pattern_generations(tmp_path) -> None:
    tree = PredictionTree(tmp_path / "prediction")
    evidence = _sample(tree)
    sample_uri = str(PredictionURI.from_address(evidence.address))
    factory = PredictionPatternFactory()
    first = _state(factory, sample_uri, summary="主人回家", identity="home")
    second = factory.build_state(
        pattern_generation="2" * 64,
        identity={"state": "comfortable"},
        fields={
            key: value
            for key, value in first.fields.items()
            if key
            not in {
                "state_key",
                "logical_state_key",
                "pattern_generation",
                "identity_material",
            }
        },
    )
    branch = _branch(
        factory,
        sample_uri,
        state=first,
        behavior="打开空调",
        probability=0.7,
    )

    with pytest.raises(PredictionTreeIntegrityError, match="parent StatePattern"):
        tree.create_pattern(branch)

    with pytest.raises(ValueError, match="one generation ID"):
        PredictionPatternGenerationPublisher(tree).publish((first, second, branch))


def test_pattern_graph_rejects_outgoing_branch_probability_overflow(tmp_path) -> None:
    tree = PredictionTree(tmp_path / "prediction")
    evidence = _sample(tree)
    sample_uri = str(PredictionURI.from_address(evidence.address))
    factory = PredictionPatternFactory()
    state = _state(factory, sample_uri, summary="主人回家", identity="home")
    branches = tuple(
        _branch(
                factory,
                sample_uri,
                state=state,
                behavior=behavior,
                probability=probability,
            )
        for behavior, probability in (("打开空调", 0.7), ("直接去洗手", 0.6))
    )

    with pytest.raises(ValueError, match="probabilities cannot exceed one"):
        PredictionPatternGenerationPublisher(tree).publish((state, *branches))


def test_pattern_generation_interruption_stays_invisible_and_replay_is_idempotent(
    tmp_path,
    monkeypatch,
) -> None:
    tree = PredictionTree(tmp_path / "prediction")
    evidence = _sample(tree)
    sample_uri = str(PredictionURI.from_address(evidence.address))
    factory = PredictionPatternFactory()
    state = _state(factory, sample_uri, summary="主人回家", identity="home")
    branch = _branch(
        factory,
        sample_uri,
        state=state,
        behavior="打开空调",
        probability=0.7,
    )
    publisher = PredictionPatternGenerationPublisher(tree)
    original_create = tree.create_pattern

    def interrupted_create(document):
        if document.kind is PredictionPatternKind.BRANCH:
            raise RuntimeError("simulated publication interruption")
        return original_create(document)

    monkeypatch.setattr(tree, "create_pattern", interrupted_create)
    first_created_at = datetime(2026, 8, 9, 10, tzinfo=UTC)
    with pytest.raises(RuntimeError, match="interruption"):
        publisher.publish((state, branch), created_at=first_created_at)
    logical_key = state.fields["logical_state_key"]
    prepared = publisher.store.read(logical_key, PATTERN_GENERATION)
    assert prepared.state is PredictionPatternGenerationState.PREPARED
    with pytest.raises(PredictionPatternGenerationStoreError, match="no active generation"):
        publisher.store.read_active_state(logical_key)

    monkeypatch.setattr(tree, "create_pattern", original_create)
    replayed = publisher.publish(
        (state, branch),
        created_at=datetime(2026, 8, 9, 11, tzinfo=UTC),
    )

    assert replayed[0].state is PredictionPatternGenerationState.READY
    assert replayed[0].manifest.created_at == first_created_at
    assert PredictionPatternGraph(tree).expand(logical_key).branches == (branch,)


def test_pattern_graph_ignores_documents_outside_active_manifest(tmp_path) -> None:
    tree = PredictionTree(tmp_path / "prediction")
    evidence = _sample(tree)
    sample_uri = str(PredictionURI.from_address(evidence.address))
    factory = PredictionPatternFactory()
    active = _state(factory, sample_uri, summary="主人回家", identity="active")
    hidden = _state(factory, sample_uri, summary="未发布的临时状态", identity="hidden")
    PredictionPatternGenerationPublisher(tree).publish(
        (active,),
        created_at=datetime(2026, 8, 9, tzinfo=UTC),
    )
    tree.create_pattern(hidden)

    graph = PredictionPatternGraph(tree)

    assert graph.search_states(semantic_terms=("临时",)) == ()
    with pytest.raises(PredictionPatternGenerationStoreError, match="no active generation"):
        graph.expand(hidden.fields["logical_state_key"])


def test_delayed_generation_replay_cannot_roll_back_active_and_history_remains_readable(
    tmp_path,
) -> None:
    tree = PredictionTree(tmp_path / "prediction")
    evidence = _sample(tree)
    sample_uri = str(PredictionURI.from_address(evidence.address))
    factory = PredictionPatternFactory()
    first = _state(factory, sample_uri, summary="第一代状态", identity="same", generation="1" * 64)
    second = _state(factory, sample_uri, summary="第二代状态", identity="same", generation="2" * 64)
    publisher = PredictionPatternGenerationPublisher(tree)
    publisher.publish((first,), created_at=datetime(2026, 8, 9, 10, tzinfo=UTC))
    publisher.publish((second,), created_at=datetime(2026, 8, 9, 11, tzinfo=UTC))

    replay_result = publisher.publish(
        (first,),
        created_at=datetime(2026, 8, 9, 12, tzinfo=UTC),
    )

    logical_key = first.fields["logical_state_key"]
    assert second.fields["logical_state_key"] == logical_key
    assert replay_result[0].manifest.generation_id == "2" * 64
    assert PredictionPatternGraph(tree).expand(logical_key).pattern_generation == "2" * 64
    history = publisher.store.read(logical_key, "1" * 64)
    assert history.manifest.generation_id == "1" * 64


def test_oversize_generation_fails_before_any_disk_mutation_and_can_retry(tmp_path) -> None:
    tree = PredictionTree(tmp_path / "prediction")
    evidence = _sample(tree)
    sample_uri = str(PredictionURI.from_address(evidence.address))
    state = _state(
        PredictionPatternFactory(),
        sample_uri,
        summary="超限清单",
        identity="oversize",
    )
    root = tmp_path / "pattern-generations"
    limited_store = PredictionPatternGenerationStore(root, max_record_bytes=100)

    with pytest.raises(PredictionPatternGenerationStoreError, match="size limit"):
        PredictionPatternGenerationPublisher(tree, store=limited_store).publish((state,))

    assert not any(root.rglob(f"{PATTERN_GENERATION}.json"))
    assert not tree.pattern_exists(state.address)
    recovered = PredictionPatternGenerationPublisher(
        tree,
        store=PredictionPatternGenerationStore(root),
    ).publish((state,), created_at=datetime(2026, 8, 9, tzinfo=UTC))
    assert recovered[0].state is PredictionPatternGenerationState.READY


def test_two_fresh_publishers_can_activate_the_same_generation_concurrently(tmp_path) -> None:
    root = tmp_path / "prediction"
    first_tree = PredictionTree(root)
    evidence = _sample(first_tree)
    sample_uri = str(PredictionURI.from_address(evidence.address))
    state = _state(
        PredictionPatternFactory(),
        sample_uri,
        summary="并发发布状态",
        identity="concurrent",
    )
    stores_root = tmp_path / "pattern-generations"
    publishers = (
        PredictionPatternGenerationPublisher(
            PredictionTree(root),
            store=PredictionPatternGenerationStore(stores_root),
        ),
        PredictionPatternGenerationPublisher(
            PredictionTree(root),
            store=PredictionPatternGenerationStore(stores_root),
        ),
    )
    barrier = Barrier(2)

    def publish(publisher: PredictionPatternGenerationPublisher):
        barrier.wait()
        return publisher.publish(
            (state,),
            created_at=datetime(2026, 8, 9, tzinfo=UTC),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(publish, publishers))

    assert {record.manifest.generation_id for result in results for record in result} == {PATTERN_GENERATION}
    assert PredictionPatternGenerationStore(stores_root).read_active_state(
        state.fields["logical_state_key"]
    ).state is PredictionPatternGenerationState.READY


def test_pattern_generation_preserves_logical_state_but_creates_new_materialization(tmp_path) -> None:
    tree = PredictionTree(tmp_path / "prediction")
    evidence = _sample(tree)
    sample_uri = str(PredictionURI.from_address(evidence.address))
    factory = PredictionPatternFactory()
    first = _state(factory, sample_uri, summary="主人回家", identity="same-state")
    second = factory.build_state(
        pattern_generation="2" * 64,
        identity={"state": "same-state"},
        fields={key: value for key, value in first.fields.items() if key not in {
            "state_key",
            "logical_state_key",
            "pattern_generation",
            "identity_material",
        }},
    )

    assert first.fields["logical_state_key"] == second.fields["logical_state_key"]
    assert first.address.state_key != second.address.state_key


def test_sample_identity_separates_logical_question_from_projection_materialization() -> None:
    material = {"event_uri": "behavior://example", "target": "action:act_0001"}
    first = derive_sample_identity(PredictionKind.TRANSITION, material, "projection-v1", {})
    second = derive_sample_identity(PredictionKind.TRANSITION, material, "projection-v2", {})

    assert first.logical_sample_id == second.logical_sample_id
    assert first.materialization_id != second.materialization_id

    payload = transition_payload()
    payload["materialization_id"] = "f" * 64
    with pytest.raises(PredictionSchemaError, match="materialization ID"):
        PredictionSchemaRegistry.load_default().validate(PredictionKind.TRANSITION, payload)


def test_exact_cutoff_rejects_future_available_fact_and_completed_history() -> None:
    registry = PredictionSchemaRegistry.load_default()
    future = datetime(2026, 8, 8, 10, 31, tzinfo=UTC)
    payload = transition_payload()
    payload["input"]["observation_frame"]["facts"][0]["available_at"] = future
    with pytest.raises(PredictionSchemaError, match="later than the prediction cutoff"):
        registry.validate(PredictionKind.TRANSITION, payload)

    payload = deepcopy(transition_payload())
    step: dict[str, object] = {
        "step_kind": "action",
        "step_ref": "action:act_0000",
        "source_uri": None,
        "local_id": "act_0000",
        "sequence": 1,
        "semantics": "仍在进行的动作",
        "actor": "主人",
        "behavior_type": "daily_activity",
        "target_refs": [],
        "status": "active",
        "started_at": datetime(2026, 8, 8, 10, 29, tzinfo=UTC),
        "ended_at": None,
        "available_at": datetime(2026, 8, 8, 10, 29, tzinfo=UTC),
    }
    payload["input"]["behavior_history"]["completed_actions"] = [step]
    with pytest.raises(PredictionSchemaError, match="without ended_at"):
        registry.validate(PredictionKind.TRANSITION, payload)


def test_updating_one_state_leaves_every_other_state_untouched(tmp_path) -> None:
    """一致性单元是单个 State：重建其中一个不得影响其余 State 的激活版本。"""

    tree = PredictionTree(tmp_path / "prediction")
    evidence = _sample(tree)
    sample_uri = str(PredictionURI.from_address(evidence.address))
    factory = PredictionPatternFactory()
    hot = _state(factory, sample_uri, summary="室温较高", identity="hot")
    quiet = _state(factory, sample_uri, summary="家里安静", identity="quiet")
    publisher = PredictionPatternGenerationPublisher(tree)
    publisher.publish((hot, quiet), created_at=datetime(2026, 8, 9, 10, tzinfo=UTC))

    hot_key = hot.fields["logical_state_key"]
    quiet_key = quiet.fields["logical_state_key"]
    graph = PredictionPatternGraph(tree)
    assert graph.expand(hot_key).pattern_generation == PATTERN_GENERATION
    assert graph.expand(quiet_key).pattern_generation == PATTERN_GENERATION

    refreshed_hot = _state(
        factory,
        sample_uri,
        summary="室温较高",
        identity="hot",
        generation="2" * 64,
    )
    publisher.publish((refreshed_hot,), created_at=datetime(2026, 8, 9, 11, tzinfo=UTC))

    assert refreshed_hot.fields["logical_state_key"] == hot_key
    assert refreshed_hot.address.state_key != hot.address.state_key
    assert graph.expand(hot_key).pattern_generation == "2" * 64
    assert graph.expand(quiet_key).pattern_generation == PATTERN_GENERATION
    assert graph.expand(quiet_key).state == quiet
    assert set(publisher.store.active_logical_state_keys()) == {hot_key, quiet_key}
