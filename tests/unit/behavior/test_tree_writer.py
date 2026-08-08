"""Behavior Tree 安全持久化、add-only 和 Outcome CAS 追加测试。"""

from copy import deepcopy
from datetime import datetime, timezone

import pytest
from behavior_test_payloads import episode_payload, event_payload, event_uri, outcome_payload, outcome_uri

from behavior import (
    BehaviorCASConflictError,
    BehaviorDirectory,
    BehaviorDocumentConfig,
    BehaviorDocumentLimitError,
    BehaviorDocumentWriter,
    BehaviorKind,
    BehaviorLevel,
    BehaviorPublishConflictError,
    BehaviorSnapshotReader,
    BehaviorTree,
    BehaviorTreeConfig,
    BehaviorTreeIntegrityError,
    BehaviorURI,
)
from infrastructure.store.locks.process_local import ProcessLocalLockStore


def _writer(tmp_path):
    tree = BehaviorTree(tmp_path / "behavior-tree")
    writer = BehaviorDocumentWriter(
        tree,
        ProcessLocalLockStore(),
        clock=lambda: datetime(2026, 8, 8, 12, tzinfo=timezone.utc),
    )
    return tree, writer


def test_publish_read_list_and_snapshot_use_atomic_event_documents(tmp_path) -> None:
    tree, writer = _writer(tmp_path)
    document = writer.publish(BehaviorKind.EVENT, event_payload())
    uri = BehaviorURI.from_address(document.address)

    assert tree.read(document.address) == document
    assert tree.path_for_uri(uri).name == "主人回家后查看并打开空调--20260808T103000000000+0000.md"
    assert tree.list_addresses() == (document.address,)

    snapshot = BehaviorSnapshotReader(tree).read(uri)
    assert snapshot.exists
    assert snapshot.revision == 1
    assert snapshot.value == document

    with pytest.raises(BehaviorPublishConflictError):
        writer.publish(BehaviorKind.EVENT, event_payload())


def test_same_semantic_event_name_uses_started_at_for_distinct_identities(tmp_path) -> None:
    tree, writer = _writer(tmp_path)
    first = writer.publish(BehaviorKind.EVENT, event_payload("主人回家", minute=30))
    second = writer.publish(BehaviorKind.EVENT, event_payload("主人回家", minute=40))

    assert first.address != second.address
    assert first.address.name == second.address.name == "主人回家"
    assert first.address.identity_name.endswith("--20260808T103000000000+0000")
    assert second.address.identity_name.endswith("--20260808T104000000000+0000")
    assert tree.list_addresses(kind=BehaviorKind.EVENT) == (first.address, second.address)


def test_outcome_publish_validates_action_and_append_uses_revision_cas(tmp_path) -> None:
    tree, writer = _writer(tmp_path)
    event = writer.publish(BehaviorKind.EVENT, event_payload())
    event_uri = str(BehaviorURI.from_address(event.address))
    outcome = writer.publish(BehaviorKind.OUTCOME, outcome_payload(event_uri))
    outcome_uri = BehaviorURI.from_address(outcome.address)

    appended = writer.append_outcomes(
        outcome_uri,
        [
            {
                "outcome_id": "out_0002",
                "occurred_at": datetime(2026, 8, 8, 10, 40, tzinfo=timezone.utc),
                "outcome_type": "delayed_effect",
                "target_type": "event",
                "target_action_id": None,
                "semantics": "室温下降到舒适范围",
                "valence": "positive",
                "knowledge_state": "observed",
                "confidence": 0.9,
                "evidence_refs": ["sensor:temperature"],
            }
        ],
        expected_revision=1,
    )
    assert appended.metadata.revision == 2
    assert len(appended.fields["outcomes"]) == 2
    assert tree.read(outcome.address) == appended

    with pytest.raises(BehaviorCASConflictError):
        writer.append_outcomes(
            outcome_uri,
            [
                {
                    "outcome_id": "out_0003",
                    "occurred_at": datetime(2026, 8, 8, 10, 45, tzinfo=timezone.utc),
                    "outcome_type": "human_feedback",
                    "target_type": "event",
                    "target_action_id": None,
                    "semantics": "过期写入不应生效",
                    "valence": "negative",
                    "knowledge_state": "reported",
                    "confidence": 1.0,
                    "evidence_refs": ["utterance:stale"],
                }
            ],
            expected_revision=1,
        )


def test_outcome_rejects_action_absent_from_target_event(tmp_path) -> None:
    _tree, writer = _writer(tmp_path)
    event = writer.publish(BehaviorKind.EVENT, event_payload())
    payload = outcome_payload(str(BehaviorURI.from_address(event.address)))
    payload["outcomes"][0]["target_action_id"] = "act_missing"
    with pytest.raises(ValueError, match="absent"):
        writer.publish(BehaviorKind.OUTCOME, payload)


def test_episode_publishes_only_after_all_referenced_l2_documents_exist(tmp_path) -> None:
    tree, writer = _writer(tmp_path)
    first = writer.publish(BehaviorKind.EVENT, event_payload())
    second = writer.publish(
        BehaviorKind.EVENT,
        event_payload("主人洗手", minute=40),
    )
    first_uri = str(BehaviorURI.from_address(first.address))
    second_uri = str(BehaviorURI.from_address(second.address))
    outcome = writer.publish(BehaviorKind.OUTCOME, outcome_payload(first_uri))
    outcome_uri = str(BehaviorURI.from_address(outcome.address))

    episode = writer.publish(
        BehaviorKind.EPISODE,
        episode_payload(first_uri, second_uri, outcome_uri),
    )
    assert episode.kind is BehaviorKind.EPISODE
    assert tuple(episode.fields["ordered_event_uris"]) == (first_uri, second_uri)
    assert episode.fields["outcome_snapshots"][0]["uri"] == outcome_uri
    assert episode.fields["outcome_snapshots"][0]["revision"] == 1
    assert len(episode.fields["outcome_snapshots"][0]["digest"]) == 64
    assert tree.read(episode.address) == episode


def test_episode_rejects_missing_event_reference(tmp_path) -> None:
    _tree, writer = _writer(tmp_path)
    first = event_uri("不存在一", minute=30)
    second = event_uri("不存在二", minute=40)
    payload = episode_payload(first, second, outcome_uri("不存在一", minute=30))
    with pytest.raises(FileNotFoundError):
        writer.publish(BehaviorKind.EPISODE, payload)


def test_delayed_outcome_is_merged_by_actual_occurrence_time(tmp_path) -> None:
    tree, writer = _writer(tmp_path)
    event = writer.publish(BehaviorKind.EVENT, event_payload())
    event_uri = str(BehaviorURI.from_address(event.address))
    outcome = writer.publish(BehaviorKind.OUTCOME, outcome_payload(event_uri))
    delayed = deepcopy(outcome_payload(event_uri)["outcomes"][0])
    delayed.update(
        {
            "outcome_id": "out_delayed",
            "occurred_at": datetime(2026, 8, 8, 10, 34, tzinfo=timezone.utc),
        }
    )

    updated = writer.append_outcomes(
        BehaviorURI.from_address(outcome.address),
        [delayed],
        expected_revision=1,
    )

    assert [item["outcome_id"] for item in updated.fields["outcomes"]] == ["out_delayed", "out_0001"]
    assert tree.read(outcome.address) == updated


def test_outcome_rejects_result_before_target_event_or_action(tmp_path) -> None:
    _tree, writer = _writer(tmp_path)
    event = writer.publish(BehaviorKind.EVENT, event_payload())
    payload = outcome_payload(str(BehaviorURI.from_address(event.address)))
    payload["outcomes"][0]["occurred_at"] = datetime(2026, 8, 8, 10, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="before its target"):
        writer.publish(BehaviorKind.OUTCOME, payload)

    action_timed_payload = event_payload("action-timed", minute=40)
    action_timed_payload["actions"][0]["started_at"] = datetime(2026, 8, 8, 10, 41, tzinfo=timezone.utc)
    action_timed_payload["actions"][0]["ended_at"] = datetime(2026, 8, 8, 10, 42, tzinfo=timezone.utc)
    action_timed = writer.publish(BehaviorKind.EVENT, action_timed_payload)
    action_outcome = outcome_payload(str(BehaviorURI.from_address(action_timed.address)), "action-timed")
    action_outcome["outcomes"][0]["occurred_at"] = datetime(2026, 8, 8, 10, 40, 30, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="before its target"):
        writer.publish(BehaviorKind.OUTCOME, action_outcome)


def test_episode_rejects_unrelated_outcome_reversed_events_and_out_of_range_event(tmp_path) -> None:
    _tree, writer = _writer(tmp_path)
    first = writer.publish(BehaviorKind.EVENT, event_payload("first", minute=30))
    second = writer.publish(BehaviorKind.EVENT, event_payload("second", minute=40))
    unrelated = writer.publish(BehaviorKind.EVENT, event_payload("unrelated", minute=50))
    first_uri = str(BehaviorURI.from_address(first.address))
    second_uri = str(BehaviorURI.from_address(second.address))
    unrelated_uri = str(BehaviorURI.from_address(unrelated.address))

    unrelated_outcome_payload = outcome_payload(unrelated_uri, "unrelated")
    unrelated_outcome_payload["outcomes"][0]["occurred_at"] = datetime(2026, 8, 8, 10, 55, tzinfo=timezone.utc)
    unrelated_outcome = writer.publish(BehaviorKind.OUTCOME, unrelated_outcome_payload)
    unrelated_outcome_uri = str(BehaviorURI.from_address(unrelated_outcome.address))
    with pytest.raises(ValueError, match="outside the Episode"):
        writer.publish(
            BehaviorKind.EPISODE,
            episode_payload(first_uri, second_uri, unrelated_outcome_uri),
        )

    first_outcome = writer.publish(BehaviorKind.OUTCOME, outcome_payload(first_uri, "first"))
    first_outcome_uri = str(BehaviorURI.from_address(first_outcome.address))
    with pytest.raises(ValueError, match="real Event order"):
        writer.publish(
            BehaviorKind.EPISODE,
            episode_payload(second_uri, first_uri, first_outcome_uri),
        )

    out_of_range = episode_payload(first_uri, second_uri, first_outcome_uri)
    out_of_range["ended_at"] = datetime(2026, 8, 8, 10, 35, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="time window"):
        writer.publish(BehaviorKind.EPISODE, out_of_range)


def test_episode_preserves_semantic_order_for_overlapping_events(tmp_path) -> None:
    _tree, writer = _writer(tmp_path)
    semantic_first_payload = event_payload("semantic-first", minute=31)
    semantic_first_payload["ended_at"] = datetime(2026, 8, 8, 10, 33, tzinfo=timezone.utc)
    overlapping_second_payload = event_payload("overlapping-second", minute=30)
    overlapping_second_payload["ended_at"] = datetime(2026, 8, 8, 10, 32, tzinfo=timezone.utc)
    semantic_first = writer.publish(BehaviorKind.EVENT, semantic_first_payload)
    overlapping_second = writer.publish(BehaviorKind.EVENT, overlapping_second_payload)
    first_uri = str(BehaviorURI.from_address(semantic_first.address))
    second_uri = str(BehaviorURI.from_address(overlapping_second.address))
    outcome_data = outcome_payload(first_uri, "semantic-first")
    outcome_data["outcomes"][0]["occurred_at"] = datetime(2026, 8, 8, 10, 34, tzinfo=timezone.utc)
    outcome = writer.publish(BehaviorKind.OUTCOME, outcome_data)

    episode = writer.publish(
        BehaviorKind.EPISODE,
        episode_payload(first_uri, second_uri, str(BehaviorURI.from_address(outcome.address))),
    )

    assert episode.fields["ordered_event_uris"] == (first_uri, second_uri)


def test_semantic_layers_reject_oversize_before_writing_either_file(tmp_path) -> None:
    tree = BehaviorTree(
        tmp_path / "behavior-tree",
        document_config=BehaviorDocumentConfig(max_encoded_bytes=32),
    )
    tree.initialize()
    directory = BehaviorDirectory.behaviors()

    with pytest.raises(BehaviorDocumentLimitError, match="semantic layer"):
        tree.write_layers(directory, abstract="x" * 33, overview="overview")

    assert not tree.layer_exists(directory, BehaviorLevel.ABSTRACT)
    assert not tree.layer_exists(directory, BehaviorLevel.OVERVIEW)


def test_default_snapshot_accepts_every_valid_default_size_l2(tmp_path) -> None:
    tree, writer = _writer(tmp_path)
    payload = event_payload("large-event")
    payload["actions"][0]["parameters"] = {"blob": "x" * 280_000}
    document = writer.publish(BehaviorKind.EVENT, payload)

    snapshot = BehaviorSnapshotReader(tree).read(BehaviorURI.from_address(document.address))

    assert snapshot.exists
    assert snapshot.value == document
    assert snapshot.size_bytes <= tree.document_config.max_encoded_bytes


def test_list_cursor_uses_canonical_identity_order_not_physical_case(tmp_path) -> None:
    tree, writer = _writer(tmp_path)
    zulu = writer.publish(BehaviorKind.EVENT, event_payload("z", minute=30))
    alpha = writer.publish(BehaviorKind.EVENT, event_payload("a", minute=40))
    zulu_path = tree.path_for(zulu.address)
    intermediate = zulu_path.with_name("rename-in-progress.md")
    zulu_path.rename(intermediate)
    intermediate.rename(zulu_path.with_name(zulu_path.name.replace("z--", "Z--", 1)))

    first_page = tree.list_addresses(limit=1)
    second_page = tree.list_addresses(limit=1, after=first_page[0])

    assert first_page == (alpha.address,)
    assert second_page == (zulu.address,)


def test_directory_capacity_rejects_before_persisting_an_unreadable_child(tmp_path) -> None:
    tree = BehaviorTree(
        tmp_path / "behavior-tree",
        tree_config=BehaviorTreeConfig(max_children_per_directory=2),
    )
    writer = BehaviorDocumentWriter(
        tree,
        ProcessLocalLockStore(),
        clock=lambda: datetime(2026, 8, 8, 12, tzinfo=timezone.utc),
    )
    first = writer.publish(BehaviorKind.EVENT, event_payload("one", minute=10))
    second = writer.publish(BehaviorKind.EVENT, event_payload("two", minute=20))

    with pytest.raises(BehaviorTreeIntegrityError, match="capacity"):
        writer.publish(BehaviorKind.EVENT, event_payload("three", minute=30))

    assert tree.list_addresses() == (first.address, second.address)
    day = tree.root / "behaviors" / "events" / "2026" / "08" / "08"
    assert sorted(path.name for path in day.iterdir()) == sorted(
        [
            tree.path_for(first.address).name,
            tree.path_for(second.address).name,
        ]
    )


def test_episode_rejects_caller_supplied_outcome_snapshot(tmp_path) -> None:
    _tree, writer = _writer(tmp_path)
    first = writer.publish(BehaviorKind.EVENT, event_payload("first", minute=30))
    second = writer.publish(BehaviorKind.EVENT, event_payload("second", minute=40))
    first_uri = str(BehaviorURI.from_address(first.address))
    second_uri = str(BehaviorURI.from_address(second.address))
    outcome = writer.publish(BehaviorKind.OUTCOME, outcome_payload(first_uri, "first"))
    payload = episode_payload(first_uri, second_uri, str(BehaviorURI.from_address(outcome.address)))
    payload["outcome_snapshots"] = []

    with pytest.raises(ValueError, match="system-owned"):
        writer.publish(BehaviorKind.EPISODE, payload)


def test_episode_rejects_outcome_revision_change_between_snapshot_and_commit(tmp_path, monkeypatch) -> None:
    tree = BehaviorTree(tmp_path / "behavior-tree")
    lock_store = ProcessLocalLockStore()
    writer = BehaviorDocumentWriter(
        tree,
        lock_store,
        clock=lambda: datetime(2026, 8, 8, 12, tzinfo=timezone.utc),
    )
    concurrent = BehaviorDocumentWriter(
        tree,
        lock_store,
        clock=lambda: datetime(2026, 8, 8, 12, 1, tzinfo=timezone.utc),
    )
    first = writer.publish(BehaviorKind.EVENT, event_payload("first", minute=30))
    second = writer.publish(BehaviorKind.EVENT, event_payload("second", minute=40))
    first_uri = str(BehaviorURI.from_address(first.address))
    second_uri = str(BehaviorURI.from_address(second.address))
    outcome = writer.publish(BehaviorKind.OUTCOME, outcome_payload(first_uri, "first"))
    target_outcome_uri = BehaviorURI.from_address(outcome.address)
    payload = episode_payload(first_uri, second_uri, str(target_outcome_uri))
    original = writer._with_system_fields

    def mutate_after_snapshot(kind, source):
        enriched = original(kind, source)
        if kind is BehaviorKind.EPISODE:
            concurrent.append_outcomes(
                target_outcome_uri,
                [
                    {
                        "outcome_id": "out_racing",
                        "occurred_at": datetime(2026, 8, 8, 10, 50, tzinfo=timezone.utc),
                        "outcome_type": "delayed_effect",
                        "target_type": "event",
                        "target_action_id": None,
                        "semantics": "快照形成后出现了新的结果",
                        "valence": "mixed",
                        "knowledge_state": "observed",
                        "confidence": 0.8,
                        "evidence_refs": ["sensor:racing"],
                    }
                ],
                expected_revision=1,
            )
        return enriched

    monkeypatch.setattr(writer, "_with_system_fields", mutate_after_snapshot)

    with pytest.raises(BehaviorCASConflictError, match="changed before"):
        writer.publish(BehaviorKind.EPISODE, payload)
    assert tree.list_addresses(kind=BehaviorKind.EPISODE) == ()


def test_episode_tracks_latest_outcome_and_preserves_creation_snapshot(tmp_path) -> None:
    tree, writer = _writer(tmp_path)
    first = writer.publish(BehaviorKind.EVENT, event_payload("first", minute=30))
    second = writer.publish(BehaviorKind.EVENT, event_payload("second", minute=40))
    first_uri = str(BehaviorURI.from_address(first.address))
    second_uri = str(BehaviorURI.from_address(second.address))
    outcome = writer.publish(BehaviorKind.OUTCOME, outcome_payload(first_uri, "first"))
    target_outcome_uri = BehaviorURI.from_address(outcome.address)
    revision_two = writer.append_outcomes(
        target_outcome_uri,
        [
            {
                "outcome_id": "out_0002",
                "occurred_at": datetime(2026, 8, 8, 10, 45, tzinfo=timezone.utc),
                "outcome_type": "delayed_effect",
                "target_type": "event",
                "target_action_id": None,
                "semantics": "稍后出现新的结果",
                "valence": "mixed",
                "knowledge_state": "observed",
                "confidence": 0.9,
                "evidence_refs": ["sensor:later"],
            }
        ],
        expected_revision=1,
    )
    episode = writer.publish(
        BehaviorKind.EPISODE,
        episode_payload(first_uri, second_uri, str(target_outcome_uri)),
    )
    frozen = episode.fields["outcome_snapshots"][0]
    revision_two_snapshot = BehaviorSnapshotReader(tree).read(target_outcome_uri)

    revision_three = writer.append_outcomes(
        target_outcome_uri,
        [
            {
                "outcome_id": "out_0003",
                "occurred_at": datetime(2026, 8, 8, 10, 50, tzinfo=timezone.utc),
                "outcome_type": "human_feedback",
                "target_type": "event",
                "target_action_id": None,
                "semantics": "主人后来补充了反馈",
                "valence": "negative",
                "knowledge_state": "reported",
                "confidence": 1.0,
                "evidence_refs": ["utterance:later"],
            }
        ],
        expected_revision=2,
    )

    assert revision_two.metadata.revision == frozen["revision"] == 2
    assert frozen["digest"] == revision_two_snapshot.source_digest
    assert tree.read(episode.address).fields["outcome_snapshots"][0] == frozen
    assert revision_three.metadata.revision == tree.read(outcome.address).metadata.revision == 3
    assert BehaviorSnapshotReader(tree).read(target_outcome_uri).source_digest != frozen["digest"]
