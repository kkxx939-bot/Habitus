"""Prediction Tree 不可变持久化、容量和发布测试。"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
from prediction_test_payloads import transition_payload

from prediction import (
    PredictionDirectory,
    PredictionDocument,
    PredictionDocumentCodec,
    PredictionKind,
    PredictionLevel,
    PredictionPublicationState,
    PredictionPublisher,
    PredictionSchemaRegistry,
    PredictionTree,
    PredictionTreeConfig,
    PredictionTreeConflictError,
    PredictionTreeIntegrityError,
)


def _document(identity_token: str):
    return PredictionDocumentCodec(PredictionSchemaRegistry.load_default()).build(
        PredictionKind.TRANSITION,
        transition_payload(identity_token),
    )


def test_publisher_is_idempotent_and_reads_back_canonical_samples(tmp_path) -> None:
    tree = PredictionTree(tmp_path / "prediction")
    publisher = PredictionPublisher(tree)
    first = _document("a" * 64)
    second = _document("b" * 64)

    ordered = tuple(sorted((first, second), key=lambda item: str(item.address.materialization_id)))
    assert publisher.publish((second, first, first)) == ordered
    assert publisher.publish((first, second)) == ordered
    assert tree.list_addresses(limit=1) == (ordered[0].address,)
    assert tree.list_addresses(after=ordered[0].address) == (ordered[1].address,)
    assert tree.read(first.address) == first


def test_tree_rejects_same_sample_identity_with_different_content(tmp_path) -> None:
    tree = PredictionTree(tmp_path / "prediction")
    original = _document("a" * 64)
    changed_payload = deepcopy(transition_payload("a" * 64))
    changed_payload["label"]["semantics"] = "关闭空调"
    changed = PredictionDocumentCodec(PredictionSchemaRegistry.load_default()).build(
        PredictionKind.TRANSITION,
        changed_payload,
    )

    tree.create(original)
    with pytest.raises(PredictionTreeConflictError):
        tree.create(changed)
    assert tree.read(original.address) == original


def test_semantic_layers_are_bounded_rebuildable_sidecars(tmp_path) -> None:
    tree = PredictionTree(tmp_path / "prediction")
    tree.initialize()
    directory = PredictionDirectory.branch(PredictionKind.TRANSITION)

    abstract_path, overview_path = tree.write_layers(
        directory,
        abstract="动作转换样本",
        overview="按日期保存动作转换监督样本。",
    )

    assert abstract_path.name == ".abstract.md"
    assert overview_path.name == ".overview.md"
    assert tree.read_layer(directory, PredictionLevel.ABSTRACT) == "动作转换样本"
    assert tree.read_layer(directory, PredictionLevel.OVERVIEW) == "按日期保存动作转换监督样本。"


def test_tree_checks_capacity_before_creating_an_extra_leaf(tmp_path) -> None:
    tree = PredictionTree(
        tmp_path / "prediction",
        tree_config=PredictionTreeConfig(max_children_per_directory=3),
    )
    for character in "abc":
        tree.create(_document(character * 64))

    with pytest.raises(PredictionTreeIntegrityError, match="no remaining child capacity"):
        tree.create(_document("d" * 64))
    assert len(tree.list_addresses()) == 3


def test_batch_capacity_failure_does_not_publish_a_partial_sample_set(tmp_path) -> None:
    tree = PredictionTree(
        tmp_path / "prediction",
        tree_config=PredictionTreeConfig(max_children_per_directory=3),
    )
    existing = _document("batch-existing")
    tree.create(existing)

    with pytest.raises(PredictionTreeIntegrityError, match="no remaining child capacity"):
        PredictionPublisher(tree).publish(
            (_document("batch-a"), _document("batch-b"), _document("batch-c"))
        )

    assert tree.list_addresses() == (existing.address,)


def test_batch_preflights_existing_conflicts_before_creating_any_new_sample(tmp_path) -> None:
    tree = PredictionTree(tmp_path / "prediction")
    original = _document("batch-conflict")
    tree.create(original)
    changed_payload = deepcopy(transition_payload("batch-conflict"))
    changed_payload["label"]["semantics"] = "关闭空调"
    changed = PredictionDocumentCodec(PredictionSchemaRegistry.load_default()).build(
        PredictionKind.TRANSITION,
        changed_payload,
    )
    candidate = _document("must-not-be-partial")

    with pytest.raises(PredictionTreeConflictError):
        tree.create_many((candidate, changed))

    assert tree.list_addresses() == (original.address,)


def test_pattern_listing_does_not_create_a_missing_tree(tmp_path) -> None:
    root = tmp_path / "prediction"
    tree = PredictionTree(root)

    assert tree.list_pattern_addresses() == ()
    assert not root.exists()


def test_tree_rejects_symbolic_link_root(tmp_path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "prediction-link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(PredictionTreeIntegrityError, match="symbolic link"):
        PredictionTree(link)


def test_two_tree_instances_cannot_overfill_one_directory_concurrently(tmp_path) -> None:
    root = tmp_path / "prediction"
    config = PredictionTreeConfig(max_children_per_directory=3)
    first_tree = PredictionTree(root, tree_config=config)
    second_tree = PredictionTree(root, tree_config=config)
    first_tree.create(_document("prefill-a"))
    first_tree.create(_document("prefill-b"))

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(first_tree.create, _document("concurrent-c")),
            executor.submit(second_tree.create, _document("concurrent-d")),
        )
        results: list[PredictionDocument | None] = []
        for future in futures:
            try:
                results.append(future.result())
            except PredictionTreeIntegrityError:
                results.append(None)

    assert sum(result is not None for result in results) == 1
    assert len(first_tree.list_addresses()) == 3


def test_prepared_publication_replays_after_partial_failure(tmp_path) -> None:
    class FailSecondCreateTree(PredictionTree):
        def create_many(self, documents):
            super().create_many((documents[0],))
            raise RuntimeError("injected publication failure")

    root = tmp_path / "prediction"
    failing = FailSecondCreateTree(root)
    documents = (_document("resume-a"), _document("resume-b"))
    publisher = PredictionPublisher(failing)
    with pytest.raises(RuntimeError, match="injected"):
        publisher.publish(documents)

    job_files = tuple(publisher.journal.root.glob("*.json"))
    assert len(job_files) == 1
    assert publisher.journal.read(job_files[0].stem).state is PredictionPublicationState.PREPARED

    replay = PredictionPublisher(PredictionTree(root), journal=publisher.journal)
    assert {item.address for item in replay.publish(documents)} == {item.address for item in documents}
    assert publisher.journal.read(job_files[0].stem).state is PredictionPublicationState.COMMITTED
