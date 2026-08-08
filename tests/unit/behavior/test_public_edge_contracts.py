"""Behavior 公开 URI、关系、派生层和快照边界的正反向合同测试。"""

from dataclasses import replace
from datetime import date, datetime, timezone

import pytest
from behavior_test_payloads import event_payload

from behavior import (
    BehaviorAddress,
    BehaviorDirectory,
    BehaviorDocumentCodec,
    BehaviorDocumentConfig,
    BehaviorDocumentLimitError,
    BehaviorDocumentMetadata,
    BehaviorKind,
    BehaviorLevel,
    BehaviorLinkType,
    BehaviorSchemaRegistry,
    BehaviorSnapshotReader,
    BehaviorStoredLink,
    BehaviorTree,
    BehaviorURI,
    BehaviorURIError,
)
from behavior.document.link import normalize_stored_links, parse_stored_links


def _event_uri(name: str) -> BehaviorURI:
    started_at = datetime(2026, 8, 8, 10, 30, tzinfo=timezone.utc)
    return BehaviorURI.from_address(BehaviorAddress.event(started_at.date(), name, started_at))


def test_stored_links_normalize_direction_round_trip_and_reject_invalid_edges() -> None:
    alpha = _event_uri("alpha")
    zulu = _event_uri("zulu")
    related = BehaviorStoredLink(zulu, alpha, BehaviorLinkType.RELATED_TO)
    directed = BehaviorStoredLink(zulu, alpha, BehaviorLinkType.CAUSED_BY)

    assert related.from_uri == alpha
    assert related.to_uri == zulu
    assert related.identity == (str(alpha), str(zulu), "related_to")
    assert BehaviorStoredLink.from_dict(related.to_dict()) == related
    assert directed.from_uri == zulu
    assert normalize_stored_links((directed, related), label="links") == (related, directed)
    assert parse_stored_links([related.to_dict()], label="links") == (related,)

    with pytest.raises(ValueError, match="same URI"):
        BehaviorStoredLink(alpha, alpha, BehaviorLinkType.RELATED_TO)
    with pytest.raises(ValueError, match="L2 behavior document"):
        BehaviorStoredLink(BehaviorURI.root(), alpha, BehaviorLinkType.RELATED_TO)
    with pytest.raises(ValueError, match="invalid shape"):
        BehaviorStoredLink.from_dict({"from_uri": str(alpha)})
    with pytest.raises(ValueError, match="duplicate"):
        normalize_stored_links((related, related), label="links")
    with pytest.raises(TypeError, match="tuple"):
        normalize_stored_links([related], label="links")
    with pytest.raises(ValueError, match="array"):
        parse_stored_links({}, label="links")


def test_document_rejects_link_orientation_that_does_not_belong_to_its_address() -> None:
    codec = BehaviorDocumentCodec(BehaviorSchemaRegistry.load_default())
    document = codec.build(
        BehaviorKind.EVENT,
        event_payload("alpha"),
        metadata=BehaviorDocumentMetadata.initial(datetime(2026, 8, 8, 12, tzinfo=timezone.utc)),
    )
    foreign_link = BehaviorStoredLink(_event_uri("zulu"), _event_uri("beta"), BehaviorLinkType.CAUSED_BY)

    with pytest.raises(ValueError, match="wrong source"):
        replace(document, links=(foreign_link,))
    with pytest.raises(ValueError, match="wrong target"):
        replace(document, backlinks=(foreign_link,))


def test_uri_public_helpers_preserve_node_types_and_immutability() -> None:
    root = BehaviorURI.root()
    events = BehaviorURI.build("behaviors", "events")
    day = BehaviorURI.from_directory(BehaviorDirectory.events(2026, 8, 8))
    document = day.join("event--20260808T103000000000+0000.md")

    assert root.is_root
    assert root.parent is None
    assert events.to_directory() == BehaviorDirectory.events()
    assert document.to_address() == BehaviorAddress.event(
        date(2026, 8, 8),
        "event",
        datetime(2026, 8, 8, 10, 30, tzinfo=timezone.utc),
    )
    assert document.containing_directory == BehaviorDirectory.events(2026, 8, 8)
    assert document.name == "event--20260808T103000000000+0000.md"
    assert document.full_path.endswith("/event--20260808T103000000000%2B0000.md")
    assert BehaviorURI.normalize(str(document)) == str(document)
    assert BehaviorURI.parse(document) is document
    assert hash(document) == hash(str(document))
    assert repr(document).startswith("BehaviorURI(")

    with pytest.raises(BehaviorURIError, match="directory"):
        document.join("child")
    with pytest.raises(BehaviorURIError, match="directory"):
        document.to_directory()
    with pytest.raises(BehaviorURIError, match="semantic layer"):
        document.to_layer()
    with pytest.raises(AttributeError, match="immutable"):
        document._uri = "behavior://"  # type: ignore[misc]


def test_tree_layer_paths_and_snapshot_batch_use_one_canonical_codec(tmp_path) -> None:
    tree = BehaviorTree(tmp_path / "behavior-tree")
    tree.initialize()
    directory = BehaviorDirectory.behaviors()
    abstract_path, overview_path = tree.write_layers(
        directory,
        abstract="行为摘要",
        overview="行为概览",
    )

    assert tree.path_for_uri(BehaviorURI.from_directory(directory)) == tree.directory_path(directory)
    assert tree.path_for_uri(BehaviorURI.from_layer(directory, BehaviorLevel.ABSTRACT)) == abstract_path
    assert tree.layer_path(directory, BehaviorLevel.OVERVIEW) == overview_path
    assert tree.read_layer(directory, BehaviorLevel.ABSTRACT) == "行为摘要"
    assert tree.read_layer(directory, BehaviorLevel.OVERVIEW) == "行为概览"
    assert tree.layer_exists(directory, BehaviorLevel.ABSTRACT)

    missing = _event_uri("missing")
    snapshots = BehaviorSnapshotReader(tree).read_many((missing, missing))
    assert len(snapshots.snapshots) == 1
    assert not snapshots.snapshots[0].exists
    assert snapshots.total_bytes == 0


def test_document_limits_reject_invalid_configuration_and_each_resource_boundary() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        BehaviorDocumentConfig(max_encoded_bytes=0)
    config = BehaviorDocumentConfig(
        max_markdown_body_chars=4,
        max_encoded_bytes=4,
        max_relations_per_document=1,
    )

    with pytest.raises(BehaviorDocumentLimitError, match="body"):
        config.validate_body("12345")
    with pytest.raises(BehaviorDocumentLimitError, match="document"):
        config.validate_encoded(b"12345")
    with pytest.raises(BehaviorDocumentLimitError, match="semantic layer"):
        config.validate_semantic_layer(b"12345")
    with pytest.raises(BehaviorDocumentLimitError, match="relation"):
        config.validate_relations(links=2, backlinks=0)
    with pytest.raises(ValueError, match="non-negative"):
        config.validate_relations(links=-1, backlinks=0)
