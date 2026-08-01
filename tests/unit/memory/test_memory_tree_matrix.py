"""物理记忆树的路径、原子读写、枚举、容量和损坏防御矩阵。"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

import infrastructure.store.filesystem.durable_io.atomic_file as atomic_file_module
from memory.document import MemoryDocumentConfig, MemoryDocumentMetadata
from memory.editor import MemoryDocumentLockKeyspace
from memory.model import MemoryAddress, MemoryDirectory, MemoryKind, MemoryLevel
from memory.tree import MemoryTree, MemoryTreeConfig, MemoryTreeIntegrityError
from memory.uri import MemoryURI
from tests.helpers import BASE_TIME, codec, document, memory_fields

ADDRESSES = (
    MemoryAddress.profile(),
    MemoryAddress.preference("回答风格"),
    MemoryAddress.entity("项目", "m2bOS"),
    MemoryAddress.tool("workspace.inspect"),
    MemoryAddress.event(date(2026, 7, 28), "确认方案"),
    MemoryAddress.intention("完成重构"),
)
DIRECTORIES = (
    MemoryDirectory.root(),
    MemoryDirectory.preferences(),
    MemoryDirectory.entities(),
    MemoryDirectory.entities("项目"),
    MemoryDirectory.tools(),
    MemoryDirectory.events(),
    MemoryDirectory.events(2026),
    MemoryDirectory.events(2026, 7),
    MemoryDirectory.events(2026, 7, 28),
    MemoryDirectory.intentions(),
)


@pytest.mark.parametrize("limit", [1, 2, 10_000, 1_000_000])
def test_tree_config_accepts_documented_capacity_bounds(limit: int) -> None:
    assert MemoryTreeConfig(limit).max_children_per_directory == limit


@pytest.mark.parametrize("limit", [0, -1, 1_000_001, True, False, 1.5, "100", None])
def test_tree_config_rejects_invalid_capacity(limit: object) -> None:
    with pytest.raises(ValueError):
        MemoryTreeConfig(limit)  # type: ignore[arg-type]


@pytest.mark.parametrize("argument", ["document_codec", "document_config", "tree_config"])
@pytest.mark.parametrize("value", [object(), "config", 1, True, []])
def test_tree_constructor_rejects_wrong_collaborator_types(tmp_path: Path, argument: str, value: object) -> None:
    kwargs = {argument: value}
    with pytest.raises(TypeError):
        MemoryTree(tmp_path / "memory", **kwargs)  # type: ignore[arg-type]


def test_tree_constructor_rejects_symbolic_link_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "memory"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(MemoryTreeIntegrityError, match="root"):
        MemoryTree(link)


@pytest.mark.parametrize(
    ("address", "relative"),
    [
        (MemoryAddress.profile(), "profile.md"),
        (MemoryAddress.preference("回答风格"), "preferences/回答风格.md"),
        (MemoryAddress.entity("项目", "m2bOS"), "entities/项目/m2bos.md"),
        (MemoryAddress.tool("workspace.inspect"), "tools/workspace.inspect.md"),
        (MemoryAddress.event(date(1, 1, 1), "开始"), "events/0001/01/01/开始.md"),
        (MemoryAddress.event(date(9999, 12, 31), "结束"), "events/9999/12/31/结束.md"),
        (MemoryAddress.intention("完成重构"), "intentions/完成重构.md"),
    ],
)
def test_tree_maps_every_address_to_exact_confirmed_physical_path(
    tmp_path: Path,
    address: MemoryAddress,
    relative: str,
) -> None:
    tree = MemoryTree(tmp_path / "memory")
    assert tree.path_for(address) == tree.root / relative
    assert tree.path_for_uri(MemoryURI.from_address(address)) == tree.root / relative


@pytest.mark.parametrize(
    ("first_name", "alias_name"),
    [
        ("Theme", "theme"),
        ("Café", "Cafe\u0301"),
    ],
)
def test_filesystem_aliases_share_one_memory_identity_and_lock(
    tmp_path: Path,
    first_name: str,
    alias_name: str,
) -> None:
    tree = MemoryTree(tmp_path / "memory")
    first = document(
        MemoryKind.PREFERENCE,
        fields={"topic": first_name, "content": "- first"},
    )
    try:
        alias = document(
            MemoryKind.PREFERENCE,
            fields={"topic": alias_name, "content": "- second"},
        )
    except ValueError:
        return
    tree.write(first)
    first_path = tree.path_for(first.address)
    alias_path = tree.path_for(alias.address)
    if not alias_path.exists() or not first_path.samefile(alias_path):
        pytest.skip("filesystem does not alias these names")

    first_uri = MemoryURI.from_address(first.address)
    alias_uri = MemoryURI.from_address(alias.address)
    keyspace = MemoryDocumentLockKeyspace(tree.root)
    assert first_uri == alias_uri
    assert keyspace.key(first_uri) == keyspace.key(alias_uri)


def test_tree_reads_and_updates_one_legacy_case_preserving_path(tmp_path: Path) -> None:
    document_codec = codec()
    tree = MemoryTree(tmp_path / "memory", document_codec=document_codec)
    tree.initialize()
    legacy = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "Theme", "content": "- legacy"},
    )
    legacy_path = tree.root / "preferences" / "Theme.md"
    legacy_path.write_text(document_codec.encode(legacy), encoding="utf-8")

    canonical_address = MemoryAddress.preference("theme")
    assert tree.exists(canonical_address)
    assert tree.read(canonical_address).fields["content"] == "- legacy"
    assert tree.list_addresses(MemoryKind.PREFERENCE) == (legacy.address,)

    updated = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "theme", "content": "- updated"},
        revision=2,
    )
    tree.write(updated)
    assert legacy_path.exists()
    assert [path.name for path in (tree.root / "preferences").iterdir()] == ["Theme.md"]
    assert tree.read(canonical_address).fields["content"] == "- updated"


def test_tree_rejects_multiple_physical_aliases_for_one_identity(tmp_path: Path) -> None:
    document_codec = codec()
    tree = MemoryTree(tmp_path / "memory", document_codec=document_codec)
    tree.initialize()
    first = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "Theme", "content": "- first"},
    )
    alias = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "theme", "content": "- alias"},
    )
    first_path = tree.root / "preferences" / "Theme.md"
    alias_path = tree.root / "preferences" / "theme.md"
    first_path.write_text(document_codec.encode(first), encoding="utf-8")
    alias_path.write_text(document_codec.encode(alias), encoding="utf-8")
    if first_path.samefile(alias_path):
        return

    with pytest.raises(MemoryTreeIntegrityError, match="aliases"):
        tree.read(MemoryAddress.preference("theme"))


@pytest.mark.parametrize("operation", ["read", "delete"])
def test_tree_rejects_parent_swapped_to_symlink_before_leaf_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    document_codec = codec()
    tree = MemoryTree(tmp_path / "memory", document_codec=document_codec)
    outside = MemoryTree(tmp_path / "outside", document_codec=document_codec)
    inside_document = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答风格", "content": "- 内部可信内容"},
    )
    outside_document = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答风格", "content": "- 外部替换内容"},
    )
    tree.write(inside_document)
    outside.write(outside_document)
    inside_filename = tree.path_for(inside_document.address).name
    inside_parent = tree.root / "preferences"
    outside_parent = outside.root / "preferences"
    parked = tree.root / "preferences-parked"
    original = atomic_file_module.require_safe_artifact_path
    swapped = False

    def validate_then_swap(root_value, path_value, *, label):
        nonlocal swapped
        candidate = original(root_value, path_value, label=label)
        if not swapped:
            inside_parent.rename(parked)
            inside_parent.symlink_to(outside_parent, target_is_directory=True)
            swapped = True
        return candidate

    monkeypatch.setattr(
        atomic_file_module,
        "require_safe_artifact_path",
        validate_then_swap,
    )

    with pytest.raises(MemoryTreeIntegrityError, match="safely"):
        getattr(tree, operation)(inside_document.address)
    assert outside.path_for(outside_document.address).exists()
    assert (parked / inside_filename).exists()


def test_tree_rejects_parent_swapped_to_symlink_before_directory_enumeration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = MemoryTree(tmp_path / "memory")
    outside = MemoryTree(tmp_path / "outside")
    tree.write(document(MemoryKind.PREFERENCE, fields={"topic": "内部", "content": "- inside"}))
    outside.write(document(MemoryKind.PREFERENCE, fields={"topic": "外部", "content": "- outside"}))
    inside_parent = tree.root / "preferences"
    outside_parent = outside.root / "preferences"
    parked = tree.root / "preferences-parked"
    original = atomic_file_module.require_safe_artifact_path
    swapped = False

    def validate_then_swap(root_value, path_value, *, label):
        nonlocal swapped
        candidate = original(root_value, path_value, label=label)
        if not swapped:
            inside_parent.rename(parked)
            inside_parent.symlink_to(outside_parent, target_is_directory=True)
            swapped = True
        return candidate

    monkeypatch.setattr(
        atomic_file_module,
        "require_safe_artifact_path",
        validate_then_swap,
    )

    with pytest.raises(MemoryTreeIntegrityError, match="safely"):
        tree.list_addresses(MemoryKind.PREFERENCE)
    assert (outside_parent / "外部.md").exists()
    assert (parked / "内部.md").exists()


@pytest.mark.parametrize("directory", DIRECTORIES)
def test_tree_maps_every_directory_and_semantic_layer(tmp_path: Path, directory: MemoryDirectory) -> None:
    tree = MemoryTree(tmp_path / "memory")
    assert tree.directory_path(directory) == tree.root.joinpath(*directory.parts)
    assert tree.path_for_uri(MemoryURI.from_directory(directory)) == tree.directory_path(directory)
    for level in (MemoryLevel.ABSTRACT, MemoryLevel.OVERVIEW):
        expected = tree.directory_path(directory) / level.sidecar_filename
        assert tree.layer_path(directory, level) == expected
        assert tree.path_for_uri(MemoryURI.from_layer(directory, level)) == expected


@pytest.mark.parametrize("value", [None, 1, True, "address", object()])
def test_tree_path_functions_reject_wrong_domain_types(tmp_path: Path, value: object) -> None:
    tree = MemoryTree(tmp_path / "memory")
    with pytest.raises(TypeError):
        tree.path_for(value)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        tree.directory_path(value)  # type: ignore[arg-type]


def test_initialize_is_idempotent_and_creates_only_static_tree(tmp_path: Path) -> None:
    tree = MemoryTree(tmp_path / "memory")
    assert tree.initialize() == tree.initialize() == tree.root
    assert sorted(item.name for item in tree.root.iterdir()) == [
        "entities",
        "events",
        "intentions",
        "preferences",
        "tools",
    ]
    assert all(item.is_dir() for item in tree.root.iterdir())


@pytest.mark.parametrize("kind", tuple(MemoryKind))
@pytest.mark.parametrize("revision", [1, 2, 10])
def test_tree_write_read_and_overwrite_round_trip_each_kind_revision(
    tmp_path: Path, kind: MemoryKind, revision: int
) -> None:
    tree = MemoryTree(tmp_path / "memory")
    item = document(kind, revision=revision)
    assert tree.write(item) == item
    assert tree.exists(item.address)
    assert tree.read(item.address) == item
    assert tree.path_for(item.address).read_text(encoding="utf-8") == tree.document_codec.encode(item)


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_tree_missing_read_exists_and_delete_have_explicit_semantics(tmp_path: Path, kind: MemoryKind) -> None:
    tree = MemoryTree(tmp_path / "memory")
    address = document(kind).address
    assert tree.exists(address) is False
    assert tree.delete(address) is False
    with pytest.raises(FileNotFoundError):
        tree.read(address)


@pytest.mark.parametrize("value", [None, 1, True, {}, [], "document"])
def test_tree_write_rejects_non_document_values(tmp_path: Path, value: object) -> None:
    with pytest.raises(TypeError):
        MemoryTree(tmp_path / "memory").write(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_tree_enforces_markdown_body_and_encoded_size_limits(tmp_path: Path, kind: MemoryKind) -> None:
    item = document(kind)
    with pytest.raises(Exception, match="body"):
        MemoryTree(
            tmp_path / f"body-{kind.value}",
            document_config=MemoryDocumentConfig(max_markdown_body_chars=1),
        ).write(item)
    with pytest.raises(Exception, match="encoded"):
        MemoryTree(
            tmp_path / f"encoded-{kind.value}",
            document_config=MemoryDocumentConfig(max_encoded_bytes=1),
        ).write(item)


@pytest.mark.parametrize("directory", DIRECTORIES)
@pytest.mark.parametrize(("abstract", "overview"), [("L0", "L1"), ("摘要\n第二行", "概览\n第二行"), (" x ", " y ")])
def test_tree_layers_round_trip_for_every_valid_directory(
    tmp_path: Path,
    directory: MemoryDirectory,
    abstract: str,
    overview: str,
) -> None:
    tree = MemoryTree(tmp_path / "memory")
    tree.initialize()
    current = MemoryDirectory.root()
    for part in directory.parts:
        current = MemoryDirectory((*current.parts, part))
        tree.directory_path(current).mkdir(exist_ok=True)
    paths = tree.write_layers(directory, abstract=abstract, overview=overview)
    assert paths == (
        tree.layer_path(directory, MemoryLevel.ABSTRACT),
        tree.layer_path(directory, MemoryLevel.OVERVIEW),
    )
    assert tree.read_layer(directory, MemoryLevel.ABSTRACT) == abstract
    assert tree.read_layer(directory, MemoryLevel.OVERVIEW) == overview
    assert tree.layer_exists(directory, MemoryLevel.ABSTRACT)
    assert tree.layer_exists(directory, MemoryLevel.OVERVIEW)
    assert tree.delete_layers(directory) is True
    assert tree.delete_layers(directory) is False


@pytest.mark.parametrize(
    ("abstract", "overview"), [("", "overview"), (" ", "overview"), ("abstract", ""), ("abstract", "\n")]
)
def test_tree_layers_reject_empty_semantics(tmp_path: Path, abstract: str, overview: str) -> None:
    tree = MemoryTree(tmp_path / "memory")
    tree.initialize()
    with pytest.raises(ValueError, match="non-empty"):
        tree.write_layers(MemoryDirectory.root(), abstract=abstract, overview=overview)


@pytest.mark.parametrize(
    ("abstract", "overview"), [(1, "overview"), ("abstract", 1), (None, "overview"), ("abstract", None)]
)
def test_tree_layers_reject_non_text_semantics(tmp_path: Path, abstract: object, overview: object) -> None:
    tree = MemoryTree(tmp_path / "memory")
    tree.initialize()
    with pytest.raises(TypeError):
        tree.write_layers(MemoryDirectory.root(), abstract=abstract, overview=overview)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_bytes", [0, -1, True, False, 1.5, "10", None])
def test_bounded_layer_read_rejects_invalid_limit(tmp_path: Path, max_bytes: object) -> None:
    tree = MemoryTree(tmp_path / "memory")
    tree.initialize()
    tree.write_layers(MemoryDirectory.root(), abstract="摘要", overview="概览")
    with pytest.raises(ValueError):
        tree.read_layer_bounded(MemoryDirectory.root(), MemoryLevel.ABSTRACT, max_bytes=max_bytes)  # type: ignore[arg-type]


@pytest.mark.parametrize("maximum", [1, 2, 5, 10])
def test_bounded_layer_read_accepts_exact_size_and_rejects_one_byte_less(tmp_path: Path, maximum: int) -> None:
    content = "a" * maximum
    tree = MemoryTree(tmp_path / "memory")
    tree.initialize()
    tree.write_layers(MemoryDirectory.root(), abstract=content, overview="overview")
    assert tree.read_layer_bounded(MemoryDirectory.root(), MemoryLevel.ABSTRACT, max_bytes=maximum) == content
    if maximum > 1:
        with pytest.raises(MemoryTreeIntegrityError, match="bound"):
            tree.read_layer_bounded(MemoryDirectory.root(), MemoryLevel.ABSTRACT, max_bytes=maximum - 1)


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_direct_addresses_lists_only_direct_l2_for_each_leaf(tmp_path: Path, kind: MemoryKind) -> None:
    tree = MemoryTree(tmp_path / "memory")
    item = document(kind)
    tree.write(item)
    directory = MemoryDirectory.for_address(item.address)
    assert tree.direct_addresses(directory) == (item.address,)


@pytest.mark.parametrize("count", [1, 2, 5, 20])
@pytest.mark.parametrize("kind", [MemoryKind.PREFERENCE, MemoryKind.TOOL, MemoryKind.INTENTION])
def test_leaf_enumeration_is_sorted_and_bounded(tmp_path: Path, count: int, kind: MemoryKind) -> None:
    tree = MemoryTree(tmp_path / "memory")
    addresses = []
    for index in reversed(range(count)):
        fields = memory_fields(kind)
        key = {
            MemoryKind.PREFERENCE: "topic",
            MemoryKind.TOOL: "tool_name",
            MemoryKind.INTENTION: "intent_name",
        }[kind]
        fields[key] = f"item-{index:03d}"
        item = codec().build(
            kind,
            fields,
            metadata=MemoryDocumentMetadata.initial(BASE_TIME, confirmed=kind is MemoryKind.INTENTION),
        )
        tree.write(item)
        addresses.append(item.address)
    expected = tuple(sorted(addresses, key=lambda item: item.name))
    directory = MemoryDirectory.for_address(addresses[0])
    assert tree.direct_addresses(directory, limit=count) == expected
    if count > 1:
        with pytest.raises(MemoryTreeIntegrityError, match="direct L2 bound"):
            tree.direct_addresses(directory, limit=count - 1)


@pytest.mark.parametrize("limit", [0, -1, 10_001, True, False, 1.5, "1", "x", None])
def test_direct_and_child_directory_limits_reject_invalid_values(tmp_path: Path, limit: object) -> None:
    tree = MemoryTree(tmp_path / "memory")
    tree.initialize()
    with pytest.raises((TypeError, ValueError)):
        tree.direct_addresses(MemoryDirectory.root(), limit=limit)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        tree.child_directories(MemoryDirectory.root(), limit=limit)  # type: ignore[arg-type]


def test_child_directories_follow_entity_and_event_hierarchy(tmp_path: Path) -> None:
    tree = MemoryTree(tmp_path / "memory")
    items = (
        document(MemoryKind.ENTITY),
        document(MemoryKind.EVENT),
    )
    for item in items:
        tree.write(item)
    assert tree.child_directories(MemoryDirectory.root()) == tuple(
        MemoryDirectory((name,)) for name in tree._STATIC_DIRECTORIES
    )
    assert tree.child_directories(MemoryDirectory.entities()) == (MemoryDirectory.entities("项目"),)
    assert tree.child_directories(MemoryDirectory.events()) == (MemoryDirectory.events(2026),)
    assert tree.child_directories(MemoryDirectory.events(2026)) == (MemoryDirectory.events(2026, 7),)
    assert tree.child_directories(MemoryDirectory.events(2026, 7)) == (MemoryDirectory.events(2026, 7, 1),)
    assert tree.child_directories(MemoryDirectory.events(2026, 7, 1)) == ()


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_list_addresses_kind_filter_and_global_limit(tmp_path: Path, kind: MemoryKind) -> None:
    tree = MemoryTree(tmp_path / "memory")
    items = tuple(document(item_kind) for item_kind in MemoryKind)
    for item in items:
        tree.write(item)
    assert tree.list_addresses(kind) == (next(item.address for item in items if item.kind is kind),)
    assert tree.list_addresses(limit=1) == (items[0].address,)
    assert tree.list_addresses(limit=6) == tuple(item.address for item in items)


@pytest.mark.parametrize("limit", [0, -1, 10_001, 1_000_000, True, False, 1.5, "1"])
def test_global_list_rejects_invalid_limit(tmp_path: Path, limit: object) -> None:
    with pytest.raises(ValueError):
        MemoryTree(tmp_path / "memory").list_addresses(limit=limit)  # type: ignore[arg-type]


def test_non_integer_global_limit_cannot_silently_truncate_a_populated_tree(tmp_path: Path) -> None:
    tree = MemoryTree(tmp_path / "memory")
    tree.write(document(MemoryKind.PROFILE))
    tree.write(document(MemoryKind.PREFERENCE))

    with pytest.raises(ValueError):
        tree.list_addresses(limit=1.5)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "uri",
    [
        "memory://events/26/07/28/event.md",
        "memory://events/2026/7/28/event.md",
        "memory://events/2026/07/8/event.md",
    ],
)
def test_tree_uri_boundary_rejects_noncanonical_event_date_widths(tmp_path: Path, uri: str) -> None:
    tree = MemoryTree(tmp_path / "memory")

    with pytest.raises(ValueError):
        tree.path_for_uri(uri)


@pytest.mark.parametrize("kind", [MemoryKind.ENTITY, MemoryKind.EVENT])
def test_delete_prunes_only_empty_dynamic_directories_and_their_layers(tmp_path: Path, kind: MemoryKind) -> None:
    tree = MemoryTree(tmp_path / "memory")
    item = document(kind)
    tree.write(item)
    directory = MemoryDirectory.for_address(item.address)
    tree.write_layers(directory, abstract="摘要", overview="概览")
    assert tree.delete(item.address) is True
    assert tree.directory_exists(directory) is False
    static_root = MemoryDirectory.entities() if kind is MemoryKind.ENTITY else MemoryDirectory.events()
    assert tree.directory_exists(static_root) is True


@pytest.mark.parametrize("kind", [MemoryKind.PREFERENCE, MemoryKind.TOOL, MemoryKind.INTENTION])
def test_delete_never_prunes_static_leaf_directory(tmp_path: Path, kind: MemoryKind) -> None:
    tree = MemoryTree(tmp_path / "memory")
    item = document(kind)
    tree.write(item)
    assert tree.delete(item.address)
    assert tree.directory_exists(MemoryDirectory.for_address(item.address))


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_tree_detects_non_utf8_document_before_decode(tmp_path: Path, kind: MemoryKind) -> None:
    tree = MemoryTree(tmp_path / "memory")
    item = document(kind)
    tree.write(item)
    tree.path_for(item.address).write_bytes(b"\xff\xfe")
    with pytest.raises(MemoryTreeIntegrityError, match="UTF-8"):
        tree.read(item.address)


@pytest.mark.parametrize(
    "directory",
    [MemoryDirectory.root(), MemoryDirectory.preferences(), MemoryDirectory.entities(), MemoryDirectory.events()],
)
def test_tree_detects_unsupported_hidden_entry(tmp_path: Path, directory: MemoryDirectory) -> None:
    tree = MemoryTree(tmp_path / "memory")
    tree.initialize()
    path = tree.directory_path(directory)
    (path / ".unknown").write_text("x", encoding="utf-8")
    operation = (
        tree.direct_addresses
        if directory not in {MemoryDirectory.entities(), MemoryDirectory.events()}
        else tree.child_directories
    )
    with pytest.raises(MemoryTreeIntegrityError, match="hidden"):
        operation(directory)


@pytest.mark.parametrize(
    "directory", [MemoryDirectory.preferences(), MemoryDirectory.tools(), MemoryDirectory.intentions()]
)
def test_tree_detects_subdirectory_in_leaf_directory(tmp_path: Path, directory: MemoryDirectory) -> None:
    tree = MemoryTree(tmp_path / "memory")
    tree.initialize()
    (tree.directory_path(directory) / "unexpected").mkdir()
    with pytest.raises(MemoryTreeIntegrityError):
        tree.direct_addresses(directory)
    with pytest.raises(MemoryTreeIntegrityError, match="leaf"):
        tree.child_directories(directory)


@pytest.mark.parametrize("branch", [MemoryDirectory.entities(), MemoryDirectory.events(), MemoryDirectory.events(2026)])
def test_tree_detects_file_in_branch_directory(tmp_path: Path, branch: MemoryDirectory) -> None:
    tree = MemoryTree(tmp_path / "memory")
    tree.initialize()
    path = tree.directory_path(branch)
    path.mkdir(parents=True, exist_ok=True)
    (path / "unexpected.md").write_text("x", encoding="utf-8")
    with pytest.raises(MemoryTreeIntegrityError):
        tree.child_directories(branch)


@pytest.mark.parametrize("target", ["document", "directory", "layer"])
def test_tree_detects_symbolic_links_at_all_persistence_boundaries(tmp_path: Path, target: str) -> None:
    tree = MemoryTree(tmp_path / "memory")
    tree.initialize()
    outside = tmp_path / "outside"
    outside.write_text("outside", encoding="utf-8")
    if target == "document":
        path = tree.path_for(MemoryAddress.preference("主题"))
        path.symlink_to(outside)
        with pytest.raises(MemoryTreeIntegrityError):
            tree.exists(MemoryAddress.preference("主题"))
    elif target == "directory":
        path = tree.root / "entities" / "项目"
        path.symlink_to(tmp_path, target_is_directory=True)
        with pytest.raises(MemoryTreeIntegrityError):
            tree.directory_exists(MemoryDirectory.entities("项目"))
    else:
        path = tree.layer_path(MemoryDirectory.root(), MemoryLevel.ABSTRACT)
        path.symlink_to(outside)
        with pytest.raises(MemoryTreeIntegrityError):
            tree.layer_exists(MemoryDirectory.root(), MemoryLevel.ABSTRACT)


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_tree_read_detects_canonical_body_tampering(tmp_path: Path, kind: MemoryKind) -> None:
    tree = MemoryTree(tmp_path / "memory")
    item = document(kind)
    tree.write(item)
    path = tree.path_for(item.address)
    raw = path.read_text(encoding="utf-8")
    path.write_text("篡改\n" + raw, encoding="utf-8")
    with pytest.raises(MemoryTreeIntegrityError, match="integrity"):
        tree.read(item.address)


def test_tree_overwrite_preserves_address_and_accepts_next_revision(tmp_path: Path) -> None:
    tree = MemoryTree(tmp_path / "memory")
    first = document(MemoryKind.PREFERENCE)
    updated_fields = dict(first.fields)
    updated_fields["content"] = "- 偏好结构化回答"
    second = codec().build(
        MemoryKind.PREFERENCE,
        updated_fields,
        metadata=first.metadata.next_revision(BASE_TIME + timedelta(days=1)),
    )
    tree.write(first)
    tree.write(second)
    assert tree.read(first.address) == second
    assert tree.list_addresses(MemoryKind.PREFERENCE) == (first.address,)
