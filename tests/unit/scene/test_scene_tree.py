"""情景树存储：按天两阶段发布、多代留存、逻辑地址不随重建改变、读侧核对指针摘要。"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import pytest

from habitus.scene import (
    SceneAddress,
    SceneDocumentIntegrityError,
    SceneDocumentMetadata,
    SceneLinkType,
    SceneStoredLink,
    SceneTree,
    SceneTreeConfig,
    SceneTreeConflictError,
    SceneTreeIntegrityError,
)
from habitus.scene.tree import GENERATIONS_DIRECTORY, POINTER_FILENAME
from tests.unit.scene.scene_payloads import BUY_GROCERIES, DAY, DIGEST, local, scene_payload, shopping_payload

PUBLISHED = datetime(2026, 8, 17, 1, 0, tzinfo=UTC)
VERSION = "scene_grouping_v1+schema0000"


def build(tree: SceneTree, payload: dict, links: tuple[SceneStoredLink, ...] = ()):
    return tree.document_codec.build(payload, metadata=SceneDocumentMetadata(created_at=PUBLISHED), links=links)


def publish(tree: SceneTree, documents: list, *, at: datetime = PUBLISHED, day: date = DAY):
    return tree.publish_day(day, documents, published_at=at, source_digest=DIGEST, scene_version=VERSION)


def day_dir(tmp_path, day: date = DAY):
    return tmp_path / "scene-tree" / "scenes" / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"


def test_publish_day_then_read_back(tmp_path) -> None:
    tree = SceneTree(tmp_path / "scene-tree")
    shopping = build(tree, shopping_payload())
    dinner_uri = build(tree, scene_payload()).uri
    dinner = build(tree, scene_payload(), (SceneStoredLink.between(dinner_uri, shopping.uri, SceneLinkType.NEEDS),))
    assert tree.list_days() == ()
    assert tree.day_state(DAY) is None
    published = publish(tree, [dinner, shopping])
    assert published.document_count == 2 and published.day == DAY
    assert tree.list_days() == (DAY,)
    assert tree.day_state(DAY) == published
    documents = tree.read_day(DAY)
    assert [d.address.label for d in documents] == ["去超市采购", "准备晚饭"]  # 按开始瞬时排序
    assert tree.read(dinner.address) == dinner
    assert tree.exists(dinner.address) and not tree.exists(SceneAddress(DAY, "不存在", local(1, 0)))
    with pytest.raises(FileNotFoundError):
        tree.read(SceneAddress(DAY, "不存在", local(1, 0)))
    assert tree.path_for(dinner.address).is_file()
    assert tree.generations_of(DAY) == (published.generation,)
    pointer = json.loads((day_dir(tmp_path) / POINTER_FILENAME).read_text())
    assert pointer["generation"] == published.generation and pointer["document_count"] == 2
    assert published.generation.endswith(published.digest[:12])


def test_republish_replaces_generation_keeps_address_and_prunes(tmp_path) -> None:
    tree = SceneTree(tmp_path / "scene-tree", tree_config=SceneTreeConfig(retained_generations=2))
    first = build(tree, scene_payload())
    publish(tree, [first])
    second = build(tree, scene_payload(effects=("晚饭做好了", "厨房收拾过")))
    generation_2 = publish(tree, [second], at=PUBLISHED + timedelta(hours=1))
    assert tree.read(first.address) == second  # 地址不变，内容换成新一代
    assert len(tree.generations_of(DAY)) == 2
    third = build(tree, scene_payload(effects=("第三版",)))
    generation_3 = publish(tree, [third], at=PUBLISHED + timedelta(hours=2))
    remaining = tree.generations_of(DAY)
    assert set(remaining) == {generation_2.generation, generation_3.generation}
    assert tree.day_state(DAY) == generation_3


def test_prune_at_the_minimum_retention_removes_leftover_temp_files(tmp_path) -> None:
    with pytest.raises(ValueError, match="between 2"):
        SceneTreeConfig(retained_generations=1)  # 翻指针后立刻删旧代会撞不持锁的读侧
    tree = SceneTree(tmp_path / "scene-tree", tree_config=SceneTreeConfig(retained_generations=2))
    first = publish(tree, [build(tree, scene_payload())])
    # 模拟上一次崩溃留下的本模块临时文件
    leftover = day_dir(tmp_path) / GENERATIONS_DIRECTORY / first.generation / ".x.md.0123456789abcdef0123456789abcdef.tmp"
    leftover.write_bytes(b"")
    second = publish(tree, [build(tree, scene_payload(effects=("v2",)))], at=PUBLISHED + timedelta(hours=1))
    third = publish(tree, [build(tree, scene_payload(effects=("v3",)))], at=PUBLISHED + timedelta(hours=2))
    assert set(tree.generations_of(DAY)) == {second.generation, third.generation}
    assert not leftover.exists()


def test_zero_document_day_is_processed_but_empty(tmp_path) -> None:
    tree = SceneTree(tmp_path / "scene-tree")
    published = publish(tree, [])
    assert published.document_count == 0
    assert tree.read_day(DAY) == ()
    assert tree.day_state(DAY) is not None
    assert tree.list_days() == (DAY,)


def test_publish_is_idempotent_and_conflicts_on_different_bytes(tmp_path) -> None:
    tree = SceneTree(tmp_path / "scene-tree")
    document = build(tree, scene_payload())
    first = publish(tree, [document])
    again = publish(tree, [document])
    assert again == first and tree.generations_of(DAY) == (first.generation,)
    generation_dir = day_dir(tmp_path) / GENERATIONS_DIRECTORY / first.generation
    (generation_dir / f"{document.address.identity_name}.md").write_text("tampered\n")
    with pytest.raises(SceneTreeConflictError):
        publish(tree, [document])
    # 冲突不清理别人的那一代
    assert generation_dir.is_dir()


def test_publish_validates_inputs_before_writing(tmp_path) -> None:
    tree = SceneTree(tmp_path / "scene-tree")
    document = build(tree, scene_payload())
    with pytest.raises(SceneTreeConflictError, match="does not belong"):
        publish(tree, [document], day=date(2026, 8, 17))
    twin = build(tree, scene_payload(label="准备晚饭"))
    with pytest.raises(SceneTreeConflictError, match="same identity"):
        publish(tree, [document, twin])
    with pytest.raises(ValueError):
        tree.publish_day(DAY, [document], published_at=datetime(2026, 8, 17), source_digest=DIGEST, scene_version=VERSION)
    with pytest.raises(ValueError):
        tree.publish_day(DAY, [document], published_at=PUBLISHED, source_digest="not-a-digest", scene_version=VERSION)
    with pytest.raises(ValueError):
        tree.publish_day(DAY, [document], published_at=PUBLISHED, source_digest=DIGEST, scene_version="")
    # 文档溯源与指针输入必须一致
    with pytest.raises(SceneTreeConflictError, match="provenance"):
        tree.publish_day(DAY, [document], published_at=PUBLISHED, source_digest="e" * 64, scene_version=VERSION)
    with pytest.raises(SceneTreeConflictError, match="provenance"):
        tree.publish_day(DAY, [document], published_at=PUBLISHED, source_digest=DIGEST, scene_version="other")
    # 全部拒绝都发生在写入之前
    assert not day_dir(tmp_path).exists() or tree.generations_of(DAY) == ()


def test_failed_read_back_discards_the_orphan_generation(tmp_path, monkeypatch) -> None:
    tree = SceneTree(tmp_path / "scene-tree")
    document = build(tree, scene_payload())
    original = tree.document_codec.decode

    def broken(raw: str, *, expected_address):
        raise SceneDocumentIntegrityError("simulated read-back failure")

    monkeypatch.setattr(tree.document_codec, "decode", broken)
    with pytest.raises(SceneTreeIntegrityError):
        publish(tree, [document])
    monkeypatch.setattr(tree.document_codec, "decode", original)
    assert tree.day_state(DAY) is None
    assert tree.generations_of(DAY) == ()


def test_read_side_rejects_substituted_document_and_tampered_pointer(tmp_path) -> None:
    tree = SceneTree(tmp_path / "scene-tree", tree_config=SceneTreeConfig(retained_generations=3))
    v1 = build(tree, scene_payload(effects=("v1",)))
    generation_1 = publish(tree, [v1])
    v2 = build(tree, scene_payload(effects=("v2",)))
    generation_2 = publish(tree, [v2], at=PUBLISHED + timedelta(hours=1))
    # 把旧代里同址的合法文档拷到当前代：codec 认，摘要不认
    name = f"{v1.address.identity_name}.md"
    old_file = day_dir(tmp_path) / GENERATIONS_DIRECTORY / generation_1.generation / name
    new_file = day_dir(tmp_path) / GENERATIONS_DIRECTORY / generation_2.generation / name
    new_file.write_bytes(old_file.read_bytes())
    with pytest.raises(SceneTreeIntegrityError, match="does not match its pointer"):
        tree.read_day(DAY)
    with pytest.raises(SceneTreeIntegrityError):
        tree.exists(v1.address)  # 完整性问题不得伪装成"不存在"
    new_file.write_text("broken\n")
    with pytest.raises(SceneTreeIntegrityError):
        tree.read_day(DAY)
    pointer = day_dir(tmp_path) / POINTER_FILENAME
    pointer.write_text("{not json")
    with pytest.raises(SceneTreeIntegrityError):
        tree.day_state(DAY)
    pointer.write_text(json.dumps({"generation": "x"}))
    with pytest.raises(SceneTreeIntegrityError):
        tree.day_state(DAY)
    bad = json.loads(json.dumps(generation_2.to_dict()))
    bad["generation"] = "../escape"
    pointer.write_text(json.dumps(bad))
    with pytest.raises(SceneTreeIntegrityError):
        tree.day_state(DAY)
    bad["generation"] = generation_2.generation
    bad["digest"] = "0" * 64
    pointer.write_text(json.dumps(bad))
    with pytest.raises(SceneTreeIntegrityError):
        tree.day_state(DAY)


def test_pointer_to_missing_generation_is_an_integrity_error(tmp_path) -> None:
    tree = SceneTree(tmp_path / "scene-tree")
    published = publish(tree, [build(tree, scene_payload())])
    generation_dir = day_dir(tmp_path) / GENERATIONS_DIRECTORY / published.generation
    for child in generation_dir.iterdir():
        child.unlink()
    generation_dir.rmdir()
    assert tree.list_days() == (DAY,)
    with pytest.raises(SceneTreeIntegrityError, match="missing"):
        tree.read_day(DAY)


def test_links_round_trip_through_the_store(tmp_path) -> None:
    """存储不校验边目标是否存在（那是 M2 归组校验的职责）；只保证边能往返。"""

    tree = SceneTree(tmp_path / "scene-tree")
    dinner = build(tree, scene_payload())
    linked = build(tree, scene_payload(), (SceneStoredLink.between(dinner.uri, BUY_GROCERIES, SceneLinkType.RESULTS_FROM),))
    publish(tree, [linked])
    assert tree.read(linked.address).links == linked.links


def test_read_day_orders_by_instant_across_offsets(tmp_path) -> None:
    """同一天里不同偏移的两个情景按瞬时排序，不按墙钟。"""

    from datetime import timezone

    tree = SceneTree(tmp_path / "scene-tree")
    from tests.unit.scene.scene_payloads import occurrence_uri

    west = occurrence_uri("west", datetime(2026, 8, 16, 9, 0, tzinfo=timezone(timedelta(hours=-7))))  # 16:00Z
    east = occurrence_uri("east", datetime(2026, 8, 16, 22, 0, tzinfo=timezone(timedelta(hours=8))))  # 14:00Z
    west_scene = build(tree, scene_payload(label="西", started_at=datetime(2026, 8, 16, 9, 0, tzinfo=timezone(timedelta(hours=-7))), ended_at=datetime(2026, 8, 16, 9, 30, tzinfo=timezone(timedelta(hours=-7))), members=({"uri": west, "role": "essential"},), pending_effects=()))
    east_scene = build(tree, scene_payload(label="东", started_at=datetime(2026, 8, 16, 22, 0, tzinfo=timezone(timedelta(hours=8))), ended_at=datetime(2026, 8, 16, 22, 30, tzinfo=timezone(timedelta(hours=8))), members=({"uri": east, "role": "essential"},), pending_effects=()))
    publish(tree, [west_scene, east_scene])
    assert [d.address.label for d in tree.read_day(DAY)] == ["东", "西"]
