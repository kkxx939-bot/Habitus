"""预测树读情景树的唯一入口：情景流、带 kind_token 与偏移的成员、关系实例、覆盖日。"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from habitus.behavior import BehaviorDocumentWriter, BehaviorKind, BehaviorTree
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.prediction import scene_source
from habitus.prediction.errors import PredictionTreeError
from habitus.prediction.model import ObservedSceneMember, SceneSnapshot
from habitus.scene import SceneDocumentMetadata, SceneLinkType, SceneStoredLink, SceneTree
from tests.unit.scene.scene_payloads import DAY, DIGEST, local, occurrence_uri, scene_payload, shopping_payload

PUBLISHED = datetime(2026, 8, 17, 1, 0, tzinfo=UTC)
VERSION = "scene_grouping_v1+schema0000"


def _occurrence(name: str, started_at: datetime, token: str, original_name: str | None = None) -> dict:
    return {
        "occurred_on": started_at.date(),
        "name": name,
        "started_at": started_at,
        "kind_token": token,
        "status": "completed",
        "status_basis": "observed",
        "last_observed_at": started_at,
        "onset_available_at": started_at,
        "reminded": False,
        "goal": None,
        "summary": f"{name}。",
        "subjects": ("Jake",),
        "place": None,
        "original_name": original_name,
        "basis": (),
        "chain_digest": DIGEST,
        "fusion_version": "fusion_v1",
        "reduction_version": "behavior_reduction_v1",
    }


@pytest.fixture
def trees(tmp_path):
    behavior_tree = BehaviorTree(tmp_path / "behavior-tree")
    writer = BehaviorDocumentWriter(behavior_tree, ProcessLocalLockStore(), clock=lambda: PUBLISHED)
    writer.publish(BehaviorKind.OCCURRENCE, _occurrence("去超市买菜", local(15, 20), "买菜"))
    writer.publish(BehaviorKind.OCCURRENCE, _occurrence("商量晚餐", local(19, 12), "商量晚餐安排"))
    writer.publish(BehaviorKind.OCCURRENCE, _occurrence("查配方", local(19, 41), "查配方"))
    # 撞车消歧的重复记录：original_name 非空，语义层一律跳过
    writer.publish(BehaviorKind.OCCURRENCE, _occurrence("洗菜", local(19, 43), "洗菜", original_name="洗菜原名"))
    # 看手机故意不落树：成员解析不到要计数跳过，不硬失败
    scene_tree = SceneTree(tmp_path / "scene-tree")
    codec = scene_tree.document_codec
    shopping = codec.build(shopping_payload(), metadata=SceneDocumentMetadata(created_at=PUBLISHED))
    dinner_uri = codec.build(scene_payload(), metadata=SceneDocumentMetadata(created_at=PUBLISHED)).uri
    dinner = codec.build(
        scene_payload(),
        metadata=SceneDocumentMetadata(created_at=PUBLISHED),
        links=(SceneStoredLink.between(dinner_uri, shopping.uri, SceneLinkType.NEEDS),),
    )
    scene_tree.publish_day(DAY, [shopping, dinner], published_at=PUBLISHED, source_digest=DIGEST, scene_version=VERSION)
    # 次日：处理过但零情景
    scene_tree.publish_day(DAY + timedelta(days=1), [], published_at=PUBLISHED, source_digest=DIGEST, scene_version=VERSION)
    return scene_tree, behavior_tree


def test_read_produces_scenes_members_relations_and_coverage(trees) -> None:
    scene_tree, behavior_tree = trees
    snapshot = scene_source.read(scene_tree, behavior_tree)
    assert isinstance(snapshot, SceneSnapshot)
    assert [scene.label for scene in snapshot.scenes] == ["去超市采购", "准备晚饭"]
    assert snapshot.covered_days == (DAY, DAY + timedelta(days=1))
    assert snapshot.latest_covered_day == DAY + timedelta(days=1)
    assert snapshot.unresolved_members == 1 and snapshot.skipped_duplicates == 1
    by_scene: dict[int, list[ObservedSceneMember]] = {}
    for member in snapshot.members:
        by_scene.setdefault(member.scene_index, []).append(member)
    assert [m.action for m in by_scene[0]] == ["买菜"]
    dinner_members = {m.action: m for m in by_scene[1]}
    assert set(dinner_members) == {"商量晚餐安排", "查配方"}
    assert dinner_members["商量晚餐安排"].offset_seconds == 0.0
    assert dinner_members["查配方"].offset_seconds == 29 * 60
    assert dinner_members["查配方"].role == "essential"
    assert len(snapshot.relations) == 1
    relation = snapshot.relations[0]
    assert relation.kind == "needs" and relation.from_uri == snapshot.scenes[1].uri
    assert relation.to_uri == snapshot.scenes[0].uri and relation.lag_seconds == 13920.0


def test_read_orders_scenes_by_instant_across_days_with_different_offsets(tmp_path) -> None:
    """日期目录是本地日历日：西偏移的当天晚场比东偏移的次日凌晨更晚，快照必须按瞬时排。"""

    behavior_tree = BehaviorTree(tmp_path / "behavior-tree")
    scene_tree = SceneTree(tmp_path / "scene-tree")
    codec = scene_tree.document_codec
    west_tz, east_tz = timezone(timedelta(hours=-7)), timezone(timedelta(hours=8))
    day_1, day_2 = date(2026, 8, 16), date(2026, 8, 17)
    west_start = datetime(2026, 8, 16, 23, 0, tzinfo=west_tz)  # 17 日 06:00Z
    east_start = datetime(2026, 8, 17, 1, 0, tzinfo=east_tz)  # 16 日 17:00Z
    west = codec.build(
        scene_payload(occurred_on=day_1, label="西", started_at=west_start, ended_at=west_start + timedelta(minutes=10), members=({"uri": occurrence_uri("w", west_start), "role": "essential"},), pending_effects=()),
        metadata=SceneDocumentMetadata(created_at=PUBLISHED),
    )
    east = codec.build(
        scene_payload(occurred_on=day_2, label="东", started_at=east_start, ended_at=east_start + timedelta(minutes=10), members=({"uri": occurrence_uri("e", east_start), "role": "essential"},), pending_effects=()),
        metadata=SceneDocumentMetadata(created_at=PUBLISHED),
    )
    scene_tree.publish_day(day_1, [west], published_at=PUBLISHED, source_digest=DIGEST, scene_version=VERSION)
    scene_tree.publish_day(day_2, [east], published_at=PUBLISHED, source_digest=DIGEST, scene_version=VERSION)
    snapshot = scene_source.read(scene_tree, behavior_tree)
    assert [scene.label for scene in snapshot.scenes] == ["东", "西"]
    assert snapshot.unresolved_members == 2  # 行为树是空的


def test_read_rejects_wrong_types(trees) -> None:
    scene_tree, behavior_tree = trees
    with pytest.raises(PredictionTreeError):
        scene_source.read(behavior_tree, behavior_tree)  # type: ignore[arg-type]
    with pytest.raises(PredictionTreeError):
        scene_source.read(scene_tree, scene_tree)  # type: ignore[arg-type]


def test_a_relation_whose_scene_target_is_no_longer_current_is_dropped_and_counted(tmp_path) -> None:
    """某天被重建、后续日尚未级联重算时，指向旧情景地址的边悬空：读侧丢弃只计数，不交给预测。"""

    behavior_tree = BehaviorTree(tmp_path / "behavior-tree")
    writer = BehaviorDocumentWriter(behavior_tree, ProcessLocalLockStore(), clock=lambda: PUBLISHED)
    writer.publish(BehaviorKind.OCCURRENCE, _occurrence("商量晚餐", local(19, 12), "商量晚餐安排"))
    scene_tree = SceneTree(tmp_path / "scene-tree")
    codec = scene_tree.document_codec
    shopping = codec.build(shopping_payload(), metadata=SceneDocumentMetadata(created_at=PUBLISHED))
    dinner_uri = codec.build(scene_payload(), metadata=SceneDocumentMetadata(created_at=PUBLISHED)).uri
    dinner = codec.build(
        scene_payload(),
        metadata=SceneDocumentMetadata(created_at=PUBLISHED),
        links=(SceneStoredLink.between(dinner_uri, shopping.uri, SceneLinkType.NEEDS),),
    )
    # 只发布晚饭，不发布采购：needs 的目标情景不在任何一天的当前一代里
    scene_tree.publish_day(DAY, [dinner], published_at=PUBLISHED, source_digest=DIGEST, scene_version=VERSION)

    snapshot = scene_source.read(scene_tree, behavior_tree)

    assert snapshot.relations == ()
    assert snapshot.dangling_relations == 1
