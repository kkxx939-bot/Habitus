"""预测层读语义树关系的那一侧：每条 lane 最近一晚、按晚缓存；组合根缺省就接上本 Runtime 的语义树根。"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from habitus.foresight import ForesightError
from habitus.runtime.scene_relations import SceneRelations
from habitus.scene.concepts import ConceptStore
from habitus.scene.relations import LaneState, Relation, RelationKey, Segment, Status
from habitus.scene.relations.engine import Subset
from habitus.scene.relations.stats import Effect
from habitus.scene.relations.store import RelationStore
from tests.unit.kind_ids import kind_id
from tests.unit.runtime.test_foresight_wiring import EVENING, assembled
from tests.unit.scene.concept_fixtures import base

BALL = kind_id("打球")


def established(antecedent: str, consequent: str, segment: Segment) -> Relation:
    reading = Subset(
        antecedents=10,
        blocks=4,
        effect=Effect(antecedents=10, treated_rate=0.9, control_rate=0.2),
        p_value=0.001,
        interval=(0.4, 0.9),
        relative_interval=None,
    )
    return Relation(
        key=RelationKey("session", antecedent, consequent, segment),
        status=Status.ESTABLISHED,
        upward=True,
        maintenance=reading,
    )


def write_night(root, night: date, *relations: Relation) -> None:  # type: ignore[no-untyped-def]
    RelationStore(root).write(LaneState(lane="session", night=night, family=len(relations), relations=relations), ())


def test_reads_the_latest_night_and_rereads_only_when_a_new_night_lands(tmp_path, monkeypatch) -> None:
    root = tmp_path / "scene"
    ConceptStore(root).write(base("打球"))
    night = date(2026, 8, 24)
    write_night(root, night - timedelta(days=1))
    write_night(root, night, established(BALL, BALL, Segment.DAYS_2_7))
    source = SceneRelations(RelationStore(root), ConceptStore(root))
    reads: list[date] = []
    original = RelationStore.read

    def counting(self, lane, when):  # type: ignore[no-untyped-def]
        reads.append(when)
        return original(self, lane, when)

    monkeypatch.setattr(RelationStore, "read", counting)
    table = source()
    assert [relation.key.segment for relation in table.relations] == [Segment.DAYS_2_7] and BALL in table.concepts
    assert table.night == night
    source()
    assert reads == [night]  # 同一晚第二拍不重读
    write_night(root, night + timedelta(days=1))
    assert source().relations == ()
    assert reads == [night, night + timedelta(days=1)]


def test_nothing_on_disk_reads_as_an_empty_table(tmp_path) -> None:
    source = SceneRelations(RelationStore(tmp_path / "scene"), ConceptStore(tmp_path / "scene"))
    table = source()
    assert table.relations == () and len(table.concepts) == 0 and table.night is None


def test_the_assembled_layer_shows_a_present_relation_from_the_runtime_scene_root(tmp_path) -> None:
    """组合根缺省读本 Runtime 语义树根下的关系表：上周一 19:00 打过球，「之后第 2–7 天」那条关系此刻在场。"""

    config, components = assembled(tmp_path)
    ConceptStore(config.scene_root).write(base("打球"))
    write_night(config.scene_root, EVENING.date(), established(BALL, BALL, Segment.DAYS_2_7))
    pack = components.assembler.assemble(now=EVENING)
    (ball,) = [item for item in pack.candidates if item.kind_token == BALL]
    (note,) = ball.relations
    assert note.segment is Segment.DAYS_2_7 and note.seen_at.date() == EVENING.date() - timedelta(days=7)


def test_an_unreadable_relation_table_is_refused_not_skipped(tmp_path) -> None:
    """关系表坏了不能悄悄少一节：判断者会把"没读到"当成"没有关系"。"""

    config, components = assembled(tmp_path)
    write_night(config.scene_root, EVENING.date())
    (config.scene_root / "relations" / "session" / f"{EVENING.date().isoformat()}.json").write_text("{", encoding="utf-8")
    with pytest.raises(ForesightError, match="relations cannot be read"):
        components.assembler.assemble(now=EVENING)



def test_a_rewritten_night_is_read_again(tmp_path) -> None:
    """重跑最晚那一晚会重写那一晚的文件：缓存按文件指纹认，重写之后下一拍读到新的（E14）。"""

    root = tmp_path / "scene"
    ConceptStore(root).write(base("打球"))
    night = date(2026, 8, 24)
    write_night(root, night)
    source = SceneRelations(RelationStore(root), ConceptStore(root))
    assert source().relations == ()
    RelationStore(root).write(
        LaneState(lane="session", night=night, family=1, relations=(established(BALL, BALL, Segment.DAYS_2_7),)), ()
    )
    assert len(source().relations) == 1
