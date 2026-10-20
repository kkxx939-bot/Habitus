"""类的成员被拆分 / 合并改了，涉及它的关系退回候选、重新做前向验证（裁定 59）。"""

from __future__ import annotations

from datetime import date

from habitus.runtime.scene_night import _changed_classes, _touched
from habitus.scene.concepts.catalog import CatalogChanges, MovedOccurrence
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.relations import RelationKey, Segment
from habitus.scene.relations.state import LaneState, Relation, Status, revert_touched
from tests.unit.scene.concept_fixtures import BALL, EXERCISE, RUNNING, cid

NIGHT = date(2026, 9, 1)


def _state() -> LaneState:
    wake = RelationKey("session", "起床", cid("打球"), Segment.REST_OF_DAY)
    other = RelationKey("session", "起床", cid("跑步"), Segment.REST_OF_DAY)
    waiting = RelationKey("session", "周三", cid("打球"), Segment.SITUATION)
    return LaneState(
        lane="session",
        night=date(2026, 8, 31),
        family=3,
        relations=(
            Relation(
                wake, Status.ESTABLISHED, True, established_on=date(2026, 7, 1), maintained_through=date(2026, 8, 1)
            ),
            Relation(other, Status.ESTABLISHED, True, established_on=date(2026, 7, 1)),
            Relation(waiting, Status.CANDIDATE, True, discovered_on=date(2026, 8, 20), forward_needed=6),
        ),
    )


def test_relations_touching_a_changed_class_go_back_to_candidate() -> None:
    reverted, transitions = revert_touched(_state(), frozenset({cid("打球")}), NIGHT)
    by_consequent = {(item.key.antecedent, item.key.consequent): item for item in reverted.relations}
    ball = by_consequent[("起床", cid("打球"))]
    assert ball.status is Status.CANDIDATE and ball.discovered_on == NIGHT and ball.established_on is None
    assert by_consequent[("周三", cid("打球"))].discovered_on == NIGHT  # 候选也从这一晚重新攒新数据
    assert by_consequent[("起床", cid("跑步"))].status is Status.ESTABLISHED  # 没牵到的不动
    assert len(transitions) == 2 and all(item.after is Status.CANDIDATE for item in transitions)


def test_changed_classes_and_the_concepts_they_touch() -> None:
    changes = CatalogChanges(
        since=1,
        version=2,
        split_from={cid("跑步"): cid("打球")},
        moved=(MovedOccurrence("behavior://x", cid("打球"), cid("跑步")),),
    )
    changed = _changed_classes(changes)
    assert changed == frozenset({cid("打球"), cid("跑步")})
    touched = _touched(ConceptSet((BALL, RUNNING, EXERCISE)), changed)
    assert touched == frozenset({cid("打球"), cid("跑步"), "运动"})  # 汇总概念也牵到
    assert _touched(ConceptSet((BALL,)), frozenset()) == frozenset()
