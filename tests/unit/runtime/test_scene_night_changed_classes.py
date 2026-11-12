"""词表变更牵到的关系退回候选（裁定 59 第 1 条；缺陷 2，用户 11-12）。

只有拆分、合并真的改了一个类的成员：以它为前因或后果的候选与成立退回候选、在新成员上重新做前向验证。从「待定」搬进来的
条目（待定归入已有类、长新类）不算成员变化，不打回；改判据时在两个已有类之间的搬动这一次照旧打回。

走 ``RelationStage`` 折叠前备检验对象的那一步（补问与折叠共用）：盘上接的那一代 + 词表口给的变更 → 退回之后的那一份与退回的迁移。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytest

from habitus.behavior.kinds.ids import Lane, pending_token
from habitus.conditions import LocalCalendar
from habitus.runtime.scene_night_relations import RelationStage
from habitus.scene.concepts import ConceptSet
from habitus.scene.concepts.catalog import CatalogChanges, CatalogClass, MovedOccurrence
from habitus.scene.concepts.store import ConceptStore
from habitus.scene.occurrences import ConceptHitStore
from habitus.scene.relations import LaneState, Relation, RelationKey, Segment, Status, build_timelines
from habitus.scene.relations.longspan import MEASURE_VERSIONS
from habitus.scene.relations.store import RelationStore
from tests.unit.kind_ids import kind_id
from tests.unit.scene.concept_fixtures import BALL, EXERCISE, RUNNING, base
from tests.unit.scene.relation_fixtures import CODE, CONFIG, DISCUSS, DOCS, PUSH, RESEARCH, TITLES, planted

PENDING = pending_token(Lane.SESSION)
#: 从「讨论方案」里拆出来的新类。
SPLIT_OFF = kind_id("看日程")
SERIES = planted(12)
TONIGHT = SERIES.cutoff
STORED = TONIGHT - timedelta(days=1)
FOUND = TONIGHT - timedelta(days=20)
CONCEPTS = ConceptSet((*(base(title) for title in TITLES), BALL, RUNNING, EXERCISE))


def _key(antecedent: str, consequent: str) -> RelationKey:
    return RelationKey("session", antecedent, consequent, Segment.CHAIN)


DISCUSS_CODE = _key(DISCUSS, CODE)  # 成立
RESEARCH_DISCUSS = _key(RESEARCH, DISCUSS)  # 候选
PUSH_DOCS = _key(PUSH, DOCS)  # 候选
EXERCISE_DOCS = _key("运动", DOCS)  # 成立；「运动」= 打球 + 跑步
ALL = (DISCUSS_CODE, RESEARCH_DISCUSS, PUSH_DOCS, EXERCISE_DOCS)


def _established(key: RelationKey) -> Relation:
    return Relation(
        key=key,
        status=Status.ESTABLISHED,
        upward=True,
        discovered_on=FOUND,
        forward_passed_on=FOUND + timedelta(days=5),
        established_on=FOUND + timedelta(days=5),
        maintained_through=FOUND + timedelta(days=10),
        maintenance_needed=6,
        forward_needed=4,
    )


def _candidate(key: RelationKey) -> Relation:
    return Relation(key=key, status=Status.CANDIDATE, upward=True, discovered_on=FOUND, forward_needed=8)


STORED_STATE = LaneState(
    lane="session",
    night=STORED,
    family=4,
    relations=(
        _established(DISCUSS_CODE),
        _candidate(RESEARCH_DISCUSS),
        _candidate(PUSH_DOCS),
        _established(EXERCISE_DOCS),
    ),
    vocabulary=1,
    measure_versions=MEASURE_VERSIONS,
)


@dataclass
class Catalog:
    """词表口：接的那一代（版本 1）之后的变更由用例直接给。"""

    changes: CatalogChanges

    def classes(self) -> tuple[CatalogClass, ...]:
        return ()

    def version(self) -> int:
        return self.changes.version

    def changes_since(self, version: int) -> CatalogChanges:
        assert version == STORED_STATE.vocabulary
        return self.changes


def _examine(tmp_path: Path, changes: CatalogChanges) -> tuple[LaneState, dict[RelationKey, Status | None]]:
    """接盘上那一代、按变更退回；返回退回之后的那一份与退回的迁移（键 → 迁移前的状态）。"""

    relations = RelationStore(tmp_path / "scene")
    relations.write(STORED_STATE, ())
    stage = RelationStage(
        catalog=Catalog(changes),
        concepts=ConceptStore(tmp_path / "scene"),
        hits=ConceptHitStore(tmp_path / "scene"),
        calendar=LocalCalendar(),
        relations=relations,
        relation_config=CONFIG,
        advice=lambda: None,
    )
    line = build_timelines(SERIES, CONCEPTS, {}, CONFIG)["session"]
    examined = stage._examine("session", line, SERIES, CONCEPTS, [])
    assert examined.previous is not None
    reverted = {item.key: item.before for item in examined.reverted}
    assert all(item.after is Status.CANDIDATE and item.night == TONIGHT for item in examined.reverted)
    return examined.previous, reverted


def _changes(
    *,
    moved: tuple[tuple[str, str], ...] = (),
    split_from: dict[str, str] | None = None,
    merged_into: dict[str, str] | None = None,
) -> CatalogChanges:
    return CatalogChanges(
        since=1,
        version=2,
        split_from=split_from or {},
        merged_into=merged_into or {},
        moved=tuple(
            MovedOccurrence(f"behavior://x/{index}", source, target) for index, (source, target) in enumerate(moved)
        ),
    )


def _by_key(state: LaneState) -> dict[RelationKey, Relation]:
    return {item.key: item for item in state.relations}


def _untouched(state: LaneState, *keys: RelationKey) -> None:
    """这几条和盘上接的那一代一模一样（状态、发现日、成立日都没动）。"""

    stored, now = _by_key(STORED_STATE), _by_key(state)
    for key in keys:
        assert now[key] == stored[key]


def _sent_back(state: LaneState, *keys: RelationKey) -> None:
    now = _by_key(state)
    for key in keys:
        assert now[key].status is Status.CANDIDATE and now[key].discovered_on == TONIGHT
        assert now[key].established_on is None


# ── 从「待定」搬出来的：不打回 ─────────────────────────────────────────────────


def test_pending_records_filed_into_an_existing_class_send_nothing_back(tmp_path: Path) -> None:
    """待定归入已有类「讨论方案」：它是前因的成立、是后果的候选都不动，发现日不变。"""

    state, reverted = _examine(tmp_path, _changes(moved=((PENDING, DISCUSS), (PENDING, DISCUSS))))
    assert reverted == {}
    _untouched(state, *ALL)


def test_pending_records_forming_a_new_class_send_nothing_back(tmp_path: Path) -> None:
    """长新类：待定条目搬进新类「跑步」。它是汇总概念「运动」的成员，「运动 → 写文档」这条成立不因此打回。"""

    state, reverted = _examine(tmp_path, _changes(moved=((PENDING, RUNNING.name),)))
    assert reverted == {}
    _untouched(state, *ALL)


def test_a_split_still_sends_back_alongside_pending_filing(tmp_path: Path) -> None:
    """同一段变更里既有待定归入「讨论方案」、又有「提交推送」被拆：只打回牵到「提交推送」的。"""

    changes = _changes(
        moved=((PENDING, DISCUSS), (PUSH, SPLIT_OFF)),
        split_from={SPLIT_OFF: PUSH},
    )
    state, reverted = _examine(tmp_path, changes)
    assert reverted == {PUSH_DOCS: Status.CANDIDATE}
    _sent_back(state, PUSH_DOCS)
    _untouched(state, DISCUSS_CODE, RESEARCH_DISCUSS, EXERCISE_DOCS)


# ── 拆分、合并、类与类之间的搬动：照旧打回 ─────────────────────────────────────


def test_a_split_sends_back_relations_on_both_classes(tmp_path: Path) -> None:
    changes = _changes(moved=((DISCUSS, SPLIT_OFF),), split_from={SPLIT_OFF: DISCUSS})
    state, reverted = _examine(tmp_path, changes)
    assert reverted == {DISCUSS_CODE: Status.ESTABLISHED, RESEARCH_DISCUSS: Status.CANDIDATE}
    _sent_back(state, DISCUSS_CODE, RESEARCH_DISCUSS)
    _untouched(state, PUSH_DOCS, EXERCISE_DOCS)


def test_a_merge_sends_back_relations_on_both_classes(tmp_path: Path) -> None:
    changes = _changes(moved=((PUSH, DOCS),), merged_into={PUSH: DOCS})
    state, reverted = _examine(tmp_path, changes)
    assert reverted == {PUSH_DOCS: Status.CANDIDATE, EXERCISE_DOCS: Status.ESTABLISHED}
    _sent_back(state, PUSH_DOCS, EXERCISE_DOCS)
    _untouched(state, DISCUSS_CODE, RESEARCH_DISCUSS)


@pytest.mark.parametrize("count", [1, 3])
def test_records_moved_between_two_existing_classes_still_send_back(tmp_path: Path, count: int) -> None:
    """改判据：条目从「讨论方案」搬到「修改代码」（没有拆分合并）。两头都算成员变了（用户 11-12：这次照旧打回）。"""

    state, reverted = _examine(tmp_path, _changes(moved=((DISCUSS, CODE),) * count))
    assert reverted == {DISCUSS_CODE: Status.ESTABLISHED, RESEARCH_DISCUSS: Status.CANDIDATE}
    _sent_back(state, DISCUSS_CODE, RESEARCH_DISCUSS)
    _untouched(state, PUSH_DOCS, EXERCISE_DOCS)


def test_a_group_concept_is_sent_back_when_a_member_class_is_split(tmp_path: Path) -> None:
    changes = _changes(moved=((BALL.name, RUNNING.name),), split_from={RUNNING.name: BALL.name})
    state, reverted = _examine(tmp_path, changes)
    assert reverted == {EXERCISE_DOCS: Status.ESTABLISHED}
    _sent_back(state, EXERCISE_DOCS)
