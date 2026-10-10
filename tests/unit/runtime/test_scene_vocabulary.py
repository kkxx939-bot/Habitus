"""词表口：基础词表的只读口装成语义树要的 ``ClassCatalog``——编号的三种读法与 lane、全部类、变更日志读成新增 / 拆分 / 合并。

改名、改判据不进变更流（同步时按全部类逐个核对）；"编号 → 现在对应哪些编号"语义树不再要（汇总概念的成员在同步时改写成现编号）。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from habitus.behavior.kinds.changes import (
    AddClass,
    ChangeReason,
    Move,
    ReviseClass,
    VersionRecord,
    merge_operations,
    split_operations,
)
from habitus.behavior.kinds.ids import ClassId, Lane
from habitus.behavior.kinds.model import BehaviorClass, ClassOrigin
from habitus.behavior.kinds.reader import VocabularyReader
from habitus.behavior.kinds.store import BehaviorKindStore
from habitus.runtime.scene_vocabulary import VocabularyCatalog
from habitus.scene.concepts.catalog import ClassCatalog, MovedOccurrence
from tests.unit.behavior.kinds_fixtures import EDIT, NOW, RESEARCH, edit_class, research_class

PROMPTS = ClassId(Lane.SESSION, 3)
URI = "behavior://occurrences/2026/10/03/调整提示词@2026-10-03T10:00:00+08:00.md"


def prompt_class() -> BehaviorClass:
    return BehaviorClass(PROMPTS, "改提示词", "改写给模型的提示词", "该改提示词了", origin=ClassOrigin.PROMOTED)


def catalog_with_history(tmp_path: Path) -> VocabularyCatalog:
    """v1 起步（修改代码、调研）→ v2 从修改代码拆出改提示词（迁一条）→ v3 调研并进修改代码、修改代码改名。"""

    store = BehaviorKindStore(tmp_path)
    store.append(
        VersionRecord(1, NOW, ChangeReason.NIGHTLY, (AddClass(edit_class()), AddClass(research_class()))),
        expected_version=0,
    )
    narrowed = replace(edit_class(), criterion="动手改代码，不含改提示词")
    store.append(
        VersionRecord(
            2,
            NOW,
            ChangeReason.REVISION,
            split_operations(narrowed, prompt_class()),
            moves=(Move(URI, str(EDIT), str(PROMPTS)),),
        ),
        expected_version=1,
    )
    renamed = replace(narrowed, name="写代码")
    store.append(
        VersionRecord(3, NOW, ChangeReason.REVISION, (*merge_operations(RESEARCH, EDIT), ReviseClass(renamed))),
        expected_version=2,
    )
    return VocabularyCatalog(VocabularyReader(store))


def test_the_adapter_implements_the_catalog_protocol(tmp_path: Path) -> None:
    assert isinstance(VocabularyCatalog(VocabularyReader(BehaviorKindStore(tmp_path))), ClassCatalog)


def test_the_classes_include_retired_ones_with_their_lane_and_state(tmp_path: Path) -> None:
    catalog = catalog_with_history(tmp_path)
    classes = {item.id: item for item in catalog.classes()}
    assert set(classes) == {"s-k0001", "s-k0002", "s-k0003"}
    assert classes["s-k0001"].name == "写代码" and classes["s-k0001"].active and classes["s-k0001"].lane == "session"
    assert not classes["s-k0002"].active
    assert catalog.version() == 3
    assert classes["s-k0003"].name == "改提示词" and classes["s-k0003"].active
    assert not hasattr(catalog, "current_ids")  # 语义树不再要这一口


def test_the_change_log_reads_as_split_merged_and_moved(tmp_path: Path) -> None:
    catalog = catalog_with_history(tmp_path)
    everything = catalog.changes_since(0)
    assert (everything.since, everything.version) == (0, 3)
    assert dict(everything.split_from) == {"s-k0003": "s-k0001"}
    assert dict(everything.merged_into) == {"s-k0002": "s-k0001"}
    assert everything.moved == (MovedOccurrence(uri=URI, source="s-k0001", target="s-k0003"),)
    # 只读上次同步之后的
    later = catalog.changes_since(2)
    assert dict(later.split_from) == {} and dict(later.merged_into) == {"s-k0002": "s-k0001"}
    assert later.moved == ()


def test_no_changes_keeps_the_current_version(tmp_path: Path) -> None:
    catalog = catalog_with_history(tmp_path)
    nothing = catalog.changes_since(3)
    assert (nothing.since, nothing.version) == (3, 3)
    assert (dict(nothing.split_from), dict(nothing.merged_into), nothing.moved) == ({}, {}, ())
    empty = VocabularyCatalog(VocabularyReader(BehaviorKindStore(tmp_path / "empty")))
    blank = empty.changes_since(0)
    assert blank.version == 0 and blank.moved == ()
