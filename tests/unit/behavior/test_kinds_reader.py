"""对外只读：编号查条目、顺着去向找现在的编号、某版之后的变更。"""

from __future__ import annotations

from pathlib import Path

import pytest

from habitus.behavior.kinds.changes import AddClass, ChangeReason, VersionRecord, merge_operations
from habitus.behavior.kinds.reader import VocabularyReader
from habitus.behavior.kinds.store import BehaviorKindStore
from tests.unit.behavior.kinds_fixtures import EDIT, NOW, RESEARCH, edit_class, research_class


def test_reader_describes_ids_and_follows_merges(tmp_path: Path) -> None:
    store = BehaviorKindStore(tmp_path)
    store.append(
        VersionRecord(1, NOW, ChangeReason.NIGHTLY, (AddClass(edit_class()), AddClass(research_class()))),
        expected_version=0,
    )
    store.append(VersionRecord(2, NOW, ChangeReason.REVISION, merge_operations(RESEARCH, EDIT)), expected_version=1)
    reader = VocabularyReader(store)
    assert reader.version() == 2
    described = reader.describe("s-k0002")
    assert described is not None and described.name == "调研" and not described.active
    assert reader.current_ids("s-k0002") == ("s-k0001",)
    assert reader.describe("s-待定") is None and reader.current_ids("s-非事件") == ()


def test_changes_since_returns_the_later_versions_in_order(tmp_path: Path) -> None:
    """派生树（语义树）只读上次同步之后的那几版；版本号不合法硬失败。"""

    store = BehaviorKindStore(tmp_path)
    store.append(VersionRecord(1, NOW, ChangeReason.NIGHTLY, (AddClass(edit_class()),)), expected_version=0)
    store.append(VersionRecord(2, NOW, ChangeReason.NIGHTLY, (AddClass(research_class()),)), expected_version=1)
    reader = VocabularyReader(store)
    assert [record.version for record in reader.changes_since(0)] == [1, 2]
    assert [record.version for record in reader.changes_since(1)] == [2]
    assert reader.changes_since(2) == ()
    for bad in (-1, True, "1"):
        with pytest.raises(ValueError, match="non-negative integer"):
            reader.changes_since(bad)  # type: ignore[arg-type]
