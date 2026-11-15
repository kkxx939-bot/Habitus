"""词表口的实现（``runtime.scene_vocabulary``）：在用状态读成语义树要的在用的词与修订号（词表方案 2026-11-14）。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from habitus.behavior.kinds.ids import ClassId, Lane
from habitus.behavior.kinds.store import BehaviorKindStore
from habitus.runtime.scene_vocabulary import VocabularyCatalog
from habitus.scene.concepts.catalog import ClassCatalog


def test_the_adapter_implements_the_catalog_protocol(tmp_path: Path) -> None:
    assert isinstance(VocabularyCatalog(BehaviorKindStore(tmp_path, lane=Lane.SESSION)), ClassCatalog)


def test_only_words_in_use_are_handed_over_with_the_state_revision(tmp_path: Path) -> None:
    store = BehaviorKindStore(tmp_path, lane=Lane.SESSION)
    catalog = VocabularyCatalog(store)
    fresh = catalog.current()
    assert fresh.version == 0 and fresh.words == ()
    vocabulary = store.read()
    word = ClassId(Lane.SESSION, 2)
    store.replace(vocabulary.with_active({word: date(2026, 4, 16)}), expected_revision=0)
    current = catalog.current()
    (only,) = current.words
    assert current.version == 1
    assert (only.id, only.name, only.lane) == ("s-k0002", vocabulary.get(word).name, "session")
    assert only.definition == vocabulary.get(word).definition
