"""基础词表的值对象、初始词表、在用状态存储与计数（词表方案 2026-11-14）。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from habitus.behavior.kinds.active import count_active
from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.kinds.ids import (
    ClassId,
    KindIdError,
    Lane,
    continued_token,
    is_word,
    lane_of_token,
    missing_token,
    not_event_token,
    pending_token,
)
from habitus.behavior.kinds.model import BehaviorKindError, Vocabulary, Word, reminder_sentence
from habitus.behavior.kinds.store import (
    LEGACY_FILENAMES,
    BehaviorKindConflictError,
    BehaviorKindStore,
    BehaviorKindStoreError,
)
from habitus.behavior.kinds.words import initial_words, parse_words

S1, S2, S3 = (ClassId(Lane.SESSION, number) for number in (1, 2, 3))


def _words() -> Vocabulary:
    return Vocabulary(
        Lane.SESSION,
        (
            Word(S1, 1, "写文档", None, "新写一份文档", "文档"),
            Word(S2, 2, "写周报", S1, "写周期性的工作汇报", "文档"),
            Word(S3, 1, "查数据", None, "查一份数据", "数据"),
        ),
    )


def test_ids_round_trip_and_the_four_markers_belong_to_a_lane() -> None:
    assert ClassId.parse("s-k0007") == ClassId(Lane.SESSION, 7)
    with pytest.raises(KindIdError):
        ClassId.parse("s-k00007")
    for token in (pending_token, not_event_token, continued_token, missing_token):
        marker = token(Lane.PHYSICAL)
        assert not is_word(marker) and lane_of_token(marker) is Lane.PHYSICAL
    assert is_word("p-k0003")
    with pytest.raises(KindIdError):
        lane_of_token("s-迁移")


def test_the_packaged_initial_words_match_the_two_tables() -> None:
    session, physical = initial_words(Lane.SESSION), initial_words(Lane.PHYSICAL)
    assert len(session.words) == 279 and len(physical.words) == 223
    assert session.revision == 0 and not session.active
    assert [word.level for word in physical.words].count(3) == 12


def test_a_word_table_must_hang_together() -> None:
    with pytest.raises(BehaviorKindError, match="unknown parent"):
        Vocabulary(Lane.SESSION, (Word(S2, 2, "写周报", S1, "写周报", "文档"),))
    with pytest.raises(BehaviorKindError, match="does not follow"):
        Vocabulary(
            Lane.SESSION, (Word(S1, 1, "写文档", None, "写", "文档"), Word(S2, 3, "写周报", S1, "写", "文档"))
        )
    with pytest.raises(BehaviorKindError, match="name repeats"):
        Vocabulary(Lane.SESSION, (Word(S1, 1, "写文档", None, "写", "文档"), Word(S3, 1, "写文档", None, "写", "文档")))
    with pytest.raises(BehaviorKindError, match="only level-1"):
        Word(S1, 2, "写文档", None, "写", "文档")
    assert reminder_sentence(_words().get(S2)) == "该写周报了"


def test_the_data_file_decoder_names_the_file_on_every_error() -> None:
    body = {"schema_version": "behavior_initial_words_v1", "lane": "session", "words": []}
    with pytest.raises(BehaviorKindError, match="here.json"):
        parse_words(json.dumps(body), lane=Lane.SESSION, source="here.json")
    body["words"] = [{"id": "s-k0001", "level": "1", "name": "写", "parent": None, "definition": "写", "domain": "文档"}]
    with pytest.raises(BehaviorKindError, match="here.json.*level must be an integer"):
        parse_words(json.dumps(body), lane=Lane.SESSION, source="here.json")


def test_counting_needs_two_distinct_days_and_waits_when_full() -> None:
    words = _words()
    picks = [
        ("s-k0001", date(2026, 4, 13)),
        ("s-k0001", date(2026, 4, 13)),
        ("s-k0002", date(2026, 4, 13)),
        ("s-k0002", date(2026, 4, 15)),
        ("s-k0003", date(2026, 4, 14)),
        ("s-k0003", date(2026, 4, 15)),
        ("s-续话", date(2026, 4, 16)),
    ]
    counted = count_active(words, picks, today=date(2026, 4, 16), min_days=2, cap=1)
    # 「写文档」只在一天里被选：不在用；另两个的"第二个日子"都是 04-15，同一天按编号，写周报（s-k0002）先进
    assert dict(counted.vocabulary.active) == {S2: date(2026, 4, 16)}
    assert counted.waiting == (S3,) and counted.vocabulary.revision == 1
    again = count_active(counted.vocabulary, picks, today=date(2026, 4, 17), min_days=2, cap=1)
    assert again.vocabulary is counted.vocabulary  # 集合没变：修订号不动


def test_the_state_store_is_strict_and_compare_and_set(tmp_path: Path) -> None:
    store = BehaviorKindStore(tmp_path, lane=Lane.SESSION, config=BehaviorKindConfig())
    fresh = store.read()
    assert fresh.revision == 0
    updated = store.replace(fresh.with_active({ClassId(Lane.SESSION, 2): date(2026, 4, 16)}), expected_revision=0)
    assert store.read() == updated and (tmp_path / "kinds.md").read_text(encoding="utf-8").count("起在用") == 1
    with pytest.raises(BehaviorKindConflictError):
        store.replace(updated.with_active({}), expected_revision=0)
    state = tmp_path / "kinds.state.json"
    raw = json.loads(state.read_text(encoding="utf-8"))
    for broken in ({**raw, "extra": 1}, {**raw, "active": [{"id": "s-k9999", "since": "2026-04-16"}]}):
        state.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(BehaviorKindStoreError, match="kinds.state.json"):
            store.read()
    state.write_text(state.read_text(encoding="utf-8")[:20], encoding="utf-8")
    with pytest.raises(BehaviorKindStoreError, match="kinds.state.json"):
        store.read()


@pytest.mark.parametrize("name", LEGACY_FILENAMES)
def test_an_old_vocabulary_file_on_disk_is_refused_by_name(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_text("{}", encoding="utf-8")
    with pytest.raises(BehaviorKindStoreError, match=name):
        BehaviorKindStore(tmp_path, lane=Lane.SESSION).check()
