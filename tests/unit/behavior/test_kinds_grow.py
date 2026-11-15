"""长新节点与生命周期的几处细节（阶段 C、D 复核补的测试）：重试补回的旧一轮不倒拨"最后被选中"、读状态文件按配置查名字长度、
同一代里被删节点的日子帮上级凑够两天、换代读数进指标。"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from habitus.behavior.kinds.active import count_active
from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.kinds.grow import Proposal, settle
from habitus.behavior.kinds.ids import ClassId, Lane
from habitus.behavior.kinds.model import Vocabulary, Word, WordOrigin
from habitus.behavior.kinds.store import BehaviorKindStore, BehaviorKindStoreError
from habitus.behavior.kinds.words import initial_words
from habitus.foundation.observability import ObservationEvent, ObservationStatus
from habitus.infrastructure.observability.metrics import project_metric_updates
from habitus.infrastructure.store.sqlite import SQLiteLockStore

CST = timezone(timedelta(hours=8))
WRITE_DOC = ClassId(Lane.SESSION, 1)
CONFIG = BehaviorKindConfig()


def at(month: int, day: int, hour: int = 9) -> datetime:
    return datetime(2026, month, day, hour, tzinfo=CST)


def grown(vocabulary: Vocabulary, name: str, parent: ClassId, moment: datetime) -> tuple[Vocabulary, ClassId]:
    """在 ``parent`` 下面长一个叫 ``name`` 的节点（记录挂在它上面）。"""

    return settle(vocabulary, parent, (Proposal(name, f"{name}这一种"),), at=moment, config=CONFIG)


def test_a_late_replay_of_an_earlier_turn_does_not_rewind_the_last_pick() -> None:
    vocabulary, x = grown(initial_words(Lane.SESSION), "写需求书", WRITE_DOC, at(4, 13))
    vocabulary, again = settle(vocabulary, x, (), at=at(4, 30), config=CONFIG)
    assert again == x and vocabulary.get(x).last_picked == at(4, 30)
    # 04-14 那一轮的选词在 05-04 才由重试堆补回、又复用了 X：最后被选中仍是 04-30
    replayed, same = grown(vocabulary, "写需求书", WRITE_DOC, at(4, 14))
    assert same == x and replayed.get(x).last_picked == at(4, 30)
    counted = count_active(replayed, (), today=date(2026, 5, 4), min_days=2, cap=100, retire_days=20)
    assert x in counted.vocabulary and counted.deleted == ()


def test_reading_a_state_file_checks_the_name_length_against_the_config(tmp_path: Path) -> None:
    store = BehaviorKindStore(
        tmp_path, lane=Lane.SESSION, lock_store=SQLiteLockStore(tmp_path / "locks.sqlite3"), lock_ttl_seconds=30
    )
    store.update(lambda vocabulary: grown(vocabulary, "写需求书文档", WRITE_DOC, at(4, 13))[0])
    assert store.read().named("写需求书文档") is not None
    state = tmp_path / "kinds.state.json"
    raw = json.loads(state.read_text(encoding="utf-8"))
    raw["grown"][0]["name"] = "写一份需求书文档"
    state.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(BehaviorKindStoreError, match="kinds.state.json.*longer than 6"):
        store.read()


def test_a_child_deleted_this_generation_lends_its_days_to_its_parent_in_the_same_generation() -> None:
    vocabulary, x = grown(initial_words(Lane.SESSION), "写需求书", WRITE_DOC, at(4, 1))
    y = vocabulary.next_id()
    vocabulary = vocabulary.with_word(
        Word(y, 3, "写验收段", x, "写验收段这一种", "文档", WordOrigin.GROWN, created_at=at(4, 1), last_picked=at(4, 2))
    )
    # X 自己最近还被选过（不到 20 天），Y 的两天都在 20 天前：这一代 Y 删掉，它的两天换算到 X 上，X 同一代转在用
    vocabulary = vocabulary.with_word(Word(**{**vars(vocabulary.get(x)), "last_picked": at(4, 20)}))
    picks = [(str(y), date(2026, 4, 1)), (str(y), date(2026, 4, 2))]
    counted = count_active(vocabulary, picks, today=date(2026, 4, 25), min_days=2, cap=100, retire_days=20)
    assert counted.deleted == (y,) and counted.added == (x,)
    assert dict(counted.vocabulary.active) == {x: date(2026, 4, 25)}
    assert counted.vocabulary.revision == vocabulary.revision + 1


def test_the_generation_word_counts_become_metrics() -> None:
    event = ObservationEvent(
        category="generation",
        operation="run",
        status=ObservationStatus.SUCCESS,
        duration_seconds=1.0,
        attributes={
            "outcome": "complete",
            "duration_seconds": 1.0,
            "active_words": 3,
            "waiting_words": 0,
            "deleted_words": 2,
            "exited_words": 1,
            "candidate_words": 4,
        },
    )
    gauges = {update.name: update.value for update in project_metric_updates(event) if update.kind == "gauge"}
    assert {name: gauges[name] for name in gauges if name.endswith("_words")} == {
        "generation_active_words": 3.0,
        "generation_waiting_words": 0.0,
        "generation_deleted_words": 2.0,
        "generation_exited_words": 1.0,
        "generation_candidate_words": 4.0,
    }
