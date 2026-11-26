"""关系卡片（裁定 115）：关系表里跨代带着的与退休的每一行一张；行没了，成立过的留着标结束、只当过候选的删；退休行保持旧读数；
重新当候选时"以前成立过"不抹掉；字段没变不写；文件名是关系的纯函数；换代里这一步尽力而为、读不到表一张都不删。"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from habitus.behavior.kinds.ids import ClassId, Lane
from habitus.behavior.kinds.model import Vocabulary, Word
from habitus.runtime.generation_result import (
    DEGRADED,
    DONE,
    FALLBACK_RELATION_CARDS,
    RELATION_CARDS,
    SKIPPED,
)
from habitus.runtime.record_files import RecordFileError, StoredRecord
from habitus.runtime.relation_cards import RelationCards
from habitus.scene.relations import stats
from habitus.scene.relations.cards import ENDED, card_location, render_card
from habitus.scene.relations.engine import RelationKey, RelationTest, Verdict
from habitus.scene.relations.spans import Segment
from habitus.scene.relations.state import LaneState, Relation, Status

NOW = datetime(2026, 3, 20, 12, tzinfo=UTC)
NIGHT = date(2026, 3, 20)
EAT, WALK, READ = "s-k0001", "s-k0002", "s-k0003"
NAMES = {EAT: "吃饭", WALK: "散步", READ: "看书"}
A_TO_B = RelationKey("session", EAT, WALK, Segment.CHAIN)
B_TO_C = RelationKey("session", WALK, READ, Segment.REST_OF_DAY)
WEDNESDAY = RelationKey("session", "周三", READ, Segment.SITUATION)


def reading(treated: float = 0.9, control: float = 0.15, antecedents: int = 12) -> RelationTest:
    return RelationTest(
        key=A_TO_B,
        verdict=Verdict.SIGNIFICANT,
        antecedents=antecedents,
        blocks=6,
        block_days=1.0,
        effect=stats.Effect(10, treated, control),
        p_value=0.001,
        minimum_p=0.0001,
        upward=True,
        interval=(0.4, 0.9),
    )


def row(key: RelationKey, status: Status, **dates: date | None) -> Relation:
    if status is Status.RETIRED:
        return Relation(key=key, status=status, upward=True, retired_on=dates.get("retired_on", NIGHT))
    found = {"discovered_on": date(2026, 3, 1), **dates}
    return Relation(key=key, status=status, upward=True, tonight=reading(), **found)  # type: ignore[arg-type]


def table(*rows: Relation, night: date = NIGHT) -> LaneState:
    return LaneState(lane="session", night=night, family=10, relations=rows)


def cards(tmp_path) -> RelationCards:  # type: ignore[no-untyped-def]
    return RelationCards(tmp_path / "habitus", lane="session", max_children=100, max_bytes=64_000)


def maybe_card(store: RelationCards, key: RelationKey) -> StoredRecord | None:
    return store.read(*card_location(key))


def card(store: RelationCards, key: RelationKey) -> StoredRecord:
    found = maybe_card(store, key)
    assert found is not None
    return found


def test_carried_and_retired_rows_get_cards_and_tested_rows_do_not(tmp_path) -> None:
    store = cards(tmp_path)
    tested = Relation(key=RelationKey("session", READ, EAT, Segment.CHAIN), status=Status.TESTED, upward=True, tonight=reading())
    established = row(A_TO_B, Status.ESTABLISHED, established_on=date(2026, 3, 10))
    synced = store.sync(
        table(established, row(B_TO_C, Status.CANDIDATE), row(WEDNESDAY, Status.REJECTED, rejected_on=NIGHT), tested),
        NAMES,
        now=NOW,
    )
    assert (synced.written, synced.deleted, synced.ended) == (3, 0, 0)
    assert sorted(path.relative_to(store.directory).as_posix() for path in store.directory.rglob("*.md")) == [
        "s-k0001/s-k0002--chain.md",
        "s-k0002/s-k0003--rest_of_day.md",
        "周三/s-k0003--situation.md",
    ]
    fields = card(store, A_TO_B).fields
    assert fields["status"] == "established" and fields["established_on"] == "2026-03-10"
    assert (fields["antecedent_name"], fields["consequent_name"]) == ("吃饭", "散步")
    assert (fields["antecedents"], fields["units"]) == (12, 10)
    assert (fields["treated_rate"], fields["control_rate"], fields["difference"]) == (0.9, 0.15, 0.75)
    # 指向：后果那个词的关系目录（顺着就是链）与两张习惯卡片；日历前因没有习惯卡片
    assert fields["next"] == "habitus://relations/session/s-k0002"
    assert fields["antecedent_habit"] == "habitus://habits/session/s-k0001.md"
    situation = card(store, WEDNESDAY).fields
    assert situation["antecedent_habit"] is None and situation["antecedent_name"] == "周三"
    body = (store.directory / "s-k0001" / "s-k0002--chain.md").read_text(encoding="utf-8")
    assert "做完「吃饭」之后，紧接着更常做「散步」。" in body and "已经成立" in body


def test_the_same_table_twice_rewrites_nothing(tmp_path) -> None:
    store = cards(tmp_path)
    rows = table(row(A_TO_B, Status.ESTABLISHED, established_on=date(2026, 3, 10)), row(B_TO_C, Status.CANDIDATE))
    store.sync(rows, NAMES, now=NOW)
    path = store.directory / "s-k0001" / "s-k0002--chain.md"
    before = path.stat().st_mtime_ns, card(store, A_TO_B).metadata.revision
    synced = store.sync(rows, NAMES, now=NOW + timedelta(days=1))
    assert (synced.written, synced.deleted) == (0, 0)
    assert (path.stat().st_mtime_ns, card(store, A_TO_B).metadata.revision) == before


def test_a_cleared_relation_stays_if_it_was_ever_established_and_goes_if_it_was_only_a_candidate(tmp_path) -> None:
    store = cards(tmp_path)
    store.sync(table(row(A_TO_B, Status.ESTABLISHED, established_on=date(2026, 3, 10)), row(B_TO_C, Status.CANDIDATE)), NAMES, now=NOW)
    later = NIGHT + timedelta(days=1)
    synced = store.sync(table(night=later), NAMES, now=NOW)
    assert (synced.deleted, synced.ended) == (1, 1)
    assert maybe_card(store, B_TO_C) is None and not (store.directory / WALK).exists()  # 前因的卡片删光了，目录也删
    fields = card(store, A_TO_B).fields
    assert (fields["status"], fields["ended_on"], fields["established_on"]) == (ENDED, later.isoformat(), "2026-03-10")
    assert fields["treated_rate"] == 0.9  # 读数保持最后一次的
    # 之后的每一代都不再动它
    assert store.sync(table(night=later + timedelta(days=1)), NAMES, now=NOW).written == 0
    assert card(store, A_TO_B).fields["ended_on"] == later.isoformat()


def test_an_expired_relation_keeps_its_expiry_day_when_it_is_cleared(tmp_path) -> None:
    store = cards(tmp_path)
    expired = row(A_TO_B, Status.EXPIRED, established_on=date(2026, 3, 10), expired_on=date(2026, 3, 18))
    store.sync(table(expired), NAMES, now=NOW)
    assert card(store, A_TO_B).fields["ended_on"] == "2026-03-18"
    store.sync(table(night=NIGHT + timedelta(days=2)), NAMES, now=NOW)
    assert (card(store, A_TO_B).fields["status"], card(store, A_TO_B).fields["ended_on"]) == (ENDED, "2026-03-18")


def test_a_retired_row_keeps_the_old_readings_and_a_first_seen_retired_row_has_none(tmp_path) -> None:
    store = cards(tmp_path)
    store.sync(table(row(A_TO_B, Status.REJECTED, rejected_on=date(2026, 3, 12))), NAMES, now=NOW)
    retired_on = NIGHT + timedelta(days=1)
    store.sync(table(row(A_TO_B, Status.RETIRED, retired_on=retired_on), row(B_TO_C, Status.RETIRED), night=retired_on), NAMES, now=NOW)
    kept = card(store, A_TO_B).fields
    assert (kept["status"], kept["retired_on"], kept["rejected_on"]) == ("retired", retired_on.isoformat(), "2026-03-12")
    assert (kept["treated_rate"], kept["antecedents"]) == (0.9, 12)
    fresh = card(store, B_TO_C).fields
    assert fresh["status"] == "retired" and fresh["antecedents"] is None and fresh["discovered_on"] is None
    assert "读数：没有" in render_card(fresh)


def test_a_relation_that_comes_back_as_a_candidate_keeps_its_established_history(tmp_path) -> None:
    store = cards(tmp_path)
    store.sync(table(row(A_TO_B, Status.EXPIRED, established_on=date(2026, 3, 10), expired_on=date(2026, 3, 15))), NAMES, now=NOW)
    # 失效之后久没来退休，再后来重新当候选：表里是新建的一行，旧日期都没了
    store.sync(table(row(A_TO_B, Status.RETIRED, retired_on=date(2026, 3, 22)), night=date(2026, 3, 22)), NAMES, now=NOW)
    back = row(A_TO_B, Status.CANDIDATE, discovered_on=date(2026, 4, 2))
    store.sync(table(back, night=date(2026, 4, 2)), NAMES, now=NOW)
    fields = card(store, A_TO_B).fields
    assert fields["status"] == "candidate" and fields["established_on"] is None and fields["retired_on"] is None
    assert fields["past"] == [{"established_on": "2026-03-10", "ended_on": "2026-03-15"}]
    assert "以前成立过：2026-03-10 起，到 2026-03-15" in render_card(fields)
    # 这一段还只是候选，行又被清掉：因为以前成立过，卡片留着
    store.sync(table(night=date(2026, 4, 3)), NAMES, now=NOW)
    assert card(store, A_TO_B).fields["status"] == ENDED
    assert card(store, A_TO_B).fields["past"] == [{"established_on": "2026-03-10", "ended_on": "2026-03-15"}]


def test_file_names_are_legal_distinct_and_stable_for_awkward_conditions(tmp_path) -> None:
    conditions = ("field:place=家:书房", "field:place=a|b", "field:place=公司 ", "field:place=Home", "field:place=home")
    keys = [RelationKey("session", EAT, WALK, Segment.CHAIN, text) for text in conditions]
    names = [card_location(key) for key in keys]
    assert len(set(names)) == len(keys) and len({name.casefold() for _, name in names}) == len(keys)
    assert names == [card_location(key) for key in keys]
    store = cards(tmp_path)
    store.sync(table(*(row(key, Status.CANDIDATE) for key in keys)), NAMES, now=NOW)
    assert len(list((store.directory / EAT).glob("*.md"))) == len(keys)
    assert {card(store, key).fields["condition"] for key in keys} == set(conditions)


def test_two_relations_landing_on_one_file_fail_instead_of_overwriting(tmp_path, monkeypatch) -> None:
    """条件摘要是截断的 sha256：真撞上时两条关系会落到同一个文件——这一遍明确失败，一张都不盖。"""

    import habitus.runtime.relation_cards as module

    store = cards(tmp_path)
    monkeypatch.setattr(module, "card_location", lambda key: (key.antecedent, "same"))
    first = RelationKey("session", EAT, WALK, Segment.CHAIN, "field:place=家")
    second = RelationKey("session", EAT, WALK, Segment.CHAIN, "field:place=公司")
    with pytest.raises(RecordFileError, match="two relations map to"):
        store.sync(table(row(first, Status.CANDIDATE), row(second, Status.CANDIDATE)), NAMES, now=NOW)


def test_a_card_whose_name_does_not_match_its_relation_is_refused(tmp_path) -> None:
    store = cards(tmp_path)
    store.sync(table(row(A_TO_B, Status.CANDIDATE)), NAMES, now=NOW)
    path = store.directory / EAT / "s-k0002--chain.md"
    path.rename(path.with_name("s-k0002--next_day.md"))
    with pytest.raises(RecordFileError, match="not named after its relation"):
        store.sync(table(row(A_TO_B, Status.CANDIDATE)), NAMES, now=NOW)


# ── 换代里这一步 ─────────────────────────────────────────────────────────


def _vocabulary() -> Vocabulary:
    words = tuple(Word(ClassId(Lane.SESSION, number), 1, name, None, name, "测试") for number, name in ((1, "吃饭"), (2, "散步")))
    return Vocabulary(Lane.SESSION, words, {}, revision=1)


def _run(tmp_path):  # type: ignore[no-untyped-def]
    """一个模型熔断开着的换代（语义树不写新的关系表），盘上已经有一张前天的关系表。"""

    from tests.unit.runtime.test_generation import TODAY, assembled

    store = cards(tmp_path)
    _site, run = assembled(tmp_path, relation_cards=store, models_down=lambda: True)
    run.vocabulary.store = type("Store", (), {"read": lambda self: _vocabulary()})()  # type: ignore[attr-defined]
    night = TODAY - timedelta(days=2)
    rows = LaneState(lane="session", night=night, family=1, relations=(row(A_TO_B, Status.ESTABLISHED, established_on=night),))
    run.scene.relations.write(rows, ())
    return run, store, night


def test_the_generation_writes_the_cards_from_the_latest_table_after_the_habits(tmp_path) -> None:
    from tests.unit.runtime.test_generation import generate

    run, store, night = _run(tmp_path)
    result = generate(run)
    step = result.step(RELATION_CARDS)
    assert step is not None and step.outcome == DONE and step.report["relation_night"] == night.isoformat()
    assert card(store, A_TO_B).fields["status"] == "established"
    # 熔断开着的下一代读到的还是这张表：卡片一张都不重写、不报错
    run.marks.path(run.today()).unlink()
    again = generate(run)
    assert again.step(RELATION_CARDS).outcome == DONE and again.step(RELATION_CARDS).report["written"] == 0  # type: ignore[union-attr]


def test_an_unreadable_table_deletes_no_card_and_degrades_the_step(tmp_path) -> None:
    from tests.unit.runtime.test_generation import generate

    run, store, night = _run(tmp_path)
    store.sync(table(row(B_TO_C, Status.CANDIDATE)), NAMES, now=NOW)  # 一张只当过候选的：表若被当成空的，它会被删
    path = run.scene.relations.directory / "session" / f"{night.isoformat()}.json"
    path.write_bytes(path.read_bytes() + b" ")
    result = generate(run)
    step = result.step(RELATION_CARDS)
    assert step is not None and step.outcome == DEGRADED and result.fallbacks[FALLBACK_RELATION_CARDS] == 1
    assert result.complete and maybe_card(store, B_TO_C) is not None


def test_without_any_table_the_step_is_skipped_and_nothing_is_created(tmp_path) -> None:
    from tests.unit.runtime.test_generation import assembled, generate

    store = cards(tmp_path)
    _site, run = assembled(tmp_path, relation_cards=store, models_down=lambda: True)
    result = generate(run)
    assert result.step(RELATION_CARDS).outcome == SKIPPED and not store.directory.exists()  # type: ignore[union-attr]
    assert result.fallbacks[FALLBACK_RELATION_CARDS] == 0
