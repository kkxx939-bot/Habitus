"""关系的状态逐晚折叠与关系表落盘（语义树新方案 ``13`` ③④，第 5 步）。

合成数据见 ``relation_fixtures``：一晚一晚往前截序列（与回放同一个做法），每晚接上前一晚折叠。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import pytest

from habitus.scene.concepts import ConceptSet
from habitus.scene.relations import (
    LaneState,
    LaneTests,
    RelationKey,
    Segment,
    StateError,
    Status,
    Transition,
    build_timelines,
    fold,
)
from habitus.scene.relations.document import RelationRecordError, decode_night, encode_night
from habitus.scene.relations.store import RelationStore, RelationStoreError
from habitus.series import EventSeries
from tests.unit.scene.concept_fixtures import base
from tests.unit.scene.relation_fixtures import (
    CODE,
    CONCEPTS,
    CONFIG,
    DISCUSS,
    TITLES,
    key,
    planted,
    record,
    workdays,
)

PLANTED = key(DISCUSS, CODE)


def nights_of(
    series: EventSeries, days: int, *, concepts: ConceptSet = CONCEPTS
) -> list[tuple[LaneState, tuple[Transition, ...]]]:
    """从第 6 个工作日起每晚折叠一次（截止日 = 那天的次日）。"""

    previous: LaneState | None = None
    found = []
    for day in workdays(days)[5:]:
        tonight = series.until(day + timedelta(days=1))
        line = build_timelines(tonight, concepts, {}, CONFIG)["session"]
        state, transitions = fold(previous, LaneTests(line, concepts, CONFIG, tonight.cutoff))
        found.append((state, transitions))
        previous = state
    return found


def history(
    nights: list[tuple[LaneState, tuple[Transition, ...]]], relation: RelationKey
) -> list[tuple[Status | None, Status]]:
    return [(item.before, item.after) for _state, moved in nights for item in moved if item.key == relation]


def status_of(state: LaneState, relation: RelationKey) -> Status:
    return next(item.status for item in state.relations if item.key == relation)


#: 这几条考"干净路径"的测试用的种子：同一条链配两道对照之后（2026-10-12），默认种子 7 的世界里这条关系在前向验证攒到第 6 次新数据时
#: p = 0.11 被退回一次、再在新数据上重新发现才成立（引擎按设计同一晚只判一次）；那条"退回再回来"的路径见下面单独一条测试。
CLEAN_SEED = 17


def test_a_real_relation_becomes_a_candidate_then_is_established_after_forward_validation() -> None:
    nights = nights_of(planted(60, seed=CLEAN_SEED), 60)
    assert history(nights, PLANTED) == [(Status.TESTED, Status.CANDIDATE), (Status.CANDIDATE, Status.ESTABLISHED)]
    final = nights[-1][0]
    (established,) = final.of(Status.ESTABLISHED)
    assert established.key == PLANTED and established.upward
    assert established.discovered_on is not None and established.established_on is not None
    assert established.discovered_on < established.forward_passed_on <= established.established_on  # type: ignore[operator]
    assert established.maintenance is not None and established.maintenance.antecedents > 0
    # 维持检验每段攒几次：成立那晚按"回落到对照率"的功效算好、存下（26 号清单 5.3）
    assert (
        established.maintenance_needed is not None
        and 1 <= established.maintenance_needed <= CONFIG.thresholds.forward_max_antecedents
    )
    # 成立之后不再进当晚的检验族
    assert established.tonight is not None and established.tonight.verdict.value == "tested"


def test_a_relation_that_stops_expires_and_does_not_flap_back_on_old_data() -> None:
    """埋 25 天后撤掉：成立 → 一段新数据说没了（再看下一段）→ 第二段也说没了才失效；之后只有失效那晚之后的数据算数，
    旧数据里那段强关系不会让它每周"又被发现"一次。"""

    nights = nights_of(planted(80, seed=CLEAN_SEED, linked=(0, 25)), 80)
    assert history(nights, PLANTED) == [
        (Status.TESTED, Status.CANDIDATE),
        (Status.CANDIDATE, Status.ESTABLISHED),
        (Status.ESTABLISHED, Status.ESTABLISHED),
        (Status.ESTABLISHED, Status.EXPIRED),
    ]
    expired = next(item for item in nights[-1][0].relations if item.key == PLANTED)
    assert expired.status is Status.EXPIRED and expired.reset_on == expired.expired_on
    # 失效之后的读数只看失效那晚以后的数据：A 之后几乎从不接修改代码
    assert expired.tonight is not None and expired.tonight.effect.treated_rate < 0.4


def test_a_candidate_that_does_not_replicate_is_remembered_as_rejected() -> None:
    nights = nights_of(planted(30, seed=CLEAN_SEED, linked=(0, 6)), 30)
    assert history(nights, PLANTED) == [(Status.TESTED, Status.CANDIDATE), (Status.CANDIDATE, Status.REJECTED)]
    rejected = next(item for item in nights[-1][0].relations if item.key == PLANTED)
    assert rejected.rejected_on is not None and rejected.reset_on == rejected.rejected_on
    reason = next(item.reason for _state, moved in nights for item in moved if item.after is Status.REJECTED)
    assert "前向验证没复现" in reason


def test_a_rejected_candidate_comes_back_only_on_new_data_and_can_still_be_established() -> None:
    """种子 21 的世界：发现之后 6 次新数据 p = 0.139（全中本可到 α，是真没复现）→ 退回；退回那晚之后的新数据上又过第 1 道
    → 重新候选（要攒更多次）→ 成立。退回不是终点，但回来只能靠新数据。（种子 7 原先在 p = 0.11 退回，那是每层对照太少、
    全中也到不了 α 的必输局，L10.15 之后不判、接着攒，直接成立。）"""

    nights = nights_of(planted(60, seed=21), 60)
    assert history(nights, PLANTED) == [
        (Status.TESTED, Status.CANDIDATE),
        (Status.CANDIDATE, Status.REJECTED),
        (Status.REJECTED, Status.CANDIDATE),
        (Status.CANDIDATE, Status.ESTABLISHED),
    ]
    rejected_night = next(state.night for state, moved in nights for item in moved if item.after is Status.REJECTED)
    final = next(item for item in nights[-1][0].relations if item.key == PLANTED)
    assert (
        final.status is Status.ESTABLISHED and final.discovered_on is not None and final.discovered_on > rejected_night
    )


def test_a_forward_validation_that_could_not_reach_alpha_waits_instead_of_rejecting() -> None:
    """种子 7 的世界：攒够算出的次数那晚，每层对照太少、全中也到不了 α（原先在 p = 0.11 退回）。L10.15 之后不判、接着攒，
    下一批新数据上复现、直接成立，不再白白退回一次。"""

    nights = nights_of(planted(60, seed=7), 60)
    assert history(nights, PLANTED) == [(Status.TESTED, Status.CANDIDATE), (Status.CANDIDATE, Status.ESTABLISHED)]


def test_forward_validation_waits_until_the_new_data_spans_more_than_one_block() -> None:
    """发现之后第一天就来了 6 次讨论方案：次数够了，但都在同一个块里（同一天的 6 次不是 6 个独立样本），继续等。"""

    series = planted(60)
    nights = nights_of(series, 60)
    discovered = next(state.night for state, moved in nights for item in moved if item.after is Status.CANDIDATE)
    before = next(state for state, _moved in nights if state.night == discovered)
    burst = tuple(record(discovered, hour, 0, "讨论方案", number=100 + hour) for hour in range(9, 15))
    tonight = EventSeries(
        cutoff=discovered + timedelta(days=1),
        records=tuple(
            sorted((*series.until(discovered).records, *burst), key=lambda item: (item.started_at, item.uri))
        ),
    )
    line = build_timelines(tonight, CONCEPTS, {}, CONFIG)["session"]
    state, moved = fold(before, LaneTests(line, CONCEPTS, CONFIG, tonight.cutoff))
    waiting = next(item for item in state.relations if item.key == PLANTED)
    assert waiting.status is Status.CANDIDATE and waiting.forward is not None
    assert waiting.forward_needed is not None and waiting.forward.antecedents >= waiting.forward_needed
    assert waiting.forward.blocks == 1  # 次数够了、块数不够：一天里的几次不是几个独立样本
    assert all(item.key != PLANTED for item in moved)


@pytest.mark.parametrize("seed", [3, 7])
def test_nothing_is_established_where_nothing_was_planted(seed: int) -> None:
    nights = nights_of(planted(60, seed=seed, follow=0.0), 60)
    assert all(not state.of(Status.ESTABLISHED) for state, _moved in nights)


def test_folding_runs_forward_only_and_carries_relations_it_cannot_test() -> None:
    series = planted(60, seed=CLEAN_SEED)
    nights = nights_of(series, 30)
    last = nights[-1][0]
    line = build_timelines(series.until(last.night), CONCEPTS, {}, CONFIG)["session"]
    with pytest.raises(StateError, match="does not come after"):
        fold(last, LaneTests(line, CONCEPTS, CONFIG, last.night))
    # 前因的概念从概念集里没了：带着的关系原样留着（不改状态、不报迁移）
    fewer = ConceptSet([item for item in CONCEPTS.values() if item.name != DISCUSS])
    later = series.until(last.night + timedelta(days=3))
    line = build_timelines(later, fewer, {}, CONFIG)["session"]
    state, moved = fold(last, LaneTests(line, fewer, CONFIG, later.cutoff))
    kept = next(item for item in state.relations if item.key == PLANTED)
    assert kept.status is Status.ESTABLISHED and kept == next(item for item in last.relations if item.key == PLANTED)
    assert all(item.key != PLANTED for item in moved)


def test_a_night_round_trips_through_its_canonical_document() -> None:
    state = nights_of(planted(40, linked=(0, 25)), 40)[-1][0]
    text = encode_night(state)
    assert decode_night(text, lane="session", night=state.night) == state
    with pytest.raises(RelationRecordError, match="canonical"):
        decode_night(text.replace(":", ": ", 1), lane="session", night=state.night)
    with pytest.raises(RelationRecordError, match="does not match"):
        decode_night(text, lane="physical", night=state.night)


def test_the_store_writes_forward_only_and_keeps_an_append_only_transition_log(tmp_path) -> None:
    store = RelationStore(tmp_path / "scene")
    nights = nights_of(planted(30), 30)
    for state, moved in nights:
        store.write(state, moved)
    assert store.lanes() == ("session",)
    assert store.nights("session") == tuple(state.night for state, _moved in nights[-3:])  # 最晚一晚 + 两代备用
    last, last_moved = nights[-1]
    assert store.read("session", last.night) == last
    assert store.previous("session", last.night) == nights[-2][0]
    logged = store.transitions("session")
    # 删掉的那几晚的迁移照样认得出来：删之前记下了摘要
    assert logged == tuple(item for _state, moved in nights for item in moved) and logged
    log = tmp_path / "scene" / "relations" / "session" / "transitions.jsonl"
    before = log.read_bytes()
    # 同一晚重跑、结果相同：不重复记
    store.write(last, last_moved)
    assert log.read_bytes() == before
    # 写更早的一晚：拒绝（不往回改）
    with pytest.raises(RelationStoreError, match="history is not rewritten"):
        store.write(nights[0][0], nights[0][1])
    # 同一晚重跑、结果不同：整批追加、带第几次写，读的人取最后一次；已写的行不动
    different = (
        Transition(night=last.night, key=PLANTED, before=Status.TESTED, after=Status.CANDIDATE, reason="重跑"),
    )
    store.write(last, different)
    assert log.read_bytes().startswith(before)
    assert store.transitions("session")[-1] == different[0]
    assert all(item.reason != "重跑" or item.night == last.night for item in store.transitions("session"))
    # 再重跑一次、这一晚没有迁移了：记空标记
    store.write(last, ())
    assert all(item.night != last.night for item in store.transitions("session"))
    assert store.read("session", last.night) == last


def test_a_batch_whose_night_was_never_written_is_not_mistaken_for_a_pruned_one(tmp_path) -> None:
    """26 号清单 5.8：第 N 晚追加了迁移、写状态文件之前被杀，下一次写的是 N + 1 晚——N 晚永远没有状态文件，也永远没有
    "删了"那一行；之后把更早的晚删掉，也不会把 N 晚那一批当成真的。"""

    store = RelationStore(tmp_path / "scene", fallback_generations=1)
    nights = nights_of(planted(30), 30)
    crashed_state, crashed_moved = next((state, moved) for state, moved in nights[5:] if moved)
    for state, moved in nights:
        if state is crashed_state:
            store._append_transitions(state.lane, state.night, moved, "f" * 64)  # 写到一半被杀
            continue
        store.write(state, moved)
    nights_on_disk = store.nights("session")
    assert len(nights_on_disk) == 2 and crashed_state.night not in nights_on_disk
    logged = store.transitions("session")
    assert not any(item.night == crashed_state.night for item in logged)
    assert logged == tuple(item for state, moved in nights if state is not crashed_state for item in moved)
    with pytest.raises(RelationStoreError, match="fallback_generations"):
        RelationStore(tmp_path / "other", fallback_generations=0)


def test_the_store_refuses_odd_lanes_and_foreign_transitions(tmp_path) -> None:
    store = RelationStore(tmp_path / "scene")
    state = LaneState(lane="session", night=date(2026, 6, 10), family=0, relations=())
    stray = Transition(night=date(2026, 6, 9), key=PLANTED, before=None, after=Status.CANDIDATE, reason="x")
    with pytest.raises(RelationStoreError, match="belong to the night"):
        store.write(state, (stray,))
    with pytest.raises(RelationStoreError):
        store.write(replace(state, lane="../x"), ())
    assert RelationKey("session", DISCUSS, CODE, Segment.CHAIN) == PLANTED


def test_a_retired_concept_takes_its_relations_down_with_a_clear_reason() -> None:
    """词表拆改把后果的类停用了：已成立的关系当晚转失效，原因写"概念停用"——不是"效应没了"，也不留成僵尸（裁定 27 第 9 条）。"""

    series = planted(60)
    nights = nights_of(series, 60)
    previous = nights[-1][0]
    assert status_of(previous, PLANTED) is Status.ESTABLISHED
    retired = ConceptSet(base(title, retired=title == "修改代码") for title in TITLES)
    tonight = series.until(series.cutoff)
    later = EventSeries(cutoff=previous.night + timedelta(days=1), records=tonight.records)
    line = build_timelines(later, retired, {}, CONFIG)["session"]
    state, moved = fold(previous, LaneTests(line, retired, CONFIG, later.cutoff))
    assert status_of(state, PLANTED) is Status.EXPIRED
    (reason,) = [item.reason for item in moved if item.key == PLANTED]
    assert "概念停用" in reason
    assert not any(item.key.consequent == PLANTED.consequent for item in moved if item.after is Status.CANDIDATE)


def test_a_refinement_whose_source_class_was_merged_away_is_no_longer_frozen_as_a_candidate() -> None:
    """26 号清单 5.5：合并把源类停了，挂在它上面的细分概念不会再有新记录——它的关系按停用处理、不进检验族，也不再给作者看；
    原来它永远冻在候选。"""

    from habitus.scene.concepts.model import members_retired
    from habitus.scene.relations import LaneState, Relation
    from tests.unit.scene.concept_fixtures import REWORK

    series = planted(60)
    retired = ConceptSet([*(base(title, retired=title == "修改代码") for title in TITLES), REWORK])
    assert members_retired(retired, REWORK.identity) and not members_retired(retired, base("讨论方案").identity)
    frozen = Relation(
        key=RelationKey("session", base("讨论方案").identity, REWORK.identity, Segment.CHAIN),
        status=Status.CANDIDATE,
        upward=True,
        discovered_on=series.cutoff - timedelta(days=30),
    )
    previous = LaneState(lane="session", night=series.cutoff - timedelta(days=1), family=1, relations=(frozen,))
    line = build_timelines(series, retired, {}, CONFIG)["session"]
    tests = LaneTests(line, retired, CONFIG, series.cutoff)
    assert not any(REWORK.identity in (key.antecedent, key.consequent) for key in tests.keys())
    state, moved = fold(previous, tests)
    assert status_of(state, frozen.key) is Status.EXPIRED
    assert any(item.key == frozen.key and "概念停用" in item.reason for item in moved)


def test_rows_that_can_never_be_used_again_are_cleared_and_quiet_ones_stay_frozen() -> None:
    """26 号清单 5.6、5.7：集中期随变点重算整段没了、带的条件已不在列举与提议里——这一行清掉，清掉成立的写一条迁移；
    前因只是这阵子没有新链的照旧冻着；停用概念的休眠行清掉，别的休眠行连 ``reset_on`` 一起留着。"""

    from habitus.scene.relations import LaneState, Relation

    series = planted(60)
    quiet = base("整理相册")  # 词表里有、这阵子没做过
    retired_title = "修改代码"
    concepts = ConceptSet([*(base(title, retired=title == retired_title) for title in TITLES), quiet])
    night = series.cutoff - timedelta(days=1)
    discover = series.cutoff - timedelta(days=40)

    def row(key: RelationKey, status: Status, **fields: object) -> Relation:
        return Relation(key=key, status=status, upward=True, discovered_on=discover, **fields)  # type: ignore[arg-type]

    burst = row(
        RelationKey("session", DISCUSS, base("调研").identity, Segment.HIGH_STATE),
        Status.ESTABLISHED,
        established_on=discover,
        forward_passed_on=discover,
        maintained_through=discover,
    )
    conditioned = row(
        RelationKey("session", DISCUSS, base("调研").identity, Segment.CHAIN, "timeline:s-k000000"), Status.CANDIDATE
    )
    frozen = row(RelationKey("session", quiet.identity, base("调研").identity, Segment.CHAIN), Status.CANDIDATE)
    gone_dormant = row(
        RelationKey("session", DISCUSS, base(retired_title).identity, Segment.CHAIN),
        Status.EXPIRED,
        expired_on=discover,
    )
    kept_dormant = row(
        RelationKey("session", DISCUSS, base("调研").identity, Segment.NEXT_DAY), Status.REJECTED, rejected_on=discover
    )
    previous = LaneState(
        lane="session", night=night, family=1, relations=(burst, conditioned, frozen, gone_dormant, kept_dormant)
    )
    line = build_timelines(series, concepts, {}, CONFIG)["session"]
    state, moved = fold(previous, LaneTests(line, concepts, CONFIG, series.cutoff))
    kept = {relation.key: relation for relation in state.relations}
    assert burst.key not in kept and conditioned.key not in kept and gone_dormant.key not in kept
    assert kept[frozen.key].status is Status.CANDIDATE
    assert kept_dormant.key in kept and kept[kept_dormant.key].reset_on == discover
    reasons = {item.key: item.reason for item in moved}
    assert "不在了" in reasons[burst.key] and "不在了" in reasons[conditioned.key]
    assert gone_dormant.key not in reasons  # 休眠行清掉不必留迁移


def test_a_transition_batch_without_its_state_file_is_not_read(tmp_path) -> None:
    """先追加迁移、再写状态文件：中途被杀，日志里会多一批没有状态文件的迁移；读的时候只认与盘上状态文件对得上的那一批（E14）。"""

    nights = nights_of(planted(60), 60)
    store = RelationStore(tmp_path)
    for state, moved in nights[:12]:
        store.write(state, moved)
    recorded = store.transitions("session")
    state, moved = nights[12]
    crashed = tuple(replace(item, night=state.night) for item in recorded[:1])
    store._append_transitions("session", state.night, crashed, "0" * 16)  # 被杀在写状态文件之前
    assert store.transitions("session") == recorded


def test_folding_onto_a_night_with_an_older_return_measure_drops_only_those_rows() -> None:
    """3.5：``fold`` 接上一晚时，量法版本旧了的那一段（久别重来）的行清掉、在新判据下重新发现，别的段原样（L10.14 补测：
    原先只测了清理函数本身，把 ``fold`` 里那一次调用去掉测试照样全过）。"""

    from habitus.scene.relations.longspan import MEASURE_VERSIONS
    from habitus.scene.relations.state import Relation

    series = planted(60)
    nights = nights_of(series, 60)
    previous = nights[-1][0]
    assert status_of(previous, PLANTED) is Status.ESTABLISHED
    stale = Relation(
        key=key(DISCUSS, CODE, Segment.RETURN), status=Status.CANDIDATE, upward=True, discovered_on=previous.night
    )
    older = replace(previous, relations=(*previous.relations, stale), measure_versions={})
    tonight = series.until(series.cutoff)
    later = EventSeries(cutoff=previous.night + timedelta(days=1), records=tonight.records)
    line = build_timelines(later, CONCEPTS, {}, CONFIG)["session"]

    state, _moved = fold(older, LaneTests(line, CONCEPTS, CONFIG, later.cutoff))

    assert stale.key not in {relation.key for relation in state.relations if relation.status is Status.CANDIDATE}
    assert status_of(state, PLANTED) is Status.ESTABLISHED
    assert state.measure_versions[Segment.RETURN.value] == MEASURE_VERSIONS[Segment.RETURN.value]
