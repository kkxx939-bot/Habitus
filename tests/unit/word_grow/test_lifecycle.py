"""第二步乙：节点的生命周期（预期表 C20–C34、C36、C38、D2 的"已删"记号与退出日、D5；共享说明 3.3）。

换代用组合根装好的 ``Generation.run``（可以跳着日子跑：每一代按"这一代的日子 − 最后一次被直接选中的日子 ≥ 20"数）。
看：下一次选词提示词里的词表（删掉的不出现）、语义树概念（在用 = 没停用的行为概念；退出 / 删掉以后概念停用）、
换代那份常驻序列读者读出来的 ``kind_token``（已删节点换成最近的活着的上级）、行为树文件（不改）、词表状态文件（按值宽口径）。

时刻口径：节点"写出 / 最后被选中"的日子取那一轮的时刻（测试里的轮都在 2026-04、05，墙钟是别的日子）。
旧代码上应当失败：会话答复带 ``new`` 不合格（不建节点），在用的词永不退出。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from habitus.runtime.foresight import unsealed_bridge
from habitus.runtime.generation_report import GenerationReports
from habitus.runtime.generation_result import generation_attributes
from tests.unit.word_grow.site import (
    NAMES,
    QUERY,
    SUMMARY,
    WEEKLY,
    WRITE_DOC,
    Grow,
    GrowSite,
    Hook,
    Named,
    at,
    entries,
    flat,
    grown,
    mentions_day,
    names,
    node,
)
from tests.unit.word_pick.site import Word, record_lines


def d(month: int, day: int) -> date:
    return date(2026, month, day)


X, Y = "写需求书", "写验收段"


@pytest.fixture
def site(tmp_path: Path):  # type: ignore[no-untyped-def]
    built = GrowSite(tmp_path)
    yield built
    built.close()


def _grow_x(site: GrowSite, moment: datetime) -> None:
    site.record("conv-x", moment, "帮我写一份产品需求文档", Grow("写文档", node(X, "写产品需求文档")))


def _deleted_mark(site: GrowSite, token: str, parent: str, day: date) -> bool:
    """状态文件里有一条带着这个编号、它的上级、删掉那天的条目（"已删"的记号）。"""

    return any(parent in flat(item) and mentions_day(item, day) for item in entries(site.state(), token))


# ── 转在用、退出、删掉 ─────────────────────────────────────────────────────


def test_c20_a_candidate_picked_on_two_days_is_in_use_with_the_same_id(site: GrowSite) -> None:
    _grow_x(site, at(d(4, 13), 9))
    site.record("conv-x", at(d(4, 15), 9), "再写一份需求书", Named(X))
    site.generate(d(4, 15))
    assert grown(1) not in site.in_use()
    site.generate(d(4, 16))
    assert grown(1) in site.in_use()
    concept = site.concepts()[grown(1)]
    assert (concept.title, concept.lane) == (X, "session")
    assert site.token_at(at(d(4, 15), 9)) == grown(1)


def test_c21_d2_a_candidate_not_picked_for_twenty_days_is_deleted_and_leaves_a_mark(site: GrowSite) -> None:
    _grow_x(site, at(d(4, 13), 9))
    site.generate(d(5, 2))
    assert X in names(site.probe(d(5, 2)))
    site.generate(d(5, 3))
    assert X not in names(site.probe(d(5, 3)))
    assert _deleted_mark(site, grown(1), WRITE_DOC, d(5, 3))


def test_c22_a_deleted_nodes_record_reads_as_its_parent_and_the_tree_file_is_untouched(site: GrowSite) -> None:
    _grow_x(site, at(d(4, 13), 9))
    path = site.occurrence_path(at(d(4, 13), 9))
    original = path.read_bytes()
    assert site.series_tokens(d(4, 14)).get(at(d(4, 13), 9)) == grown(1)
    site.generate(d(5, 3))
    assert site.series_tokens(d(5, 4)).get(at(d(4, 13), 9)) == WRITE_DOC
    assert path.read_bytes() == original
    # 选词器显示"最近的记录"归到哪个词也用同一个换算
    site.record("conv-later", at(d(5, 3), 10), "帮我写一份本周周报", Word(WEEKLY))
    (line,) = [item for item in record_lines(site.router.texts("pick")[-1]) if "产品需求文档" in item]
    assert line.endswith(NAMES[WRITE_DOC]) and X not in line


def test_c23_a_grown_node_in_use_not_picked_for_twenty_days_is_deleted_and_its_concept_retired(
    site: GrowSite,
) -> None:
    _grow_x(site, at(d(4, 13), 9))
    site.record("conv-x", at(d(4, 14), 9), "再写一份需求书", Named(X))
    site.generate(d(4, 15))
    assert grown(1) in site.in_use()
    generation = site.runtime.components.generation.generation
    site.generate(d(5, 3))
    kept = generation.marks.read(d(5, 3))
    assert kept is not None and X in names(site.probe(d(5, 3)))
    result = site.generate(d(5, 4))
    assert X not in names(site.probe(d(5, 4)))
    assert grown(1) not in site.in_use()
    assert result.vocabulary_version is not None and result.vocabulary_version > kept.vocabulary_version
    site.generate(d(5, 5))
    assert site.retired(grown(1))


def _x_in_use_with_a_child_picked_yesterday(site: GrowSite) -> None:
    _grow_x(site, at(d(4, 13), 9))
    site.record("conv-x", at(d(4, 14), 9), "再写一份需求书", Named(X))
    site.generate(d(4, 15))
    assert grown(1) in site.in_use()
    # Y 昨天写出（这一轮记在 Y 上，是 Y 的第一个日子）：只一个日子，它自己不会在 05-04 转在用
    site.record("conv-y", at(d(5, 3), 9), "写需求书里的验收标准", Grow(X, node(Y, "写验收标准那一段")))


def test_c24_a_second_level_node_with_a_live_child_is_not_deleted_but_leaves_use(site: GrowSite) -> None:
    _x_in_use_with_a_child_picked_yesterday(site)
    site.generate(d(5, 4))
    block = names(site.probe(d(5, 4)))
    assert X in block and Y in block
    assert grown(1) not in site.in_use() and grown(2) not in site.in_use()
    site.generate(d(5, 5))
    assert site.retired(grown(1))
    assert site.token_at(at(d(5, 3), 9)) == grown(2)


def _x_and_its_only_child_both_idle(site: GrowSite) -> None:
    _grow_x(site, at(d(4, 13), 9))
    site.record("conv-x", at(d(4, 13), 10), "写需求书里的验收标准", Grow(X, node(Y, "写验收标准那一段")))
    site.generate(d(5, 2))
    assert {X, Y} <= set(names(site.probe(d(5, 2))))


def test_c25_a_third_level_node_goes_first_and_its_idle_parent_follows_in_the_same_generation(
    site: GrowSite,
) -> None:
    _x_and_its_only_child_both_idle(site)
    site.generate(d(5, 3))
    block = names(site.probe(d(5, 3)))
    assert X not in block and Y not in block
    assert _deleted_mark(site, grown(2), grown(1), d(5, 3))
    assert _deleted_mark(site, grown(1), WRITE_DOC, d(5, 3))


def test_c30_a_deleted_third_level_nodes_record_climbs_to_the_nearest_live_ancestor(site: GrowSite) -> None:
    _x_and_its_only_child_both_idle(site)
    assert site.series_tokens(d(4, 14)).get(at(d(4, 13), 10)) == grown(2)
    site.generate(d(5, 3))
    tokens = site.series_tokens(d(5, 4))
    assert tokens[at(d(4, 13), 10)] == WRITE_DOC and tokens[at(d(4, 13), 9)] == WRITE_DOC


def _query_in_use_then_exits(site: GrowSite) -> None:
    site.record("conv-q", at(d(4, 13), 9), "查一下上个月的订单数", Named(NAMES[QUERY]))
    site.record("conv-q", at(d(4, 14), 9), "再查一下退货数", Named(NAMES[QUERY]))
    site.generate(d(4, 15))
    assert QUERY in site.in_use()
    site.generate(d(5, 3))
    assert QUERY in site.in_use()
    site.generate(d(5, 4))


def test_c26_d2_an_initial_word_in_use_leaves_use_after_twenty_days_but_stays_in_the_table(site: GrowSite) -> None:
    _query_in_use_then_exits(site)
    assert QUERY not in site.in_use()
    assert NAMES[QUERY] in names(site.probe(d(5, 4)))
    assert any(mentions_day(item, d(5, 4)) for item in entries(site.state(), QUERY))  # 记着哪天退出
    site.generate(d(5, 5))
    assert site.retired(QUERY)


def test_c27_after_leaving_use_one_more_day_does_not_bring_it_back(site: GrowSite) -> None:
    _query_in_use_then_exits(site)
    assert QUERY not in site.in_use()
    site.record("conv-q", at(d(5, 5), 9), "查一下这个月的订单数", Named(NAMES[QUERY]))
    site.generate(d(5, 6))
    assert QUERY not in site.in_use()  # 退出前的日子作废


def test_c28_after_leaving_use_two_new_days_bring_it_back_with_the_same_id(site: GrowSite) -> None:
    _query_in_use_then_exits(site)
    assert QUERY not in site.in_use()
    site.record("conv-q", at(d(5, 5), 9), "查一下这个月的订单数", Named(NAMES[QUERY]))
    site.record("conv-q", at(d(5, 6), 9), "再查一下这个月的退货数", Named(NAMES[QUERY]))
    site.generate(d(5, 7))
    assert QUERY in site.in_use() and not site.retired(QUERY)


def test_c29_a_deleted_name_grown_again_is_a_new_node_with_a_new_id(site: GrowSite) -> None:
    _grow_x(site, at(d(4, 13), 9))
    site.generate(d(5, 3))
    assert X not in names(site.probe(d(5, 3)))
    site.record("conv-x2", at(d(5, 4), 9), "帮我写一份产品需求文档", Grow("写文档", node(X)))
    assert site.token_at(at(d(5, 4), 9)) == grown(2)
    assert names(site.probe(d(5, 4))).count(X) == 1
    site.generate(d(5, 5))
    assert grown(2) not in site.in_use()  # 从候选开始


def test_c31_a_deleted_nodes_day_counts_for_its_parent_turning_in_use(site: GrowSite) -> None:
    _grow_x(site, at(d(4, 13), 9))
    site.record("conv-w", at(d(4, 20), 9), "帮我新写一份文档", Named(NAMES[WRITE_DOC]))
    site.generate(d(5, 2))
    assert WRITE_DOC not in site.in_use()
    site.generate(d(5, 3))  # X 删掉
    site.generate(d(5, 4))
    assert WRITE_DOC in site.in_use()  # 04-13（换算过来的）+ 04-20


def test_c32_with_the_cap_full_a_word_leaving_use_makes_room_in_the_same_generation(tmp_path: Path) -> None:
    def cap_two(raw: dict[str, Any]) -> None:
        raw["behavior"]["kinds_active_cap"] = 2

    site = GrowSite(tmp_path, edit=cap_two)
    try:
        for token, said in ((QUERY, "查一下订单数"), (WEEKLY, "帮我写一份周报")):
            site.record(f"conv-{token}", at(d(4, 13), 9), said, Named(NAMES[token]))
            site.record(f"conv-{token}", at(d(4, 14), 9), said, Named(NAMES[token]))
        site.generate(d(4, 15))
        site.record("conv-s", at(d(4, 16), 9), "写个总结", Named(NAMES[SUMMARY]))
        site.record("conv-s", at(d(4, 17), 9), "再写个总结", Named(NAMES[SUMMARY]))
        site.record(f"conv-{QUERY}", at(d(4, 20), 9), "查一下退货数", Named(NAMES[QUERY]))
        waiting = site.generate(d(4, 18))
        assert (waiting.active_words, waiting.waiting_words) == (2, 1)
        result = site.generate(d(5, 4))  # 写周报最后一次是 04-14：退出；等着的写总结同一代进来
        assert (result.active_words, result.waiting_words) == (2, 0)
        site.generate(d(5, 5))
        assert site.in_use() == {QUERY, SUMMARY}
    finally:
        site.close()


def test_c33_an_unsealed_judgement_on_a_deleted_node_reads_as_its_nearest_live_ancestor(site: GrowSite) -> None:
    _grow_x(site, at(d(4, 13), 9))
    site.router.picks.append(Named(X))
    site.consume(site.turn("conv-x", at(d(4, 13), 10), "再写一份需求书"))  # 写了判断，还没归约
    bridge = unsealed_bridge(site.session)
    before = [row.kind_token for row in bridge.rows(since=at(d(4, 13), 0), until=at(d(4, 13), 23))]
    assert before == [grown(1)]
    site.generate(d(5, 3))
    asyncio.run(bridge.prepare(until=at(d(5, 3), 12)))
    after = [
        row.kind_token for row in unsealed_bridge(site.session).rows(since=at(d(4, 13), 0), until=at(d(4, 13), 23))
    ]
    assert after == [WRITE_DOC]


def test_c34_a_candidate_never_picked_is_deleted_twenty_days_after_it_was_written(
    site: GrowSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    site.router.picks.append(Grow("写文档", node(X)))
    turn = site.turn("conv-c34", at(d(4, 13), 9), "帮我写一份产品需求文档")

    def killed(*_: object, **__: object) -> None:
        raise RuntimeError("进程被杀")

    with monkeypatch.context() as patched:
        patched.setattr(site.session.judgements, "put_payload", killed)
        with pytest.raises(RuntimeError):
            site.consume(turn)
    site.generate(d(5, 2))
    assert X in names(site.probe(d(5, 2)))
    site.generate(d(5, 3))
    assert X not in names(site.probe(d(5, 3)))


def test_c36_a_node_deleted_while_the_model_answers_sends_the_record_to_its_parent_without_revival(
    site: GrowSite,
) -> None:
    _grow_x(site, at(d(4, 13), 9))
    site.record("conv-race", at(d(5, 3), 9), "再写一份需求书", Hook(site.generate_in_thread(d(5, 3)), Named(X)))
    assert site.router.count("pick") == 2  # 不重问
    assert site.token_at(at(d(5, 3), 9)) == WRITE_DOC
    assert X not in names(site.probe(d(5, 3)))  # 没有被复活
    assert _deleted_mark(site, grown(1), WRITE_DOC, d(5, 3))


def _hit_files(site: GrowSite, day: date) -> dict[str, bytes]:
    folder = Path(site.runtime.config.scene_root) / "occurrences" / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}"
    if not folder.is_dir():
        return {}
    return {path.name: path.read_bytes() for path in sorted(folder.iterdir()) if path.is_file()}


def test_c38_the_days_a_deleted_node_had_records_on_are_resealed_with_the_parent_and_other_days_are_untouched(
    site: GrowSite,
) -> None:
    _grow_x(site, at(d(4, 13), 9))
    site.record("conv-q", at(d(4, 14), 9), "查一下上个月的订单数", Named(NAMES[QUERY]))
    site.generate(d(5, 2))
    before_13, before_14 = _hit_files(site, d(4, 13)), _hit_files(site, d(4, 14))
    assert any(grown(1).encode() in body for body in before_13.values())
    site.generate(d(5, 3))
    site.generate(d(5, 4))
    after_13 = _hit_files(site, d(4, 13))
    hits = [body for name, body in after_13.items() if not name.startswith(".")]
    assert hits and all(grown(1).encode() not in body and WRITE_DOC.encode() in body for body in hits)
    assert _hit_files(site, d(4, 14)) == before_14


# ── D5：读数 ──────────────────────────────────────────────────────────────

OLD_ATTRIBUTE_KEYS = frozenset(
    {
        "active_words",
        "complete",
        "failed_step",
        "held_reason",
        "held_step",
        "lag_seconds",
        "reason",
        "series_cutoff",
        "stamped_days",
        "unreduced_gaps",
        "unwatched_minutes",
        "unwatched_spans",
        "vocabulary_version",
        "waiting_words",
    }
)
OLD_PREFIXES = ("fallback.", "outcome.", "queue_owed.", "relation_night.")
OLD_INPUT_KEYS = frozenset(
    {
        "series_cutoff",
        "vocabulary_version",
        "active_words",
        "waiting_words",
        "relation_nights",
        "stamped_days",
        "unreduced_gaps",
    }
)
OLD_COUNT_REPORT_KEYS = frozenset({"vocabulary_version", "active_words", "added_words", "waiting_words"})


def _new_ints(mapping: dict[str, Any], old: frozenset[str], prefixes: tuple[str, ...] = ()) -> list[int]:
    return [
        value
        for key, value in mapping.items()
        if key not in old and not key.startswith(prefixes) and isinstance(value, int) and not isinstance(value, bool)
    ]


def _contains(found: list[int], expected: list[int]) -> bool:
    pool = list(found)
    for value in expected:
        if value not in pool:
            return False
        pool.remove(value)
    return True


def test_d5_the_generation_reports_deleted_left_use_and_candidate_counts(site: GrowSite, tmp_path: Path) -> None:
    site.record("conv-q", at(d(4, 13), 9), "查一下上个月的订单数", Named(NAMES[QUERY]))
    site.record("conv-q", at(d(4, 14), 9), "再查一下退货数", Named(NAMES[QUERY]))
    for index, name in enumerate(("写需求书", "写设计书")):
        site.record(f"conv-old-{index}", at(d(4, 13), 10 + index), f"帮我{name}", Grow("写文档", node(name)))
    site.generate(d(4, 15))
    for index, name in enumerate(("写测试书", "写用户书", "写运维书")):
        site.record(f"conv-new-{index}", at(d(4, 30), 10 + index), f"帮我{name}", Grow("写文档", node(name)))
    result = site.generate(d(5, 4))  # 删 2 个、退出在用 1 个（查数据）、此刻候选 3 个
    expected = [2, 1, 3]

    steps = [step.step for step in result.steps]
    between = steps[steps.index("series") + 1 : steps.index("scene")]
    counted: list[int] = []
    for step in result.steps:
        if step.step in between:
            counted.extend(_new_ints(dict(step.report), OLD_COUNT_REPORT_KEYS))
    assert _contains(counted, expected), counted

    attributes = _new_ints(generation_attributes(result), OLD_ATTRIBUTE_KEYS, OLD_PREFIXES)
    assert _contains(attributes, expected), attributes

    root = tmp_path.resolve()
    reports = GenerationReports(root / "reports", artifact_root=root, relations=None)
    reports.write(result, now=datetime(2026, 5, 4, 6, tzinfo=UTC))
    written = reports.read(d(5, 4))
    assert written is not None
    inputs = written["inputs"]
    assert isinstance(inputs, dict)
    assert _contains(_new_ints(inputs, OLD_INPUT_KEYS), expected), inputs
