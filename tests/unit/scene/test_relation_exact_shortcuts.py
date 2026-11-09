"""关系检验的精确捷径守住"结果逐字不变"（relations-incremental：E1 条件判定按链共用、E3 门 1 整组跳过、E4 读数只算一遍、
E5 稀疏的只计数、E6 加权 BH 的归一）。

这里只看对外结果——``fold`` 产出的 ``LaneState``（经 ``document.encode_night`` 的规范 JSON）与迁移——不看内部结构。期望值是在
基线（7dffb51 + 裁定 95 的分族层级与 V2.2、不含 E1–E5 捷径的参照代码）上跑出来写死的：改了检验的语义（不是只改用时与内存）才该
重新生成它们。

合成数据：40 个工作日，讨论方案每天一次（11:00–15:00 随机），只在第 4、19 个工作日之前 20 分钟先提交推送——于是
"讨论方案 | 之前提交推送过"这组条件下只有 2 条链，它的全部后果、全部段都到不了每边 3 次（``MIN_PER_SIDE``）。
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from datetime import date, timedelta

import pytest

from habitus.scene.relations import (
    LaneState,
    LaneTests,
    Relation,
    RelationKey,
    Segment,
    Status,
    Transition,
    build_timelines,
    fold,
)
from habitus.scene.relations.conditions import Condition, Template
from habitus.scene.relations.document import encode_night, encode_transition
from habitus.scene.relations.longspan import MEASURE_VERSIONS
from habitus.scene.relations.spans import SEGMENTS
from habitus.series import EventSeries
from tests.unit.scene import test_relation_family_hierarchy as hierarchy
from tests.unit.scene.relation_fixtures import (
    CODE,
    CONCEPTS,
    CONFIG,
    DISCUSS,
    DOCS,
    PUSH,
    RESEARCH,
    key,
    record,
    workdays,
)

DAYS = 40
#: 讨论方案之前先提交推送的那两天（第几个工作日）。
PRIMED_DAYS = (4, 19)
PUSHED = Condition(Template.TIMELINE, PUSH).text


def _series(seed: int) -> EventSeries:
    generator = random.Random(seed)
    records = []
    for index, day in enumerate(workdays(DAYS)):
        number = 0

        def add(hour: int, minute: int, title: str, day: date = day) -> None:
            nonlocal number
            number += 1
            records.append(record(day, hour, minute, title, number=number))

        add(9, 0, "调研")
        add(17, 50, "写文档")
        hour, minute = divmod(generator.randint(11 * 60, 15 * 60), 60)
        if index in PRIMED_DAYS:
            add(*divmod(hour * 60 + minute - 20, 60), "提交推送")
        add(hour, minute, "讨论方案")
        if generator.random() < 0.7:
            add(*divmod(hour * 60 + minute + 20, 60), "修改代码")
        add(*divmod(generator.randint(9 * 60 + 5, 10 * 60 + 30), 60), "修改代码")
        add(*divmod(generator.randint(16 * 60, 17 * 60 + 15), 60), "调研")
        if generator.random() < 0.5:
            add(*divmod(generator.randint(16 * 60, 17 * 60 + 15), 60), "提交推送")
    records.sort(key=lambda item: (item.started_at, item.uri))
    return EventSeries(cutoff=workdays(DAYS)[-1] + timedelta(days=1), records=tuple(records))


def _tests(seed: int, previous: LaneState | None = None) -> LaneTests:
    series = _series(seed)
    line = build_timelines(series, CONCEPTS, {}, CONFIG)["session"]
    established = tuple(item.key for item in previous.of(Status.ESTABLISHED)) if previous is not None else ()
    return LaneTests(line, CONCEPTS, CONFIG, series.cutoff, established=established)


def _digest(state: LaneState, transitions: tuple[Transition, ...]) -> str:
    text = encode_night(state) + "\n" + "".join(encode_transition(item) + "\n" for item in transitions)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _uniform(tests: LaneTests, weight: float) -> dict[RelationKey, float]:
    """先验只给不带条件的键（``runtime/scene_advice.weights``），带条件的按 1。"""

    return {item: weight for item in tests.keys() if not item.condition}


def _candidates(state: LaneState) -> set[str]:
    return {item.key.text for item in state.relations if item.status is Status.CANDIDATE}


def _short(item: RelationKey) -> str:
    names = {DISCUSS: "讨论", CODE: "代码", RESEARCH: "调研", DOCS: "文档", PUSH: "推送"}
    condition = (
        "+".join(names.get(part.split(":", 1)[1], part) for part in item.condition.split("+")) if item.condition else ""
    )
    return (
        f"{names[item.antecedent]}→{names[item.consequent]} {item.segment.value}{'｜' + condition if condition else ''}"
    )


# ── 门 1 整组都过不了：全部计为稀疏，计数与逐条量一致 ─────────────────────────────


def test_a_condition_with_two_chains_on_one_side_leaves_every_consequent_and_segment_sparse() -> None:
    """ "讨论方案｜之前提交推送过"条件下只有 2 条链：这一组的每个后果、每一段都计进稀疏、不留行；各段稀疏计数、族的大小、
    整份关系表与基线逐字相同（第一晚：每条键要么是一行、要么计进稀疏）。"""

    tests = _tests(2)
    keys = tests.keys()
    group = [item for item in keys if item.antecedent == DISCUSS and item.condition == PUSHED]
    assert {item.segment for item in group} == set(SEGMENTS)
    assert {item.consequent for item in group} == {CODE, RESEARCH, DOCS}  # 推送自己是条件，不当后果
    state, transitions = fold(None, tests)
    stored = {item.key for item in state.relations}
    assert not stored & set(group)
    assert sum(state.sparse.values()) + len(state.relations) == len(set(keys))
    assert state.sparse == GOLDEN_SPARSE
    assert state.family == GOLDEN_FAMILY
    assert _digest(state, transitions) == GOLDEN_PLAIN_DIGEST


# ── 跨夜带着的键：当晚两边不够也照样有完整读数 ─────────────────────────────────


def _carried_previous(cutoff: date) -> LaneState:
    days = workdays(DAYS)
    return LaneState(
        lane="session",
        night=cutoff - timedelta(days=1),
        family=0,
        relations=(
            Relation(
                key=key(DISCUSS, CODE, condition=PUSHED),
                status=Status.CANDIDATE,
                upward=True,
                discovered_on=days[20],
                forward_needed=5,
            ),
            Relation(
                key=key(DISCUSS, RESEARCH, condition=PUSHED),
                status=Status.ESTABLISHED,
                upward=True,
                discovered_on=days[5],
                forward_passed_on=days[15],
                established_on=days[15],
                maintained_through=days[30],
                maintenance_needed=4,
            ),
            Relation(
                key=key(DISCUSS, DOCS, condition=PUSHED),
                status=Status.EXPIRED,
                upward=True,
                discovered_on=days[2],
                established_on=days[6],
                expired_on=days[10],
            ),
            Relation(
                key=key(DISCUSS, CODE, Segment.NEXT_DAY, condition=PUSHED),
                status=Status.REJECTED,
                upward=False,
                discovered_on=days[2],
                rejected_on=days[3],
            ),
        ),
        measure_versions=MEASURE_VERSIONS,
    )


def test_carried_rows_keep_a_full_reading_even_when_one_side_has_two_chains() -> None:
    """候选、成立、失效、前向未复现跨夜带着历史，当晚必须有完整读数（候选做前向验证与第 2、3 道，成立做维持检验与第 2、3
    道，休眠的留当晚的读数）——哪怕它们所在的那组条件下只有 2 条链、同组没带着的键全部计为稀疏。"""

    series = _series(2)
    previous = _carried_previous(series.cutoff)
    tests = _tests(2, previous)
    state, transitions = fold(previous, tests)
    rows = {item.key: item for item in state.relations}
    carried = {item.key: item for item in previous.relations}
    assert set(carried) <= set(rows)
    found = {}
    for item_key in carried:
        row = rows[item_key]
        assert row.tonight is not None
        found[_short(item_key)] = (
            row.status.value,
            row.tonight.verdict.value,
            row.tonight.antecedents,
            row.tonight.blocks,
            row.tonight.p_value,
            row.tonight.minimum_p,
            row.tonight.stable,
            row.tonight.large_enough,
        )
    assert found == GOLDEN_CARRIED
    for status in (Status.CANDIDATE, Status.ESTABLISHED):
        row = next(item for item in state.relations if item.key in carried and carried[item.key].status is status)
        assert row.tonight is not None and row.tonight.stable is not None and row.tonight.leave_one_out is not None
    # 同组没带着的键照旧只计数
    others = [
        item for item in tests.keys() if item.antecedent == DISCUSS and item.condition == PUSHED and item not in carried
    ]
    assert others and not {item.key for item in state.relations} & set(others)
    assert state.sparse == GOLDEN_CARRIED_SPARSE
    assert _digest(state, transitions) == GOLDEN_CARRIED_DIGEST


# ── 带先验：读数只算一遍之后，"靠先验"仍按不加权的显著集打 ─────────────────────────


@pytest.mark.parametrize(("seed", "weight"), [(5, 0.5), (2, 1.5), (5, 2.0)])
def test_prior_only_marks_exactly_the_candidates_the_unweighted_night_would_not_find(seed: int, weight: float) -> None:
    """带一份非 1 的权重折叠：新发现的候选里，不加权折叠找不到的才打"靠先验"，其余不打；不加权能找到、加权丢了的不是候选。
    结果与基线逐字相同。"""

    plain, _ = fold(None, _tests(seed))
    tests = _tests(seed)
    state, transitions = fold(None, tests, _uniform(tests, weight))
    unweighted = _candidates(plain)
    for row in state.relations:
        if row.status is Status.CANDIDATE:
            assert row.prior_only is (row.key.text not in unweighted)
        else:
            assert not row.prior_only
    prior_only = sorted(_short(item.key) for item in state.relations if item.prior_only)
    lost = sorted(_short(item.key) for item in plain.relations if item.key.text in unweighted - _candidates(state))
    assert (prior_only, lost, state.family) == GOLDEN_PRIOR[(seed, weight)]
    assert _digest(state, transitions) == GOLDEN_PRIOR_DIGEST[(seed, weight)]


def test_prior_only_on_a_second_night_reads_the_unweighted_night_on_the_same_new_data() -> None:
    """第二晚：带着候选、成立、休眠的行（``outside`` 与 ``since`` 都起作用）再带一份权重折叠，"靠先验"的判法与第一晚相同，
    结果与基线逐字相同。"""

    series = _series(5)
    previous = _carried_previous(series.cutoff)
    plain, _ = fold(previous, _tests(5, previous))
    tests = _tests(5, previous)
    state, transitions = fold(previous, tests, _uniform(tests, 0.5))
    unweighted = _candidates(plain)
    for row in state.relations:
        if row.status is Status.CANDIDATE and row.discovered_on == state.night:
            assert row.prior_only is (row.key.text not in unweighted)
    assert sorted(_short(item.key) for item in state.relations if item.prior_only) == GOLDEN_SECOND_NIGHT_PRIOR_ONLY
    assert _digest(state, transitions) == GOLDEN_SECOND_NIGHT_DIGEST


#: 先验经父关系影响带条件关系的几份情形：（工作日数, 先调研过的比例, 先调研过之后接修改代码的比例, 父关系的权重,
#: 其余不带条件的权重, 是否第二晚）。
PRIOR_THROUGH_PARENT = {
    "boosted": (40, 0.6, 0.65, 2.0, 0.5, False),
    "damped": (70, 0.6, 0.5, 0.5, 2.0, False),
    "boosted-second-night": (40, 0.6, 0.65, 2.0, 0.5, True),
}


@pytest.mark.parametrize("name", sorted(PRIOR_THROUGH_PARENT))
def test_prior_only_reaches_a_moderated_relation_through_its_parent(name: str) -> None:
    """裁定 95：带条件的关系经父关系（过了第 1 道或跨夜带着）或经它所在的（前因, 后果, 段）组进族；组过不了时先验经父关系
    影响带条件的那条。数据取层级测试的"六成先调研过、先调研过之后才可能接修改代码"，带条件那条中等强、它那一组的 Simes p
    过不了组间 BH（不然它不加权也经组进族，先验就碰不到它）：40 天、接的比例 65% 时父关系不加权过不了、先验给它 2 就过——
    父关系与"｜先调研过"都打"靠先验"；70 天、接的比例 50% 时父关系不加权过得了、先验给它 0.5 就过不了——两条都是"不加权
    能找到、加权丢了"。第二晚再带着 ``_carried_previous`` 那几行（``outside`` 与 ``since`` 起作用）。"靠先验"的判法与
    上面相同，结果与基线逐字相同。"""

    days, primed, on, boost, damp, second = PRIOR_THROUGH_PARENT[name]
    series = hierarchy._series(days, primed=primed, on=on, off=0.0)
    previous = _carried_previous(series.cutoff) if second else None
    established = tuple(item.key for item in previous.of(Status.ESTABLISHED)) if previous is not None else ()
    plain, _ = fold(previous, hierarchy._tests(series, established=established))
    tests = hierarchy._tests(series, established=established)
    weights = {item: (boost if item == hierarchy.PARENT else damp) for item in tests.keys() if not item.condition}
    state, transitions = fold(previous, tests, weights)
    unweighted = _candidates(plain)
    for row in state.relations:
        if row.status is Status.CANDIDATE and row.discovered_on == state.night:
            assert row.prior_only is (row.key.text not in unweighted)
        else:
            assert not row.prior_only
    prior_only = sorted(_short(item.key) for item in state.relations if item.prior_only)
    lost = sorted(_short(item.key) for item in plain.relations if item.key.text in unweighted - _candidates(state))
    assert prior_only or lost  # 前提：这份数据上先验确实改了结果
    assert (prior_only, lost, plain.family, state.family) == GOLDEN_PRIOR_THROUGH_PARENT[name]
    assert _digest(state, transitions) == GOLDEN_PRIOR_THROUGH_PARENT_DIGEST[name]


# ── 权重归一：必然稀疏的带条件键也按"个数 × 权重 1"留在归一里 ────────────────────────


def test_weights_are_normalised_over_every_key_including_the_hopeless_moderated_ones() -> None:
    """加权 BH 的权重在全部进族候选的键上归一到平均 1（``stats.tarone_bh``）：不带条件的都给 0.1、带条件的按 1 时，那一大批
    门 1 都过不了的带条件键也算在归一里（裁定 95：计入带条件那一族的归一）。族的大小与过第 1 道的那几条与基线逐字相同。"""

    tests = _tests(2)
    hopeless = [item for item in tests.keys() if item.antecedent == DISCUSS and item.condition == PUSHED]
    assert len(hopeless) == 12
    plain, _ = fold(None, _tests(2))
    state, transitions = fold(None, tests, _uniform(tests, 0.1))
    assert (plain.family, state.family) == GOLDEN_NORMALISED_FAMILY
    assert sorted(_short(item.key) for item in state.relations if item.status is Status.CANDIDATE) == GOLDEN_NORMALISED
    assert Counter(item.status for item in state.relations) == GOLDEN_NORMALISED_COUNTS
    assert state.sparse == GOLDEN_NORMALISED_SPARSE
    assert _digest(state, transitions) == GOLDEN_NORMALISED_DIGEST


# ── 基线（7dffb51 + 裁定 95 分族层级与 V2.2、不含 E1–E5）上跑出的期望值（生成脚本：V22实现/regen/regen.py）

GOLDEN_SPARSE = {"chain": 42, "days_2_7": 47, "next_day": 47, "rest_of_day": 47}
GOLDEN_FAMILY = 5
GOLDEN_PLAIN_DIGEST = "130a5804cf33cfc5b56b8caf4cc9483054e68897284a2891cfd34531109513a4"
#: （状态, 当晚判定, 前因次数, 块数, p, 最小可达 p, 时间上稳, 影响够大）
GOLDEN_CARRIED: dict[str, tuple[object, ...]] = {
    "讨论→代码 chain｜推送": ("candidate", "tested", 40, 40, 1.0, 1.0, False, False),
    "讨论→调研 chain｜推送": ("established", "tested", 40, 40, 1.0, 1.0, False, False),
    "讨论→文档 chain｜推送": ("expired", "sparse", 30, 30, 1.0, 1.0, None, None),
    "讨论→代码 next_day｜推送": ("rejected", "sparse", 0, 0, 1.0, 1.0, None, None),
}
GOLDEN_CARRIED_SPARSE = {"chain": 40, "days_2_7": 47, "next_day": 46, "rest_of_day": 47}
GOLDEN_CARRIED_DIGEST = "4d98a6185c147ebca65ff9d350701ed9c2d06fe4cfe2ed84736973d89f53f202"
#: （打了"靠先验"的, 不加权能找到、加权丢了的, 族的大小）
GOLDEN_PRIOR: dict[tuple[int, float], tuple[list[str], list[str], int]] = {
    (5, 0.5): ([], [], 4),
    (2, 1.5): ([], [], 5),
    (5, 2.0): ([], [], 4),
}
GOLDEN_PRIOR_DIGEST = {
    (5, 0.5): "38d887d9590fd10f6d1e0f026fc670d8014cfd3923d0755acd47f33669b77c44",
    (2, 1.5): "524baaf2e5bb7280bc2ab19a3373a748082602a9342a375596e5bd5f5702f48b",
    (5, 2.0): "b658f4dc6ca2fe4d5bc623131d3cf5848949a87dfb5149361f62b81153ba5f18",
}
GOLDEN_SECOND_NIGHT_PRIOR_ONLY: list[str] = []
GOLDEN_SECOND_NIGHT_DIGEST = "cc2e0e398572707ff82896bb8e76a111288695e0e9c78327ccee4efb9629f286"
#: （不加权的族, 加权的族）
GOLDEN_NORMALISED_FAMILY = (5, 5)
GOLDEN_NORMALISED = ["讨论→代码 chain"]
GOLDEN_NORMALISED_SPARSE = {"chain": 42, "days_2_7": 47, "next_day": 47, "rest_of_day": 47}
GOLDEN_NORMALISED_COUNTS = Counter({Status.TESTED: 4, Status.CANDIDATE: 1})
GOLDEN_NORMALISED_DIGEST = "b00fbdb0ab1fedd1c440d91d79dc2510c5e86f07fe539e42bb7151769419ca0e"
#: （打了"靠先验"的, 不加权能找到、加权丢了的, 不加权的族, 加权的族）
GOLDEN_PRIOR_THROUGH_PARENT: dict[str, tuple[list[str], list[str], int, int]] = {
    "boosted": (["讨论→代码 chain", "讨论→代码 chain｜调研"], [], 15, 16),
    "boosted-second-night": (["讨论→代码 chain", "讨论→代码 chain｜调研"], [], 15, 16),
    "damped": ([], ["讨论→代码 chain", "讨论→代码 chain｜调研"], 17, 16),
}
GOLDEN_PRIOR_THROUGH_PARENT_DIGEST = {
    "boosted": "18382afc8379b4d7871ed028abd7704fbd2c46f307948fa19fa3b57cddfc5768",
    "boosted-second-night": "ddf7352789e7a0da1853201d0c8c9ea76821c1bef5bab00e1f926dbed61602d2",
    "damped": "7d2c007f496a62ed28b7afe523ab8aae382f2ab29adc3d223da3a0ba7ac4bf08",
}
