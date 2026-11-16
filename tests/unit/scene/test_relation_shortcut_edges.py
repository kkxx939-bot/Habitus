"""关系检验精确捷径的边缘（relations-incremental 验证轮对抗测试）：开测门第 1 道"整组到不了就不逐条量"的判据只在
**链数**上数，这里专挑链数恰好卡在 ``MIN_PER_SIDE``（3）与差一条（2）的组，以及几种容易数错链的情形——几个条件合起来的组合键、非同一条链段合并之后链变少、条件判不了（``None``）的链——再加上
"跳过的键在加权归一之后可能进族、必须回退去逐条量"的那条路。

只看对外结果：``fold`` 产出的 ``LaneState``（``document.encode_night``）与迁移的摘要、各段稀疏计数与族的大小。期望值一律
在基线（7dffb51 + 裁定 95 分族层级与 V2.2、不含 E1–E5 捷径的参照代码）上跑出来写死；每个情形另用基线也有的公开函数（``chains`` / ``merge_for`` / ``conditions.evaluator``）
核对它确实卡在所说的边上，免得数据一改就悄悄不再考边缘。
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, timedelta

import pytest

from habitus.scene.concepts import ConceptSet
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
from habitus.scene.relations.conditions import Condition, Field, Template, evaluator
from habitus.scene.relations.document import encode_night, encode_transition
from habitus.scene.relations.longspan import MEASURE_VERSIONS
from habitus.scene.relations.spans import merge_for
from habitus.series import EventSeries, SeriesRecord
from tests.unit.scene import test_relation_exact_shortcuts as exact
from tests.unit.scene.concept_fixtures import base
from tests.unit.scene.hit_fixtures import uri_for
from tests.unit.scene.relation_fixtures import CODE, CONFIG, DISCUSS, PUSH, TITLES, record, watching, workdays

PUSHED = Condition(Template.TIMELINE, PUSH)
OFFICE = Condition(Template.FIELD, "公司", Field.PLACE)
HOME = Condition(Template.FIELD, "家里", Field.PLACE)
Outcome = tuple[str, str, dict[str, int], int, int]
RARE_TITLES = tuple(f"杂事{index}" for index in range(10))


@dataclass(frozen=True)
class Plan:
    """一份合成数据：40 个工作日，讨论方案每天一次（``doubled`` 的那几天相隔两小时再一次）。集合里是第几个工作日。"""

    seed: int = 3
    #: 第一次讨论方案之前 20 分钟先提交推送；``late_primed``：第二次之前。
    primed: frozenset[int] = frozenset()
    late_primed: frozenset[int] = frozenset()
    doubled: frozenset[int] = frozenset()
    #: 第一次讨论方案的地点；都没列的那天没有地点（地点条件判不了）。
    office: frozenset[int] = frozenset()
    home: frozenset[int] = frozenset()
    #: 讨论方案之后几分钟接修改代码：七成的日子；``follows`` 给了就恰好是那几天（第一次讨论之后）。
    code_after: int = 20
    follows: frozenset[int] | None = None
    #: 另有几个很少出现的类（每个两次，清早 7 点），只为让不带条件的键多起来。
    rare: int = 0
    #: 几个同事各以一半的概率出现在每条记录上（机械列举的"同在的人"条件，用来批量造带条件的键）；另一个随机数发生器，
    #: 行为时间线与同事多少无关。
    people: int = 0


def _concepts(plan: Plan) -> ConceptSet:
    return ConceptSet([base(title) for title in (*TITLES, *RARE_TITLES[: plan.rare])])


def _series(plan: Plan) -> EventSeries:
    generator = random.Random(plan.seed)
    days = workdays(40)
    records: list[SeriesRecord] = []
    for index, day in enumerate(days):
        number = 0

        def add(minutes: int, title: str, day: date = day) -> SeriesRecord:
            nonlocal number
            number += 1
            item = record(day, *divmod(minutes, 60), title, number=number)
            records.append(item)
            return item

        add(9 * 60, "调研")
        start = generator.randint(11 * 60, 13 * 60 + 30)
        if index in plan.primed:
            add(start - 20, "提交推送")
        first = add(start, "讨论方案")
        # 第一次讨论方案换成按名字编号的行为 URI（期望值按这份 URI 写死）；地点没列的那天留空
        place = "公司" if index in plan.office else "家里" if index in plan.home else None
        uri = uri_for(day, f"讨论方案{number}", *divmod(start, 60))
        records[-1] = first = replace(first, uri=uri, place=place)
        chance = generator.random()
        if index in plan.follows if plan.follows is not None else chance < 0.7:
            add(start + plan.code_after, "修改代码")
        if index in plan.doubled:
            second = start + 120
            if index in plan.late_primed:
                add(second - 20, "提交推送")
            add(second, "讨论方案")
            if generator.random() < 0.7:
                add(second + plan.code_after, "修改代码")
        add(17 * 60 + 50, "写文档")
        for rare, title in enumerate(RARE_TITLES[: plan.rare]):
            if index in (rare * 3 % 40, (rare * 3 + 17) % 40):
                add(7 * 60 + rare, title)
    if plan.people:
        company = random.Random(plan.seed + 1000)
        names = [f"同事{index}" for index in range(plan.people)]
        records = [replace(item, subjects=tuple(name for name in names if company.random() < 0.5)) for item in records]
    records.sort(key=lambda item: (item.started_at, item.uri))
    return EventSeries(cutoff=days[-1] + timedelta(days=1), records=tuple(records))


def _tests(
    plan: Plan,
    *,
    established: tuple[RelationKey, ...] = (),
) -> LaneTests:
    series = _series(plan)
    concepts = _concepts(plan)
    line = build_timelines(series, concepts, {}, CONFIG)["session"]
    return LaneTests(line, concepts, CONFIG, series.cutoff, established=established)


def _sides(tests: LaneTests, conditions: tuple[Condition, ...], segment: Segment) -> tuple[int, tuple[int, ...]]:
    """（条件全成立的链数, 每个条件"其余成立、它不成立"的链数）：按 ``split`` 的分法在这一段合并后的链上数，判不了的不计。"""

    window = CONFIG.transition_window_slots
    judges = [evaluator(tests.line, condition, window=window) for condition in conditions]
    inside = 0
    outsides = [0] * len(judges)
    for chain in merge_for(segment, tests.chains(DISCUSS)):
        flags = [judge(chain) for judge in judges]
        if any(flag is None for flag in flags):
            continue
        failed = [index for index, flag in enumerate(flags) if flag is False]
        if not failed:
            inside += 1
        elif len(failed) == 1:
            outsides[failed[0]] += 1
    return inside, tuple(outsides)


def _hashed(tests: LaneTests) -> dict[RelationKey, float]:
    """先验三档（2 / 1 / 0.5）按键文本摘要分配，只给不带条件的键（``runtime/scene_advice.weights``）。"""

    return {
        item: (0.5, 1.0, 2.0)[int(hashlib.sha256(item.text.encode()).hexdigest()[:8], 16) % 3]
        for item in tests.keys()
        if not item.condition
    }


def _digest(state: LaneState, transitions: tuple[Transition, ...]) -> str:
    text = encode_night(state) + "\n" + "".join(encode_transition(item) + "\n" for item in transitions)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rounded(value: object) -> object:
    if isinstance(value, float):
        return float(f"{value:.9g}")
    if isinstance(value, dict):
        return {name: _rounded(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


def _stable_text(state: LaneState, transitions: tuple[Transition, ...]) -> str:
    """折叠结果的全文，浮点取 9 位有效数字：带先验权重的那几份情形在 Linux、Windows 上与 macOS 只差浮点末位（族、稀疏、
    候选都相同），按全文逐位比的摘要在那两个平台上对不上（远端 CI，2026-11-09）。断言失败时把这份全文打出来，能直接看差在哪。"""

    lines = [encode_night(state), *(encode_transition(item) for item in transitions)]
    return "\n".join(
        json.dumps(_rounded(json.loads(line)), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        for line in lines
    )


def _stable_digest(state: LaneState, transitions: tuple[Transition, ...]) -> str:
    return hashlib.sha256(_stable_text(state, transitions).encode("utf-8")).hexdigest()


def _outcome(
    make: Callable[[], LaneTests],
    previous: LaneState | None = None,
    weights: Callable[[LaneTests], dict[RelationKey, float]] = _hashed,
    *,
    series: EventSeries,
) -> Outcome:
    """（不加权折叠的摘要, 加权折叠的摘要, 不加权的各段稀疏, 不加权的族, 加权的族）。``make`` 每次造一份新的 ``LaneTests``；
    ``series`` 是它的序列（判退休的观测材料从它取）。"""

    plain, plain_moves = fold(previous, make(), watching=watching(series))
    tests = make()
    weighted, weighted_moves = fold(previous, tests, weights(tests), watching=watching(series))
    return (
        _digest(plain, plain_moves),
        _digest(weighted, weighted_moves),
        dict(plain.sparse),
        plain.family,
        weighted.family,
    )


def _rows(state: LaneState, keys: set[RelationKey]) -> dict[str, str]:
    return {item.key.text: item.status.value for item in state.relations if item.key in keys}


#: 强关系那几天（对照组用）。
STRONG = frozenset(range(8, 40, 3))


# ── 条件判不了（None）的链：地点没给的日子 ──────────────────────────────────────

PLACE_PLANS = {
    "3-3": Plan(office=frozenset({1, 5, 9}), home=frozenset({13, 17, 21}), primed=frozenset({5, 30})),
    "3-2": Plan(office=frozenset({1, 5, 9}), home=frozenset({13, 17}), primed=frozenset({5, 30})),
    "2-3": Plan(office=frozenset({1, 5}), home=frozenset({13, 17, 21}), primed=frozenset({5, 30})),
    # 对照组：在公司讨论完必接修改代码，过门且成了候选；第 2–7 天合并后两边各 2 条（跳过）
    "strong": Plan(office=STRONG, home=frozenset(range(9, 40, 3)), follows=STRONG, primed=frozenset({5, 30})),
}
PLACE_EDGES = {
    "3-3": ((3, (3,)), (3, (3,))),
    "3-2": ((3, (2,)), (2, (3,))),
    "2-3": ((2, (3,)), (3, (2,))),
    "strong": ((11, (11,)), (11, (11,))),
}


@pytest.mark.parametrize("name", sorted(PLACE_PLANS))
def test_conditions_with_undecidable_chains_on_the_gate_edge_match_the_baseline(name: str) -> None:
    """地点是机械列举的条件（公司 / 家里），只有六七天给了地点，其余三十多天判不了、不进任何一组：可判的链恰好 3 / 2 条。
    整份关系表与迁移与基线逐字相同。"""

    plan = PLACE_PLANS[name]

    def make() -> LaneTests:
        return _tests(plan)

    tests = make()
    assert {OFFICE.text, HOME.text} <= {item.condition for item in tests.keys()}
    assert (_sides(tests, (OFFICE,), Segment.CHAIN), _sides(tests, (HOME,), Segment.CHAIN)) == PLACE_EDGES[name]
    if name == "strong":
        assert _sides(tests, (OFFICE,), Segment.DAYS_2_7) == (2, (2,))
        state, _moves = fold(None, make(), watching=watching(_series(plan)))
        assert _rows(state, {item for item in tests.keys() if item.condition}) == GOLDEN_STRONG_ROWS["place"]
    assert _outcome(make, series=_series(plan)) == GOLDEN_PLACE[name]


# ── 几个条件合起来的组合键（裁定 62 第 1 条）在门 1 边缘 ───────────────────────────

PARENT = RelationKey("session", DISCUSS, CODE, Segment.CHAIN, OFFICE.text)
#: 公司 × 推送：（都成立, 推送成立但不在公司, 在公司但没推送）的链数。
GROWN_PLANS = {
    "3-3-3": Plan(
        office=frozenset({1, 3, 5, 7, 9, 11}),
        home=frozenset({13, 15, 17, 19}),
        primed=frozenset({1, 3, 5, 13, 15, 17}),
    ),
    "2-3-3": Plan(
        office=frozenset({1, 3, 7, 9, 11}),
        home=frozenset({13, 15, 17, 19}),
        primed=frozenset({1, 3, 13, 15, 17}),
    ),
    "3-2-3": Plan(
        office=frozenset({1, 3, 5, 7, 9, 11}),
        home=frozenset({13, 15, 19}),
        primed=frozenset({1, 3, 5, 13, 15}),
    ),
    "3-3-2": Plan(
        office=frozenset({1, 3, 5, 7, 9}),
        home=frozenset({13, 15, 17, 19}),
        primed=frozenset({1, 3, 5, 13, 15, 17}),
    ),
}
GROWN_EDGES = {"3-3-3": (3, (3, 3)), "2-3-3": (2, (3, 3)), "3-2-3": (3, (2, 3)), "3-3-2": (3, (3, 2))}


def _with_parent(cutoff: date) -> LaneState:
    days = workdays(40)
    return LaneState(
        lane="session",
        night=cutoff - timedelta(days=1),
        family=0,
        relations=(
            Relation(
                key=PARENT,
                status=Status.ESTABLISHED,
                upward=True,
                discovered_on=days[5],
                forward_passed_on=days[15],
                established_on=days[15],
                maintained_through=days[30],
                maintenance_needed=4,
            ),
        ),
        measure_versions=MEASURE_VERSIONS,
    )


@pytest.mark.parametrize("name", sorted(GROWN_PLANS))
def test_a_grown_combination_on_the_gate_edge_matches_the_baseline(name: str) -> None:
    """「在公司」那条已成立，第二晚在它上面长出「在公司 + 之前推送过」：条件全成立与两组"少一个"的链数分别卡在 3 / 2。
    组合键不是跨夜带着的，可以走捷径；父关系是带着的，必须完整读数。整份关系表与迁移与基线逐字相同。"""

    plan = GROWN_PLANS[name]
    series = _series(plan)
    previous = _with_parent(series.cutoff)

    def make() -> LaneTests:
        return _tests(plan, established=(PARENT,))

    tests = make()
    grown = [item for item in tests.keys() if len(item.conditions) == 2]
    assert grown and {item.condition for item in grown} >= {f"{OFFICE.text}+{PUSHED.text}"}
    assert _sides(tests, (OFFICE, PUSHED), Segment.CHAIN) == GROWN_EDGES[name]
    assert _outcome(make, previous, series=series) == GOLDEN_GROWN[name]


# ── 非同一条链段：合并之后链比原来少 ─────────────────────────────────────────────

#: 一半日子讨论两次；推送有时在第一次之前、有时只在第二次之前——同一条链段数到的链，到了当天剩下 / 次日合并后只剩锚
#: （当天第一次）那一条，条件只在锚上判。
MERGED_PLANS = {
    "4-to-3": Plan(
        doubled=frozenset(range(0, 40, 2)),
        primed=frozenset({4, 14, 24}),
        late_primed=frozenset({8}),
    ),
    "4-to-2": Plan(
        doubled=frozenset(range(0, 40, 2)),
        primed=frozenset({4, 24}),
        late_primed=frozenset({8, 30}),
    ),
}
MERGED_EDGES = {
    "4-to-3": {Segment.CHAIN: 4, Segment.REST_OF_DAY: 3, Segment.NEXT_DAY: 3},
    "4-to-2": {Segment.CHAIN: 4, Segment.REST_OF_DAY: 2, Segment.NEXT_DAY: 2},
}


@pytest.mark.parametrize("name", sorted(MERGED_PLANS))
def test_segments_whose_merged_chains_fall_to_the_gate_edge_match_the_baseline(name: str) -> None:
    """同一条链段里"之前推送过"有 4 条链（过门），当天剩下与次日合并后只剩 3 / 2 条（卡在门上 / 差一条）。按段数链，
    不按同一条链段的链数推。整份关系表与迁移与基线逐字相同。"""

    plan = MERGED_PLANS[name]

    def make() -> LaneTests:
        return _tests(plan)

    tests = make()
    found = {segment: _sides(tests, (PUSHED,), segment)[0] for segment in MERGED_EDGES[name]}
    assert found == MERGED_EDGES[name]
    assert _outcome(make, series=_series(plan)) == GOLDEN_MERGED[name]


# ── 回退：跳过的键在加权归一之后可能进族，必须逐条量回来 ─────────────────────────────


def test_skipped_keys_that_could_enter_the_family_after_normalisation_are_measured_like_the_baseline() -> None:
    """不带条件的键（十几个类两两 × 四段）先验权重都极小，带条件的只有"讨论方案｜之前推送过"一组、每边链不够：
    权重归一之后这组键的有效权重被抬到 1 / q 以上（"最小可达 p / 权重 ≤ q"不再必不成立），不能再当"必不进族"略过，
    得逐条量回来。结果与基线逐字相同。"""

    plan = Plan(primed=frozenset({4, 19}), code_after=60, rare=10)

    def make() -> LaneTests:
        return _tests(plan)

    def tiny(tests: LaneTests) -> dict[RelationKey, float]:
        return {item: 0.001 for item in tests.keys() if not item.condition}

    tests = make()
    keys = set(tests.keys())
    moderated = {item for item in keys if item.condition}
    assert moderated and {item.condition for item in moderated} == {PUSHED.text}
    assert all(_sides(tests, (PUSHED,), segment)[0] < 3 for segment in {item.segment for item in moderated})
    # 归一系数下这组"权重 1、最小可达 p 1"的键比值 ≤ q：可能进族
    short = [item for item in keys if not item.segment.long]
    total = sum(0.001 if not item.condition else 1.0 for item in short)
    assert total / len(short) <= CONFIG.thresholds.fdr
    assert _outcome(make, weights=tiny, series=_series(plan)) == GOLDEN_FALLBACK


# ── 调用方给跳过的带条件键非 1 的权重；``night`` 直接传 ``outside`` 不传 ``keep`` ─────────────────


@pytest.mark.parametrize(("hopeless", "plain_rest"), [(0.1, True), (0.1, False), (10.0, True), (10.0, False)])
def test_weights_given_to_skipped_moderated_keys_are_normalised_like_the_baseline(
    hopeless: float, plain_rest: bool
) -> None:
    """runtime 只给不带条件的键权重，但 ``fold`` 接受任意权重：给"讨论方案｜之前推送过"那 12 条门 1 整组到不了的键
    非 1 的权重（另外不带条件的按 1 或都给 0.5），它们按各自的权重留在归一里，结果与基线逐字相同。"""

    tests = exact._tests(2)
    group = sorted(item for item in tests.keys() if item.antecedent == DISCUSS and item.condition == exact.PUSHED)
    assert len(group) == 12
    weights = {} if plain_rest else exact._uniform(tests, 0.5)
    weights.update({item: hopeless for item in group})
    state, transitions = fold(None, tests, weights, watching=watching(exact._series(2)))
    assert (state.family, dict(state.sparse), _digest(state, transitions)) == GOLDEN_HOPELESS_WEIGHTS[
        (hopeless, plain_rest)
    ]


def test_outside_moderated_keys_keep_a_full_reading_without_keep() -> None:
    """``LaneTests.night`` 直接传 ``outside``、不传 ``keep``：outside 里两边链不够的带条件键仍要完整读数（units > 0 的判
    TESTED），与基线逐字相同。"""

    tests = exact._tests(2)
    target = RelationKey("session", DISCUSS, CODE, Segment.CHAIN, exact.PUSHED)
    night = tests.night(outside=frozenset({target}))
    found = [
        (test.verdict.value, test.antecedents, test.blocks, test.p_value, test.minimum_p)
        for test in night.tests
        if test.key == target
    ]
    assert found == GOLDEN_OUTSIDE_READING
    assert night.family == GOLDEN_OUTSIDE_FAMILY


# ── 父关系过了第 1 道：跳过的带条件键进了带条件那一族的归一（裁定 95） ───────────────────────

#: 「在公司」那份对照组数据（讨论方案 → 修改代码当晚过第 1 道），"之前推送过"只有 2 条链（门 1 整组到不了）。
ADMITTED = replace(PLACE_PLANS["strong"], people=3)
#: 同一份数据加 20 个同事：讨论方案 → 修改代码底下多出二十来条量得了的"｜某同事在"。
CROWDED = replace(PLACE_PLANS["strong"], people=20)
PUSHED_CHILD = RelationKey("session", DISCUSS, CODE, Segment.CHAIN, PUSHED.text)


def _admitted_premise(tests: LaneTests) -> list[RelationKey]:
    """前提：父关系（讨论方案 → 修改代码 同一条链）不加权当晚过第 1 道；它底下"之前推送过"那条两边链不够（整组到不了）。
    返回"讨论方案｜之前推送过"那一组键。"""

    parent = RelationKey("session", DISCUSS, CODE, Segment.CHAIN, "")
    assert {test.key: test.verdict.value for test in tests.night().tests}[parent] == "significant"
    assert _sides(tests, (PUSHED,), Segment.CHAIN)[0] < 3
    group = sorted(item for item in tests.keys() if item.antecedent == DISCUSS and item.condition == PUSHED.text)
    assert PUSHED_CHILD in group
    return group


@pytest.mark.parametrize("hopeless", [0.1, 10.0])
def test_skipped_keys_under_a_passing_parent_count_in_the_moderated_normalisation(hopeless: float) -> None:
    """父关系当晚过了第 1 道，"讨论方案 → 修改代码｜之前推送过"放得进带条件那一族，但门 1 整组到不了、不逐条量。给那一组
    0.1 或 10 的权重：它按自己的权重计入带条件那一族的归一，把"｜在公司""｜某同事在"这些量了的键的有效权重抬高或压低，
    族的大小随之不同（0.1 时 6、10 时 5）。拿掉它的归一（只按量了的键归一），两种权重的结果就一样了。结果与基线逐字相同。"""

    tests = _tests(ADMITTED)
    group = _admitted_premise(tests)
    state, transitions = fold(
        None, _tests(ADMITTED), {item: hopeless for item in group}, watching=watching(_series(ADMITTED))
    )
    family, sparse, digest = GOLDEN_ADMITTED_WEIGHTS[hopeless]
    assert (state.family, dict(state.sparse)) == (family, sparse)
    assert _stable_digest(state, transitions) == digest, _stable_text(state, transitions)
    assert GOLDEN_ADMITTED_WEIGHTS[0.1] != GOLDEN_ADMITTED_WEIGHTS[10.0]  # 前提：归一真的改了结果


def test_a_skipped_key_that_may_enter_the_moderated_family_is_measured_like_the_baseline() -> None:
    """父关系当晚过了第 1 道；底下二十来条量了的带条件键调用方都给 0.001，"｜之前推送过"那一组按 1。带条件那一族归一之后，
    "讨论方案 → 修改代码｜之前推送过"的有效权重约 22.5、比值约 0.044 ≤ q（0.1）：它可能进族，不能当"必不进族"略过，得逐条量回来——
    当晚它进了族（判"没过"，不是稀疏）。结果与基线逐字相同。"""

    tests = _tests(CROWDED)
    _admitted_premise(tests)
    weights = {item: 0.001 for item in tests.keys() if item.condition and item.condition != PUSHED.text}
    night = _tests(CROWDED).night(weights)
    verdicts = {test.key: test.verdict.value for test in night.tests}
    assert verdicts[PUSHED_CHILD] == "tested"
    state, transitions = fold(None, _tests(CROWDED), weights, watching=watching(_series(CROWDED)))
    assert (night.family, state.family, dict(state.sparse), _digest(state, transitions)) == GOLDEN_MAY_ENTER


# ── 基线（7dffb51 + 裁定 95 分族层级与 V2.2、不含 E1–E5）上跑出的期望值（生成脚本：V22实现/regen/regen.py）

#: （不加权折叠的摘要, 加权折叠的摘要, 不加权的各段稀疏, 不加权的族, 加权的族）
GOLDEN_STRONG_ROWS: dict[str, dict[str, str]] = {
    "place": {"session|s-k839254|s-k778238|chain|field:place=公司": "candidate"},
}
GOLDEN_PLACE: dict[str, Outcome] = {
    "2-3": (
        "82165d92fea375cab72908c995916c396579617c2b5750cd3b411ab0fa0e8fc7",
        "9856fa03bc08185d6f73c934fd84dda8bf687699fb391f8bb2b6e0a42e619b5f",
        {"chain": 33, "days_2_7": 34, "next_day": 33, "rest_of_day": 34},
        2,
        2,
    ),
    "3-2": (
        "82165d92fea375cab72908c995916c396579617c2b5750cd3b411ab0fa0e8fc7",
        "9856fa03bc08185d6f73c934fd84dda8bf687699fb391f8bb2b6e0a42e619b5f",
        {"chain": 33, "days_2_7": 34, "next_day": 33, "rest_of_day": 34},
        2,
        2,
    ),
    "3-3": (
        "82165d92fea375cab72908c995916c396579617c2b5750cd3b411ab0fa0e8fc7",
        "9856fa03bc08185d6f73c934fd84dda8bf687699fb391f8bb2b6e0a42e619b5f",
        {"chain": 33, "days_2_7": 34, "next_day": 33, "rest_of_day": 34},
        2,
        2,
    ),
    "strong": (
        "f8ba0fa4e002f449689c0e51d9e74a68f03908f2eda5de1bf222c3b97415a328",
        "662ac666e1887e68d65e2e684c0b31d02b0bf46f008eea4be41d813049e18ca9",
        {"chain": 29, "days_2_7": 30, "next_day": 31, "rest_of_day": 30},
        4,
        4,
    ),
}
GOLDEN_GROWN: dict[str, Outcome] = {
    "2-3-3": (
        "82b4cf16f130cba2fd49a1b7cd5ccd6d911fa43a3222d8fed0294c825558eb98",
        "27780a9431de3c3c403cbd3888a9bc8b91a27a0d205875f7062ad72ef6da3f9d",
        {"chain": 31, "days_2_7": 34, "high_state": 3, "next_day": 32, "rest_of_day": 33},
        8,
        7,
    ),
    "3-2-3": (
        "1bbcd3b42f7095b53315a39238cdb2466e42c9af8e899c6ff3a3c3571d1a0ed8",
        "77b847d9dfda4020b6c6296c1808d35eb73f30bb89c0b3425343f5c7ee3ddbe6",
        {"chain": 31, "days_2_7": 34, "high_state": 3, "next_day": 33, "rest_of_day": 34},
        6,
        6,
    ),
    "3-3-2": (
        "78ce83e6401c0e9119d0cb0172e68af88d86fc4c82c460208188d392331c3e9d",
        "19c463c1899fbd5f064b68ba7c41cc285b3559c6e0f148978bbdb7d4833db3de",
        {"chain": 31, "days_2_7": 34, "high_state": 3, "next_day": 32, "rest_of_day": 33},
        8,
        8,
    ),
    "3-3-3": (
        "6306b0d1387e60a09b190d61a97e3df94a889dc97b01aebf30e940a735f8cfff",
        "058a438838e0e413280cd062740d389398cfa794da02729b9a686d2750eb4bc5",
        {"chain": 31, "days_2_7": 34, "high_state": 3, "next_day": 32, "rest_of_day": 33},
        8,
        8,
    ),
}
GOLDEN_MERGED: dict[str, Outcome] = {
    "4-to-2": (
        "f3ccc1cdd3fe789e74e38209e1b2375caa5b3867fa004b35727002862ddd0267",
        "d9578d563a72fd2e8ca1f13613b4db4de473018f788b0667f960462035dd9bfe",
        {"chain": 23, "days_2_7": 26, "next_day": 26, "rest_of_day": 25},
        4,
        4,
    ),
    "4-to-3": (
        "587ff8015e57cec5c2a27461d805c38ea90c5e104dc6a1072722bb9945036b0f",
        "27a102a29c2d2a501d6009f25533746cb904c23ce0aa87f404240835a2a6d828",
        {"chain": 23, "days_2_7": 26, "next_day": 25, "rest_of_day": 25},
        5,
        5,
    ),
}
GOLDEN_FALLBACK: Outcome = (
    "07993f9d31564e3a78a131fe3c954fa7a1a8e5e566ec47439367957df8e074d5",
    "297e76a2031edc52f1c9055d64f098400df674bdf2c42bce7491b33de996e81f",
    {"chain": 222, "days_2_7": 223, "next_day": 220, "rest_of_day": 223},
    4,
    4,
)
GOLDEN_HOPELESS_WEIGHTS: dict[tuple[float, bool], tuple[int, dict[str, int], str]] = {
    (0.1, True): (
        5,
        {"chain": 42, "days_2_7": 47, "next_day": 47, "rest_of_day": 47},
        "398f69525e012108a09de2e91ea7bd1757451b5cf9198eeedae5126a0742abfc",
    ),
    (0.1, False): (
        5,
        {"chain": 42, "days_2_7": 47, "next_day": 47, "rest_of_day": 47},
        "51cca9a1689e32a0bf24a8598c7dbb94e4883841f183caf668e70bd383853df8",
    ),
    (10.0, True): (
        5,
        {"chain": 42, "days_2_7": 47, "next_day": 47, "rest_of_day": 47},
        "398f69525e012108a09de2e91ea7bd1757451b5cf9198eeedae5126a0742abfc",
    ),
    (10.0, False): (
        5,
        {"chain": 42, "days_2_7": 47, "next_day": 47, "rest_of_day": 47},
        "51cca9a1689e32a0bf24a8598c7dbb94e4883841f183caf668e70bd383853df8",
    ),
}
GOLDEN_OUTSIDE_READING: list[tuple[str, int, int, float, float]] = [("tested", 40, 40, 1.0, 1.0)]
GOLDEN_OUTSIDE_FAMILY: int = 5
#: 摘要是 ``_stable_digest``（浮点取 9 位有效数字）：这两份带权重的情形跨平台只差浮点末位。
GOLDEN_ADMITTED_WEIGHTS: dict[float, tuple[int, dict[str, int], str]] = {
    0.1: (
        6,
        {"chain": 75, "days_2_7": 78, "next_day": 79, "rest_of_day": 78},
        "9567efd6c139fe71a3e2077a10d4062d5b72ab7063eecf381a1836fd2b921648",
    ),
    10.0: (
        5,
        {"chain": 76, "days_2_7": 78, "next_day": 79, "rest_of_day": 78},
        "b452f1eb58ec1b727f72a80cf48c738275329902a6ee42753abfe73a9297ed21",
    ),
}
#: （当晚加权的族, 折叠后的族, 各段稀疏, 摘要）
GOLDEN_MAY_ENTER: tuple[int, int, dict[str, int], str] = (
    5,
    5,
    {"chain": 348, "days_2_7": 350, "next_day": 351, "rest_of_day": 350},
    "3398b8f39176cd7e62577da89a3260cb0f2d0797de5a9feee9298089acb49deb",
)


# ── 机械列举的结构字段条件在门 1 边缘：门 1 数的链与量的链是同一批全部历史 ─────────────────


@pytest.mark.parametrize("name", ["3-3", "3-2", "2-3"])
def test_a_mechanical_place_condition_on_the_gate_edge_keeps_the_full_history(name: str) -> None:
    """可判的链恰好卡在门 1 边上（3 / 2 条）的组上，"在公司"是机械列举的条件（链上公司、家里两面都有），门 1 数的链与量的链
    是同一批全部历史。整份关系表与迁移与冻住的金样逐字相同。（原来这一族守"模型很晚才提、与机械列举撞上的条件不截成提出之后"，
    问模型提条件那一路 2026-11-15 删了，这里只留机械列举那一半。）"""

    plan = PLACE_PLANS[name]
    series = _series(plan)
    concepts = _concepts(plan)

    def build() -> LaneTests:
        line = build_timelines(series, concepts, {}, CONFIG)["session"]
        return LaneTests(line, concepts, CONFIG, series.cutoff)

    tests = build()
    group = {item for item in tests.keys() if item.condition == OFFICE.text}
    assert group and all(item.antecedent == DISCUSS for item in group)
    assert _sides(tests, (OFFICE,), Segment.CHAIN) == PLACE_EDGES[name][0]
    assert _outcome(build, series=series) == GOLDEN_PLACE[name]
