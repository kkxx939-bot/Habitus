"""语义树的门槛按 α 与功效定（裁定 60 第 9 条；26 号清单 5.2–5.4）：前向验证没有"至少 5 次"的下限；维持检验每段要攒几次
成立那晚按"回落到对照率"算好；模拟里效应恒定的关系不比旧规则（8 块一段）更容易被判失效，真消失了几段之内判出来。"""

from __future__ import annotations

import random
from datetime import date
from types import SimpleNamespace

import pytest

from habitus.scene.relations import RelationKey, Segment, stats
from habitus.scene.relations.engine import RelationTest, Subset, Verdict
from habitus.scene.relations.state import Relation, Status, _candidate, _forward_needed, _maintenance_needed
from tests.unit.scene.relation_fixtures import CONFIG

TESTS = SimpleNamespace(config=CONFIG)
KEY = RelationKey("session", "a", "b", Segment.CHAIN)


def discovered(
    treated: float, control: float, *, interval: tuple[float, float] | None, upward: bool = True
) -> RelationTest:
    return RelationTest(
        key=KEY,
        verdict=Verdict.SIGNIFICANT,
        antecedents=20,
        blocks=6,
        block_days=7.0,
        effect=stats.Effect(20, treated, control),
        p_value=0.001,
        minimum_p=0.0001,
        upward=upward,
        interval=interval,
    )


def test_forward_validation_needs_only_what_its_power_asks_for() -> None:
    """发现期全部阳性（做了 A 都做了 B、平时一成）：两次就够，不再被"至少 5 次"拖着；九成对一成、区间近端 0.61–0.71 的
    要 3–5 次（离 0 近的一端越近要得越多）。"""

    assert _forward_needed(discovered(1.0, 0.1, interval=(0.8, 1.0)), TESTS) == 2  # type: ignore[arg-type]
    assert _forward_needed(discovered(0.9, 0.1, interval=(0.71, 0.9)), TESTS) == 3  # type: ignore[arg-type]
    assert _forward_needed(discovered(0.9, 0.1, interval=(0.61, 0.9)), TESTS) == 5  # type: ignore[arg-type]
    assert (
        _forward_needed(discovered(0.12, 0.1, interval=(-0.1, 0.2)), TESTS) == CONFIG.thresholds.forward_max_antecedents
    )  # type: ignore[arg-type]


def _forward(antecedents: int, *, p_value: float, minimum_p: float) -> Subset:
    return Subset(antecedents, 2, stats.Effect(antecedents, 1.0, 0.0), p_value, None, None, minimum_p)


def test_forward_validation_waits_while_even_all_hits_could_not_reach_alpha() -> None:
    """要攒的次数到了，但每层只有 2 个对照、全中也只有 (1/3)² ≈ 0.11：判了必输，再攒、不判（L10.15，27 号调研）；
    全中能到 α 的那一晚照常判，不过就退回。"""

    before = Relation(KEY, Status.CANDIDATE, True, discovered_on=date(2026, 10, 1), forward_needed=2)
    test = discovered(1.0, 0.1, interval=(0.8, 1.0))

    def night(forward: Subset) -> tuple[Relation, str | None]:
        tests = SimpleNamespace(config=CONFIG, assess=lambda item, measure=None: item, evaluate=lambda *a, **k: forward)
        return _candidate(before, test, tests, date(2026, 10, 9))  # type: ignore[arg-type]

    waiting, note = night(_forward(2, p_value=(1 / 3) ** 2, minimum_p=(1 / 3) ** 2))
    assert waiting.status is Status.CANDIDATE and note is None

    judged, note = night(_forward(3, p_value=0.2, minimum_p=(1 / 3) ** 3))
    assert judged.status is Status.REJECTED and note is not None and "没复现" in note


def test_maintenance_asks_how_many_it_takes_to_see_a_fall_back_to_the_control_rate() -> None:
    """零假设"效应还像成立那晚"、备择"回落到对照率"，方向朝回落：九成对一成两次就看得出回落，六成对两成要多攒几次；压低的
    关系反过来朝上看。"""

    strong = _maintenance_needed(discovered(0.9, 0.1, interval=(0.7, 0.9)), TESTS)  # type: ignore[arg-type]
    weak = _maintenance_needed(discovered(0.6, 0.2, interval=(0.2, 0.6)), TESTS)  # type: ignore[arg-type]
    down = _maintenance_needed(discovered(0.05, 0.5, interval=(-0.6, -0.3), upward=False), TESTS)  # type: ignore[arg-type]
    assert strong == stats.required_antecedents(0.9, 0.8, upward=False, alpha=0.05, power=0.8, minimum=1, maximum=40)
    assert 1 <= strong < weak <= CONFIG.thresholds.forward_max_antecedents
    assert down == stats.required_antecedents(0.05, 0.45, upward=True, alpha=0.05, power=0.8, minimum=1, maximum=40)


# --- 模拟（26 号清单 5.4）：纯函数生成每次前因之后后果来没来，按两种分段规则走维持检验 ---------------------------------


def _segments(outcomes: list[bool], *, size: int):  # type: ignore[no-untyped-def]
    for start in range(0, len(outcomes) - size + 1, size):
        yield outcomes[start : start + size]


def _expired(outcomes: list[bool], *, size: int, established: float, control: float) -> int | None:
    """连续两段都说"没了"（这一段的发生率按 95% 区间够不上成立那晚效应的一半）就失效；返回失效在第几段（没失效为 None）。
    区间与 ``gone()`` 同一个意思：上端离对照率不到效应的一半就是"影响不够大了"。"""

    misses = 0
    for index, segment in enumerate(_segments(outcomes, size=size)):
        rate = sum(segment) / len(segment)
        half = 1.96 * (rate * (1 - rate) / len(segment)) ** 0.5
        if rate + half < control + (established - control) / 2:
            misses += 1
        elif rate - half > control:
            misses = 0
        if misses >= 2:
            return index
    return None


@pytest.mark.parametrize("established, control", [(0.9, 0.1), (0.7, 0.2), (0.6, 0.2)])
def test_a_constant_effect_is_no_more_often_expired_than_under_the_old_eight_block_rule(
    established: float, control: float
) -> None:
    generator = random.Random(20261019)
    size = max(
        _maintenance_needed(discovered(established, control, interval=None), TESTS),  # type: ignore[arg-type]
        2,
    )
    old = 8  # 旧规则：每段 8 块（每块一次前因）
    runs = 400
    new_expired = old_expired = 0
    for _ in range(runs):
        outcomes = [generator.random() < established for _ in range(200)]
        new_expired += _expired(outcomes, size=size, established=established, control=control) is not None
        old_expired += _expired(outcomes, size=old, established=established, control=control) is not None
    assert new_expired <= old_expired + runs * 0.02


@pytest.mark.parametrize("established, control", [(0.9, 0.1), (0.7, 0.2)])
def test_a_relation_that_really_disappears_is_expired_within_a_few_segments(established: float, control: float) -> None:
    generator = random.Random(7)
    size = max(_maintenance_needed(discovered(established, control, interval=None), TESTS), 2)  # type: ignore[arg-type]
    late = []
    for _ in range(200):
        outcomes = [generator.random() < control for _ in range(size * 10)]  # 成立之后效应整个没了
        found = _expired(outcomes, size=size, established=established, control=control)
        late.append(found if found is not None else 99)
    assert sorted(late)[int(len(late) * 0.9)] <= 5  # 九成在 6 段之内判出来（效应 ≥ 0.5 的；九成对一成的 4 段之内）
