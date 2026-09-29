"""触点②：分型与逐峰展开由节律定、零条是合法答复、穷举被闸挡住、失败这一轮不写。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from habitus.scene.concepts.rhythm import Rhythm, RhythmPeak
from habitus.scene.hypotheses.author import (
    MAX_DRAFTS_PER_CONSEQUENT,
    HypothesisAuthor,
    HypothesisAuthorError,
    assemble_hypotheses,
    build_hypothesis_request,
    hypothesis_author_json_schema,
)
from habitus.scene.hypotheses.model import Aspect, HypothesisOrigin
from tests.unit.foresight.scripted_model import recording_client
from tests.unit.scene.concept_fixtures import concept_set

NOW = datetime(2026, 9, 28, 3, 0, tzinfo=UTC)
CONCEPTS = concept_set()
#: 早餐一天一个机会；把它当"一天三个机会"的后件用来验逐峰展开（真实咖啡的形状）。
ONE_PEAK = Rhythm("早餐", (RhythmPeak(1, 420, 510, 0.8),), 7, 24.0)
THREE_PEAKS = Rhythm(
    "早餐",
    (RhythmPeak(1, 420, 510, 0.55), RhythmPeak(2, 780, 840, 0.40), RhythmPeak(3, 1200, 1260, 0.30)),
    7,
    8.0,
)
#: 打球只在周末冒头 → 无节律型（不数机会、没有时效）。
NO_RHYTHM = Rhythm("打球", (RhythmPeak(1, 900, 990, 0.4),), 2, None)

def draft(**overrides: object) -> dict[str, object]:
    """一条答复。**每个键都要有**：schema 是严格模式的（`required` 列全部属性），"可选"用 null / [] 表达。"""

    entry: dict[str, object] = {
        "antecedents": [{"concept": "晚睡", "grade": "重"}],
        "aspect": "probability",
        "direction": "down",
        "type_prior": "inhibiting",
        "expected_at": 1,
        "released_by": [],
        "split_by": [],
        "why": "前一晚睡得很晚，早上会睡过早餐的时间",
    }
    entry.update(overrides)
    return entry


LATE_TO_BREAKFAST: dict[str, object] = {"hypotheses": [draft()]}


def test_a_rhythmic_consequent_gets_one_account_per_peak() -> None:
    """一天三个咖啡机会 = 三条假设、三本账：三个峰的本来概率不同（0.55 / 0.40 / 0.30），混一本就读不出东西。"""

    hypotheses, signals = assemble_hypotheses(LATE_TO_BREAKFAST, CONCEPTS, "早餐", THREE_PEAKS, now=NOW)
    assert [item.expected_at for item in hypotheses] == [1, 2, 3]
    assert len({item.identity for item in hypotheses}) == 3
    assert all(item.identity.endswith(f"--probability--{item.expected_at}") for item in hypotheses)
    assert all(item.aspect is Aspect.PROBABILITY and item.horizon == 1 and not item.released_by for item in hypotheses)
    assert any("逐峰各一条" in signal for signal in signals)
    # 模型的猜只进理由，不定形状。
    assert "基准猜第 1 次机会" in hypotheses[0].note

    single, _ = assemble_hypotheses(LATE_TO_BREAKFAST, CONCEPTS, "早餐", ONE_PEAK, now=NOW)
    assert [item.expected_at for item in single] == [1]


def test_the_count_aspect_takes_one_account_whose_horizon_is_the_whole_day() -> None:
    """次数方面不分峰："晚睡当天咖啡喝几杯" = 一条，地平线 = 一天的机会数。"""

    answer = {
        "hypotheses": [
            draft(
                antecedents=[{"concept": "晚睡", "grade": None}],
                aspect="count",
                direction="up",
                type_prior="none",
                expected_at=None,
                why="困了会多喝几次",
            )
        ]
    }
    (hypothesis,), _signals = assemble_hypotheses(answer, CONCEPTS, "早餐", THREE_PEAKS, now=NOW)
    assert hypothesis.expected_at == 1 and hypothesis.horizon == 3 and "说不准" in hypothesis.note


def test_a_consequent_without_a_rhythm_becomes_one_open_ended_account() -> None:
    """无节律型：不数机会、没有时效，只有"来了"和 released_by 两种结法；模型猜的机会数被报出来。"""

    answer = {
        "hypotheses": [
            draft(
                antecedents=[{"concept": "晚睡", "grade": None}],
                direction="up",
                type_prior="promoting",
                expected_at=2,
                released_by=["早餐"],
                why="熬夜之后想动一动",
            )
        ]
    }
    (hypothesis,), signals = assemble_hypotheses(answer, CONCEPTS, "打球", NO_RHYTHM, now=NOW)
    assert hypothesis.is_open_ended and hypothesis.expected_at is None and hypothesis.released_by == ("早餐",)
    assert hypothesis.identity.endswith("--probability--open")
    assert any("树上没节律" in signal for signal in signals)


def test_timing_and_count_are_refused_on_a_consequent_with_no_rhythm() -> None:
    """时刻与次数量的是"哪一次"，后件没有机会序列就无从量起——报回去让模型改，不静默改成别的量。"""

    answer = {"hypotheses": [draft(antecedents=[{"concept": "晚睡", "grade": None}], aspect="timing", direction="up", type_prior="none", why="累了会拖后")]}
    with pytest.raises(ValueError, match="cannot be measured"):
        assemble_hypotheses(answer, CONCEPTS, "打球", NO_RHYTHM, now=NOW)


def test_an_out_of_range_guess_is_reported_but_the_expansion_goes_on() -> None:
    """A4："模型写的 expected_at 与节律对不上就报出来"——报出来，不丢掉这条假设。"""

    answer = {"hypotheses": [draft(expected_at=9)]}
    hypotheses, signals = assemble_hypotheses(answer, CONCEPTS, "早餐", ONE_PEAK, now=NOW)
    assert [item.expected_at for item in hypotheses] == [1]
    assert any("一天只有 1 个" in signal for signal in signals)


def test_zero_hypotheses_is_a_legitimate_answer() -> None:
    """安慰剂输入（编出来的无关配对）上它就该一条都不给；这不是失败。"""

    hypotheses, signals = assemble_hypotheses({"hypotheses": []}, CONCEPTS, "早餐", ONE_PEAK, now=NOW)
    assert hypotheses == () and any("一条都没写" in signal for signal in signals)


def test_the_algorithm_caps_its_own_fan_out_instead_of_blaming_the_model() -> None:
    """一个后件最多 24 本账，而"逐峰各一条"是**算法**的展开——超了就截断并报出来。

    探针实测：4 组前因 × 12 个峰 = 48 本 → 老写法整轮拒掉，报错还写"少写几组前因"，
    把算法的展开算到模型头上（模型写 4 组前因本身没错）。
    """

    many = Rhythm("早餐", tuple(RhythmPeak(i, 60 * i, 60 * i + 30, 0.3) for i in range(1, 13)), 7, 2.0)
    sets = ([{"concept": "晚睡", "grade": None}], [{"concept": "打球", "grade": None}], [{"concept": "运动", "grade": None}],
            [{"concept": "晚睡", "grade": None}, {"concept": "出差中", "grade": None}])
    four = {"hypotheses": [draft(antecedents=items) for items in sets]}
    hypotheses, signals = assemble_hypotheses(four, CONCEPTS, "早餐", many, now=NOW)
    # 4 组 × 每组最多 24//4=6 次机会 = 24 本，正好压在上限上
    assert len(hypotheses) == 24 and len({item.identity for item in hypotheses}) == 24
    assert max(item.expected_at or 0 for item in hypotheses) == 6
    assert any("只建前 6 次" in signal for signal in signals)


def test_enumerating_the_concept_set_is_refused() -> None:
    """不许穷举连线——门槛能定 3 的前提就挂在这上面（穷举时"说有影响"里九成是假的）。"""

    one = draft()
    answer = {"hypotheses": [one] * (MAX_DRAFTS_PER_CONSEQUENT + 1)}
    with pytest.raises(ValueError, match="at most"):
        assemble_hypotheses(answer, CONCEPTS, "早餐", ONE_PEAK, now=NOW)
    # 两条展开成同一身份（同一前件集合、同一方面）：一本账混两种量，拒。
    with pytest.raises(ValueError, match="share one identity"):
        assemble_hypotheses({"hypotheses": [one, one]}, CONCEPTS, "早餐", ONE_PEAK, now=NOW)


def test_a_tautology_or_an_unknown_grade_does_not_pass_the_concept_set() -> None:
    """「打球 → 运动」这种重言、以及没定义过的档，由 ``validate_against`` 挡住。"""

    answer = {"hypotheses": [draft(antecedents=[{"concept": "打球", "grade": None}], direction="up", type_prior="promoting", why="重言")]}
    with pytest.raises(ValueError, match="ancestor chain"):
        assemble_hypotheses(answer, CONCEPTS, "运动", ONE_PEAK, now=NOW)
    wrong_grade = {"hypotheses": [draft(antecedents=[{"concept": "晚睡", "grade": "中"}])]}
    with pytest.raises(ValueError, match="grade"):
        assemble_hypotheses(wrong_grade, CONCEPTS, "早餐", ONE_PEAK, now=NOW)


def test_the_prompt_and_schema_pin_the_concept_names_and_show_the_rhythm() -> None:
    request = build_hypothesis_request(CONCEPTS, "早餐", {"早餐": THREE_PEAKS, "晚睡": Rhythm("晚睡", (RhythmPeak(1, 1380, 1440, 0.6),), 7, 24.0)})
    rendered = request.messages[-1].content or ""
    assert "## 后件（要找它的前因）" in rendered and "一天 3 个机会" in rendered
    assert "- 晚睡（behavior）：入睡时刻晚于常态两小时以上；档：轻/重" in rendered
    assert "早餐" not in rendered.split("## 可以当前件的概念")[1]  # 后件不在候选前件里
    schema = hypothesis_author_json_schema(CONCEPTS, "早餐")
    antecedents = schema["properties"]["hypotheses"]["items"]["properties"]["antecedents"]
    assert "早餐" not in antecedents["items"]["properties"]["concept"]["enum"]
    assert schema["properties"]["hypotheses"]["items"]["properties"]["split_by"]["items"]["enum"] == ["出差中", "周末"]
    with pytest.raises(HypothesisAuthorError, match="behaviour concept"):
        build_hypothesis_request(CONCEPTS, "出差中")


def test_a_failed_answer_writes_nothing_this_round() -> None:
    """与触点① 同一条纪律：传输 / 结构 / 核对任一不过，这一轮一条都不写、留信号。"""

    client, provider = recording_client([{"hypotheses": [draft(antecedents=[], why="空前件")]}])
    proposal = asyncio.run(HypothesisAuthor(client, clock=lambda: NOW).propose(CONCEPTS, "早餐", {"早餐": ONE_PEAK}))
    assert proposal.wrote_nothing and provider.calls == 2 and proposal.consequent == "早餐"
    assert any(signal.startswith("model: ") for signal in proposal.signals)

    good, good_provider = recording_client([LATE_TO_BREAKFAST])
    written = asyncio.run(
        HypothesisAuthor(good, clock=lambda: NOW).propose(CONCEPTS, "早餐", {"早餐": ONE_PEAK}, origin=HypothesisOrigin.NEW_CONCEPT)
    )
    assert good_provider.calls == 1 and len(written.hypotheses) == 1
    assert written.hypotheses[0].source.origin is HypothesisOrigin.NEW_CONCEPT
    assert any(signal.startswith("author: ") for signal in written.signals)
