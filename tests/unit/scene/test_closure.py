"""闭环：安慰剂那把尺子 · 触点③ 的材料不带数字 · 提的假设从写入日起攒账。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from habitus.scene.hypotheses.closure import (
    MAX_CLOSURE_HYPOTHESES,
    ClosureAuthor,
    ClosureError,
    DriftFact,
    SplitFact,
    Structure,
    assemble_closure,
    build_closure_request,
    closure_json_schema,
)
from habitus.scene.hypotheses.model import HypothesisOrigin
from habitus.scene.hypotheses.placebo import PLACEBO_NOTE, placebo_hypotheses, split_by_origin
from habitus.scene.views.placebo import MIN_PLACEBOS, PlaceboReport
from tests.unit.foresight.scripted_model import recording_client
from tests.unit.scene.ledger_fixtures import CONCEPTS, LATE_TO_BREAKFAST, LATE_TO_WAKE

NOW = datetime(2026, 9, 29, 3, 0, tzinfo=UTC)
SPLIT = SplitFact(
    consequent="早餐",
    antecedents=("晚睡",),
    situation="出差中",
    with_situation=("07-11 02:10 晚睡（就寝）", "07-11 09:40 起床（起床）"),
    without_situation=("07-12 23:30 就寝（就寝）", "07-13 07:30 早餐（吃了碗面）"),
)


def answer(**overrides: object) -> dict[str, object]:
    """缺省是一条**调节体**：情境「出差中」加进原来的前件集合 {晚睡}（设计稿七f-7 的写法）。"""

    entry: dict[str, object] = {
        "structure": "moderator",
        "antecedents": [{"concept": "出差中", "grade": None}, {"concept": "晚睡", "grade": None}],
        "consequent": "早餐",
        "aspect": "probability",
        "direction": "down",
        "type_prior": "inhibiting",
        "why": "出差那几天他连着熬夜，早上起不来",
    }
    entry.update(overrides)
    return {"explanation": "出差的那几天他都过了午夜才睡、第二天九点半才起；不出差那几天十一点半就睡了。", "hypotheses": [entry]}


# ── 安慰剂：尺子 ──────────────────────────────────────────────────────────


def test_a_placebo_swaps_the_antecedent_for_an_unrelated_behaviour_and_is_reproducible() -> None:
    """七c ⑭：前件换成与后件无关的行为。方向/方面/后果峰号/峰表抄真假设——尺子要和被量的东西同一把。"""

    real = [LATE_TO_BREAKFAST, LATE_TO_WAKE]
    made = placebo_hypotheses(CONCEPTS, real, now=NOW)
    assert made and all(item.source.origin is HypothesisOrigin.PLACEBO for item in made)
    assert all(item.note == PLACEBO_NOTE for item in made)
    for item in made:
        template = next(h for h in real if h.consequent_identity == item.consequent_identity)
        assert (item.aspect, item.direction, item.consequent_peak) == (template.aspect, template.direction, template.consequent_peak)
        assert item.windows[item.consequent_identity] == template.windows[template.consequent_identity] and all(a.peak is None for a in item.antecedents)
        # 前件确实与后件无关：不是真假设用过的、不在同一条祖先链上
        names = {a.identity for a in item.antecedents}
        assert not names & {a.identity for a in template.antecedents}
        assert not any(CONCEPTS.is_ancestor(name, item.consequent_identity) for name in names)
    # 可重放：同一套输入永远挑出同一条（安慰剂一旦写盘就攒账，每晚换一批就永远量不出东西）
    assert [item.identity for item in placebo_hypotheses(CONCEPTS, real, now=NOW)] == [item.identity for item in made]
    # 只补差额
    assert placebo_hypotheses(CONCEPTS, [*real, *made], now=NOW) == ()


def test_the_placebo_report_gives_a_reference_distribution_and_no_verdict() -> None:
    """裁定六：不按标签判真假。报告只给计数与按机会分档的参照分位（真假设超过安慰剂 p90 的有几条），不印"误报率 / 高出"的判词。"""

    from habitus.scene.views.placebo import StepReference

    assert PlaceboReport(0, 0, MIN_PLACEBOS - 1, 0).false_alarm_rate is None  # 不拿 0% 当结论
    report = PlaceboReport(
        real_tested=20,
        real_significant=12,
        placebo_tested=10,
        placebo_significant=4,
        references=(StepReference(step=1, placebo_tested=10, real_tested=20, placebo_p50=0.18, placebo_reference=0.5, real_above_reference=8),),
    )
    text = report.render()
    assert "判词待裁定六的数值" in text and "p50 +18pp" in text and "p90 +50pp" in text and "超过 p90 的 8 条（碰巧该约 2.0 条）" in text
    assert "误报率" not in text and "高出" not in text and "没起作用" not in text


def test_placebos_are_kept_out_of_the_human_views() -> None:
    """安慰剂不是读数是尺子：印进 behaviours/ 会被当成真因果读。"""

    made = placebo_hypotheses(CONCEPTS, [LATE_TO_BREAKFAST], now=NOW)
    table = {item.identity: item for item in (LATE_TO_BREAKFAST, *made)}
    real, placebo = split_by_origin(table)
    assert set(real) == {LATE_TO_BREAKFAST.identity} and len(placebo) == len(made)


# ── 触点③：材料不带数字 ─────────────────────────────────────────────────


def test_the_closure_prompt_carries_the_narrative_and_no_numbers() -> None:
    """B8：只给"这两层不一样"这个事实 + 两边的叙事，不给强度、区间、兑现率、样本数。"""

    rendered = build_closure_request(SPLIT, CONCEPTS).messages[-1].content or ""
    assert "按「出差中」分成两边之后，两边**明显不一样**" in rendered
    assert "07-11 02:10 晚睡（就寝）" in rendered and "07-13 07:30 早餐（吃了碗面）" in rendered
    for forbidden in ("pp", "n=", "区间", "兑现率", "%"):
        assert forbidden not in rendered, forbidden
    # 一边没有叙事就问不出东西
    with pytest.raises(ClosureError, match="both layers"):
        SplitFact(consequent="早餐", antecedents=("晚睡",), situation="出差中", with_situation=("x",))


def test_a_drift_fact_gives_the_direction_but_not_the_size() -> None:
    drift = DriftFact(concept="就寝", quantity="时刻", direction="后", recent=("09-20 01:00 就寝",), earlier=("08-20 22:30 就寝",))
    rendered = drift.render()
    assert "「就寝」的时刻在**往后移**" in rendered and "移了多少不告诉你" in rendered
    assert "分钟" not in rendered


def test_the_schema_pins_the_concept_names_and_the_three_structures() -> None:
    """后件的取值是**全部行为概念**：链要写中间那一段（{甲} → 乙），乙不是事实里的后件（评审 C-10 ②）。"""

    schema = closure_json_schema(CONCEPTS)
    item = schema["properties"]["hypotheses"]["items"]
    assert item["properties"]["structure"]["enum"] == [s.value for s in Structure]
    assert set(item["properties"]["consequent"]["enum"]) == {CONCEPTS[i].name for i in CONCEPTS.behaviors()}
    assert set(item["properties"]) == set(item["required"])  # 严格模式
    assert "split_by" not in item["properties"]  # 调节体 = 情境进前件集合，不再另写 split_by


def test_each_structure_is_checked_against_the_fact_it_answers() -> None:
    """structure 不是标签（评审 C-10 ③）：直接因/调节体的后件必须是事实里的后件、前件里要有那个情境；链的后件必须是别的行为。"""

    # 调节体：情境进前件集合 → 与原假设 {晚睡}→早餐 身份不同，写得进去（旧写法同身份、永远被丢，评审 A-5）
    _e, built, signals = assemble_closure(answer(), CONCEPTS, SPLIT, now=NOW)
    (hypothesis,) = built
    assert {item.concept for item in hypothesis.antecedents} == {"出差中", "晚睡"} and hypothesis.note.startswith("调节体：")
    assert hypothesis.split_by == () and signals == ()
    # 链：后件还是事实里的后件 → 丢
    _e, built, signals = assemble_closure(answer(structure="chain"), CONCEPTS, SPLIT, now=NOW)
    assert built == () and "链要写中间那一段" in signals[0]
    # 链写对了：{出差中} → 晚睡（中间那一段），后件是别的行为
    chain = answer(
        structure="chain",
        antecedents=[{"concept": "出差中", "grade": None}, {"concept": "熬夜", "grade": None}],
        consequent="晚睡",
        type_prior="promoting",
        direction="up",
    )
    _e, built, signals = assemble_closure(chain, CONCEPTS, SPLIT, now=NOW)
    assert len(built) == 1 and built[0].consequent == "晚睡" and built[0].note.startswith("链：")
    # 调节体没把原前件带上 → 丢
    lone = answer(antecedents=[{"concept": "出差中", "grade": None}])
    _e, built, signals = assemble_closure(lone, CONCEPTS, SPLIT, now=NOW)
    assert built == () and "加进原来的前件集合" in signals[0]
    # 直接因：{出差中, 打球} → 早餐——前件里有情境即可，但集合里至少要有一个行为当锚（B14）
    direct = answer(structure="direct", antecedents=[{"concept": "出差中", "grade": None}, {"concept": "打球", "grade": None}])
    _e, built, _s = assemble_closure(direct, CONCEPTS, SPLIT, now=NOW)
    assert len(built) == 1 and built[0].note.startswith("直接因：")


def test_closure_hypotheses_are_typed_and_expanded_by_the_consequent_rhythm() -> None:
    """分型与逐峰是算法的、与触点② 同一段代码：后件三个峰就三本账，没节律就无节律型，上限（配置化）截后果峰（评审 C-10 ①、C-4）。"""

    from habitus.scene.concepts.rhythm import Rhythm, RhythmPeak
    from habitus.scene.hypotheses.author import ExpansionLimits

    three_peaks = Rhythm(
        concept="早餐",
        peaks=(RhythmPeak(1, 420, 480, 0.5), RhythmPeak(2, 720, 780, 0.4), RhythmPeak(3, 930, 990, 0.3)),
        days_with_peaks=7,
        recurrence_hours=8.0,
    )
    _e, built, _s = assemble_closure(answer(), CONCEPTS, SPLIT, now=NOW, rhythms={"早餐": three_peaks})
    assert [item.consequent_peak for item in built] == [1, 2, 3]
    _e, capped, notes = assemble_closure(answer(), CONCEPTS, SPLIT, now=NOW, rhythms={"早餐": three_peaks}, limits=ExpansionLimits(max_accounts_per_consequent=2))
    assert [item.consequent_peak for item in capped] == [1, 2] and any("只建前 2 个" in note for note in notes)
    _e, open_ended, _s = assemble_closure(answer(), CONCEPTS, SPLIT, now=NOW)  # 没给节律 → 无节律型
    assert open_ended[0].consequent_peak is None


# ── 组装与写入 ───────────────────────────────────────────────────────────


def test_a_proposed_hypothesis_is_dated_now_so_it_starts_from_scratch() -> None:
    """C3 不回填：用来发现它的那批观测**不算**它的证据，所以 created_at 是此刻、来源是 moderation。"""

    explanation, built, signals = assemble_closure(answer(), CONCEPTS, SPLIT, now=NOW)
    assert "出差的那几天" in explanation and signals == ()
    (hypothesis,) = built
    assert hypothesis.created_at == NOW and hypothesis.source.origin is HypothesisOrigin.MODERATION
    assert hypothesis.note.startswith("调节体：") and "出差那几天他连着熬夜" in hypothesis.note
    assert {item.concept for item in hypothesis.antecedents} == {"出差中", "晚睡"}
    # 写入时刻是夜批传进来的那一刻，不是墙钟（评审 B-7；重放 7 月的历史时墙钟是 9 月）
    client, _provider = recording_client([answer()])
    replayed = asyncio.run(ClosureAuthor(client, clock=lambda: datetime(2030, 1, 1, tzinfo=UTC)).propose(SPLIT, CONCEPTS, now=NOW))
    assert replayed.hypotheses[0].created_at == NOW


def test_one_unusable_proposal_is_dropped_and_the_round_goes_on() -> None:
    """单条写不成就丢那一条并报出来（与另外两个触点同一条纪律），不否决整轮。"""

    tautology = answer(structure="chain", antecedents=[{"concept": "打球", "grade": None}], consequent="运动", type_prior="promoting", direction="up")
    _explanation, built, signals = assemble_closure(tautology, CONCEPTS, SPLIT, now=NOW)
    assert built == () and len(signals) == 1 and "ancestor chain" in signals[0]
    # 已经有账的不重复写
    _e, again, notes = assemble_closure(answer(), CONCEPTS, SPLIT, now=NOW, known=assemble_closure(answer(), CONCEPTS, SPLIT, now=NOW)[1])
    assert again == () and "已经有账了" in notes[0]


def test_a_round_that_proposes_too_many_is_refused() -> None:
    body = answer()
    body["hypotheses"] = list(body["hypotheses"]) * (MAX_CLOSURE_HYPOTHESES + 1)  # type: ignore[index]
    with pytest.raises(ValueError, match="at most"):
        assemble_closure(body, CONCEPTS, SPLIT, now=NOW)


def test_the_author_writes_nothing_when_the_model_fails() -> None:
    client, provider = recording_client([{"explanation": "", "hypotheses": []}])  # 解释是空的 → 核对不过
    proposal = asyncio.run(ClosureAuthor(client, clock=lambda: NOW).propose(SPLIT, CONCEPTS))
    assert proposal.wrote_nothing and provider.calls == 2
    assert any(signal.startswith("model: ") and "explanation" in signal for signal in proposal.signals)

    good, good_provider = recording_client([answer()])
    written = asyncio.run(ClosureAuthor(good, clock=lambda: NOW).propose(SPLIT, CONCEPTS))
    assert good_provider.calls == 1 and len(written.hypotheses) == 1 and written.explanation
