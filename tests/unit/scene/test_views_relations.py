"""⑤ relations/：累积度门槛、KM 概率 + 兑现间隔、时刻、次数、类型 PN/PS、上一级指针、稳定性扫、剂量、共享证据的调节对照、干预分层、混口径。"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from habitus.scene.hypotheses import Antecedent, Aspect, Direction, TypePrior
from habitus.scene.ledger import Claim, Intervention, LedgerStore, OpportunityPass, Outcome, Settlement
from habitus.scene.occurrences import ConceptHit
from habitus.scene.views import TypeReading, ViewsConfig, read_relation, read_relations
from habitus.scene.views.materialize import render_relation
from habitus.scene.views.relations import borrow_candidates
from tests.unit.scene.fixtures import DAY1, at
from tests.unit.scene.ledger_fixtures import (
    BOOKING_TO_BALL,
    CONCEPTS,
    EXERCISE_TO_COFFEE,
    LATE_TO_BREAKFAST,
    LATE_TO_WAKE,
    NOW,
    daily_snapshot,
    hypothesis,
    uri_for,
)

BALL_TO_BREAKFAST = hypothesis("打球", direction=Direction.UP, type_prior=TypePrior.PROMOTING, note="打完球饿")
EXERCISE_TO_BREAKFAST = hypothesis("运动", direction=Direction.UP, type_prior=TypePrior.PROMOTING, note="动完饿")
OWL_TO_BREAKFAST = hypothesis("熬夜", note="熬得越晚越不吃")
CONFIG = ViewsConfig(min_count=6, min_blocks=3, split_min=3, split_min_blocks=2)


def book(
    ledger: LedgerStore,
    hyp,
    day: date,
    name: str,
    *,
    outcome: Outcome | None,
    index: int = 1,
    released_by_day: date | None = None,
    steps: int = 3,
    observed_at: datetime | None = None,
    count: int | None = None,
    hits: tuple[ConceptHit, ...] | None = None,
    situations: tuple[str, ...] = (),
    fingerprint: str | None = None,
    reminded: bool = False,
    generation: str = "gen-1",
    control: bool = True,
) -> Claim:
    """开一条承诺（锚在 ``day`` 02:10）并直接结算——views 只读账，账怎么来的不重要。``outcome=None`` 就只开不结。"""

    anchor = at(day, 2, 10)
    trigger = uri_for(day, name, 2, 10)
    snapshot = daily_snapshot(hyp.consequent, anchor, 8, generation=generation) if control else None
    claim = Claim(
        hypothesis_identity=hyp.identity,
        hypothesis_fingerprint=fingerprint or hyp.fingerprint,
        aspect=hyp.aspect,
        trigger_uri=trigger,
        antecedent_hits=hits or tuple(ConceptHit(item.concept, item.grade) for item in hyp.antecedents),
        antecedent_uris=(trigger,),
        situation_snapshot=situations,
        control=snapshot,
        created_at=NOW,
    )
    ledger.write_claim(claim)
    if outcome is None:
        return claim
    if outcome is Outcome.OCCURRED:
        assert snapshot is not None
        when = observed_at or snapshot.at(index).at  # type: ignore[union-attr]
        passes = tuple(OpportunityPass(snapshot.at(i).at, True) for i in range(1, index))  # type: ignore[union-attr]
        settlement = Settlement(
            claim.ref, outcome, NOW, observed_at=when, fulfilling_uri=uri_for(when.date(), "后件", when.hour, when.minute),
            latency_hours=(when - anchor).total_seconds() / 3600.0, opportunity_index=index, passes=passes,
        )
    elif outcome is Outcome.CENSORED:
        assert snapshot is not None
        settlement = Settlement(claim.ref, outcome, NOW, passes=tuple(OpportunityPass(snapshot.at(i).at, True) for i in range(1, steps + 1)))  # type: ignore[union-attr]
    elif outcome is Outcome.OBSERVED:
        assert observed_at is not None
        settlement = Settlement(claim.ref, outcome, NOW, observed_at=observed_at, fulfilling_uri=uri_for(observed_at.date(), "后件", observed_at.hour, observed_at.minute), opportunity_index=index)
    elif outcome is Outcome.ABSENT:
        settlement = Settlement(claim.ref, outcome, NOW, opportunity_index=index)
    elif outcome is Outcome.RELEASED:
        assert released_by_day is not None
        settlement = Settlement(claim.ref, outcome, NOW, releasing_uri=uri_for(released_by_day, "又约球", 11, 0))
    else:
        settlement = Settlement(claim.ref, outcome, NOW, count=count)
    ledger.write_settlement(settlement)
    if reminded:
        ledger.record_intervention(Intervention(claim.ref, reminded_at=at(day, 6, 30), recorded_at=NOW))
    return claim


def week(offset: int) -> date:
    """一周一次的前件：块长 7 天，六次就跨六块。"""

    return DAY1 + timedelta(days=7 * offset)


def day(offset: int) -> date:
    return DAY1 + timedelta(days=offset)


def test_the_read_side_truncation_point_must_equal_the_write_side_censoring_point() -> None:
    """``settlement_horizon`` 与 ``censor_after`` 是同一件事的两头，必须相等；两个数都待定，"相等"与定成几无关。

    不等的后果是系统性偏：读侧比写侧短，最近那段只进事件不进删失、p1 偏高；读侧比写侧长，已删失的承诺
    还被当"命运未定"扣掉、n 偏小。组合根在刀 5 里同时设这两个数，所以这里留一个守门人。
    """

    from habitus.scene.ledger import LedgerConfig
    from habitus.scene.views import require_aligned

    require_aligned(ViewsConfig(settlement_horizon=8), LedgerConfig(censor_after=8))
    with pytest.raises(ValueError, match="must equal LedgerConfig.censor_after"):
        require_aligned(ViewsConfig(settlement_horizon=8), LedgerConfig(censor_after=10))


def test_probability_is_kaplan_meier_at_the_expected_opportunity_minus_the_control(tmp_path) -> None:
    """七h 第 4 步：6 次机会、2 次在第 1 次机会上来了、4 次等过 3 个机会没来 → 实际 33% vs 本来 88% → −55pp，抑制。"""

    ledger = LedgerStore(tmp_path / "scene")
    for i, outcome in enumerate((Outcome.OCCURRED, Outcome.CENSORED, Outcome.CENSORED, Outcome.OCCURRED, Outcome.CENSORED, Outcome.CENSORED)):
        book(ledger, LATE_TO_BREAKFAST, week(i), "就寝", outcome=outcome)
    reading = read_relation(LATE_TO_BREAKFAST, ledger=ledger, concepts=CONCEPTS, config=CONFIG)
    strength = reading.strength
    assert (strength.accumulation.count, strength.accumulation.blocks, strength.accumulation.block_hours) == (6, 6, 7 * 24.0)
    assert strength.p1 == pytest.approx(2 / 6) and strength.control == pytest.approx(0.88) and (strength.events, strength.censored) == (2, 4)
    assert strength.interval is not None and strength.interval.point == pytest.approx(2 / 6 - 0.88) and strength.interval.high < 0
    assert reading.type_reading is not None and reading.type_reading.reading is TypeReading.INHIBITING
    assert strength.agrees_with_prior is True and reading.unrefuted and reading.open_claims == 0 and reading.fallback is None
    assert strength.median_steps is None  # S 停在 0.67，中位没到——不拿两个到达者的中位冒充
    assert strength.median_hours == pytest.approx(5.5833, abs=1e-3)  # 到达者：02:10 → 07:45


def test_below_the_accumulation_gate_there_is_no_number_only_how_far_and_a_fallback(tmp_path) -> None:
    """8 次机会挤在 2 天里：次数够、块不够 → 不出数，报还差 1 块；账薄时给上一级（留一法后）的读数当指针。"""

    ledger = LedgerStore(tmp_path / "scene")
    for i in range(8):
        anchor_day = day(i // 4)
        name = f"就寝{i}"
        trigger = uri_for(anchor_day, name, 2, 10)
        claim = Claim(
            hypothesis_identity=BALL_TO_BREAKFAST.identity, hypothesis_fingerprint=BALL_TO_BREAKFAST.fingerprint, aspect=Aspect.PROBABILITY,
            trigger_uri=trigger, antecedent_hits=(ConceptHit("打球"),), antecedent_uris=(trigger,), situation_snapshot=(),
            control=daily_snapshot("早餐", at(anchor_day, 2, 10), 8), created_at=NOW,
        )
        ledger.write_claim(claim)
        ledger.write_settlement(Settlement(claim.ref, Outcome.OCCURRED, NOW, observed_at=at(anchor_day, 7, 45), fulfilling_uri=uri_for(anchor_day, "x", 7, 45), latency_hours=5.58, opportunity_index=1))
    thin = read_relation(BALL_TO_BREAKFAST, ledger=ledger, concepts=CONCEPTS, config=CONFIG)
    assert thin.strength.interval is None and not thin.unrefuted and thin.type_reading is None
    assert (thin.strength.accumulation.count, thin.strength.accumulation.blocks, thin.strength.accumulation.short_by_blocks) == (8, 2, 1)
    assert thin.strength.p1 == pytest.approx(1.0)  # 计数照报，只是不下结论
    assert thin.fallback is None  # 上一级还没有账

    # 上一级「运动→早餐」六周 8 条，其中 2 条与本条共用出力 occurrence → 留一法后按 6 条读（4 到 2 没到）。
    for i in range(8):
        shared = i < 2
        book(
            ledger, EXERCISE_TO_BREAKFAST, day(i // 4) if shared else week(i), "就寝0" if shared else "跑步",
            outcome=Outcome.OCCURRED if i % 4 != 3 else Outcome.CENSORED, hits=(ConceptHit("运动"),),
        )
    assert borrow_candidates(BALL_TO_BREAKFAST, CONCEPTS) == (EXERCISE_TO_BREAKFAST.identity,)
    # 指针要**父假设本体**才给：只有账、没有假设的上一级解释不了它在量什么（父可能猜的是别的机会数）。
    assert read_relation(BALL_TO_BREAKFAST, ledger=ledger, concepts=CONCEPTS, config=CONFIG).fallback is None
    known = {EXERCISE_TO_BREAKFAST.identity: EXERCISE_TO_BREAKFAST}
    thin = read_relation(BALL_TO_BREAKFAST, ledger=ledger, concepts=CONCEPTS, config=CONFIG, hypotheses=known)
    assert thin.fallback is not None and thin.fallback.identity == EXERCISE_TO_BREAKFAST.identity
    assert thin.fallback.strength.accumulation.count == 6 and thin.fallback.strength.interval is not None
    # 多前件退到少一个元素的子集。
    pair = hypothesis("打球", "晚睡", note="x")
    assert set(borrow_candidates(pair, CONCEPTS)) == {LATE_TO_BREAKFAST.identity, BALL_TO_BREAKFAST.identity}


def test_stability_scan_flags_a_situation_that_splits_the_account_and_reports_what_it_skipped(tmp_path) -> None:
    """出差的 5 周全没来、不出差的 5 周全来了 → 被「出差中」调节；「周末」只出现 1 次、那一层没攒够 → 记进 skipped。"""

    ledger = LedgerStore(tmp_path / "scene")
    for i in range(5):
        book(ledger, LATE_TO_BREAKFAST, week(i), "就寝", outcome=Outcome.CENSORED, situations=("出差中",))
        book(ledger, LATE_TO_BREAKFAST, week(i + 5), "就寝", outcome=Outcome.OCCURRED, situations=() if i else ("周末",))
    reading = read_relation(LATE_TO_BREAKFAST, ledger=ledger, concepts=CONCEPTS, config=CONFIG)
    stability = reading.stability
    assert [m.situation for m in stability.moderations] == ["出差中"] and not reading.unrefuted
    travel = stability.moderations[0]
    assert travel.with_situation.interval is not None and travel.with_situation.interval.point < travel.without_situation.interval.point  # type: ignore[union-attr]
    assert stability.skipped == ("周末",) and stability.tested == ("出差中",) and not stability.control_split_by_situation


def test_dose_orders_grades_by_their_numeric_bound_and_needs_a_trend_that_agrees_with_the_prior(tmp_path) -> None:
    """三档且越重越不吃 → 单调可信度升一档；反过来摆不算；两档谈不上曲线。每档三周各一次才过分层门槛。"""

    def fill(store: LedgerStore, plan: dict[str, tuple[Outcome, ...]]) -> None:
        offset = 0
        for grade, outcomes in plan.items():
            for outcome in outcomes:
                book(store, OWL_TO_BREAKFAST, week(offset), "就寝", outcome=outcome, hits=(ConceptHit("熬夜", grade),))
                offset += 1

    ledger = LedgerStore(tmp_path / "trend")
    # 声明顺序故意打乱成 重、轻、中：读出来仍按数值下界排 轻 → 中 → 重。
    fill(ledger, {"重": (Outcome.CENSORED,) * 3, "轻": (Outcome.OCCURRED, Outcome.OCCURRED, Outcome.CENSORED), "中": (Outcome.OCCURRED, Outcome.CENSORED, Outcome.CENSORED)})
    (dose,) = read_relation(OWL_TO_BREAKFAST, ledger=ledger, concepts=CONCEPTS, config=CONFIG).dose
    assert dose.concept == "熬夜" and [grade for grade, _ in dose.by_grade] == ["轻", "中", "重"]
    assert [round(s.p1, 2) for _g, s in dose.by_grade] == [0.67, 0.33, 0.0]  # type: ignore[arg-type]
    assert dose.monotonic

    reversed_ledger = LedgerStore(tmp_path / "reversed")
    fill(reversed_ledger, {"轻": (Outcome.CENSORED,) * 3, "中": (Outcome.OCCURRED, Outcome.CENSORED, Outcome.CENSORED), "重": (Outcome.OCCURRED, Outcome.OCCURRED, Outcome.CENSORED)})
    (flipped,) = read_relation(OWL_TO_BREAKFAST, ledger=reversed_ledger, concepts=CONCEPTS, config=CONFIG).dose
    assert not flipped.monotonic  # 数值上单调，但"档越重越吃"与基准（↓）相反

    two = LedgerStore(tmp_path / "two")
    for i in range(3):
        book(two, LATE_TO_BREAKFAST, week(i), "就寝", outcome=Outcome.OCCURRED, hits=(ConceptHit("晚睡", "轻"),))
        book(two, LATE_TO_BREAKFAST, week(i + 3), "就寝", outcome=Outcome.CENSORED, hits=(ConceptHit("晚睡", "重"),))
    (pair,) = read_relation(LATE_TO_BREAKFAST, ledger=two, concepts=CONCEPTS, config=CONFIG).dose
    assert [grade for grade, _ in pair.by_grade] == ["轻", "重"] and not pair.monotonic
    graded = hypothesis(Antecedent("晚睡", "重"), note="x")
    assert read_relation(graded, ledger=two, concepts=CONCEPTS, config=CONFIG).dose == ()


def test_a_grade_that_wraps_midnight_gets_no_dose_ladder_and_says_so(tmp_path) -> None:
    """评审 C-12：档按 ``grade.lower`` 在数轴上排，而 22:00–02:00 的档 ``lower`` 是 1320。

    和一个 02:00–04:00 的档（``lower`` 120）比大小，排出来是"凌晨在深夜之前"，那条"趋势"是假的。
    所以这一项不出数，而且要说出为什么（不静默）。
    """

    from habitus.scene.concepts import ConceptGrade, ConceptSet, GradeMeasure
    from habitus.scene.hypotheses import Direction, Hypothesis, TypePrior
    from tests.unit.scene.concept_fixtures import concept

    start = GradeMeasure.START_MINUTE_OF_DAY
    bedtime = concept(
        "上床",
        "躺下准备睡",
        grades=(ConceptGrade("凌晨", start, 120, 240), ConceptGrade("深夜", start, 22 * 60, 2 * 60)),
    )
    concepts = ConceptSet((*CONCEPTS.values(), bedtime))
    wrapped = Hypothesis(
        antecedents=(Antecedent("上床"),),
        consequent="早餐",
        aspect=Aspect.PROBABILITY,
        direction=Direction.DOWN,
        type_prior=TypePrior.INHIBITING,
        note="档跨午夜",
        source=LATE_TO_BREAKFAST.source,
        created_at=LATE_TO_BREAKFAST.created_at,
    )
    wrapped.validate_against(concepts)
    ledger = LedgerStore(tmp_path / "wrapped")
    for i in range(3):
        book(ledger, wrapped, week(i), "就寝", outcome=Outcome.OCCURRED, hits=(ConceptHit("上床", "深夜"),))
        book(ledger, wrapped, week(i + 3), "就寝", outcome=Outcome.CENSORED, hits=(ConceptHit("上床", "凌晨"),))
    reading = read_relation(wrapped, ledger=ledger, concepts=concepts, config=CONFIG)
    assert reading.dose == () and reading.unordered_dose == ("上床",)


def test_shared_evidence_runs_the_moderation_contrast_between_the_two_hypotheses(tmp_path) -> None:
    """七i 四：同一晚的就寝同时命中「晚睡」与「熬夜」；晚睡→早餐按"那晚是否也熬夜"分两层——熬夜在场的全没来、不在场的全来了 → 分得开。"""

    ledger = LedgerStore(tmp_path / "scene")
    for i in range(8):
        with_owl = i % 2 == 0
        book(ledger, LATE_TO_BREAKFAST, week(i), "就寝", outcome=Outcome.CENSORED if with_owl else Outcome.OCCURRED)
        if with_owl:
            book(ledger, OWL_TO_BREAKFAST, week(i), "就寝", outcome=Outcome.CENSORED, hits=(ConceptHit("熬夜", "重"),))  # 同一个触发 occurrence
    reading, _owl = read_relations((LATE_TO_BREAKFAST, OWL_TO_BREAKFAST), ledger=ledger, concepts=CONCEPTS, config=CONFIG)
    (shared,) = reading.shared
    assert shared.other == OWL_TO_BREAKFAST.identity and shared.shared_claims == 4
    assert shared.separable is True and shared.with_other.p1 == pytest.approx(0.0) and shared.without_other.p1 == pytest.approx(1.0)
    # 同一前件的兄弟假设（晚睡→起床）不是"另一个原因"：证据 100% 重合，列出来只有"分不开"这句噪音。
    for day_offset in range(8):
        book(ledger, LATE_TO_WAKE, week(day_offset), "就寝", outcome=Outcome.OBSERVED, observed_at=at(week(day_offset), 9, 30 + day_offset))
    (again,) = read_relations((LATE_TO_BREAKFAST,), ledger=ledger, concepts=CONCEPTS, config=CONFIG)
    assert [item.other for item in again.shared] == []  # 兄弟假设与未在读的假设都不列
    with_siblings = read_relations((LATE_TO_BREAKFAST, LATE_TO_WAKE, OWL_TO_BREAKFAST), ledger=ledger, concepts=CONCEPTS, config=CONFIG)
    assert [item.other for item in with_siblings[0].shared] == [OWL_TO_BREAKFAST.identity]  # 起床那条后件不同，不算共享证据


def test_reminded_split_and_mixed_fingerprints(tmp_path) -> None:
    ledger = LedgerStore(tmp_path / "scene")
    for i in range(6):
        book(ledger, LATE_TO_BREAKFAST, week(i), "就寝", outcome=Outcome.OCCURRED if i % 2 else Outcome.CENSORED, reminded=i < 3)
    reading = read_relation(LATE_TO_BREAKFAST, ledger=ledger, concepts=CONCEPTS, config=CONFIG)
    reminded, quiet = reading.reminded
    assert reminded.accumulation.count == 3 and quiet.accumulation.count == 3 and reminded.interval is not None and quiet.interval is not None
    book(ledger, LATE_TO_BREAKFAST, week(9), "就寝", outcome=Outcome.CENSORED, fingerprint="0000000000000000")
    mixed = read_relation(LATE_TO_BREAKFAST, ledger=ledger, concepts=CONCEPTS, config=CONFIG)
    assert mixed.mixed_fingerprints and not mixed.unrefuted
    # 树每晚重建 → 代名每晚不同：**跨天**多代是正常的，不该否掉读数（否则 profile 永远空）。
    book(ledger, LATE_TO_BREAKFAST, week(10), "就寝", outcome=Outcome.CENSORED, generation="gen-2")
    across_days = read_relation(LATE_TO_BREAKFAST, ledger=ledger, concepts=CONCEPTS, config=CONFIG)
    assert across_days.control_generations == ("gen-1", "gen-2") and across_days.mixed_control_generations == ()
    # 同一个触发日里两代：那一晚崩在开承诺中途、重建树后又跑了一次 —— 这一批才不可信。
    book(ledger, LATE_TO_BREAKFAST, week(11), "就寝", outcome=Outcome.CENSORED, generation="gen-3")
    book(ledger, LATE_TO_BREAKFAST, week(11), "又睡", outcome=Outcome.CENSORED, generation="gen-4")
    same_day = read_relation(LATE_TO_BREAKFAST, ledger=ledger, concepts=CONCEPTS, config=CONFIG)
    assert same_day.mixed_control_generations == ("gen-3", "gen-4") and not same_day.unrefuted


def test_timing_count_and_latency_use_their_own_arithmetic(tmp_path) -> None:
    ledger = LedgerStore(tmp_path / "scene")
    # 时刻：起床 09:40 / 10:00 / 09:20 对着第 1 次机会 08:10 → +90 / +110 / +70 分，中位 +90；两条缺席另计。
    for i, (hour, minute) in enumerate(((9, 40), (10, 0), (9, 20))):
        book(ledger, LATE_TO_WAKE, week(i), "就寝", outcome=Outcome.OBSERVED, observed_at=at(week(i), hour, minute))
    for i in (3, 4, 5):
        book(ledger, LATE_TO_WAKE, week(i), "就寝", outcome=Outcome.OBSERVED if i == 5 else Outcome.ABSENT, observed_at=at(week(i), 9, 40))
    timing = read_relation(LATE_TO_WAKE, ledger=ledger, concepts=CONCEPTS, config=CONFIG).strength
    assert timing.unit == "minutes" and timing.interval is not None and timing.interval.point == pytest.approx(90.0) and timing.absent == 2
    assert timing.accumulation.count == 6 and timing.agrees_with_prior is True
    # 次数：3、1、2、2、1、3 杯，对照 = 第 1–3 次机会的概率之和 0.5+0.4+0.3 → 均值 2 − 1.2 = +0.8。
    for i, n in enumerate((3, 1, 2, 2, 1, 3)):
        book(ledger, EXERCISE_TO_COFFEE, week(i), "打球", outcome=Outcome.COUNTED, count=n, hits=(ConceptHit("运动"),))
    count_reading = read_relation(EXERCISE_TO_COFFEE, ledger=ledger, concepts=CONCEPTS, config=CONFIG)
    count = count_reading.strength
    assert count.unit == "times" and count.interval is not None and count.interval.point == pytest.approx(2.0 - 1.2)
    # 次数留一位小数：按整数印 +0.8 会成 "+1 次"，把差额说反（评审 C-9）。
    rendered = render_relation(count_reading, EXERCISE_TO_COFFEE)
    assert "+0.8 次" in rendered and "注：区间未修天与天之间的相关" in rendered


def test_a_open_ended_hypothesis_is_read_as_a_fulfilment_rate_not_as_a_rate_against_the_tree(tmp_path) -> None:
    """评审 C-2/A-13：约球→打球 4 次第 2–3 次机会去了、2 次没去，旧读法按"第 1 次机会"算成实际 0% vs 本来 20%
    → "抑制、未被推翻"。无节律型改成只读兑现（用户 09-27）：4/6 兑现、到达者中位 52.8 小时，不读 p1−p0、不读类型。
    """

    ledger = LedgerStore(tmp_path / "scene")
    for i, index in enumerate((2, 3, 3, 2)):
        book(ledger, BOOKING_TO_BALL, week(i), "约球", outcome=Outcome.OCCURRED, index=index)
    # 无节律型的"没去"不是删失（不按机会数删失）：它就是**还立着**。立了 1–2 周 > 对照 120 小时 → 进分母。
    for i in (4, 5):
        book(ledger, BOOKING_TO_BALL, week(i), "约球", outcome=None)
    reading = read_relation(BOOKING_TO_BALL, ledger=ledger, concepts=CONCEPTS, config=CONFIG, now=at(week(6), 12, 0))
    assert reading.type_reading is None and reading.strength.interval is None and reading.strength.unit == "fulfilment"
    fulfilment = reading.fulfilment
    assert fulfilment is not None
    assert (fulfilment.fulfilled, fulfilment.released, fulfilment.closed, fulfilment.standing_total) == (4, 0, 0, 2)
    assert fulfilment.standing == 2  # 两条都立过了对照间隔
    assert fulfilment.accumulation.count == 6 and fulfilment.accumulation.blocks == 6 and fulfilment.sufficient
    assert fulfilment.rate == pytest.approx(4 / 6)
    assert fulfilment.interval is not None and 0.0 < fulfilment.interval.low < 4 / 6 < fulfilment.interval.high < 1.0
    # 第 2 次机会 40.83h、第 3 次 64.83h（打球峰 19:00，锚 02:10）→ 到达者中位 52.83h。
    assert fulfilment.median_hours == pytest.approx(52.8333, abs=1e-3)
    # 对照 = 后件本来多久真的发生一次 = 间隔 ÷ 峰概率（用户 09-27）：打球峰一天一个、概率 0.20 → 24 / 0.20 = 120 小时。
    # 中位兑现 52.8h 只有它的 44% → "约了就去，比平时快得多"，算未被推翻。按间隔本身（24h）当对照会判反。
    assert fulfilment.control_hours == pytest.approx(120.0)
    assert fulfilment.unrefuted and reading.unrefuted
    # 渲染：无节律型印兑现那一行，不印强度、不印类型、不印干预分层。
    text = render_relation(reading, BOOKING_TO_BALL)
    assert "· 无节律型 ·" in text and "兑现：67% [" in text and "兑现 4 · 释放 0 · 关掉 0 · 仍立着 2（其中 2 已过对照间隔）" in text
    assert "到达者中位 53 小时" in text and "对照：本来约 120 小时才发生一次" in text
    # R-3：出了区间就要标"未修天间相关"（裁定：承认不修，渲染上标出）。
    assert "注：区间未修天与天之间的相关" in text
    lines = text.splitlines()
    assert not [line for line in lines if line.startswith(("强度：", "类型：", "退回上一级："))]
    assert "干预分层" not in text  # 这本账从没提醒过（C-17：没提醒过不印这一节）
    # P-1（用户 09-27 定"四处"）：四处分层对无节律型也照跑，每层读的是兑现率。
    from habitus.scene.views.relations import layer_reading, load_account
    layer = layer_reading(load_account(ledger, BOOKING_TO_BALL.identity), BOOKING_TO_BALL, CONFIG, now=at(week(6), 12, 0))
    assert layer.__class__.__name__ == "FulfilmentReading" and layer.interval is not None


def test_a_released_open_ended_is_in_the_denominator_and_standing_ones_count_once_overdue(tmp_path) -> None:
    """释放（再次约球作废上一次）与仍立着的分别怎么进分母；``now`` 不给就一条仍立着的都不算（不凭空抬高兑现率）。"""

    from habitus.scene.views.relations import fulfilment_of, load_account

    ledger = LedgerStore(tmp_path / "scene")
    for i in (0, 1, 2):
        book(ledger, BOOKING_TO_BALL, week(i), "约球", outcome=Outcome.OCCURRED, index=2)
    for i in (3, 4):
        book(ledger, BOOKING_TO_BALL, week(i), "约球", outcome=Outcome.RELEASED, released_by_day=week(i) + timedelta(days=1))
    book(ledger, BOOKING_TO_BALL, week(5), "约球", outcome=None)  # 还立着
    account = load_account(ledger, BOOKING_TO_BALL.identity)
    # 立了 10 天（249.8h）> 对照 120 小时 → 进分母：3 兑现 / (3 + 2 释放 + 1 仍立着) = 50%。
    overdue = fulfilment_of(account, BOOKING_TO_BALL, CONFIG, now=at(week(5) + timedelta(days=10), 12, 0))
    assert (overdue.fulfilled, overdue.released, overdue.standing, overdue.standing_total) == (3, 2, 1, 1)
    assert overdue.accumulation.count == 6 and overdue.rate == pytest.approx(0.5)
    assert overdue.longest_standing_hours == pytest.approx(10 * 24 + 9.8333, abs=1e-3)
    # ``now`` 不给 → 没有"立了多久"这个量，仍立着的不进分母：3/5。
    blind = fulfilment_of(account, BOOKING_TO_BALL, CONFIG, now=None)
    assert blind.standing == 0 and blind.standing_total == 1 and blind.rate == pytest.approx(3 / 5)
    # 被生命周期关掉的（CENSORED）只报数、不进分母：裁定的分母只有兑现 + 释放 + 超期仍立着三项，而
    # ``close_claim`` 的关闭理由是自由文本，账本分不出"等太久了"与"这条假设整个退役了"。
    book(ledger, BOOKING_TO_BALL, week(6), "约球", outcome=Outcome.CENSORED, steps=2)
    with_shut = fulfilment_of(load_account(ledger, BOOKING_TO_BALL.identity), BOOKING_TO_BALL, CONFIG, now=at(week(5) + timedelta(days=10), 12, 0))
    assert with_shut.closed == 1 and with_shut.accumulation.count == overdue.accumulation.count and with_shut.rate == overdue.rate


def test_type_readout_prints_pn_and_ps_and_tells_enabling_from_promoting(tmp_path) -> None:
    """挂号→就诊那类：不挂号几乎不去（p0 低）、挂了就去（p1 高）→ PN 高 PS 高 → 促进；PS 公式是 (p1−p0)/(1−p0)。"""

    from habitus.scene.views.relations import Strength, read_type
    from habitus.scene.views.stats import Interval
    from tests.unit.scene.ledger_fixtures import BEDTIME  # noqa: F401  # 保持导入路径可用

    def strength(p1: float, p0: float) -> Strength:
        from habitus.scene.views.relations import Accumulation

        return Strength(Aspect.PROBABILITY, Accumulation(6, 3, 24.0, 6, 3), Interval(p1 - p0, p1 - p0 - 0.05, p1 - p0 + 0.05), p0, "probability", p1=p1)

    promoting = read_type(strength(0.9, 0.3), CONFIG)
    assert promoting is not None and promoting.reading is TypeReading.PROMOTING
    assert promoting.pn == pytest.approx((0.9 - 0.3) / 0.9) and promoting.ps == pytest.approx((0.9 - 0.3) / 0.7)
    enabling = read_type(strength(0.4, 0.05), CONFIG)
    assert enabling is not None and enabling.reading is TypeReading.ENABLING and enabling.pn == pytest.approx(0.875) and enabling.ps == pytest.approx(0.35 / 0.95)
    assert read_type(strength(0.5, 0.5), CONFIG).reading is TypeReading.NONE  # type: ignore[union-attr]
    assert read_type(strength(0.2, 0.8), CONFIG).reading is TypeReading.INHIBITING  # type: ignore[union-attr]


def test_all_six_arriving_gets_a_width_from_jeffreys_not_a_zero_width_certainty(tmp_path) -> None:
    """六周早餐全来了：只用百分位 bootstrap 会给 ``+12pp [+12, +12]``（零宽 → 必然"促进 · 未被推翻"，零效应假显著率
    p0=0.88 时实测 50.7%）。区间取 Jeffreys ∪ 分块 bootstrap 之后有宽度，而且含零 → 读成"无"。"""

    ledger = LedgerStore(tmp_path / "scene")
    for i in range(6):
        book(ledger, LATE_TO_BREAKFAST, week(i), "就寝", outcome=Outcome.OCCURRED)
    reading = read_relation(LATE_TO_BREAKFAST, ledger=ledger, concepts=CONCEPTS, config=CONFIG)
    strength = reading.strength
    assert strength.p1 == pytest.approx(1.0) and strength.control == pytest.approx(0.88)
    assert strength.interval is not None and strength.interval.width > 0
    assert strength.interval.low < 0.12 and strength.interval.contains_zero  # 六次全到也不该宣称"促进"
    assert reading.type_reading is not None and reading.type_reading.reading is TypeReading.NONE and not reading.unrefuted


def test_a_zero_width_interval_is_not_an_interval(tmp_path) -> None:
    """连续方面六次完全一样（真实数据不会这样，合成数据会）：bootstrap 给不出宽度 → 当"算不出"，不进 profile。"""

    ledger = LedgerStore(tmp_path / "scene")
    for i in range(6):
        book(ledger, LATE_TO_WAKE, week(i), "就寝", outcome=Outcome.OBSERVED, observed_at=at(week(i), 9, 40))
    flat = read_relation(LATE_TO_WAKE, ledger=ledger, concepts=CONCEPTS, config=CONFIG)
    assert flat.strength.accumulation.count == 6 and flat.strength.interval is None and not flat.unrefuted
    # 时刻有抖动就照常出区间。
    jittered = LedgerStore(tmp_path / "jitter")
    for i in range(6):
        book(jittered, LATE_TO_WAKE, week(i), "就寝", outcome=Outcome.OBSERVED, observed_at=at(week(i), 9, 30 + 5 * i))
    varied = read_relation(LATE_TO_WAKE, ledger=jittered, concepts=CONCEPTS, config=CONFIG)
    assert varied.strength.interval is not None and varied.strength.interval.width > 0


def test_probability_reads_only_claims_whose_fate_is_already_decided(tmp_path) -> None:
    """对称截断：后件来了当晚就 OCCURRED 进账、没来要等 ``settlement_horizon`` 次机会才 CENSORED。不截断的话最近一段
    只进事件，p1 系统偏高（真 1/3 读 0.375，评审 C-4 实测）。"""

    ledger = LedgerStore(tmp_path / "scene")
    for i in range(6):
        book(ledger, LATE_TO_BREAKFAST, week(i), "就寝", outcome=Outcome.CENSORED)
    fresh = book(ledger, LATE_TO_BREAKFAST, week(6), "就寝", outcome=Outcome.OCCURRED)  # 刚开就兑现，命运还没到期
    config = ViewsConfig(min_count=6, min_blocks=3, split_min=3, split_min_blocks=2, settlement_horizon=8)
    naive = read_relation(LATE_TO_BREAKFAST, ledger=ledger, concepts=CONCEPTS, config=config)
    assert naive.strength.accumulation.count == 7 and naive.strength.immature == 0  # 不给 now：不截断（旧行为）
    assert fresh.control is not None
    horizon = fresh.control.at(8)
    assert horizon is not None
    just_after = horizon.span.start  # 第 8 次机会还没过完 → 这一条的命运还没定
    truncated = read_relation(LATE_TO_BREAKFAST, ledger=ledger, concepts=CONCEPTS, config=config, now=just_after)
    assert truncated.strength.accumulation.count == 6 and truncated.strength.immature == 1
    assert truncated.strength.events == 0 and truncated.strength.censored == 6
