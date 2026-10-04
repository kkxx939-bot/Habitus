"""④ 结算：按这本账自己的钟面窗口走（2026-10-01）。概率：窗口里来了 / 过完没来 / 没看清；时刻看同一个窗口；次数数 horizon 个窗口；逐窗口判没看清。"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from habitus.scene.hypotheses import Aspect
from habitus.scene.ledger import (
    Claim,
    LedgerConfig,
    LedgerStore,
    Outcome,
    WindowSpan,
    close_claim,
    open_claims_for_day,
    settle_claim,
    settle_due,
)
from habitus.scene.ledger.settlement import mapped_until, passes_until
from habitus.scene.occurrences import ConceptHit, ConceptHitStore
from tests.unit.scene.fixtures import DAY1, DAY2, DAY3, at
from tests.unit.scene.ledger_fixtures import (
    BOOKING_TO_BALL,
    CONCEPTS,
    EXERCISE_TO_COFFEE,
    LATE_TO_BEDTIME,
    LATE_TO_BREAKFAST,
    LATE_TO_WAKE,
    NOW,
    REBOOKING_TO_BALL,
    FixedCoverage,
    TableOpportunities,
    daily_snapshot,
    hypothesis,
    record,
)

LATE = record(DAY1, "就寝", 2, 10, ConceptHit("晚睡", "轻"), lasts_minutes=7 * 60)  # 被认到 09:10
CONFIG = LedgerConfig()


def claim_for(hypothesis, *, control=True) -> Claim:
    return Claim(
        hypothesis_identity=hypothesis.identity,
        hypothesis_fingerprint=hypothesis.fingerprint,
        aspect=hypothesis.aspect,
        trigger_uri=LATE.occurrence_uri,
        antecedent_hits=(ConceptHit("晚睡", "轻"),),
        antecedent_uris=(LATE.occurrence_uri,),
        situation_snapshot=(),
        control=daily_snapshot(hypothesis.consequent, LATE.started_at, hypothesis.horizon) if control else None,
        created_at=NOW,
    )


def settle(claim, hypothesis, records, *, now, coverage=None, config=CONFIG, until=None):
    """单测里 ``until`` 缺省 = ``now``（"映射跟得上"）；B2 那条前置条件由 ``settle_due`` 的 ``mapped_until`` 管，另有测试。"""

    return settle_claim(claim, hypothesis, records, CONCEPTS, now=now, until=until or now, coverage=coverage or FixedCoverage(1.0), config=config)


# ── 概率（窗口账，2026-10-01） ────────────────────────────────────────────────────


def test_probability_settles_the_moment_the_consequent_arrives_inside_its_window() -> None:
    """七h：02:10 晚睡，07:00 吃了碗面——落在这本账自己那个早餐窗（06:15–09:15）里，隔 4.8 小时；「就寝」还被认着（到 09:10）也照样算。"""

    claim = claim_for(LATE_TO_BREAKFAST)
    breakfast = record(DAY1, "吃了碗面", 7, 0, "早餐")
    settlement = settle(claim, LATE_TO_BREAKFAST, (LATE, breakfast), now=at(DAY1, 7, 5))
    assert settlement is not None and settlement.outcome is Outcome.OCCURRED
    assert settlement.observed_at == at(DAY1, 7, 0) and settlement.fulfilling_uri == breakfast.occurrence_uri
    assert settlement.latency_hours == pytest.approx(4.8333, abs=1e-3) and settlement.opportunity_index == 1 and settlement.passes == ()
    # 第二天才吃：不算进这本账——那是"早餐#1"下一天的事；这本账在窗口过完后读成"没来"。往后推与不做分不分得开，
    # 由同一前因的几本峰账摆在一起看（behaviours 面），不在一本账里数第几次。
    late_breakfast = record(DAY2, "早饭", 7, 50, "早餐")
    second = settle(claim, LATE_TO_BREAKFAST, (LATE, late_breakfast), now=at(DAY2, 8, 0))
    assert second is not None and second.outcome is Outcome.ABSENT and second.opportunity_index == 1 and second.passes[0].at == at(DAY1, 7, 45)
    # 触发那条自己、锚之前的不算后件；窗口还没过完 → 不结。
    early = record(DAY1, "夜宵", 1, 0, "早餐")
    assert settle(claim, LATE_TO_BREAKFAST, (LATE, early), now=at(DAY1, 9, 0)) is None
    assert settle(claim, LATE_TO_BREAKFAST, (LATE, early), now=at(DAY1, 10, 0)).outcome is Outcome.ABSENT  # type: ignore[union-attr]


def test_a_window_whose_consequent_the_mapper_could_not_judge_does_not_count(tmp_path) -> None:
    """评审 A-4 / 用户 09-27「没看到就不算」：后件被映射器记成「未决」的窗口不算它过了、也不算缺席——没看清（CENSORED）。

    映射侧改成三态之后模型不再答"没有"，但账本这边如果照旧把那个窗口算成"后件没来"，假负样本还是进来了——
    分母不该有它。这里造：DAY1 早餐时段有一条记录把「早餐」列进 unresolved（判不了）。
    """

    claim = claim_for(LATE_TO_BREAKFAST)
    puzzling = replace(record(DAY1, "吃了点东西", 7, 30, kind="吃饭"), unresolved=("早餐",))
    assert settle(claim, LATE_TO_BREAKFAST, (LATE,), now=at(DAY1, 10, 0)).outcome is Outcome.ABSENT  # type: ignore[union-attr]
    censored = settle(claim, LATE_TO_BREAKFAST, (LATE, puzzling), now=at(DAY1, 10, 0))
    assert censored is not None and censored.outcome is Outcome.CENSORED and censored.reason == "窗口没看清"
    assert [item.observed for item in censored.passes] == [False] and censored.observed_passes == 0
    # 时刻方面同理：窗口没看清 → 删失，不是缺席。
    timing_claim = claim_for(LATE_TO_WAKE)
    wake_unresolved = replace(record(DAY1, "起了一下", 8, 0, kind="起床"), unresolved=("起床",))
    timing = settle(timing_claim, LATE_TO_WAKE, (LATE, wake_unresolved), now=at(DAY2, 12, 0))
    assert timing is not None and timing.outcome is Outcome.CENSORED
    # 未决的是别的概念 → 与这本账无关，照常缺席。
    other = replace(record(DAY1, "起了一下", 8, 0, kind="起床"), unresolved=("咖啡",))
    assert settle(timing_claim, LATE_TO_WAKE, (LATE, other), now=at(DAY2, 12, 0)).outcome is Outcome.ABSENT  # type: ignore[union-attr]


def test_probability_waits_for_its_window_then_reads_absent_or_censored() -> None:
    """没来就等到窗口末尾；过了、看清了 → 没来（ABSENT）；过了、没看清 → 删失。每本账只等自己那一个窗口，不数"第几次机会"。"""

    claim = claim_for(LATE_TO_BREAKFAST)
    assert settle(claim, LATE_TO_BREAKFAST, (LATE,), now=at(DAY1, 9, 0)) is None  # 窗口 06:15–09:15 还没过完
    absent = settle(claim, LATE_TO_BREAKFAST, (LATE,), now=at(DAY1, 9, 15))
    assert absent is not None and absent.outcome is Outcome.ABSENT and absent.observed_passes == 1 and not absent.is_right_censored
    assert [item.at for item in absent.passes] == [at(DAY1, 7, 45)]
    # 早餐时段没在看：没看清 → 删失（右删失），带原因。
    dark = FixedCoverage(1.0, dark=(WindowSpan(at(DAY1, 6, 0), at(DAY1, 10, 0)),))
    censored = settle(claim, LATE_TO_BREAKFAST, (LATE,), now=at(DAY1, 10, 0), coverage=dark)
    assert censored is not None and censored.outcome is Outcome.CENSORED and censored.is_right_censored
    passes = passes_until(claim.control, before_index=None, until=at(DAY1, 10, 0), coverage=dark, config=CONFIG)  # type: ignore[arg-type]
    assert [item.observed for item in passes] == [False]
    # 没有快照（树上没这个后件的数）：只能在后件到来时结，等不到就交生命周期。
    blind = claim_for(LATE_TO_BREAKFAST, control=False)
    assert settle(blind, LATE_TO_BREAKFAST, (LATE,), now=at(DAY3, 10, 0)) is None
    assert settle(blind, LATE_TO_BREAKFAST, (LATE, record(DAY1, "面", 7, 0, "早餐")), now=at(DAY1, 8, 0)) is None  # 没窗口就不知道"落在窗口里"
    closed = close_claim(claim, reason="约了之后一个月没去", now=at(DAY2, 12, 0), coverage=FixedCoverage(1.0), config=CONFIG)
    assert closed.outcome is Outcome.CENSORED and closed.reason == "约了之后一个月没去" and closed.observed_passes == 1


def test_an_enabling_hypothesis_is_the_same_probability_ledger_read_as_latency() -> None:
    """约球→打球：说不准第几次机会。打球是每天 19:00 一个机会；DAY3 19:00 去了 → 第 3 次机会、隔 57 小时。"""

    booking = record(DAY1, "约球", 10, 0, "约球")
    claim = Claim(
        hypothesis_identity=BOOKING_TO_BALL.identity,
        hypothesis_fingerprint=BOOKING_TO_BALL.fingerprint,
        aspect=Aspect.PROBABILITY,
        trigger_uri=booking.occurrence_uri,
        antecedent_hits=(ConceptHit("约球"),),
        antecedent_uris=(booking.occurrence_uri,),
        situation_snapshot=(),
        control=daily_snapshot("打球", booking.started_at, 16),
        created_at=NOW,
    )
    ball = record(DAY3, "打球", 19, 0, "打球")
    settlement = settle(claim, BOOKING_TO_BALL, (booking, ball), now=at(DAY3, 20, 0))
    assert settlement is not None and settlement.outcome is Outcome.OCCURRED
    assert settlement.latency_hours == 57.0 and settlement.opportunity_index == 3 and settlement.observed_passes == 2


def booking_claim(hyp, day=DAY1, hour: int = 10) -> tuple[Claim, object]:
    """一条无节律型承诺（约球 → 打球），返回 (承诺, 触发记录)。"""

    booking = record(day, "约球", hour, 0, "约球")
    claim = Claim(
        hypothesis_identity=hyp.identity,
        hypothesis_fingerprint=hyp.fingerprint,
        aspect=Aspect.PROBABILITY,
        trigger_uri=booking.occurrence_uri,
        antecedent_hits=(ConceptHit("约球"),),
        antecedent_uris=(booking.occurrence_uri,),
        situation_snapshot=(),
        control=daily_snapshot("打球", booking.started_at, 8),
        created_at=NOW,
    )
    return claim, booking


def test_a_open_ended_claim_is_never_censored_by_the_count_of_opportunities() -> None:
    """用户 09-27："很多行为没有机会时效。"挂号→就诊两周才去也要记上——无节律型过多少个机会都不删失。

    对照：节律型的账在自己那个窗口过完就收口了（没来）。
    """

    claim, booking = booking_claim(BOOKING_TO_BALL)
    # 快照 8 个机会（每天 19:00 一个）全部过完、一次没打球：节律型这时早收口了，无节律型仍然开着。
    far = at(DAY1, 10, 0) + timedelta(days=30)
    assert settle(claim, BOOKING_TO_BALL, (booking,), now=far) is None
    rhythmic = claim_for(LATE_TO_BREAKFAST)
    absent = settle(rhythmic, LATE_TO_BREAKFAST, (LATE,), now=at(DAY1, 10, 0) + timedelta(days=30))
    assert absent is not None and absent.outcome is Outcome.ABSENT
    # 两周后去了 → 照样兑现，记隔了多久。
    ball = record(DAY1 + timedelta(days=13), "打球", 19, 0, "打球")
    settlement = settle(claim, BOOKING_TO_BALL, (booking, ball), now=far)
    assert settlement is not None and settlement.outcome is Outcome.OCCURRED and settlement.latency_hours == pytest.approx(13 * 24 + 9.0)


def test_a_release_concept_voids_a_standing_open_ended() -> None:
    """再次约球取代上一次那次约：``released_by`` 命中 → RELEASED，记是哪条 occurrence 作废的；后件来了则兑现优先。"""

    claim, booking = booking_claim(REBOOKING_TO_BALL)
    again = record(DAY2, "又约球", 11, 0, "约球")
    released = settle(claim, REBOOKING_TO_BALL, (booking, again), now=at(DAY3, 1, 0))
    assert released is not None and released.outcome is Outcome.RELEASED and released.releasing_uri == again.occurrence_uri
    assert released.observed_at is None and released.fulfilling_uri is None and released.latency_hours is None
    # 作废之前就去打了球 → 兑现优先（那件事确实做了）。
    ball = record(DAY1, "打球", 19, 0, "打球")
    fulfilled = settle(claim, REBOOKING_TO_BALL, (booking, ball, again), now=at(DAY3, 1, 0))
    assert fulfilled is not None and fulfilled.outcome is Outcome.OCCURRED and fulfilled.fulfilling_uri == ball.occurrence_uri
    # 没写 released_by 的无节律型不认这条记录，继续开着（B9：没写就一直立着）。
    plain, plain_booking = booking_claim(BOOKING_TO_BALL)
    assert settle(plain, BOOKING_TO_BALL, (plain_booking, again), now=at(DAY3, 1, 0)) is None


def test_one_consequent_occurrence_fulfils_only_the_earliest_open_open_ended(tmp_path) -> None:
    """FIFO（评审 A-3）：约球两次、打球一次 → 只有最早那条兑现，第二条还开着；下一晚也不会被同一条打球再兑现一次。"""

    ledger = LedgerStore(tmp_path / "scene")
    hits = ConceptHitStore(tmp_path / "scene")
    first, first_booking = booking_claim(BOOKING_TO_BALL, DAY1, 10)
    second, second_booking = booking_claim(BOOKING_TO_BALL, DAY2, 11)
    ledger.write_claim(first)
    ledger.write_claim(second)
    ball = record(DAY3, "打球", 19, 0, "打球")
    for item in (first_booking, second_booking, ball):
        hits.write(item)
    night = at(DAY3, 23, 0)
    report = settle_due(
        BOOKING_TO_BALL, concepts=CONCEPTS, hits=hits, ledger=ledger, coverage=FixedCoverage(1.0), now=night, until=night, config=CONFIG
    )
    assert (report.settled, report.pending) == (1, 1)
    settled = ledger.settlements_for(BOOKING_TO_BALL.identity)
    assert len(settled) == 1 and settled[0].ref == first.ref and settled[0].fulfilling_uri == ball.occurrence_uri
    # 再跑一晚：盘上那条打球已经用掉了，第二条承诺不会被它兑现（``consumed`` 从盘上装起来）。
    later = at(DAY3, 23, 30)
    again = settle_due(
        BOOKING_TO_BALL, concepts=CONCEPTS, hits=hits, ledger=ledger, coverage=FixedCoverage(1.0), now=later, until=later, config=CONFIG
    )
    assert (again.settled, again.pending) == (0, 1)
    assert ledger.open_claims(BOOKING_TO_BALL.identity) == (second,)


# ── 时刻 ──────────────────────────────────────────────────────────────────────


def test_timing_reads_the_consequent_inside_its_window_or_records_absence() -> None:
    """七i 一：晚睡→起床、晚睡→就寝（补偿）都看这本账自己那个窗口——02:10 的锚之后第一个就寝窗口就是当晚，不再需要 12–36h 那种档。

    窗口之外的到来不归这本账（容差由机会口展宽窗口来给，不在这里再放宽）。"""

    wake_claim = claim_for(LATE_TO_WAKE)
    wake = record(DAY1, "起床", 9, 30, "起床")  # 窗口 06:40–09:40 之内
    observed = settle(wake_claim, LATE_TO_WAKE, (LATE, wake, record(DAY2, "再起", 8, 0, "起床")), now=at(DAY2, 9, 0))
    assert observed is not None and observed.outcome is Outcome.OBSERVED and observed.observed_at == at(DAY1, 9, 30) and observed.opportunity_index == 1
    # 窗口过完了没起床 → 缺席，不是"很晚"；12:00 才起也是缺席（这个窗口里没有）。
    absent = settle(wake_claim, LATE_TO_WAKE, (LATE,), now=at(DAY1, 12, 0))
    assert absent is not None and absent.outcome is Outcome.ABSENT and absent.opportunity_index == 1
    assert settle(wake_claim, LATE_TO_WAKE, (LATE, record(DAY1, "起床", 12, 0, "起床")), now=at(DAY1, 12, 30)).outcome is Outcome.ABSENT  # type: ignore[union-attr]
    assert settle(wake_claim, LATE_TO_WAKE, (LATE,), now=at(DAY1, 9, 0)) is None  # 窗口还没过完，不下结论
    # 那次机会没看清 → 删失。
    dark = FixedCoverage(1.0, dark=(WindowSpan(at(DAY1, 7, 0), at(DAY1, 9, 0)),))
    censored = settle(wake_claim, LATE_TO_WAKE, (LATE,), now=at(DAY2, 7, 0), coverage=dark)
    assert censored is not None and censored.outcome is Outcome.CENSORED

    # 补偿就寝：02:10 的锚之后第一个就寝窗口就是**当晚** 23:30（锚后 20.5 小时），所以 consequent_peak=1；
    # 22:40 睡 = 比常态早 50 分。
    bedtime_claim = claim_for(LATE_TO_BEDTIME)
    tonight = record(DAY1, "就寝", 22, 40, "就寝")
    compensation = settle(bedtime_claim, LATE_TO_BEDTIME, (LATE, tonight), now=at(DAY2, 12, 0))
    assert compensation is not None and compensation.outcome is Outcome.OBSERVED and compensation.observed_at == at(DAY1, 22, 40) and compensation.opportunity_index == 1
    assert settle(bedtime_claim, LATE_TO_BEDTIME, (LATE,), now=at(DAY1, 12, 0)) is None  # 当晚那个窗口还没开始


# ── 次数 ──────────────────────────────────────────────────────────────────────


def test_count_waits_until_the_horizon_of_opportunities_has_passed() -> None:
    """运动→咖啡（账挂在咖啡 #1 上），数接下来 3 个窗口：DAY1 19:00 打球后 #1 的第一次落地是次日 08:30，再轮 14:00、20:00。"""

    ball = record(DAY1, "打球", 19, 0, "打球")
    claim = Claim(
        hypothesis_identity=EXERCISE_TO_COFFEE.identity,
        hypothesis_fingerprint=EXERCISE_TO_COFFEE.fingerprint,
        aspect=Aspect.COUNT,
        trigger_uri=ball.occurrence_uri,
        antecedent_hits=(ConceptHit("运动"),),
        antecedent_uris=(ball.occurrence_uri,),
        situation_snapshot=(),
        control=daily_snapshot("咖啡", ball.started_at, 3),
        created_at=NOW,
    )
    coffees = (record(DAY1, "咖啡", 20, 30, "咖啡"), record(DAY2, "咖啡", 8, 0, "咖啡"), record(DAY2, "咖啡", 13, 30, "咖啡"), record(DAY2, "咖啡", 20, 0, "咖啡"))
    assert settle(claim, EXERCISE_TO_COFFEE, (ball, *coffees), now=at(DAY2, 20, 30)) is None  # 第 3 个窗口（20:00，到 20:45）还没过完
    counted = settle(claim, EXERCISE_TO_COFFEE, (ball, *coffees), now=at(DAY2, 21, 0))
    assert counted is not None and counted.outcome is Outcome.COUNTED and counted.count == 3  # DAY1 20:30 那杯在第一个窗口之前，不算
    zero = settle(claim, EXERCISE_TO_COFFEE, (ball,), now=at(DAY2, 21, 0))
    assert zero is not None and zero.count == 0  # 0 也是计数
    dark = FixedCoverage(1.0, dark=(WindowSpan(at(DAY2, 8, 0), at(DAY2, 9, 0)),))
    assert settle(claim, EXERCISE_TO_COFFEE, (ball, *coffees), now=at(DAY2, 21, 0), coverage=dark).outcome is Outcome.CENSORED  # type: ignore[union-attr]


# ── 整条走 ────────────────────────────────────────────────────────────────────


def test_settle_due_walks_a_hypothesis_ledger_end_to_end(tmp_path) -> None:
    hits = ConceptHitStore(tmp_path / "scene")
    ledger = LedgerStore(tmp_path / "scene")
    hits.write(LATE)
    hits.write(record(DAY1, "吃了碗面", 7, 0, "早餐"))
    hits.write(record(DAY2, "就寝", 1, 0, ConceptHit("晚睡", "轻")))  # 第二晚没吃早饭
    for day in (DAY1, DAY2):
        open_claims_for_day(day, hypotheses=(LATE_TO_BREAKFAST,), concepts=CONCEPTS, hits=hits, ledger=ledger, opportunities=TableOpportunities(), now=NOW)
    assert len(ledger.open_claims(LATE_TO_BREAKFAST.identity)) == 2

    report = settle_due(LATE_TO_BREAKFAST, concepts=CONCEPTS, hits=hits, ledger=ledger, coverage=FixedCoverage(1.0), now=at(DAY2, 8, 0), until=at(DAY2, 8, 0), config=CONFIG)
    assert (report.settled, report.pending) == (1, 1)  # 第一条来了；第二条的窗口（DAY2 06:15–09:15）还没过完
    report = settle_due(LATE_TO_BREAKFAST, concepts=CONCEPTS, hits=hits, ledger=ledger, coverage=FixedCoverage(1.0), now=at(DAY2, 12, 0), until=at(DAY2, 12, 0), config=CONFIG)
    assert (report.settled, report.pending) == (1, 0)  # 窗口过完、看清了、没来
    outcomes = [s.outcome for s in ledger.settlements_for(LATE_TO_BREAKFAST.identity)]
    assert outcomes == [Outcome.OCCURRED, Outcome.ABSENT]
    assert settle_due(LATE_TO_BREAKFAST, concepts=CONCEPTS, hits=hits, ledger=ledger, coverage=FixedCoverage(1.0), now=at(DAY3, 0, 0), until=at(DAY3, 0, 0), config=CONFIG).settled == 0


def test_a_deleted_consequent_concept_stops_the_account_instead_of_fabricating_censors(tmp_path) -> None:
    from habitus.scene.concepts import ConceptSet

    hits = ConceptHitStore(tmp_path / "scene")
    ledger = LedgerStore(tmp_path / "scene")
    hits.write(LATE)
    hits.write(record(DAY1, "吃了碗面", 7, 0, "早餐"))
    open_claims_for_day(DAY1, hypotheses=(LATE_TO_BREAKFAST,), concepts=CONCEPTS, hits=hits, ledger=ledger, opportunities=TableOpportunities(), now=NOW)
    without = ConceptSet([CONCEPTS[identity] for identity in CONCEPTS if identity != "早餐"])
    report = settle_due(LATE_TO_BREAKFAST, concepts=without, hits=hits, ledger=ledger, coverage=FixedCoverage(1.0), now=at(DAY3, 12, 0), until=at(DAY3, 12, 0), config=CONFIG)
    assert (report.settled, report.pending) == (0, 1) and ledger.settlements_for(LATE_TO_BREAKFAST.identity) == ()
    assert settle_due(LATE_TO_BREAKFAST, concepts=CONCEPTS, hits=hits, ledger=ledger, coverage=FixedCoverage(1.0), now=at(DAY1, 14, 30), until=at(DAY1, 14, 30), config=CONFIG).settled == 1
    assert ledger.settlements_for(LATE_TO_BREAKFAST.identity)[0].outcome is Outcome.OCCURRED


# ── 机会序列的边界（评审 A-2 / B5） ─────────────────────────────────────────────


def test_the_first_opportunity_is_the_peak_the_anchor_sits_in() -> None:
    """就寝 07:00 命中「晚睡·重」（设计稿 十③ 的例子），起床 09:30：当天 06:40–09:40 那个窗口就是这本账的窗口。

    按"峰的开始晚于锚"铺会把这个窗口丢掉，起床被算到次日 08:10 那个窗口上 → 时刻差 −1360 分（真值 +80）。
    12:00 才起则是窗口里没有（ABSENT）：容差由机会口展宽窗口来给，账本不再第二次放宽。
    """

    late = record(DAY1, "就寝", 7, 0, ConceptHit("晚睡", "重"))
    claim = Claim(
        hypothesis_identity=LATE_TO_WAKE.identity,
        hypothesis_fingerprint=LATE_TO_WAKE.fingerprint,
        aspect=LATE_TO_WAKE.aspect,
        trigger_uri=late.occurrence_uri,
        antecedent_hits=(ConceptHit("晚睡", "重"),),
        antecedent_uris=(late.occurrence_uri,),
        situation_snapshot=(),
        control=daily_snapshot("起床", late.started_at, 16),
        created_at=NOW,
    )
    first = claim.control.at(1)
    assert first is not None and first.at == at(DAY1, 8, 10) and first.span.start < claim.anchor < first.span.end
    wake = record(DAY1, "起床", 9, 30, "起床")
    settlement = settle(claim, LATE_TO_WAKE, (late, wake), now=at(DAY2, 12, 0))
    assert settlement is not None and settlement.outcome is Outcome.OBSERVED and settlement.opportunity_index == 1
    assert (settlement.observed_at - first.at).total_seconds() / 60.0 == 80.0
    noon = settle(claim, LATE_TO_WAKE, (late, record(DAY1, "起床", 12, 0, "起床")), now=at(DAY2, 12, 0))
    assert noon is not None and noon.outcome is Outcome.ABSENT


def test_a_snapshot_too_short_for_the_horizon_settles_instead_of_hanging_forever() -> None:
    """次数账要数 horizon 个窗口，而快照只有 2 个：快照事后不改，等下去也等不出来 → 删失带原因，不挂在 pending 里。"""

    ball = record(DAY1, "打球", 19, 0, "打球")
    claim = Claim(
        hypothesis_identity=EXERCISE_TO_COFFEE.identity,
        hypothesis_fingerprint=EXERCISE_TO_COFFEE.fingerprint,
        aspect=EXERCISE_TO_COFFEE.aspect,
        trigger_uri=ball.occurrence_uri,
        antecedent_hits=(ConceptHit("运动"),),
        antecedent_uris=(ball.occurrence_uri,),
        situation_snapshot=(),
        control=daily_snapshot("咖啡", ball.started_at, 2),
        created_at=NOW,
    )
    settlement = settle(claim, EXERCISE_TO_COFFEE, (ball,), now=at(DAY1, 12, 0))
    assert settlement is not None and settlement.outcome is Outcome.CENSORED and settlement.reason is not None and "数不到第 3 个" in settlement.reason


def test_settling_never_runs_past_the_day_the_hits_are_mapped_to(tmp_path) -> None:
    """``until`` 不给就取"最后一个已映射日的日界"：夜批用墙钟当 now 会越过还没映射的日子，跨午夜的后件就被判成
    "没来"，而 add-only 撤不掉（评审 B2）。"""

    hits = ConceptHitStore(tmp_path / "scene")
    ledger = LedgerStore(tmp_path / "scene")
    hits.write(LATE)
    open_claims_for_day(DAY1, hypotheses=(LATE_TO_BREAKFAST,), concepts=CONCEPTS, hits=hits, ledger=ledger, opportunities=TableOpportunities(), now=NOW, config=CONFIG)
    # 一天都没盖章 → 什么都不可信，一条都不结。
    assert mapped_until(hits, at(DAY3, 12, 0)) is None
    assert settle_due(LATE_TO_BREAKFAST, concepts=CONCEPTS, hits=hits, ledger=ledger, coverage=FixedCoverage(1.0), now=at(DAY3, 12, 0), config=CONFIG).settled == 0
    # 映射了 DAY1：DAY1 的早餐窗（06:15–09:15）整个在可信范围里 → 可以读成"没来"；跨午夜的窗口要等到次日映射完才行。
    hits.complete_day(DAY1, records=1, completed_at=NOW, mapper=LATE.mapper)
    assert mapped_until(hits, at(DAY3, 12, 0)) == at(DAY2, 0, 0)
    report = settle_due(LATE_TO_BREAKFAST, concepts=CONCEPTS, hits=hits, ledger=ledger, coverage=FixedCoverage(1.0), now=at(DAY3, 12, 0), config=CONFIG)
    assert report.settled == 1 and ledger.settlements_for(LATE_TO_BREAKFAST.identity)[0].outcome is Outcome.ABSENT
    # 跨午夜的补偿就寝（窗口 22:00–01:00）：DAY1 映射完还不能结，要等 DAY2 也映射完。
    hits.write(record(DAY2, "就寝", 1, 0, ConceptHit("晚睡", "轻")))
    open_claims_for_day(DAY1, hypotheses=(LATE_TO_BEDTIME,), concepts=CONCEPTS, hits=hits, ledger=ledger, opportunities=TableOpportunities(), now=NOW, config=CONFIG)
    assert settle_due(LATE_TO_BEDTIME, concepts=CONCEPTS, hits=hits, ledger=ledger, coverage=FixedCoverage(1.0), now=at(DAY3, 12, 0), config=CONFIG).settled == 0
    hits.complete_day(DAY2, records=1, completed_at=NOW, mapper=LATE.mapper)
    assert settle_due(LATE_TO_BEDTIME, concepts=CONCEPTS, hits=hits, ledger=ledger, coverage=FixedCoverage(1.0), now=at(DAY3, 12, 0), config=CONFIG).settled == 1
