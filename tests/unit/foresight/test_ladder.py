"""递法（裁定 48–51、58、61）：边的读数从单元历史里这条边的那几刻 fold 出来；两档；下界过线升、上界掉线降、中间维持；所有边
同一条线；人的设置按行为；同一条预期后写的覆盖先写的；递送决定的名额。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from habitus.foresight import (
    Evidence,
    ForesightError,
    Tolerance,
    account_key,
    climb,
    current_facts,
    decide,
    ranked_account,
)
from habitus.foresight.edges import PastMoment
from habitus.foresight.ledger import AccountFact, AccountLevel, Expectation, UnitSetting

CST = timezone(timedelta(hours=8))
NOW = datetime(2026, 9, 1, 21, 0, tzinfo=CST)
KIND = "s-k0058"
ACCOUNT = account_key(KIND, "日子", "weekday=1")
LOSS = Tolerance(line=2 / 3)  # 线 = 1 − 可接受的误提醒率（0.33），所有边同一条
EVIDENCE = Evidence(alpha=0.05, half_life=8.0)


def moment(days_ago: float, came: bool, *, strength: float = 1.0, others: frozenset[str] = frozenset()) -> PastMoment:
    return PastMoment(NOW - timedelta(days=days_ago), frozenset({ACCOUNT}) | others, came, strength)


def run(results: list[bool], *, start: float = 70.0, step: float = 7.0) -> list[PastMoment]:
    return [moment(start - step * index, came) for index, came in enumerate(results)]


def fact(days_ago: float, good: bool, *, kind: str | None = None, strength: float = 1.0, n: int = 0) -> AccountFact:
    chosen = kind or ("准时" if good else "过期待决")
    return AccountFact(ACCOUNT, NOW - timedelta(days=days_ago), good, strength, chosen, f"e{n}-{days_ago}")


def standing(moments, *, now: datetime = NOW, **kwargs):  # type: ignore[no-untyped-def]
    return climb(tuple(moments), ACCOUNT, now=now, tolerance=LOSS, evidence=EVIDENCE, **kwargs)


def test_a_new_edge_is_shadow_and_enough_hits_lift_it_past_the_line() -> None:
    empty = standing(())
    assert empty.level == 0 and empty.prediction.n == 0 and empty.threshold == pytest.approx(2 / 3)
    # 线 0.667：两次全中下界只有 0.4 多，不够；连中十次够了
    assert standing(run([True, True])).level == 0
    assert standing(run([True] * 10)).level == 1
    # 一半一半的边：上界也掉到线下，回影子
    assert standing(run([True, False] * 8)).level == 0


def test_only_the_moments_that_carry_this_edge_count_and_only_before_now() -> None:
    """一条边的历史是签名里含它的那几刻：别的输入独自在场的那几刻不算；此刻之后的刻也不算（诊断重建"那一刻的边"）。"""

    other = account_key(KIND, "事件", "session|x|y|chain")
    mine = run([True] * 10)
    theirs = [PastMoment(NOW - timedelta(days=3 - i), frozenset({other}), False) for i in range(3)]
    assert standing([*mine, *theirs]).prediction == standing(mine).prediction
    future = [moment(-1.0, False), moment(-2.0, False)]
    assert standing([*mine, *future]).prediction == standing(mine).prediction


def test_between_the_bounds_the_previous_level_holds() -> None:
    moments = run([True] * 6 + [False, True, True])
    middle = standing(moments)
    assert middle.prediction.lower < middle.threshold <= middle.prediction.upper
    assert middle.level == 0
    held = standing(moments, previous=AccountLevel(ACCOUNT, NOW - timedelta(days=3), 1))
    assert held.level == 1


@pytest.mark.parametrize("history", [8, 20, 100])
def test_a_perfect_edge_that_stops_falls_to_shadow_on_the_fourth_miss_however_long_its_history(history: int) -> None:
    """按机会数遗忘（裁定 57）：全中的边停了，连落空 3 次仍在提醒档，第 4 次掉进影子——历史 8 次、20 次、100 次都一样，
    旧证据攒得再多也压不住最近几次机会（只读最近 M 刻，窗外的分量本来就低于 α）。前因不再出现、没有新机会的边读数不褪。"""

    held = AccountLevel(ACCOUNT, NOW - timedelta(days=600), 1)
    hits = [moment(500.0 - index, True) for index in range(history)]

    def after(misses: int):  # type: ignore[no-untyped-def]
        return standing([*hits, *[moment(10.0 - index, False) for index in range(misses)]], previous=held)

    assert after(3).level == 1 and after(3).prediction.upper >= LOSS.threshold
    assert after(4).level == 0 and after(4).prediction.upper < LOSS.threshold
    if history == 100:
        assert after(3).prediction.upper == pytest.approx(0.78, abs=0.01)
        assert after(4).prediction.upper == pytest.approx(0.56, abs=0.01)
    quiet = standing(hits, now=NOW + timedelta(days=365), previous=held)
    assert quiet.prediction == standing(hits, previous=held).prediction  # 一年没有新机会：读数原样


def test_strength_weights_the_moment() -> None:
    full = standing(run([True] * 4))
    weak = standing([moment(30 - i, True, strength=0.1) for i in range(4)])
    assert weak.prediction.n < full.prediction.n / 5
    assert weak.prediction.lower < full.prediction.lower


def test_a_run_of_misses_that_the_old_rate_cannot_explain_discounts_what_came_before() -> None:
    """天天做、命中 88% 的边：连漏两次在它的读数下不稀奇，不算断；连漏六次就是规律断了，断点之前的刻打折、边降下来；
    之后又做了，那串连错不再是"最近的"，打折解除（扣分可恢复）。从没漏过的边连漏三次就会轻罚一下。"""

    history = [moment(60.0 - i, i % 8 != 7) for i in range(50)]
    two = standing([*history, *[moment(1.5 - 0.5 * i, False) for i in range(2)]], previous=None)
    assert two.break_at is None and two.discount == 1.0
    six = [*history, *[moment(6.0 - i, False) for i in range(6)]]
    broke = standing(six, previous=AccountLevel(ACCOUNT, NOW - timedelta(days=30), 1))
    assert broke.break_at == NOW - timedelta(days=6.0) and broke.discount < 0.3
    assert broke.level == 0
    healed = standing([*six, moment(0.5, True)], previous=None)
    assert healed.break_at is None and healed.discount == 1.0
    perfect = [moment(60.0 - i, True) for i in range(50)]
    nudged = standing([*perfect, *[moment(1.5 - 0.5 * i, False) for i in range(3)]])
    assert nudged.break_at is not None and 0.1 < nudged.discount < 1.0
    # 一条本来就弱的边：同样六次连错在它的读数下不算断
    weak = [moment(60.0 - i, i % 2 == 0) for i in range(50)]
    assert standing([*weak, *[moment(6.0 - i, False) for i in range(6)]]).break_at is None


def test_a_run_of_misses_is_counted_by_weight_and_ended_by_anything_he_did() -> None:
    """天天做、命中 88%：连漏四次不算断、五次算；后补只算它打的那一折；做了的一刻哪怕分量是 0 也打断连错。"""

    history = [moment(60.0 - i, i % 8 != 7) for i in range(57)]
    four = [moment(3.0 - 0.5 * i, False) for i in range(4)]
    assert standing([*history, *four]).break_at is None
    assert standing([*history, *four, moment(0.5, False)]).break_at is not None
    weak = moment(0.5, False, strength=0.1)
    assert standing([*history, *four, weak]).break_at is None
    odd_hit = moment(3.2, True, strength=0.0)
    misses = [moment(3.4, False), moment(3.3, False), odd_hit, moment(2.0, False), moment(1.0, False)]
    assert standing([*history, *misses]).break_at is None


def test_settings_are_per_behaviour_and_override_the_reading() -> None:
    moments = run([True] * 12, start=90.0)
    always = UnitSetting(KIND, NOW - timedelta(days=1), "以后都提醒我")
    never = UnitSetting(KIND, NOW - timedelta(days=1), "以后别提了")
    assert standing((), setting=always).level == 1 and standing((), setting=always).setting == "以后都提醒我"
    assert standing(moments, setting=never).level == 0
    assert standing(moments, setting=UnitSetting(KIND, NOW - timedelta(days=1), "照常")).level == 1
    # 还没生效的设置不算
    assert standing(moments, setting=UnitSetting(KIND, NOW + timedelta(days=1), "以后别提了")).level == 1


def test_a_later_fact_for_the_same_expectation_replaces_the_earlier_one() -> None:
    overdue = AccountFact(ACCOUNT, NOW - timedelta(days=3), False, 1.0, "过期待决", "e1")
    caught = AccountFact(ACCOUNT, NOW - timedelta(days=2), False, 0.0, "后补", "e1")
    other = AccountFact(ACCOUNT, NOW - timedelta(days=3), True, 1.0, "准时", "e2")
    assert current_facts([overdue, other, caught]) == (other, caught)
    # 同一晚先结成过期待决、又裁成后补：同一时刻，改判在后
    same = AccountFact(ACCOUNT, overdue.at, False, 0.0, "后补", "e1")
    assert current_facts([same, overdue]) == (same,)


def test_every_edge_has_the_same_line_this_version() -> None:
    """裁定 58 第 1 条：扣发与急缓都先不做，线就是 1 − 可接受的误提醒率，不随边变。"""

    assert LOSS.threshold == pytest.approx(2 / 3)
    assert standing(()).threshold == standing(run([True] * 20)).threshold == pytest.approx(2 / 3)


def test_tolerance_and_evidence_refuse_nonsense() -> None:
    with pytest.raises(ForesightError):
        Tolerance(line=1.0)
    with pytest.raises(ForesightError):
        Tolerance(line=True)  # type: ignore[arg-type]
    with pytest.raises(ForesightError):
        Evidence(alpha=0.6, half_life=8.0)
    with pytest.raises(ForesightError):
        Evidence(alpha=0.05, half_life=0.0)
    with pytest.raises(ForesightError):
        ranked_account((), visible_at=NOW.replace(tzinfo=None))
    assert EVIDENCE.window == 35


def expectation() -> Expectation:
    return Expectation(
        expectation_id="e",
        kind_token=KIND,
        day=NOW.date(),
        formed_at=NOW,
        source="日子",
        trigger="weekday=1",
        trigger_instance=NOW.date().isoformat(),
        window=(NOW, NOW + timedelta(minutes=30)),
        deadline=NOW + timedelta(hours=3),
        basis=(),
        opportunities=10,
        hits=9,
        observed_from=NOW.replace(hour=0),
    )


def test_the_delivery_decision_records_the_level_and_the_cap() -> None:
    """这一版不扣发（裁定 58 第 1 条）：提醒档没满名额就说，说出的概率是 1。"""

    item = expectation()
    shadow = decide(item, 0, now=NOW, spoken_today=0, daily_cap=6)
    assert (shadow.reason, shadow.spoken, shadow.level) == ("影子", False, 0)
    spoken = decide(item, 1, now=NOW, spoken_today=0, daily_cap=6, text="往常这会儿你会跑步。")
    assert (spoken.reason, spoken.spoken, spoken.text) == ("说了", True, "往常这会儿你会跑步。")
    full = decide(item, 1, now=NOW, spoken_today=6, daily_cap=6)
    assert full.reason == "名额已满"
    assert spoken.p_speak == 1.0
    with pytest.raises(ForesightError):
        decide(item, 2, now=NOW, spoken_today=0, daily_cap=6)


def test_a_revision_overrides_by_when_it_was_written_but_stays_at_its_own_opportunity() -> None:
    """后补、重结写在几天之后：谁覆盖谁按写下的时刻，先后按那次机会的时刻（改判不能把旧机会变成最新的证据）。"""

    original = fact(10.0, False, n=1)
    caught = replace(fact(10.0, False, kind="后补", strength=0.5, n=1), written_at=NOW - timedelta(days=7))
    (live,) = current_facts([caught, original])
    assert live is caught and live.at == NOW - timedelta(days=10.0)
    # 重结写下的不是改判种类：同一机会时刻，读回的先后也不定，只能靠写下的时刻分先后
    resettled = replace(fact(10.0, True, n=1), written_at=NOW - timedelta(days=1))
    (live,) = current_facts([resettled, original])
    assert live is resettled
    # 写下的时刻早于机会的时刻说不通
    with pytest.raises(ForesightError):
        replace(original, written_at=NOW - timedelta(days=11))
