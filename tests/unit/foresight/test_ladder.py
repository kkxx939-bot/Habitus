"""递法（裁定 48–51）：两档；预测轴下界过线升、上界掉线降、中间维持；重的线低；「这次不用」压回并要重新攒；人的设置覆盖；
同一条预期同一条轴后写的覆盖先写的；递送决定的名额与扣发只扣轻的。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from habitus.foresight import Evidence, ForesightError, Tolerance, account_key, climb, current_facts, decide
from habitus.foresight.ledger import AccountFact, AccountLevel, AccountSetting, AccountSeverity, Expectation

CST = timezone(timedelta(hours=8))
NOW = datetime(2026, 9, 1, 21, 0, tzinfo=CST)
ACCOUNT = account_key("s-k0058", "钟面", "weekday=1")
LOSS = Tolerance(floor=0.2, ceiling=0.9, unjudged=0.5)  # 线 = 1 / (1 + 代价比 × 效果)：代价比 2 → 0.33、没判过 0.5 → 0.67
EVIDENCE = Evidence(alpha=0.05, half_life_days=60.0)


def fact(days_ago: float, good: bool, *, axis: str = "预测", kind: str | None = None, strength: float = 1.0, n: int = 0) -> AccountFact:
    chosen = kind or ("准时" if good else ("过期待决" if axis == "预测" else "落空"))
    return AccountFact(ACCOUNT, NOW - timedelta(days=days_ago), axis, good, strength, chosen, f"e{n}-{days_ago}")


def run(results: list[bool], *, start: float = 70.0, step: float = 7.0, axis: str = "预测") -> list[AccountFact]:
    return [fact(start - step * index, good, axis=axis, n=index) for index, good in enumerate(results)]


def standing(facts, **kwargs):  # type: ignore[no-untyped-def]
    return climb(facts, tolerance=LOSS, evidence=EVIDENCE, now=NOW, **kwargs)


def heavy(at: datetime = NOW - timedelta(days=1)) -> AccountSeverity:
    return AccountSeverity(ACCOUNT, at, 2.0, "漏了要补办", at + timedelta(days=60))


def test_a_new_account_is_shadow_and_enough_hits_lift_it_past_the_light_line() -> None:
    empty = standing(())
    assert empty.level == 0 and empty.prediction.n == 0 and empty.threshold == pytest.approx(2 / 3)
    # 轻的线 0.667：两次全中下界只有 0.4 多，不够；连中十次够了
    assert standing(run([True, True])).level == 0
    assert standing(run([True] * 10)).level == 1
    # 一半一半的账：上界也掉到线下，回影子
    assert standing(run([True, False] * 8)).level == 0


def test_a_grave_matter_has_a_low_line_and_speaks_after_two_hits_not_one() -> None:
    two = run([True, True], start=10.0, step=3.0)
    assert standing(two).level == 0
    lifted = standing(two, severity=heavy())
    assert lifted.level == 1 and lifted.threshold == pytest.approx(1 / 3)
    assert standing(run([True], start=3.0), severity=heavy()).level == 0  # 一次不够（下界 0.23）
    # 过期的重轻不算：回到轻的线
    stale = AccountSeverity(ACCOUNT, NOW - timedelta(days=90), 2.0, "", NOW - timedelta(days=30))
    assert standing(two, severity=stale).level == 0


def test_between_the_bounds_the_previous_level_holds() -> None:
    facts = run([True] * 6 + [False, True, True])
    middle = standing(facts)
    assert middle.prediction.lower < middle.threshold <= middle.prediction.upper
    assert middle.level == 0
    held = standing(facts, previous=AccountLevel(ACCOUNT, NOW - timedelta(days=3), 1))
    assert held.level == 1


def test_forgetting_lets_a_quiet_account_fall_back_to_shadow() -> None:
    fresh = run([True] * 10, start=20.0, step=2.0)
    assert standing(fresh).level == 1
    old = run([True] * 10, start=400.0, step=2.0)
    reading = standing(old)
    assert reading.prediction.n < 1.0 and reading.level == 0


def test_strength_weights_the_fact() -> None:
    full = standing(run([True] * 4))
    weak = standing([fact(30 - i, True, strength=0.1, n=i) for i in range(4)])
    assert weak.prediction.n < full.prediction.n / 5
    assert weak.prediction.lower < full.prediction.lower


def test_a_decline_is_one_bad_fact_not_a_reset() -> None:
    """「这次不用」只是提醒组上一笔坏（裁定 54、58）：预测轴的证据不清空，账在原地往下移，不回影子重攒。"""

    facts = run([True] * 12, start=90.0)
    previous = AccountLevel(ACCOUNT, NOW - timedelta(days=10), 1)
    before = standing(facts, previous=previous)
    declined = [*facts, fact(2.0, False, axis="提醒组", kind="这次不用")]
    after = standing(declined, previous=previous)
    assert after.prediction.n == before.prediction.n  # 预测轴没被清空
    assert after.reminded.n == pytest.approx(1.0, abs=0.05) and after.reminded.k == 0.0
    assert after.level == 1  # 一次不用，提醒组上界还在线上：不降
    # 一而再地不用：提醒组上界掉到线下，才降
    many = [*facts, *[fact(5.0 - i, False, axis="提醒组", kind="这次不用", n=200 + i) for i in range(5)]]
    assert standing(many, previous=previous).level == 0


def test_a_run_of_misses_that_the_old_rate_cannot_explain_discounts_what_came_before() -> None:
    """天天做、命中 88% 的账：连漏两次在它的读数下不稀奇，不算断；连漏六次就是规律断了，断点之前的事实打折、账降下来；
    之后又做了，那串连错不再是"最近的"，打折解除（扣分可恢复）。从没漏过的账连漏两次就会轻罚一下（它自己的读数下确实反常）。"""

    history = [fact(60.0 - i, i % 8 != 7, n=i) for i in range(50)]
    two = standing([*history, *[fact(1.5 - 0.5 * i, False, n=100 + i) for i in range(2)]], previous=None)
    assert two.break_at is None and two.discount == 1.0
    six = [*history, *[fact(6.0 - i, False, n=100 + i) for i in range(6)]]
    broke = standing(six, previous=AccountLevel(ACCOUNT, NOW - timedelta(days=30), 1))
    assert broke.break_at == NOW - timedelta(days=6.0) and broke.discount < 0.05
    assert broke.level == 0
    healed = standing([*six, fact(0.5, True, n=300)], previous=None)
    assert healed.break_at is None and healed.discount == 1.0
    perfect = [fact(60.0 - i, True, n=i) for i in range(50)]
    nudged = standing([*perfect, *[fact(1.5 - 0.5 * i, False, n=100 + i) for i in range(2)]])
    assert nudged.break_at is not None and 0.1 < nudged.discount < 1.0
    # 一本本来就弱的账：同样六次连错在它的读数下不算断
    weak = [fact(60.0 - i, i % 2 == 0, n=i) for i in range(50)]
    assert standing([*weak, *[fact(6.0 - i, False, n=100 + i) for i in range(6)]]).break_at is None


def test_the_reminded_group_demotes_when_reminders_keep_being_ignored() -> None:
    """说了之后他老不做：提醒组的完成率（就是这本账说出口之后的误提醒率的反面）上界掉到线下，降回影子——预测轴再好也一样。"""

    facts = run([True] * 12, start=90.0) + run([False] * 10, start=40.0, step=3.0, axis="提醒组")
    reading = standing(facts, previous=AccountLevel(ACCOUNT, NOW - timedelta(days=50), 1))
    assert reading.prediction.lower >= reading.threshold and reading.reminded.upper < reading.threshold
    assert reading.level == 0
    # 扣发组只是对照，不进升降：扣下没说之后他全没做，账照样在提醒档
    held = run([True] * 12, start=90.0) + run([False] * 10, start=40.0, step=3.0, axis="扣发组")
    assert standing(held, previous=AccountLevel(ACCOUNT, NOW - timedelta(days=50), 1)).level == 1


def test_the_effect_of_reminding_is_the_reminded_minus_the_held_out_completion() -> None:
    """提醒的效果 = 说了之后做了的比例 − 扣下没说之后做了的比例；两组都有过才有读数。"""

    base = run([True] * 12, start=90.0)
    assert standing(base).effect is None
    reminded = run([True] * 8 + [False] * 2, start=60.0, step=3.0, axis="提醒组")
    assert standing([*base, *reminded]).effect is None  # 只有提醒组
    held = run([True] * 3 + [False] * 7, start=59.0, step=3.0, axis="扣发组")
    effect = standing([*base, *reminded, *held]).effect
    assert effect is not None and effect > 0.3


def test_each_account_has_its_own_line_that_moves_with_the_cost_ratio_and_the_effect_of_reminding() -> None:
    """裁定 58 第 1 条：线 = 1 / (1 + c × e)，夹在上下限之间。c 是这件事本身的代价比（连续、按账判），e 是提醒效果的上界；
    不分档、不写死：代价比 2 → 0.33、0.5 → 0.67、1 → 0.5，3 → 0.25；提醒对他没用（e 掉下来）时线抬起来。"""

    assert LOSS.threshold(None) == pytest.approx(1 / 1.5)  # 没判过：按默认代价比 0.5
    assert LOSS.threshold(0.5) == pytest.approx(1 / 1.5)
    assert LOSS.threshold(1.0) == pytest.approx(0.5)
    assert LOSS.threshold(2.0) == pytest.approx(1 / 3)
    assert LOSS.threshold(3.0) == pytest.approx(0.25)
    assert LOSS.threshold(2.0, effect=0.5) == pytest.approx(0.5)  # 同一件重的事，提醒只有一半效果：线从 0.33 升到 0.5
    assert LOSS.threshold(10.0) == pytest.approx(0.2)  # 下限
    assert LOSS.threshold(0.1, effect=0.1) == pytest.approx(0.9)  # 上限
    with pytest.raises(ForesightError):
        LOSS.threshold(2.0, effect=1.5)


def test_the_effect_bound_is_one_until_the_held_out_group_shows_reminding_does_not_help() -> None:
    """效果取上界：两组样本少时上界宽、线只动一点；扣下没说他也照做（提醒组 ≈ 扣发组）攒多了，上界掉下来、线明显抬起来。"""

    base = run([True] * 20, start=90.0, step=2.0)
    assert standing(base).effect_bound == 1.0  # 没有扣发组
    few = [*base, *run([True, True], start=30.0, axis="提醒组"), *run([True, True], start=29.0, axis="扣发组")]
    assert 0.6 < standing(few).effect_bound < 1.0  # 各两次：扣下没说那两次他也做了，上界已经不到 1，但还宽
    same = [*base, *run([True] * 30, start=60.0, step=1.0, axis="提醒组"), *run([True] * 30, start=59.5, step=1.0, axis="扣发组")]
    useless = standing(same)
    assert useless.effect_bound < 0.3 and useless.threshold > standing(base).threshold

def test_settings_override_the_reading() -> None:
    facts = run([True] * 12, start=90.0)
    always = AccountSetting(ACCOUNT, NOW - timedelta(days=1), "以后都提醒我")
    never = AccountSetting(ACCOUNT, NOW - timedelta(days=1), "以后别提了")
    assert standing((), setting=always).level == 1 and standing((), setting=always).setting == "以后都提醒我"
    assert standing(facts, setting=never).level == 0
    assert standing(facts, setting=AccountSetting(ACCOUNT, NOW - timedelta(days=1), "照常")).level == 1


def test_a_later_fact_for_the_same_expectation_and_axis_replaces_the_earlier_one() -> None:
    overdue = AccountFact(ACCOUNT, NOW - timedelta(days=3), "预测", False, 1.0, "过期待决", "e1")
    caught = AccountFact(ACCOUNT, NOW - timedelta(days=2), "预测", False, 0.0, "后补", "e1")
    other = AccountFact(ACCOUNT, NOW - timedelta(days=3), "提醒组", False, 1.0, "落空", "e1")
    assert current_facts([overdue, other, caught]) == (other, caught)
    # 同一晚先结成过期待决、又裁成后补：同一时刻，改判在后
    same = AccountFact(ACCOUNT, overdue.at, "预测", False, 0.0, "后补", "e1")
    assert current_facts([same, overdue]) == (same,)


def test_tolerance_and_evidence_refuse_nonsense() -> None:
    with pytest.raises(ForesightError):
        Tolerance(floor=0.6, ceiling=0.5, unjudged=0.5)  # 下限要在上限之下
    with pytest.raises(ForesightError):
        Tolerance(floor=0.2, ceiling=1.0, unjudged=0.5)
    with pytest.raises(ForesightError):
        Evidence(alpha=0.6, half_life_days=60.0)
    with pytest.raises(ForesightError):
        Evidence(alpha=0.05, half_life_days=0.0)
    with pytest.raises(ForesightError):
        climb((), tolerance=LOSS, evidence=EVIDENCE, now=NOW.replace(tzinfo=None))


def expectation() -> Expectation:
    return Expectation(
        expectation_id="e",
        kind_token="s-k0058",
        day=NOW.date(),
        formed_at=NOW,
        source="钟面",
        trigger="weekday=1",
        trigger_instance=NOW.date().isoformat(),
        window=(NOW, NOW + timedelta(minutes=30)),
        deadline=NOW + timedelta(hours=3),
        basis=(),
        opportunities=10,
        hits=9,
        observed_from=NOW.replace(hour=0),
    )


def test_the_delivery_decision_records_the_level_the_cap_and_the_holdout_rate_it_was_given() -> None:
    """扣发比例由调用方按这本账的轻重缓急取（裁定 58 第 7 条：重的也扣、比例更小），这里只按给的比例抽。"""

    item = expectation()
    shadow = decide(item, 0, now=NOW, spoken_today=0, daily_cap=6)
    assert (shadow.reason, shadow.spoken, shadow.level) == ("影子", False, 0)
    spoken = decide(item, 1, now=NOW, spoken_today=0, daily_cap=6, text="往常这会儿你会跑步。")
    assert (spoken.reason, spoken.spoken, spoken.text) == ("说了", True, "往常这会儿你会跑步。")
    full = decide(item, 1, now=NOW, spoken_today=6, daily_cap=6)
    assert full.reason == "名额已满"
    held = decide(item, 1, now=NOW, spoken_today=0, daily_cap=6, holdout_rate=0.2, draw=0.1)
    assert (held.reason, held.p_speak) == ("随机扣发", pytest.approx(0.8))
    kept = decide(item, 1, now=NOW, spoken_today=0, daily_cap=6, holdout_rate=0.05, draw=0.1, text="话")
    assert kept.spoken and kept.p_speak == pytest.approx(0.95)
    with pytest.raises(ForesightError):
        decide(item, 2, now=NOW, spoken_today=0, daily_cap=6)
    with pytest.raises(ForesightError):
        decide(item, 1, now=NOW, spoken_today=0, daily_cap=6, holdout_rate=1.0)


def test_a_later_miss_on_the_same_expectation_does_not_undo_a_decline() -> None:
    facts = run([True] * 20, start=90.0)
    declined = AccountFact(ACCOUNT, NOW - timedelta(hours=5), "提醒组", False, 1.0, "这次不用", "e-today")
    # 当晚结算：同一条预期结成过期待决、说过所以提醒组落空——时刻在回话之后
    overdue = AccountFact(ACCOUNT, NOW - timedelta(hours=1), "预测", False, 1.0, "过期待决", "e-today")
    missed = AccountFact(ACCOUNT, NOW - timedelta(hours=1), "提醒组", False, 1.0, "落空", "e-today")
    kept = current_facts([*facts, declined, overdue, missed])
    assert declined in kept and missed not in kept


def test_a_revision_overrides_by_when_it_was_written_but_stays_at_its_own_opportunity() -> None:
    """后补、重结写在几天之后：谁覆盖谁按写下的时刻，遗忘与连错的先后按那次机会的时刻（改判不能把旧机会变成最新的证据）。"""

    from dataclasses import replace

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


def test_a_run_of_misses_is_counted_by_weight_and_ended_by_anything_he_did() -> None:
    """天天做、命中 88%：连漏三次不算断、四次算。几天后才写下的后补留在它那次机会的位置，不排到之后那次做了的后面；
    后补只算它打的那一折；做了的一笔哪怕强度是 0（偏得比历史上哪次都远）也打断连错。"""

    from dataclasses import replace

    history = [fact(60.0 - i, i % 8 != 7, n=i) for i in range(57)]
    three = [fact(3.0 - 0.5 * i, False, n=100 + i) for i in range(3)]
    assert standing([*history, *three]).break_at is None
    assert standing([*history, *three, fact(0.5, False, n=200)]).break_at is not None
    weak = replace(fact(0.5, False, kind="后补", strength=0.1, n=200), written_at=NOW - timedelta(hours=1))
    assert standing([*history, *three, weak]).break_at is None
    caught = replace(fact(3.5, False, kind="后补", strength=0.5, n=300), written_at=NOW - timedelta(hours=2))
    assert standing([*history, caught, fact(3.2, True, n=301), *three]).break_at is None
    odd_hit = fact(3.2, True, kind="晚了但做了", strength=0.0, n=3)
    misses = [fact(3.4, False, n=1), fact(3.3, False, n=2), odd_hit, fact(2.0, False, n=4), fact(1.0, False, n=5)]
    assert standing([*history, *misses]).break_at is None
