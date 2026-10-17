"""递送那一拍（裁定 48–51）：到了时窗的预期按这本账从事实折出的档决定、记进账本；一条预期只决定一次（「等时窗末」除外）；
没渠道记「不可达」也不问模型；窗与拍点取交集；一类一天一句；轻的到时窗末才说、重的时窗一开就说；名额按决定的日子数；
重轻按账判一次、到期才重判；档变了记一条；可见面能读到每本账。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta, timezone

import pytest

from habitus.foresight import Evidence, Tolerance
from habitus.foresight.judge import Judge
from habitus.foresight.ledger import AccountFact, AccountSeverity, Expectation, Omission, account_key
from habitus.runtime.foresight_delivery import DeliveryStage, JudgeMaterials
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from tests.unit.foresight.scripted_model import recording_client

CST = timezone(timedelta(hours=8))
DAY0 = datetime(2026, 10, 1, 9, 0, tzinfo=CST)
LOSS = Tolerance(floor=0.2, ceiling=0.9, unjudged=0.5)  # 线 = 1 / (1 + 代价比 × 效果)：代价比 2 → 0.33、0.5 → 0.67
EVIDENCE = Evidence(alpha=0.05, half_life_days=60.0)
NO_ROWS = JudgeMaterials(None, lambda uris: {})
GRAVE = {"verdicts": [{"item": 1, "cost_ratio": 2.0, "basis": "就医", "basis_number": 0}]}
LIKE = {"verdicts": [{"candidate": 1, "like": "yes", "like_basis": 0, "text": "该去医院了"}], "order": [1]}


def expectation(
    identity: str,
    formed: datetime,
    *,
    kind: str = "s-k0002",
    trigger: str = "rel",
    window: tuple[int, int] = (10, 40),
    deadline_hours: int = 2,
) -> Expectation:
    return Expectation(
        expectation_id=identity,
        kind_token=kind,
        day=formed.date(),
        formed_at=formed,
        source="事件",
        trigger=trigger,
        trigger_instance=f"behavior://{identity}.md",
        window=(formed + timedelta(minutes=window[0]), formed + timedelta(minutes=window[1])),
        deadline=formed + timedelta(hours=deadline_hours),
        basis=(),
        opportunities=3,
        hits=3,
        observed_from=formed,
    )


def earn(ledger: ForesightLedgerStore, *, kind: str = "s-k0002", trigger: str = "rel", hits: int = 8) -> str:
    """往这本账上写前几天各一笔预测轴的好。八次全中，轻的线也过得去。"""

    account = account_key(kind, "事件", trigger)
    for offset in range(hits, 0, -1):
        ledger.append_fact(
            AccountFact(account, DAY0 - timedelta(days=offset), "预测", True, 1.0, "准时", f"old-{offset}")
        )
    return account


def stage(tmp_path, **overrides):  # type: ignore[no-untyped-def]
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    values = dict(
        tolerance=LOSS,
        evidence=EVIDENCE,
        daily_cap=2,
        severity_review_days=60.0,
        holdout_rate=0.0,
        tick_seconds=60.0,
        reachable=True,
    )
    values.update(overrides)
    return DeliveryStage(ledger, **values), ledger


def test_due_expectations_are_decided_once_by_the_level_their_account_has_earned(tmp_path) -> None:
    run, ledger = stage(tmp_path)
    account = earn(ledger)
    ledger.record_expectation(expectation("today", DAY0))
    ledger.record_expectation(expectation("fresh", DAY0, kind="s-k0009", trigger="other"))  # 另一本账，从影子起
    ledger.record_expectation(expectation("done", DAY0, kind="s-k0010", trigger="done"))  # 到时窗时已经做了
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=5))).decided == 0  # 时窗还没开
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=50))).decided == 0  # 时窗过了、这一拍没盖到：等结算
    materials = JudgeMaterials(None, lambda uris: {}, happened=lambda item: item.expectation_id == "done")
    report = asyncio.run(run.run_once(DAY0 + timedelta(minutes=16), materials))
    # 轻的事：到时窗末还没做再说，那时才写话（话里的"今天""这会儿"说的是开口那一刻）
    assert report.decided == 3 and report.spoken == 0 and report.level_changes == 1
    assert report.by_reason == {"等时窗末": 1, "影子": 1, "已发生": 1} and report.by_level == {1: 1, 0: 2}
    stored = {item.expectation_id: item for item in ledger.deliveries_on(DAY0.date())}
    assert stored["today"].level == 1 and stored["today"].text is None and not stored["today"].spoken
    assert stored["fresh"].reason == "影子" and stored["done"].reason == "已发生"
    level = ledger.level_of(account)
    assert level is not None and level.level == 1 and level.at == DAY0 + timedelta(minutes=16)
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=20), materials)).decided == 0  # 一条只决定一次
    closing = asyncio.run(run.run_once(DAY0 + timedelta(minutes=41), materials))
    assert closing.by_reason == {"说了": 1} and closing.level_changes == 0  # 该来了还没来
    (said,) = [item for item in ledger.deliveries_on(DAY0.date()) if item.expectation_id == "today" and item.spoken]
    assert said.text
    assert run.standing_of(expectation("today", DAY0), DAY0 + timedelta(minutes=41)).level == 1


def test_without_a_channel_the_decision_is_unreachable_and_the_model_is_not_asked(tmp_path) -> None:
    client, provider = recording_client([GRAVE, GRAVE, LIKE, LIKE])
    run, ledger = stage(tmp_path, reachable=False, judge=Judge(client, retries=0, delay_seconds=0.0))
    earn(ledger)
    ledger.record_expectation(expectation("today", DAY0))
    report = asyncio.run(run.run_once(DAY0 + timedelta(minutes=15), NO_ROWS))
    assert report.by_reason == {"不可达": 1} and report.by_level == {1: 1} and report.spoken == 0
    assert report.judged == 0 and report.severities == 0 and provider.calls == 0  # 送不到：问了也说不出口
    (stored,) = ledger.deliveries_on(DAY0.date())
    assert not stored.spoken and stored.text is None


def test_a_window_shorter_than_a_tick_is_caught_by_the_tick_that_covers_it(tmp_path) -> None:
    run, ledger = stage(tmp_path, tick_seconds=900.0)
    ledger.record_expectation(expectation("point", DAY0, window=(7, 22)))  # 09:07–09:22，一槽宽
    assert asyncio.run(run.run_once(DAY0)).decided == 0  # 09:00：还没开
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=15))).by_reason == {"影子": 1}  # 09:15 这一拍盖到了
    ledger.record_expectation(expectation("gone", DAY0, kind="s-k0003", window=(1, 16)))  # 09:01–09:16
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=45))).decided == 0  # 09:45 的那一拍已经盖不到


def test_light_reminders_wait_for_the_window_end_and_the_cap_counts_by_decision_day(tmp_path) -> None:
    run, ledger = stage(tmp_path)
    for kind in ("s-k0002", "s-k0003", "s-k0004"):
        earn(ledger, kind=kind)
        ledger.record_expectation(expectation(f"t-{kind}", DAY0, kind=kind))
    opened = asyncio.run(run.run_once(DAY0 + timedelta(minutes=15)))
    assert opened.by_reason == {"等时窗末": 3} and opened.spoken == 0
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=30))).decided == 0  # 时窗里不再问
    done = JudgeMaterials(None, lambda uris: {}, happened=lambda item: item.kind_token == "s-k0004")
    closing = asyncio.run(run.run_once(DAY0 + timedelta(minutes=41), done))
    assert closing.by_reason == {"说了": 2, "已发生": 1} and closing.spoken == 2  # 名额 2
    earn(ledger, kind="s-k0005")
    ledger.record_expectation(expectation("t-late", DAY0 + timedelta(minutes=50), kind="s-k0005"))
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=65)))
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=91))).by_reason == {"名额已满": 1}
    # 过了最晚时限的不再说
    ledger.record_expectation(expectation("t-over", DAY0 + timedelta(minutes=100), kind="s-k0006", deadline_hours=1))
    earn(ledger, kind="s-k0006")
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=115)))
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=161))).decided == 0


def test_severity_is_judged_once_per_account_and_a_grave_matter_speaks_after_two_hits_at_once(tmp_path) -> None:
    """第一次到了该问时就判轻重缓急（裁定 58 第 1 条），不等结算过一次；判成最重一级后两次全中就过线、时窗一开就说。"""

    client, provider = recording_client([GRAVE, GRAVE, LIKE, LIKE])
    run, ledger = stage(tmp_path, judge=Judge(client, retries=0, delay_seconds=0.0))
    ledger.record_expectation(expectation("fresh", DAY0, kind="s-k0077", trigger="doctor"))  # 账上还什么都没有
    first = asyncio.run(run.run_once(DAY0 + timedelta(minutes=15), NO_ROWS))
    assert first.by_reason == {"影子": 1} and first.severities == 1 and provider.calls == 2  # 判了，但没证据：影子
    account = earn(ledger, kind="s-k0077", trigger="doctor", hits=2)  # 两次全中：最轻的线过不了，最重的过得了
    ledger.record_expectation(expectation("grave", DAY0 + timedelta(minutes=20), kind="s-k0077", trigger="doctor"))
    report = asyncio.run(run.run_once(DAY0 + timedelta(minutes=35), NO_ROWS))
    assert report.severities == 0 and report.judged == 1 and report.calls == 2  # 已判过、没到期：只问像不像
    assert report.by_reason == {"说了": 1} and report.by_level == {1: 1}  # 最重的时窗一开就说，不等时窗末
    (grave,) = [item for item in ledger.deliveries_on(DAY0.date()) if item.expectation_id == "grave"]
    assert grave.spoken and grave.text == "该去医院了"
    judged = ledger.severity_of(account)
    assert judged is not None and judged.cost_ratio == pytest.approx(2.0) and judged.expires_at == judged.judged_at + timedelta(days=60)
    # 同一类今天已经说过：不再问模型（问了也是「同类已说」）
    ledger.record_expectation(expectation("again", DAY0 + timedelta(hours=3), kind="s-k0077", trigger="doctor"))
    later = asyncio.run(run.run_once(DAY0 + timedelta(hours=3, minutes=15), NO_ROWS))
    assert later.by_reason == {"同类已说": 1} and later.severities == 0 and provider.calls == 4
    # 第二天同一本账：重轻已经判过、没到期，不再问重轻，只问像不像
    tomorrow = DAY0 + timedelta(days=1)
    ledger.record_expectation(expectation("next", tomorrow, kind="s-k0077", trigger="doctor"))
    next_day = asyncio.run(run.run_once(tomorrow + timedelta(minutes=15), NO_ROWS))
    assert next_day.severities == 0 and next_day.by_reason == {"说了": 1} and provider.calls == 6


def test_an_expired_severity_is_judged_again(tmp_path) -> None:
    client, provider = recording_client([GRAVE, GRAVE, LIKE, LIKE])
    run, ledger = stage(tmp_path, judge=Judge(client, retries=0, delay_seconds=0.0))
    account = earn(ledger, kind="s-k0077", trigger="doctor", hits=2)
    ledger.record_severity(AccountSeverity(account, DAY0 - timedelta(days=90), 0.5, "", DAY0 - timedelta(days=30)))
    ledger.record_expectation(expectation("grave", DAY0, kind="s-k0077", trigger="doctor"))
    report = asyncio.run(run.run_once(DAY0 + timedelta(minutes=15), NO_ROWS))
    assert report.severities == 1 and provider.calls == 4
    current = ledger.severity_of(account)
    assert current is not None and current.cost_ratio == pytest.approx(2.0) and current.current(DAY0 + timedelta(minutes=15))


def test_one_class_is_spoken_once_a_day_even_from_several_accounts(tmp_path) -> None:
    """同一刻同一个行为的两本账是一个单元：一条说、另一条「单元已选」引它（裁定 56 第 12 条）；同一天之后另一刻再来，
    同一类今天已经说过：「同类已说」。"""

    run, ledger = stage(tmp_path)
    for trigger in ("rel", "interval"):
        account = earn(ledger, trigger=trigger)
        ledger.record_severity(AccountSeverity(account, DAY0 - timedelta(days=1), 2.0, "", DAY0 + timedelta(days=59)))
    ledger.record_expectation(expectation("a", DAY0, trigger="rel"))
    ledger.record_expectation(expectation("b", DAY0, trigger="interval"))
    report = asyncio.run(run.run_once(DAY0 + timedelta(minutes=15)))  # 重的：时窗一开就说
    assert report.by_reason == {"说了": 1, "单元已选": 1} and report.spoken == 1
    (subsumed,) = [item for item in ledger.deliveries_on(DAY0.date()) if item.reason == "单元已选"]
    (said,) = [item for item in ledger.deliveries_on(DAY0.date()) if item.spoken]
    assert subsumed.evidence == (f"单元:{said.expectation_id}",) and not subsumed.spoken
    ledger.record_expectation(expectation("c", DAY0 + timedelta(minutes=30), trigger="rel"))
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=45))).by_reason == {"同类已说": 1}


def test_the_account_views_show_every_account_with_both_axes(tmp_path) -> None:
    run, ledger = stage(tmp_path)
    account = earn(ledger, hits=3)
    other = earn(ledger, kind="s-k0003", trigger="weekday=1", hits=8)
    views = {view.account: view for view in run.account_views(DAY0)}
    assert set(views) == {account, other}
    assert views[account].standing.level == 0 and views[other].standing.level == 1
    assert views[account].facts == 3 and len(views[other].recent) == 5 and "预测好 准时" in views[other].recent[-1]


def test_the_stage_checks_its_inputs(tmp_path) -> None:
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    common = dict(tolerance=LOSS, evidence=EVIDENCE, daily_cap=2, severity_review_days=60.0)
    with pytest.raises(TypeError):
        DeliveryStage(object(), **common)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        DeliveryStage(ledger, **{**common, "tolerance": object()})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DeliveryStage(ledger, **{**common, "daily_cap": 0})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DeliveryStage(ledger, **common, holdout_rate=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DeliveryStage(ledger, **common, tick_seconds=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DeliveryStage(ledger, **{**common, "severity_review_days": 0})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        DeliveryStage(ledger, **common, reachable=(1, 2))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        asyncio.run(DeliveryStage(ledger, **common).run_once(datetime(2026, 10, 1, 9, 0)))  # type: ignore[arg-type]


def test_the_omission_views_list_what_happened_without_a_covering_expectation(tmp_path) -> None:
    run, ledger = stage(tmp_path)
    for offset, kind, had in (
        (1, "s-k0003", False),
        (2, "s-k0003", True),
        (3, "s-k0003", False),
        (5, "s-k0004", False),
        (60, "s-k0004", False),
    ):
        day = DAY0 - timedelta(days=offset)
        ledger.record_omission(Omission(f"behavior://{kind}-{offset}.md", kind, day.date(), day, had))
    views = run.omission_views(DAY0)
    assert [(view.kind_token, view.count, view.with_expectation) for view in views] == [
        ("s-k0003", 3, 1),
        ("s-k0004", 1, 0),
    ]
    assert views[0].latest_uri == "behavior://s-k0003-1.md"


def test_at_the_window_end_the_account_is_read_again_and_the_model_is_asked_then(tmp_path) -> None:
    """轻的事：时窗开时不问模型、不写话；窗里设了「以后别提了」，到时窗末记影子、不说；没设的那本到时窗末才问模型、用它的话。"""

    from habitus.foresight.ledger import AccountSetting

    client, provider = recording_client([LIKE, LIKE])
    run, ledger = stage(tmp_path, judge=Judge(client, retries=0, delay_seconds=0.0))
    quiet = earn(ledger, kind="s-k0003")
    earn(ledger, kind="s-k0002")
    ledger.record_severity(AccountSeverity(quiet, DAY0, 0.5, "", DAY0 + timedelta(days=60)))
    ledger.record_severity(AccountSeverity(account_key("s-k0002", "事件", "rel"), DAY0, 0.5, "", DAY0 + timedelta(days=60)))
    ledger.record_expectation(expectation("silenced", DAY0, kind="s-k0003"))
    ledger.record_expectation(expectation("spoken", DAY0, kind="s-k0002"))
    opened = asyncio.run(run.run_once(DAY0 + timedelta(minutes=15), NO_ROWS))
    assert opened.by_reason == {"等时窗末": 2} and provider.calls == 0
    ledger.record_setting(AccountSetting(quiet, DAY0 + timedelta(minutes=20), "以后别提了"))
    closing = asyncio.run(run.run_once(DAY0 + timedelta(minutes=41), NO_ROWS))
    assert closing.by_reason == {"影子": 1, "说了": 1} and provider.calls == 2
    stored = {item.expectation_id: item for item in ledger.deliveries_on(DAY0.date()) if item.reason != "等时窗末"}
    assert stored["silenced"].level == 0 and not stored["silenced"].spoken
    assert stored["spoken"].text == "该去医院了"


def test_a_window_longer_than_a_week_is_still_closed_at_its_end(tmp_path) -> None:
    run, ledger = stage(tmp_path, tick_seconds=900.0)
    earn(ledger)
    ledger.record_expectation(expectation("long", DAY0, window=(10, 12 * 24 * 60), deadline_hours=13 * 24))
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=15))).by_reason == {"等时窗末": 1}
    end = DAY0 + timedelta(days=12, minutes=15)
    assert asyncio.run(run.run_once(end)).by_reason == {"说了": 1}


def test_before_speaking_an_event_reminder_checks_its_antecedent_again(tmp_path) -> None:
    """事件触发在前因没封口时建；到开口那一拍，前因已经归成了别的类：不说，记「前因不成立」（裁定 54）。"""

    from habitus.foresight import Seen
    from habitus.foresight.opportunities import OpportunityTable, RelationOpportunities

    run, ledger = stage(tmp_path)
    earn(ledger)
    started = DAY0 - timedelta(minutes=5)
    item = expectation("event", DAY0)
    item = Expectation(**{**item.__dict__, "trigger_instance": started.astimezone(UTC).isoformat()})
    ledger.record_expectation(item)
    relation = RelationOpportunities(
        key="rel",
        antecedent="c-flight",
        consequent="c-hotel",
        antecedent_label="订机票",
        consequent_label="订酒店",
        antecedent_classes=("s-k0001",),
        consequent_classes=("s-k0002",),
        segment="chain",
        status="established",
        upward=True,
        items=(),
    )
    table = OpportunityTable("session", DAY0.date(), DAY0.date(), 15, 3, (DAY0.date(),), (relation,), ())

    def rows(kind: str):  # type: ignore[no-untyped-def]
        seen = Seen(started, started, kind, "behavior://x.md", "订机票", "")
        return lambda uris: {uri: seen for uri in uris}

    grave = AccountSeverity(item.account, DAY0, 2.0, "", DAY0 + timedelta(days=60))
    ledger.record_severity(grave)
    changed = asyncio.run(run.run_once(DAY0 + timedelta(minutes=15), JudgeMaterials(table, rows("s-k0009"))))
    assert changed.by_reason == {"前因不成立": 1} and changed.spoken == 0
    run2, ledger2 = stage(tmp_path / "b")
    earn(ledger2)
    ledger2.record_expectation(item)
    ledger2.record_severity(grave)
    held = asyncio.run(run2.run_once(DAY0 + timedelta(minutes=15), JudgeMaterials(table, rows("s-k0001"))))
    assert held.by_reason == {"说了": 1}


def test_when_last_misss_cause_is_back_it_holds_back_and_the_next_answer_can_retire_the_cause(tmp_path) -> None:
    """裁定 55「再预测」：上次没做是因为惯常的前因没来；今天这条预期上同一条前因又不在 → 不说，记「上次的原因今天又在」并引出处。
    因此没说的那一次他其实自己做了（准时）→ 这条原因错了，之后不再用，照常说。"""

    from dataclasses import replace

    from habitus.foresight import Seen
    from habitus.foresight.ledger import Cause
    from habitus.foresight.opportunities import OpportunityTable

    run, ledger = stage(tmp_path)
    account = earn(ledger, hits=14)
    cue = "session|s-k0077|s-k0002|rest_of_day"
    ledger.append_fact(
        AccountFact(
            account, DAY0 - timedelta(hours=20), "预测", False, 1.0, "过期待决", "last-miss",
            written_at=DAY0 - timedelta(hours=10),
            cause=Cause("线索没了", classes=("s-k0077",), relation=cue, segment="rest_of_day"),
        )
    )
    table = OpportunityTable("session", DAY0.date(), DAY0.date(), 15, 3, (DAY0.date(),), (), ())
    materials = JudgeMaterials(table, lambda uris: {})  # 今天到此刻没有那条前因
    ledger.record_expectation(expectation("today", DAY0))
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=16), materials))
    report = asyncio.run(run.run_once(DAY0 + timedelta(minutes=41), materials))
    assert report.by_reason == {"上次的原因今天又在": 1}
    (held,) = [item for item in ledger.deliveries_on(DAY0.date()) if item.reason == "上次的原因今天又在"]
    assert held.evidence == ("原因:last-miss",) and not held.spoken
    # 前因今天早上来过（当天剩下的时间里）：上次的原因不成立，照常说
    tomorrow = DAY0 + timedelta(days=1)
    early = tomorrow - timedelta(hours=2)
    present = replace(materials, seen=(Seen(early, early + timedelta(minutes=10), "s-k0077", "behavior://c.md"),))
    ledger.record_expectation(expectation("present", tomorrow))
    asyncio.run(run.run_once(tomorrow + timedelta(minutes=16), present))
    assert asyncio.run(run.run_once(tomorrow + timedelta(minutes=41), present)).by_reason == {"说了": 1}
    # 因此没说的那一次他其实自己做了：原因错了，之后同样的情形也照常说
    ledger.append_fact(AccountFact(account, DAY0, "预测", True, 1.0, "准时", "today"))
    later = DAY0 + timedelta(days=2)
    ledger.record_expectation(expectation("again", later))
    asyncio.run(run.run_once(later + timedelta(minutes=16), materials))
    assert asyncio.run(run.run_once(later + timedelta(minutes=41), materials)).by_reason == {"说了": 1}


def test_the_holdout_rate_is_smaller_for_matters_where_a_miss_costs_more_than_a_false_alarm(tmp_path) -> None:
    """裁定 58 第 7 条：都随机扣发；漏一次比错提一次代价大的（代价比 > 1）扣得少；没判过、过期的按较大的那个。"""

    run, _ledger = stage(tmp_path, holdout_rate=0.2, holdout_rate_heavy=0.04)
    account = account_key("s-k0002", "事件", "rel")

    def judged(ratio: float) -> AccountSeverity:
        return AccountSeverity(account, DAY0 - timedelta(days=1), ratio, "", DAY0 + timedelta(days=30))

    assert run._holdout(None, DAY0) == pytest.approx(0.2)
    assert run._holdout(judged(0.5), DAY0) == pytest.approx(0.2)
    assert run._holdout(judged(1.0), DAY0) == pytest.approx(0.2)  # 1 是好处等于代价的临界点，不算
    assert run._holdout(judged(3.0), DAY0) == pytest.approx(0.04)
    expired = AccountSeverity(account, DAY0 - timedelta(days=90), 3.0, "", DAY0 - timedelta(days=1))
    assert run._holdout(expired, DAY0) == pytest.approx(0.2)  # 过期的按没判过
    with pytest.raises(ValueError):
        stage(tmp_path / "b", holdout_rate=0.05, holdout_rate_heavy=0.1)  # 代价大的不比轻的扣得多


def test_a_unit_decides_once_for_every_input_of_the_same_moment_including_those_not_yet_due(tmp_path) -> None:
    """裁定 56 第 12 条：同一个行为这一刻的输入拢成一个单元——到了时窗的那条决定，说中的范围和它连成一片、还没到时窗的那条
    也一并记「单元已选」；它自己的时窗开了也不再决定第二次。"""

    run, ledger = stage(tmp_path)
    earn(ledger, trigger="rel")
    earn(ledger, trigger="clock")
    ledger.record_expectation(expectation("now", DAY0, trigger="rel"))
    ledger.record_expectation(expectation("later", DAY0 + timedelta(minutes=30), trigger="clock"))  # 时窗 09:40 才开
    first = asyncio.run(run.run_once(DAY0 + timedelta(minutes=16)))
    assert first.by_reason == {"等时窗末": 1, "单元已选": 1}
    later = next(item for item in ledger.deliveries_on(DAY0.date()) if item.expectation_id == "later")
    assert later.reason == "单元已选" and later.evidence == ("单元:now",)
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=41))).by_reason == {"说了": 1}  # 只有依据那条在时窗末说


def test_a_suppressor_present_holds_the_unit_back_even_when_the_account_alone_would_speak(tmp_path) -> None:
    """取消出差在场（压低确认行李）的那几刻，历史上确认行李都没来：这一刻同样的组合在场，单元读数压在线下、不说；
    不在场时退到这本账自己的读数，照常说。"""

    from dataclasses import replace

    from habitus.foresight.ledger import Antecedent

    run, ledger = stage(tmp_path)
    account = earn(ledger, trigger="rel", hits=30)
    cancel = Antecedent("session|cancel|s-k0002|next_day", True, False, "behavior://cancel.md")
    for back in range(1, 6):
        past = replace(expectation(f"past-{back}", DAY0 - timedelta(days=back), trigger="rel"), antecedents=(cancel,))
        ledger.record_expectation(past)
        ledger.append_fact(AccountFact(account, past.formed_at, "预测", False, 1.0, "过期待决", past.expectation_id))
    ledger.record_expectation(replace(expectation("today", DAY0, trigger="rel"), antecedents=(cancel,)))
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=16)))
    (decided,) = [item for item in ledger.deliveries_on(DAY0.date()) if item.expectation_id == "today"]
    assert decided.reason == "影子"
    tomorrow = DAY0 + timedelta(days=1)
    ledger.record_expectation(expectation("plain", tomorrow, trigger="rel"))
    asyncio.run(run.run_once(tomorrow + timedelta(minutes=16)))
    assert asyncio.run(run.run_once(tomorrow + timedelta(minutes=41))).by_reason == {"说了": 1}
