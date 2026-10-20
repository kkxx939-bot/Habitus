"""递送那一拍（裁定 48–51、58、61）：到了时窗的预期按这条边从单元历史读出的档决定、记进账本；一条预期只决定一次（「等时窗末」
除外）；没渠道记「不可达」也不问模型；窗与拍点取交集；一类一天一句；一律到时窗末还没做才说（这一版不分急缓、不扣发）；名额按
决定的日子数；档变了记一条；可见面能读到每本账；设置与提醒反馈按行为。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta, timezone

import pytest

from habitus.foresight import Evidence, Tolerance
from habitus.foresight.edges import PastMoment
from habitus.foresight.history import Moment, UnitHistory
from habitus.foresight.judge import Judge
from habitus.foresight.ledger import AccountFact, Expectation, Omission, account_key
from habitus.runtime.foresight_delivery import DeliveryStage, JudgeMaterials
from habitus.runtime.foresight_history_store import HistoryStore
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from tests.unit.foresight.scripted_model import recording_client

CST = timezone(timedelta(hours=8))
DAY0 = datetime(2026, 10, 1, 9, 0, tzinfo=CST)
LOSS = Tolerance(line=2 / 3)  # 所有账同一条线（裁定 58 第 1 条）
EVIDENCE = Evidence(alpha=0.05, half_life=8.0)
NO_ROWS = JudgeMaterials(None, lambda uris: {})
LIKE = {"verdicts": [{"candidate": 1, "like": "yes", "like_basis": 0, "text": "该去医院了"}], "order": [1]}


def expectation(
    identity: str,
    formed: datetime,
    *,
    kind: str = "s-k0002",
    trigger: str = "rel",
    window: tuple[int, int] = (10, 40),
    deadline_hours: int = 2,
    source: str = "日子",
) -> Expectation:
    """默认日子触发（离上次够久）：递送这几条测的是时机与名额，不是依据（事件触发要关系已成立才能开口，见 ``test_units``）。"""

    return Expectation(
        expectation_id=identity,
        kind_token=kind,
        day=formed.date(),
        formed_at=formed,
        source=source,
        trigger=trigger,
        trigger_instance=f"behavior://{identity}.md",
        window=(formed + timedelta(minutes=window[0]), formed + timedelta(minutes=window[1])),
        deadline=formed + timedelta(hours=deadline_hours),
        basis=(),
        opportunities=3,
        hits=3,
        observed_from=formed,
    )


class Seeded:
    """账本旁边那一份单元历史（夜里重标的快照）：边的位置从它读（裁定 61）。``earn`` 给一条边写前几天各一刻的来了
    （账上同时留流水），写进下一晚的快照；每次写完下一次读到的是更晚的一晚。"""

    def __init__(self, root, ledger: ForesightLedgerStore) -> None:  # type: ignore[no-untyped-def]
        self.store = HistoryStore(root)
        self.ledger = ledger
        self.moments: dict[str, list[PastMoment]] = {}
        self.nights = 0

    def earn(self, *, kind: str = "s-k0002", trigger: str = "rel", hits: int = 8, source: str = "日子") -> str:
        """这条边前几天各一刻的来了。八次全中，线也过得去。"""

        account = account_key(kind, source, trigger)
        for offset in range(hits, 0, -1):
            self.ledger.append_fact(
                AccountFact(account, DAY0 - timedelta(days=offset), True, 1.0, "准时", f"old-{offset}")
            )
        self.put(
            kind,
            [PastMoment(DAY0 - timedelta(days=offset), frozenset({account}), True) for offset in range(hits, 0, -1)],
        )
        return account

    def put(self, kind: str, moments) -> None:  # type: ignore[no-untyped-def]
        self.moments.setdefault(kind, []).extend(moments)
        self.flush()

    def flush(self) -> None:
        self.nights += 1
        night = DAY0.date() - timedelta(days=30) + timedelta(days=self.nights)
        self.store.write(
            UnitHistory(
                "session",
                night,
                {
                    kind: tuple(Moment(item, (f"{kind}-{index}",)) for index, item in enumerate(found))
                    for kind, found in self.moments.items()
                },
            )
        )


_SEEDED: dict[str, Seeded] = {}


def stage(tmp_path, **overrides):  # type: ignore[no-untyped-def]
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    seeded = Seeded(tmp_path / "ledger", ledger)
    _SEEDED[str(ledger.root)] = seeded
    values = dict(
        tolerance=LOSS,
        evidence=EVIDENCE,
        daily_cap=2,
        tick_seconds=60.0,
        reachable=True,
        history=seeded.store,
    )
    values.update(overrides)
    return DeliveryStage(ledger, **values), ledger


def earn(ledger: ForesightLedgerStore, **kwargs) -> str:  # type: ignore[no-untyped-def]
    return seeded_of(ledger).earn(**kwargs)


def seeded_of(ledger: ForesightLedgerStore) -> Seeded:
    return _SEEDED[str(ledger.root)]


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
    # 档变了记两条：这本账自己的级、这个行为单元的档（两者分开记，2026-10-17 四方评审）
    assert report.decided == 3 and report.spoken == 0 and report.level_changes == 2
    assert report.by_reason == {"等时窗末": 1, "影子": 1, "已发生": 1} and report.by_level == {1: 1, 0: 2}
    stored = {item.expectation_id: item for item in ledger.deliveries_on(DAY0.date())}
    assert stored["today"].level == 1 and stored["today"].text is None and not stored["today"].spoken
    assert stored["fresh"].reason == "影子" and stored["done"].reason == "已发生"
    level = ledger.level_of(account)
    assert level is not None and level.level == 1 and level.at == DAY0 + timedelta(minutes=16)
    unit = ledger.unit_level_of("s-k0002")
    assert unit is not None and unit.level == 1
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=20), materials)).decided == 0  # 一条只决定一次
    closing = asyncio.run(run.run_once(DAY0 + timedelta(minutes=41), materials))
    assert closing.by_reason == {"说了": 1} and closing.level_changes == 0  # 该来了还没来
    (said,) = [item for item in ledger.deliveries_on(DAY0.date()) if item.expectation_id == "today" and item.spoken]
    assert said.text
    assert run.standing_of(expectation("today", DAY0), DAY0 + timedelta(minutes=41)).level == 1


def test_without_a_channel_the_decision_is_unreachable_and_the_model_is_not_asked(tmp_path) -> None:
    client, provider = recording_client([LIKE, LIKE])
    run, ledger = stage(tmp_path, reachable=False, judge=Judge(client, retries=0, delay_seconds=0.0))
    earn(ledger)
    ledger.record_expectation(expectation("today", DAY0))
    report = asyncio.run(run.run_once(DAY0 + timedelta(minutes=15), NO_ROWS))
    assert report.by_reason == {"不可达": 1} and report.by_level == {1: 1} and report.spoken == 0
    assert report.judged == 0 and provider.calls == 0  # 送不到：问了也说不出口
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


def test_one_class_is_spoken_once_a_day_even_from_several_accounts(tmp_path) -> None:
    """同一刻同一个行为的两本账是一个单元：一条说、另一条「单元已选」引它（裁定 56 第 12 条）；同一天之后另一刻再来，
    同一类今天已经说过：「同类已说」。"""

    run, ledger = stage(tmp_path)
    for trigger in ("rel", "interval"):
        earn(ledger, trigger=trigger)
    ledger.record_expectation(expectation("a", DAY0, trigger="rel"))
    ledger.record_expectation(expectation("b", DAY0, trigger="interval"))
    opened = asyncio.run(run.run_once(DAY0 + timedelta(minutes=15)))
    assert opened.by_reason == {"等时窗末": 1, "单元已选": 1}
    report = asyncio.run(run.run_once(DAY0 + timedelta(minutes=41)))  # 时窗末还没做：说
    assert report.by_reason == {"说了": 1} and report.spoken == 1
    (subsumed,) = [item for item in ledger.deliveries_on(DAY0.date()) if item.reason == "单元已选"]
    (said,) = [item for item in ledger.deliveries_on(DAY0.date()) if item.spoken]
    assert subsumed.evidence == (f"单元:{said.expectation_id}",) and not subsumed.spoken
    # 同一刻后来才形成的那条：并进已经决定过的单元，不另起一个（2026-10-17 四方评审）
    ledger.record_expectation(expectation("c", DAY0 + timedelta(minutes=30), trigger="rel"))
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=45))).by_reason == {"单元已选": 1}
    # 同一天另一刻（说中的范围不相连）再来：同一类今天已经说过
    ledger.record_expectation(expectation("d", DAY0 + timedelta(hours=3), trigger="rel"))
    asyncio.run(run.run_once(DAY0 + timedelta(hours=3, minutes=15)))
    assert asyncio.run(run.run_once(DAY0 + timedelta(hours=3, minutes=41))).by_reason == {"同类已说": 1}


def test_the_account_views_show_every_account_with_the_edge_read_from_the_unit_history(tmp_path) -> None:
    run, ledger = stage(tmp_path)
    account = earn(ledger, hits=3)
    other = earn(ledger, kind="s-k0003", trigger="weekday=1", hits=8)
    views = {view.account: view for view in run.account_views(DAY0)}
    assert set(views) == {account, other}
    assert views[account].standing.level == 0 and views[other].standing.level == 1
    assert views[account].facts == 3 and len(views[other].recent) == 5 and "好 准时" in views[other].recent[-1]


def test_the_stage_checks_its_inputs(tmp_path) -> None:
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    common = dict(tolerance=LOSS, evidence=EVIDENCE, daily_cap=2)
    with pytest.raises(TypeError):
        DeliveryStage(object(), **common)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        DeliveryStage(ledger, **{**common, "tolerance": object()})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DeliveryStage(ledger, **{**common, "daily_cap": 0})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DeliveryStage(ledger, **common, tick_seconds=0)  # type: ignore[arg-type]
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
    """轻的事：时窗开时不问模型、不写话；窗里给那个行为设了「以后别提了」，到时窗末记影子、不说；没设的那个到时窗末才问模型、
    用它的话（设置按行为，裁定 61）。"""

    from habitus.foresight.ledger import UnitSetting

    client, provider = recording_client([LIKE, LIKE])
    run, ledger = stage(tmp_path, judge=Judge(client, retries=0, delay_seconds=0.0))
    earn(ledger, kind="s-k0003")
    earn(ledger, kind="s-k0002")
    ledger.record_expectation(expectation("silenced", DAY0, kind="s-k0003"))
    ledger.record_expectation(expectation("spoken", DAY0, kind="s-k0002"))
    opened = asyncio.run(run.run_once(DAY0 + timedelta(minutes=15), NO_ROWS))
    assert opened.by_reason == {"等时窗末": 2} and provider.calls == 0
    ledger.record_setting(UnitSetting("s-k0003", DAY0 + timedelta(minutes=20), "以后别提了"))
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
    earn(ledger, source="事件")
    started = DAY0 - timedelta(minutes=5)
    item = expectation("event", DAY0, source="事件")
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
    table = OpportunityTable("session", DAY0.date(), DAY0.date(), 15, 3, (relation,), ())

    def rows(kind: str):  # type: ignore[no-untyped-def]
        seen = Seen(started, started, kind, "behavior://x.md", "订机票", "")
        return lambda uris: {uri: seen for uri in uris}

    changed = JudgeMaterials(table, rows("s-k0009"))
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=15), changed))
    closing = asyncio.run(run.run_once(DAY0 + timedelta(minutes=41), changed))
    assert closing.by_reason == {"前因不成立": 1} and closing.spoken == 0
    run2, ledger2 = stage(tmp_path / "b")
    earn(ledger2, source="事件")
    ledger2.record_expectation(item)
    holds = JudgeMaterials(table, rows("s-k0001"))
    asyncio.run(run2.run_once(DAY0 + timedelta(minutes=15), holds))
    assert asyncio.run(run2.run_once(DAY0 + timedelta(minutes=41), holds)).by_reason == {"说了": 1}


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
    不在场时照常说。历史是夜里重标过、落在单元历史里的那一份（裁定 55 新版第 2 条），白天不扫账本。"""

    from dataclasses import replace

    from habitus.foresight.edges import DOWN
    from habitus.foresight.ledger import Antecedent
    from habitus.foresight.opportunities import OpportunityTable

    run, ledger = stage(tmp_path)
    account = account_key("s-k0002", "日子", "rel")
    cancel = Antecedent("session|cancel|s-k0002|next_day", True, False, "behavior://cancel.md")
    suppressed = frozenset({account, f"{DOWN}session|cancel|s-k0002|next_day"})
    plain = frozenset({account})
    # 约了游泳（压低）的那几刻在前、平常的二十刻在后：这条边自己读高，压低的那一组读低
    past = [PastMoment(DAY0 - timedelta(days=back), suppressed, False) for back in range(21, 26)]
    past += [PastMoment(DAY0 - timedelta(days=back), plain, True) for back in range(1, 21)]
    seeded_of(ledger).put("s-k0002", past)
    table = OpportunityTable("session", DAY0.date(), DAY0.date(), 15, 3, (), ())
    materials = JudgeMaterials(table, lambda uris: {})
    ledger.record_expectation(replace(expectation("today", DAY0, trigger="rel"), antecedents=(cancel,)))
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=16), materials))
    (decided,) = [item for item in ledger.deliveries_on(DAY0.date()) if item.expectation_id == "today"]
    assert decided.reason == "影子"
    tomorrow = DAY0 + timedelta(days=1)
    ledger.record_expectation(expectation("plain", tomorrow, trigger="rel"))
    asyncio.run(run.run_once(tomorrow + timedelta(minutes=16), materials))
    assert asyncio.run(run.run_once(tomorrow + timedelta(minutes=41), materials)).by_reason == {"说了": 1}


def test_at_the_window_end_the_unit_is_read_again_not_the_basis_account_alone(tmp_path) -> None:
    """几个前因一起才稳的轻事：每本账单独都过不了线，这一组在场时历史上都来了——时窗开时单元说该说（等时窗末），到时窗末仍按
    单元读、照说（2026-10-17 四方评审：原来时窗末改按依据那本账自己的档判，记成影子、永远说不出口）；单元的档按行为另记。"""

    from habitus.foresight.opportunities import OpportunityTable

    run, ledger = stage(tmp_path)
    weak = []
    moments: list[PastMoment] = []
    for trigger in ("rel", "clock"):
        account = account_key("s-k0002", "日子", trigger)
        weak.append(account)
        # 单独在场的二十刻一半一半（最近）：这条边自己下界不过线
        moments += [
            PastMoment(DAY0 - timedelta(days=offset), frozenset({account}), offset % 2 == 0) for offset in range(1, 21)
        ]
    together = frozenset(weak)
    moments += [PastMoment(DAY0 - timedelta(days=d), together, True) for d in range(21, 36)]  # 一起在场的那几刻全来了
    seeded_of(ledger).put("s-k0002", moments)
    table = OpportunityTable("session", DAY0.date(), DAY0.date(), 15, 3, (), ())
    materials = JudgeMaterials(table, lambda uris: {})
    ledger.record_expectation(expectation("a", DAY0, trigger="rel"))
    ledger.record_expectation(expectation("b", DAY0, trigger="clock"))
    opened = asyncio.run(run.run_once(DAY0 + timedelta(minutes=16), materials))
    assert opened.by_reason == {"等时窗末": 1, "单元已选": 1}
    assert all(
        run.standing_of(expectation(n, DAY0, trigger=t), DAY0).level == 0 for n, t in (("a", "rel"), ("b", "clock"))
    )
    closing = asyncio.run(run.run_once(DAY0 + timedelta(minutes=41), materials))
    assert closing.by_reason == {"说了": 1}
    unit = ledger.unit_level_of("s-k0002")
    assert unit is not None and unit.level == 1
    assert all(ledger.level_of(account) is None for account in weak)  # 账自己的级没被单元的档改写


# --- 久别回来（26 号清单 3.3）----------------------------------------------------------------------


def _absence_table():  # type: ignore[no-untyped-def]
    from habitus.foresight.opportunities import DayWatch, OpportunityTable

    watch = tuple(DayWatch(DAY0.date() - timedelta(days=offset), 1.0) for offset in range(30, 0, -1))
    return OpportunityTable("session", DAY0.date() - timedelta(days=1), None, 15, 3, (), (), watch=watch)


def _middle_band(ledger: ForesightLedgerStore) -> str:
    """4/4 的账（下界不到线、上界过线）、上一次在提醒档：靠滞回留在提醒档。四次隔一天一次，之后隔了 8 天才又来。"""

    from habitus.foresight.ledger import AccountLevel, UnitLevel

    account = account_key("s-k0002", "日子", "rel")
    for offset in (14, 12, 10, 8):
        ledger.append_fact(AccountFact(account, DAY0 - timedelta(days=offset), True, 1.0, "准时", f"old-{offset}"))
    seeded_of(ledger).put(
        "s-k0002", [PastMoment(DAY0 - timedelta(days=offset), frozenset({account}), True) for offset in (14, 12, 10, 8)]
    )
    ledger.record_level(AccountLevel(account, DAY0 - timedelta(days=15), 1))
    ledger.record_unit_level(UnitLevel("s-k0002", DAY0 - timedelta(days=15), 1))
    return account


def test_a_return_after_absence_stays_quiet_once_and_the_next_one_is_back_at_the_reminder_level(tmp_path) -> None:
    run, ledger = stage(tmp_path)
    account = _middle_band(ledger)
    materials = JudgeMaterials(_absence_table(), lambda uris: {})
    ledger.record_expectation(expectation("back", DAY0))
    opening = asyncio.run(run.run_once(DAY0 + timedelta(minutes=15), materials))
    assert opening.by_reason == {"久别回来": 1} and opening.by_level == {1: 1} and opening.spoken == 0
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=41), materials)).decided == 0  # 终局：时窗末不再说
    level = ledger.level_of(account)
    unit = ledger.unit_level_of("s-k0002")
    assert level is not None and level.level == 1 and unit is not None and unit.level == 1  # 档没动
    # 这次照常结算；下一次（隔一天，和往常一样）照常在提醒档
    ledger.append_fact(AccountFact(account, DAY0, True, 1.0, "准时", "back"))
    following = DAY0 + timedelta(days=2)
    ledger.record_expectation(expectation("next", following))
    report = asyncio.run(run.run_once(following + timedelta(minutes=15), materials))
    assert report.by_reason == {"等时窗末": 1} and report.by_level == {1: 1}


def test_always_remind_me_wins_over_a_return_after_absence(tmp_path) -> None:
    from habitus.foresight.ledger import UnitSetting

    run, ledger = stage(tmp_path)
    _middle_band(ledger)
    ledger.record_setting(UnitSetting("s-k0002", DAY0 - timedelta(days=1), "以后都提醒我"))
    ledger.record_expectation(expectation("back", DAY0))
    report = asyncio.run(run.run_once(DAY0 + timedelta(minutes=15), JudgeMaterials(_absence_table(), lambda uris: {})))
    assert report.by_reason == {"等时窗末": 1}


def test_a_return_seen_only_at_the_window_end_is_not_put_to_the_model(tmp_path) -> None:
    """时窗开时还没有机会表（判不了久别）、到时窗末有了：时窗末记「久别回来」，也不拿它去问模型。"""

    client, provider = recording_client([LIKE, LIKE])
    run, ledger = stage(tmp_path, judge=Judge(client, retries=0, delay_seconds=0.0))
    _middle_band(ledger)
    ledger.record_expectation(expectation("back", DAY0))
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=15), NO_ROWS)).by_reason == {"等时窗末": 1}
    closing = asyncio.run(run.run_once(DAY0 + timedelta(minutes=41), JudgeMaterials(_absence_table(), lambda uris: {})))
    assert closing.by_reason == {"久别回来": 1} and closing.spoken == 0
    assert provider.calls == 0
