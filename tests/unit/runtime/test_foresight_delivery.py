"""递送那一拍（裁定 48–51、58、61）：到了时窗的预期按这条边从单元历史读出的档决定、记进账本；一条预期只决定一次（「等时窗末」
除外）；没有宿主活着记「不可达」也不问模型；窗与拍点取交集；一类一天一句；一律到时窗末还没做才说（这一版不分急缓、不扣发）：决定说的
记「待取」、话交给出口（方案 32），「说了」由出口追加；「待取」之后做了记「已发生」、过时限记「宿主没取」；名额按决定的日子数、「待取」
也占；档变了记一条；可见面能读到每本账；设置与提醒反馈按行为。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta, timezone

import pytest

from habitus.foresight import Evidence, Tolerance
from habitus.foresight.edges import PastMoment
from habitus.foresight.history import Moment, UnitHistory
from habitus.foresight.judge import Judge
from habitus.foresight.ledger import AccountFact, Delivery, Expectation, Omission, account_key
from habitus.runtime.foresight_delivery import DeliveryStage, JudgeMaterials
from habitus.runtime.foresight_history_store import HistoryStore
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_outlet import OpenRequest
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


class Outlet:
    """出口的替身：有没有宿主活着由测试定；记下每拍交过来的可取清单。"""

    def __init__(self, alive: bool = True) -> None:
        self.is_alive = alive
        self.published: list[tuple[OpenRequest, ...]] = []

    def alive(self, now: datetime) -> bool:
        return self.is_alive

    def publish(self, rows, *, now: datetime) -> int:  # type: ignore[no-untyped-def]
        self.published.append(tuple(rows))
        return len(rows)


def stage(tmp_path, **overrides):  # type: ignore[no-untyped-def]
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    seeded = Seeded(tmp_path / "ledger", ledger)
    _SEEDED[str(ledger.root)] = seeded
    values = dict(
        tolerance=LOSS,
        evidence=EVIDENCE,
        daily_cap=2,
        tick_seconds=60.0,
        outlet=Outlet(),
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
    assert closing.by_reason == {"待取": 1} and closing.level_changes == 0  # 该来了还没来：话写好、等宿主来取
    (said,) = [
        item for item in ledger.deliveries_on(DAY0.date()) if item.expectation_id == "today" and item.reason == "待取"
    ]
    assert said.text
    assert run.standing_of(expectation("today", DAY0), DAY0 + timedelta(minutes=41)).level == 1


def test_without_a_live_host_the_window_end_is_unreachable_and_the_model_is_not_asked(tmp_path) -> None:
    """活性到时窗末才看（方案 32 第 3.3 条）：时窗开时照常「等时窗末」，末了还没有宿主活着才记「不可达」、不问模型。"""

    client, provider = recording_client([LIKE, LIKE])
    outlet = Outlet(alive=False)
    run, ledger = stage(tmp_path, outlet=outlet, judge=Judge(client, retries=0, delay_seconds=0.0))
    earn(ledger)
    ledger.record_expectation(expectation("today", DAY0))
    opened = asyncio.run(run.run_once(DAY0 + timedelta(minutes=15), NO_ROWS))
    assert opened.by_reason == {"等时窗末": 1} and opened.by_level == {1: 1}
    report = asyncio.run(run.run_once(DAY0 + timedelta(minutes=41), NO_ROWS))
    assert report.by_reason == {"不可达": 1} and report.by_level == {1: 1} and report.spoken == 0
    assert report.judged == 0 and provider.calls == 0  # 没人来取：问了也说不出口
    stored = {item.reason: item for item in ledger.deliveries_on(DAY0.date())}
    assert not stored["不可达"].spoken and stored["不可达"].text is None
    assert outlet.published[-1] == ()  # 可取清单是空的
    # 没接出口（None）与没有宿主活着一样
    run2, ledger2 = stage(tmp_path / "b", outlet=None)
    earn(ledger2)
    ledger2.record_expectation(expectation("today", DAY0))
    asyncio.run(run2.run_once(DAY0 + timedelta(minutes=15), NO_ROWS))
    assert asyncio.run(run2.run_once(DAY0 + timedelta(minutes=41), NO_ROWS)).by_reason == {"不可达": 1}


def test_an_awaiting_request_is_handed_to_the_outlet_and_closed_by_doing_it_spoken_or_the_deadline(tmp_path) -> None:
    """「待取」之后每拍只做两件事：做了追加「已发生」、过了最晚时限追加「宿主没取」；出口写了「说了」的不再动；
    可取清单每拍重写、只含还等着的。"""

    outlet = Outlet()
    run, ledger = stage(tmp_path, outlet=outlet)
    for kind in ("s-k0002", "s-k0003", "s-k0004"):
        earn(ledger, kind=kind)
        ledger.record_expectation(expectation(f"t-{kind}", DAY0, kind=kind))
    run.daily_cap = 3
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=15)))
    closing = asyncio.run(run.run_once(DAY0 + timedelta(minutes=41)))
    assert closing.by_reason == {"待取": 3} and closing.open_requests == 3
    rows = outlet.published[-1]
    assert sorted(row.request_id for row in rows) == ["t-s-k0002", "t-s-k0003", "t-s-k0004"]
    assert all(row.text and row.level == 1 and row.deadline == DAY0 + timedelta(hours=2) for row in rows)
    # 出口认出了提问：追加「说了」（出口写，不是这一拍）
    ledger.record_delivery(
        DAY0.date(),
        Delivery("t-s-k0002", DAY0 + timedelta(minutes=50), 1, True, "说了", 1.0, "题目", via="codex", turn="c1"),
    )
    done = JudgeMaterials(None, lambda uris: {}, happened=lambda item: item.kind_token == "s-k0003")
    report = asyncio.run(run.run_once(DAY0 + timedelta(minutes=55), done))
    assert report.by_reason == {"已发生": 1} and report.open_requests == 1  # 做了的、说了的都不在清单上
    assert [row.request_id for row in outlet.published[-1]] == ["t-s-k0004"]
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=70), done)).decided == 0  # 没变就不再写
    over = asyncio.run(run.run_once(DAY0 + timedelta(hours=2, minutes=1), done))
    assert over.by_reason == {"宿主没取": 1} and over.open_requests == 0
    stored = {(item.expectation_id, item.reason) for item in ledger.deliveries_on(DAY0.date())}
    assert ("t-s-k0004", "宿主没取") in stored and ("t-s-k0002", "宿主没取") not in stored
    assert asyncio.run(run.run_once(DAY0 + timedelta(hours=2, minutes=2), done)).decided == 0
    # 名额：「待取」与它之后的「说了」同一条预期只算一次——今天三条都已占满，再来一条记「名额已满」
    earn(ledger, kind="s-k0005")
    ledger.record_expectation(expectation("t-late", DAY0 + timedelta(minutes=50), kind="s-k0005"))
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=65)))
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=91))).by_reason == {"名额已满": 1}


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
    assert closing.by_reason == {"待取": 2, "已发生": 1} and closing.open_requests == 2  # 名额 2，「待取」占名额
    earn(ledger, kind="s-k0005")
    ledger.record_expectation(expectation("t-late", DAY0 + timedelta(minutes=50), kind="s-k0005"))
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=65)))
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=91))).by_reason == {"名额已满": 1}
    # 过了最晚时限的不再说
    ledger.record_expectation(expectation("t-over", DAY0 + timedelta(minutes=100), kind="s-k0006", deadline_hours=1))
    earn(ledger, kind="s-k0006")
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=115)))
    # 11:41：t-over 过了最晚时限、没决定过，不再递；上午那两条「待取」到这时也没人问出口，记「宿主没取」
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=161))).by_reason == {"宿主没取": 2}
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=162))).decided == 0


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
    assert report.by_reason == {"待取": 1} and report.open_requests == 1
    (subsumed,) = [item for item in ledger.deliveries_on(DAY0.date()) if item.reason == "单元已选"]
    (said,) = [item for item in ledger.deliveries_on(DAY0.date()) if item.reason == "待取"]
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
    with pytest.raises(ValueError):
        DeliveryStage(ledger, **{**common, "daily_cap": 0})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DeliveryStage(ledger, **common, tick_seconds=0)  # type: ignore[arg-type]
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
    assert closing.by_reason == {"影子": 1, "待取": 1} and provider.calls == 2
    stored = {item.expectation_id: item for item in ledger.deliveries_on(DAY0.date()) if item.reason != "等时窗末"}
    assert stored["silenced"].level == 0 and not stored["silenced"].spoken
    assert stored["spoken"].text == "该去医院了"


def test_a_window_longer_than_a_week_is_still_closed_at_its_end(tmp_path) -> None:
    run, ledger = stage(tmp_path, tick_seconds=900.0)
    earn(ledger)
    ledger.record_expectation(expectation("long", DAY0, window=(10, 12 * 24 * 60), deadline_hours=13 * 24))
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=15))).by_reason == {"等时窗末": 1}
    end = DAY0 + timedelta(days=12, minutes=15)
    assert asyncio.run(run.run_once(end)).by_reason == {"待取": 1}


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

    def read(kind: str) -> tuple[Seen, ...]:
        return (Seen(started, started, kind, "behavior://x.md", "订机票", ""),)

    changed = JudgeMaterials(table, rows("s-k0009"), seen=read("s-k0009"))
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=15), changed))
    closing = asyncio.run(run.run_once(DAY0 + timedelta(minutes=41), changed))
    assert closing.by_reason == {"前因不成立": 1} and closing.spoken == 0
    run2, ledger2 = stage(tmp_path / "b")
    earn(ledger2, source="事件")
    ledger2.record_expectation(item)
    holds = JudgeMaterials(table, rows("s-k0001"), seen=read("s-k0001"))
    asyncio.run(run2.run_once(DAY0 + timedelta(minutes=15), holds))
    assert asyncio.run(run2.run_once(DAY0 + timedelta(minutes=41), holds)).by_reason == {"待取": 1}


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
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=41))).by_reason == {"待取": 1}  # 只有依据那条在时窗末说


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
    assert asyncio.run(run.run_once(tomorrow + timedelta(minutes=41), materials)).by_reason == {"待取": 1}


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
    assert closing.by_reason == {"待取": 1}
    unit = ledger.unit_level_of("s-k0002")
    assert unit is not None and unit.level == 1
    assert all(ledger.level_of(account) is None for account in weak)  # 账自己的级没被单元的档改写


# --- 久别回来（26 号清单 3.3）----------------------------------------------------------------------


def _absence_table():  # type: ignore[no-untyped-def]
    from habitus.foresight.opportunities import DayWatch, OpportunityTable

    watch = tuple(DayWatch(DAY0.date() - timedelta(days=offset), 1.0) for offset in range(30, 0, -1))
    return OpportunityTable("session", DAY0.date() - timedelta(days=1), None, 15, 3, (), (), watch=watch)


def _middle_band(ledger: ForesightLedgerStore) -> str:
    """4/4 的账（下界不到线、上界过线）、上一次在提醒档：靠滞回留在提醒档。四次隔一天一次，之后隔了 16 天（8 个典型间隔，裁定 88）才又来。"""

    from habitus.foresight.ledger import AccountLevel, UnitLevel

    account = account_key("s-k0002", "日子", "rel")
    for offset in (22, 20, 18, 16):
        ledger.append_fact(AccountFact(account, DAY0 - timedelta(days=offset), True, 1.0, "准时", f"old-{offset}"))
    seeded_of(ledger).put(
        "s-k0002", [PastMoment(DAY0 - timedelta(days=offset), frozenset({account}), True) for offset in (22, 20, 18, 16)]
    )
    ledger.record_level(AccountLevel(account, DAY0 - timedelta(days=23), 1))
    ledger.record_unit_level(UnitLevel("s-k0002", DAY0 - timedelta(days=23), 1))
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


def _daily_habit(ledger: ForesightLedgerStore, offsets: range) -> None:
    """天天做的事（``offsets`` 是 DAY0 往前的天数），上一次在提醒档。"""

    from habitus.foresight.ledger import AccountLevel, UnitLevel

    account = account_key("s-k0002", "日子", "rel")
    for offset in offsets:
        ledger.append_fact(AccountFact(account, DAY0 - timedelta(days=offset), True, 1.0, "准时", f"old-{offset}"))
    seeded_of(ledger).put(
        "s-k0002", [PastMoment(DAY0 - timedelta(days=offset), frozenset({account}), True) for offset in offsets]
    )
    first = DAY0 - timedelta(days=max(offsets) + 1)
    ledger.record_level(AccountLevel(account, first, 1))
    ledger.record_unit_level(UnitLevel("s-k0002", first, 1))


@pytest.mark.parametrize(
    ("offsets", "reason"),
    [
        (range(21, 1, -1), "等时窗末"),  # 漏了昨天一天：空白 2 天，照常（裁定 88 第 1 条，E1）
        (range(27, 6, -1), "等时窗末"),  # 空白 7 天（中间 6 天没做）：不到 8 个典型间隔
        (range(28, 7, -1), "久别回来"),  # 空白 8 天（中间 7 天没做）：久别
    ],
)
def test_the_configured_half_life_decides_a_daily_habits_return_after_absence(tmp_path, offsets, reason) -> None:
    """组合根那一份 H（``example.yaml`` → ``HabitusConfig`` → ``evidence_of``）一路传到递送那一拍的久别判法：
    天天做的事漏一天不算久别、照常等时窗末；空白满 8 天才「久别回来」。"""

    import yaml

    from habitus.config import HabitusConfig
    from habitus.runtime.evidence import evidence_of
    from tests.integration.test_runtime_assembly import REPOSITORY_ROOT

    raw = yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))
    raw["storage"]["root"] = str(tmp_path / "data")
    evidence = evidence_of(HabitusConfig.from_mapping(raw))
    assert evidence.half_life == 8.0
    run, ledger = stage(tmp_path, evidence=evidence)
    _daily_habit(ledger, offsets)
    ledger.record_expectation(expectation("today", DAY0))
    opening = asyncio.run(run.run_once(DAY0 + timedelta(minutes=15), JudgeMaterials(_absence_table(), lambda uris: {})))
    assert opening.by_reason == {reason: 1} and opening.by_level == {1: 1} and opening.spoken == 0


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


def test_a_closing_window_with_an_unreadable_stretch_is_not_spoken(tmp_path) -> None:
    """裁定 67 第 14 条：时窗末说中的范围里有「没读懂」——他可能已经做了、我们没看到，记「说不准」、不说、出口取不到。"""

    run, ledger = stage(tmp_path)
    earn(ledger)
    ledger.record_expectation(expectation("today", DAY0))
    plain = JudgeMaterials(None, lambda uris: {})
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=16), plain))  # 时窗开：等时窗末
    blind = JudgeMaterials(None, lambda uris: {}, unobserved=lambda item: item.expectation_id == "today")
    closing = asyncio.run(run.run_once(DAY0 + timedelta(minutes=41), blind))
    assert closing.by_reason == {"说不准": 1} and closing.spoken == 0
    assert not any(item.reason == "待取" for item in ledger.deliveries_on(DAY0.date()))


def test_a_closing_window_with_an_unreadable_stretch_is_reread_before_deciding(tmp_path) -> None:
    """裁定 68 第 1 条：范围里有「没读懂」先当场重读——读出了什么，这一拍不决定，下一拍按读出的再判（正是这件事就
    「已发生」）；读不出来（模型用不了、堆里没有）才记「说不准」。"""

    run, ledger = stage(tmp_path)
    earn(ledger)
    ledger.record_expectation(expectation("today", DAY0))
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=16), JudgeMaterials(None, lambda uris: {})))
    asked: list[str] = []

    async def recovered(item) -> bool:  # type: ignore[no-untyped-def]
        asked.append(item.expectation_id)
        return True

    blind = JudgeMaterials(None, lambda uris: {}, unobserved=lambda item: True, reread=recovered)
    deferred = asyncio.run(run.run_once(DAY0 + timedelta(minutes=41), blind))
    assert asked == ["today"] and deferred.by_reason == {} and deferred.decided == 0
    seen = JudgeMaterials(None, lambda uris: {}, happened=lambda item: True)
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=42), seen)).by_reason == {"已发生": 1}

    run2, ledger2 = stage(tmp_path / "unavailable")
    earn(ledger2)
    ledger2.record_expectation(expectation("today", DAY0))
    asyncio.run(run2.run_once(DAY0 + timedelta(minutes=16), JudgeMaterials(None, lambda uris: {})))

    async def unavailable(_item) -> bool:  # type: ignore[no-untyped-def]
        return False

    stuck = JudgeMaterials(None, lambda uris: {}, unobserved=lambda item: True, reread=unavailable)
    assert asyncio.run(run2.run_once(DAY0 + timedelta(minutes=41), stuck)).by_reason == {"说不准": 1}


# --- 递送与结算对同一份行的读法（段 6 Q4）---------------------------------------------------------------------

_Q4_CASES = {
    "对照": ("讨论方案", (9, 0), "修改代码"),
    "前因开始被更正 09:00→08:57": ("讨论方案", (8, 57), "修改代码"),
    "前因拆改": ("讨论方案-拆A", (9, 0), "修改代码"),
    "后果拆改、他已经做了": ("讨论方案", (9, 0), "修改代码-拆"),
    "前因开始被更正、没做": ("讨论方案", (8, 57), None),
    "前因拆改、没做": ("讨论方案-拆A", (9, 0), None),
}


@pytest.mark.parametrize("case", list(_Q4_CASES))
def test_delivery_and_settlement_read_the_same_rows_the_same_way(tmp_path, case) -> None:
    """段 6 Q4：同一棵树、同一张表，递送（这一拍读到的行 → 开口前核前因、判做没做）与结算（树上的行 → 前因成立吗、第一次
    发生）结论相同。原来前因开始被更正 / 拆改后递送说不成立（不说）、结算说成立（他没做就记过期待决，账上一笔坏）；后果拆改后
    递送说没做（他做完了还被问）、结算说准时。"""

    from datetime import date

    from habitus.conditions import LocalCalendar
    from habitus.foresight.delivery import done_since
    from habitus.foresight.ledger import AntecedentRow, antecedent_holds_when_settling, first_occurrence
    from habitus.foresight.model import Moment as TickMoment
    from habitus.foresight.opportunities import OpportunityTable, RelationOpportunities
    from habitus.foresight.triggers.assemble import seen_rows
    from habitus.foresight.triggers.draft import instance_at
    from habitus.runtime.foresight_tick import ForesightTick
    from habitus.scene.views import DayIndexCache
    from habitus.series.reader import admitted
    from tests.unit.kind_ids import kind_id
    from tests.unit.scene.fixtures import SUBJECT, Site, at, publish

    day = date(2026, 12, 10)
    discuss, code = kind_id("讨论方案"), kind_id("修改代码")
    split = {discuss: (kind_id("讨论方案-拆A"),), code: (kind_id("修改代码-拆"),)}

    def current(token: str) -> tuple[str, ...]:
        return split.get(token, (token,))

    relation = RelationOpportunities(
        key="session|A|C|chain",
        antecedent="A",
        consequent="C",
        antecedent_label="讨论方案",
        consequent_label="修改代码",
        antecedent_classes=(discuss,),
        consequent_classes=(code,),
        segment="chain",
        status="established",
        upward=True,
        items=(),
    )
    table = OpportunityTable("session", day, day, 15, 3, (relation,), ())
    item = Expectation(
        expectation_id="e1",
        kind_token=code,
        day=day,
        formed_at=at(day, 9, 25),
        source="事件",
        trigger=relation.key,
        trigger_instance=instance_at(at(day, 9, 0)),  # 建预期时（前因还没封口）前因开始在 09:00
        window=(at(day, 9, 30), at(day, 9, 50)),
        deadline=at(day, 10, 25),
        basis=(),
        opportunities=10,
        hits=8,
        observed_from=at(day, 9, 20),
    )
    antecedent, start, consequent = _Q4_CASES[case]
    site = Site(tmp_path, now=at(day, 23, 0))
    publish(site.behavior_tree, day, "前因", *start, kind=antecedent, lasts_minutes=20)
    if consequent is not None:
        publish(site.behavior_tree, day, "后果", 9, 40, kind=consequent)
    cache = DayIndexCache(site.behavior_tree, subject=SUBJECT, admits=admitted, calendar=LocalCalendar())
    now = at(day, 9, 51)
    seen = seen_rows(TickMoment(at=now, slot=0), cache, ())
    run, _ledger = stage(tmp_path / "stage", current=current)
    materials = JudgeMaterials(table, lambda uris: ForesightTick._rows_for(uris, cache, (), seen, now), seen=seen)
    settling = antecedent_holds_when_settling(
        item,
        relation,
        official=False,
        calendar_on=lambda _day: (),
        rows_on=lambda found: [AntecedentRow(r.at, r.last_observed_at, r.kind_token) for r in cache.day(found).rows],
        current=current,
    )
    assert run._antecedent_holds(item, materials) is settling is True
    done = first_occurrence(item, cache.day(day).rows, current=current) is not None
    assert done_since(item, seen, now=now, current=current) is done is (consequent is not None)


@pytest.mark.parametrize(
    ("deadline_minutes", "closing_at"),
    [(60, "10:06"), (52, "10:06"), (57, "10:06"), (70, "10:06")],
    ids=["终点到时限 8 分钟", "终点等于时限", "5 分钟", "对照 18 分钟"],
)
def test_the_window_end_is_caught_by_the_tick_whose_interval_meets_it_and_waiting_always_ends(
    tmp_path, deadline_minutes, closing_at
) -> None:
    """段 6 Q7：时窗末那一拍按拍区间与 (时窗终点, 最晚时限] 取交集，不再要求"此刻"落在区间里——区间短于一拍、为 0 时
    原来永远到不了时窗末，账停在「等时窗末」。拍长 15 分钟、拍点在 :06 :21 :36 :51，时窗 09:10–09:52。"""

    from dataclasses import replace

    run, ledger = stage(tmp_path, tick_seconds=900.0)
    earn(ledger)
    item = replace(expectation("e", DAY0, window=(10, 52)), deadline=DAY0 + timedelta(minutes=deadline_minutes))
    ledger.record_expectation(item)
    for minutes in (6, 21, 36, 51, 66, 81):
        asyncio.run(run.run_once(DAY0 + timedelta(minutes=minutes), NO_ROWS))
    decided = sorted(ledger.deliveries_on(DAY0.date()), key=lambda found: found.decided_at)
    assert [found.reason for found in decided][0] == "等时窗末"
    assert len(decided) >= 2 and decided[1].decided_at.strftime("%H:%M") == closing_at
    assert decided[-1].reason not in ("等时窗末", "待取")  # 有终局


def test_a_window_end_reached_already_in_the_opening_tick_is_decided_in_the_next_tick(tmp_path) -> None:
    """开时窗那一拍就已越过终点（时窗 09:03–09:05，09:06 那一拍开窗时就过了终点）：终点之后的第一拍是下一拍，照样决定；
    中间停过机、(终点, 最晚时限] 整段落在没跑的拍里的不再决定（过了时限的不再说）。"""

    from dataclasses import replace

    run, ledger = stage(tmp_path, tick_seconds=900.0)
    earn(ledger)
    early = replace(expectation("early", DAY0, window=(3, 5)), deadline=DAY0 + timedelta(minutes=5))
    ledger.record_expectation(early)
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=6), NO_ROWS)).by_reason == {"等时窗末": 1}
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=21), NO_ROWS)).decided == 1
    # 终点与最晚时限重合、上一拍正好落在终点：越过终点的这一拍照样决定
    run3, ledger3 = stage(tmp_path / "c", tick_seconds=900.0)
    earn(ledger3)
    edge = replace(expectation("edge", DAY0, window=(1, 5)), deadline=DAY0 + timedelta(minutes=5))
    ledger3.record_expectation(edge)
    assert asyncio.run(run3.run_once(DAY0 + timedelta(minutes=2), NO_ROWS)).by_reason == {"等时窗末": 1}
    assert asyncio.run(run3.run_once(DAY0 + timedelta(minutes=5), NO_ROWS)).decided == 0
    assert asyncio.run(run3.run_once(DAY0 + timedelta(minutes=20), NO_ROWS)).decided == 1
    run2, ledger2 = stage(tmp_path / "b", tick_seconds=900.0)
    earn(ledger2)
    missed = replace(expectation("missed", DAY0, window=(1, 20)), deadline=DAY0 + timedelta(minutes=30))
    ledger2.record_expectation(missed)
    assert asyncio.run(run2.run_once(DAY0 + timedelta(minutes=6), NO_ROWS)).by_reason == {"等时窗末": 1}
    assert asyncio.run(run2.run_once(DAY0 + timedelta(hours=3), NO_ROWS)).decided == 0


@pytest.mark.parametrize(("done_minutes", "reason"), [(None, "宿主没取"), (130, "宿主没取"), (90, "已发生")])
def test_an_awaiting_request_done_only_after_its_deadline_ends_as_never_claimed(tmp_path, done_minutes, reason) -> None:
    """段 6 Q15：「待取」过了最晚时限（11:00）才做的不算这一回做了（结算结成过期待决），记「宿主没取」；原来 11:11 那一拍
    把 11:10 的那次读成「已发生」。时限内（10:30）做了的照常记「已发生」。"""

    from habitus.foresight import Seen
    from habitus.foresight.delivery import done_since

    run, ledger = stage(tmp_path, tick_seconds=900.0)
    earn(ledger)
    ledger.record_expectation(expectation("e", DAY0, window=(10, 40)))  # 时窗 09:10–09:40，最晚时限 11:00
    rows = (
        ()
        if done_minutes is None
        else (Seen(DAY0 + timedelta(minutes=done_minutes), DAY0 + timedelta(minutes=done_minutes + 5), "s-k0002", "u"),)
    )

    def materials(now: datetime) -> JudgeMaterials:
        seen = tuple(row for row in rows if row.started_at <= now)
        return JudgeMaterials(
            None, lambda uris: {}, happened=lambda item: done_since(item, seen, now=now, current=run.current)
        )

    for minutes in (11, 41, 56, 71, 86, 101, 116, 131, 146):
        now = DAY0 + timedelta(minutes=minutes)
        asyncio.run(run.run_once(now, materials(now)))
    decided = [item.reason for item in sorted(ledger.deliveries_on(DAY0.date()), key=lambda found: found.decided_at)]
    assert decided[:2] == ["等时窗末", "待取"] and decided[-1] == reason and len(decided) == 3


def test_an_input_not_yet_due_joins_only_one_unit_and_the_tick_does_not_fail(tmp_path) -> None:
    """段 6 Q19：同一类两组到了时窗、互不相连（A 09:00 起、09:50 止；B 09:51 起），还没到时窗的 W 和两组都连着——原来
    W 并进两个单元、两边各记一条「单元已选」，同一刻同一身份两份内容，这一拍整个抛（所有类的递送中断，A 丢一次决定）。
    一个输入只进一个单元。"""

    from dataclasses import replace

    def at(minutes: int) -> datetime:
        return DAY0 + timedelta(minutes=minutes)

    def made(identity: str, trigger: str, observed: int, window: tuple[int, int], deadline: int) -> Expectation:
        item = expectation(identity, DAY0, trigger=trigger)
        return replace(
            item,
            observed_from=at(observed),
            formed_at=at(observed),
            window=(at(window[0]), at(window[1])),
            deadline=at(deadline),
        )

    run, ledger = stage(tmp_path, tick_seconds=900.0)
    for trigger in ("a", "b", "w"):
        earn(ledger, trigger=trigger)
    for item in (made("A", "a", 0, (41, 45), 50), made("B", "b", 51, (52, 90), 120), made("W", "w", 0, (90, 105), 180)):
        ledger.record_expectation(item)
    report = asyncio.run(run.run_once(at(55)))
    assert report.by_reason.get("单元已选") == 1
    subsumed = [item for item in ledger.deliveries_on(DAY0.date()) if item.expectation_id == "W"]
    assert len(subsumed) == 1 and subsumed[0].reason == "单元已选"
    assert {item.expectation_id for item in ledger.deliveries_on(DAY0.date())} == {"A", "B", "W"}


# --- 段 6 Q18：存活变异补的断言 ----------------------------------------------------------------------------------


def test_a_window_that_closed_between_two_ticks_is_still_opened_by_the_later_tick(tmp_path) -> None:
    """时窗在两拍之间开、又在两拍之间关（09:07–09:12，拍点 09:00、09:15）：09:15 那一拍的区间盖到了它，照样决定
    （M59：改成"终点不早于此刻"就丢了这次机会）。"""

    run, ledger = stage(tmp_path, tick_seconds=900.0)
    ledger.record_expectation(expectation("between", DAY0, window=(7, 12)))
    assert asyncio.run(run.run_once(DAY0)).decided == 0
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=15))).by_reason == {"影子": 1}


def test_the_window_end_moment_itself_is_still_inside_the_window(tmp_path) -> None:
    """拍点恰在时窗终点：时窗还没过完，不是时窗末那一拍（M60：``<`` 改 ``<=``）；下一拍才是。"""

    run, ledger = stage(tmp_path, tick_seconds=900.0)
    earn(ledger)
    ledger.record_expectation(expectation("e", DAY0, window=(10, 30)))
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=15), NO_ROWS)).by_reason == {"等时窗末": 1}
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=30), NO_ROWS)).decided == 0
    assert asyncio.run(run.run_once(DAY0 + timedelta(minutes=45), NO_ROWS)).by_reason == {"待取": 1}


def test_the_model_is_not_asked_once_the_days_cap_is_full(tmp_path) -> None:
    """名额满了不问模型（成本保护，M63：去掉这一道也全绿）：记「名额已满」，一次也没调。"""

    client, provider = recording_client([LIKE, LIKE])
    run, ledger = stage(tmp_path, judge=Judge(client, retries=0, delay_seconds=0.0))
    for index, kind in enumerate(("s-k0007", "s-k0008")):
        said = expectation(f"said-{index}", DAY0, kind=kind)
        ledger.record_expectation(said)
        ledger.record_delivery(said.day, Delivery(said.expectation_id, DAY0, 1, True, "说了", 1.0, "话"))
    earn(ledger)
    ledger.record_expectation(expectation("e", DAY0))
    asyncio.run(run.run_once(DAY0 + timedelta(minutes=15), NO_ROWS))
    closing = asyncio.run(run.run_once(DAY0 + timedelta(minutes=41), NO_ROWS))
    assert closing.by_reason == {"名额已满": 1} and provider.calls == 0


def test_a_later_input_joins_the_basis_not_a_member_the_unit_already_decided_for(tmp_path) -> None:
    """后来才到时窗的同一类输入只并进依据，不并进「单元已选」的成员（M66：成员也被当成依据，后来并进的那条引错依据，
    结算的说出时刻就跟着错）：和成员连着、和依据不连着的那条自己开一个单元。"""

    from dataclasses import replace

    def at(minutes: int) -> datetime:
        return DAY0 + timedelta(minutes=minutes)

    def made(identity: str, trigger: str, observed: int, window: tuple[int, int], deadline: int) -> Expectation:
        item = expectation(identity, DAY0, trigger=trigger)
        return replace(
            item,
            observed_from=at(observed),
            formed_at=at(observed),
            window=(at(window[0]), at(window[1])),
            deadline=at(deadline),
        )

    run, ledger = stage(tmp_path, tick_seconds=900.0)
    for trigger in ("a", "m", "f"):
        earn(ledger, trigger=trigger)
    ledger.record_expectation(made("A", "a", 0, (10, 40), 60))  # 依据：09:00 → 10:00
    ledger.record_expectation(made("M", "m", 30, (90, 120), 180))  # 和 A 连着、还没到时窗：成员
    asyncio.run(run.run_once(at(15)))
    member = next(item for item in ledger.deliveries_on(DAY0.date()) if item.expectation_id == "M")
    assert member.reason == "单元已选" and member.evidence == ("单元:A",)
    ledger.record_expectation(made("F", "f", 90, (100, 130), 150))  # 和 M 连着、和 A 不连着
    asyncio.run(run.run_once(at(105)))
    later = [item for item in ledger.deliveries_on(DAY0.date()) if item.expectation_id == "F"]
    assert later and all(item.evidence != ("单元:M",) for item in later)
