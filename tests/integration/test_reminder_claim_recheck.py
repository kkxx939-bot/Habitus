"""宿主来取提醒的那一刻再判一次（Q19；裁定 65 第 2 条、裁定 104）：

已经做了的不给；有已经交进来但还没判完的轮，这一次先不给，下一次再取；被拦下的这条不写取走记录、放回次数不变、不占 ``limit``、
还在可取清单上。已经交到宿主手里的（同一会话租约内重取、租约过期后续租）不算这一次新给，不再判。两支各看自己的。

入口一律是 ``Runtime.claim_reminders``（``tests.unit.physical_downstream.site.World`` 那一套真实组装的 Runtime）：「待取」由真实的
预测拍写出来，做过的那一行走真实的会话 lane、真实的行为树，交进来的会话源走真实的 Source Store 与 Consumer 交付。断言只落在
``claim`` 返回的 ``request_id``、账本、可取清单、取走记录与放回次数上。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from habitus.behavior.reduction.kinds_step import record_material
from habitus.behavior.reduction.pending import pending_judgements
from habitus.conversation import (
    ConversationConsumerOutcomeState,
    ConversationSourceConsumer,
    ConversationSourceEnvelope,
    conversation_source_request_digest,
)
from habitus.foresight.edges import PastMoment
from habitus.foresight.history import Moment, UnitHistory
from habitus.foresight.ledger import (
    AccountFact,
    AntecedentRow,
    Expectation,
    account_key,
    antecedent_holds_when_delivering,
)
from habitus.foresight.opportunities import OpportunityTable, RelationOpportunities
from habitus.pre.conversation import (
    ConversationAdapterRegistry,
    ConversationBatch,
    ConversationMessage,
    ConversationMessageRole,
)
from habitus.runtime.foresight import unsealed_bridge
from habitus.runtime.foresight_history_store import HistoryStore
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.runtime.foresight_outlet import OPTIONS, HostMessage
from habitus.runtime.foresight_settlement import Asked
from tests.unit.physical_downstream.site import HOST, MEDICINE, SESSION, Named, World, at, delivery
from tests.unit.scene.fixtures import publish

DAY = date(2026, 4, 14)
METRIC = SESSION["记指标"]  # 第 1 层，记体重、记血压的上级
WEIGHT = SESSION["记体重"]  # 第 2 层
PRESSURE = SESSION["记血压"]  # 第 2 层，记体重的兄弟
WEEKLY = SESSION["写周报"]
CONVERSATION = "conv-recheck"
#: 事件预期那几条用的关系身份（前因写周报 → 后果记体重）；日子预期的 ``trigger`` 是 ``World.expect`` 写死的 ``interval``。
RELATION = f"session|{WEEKLY}|{WEIGHT}|chain"
DAILY = "interval"


@pytest.fixture
def world(tmp_path: Path) -> Iterator[World]:
    site = World(tmp_path, start=at(DAY, 0))
    try:
        yield site
    finally:
        site.close()


# ── 走拍 ──────────────────────────────────────────────────────────────────────


def tick(world: World, moment: datetime) -> None:
    """``moment`` 这一拍：两支真实的预测拍（建预期、递送、重写可取清单）。不看换代：这里的日子不需要它，机会表由用例自己写。"""

    world.set_clock(moment)
    worker = world.runtime.components.slot_worker
    assert worker is not None
    for branch in worker.branches:
        assert branch.tick is not None
        asyncio.run(branch.tick.run_once())


def ticks(world: World, start: datetime, end: datetime) -> None:
    moment = start
    while moment <= end:
        tick(world, moment)
        moment += timedelta(minutes=15)


def ready(
    world: World, lane: str, identity: str, kind: str, *, hour: int = 7, deadline: datetime | None = None
) -> None:
    """这一支一条预期：时窗 ``hour``:25–``hour``:40，账到提醒档、节点在用。"""

    world.use(lane, kind)
    world.earn(lane, kind, at(DAY, 0))
    world.expect(
        lane,
        identity,
        kind,
        window=(at(DAY, hour, 25), at(DAY, hour, 40)),
        deadline=deadline or at(DAY + timedelta(days=1), 0),
    )


def awaiting(
    world: World,
    *,
    lane: str = "session",
    identity: str = "weight",
    kind: str = WEIGHT,
    hour: int = 7,
    deadline: datetime | None = None,
) -> None:
    """共同摆法：宿主活着，``hour``:30 那一拍记「等时窗末」、``hour``:45 那一拍记「待取」，这条在可取清单上。"""

    ready(world, lane, identity, kind, hour=hour, deadline=deadline)
    assert world.claim(at(DAY, hour), session="probe") == ()
    tick(world, at(DAY, hour, 30))
    tick(world, at(DAY, hour, 45))
    assert world.reasons(lane, DAY)[identity] == ["等时窗末", "待取"]
    assert identity in listed(world)


def moment_of(hour: int, minute: int, second: int = 0) -> datetime:
    return at(DAY, hour, minute).replace(second=second)


def did(world: World, word: str, started: datetime, *, arrived: datetime) -> None:
    """他 ``started`` 做的这一轮 ``arrived`` 才交进系统：真实的会话 lane 写下判断（带选词结果），还没封口。"""

    world.router.pick_default = Named(word)
    world.set_clock(arrived)
    world.turn("conv", started, f"帮我{word}")


# ── 落盘的读法 ────────────────────────────────────────────────────────────────


def outlet(world: World) -> Any:
    found = world.runtime.components.outlet
    assert found is not None
    return found


def listed(world: World) -> list[str]:
    """可取清单上的请求（两支的都在）。"""

    return [str(row["request_id"]) for row in outlet(world).store.read_open()]


def claim_file(world: World, identity: str, *, lane: str = "session") -> Path:
    return world.foresight(lane) / "outlet" / "claims" / f"{identity}.json"


def given(claimed: tuple[Any, ...]) -> list[str]:
    return [item.request_id for item in claimed]


def spoken_on_ledger(world: World, lane: str = "session") -> list[str]:
    return [item.expectation_id for item in world.ledger(lane).deliveries_on(DAY) if item.spoken]


def assert_held_back(world: World, identity: str, *, lane: str = "session") -> None:
    """被拦下的这条：没有取走记录、没有放回次数、还在可取清单上、账上没有「说了」。"""

    assert not claim_file(world, identity, lane=lane).exists()
    assert identity not in outlet(world).trails()
    assert identity in listed(world)
    assert spoken_on_ledger(world, lane) == []


# ── 交进来的会话源（真实的 Source Store 与 Consumer 交付）─────────────────────


def envelope(moment: datetime, said: str) -> ConversationSourceEnvelope:
    messages = (
        ConversationMessage(
            message_id="recheck-0", sequence=0, role=ConversationMessageRole.PROMPT, occurred_at=moment, content=said
        ),
        ConversationMessage(
            message_id="recheck-1",
            sequence=1,
            role=ConversationMessageRole.COMPLETION,
            occurred_at=moment + timedelta(minutes=1),
            content="好的，办好了。",
        ),
    )
    batch = ConversationBatch(CONVERSATION, messages)
    digest = conversation_source_request_digest(
        conversation_id=CONVERSATION,
        started_on=DAY,
        protocol="normalized",
        batch=batch,
        after_turn=True,
        omit_tool_call_ids=frozenset(),
    )
    return ConversationSourceEnvelope.create(
        conversation_id=CONVERSATION,
        started_on=DAY,
        protocol="normalized",
        batch=batch,
        after_turn=True,
        omit_tool_call_ids=frozenset(),
        delivery_id=digest,
        request_digest=digest,
        recorded_at=moment + timedelta(minutes=1),
    )


def handed_in(world: World, moment: datetime, said: str) -> ConversationSourceEnvelope:
    """这一份会话源交进来了（耐久落盘），两个 Consumer 都还没判。"""

    world.set_clock(moment + timedelta(minutes=1))
    return world.runtime.components.conversation.sources.put(envelope(moment, said))


def judged(world: World, source: ConversationSourceEnvelope, word: str, moment: datetime) -> None:
    """会话 lane 这个 Consumer 把这一份判完了（判出来是 ``word``）。"""

    world.router.pick_default = Named(word)
    world.set_clock(moment)
    delivery_port = world.runtime.components.conversation.source_delivery

    async def run() -> None:
        ensured = await delivery_port.ensure_outcome(source, ConversationSourceConsumer.BEHAVIOR_SESSION)
        assert ensured.outcome.state is ConversationConsumerOutcomeState.COMMITTED

    asyncio.run(run())


# ── 机会表（换代之后依据变回候选那几条）─────────────────────────────────────


def relation() -> RelationOpportunities:
    return RelationOpportunities(
        key=RELATION,
        antecedent=WEEKLY,
        consequent=WEIGHT,
        antecedent_label="写周报",
        consequent_label="记体重",
        antecedent_classes=(WEEKLY,),
        consequent_classes=(WEIGHT,),
        segment="chain",
        status="established",
        upward=True,
        items=(),
    )


def write_table(world: World, night: date, *relations: RelationOpportunities) -> None:
    OpportunityStore(world.foresight("session")).write(OpportunityTable("session", night, night, 15, 3, relations, ()))


def earn_together(world: World, kind: str, *accounts: str) -> None:
    """这几本账前八天每一刻都一起在场、后果都来了：单元到提醒档。"""

    ledger, store = world.ledger("session"), HistoryStore(world.foresight("session"))
    moments = []
    for offset in range(8, 0, -1):
        moment = at(DAY, 0) - timedelta(days=offset)
        for index, account in enumerate(accounts):
            ledger.append_fact(AccountFact(account, moment, True, 1.0, "准时", f"old-{index}-{offset}"))
        moments.append(PastMoment(moment, frozenset(accounts), True))
    store.write(
        UnitHistory(
            "session",
            DAY - timedelta(days=2),
            {kind: tuple(Moment(item, (f"{kind}-{index}",)) for index, item in enumerate(moments))},
        )
    )


def event_expectation(world: World, identity: str, kind: str, antecedent_at: datetime) -> Expectation:
    """一条事件预期：前因是 ``antecedent_at`` 开始的那一行，时窗 07:25–07:40，最晚时限次日零点。"""

    formed = at(DAY, 0)
    item = Expectation(
        expectation_id=identity,
        kind_token=kind,
        day=DAY,
        formed_at=formed,
        source="事件",
        trigger=RELATION,
        trigger_instance=antecedent_at.astimezone(UTC).isoformat(),
        window=(at(DAY, 7, 25), at(DAY, 7, 40)),
        deadline=at(DAY + timedelta(days=1), 0),
        basis=(),
        opportunities=8,
        hits=8,
        observed_from=formed,
    )
    world.ledger("session").record_expectation(item)
    return item


def event_awaiting(world: World) -> None:
    """一条事件预期走到「待取」：关系成立、前因（07:00 写周报）在，07:45 那一拍记「待取」。"""

    world.use("session", WEIGHT, WEEKLY)
    earn_together(world, WEIGHT, account_key(WEIGHT, "事件", RELATION), account_key(WEIGHT, "日子", DAILY))
    write_table(world, DAY - timedelta(days=1), relation())
    started = at(DAY, 7)
    did(world, "写周报", started, arrived=at(DAY, 7, 2))
    event_expectation(world, "weight", WEIGHT, started)
    assert world.claim(at(DAY, 7, 3), session="probe") == ()


# ── 复现：他已经做了，取的这一刻不给 ──────────────────────────────────────────


def test_q19_1_a_turn_not_sealed_yet_keeps_the_reminder_from_being_handed_out(world: World) -> None:
    """Q19-1（实盘 03-18）：07:43:18 开始的这一轮 07:45:37 才交进系统，07:45 那一拍没看到、记了「待取」；07:54:30 宿主来取。
    不给、不写取走记录、活性照记；08:00 那一拍账上追加「已发生」，没有「说了」。"""

    awaiting(world)
    did(world, "记体重", moment_of(7, 43, 18), arrived=moment_of(7, 45, 37))
    claimed_at = moment_of(7, 54, 30)
    assert given(world.claim(claimed_at)) == []
    assert_held_back(world, "weight")
    assert outlet(world).store.hosts()[HOST] == claimed_at  # 来取这件事本身照记活性
    tick(world, at(DAY, 8))
    assert world.reasons("session", DAY)["weight"] == ["等时窗末", "待取", "已发生"]
    assert spoken_on_ledger(world) == []


def test_q19_2_a_turn_that_arrived_after_the_awaiting_tick_keeps_the_reminder_back(world: World) -> None:
    """Q19-2（实盘 04-11）：09:45 记「待取」，这一轮 09:44:35 开始、09:46:26 进系统；09:49:44 来取不给，下一拍「已发生」。"""

    awaiting(world, hour=9)
    did(world, "记体重", moment_of(9, 44, 35), arrived=moment_of(9, 46, 26))
    assert given(world.claim(moment_of(9, 49, 44))) == []
    assert_held_back(world, "weight")
    tick(world, at(DAY, 10))
    assert world.reasons("session", DAY)["weight"] == ["等时窗末", "待取", "已发生"]


def test_q19_3_a_sealed_row_on_the_tree_keeps_the_reminder_back(world: World) -> None:
    """Q19-3：同 Q19-1，但那一行已经封口落树。"""

    awaiting(world)
    world.set_clock(moment_of(7, 50))
    publish(world.session.tree, DAY, "记体重", 7, 43, kind=WEIGHT)
    assert given(world.claim(moment_of(7, 54, 30))) == []
    assert_held_back(world, "weight")


def test_q19_4_doing_the_finer_kind_counts_for_its_parent(world: World) -> None:
    """Q19-4：E 是上级（记指标），他做的是下级（记体重）：下级算上级做了，不给。"""

    world.use("session", WEIGHT)
    awaiting(world, identity="metric", kind=METRIC)
    did(world, "记体重", moment_of(7, 43, 18), arrived=moment_of(7, 45, 37))
    assert given(world.claim(moment_of(7, 54, 30))) == []
    assert_held_back(world, "metric")


def test_q19_14_a_held_back_request_does_not_take_a_place_under_the_limit(world: World) -> None:
    """Q19-14：两条请求 A、B，A（时限更近）他做了、B 没做，``limit=1``：交出 B，A 不占名额、不写取走记录、还在清单上。"""

    world.use("session", WEIGHT, PRESSURE)
    world.earn("session", WEIGHT, at(DAY, 0))
    world.earn("session", PRESSURE, at(DAY, 0))
    window = (at(DAY, 7, 25), at(DAY, 7, 40))
    world.expect("session", "a", WEIGHT, window=window, deadline=at(DAY, 20))
    world.expect("session", "b", PRESSURE, window=window, deadline=at(DAY, 22))
    assert world.claim(at(DAY, 7), session="probe") == ()
    tick(world, at(DAY, 7, 30))
    tick(world, at(DAY, 7, 45))
    did(world, "记体重", moment_of(7, 43, 18), arrived=moment_of(7, 45, 37))
    assert given(world.claim(moment_of(7, 54, 30), limit=1)) == ["b"]
    assert_claimed_only(world, "b", held_back="a")


def assert_claimed_only(world: World, taken: str, *, held_back: str) -> None:
    assert claim_file(world, taken).is_file()
    assert not claim_file(world, held_back).exists()
    assert held_back in listed(world)


def test_q19_18_a_request_put_back_once_is_not_handed_out_again_once_he_has_done_it(world: World) -> None:
    """Q19-18：取走过一次、这一轮没问出口放回了（放回一次）；再取时他已经做了：不给，放回次数还是 1。"""

    awaiting(world)
    (first,) = world.claim(at(DAY, 7, 50))
    assert first.request_id == "weight"
    report = outlet(world).observe(
        "s1",
        (
            HostMessage(0, at(DAY, 7, 50).replace(second=5), "prompt", "帮我看看这个报错"),
            HostMessage(1, at(DAY, 7, 55), "completion", "报错看完了。"),
        ),
        host=HOST,
    )
    assert report.released == 1
    did(world, "记体重", at(DAY, 7, 56), arrived=at(DAY, 7, 57))
    assert given(world.claim(at(DAY, 7, 58))) == []
    trail = outlet(world).trails()["weight"]
    assert trail.attempts == 1 and len(trail.handoffs) == 1 and trail.handoffs[0].ended == "released"
    assert "weight" in listed(world)


# ── 复现：交进来了还没判完，这一次先不给 ──────────────────────────────────────


def test_q19_8_a_handed_in_source_not_judged_yet_holds_the_reminder_this_time(world: World) -> None:
    """Q19-8：会话一支有一份已交进来的会话源、会话 lane 还没判完：这一次不给、不写取走记录、放回次数不变。"""

    awaiting(world)
    handed_in(world, moment_of(7, 50), "帮我看一下")
    assert given(world.claim(moment_of(7, 54, 30))) == []
    assert_held_back(world, "weight")


def test_q19_9_once_that_source_is_judged_as_something_else_the_reminder_is_handed_out(world: World) -> None:
    """Q19-9（接 Q19-8）：这一份判完了、判出来是别的事：再取交出 E。"""

    awaiting(world)
    source = handed_in(world, moment_of(7, 50), "帮我量一下血压")
    assert given(world.claim(moment_of(7, 54, 30))) == []
    judged(world, source, "记血压", moment_of(7, 56))
    assert given(world.claim(moment_of(7, 58))) == ["weight"]
    assert claim_file(world, "weight").is_file()


def test_q19_10_once_that_source_is_judged_as_this_very_thing_it_stays_held_back(world: World) -> None:
    """Q19-10（接 Q19-8）：这一份判完了、判出来就是记体重：再取还是不给，下一拍「已发生」。"""

    awaiting(world)
    source = handed_in(world, moment_of(7, 50), "帮我记一下体重")
    assert given(world.claim(moment_of(7, 54, 30))) == []
    judged(world, source, "记体重", moment_of(7, 56))
    assert given(world.claim(moment_of(7, 58))) == []
    assert_held_back(world, "weight")
    tick(world, at(DAY, 8))
    assert world.reasons("session", DAY)["weight"] == ["等时窗末", "待取", "已发生"]


def test_q19_13_an_unpicked_physical_judgement_holds_a_physical_request_this_time(world: World) -> None:
    """Q19-13：物理一支有判断还没选上词，来取的是物理一支的请求：这一次不给；下一拍选上词（是别的事）之后再取交出。"""

    awaiting(world, lane="physical", identity="pill", kind=MEDICINE)
    world.deliver(
        delivery(
            "d-run",
            [("跑步", at(DAY, 7, 30), at(DAY, 7, 40))],
            [(at(DAY, 7, 20), at(DAY, 7, 50), "observing")],
            sent_at=at(DAY, 7, 50),
        ),
        now=at(DAY, 7, 50),
    )
    assert unsealed_bridge(world.physical).unclassified() > 0
    assert given(world.claim(moment_of(7, 54, 30))) == []
    assert_held_back(world, "pill", lane="physical")
    tick(world, at(DAY, 8))
    assert given(world.claim(at(DAY, 8, 5))) == ["pill"]


# ── 不变：不算这件事做了的，照常交出 ──────────────────────────────────────────


@pytest.mark.parametrize("word", ["记指标", "记血压"], ids=["上级", "兄弟"])
def test_q19_5_doing_the_parent_or_a_sibling_does_not_count_for_the_finer_kind(world: World, word: str) -> None:
    """Q19-5：E 是下级（记体重），他做的是它的上级或兄弟：不算下级做了，照常交出。"""

    awaiting(world)
    did(world, word, moment_of(7, 43, 18), arrived=moment_of(7, 45, 37))
    assert given(world.claim(moment_of(7, 54, 30))) == ["weight"]


def test_q19_6_an_abandoned_row_does_not_count_as_done(world: World) -> None:
    """Q19-6：那一行是放弃了的（abandoned）：不算做了，照常交出。"""

    awaiting(world)
    world.set_clock(moment_of(7, 50))
    publish(world.session.tree, DAY, "记体重", 7, 43, kind=WEIGHT, status="abandoned")
    assert given(world.claim(moment_of(7, 54, 30))) == ["weight"]


def test_q19_7_a_row_started_before_the_expectation_began_does_not_count(world: World) -> None:
    """Q19-7：那一行开始于起算之前（昨天做的）：不算这一回做了，照常交出。"""

    awaiting(world)
    did(world, "记体重", at(DAY - timedelta(days=1), 23, 30), arrived=moment_of(7, 50))
    assert given(world.claim(moment_of(7, 54, 30))) == ["weight"]


def test_q19_11_a_leftover_not_terminal_marker_of_the_memory_consumer_does_not_hold_the_reminder(world: World) -> None:
    """Q19-11（专门用例）：这份会话源只有 MEMORY 没到终态、BEHAVIOR_SESSION 已经终态。"还没终态"标记是两个 Consumer 共用的、
    惰性清除的残留：它还在，但会话 lane 已经判完了，不挡，照常交出。"""

    awaiting(world)
    source = handed_in(world, moment_of(7, 50), "帮我量一下血压")
    judged(world, source, "记血压", moment_of(7, 52))
    conversation = world.runtime.components.conversation
    marker = {item[0] for item in conversation.source_ledger.pending.listing().sources}
    assert source.source_id in marker  # 残留标记确实还在
    states = {
        consumer: conversation.source_delivery.inspect(source, consumer).state.value
        for consumer in (ConversationSourceConsumer.MEMORY, ConversationSourceConsumer.BEHAVIOR_SESSION)
    }
    assert states[ConversationSourceConsumer.BEHAVIOR_SESSION] == "committed"
    assert states[ConversationSourceConsumer.MEMORY] != "committed"
    assert given(world.claim(moment_of(7, 54, 30))) == ["weight"]
    assert claim_file(world, "weight").is_file()


def test_q19_12_an_unpicked_physical_judgement_does_not_hold_a_session_request(world: World) -> None:
    """Q19-12：物理一支有判断还没选上词，来取的是会话一支的请求：各支只看自己的，照常交出。"""

    awaiting(world)
    world.deliver(
        delivery(
            "d-run",
            [("跑步", at(DAY, 7, 30), at(DAY, 7, 40))],
            [(at(DAY, 7, 20), at(DAY, 7, 50), "observing")],
            sent_at=at(DAY, 7, 50),
        ),
        now=at(DAY, 7, 50),
    )
    assert unsealed_bridge(world.physical).unclassified() > 0
    assert given(world.claim(moment_of(7, 54, 30))) == ["weight"]


# ── 不变：已经交到宿主手里的 ──────────────────────────────────────────────────


def test_q19_15_taking_again_within_the_lease_is_not_a_new_handout(world: World) -> None:
    """Q19-15（已被取走）：同一会话租约内再取，这期间他做了：还给它，``claimed_at`` 不变。"""

    awaiting(world)
    (first,) = world.claim(at(DAY, 7, 50))
    did(world, "记体重", at(DAY, 7, 51), arrived=at(DAY, 7, 53))
    (again,) = world.claim(at(DAY, 7, 55))
    assert (again.request_id, again.claimed_at) == ("weight", first.claimed_at)


def test_q19_16_renewing_an_expired_lease_is_not_a_new_handout(world: World) -> None:
    """Q19-16（已被取走）：同一会话租约过期后再取（这期间他做了）：续租，``claimed_at`` 不变、租约往后延。"""

    awaiting(world)
    (first,) = world.claim(at(DAY, 7, 50))
    did(world, "记体重", at(DAY, 7, 51), arrived=at(DAY, 7, 53))
    (again,) = world.claim(at(DAY, 8, 25))
    assert (again.request_id, again.claimed_at) == ("weight", first.claimed_at)
    assert again.lease_until > first.lease_until


def test_q19_17_done_after_it_was_taken_is_settled_when_the_question_is_seen(world: World) -> None:
    """Q19-17（已被取走，裁定 65 第 2 条）：取走后他做了，下一拍「已发生」，agent 照问：不写「说了」、取走记录删掉、``already_done=1``。"""

    awaiting(world)
    (first,) = world.claim(at(DAY, 7, 50))
    did(world, "记体重", at(DAY, 7, 51), arrived=at(DAY, 7, 53))
    tick(world, at(DAY, 8))
    assert world.reasons("session", DAY)["weight"] == ["等时窗末", "待取", "已发生"]
    tool = ConversationAdapterRegistry.with_builtins().question_tools()[0]
    arguments = {"questions": [{tool.title_key: first.text, "options": [{"label": label} for label in OPTIONS]}]}
    report = outlet(world).observe(
        "s1",
        (
            HostMessage(
                0,
                at(
                    DAY,
                    7,
                    50,
                ).replace(second=5),
                "prompt",
                "帮我看看这个报错",
            ),
            HostMessage(
                1,
                at(DAY, 8, 2),
                "tool_call",
                json.dumps(arguments, ensure_ascii=False),
                tool_name=tool.tool,
                tool_call_id="call-1",
            ),
        ),
        host=HOST,
    )
    assert report.already_done == 1 and report.spoken == 0
    assert spoken_on_ledger(world) == []
    assert not claim_file(world, "weight").exists()


# ── 不变：不能开口的、前因不成立的，取的这一刻不多管 ──────────────────────────


def test_q19_19_a_basis_that_went_back_to_candidate_is_withdrawn_and_its_member_takes_over(world: World) -> None:
    """Q19-19（不能开口）：E 换代后依据变回候选、还没被取走：下一拍「依据失效」撤下，被它吸收的成员 M 到自己时窗末照常接手。"""

    event_awaiting(world)
    world.expect("session", "member", WEIGHT, window=(at(DAY, 10), at(DAY, 10, 20)), deadline=at(DAY, 20))
    tick(world, at(DAY, 7, 30))
    tick(world, at(DAY, 7, 45))
    assert world.reasons("session", DAY)["weight"] == ["等时窗末", "待取"]
    assert world.reasons("session", DAY)["member"] == ["单元已选"]
    write_table(world, DAY)  # 换代之后这条关系不在表上了
    ticks(world, at(DAY, 8), at(DAY, 10, 30))
    reasons = world.reasons("session", DAY)
    assert reasons["weight"] == ["等时窗末", "待取", "依据失效"]
    assert reasons["member"] == ["单元已选", "待取"]
    assert "weight" not in listed(world) and "member" in listed(world)


def test_q19_20_a_basis_held_back_until_its_deadline_does_not_block_its_member(world: World) -> None:
    """Q19-20（不能开口）：成员 M 被依据 E 吸收，E 因为有一份会话源一直没判完而取不走，到最晚时限记「宿主没取」；M 到自己的时窗末
    照常接手、记「待取」。这份源判完之后 M 取得走。"""

    world.use("session", WEIGHT)
    earn_together(world, WEIGHT, account_key(WEIGHT, "日子", DAILY))
    world.expect("session", "basis", WEIGHT, window=(at(DAY, 7, 25), at(DAY, 7, 40)), deadline=at(DAY, 9))
    world.expect("session", "held", WEIGHT, window=(at(DAY, 9, 10), at(DAY, 9, 40)), deadline=at(DAY, 20))
    assert world.claim(at(DAY, 7), session="probe") == ()
    tick(world, at(DAY, 7, 30))
    tick(world, at(DAY, 7, 45))
    source = handed_in(world, moment_of(7, 50), "帮我量一下血压")
    world.claim(at(DAY, 8))  # 他这时候敲了一句话：这一次拿不拿得到这里不断言（另见 Q19-8）
    ticks(world, at(DAY, 8), at(DAY, 9, 45))
    reasons = world.reasons("session", DAY)
    assert reasons["basis"][-1] == "宿主没取" and "待取" in reasons["basis"]
    assert reasons["held"][-1] == "待取"
    judged(world, source, "记血压", at(DAY, 9, 50))
    assert "held" in given(world.claim(at(DAY, 9, 55), session="s2"))


def test_q19_21_an_antecedent_that_no_longer_holds_is_not_rechecked_at_the_claim(world: World) -> None:
    """Q19-21（前因不成立）：事件预期 E 记了「待取」之后，它的前因那一行封口时归成了别的类（前因此刻不成立），这件事也没做：
    取的这一刻只判做没做，照常交出。"""

    event_awaiting(world)
    tick(world, at(DAY, 7, 30))
    tick(world, at(DAY, 7, 45))
    assert world.reasons("session", DAY)["weight"] == ["等时窗末", "待取"]
    world.set_clock(at(DAY, 7, 48))
    publish(world.session.tree, DAY, "记血压", 7, 0, kind=PRESSURE)  # 07:00 那一行封口后是记血压，不是写周报
    worker = world.runtime.components.slot_worker
    assert worker is not None and worker.branches[0].tick is not None
    recent = asyncio.run(worker.branches[0].tick.forming.read(at(DAY, 7, 48)))
    item = next(found for found in world.ledger("session").expectations_on(DAY) if found.expectation_id == "weight")
    rows = tuple(AntecedentRow(row.started_at, row.ended_at, row.kind_token) for row in recent.seen)
    assert not antecedent_holds_when_delivering(item, relation(), rows_on=lambda _day: rows)
    assert given(world.claim(at(DAY, 7, 50))) == ["weight"]


def test_q19_22_held_back_until_the_deadline_ends_as_never_claimed(world: World) -> None:
    """Q19-22：E 一直被拦到最晚时限：记「宿主没取」，结算按没说过。"""

    awaiting(world, deadline=at(DAY, 9))
    handed_in(world, moment_of(7, 50), "帮我看一下")
    world.claim(at(DAY, 8))  # 他这时候敲了一句话：这一次拿不拿得到这里不断言（另见 Q19-8）
    ticks(world, at(DAY, 8), at(DAY, 9, 15))
    assert world.reasons("session", DAY)["weight"] == ["等时窗末", "待取", "宿主没取"]
    assert spoken_on_ledger(world) == []
    assert Asked.of(world.ledger("session").deliveries_on(DAY), "weight") == Asked(None)


def test_q19_23_a_host_without_a_question_tool_gets_nothing_and_does_not_count_as_alive(world: World) -> None:
    """Q19-23：没有提问工具的宿主来取：空，不记活性，也不写取走记录。"""

    awaiting(world)
    stranger = "host-without-question-tool"
    world.set_clock(at(DAY, 7, 50))
    claimed = asyncio.run(world.runtime.claim_reminders(host=stranger, session="s9", at=at(DAY, 7, 50), limit=3))
    assert claimed == ()
    assert stranger not in outlet(world).store.hosts()
    assert not claim_file(world, "weight").exists()


# ── Q19-24、Q19-25：取的中途读材料出错；永远选不上词的判断 ────────────────────


def test_q19_24_a_claim_that_fails_halfway_leaves_nothing_behind(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """Q19-24：会话一支 weight（时限更早）、物理一支 pill 都在可取清单上；第一个会话来取时，判定物理一支的材料抛 ``OSError``：
    这一次取失败，什么都不留下——weight 没有取走记录、放回次数不变，第二个会话来取能拿到 weight。"""

    ready(world, "session", "weight", WEIGHT, deadline=at(DAY, 20))
    ready(world, "physical", "pill", MEDICINE)
    assert world.claim(at(DAY, 7), session="probe") == ()
    tick(world, at(DAY, 7, 30))
    tick(world, at(DAY, 7, 45))
    assert {"weight", "pill"} <= set(listed(world))

    def broken() -> tuple[dict[str, Any], ...]:
        raise OSError("judgement store is unreadable")

    with monkeypatch.context() as patch:
        patch.setattr(world.physical.judgements, "list", broken)
        with pytest.raises(OSError, match="unreadable"):
            world.claim(at(DAY, 7, 50), session="s1")
    assert not claim_file(world, "weight").exists()
    assert "weight" not in outlet(world).trails()
    assert "weight" in given(world.claim(at(DAY, 7, 52), session="s2"))


def test_q19_25_a_physical_judgement_that_can_never_be_picked_does_not_hold_the_request(world: World) -> None:
    """Q19-25：物理一支有一条判断没有可选词的材料（名字是空白，``record_material`` 给 None，永远选不上词）：它不是"还没判完"，
    来取物理一支的请求不挡，照常交出。"""

    awaiting(world, lane="physical", identity="pill", kind=MEDICINE)
    world.deliver(
        delivery(
            "d-blank",
            [("跑步", at(DAY, 7, 30), at(DAY, 7, 40))],
            [(at(DAY, 7, 20), at(DAY, 7, 50), "observing")],
            sent_at=at(DAY, 7, 50),
        ),
        now=at(DAY, 7, 50),
    )
    store = world.physical.judgements
    (raw,) = store.list()
    store.discard(str(raw["judgement_id"]))
    store.put_payload({**raw, "behavior": "   "})
    runner = world.physical.reduction_runner
    (record,) = pending_judgements(store, runner.ledger).occurrences
    picker = runner.kinds.picker
    assert picker is not None
    assert record.kind_token is None
    assert record_material(record, max_steps=picker.config.prompt_max_steps) is None
    assert given(world.claim(moment_of(7, 54, 30))) == ["pill"]
