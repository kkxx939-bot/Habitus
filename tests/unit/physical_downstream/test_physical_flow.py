"""第二步丙二：物理一支走通（场景 C8–C17；共享说明 3.3）。

物理一支自己的换代、语义树、机会表、日子预期、递送、结算；未封口的判断由模型队列的 ``unsealed`` 认欠账；上游送来的"没在看"
落「未观测」。提醒出口只有一个（``runtime.claim_reminders``），两支的预期都从它说。预期、递送、回话、结算按目录读写这一支的
账本（``<foresight_root>/physical/``）。改动前物理一支没有下游，这几条在"物理一支做了"那一处失败。
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from habitus.model_client.errors import ModelContentSafetyError
from habitus.runtime.foresight_opportunities import OpportunityStore
from tests.unit.physical_downstream.site import (
    MEDICINE,
    WEEKLY,
    Named,
    World,
    at,
    day_delivery,
    delivery,
    texts_under,
)
from tests.unit.word_pick.site import description

D13 = date(2026, 4, 13)
MORNING = [("吃药", 8, 0, 10)]


def _session(world: World) -> None:
    world.router.pick_default = Named("写周报")


def test_c8_a_physical_word_picked_on_two_days_is_in_use_next_generation_with_a_concept(tmp_path: Path) -> None:
    world = World(tmp_path, start=at(D13, 0))
    try:
        world.physical_days(D13, [MORNING, MORNING, []])
        assert MEDICINE in world.in_use("physical")
        assert (world.scene("physical") / "concepts" / f"{MEDICINE}.md").is_file()
    finally:
        world.close()


def test_c9_a_physical_daily_habit_gets_into_the_opportunity_table_and_a_day_expectation(tmp_path: Path) -> None:
    world = World(tmp_path, start=at(D13, 0))
    try:
        world.physical_days(D13, [MORNING] * 5)
        today = D13 + timedelta(days=5)
        table = OpportunityStore(world.foresight("physical")).latest("physical")
        assert table is not None, "the physical branch has no opportunity table"
        assert MEDICINE in texts_under(world.foresight("physical") / "opportunities")
        kinds = {item.kind_token for item in world.ledger("physical").expectations_on(today)}
        assert MEDICINE in kinds
        assert not any(item.kind_token == MEDICINE for item in world.ledger("session").expectations_on(today))
    finally:
        world.close()


def _due(world: World, lane: str, identity: str, kind: str, day: date) -> None:
    """这一支一条今天 09:00–09:30 的日子预期，账到提醒档、在用。"""

    world.use(lane, kind)
    world.earn(lane, kind, at(day, 0))
    world.expect(lane, identity, kind, window=(at(day, 9), at(day, 9, 30)), deadline=at(day, 23, 59))


def test_c10_a_physical_reminder_comes_out_of_the_one_outlet_and_is_booked_on_the_physical_ledger(
    tmp_path: Path,
) -> None:
    day = D13 + timedelta(days=1)
    world = World(tmp_path, start=at(day, 0))
    try:
        _due(world, "physical", "pill", MEDICINE, day)
        assert world.claim(at(day, 8), session="probe") == ()  # 宿主来过（出口活着）
        world.slots_between(at(day, 9), at(day, 10), reduce=False)
        claimed = world.claim(at(day, 10, 5))
        assert [item.request_id for item in claimed] == ["pill"]
        assert "待取" in world.reasons("physical", day).get("pill", [])
        assert world.reasons("session", day) == {}
    finally:
        world.close()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "待主会话定（阶段 A 报告「要主会话定的事」第 1 条）：共享说明 3.3 写'同一刻最多一条请求跨两支也成立'，"
        "现有口径（foresight_delivery 模块文档）是同一个行为单元之内最多一条；实现按单元口径、两支之间不仲裁"
        "（runtime/foresight_outlet_ledgers.py:8）。定了 (b) 就去掉 xfail，定了 (a) 就删掉这一条、留 C11a。"
    ),
)
def test_c11_a_session_and_a_physical_reminder_due_at_the_same_moment_put_one_request_at_the_outlet(
    tmp_path: Path,
) -> None:
    day = D13 + timedelta(days=1)
    world = World(tmp_path, start=at(day, 0))
    try:
        _due(world, "session", "weekly", WEEKLY, day)
        _due(world, "physical", "pill", MEDICINE, day)
        assert world.claim(at(day, 8), session="probe") == ()
        world.slots_between(at(day, 9), at(day, 10), reduce=False)
        claimed = world.claim(at(day, 10, 5))
        assert len(claimed) == 1, [item.request_id for item in claimed]
        waiting = [
            identity
            for lane in ("session", "physical")
            for identity, reasons in world.reasons(lane, day).items()
            if "待取" in reasons
        ]
        assert len(waiting) == 1, waiting
        # 另一条照"同一刻最多一条"的口径处理：它那一支的账上也有这一刻的决定（不是没人管）
        assert world.reasons("session", day).get("weekly"), world.reasons("session", day)
        assert world.reasons("physical", day).get("pill"), world.reasons("physical", day)
    finally:
        world.close()


def test_c11a_under_the_unit_rule_a_session_and_a_physical_reminder_each_get_their_own_request(tmp_path: Path) -> None:
    """现有口径（单元 = 同一个行为这一刻的全部输入）下的 C11：两支各是一个单元，各出一条、各记在自己那一支的账上。"""

    day = D13 + timedelta(days=1)
    world = World(tmp_path, start=at(day, 0))
    try:
        _due(world, "session", "weekly", WEEKLY, day)
        _due(world, "physical", "pill", MEDICINE, day)
        assert world.claim(at(day, 8), session="probe") == ()
        world.slots_between(at(day, 9), at(day, 10), reduce=False)
        claimed = world.claim(at(day, 10, 5))
        assert sorted(item.request_id for item in claimed) == ["pill", "weekly"]
        assert "待取" in world.reasons("session", day)["weekly"]
        assert "待取" in world.reasons("physical", day)["pill"]
        assert "pill" not in world.reasons("session", day) and "weekly" not in world.reasons("physical", day)
    finally:
        world.close()


def test_c12_his_reply_in_the_conversation_to_a_physical_reminder_lands_on_the_physical_ledger(
    tmp_path: Path,
) -> None:
    day = D13 + timedelta(days=1)
    world = World(tmp_path, start=at(day, 0))
    try:
        world.use("physical", MEDICINE)
        item = world.expect("physical", "pill", MEDICINE, window=(at(day, 8), at(day, 8, 30)), deadline=at(day, 12))
        world.spoken("physical", item, at(day, 8, 30))
        world.router.describes.append({**description("回复提醒"), "responses": [{"reminder": 1, "polarity": "已做"}]})
        world.set_clock(at(day, 8, 42))
        world.turn("conv", at(day, 8, 40), "已经吃了")
        responses = world.ledger("physical").responses_on(day)
        assert [(item.expectation_id, item.polarity) for item in responses] == [("pill", "已做")]
        assert world.ledger("session").responses_on(day) == ()
    finally:
        world.close()


def _spoken_pill(world: World, day: date) -> None:
    world.use("physical", MEDICINE)
    item = world.expect("physical", "pill", MEDICINE, window=(at(day, 7, 30), at(day, 8)), deadline=at(day, 10))
    world.spoken("physical", item, at(day, 8))


def test_c13_a_physical_record_after_the_physical_reminder_settles_it_as_done(tmp_path: Path) -> None:
    world = World(tmp_path, start=at(D13, 0))
    try:
        world.physical_days(D13, [MORNING])
        day = D13 + timedelta(days=1)
        _spoken_pill(world, day)
        world.physical_days(day, [[("吃药", 8, 20, 5)], []])
        settled = world.ledger("physical").settlements_on(day)
        assert [(item.expectation_id, item.outcome) for item in settled] == [("pill", "提醒后发生")]
        assert settled[0].occurrence_uri is not None
    finally:
        world.close()


def test_c14_a_session_record_does_not_settle_a_physical_expectation(tmp_path: Path) -> None:
    world = World(tmp_path, start=at(D13, 0))
    try:
        _session(world)
        world.physical_days(D13, [MORNING])
        day = D13 + timedelta(days=1)
        _spoken_pill(world, day)
        # 阶段 C 改的造法：物理那条说出去的提醒开着，这一轮"描述这一轮"的输出形状里必有 ``responses``（C12 的那条口），
        # 替身默认的描述没有它、不合格，这一轮落成 ERROR、不进树。这里排一份不回应任何提醒的描述（他没回提醒，只是说了一句）。
        world.router.describes.append({**description("吃药"), "responses": []})
        world.set_clock(at(day, 8, 22))
        world.turn("conv", at(day, 8, 20), "我刚吃了药")  # 会话一支的一条记录（落在说中的范围里）
        world.physical_days(day, [[], []])
        assert world.tokens("session"), "the session turn did not land"
        settled = world.ledger("physical").settlements_on(day)
        assert [item.expectation_id for item in settled] == ["pill"]
        assert settled[0].occurrence_uri is None
        assert settled[0].outcome not in ("提醒后发生", "准时", "早到", "晚到")
    finally:
        world.close()


def _unsealed_today(world: World) -> date:
    """两天物理记录、两代之后，今天 10:00–10:10 一条还没封口的物理判断（10:15 送到）。返回今天。

    今天这一代已经做完：这条判断归预测拍自己选（``runtime.model_queue`` 模块文档 ``unsealed`` 一行与第 19 行）。"""

    world.physical_days(D13, [MORNING, MORNING])
    today = D13 + timedelta(days=2)
    assert today.isoformat() in world.marks("physical"), "the physical branch has no generation today"
    world.deliver(
        delivery("late", [("吃药", at(today, 10), at(today, 10, 10))],
                 [(at(today, 0), at(today, 10, 15), "observing")], sent_at=at(today, 10, 15)),
        now=at(today, 10, 15),
    )  # fmt: skip
    return today


def _unsealed_before_generation(world: World) -> date:
    """今天这一代还没做、触发条件里除完成标记之外都满足的那一刻（今天 01:00 归约刚把昨天定了稿、预测 worker 还没跑这一槽），
    一条今天 00:40–00:50 还没封口的物理判断（01:00 送到）。返回今天。

    阶段 C 改的造法：原来的 ``_unsealed_today`` 在今天这一代**之后**送判断，而模型队列的 ``unsealed`` 一类按设计只在今天这一代
    还没做时认欠账（``runtime/model_queue.py:10,19``、``runtime/model_queue_work.py:76-79``：``not self._generated(night)``），
    之后归预测拍自己选；这里换成换代之前的那一刻。"""

    world.physical_days(D13, [MORNING])
    yesterday = D13 + timedelta(days=1)
    today = yesterday + timedelta(days=1)
    world.deliver(day_delivery(yesterday, MORNING), now=at(today, 0, 5))
    world.slots_between(at(today, 0), at(today, 0, 45))
    world.set_clock(at(today, 1))
    world.deliver(
        delivery("late", [("吃药", at(today, 0, 40), at(today, 0, 50))],
                 [(at(today, 0), at(today, 1), "observing")], sent_at=at(today, 1)),
    )  # fmt: skip
    world.reduce()
    pipeline = world.runtime.components.pipeline("physical")
    assert pipeline is not None
    assert pipeline.generation.generation.sealed_day() == today, pipeline.generation.generation.ready()
    assert today.isoformat() not in world.marks("physical")
    return today


def _bridge_rows(world: World, today: date) -> list[str | None]:
    """物理一支那座未封口的桥此刻给场景的行（只读缓存，不问模型）：每行的词。"""

    pipeline = world.runtime.components.pipeline("physical")
    assert pipeline is not None
    bridge = pipeline.queue.daily.bridge
    assert bridge is not None
    return [row.kind_token for row in bridge.rows(since=at(today, 0), until=at(today, 23))]


def test_c15_an_unsealed_physical_judgement_is_owed_to_the_queue_picked_once_and_read_from_the_cache(
    tmp_path: Path,
) -> None:
    world = World(tmp_path, start=at(D13, 0))
    try:
        today = _unsealed_before_generation(world)
        before = len(world.router.physical_calls())
        report = world.drain(at(today, 1))
        assert [item.category for item in report["physical"].done] == ["unsealed"], report
        assert len(world.router.physical_calls()) == before + 1
        assert _bridge_rows(world, today) == [MEDICINE]  # 此刻场景里这一行带着词
        world.slot(at(today, 1), reduce=False)  # 换代（日子预期只读缓存）+ 预测拍（读缓存）
        assert today.isoformat() in world.marks("physical")
        world.slots_between(at(today, 1, 15), at(today, 1, 30))
        world.drain(at(today, 1, 35))
        assert len(world.router.physical_calls()) == before + 1
    finally:
        world.close()


def test_c15b_after_todays_generation_the_tick_picks_an_unsealed_physical_judgement_once(tmp_path: Path) -> None:
    """今天这一代做完之后的那一条：队列不再认它，预测拍选一次、之后读缓存。"""

    world = World(tmp_path, start=at(D13, 0))
    try:
        today = _unsealed_today(world)
        before = len(world.router.physical_calls())
        report = world.drain(at(today, 10, 20))
        assert report["physical"].owed["unsealed"] == 0, report
        assert len(world.router.physical_calls()) == before
        world.slots_between(at(today, 10, 30), at(today, 10, 45))  # 视界未过，归约不封它
        assert len(world.router.physical_calls()) == before + 1
        assert _bridge_rows(world, today) == [MEDICINE]
        world.drain(at(today, 10, 50))
        assert len(world.router.physical_calls()) == before + 1
    finally:
        world.close()


def test_c16_an_unsealed_physical_judgement_refused_by_content_safety_is_not_asked_again(tmp_path: Path) -> None:
    world = World(tmp_path, start=at(D13, 0))
    try:
        today = _unsealed_today(world)
        before = len(world.router.physical_calls())
        world.router.physical.append(ModelContentSafetyError("blocked"))
        world.drain(at(today, 10, 20))
        world.slots_between(at(today, 10, 30), at(today, 10, 45))
        world.drain(at(today, 10, 50))
        world.drain(at(today, 11, 5))
        assert len(world.router.physical_calls()) == before + 1
        assert _bridge_rows(world, today) == [None]  # 按错误处理：这一行不带词
    finally:
        world.close()


def test_c16b_refused_in_the_queue_before_todays_generation_is_not_asked_again(tmp_path: Path) -> None:
    world = World(tmp_path, start=at(D13, 0))
    try:
        today = _unsealed_before_generation(world)
        before = len(world.router.physical_calls())
        world.router.physical.append(ModelContentSafetyError("blocked"))
        report = world.drain(at(today, 1))
        assert report["physical"].owed["unsealed"] == 0, report  # 不算欠
        world.drain(at(today, 1, 5))
        world.slot(at(today, 1), reduce=False)
        world.slots_between(at(today, 1, 15), at(today, 1, 30))
        assert len(world.router.physical_calls()) == before + 1
        assert _bridge_rows(world, today) == [None]
        assert today.isoformat() in world.marks("physical")
    finally:
        world.close()


def test_c17_an_upstream_not_watching_day_is_unobserved_and_the_physical_expectation_is_not_a_miss(
    tmp_path: Path,
) -> None:
    world = World(tmp_path, start=at(D13, 0))
    try:
        world.physical_days(D13, [MORNING])
        day = D13 + timedelta(days=1)
        world.use("physical", MEDICINE)
        world.expect("physical", "pill", MEDICINE, window=(at(day, 8), at(day, 8, 30)), deadline=at(day, 10))
        world.deliver(day_delivery(day, [], state="not_worn"), now=at(day + timedelta(days=1), 0, 5))
        world.night(day + timedelta(days=1))
        world.physical_days(day + timedelta(days=1), [MORNING])
        assert any(day.isoformat() in str(gap.get("started_at")) for gap in world.gaps("physical"))
        facts = [
            fact
            for account in world.ledger("physical").accounts()
            for fact in world.ledger("physical").facts_of(account)
            if fact.expectation_id == "pill"
        ]
        assert facts, "the physical expectation was never settled"
        assert all(fact.strength == 0.0 for fact in facts), [(f.kind, f.strength) for f in facts]
    finally:
        world.close()
