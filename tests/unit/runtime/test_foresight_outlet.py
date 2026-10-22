"""提醒出口（方案 32；裁定 64）：取 / 租约 / 幂等重取 / 过期 / 放回计数 / 时限后取不到；认提问（问了且配上、问了配不上、没问、
问在时限后）与作答（三个固定选项 / 自己填 / 没答 / Desktop 晚来的作答）；活性；一条预期最多一条「说了」。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from habitus.foresight import ForesightError
from habitus.foresight.ledger import Delivery, Expectation
from habitus.foundation.observability import ObservationEvent
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_outlet import (
    OPTIONS,
    HostMessage,
    OpenRequest,
    ReminderOutlet,
    answer_in_result,
    fixed_polarity_in,
    parse_reply,
)
from habitus.runtime.foresight_outlet_store import OutletStore

CST = timezone(timedelta(hours=8))
FORMED = datetime(2026, 10, 22, 14, 0, tzinfo=CST)
ASKED = FORMED + timedelta(minutes=100)  # 15:40 时窗末记「待取」
TEXT = "周四下午你一般会写周报，这周的还没写。"
SESSION = "codex-thread-1"


class Recorder:
    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)


def expectation(identity: str = "e1", *, kind: str = "s-k0002", deadline_hours: int = 4) -> Expectation:
    return Expectation(
        expectation_id=identity,
        kind_token=kind,
        day=FORMED.date(),
        formed_at=FORMED,
        source="日子",
        trigger="rel",
        trigger_instance=f"behavior://{identity}.md",
        window=(FORMED, FORMED + timedelta(hours=2)),
        deadline=FORMED + timedelta(hours=deadline_hours),
        basis=(),
        opportunities=3,
        hits=3,
        observed_from=FORMED,
    )


def request(item: Expectation, text: str = TEXT) -> OpenRequest:
    return OpenRequest(item.expectation_id, item.kind_token, text, item.deadline, item.formed_at, 1, item.day)


def outlet(tmp_path, **overrides):  # type: ignore[no-untyped-def]
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    recorder = Recorder()
    values = dict(
        liveness=timedelta(hours=48),
        lease=timedelta(minutes=30),
        max_attempts=3,
        claim_limit=3,
        similarity=0.6,
        kind_label=lambda token: {"s-k0002": "写周报"}.get(token),
        observer=recorder,
    )
    values.update(overrides)
    return ReminderOutlet(OutletStore(tmp_path / "ledger" / "outlet"), ledger, **values), ledger, recorder


def awaiting(ledger: ForesightLedgerStore, item: Expectation) -> None:
    ledger.record_expectation(item)
    ledger.record_delivery(item.day, Delivery(item.expectation_id, ASKED, 1, False, "待取", 1.0, TEXT))


def claude_question(title: str, call_id: str = "toolu_1", at: datetime = ASKED + timedelta(minutes=12)) -> HostMessage:
    arguments = {
        "questions": [
            {"question": title, "header": "提醒", "multiSelect": False, "options": [{"label": o} for o in OPTIONS]}
        ]
    }
    return HostMessage(
        3, at, "tool_call", json.dumps(arguments, ensure_ascii=False), tool_name="AskUserQuestion", tool_call_id=call_id
    )


def claude_answer(text: str, call_id: str = "toolu_1", at: datetime = ASKED + timedelta(minutes=13)) -> HostMessage:
    return HostMessage(4, at, "tool_result", text, tool_name="AskUserQuestion", tool_call_id=call_id)


def turn(*middle: HostMessage, at: datetime = ASKED + timedelta(minutes=11)) -> tuple[HostMessage, ...]:
    """他敲了一句话（hook 这时取走）、中间几条、助手最后说了话。"""

    return (
        HostMessage(1, at, "prompt", "帮我看看这个报错"),
        *middle,
        HostMessage(9, at + timedelta(minutes=5), "completion", "看完了，是依赖版本不对。"),
    )


# ── 取 ────────────────────────────────────────────────────────────────────────


def test_claims_hand_out_open_requests_by_deadline_with_a_lease_and_keep_the_host_alive(tmp_path) -> None:
    run, ledger, _ = outlet(tmp_path)
    first, second, third, fourth = (
        expectation(f"e{i}", kind=f"s-k000{i}", deadline_hours=h) for i, h in ((1, 4), (2, 3), (3, 5), (4, 6))
    )
    for item in (first, second, third, fourth):
        awaiting(ledger, item)
    now = ASKED + timedelta(minutes=10)
    assert not run.alive(now)
    assert run.publish([request(first), request(second), request(third), request(fourth)], now=now) == 4
    given = run.claim("codex", SESSION, now, 3)
    assert [item.request_id for item in given] == ["e2", "e1", "e3"]  # 按最晚时限由近到远，最多 3 条
    assert (
        given[0].options == OPTIONS and given[0].kind == "写周报" and given[1].kind == "s-k0001"
    )  # 词表没有类名的给编号
    assert given[0].lease_until == now + timedelta(minutes=30) and given[0].claimed_at == now
    assert run.alive(now) and run.alive(now + timedelta(hours=47)) and not run.alive(now + timedelta(hours=49))
    # 同一会话租约内重取：拿回自己手里那几条（claimed_at 不变，宿主据此不重注入），不另发第四条
    again = run.claim("codex", SESSION, now + timedelta(minutes=5), 3)
    assert [(item.request_id, item.claimed_at) for item in again] == [(i.request_id, now) for i in given]
    # 别的会话这时只拿得到没被租着的那一条
    other = run.claim("claude-code", "claude-s2", now + timedelta(minutes=5), 3)
    assert [item.request_id for item in other] == ["e4"]
    # 没有提问工具的宿主问不了他：不给，也不点亮活性（方案 36 第四节）
    assert run.claim("pi", "pi-s1", now + timedelta(minutes=5), 3) == ()
    with pytest.raises(ForesightError, match="limit"):
        run.claim("codex", SESSION, now, 4)
    with pytest.raises(ForesightError, match="timezone"):
        run.claim("codex", SESSION, now.replace(tzinfo=None), 1)


def test_an_expired_lease_frees_the_request_for_another_session_without_counting_an_attempt(tmp_path) -> None:
    run, ledger, _ = outlet(tmp_path)
    item = expectation()
    awaiting(ledger, item)
    now = ASKED + timedelta(minutes=10)
    run.publish([request(item)], now=now)
    assert run.claim("codex", SESSION, now, 3)
    assert run.claim("codex", "codex-thread-2", now + timedelta(minutes=29), 3) == ()
    later = run.claim("codex", "codex-thread-2", now + timedelta(minutes=31), 3)
    assert [item.request_id for item in later] == ["e1"]
    # 过了最晚时限谁都取不到，取走记录也清掉
    assert run.claim("codex", "codex-thread-3", item.deadline, 3) == ()
    assert run.store.claims() == {}


def test_a_request_put_back_three_times_is_not_handed_out_again(tmp_path) -> None:
    run, ledger, _ = outlet(tmp_path)
    item = expectation()
    awaiting(ledger, item)
    now = ASKED + timedelta(minutes=10)
    run.publish([request(item)], now=now)
    for attempt in range(3):
        at = now + timedelta(minutes=attempt)
        assert run.claim("codex", SESSION, at, 3), attempt
        # 取走了、他那一轮过了、agent 没问：放回
        report = run.observe(SESSION, turn(at=at + timedelta(seconds=5)), host="codex")
        assert (report.released, report.spoken) == (1, 0)
    assert run.claim("codex", SESSION, now + timedelta(minutes=5), 3) == ()
    assert run.store.read_claim("e1")["attempts"] == 3


# ── 认提问与作答 ──────────────────────────────────────────────────────────────


def test_a_question_carrying_the_reminder_is_recorded_as_spoken_and_a_fixed_option_as_a_response(tmp_path) -> None:
    run, ledger, recorder = outlet(tmp_path)
    item = expectation()
    awaiting(ledger, item)
    now = ASKED + timedelta(minutes=10)
    run.publish([request(item)], now=now)
    run.claim("claude-code", "claude-s1", now, 3)
    asked_at = now + timedelta(minutes=2)
    answered_at = asked_at + timedelta(seconds=40)
    messages = turn(
        claude_question(TEXT, at=asked_at),
        claude_answer(f'User has answered your questions: "{TEXT}"="这次不用". You can now continue.', at=answered_at),
        at=now + timedelta(seconds=3),
    )
    report = run.observe("claude-s1", messages, host="claude-code")
    assert (report.spoken, report.responded, report.released, report.awaiting_reply) == (1, 1, 0, 0)
    spoken = [d for d in ledger.deliveries_on(item.day) if d.spoken]
    assert len(spoken) == 1
    assert (spoken[0].decided_at, spoken[0].text, spoken[0].via, spoken[0].turn) == (
        asked_at,
        TEXT,
        "claude-code",
        "toolu_1",
    )
    (response,) = ledger.responses_on(item.day)
    assert (response.polarity, response.responded_at, response.turn) == ("负", answered_at, "toolu_1")
    assert run.store.read_claim("e1")["spoken"] is True and run.store.read_open() == ()  # 认出之后别的会话再也取不到
    assert run.claim("codex", "codex-other", answered_at, 3) == ()
    # 同一份转写再来一遍（compact 前后、重放）：不再写第二条
    again = run.observe("claude-s1", messages, host="claude-code")
    assert again.spoken == 0 and len([d for d in ledger.deliveries_on(item.day) if d.spoken]) == 1
    assert any(event.operation == "outlet_observe" for event in recorder.events)


def test_a_free_text_answer_is_left_to_the_session_lane_model(tmp_path) -> None:
    run, ledger, _ = outlet(tmp_path)
    item = expectation()
    awaiting(ledger, item)
    now = ASKED + timedelta(minutes=10)
    run.publish([request(item)], now=now)
    run.claim("claude-code", "claude-s1", now, 3)
    report = run.observe(
        "claude-s1",
        turn(
            claude_question(TEXT),
            claude_answer(f'User has answered your questions: "{TEXT}"="下周再弄吧". You can now continue.'),
        ),
        host="claude-code",
    )
    assert (report.spoken, report.responded) == (1, 0)
    assert ledger.responses_on(item.day) == ()  # 桥把它列为开着的提醒、模型判
    # 他没答就关了会话：「说了」照记，没有回话
    other = expectation("e2", kind="s-k0003")
    awaiting(ledger, other)
    run.publish([request(other, "周四了，账还没对。")], now=now)
    run.claim("claude-code", "claude-s2", now, 3)
    report = run.observe(
        "claude-s2", turn(claude_question("周四了，账还没对。", call_id="toolu_9")), host="claude-code"
    )
    assert (report.spoken, report.responded, report.awaiting_reply) == (1, 0, 1)


def test_a_question_about_something_else_does_not_count_and_the_request_goes_back(tmp_path) -> None:
    run, ledger, _ = outlet(tmp_path)
    item = expectation()
    awaiting(ledger, item)
    now = ASKED + timedelta(minutes=10)
    run.publish([request(item)], now=now)
    run.claim("claude-code", "claude-s1", now, 3)
    unrelated = HostMessage(
        3,
        now + timedelta(minutes=1),
        "tool_call",
        json.dumps(
            {"questions": [{"question": "用哪个数据库？", "options": [{"label": "Postgres"}, {"label": "SQLite"}]}]}
        ),
        tool_name="AskUserQuestion",
        tool_call_id="toolu_2",
    )
    report = run.observe("claude-s1", turn(unrelated), host="claude-code")
    assert (report.spoken, report.released) == (0, 1)
    assert not any(d.spoken for d in ledger.deliveries_on(item.day))
    # 题目被 agent 改写了、但选项是我们给的三个，只取走了这一条：直接配上
    run.claim("claude-code", "claude-s1", now + timedelta(minutes=10), 3)
    reworded = claude_question("要不要现在写周报？", call_id="toolu_3", at=now + timedelta(minutes=11))
    report = run.observe("claude-s1", turn(reworded, at=now + timedelta(minutes=10, seconds=2)), host="claude-code")
    assert report.spoken == 1
    (spoken,) = [d for d in ledger.deliveries_on(item.day) if d.spoken]
    assert spoken.text == "要不要现在写周报？"


def test_a_question_asked_after_the_deadline_is_not_spoken(tmp_path) -> None:
    run, ledger, _ = outlet(tmp_path)
    item = expectation()
    awaiting(ledger, item)
    now = item.deadline - timedelta(minutes=5)
    run.publish([request(item)], now=now)
    run.claim("claude-code", "claude-s1", now, 3)
    late = item.deadline + timedelta(minutes=1)
    report = run.observe(
        "claude-s1", turn(claude_question(TEXT, at=late), at=now + timedelta(seconds=1)), host="claude-code"
    )
    assert report.spoken == 0 and not any(d.spoken for d in ledger.deliveries_on(item.day))
    assert run.store.claims() == {}


def test_codex_desktop_answers_arrive_in_the_next_transcript_and_still_write_the_fixed_option(tmp_path) -> None:
    run, ledger, _ = outlet(tmp_path)
    item = expectation()
    awaiting(ledger, item)
    now = ASKED + timedelta(minutes=10)
    run.publish([request(item)], now=now)
    run.claim("codex", SESSION, now, 3)
    arguments = json.dumps({"questions": [{"title": TEXT, "options": list(OPTIONS)}]}, ensure_ascii=False)
    asked_at = now + timedelta(minutes=1)
    first = (
        HostMessage(1, now + timedelta(seconds=2), "prompt", "帮我看看这个报错"),
        HostMessage(2, asked_at, "tool_call", arguments, tool_name="request_user_input_async", tool_call_id="call_a1"),
        HostMessage(
            3,
            asked_at,
            "tool_result",
            '{"accepted":true}',
            tool_name="request_user_input_async",
            tool_call_id="call_a1",
        ),
        HostMessage(4, asked_at + timedelta(minutes=3), "completion", "报错看完了。"),
    )
    report = run.observe(SESSION, first, host="codex")
    assert (report.spoken, report.responded, report.awaiting_reply, report.released) == (1, 0, 1, 0)
    assert ledger.responses_on(item.day) == ()
    reply = {
        "questionItemId": json.dumps(["request_user_input_async", "call_a1", 0]),
        "question": TEXT,
        "answer": "现在就做",
    }
    answered_at = asked_at + timedelta(minutes=6)
    second = (
        HostMessage(
            5,
            answered_at,
            "prompt",
            f"{REPLY_TAG}\n{json.dumps([reply], ensure_ascii=False)}\n</send_user_message_question_reply>\n",
        ),
        HostMessage(6, answered_at + timedelta(seconds=30), "completion", "好，那先写周报。"),
    )
    report = run.observe(SESSION, second, host="codex")
    assert report.responded == 1 and run.store.asked() == {}
    (response,) = ledger.responses_on(item.day)
    assert (response.polarity, response.responded_at, response.turn) == ("正", answered_at, "call_a1")
    # 同一份作答再来：问过的记录已清，不再写；自己填的话也不写
    assert run.observe(SESSION, second, host="codex").responded == 0


def test_only_one_spoken_record_per_expectation_even_when_two_sessions_both_asked(tmp_path) -> None:
    run, ledger, _ = outlet(tmp_path)
    item = expectation()
    awaiting(ledger, item)
    now = ASKED + timedelta(minutes=10)
    run.publish([request(item)], now=now)
    run.claim("claude-code", "claude-s1", now, 3)
    later = now + timedelta(minutes=31)
    run.claim("claude-code", "claude-s2", later, 3)  # 第一个会话的租约过期了，第二个也取走了、也问了
    second = run.observe(
        "claude-s2", turn(claude_question(TEXT, "toolu_b", later + timedelta(minutes=1)), at=later), host="claude-code"
    )
    assert second.spoken == 1
    # 第一个会话的转写晚到：它也问过——认得出（历任持有者），但账上已有一条「说了」，不再写第二条
    first = run.observe(
        "claude-s1", turn(claude_question(TEXT, "toolu_a", now + timedelta(minutes=1)), at=now), host="claude-code"
    )
    assert (first.spoken, first.already_spoken) == (0, 1)
    assert len([d for d in ledger.deliveries_on(item.day) if d.spoken]) == 1


def test_the_outlet_checks_its_inputs(tmp_path) -> None:
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    store = OutletStore(tmp_path / "ledger" / "outlet")
    common = dict(
        liveness=timedelta(hours=1), lease=timedelta(minutes=1), max_attempts=1, claim_limit=1, similarity=0.5
    )
    with pytest.raises(TypeError):
        ReminderOutlet(object(), ledger, **common)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ReminderOutlet(store, ledger, **{**common, "lease": timedelta(0)})
    with pytest.raises(ValueError):
        ReminderOutlet(store, ledger, **{**common, "similarity": 0.0})
    with pytest.raises(ForesightError, match="role"):
        HostMessage(1, ASKED, "system", "x")
    with pytest.raises(ForesightError, match="request_id"):
        OpenRequest("", "s-k0002", TEXT, ASKED, ASKED, 1, ASKED.date())
    with pytest.raises(ForesightError, match="safe file name"):
        store.write_claim("../x", {})


def test_the_pure_readers_recognise_fixed_options_and_desktop_replies() -> None:
    assert fixed_polarity_in('User has answered your questions: "问题"="知道了". You can now continue.') == "中"
    assert fixed_polarity_in(json.dumps({"answers": {"q1": "现在就做"}})) == "正"
    assert fixed_polarity_in("这次不用") == "负"
    assert fixed_polarity_in("The user doesn't want to proceed with this tool use.") is None
    assert fixed_polarity_in('"问题"="知道了"；"问题2"="这次不用"') is None  # 两个选项同时出现：说不清
    assert fixed_polarity_in("下周再弄") is None
    replies = parse_reply(
        f'{REPLY_TAG}\n[{{"questionItemId":"[\\"request_user_input_async\\",\\"call_1\\",0]","question":"q","answer":"a"}}]\n</send_user_message_question_reply>\n'
    )
    assert replies == (("call_1", 0, "a"),)
    assert parse_reply("<send_user_message_question_reply>\nnot json\n</send_user_message_question_reply>") == ()
    assert parse_reply("随便一句话") == ()


# ── 五路评审 + 测试工程师确认过的九条 bug 的回归（2026-10-20） ──────────────────────────


def codex_question(
    titles: list[str], call_id: str, at: datetime, *, tool: str = "request_user_input_async"
) -> HostMessage:
    arguments = {"questions": [{"title": title, "options": list(OPTIONS)} for title in titles]}
    return HostMessage(
        3, at, "tool_call", json.dumps(arguments, ensure_ascii=False), tool_name=tool, tool_call_id=call_id
    )


def desktop_reply(items: list[tuple[str, int, str, str]], at: datetime, sequence: int = 7) -> HostMessage:
    """``(call_id, 题序, 题目, 作答)`` → Desktop 那条用户消息。"""

    body = [
        {
            "questionItemId": json.dumps(["request_user_input_async", call_id, index]),
            "question": title,
            "answer": answer,
        }
        for call_id, index, title, answer in items
    ]
    return HostMessage(
        sequence,
        at,
        "prompt",
        f"{REPLY_TAG}\n{json.dumps(body, ensure_ascii=False)}\n</send_user_message_question_reply>\n",
    )


def test_bug1_a_desktop_answer_in_the_same_transcript_as_its_question_is_recorded(tmp_path) -> None:
    run, ledger, _ = outlet(tmp_path)
    item = expectation()
    awaiting(ledger, item)
    now = ASKED + timedelta(minutes=10)
    run.publish([request(item)], now=now)
    run.claim("codex", SESSION, now, 3)
    asked_at = now + timedelta(minutes=1)
    messages = (
        HostMessage(1, now + timedelta(seconds=2), "prompt", "帮我看看这个报错"),
        codex_question([TEXT], "call_a1", asked_at),
        HostMessage(
            4,
            asked_at,
            "tool_result",
            '{"accepted":true}',
            tool_name="request_user_input_async",
            tool_call_id="call_a1",
        ),
        desktop_reply([("call_a1", 0, TEXT, "这次不用")], asked_at + timedelta(minutes=2)),
        HostMessage(8, asked_at + timedelta(minutes=3), "completion", "好。"),
    )
    report = run.observe(SESSION, messages, host="codex")
    assert (report.spoken, report.responded, report.awaiting_reply) == (1, 1, 1)
    (response,) = ledger.responses_on(item.day)
    assert response.polarity == "负" and run.store.asked() == {}


def test_bug2_two_sessions_claiming_at_the_same_instant_never_both_get_the_request(tmp_path) -> None:
    import threading

    run, ledger, _ = outlet(tmp_path)
    for round_number in range(20):
        item = expectation(f"e{round_number}", kind=f"s-k{round_number:04d}")
        awaiting(ledger, item)
        now = ASKED + timedelta(minutes=10, seconds=round_number)
        run.publish([request(item)], now=now)
        barrier = threading.Barrier(2)
        got: dict[str, tuple[str, ...]] = {}

        def take(session: str, barrier=barrier, got=got, now=now) -> None:  # type: ignore[no-untyped-def]
            barrier.wait()
            got[session] = tuple(r.request_id for r in run.claim("codex", session, now, 3))

        threads = [threading.Thread(target=take, args=(f"s-{k}",)) for k in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(got["s-0"]) + len(got["s-1"]) == 1, (round_number, got)


def test_bug3_an_asked_record_names_the_session_that_asked_not_the_latest_holder(tmp_path) -> None:
    run, ledger, _ = outlet(tmp_path)
    item = expectation()
    awaiting(ledger, item)
    now = ASKED + timedelta(minutes=10)
    run.publish([request(item)], now=now)
    run.claim("codex", "codex-s1", now, 3)
    later = now + timedelta(minutes=31)
    run.claim("codex", "codex-s2", later, 3)  # s1 租约过期，s2 取走
    asked_at = now + timedelta(minutes=1)
    first = (
        HostMessage(1, now + timedelta(seconds=2), "prompt", "帮我看看"),
        codex_question([TEXT], "call_s1", asked_at),
        HostMessage(
            4,
            asked_at,
            "tool_result",
            '{"accepted":true}',
            tool_name="request_user_input_async",
            tool_call_id="call_s1",
        ),
        HostMessage(5, asked_at + timedelta(minutes=2), "completion", "看完了。"),
    )
    report = run.observe("codex-s1", first, host="codex")  # s1 的转写晚到：它问过
    assert (report.spoken, report.awaiting_reply) == (1, 1)
    (asked,) = run.store.asked().values()
    assert asked["session"] == "codex-s1"
    answered_at = later + timedelta(minutes=5)
    second = (
        desktop_reply([("call_s1", 0, TEXT, "现在就做")], answered_at),
        HostMessage(8, answered_at + timedelta(seconds=5), "completion", "好"),
    )
    assert run.observe("codex-s1", second, host="codex").responded == 1


def test_bug4_one_call_with_two_questions_pairs_both_reminders_and_both_answers(tmp_path) -> None:
    run, ledger, _ = outlet(tmp_path)
    first, second = expectation("e1"), expectation("e2", kind="s-k0003")
    text2 = "周四了，这周的账还没对。"
    for item in (first, second):
        awaiting(ledger, item)
    now = ASKED + timedelta(minutes=10)
    run.publish([request(first), request(second, text2)], now=now)
    assert len(run.claim("codex", SESSION, now, 3)) == 2
    asked_at = now + timedelta(minutes=1)
    messages = (
        HostMessage(1, now + timedelta(seconds=2), "prompt", "帮我看看"),
        codex_question([text2, TEXT], "call_m", asked_at),
        HostMessage(
            4, asked_at, "tool_result", '{"accepted":true}', tool_name="request_user_input_async", tool_call_id="call_m"
        ),
        HostMessage(5, asked_at + timedelta(minutes=2), "completion", "看完了。"),
    )
    report = run.observe(SESSION, messages, host="codex")
    assert (report.spoken, report.awaiting_reply, report.released) == (2, 2, 0)
    spoken = {d.expectation_id: d.text for d in ledger.deliveries_on(first.day) if d.spoken}
    assert spoken == {"e1": TEXT, "e2": text2}
    answered_at = asked_at + timedelta(minutes=5)
    reply = (
        desktop_reply([("call_m", 0, text2, "下周再弄"), ("call_m", 1, TEXT, "这次不用")], answered_at),
        HostMessage(8, answered_at + timedelta(seconds=5), "completion", "好"),
    )
    assert run.observe(SESSION, reply, host="codex").responded == 1  # 自由作答那题交模型，固定选项那题写负
    (response,) = ledger.responses_on(first.day)
    assert (response.expectation_id, response.polarity) == ("e1", "负")
    assert run.store.asked() == {}


def test_bug5_similar_texts_pair_by_best_score_not_by_list_order(tmp_path) -> None:
    run, ledger, _ = outlet(tmp_path, kind_label=lambda token: None)
    text1, text2 = "往常周四下午你会写周报，今天还没有。", "往常周四下午你会对账，今天还没有。"
    first, second = expectation("e1"), expectation("e2", kind="s-k0003")
    for item in (first, second):
        awaiting(ledger, item)
    now = ASKED + timedelta(minutes=10)
    run.publish([request(first, text1), request(second, text2)], now=now)
    run.claim("claude-code", "claude-s1", now, 3)
    messages = turn(
        claude_question(text2, "toolu_b", now + timedelta(minutes=1)),
        claude_answer(f'"{text2}"="这次不用"', "toolu_b", now + timedelta(minutes=1, seconds=30)),
        claude_question(text1, "toolu_a", now + timedelta(minutes=2)),
        claude_answer(f'"{text1}"="现在就做"', "toolu_a", now + timedelta(minutes=2, seconds=30)),
        at=now + timedelta(seconds=2),
    )
    report = run.observe("claude-s1", messages, host="claude-code")
    assert (report.spoken, report.responded) == (2, 2)
    spoken = {d.expectation_id: (d.text, d.turn) for d in ledger.deliveries_on(first.day) if d.spoken}
    assert spoken == {"e1": (text1, "toolu_a"), "e2": (text2, "toolu_b")}
    assert {r.expectation_id: r.polarity for r in ledger.responses_on(first.day)} == {"e1": "正", "e2": "负"}


def test_bug6_a_failed_ledger_write_leaves_the_request_claimable_and_a_replay_writes_it(tmp_path, monkeypatch) -> None:
    run, ledger, recorder = outlet(tmp_path)
    item = expectation()
    awaiting(ledger, item)
    now = ASKED + timedelta(minutes=10)
    run.publish([request(item)], now=now)
    run.claim("claude-code", "claude-s1", now, 3)
    messages = turn(claude_question(TEXT, at=now + timedelta(minutes=1)), at=now + timedelta(seconds=2))

    def broken(day, delivery):  # type: ignore[no-untyped-def]
        raise OSError("disk full")

    monkeypatch.setattr(ledger, "record_delivery", broken)
    report = run.observe("claude-s1", messages, host="claude-code")
    assert (report.spoken, report.already_spoken) == (0, 1)
    assert run.store.read_claim("e1")["spoken"] is False and run.store.read_open() != ()
    monkeypatch.undo()
    replay = run.observe("claude-s1", messages, host="claude-code")
    assert replay.spoken == 1 and len([d for d in ledger.deliveries_on(item.day) if d.spoken]) == 1
    assert any(event.operation == "outlet_spoken" and event.status.value == "failure" for event in recorder.events)


def test_bug7_a_late_transcript_of_the_previous_turn_does_not_release_the_claim(tmp_path) -> None:
    run, ledger, _ = outlet(tmp_path)
    item = expectation()
    awaiting(ledger, item)
    now = ASKED + timedelta(minutes=10)
    run.publish([request(item)], now=now)
    run.claim("claude-code", "claude-s1", now, 3)
    previous = (
        HostMessage(1, now - timedelta(seconds=60), "prompt", "上一句话"),
        HostMessage(2, now - timedelta(seconds=30), "completion", "上一轮取走之前就答完了。"),
    )
    report = run.observe("claude-s1", previous, host="claude-code")
    assert report.released == 0 and run.store.read_claim("e1")["holder"] is not None
    assert run.claim("claude-code", "claude-other", now + timedelta(seconds=10), 3) == ()


def test_bug9_the_same_session_reclaiming_after_its_lease_expired_keeps_the_claimed_at(tmp_path) -> None:
    run, ledger, _ = outlet(tmp_path)
    item = expectation()
    awaiting(ledger, item)
    now = ASKED + timedelta(minutes=10)
    run.publish([request(item)], now=now)
    (first,) = run.claim("codex", SESSION, now, 3)
    (again,) = run.claim("codex", SESSION, now + timedelta(minutes=31), 3)
    assert again.claimed_at == first.claimed_at == now  # 续租：宿主按它判是同一次，不再注入
    assert again.lease_until == now + timedelta(minutes=61)
    assert run.store.read_claim("e1")["attempts"] == 0
    # 真正放回之后再取：换 claimed_at
    run.observe(SESSION, turn(at=now + timedelta(minutes=31, seconds=5)), host="codex")
    (third,) = run.claim("codex", SESSION, now + timedelta(minutes=40), 3)
    assert third.claimed_at == now + timedelta(minutes=40) and run.store.read_claim("e1")["attempts"] == 1


def test_answers_are_read_per_question_in_a_sync_result() -> None:
    both = '"题一"="知道了"；"题二"="这次不用"'
    assert answer_in_result(both, "题一", single=False) == "中"
    assert answer_in_result(both, "题二", single=False) == "负"
    assert answer_in_result(both, "题三", single=False) is None
    assert answer_in_result(json.dumps({"answers": ["现在就做"]}), "题一", single=True) == "正"
    assert (
        answer_in_result(json.dumps({"answers": ["现在就做"]}), "题一", single=False) is None
    )  # 多题的结果对不上题就不猜


def test_ruling65a_the_agents_own_question_mentioning_the_class_is_not_our_reminder(tmp_path) -> None:
    """题目含类名、选项却不是我们那三个："写周报的模板用哪一个？"是 agent 自己的事，不记「说了」、放回（裁定 65）。"""

    run, ledger, _ = outlet(tmp_path)
    item = expectation()
    awaiting(ledger, item)
    now = ASKED + timedelta(minutes=10)
    run.publish([request(item)], now=now)
    run.claim("claude-code", "claude-s1", now, 3)
    own = HostMessage(
        3,
        now + timedelta(minutes=1),
        "tool_call",
        json.dumps(
            {
                "questions": [
                    {"question": "写周报的模板用哪一个？", "options": [{"label": "上周的"}, {"label": "新建"}]}
                ]
            },
            ensure_ascii=False,
        ),
        tool_name="AskUserQuestion",
        tool_call_id="toolu_own",
    )
    report = run.observe("claude-s1", turn(own, at=now + timedelta(seconds=2)), host="claude-code")
    assert (report.spoken, report.released) == (0, 1)
    assert not any(d.spoken for d in ledger.deliveries_on(item.day))
    # 我们的选项 + 含类名：照常配上（题目被改写也认得）
    run.claim("claude-code", "claude-s1", now + timedelta(minutes=10), 3)
    ours = claude_question("要不要现在写周报？", "toolu_ours", now + timedelta(minutes=11))
    report = run.observe("claude-s1", turn(ours, at=now + timedelta(minutes=10, seconds=2)), host="claude-code")
    assert report.spoken == 1


def test_ruling65b_a_question_asked_after_the_thing_was_already_done_is_not_spoken(tmp_path) -> None:
    """「待取」后 tick 记了「已发生」、取走的请求没撤、agent 照问：不算「说了」、取走记录删掉；账上没有 spoken，结算不会把功劳记给提醒。"""

    run, ledger, _ = outlet(tmp_path)
    item = expectation()
    awaiting(ledger, item)
    now = ASKED + timedelta(minutes=10)
    run.publish([request(item)], now=now)
    run.claim("claude-code", "claude-s1", now, 3)
    ledger.record_delivery(
        item.day, Delivery("e1", now + timedelta(minutes=5), 1, False, "已发生", 0.0)
    )  # 15:55 他自己做了
    run.publish([], now=now + timedelta(minutes=6))  # 下一拍：清单里没有它了，但取走记录还在
    asked_at = now + timedelta(minutes=8)
    report = run.observe(
        "claude-s1",
        turn(
            claude_question(TEXT, at=asked_at),
            claude_answer(f'"{TEXT}"="现在就做"', at=asked_at + timedelta(seconds=30)),
            at=now + timedelta(seconds=2),
        ),
        host="claude-code",
    )
    assert (report.spoken, report.responded, report.already_done) == (0, 0, 1)
    assert not any(d.spoken for d in ledger.deliveries_on(item.day)) and ledger.responses_on(item.day) == ()
    assert run.store.claims() == {}
    assert run.claim("claude-code", "claude-other", asked_at, 3) == ()


def test_opencode_question_tool_is_recognised_with_its_answer(tmp_path) -> None:
    """OpenCode 的 question 工具：题目在 questions[].question、选项是 {label}；作答由插件写成 "<题目>"="<作答>"。"""

    run, ledger, _ = outlet(tmp_path)
    item = expectation()
    awaiting(ledger, item)
    now = ASKED + timedelta(minutes=10)
    run.publish([request(item)], now=now)
    run.claim("opencode", "opencode-ses_1", now, 3)
    arguments = {
        "questions": [
            {"question": TEXT, "header": "提醒", "options": [{"label": o, "description": ""} for o in OPTIONS]}
        ]
    }
    call = HostMessage(
        3,
        now + timedelta(minutes=1),
        "tool_call",
        json.dumps(arguments, ensure_ascii=False),
        tool_name="question",
        tool_call_id="call_q",
    )
    answer = HostMessage(
        4,
        now + timedelta(minutes=2),
        "tool_result",
        f'"{TEXT}"="现在就做"',
        tool_name="question",
        tool_call_id="call_q",
    )
    report = run.observe("opencode-ses_1", turn(call, answer, at=now + timedelta(seconds=2)), host="opencode")
    assert (report.spoken, report.responded) == (1, 1)
    (spoken,) = [d for d in ledger.deliveries_on(item.day) if d.spoken]
    assert (spoken.via, spoken.turn) == ("opencode", "call_q")
    (response,) = ledger.responses_on(item.day)
    assert response.polarity == "正"


def test_a_dismissed_synchronous_question_is_spoken_but_never_waits_for_a_late_answer(tmp_path) -> None:
    """他把 OpenCode 的提问关掉了（工具结果是失败 "The user dismissed this question"）：问出口了记「说了」，不写回话，
    也不记成"等晚到的作答"——同步工具的作答不会再来。"""

    run, ledger, _ = outlet(tmp_path)
    item = expectation()
    awaiting(ledger, item)
    now = ASKED + timedelta(minutes=10)
    run.publish([request(item)], now=now)
    run.claim("opencode", "opencode-ses_1", now, 3)
    arguments = {"questions": [{"question": TEXT, "header": "提醒", "options": [{"label": o} for o in OPTIONS]}]}
    call = HostMessage(3, now + timedelta(minutes=1), "tool_call", json.dumps(arguments, ensure_ascii=False), tool_name="question", tool_call_id="call_q")
    dismissed = HostMessage(
        4, now + timedelta(minutes=2), "tool_result", "The user dismissed this question", tool_name="question", tool_call_id="call_q", failed=True
    )
    report = run.observe("opencode-ses_1", turn(call, dismissed, at=now + timedelta(seconds=2)), host="opencode")
    assert (report.spoken, report.responded, report.awaiting_reply) == (1, 0, 0)
    assert ledger.responses_on(item.day) == () and run.store.asked() == {}


from habitus.runtime.foresight_outlet import REPLY_TAG  # noqa: E402 - 放在最后：上面的夹具只用名字


# ── habitus reminders：某天每条提醒走到了哪一步（方案 36 第 2 步）────────────────────────────────


def test_the_reminder_day_reads_each_stage_from_the_ledger_and_the_claim_trail(tmp_path) -> None:
    from habitus.runtime.foresight_reminders import reminder_day

    run, ledger, _ = outlet(tmp_path)
    spoken, taken, waiting, missed, quiet = (
        expectation(f"e{i}", kind="s-k0002" if i == 1 else f"s-k000{i + 4}", deadline_hours=4 + i) for i in range(1, 6)
    )
    for item in (spoken, taken, waiting, missed):
        awaiting(ledger, item)
    ledger.record_expectation(quiet)  # 只等时窗末、没进出口的不列
    ledger.record_delivery(quiet.day, Delivery("e5", ASKED, 1, False, "等时窗末", 1.0, TEXT))
    ledger.record_delivery(missed.day, Delivery("e4", missed.deadline, 1, False, "宿主没取", 1.0))
    now = ASKED + timedelta(minutes=10)
    run.publish([request(spoken), request(taken), request(waiting)], now=now)
    run.claim("claude-code", "claude-s1", now, 1)  # 取走 e1
    asked_at = now + timedelta(minutes=2)
    run.observe(
        "claude-s1",
        turn(
            claude_question(TEXT, at=asked_at),
            claude_answer(f'User has answered your questions: "{TEXT}"="现在就做". You can now continue.', at=asked_at + timedelta(seconds=30)),
            at=now + timedelta(seconds=3),
        ),
        host="claude-code",
    )
    run.claim("codex", SESSION, now + timedelta(minutes=5), 1)  # 取走 e2，还在租约里

    report = reminder_day(ledger, run, FORMED.date())
    stages = {item.expectation_id: item.stage for item in report.reminders}
    assert stages == {"e1": "说了", "e2": "取走了", "e3": "待取", "e4": "宿主没取"}
    first = report.reminders[0]
    assert (first.kind, first.text, first.spoken_via, first.spoken_at, first.response) == (
        "写周报", TEXT, "claude-code", asked_at, "正",
    )
    second = next(item for item in report.reminders if item.expectation_id == "e2")
    assert second.attempts == 0 and [(h.host, h.session, h.ended) for h in second.handoffs] == [("codex", SESSION, None)]
    assert (report.count("说了"), report.count("宿主没取"), report.responded) == (1, 1, 1)
    assert reminder_day(ledger, None, FORMED.date()).reminders[0].kind == "s-k0002"  # 没有出口就给编号
