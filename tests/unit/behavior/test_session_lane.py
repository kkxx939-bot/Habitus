"""会话 lane 的融合：拆轮、取材料、抹密钥、输出校验、一轮一条判断。

依据是方案第五、六节与裁定 31–37；每条用例对应其中一条规则，用例名写的就是那条规则。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import pytest

from habitus.behavior.fusion import BehaviorFusionEnqueuer, BehaviorFusionJobStore
from habitus.behavior.fusion.errors import BehaviorFusionError
from habitus.behavior.fusion.lanes.session import (
    SESSION_PROMPT_VERSION,
    SESSION_PROTOCOL,
    SessionLaneConfig,
    SessionMessage,
    SessionTurn,
    TurnDisposition,
    assemble_description,
    redact,
    render_turn,
    session_json_schema,
    session_observer,
    split_turns,
)
from habitus.behavior.fusion.lanes.session.material import ANSWER_PREFIX, WAKE_LABEL
from habitus.behavior.fusion.lanes.session.redaction import REDACTED
from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.model_client.contracts import ModelStructuredOutputError
from tests.unit.behavior.session_fixtures import (
    START,
    ZONE,
    Stores,
    at,
    call,
    description,
    lane,
    prompt,
    reply,
    result,
)

CONVERSATION = "codex-rollout-0bc9ee2f"


def one_turn(*messages: SessionMessage) -> SessionTurn:
    (turn,) = split_turns(CONVERSATION, messages)
    return turn


# ── 拆轮 ─────────────────────────────────────────────────────────────────────────────


def test_a_turn_runs_from_one_instruction_to_just_before_the_next() -> None:
    turns = split_turns(
        CONVERSATION,
        (
            prompt(0, "读取层应该写在哪里？"),
            reply(1, "建议放在 editor 下。", minutes=1),
            prompt(2, "那写吧", minutes=4),
            call(3, "apply_patch", "{}", call_id="c1", minutes=5),
            reply(4, "写好了。", minutes=9),
        ),
    )
    assert [(turn.start_sequence, turn.end_sequence) for turn in turns] == [(0, 1), (2, 4)]
    assert [turn.key for turn in turns] == [f"{CONVERSATION}#0-1", f"{CONVERSATION}#2-4"]
    # 开始是人开口的时刻，结束是助手做完的时刻。
    assert (turns[1].instructed_at, turns[1].completed_at) == (at(4), at(9))


def test_synthetic_host_messages_do_not_open_a_turn_nor_close_the_current_one() -> None:
    turns = split_turns(
        CONVERSATION,
        (
            prompt(0, "继续"),
            prompt(1, "<system-reminder>后台任务结束</system-reminder>", minutes=1),
            reply(2, "接着拆完了。", minutes=2),
        ),
    )
    assert len(turns) == 1
    assert turns[0].instruction == "继续"
    assert turns[0].end_sequence == 2


def test_three_instructions_in_the_same_second_are_a_replay_not_speech() -> None:
    replayed = tuple(prompt(index, f"重放的第 {index} 句") for index in range(3))
    assert split_turns(CONVERSATION, (*replayed, reply(3, "……"))) == ()


def test_host_messages_in_the_same_second_do_not_make_his_one_sentence_a_replay() -> None:
    """宿主常在他开口的那一秒里连写几条合成消息；重放只数人说的话，他真说的那一句照样开轮。"""

    (turn,) = split_turns(
        CONVERSATION,
        (
            prompt(0, "<environment_context>cwd=/Users/a</environment_context>"),
            prompt(1, "<recommended_plugins>…</recommended_plugins>"),
            prompt(2, "把词表的归类逻辑讲一下"),
            reply(3, "归类分三步……"),
        ),
    )
    assert turn.instruction == "把词表的归类逻辑讲一下"


def test_messages_before_the_first_instruction_belong_to_no_turn() -> None:
    turns = split_turns(
        CONVERSATION, (reply(0, "上一轮剩下的答复"), prompt(1, "下一步是什么？"), reply(2, "先做读取层。"))
    )
    assert [(turn.start_sequence, turn.end_sequence) for turn in turns] == [(1, 2)]


def test_a_turn_without_any_reply_or_call_is_unanswered() -> None:
    interrupted, answered = split_turns(
        CONVERSATION,
        (prompt(0, "配置统一：集中管理"), prompt(1, "配置统一：集中管理", minutes=1), reply(2, "好。", minutes=2)),
    )
    assert not interrupted.answered
    assert answered.answered
    # 只有调用、没有文字答复，也算助手动了手。
    assert one_turn(prompt(0, "继续"), call(1, "bash", "ls", call_id="c1")).answered


# ── 取材料 ───────────────────────────────────────────────────────────────────────────


def test_material_keeps_everything_but_tool_result_bodies() -> None:
    text = render_turn(
        one_turn(
            prompt(0, "查一下 CI 为什么失败"),
            reply(1, "我先看远端的运行记录。", minutes=0.2),
            call(2, "bash", "gh run view 123", call_id="c1"),
            result(3, "这里是一大段成功的输出正文", call_id="c1"),
            call(4, "bash", "ruff check .", call_id="c2"),
            result(5, "E501 line too long", call_id="c2", failed=True),
            reply(6, "是 ci.yml 里的路径没跟着改名。", minutes=2),
        )
    )
    assert "他说：查一下 CI 为什么失败" in text
    assert "我先看远端的运行记录。" in text  # 中途的答复留着：流程就在这里
    assert "bash gh run view 123" in text
    assert "这里是一大段成功的输出正文" not in text  # 成功的结果正文不给模型
    assert "bash ruff check . → 失败：E501 line too long" in text  # 失败的留报错开头
    assert "助手最后说：是 ci.yml 里的路径没跟着改名。" in text


def test_identical_consecutive_tool_calls_fold_into_one_line() -> None:
    calls = [call(index, "bash", "pytest -q", call_id=f"c{index}") for index in range(1, 4)]
    text = render_turn(one_turn(prompt(0, "跑测试"), *calls, reply(4, "都过了。")))
    assert text.count("bash pytest -q") == 1
    assert "（连续 3 次）" in text


def test_an_oversized_turn_is_trimmed_tools_first_and_the_instruction_last() -> None:
    config = SessionLaneConfig(max_turn_chars=900, max_instruction_chars=400, kept_tool_lines=4)
    calls = [call(index, "bash", f"step-{index} " + "x" * 80, call_id=f"c{index}") for index in range(1, 41)]
    text = render_turn(
        one_turn(prompt(0, "把这四十步都跑完"), *calls, reply(41, "全部完成。" + "y" * 50)),
        config=config,
    )
    assert len(text) <= config.max_turn_chars
    assert "他说：把这四十步都跑完" in text
    assert "（中间省略 36 次调用）" in text
    assert "step-1 " in text and "step-40 " in text


def test_too_many_interim_replies_keep_only_the_head_and_tail_ones() -> None:
    config = SessionLaneConfig(max_turn_chars=900, max_instruction_chars=400, kept_reply_lines=4)
    interim = [reply(index, f"第{index}段进展：" + "z" * 60) for index in range(1, 41)]
    text = render_turn(one_turn(prompt(0, "一口气做完"), *interim, reply(41, "全部完成。")), config=config)
    assert len(text) <= config.max_turn_chars
    assert "（中间省略 36 条）" in text
    assert "第1段进展" in text and "第40段进展" in text
    assert "第20段进展" not in text
    assert "助手最后说：全部完成。" in text


def test_answering_the_assistants_question_opens_a_new_turn() -> None:
    asked, answered = split_turns(
        CONVERSATION,
        (
            prompt(0, "把配置集中起来", minutes=0),
            reply(1, "有两种放法。", minutes=1),
            call(2, "AskUserQuestion", '{"questions": [{"question": "放哪？"}]}', call_id="q1", minutes=2),
            result(
                3,
                'Your questions have been answered: "放哪？"="独立模块"',
                call_id="q1",
                name="AskUserQuestion",
                minutes=9,
            ),
            reply(4, "按独立模块改完了。", minutes=12),
        ),
    )
    assert (asked.start_sequence, asked.end_sequence) == (0, 2)
    assert asked.completed_at == at(2)  # 助手发问时上一轮到此为止
    assert asked.answered
    assert answered.instructed_at == at(9)  # 他作答的时刻
    assert answered.instruction == f'{ANSWER_PREFIX}Your questions have been answered: "放哪？"="独立模块"'
    assert answered.completed_at == at(12)
    assert "他说：（回答助手的提问）" in render_turn(answered)


def test_a_rejected_or_invalid_question_is_not_an_answer() -> None:
    (turn,) = split_turns(
        CONVERSATION,
        (
            prompt(0, "把配置集中起来"),
            call(1, "AskUserQuestion", "{bad", call_id="q1"),
            result(2, "InputValidationError", call_id="q1", name="AskUserQuestion", failed=True),
            call(3, "AskUserQuestion", "{}", call_id="q2"),
            result(4, "The user doesn't want to proceed", call_id="q2", name="AskUserQuestion", failed=True),
            reply(5, "那你想先澄清哪一点？"),
        ),
    )
    assert turn.end_sequence == 5


def test_other_tool_results_never_open_a_turn() -> None:
    (turn,) = split_turns(
        CONVERSATION,
        (prompt(0, "跑测试"), call(1, "bash", "pytest", call_id="c1"), result(2, "ok", call_id="c1", name="bash")),
    )
    assert turn.end_sequence == 2
    assert split_turns(
        CONVERSATION,
        (prompt(0, "跑"), call(1, "Ask", "{}", call_id="c"), result(2, "选 1", call_id="c", name="Ask")),
        question_tools=("Ask",),
    )[1].instruction.endswith("选 1")


def test_work_done_after_being_woken_is_its_own_turn() -> None:
    turns = split_turns(
        CONVERSATION,
        (
            prompt(0, "跑全量回归", minutes=0),
            call(1, "bash", "pytest &", call_id="c1", minutes=1),
            reply(2, "已在后台跑，等结果。", minutes=2),
            prompt(3, "<task-notification>done</task-notification>", minutes=300),
            call(4, "bash", "cat result", call_id="c2", minutes=301),
            reply(5, "回归全过。", minutes=302),
            prompt(6, "<task-notification>another</task-notification>", minutes=350),
            prompt(7, "那提交吧", minutes=400),
            reply(8, "已提交。", minutes=401),
        ),
    )
    assert [(turn.start_sequence, turn.end_sequence) for turn in turns] == [(0, 2), (3, 5), (6, 6), (7, 8)]
    assert turns[0].completed_at == at(2)  # 上一轮到助手停下为止
    woken = turns[1]
    assert (woken.instructed_at, woken.completed_at) == (at(300), at(302))
    assert woken.instruction.startswith(WAKE_LABEL) and woken.answered
    assert not turns[2].answered  # 被叫醒之后什么也没做：不出记录


def test_a_source_that_starts_with_a_wake_up_still_yields_its_turn() -> None:
    (turn,) = split_turns(
        CONVERSATION, (prompt(0, "<task-notification>done</task-notification>"), reply(1, "回归全过。"))
    )
    assert turn.instruction.startswith(WAKE_LABEL)


def test_a_notification_while_the_assistant_is_still_working_does_not_end_the_turn() -> None:
    (turn,) = split_turns(
        CONVERSATION,
        (
            prompt(0, "跑全量回归"),
            call(1, "bash", "pytest", call_id="c1"),
            prompt(2, "<task-notification>done</task-notification>"),
            reply(3, "回归全过。"),
        ),
    )
    assert turn.end_sequence == 3


def test_secrets_never_reach_the_model() -> None:
    text = render_turn(
        one_turn(prompt(0, "现在改用 deepseek apikey： sk-bff31c8581df478187656abcd"), reply(1, "已切换。"))
    )
    assert "sk-bff31c8581df478187656abcd" not in text
    assert REDACTED in text


def test_a_secret_is_redacted_before_the_text_around_it_is_cut_short() -> None:
    """先截再抹的话，被截掉后半截的密钥不再是密钥的形状，开头几位会漏给模型。"""

    secret = "sk-bff31c8581df478187656abcd"
    config = SessionLaneConfig(tool_argument_chars=30, tool_error_chars=30)
    text = render_turn(
        one_turn(
            prompt(0, "调一下接口"),
            call(1, "bash", f"curl -H 'x-api-key: {secret}' https://example.invalid", call_id="c1"),
            result(2, f"401 invalid key {secret} rejected", call_id="c1", failed=True),
            reply(3, "密钥不对。"),
        ),
        config=config,
    )
    assert "sk-bff" not in text
    (woken,) = split_turns(
        CONVERSATION,
        (prompt(0, "<task-notification>" + "进度 " * 95 + secret + "</task-notification>"), reply(1, "看到了。")),
    )
    assert "sk-bff" not in woken.instruction


# ── 抹密钥 ───────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "secret",
    [
        "sk-ant-api03-abcdefghijklmnopqrstuvwx",
        "ark-71e28876-20e7-4a4b-9f3c-0123456789ab",
        "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        "AKIAIOSFODNN7EXAMPLE",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N",
        "Bearer abcdefghijklmnopqrstuvwxyz012345",
    ],
)
def test_recognisable_secret_shapes_are_redacted(secret: str) -> None:
    assert redact(f"用这个 {secret} 调接口") == f"用这个 {REDACTED} 调接口"


def test_assignment_shaped_secrets_keep_the_name_and_lose_the_value() -> None:
    assert redact("password=correct-horse-battery-staple") == f"password={REDACTED}"
    assert redact("普通的一句话，没有密钥") == "普通的一句话，没有密钥"


@pytest.mark.parametrize(
    ("text", "cleaned"),
    [
        # 密钥紧跟在中文后面
        ("密钥是sk-bff31c8581df478187656abcd。", f"密钥是{REDACTED}。"),
        # 名字是更长的变量名的结尾
        ("ARK_API_KEY=0f8fad5b-d9cb-469f-a165-70867728950e", f"ARK_API_KEY={REDACTED}"),
        # JSON，以及再转义一层的 JSON
        ('{"api_key":"0f8fad5b-d9cb-469f-a165-70867728950e"}', f'{{"api_key":"{REDACTED}"}}'),
        ('{\\"api_key\\": \\"0f8fad5b-d9cb-469f\\"}', f'{{\\"api_key\\": \\"{REDACTED}\\"}}'),
        ("apikey：0f8fad5b-d9cb-469f-a165-70867728950e", f"apikey：{REDACTED}"),
    ],
)
def test_secrets_are_recognised_wherever_they_sit(text: str, cleaned: str) -> None:
    assert redact(text) == cleaned


@pytest.mark.parametrize(
    "text",
    [
        "token: /Users/a/.config/tool/token.json",  # 放密钥的地方，不是密钥
        "password=~/.secrets/password.txt",
        "max_tokens: 123456789012345",
        "tokenizer: sentencepiece_model_v2",
        "task-abcdefghijklmnopqrstu",
    ],
)
def test_text_that_only_looks_like_a_secret_is_left_alone(text: str) -> None:
    assert redact(text) == text


# ── 输出校验 ─────────────────────────────────────────────────────────────────────────


def test_the_output_shape_has_no_decision_about_other_turns() -> None:
    schema = session_json_schema()
    assert set(schema["properties"]) == {"name", "summary", "goal", "steps"}  # 融合只描述，不判归属
    assert schema["additionalProperties"] is False


def test_a_description_is_capped_and_redacted_before_it_is_stored() -> None:
    config = SessionLaneConfig(max_steps=3, max_summary_chars=40, max_step_chars=10)
    described = assemble_description(
        description(
            summary="他把密钥 sk-bff31c8581df478187656abcd 交给助手；" + "后面还有很长的一段" * 20,
            goal=None,
            steps=["第一步" * 10, "  ", "第二步", "第三步", "第四步"],
        ),
        config=config,
    )
    assert "sk-bff31c8581df478187656abcd" not in described.summary
    assert len(described.summary) <= 40
    assert described.goal is None
    assert described.steps == ("第一步第一步第一步…", "第二步", "第三步")  # 空步骤丢掉，最多 3 条，每条封顶


def test_text_over_its_cap_is_cut_at_a_pause_and_marked() -> None:
    config = SessionLaneConfig(max_summary_chars=30)
    described = assemble_description(
        description(summary="他要求把配置拆成两层，助手照做并跑通了检查；随后又补了三处遗漏的引用并重新验证。"), config=config
    )
    assert described.summary == "他要求把配置拆成两层，助手照做并跑通了检查…"  # 截在停顿上，不从词中间切


def test_path_symbols_in_a_name_are_replaced_not_sent_back_to_the_model() -> None:
    described = assemble_description(description(name="对比 A/B 两种方案: 先看 habitus\\runtime"))
    assert described.name == "对比 A B 两种方案 先看 habitus runtime"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ([], "output must be an object"),
        (description(name="  "), "name is required"),
        (description(name=".."), "name is not usable"),
        (description(summary=""), "summary must be a non-empty sentence"),
        (description(goal=""), "goal must be text or null"),
        (description(steps="不是数组"), "steps must be an array of text"),
    ],
)
def test_malformed_descriptions_are_rejected_so_the_model_is_asked_again(body: object, message: str) -> None:
    with pytest.raises(BehaviorFusionError, match=message):
        assemble_description(body)


# ── 一轮一条判断 ─────────────────────────────────────────────────────────────────────


def test_one_turn_becomes_one_judgement_spanning_instruction_to_completion(tmp_path: Path) -> None:
    stores = Stores(tmp_path)
    session, provider = lane(stores, [description()])
    turn = one_turn(prompt(0, "查看我本地 m2bos 有多少行代码？"), reply(1, "8,792 行。", minutes=117))

    (outcome,) = asyncio.run(session.consume((turn,)))

    assert outcome.disposition is TurnDisposition.RECORDED
    assert provider.calls == 1
    (judgement,) = stores.judgements.list()
    assert judgement["judgement_id"] == outcome.judgement_id
    assert judgement["behavior"] == "统计本地项目代码行数"
    assert judgement["relations"] == []  # 不判续、不合并：一轮一条，互不相认
    assert judgement["prompt_version"] == SESSION_PROMPT_VERSION
    # 开始是人开口的时刻，最后所见是助手做完的时刻，都带本地偏移。
    assert datetime.fromisoformat(judgement["started_at"]) == START.astimezone(ZONE)
    assert datetime.fromisoformat(judgement["last_observed_at"]) == at(117).astimezone(ZONE)
    assert judgement["started_at"].endswith("+08:00")


def test_the_evidence_holds_times_and_a_reference_but_no_conversation_content(tmp_path: Path) -> None:
    stores = Stores(tmp_path)
    session, _provider = lane(stores, [description()])
    turn = one_turn(prompt(0, "这句话不该出现在凭据里"), reply(1, "这句答复也不该出现"))

    asyncio.run(session.consume((turn,)))

    (envelope,) = stores.observations.list()
    assert envelope.protocol == SESSION_PROTOCOL
    assert envelope.observer_id == session_observer(CONVERSATION)  # 会话身份带在凭据的来源上
    opened, closed = sorted(envelope.batch.observations, key=lambda item: item.occurred_at)
    assert (opened.occurred_at, closed.occurred_at) == (turn.instructed_at, turn.completed_at)
    for observation in (opened, closed):
        assert observation.evidence_refs == (turn.key,)  # 指向会话源：哪个会话、第几条到第几条
        assert "这句话不该出现在凭据里" not in observation.semantics
        assert "这句答复也不该出现" not in observation.semantics


def test_an_unanswered_turn_produces_no_record_and_no_model_call(tmp_path: Path) -> None:
    stores = Stores(tmp_path)
    session, provider = lane(stores, [description()])
    interrupted, resent = split_turns(
        CONVERSATION,
        (prompt(0, "配置统一：集中管理"), prompt(1, "配置统一：集中管理", minutes=1), reply(2, "好。", minutes=117)),
    )

    outcomes = asyncio.run(session.consume((interrupted, resent)))

    # 被打断后原样重发：只有得到答复的那一次出记录，不另设"重发去重"。
    assert [item.disposition for item in outcomes] == [TurnDisposition.UNANSWERED, TurnDisposition.RECORDED]
    assert provider.calls == 1
    assert len(stores.judgements.list()) == 1


def test_redoing_a_turn_does_not_write_it_twice(tmp_path: Path) -> None:
    stores = Stores(tmp_path)
    session, provider = lane(stores, [description(), description(name="重做时模型换了个说法")])
    turn = one_turn(prompt(0, "继续"), reply(1, "拆完了。"))

    first = asyncio.run(session.consume((turn,)))
    again = asyncio.run(session.consume((turn,)))

    assert first[0].disposition is TurnDisposition.RECORDED
    assert again[0].disposition is TurnDisposition.ALREADY_RECORDED
    assert provider.calls == 1  # 写过的轮连模型都不再问
    assert len(stores.judgements.list()) == 1


def test_a_failed_description_leaves_nothing_behind_so_the_turn_can_be_retried(tmp_path: Path) -> None:
    stores = Stores(tmp_path)
    bad = description(name="..")
    session, provider = lane(stores, [bad, bad, description()])
    turn = one_turn(prompt(0, "继续"), reply(1, "拆完了。"))

    with pytest.raises(ModelStructuredOutputError, match="domain validation"):
        asyncio.run(session.consume((turn,)))
    assert stores.judgements.list() == ()
    assert stores.observations.list() == ()  # 不写"没读懂"的空白

    (outcome,) = asyncio.run(session.consume((turn,)))
    assert outcome.disposition is TurnDisposition.RECORDED
    assert provider.calls == 3


def test_the_model_sees_this_turn_only_and_never_the_person_s_secrets(tmp_path: Path) -> None:
    stores = Stores(tmp_path)
    session, provider = lane(stores, [description(), description()])
    first = one_turn(prompt(0, "用 token=abcdefghijklmnop1234 登录"), reply(1, "登录成功。"))
    (second,) = split_turns(CONVERSATION, (prompt(2, "继续", minutes=5), reply(3, "做完了。", minutes=6)))

    asyncio.run(session.consume((first, second)))

    assert "abcdefghijklmnop1234" not in provider.prompts[0]
    assert "【这一轮】2026-07-26 17:58" in provider.prompts[0]  # 时刻按本地时间给
    assert "登录" not in provider.prompts[1]  # 不带别的轮：没有清单，没有上文


# ── 写到一半崩了，重做时接着写完 ─────────────────────────────────────────────────────


@pytest.mark.parametrize("crash_at", ["receipts.put", "coverage.record"])
def test_a_turn_interrupted_after_its_judgement_is_finished_not_judged_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_at: str
) -> None:
    """判断落了盘、回执还没落就崩了：重做时再问模型，措辞一变就是第二条判断。认回原来那条，补上回执。"""

    stores = Stores(tmp_path)
    session, provider = lane(stores, [description("第一次的说法"), description("第二次的说法")])
    turn = one_turn(prompt(0, "继续"), reply(1, "拆完了。"))
    store, method = crash_at.split(".")

    def crash(*_: object) -> None:
        raise RuntimeError("进程崩溃")

    with monkeypatch.context() as patched:
        patched.setattr(getattr(stores, store), method, crash)
        with pytest.raises(RuntimeError):
            asyncio.run(session.consume((turn,)))

    stores.now = at(200)  # 重做发生在另一个时刻
    (outcome,) = asyncio.run(session.consume((turn,)))

    assert outcome.disposition is TurnDisposition.ALREADY_RECORDED
    assert [item["behavior"] for item in stores.judgements.list()] == ["第一次的说法"]
    assert outcome.judgement_id == stores.judgements.list()[0]["judgement_id"]
    assert provider.calls == 1  # 没有重问模型
    assert session.recorder.is_recorded(turn)
    assert len(stores.receipts.list()) == 1


def test_a_turn_interrupted_before_its_judgement_is_simply_judged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stores = Stores(tmp_path)
    session, _ = lane(stores, [description("第一次的说法"), description("第二次的说法")])
    turn = one_turn(prompt(0, "继续"), reply(1, "拆完了。"))

    def crash(*_: object) -> None:
        raise RuntimeError("进程崩溃")

    with monkeypatch.context() as patched:
        patched.setattr(stores.judgements, "put_payload", crash)
        with pytest.raises(RuntimeError):
            asyncio.run(session.consume((turn,)))

    (outcome,) = asyncio.run(session.consume((turn,)))
    assert outcome.disposition is TurnDisposition.RECORDED
    assert [item["behavior"] for item in stores.judgements.list()] == ["第二次的说法"]


def test_coverage_is_read_once_for_a_batch_of_turns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """覆盖记录每轮读一遍的话，读的量随已有记录数涨，一份长会话越做越慢。"""

    stores = Stores(tmp_path)
    session, _ = lane(stores, [description(f"第 {index} 件事") for index in range(4)])
    turns = split_turns(
        CONVERSATION,
        tuple(
            message
            for index in range(4)
            for message in (prompt(2 * index, f"第 {index} 句", minutes=index), reply(2 * index + 1, "好。", minutes=index + 0.5))
        ),
    )
    reads = 0
    original = stores.coverage.covered_observation_ids

    def counted(now: datetime | None = None) -> frozenset[str]:
        nonlocal reads
        reads += 1
        return original(now)

    monkeypatch.setattr(stores.coverage, "covered_observation_ids", counted)
    checks: list[int] = []
    asyncio.run(session.consume(turns, before_each=lambda: checks.append(1)))
    assert reads == 1
    assert len(checks) == 4  # 每一轮之前都问一次调用方还该不该接着做


# ── 与逐帧融合互不相扰 ───────────────────────────────────────────────────────────────


def test_the_frame_pipeline_never_enqueues_session_evidence(tmp_path: Path) -> None:
    """凭据落盘与回执落盘之间崩溃，会留下两条没被覆盖的凭据；它们不是帧，逐帧融合不许拿去切段。"""

    stores = Stores(tmp_path)
    recorder = stores.recorder()
    turn = one_turn(prompt(0, "继续"), reply(1, "拆完了。"))
    stores.observations.put(recorder._envelope(turn, recorder._evidence(turn)))  # noqa: SLF001 - 模拟崩溃留下的半截
    assert not recorder.is_recorded(turn)

    enqueuer = BehaviorFusionEnqueuer(
        stores.observations,
        BehaviorFusionJobStore(tmp_path, PathLock(ProcessLocalLockStore()), clock=lambda: stores.now),
        stores.receipts,
        coverage=stores.coverage,
        quiet_period_seconds=0,
        clock=lambda: stores.now,
    )
    assert enqueuer.enqueue_ready().count == 0
