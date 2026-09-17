"""判断包：schema 把条数与名字钉死；装配把卡号换回 URI、对记账疏漏逐项降级；服务一包一问、瞬态错有界重试。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from habitus.foresight import EvidencePack
from habitus.foresight.judge import (
    ASSEMBLY_VERSION,
    JUDGE_JSON_SCHEMA,
    JUDGE_PROMPT_VERSION,
    JUDGE_VERSION,
    SCHEMA_FINGERPRINT,
    CandidateVerdict,
    JudgeAssemblyError,
    JudgeConfig,
    Judgement,
    LLMJudge,
    assemble_judgement,
    judge_json_schema,
)
from habitus.foresight.render import render_pack
from habitus.model_client import ChatClient, ModelTransportError, StructuredChatClient
from tests.unit.foresight.fixtures import MONDAY, Ground, ScriptedJudge, at
from tests.unit.scene.association_payloads import ScriptedProvider, model_config, recording_client

NOW = MONDAY + timedelta(days=28)
JUDGED_AT = datetime(2026, 8, 31, 11, 5, tzinfo=UTC)


def ground_for(tmp_path) -> Ground:
    """周一 18:40 收拾球包、19:00 打球、20:10 洗澡，四周；早饭每天 8:00。19:05 时打球与收拾球包摊开，早饭与洗澡只列名。"""

    ground = Ground(tmp_path, now=at(NOW, 23, 0))
    for week in range(4):
        day = MONDAY + timedelta(days=7 * week)
        ground.record(day, "收拾球包", 18, 40, kind="收拾球包")
        play = ground.record(day, "打球", 19, 0, kind="打球", lasts_minutes=60)
        ground.record(day, "洗澡", 20, 10, kind="洗澡")
        ground.record(day, "吃早饭", 8, 0, kind="吃早饭")
        if week < 2:
            ground.associate(play, kind="打球", context=f"第 {week + 1} 周", situation="周一下班后自己去")
    return ground


def pack_for(tmp_path) -> EvidencePack:
    return ground_for(tmp_path).pack(at(NOW, 19, 5))


def verdict(kind: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "kind_token": kind,
        "verdict": "会",
        "window": {"from_slot": 76, "to_slot": 78},
        "next": ["洗澡"],
        "basis": [1, 2],
        "note": "此刻像 #1、#2 之前",
    }
    row.update(overrides)
    return row


def output(*verdicts: dict[str, object], day_state: str = "正常", day_note: str | None = None) -> dict[str, object]:
    return {"verdicts": list(verdicts), "day_state": day_state, "day_note": day_note}


def good_output() -> dict[str, object]:
    return output(verdict("打球"), verdict("收拾球包", verdict="不会", window=None, next=[], basis=[1]))


def test_schema_pins_one_verdict_per_expanded_candidate_by_name() -> None:
    schema = judge_json_schema(("打球", "收拾球包"))
    verdicts = schema["properties"]["verdicts"]
    assert (verdicts["minItems"], verdicts["maxItems"]) == (2, 2)
    assert verdicts["items"]["properties"]["kind_token"]["enum"] == ["打球", "收拾球包"]
    assert "enum" not in JUDGE_JSON_SCHEMA["properties"]["verdicts"]["items"]["properties"]["kind_token"]
    # 没有摊开的候选时条数钉成 0，enum 不能为空所以不设。
    empty = judge_json_schema(())["properties"]["verdicts"]
    assert (empty["minItems"], empty["maxItems"]) == (0, 0)
    assert "enum" not in empty["items"]["properties"]["kind_token"]
    with pytest.raises(ValueError):
        judge_json_schema(("打球", "打球"))
    assert JUDGE_VERSION == f"{JUDGE_PROMPT_VERSION}+schema{SCHEMA_FINGERPRINT}+{ASSEMBLY_VERSION}"


def test_a_clean_answer_maps_card_numbers_back_to_uris(tmp_path) -> None:
    pack = pack_for(tmp_path)
    assert [item.kind_token for item in pack.expanded] == ["打球", "收拾球包"]
    play = pack.expanded[0]
    judgement = assemble_judgement(good_output(), pack, judged_at=JUDGED_AT, version=JUDGE_VERSION)

    assert judgement.signals == ()
    assert judgement.judge_version == JUDGE_VERSION
    assert (judgement.generation, judgement.moment) == (pack.generation, pack.moment)
    assert (judgement.day_state, judgement.day_note) == ("正常", None)
    first = judgement.verdict_for("打球")
    assert first is not None
    assert first.basis == (play.background.cards[0].uri, play.background.cards[1].uri)
    assert first.window == (76, 78) and first.next == ("洗澡",) and first.note == "此刻像 #1、#2 之前"
    assert judgement.expected == (first,)
    second = judgement.verdict_for("收拾球包")
    assert second is not None and second.verdict == "不会" and second.window is None
    assert second.basis == (pack.expanded[1].background.cards[0].uri,)


def test_unknown_and_repeated_candidates_are_dropped_and_unanswered_ones_become_undecided(tmp_path) -> None:
    pack = pack_for(tmp_path)
    judgement = assemble_judgement(
        output(verdict("打球"), verdict("打球", verdict="不会", window=None), verdict("跳舞"), "nonsense"),
        pack,
        judged_at=JUDGED_AT,
        version=JUDGE_VERSION,
    )
    assert [item.kind_token for item in judgement.verdicts] == ["打球", "收拾球包"]
    play = judgement.verdict_for("打球")
    assert play is not None and play.verdict == "会"
    left = judgement.verdict_for("收拾球包")
    assert left == CandidateVerdict(kind_token="收拾球包", verdict="说不准", window=None, next=(), basis=(), note="")
    assert "verdict_dropped: 打球 appears more than once" in judgement.signals
    assert "verdict_dropped: '跳舞' is not an expanded candidate in this pack" in judgement.signals
    assert "verdict_dropped: malformed entry" in judgement.signals
    assert "unanswered: 收拾球包 got no verdict; recorded as 说不准" in judgement.signals


def test_a_yes_without_cards_becomes_undecided_and_unknown_cards_are_dropped(tmp_path) -> None:
    pack = pack_for(tmp_path)
    judgement = assemble_judgement(
        output(verdict("打球", basis=[]), verdict("收拾球包", basis=[1, 99, 1, "x"], next=[])),
        pack,
        judged_at=JUDGED_AT,
        version=JUDGE_VERSION,
    )
    play = judgement.verdict_for("打球")
    assert play is not None and play.verdict == "说不准" and play.basis == () and play.next == ()
    assert "verdict_degraded: 打球 says 会 but cites no card; recorded as 说不准" in judgement.signals
    left = judgement.verdict_for("收拾球包")
    assert left is not None and left.verdict == "会"
    assert left.basis == (pack.expanded[1].background.cards[0].uri,)
    assert "basis_dropped: 收拾球包 cites card 99, which is not in this pack" in judgement.signals
    assert "basis_dropped: 收拾球包 repeats card #1" in judgement.signals
    assert "basis_dropped: 收拾球包 cites card 'x', which is not in this pack" in judgement.signals


def test_next_steps_must_follow_a_cited_card(tmp_path) -> None:
    """收拾球包的卡之后是打球；打球的卡之后是洗澡。名字不在所引卡"之后"里的丢。"""

    pack = pack_for(tmp_path)
    judgement = assemble_judgement(
        output(verdict("打球", next=["洗澡", "跳舞", "洗澡"]), verdict("收拾球包", basis=[1], next=["洗澡", "打球"])),
        pack,
        judged_at=JUDGED_AT,
        version=JUDGE_VERSION,
    )
    play = judgement.verdict_for("打球")
    assert play is not None and play.next == ("洗澡",)
    left = judgement.verdict_for("收拾球包")
    assert left is not None and left.next == ("打球",)
    assert "next_dropped: 打球 names '跳舞', which follows none of the cited cards" in judgement.signals
    assert "next_dropped: 收拾球包 names '洗澡', which follows none of the cited cards" in judgement.signals


def test_windows_must_fit_today_and_not_end_before_now(tmp_path) -> None:
    pack = pack_for(tmp_path)
    assert pack.moment.slot == 76 and pack.slots_per_day == 96
    cases = {
        "outside": {"from_slot": 90, "to_slot": 100},
        "reversed": {"from_slot": 80, "to_slot": 78},
        "past": {"from_slot": 70, "to_slot": 75},
        "shape": [76, 78],
    }
    for label, window in cases.items():
        judgement = assemble_judgement(output(verdict("打球", window=window), verdict("收拾球包", basis=[1])), pack, judged_at=JUDGED_AT, version=JUDGE_VERSION)
        play = judgement.verdict_for("打球")
        assert play is not None and play.verdict == "会" and play.window is None, label
        assert any(note.startswith("window_dropped: 打球") for note in judgement.signals), label
    # 起槽早于此刻的截到此刻：时窗只说从此刻起的事，此刻之前那一截没意义，但止槽仍是有效信息。
    judgement = assemble_judgement(output(verdict("打球", window={"from_slot": 60, "to_slot": 78}), verdict("收拾球包", basis=[1])), pack, judged_at=JUDGED_AT, version=JUDGE_VERSION)
    play = judgement.verdict_for("打球")
    assert play is not None and play.window == (76, 78)
    assert any(note.startswith("window_clamped: 打球") for note in judgement.signals)
    # 到此刻所在的槽为止仍算未来；「不会」不带时窗。
    judgement = assemble_judgement(
        output(verdict("打球", window={"from_slot": 76, "to_slot": 76}), verdict("收拾球包", verdict="不会", basis=[1])),
        pack,
        judged_at=JUDGED_AT,
        version=JUDGE_VERSION,
    )
    play = judgement.verdict_for("打球")
    assert play is not None and play.window == (76, 76)
    left = judgement.verdict_for("收拾球包")
    assert left is not None and left.window is None
    assert "window_dropped: 收拾球包 is judged 不会 yet carries a window" in judgement.signals


def test_unknown_verdicts_and_day_states_degrade_and_notes_are_cleaned(tmp_path) -> None:
    pack = pack_for(tmp_path)
    judgement = assemble_judgement(
        output(verdict("打球", verdict="maybe"), verdict("收拾球包", basis=[1], note="  会​的 "), day_state="odd", day_note=" \t"),
        pack,
        judged_at=JUDGED_AT,
        version=JUDGE_VERSION,
    )
    play = judgement.verdict_for("打球")
    assert play is not None and play.verdict == "说不准" and play.window == (76, 78)  # 说不准也可以带时窗
    left = judgement.verdict_for("收拾球包")
    assert left is not None and left.note == "会 的"
    # "没答"记 None，不写成"正常"——那是在替模型断言今天没事。
    assert (judgement.day_state, judgement.day_note) == (None, None)
    assert "verdict_degraded: 打球 says 'maybe'; recorded as 说不准" in judgement.signals
    assert "day_state_dropped: 'odd' is not a day state; recorded as not answered" in judgement.signals
    abnormal = assemble_judgement(good_output() | {"day_state": "反常", "day_note": "往常这时已经收拾了球包"}, pack, judged_at=JUDGED_AT, version=JUDGE_VERSION)
    assert (abnormal.day_state, abnormal.day_note) == ("反常", "往常这时已经收拾了球包")


def test_non_array_references_and_integer_valued_floats_are_handled_not_crashed(tmp_path) -> None:
    pack = pack_for(tmp_path)
    judgement = assemble_judgement(
        output(verdict("打球", basis="1", next="洗澡"), verdict("收拾球包", basis=[1.0], window={"from_slot": 76.0, "to_slot": 77.0}, next=[])),
        pack,
        judged_at=JUDGED_AT,
        version=JUDGE_VERSION,
    )
    play = judgement.verdict_for("打球")
    assert play is not None and play.verdict == "说不准" and play.basis == () and play.next == ()
    assert "basis_dropped: 打球 basis is not an array" in judgement.signals
    assert "next_dropped: 打球 next is not an array" in judgement.signals
    left = judgement.verdict_for("收拾球包")
    assert left is not None and left.basis == (pack.expanded[1].background.cards[0].uri,) and left.window == (76, 77)
    assert "index_coerced: 收拾球包 basis arrived as 1.0" in judgement.signals
    assert "index_coerced: 收拾球包 window from_slot arrived as 76.0" in judgement.signals


def test_an_answer_that_is_not_about_this_pack_is_rejected_outright(tmp_path) -> None:
    pack = pack_for(tmp_path)
    with pytest.raises(JudgeAssemblyError):
        assemble_judgement(["not", "an", "object"], pack, judged_at=JUDGED_AT, version=JUDGE_VERSION)
    with pytest.raises(JudgeAssemblyError):
        assemble_judgement({"verdicts": "none"}, pack, judged_at=JUDGED_AT, version=JUDGE_VERSION)
    with pytest.raises(JudgeAssemblyError):
        assemble_judgement(output(verdict("跳舞")), pack, judged_at=JUDGED_AT, version=JUDGE_VERSION)
    # 产物类型的自洽错误也归成装配错误（让结构层纠正重问），不漏成 ForesightError 让整拍失败。
    with pytest.raises(JudgeAssemblyError, match="does not assemble"):
        assemble_judgement(good_output(), pack, judged_at=datetime(2026, 8, 31, 11, 5), version=JUDGE_VERSION)


def test_the_verdict_shape_refuses_self_contradiction() -> None:
    with pytest.raises(Exception, match="cite at least one card"):
        CandidateVerdict(kind_token="打球", verdict="会", window=None, next=(), basis=(), note="")
    with pytest.raises(Exception, match="has no window"):
        CandidateVerdict(kind_token="打球", verdict="不会", window=(1, 2), next=(), basis=(), note="")
    with pytest.raises(Exception, match="only come from cited cards"):
        CandidateVerdict(kind_token="打球", verdict="说不准", window=None, next=("洗澡",), basis=(), note="")
    with pytest.raises(Exception, match="end before it starts"):
        CandidateVerdict(kind_token="打球", verdict="会", window=(5, 4), next=(), basis=("u",), note="")


def test_the_llm_judge_asks_once_with_the_pinned_schema_and_the_rendered_pack(tmp_path) -> None:
    pack = pack_for(tmp_path)
    client, provider = recording_client([good_output()])
    judge = LLMJudge(client, clock=lambda: JUDGED_AT)

    judgement = asyncio.run(judge.judge(pack))

    assert judge.version == JUDGE_VERSION
    assert provider.calls == 1 and client.names == ["foresight_judgement"]
    pinned = client.schemas[0]["properties"]["verdicts"]
    assert (pinned["minItems"], pinned["maxItems"]) == (2, 2)
    assert pinned["items"]["properties"]["kind_token"]["enum"] == ["打球", "收拾球包"]
    assert provider.prompts[-1] == render_pack(pack)
    assert "- #1 " in provider.prompts[-1] and "钟面：槽宽 15 分钟" in provider.prompts[-1]
    assert judgement.judged_at == JUDGED_AT
    assert [item.verdict for item in judgement.verdicts] == ["会", "不会"]


def test_a_pack_with_nothing_expanded_is_still_asked_about_the_day(tmp_path) -> None:
    """没有摊开的候选时只剩"今天正不正常"要判，那也要读了此刻场景才能答，不由本层替模型填。"""

    pack = ground_for(tmp_path).pack(at(NOW, 3, 0))
    assert pack.expanded == ()
    client, provider = recording_client([output(day_state="反常", day_note="凌晨三点还在活动")])
    judgement = asyncio.run(LLMJudge(client, clock=lambda: JUDGED_AT).judge(pack))
    assert provider.calls == 1
    assert judgement.verdicts == () and (judgement.day_state, judgement.day_note) == ("反常", "凌晨三点还在活动")


def test_an_unusable_answer_is_corrected_by_the_structured_layer_and_the_judgement_says_so(tmp_path) -> None:
    pack = pack_for(tmp_path)
    client, provider = recording_client([output(verdict("跳舞")), good_output()])
    judgement = asyncio.run(LLMJudge(client, clock=lambda: JUDGED_AT).judge(pack))
    assert provider.calls == 2
    assert judgement.expected[0].kind_token == "打球"
    # 第二轮才答对要留在判断的信号里：读判断的人得知道这不是一问就对的答复。
    assert "structured: answered on attempt 2" in judgement.signals
    assert judgement.judge_version == JUDGE_VERSION


class _FlakyClient(StructuredChatClient):
    """前几次在结构化客户端这一层就断线：路由层自己的重试够不着，只能由判断服务的那一层接住。"""

    def __init__(self, *args, failures: int, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.failures = failures
        self.attempts = 0

    async def complete_json_async(self, request, **kwargs):  # type: ignore[no-untyped-def, override]
        self.attempts += 1
        if self.attempts <= self.failures:
            raise ModelTransportError("boom")
        return await super().complete_json_async(request, **kwargs)


def _flaky(failures: int) -> _FlakyClient:
    return _FlakyClient(ChatClient(model_config(), ScriptedProvider([good_output()])), validation_retries=1, failures=failures)


def test_transient_errors_are_retried_a_bounded_number_of_times(tmp_path, monkeypatch) -> None:
    pack = pack_for(tmp_path)
    naps: list[float] = []

    async def nap(seconds: float) -> None:
        naps.append(seconds)

    monkeypatch.setattr("habitus.foresight.judge.service.asyncio.sleep", nap)
    client = _flaky(1)
    judge = LLMJudge(client, config=JudgeConfig(transient_retries=1, transient_retry_delay_seconds=2.0), clock=lambda: JUDGED_AT)
    assert asyncio.run(judge.judge(pack)).expected[0].kind_token == "打球"
    assert client.attempts == 2 and naps == [2.0]

    exhausted = _flaky(5)
    strict = LLMJudge(exhausted, config=JudgeConfig(transient_retries=1, transient_retry_delay_seconds=0), clock=lambda: JUDGED_AT)
    with pytest.raises(ModelTransportError):
        asyncio.run(strict.judge(pack))
    assert exhausted.attempts == 2


def test_the_scripted_judge_replays_its_script_against_each_pack(tmp_path) -> None:
    pack = pack_for(tmp_path)
    script = Judgement(judged_at=JUDGED_AT, generation="other", moment=pack.moment, verdicts=(), day_state="正常", day_note=None, judge_version="scripted-judge")
    judge = ScriptedJudge(script)
    replayed = asyncio.run(judge.judge(pack))
    assert replayed.generation == pack.generation and judge.packs == [pack]


def test_config_rejects_negative_retries() -> None:
    with pytest.raises(ValueError):
        JudgeConfig(transient_retries=-1)
    with pytest.raises(ValueError):
        JudgeConfig(transient_retry_delay_seconds=-0.5)
