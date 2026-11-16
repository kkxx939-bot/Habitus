"""第二步甲阶段 C 补的测试：预期表里阶段 A 没写成的几条、撤回即失败没变红的两处、对抗输入。

- C18 后半句：在用上限满了，够条件却等着的词数报在换代结果、"数在用的词"那一步的报告、换代读数与每代报告文件里；
- D3：新的选词触点只有一个结构化输出名（``behavior_word_pick``），会话一轮一次、物理一条一次；
- 叫醒标记是轮上的结构信息，进重试堆的编解码（共享说明 3.2）；
- 选词结果进判断的内容身份（撤回即失败：去掉它时阶段 A 的测试不红）；
- 在用状态落盘、带"哪一天转的在用"，下一代不重算日子（撤回即失败：改成每次现算时阶段 A 的测试不红）；
- 对抗：初始词表数据文件、在用状态文件、判断上的选词结果、树上记录的 ``kind_token`` 坏了都明确报错并点出文件。
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from habitus.behavior.fusion import BehaviorJudgementStore, derive_judgements, judgement_payload
from habitus.behavior.fusion.errors import BehaviorFusionError
from habitus.behavior.fusion.judgement import BehaviorClaim
from habitus.behavior.fusion.lanes.session import SessionMessageRole
from habitus.behavior.fusion.lanes.session.model import decode_turn, encode_turn
from habitus.behavior.kinds.ids import ClassId, Lane
from habitus.behavior.kinds.model import BehaviorKindError
from habitus.behavior.kinds.store import STATE_FILENAME, BehaviorKindStore, BehaviorKindStoreError
from habitus.behavior.kinds.words import initial_words, parse_words
from habitus.behavior.model import BehaviorKind
from habitus.behavior.schema import fields as schema_fields
from habitus.behavior.schema.model import BehaviorFieldType
from habitus.behavior.tree import BehaviorTree
from habitus.behavior.tree.writer import BehaviorDocumentWriter
from habitus.behavior.uri import BehaviorURI
from habitus.infrastructure.store.sqlite import SQLiteLockStore
from habitus.runtime.generation_report import GenerationReports
from habitus.runtime.generation_result import generation_attributes
from habitus.series.reader import read_series
from tests.lock_store import ProcessLocalLockStore
from tests.unit.behavior.judgement_builders import batch_from
from tests.unit.behavior.test_fusion_judgement_durability import FRAGMENTS, JUDGED_AT, LANE_VERSION, SOURCE_REFS, body
from tests.unit.behavior.test_fusion_judgement_durability import PROMPT_VERSION as JUDGEMENT_PROMPT_VERSION
from tests.unit.behavior.tree_payloads import occurrence_payload
from tests.unit.word_pick.site import CST, REPOSITORY_ROOT, Site, Word, at
from tests.unit.word_pick.test_in_use import D13, D14, D15, D16, D17, _cap_two, behaviours
from tests.unit.word_pick.test_physical_pick import delivery

WORDS_DIR = REPOSITORY_ROOT / "habitus" / "behavior" / "kinds" / "words"
PICK_NAME = "behavior_word_pick"


@pytest.fixture
def site(tmp_path: Path):  # type: ignore[no-untyped-def]
    built = Site(tmp_path)
    yield built
    built.close()


# ── C18 后半句：等着的个数报出来 ────────────────────────────────────────────


def test_c18_the_generation_reports_how_many_qualified_words_wait_behind_the_cap(tmp_path: Path) -> None:
    site = Site(tmp_path / "site", edit=_cap_two)
    try:
        site.record("conv-a", at(D13, 9), "帮我写一篇投稿论文", Word("s-k0006"))
        site.record("conv-a", at(D14, 9), "论文再写一节", Word("s-k0006"))
        site.record("conv-b", at(D13, 10), "帮我写一份调研报告", Word("s-k0005"))
        site.record("conv-b", at(D15, 10), "再写一份测试报告", Word("s-k0005"))
        site.record("conv-c", at(D14, 11), "把这个季度的工作写个总结", Word("s-k0003"))
        site.record("conv-c", at(D16, 11), "再写一个项目总结", Word("s-k0003"))

        result = site.generate(D17)
        assert set(behaviours(site)) == {"s-k0006", "s-k0005"}
        assert (result.active_words, result.waiting_words) == (2, 1)
        (counting,) = [step for step in result.steps if step.step == "active"]
        assert counting.report["waiting_words"] == 1 and counting.report["active_words"] == 2
        attributes = generation_attributes(result)
        assert (attributes["active_words"], attributes["waiting_words"]) == (2, 1)
        reports = GenerationReports(tmp_path / "reports", artifact_root=tmp_path, relations=None)
        reports.write(result, now=datetime(2026, 4, 17, 6, tzinfo=UTC))
        written = reports.read(D17)
        assert written is not None
        inputs = written["inputs"]
        assert isinstance(inputs, dict) and inputs["waiting_words"] == 1
    finally:
        site.close()


# ── D3：新的选词触点 ──────────────────────────────────────────────────────


def test_d3_one_session_turn_and_one_physical_record_are_each_one_word_pick_call(site: Site) -> None:
    site.router.picks.extend([Word("s-k0002"), Word("p-k0005")])
    site.consume(site.turn("conv-d3", at(D13, 9), "帮我写一份本周周报"))
    site.reduce()
    asyncio.run(site.runtime.deliver_physical(delivery("换衣服")))
    site.reduce(site.physical)
    picks = [call for call in site.router.calls if call.kind == "pick"]
    assert [call.name for call in picks] == [PICK_NAME, PICK_NAME]
    # 选词这一种调用只在一个地方发出
    sources = [
        path
        for path in (REPOSITORY_ROOT / "habitus").rglob("*.py")
        if f'"{PICK_NAME}"' in path.read_text(encoding="utf-8")
    ]
    assert [path.relative_to(REPOSITORY_ROOT).as_posix() for path in sources] == [
        "habitus/behavior/kinds/pick/picker.py"
    ]


# ── 叫醒标记进编解码 ──────────────────────────────────────────────────────


def _woken_turn(site: Site) -> Any:
    prompt, completion = SessionMessageRole.PROMPT, SessionMessageRole.COMPLETION
    _first, woken = site.turns(
        "conv-codec",
        [
            (0.0, prompt, "帮我在后台把全量测试跑一遍", False),
            (1.0, completion, "已经在后台跑了。", False),
            (5.0, prompt, "<task-notification>后台任务完成</task-notification>", False),
            (6.0, completion, "测试全部通过了。", False),
        ],
        at(D13, 15),
    )
    return woken


def test_the_wake_up_mark_is_structural_and_survives_the_retry_codec(site: Site) -> None:
    woken = _woken_turn(site)
    assert woken.woken and woken.notice_only
    decoded = decode_turn(encode_turn(woken))
    assert decoded == woken and decoded.woken and decoded.notice_only
    encoded = encode_turn(woken)
    encoded.pop("woken")
    with pytest.raises(BehaviorFusionError):
        decode_turn(encoded)
    bad = {**encode_turn(woken), "woken": "true"}
    with pytest.raises(BehaviorFusionError):
        decode_turn(bad)


# ── 选词结果进判断的内容身份 ──────────────────────────────────────────────


def _with_token(token: str | None) -> Any:
    batch = batch_from(body())
    first = batch.judgements[0]
    judgement = dataclasses.replace(first, claim=dataclasses.replace(first.claim, kind_token=token))
    return derive_judgements(
        dataclasses.replace(batch, judgements=(judgement,)),
        FRAGMENTS,
        source_refs=SOURCE_REFS,
        judged_at=JUDGED_AT,
        fusion_version=LANE_VERSION,
        prompt_version=JUDGEMENT_PROMPT_VERSION,
    )[0]


def test_the_picked_word_is_part_of_the_judgement_identity_and_round_trips(tmp_path: Path) -> None:
    one, other, none = _with_token("s-k0002"), _with_token("s-k0003"), _with_token(None)
    assert len({one.judgement_id, other.judgement_id, none.judgement_id}) == 3
    store = BehaviorJudgementStore(tmp_path)
    store.put(one)
    store.put(other)  # 同一份描述配上另一个词是另一条判断，不撞车
    read = store.read(one.judgement_id)
    assert read is not None and read["kind_token"] == "s-k0002"


# ── 在用状态落盘 ──────────────────────────────────────────────────────────


def test_the_in_use_state_is_on_disk_with_the_day_it_turned_and_is_not_recounted(site: Site) -> None:
    site.record("conv-a", at(D13, 9), "帮我写一份本周周报", Word("s-k0002"))
    site.record("conv-a", at(D14, 9), "这周的周报也帮我写一下", Word("s-k0002"))
    site.generate(D15)
    store = site.session.kind_store
    assert (store.root / STATE_FILENAME).is_file()
    first = store.read()
    assert dict(first.active) == {ClassId(Lane.SESSION, 2): D15} and first.revision == 1

    site.record("conv-b", at(D15, 9), "再写一份周报", Word("s-k0002"))
    site.generate(D16)
    again = store.read()
    assert dict(again.active) == {ClassId(Lane.SESSION, 2): D15}  # 转在用的那一天不随下一代变
    assert again.revision == 1


def test_a_word_in_use_stays_in_use_even_if_the_series_no_longer_shows_two_days(site: Site) -> None:
    """盘上已经在用的词，下一代即使序列里数不到两天也不退出（二乙起只有退出天数没被选中才退出，见 word_grow）。"""

    store = site.session.kind_store
    store.update(lambda current: current.with_active({ClassId(Lane.SESSION, 4): D13}))
    site.record("conv-a", at(D14, 9), "帮我写一份本周周报", Word("s-k0002"))
    site.generate(D15)
    assert set(behaviours(site)) == {"s-k0004"}
    assert dict(store.read().active) == {ClassId(Lane.SESSION, 4): D13}


# ── 对抗：初始词表数据文件 ────────────────────────────────────────────────


def _session_words() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((WORDS_DIR / "session.json").read_text(encoding="utf-8"))
    return loaded


def _second_level(raw: dict[str, Any]) -> dict[str, Any]:
    return next(word for word in raw["words"] if word["parent"] is not None)


def _parse(raw: dict[str, Any]) -> None:
    parse_words(json.dumps(raw, ensure_ascii=False), lane=Lane.SESSION, source="words/session.json")


def test_the_packaged_word_files_parse_and_match_the_expected_counts() -> None:
    assert len(initial_words(Lane.SESSION).words) == 279
    assert len(initial_words(Lane.PHYSICAL).words) == 223
    assert initial_words(Lane.SESSION).active == {} and initial_words(Lane.SESSION).revision == 0


@pytest.mark.parametrize(
    ("label", "corrupt", "pattern"),
    [
        ("missing_parent", lambda raw: _second_level(raw).update(parent="s-k9999"), "unknown parent"),
        ("level_mismatch", lambda raw: _second_level(raw).update(level=3), "does not follow its parent"),
        ("duplicate_id", lambda raw: raw["words"][1].update(id=raw["words"][0]["id"]), "id repeats"),
        ("duplicate_name", lambda raw: raw["words"][1].update(name=raw["words"][0]["name"]), "name repeats"),
        ("unknown_field", lambda raw: raw["words"][0].update(reminder="该写了"), "keys must be"),
        ("truncated", None, "not valid JSON"),
    ],
)
def test_a_broken_initial_word_file_is_refused_and_names_the_file(label: str, corrupt: Any, pattern: str) -> None:
    raw = _session_words()
    if corrupt is None:
        text = json.dumps(raw, ensure_ascii=False)[:500]
        with pytest.raises(BehaviorKindError, match=pattern) as caught:
            parse_words(text, lane=Lane.SESSION, source="words/session.json")
    else:
        corrupt(raw)
        with pytest.raises(BehaviorKindError, match=pattern) as caught:
            _parse(raw)
    assert "words/session.json" in str(caught.value), label


# ── 对抗：在用状态文件 ────────────────────────────────────────────────────


def _state_store(tmp_path: Path) -> tuple[BehaviorKindStore, Path, dict[str, Any]]:
    store = BehaviorKindStore(
        tmp_path, lane=Lane.SESSION, lock_store=SQLiteLockStore(tmp_path / "locks.sqlite3"), lock_ttl_seconds=30
    )
    store.update(lambda current: current.with_active({ClassId(Lane.SESSION, 2): D15}))
    path = tmp_path / STATE_FILENAME
    return store, path, json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("label", "corrupt"),
    [
        ("truncated", lambda text, raw: text[: len(text) // 2]),
        ("unknown_field", lambda text, raw: json.dumps({**raw, "extra": 1})),
        ("unknown_word", lambda text, raw: json.dumps({**raw, "active": [{"id": "s-k9999", "since": "2026-04-15"}]})),
        (
            "other_lane_word",
            lambda text, raw: json.dumps({**raw, "active": [{"id": "p-k0001", "since": "2026-04-15"}]}),
        ),
        ("marker_as_word", lambda text, raw: json.dumps({**raw, "active": [{"id": "s-续话", "since": "2026-04-15"}]})),
        ("bad_since", lambda text, raw: json.dumps({**raw, "active": [{"id": "s-k0002", "since": "昨天"}]})),
        ("negative_revision", lambda text, raw: json.dumps({**raw, "revision": -1})),
    ],
)
def test_a_broken_in_use_state_file_is_refused_and_names_the_file(tmp_path: Path, label: str, corrupt: Any) -> None:
    store, path, raw = _state_store(tmp_path)
    path.write_text(corrupt(path.read_text(encoding="utf-8"), raw), encoding="utf-8")
    with pytest.raises(BehaviorKindStoreError) as caught:
        store.read()
    assert str(path) in str(caught.value), label


def test_a_broken_in_use_state_file_stops_the_generation_count(site: Site) -> None:
    site.record("conv-a", at(D13, 9), "帮我写一份本周周报", Word("s-k0002"))
    site.record("conv-a", at(D14, 9), "这周的周报也帮我写一下", Word("s-k0002"))
    path = site.session.tree.root / STATE_FILENAME
    path.write_text('{"schema_version": "behavior_kinds_state_v1", "lane": "session"', encoding="utf-8")
    result = site.generate(D15)
    # 二乙起读序列也要过词表状态的换算口：状态文件坏了，必须成的读序列一步就没成，这一代没有、不写完成标记
    assert not result.complete
    assert [step.outcome for step in result.steps if step.step == "series"] == ["failed"]
    assert behaviours(site) == {}  # 没数成：语义树没有凭空长出概念


# ── 对抗：判断上的选词结果 ────────────────────────────────────────────────


@pytest.mark.parametrize("token", ["s-未知", "s-k0002x", "写周报", "s-k00002", "s-k0000", ""])
def test_a_claim_with_an_unknown_kind_token_is_refused(token: str) -> None:
    with pytest.raises(BehaviorFusionError):
        BehaviorClaim(behavior="写周报", goal=None, summary="他让助手写周报", basis=(), kind_token=token)


@pytest.mark.parametrize("token", ["s-未知", "s-k0002x", "写周报", 7])
def test_a_judgement_file_with_an_unknown_kind_token_is_refused_and_names_the_file(
    tmp_path: Path, token: object
) -> None:
    store = BehaviorJudgementStore(tmp_path)
    judgement = _with_token("s-k0002")
    store.put(judgement)
    (path,) = [item for item in tmp_path.rglob("*.json") if judgement.judgement_id in item.name]
    payload = {**judgement_payload(judgement), "kind_token": token}
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update({key: value for key, value in payload.items() if key == "kind_token"})
    path.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="",  # 规范编码只认 \n：Windows 文本模式会写成 \r\n
    )
    with pytest.raises(BehaviorFusionError) as caught:
        store.read(judgement.judgement_id)
    assert judgement.judgement_id in str(caught.value) or path.name in str(caught.value)


# ── 对抗：树上记录的 kind_token ───────────────────────────────────────────


def _publish_raw(tree: Any, token: str) -> str:
    """绕过写入守卫把一条 kind_token 坏了的记录规范地写进树（模拟盘上的坏数据），返回它的 URI。"""

    started = datetime(2026, 4, 13, 10, tzinfo=CST)
    writer = BehaviorDocumentWriter(tree, ProcessLocalLockStore(), clock=lambda: started + timedelta(hours=3))
    payload = occurrence_payload(
        occurred_on=started.date(),
        name="写周报",
        kind_token=token,
        started_at=started,
        last_observed_at=started + timedelta(minutes=10),
        onset_available_at=started + timedelta(seconds=2),
        basis=(),
        goal=None,
    )
    document = writer.publish(BehaviorKind.OCCURRENCE, payload)
    return str(BehaviorURI.from_address(document.address))


def _chain(error: BaseException | None) -> list[str]:
    found = []
    while error is not None:
        found.append(f"{type(error).__name__}: {error}")
        error = error.__cause__ or error.__context__
    return found


#: 词编号与四个占位标记以外的写法（含旧形状的类编号 ``s-c0001``、英文标记、序号 0）。
BAD_TOKENS = ["s-垃圾", "s-c0001", "写周报", "s-pending", "s-k0000", "s-k00002"]


@pytest.mark.parametrize("token", BAD_TOKENS)
def test_writing_a_tree_record_with_an_arbitrary_kind_token_is_refused(tmp_path: Path, token: str) -> None:
    tree = BehaviorTree(tmp_path / "tree")
    with pytest.raises(Exception) as caught:
        _publish_raw(tree, token)
    assert any("kind_token" in text for text in _chain(caught.value)), _chain(caught.value)


def _bad_record_on_disk(tmp_path: Path, token: str, monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], str]:
    tree = BehaviorTree(tmp_path / "tree")
    with monkeypatch.context() as patched:
        patched.setitem(schema_fields._VALIDATORS, BehaviorFieldType.KIND_TOKEN, lambda value, label: value)
        uri = _publish_raw(tree, token)
    with pytest.raises(Exception) as caught:
        read_series(tree, cutoff=date(2026, 4, 14))
    return _chain(caught.value), uri


@pytest.mark.parametrize("token", BAD_TOKENS)
def test_a_tree_record_with_an_arbitrary_kind_token_on_disk_is_refused(
    tmp_path: Path, token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain, _uri = _bad_record_on_disk(tmp_path, token, monkeypatch)
    assert any("kind_token" in text and repr(token) in text for text in chain), chain


@pytest.mark.xfail(strict=True, reason="阶段 C 发现：树的读路径报出坏值但不点出是哪条记录（URI / 文件），见报告")
def test_a_tree_record_with_an_arbitrary_kind_token_on_disk_names_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain, uri = _bad_record_on_disk(tmp_path, "s-垃圾", monkeypatch)
    leaf = uri.rsplit("/", 1)[-1]
    assert any(uri in text or leaf in text for text in chain), chain


# ── 选词上文（【最近的记录】）的两处偏差：钉住现状，修了就翻 ────────────────


def _records_block(text: str) -> list[str]:
    lines = text.splitlines()
    start = lines.index(next(line for line in lines if line.startswith("【最近的记录】")))
    block: list[str] = []
    for line in lines[start + 1 :]:
        if not line.strip():
            break
        block.append(line)
    return block


@pytest.mark.xfail(
    strict=True, reason="阶段 C 发现：最近的记录只在进程里记，重启之后为空（续话接不上重启前的记录），见报告"
)
def test_recent_records_survive_a_restart(tmp_path: Path) -> None:
    first = Site(tmp_path)
    try:
        first.router.picks.append(Word("s-k0002"))
        first.consume(first.turn("conv-r", at(D13, 9), "帮我写一份本周周报"))
        first.reduce()
    finally:
        first.close()
    second = Site(tmp_path)
    try:
        second.router.picks.append(Word("s-k0002"))
        second._sequences["conv-r"] = 2
        second.consume(second.turn("conv-r", at(D13, 10), "周报写得怎么样了"))
        (text,) = second.router.texts("pick")
        assert any("帮我写一份本周周报" in line for line in _records_block(text))
    finally:
        second.close()


def test_recent_records_show_the_local_time_when_messages_arrive_in_utc(site: Site) -> None:
    # 生产里会话源的消息时刻归一成 UTC（``pre/conversation/adapters.py``）；09:00（东八区）= 01:00Z
    site.router.picks.extend([Word("s-k0002"), Word("s-k0003")])
    site.consume(site.turn("conv-utc", at(D13, 9).astimezone(UTC), "帮我写一份本周周报"))
    site.reduce()
    site.consume(site.turn("conv-utc", at(D13, 10).astimezone(UTC), "把这个季度的工作写个总结"))
    text = site.router.texts("pick")[1]
    (line,) = _records_block(text)
    assert "04-13 09:00" in line, line


def test_an_in_use_state_with_a_non_text_since_is_refused(tmp_path: Path) -> None:
    store, path, raw = _state_store(tmp_path)
    path.write_text(json.dumps({**raw, "active": [{"id": "s-k0002", "since": 20260415}]}), encoding="utf-8")
    with pytest.raises(BehaviorKindStoreError):
        store.read()
