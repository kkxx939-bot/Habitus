"""第二步甲"应当不变"的几条（预期表第 1 节 S1–S5）：改动前后都必须通过。

- S1：会话 lane "描述这一轮"的提示词、输出形状、对提醒的回应逐字不变——按组合根装出的会话 lane 真跑一轮，钉住发给模型的
  系统提示词、这一轮的材料与要求的输出形状（快照取自基线 ``9f23a13`` 加第一步未提交改动的工作区）。
- S2：「非事件」的记录落树、下游（序列）不算。旧代码里「非事件」来自归约里的白天归类，新代码里来自选词答 ``none``——
  替身两种都答，所以这一条两边都能走通。
- S3：助手没答复也没调用的轮不出记录、不调模型。
- S4：描述失败的轮写成「错误」、进重试堆。
- S5：日历情境、封口（命中记录）、关系引擎的统计——给同一份序列和同一份概念集，结果逐项相同（快照同上）。
- S6（记忆主链）不另写测试：``tests/unit/memory``、``tests/unit/conversation``、``tests/unit/editor``、
  ``tests/unit/retrieval``、``tests/integration`` 里的现有测试不动就过。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from habitus.behavior.fusion.lanes.session import SESSION_PROMPT_VERSION, SessionMessageRole, TurnDisposition
from habitus.behavior.fusion.lanes.session.prompt import session_json_schema
from habitus.behavior.model import BehaviorKind
from habitus.behavior.retry import RetryCause, RetryKind
from habitus.conditions import LocalCalendar
from habitus.model_client import ModelAuthenticationError
from habitus.scene.concepts.calendar_situations import calendar_concepts
from habitus.scene.concepts.model import (
    ConceptDefinition,
    ConceptOrigin,
    ConceptRole,
    ConceptSet,
    ConceptSource,
)
from habitus.scene.occurrences.daily import plan_seal
from habitus.scene.occurrences.situations import SituationInputs, situation_hits
from habitus.scene.occurrences.store import ConceptHitStore
from habitus.scene.relations import LaneTests, RelationConfig, build_timelines
from habitus.series.model import SkipReason
from habitus.series.reader import read_series
from tests.unit.scene.fixtures import Site as TreeSite
from tests.unit.scene.fixtures import publish
from tests.unit.word_pick.site import NONE, Raw, Site, at


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


# ── S1：描述这一轮逐字不变 ─────────────────────────────────────────────────────

#: 基线上量出来的（改动后必须逐项相同）。封口那一组去掉日完成章摘要后在 4b3ab8d 上重量过一遍，与合并后的代码相同。
DESCRIBE_SYSTEM_PROMPT_SHA = "eeabe7b2cdb43526039e0faba1939a97372eb38691de2784875e0244dc7e5c16"
DESCRIBE_TURN_MATERIAL_SHA = "02e0d583ce4347d4ab8b141d9b3a140cfdbef4ab07be35f6eeb053672fc39bfd"
DESCRIBE_SCHEMA_SHA = "56b151a5f58629c68b6fc106aa438c0cf286c929024d95084a41169b77e55deb"
DESCRIBE_SCHEMA_WITH_REMINDERS_SHA = "2573d8381a6d4ffcaf6528915505a77ab0de60c19547dbb4c73dec5155091d4c"


def test_s1_the_describe_call_sends_the_same_prompt_material_and_output_shape(tmp_path: Path) -> None:
    site = Site(tmp_path)
    try:
        turn = site.turn("conv-s1", at(date(2026, 4, 13), 9, 30), "把仓库里的测试跑一遍，看看哪些失败了")
        (outcome,) = site.consume(turn)
        assert outcome.disposition is TurnDisposition.RECORDED
        (call,) = [item for item in site.router.calls if item.kind == "describe"]
        roles = [role for role, _ in call.messages]
        assert roles[-2:] == ["system", "user"]
        system, material = call.messages[-2][1], call.messages[-1][1]
        assert sha(system) == DESCRIBE_SYSTEM_PROMPT_SHA
        assert (
            material
            == "【这一轮】2026-04-13 09:30\n他说：把仓库里的测试跑一遍，看看哪些失败了\n助手最后说：好的，已经办好了。"
        )
        assert sha(material) == DESCRIBE_TURN_MATERIAL_SHA
        assert call.schema is not None and sha(canonical(call.schema)) == DESCRIBE_SCHEMA_SHA
    finally:
        site.close()


def test_s1_the_prompt_version_and_the_reminder_answer_shape_are_unchanged() -> None:
    assert SESSION_PROMPT_VERSION == "behavior_session_turn_prompt_v7"
    with_reminders = session_json_schema(reminder_numbers=(1, 2))
    assert with_reminders["required"] == ["name", "summary", "goal", "steps", "responses"]
    assert sha(canonical(with_reminders)) == DESCRIBE_SCHEMA_WITH_REMINDERS_SHA


# ── S2：「非事件」落树、下游不算 ────────────────────────────────────────────────


def test_s2_a_not_event_record_lands_on_the_tree_and_the_series_skips_it(tmp_path: Path) -> None:
    site = Site(tmp_path)
    try:
        site.router.pick_default = NONE
        site.record("conv-s2", at(date(2026, 4, 13), 10), "谢谢你", NONE)
        (address,) = site.session.tree.list_addresses(BehaviorKind.OCCURRENCE)
        assert site.session.tree.read(address).fields["kind_token"] == "s-非事件"
        series = read_series(site.session.tree, cutoff=date(2026, 4, 14))
        assert series.records == ()
        assert [item.reason for item in series.skipped] == [SkipReason.NOT_EVENT]
    finally:
        site.close()


# ── S3：没答复的轮不出记录 ──────────────────────────────────────────────────────


def test_s3_a_turn_without_reply_or_call_is_not_recorded_and_asks_no_model(tmp_path: Path) -> None:
    site = Site(tmp_path)
    try:
        (turn,) = site.turns(
            "conv-s3", [(0.0, SessionMessageRole.PROMPT, "帮我查一下明天的天气", False)], at(date(2026, 4, 13), 11)
        )
        (outcome,) = site.consume(turn)
        assert outcome.disposition is TurnDisposition.UNANSWERED
        assert site.router.calls == []
        assert site.session.judgements.list() == ()
        site.reduce()
        assert site.session.tree.list_addresses(BehaviorKind.OCCURRENCE) == ()
    finally:
        site.close()


# ── S4：描述失败写成「错误」进重试堆 ───────────────────────────────────────────


def _failed_describe(tmp_path: Path, failure: object, cause: RetryCause) -> None:
    site = Site(tmp_path)
    try:
        site.router.describes = [failure, failure, failure]
        turn = site.turn("conv-s4", at(date(2026, 4, 13), 12), "把这份合同审一遍")
        (outcome,) = site.consume(turn)
        assert outcome.disposition is TurnDisposition.ERROR
        (judgement,) = site.session.judgements.list()
        assert judgement["behavior"] is None
        backlog = site.session.retry_backlog
        assert backlog is not None
        (item,) = backlog.snapshot().items
        assert (item.kind, item.cause) == (RetryKind.TURN, cause)
        assert site.router.count("pick") == 0
    finally:
        site.close()


def test_s4_an_invalid_description_is_an_error_kept_for_the_retry(tmp_path: Path) -> None:
    _failed_describe(tmp_path, Raw("这不是 JSON"), RetryCause.INVALID)


def test_s4_an_unavailable_model_while_describing_is_an_error_kept_for_the_retry(tmp_path: Path) -> None:
    _failed_describe(tmp_path, ModelAuthenticationError("down"), RetryCause.UNAVAILABLE)


# ── S5：日历情境、封口、命中记录、关系引擎的统计 ───────────────────────────────

ZONE = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 4, 1, 1, 0, tzinfo=ZONE)
MINUTES, WEEKLY, RUN = "s-k0040", "s-k0041", "s-k0023"
RELATIONS = RelationConfig(slot_minutes=15, transition_window_slots=3, absence_typical_multiple=8.0, absence_min_gaps=2)
START = date(2026, 3, 1)
CUTOFF = date(2026, 3, 31)


def _behaviour(identity: str, title: str, definition: str) -> ConceptDefinition:
    return ConceptDefinition(
        name=identity,
        definition=definition,
        role=ConceptRole.BEHAVIOR,
        source=ConceptSource(ConceptOrigin.VOCABULARY),
        created_at=NOW,
        classes=(identity,),
        lane="session",
        title=title,
    )


def _tree(tmp_path: Path) -> TreeSite:
    """「写会议纪要」20 天 09:00 做、其中 18 天 09:30 接「写周报」；「写周报」另有两天 16:00；每天 08:00、18:00 跑步。"""

    site = TreeSite(tmp_path, now=NOW)
    for offset in range(30):
        day = START + timedelta(days=offset)
        if offset < 20:
            publish(site.behavior_tree, day, "写会议纪要", 9, 0, kind=MINUTES, lasts_minutes=20)
            if offset not in (6, 13):
                publish(site.behavior_tree, day, "写周报", 9, 30, kind=WEEKLY, lasts_minutes=20)
        if offset in (22, 27):
            publish(site.behavior_tree, day, "写周报", 16, 0, kind=WEEKLY, lasts_minutes=20)
        publish(site.behavior_tree, day, "晨跑", 8, 0, kind=RUN, lasts_minutes=20)
        publish(site.behavior_tree, day, "夜跑", 18, 0, kind=RUN, lasts_minutes=20)
    return site


def _concepts(days: Sequence[date]) -> ConceptSet:
    observed = {item for day in days for item in dict(LocalCalendar().of_day(day)).items()}
    return ConceptSet(
        (
            _behaviour(MINUTES, "写会议纪要", "整理会议的纪要"),
            _behaviour(WEEKLY, "写周报", "写这一周的工作周报"),
            _behaviour(RUN, "跑步", "户外或跑步机上跑步"),
            *calendar_concepts(now=NOW, observed=sorted(observed)),
        )
    )


def test_s5_calendar_situations_seal_hits_and_relation_statistics_are_unchanged(tmp_path: Path) -> None:
    site = _tree(tmp_path)
    series = read_series(site.behavior_tree, cutoff=CUTOFF)
    days = sorted({record.day for record in series.records})
    concepts = _concepts(days)
    store = ConceptHitStore(tmp_path / "scene", lane="session")
    sealed_at = datetime(2026, 4, 1, 0, 30, tzinfo=UTC)
    hits = {}
    seals = []
    for day in days:
        records = [record for record in series.records if record.day == day]
        outcome = situation_hits(concepts, SituationInputs(day=day, calendar=dict(LocalCalendar().of_day(day))))
        plan = plan_seal(store, day, records, concepts, outcome, now=sealed_at)
        seals.append(
            (
                day.isoformat(),
                # 日完成章的摘要值把这天每条记录的地址算进去，地址换成 habitus:// 之后它本来就变（裁定 112），不比
                sorted(hit.concept for hit in outcome.hits),
                sorted(outcome.checked),
                [(item.kind_token, item.classified, item.lane, list(item.situation_hits)) for item in plan.records],
            )
        )
        hits.update({item.occurrence_uri: item for item in plan.records})
    line = build_timelines(series, concepts, hits, RELATIONS)["session"]
    tests = LaneTests(line, concepts, RELATIONS, series.cutoff, established=())
    night = tests.night()
    readings = sorted(
        (
            test.key.antecedent,
            test.key.consequent,
            str(test.key.segment),
            test.key.condition,
            test.verdict.value,
            test.antecedents,
            test.blocks,
            round(test.effect.treated_rate, 6),
            round(test.effect.control_rate, 6),
            None if test.interval is None else [round(value, 6) for value in test.interval],
        )
        for test in night.tests
    )
    assert sha(canonical(seals)) == SNAPSHOT_SEALS
    assert sha(canonical(readings)) == SNAPSHOT_RELATIONS
    assert len(readings) == SNAPSHOT_RELATION_COUNT


#: 基线上量出来的（改动后必须逐项相同）。封口那一组去掉日完成章摘要后在 4b3ab8d 上重量过一遍，与合并后的代码相同。
SNAPSHOT_SEALS = "38029f789059331335361913c82f24fb8fdc1311b7a5100eeee702783779f251"
SNAPSHOT_RELATIONS = "c5b3b1579b270a51245b37603dc4ec4e010ac3a8677160340fd1c0fe7ba374ad"
SNAPSHOT_RELATION_COUNT = 137
