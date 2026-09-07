"""情景归组：schema 钉死穷尽性、装配层只降级不整批拒、提示词渲染确定、服务层有界重试。"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Any

import pytest

from habitus.model_client import ChatClient, ModelTransportError, StructuredChatClient
from habitus.scene.grouping import (
    SCENE_VERSION,
    DraftRelation,
    LastOccurrenceReference,
    LLMSceneGrouper,
    OccurrenceRow,
    PendingReference,
    SceneGroupingAssemblyError,
    SceneGroupingConfig,
    SceneGroupingInput,
    SceneGroupingLimitError,
    SceneReference,
    assemble_grouping,
    build_request,
    grouping_json_schema,
    render_occurrences,
    render_references,
)
from habitus.scene.grouping.model import GapRow
from habitus.scene.model import SceneLinkType, SceneRole
from tests.unit.behavior.test_kinds import ScriptedProvider
from tests.unit.scene.fixtures import structured_client
from tests.unit.scene.scene_payloads import DAY, local, occurrence_uri

YESTERDAY = DAY - timedelta(days=1)


def row(no: int, name: str, hour: int, minute: int, *, kind: str | None = None, **overrides: Any) -> OccurrenceRow:
    started = local(hour, minute)
    values: dict[str, Any] = {
        "no": no,
        "uri": occurrence_uri(name, started),
        "name": name,
        "kind_token": kind or name,
        "started_at": started,
        "last_observed_at": started + timedelta(minutes=5),
        "status": "completed",
        "status_basis": "observed",
        "goal": None,
        "summary": f"{name}的一句话摘要",
        "subjects": ("家庭成员A",),
        "basis": (),
    }
    values.update(overrides)
    return OccurrenceRow(**values)


def payload(**overrides: Any) -> SceneGroupingInput:
    rows = (
        row(1, "去超市买菜", 15, 20, goal="买晚饭食材", basis=("挑菜", "买肉")),
        row(2, "与Tasha交谈", 19, 40),
        row(3, "查询做汤配方", 19, 41, links=(("results_from", 2),)),
        row(4, "做晚饭", 19, 52, status="ongoing", status_basis="observation_lost", goal="做汤和炒菜"),
        row(5, "使用手机", 20, 15, links=(("concurrent_with", 4),)),
        row(6, "喝水", 20, 35),
    )
    previous_started = local(10, 0, day=YESTERDAY)
    values: dict[str, Any] = {
        "day": DAY,
        "subject": "家庭成员A",
        "occurrences": rows,
        "gaps": (GapRow(local(17, 0), local(17, 30)),),
        "scene_references": (
            SceneReference(1, "scene://scenes/2026/08/15/去药店买药--2026-08-15T10:00:00+08:00.md", YESTERDAY, "去药店买药", previous_started, ("买到了药",), ()),
        ),
        "pending_references": (
            PendingReference(1, "预约了周六理发", occurrence_uri("打电话预约理发", local(9, 0, day=YESTERDAY)), local(9, 0, day=YESTERDAY), "scene://scenes/2026/08/15/预约理发--2026-08-15T09:00:00+08:00.md", YESTERDAY),
        ),
        "last_occurrences": (LastOccurrenceReference("喝水", 1, None, today_nos=(6,)),),
    }
    values.update(overrides)
    return SceneGroupingInput(**values)


def relation(kind: str, **target: int) -> dict[str, Any]:
    return {"kind": kind, "occurrence_no": None, "scene_no": None, "reference_no": None, "pending_no": None, **target}


def good_output() -> dict[str, Any]:
    return {
        "scenes": [
            {
                "scene_no": 2,
                "label": "准备晚饭",
                "effects": ["晚饭做到一半"],
                "pending_effects": [],
                "relations": [relation("needs", occurrence_no=1), relation("results_from", reference_no=1), relation("needs", pending_no=1)],
            },
            {
                "scene_no": 1,
                "label": "去超市采购晚饭食材",
                "effects": ["买到了晚饭食材"],
                "pending_effects": [{"text": "家里有今晚要用的食材", "occurrence_no": 1}],
                "relations": [],
            },
        ],
        "assignments": [
            {"no": 1, "scene_no": 1, "role": "essential"},
            {"no": 2, "scene_no": 2, "role": "essential"},
            {"no": 3, "scene_no": 2, "role": "essential"},
            {"no": 4, "scene_no": 2, "role": "essential"},
            {"no": 5, "scene_no": 2, "role": "irrelevant"},
            {"no": 6, "scene_no": None, "role": None},
        ],
    }


# ── schema ───────────────────────────────────────────────────────────────────


def test_schema_pins_the_assignment_count_to_the_day() -> None:
    schema = grouping_json_schema(6)
    assert schema["properties"]["assignments"]["minItems"] == 6
    assert schema["properties"]["assignments"]["maxItems"] == 6
    assert "maxItems" not in grouping_json_schema(3)["properties"]["scenes"]
    with pytest.raises(ValueError):
        grouping_json_schema(0)


# ── assembly ─────────────────────────────────────────────────────────────────


def test_good_output_assembles_ordered_scenes_and_all_four_target_kinds() -> None:
    assembly = assemble_grouping(good_output(), payload())

    assert [draft.label for draft in assembly.scenes] == ["去超市采购晚饭食材", "准备晚饭"]
    dinner = assembly.scenes[1]
    assert dinner.members == ((2, SceneRole.ESSENTIAL), (3, SceneRole.ESSENTIAL), (4, SceneRole.ESSENTIAL), (5, SceneRole.IRRELEVANT))
    assert dinner.relations == (
        DraftRelation(SceneLinkType.NEEDS, occurrence_no=1),
        DraftRelation(SceneLinkType.RESULTS_FROM, reference_no=1),
        DraftRelation(SceneLinkType.NEEDS, pending_no=1),
    )
    assert assembly.scenes[0].pending_effects == (("家里有今晚要用的食材", 1),)
    assert assembly.unassigned == (6,)
    assert assembly.signals == ()


def test_bookkeeping_slips_degrade_with_signals_instead_of_rejecting_the_batch() -> None:
    output = good_output()
    scenes = output["scenes"]
    dinner, shopping = scenes
    dinner["relations"] = [
        relation("needs", occurrence_no=2),  # 自己的成员
        relation("needs", scene_no=2),  # 自己
        relation("results_from", scene_no=9),  # 不存在
        relation("needs", occurrence_no=1),
        relation("needs", occurrence_no=1),  # 重复
        relation("needs", reference_no=7),  # 不存在的 C
        relation("needs", occurrence_no=1, pending_no=1),  # 两个目标
        {"kind": "continues", "occurrence_no": 1, "scene_no": None, "reference_no": None, "pending_no": None},
    ]
    shopping["relations"] = [relation("needs", scene_no=2)]  # 指向更晚的情景
    shopping["pending_effects"] = [
        {"text": "家里有今晚要用的食材", "occurrence_no": 1},
        {"text": "家里有今晚要用的食材", "occurrence_no": 1},
        {"text": "别人的前提", "occurrence_no": 4},
        {"text": "   ", "occurrence_no": 1},
    ]
    shopping["effects"] = ["买到了晚饭食材", "买到了晚饭食材", ""]
    scenes.append({"scene_no": 3, "label": "空情景", "effects": [], "pending_effects": [], "relations": []})
    output["assignments"][5] = {"no": 6, "scene_no": 8, "role": "essential"}
    output["assignments"][4] = {"no": 5, "scene_no": 2, "role": None}

    assembly = assemble_grouping(output, payload())

    dinner_draft = assembly.scenes[1]
    assert dinner_draft.relations == (DraftRelation(SceneLinkType.NEEDS, occurrence_no=1),)
    assert assembly.scenes[0].relations == ()
    assert assembly.scenes[0].pending_effects == (("家里有今晚要用的食材", 1),)
    assert assembly.scenes[0].effects == ("买到了晚饭食材",)
    assert (5, SceneRole.OPTIONAL) in dinner_draft.members
    assert assembly.unassigned == (6,)
    assert len(assembly.scenes) == 2
    kinds = {note.split(":")[0] for note in assembly.signals}
    assert kinds == {"relation_dropped", "pending_dropped", "effect_dropped", "scene_dropped", "assignment_degraded", "role_degraded"}


def test_same_identity_and_start_merge_into_one_scene_keeping_everything() -> None:
    output = good_output()
    # 标签只差大小写/空白：存储层按 NFC+casefold 判同一身份，装配层必须按同一口径合并
    output["scenes"].append(
        {
            "scene_no": 3,
            "label": " 准备晚饭 ",
            "effects": ["汤煮好了"],
            "pending_effects": [{"text": "剩了半锅汤", "occurrence_no": 2}],
            "relations": [relation("needs", occurrence_no=1)],
        }
    )
    output["assignments"][1] = {"no": 2, "scene_no": 3, "role": "essential"}
    # #2 与 #3 同刻开始：情景 2（#3 起）与情景 3（#2 起）同身份同刻——地址上就是同一件事
    rows = list(payload().occurrences)
    rows[2] = row(3, "查询做汤配方", 19, 40, links=(("results_from", 2),))

    assembly = assemble_grouping(output, payload(occurrences=tuple(rows)))

    assert [draft.label for draft in assembly.scenes] == ["去超市采购晚饭食材", "准备晚饭"]
    dinner = assembly.scenes[1]
    assert dinner.member_nos == (2, 3, 4, 5)
    assert dinner.effects == ("晚饭做到一半", "汤煮好了")
    assert dinner.pending_effects == (("剩了半锅汤", 2),)
    assert dinner.relations.count(DraftRelation(SceneLinkType.NEEDS, occurrence_no=1)) == 1  # 并入后去重
    assert any(note.startswith("scene_merged") for note in assembly.signals)


def test_texts_are_cleaned_like_the_storage_layer_and_same_second_targets_are_allowed() -> None:
    output = good_output()
    output["scenes"][1]["label"] = "去超市\u200b采购晚饭食材"
    output["scenes"][1]["effects"] = ["买到了\u200b晚饭食材", "\u200b"]
    output["scenes"][0]["relations"] = [relation("needs", occurrence_no=1)]
    rows = list(payload().occurrences)
    rows[0] = row(1, "去超市买菜", 19, 40, goal="买晚饭食材")  # 与情景 2 首成员同秒开始
    rows[1] = row(2, "与Tasha交谈", 19, 40)
    rows.sort(key=lambda item: item.started_at)

    assembly = assemble_grouping(output, payload(occurrences=tuple(rows)))

    shopping, dinner = assembly.scenes
    assert shopping.label == "去超市 采购晚饭食材"
    assert shopping.effects == ("买到了 晚饭食材",)
    assert dinner.relations[0] == DraftRelation(SceneLinkType.NEEDS, occurrence_no=1)  # 同秒的目标不丢
    assert any(note.startswith("effect_dropped") for note in assembly.signals)


def test_an_unaddressable_label_drops_the_scene_and_frees_its_members() -> None:
    output = good_output()
    output["scenes"][1]["label"] = "超" * 200

    assembly = assemble_grouping(output, payload())

    assert [draft.label for draft in assembly.scenes] == ["准备晚饭"]
    assert assembly.unassigned == (1, 6)


def test_broken_exhaustiveness_is_the_only_hard_failure() -> None:
    output = good_output()
    output["assignments"].reverse()
    reordered = assemble_grouping(output, payload())  # 排列只是乱序：机械排好、留信号
    assert reordered.unassigned == (6,)
    assert any(note.startswith("assignments_reordered") for note in reordered.signals)
    output["assignments"][0] = {"no": 2, "scene_no": 2, "role": "essential"}  # 重号：#1 缺了
    with pytest.raises(SceneGroupingAssemblyError, match="exactly once"):
        assemble_grouping(output, payload())
    with pytest.raises(SceneGroupingAssemblyError):
        assemble_grouping([], payload())


# ── prompt ───────────────────────────────────────────────────────────────────


def test_rendering_numbers_occurrences_and_lists_every_reference_channel() -> None:
    body = render_occurrences(payload())
    assert "#1 15:20:00–15:25:00  去超市买菜" in body
    assert "results_from→#2" in body and "concurrent_with→#4" in body
    assert "观测空白" in body and "17:00:00–17:30:00" in body
    references = render_references(payload())
    assert "C1  2026-08-15 10:00:00  去药店买药  留下：买到了药" in references
    assert "P1  2026-08-15  预约了周六理发" in references
    assert "#6（喝水）：上一次 1 天前" in references
    assert "待用：" not in references  # C 行不再列待用前提：P 是唯一的待用通道
    assert "【先前的事】（无）" in render_references(payload(scene_references=(), pending_references=(), last_occurrences=()))
    request = build_request(payload())
    assert [message.role for message in request.messages] == ["system", "user"]
    assert render_occurrences(payload()) == body  # 确定性：同输入同字节
    assert SCENE_VERSION.startswith("scene_grouping_prompt_v1+schema")


def test_long_summaries_and_step_lists_are_budgeted_in_the_rendering_only() -> None:
    long = payload(occurrences=(row(1, "长链", 8, 0, summary="很" * 400, basis=tuple(f"步{i}" for i in range(20))),))
    body = render_occurrences(long)
    assert "…（共 20 步）" in body
    assert "很" * 241 not in body


# ── model / input guards ────────────────────────────────────────────────────


def test_input_rejects_misnumbered_or_unordered_rows() -> None:
    with pytest.raises(ValueError, match="numbered"):
        payload(occurrences=(row(2, "a", 8, 0),))
    with pytest.raises(ValueError, match="ordered"):
        payload(occurrences=(row(1, "a", 9, 0), row(2, "b", 8, 0)))
    with pytest.raises(ValueError, match="exactly one target"):
        DraftRelation(SceneLinkType.NEEDS)


# ── service ──────────────────────────────────────────────────────────────────


def test_the_service_returns_an_assembly_and_enforces_its_limits() -> None:
    provider = ScriptedProvider([good_output()])
    grouper = LLMSceneGrouper(structured_client(provider), config=SceneGroupingConfig(transient_retry_delay_seconds=0))

    assembly = asyncio.run(grouper.group(payload()))

    assert [draft.label for draft in assembly.scenes] == ["去超市采购晚饭食材", "准备晚饭"]
    assert provider.calls == 1
    assert "#6 20:35:00–20:40:00  喝水" in provider.prompts[0]
    assert grouper.version == SCENE_VERSION
    small = LLMSceneGrouper(structured_client(provider), config=SceneGroupingConfig(max_occurrences_per_call=2))
    with pytest.raises(SceneGroupingLimitError, match="max_occurrences_per_call"):
        asyncio.run(small.group(payload()))
    tight = LLMSceneGrouper(structured_client(provider), config=SceneGroupingConfig(max_prompt_chars=1_000))
    with pytest.raises(SceneGroupingLimitError, match="max_prompt_chars"):
        asyncio.run(tight.group(payload()))


def test_transient_errors_are_retried_a_bounded_number_of_times() -> None:
    class Flaky(StructuredChatClient):
        failures = 2
        attempts = 0

        async def complete_json_async(self, request, **kwargs):  # type: ignore[override]
            Flaky.attempts += 1
            if Flaky.attempts <= Flaky.failures:
                raise ModelTransportError("boom")
            return await super().complete_json_async(request, **kwargs)

    provider = ScriptedProvider([good_output()])
    client = Flaky(ChatClient(structured_client(provider).client.config, provider), validation_retries=1)
    grouper = LLMSceneGrouper(client, config=SceneGroupingConfig(transient_retries=2, transient_retry_delay_seconds=0))
    assert len(asyncio.run(grouper.group(payload())).scenes) == 2
    Flaky.attempts = 0
    exhausted = LLMSceneGrouper(client, config=SceneGroupingConfig(transient_retries=1, transient_retry_delay_seconds=0))
    with pytest.raises(ModelTransportError):
        asyncio.run(exhausted.group(payload()))


def test_service_config_bounds() -> None:
    with pytest.raises(ValueError):
        SceneGroupingConfig(max_occurrences_per_call=0)
    with pytest.raises(ValueError):
        SceneGroupingConfig(transient_retry_delay_seconds=-1)
    with pytest.raises(TypeError):
        LLMSceneGrouper(object())  # type: ignore[arg-type]
    assert date(2026, 8, 16) == DAY
