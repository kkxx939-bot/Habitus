"""Behavior Event/Episode/Outcome 到三类 PredictionSample 的确定性投影测试。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from behavior import BehaviorDocumentWriter, BehaviorKind, BehaviorTree, BehaviorURI
from infrastructure.store.locks.process_local import ProcessLocalLockStore
from prediction import (
    PredictionContext,
    PredictionKind,
    PredictionPublisher,
    PredictionSampleView,
    PredictionTree,
)
from prediction.projection import BehaviorPredictionProjector

UTC = timezone.utc


def _action(
    action_id: str,
    sequence: int,
    semantics: str,
    *,
    minute: int,
    second: int = 0,
    available_delay_seconds: int = 0,
) -> dict[str, Any]:
    started_at = datetime(2026, 8, 8, 10, minute, second, tzinfo=UTC)
    return {
        "action_id": action_id,
        "sequence": sequence,
        "actor": "主人",
        "action_type": "device_control" if "空调" in semantics else "daily_activity",
        "semantics": semantics,
        "target_refs": ["空调"] if "空调" in semantics else [semantics],
        "method": None,
        "parameters": {"mode": "cool"} if semantics == "打开空调" else {},
        "started_at": started_at,
        "ended_at": started_at + timedelta(seconds=20),
        "available_at": started_at + timedelta(seconds=available_delay_seconds),
        "status": "completed",
        "knowledge_state": "observed",
        "evidence_refs": [f"frame:{action_id}"],
    }


def _event_payload(
    name: str,
    *,
    minute: int,
    actions: list[dict[str, Any]] | None = None,
    summary: str | None = None,
    facts: list[str] | None = None,
    onset: str = "主人开始执行当前事件",
) -> dict[str, Any]:
    started_at = datetime(2026, 8, 8, 10, minute, tzinfo=UTC)
    resolved_actions = actions or [_action("act_0001", 1, name, minute=minute, second=10)]
    return {
        "event_date": date(2026, 8, 8),
        "event_name": name,
        "started_at": started_at,
        "onset_semantics": onset,
        "onset_available_at": started_at,
        "ended_at": started_at + timedelta(minutes=4),
        "participants": ["主人"],
        "semantic_summary": summary or f"主人完成了{name}。",
        "semantic_facts": facts or [f"{name}已经完成"],
        "trigger": "主人已经回家",
        "goal": "完成回家后的生活安排",
        "constraints": [],
        "actions": resolved_actions,
        "status": "completed",
        "closure_reason": None,
        "evidence_refs": [f"frame:event-{minute}"],
        "source_refs": [f"fusion:event-{minute}"],
        "fusion_version": "fusion-v1",
        "confidence": 0.95,
        "conflicts": [],
    }


def _writer(tmp_path):
    tree = BehaviorTree(tmp_path / "behavior")
    writer = BehaviorDocumentWriter(
        tree,
        ProcessLocalLockStore(),
        clock=lambda: datetime(2026, 8, 8, 12, tzinfo=UTC),
    )
    return tree, writer


def _uri(document) -> str:
    return str(BehaviorURI.from_address(document.address))


def test_event_projection_keeps_future_action_and_final_event_facts_out_of_input(tmp_path) -> None:
    tree, writer = _writer(tmp_path)
    event = writer.publish(
        BehaviorKind.EVENT,
        _event_payload(
            "主人查看后打开空调",
            minute=30,
            actions=[
                _action("act_0001", 1, "看向空调", minute=31),
                _action("act_0002", 2, "打开空调", minute=32),
            ],
            summary="主人查看空调后打开空调并感到舒适。",
            facts=["空调已经启动", "主人后来感觉舒适"],
        ),
    )
    documents = BehaviorPredictionProjector(tree).project_event(_uri(event))

    assert [document.kind for document in documents] == [PredictionKind.TRANSITION] * 3
    first_input = str(documents[0].fields["input"])
    second_input = str(documents[1].fields["input"])
    terminal_input = str(documents[2].fields["input"])
    assert "打开空调" not in first_input
    assert "感到舒适" not in first_input
    assert "后来感觉舒适" not in first_input
    assert documents[0].fields["label"]["semantics"] == "看向空调"
    assert "看向空调" in second_input
    assert "打开空调" not in second_input
    assert "看向空调" in terminal_input and "打开空调" in terminal_input
    runtime_context = PredictionContext.from_document(documents[1]).to_fields()
    assert set(runtime_context) == {"prediction_type", "prediction_scope", "anchor", "input"}
    assert "label" not in runtime_context and "provenance" not in runtime_context
    assert documents == BehaviorPredictionProjector(tree).project_event(_uri(event))


def test_event_projection_skips_action_started_before_delayed_event_onset(tmp_path) -> None:
    tree, writer = _writer(tmp_path)
    payload = _event_payload(
        "延迟识别事件",
        minute=30,
        actions=[
            _action("act_0001", 1, "先拿起杯子", minute=30, second=1),
            _action("act_0002", 2, "随后喝水", minute=30, second=10),
        ],
    )
    payload["onset_available_at"] = datetime(2026, 8, 8, 10, 30, 5, tzinfo=UTC)
    event = writer.publish(BehaviorKind.EVENT, payload)

    samples = BehaviorPredictionProjector(tree).project_event(_uri(event))

    assert [sample.fields["anchor"]["prefix_length"] for sample in samples] == [1, 2]
    assert samples[0].fields["label"]["semantics"] == "随后喝水"
    assert samples[-1].fields["label"]["target_kind"] == "terminal"


def test_event_projection_skips_only_target_started_before_previous_action_was_observed(
    tmp_path,
) -> None:
    tree, writer = _writer(tmp_path)
    event = writer.publish(
        BehaviorKind.EVENT,
        _event_payload(
            "动作识别发生重叠",
            minute=30,
            actions=[
                _action(
                    "act_0001",
                    1,
                    "拿起杯子",
                    minute=30,
                    second=1,
                    available_delay_seconds=5,
                ),
                _action("act_0002", 2, "走向水池", minute=30, second=3),
                _action("act_0003", 3, "开始接水", minute=30, second=10),
            ],
        ),
    )

    samples = BehaviorPredictionProjector(tree).project_event(_uri(event))

    assert [sample.fields["anchor"]["prefix_length"] for sample in samples] == [0, 1, 3]
    assert [sample.fields["label"]["semantics"] for sample in samples[:-1]] == [
        "拿起杯子",
        "开始接水",
    ]
    assert samples[1].fields["anchor"]["cutoff_at"] == "2026-08-08T10:30:03.000000Z"


def test_episode_projection_skips_only_event_started_before_previous_event_was_observed(
    tmp_path,
) -> None:
    tree, writer = _writer(tmp_path)
    first_payload = _event_payload("较晚才识别的取水事件", minute=30)
    first_payload["onset_available_at"] = datetime(2026, 8, 8, 10, 32, tzinfo=UTC)
    event_documents = [
        writer.publish(BehaviorKind.EVENT, first_payload),
        writer.publish(BehaviorKind.EVENT, _event_payload("重叠开始的查看消息事件", minute=31)),
        writer.publish(BehaviorKind.EVENT, _event_payload("后续准备晚餐事件", minute=40)),
    ]
    event_uris = [_uri(document) for document in event_documents]
    episode = writer.publish(
        BehaviorKind.EPISODE,
        {
            "episode_date": date(2026, 8, 8),
            "episode_name": "带延迟识别的晚间过程",
            "started_at": datetime(2026, 8, 8, 10, 30, tzinfo=UTC),
            "ended_at": datetime(2026, 8, 8, 10, 50, tzinfo=UTC),
            "participants": ["主人"],
            "goal": "完成取水并准备晚餐",
            "semantic_summary": "多个日常事件重叠发生，部分事件延迟识别。",
            "status": "completed",
            "closure_reason": "晚餐准备完成",
            "ordered_event_uris": event_uris,
            "phases": [
                {
                    "phase_id": f"phase_{index:04d}",
                    "sequence": index,
                    "semantics": semantics,
                    "status": "completed",
                    "event_uris": [event_uri],
                    "confidence": 0.9,
                }
                for index, (semantics, event_uri) in enumerate(
                    zip(("取水", "查看消息", "准备晚餐"), event_uris, strict=True),
                    start=1,
                )
            ],
            "unphased_events": [],
            "transitions": [
                {
                    "from_event_uri": source,
                    "to_event_uri": target,
                    "relation": "precedes",
                }
                for source, target in zip(event_uris, event_uris[1:], strict=False)
            ],
            "key_turning_points": ["延迟识别后继续主线"],
            "outcome_uris": [],
            "evidence_refs": ["fusion:episode-delayed-recognition"],
            "evidence_coverage": 0.85,
            "confidence": 0.9,
        },
    )

    samples = BehaviorPredictionProjector(tree).project_episode(_uri(episode))
    transitions = [sample for sample in samples if sample.kind is PredictionKind.TRANSITION]

    assert [sample.fields["anchor"]["prefix_length"] for sample in transitions] == [0, 1, 3]
    assert [sample.fields["label"]["semantics"] for sample in transitions[:-1]] == [
        "主人完成了较晚才识别的取水事件。",
        "主人完成了后续准备晚餐事件。",
    ]
    assert transitions[-1].fields["label"]["target_kind"] == "terminal"


def test_complex_episode_projects_weak_phase_coverage_into_explicit_future_branches(tmp_path) -> None:
    tree, writer = _writer(tmp_path)
    event_documents = [
        writer.publish(BehaviorKind.EVENT, _event_payload("调整室内环境", minute=30)),
        writer.publish(BehaviorKind.EVENT, _event_payload("门口快递员经过", minute=40)),
        writer.publish(BehaviorKind.EVENT, _event_payload("电话打断晚间流程", minute=50)),
        writer.publish(BehaviorKind.EVENT, _event_payload("恢复并准备晚餐", minute=54)),
    ]
    event_uris = [_uri(document) for document in event_documents]
    episode = writer.publish(
        BehaviorKind.EPISODE,
        {
            "episode_date": date(2026, 8, 8),
            "episode_name": "主人回家后的复杂晚间流程",
            "started_at": datetime(2026, 8, 8, 10, 30, tzinfo=UTC),
            "ended_at": datetime(2026, 8, 8, 11, 0, tzinfo=UTC),
            "participants": ["主人"],
            "goal": "完成环境调整并准备晚餐",
            "semantic_summary": "环境调整期间出现上下文事件和电话中断，随后恢复晚餐准备。",
            "status": "completed",
            "closure_reason": "晚餐准备开始",
            "ordered_event_uris": event_uris,
            "phases": [
                {
                    "phase_id": "phase_0001",
                    "sequence": 1,
                    "semantics": "环境调整",
                    "status": "completed",
                    "event_uris": [event_uris[0]],
                    "confidence": 0.95,
                },
                {
                    "phase_id": "phase_0002",
                    "sequence": 2,
                    "semantics": "恢复主线并准备晚餐",
                    "status": "completed",
                    "event_uris": [event_uris[3]],
                    "confidence": 0.93,
                },
            ],
            "unphased_events": [
                {"event_uri": event_uris[1], "role": "contextual", "reason": "只是环境上下文"},
                {"event_uri": event_uris[2], "role": "interruption", "reason": "临时电话打断主线"},
            ],
            "transitions": [
                {"from_event_uri": event_uris[0], "to_event_uri": event_uris[2], "relation": "interrupts"},
                {"from_event_uri": event_uris[2], "to_event_uri": event_uris[3], "relation": "resumes"},
            ],
            "key_turning_points": ["电话打断后恢复任务"],
            "outcome_uris": [],
            "evidence_refs": ["fusion:episode-complex"],
            "evidence_coverage": 0.88,
            "confidence": 0.91,
        },
    )
    documents = BehaviorPredictionProjector(tree).project_episode(_uri(episode))
    transitions = tuple(document for document in documents if document.kind is PredictionKind.TRANSITION)
    trajectories = tuple(document for document in documents if document.kind is PredictionKind.TRAJECTORY)

    assert len(transitions) == 5
    assert len(trajectories) == 3
    target_context_input = str(transitions[1].fields["input"])
    assert "主人开始执行当前事件" in target_context_input
    assert "主人完成了调整室内环境" not in target_context_input
    assert "门口快递员经过" not in target_context_input

    after_first_phase = trajectories[1]
    label = after_first_phase.fields["label"]
    prediction_input = str(after_first_phase.fields["input"])
    assert [step["semantics"] for step in label["mainline"]] == ["恢复主线并准备晚餐"]
    assert [step["semantics"] for step in label["future_context"]] == ["主人完成了门口快递员经过。"]
    assert [step["semantics"] for step in label["interruptions"]] == ["主人完成了电话打断晚间流程。"]
    assert [step["semantics"] for step in label["resumptions"]] == ["主人完成了恢复并准备晚餐。"]
    assert "门口快递员经过" not in prediction_input
    assert "电话打断晚间流程" not in prediction_input
    assert "完成环境调整并准备晚餐" not in prediction_input
    assert after_first_phase.fields["anchor"]["cutoff_at"] == "2026-08-08T10:30:00.000000Z"
    assert after_first_phase.fields["anchor"]["decision_basis"] == "previous_step_started"
    assert label["mainline"][0]["started_at"] == "2026-08-08T10:54:00.000000Z"
    assert trajectories[-1].fields["label"]["terminal"]["status"] == "completed"
    terminal_input = str(trajectories[-1].fields["input"])
    assert "门口快递员经过" in terminal_input
    assert "电话打断晚间流程" in terminal_input


def test_outcome_projection_uses_pre_behavior_context_and_latest_effective_revision(
    tmp_path,
    monkeypatch,
) -> None:
    tree, writer = _writer(tmp_path)
    event = writer.publish(
        BehaviorKind.EVENT,
        _event_payload(
            "主人查看后打开空调",
            minute=30,
            actions=[
                _action("act_0001", 1, "看向空调", minute=31),
                _action("act_0002", 2, "打开空调", minute=32),
            ],
            facts=["主人开完空调后感觉舒适"],
            onset="主人进行空调操作",
        ),
    )
    event_uri = _uri(event)
    outcome = writer.publish(
        BehaviorKind.OUTCOME,
        {
            "event_date": date(2026, 8, 8),
            "event_name": event.address.name,
            "event_uri": event_uri,
            "outcomes": [
                {
                    "outcome_id": "out_0001",
                    "occurred_at": datetime(2026, 8, 8, 10, 36, tzinfo=UTC),
                    "outcome_type": "human_feedback",
                    "target_type": "action",
                    "target_action_id": "act_0002",
                    "semantics": "主人感觉温度舒适",
                    "valence": "positive",
                    "knowledge_state": "reported",
                    "confidence": 1.0,
                    "evidence_refs": ["utterance:comfort"],
                }
            ],
        },
    )
    projector = BehaviorPredictionProjector(tree)
    first_revision_samples = projector.project_outcome(_uri(outcome))
    sample = first_revision_samples[0]

    assert "看向空调" in str(sample.fields["input"])
    assert "打开空调" not in str(sample.fields["input"])
    assert "感觉舒适" not in str(sample.fields["input"])
    assert sample.fields["label"]["outcome"]["semantics"] == "主人感觉温度舒适"
    assert sample.fields["treatment"]["semantics"] == "打开空调"
    runtime_context = PredictionContext.from_document(sample).to_fields()
    assert runtime_context["treatment"]["semantics"] == "打开空调"
    assert "source_uri" not in runtime_context["treatment"]
    assert "ended_at" not in runtime_context["treatment"]
    assert "label" not in runtime_context

    writer.append_outcomes(
        _uri(outcome),
        [
            {
                "outcome_id": "out_0002",
                "occurred_at": datetime(2026, 8, 8, 10, 45, tzinfo=UTC),
                "outcome_type": "delayed_effect",
                "target_type": "event",
                "target_action_id": None,
                "semantics": "房间温度保持稳定",
                "valence": "positive",
                "knowledge_state": "observed",
                "confidence": 0.9,
                "evidence_refs": ["sensor:temperature"],
            }
        ],
        expected_revision=1,
    )
    second_revision_samples = projector.project_outcome(_uri(outcome))

    assert len(second_revision_samples) == 2
    assert second_revision_samples[0].fields["logical_sample_id"] == sample.fields["logical_sample_id"]
    assert second_revision_samples[0].address.materialization_id != sample.address.materialization_id
    assert second_revision_samples[0].fields["lineage"]["consequence_group_id"] == sample.fields["lineage"][
        "consequence_group_id"
    ]
    assert second_revision_samples[0].fields["provenance"]["source_bindings"][1]["revision"] == 2
    event_treatment = PredictionContext.from_document(second_revision_samples[1]).to_fields()[
        "treatment"
    ]
    assert event_treatment["semantics"] == "主人进行空调操作"
    assert event_treatment["target_refs"] == []
    assert "房间温度保持稳定" not in str(event_treatment)

    prediction_tree = PredictionTree(tmp_path / "prediction-tree")
    PredictionPublisher(prediction_tree).publish((*first_revision_samples, *second_revision_samples))
    monkeypatch.setattr("prediction.view._PAGE_SIZE", 1)
    latest = PredictionSampleView(prediction_tree).latest_effective_consequences(
        projection_version="behavior-prediction-v3"
    )
    assert len(latest) == 2
    assert {item.fields["label"]["outcome"]["outcome_id"] for item in latest} == {"out_0001", "out_0002"}


def test_overlapping_actions_are_projected_as_active_without_future_completion(tmp_path) -> None:
    tree, writer = _writer(tmp_path)
    event = writer.publish(
        BehaviorKind.EVENT,
        _event_payload(
            "主人并行处理回家任务",
            minute=30,
            actions=[
                _action("act_0001", 1, "整理随身物品", minute=31),
                _action("act_0002", 2, "打开空调", minute=31, second=10),
            ],
        ),
    )

    samples = BehaviorPredictionProjector(tree).project_event(_uri(event))
    second = samples[1]
    active = second.fields["input"]["behavior_history"]["active_behaviors"]

    assert second.fields["anchor"]["cutoff_at"] == "2026-08-08T10:31:00.000000Z"
    assert second.fields["anchor"]["decision_basis"] == "previous_step_observed"
    assert second.fields["label"]["started_at"] == "2026-08-08T10:31:10.000000Z"
    assert second.fields["label"]["delay_seconds"] == 10.0
    assert len(active) == 1
    assert active[0]["semantics"] == "整理随身物品"
    assert active[0]["status"] == "active"
    assert active[0]["ended_at"] is None


def test_next_target_time_cannot_change_the_same_historical_anchor(tmp_path) -> None:
    def projected_second_action(root, *, second_minute: int, second_second: int):
        tree, writer = _writer(root)
        event = writer.publish(
            BehaviorKind.EVENT,
            _event_payload(
                "固定历史前缀",
                minute=30,
                actions=[
                    _action("act_0001", 1, "整理随身物品", minute=31),
                    _action(
                        "act_0002",
                        2,
                        "打开空调",
                        minute=second_minute,
                        second=second_second,
                    ),
                ],
            ),
        )
        return BehaviorPredictionProjector(tree).project_event(_uri(event))[1]

    overlapping = projected_second_action(tmp_path / "overlap", second_minute=31, second_second=10)
    sequential = projected_second_action(tmp_path / "sequential", second_minute=32, second_second=0)

    assert overlapping.fields["anchor"] == sequential.fields["anchor"]
    assert overlapping.fields["input"] == sequential.fields["input"]


def test_outcome_during_action_uses_pre_action_consequence_anchor(tmp_path) -> None:
    tree, writer = _writer(tmp_path)
    event = writer.publish(
        BehaviorKind.EVENT,
        _event_payload(
            "主人打开空调",
            minute=30,
            actions=[_action("act_0001", 1, "打开空调", minute=31)],
        ),
    )
    outcome = writer.publish(
        BehaviorKind.OUTCOME,
        {
            "event_date": date(2026, 8, 8),
            "event_name": event.address.name,
            "event_uri": _uri(event),
            "outcomes": [
                {
                    "outcome_id": "out_0001",
                    "occurred_at": datetime(2026, 8, 8, 10, 31, 30, tzinfo=UTC),
                    "outcome_type": "human_feedback",
                    "target_type": "action",
                    "target_action_id": "act_0001",
                    "semantics": "主人感受到凉风",
                    "valence": "positive",
                    "knowledge_state": "observed",
                    "confidence": 0.9,
                    "evidence_refs": ["sensor:airflow"],
                }
            ],
        },
    )

    sample = BehaviorPredictionProjector(tree).project_outcome(_uri(outcome))[0]

    assert sample.fields["anchor"]["cutoff_at"] == "2026-08-08T10:31:00.000000Z"
    assert sample.fields["anchor"]["decision_basis"] == "treatment_observed"
    assert "打开空调" not in str(sample.fields["input"])
    assert sample.fields["treatment"]["semantics"] == "打开空调"
    assert sample.fields["treatment"]["status"] == "active"
    assert sample.fields["treatment"]["ended_at"] is None
    assert sample.fields["treatment"]["available_at"] == "2026-08-08T10:31:00.000000Z"
    assert sample.fields["label"]["outcome"]["semantics"] == "主人感受到凉风"


def test_action_identity_availability_controls_context_and_consequence_cutoff(tmp_path) -> None:
    tree, writer = _writer(tmp_path)
    action = _action(
        "act_0001",
        1,
        "打开空调",
        minute=31,
        available_delay_seconds=5,
    )
    event = writer.publish(
        BehaviorKind.EVENT,
        _event_payload("主人操作空调", minute=30, actions=[action]),
    )
    outcome = writer.publish(
        BehaviorKind.OUTCOME,
        {
            "event_date": date(2026, 8, 8),
            "event_name": event.address.name,
            "event_uri": _uri(event),
            "outcomes": [
                {
                    "outcome_id": "out_0001",
                    "occurred_at": datetime(2026, 8, 8, 10, 31, 30, tzinfo=UTC),
                    "outcome_type": "state_change",
                    "target_type": "action",
                    "target_action_id": "act_0001",
                    "semantics": "空调开始送风",
                    "valence": "neutral",
                    "knowledge_state": "observed",
                    "confidence": 0.95,
                    "evidence_refs": ["sensor:airflow"],
                }
            ],
        },
    )

    consequence = BehaviorPredictionProjector(tree).project_outcome(_uri(outcome))[0]

    assert consequence.fields["anchor"]["cutoff_at"] == "2026-08-08T10:31:05.000000Z"
    assert consequence.fields["anchor"]["decision_basis"] == "treatment_observed"
    assert consequence.fields["treatment"]["started_at"] == "2026-08-08T10:31:00.000000Z"
    assert consequence.fields["treatment"]["available_at"] == "2026-08-08T10:31:05.000000Z"
    assert PredictionContext.from_document(consequence).to_fields()["treatment"][
        "semantics"
    ] == "打开空调"
