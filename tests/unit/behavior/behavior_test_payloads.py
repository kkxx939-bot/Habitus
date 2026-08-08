"""Behavior 核心合同测试共享的严格载荷工厂。"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timezone
from typing import Any

from behavior.model import BehaviorAddress
from behavior.uri import BehaviorURI


def event_payload(
    name: str = "主人回家后查看并打开空调",
    *,
    minute: int = 30,
) -> dict[str, Any]:
    return {
        "event_date": date(2026, 8, 8),
        "event_name": name,
        "started_at": datetime(2026, 8, 8, 10, minute, tzinfo=timezone.utc),
        "ended_at": datetime(2026, 8, 8, 10, minute + 2, tzinfo=timezone.utc),
        "participants": ["主人"],
        "semantic_summary": f"{name}。",
        "semantic_facts": ["主人已经回家", "主人执行了明确动作"],
        "trigger": "主人到家",
        "goal": "完成回家后的生活安排",
        "constraints": [],
        "actions": [
            {
                "action_id": "act_0001",
                "sequence": 1,
                "actor": "主人",
                "action_type": "device_control",
                "semantics": "打开空调",
                "target_refs": ["空调"],
                "method": None,
                "parameters": {"mode": "cool"},
                "started_at": None,
                "ended_at": None,
                "status": "completed",
                "knowledge_state": "observed",
                "evidence_refs": ["frame:0001"],
            }
        ],
        "status": "completed",
        "closure_reason": None,
        "evidence_refs": ["frame:0001"],
        "source_refs": ["fusion:0001"],
        "fusion_version": "fusion-v1",
        "confidence": 0.96,
        "conflicts": [],
    }


def outcome_payload(event_uri: str, name: str = "主人回家后查看并打开空调") -> dict[str, Any]:
    return {
        "event_date": date(2026, 8, 8),
        "event_name": name,
        "event_uri": event_uri,
        "outcomes": [
            {
                "outcome_id": "out_0001",
                "occurred_at": datetime(2026, 8, 8, 10, 35, tzinfo=timezone.utc),
                "outcome_type": "human_feedback",
                "target_type": "action",
                "target_action_id": "act_0001",
                "semantics": "主人接受了自动打开空调",
                "valence": "positive",
                "knowledge_state": "reported",
                "confidence": 1.0,
                "evidence_refs": ["utterance:0001"],
            }
        ],
    }


def episode_payload(
    first_event_uri: str,
    second_event_uri: str,
    outcome_uri: str,
) -> dict[str, Any]:
    return {
        "episode_date": date(2026, 8, 8),
        "episode_name": "主人回家后的晚间生活",
        "started_at": datetime(2026, 8, 8, 10, 30, tzinfo=timezone.utc),
        "ended_at": datetime(2026, 8, 8, 11, 30, tzinfo=timezone.utc),
        "participants": ["主人"],
        "goal": "完成回家后的环境调整和晚餐准备",
        "semantic_summary": "主人回家后调整环境、洗手并开始准备晚餐。",
        "status": "completed",
        "closure_reason": "晚餐准备完成",
        "ordered_event_uris": [first_event_uri, second_event_uri],
        "phases": [
            {
                "phase_id": "phase_0001",
                "sequence": 1,
                "semantics": "环境调整",
                "status": "completed",
                "event_uris": [first_event_uri],
                "confidence": 0.96,
            },
            {
                "phase_id": "phase_0002",
                "sequence": 2,
                "semantics": "个人整理",
                "status": "completed",
                "event_uris": [second_event_uri],
                "confidence": 0.93,
            },
        ],
        "unphased_events": [],
        "transitions": [
            {
                "from_event_uri": first_event_uri,
                "to_event_uri": second_event_uri,
                "relation": "precedes",
            }
        ],
        "key_turning_points": ["完成环境调整后开始个人整理"],
        "outcome_uris": [outcome_uri],
        "evidence_refs": ["fusion:episode-0001"],
        "evidence_coverage": 0.9,
        "confidence": 0.92,
    }


def event_uri(name: str = "主人回家后查看并打开空调", *, minute: int = 30) -> str:
    started_at = datetime(2026, 8, 8, 10, minute, tzinfo=timezone.utc)
    return str(BehaviorURI.from_address(BehaviorAddress.event(started_at.date(), name, started_at)))


def outcome_uri(name: str = "主人回家后查看并打开空调", *, minute: int = 30) -> str:
    started_at = datetime(2026, 8, 8, 10, minute, tzinfo=timezone.utc)
    return str(BehaviorURI.from_address(BehaviorAddress.outcome(started_at.date(), name, started_at)))


def episode_storage_payload(
    first_event_uri: str,
    second_event_uri: str,
    target_outcome_uri: str,
) -> dict[str, Any]:
    payload = deepcopy(episode_payload(first_event_uri, second_event_uri, target_outcome_uri))
    payload["outcome_snapshots"] = [
        {
            "uri": target_outcome_uri,
            "revision": 1,
            "digest": "0" * 64,
        }
    ]
    return payload


__all__ = [
    "episode_payload",
    "episode_storage_payload",
    "event_payload",
    "event_uri",
    "outcome_payload",
    "outcome_uri",
]
