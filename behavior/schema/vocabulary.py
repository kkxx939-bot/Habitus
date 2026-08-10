"""行为语义树受控词表；枚举取值属于不可配置的代码不变量。"""

from __future__ import annotations

import re

RECORD_ID = re.compile(r"^[a-z][a-z0-9_-]{0,127}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

EVENT_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted", "partial"})
ACTION_STATUSES = frozenset({"observed", "completed", "failed", "cancelled", "interrupted"})
KNOWLEDGE_STATES = frozenset({"observed", "reported", "inferred", "corrected"})
OUTCOME_TYPES = frozenset(
    {
        "state_change",
        "task_result",
        "human_response",
        "human_feedback",
        "correction",
        "delayed_effect",
        "safety_result",
    }
)
OUTCOME_TARGET_TYPES = frozenset({"event", "action"})
OUTCOME_VALENCES = frozenset({"positive", "neutral", "negative", "mixed", "not_applicable"})
EPISODE_STATUSES = frozenset({"completed", "failed", "cancelled", "partial"})
PHASE_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted", "partial"})
UNPHASED_EVENT_ROLES = frozenset({"contextual", "parallel", "interruption", "uncertain"})
TRANSITION_TYPES = frozenset({"precedes", "enables", "causes", "interrupts", "resumes", "completes", "fails"})

__all__ = [
    "ACTION_STATUSES",
    "EPISODE_STATUSES",
    "EVENT_STATUSES",
    "KNOWLEDGE_STATES",
    "OUTCOME_TARGET_TYPES",
    "OUTCOME_TYPES",
    "OUTCOME_VALENCES",
    "PHASE_STATUSES",
    "RECORD_ID",
    "SHA256",
    "TRANSITION_TYPES",
    "UNPHASED_EVENT_ROLES",
]
