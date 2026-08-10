"""把已校验的预测样本字段确定性渲染为可读 L2 正文。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from foundation.integrity import canonicalize
from prediction.model import PredictionKind


def render_sample(kind: PredictionKind, payload: Mapping[str, Any]) -> str:
    scope = payload["prediction_scope"]
    anchor = payload["anchor"]
    frame = payload["input"]["observation_frame"]
    history = payload["input"]["behavior_history"]
    lines = [
        f"# {kind.value.title()} Sample",
        "",
        f"**Logical Sample ID:** {payload['logical_sample_id']}",
        f"**Materialization ID:** {payload['materialization_id']}",
        f"**Mode:** {scope['prediction_mode']}",
        f"**Target:** {scope['target_level']}",
        "**User Model:** implicit single owner",
        f"**Anchor:** {anchor['anchor_type']} prefix={anchor['prefix_length']} precision={anchor['precision']}",
        "",
        "## Observed Facts",
    ]
    lines.extend(f"- {fact['semantics']}" for fact in frame["facts"])
    if not frame["facts"]:
        lines.append("- None explicitly available")
    lines.extend(["", "## Behavior Prefix"])
    steps = [
        *history["completed_events"],
        *history["completed_actions"],
        *history["completed_phases"],
    ]
    lines.extend(f"- [{step['step_kind']}] {step['semantics']}" for step in steps)
    if not steps:
        lines.append("- Empty prefix")
    lines.extend(
        [
            "",
            "## Label",
            "```json",
            json.dumps(canonicalize(payload["label"]), ensure_ascii=False, sort_keys=True, indent=2),
            "```",
        ]
    )
    if kind is PredictionKind.CONSEQUENCE:
        lines.extend(
            [
                "",
                "## Treatment",
                "```json",
                json.dumps(canonicalize(payload["treatment"]), ensure_ascii=False, sort_keys=True, indent=2),
                "```",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["render_sample"]
