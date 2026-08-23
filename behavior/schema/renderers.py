"""把已经校验完成的领域字段确定性渲染为人类可读的 L2 正文。

正文只呈现语义面与一行数字面摘要；system（溯源）角色不进正文，只存在于末尾结构块。
守卫测试保证每个非 system 字段都出现在渲染结果里——防止将来加字段忘了进正文。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from behavior.model import BehaviorKind

_STATUS_TEXT = {
    "ongoing": "还在进行",
    "completed": "做完了",
    "interrupted": "被打断",
    "abandoned": "放弃了",
}
_BASIS_TEXT = {
    "observed": "看到了",
    "inferred": "推断的",
    "observation_lost": "之后没看到",
}


def render_markdown(kind: BehaviorKind, payload: Mapping[str, Any]) -> str:
    """同一份字段永远渲染出同一段正文；正文是字段的确定性函数。"""

    if kind is BehaviorKind.OCCURRENCE:
        return _render_occurrence(payload)
    return _render_gap(payload)


def _render_occurrence(payload: Mapping[str, Any]) -> str:
    status_text = _STATUS_TEXT[payload["status"]]
    basis_text = _BASIS_TEXT[payload["status_basis"]]
    lines = [
        f"# {payload['name']}",
        "",
        f"**时间** {_local(payload['started_at'])} — 最后所见 {_local(payload['last_observed_at'])}",
        f"**类型** {payload['kind_token']} · **结束** {status_text}（{basis_text}）",
        f"**首次可知** {_local(payload['onset_available_at'])}"
        + ("　·　此前被提醒过" if payload["reminded"] else ""),
    ]
    if payload["original_name"] is not None:
        lines.append(f"**原始名** {payload['original_name']}　·　撞车消歧的重复记录，统计不计入")
    if payload["goal"] is not None:
        lines.append(f"**目标** {payload['goal']}")
    if payload["place"] is not None:
        lines.append(f"**地点** {payload['place']}")
    lines.extend(["", str(payload["summary"])])
    lines.extend(["", f"**主体** {'、'.join(payload['subjects'])}"])
    if payload["basis"]:
        lines.extend(["", "## 步骤"])
        for index, step in enumerate(payload["basis"], start=1):
            lines.append(
                f"{index}. {step['semantics']}"
                f"（{_local_time(step['started_at'])}–{_local_time(step['ended_at'])}，"
                f"可知于 {_local_time(step['available_at'])}）"
            )
    return "\n".join(lines).rstrip() + "\n"


def _render_gap(payload: Mapping[str, Any]) -> str:
    lines = [
        f"# {payload['gap_kind']}",
        "",
        f"**时段** {_local(payload['started_at'])} — {_local(payload['ended_at'])}",
        "",
        "这段时间我们不知道发生了什么。"
        if payload["gap_kind"] == "未观测"
        else "这段时间观测到了内容，但没能读懂。",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _local(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _local_time(value: datetime) -> str:
    return value.strftime("%H:%M:%S")


__all__ = ["render_markdown"]
