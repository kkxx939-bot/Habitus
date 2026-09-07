"""把已经校验完成的情景字段确定性渲染为人类可读的 L2 正文。

正文只呈现语义面；system（溯源）不进正文。守卫测试保证每个非 system 字段都出现在渲染结果里。
边（needs / results_from）在信封层，不进正文。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from habitus.behavior.uri import BehaviorURI
from habitus.scene.model import SceneRole

_ROLE_TEXT = {SceneRole.ESSENTIAL: "必要", SceneRole.OPTIONAL: "顺带", SceneRole.IRRELEVANT: "无关"}


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        f"# {payload['label']}",
        "",
        f"**时段** {_local(payload['started_at'])} — {_local(payload['ended_at'])}",
        "",
        f"## 成员（共 {len(payload['members'])} 条）",
    ]
    for member in payload["members"]:
        lines.append(f"- {_ROLE_TEXT[SceneRole(member['role'])]} {_leaf(member['uri'])}")
    lines.extend(["", "## 留下的改变"])
    if payload["effects"]:
        lines.extend(f"- {item}" for item in payload["effects"])
    else:
        lines.append("- （无）")
    lines.extend(["", "## 待用前提"])
    if payload["pending_effects"]:
        lines.extend(f"- {item['text']}（由 {_leaf(item['producer_uri'])} 建立）" for item in payload["pending_effects"])
    else:
        lines.append("- （无）")
    return "\n".join(lines).rstrip() + "\n"


def _leaf(uri: str) -> str:
    """成员在正文里显示为「时刻 行为名」（按 URI 解析，不带百分号编码），完整 URI 在结构块里。

    行为名是地址的规范身份（NFC + casefold），不是 occurrence 文档里的原始写法——正文只是投影，
    真相在行为树上；投影层（M3）要显示原话应读行为树文档。
    """

    address = BehaviorURI.parse(uri).to_address()
    return f"{address.started_at.strftime('%H:%M')} {address.name}"


def _local(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


__all__ = ["render_markdown"]
