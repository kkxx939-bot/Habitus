"""行为树上一条 occurrence 算不算数的唯一规则：撞车消歧的已知重复（``original_name``
非空）、「非事件」与「续话」不算（裁定 101：续话与非事件同一种待遇，原来那条记录不改）。事件序列
（``series.reader.skip_reason``）、按天读树（经它注入的 ``admits``）、每天的语义概览判空白作废（裁定 68）都按这一条。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from habitus.behavior.kinds.ids import continued_token, lane_of_token, not_event_token

Uncounted = Literal["duplicate", "not_event", "continued"]


def uncounted(fields: Mapping[str, object]) -> Uncounted | None:
    """不算数的原因（算数是 None）。``kind_token`` 不是词编号也不是标记时抛 ``KindIdError``（由读者按自己的错误类型报）。"""

    if fields.get("original_name") is not None:
        return "duplicate"
    token = str(fields.get("kind_token") or "")
    lane = lane_of_token(token)
    if token == not_event_token(lane):
        return "not_event"
    if token == continued_token(lane):
        return "continued"
    return None


__all__ = ["Uncounted", "uncounted"]
