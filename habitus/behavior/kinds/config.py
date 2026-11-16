"""词表的全部可调参数（裁定 6：门槛做成配置，不写死）。

数值由 ``Config.behavior`` 的 ``kinds_*`` 字段经组合根注入（``runtime/behavior.py``）；这里的默认值是代码内唯一出处，
YAML 留空即取之。数值来自裁定 101 第 7 条与选词实验（``2026-11-14-词表第二步/scripts/run.py``），不从造的数据里调。
"""

from __future__ import annotations

from dataclasses import dataclass

_INTEGER_BOUNDS: tuple[tuple[str, int, int], ...] = (
    ("pick_retries", 0, 5),
    ("transient_retries", 0, 20),
    ("max_encoded_bytes", 1, 256 * 1024 * 1024),
    ("active_min_days", 1, 365),
    ("active_cap", 1, 10_000),
    ("recent_records", 0, 100),
    ("recent_same_conversation", 0, 100),
    ("earlier_turns", 0, 20),
    ("earlier_chars", 1, 100_000),
    ("prompt_max_steps", 0, 200),
    ("max_name_chars", 1, 100),
    ("candidate_slots", 1, 100),
    ("retire_days", 1, 3650),
    ("max_level", 1, 10),
)


@dataclass(frozen=True)
class BehaviorKindConfig:
    """词表的门槛与运维边界。

    - 选词调用：答复结构不合格时带着错误重问几次（``pick_retries``，用尽记「待定」）、瞬态错误重试几次、隔多久。
    - 在用：一个词在 ``active_min_days`` 个不同的本地日被直接选中，下一次换代起在用；每条 lane 在用的词最多 ``active_cap`` 个，
      满了够条件的词等着、不挤别人（裁定 101 第 7 条）。
    - 会话 lane 选词给模型看的上文（实验原数）：最近 ``recent_records`` 条成了记录的轮，加同一会话最近
      ``recent_same_conversation`` 条；这场对话里之前最多 ``earlier_turns`` 轮，每轮材料截 ``earlier_chars`` 字。
    - 物理 lane 选词时每条最多列几步（``prompt_max_steps``）。
    - ``max_encoded_bytes``：词表状态文件与清单视图的字节上限（保护闸）。
    - 新节点（只在会话 lane，裁定 101 第 6、7 条）：模型写的名字最多 ``max_name_chars`` 个字；词表最多 ``max_level`` 层；每个上级
      下面最多 ``candidate_slots`` 个候选，满了挤掉最久没被选中的；``retire_days`` 天没被直接选中的节点退出在用或删掉。
    """

    pick_retries: int = 2
    transient_retries: int = 5
    transient_retry_delay_seconds: float = 5.0
    max_encoded_bytes: int = 16 * 1024 * 1024
    active_min_days: int = 2
    active_cap: int = 100
    recent_records: int = 8
    recent_same_conversation: int = 3
    earlier_turns: int = 3
    earlier_chars: int = 600
    prompt_max_steps: int = 12
    max_name_chars: int = 6
    candidate_slots: int = 5
    retire_days: int = 20
    max_level: int = 3

    def __post_init__(self) -> None:
        for name, lower, upper in _INTEGER_BOUNDS:
            value = getattr(self, name)
            if not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer between {lower} and {upper}")
        if not 0.0 <= float(self.transient_retry_delay_seconds) <= 600.0:
            raise ValueError("transient_retry_delay_seconds must be between 0.0 and 600.0")


__all__ = ["BehaviorKindConfig"]
