"""语义关联树（情景树）共享的节点、地址与目录值对象。

情景树是行为树的**解释层**：每封口日一次 LLM 归组，把当天的 occurrence 归到若干"事"（情景）
之下，并给出情景对更早行为/情景的 needs / results_from 边。它从行为树派生、按天整体重建、
可推倒重来；行为树一个字不动。

设计定稿见桌面《语义关联层实现方案 v1》（2026-09-06）。三条不变的纪律：

- **判断按事、表达按行为**：LLM 只判"哪几条是一件事、这件事依赖什么、留下什么"；每条行为的
  上下文视图由此机械投影（M3）。
- **occurrence 是原子**：本树只归组、只连边，不切分、不合并、不改写任何一条行为。
- **不存任何计数**：次数与权重归时间预测树；本树只存实例与解释。

地址身份沿行为树同一套规则（``behavior.model`` 的语义名与叶名函数直接复用）：叶名 =
``{label}--{started_at}``，label 是归组给出的原话标签（第一期不归一）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum

from habitus.behavior.model import (
    MAX_BEHAVIOR_NAME_UTF8_BYTES,
    behavior_identity_name,
    behavior_local_timestamp,
    is_ascii_digits,
    semantic_name,
    split_behavior_identity,
)

SCENES_SEGMENT = "scenes"

# 与行为名同一字节预算（同一套叶名规则）。情景**没有**撞车消歧后缀：同一天里两个情景同标签且
# 首成员同一微秒开始，就是同一件事——归组校验（M2）在落盘前把它们合并；到了存储层仍撞车即冲突。
MAX_SCENE_LABEL_UTF8_BYTES = MAX_BEHAVIOR_NAME_UTF8_BYTES


class SceneRole(str, Enum):
    """一条 occurrence 在所属情景里的角色（借 Ego4D Goal-Step 的三档）。

    ``ESSENTIAL`` 这件事必需的一步；``OPTIONAL`` 相关但可有可无；``IRRELEVANT`` 发生在这件事
    期间、与它无关（做饭时看了会儿手机）——仍归进情景，记录"那段时间他在做什么事的期间发生了
    这条"，但投影"此前步骤"时不计。
    """

    ESSENTIAL = "essential"
    OPTIONAL = "optional"
    IRRELEVANT = "irrelevant"


class SceneLinkType(str, Enum):
    """情景对更早对象的两种前向边；方向固定为晚指早，目标不设时间上限。

    ``NEEDS`` 这件事的前提由那条行为（或那件事）建立；``RESULTS_FROM`` 这件事因那件事而起。
    没有 part_of：归属是情景的 ``members`` 字段，不是边（DAY1 实测把 part_of 作为行为间链接会
    传递成 59 条的大团）；没有 continues：中断续做可由成员与时间轴机械推出。
    """

    NEEDS = "needs"
    RESULTS_FROM = "results_from"


def scene_label(value: object, field_name: str) -> str:
    """校验情景标签：可作地址叶名的语义名，且留出消歧后缀的字节余量。"""

    name = semantic_name(value, field_name)
    if len(name.encode("utf-8")) > MAX_SCENE_LABEL_UTF8_BYTES:
        raise ValueError(f"{field_name} exceeds the scene label byte budget")
    return name


def scene_static_directories() -> tuple[tuple[str, ...], ...]:
    """初始化必须存在的固定目录。"""

    return ((SCENES_SEGMENT,),)


@dataclass(frozen=True)
class SceneAddress:
    """唯一映射到一个情景 L2 文档的逻辑地址（不含物理代目录）。"""

    occurred_on: date
    label: str = field(compare=False)
    started_at: datetime = field(compare=False)
    _identity_name: str = field(init=False, repr=False, compare=True)

    def __post_init__(self) -> None:
        if isinstance(self.occurred_on, datetime) or not isinstance(self.occurred_on, date):
            raise TypeError("scene address occurred_on must be a date without a time")
        label = scene_label(self.label, "scene label")
        started_at = behavior_local_timestamp(self.started_at, "scene started_at")
        if self.occurred_on != started_at.date():
            raise ValueError("scene address date must match the local started_at date")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "_identity_name", behavior_identity_name(label, started_at, "scene label"))

    @property
    def identity_name(self) -> str:
        return self._identity_name

    @classmethod
    def from_identity(cls, occurred_on: date, identity_name: str) -> SceneAddress:
        """从叶名恢复地址；还原到的是**规范身份**（NFC + casefold），标签原话在文档 ``label`` 字段里。"""

        label, started_at = split_behavior_identity(identity_name, "scene label")
        return cls(occurred_on, label, started_at)


@dataclass(frozen=True)
class SceneDirectory:
    """严格限定于情景树的逻辑目录：根、``scenes``、``scenes/YYYY[/MM[/DD]]``。"""

    parts: tuple[str, ...] = field(default=(), compare=False)
    _identity_parts: tuple[str, ...] = field(init=False, repr=False, default=())

    def __post_init__(self) -> None:
        if isinstance(self.parts, str) or not isinstance(self.parts, tuple):
            raise TypeError("scene directory parts must be a tuple of strings")
        parts = tuple(self.parts)
        object.__setattr__(self, "parts", parts)
        self._validate(parts)
        object.__setattr__(self, "_identity_parts", parts)

    @property
    def identity_parts(self) -> tuple[str, ...]:
        return self._identity_parts

    @staticmethod
    def _validate(parts: tuple[str, ...]) -> None:
        if not parts:
            return
        if parts[0] != SCENES_SEGMENT:
            raise ValueError("scene directory is outside the confirmed tree")
        values = parts[1:]
        if len(values) > 3:
            raise ValueError("scene dated directory is deeper than day level")
        for value, width, label in zip(values, (4, 2, 2), ("year", "month", "day"), strict=False):
            if not isinstance(value, str) or len(value) != width or not is_ascii_digits(value):
                raise ValueError(f"scene {label} directory has an invalid format")
        if values and not 1 <= int(values[0]) <= 9999:
            raise ValueError("scene year directory is outside the calendar range")
        if len(values) >= 2 and not 1 <= int(values[1]) <= 12:
            raise ValueError("scene month directory is outside the calendar range")
        if len(values) == 3:
            try:
                date(int(values[0]), int(values[1]), int(values[2]))
            except ValueError as exc:
                raise ValueError("scene day directory is not a valid calendar date") from exc

    @classmethod
    def root(cls) -> SceneDirectory:
        return cls()

    @classmethod
    def scenes(cls, year: int | None = None, month: int | None = None, day: int | None = None) -> SceneDirectory:
        if year is None:
            if month is not None or day is not None:
                raise ValueError("scene month or day requires a year")
            return cls((SCENES_SEGMENT,))
        if isinstance(year, bool) or not isinstance(year, int):
            raise TypeError("scene directory year must be an integer")
        parts = [SCENES_SEGMENT, f"{year:04d}"]
        if month is None:
            if day is not None:
                raise ValueError("scene day requires a month")
            return cls(tuple(parts))
        if isinstance(month, bool) or not isinstance(month, int):
            raise TypeError("scene directory month must be an integer")
        parts.append(f"{month:02d}")
        if day is None:
            return cls(tuple(parts))
        if isinstance(day, bool) or not isinstance(day, int):
            raise TypeError("scene directory day must be an integer")
        parts.append(f"{day:02d}")
        return cls(tuple(parts))

    @classmethod
    def for_day(cls, occurred_on: date) -> SceneDirectory:
        if isinstance(occurred_on, datetime) or not isinstance(occurred_on, date):
            raise TypeError("occurred_on must be a date without a time")
        return cls.scenes(occurred_on.year, occurred_on.month, occurred_on.day)

    @classmethod
    def for_address(cls, address: SceneAddress) -> SceneDirectory:
        if not isinstance(address, SceneAddress):
            raise TypeError("address must be a SceneAddress")
        return cls.for_day(address.occurred_on)

    def day(self) -> date | None:
        """日目录对应的日历日；不是日目录返回 None。"""

        if len(self.parts) != 4:
            return None
        return date(int(self.parts[1]), int(self.parts[2]), int(self.parts[3]))

    def parent(self) -> SceneDirectory | None:
        if not self.parts:
            return None
        return SceneDirectory(self.parts[:-1])


__all__ = [
    "MAX_SCENE_LABEL_UTF8_BYTES",
    "SCENES_SEGMENT",
    "is_ascii_digits",
    "SceneAddress",
    "SceneDirectory",
    "SceneLinkType",
    "SceneRole",
    "scene_label",
    "scene_static_directories",
]
