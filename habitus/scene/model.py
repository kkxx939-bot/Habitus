"""语义关联层的地址与受控枚举。

两套地址并存了一段时间；按天归组那一套已经删掉（2026-09-13）。现在这里只有**规律级**：
``AssociationAddress``（哪个候选、哪一天、哪条行为、几点开始）与它的目录 ``KindDirectory``。

身份纪律与行为树、记忆树同一套：人写法 ``compare=False``，真正的身份是 NFC + casefold 之后的
那一份；叶名带上行为原名，因为同一 kind、同一时刻、不同名字的两条 occurrence 在行为树上完全
合法，只用时刻会让它们互相覆盖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum

from habitus.behavior.model import (
    behavior_identity_name,
    behavior_local_timestamp,
    is_ascii_digits,
    semantic_name,
    split_behavior_identity,
)
from habitus.foundation.ids import canonical_path_identity

# 与行为名同一字节预算（同一套叶名规则）。情景**没有**撞车消歧后缀：同一天里两个情景同标签且
# 首成员同一微秒开始，就是同一件事——归组校验（M2）在落盘前把它们合并；到了存储层仍撞车即冲突。




class SceneLinkType(str, Enum):
    """情景对更早对象的两种前向边；方向固定为晚指早，目标不设时间上限。

    ``NEEDS`` 这件事的前提由那条行为（或那件事）建立；``RESULTS_FROM`` 这件事因那件事而起。
    没有 part_of：归属是情景的 ``members`` 字段，不是边（DAY1 实测把 part_of 作为行为间链接会
    传递成 59 条的大团）；没有 continues：中断续做可由成员与时间轴机械推出。
    """

    NEEDS = "needs"
    RESULTS_FROM = "results_from"










# ── 规律级：按候选行为归档的那片区域 ──────────────────────────────────────────────────
#
# 语义关联是为预测树算出的候选服务的，所以归档的主键是 ``kind_token`` 而不是日历日——按天归档
# 会把同一个候选的历次发生散在几百个日子里，读侧要还原"这个候选历次分别因为什么"就得扫全树。
#
#     kinds/<kind_token>/
#         .abstract.md                     L0 一句话：这个行为通常因为什么发生
#         .overview.md                     L1 几种情境（各自覆盖哪些日期）+ 由来
#         2026/06/10/<时刻>.md              L2 这一次的上下文与前因
#
# 地址住在这里而不是 ``regularity`` 包里：``uri`` 要同时认两种文档形态，而 ``regularity`` 的
# 文档层反过来要用 ``uri``——地址与地址放在一起，这条环就不存在。

KINDS_SEGMENT = "kinds"


class RegularityLevel(int, Enum):
    """规律级的语义层；与行为树、记忆树同一套约定。"""

    ABSTRACT = 0
    OVERVIEW = 1
    DETAIL = 2

    @property
    def sidecar_filename(self) -> str:
        if self is RegularityLevel.ABSTRACT:
            return ".abstract.md"
        if self is RegularityLevel.OVERVIEW:
            return ".overview.md"
        raise ValueError("L2 uses an AssociationAddress instead of a semantic sidecar")

    @classmethod
    def from_sidecar_filename(cls, filename: object) -> RegularityLevel | None:
        if filename == ".abstract.md":
            return cls.ABSTRACT
        if filename == ".overview.md":
            return cls.OVERVIEW
        return None


def regularity_static_directories() -> tuple[tuple[str, ...], ...]:
    return ((KINDS_SEGMENT,),)


def _kind_identity(kind_token: object) -> str:
    """候选落到文件系统的目录名：规范身份（NFC + casefold），不是人写法。"""

    return canonical_path_identity(semantic_name(kind_token, "regularity kind token"), "regularity kind token")


@dataclass(frozen=True)
class AssociationAddress:
    """一次发生的关联记录：哪个候选、哪一天、哪条行为、几点开始。

    身份与 ``SceneAddress`` / ``BehaviorAddress`` / ``MemoryAddress` 同一套做法，不是另起一套：

    - ``kind_token`` 与 ``name`` 是**人写法**，比较时不看（``compare=False``）；真正的身份是
      ``canonical_path_identity``（NFC + casefold）之后的那两个。``semantic_name`` 只**校验**
      归一后的形式合法，**返回的是原名**——直接拿它当目录名，``Gym`` 与 ``gym`` 在大小写不敏感
      的文件系统上就是同一个文件，而两个地址却不相等：一个候选会把另一个的记录物理覆盖，
      幸存的那条对两边都读不出来。
    - 叶名是 ``behavior_identity_name(name, started_at)``，**带上行为的原始名**。只用时刻不行：
      同一个 kind、同一时刻、不同名字的两条 occurrence 在行为树上完全合法（撞车消歧按名字分），
      只用时刻会让它们互相覆盖。

    ``started_at`` 就是行为树上那条 occurrence 的开始时刻；``occurred_on`` 必须等于它的本地日期
    （与行为树同一条契约），否则目录与内容会说两个日子。
    """

    kind_token: str = field(compare=False)
    occurred_on: date
    name: str = field(compare=False)
    started_at: datetime = field(compare=False)
    _identity_kind: str = field(init=False, repr=False, compare=True)
    _identity_name: str = field(init=False, repr=False, compare=True)

    def __post_init__(self) -> None:
        kind_token = semantic_name(self.kind_token, "regularity kind token")
        if isinstance(self.occurred_on, datetime) or not isinstance(self.occurred_on, date):
            raise TypeError("regularity address occurred_on must be a date without a time")
        name = semantic_name(self.name, "regularity behaviour name")
        started_at = behavior_local_timestamp(self.started_at, "regularity started_at")
        if self.occurred_on != started_at.date():
            raise ValueError("regularity address date must match the local started_at date")
        object.__setattr__(self, "kind_token", kind_token)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "_identity_kind", canonical_path_identity(kind_token, "regularity kind token"))
        object.__setattr__(
            self, "_identity_name", behavior_identity_name(name, started_at, "regularity behaviour name")
        )

    @property
    def identity_kind(self) -> str:
        """落到文件系统的候选目录名。"""

        return self._identity_kind

    @property
    def identity_name(self) -> str:
        """落到文件系统的叶名（不含 ``.md``）。"""

        return self._identity_name

    @classmethod
    def from_identity(cls, kind_token: str, occurred_on: date, identity_name: str) -> AssociationAddress:
        """从叶名还原。

        走行为树那套 ``split_behavior_identity``——它比自己写的解析严格：非规范的时间戳
        （``+0060``、``+0099``、阿拉伯数字）会被拒，而不是被"修复"成另一个文件名。
        """

        name, started_at = split_behavior_identity(identity_name, "regularity behaviour name")
        return cls(kind_token, occurred_on, name, started_at)


@dataclass(frozen=True)
class KindDirectory:
    """``kinds`` 之下的目录：候选，以及它下面按年 / 月 / 日分的片。

    分片只为"按出处日直接定位"服务：预测层拿到四层出处日之后，要读的就是那几天，不该为此扫遍
    这个候选的全部历史。
    """

    parts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.parts, str) or not isinstance(self.parts, tuple):
            raise TypeError("regularity directory parts must be a tuple of strings")
        self._validate(self.parts)

    @staticmethod
    def _validate(parts: tuple[str, ...]) -> None:
        if not parts:
            return
        if parts[0] != KINDS_SEGMENT:
            raise ValueError("regularity directory is outside the confirmed tree")
        if len(parts) == 1:
            return
        # 目录名必须**已经是**规范身份：路径上出现 ``Gym`` 而不是 ``gym``，说明有人绕过了
        # ``for_kind`` 直接拼路径，那一刻大小写不敏感的文件系统上就已经埋了一次覆盖。
        if _kind_identity(parts[1]) != parts[1]:
            raise ValueError("regularity kind directory must use the canonical kind identity")
        dated = parts[2:]
        if len(dated) > 3:
            raise ValueError("regularity dated directory is deeper than day level")
        for value, width, label in zip(dated, (4, 2, 2), ("year", "month", "day"), strict=False):
            if not isinstance(value, str) or len(value) != width or not is_ascii_digits(value):
                raise ValueError(f"regularity {label} directory has an invalid format")
        if dated and not 1 <= int(dated[0]) <= 9999:
            raise ValueError("regularity year directory is outside the calendar range")
        if len(dated) >= 2 and not 1 <= int(dated[1]) <= 12:
            raise ValueError("regularity month directory is outside the calendar range")
        if len(dated) == 3:
            try:
                date(int(dated[0]), int(dated[1]), int(dated[2]))
            except ValueError as exc:
                raise ValueError("regularity day directory is not a valid calendar date") from exc

    @classmethod
    def root(cls) -> KindDirectory:
        return cls()

    @classmethod
    def kinds(cls) -> KindDirectory:
        return cls((KINDS_SEGMENT,))

    @classmethod
    def for_kind(cls, kind_token: str) -> KindDirectory:
        return cls((KINDS_SEGMENT, _kind_identity(kind_token)))

    @classmethod
    def for_day(cls, kind_token: str, occurred_on: date) -> KindDirectory:
        if isinstance(occurred_on, datetime) or not isinstance(occurred_on, date):
            raise TypeError("occurred_on must be a date without a time")
        return cls(
            (
                KINDS_SEGMENT,
                _kind_identity(kind_token),
                f"{occurred_on.year:04d}",
                f"{occurred_on.month:02d}",
                f"{occurred_on.day:02d}",
            )
        )

    @classmethod
    def for_address(cls, address: AssociationAddress) -> KindDirectory:
        if not isinstance(address, AssociationAddress):
            raise TypeError("address must be an AssociationAddress")
        return cls.for_day(address.kind_token, address.occurred_on)

    @property
    def kind_token(self) -> str | None:
        return self.parts[1] if len(self.parts) >= 2 else None

    def day(self) -> date | None:
        if len(self.parts) != 5:
            return None
        return date(int(self.parts[2]), int(self.parts[3]), int(self.parts[4]))

    def parent(self) -> KindDirectory | None:
        if not self.parts:
            return None
        return KindDirectory(self.parts[:-1])


__all__ = [
    "regularity_static_directories",
    "RegularityLevel",
    "KindDirectory",
    "AssociationAddress",
    "KINDS_SEGMENT",
    "is_ascii_digits",
    "SceneLinkType",
]
