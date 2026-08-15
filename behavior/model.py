"""行为语义树共享的节点、地址与目录值对象。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum

from foundation.ids import canonical_path_identity, require_safe_path_segment

_IDENTITY_SEPARATOR = "--"
_IDENTITY_TIMESTAMP = re.compile(
    r"^(?P<year>[0-9]{4})(?P<month>[0-9]{2})(?P<day>[0-9]{2})T"
    r"(?P<hour>[0-9]{2})(?P<minute>[0-9]{2})(?P<second>[0-9]{2})"
    r"(?P<microsecond>[0-9]{6})(?P<sign>[+-])(?P<offset_hour>[0-9]{2})(?P<offset_minute>[0-9]{2})$"
)
_MAX_IDENTITY_NAME_UTF8_BYTES = 200
_IDENTITY_SUFFIX_BYTES = len(b"--20000101T000000000000+0000")
_MAX_SEMANTIC_NAME_UTF8_BYTES = _MAX_IDENTITY_NAME_UTF8_BYTES - _IDENTITY_SUFFIX_BYTES


class BehaviorKind(str, Enum):
    """行为语义树中拥有独立 L2 文档的节点类型。"""

    EVENT = "event"
    OUTCOME = "outcome"
    EPISODE = "episode"


class BehaviorLevel(int, Enum):
    """行为语义树的目录语义层级。"""

    ABSTRACT = 0
    OVERVIEW = 1
    DETAIL = 2

    @property
    def sidecar_filename(self) -> str:
        if self is BehaviorLevel.ABSTRACT:
            return ".abstract.md"
        if self is BehaviorLevel.OVERVIEW:
            return ".overview.md"
        raise ValueError("L2 uses a BehaviorAddress instead of a semantic sidecar")

    @classmethod
    def from_sidecar_filename(cls, filename: object) -> BehaviorLevel | None:
        if filename == ".abstract.md":
            return cls.ABSTRACT
        if filename == ".overview.md":
            return cls.OVERVIEW
        return None


BEHAVIORS_SEGMENT = "behaviors"
EVENTS_SEGMENT = "events"
OUTCOMES_SEGMENT = "outcomes"
EPISODES_SEGMENT = "episodes"

_KIND_DIRECTORY_PREFIXES: dict[BehaviorKind, tuple[str, ...]] = {
    BehaviorKind.EVENT: (BEHAVIORS_SEGMENT, EVENTS_SEGMENT),
    BehaviorKind.OUTCOME: (BEHAVIORS_SEGMENT, OUTCOMES_SEGMENT),
    BehaviorKind.EPISODE: (EPISODES_SEGMENT,),
}


def kind_directory_prefix(kind: BehaviorKind) -> tuple[str, ...]:
    """返回该文档类型的固定目录前缀；这里是行为树路径结构的唯一真相来源。"""

    return _KIND_DIRECTORY_PREFIXES[BehaviorKind(kind)]


def kind_for_directory_prefix(segments: tuple[str, ...]) -> BehaviorKind | None:
    """把路径开头映射回文档类型；没有任何前缀匹配时返回 None。"""

    for kind, prefix in _KIND_DIRECTORY_PREFIXES.items():
        if segments[: len(prefix)] == prefix:
            return kind
    return None


def behavior_static_directories() -> tuple[tuple[str, ...], ...]:
    """初始化必须存在的固定目录；父目录一定排在子目录之前。"""

    directories: list[tuple[str, ...]] = []
    for prefix in _KIND_DIRECTORY_PREFIXES.values():
        for depth in range(1, len(prefix) + 1):
            candidate = prefix[:depth]
            if candidate not in directories:
                directories.append(candidate)
    return tuple(sorted(directories, key=lambda parts: (len(parts), parts)))


def semantic_name(value: object, field_name: str) -> str:
    """校验可读语义名称，同时生成路径身份所需的稳定输入。"""

    name = require_safe_path_segment(value, field_name)
    if name != name.strip() or any(not character.isprintable() for character in name):
        raise ValueError(f"{field_name} must not contain surrounding whitespace or control characters")
    identity = canonical_path_identity(name, field_name)
    if identity.startswith("."):
        raise ValueError(f"{field_name} must not use a hidden-file name")
    if len(identity.encode("utf-8")) > _MAX_SEMANTIC_NAME_UTF8_BYTES:
        raise ValueError(f"{field_name} exceeds the portable path length budget")
    if identity.endswith(".md"):
        raise ValueError(f"{field_name} must not include the .md suffix")
    reserved = {
        BehaviorLevel.ABSTRACT.sidecar_filename[:-3],
        BehaviorLevel.OVERVIEW.sidecar_filename[:-3],
    }
    if identity in reserved:
        raise ValueError(f"{field_name} conflicts with a reserved semantic layer")
    return name


def behavior_local_timestamp(value: object, field_name: str) -> datetime:
    """校验用于行为事实和地址身份的带时区本地时间。"""

    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise TypeError(f"{field_name} must be a timezone-aware datetime")
    offset = value.utcoffset()
    assert offset is not None
    if offset.microseconds or offset.total_seconds() % 60:
        raise ValueError(f"{field_name} UTC offset must use whole minutes")
    return value


def behavior_identity_name(name: object, started_at: object, field_name: str) -> str:
    """从可读语义名称和事实开始时间生成唯一、可逆的地址叶名。"""

    resolved_name = semantic_name(name, field_name)
    resolved_started_at = behavior_local_timestamp(started_at, f"{field_name} started_at")
    identity = (
        f"{canonical_path_identity(resolved_name, field_name)}"
        f"{_IDENTITY_SEPARATOR}{_format_identity_timestamp(resolved_started_at)}"
    )
    if len(identity.encode("utf-8")) > _MAX_IDENTITY_NAME_UTF8_BYTES:
        raise ValueError(f"{field_name} identity exceeds the portable path length budget")
    return identity


def split_behavior_identity(value: object, field_name: str) -> tuple[str, datetime]:
    """把 URI 叶名还原为规范语义名称和用于身份的本地开始时间。"""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} identity must be a string")
    name, separator, timestamp = value.rpartition(_IDENTITY_SEPARATOR)
    if not separator or not name or not timestamp:
        raise ValueError(f"{field_name} identity must contain a semantic name and timestamp")
    resolved_name = semantic_name(name, field_name)
    started_at = _parse_identity_timestamp(timestamp, field_name)
    return resolved_name, started_at


def _format_identity_timestamp(value: datetime) -> str:
    offset = value.utcoffset()
    assert offset is not None
    offset_minutes = int(offset.total_seconds() // 60)
    sign = "+" if offset_minutes >= 0 else "-"
    absolute_offset = abs(offset_minutes)
    offset_hour, offset_minute = divmod(absolute_offset, 60)
    return (
        f"{value.year:04d}{value.month:02d}{value.day:02d}T"
        f"{value.hour:02d}{value.minute:02d}{value.second:02d}{value.microsecond:06d}"
        f"{sign}{offset_hour:02d}{offset_minute:02d}"
    )


def _parse_identity_timestamp(value: str, field_name: str) -> datetime:
    match = _IDENTITY_TIMESTAMP.fullmatch(value)
    if match is None:
        raise ValueError(f"{field_name} identity contains an invalid timestamp")
    parts = {name: int(raw) for name, raw in match.groupdict().items() if name != "sign"}
    offset_minutes = parts["offset_hour"] * 60 + parts["offset_minute"]
    if match.group("sign") == "-":
        offset_minutes = -offset_minutes
    try:
        resolved = datetime(
            parts["year"],
            parts["month"],
            parts["day"],
            parts["hour"],
            parts["minute"],
            parts["second"],
            parts["microsecond"],
            tzinfo=timezone(timedelta(minutes=offset_minutes)),
        )
    except ValueError as exc:
        raise ValueError(f"{field_name} identity contains an invalid timestamp") from exc
    if _format_identity_timestamp(resolved) != value:
        raise ValueError(f"{field_name} identity timestamp is not canonical")
    return resolved


def is_ascii_digits(value: object) -> bool:
    """只接受行为树日期目录使用的 ASCII 十进制数字。"""

    return isinstance(value, str) and value.isascii() and value.isdigit()


# TODO(BHV-TREE-TIMEBASE-001): 地址把本地偏移编进身份，与其它两层的时间约定不一致，随行为树
# 重新设计一并统一。
# - 现状：全仓库有**三套**约定。观测层存 UTC 加每条记录自己的 ``utc_offset_minutes``，用
#   ``local_occurred_at`` 还原（``behavior/observation/model.py``）。行为树把偏移编进地址文本
#   （``洗手--20260814T003000000000+0800``），并要求 ``occurred_on`` 等于本地 ``started_at`` 的日期
#   （下面那条断言）。预测层存 UTC，再用**配置里的一个全局** ``temporal_utc_offset_minutes`` 换算回
#   本地时段桶（``prediction/learning/keys.py:77``）。
# - 具体例子：``foundation.integrity.canonicalize`` 会把 datetime 折成 UTC。东八区本地 00:30 的
#   Event 折完变成前一天 16:30，``occurred_on`` 还是 14 号而 ``started_at`` 成了 13 号，这条断言当场
#   拒绝。触发区间是本地 00:00–08:00——睡前洗漱、起夜、早起做饭全在里面，是三分之一的行为，不是边角
#   情况。目前只有 ``behavior/fusion/jobs`` 这一处耐久边界会让原始 payload 通过，靠它自己的
#   ``json_safe_payload`` 与 ``StagedEvent`` 的地址重派生挡住。
# - 影响大小：中。当前没有已知的错误路径（树自己在 ``materialize`` 里就把时间转成了保留偏移的
#   字符串，``canonicalize`` 拿到的是 str），但约定不统一会在每条新增的耐久路径上重犯同一个坑；而
#   预测层用单一配置偏移，人跨时区就直接算错时段桶。
# - 改造方案：地址改成 ``(本地 occurred_on, UTC started_at, utc_offset_minutes)`` 三元组，偏移作为
#   地址的组成部分参与自校验，目录**仍按本地日历日**——"人的一天"才是行为的自然单位，按 UTC 分目录
#   会把东八区用户的睡前与起床切到相邻两天里。这样 ``canonicalize`` 全程安全，
#   ``json_safe_payload`` 与 ``StagedEvent`` 的守卫可以直接删掉，地址还能跨时区按真实时间正确排序
#   （现在不同偏移的地址按文本排序并不等于按时间排序）。预测层的 ``temporal_utc_offset_minutes``
#   随之改成从记录取而不是从配置取。
# - 代价：地址文本与所在目录在观感上不一致（``2026/08/14/洗手--20260813T1630Z``），读树的人会觉得
#   像 bug，需要在 Schema 描述里写清楚。
# - 时机：地址方案是行为树的核心设计，与 ``TODO(BHV-EPISODE-002)`` 同批做，不要改两次。


@dataclass(frozen=True)
class BehaviorAddress:
    """唯一映射到一个行为语义 L2 Markdown 文档的地址。"""

    kind: BehaviorKind
    occurred_on: date
    name: str = field(compare=False)
    started_at: datetime = field(compare=False)
    _identity_name: str = field(init=False, repr=False, compare=True)

    def __post_init__(self) -> None:
        kind = BehaviorKind(self.kind)
        if isinstance(self.occurred_on, datetime) or not isinstance(self.occurred_on, date):
            raise TypeError("behavior address occurred_on must be a date without a time")
        name = semantic_name(self.name, f"{kind.value} name")
        started_at = behavior_local_timestamp(self.started_at, f"{kind.value} started_at")
        if self.occurred_on != started_at.date():
            raise ValueError("behavior address date must match the local started_at date")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(
            self,
            "_identity_name",
            behavior_identity_name(name, started_at, f"{kind.value} name"),
        )

    @property
    def identity_name(self) -> str:
        return self._identity_name

    @classmethod
    def event(cls, occurred_on: date, event_name: str, started_at: datetime) -> BehaviorAddress:
        return cls(BehaviorKind.EVENT, occurred_on, event_name, started_at)

    @classmethod
    def outcome(cls, event_date: date, event_name: str, event_started_at: datetime) -> BehaviorAddress:
        """Outcome 使用目标 Event 的完整地址身份形成一一镜像。"""

        return cls(BehaviorKind.OUTCOME, event_date, event_name, event_started_at)

    @classmethod
    def episode(cls, started_on: date, episode_name: str, started_at: datetime) -> BehaviorAddress:
        return cls(BehaviorKind.EPISODE, started_on, episode_name, started_at)

    @classmethod
    def from_identity(
        cls,
        kind: BehaviorKind,
        occurred_on: date,
        identity_name: str,
    ) -> BehaviorAddress:
        """从 URI 或物理文件叶名恢复完整地址值对象。"""

        normalized_kind = BehaviorKind(kind)
        name, started_at = split_behavior_identity(identity_name, f"{normalized_kind.value} name")
        return cls(normalized_kind, occurred_on, name, started_at)


@dataclass(frozen=True)
class BehaviorDirectory:
    """严格限定于已确认行为语义树的目录地址。"""

    parts: tuple[str, ...] = field(default=(), compare=False)
    _identity_parts: tuple[str, ...] = field(init=False, repr=False, default=())

    def __post_init__(self) -> None:
        if isinstance(self.parts, str) or not isinstance(self.parts, tuple):
            raise TypeError("behavior directory parts must be a tuple of strings")
        parts = tuple(self.parts)
        object.__setattr__(self, "parts", parts)
        self._validate(parts)
        object.__setattr__(self, "_identity_parts", parts)

    @property
    def identity_parts(self) -> tuple[str, ...]:
        return self._identity_parts

    @classmethod
    def _validate(cls, parts: tuple[str, ...]) -> None:
        if not parts:
            return
        kind = kind_for_directory_prefix(parts)
        if kind is not None:
            cls._validate_dated_branch(parts, prefix_length=len(kind_directory_prefix(kind)))
            return
        if parts in behavior_static_directories():
            return
        raise ValueError("behavior directory is outside the confirmed tree")

    @staticmethod
    def _validate_dated_branch(parts: tuple[str, ...], *, prefix_length: int) -> None:
        values = parts[prefix_length:]
        if len(values) > 3:
            raise ValueError("behavior dated directory is deeper than day level")
        widths = (4, 2, 2)
        labels = ("year", "month", "day")
        for value, width, label in zip(values, widths, labels, strict=False):
            if not isinstance(value, str) or len(value) != width or not is_ascii_digits(value):
                raise ValueError(f"behavior {label} directory has an invalid format")
        if values and not 1 <= int(values[0]) <= 9999:
            raise ValueError("behavior year directory is outside the calendar range")
        if len(values) >= 2 and not 1 <= int(values[1]) <= 12:
            raise ValueError("behavior month directory is outside the calendar range")
        if len(values) == 3:
            try:
                date(int(values[0]), int(values[1]), int(values[2]))
            except ValueError as exc:
                raise ValueError("behavior day directory is not a valid calendar date") from exc

    @classmethod
    def root(cls) -> BehaviorDirectory:
        return cls()

    @classmethod
    def behaviors(cls) -> BehaviorDirectory:
        return cls((BEHAVIORS_SEGMENT,))

    @classmethod
    def events(
        cls,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
    ) -> BehaviorDirectory:
        return cls._dated(kind_directory_prefix(BehaviorKind.EVENT), year, month, day)

    @classmethod
    def outcomes(
        cls,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
    ) -> BehaviorDirectory:
        return cls._dated(kind_directory_prefix(BehaviorKind.OUTCOME), year, month, day)

    @classmethod
    def episodes(
        cls,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
    ) -> BehaviorDirectory:
        return cls._dated(kind_directory_prefix(BehaviorKind.EPISODE), year, month, day)

    @classmethod
    def _dated(
        cls,
        prefix: tuple[str, ...],
        year: int | None,
        month: int | None,
        day: int | None,
    ) -> BehaviorDirectory:
        if year is None:
            if month is not None or day is not None:
                raise ValueError("behavior month or day requires a year")
            return cls(prefix)
        if isinstance(year, bool) or not isinstance(year, int):
            raise TypeError("behavior directory year must be an integer")
        parts = [*prefix, f"{year:04d}"]
        if month is None:
            if day is not None:
                raise ValueError("behavior day requires a month")
            return cls(tuple(parts))
        if isinstance(month, bool) or not isinstance(month, int):
            raise TypeError("behavior directory month must be an integer")
        parts.append(f"{month:02d}")
        if day is None:
            return cls(tuple(parts))
        if isinstance(day, bool) or not isinstance(day, int):
            raise TypeError("behavior directory day must be an integer")
        parts.append(f"{day:02d}")
        return cls(tuple(parts))

    @classmethod
    def for_address(cls, address: BehaviorAddress) -> BehaviorDirectory:
        if not isinstance(address, BehaviorAddress):
            raise TypeError("address must be a BehaviorAddress")
        occurred_on = address.occurred_on
        return cls._dated(
            kind_directory_prefix(address.kind),
            occurred_on.year,
            occurred_on.month,
            occurred_on.day,
        )

    def parent(self) -> BehaviorDirectory | None:
        if not self.parts:
            return None
        return BehaviorDirectory(self.parts[:-1])

    def lineage(self) -> tuple[BehaviorDirectory, ...]:
        directories: list[BehaviorDirectory] = []
        current: BehaviorDirectory | None = self
        while current is not None:
            directories.append(current)
            current = current.parent()
        return tuple(directories)


__all__ = [
    "BEHAVIORS_SEGMENT",
    "EPISODES_SEGMENT",
    "EVENTS_SEGMENT",
    "OUTCOMES_SEGMENT",
    "BehaviorAddress",
    "BehaviorDirectory",
    "BehaviorKind",
    "BehaviorLevel",
    "behavior_identity_name",
    "behavior_local_timestamp",
    "behavior_static_directories",
    "is_ascii_digits",
    "kind_directory_prefix",
    "kind_for_directory_prefix",
    "split_behavior_identity",
]
