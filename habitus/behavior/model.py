"""habitus 树里事件与空白那一部分共享的节点、地址与目录值对象（地址在一支之内；哪一支由地址前缀与读写视图给，见 ``behavior/uri``）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum

from habitus.foundation.ids import canonical_path_identity, require_safe_path_segment
from habitus.infrastructure.tree.layout import GAPS_DIRECTORY, OCCURRENCES_DIRECTORY, SIDECAR_FILENAMES

_IDENTITY_SEPARATOR = "--"
_IDENTITY_TIMESTAMP = re.compile(
    r"^(?P<year>[0-9]{4})(?P<month>[0-9]{2})(?P<day>[0-9]{2})T"
    r"(?P<hour>[0-9]{2})(?P<minute>[0-9]{2})(?P<second>[0-9]{2})"
    r"(?P<microsecond>[0-9]{6})(?P<sign>[+-])(?P<offset_hour>[0-9]{2})(?P<offset_minute>[0-9]{2})$"
)
_MAX_IDENTITY_NAME_UTF8_BYTES = 200
_IDENTITY_SUFFIX_BYTES = len(b"--20000101T000000000000+0000")
_MAX_SEMANTIC_NAME_UTF8_BYTES = _MAX_IDENTITY_NAME_UTF8_BYTES - _IDENTITY_SUFFIX_BYTES

# 融合层对 behavior 名的字节预算：在语义名预算上再预留撞车消歧后缀（"-2".."-9999" 级别）的
# 空间——否则贴着预算上限的合法判断名，一旦触发消歧就会在归约期撑爆地址身份、卡死整条队列。
MAX_BEHAVIOR_NAME_UTF8_BYTES = _MAX_SEMANTIC_NAME_UTF8_BYTES - 8


class BehaviorKind(str, Enum):
    """行为语义树中拥有独立 L2 文档的节点类型。

    ``OCCURRENCE`` 是一次被观测到的行为单位（可提醒或可代劳的那种；``goal`` 可空，空非空不
    决定它算不算一条）；``GAP`` 是一段观测空白（没读懂 / 未观测），与行为共享同一条时间轴。
    """

    OCCURRENCE = "occurrence"
    GAP = "gap"


OCCURRENCES_SEGMENT = OCCURRENCES_DIRECTORY
GAPS_SEGMENT = GAPS_DIRECTORY

_KIND_DIRECTORY_PREFIXES: dict[BehaviorKind, tuple[str, ...]] = {
    BehaviorKind.OCCURRENCE: (OCCURRENCES_SEGMENT,),
    BehaviorKind.GAP: (GAPS_SEGMENT,),
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
    if identity in {filename[:-3] for filename in SIDECAR_FILENAMES}:
        raise ValueError(f"{field_name} conflicts with a reserved semantic layer")
    return name


class BehaviorNameTooLongError(ValueError):
    """名字能当地址，但没给撞车消歧后缀留出字节（``MAX_BEHAVIOR_NAME_UTF8_BYTES``）。"""


def behavior_name(value: object) -> str:
    """一条行为的名字：能当行为树地址（``semantic_name``），且在融合层的字节预算之内。两条 lane 的入口与判断本身都按它查。"""

    name = semantic_name(value, "behavior")
    if len(name.encode("utf-8")) > MAX_BEHAVIOR_NAME_UTF8_BYTES:
        raise BehaviorNameTooLongError("behavior name exceeds the byte budget kept for a collision suffix")
    return name


# 名字会拿去当路径上的一段，这些符号不能有：直接换成空格（同一个说法每次换出来的都一样）。
_PATH_SYMBOLS = str.maketrans(dict.fromkeys('/\\:<>"|?*', " "))


def address_name(raw: str) -> str:
    """把一个说法洗成能当路径一段的样子：路径符号换成空格、空白收拢、去掉首尾的点与空格。洗完仍要过 ``behavior_name``。"""

    return " ".join(raw.translate(_PATH_SYMBOLS).split()).strip(". ")


def behavior_local_timestamp(value: object, field_name: str) -> datetime:
    """校验用于行为事实和地址身份的带时区本地时间。"""

    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise TypeError(f"{field_name} must be a timezone-aware datetime")
    offset = value.utcoffset()
    assert offset is not None
    if offset.microseconds or offset.total_seconds() % 60:
        raise ValueError(f"{field_name} UTC offset must use whole minutes")
    # 树上行为时间一律是发生地的本地时刻+偏移。零偏移在这里是上游把时间折成 UTC 的事故信号：
    # 放进来会把东八区凌晨的行为错到前一天，硬拒让事故当场现形。
    if not offset:
        raise ValueError(
            f"{field_name} must carry a non-zero local UTC offset (a +00:00 stamp here means something upstream folded to UTC)"
        )
    return value


def behavior_identity_name(name: object, started_at: object, field_name: str) -> str:
    """从可读语义名称和事实开始时间生成唯一的地址叶名。

    往返只保证还原到规范身份（NFC+casefold 后的名字）——原始写法保存在文档的 ``name``
    字段里，不从叶名还原。地址相等性按规范身份比较，与此一致。
    """

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


def readable_text(value: str) -> bool:
    """树上的文本字段（摘要、目标、步骤）收不收得下：除换行与制表符以外不许有不可打印字符——控制符、ANSI 转义、零宽字符
    与其他格式符（含组合表情里的零宽连接符）。树 schema 与两条 lane 的融合入口共用这一条：入口放行的文本树上一定收得下。"""

    return not any(not character.isprintable() and character not in "\n\t" for character in value)


def is_ascii_digits(value: object) -> bool:
    """只接受行为树日期目录使用的 ASCII 十进制数字。"""

    return isinstance(value, str) and value.isascii() and value.isdigit()


# TODO(BHV-TREE-REBUILD-001): 物理 lane 交付的 HTTP 端点未做；上游契约缺 place 字段、交互模态与覆盖信号。
# TODO(BHV-FEEDBACK-001): 预测要被结果推着调整，提醒上线前须把"被提醒后发生"与自然发生分开。
# TODO(BHV-EPISODE-002): episode（走完预测闭环的成熟行为的历程）未设计，代码不留占位。


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
    def occurrence(cls, occurred_on: date, name: str, started_at: datetime) -> BehaviorAddress:
        return cls(BehaviorKind.OCCURRENCE, occurred_on, name, started_at)

    @classmethod
    def gap(cls, occurred_on: date, gap_kind: str, started_at: datetime) -> BehaviorAddress:
        """空白节点的叶名以空白类型（没读懂/未观测）为语义名。"""

        return cls(BehaviorKind.GAP, occurred_on, gap_kind, started_at)

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
        if parts in _KIND_DIRECTORY_PREFIXES.values():
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
            if len(value) != width or not is_ascii_digits(value):
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
    def occurrences(
        cls,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
    ) -> BehaviorDirectory:
        return cls._dated(kind_directory_prefix(BehaviorKind.OCCURRENCE), year, month, day)

    @classmethod
    def gaps(
        cls,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
    ) -> BehaviorDirectory:
        return cls._dated(kind_directory_prefix(BehaviorKind.GAP), year, month, day)

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
        parts = [*prefix, f"{year:04d}"]
        if month is None:
            if day is not None:
                raise ValueError("behavior day requires a month")
            return cls(tuple(parts))
        parts.append(f"{month:02d}")
        if day is None:
            return cls(tuple(parts))
        parts.append(f"{day:02d}")
        return cls(tuple(parts))

    @classmethod
    def for_address(cls, address: BehaviorAddress) -> BehaviorDirectory:
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
    "GAPS_SEGMENT",
    "MAX_BEHAVIOR_NAME_UTF8_BYTES",
    "OCCURRENCES_SEGMENT",
    "BehaviorAddress",
    "BehaviorDirectory",
    "BehaviorKind",
    "BehaviorNameTooLongError",
    "address_name",
    "behavior_identity_name",
    "behavior_local_timestamp",
    "behavior_name",
    "is_ascii_digits",
    "kind_directory_prefix",
    "kind_for_directory_prefix",
    "readable_text",
    "split_behavior_identity",
]
