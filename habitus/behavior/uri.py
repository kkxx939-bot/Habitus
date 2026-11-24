"""``habitus://`` 地址里事件与空白那一部分：``<occurrences|gaps>/<支>/YYYY[/MM[/DD[/<叶名>.md]]]``。

支（``session`` / ``physical``）在路径里、不在文档字段里：同一个 ``BehaviorAddress`` 在两支里是两个地址（裁定 112）。
叶名与百分号编码和合并之前逐字节相同，同一支内地址的先后不变。
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from habitus.behavior.kinds.ids import Lane
from habitus.behavior.model import (
    BehaviorAddress,
    BehaviorDirectory,
    BehaviorKind,
    is_ascii_digits,
    kind_directory_prefix,
    kind_for_directory_prefix,
)
from habitus.infrastructure.tree.address import TreeAddressError, format_uri, split_uri
from habitus.infrastructure.tree.layout import SIDECAR_FILENAMES

_LANES = frozenset(lane.value for lane in Lane)


class BehaviorURIError(ValueError):
    """地址不是一支里的事件、空白或它们的日期目录。"""


class BehaviorURINodeType(str, Enum):
    DIRECTORY = "directory"
    DOCUMENT = "document"


class BehaviorURI:
    """支 + 一支之内的地址或目录，与 ``habitus://`` 字符串之间的不可变映射。"""

    _uri: str
    _lane: str
    _segments: tuple[str, ...]
    _node_type: BehaviorURINodeType
    _address: BehaviorAddress | None
    _directory: BehaviorDirectory | None
    __slots__ = ("_address", "_directory", "_lane", "_node_type", "_segments", "_uri")

    def __init__(self, uri: str) -> None:
        try:
            segments = split_uri(uri)
        except TreeAddressError as exc:
            raise BehaviorURIError(str(exc)) from exc
        kind = kind_for_directory_prefix(segments)
        if kind is None:
            raise BehaviorURIError("habitus URI is not inside the occurrence or gap directories")
        depth = len(kind_directory_prefix(kind))
        if len(segments) <= depth or segments[depth] not in _LANES:
            raise BehaviorURIError("behavior URI must name a lane right after its kind directory")
        lane = segments[depth]
        inner = (*segments[:depth], *segments[depth + 1 :])
        address = _address(kind, depth, inner)
        if address is not None:
            node_type, directory = BehaviorURINodeType.DOCUMENT, None
        else:
            try:
                directory = BehaviorDirectory(inner)
            except (TypeError, ValueError) as exc:
                raise BehaviorURIError("behavior URI does not map to an occurrence, gap or dated directory") from exc
            node_type = BehaviorURINodeType.DIRECTORY
        canonical = _with_lane(lane, address.kind if address else kind, _inner_segments(address, directory))
        object.__setattr__(self, "_uri", format_uri(canonical))
        object.__setattr__(self, "_lane", lane)
        object.__setattr__(self, "_segments", canonical)
        object.__setattr__(self, "_node_type", node_type)
        object.__setattr__(self, "_address", address)
        object.__setattr__(self, "_directory", directory)

    @classmethod
    def parse(cls, value: BehaviorURI | str) -> BehaviorURI:
        if isinstance(value, BehaviorURI):
            return value
        if not isinstance(value, str):
            raise TypeError("behavior URI must be a string")
        return cls(value)

    @classmethod
    def from_address(cls, address: BehaviorAddress, lane: str) -> BehaviorURI:
        return cls(format_uri(_with_lane(_lane(lane), address.kind, _inner_segments(address, None))))

    @classmethod
    def from_directory(cls, directory: BehaviorDirectory, lane: str) -> BehaviorURI:
        kind = kind_for_directory_prefix(directory.identity_parts)
        if kind is None:
            raise BehaviorURIError("behavior URI directory must be inside a kind directory")
        return cls(format_uri(_with_lane(_lane(lane), kind, directory.identity_parts)))

    @staticmethod
    def is_valid(uri: object) -> bool:
        if not isinstance(uri, str):
            return False
        try:
            BehaviorURI(uri)
        except (TypeError, ValueError):
            return False
        return True

    @property
    def uri(self) -> str:
        return self._uri

    @property
    def lane(self) -> str:
        return self._lane

    @property
    def segments(self) -> tuple[str, ...]:
        return self._segments

    @property
    def node_type(self) -> BehaviorURINodeType:
        return self._node_type

    def to_address(self) -> BehaviorAddress:
        if self._node_type is not BehaviorURINodeType.DOCUMENT or self._address is None:
            raise BehaviorURIError("behavior URI does not identify an L2 document")
        return self._address

    def to_directory(self) -> BehaviorDirectory:
        if self._node_type is not BehaviorURINodeType.DIRECTORY or self._directory is None:
            raise BehaviorURIError("behavior URI does not identify a directory")
        return self._directory

    def __str__(self) -> str:
        return self._uri

    def __repr__(self) -> str:
        return f"BehaviorURI({self._uri!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BehaviorURI):
            return self._uri == other._uri
        if isinstance(other, str):
            return self._uri == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._uri)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("BehaviorURI is immutable")


def _lane(value: object) -> str:
    if not isinstance(value, str) or value not in _LANES:
        raise BehaviorURIError(f"unknown behavior lane {value!r}")
    return value


def _with_lane(lane: str, kind: BehaviorKind, inner: tuple[str, ...]) -> tuple[str, ...]:
    depth = len(kind_directory_prefix(kind))
    return (*inner[:depth], lane, *inner[depth:])


def _inner_segments(address: BehaviorAddress | None, directory: BehaviorDirectory | None) -> tuple[str, ...]:
    """一支之内的路径段（不含支）：文档是「类型前缀 + YYYY/MM/DD + 叶文件」。"""

    if address is None:
        assert directory is not None
        return directory.identity_parts
    occurred_on = address.occurred_on
    return (
        *kind_directory_prefix(address.kind),
        f"{occurred_on.year:04d}",
        f"{occurred_on.month:02d}",
        f"{occurred_on.day:02d}",
        f"{address.identity_name}.md",
    )


def _address(kind: BehaviorKind, depth: int, inner: tuple[str, ...]) -> BehaviorAddress | None:
    if len(inner) != depth + 4:
        return None
    filename = inner[depth + 3]
    if not filename.endswith(".md") or filename in SIDECAR_FILENAMES or not filename[:-3]:
        return None
    try:
        return BehaviorAddress.from_identity(kind, _date(inner[depth : depth + 3]), filename[:-3])
    except (TypeError, ValueError):
        return None


def _date(parts: tuple[str, ...]) -> date:
    if len(parts[0]) != 4 or len(parts[1]) != 2 or len(parts[2]) != 2 or not all(is_ascii_digits(p) for p in parts):
        raise BehaviorURIError("behavior URI date must use YYYY/MM/DD")
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


__all__ = ["BehaviorURI", "BehaviorURIError", "BehaviorURINodeType"]
