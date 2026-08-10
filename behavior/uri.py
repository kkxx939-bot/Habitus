"""严格映射行为语义树的 ``behavior://`` URI。"""

from __future__ import annotations

from datetime import date
from enum import Enum
from urllib.parse import unquote

from behavior.model import (
    BehaviorAddress,
    BehaviorDirectory,
    BehaviorLevel,
    is_ascii_digits,
    kind_directory_prefix,
    kind_for_directory_prefix,
)

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_UNRESERVED_ASCII = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")


class BehaviorURIError(ValueError):
    """URI 无法映射到已确认的行为语义树。"""


class BehaviorURINodeType(str, Enum):
    DIRECTORY = "directory"
    DOCUMENT = "document"
    LAYER = "layer"


class BehaviorURI:
    """行为 URI、L2 地址、目录和派生语义层之间的不可变映射。"""

    SCHEME = "behavior"
    _uri: str
    _segments: tuple[str, ...]
    _node_type: BehaviorURINodeType
    _address: BehaviorAddress | None
    _directory: BehaviorDirectory | None
    _level: BehaviorLevel | None
    __slots__ = ("_address", "_directory", "_level", "_node_type", "_segments", "_uri")

    def __init__(self, uri: str) -> None:
        if not isinstance(uri, str):
            raise TypeError("behavior URI must be a string")
        if not uri or uri != uri.strip():
            raise BehaviorURIError("behavior URI must be non-empty without surrounding whitespace")
        prefix = f"{self.SCHEME}://"
        if not uri.startswith(prefix):
            raise BehaviorURIError(f"behavior URI must start with '{prefix}'")
        raw_path = uri[len(prefix) :]
        if raw_path.endswith("/"):
            raise BehaviorURIError("behavior URI must not contain a trailing slash")
        raw_segments = tuple(raw_path.split("/")) if raw_path else ()
        if any(not segment for segment in raw_segments):
            raise BehaviorURIError("behavior URI contains an empty path segment")
        segments = tuple(_decode_segment(segment) for segment in raw_segments)
        node_type, address, directory, level = _classify(segments)
        canonical = _canonical_segments(node_type, address, directory, level)
        encoded_path = "/".join(_encode_segment(segment) for segment in canonical)
        object.__setattr__(self, "_uri", f"{prefix}{encoded_path}")
        object.__setattr__(self, "_segments", canonical)
        object.__setattr__(self, "_node_type", node_type)
        object.__setattr__(self, "_address", address)
        object.__setattr__(self, "_directory", directory)
        object.__setattr__(self, "_level", level)

    @classmethod
    def parse(cls, value: BehaviorURI | str) -> BehaviorURI:
        if isinstance(value, BehaviorURI):
            return value
        return cls(value)

    @classmethod
    def root(cls) -> BehaviorURI:
        return cls(f"{cls.SCHEME}://")

    @classmethod
    def from_address(cls, address: BehaviorAddress) -> BehaviorURI:
        if not isinstance(address, BehaviorAddress):
            raise TypeError("address must be a BehaviorAddress")
        return cls._from_segments(_address_segments(address))

    @classmethod
    def from_directory(cls, directory: BehaviorDirectory) -> BehaviorURI:
        if not isinstance(directory, BehaviorDirectory):
            raise TypeError("directory must be a BehaviorDirectory")
        return cls._from_segments(directory.identity_parts)

    @classmethod
    def from_layer(cls, directory: BehaviorDirectory, level: BehaviorLevel) -> BehaviorURI:
        if not isinstance(directory, BehaviorDirectory):
            raise TypeError("directory must be a BehaviorDirectory")
        normalized = BehaviorLevel(level)
        return cls._from_segments((*directory.identity_parts, normalized.sidecar_filename))

    @classmethod
    def build(cls, *path_parts: str) -> BehaviorURI:
        segments: list[str] = []
        for part in path_parts:
            if not isinstance(part, str):
                raise TypeError("behavior URI path parts must be strings")
            if not part or "/" in part or "\\" in part:
                raise BehaviorURIError("behavior URI path parts must be safe non-empty segments")
            segments.append(part)
        return cls._from_segments(tuple(segments))

    @classmethod
    def _from_segments(cls, segments: tuple[str, ...]) -> BehaviorURI:
        encoded = "/".join(_encode_segment(segment) for segment in segments)
        return cls(f"{cls.SCHEME}://{encoded}")

    @staticmethod
    def is_valid(uri: object) -> bool:
        if not isinstance(uri, str):
            return False
        try:
            BehaviorURI(uri)
        except (TypeError, ValueError):
            return False
        return True

    @staticmethod
    def normalize(uri: str) -> str:
        return str(BehaviorURI(uri))

    @property
    def uri(self) -> str:
        return self._uri

    @property
    def full_path(self) -> str:
        return self._uri[len(f"{self.SCHEME}://") :]

    @property
    def decoded_path(self) -> str:
        return "/".join(self._segments)

    @property
    def segments(self) -> tuple[str, ...]:
        return self._segments

    @property
    def node_type(self) -> BehaviorURINodeType:
        return self._node_type

    @property
    def name(self) -> str | None:
        return self._segments[-1] if self._segments else None

    @property
    def is_root(self) -> bool:
        return not self._segments

    @property
    def parent(self) -> BehaviorURI | None:
        if not self._segments:
            return None
        return self._from_segments(self._segments[:-1])

    @property
    def containing_directory(self) -> BehaviorDirectory:
        if self._node_type is BehaviorURINodeType.DOCUMENT:
            assert self._address is not None
            return BehaviorDirectory.for_address(self._address)
        assert self._directory is not None
        return self._directory

    def to_address(self) -> BehaviorAddress:
        if self._node_type is not BehaviorURINodeType.DOCUMENT or self._address is None:
            raise BehaviorURIError("behavior URI does not identify an L2 document")
        return self._address

    def to_directory(self) -> BehaviorDirectory:
        if self._node_type is not BehaviorURINodeType.DIRECTORY or self._directory is None:
            raise BehaviorURIError("behavior URI does not identify a directory")
        return self._directory

    def to_layer(self) -> tuple[BehaviorDirectory, BehaviorLevel]:
        if self._node_type is not BehaviorURINodeType.LAYER or self._directory is None or self._level is None:
            raise BehaviorURIError("behavior URI does not identify a semantic layer")
        return self._directory, self._level

    def join(self, part: str) -> BehaviorURI:
        if self._node_type is not BehaviorURINodeType.DIRECTORY:
            raise BehaviorURIError("only a behavior directory URI can join a child")
        if not isinstance(part, str):
            raise TypeError("behavior URI join part must be a string")
        if not part or "/" in part or "\\" in part:
            raise BehaviorURIError("behavior URI join part must be one safe non-empty segment")
        return self._from_segments((*self._segments, part))

    def matches_prefix(self, prefix: BehaviorURI | str) -> bool:
        parsed = self.parse(prefix)
        return self._segments[: len(parsed._segments)] == parsed._segments

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


def _classify(
    segments: tuple[str, ...],
) -> tuple[BehaviorURINodeType, BehaviorAddress | None, BehaviorDirectory | None, BehaviorLevel | None]:
    directory = _directory(segments)
    if directory is not None:
        return BehaviorURINodeType.DIRECTORY, None, directory, None
    level = BehaviorLevel.from_sidecar_filename(segments[-1]) if segments else None
    if level is not None:
        owner = _directory(segments[:-1])
        if owner is not None:
            return BehaviorURINodeType.LAYER, None, owner, level
    address = _address(segments)
    if address is not None:
        return BehaviorURINodeType.DOCUMENT, address, None, None
    raise BehaviorURIError("behavior URI does not map to the confirmed behavior tree")


def _canonical_segments(
    node_type: BehaviorURINodeType,
    address: BehaviorAddress | None,
    directory: BehaviorDirectory | None,
    level: BehaviorLevel | None,
) -> tuple[str, ...]:
    if node_type is BehaviorURINodeType.DIRECTORY:
        assert directory is not None
        return directory.identity_parts
    if node_type is BehaviorURINodeType.LAYER:
        assert directory is not None and level is not None
        return (*directory.identity_parts, level.sidecar_filename)
    assert address is not None
    return _address_segments(address)


def _address_segments(address: BehaviorAddress) -> tuple[str, ...]:
    occurred_on = address.occurred_on
    return (
        *kind_directory_prefix(address.kind),
        f"{occurred_on.year:04d}",
        f"{occurred_on.month:02d}",
        f"{occurred_on.day:02d}",
        f"{address.identity_name}.md",
    )


def _directory(segments: tuple[str, ...]) -> BehaviorDirectory | None:
    try:
        return BehaviorDirectory(segments)
    except (TypeError, ValueError):
        return None


def _address(segments: tuple[str, ...]) -> BehaviorAddress | None:
    """L2 路径固定为「类型前缀 + YYYY/MM/DD + 叶文件」，长度不符即不是文档地址。"""

    kind = kind_for_directory_prefix(segments)
    if kind is None:
        return None
    depth = len(kind_directory_prefix(kind))
    if len(segments) != depth + 4:
        return None
    try:
        return BehaviorAddress.from_identity(
            kind,
            _date(segments[depth : depth + 3]),
            _markdown_stem(segments[depth + 3]),
        )
    except (TypeError, ValueError):
        return None


def _date(parts: tuple[str, ...]) -> date:
    if (
        len(parts) != 3
        or len(parts[0]) != 4
        or len(parts[1]) != 2
        or len(parts[2]) != 2
        or not all(is_ascii_digits(part) for part in parts)
    ):
        raise BehaviorURIError("behavior URI date must use YYYY/MM/DD")
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


def _markdown_stem(filename: str) -> str:
    if not filename.endswith(".md") or BehaviorLevel.from_sidecar_filename(filename) is not None:
        raise BehaviorURIError("behavior L2 URI must end with a non-reserved .md filename")
    stem = filename[:-3]
    if not stem:
        raise BehaviorURIError("behavior L2 URI filename must have a non-empty stem")
    return stem


def _decode_segment(value: str) -> str:
    index = 0
    while index < len(value):
        if value[index] != "%":
            index += 1
            continue
        if index + 2 >= len(value) or value[index + 1] not in _HEX_DIGITS or value[index + 2] not in _HEX_DIGITS:
            raise BehaviorURIError("behavior URI contains malformed percent encoding")
        index += 3
    try:
        decoded = unquote(value, encoding="utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise BehaviorURIError("behavior URI contains invalid UTF-8 percent encoding") from exc
    if not decoded:
        raise BehaviorURIError("behavior URI contains an empty decoded segment")
    return decoded


def _encode_segment(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise BehaviorURIError("behavior URI segments must be non-empty strings")
    encoded: list[str] = []
    for character in value:
        if character in _UNRESERVED_ASCII or ord(character) >= 128:
            encoded.append(character)
        else:
            encoded.extend(f"%{byte:02X}" for byte in character.encode("utf-8"))
    return "".join(encoded)


__all__ = ["BehaviorURI", "BehaviorURIError", "BehaviorURINodeType"]
