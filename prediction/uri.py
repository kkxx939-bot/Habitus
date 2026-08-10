"""严格映射预测样本树的 ``prediction://`` URI。"""

from __future__ import annotations

from datetime import date
from enum import Enum
from urllib.parse import unquote

from prediction.model import (
    PredictionAddress,
    PredictionDirectory,
    PredictionKind,
    PredictionLevel,
    PredictionPatternAddress,
    PredictionPatternKind,
    is_ascii_digits,
    prediction_shard,
)

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_UNRESERVED_ASCII = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")


class PredictionURIError(ValueError):
    """URI 无法映射到已确认的预测树。"""


class PredictionURINodeType(str, Enum):
    DIRECTORY = "directory"
    DOCUMENT = "document"
    LAYER = "layer"


class PredictionURI:
    """预测 URI、样本地址、目录和派生语义层之间的不可变映射。"""

    SCHEME = "prediction"
    _uri: str
    _segments: tuple[str, ...]
    _node_type: PredictionURINodeType
    _address: PredictionAddress | None
    _pattern_address: PredictionPatternAddress | None
    _directory: PredictionDirectory | None
    _level: PredictionLevel | None
    __slots__ = ("_address", "_directory", "_level", "_node_type", "_pattern_address", "_segments", "_uri")

    def __init__(self, uri: str) -> None:
        if not isinstance(uri, str):
            raise TypeError("prediction URI must be a string")
        if not uri or uri != uri.strip():
            raise PredictionURIError("prediction URI must be non-empty without surrounding whitespace")
        prefix = f"{self.SCHEME}://"
        if not uri.startswith(prefix):
            raise PredictionURIError(f"prediction URI must start with '{prefix}'")
        raw_path = uri[len(prefix) :]
        if raw_path.endswith("/"):
            raise PredictionURIError("prediction URI must not contain a trailing slash")
        raw_segments = tuple(raw_path.split("/")) if raw_path else ()
        if any(not segment for segment in raw_segments):
            raise PredictionURIError("prediction URI contains an empty path segment")
        segments = tuple(_decode_segment(segment) for segment in raw_segments)
        node_type, address, pattern_address, directory, level = _classify(segments)
        canonical = _canonical_segments(node_type, address, pattern_address, directory, level)
        encoded_path = "/".join(_encode_segment(segment) for segment in canonical)
        object.__setattr__(self, "_uri", f"{prefix}{encoded_path}")
        object.__setattr__(self, "_segments", canonical)
        object.__setattr__(self, "_node_type", node_type)
        object.__setattr__(self, "_address", address)
        object.__setattr__(self, "_pattern_address", pattern_address)
        object.__setattr__(self, "_directory", directory)
        object.__setattr__(self, "_level", level)

    @classmethod
    def parse(cls, value: PredictionURI | str) -> PredictionURI:
        if isinstance(value, PredictionURI):
            return value
        return cls(value)

    @classmethod
    def root(cls) -> PredictionURI:
        return cls(f"{cls.SCHEME}://")

    @classmethod
    def from_address(cls, address: PredictionAddress) -> PredictionURI:
        if not isinstance(address, PredictionAddress):
            raise TypeError("address must be a PredictionAddress")
        return cls._from_segments(_address_segments(address))

    @classmethod
    def from_pattern_address(cls, address: PredictionPatternAddress) -> PredictionURI:
        if not isinstance(address, PredictionPatternAddress):
            raise TypeError("address must be a PredictionPatternAddress")
        return cls._from_segments(_pattern_address_segments(address))

    @classmethod
    def from_directory(cls, directory: PredictionDirectory) -> PredictionURI:
        if not isinstance(directory, PredictionDirectory):
            raise TypeError("directory must be a PredictionDirectory")
        return cls._from_segments(directory.identity_parts)

    @classmethod
    def from_layer(cls, directory: PredictionDirectory, level: PredictionLevel) -> PredictionURI:
        if not isinstance(directory, PredictionDirectory):
            raise TypeError("directory must be a PredictionDirectory")
        normalized = PredictionLevel(level)
        return cls._from_segments((*directory.identity_parts, normalized.sidecar_filename))

    @classmethod
    def _from_segments(cls, segments: tuple[str, ...]) -> PredictionURI:
        encoded = "/".join(_encode_segment(segment) for segment in segments)
        return cls(f"{cls.SCHEME}://{encoded}")

    @property
    def uri(self) -> str:
        return self._uri

    @property
    def segments(self) -> tuple[str, ...]:
        return self._segments

    @property
    def node_type(self) -> PredictionURINodeType:
        return self._node_type

    @property
    def is_root(self) -> bool:
        return not self._segments

    @property
    def is_pattern_document(self) -> bool:
        return self._pattern_address is not None

    @property
    def parent(self) -> PredictionURI | None:
        if not self._segments:
            return None
        return self._from_segments(self._segments[:-1])

    @property
    def containing_directory(self) -> PredictionDirectory:
        if self._node_type is PredictionURINodeType.DOCUMENT:
            if self._address is not None:
                return PredictionDirectory.for_address(self._address)
            assert self._pattern_address is not None
            return PredictionDirectory.for_pattern_address(self._pattern_address)
        assert self._directory is not None
        return self._directory

    def to_address(self) -> PredictionAddress:
        if self._node_type is not PredictionURINodeType.DOCUMENT or self._address is None:
            raise PredictionURIError("prediction URI does not identify a sample document")
        return self._address

    def to_pattern_address(self) -> PredictionPatternAddress:
        if self._node_type is not PredictionURINodeType.DOCUMENT or self._pattern_address is None:
            raise PredictionURIError("prediction URI does not identify a pattern document")
        return self._pattern_address

    def to_directory(self) -> PredictionDirectory:
        if self._node_type is not PredictionURINodeType.DIRECTORY or self._directory is None:
            raise PredictionURIError("prediction URI does not identify a directory")
        return self._directory

    def to_layer(self) -> tuple[PredictionDirectory, PredictionLevel]:
        if self._node_type is not PredictionURINodeType.LAYER or self._directory is None or self._level is None:
            raise PredictionURIError("prediction URI does not identify a semantic layer")
        return self._directory, self._level

    def matches_prefix(self, prefix: PredictionURI | str) -> bool:
        parsed = self.parse(prefix)
        return self._segments[: len(parsed._segments)] == parsed._segments

    def __str__(self) -> str:
        return self._uri

    def __repr__(self) -> str:
        return f"PredictionURI({self._uri!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PredictionURI):
            return self._uri == other._uri
        if isinstance(other, str):
            return self._uri == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._uri)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("PredictionURI is immutable")


def _classify(
    segments: tuple[str, ...],
) -> tuple[
    PredictionURINodeType,
    PredictionAddress | None,
    PredictionPatternAddress | None,
    PredictionDirectory | None,
    PredictionLevel | None,
]:
    directory = _directory(segments)
    if directory is not None:
        return PredictionURINodeType.DIRECTORY, None, None, directory, None
    level = PredictionLevel.from_sidecar_filename(segments[-1]) if segments else None
    if level is not None:
        owner = _directory(segments[:-1])
        if owner is not None:
            return PredictionURINodeType.LAYER, None, None, owner, level
    address = _address(segments)
    if address is not None:
        return PredictionURINodeType.DOCUMENT, address, None, None, None
    pattern_address = _pattern_address(segments)
    if pattern_address is not None:
        return PredictionURINodeType.DOCUMENT, None, pattern_address, None, None
    raise PredictionURIError("prediction URI does not map to the confirmed prediction tree")


def _canonical_segments(
    node_type: PredictionURINodeType,
    address: PredictionAddress | None,
    pattern_address: PredictionPatternAddress | None,
    directory: PredictionDirectory | None,
    level: PredictionLevel | None,
) -> tuple[str, ...]:
    if node_type is PredictionURINodeType.DIRECTORY:
        assert directory is not None
        return directory.identity_parts
    if node_type is PredictionURINodeType.LAYER:
        assert directory is not None and level is not None
        return (*directory.identity_parts, level.sidecar_filename)
    if address is not None:
        return _address_segments(address)
    assert pattern_address is not None
    return _pattern_address_segments(pattern_address)


def _address_segments(address: PredictionAddress) -> tuple[str, ...]:
    sample_date = address.sample_date
    return (
        "samples",
        address.kind.branch_name,
        f"{sample_date.year:04d}",
        f"{sample_date.month:02d}",
        f"{sample_date.day:02d}",
        prediction_shard(address.materialization_id),
        f"{address.materialization_id}.md",
    )


def _pattern_address_segments(address: PredictionPatternAddress) -> tuple[str, ...]:
    shard = prediction_shard(address.state_key)
    if address.kind is PredictionPatternKind.STATE:
        return "patterns", "states", shard, f"{address.state_key}.md"
    assert address.branch_key is not None
    return "patterns", "branches", shard, address.state_key, f"{address.branch_key}.md"


def _directory(segments: tuple[str, ...]) -> PredictionDirectory | None:
    try:
        return PredictionDirectory(segments)
    except (TypeError, ValueError):
        return None


def _address(segments: tuple[str, ...]) -> PredictionAddress | None:
    try:
        if len(segments) != 7 or segments[0] != "samples":
            return None
        kind = PredictionKind.from_branch_name(segments[1])
        sample_date = _date(segments[2:5])
        filename = segments[6]
        if not filename.endswith(".md") or PredictionLevel.from_sidecar_filename(filename) is not None:
            return None
        address = PredictionAddress(kind, sample_date, filename[:-3])
        if prediction_shard(address.materialization_id) != segments[5]:
            return None
        return address
    except (TypeError, ValueError):
        return None


def _pattern_address(segments: tuple[str, ...]) -> PredictionPatternAddress | None:
    try:
        if len(segments) == 4 and segments[:2] == ("patterns", "states"):
            filename = segments[3]
            if not filename.endswith(".md"):
                return None
            address = PredictionPatternAddress(PredictionPatternKind.STATE, filename[:-3])
        elif len(segments) == 5 and segments[:2] == ("patterns", "branches"):
            filename = segments[4]
            if not filename.endswith(".md"):
                return None
            address = PredictionPatternAddress(PredictionPatternKind.BRANCH, segments[3], filename[:-3])
        else:
            return None
        if prediction_shard(address.state_key) != segments[2]:
            return None
        return address
    except (TypeError, ValueError):
        return None
    return None


def _date(parts: tuple[str, ...]) -> date:
    if (
        len(parts) != 3
        or tuple(map(len, parts)) != (4, 2, 2)
        or not all(is_ascii_digits(part) for part in parts)
    ):
        raise PredictionURIError("prediction URI date must use YYYY/MM/DD")
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


def _decode_segment(value: str) -> str:
    index = 0
    while index < len(value):
        if value[index] != "%":
            index += 1
            continue
        if index + 2 >= len(value) or value[index + 1] not in _HEX_DIGITS or value[index + 2] not in _HEX_DIGITS:
            raise PredictionURIError("prediction URI contains malformed percent encoding")
        index += 3
    try:
        decoded = unquote(value, encoding="utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise PredictionURIError("prediction URI contains invalid UTF-8 percent encoding") from exc
    if not decoded:
        raise PredictionURIError("prediction URI contains an empty decoded segment")
    return decoded


def _encode_segment(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise PredictionURIError("prediction URI segments must be non-empty strings")
    encoded: list[str] = []
    for character in value:
        if character in _UNRESERVED_ASCII or ord(character) >= 128:
            encoded.append(character)
        else:
            encoded.extend(f"%{byte:02X}" for byte in character.encode("utf-8"))
    return "".join(encoded)


__all__ = ["PredictionURI", "PredictionURIError", "PredictionURINodeType"]
