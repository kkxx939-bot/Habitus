"""严格映射情景树的 ``scene://`` URI。

逻辑地址不含物理代目录：``scene://scenes/YYYY/MM/DD/{label}--{ts}.md`` 指的是"这一天当前
生效的那一代里的这篇文档"，代由该日指针解析（见 ``scene.tree``）。这样引用它的下游（上下文
视图、结算账本）不随重建改地址。
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from urllib.parse import unquote

from habitus.behavior.model import is_ascii_digits
from habitus.scene.model import SCENES_SEGMENT, SceneAddress, SceneDirectory

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_UNRESERVED_ASCII = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")


class SceneURIError(ValueError):
    """URI 无法映射到已确认的情景树。"""


class SceneURINodeType(str, Enum):
    DIRECTORY = "directory"
    DOCUMENT = "document"


class SceneURI:
    """情景 URI 与地址、目录之间的不可变映射。"""

    SCHEME = "scene"
    _uri: str
    _segments: tuple[str, ...]
    _node_type: SceneURINodeType
    _address: SceneAddress | None
    _directory: SceneDirectory | None
    __slots__ = ("_address", "_directory", "_node_type", "_segments", "_uri")

    def __init__(self, uri: str) -> None:
        if not isinstance(uri, str):
            raise TypeError("scene URI must be a string")
        if not uri or uri != uri.strip():
            raise SceneURIError("scene URI must be non-empty without surrounding whitespace")
        prefix = f"{self.SCHEME}://"
        if not uri.startswith(prefix):
            raise SceneURIError(f"scene URI must start with '{prefix}'")
        raw_path = uri[len(prefix) :]
        if raw_path.endswith("/"):
            raise SceneURIError("scene URI must not contain a trailing slash")
        raw_segments = tuple(raw_path.split("/")) if raw_path else ()
        if any(not segment for segment in raw_segments):
            raise SceneURIError("scene URI contains an empty path segment")
        segments = tuple(_decode_segment(segment) for segment in raw_segments)
        node_type, address, directory = _classify(segments)
        canonical = directory.identity_parts if directory is not None else _address_segments(address)
        encoded_path = "/".join(_encode_segment(segment) for segment in canonical)
        object.__setattr__(self, "_uri", f"{prefix}{encoded_path}")
        object.__setattr__(self, "_segments", canonical)
        object.__setattr__(self, "_node_type", node_type)
        object.__setattr__(self, "_address", address)
        object.__setattr__(self, "_directory", directory)

    @classmethod
    def parse(cls, value: SceneURI | str) -> SceneURI:
        if isinstance(value, SceneURI):
            return value
        return cls(value)

    @classmethod
    def root(cls) -> SceneURI:
        return cls(f"{cls.SCHEME}://")

    @classmethod
    def from_address(cls, address: SceneAddress) -> SceneURI:
        if not isinstance(address, SceneAddress):
            raise TypeError("address must be a SceneAddress")
        return cls._from_segments(_address_segments(address))

    @classmethod
    def from_directory(cls, directory: SceneDirectory) -> SceneURI:
        if not isinstance(directory, SceneDirectory):
            raise TypeError("directory must be a SceneDirectory")
        return cls._from_segments(directory.identity_parts)

    @classmethod
    def _from_segments(cls, segments: tuple[str, ...]) -> SceneURI:
        encoded = "/".join(_encode_segment(segment) for segment in segments)
        return cls(f"{cls.SCHEME}://{encoded}")

    @staticmethod
    def is_valid(uri: object) -> bool:
        if not isinstance(uri, str):
            return False
        try:
            SceneURI(uri)
        except (TypeError, ValueError):
            return False
        return True

    @property
    def uri(self) -> str:
        return self._uri

    @property
    def segments(self) -> tuple[str, ...]:
        return self._segments

    @property
    def node_type(self) -> SceneURINodeType:
        return self._node_type

    @property
    def is_root(self) -> bool:
        return not self._segments

    def to_address(self) -> SceneAddress:
        if self._node_type is not SceneURINodeType.DOCUMENT or self._address is None:
            raise SceneURIError("scene URI does not identify an L2 document")
        return self._address

    def to_directory(self) -> SceneDirectory:
        if self._node_type is not SceneURINodeType.DIRECTORY or self._directory is None:
            raise SceneURIError("scene URI does not identify a directory")
        return self._directory

    def __str__(self) -> str:
        return self._uri

    def __repr__(self) -> str:
        return f"SceneURI({self._uri!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SceneURI):
            return self._uri == other._uri
        if isinstance(other, str):
            return self._uri == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._uri)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("SceneURI is immutable")


def _classify(
    segments: tuple[str, ...],
) -> tuple[SceneURINodeType, SceneAddress | None, SceneDirectory | None]:
    try:
        return SceneURINodeType.DIRECTORY, None, SceneDirectory(segments)
    except (TypeError, ValueError):
        pass
    address = _address(segments)
    if address is not None:
        return SceneURINodeType.DOCUMENT, address, None
    raise SceneURIError("scene URI does not map to the confirmed scene tree")


def _address_segments(address: SceneAddress | None) -> tuple[str, ...]:
    assert address is not None
    occurred_on = address.occurred_on
    return (
        SCENES_SEGMENT,
        f"{occurred_on.year:04d}",
        f"{occurred_on.month:02d}",
        f"{occurred_on.day:02d}",
        f"{address.identity_name}.md",
    )


def _address(segments: tuple[str, ...]) -> SceneAddress | None:
    if len(segments) != 5 or segments[0] != SCENES_SEGMENT:
        return None
    year, month, day, filename = segments[1:]
    if (
        len(year) != 4
        or len(month) != 2
        or len(day) != 2
        or not all(is_ascii_digits(part) for part in (year, month, day))
    ):
        return None
    if not filename.endswith(".md") or len(filename) <= 3:
        return None
    try:
        return SceneAddress.from_identity(date(int(year), int(month), int(day)), filename[:-3])
    except (TypeError, ValueError):
        return None


def _decode_segment(value: str) -> str:
    index = 0
    while index < len(value):
        if value[index] != "%":
            index += 1
            continue
        if index + 2 >= len(value) or value[index + 1] not in _HEX_DIGITS or value[index + 2] not in _HEX_DIGITS:
            raise SceneURIError("scene URI contains malformed percent encoding")
        index += 3
    try:
        decoded = unquote(value, encoding="utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SceneURIError("scene URI contains invalid UTF-8 percent encoding") from exc
    if not decoded:
        raise SceneURIError("scene URI contains an empty decoded segment")
    return decoded


def _encode_segment(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise SceneURIError("scene URI segments must be non-empty strings")
    encoded: list[str] = []
    for character in value:
        if character in _UNRESERVED_ASCII or ord(character) >= 128:
            encoded.append(character)
        else:
            encoded.extend(f"%{byte:02X}" for byte in character.encode("utf-8"))
    return "".join(encoded)


__all__ = ["SceneURI", "SceneURIError", "SceneURINodeType"]
