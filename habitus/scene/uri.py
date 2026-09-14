"""严格映射情景树的 ``scene://`` URI。

逻辑地址不含物理代目录：``scene://scenes/YYYY/MM/DD/{label}--{ts}.md`` 指的是"这一天当前
生效的那一代里的这篇文档"，代由该日指针解析（见 ``scene.tree``）。这样引用它的下游（上下文
视图、结算账本）不随重建改地址。
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from urllib.parse import unquote

from habitus.behavior.model import is_ascii_digits
from habitus.scene.model import (
    KINDS_SEGMENT,
    AssociationAddress,
    KindDirectory,
    RegularityLevel,
)

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_UNRESERVED_ASCII = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")


class SceneURIError(ValueError):
    """URI 无法映射到已确认的情景树。"""


class SceneURINodeType(str, Enum):
    DIRECTORY = "directory"
    DOCUMENT = "document"
    # 规律级的 L0 / L1 侧车；与 ``behavior://`` 的 LAYER 同一个位置。没有它，判断者引用不到
    # "这个候选通常因为什么发生"那一句，而它正是候选清单上每行要给的东西。
    LAYER = "layer"


class SceneURI:
    """情景 URI 与地址、目录之间的不可变映射。"""

    SCHEME = "scene"
    _uri: str
    _segments: tuple[str, ...]
    _node_type: SceneURINodeType
    _target: AssociationAddress | KindDirectory | tuple[KindDirectory, RegularityLevel]
    __slots__ = ("_node_type", "_segments", "_target", "_uri")

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
        node_type, target = _classify(segments)
        canonical = _canonical_segments(target)
        encoded_path = "/".join(_encode_segment(segment) for segment in canonical)
        object.__setattr__(self, "_uri", f"{prefix}{encoded_path}")
        object.__setattr__(self, "_segments", canonical)
        object.__setattr__(self, "_node_type", node_type)
        object.__setattr__(self, "_target", target)

    @classmethod
    def parse(cls, value: SceneURI | str) -> SceneURI:
        if isinstance(value, SceneURI):
            return value
        return cls(value)

    @classmethod
    def root(cls) -> SceneURI:
        return cls(f"{cls.SCHEME}://")



    @classmethod
    def from_directory(cls, directory: KindDirectory) -> SceneURI:
        """一个目录的 URI：候选，或它下面按年 / 月 / 日分的片。"""

        if not isinstance(directory, KindDirectory):
            raise TypeError("directory must be a KindDirectory")
        return cls._from_segments(directory.parts)

    @classmethod
    def from_layer(cls, directory: KindDirectory, level: RegularityLevel) -> SceneURI:
        """引用某个候选的 L0 / L1 侧车。"""

        if not isinstance(directory, KindDirectory):
            raise TypeError("directory must be a KindDirectory")
        return cls._from_segments((*directory.parts, RegularityLevel(level).sidecar_filename))

    @classmethod
    def from_association(cls, address: AssociationAddress) -> SceneURI:
        if not isinstance(address, AssociationAddress):
            raise TypeError("address must be an AssociationAddress")
        return cls._from_segments(_association_segments(address))

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

    @property
    def is_association(self) -> bool:
        """指向规律级的一次关联记录（``kinds/...``），而不是旧的按天情景文档。"""

        return isinstance(self._target, AssociationAddress)


    def to_association(self) -> AssociationAddress:
        if not isinstance(self._target, AssociationAddress):
            raise SceneURIError("scene URI does not identify an association record")
        return self._target


    def to_kind_directory(self) -> KindDirectory:
        if not isinstance(self._target, KindDirectory):
            raise SceneURIError("scene URI does not identify a kind directory")
        return self._target

    def to_layer(self) -> tuple[KindDirectory, RegularityLevel]:
        if self._node_type is not SceneURINodeType.LAYER or not isinstance(self._target, tuple):
            raise SceneURIError("scene URI does not identify a semantic layer")
        return self._target

    def started_at(self) -> datetime:
        """文档身份里的开始时刻。

        链接层的 lag 校验（"晚指早"、lag 等于起止时刻差）对两种文档形态是同一条契约，所以
        取时刻这件事不能按形态分叉。
        """

        if isinstance(self._target, AssociationAddress):
            return self._target.started_at
        raise SceneURIError("scene URI does not identify a document")

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
) -> tuple[
    SceneURINodeType,
    AssociationAddress | KindDirectory | tuple[KindDirectory, RegularityLevel],
]:
    """一个 scheme 两片区域：旧的按天情景（``scenes/``）与规律级（``kinds/``）。

    分成两个 scheme 会让链接层跟着分叉，而"晚指早、lag 等于起止时刻差"这两条契约对两边是
    同一条；所以同一个 scheme 认两种文档形态。
    """

    for directory_type in (KindDirectory,):
        try:
            return SceneURINodeType.DIRECTORY, directory_type(segments)
        except (TypeError, ValueError):
            continue
    for parse in (_association,):
        target = parse(segments)
        if target is not None:
            return SceneURINodeType.DOCUMENT, target
    layer = _layer(segments)
    if layer is not None:
        return SceneURINodeType.LAYER, layer
    raise SceneURIError("scene URI does not map to the confirmed scene tree")


def _layer(segments: tuple[str, ...]) -> tuple[KindDirectory, RegularityLevel] | None:
    if len(segments) < 2:
        return None
    level = RegularityLevel.from_sidecar_filename(segments[-1])
    if level is None:
        return None
    try:
        return KindDirectory(segments[:-1]), level
    except (TypeError, ValueError):
        return None


def _canonical_segments(
    target: AssociationAddress | KindDirectory | tuple[KindDirectory, RegularityLevel],
) -> tuple[str, ...]:
    if isinstance(target, tuple):
        directory, level = target
        return (*directory.parts, level.sidecar_filename)
    if isinstance(target, KindDirectory):
        return target.parts
    assert isinstance(target, AssociationAddress)
    return _association_segments(target)


def _association_segments(address: AssociationAddress) -> tuple[str, ...]:
    occurred_on = address.occurred_on
    return (
        KINDS_SEGMENT,
        address.identity_kind,
        f"{occurred_on.year:04d}",
        f"{occurred_on.month:02d}",
        f"{occurred_on.day:02d}",
        f"{address.identity_name}.md",
    )


def _association(segments: tuple[str, ...]) -> AssociationAddress | None:
    if len(segments) != 6 or segments[0] != KINDS_SEGMENT:
        return None
    kind_token, year, month, day, filename = segments[1:]
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
        address = AssociationAddress.from_identity(kind_token, date(int(year), int(month), int(day)), filename[:-3])
    except (TypeError, ValueError):
        return None
    # URI 必须**已经是**规范形式：解析出来的 segments 与规范 segments 不一致，说明这个 URI 指的
    # 是另一个文件名。此前没有这道检查，非规范的时间戳会被"修复"成一个磁盘上并不存在的路径。
    return address if _association_segments(address) == segments else None






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
