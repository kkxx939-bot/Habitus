"""``habitus://`` 地址里四类卡片那一部分的值对象：只认卡片、卡片目录和它们的 L0/L1 侧车。"""

from __future__ import annotations

from enum import Enum

from habitus.infrastructure.tree.address import SCHEME, TreeAddressError, format_uri, split_uri
from habitus.memory.model import MEMORY_DIRECTORIES, MemoryAddress, MemoryDirectory, MemoryKind, MemoryLevel


class MemoryURIError(ValueError):
    """URI 不是规范的 ``habitus://`` 地址，或不是卡片、卡片目录及其侧车。"""


class MemoryURINodeType(str, Enum):
    """URI 指向的记忆树节点类型。"""

    DIRECTORY = "directory"
    DOCUMENT = "document"
    LAYER = "layer"


class MemoryURI:
    """把 URI 与 ``MemoryAddress``、``MemoryDirectory``、语义层双向转换。

    URI 只表达记忆树内的稳定位置，不包含 owner、租户、文档 ID 或旧 Context
    scope。构造时允许合法的百分号编码和直接 Unicode，保存时统一为可读的
    规范 IRI：Unicode 保持原文，保留字符使用大写百分号编码。
    """

    SCHEME = SCHEME
    _uri: str
    _segments: tuple[str, ...]
    _node_type: MemoryURINodeType
    _address: MemoryAddress | None
    _directory: MemoryDirectory | None
    _level: MemoryLevel | None
    __slots__ = (
        "_address",
        "_directory",
        "_level",
        "_node_type",
        "_segments",
        "_uri",
    )

    def __init__(self, uri: str) -> None:
        try:
            segments = split_uri(uri)
        except TreeAddressError as exc:
            raise MemoryURIError(str(exc)) from exc
        node_type, address, directory, level = _classify(segments)
        segments = _canonical_segments(node_type, address, directory, level)

        object.__setattr__(self, "_uri", format_uri(segments))
        object.__setattr__(self, "_segments", segments)
        object.__setattr__(self, "_node_type", node_type)
        object.__setattr__(self, "_address", address)
        object.__setattr__(self, "_directory", directory)
        object.__setattr__(self, "_level", level)

    @classmethod
    def parse(cls, value: MemoryURI | str) -> MemoryURI:
        """返回已经验证并规范化的 URI 值对象。"""

        if isinstance(value, MemoryURI):
            return value
        return cls(value)

    @classmethod
    def root(cls) -> MemoryURI:
        """返回记忆树根 URI。"""

        return cls(f"{cls.SCHEME}://")

    @classmethod
    def from_address(cls, address: MemoryAddress) -> MemoryURI:
        """从一个 L2 语义地址构造唯一 URI。"""

        segments: tuple[str, ...]
        if address.kind is MemoryKind.PROFILE:
            segments = ("profile.md",)
        elif address.kind is MemoryKind.PREFERENCE:
            segments = ("preferences", f"{address.identity_name}.md")
        elif address.kind is MemoryKind.ENTITY:
            segments = (
                "entities",
                address.identity_category,
                f"{address.identity_name}.md",
            )
        else:
            segments = ("tools", f"{address.identity_name}.md")
        return cls._from_segments(segments)

    @classmethod
    def from_directory(cls, directory: MemoryDirectory) -> MemoryURI:
        """从受控目录地址构造唯一 URI。"""

        return cls._from_segments(directory.identity_parts)

    @classmethod
    def from_layer(
        cls,
        directory: MemoryDirectory,
        level: MemoryLevel,
    ) -> MemoryURI:
        """构造目录 L0 或 L1 侧车文件 URI。"""

        normalized_level = MemoryLevel(level)
        return cls._from_segments((*directory.identity_parts, normalized_level.sidecar_filename))

    @classmethod
    def build(cls, *path_parts: str) -> MemoryURI:
        """使用已经解码的单段名称构造 URI，不接受含斜杠的复合片段。"""

        segments: list[str] = []
        for part in path_parts:
            if not part or "/" in part or "\\" in part:
                raise MemoryURIError("memory URI path parts must be safe non-empty segments")
            segments.append(part)
        return cls._from_segments(tuple(segments))

    @classmethod
    def _from_segments(cls, segments: tuple[str, ...]) -> MemoryURI:
        return cls(format_uri(segments))

    @staticmethod
    def is_valid(uri: object) -> bool:
        """判断输入能否映射为一个合法记忆树节点。"""

        if not isinstance(uri, str):
            return False
        try:
            MemoryURI(uri)
        except (TypeError, ValueError):
            return False
        return True

    @staticmethod
    def normalize(uri: str) -> str:
        """严格解析 URI 后返回唯一规范字符串，不补全短路径。"""

        return str(MemoryURI(uri))

    @property
    def uri(self) -> str:
        return self._uri

    @property
    def full_path(self) -> str:
        """返回不包含 scheme 的规范路径。"""

        return self._uri[len(f"{self.SCHEME}://") :]

    @property
    def segments(self) -> tuple[str, ...]:
        return self._segments

    @property
    def node_type(self) -> MemoryURINodeType:
        return self._node_type

    @property
    def name(self) -> str | None:
        return self._segments[-1] if self._segments else None

    @property
    def is_root(self) -> bool:
        return not self._segments

    @property
    def parent(self) -> MemoryURI | None:
        """返回合法父节点；记忆根没有父节点。"""

        if not self._segments:
            return None
        return self._from_segments(self._segments[:-1])

    def to_address(self) -> MemoryAddress:
        """把 L2 文档 URI 反解为 ``MemoryAddress``。"""

        if self._node_type is not MemoryURINodeType.DOCUMENT or self._address is None:
            raise MemoryURIError("memory URI does not identify an L2 document")
        return self._address

    def to_directory(self) -> MemoryDirectory:
        """把目录 URI 反解为 ``MemoryDirectory``。"""

        if self._node_type is not MemoryURINodeType.DIRECTORY or self._directory is None:
            raise MemoryURIError("memory URI does not identify a directory")
        return self._directory

    def to_layer(self) -> tuple[MemoryDirectory, MemoryLevel]:
        """把侧车文件 URI 反解为目录与 L0/L1 层级。"""

        if self._node_type is not MemoryURINodeType.LAYER or self._directory is None or self._level is None:
            raise MemoryURIError("memory URI does not identify a semantic layer")
        return self._directory, self._level

    def join(self, part: str) -> MemoryURI:
        """在当前目录后追加一个已解码的安全单段，并验证最终树节点。"""

        if self._node_type is not MemoryURINodeType.DIRECTORY:
            raise MemoryURIError("only a memory directory URI can join a child")
        if not part or "/" in part or "\\" in part:
            raise MemoryURIError("memory URI join part must be one safe non-empty segment")
        return self._from_segments((*self._segments, part))

    def __str__(self) -> str:
        return self._uri

    def __repr__(self) -> str:
        return f"MemoryURI({self._uri!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MemoryURI):
            return self._uri == other._uri
        if isinstance(other, str):
            return self._uri == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._uri)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("MemoryURI is immutable")


def decode_memory_uri(value: object) -> MemoryURI:
    """外部数据（磁盘记录、数据库行、模型输出）读回的 URI 在构造前确认是字符串，文字照抄 ``MemoryURI.__init__``。"""

    if not isinstance(value, str):
        raise TypeError("memory URI must be a string")
    return MemoryURI(value)


def _classify(
    segments: tuple[str, ...],
) -> tuple[
    MemoryURINodeType,
    MemoryAddress | None,
    MemoryDirectory | None,
    MemoryLevel | None,
]:
    directory = _directory(segments)
    if directory is not None:
        return MemoryURINodeType.DIRECTORY, None, directory, None

    level = MemoryLevel.from_sidecar_filename(segments[-1]) if segments else None
    if level is not None:
        owner = _directory(segments[:-1])
        if owner is not None:
            return (
                MemoryURINodeType.LAYER,
                None,
                owner,
                level,
            )

    address = _address(segments)
    if address is not None:
        return MemoryURINodeType.DOCUMENT, address, None, None
    raise MemoryURIError("habitus URI does not identify a memory card, card directory or its layer")


def _canonical_segments(
    node_type: MemoryURINodeType,
    address: MemoryAddress | None,
    directory: MemoryDirectory | None,
    level: MemoryLevel | None,
) -> tuple[str, ...]:
    if node_type is MemoryURINodeType.DIRECTORY:
        assert directory is not None
        return directory.identity_parts
    if node_type is MemoryURINodeType.LAYER:
        assert directory is not None and level is not None
        return (*directory.identity_parts, level.sidecar_filename)
    assert address is not None
    if address.kind is MemoryKind.PROFILE:
        return ("profile.md",)
    if address.kind is MemoryKind.PREFERENCE:
        return ("preferences", f"{address.identity_name}.md")
    if address.kind is MemoryKind.ENTITY:
        return (
            "entities",
            address.identity_category,
            f"{address.identity_name}.md",
        )
    return ("tools", f"{address.identity_name}.md")


def _directory(segments: tuple[str, ...]) -> MemoryDirectory | None:
    if not segments:
        return MemoryDirectory.root()
    if segments[0] not in MEMORY_DIRECTORIES:
        return None
    try:
        return MemoryDirectory(segments)
    except (TypeError, ValueError):
        return None


def _address(segments: tuple[str, ...]) -> MemoryAddress | None:
    try:
        if segments == ("profile.md",):
            return MemoryAddress.profile()
        if len(segments) == 2 and segments[0] == "preferences":
            return MemoryAddress.preference(_markdown_stem(segments[1]))
        if len(segments) == 3 and segments[0] == "entities":
            return MemoryAddress.entity(segments[1], _markdown_stem(segments[2]))
        if len(segments) == 2 and segments[0] == "tools":
            return MemoryAddress.tool(_markdown_stem(segments[1]))
    except (TypeError, ValueError):
        return None
    return None


def _markdown_stem(filename: str) -> str:
    if not filename.endswith(".md") or MemoryLevel.from_sidecar_filename(filename) is not None:
        raise MemoryURIError("memory L2 URI must end with a non-reserved .md filename")
    stem = filename[:-3]
    if not stem:
        raise MemoryURIError("memory L2 URI filename must have a non-empty stem")
    return stem


__all__ = ["MemoryURI", "MemoryURIError", "MemoryURINodeType", "decode_memory_uri"]
