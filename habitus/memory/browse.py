"""按 ``memory://`` 地址浏览记忆树的公开只读视图（方案 36 第 2 步 ``habitus memory``）。

对外只用 URI 说位置——与召回命中里给的 ``memory://`` 同一套——不给本机文件路径。读走记忆树自己的公开读取：
跨正在发布的多文档事务读不到一致视图时报 ``MemoryTreeConsistencyError``（稍后重试即可）。
"""

from __future__ import annotations

from dataclasses import dataclass

from habitus.memory.document import MemoryDocument
from habitus.memory.model import MemoryLevel
from habitus.memory.tree import MemoryTree
from habitus.memory.uri import MemoryURI, MemoryURINodeType

#: 目录摘要（L0）随列目录一起给，超出这个大小只说"有"、不带原文
_ABSTRACT_MAX_BYTES = 4_096
#: 读 L0 / L1 层的上限：派生层本来就短，异常膨胀的不读
_LAYER_MAX_BYTES = 256 * 1024


class MemoryBrowseNotFoundError(LookupError):
    """URI 合法，但记忆树里此刻没有这个目录、文档或语义层。"""


@dataclass(frozen=True)
class MemoryListing:
    """一个目录的直接内容：子目录、L2 文档与这个目录的摘要（L0，没有为 ``None``）。"""

    uri: MemoryURI
    directories: tuple[MemoryURI, ...]
    documents: tuple[MemoryURI, ...]
    abstract: str | None


@dataclass(frozen=True)
class MemoryLayerText:
    """目录的一层派生文字（L0 摘要或 L1 概览）。"""

    uri: MemoryURI
    level: MemoryLevel
    text: str


class MemoryBrowser:
    """列目录、读文档与语义层；不写、不检索。"""

    def __init__(self, tree: MemoryTree) -> None:
        if not isinstance(tree, MemoryTree):
            raise TypeError("tree must be a MemoryTree")
        self.tree = tree

    def list(self, uri: MemoryURI | str | None = None) -> MemoryListing:
        """列一个目录（省略时是根）的直接子目录和文档，按名字排序。"""

        parsed = MemoryURI.root() if uri is None else MemoryURI.parse(uri)
        if parsed.node_type is not MemoryURINodeType.DIRECTORY:
            raise ValueError("memory listing needs a directory URI")
        directory = parsed.to_directory()
        if not self.tree.directory_exists(directory):
            raise MemoryBrowseNotFoundError(f"no memory directory at {parsed}")
        directories = tuple(
            sorted((MemoryURI.from_directory(child) for child in self.tree.child_directories(directory)), key=str)
        )
        documents = tuple(
            sorted((MemoryURI.from_address(address) for address in self.tree.direct_addresses(directory)), key=str)
        )
        abstract = None
        if not parsed.is_root and self.tree.layer_exists(directory, MemoryLevel.ABSTRACT):
            try:
                abstract = self.tree.read_layer_bounded(directory, MemoryLevel.ABSTRACT, max_bytes=_ABSTRACT_MAX_BYTES)
            except ValueError:
                abstract = None  # 超长或读不了：列目录不因摘要失败
        return MemoryListing(parsed, directories, documents, abstract)

    def read(self, uri: MemoryURI | str) -> MemoryDocument | MemoryLayerText:
        """读一个 L2 文档（结构化），或一个目录的 L0 / L1 层（原文）。"""

        parsed = MemoryURI.parse(uri)
        if parsed.node_type is MemoryURINodeType.DOCUMENT:
            address = parsed.to_address()
            if not self.tree.exists(address):
                raise MemoryBrowseNotFoundError(f"no memory document at {parsed}")
            return self.tree.read(address)
        if parsed.node_type is MemoryURINodeType.LAYER:
            directory, level = parsed.to_layer()
            if not self.tree.layer_exists(directory, level):
                raise MemoryBrowseNotFoundError(f"no semantic layer at {parsed}")
            return MemoryLayerText(parsed, level, self.tree.read_layer_bounded(directory, level, max_bytes=_LAYER_MAX_BYTES))
        raise ValueError("memory read needs a document or semantic layer URI; list a directory instead")


__all__ = ["MemoryBrowseNotFoundError", "MemoryBrowser", "MemoryLayerText", "MemoryListing"]
