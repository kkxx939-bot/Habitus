"""habitus 树里四类卡片那一部分的读写：档案、偏好、实体、工具（文件操作见 ``infrastructure/tree``）。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from habitus.infrastructure.store.filesystem import DurableDirectoryEntry
from habitus.infrastructure.tree.files import TreeFiles
from habitus.infrastructure.tree.layout import ROOT_DIRECTORIES, ROOT_FILES
from habitus.memory.document import (
    MemoryDocument,
    MemoryDocumentCodec,
    MemoryDocumentConfig,
    MemoryDocumentIntegrityError,
    MemoryDocumentLimitError,
)
from habitus.memory.model import MEMORY_DIRECTORIES, MemoryAddress, MemoryDirectory, MemoryKind, MemoryLevel
from habitus.memory.tree.config import MemoryTreeConfig
from habitus.memory.uri import MemoryURI, MemoryURINodeType


class MemoryTreeIntegrityError(ValueError):
    """记忆树包含路径逃逸、符号链接或不符合目录结构的条目。"""


class MemoryTree:
    """安全持久化结构化 L2 文档和可重建的目录语义层。树根下还有事件与空白的目录：根目录检查认它们，枚举和概览不走进去。"""

    _STATIC_DIRECTORIES = MEMORY_DIRECTORIES

    def __init__(
        self,
        root: str | Path,
        *,
        document_codec: MemoryDocumentCodec | None = None,
        document_config: MemoryDocumentConfig | None = None,
        tree_config: MemoryTreeConfig | None = None,
    ) -> None:
        self.tree_config = tree_config or MemoryTreeConfig()
        self._files = TreeFiles(
            root,
            max_children=self.tree_config.max_children_per_directory,
            error=MemoryTreeIntegrityError,
            conflict=MemoryTreeIntegrityError,
        )
        self.root = self._files.root
        if document_codec is None:
            from habitus.memory.schema import MemorySchemaRegistry

            document_codec = MemoryDocumentCodec(MemorySchemaRegistry.load_default())
        self._document_codec = document_codec
        self.document_config = document_config or MemoryDocumentConfig()

    @property
    def document_codec(self) -> MemoryDocumentCodec:
        """返回记忆树实际用于规范 L2 读写的同一编解码器。"""

        return self._document_codec

    def require_outside(self, path: str | Path) -> Path:
        """把树外工作目录（任务、生命周期状态）规范成绝对路径；落在 L2 记忆树内的位置一律拒绝。"""

        resolved = Path(path).expanduser().resolve(strict=False)
        if resolved.is_relative_to(self.root):
            raise ValueError("workflow state must be stored outside the habitus tree")
        return resolved

    def initialize(self) -> Path:
        """先检查树根只含一棵 habitus 树该有的条目，再创建卡片的静态目录；没有真实内容时不创建空的 profile.md。

        旧记忆树（含已退出记忆类型的目录）不兼容也不迁移，启动时在这里就明确报错，
        不等到第一条记忆任务刷新目录概览时才失败。
        """

        self._ensure_directory(self.root)
        self._require_known_root_entries()
        for name in self._STATIC_DIRECTORIES:
            self._ensure_directory(self.root / name)
        return self.root

    def path_for(self, address: MemoryAddress) -> Path:
        """把经过验证的语义地址确定性映射为唯一 Markdown 路径。"""

        relative = self._relative_path(address)
        candidate = self.root / relative
        self._require_inside_root(candidate)
        return candidate

    def _existing_document_path(self, address: MemoryAddress) -> Path:
        return self._existing_relative_path(self._relative_path(address), leaf_is_file=True)

    def _existing_directory_path(self, directory: MemoryDirectory) -> Path:
        return self._existing_relative_path(Path(*directory.identity_parts), leaf_is_file=False)

    def _existing_layer_path(self, directory: MemoryDirectory, level: MemoryLevel) -> Path:
        normalized = MemoryLevel(level)
        relative = Path(*directory.identity_parts, normalized.sidecar_filename)
        return self._existing_relative_path(relative, leaf_is_file=True)

    def _existing_relative_path(self, relative: Path, *, leaf_is_file: bool) -> Path:
        return self._files.resolve(relative, leaf_is_file=leaf_is_file)

    def directory_path(self, directory: MemoryDirectory) -> Path:
        """把受控目录地址映射到记忆树内的真实目录。"""

        candidate = self.root.joinpath(*directory.identity_parts)
        self._require_inside_root(candidate)
        return candidate

    def layer_path(self, directory: MemoryDirectory, level: MemoryLevel) -> Path:
        """返回目录 L0 或 L1 侧车文件的确定性路径。"""

        normalized = MemoryLevel(level)
        return self.directory_path(directory) / normalized.sidecar_filename

    def path_for_uri(self, uri: MemoryURI | str) -> Path:
        """把卡片、卡片目录或侧车的 ``habitus://`` 地址确定性映射为树内物理路径。"""

        parsed = MemoryURI.parse(uri)
        if parsed.node_type is MemoryURINodeType.DOCUMENT:
            return self.path_for(parsed.to_address())
        if parsed.node_type is MemoryURINodeType.DIRECTORY:
            return self.directory_path(parsed.to_directory())
        directory, level = parsed.to_layer()
        return self.layer_path(directory, level)

    def write(
        self,
        document: MemoryDocument,
    ) -> MemoryDocument:
        """原子写入已由上层构造的规范 L2；不读取旧记忆或推进版本。"""

        encoded = self._document_codec.encode(document).encode("utf-8")
        self.document_config.validate_body(document.markdown_body)
        self.document_config.validate_relations(
            links=len(document.links),
            backlinks=len(document.backlinks),
        )
        self.document_config.validate_encoded(encoded)
        self.initialize()
        path = self._existing_document_path(document.address)
        self._ensure_directory(path.parent)
        self._atomic_write(path, encoded)
        return document

    def write_layers(
        self,
        directory: MemoryDirectory,
        *,
        abstract: str,
        overview: str,
    ) -> tuple[Path, Path]:
        """在已有目录中先写 L1、再写由它派生的 L0。"""

        if not abstract.strip() or not overview.strip():
            raise ValueError("memory semantic layers must be non-empty")
        directory_path = self._existing_directory_path(directory)
        if not self.directory_exists(directory):
            raise MemoryTreeIntegrityError("memory path is not a safe directory")
        overview_path = directory_path / MemoryLevel.OVERVIEW.sidecar_filename
        abstract_path = directory_path / MemoryLevel.ABSTRACT.sidecar_filename
        self._atomic_write(overview_path, overview.encode("utf-8"))
        self._atomic_write(abstract_path, abstract.encode("utf-8"))
        return abstract_path, overview_path

    def read(self, address: MemoryAddress) -> MemoryDocument:
        """读取并完整验证一个结构化 L2 文档。"""

        return self._read_document(address)

    def read_layer(self, directory: MemoryDirectory, level: MemoryLevel) -> str:
        """读取目录 L0 或 L1 的 UTF-8 Markdown 原文。"""

        return self._read_utf8(
            self._existing_layer_path(directory, level),
            label="memory semantic layer",
        )

    def read_layer_bounded(
        self,
        directory: MemoryDirectory,
        level: MemoryLevel,
        *,
        max_bytes: int,
    ) -> str:
        """在读取前校验字节上限，避免加载损坏或异常膨胀的派生层。"""

        if max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        return self._read_utf8(
            self._existing_layer_path(directory, level),
            label="memory semantic layer",
            max_bytes=max_bytes,
        )

    def exists(self, address: MemoryAddress) -> bool:
        return self._files.file_exists(self._existing_document_path(address))

    def layer_exists(self, directory: MemoryDirectory, level: MemoryLevel) -> bool:
        return self._files.file_exists(self._existing_layer_path(directory, level))

    def directory_exists(self, directory: MemoryDirectory) -> bool:
        return self._files.directory_exists(self._existing_directory_path(directory))

    def delete(self, address: MemoryAddress) -> bool:
        """删除一个记忆文件，并只清理其下方已经为空的动态目录。"""

        path = self._existing_document_path(address)
        deleted = self._files.unlink(path)
        if not deleted:
            return False
        self._prune_dynamic_directories(address, path.parent)
        return True

    def delete_layers(self, directory: MemoryDirectory) -> bool:
        """删除目录的派生 L0/L1，不触碰任何 L2。"""

        changed = False
        if not self.directory_exists(directory):
            return False
        directory_path = self._existing_directory_path(directory)
        for level in (MemoryLevel.ABSTRACT, MemoryLevel.OVERVIEW):
            path = directory_path / level.sidecar_filename
            changed = self._files.unlink(path) or changed
        return changed

    def direct_addresses(
        self,
        directory: MemoryDirectory,
        *,
        limit: int = 1_000,
    ) -> tuple[MemoryAddress, ...]:
        """有界枚举目录中直接存在的 L2 地址，不递归进入子目录。"""

        maximum = self._directory_limit(limit)
        path = self._existing_directory_path(directory)
        if not self.directory_exists(directory):
            return ()
        parts = directory.parts
        addresses: tuple[MemoryAddress, ...]
        if not parts:
            self._require_known_root_entries()
            addresses = (MemoryAddress.profile(),) if self.exists(MemoryAddress.profile()) else ()
        elif parts == ("preferences",):
            addresses = tuple(MemoryAddress.preference(name) for name in self._markdown_names(path))
        elif parts[0] == "entities" and len(parts) == 2:
            addresses = tuple(MemoryAddress.entity(parts[1], name) for name in self._markdown_names(path))
        elif parts == ("tools",):
            addresses = tuple(MemoryAddress.tool(name) for name in self._markdown_names(path))
        else:
            if any(child.is_file() for child in self._content_children(path)):
                raise MemoryTreeIntegrityError("memory branch directory cannot contain L2 files")
            addresses = ()
        if len(addresses) > maximum:
            raise MemoryTreeIntegrityError("memory directory exceeded its direct L2 bound")
        return addresses

    def child_directories(
        self,
        directory: MemoryDirectory,
        *,
        limit: int = 1_000,
    ) -> tuple[MemoryDirectory, ...]:
        """有界枚举目录的直接子目录，不读取更深层内容。"""

        maximum = self._directory_limit(limit)
        path = self._existing_directory_path(directory)
        if not self.directory_exists(directory):
            return ()
        parts = directory.parts
        if not parts:
            children = tuple(MemoryDirectory((name,)) for name in self._STATIC_DIRECTORIES)
        elif parts == ("entities",):
            children = tuple(MemoryDirectory.entities(child.name) for child in self._directories(path))
        else:
            if any(child.is_dir() for child in self._content_children(path)):
                raise MemoryTreeIntegrityError("memory leaf directory cannot contain subdirectories")
            children = ()
        existing = tuple(child for child in children if self.directory_exists(child))
        if len(existing) > maximum:
            raise MemoryTreeIntegrityError("memory directory exceeded its child directory bound")
        return existing

    def list_addresses(
        self,
        kind: MemoryKind | None = None,
        *,
        limit: int = 256,
        after: MemoryAddress | None = None,
    ) -> tuple[MemoryAddress, ...]:
        """按固定类型顺序和路径字典序，从可选游标之后有界枚举。"""

        if limit <= 0 or limit > 10_000:
            raise ValueError("memory tree list limit must be between 1 and 10000")
        selected_kind = None if kind is None else MemoryKind(kind)
        if after is not None and selected_kind is not None and after.kind is not selected_kind:
            raise ValueError("memory tree list cursor kind does not match the selected kind")
        after_key = None if after is None else self._address_order_key(after)
        maximum = limit
        if not self.directory_exists(MemoryDirectory.root()):
            return ()
        kinds = (selected_kind,) if selected_kind is not None else tuple(MemoryKind)
        result: list[MemoryAddress] = []
        for selected in kinds:
            for address in self._iter_kind(selected):
                if after_key is not None and self._address_order_key(address) <= after_key:
                    continue
                result.append(address)
                if len(result) >= maximum:
                    return tuple(result)
        return tuple(result)

    @classmethod
    def _address_order_key(cls, address: MemoryAddress) -> tuple[int, str]:
        return tuple(MemoryKind).index(address.kind), cls._relative_path(address).as_posix()

    @staticmethod
    def _relative_path(address: MemoryAddress) -> Path:
        if address.kind is MemoryKind.PROFILE:
            return Path("profile.md")
        if address.kind is MemoryKind.PREFERENCE:
            return Path("preferences", f"{address.identity_name}.md")
        if address.kind is MemoryKind.ENTITY:
            return Path(
                "entities",
                address.identity_category,
                f"{address.identity_name}.md",
            )
        return Path("tools", f"{address.identity_name}.md")

    def _iter_kind(self, kind: MemoryKind) -> Iterator[MemoryAddress]:
        if kind is MemoryKind.PROFILE:
            if self.exists(MemoryAddress.profile()):
                yield MemoryAddress.profile()
            return
        if kind is MemoryKind.PREFERENCE:
            yield from (MemoryAddress.preference(name) for name in self._markdown_names(self.root / "preferences"))
            return
        if kind is MemoryKind.ENTITY:
            for category_path in self._directories(self.root / "entities"):
                for name in self._markdown_names(category_path):
                    yield MemoryAddress.entity(category_path.name, name)
            return
        yield from (MemoryAddress.tool(name) for name in self._markdown_names(self.root / "tools"))

    def _require_known_root_entries(self) -> None:
        for child in self._content_children(self.root):
            if (child.name in ROOT_FILES and child.is_file()) or (child.name in ROOT_DIRECTORIES and child.is_dir()):
                continue
            raise MemoryTreeIntegrityError(
                f"habitus tree root contains an unsupported entry {child.name!r}; "
                "old tree layouts are not supported, start from an empty habitus tree"
            )

    def _markdown_names(self, directory: Path) -> tuple[str, ...]:
        return self._files.markdown_names(directory)

    def _directories(self, directory: Path) -> tuple[Path, ...]:
        return self._files.directories(directory)

    def _content_children(self, directory: Path) -> tuple[DurableDirectoryEntry, ...]:
        return self._files.content_children(directory)

    def _ensure_directory(self, directory: Path) -> None:
        self._files.ensure_directory(directory, check_capacity=False)

    def _require_inside_root(self, path: Path) -> None:
        self._files.require_inside(path)

    def _read_utf8(self, path: Path, *, label: str, max_bytes: int | None = None) -> str:
        maximum = self.document_config.max_encoded_bytes if max_bytes is None else max_bytes
        return self._files.read_utf8(path, label=label, max_bytes=maximum)

    def _read_document(
        self,
        address: MemoryAddress,
    ) -> MemoryDocument:
        raw = self._read_utf8(
            self._existing_document_path(address),
            label="memory document",
            max_bytes=self.document_config.max_encoded_bytes,
        )
        try:
            document = self._document_codec.decode(raw, expected_address=address)
            self.document_config.validate_body(document.markdown_body)
            self.document_config.validate_relations(
                links=len(document.links),
                backlinks=len(document.backlinks),
            )
            return document
        except (MemoryDocumentIntegrityError, MemoryDocumentLimitError) as exc:
            raise MemoryTreeIntegrityError("memory L2 document failed integrity validation") from exc

    def _atomic_write(self, path: Path, payload: bytes) -> None:
        self._files.replace(path, payload)

    def _prune_dynamic_directories(self, address: MemoryAddress, parent: Path) -> None:
        if address.kind is not MemoryKind.ENTITY:
            return
        stop = self.root / "entities"
        current = parent
        while current != stop:
            if self._content_children(current):
                break
            for filename in (
                MemoryLevel.ABSTRACT.sidecar_filename,
                MemoryLevel.OVERVIEW.sidecar_filename,
            ):
                self._files.unlink(current / filename)
            try:
                deleted = self._files.rmdir(current)
            except OSError:
                break
            if not deleted:
                break
            current = current.parent

    @staticmethod
    def _directory_limit(limit: int) -> int:
        if limit <= 0 or limit > 10_000:
            raise ValueError("memory directory limit must be between 1 and 10000")
        return limit


__all__ = ["MemoryTree", "MemoryTreeIntegrityError"]
