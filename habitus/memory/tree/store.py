"""严格限定于已确认目录结构的 Markdown 记忆树。"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path
from typing import Protocol, TypeVar, cast

from habitus.foundation.ids import canonical_path_identity
from habitus.infrastructure.store.filesystem import (
    DurableDirectoryEntry,
    DurablePathIntegrityError,
    atomic_replace_bytes,
    atomic_temporary_destination,
    durable_rmdir,
    durable_unlink,
    ensure_real_directory,
    list_real_directory,
    read_regular_bytes,
    real_directory_exists,
    regular_file_exists,
)
from habitus.memory.document import (
    MemoryDocument,
    MemoryDocumentCodec,
    MemoryDocumentConfig,
    MemoryDocumentIntegrityError,
    MemoryDocumentLimitError,
)
from habitus.memory.model import MemoryAddress, MemoryDirectory, MemoryKind, MemoryLevel
from habitus.memory.tree.config import MemoryTreeConfig
from habitus.memory.uri import MemoryURI, MemoryURINodeType


class MemoryTreeIntegrityError(ValueError):
    """记忆树包含路径逃逸、符号链接或不符合目录结构的条目。"""


class MemoryTreeConsistencyError(RuntimeError):
    """公开读取无法跨越正在发布的多文档事务形成一致视图。"""


_VisibleValue = TypeVar("_VisibleValue")


class _VisibilityJournal(Protocol):
    root: Path

    def visibility_generation(self) -> int: ...

    def pending(self) -> tuple[object, ...]: ...


class MemoryTree:
    """安全持久化结构化 L2 文档和可重建的目录语义层。"""

    _STATIC_DIRECTORIES = ("preferences", "entities", "tools", "events", "intentions")

    def __init__(
        self,
        root: str | Path,
        *,
        document_codec: MemoryDocumentCodec | None = None,
        document_config: MemoryDocumentConfig | None = None,
        tree_config: MemoryTreeConfig | None = None,
    ) -> None:
        requested = Path(root).expanduser().absolute()
        if requested.is_symlink():
            raise MemoryTreeIntegrityError("memory tree root cannot be a symbolic link")
        self.root = requested.resolve(strict=False)
        if document_codec is None:
            from habitus.memory.schema import MemorySchemaRegistry

            document_codec = MemoryDocumentCodec(MemorySchemaRegistry.load_default())
        if not isinstance(document_codec, MemoryDocumentCodec):
            raise TypeError("document_codec must be a MemoryDocumentCodec")
        self._document_codec = document_codec
        if document_config is not None and not isinstance(document_config, MemoryDocumentConfig):
            raise TypeError("document_config must be MemoryDocumentConfig")
        if tree_config is not None and not isinstance(tree_config, MemoryTreeConfig):
            raise TypeError("tree_config must be MemoryTreeConfig")
        self.document_config = document_config or MemoryDocumentConfig()
        self.tree_config = tree_config or MemoryTreeConfig()
        self._visibility_journal: _VisibilityJournal | None = None

    @property
    def document_codec(self) -> MemoryDocumentCodec:
        """返回记忆树实际用于规范 L2 读写的同一编解码器。"""

        return self._document_codec

    @property
    def visibility_journal(self) -> _VisibilityJournal | None:
        """返回由事务层绑定的可见性日志，不赋予物理存储提交职责。"""

        if self._visibility_journal is None:
            conventional_root = self.root.parent / "workflow" / "transactions"
            from habitus.memory.editor.transaction_log import MemoryTransactionJournal

            self._visibility_journal = MemoryTransactionJournal(
                conventional_root,
                self.document_codec,
            )
        return self._visibility_journal

    def bind_visibility_journal(self, journal: object) -> None:
        """让同一 ``MemoryTree`` 上后来创建的公开 Reader 共享提交边界。"""

        root = getattr(journal, "root", None)
        if (
            not isinstance(root, Path)
            or not callable(getattr(journal, "visibility_generation", None))
            or not callable(getattr(journal, "pending", None))
        ):
            raise TypeError("journal must implement the memory visibility journal contract")
        current_root = getattr(self._visibility_journal, "root", None)
        if current_root is not None and current_root != root:
            raise ValueError("memory tree is already bound to another visibility journal")
        self._visibility_journal = cast(_VisibilityJournal, journal)

    def initialize(self) -> Path:
        """只创建静态目录；没有真实内容时不创建空的 profile.md。"""

        self._ensure_directory(self.root)
        for name in self._STATIC_DIRECTORIES:
            self._ensure_directory(self.root / name)
        return self.root

    def path_for(self, address: MemoryAddress) -> Path:
        """把经过验证的语义地址确定性映射为唯一 Markdown 路径。"""

        if not isinstance(address, MemoryAddress):
            raise TypeError("address must be a MemoryAddress")
        relative = self._relative_path(address)
        candidate = self.root / relative
        self._require_inside_root(candidate)
        return candidate

    def _existing_document_path(self, address: MemoryAddress) -> Path:
        if not isinstance(address, MemoryAddress):
            raise TypeError("address must be a MemoryAddress")
        return self._existing_relative_path(self._relative_path(address), leaf_is_file=True)

    def _existing_directory_path(self, directory: MemoryDirectory) -> Path:
        if not isinstance(directory, MemoryDirectory):
            raise TypeError("directory must be a MemoryDirectory")
        return self._existing_relative_path(Path(*directory.identity_parts), leaf_is_file=False)

    def _existing_layer_path(self, directory: MemoryDirectory, level: MemoryLevel) -> Path:
        normalized = MemoryLevel(level)
        relative = Path(*directory.identity_parts, normalized.sidecar_filename)
        return self._existing_relative_path(relative, leaf_is_file=True)

    def _existing_relative_path(self, relative: Path, *, leaf_is_file: bool) -> Path:
        current = self.root
        parts = relative.parts
        for index, part in enumerate(parts):
            try:
                entries = list_real_directory(
                    current,
                    artifact_root=self.root,
                    max_entries=self.tree_config.max_children_per_directory,
                )
            except DurablePathIntegrityError as exc:
                raise MemoryTreeIntegrityError("memory identity path cannot be resolved safely") from exc
            desired = canonical_path_identity(part, "memory path segment")
            matches: list[DurableDirectoryEntry] = []
            for entry in entries:
                try:
                    identity = canonical_path_identity(entry.name, "memory path entry")
                except ValueError:
                    continue
                if identity == desired:
                    matches.append(entry)
            if len(matches) > 1:
                raise MemoryTreeIntegrityError("memory tree contains multiple physical aliases for one identity")
            if not matches:
                return current.joinpath(*parts[index:])
            selected = matches[0]
            is_leaf = index == len(parts) - 1
            if (is_leaf and leaf_is_file and not selected.is_file()) or (
                (not is_leaf or not leaf_is_file) and not selected.is_dir()
            ):
                raise MemoryTreeIntegrityError("memory identity path has an incompatible physical entry")
            current = current / selected.name
        return current

    def directory_path(self, directory: MemoryDirectory) -> Path:
        """把受控目录地址映射到记忆树内的真实目录。"""

        if not isinstance(directory, MemoryDirectory):
            raise TypeError("directory must be a MemoryDirectory")
        candidate = self.root.joinpath(*directory.identity_parts)
        self._require_inside_root(candidate)
        return candidate

    def layer_path(self, directory: MemoryDirectory, level: MemoryLevel) -> Path:
        """返回目录 L0 或 L1 侧车文件的确定性路径。"""

        normalized = MemoryLevel(level)
        return self.directory_path(directory) / normalized.sidecar_filename

    def path_for_uri(self, uri: MemoryURI | str) -> Path:
        """把合法 ``memory://`` 节点确定性映射为树内物理路径。"""

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

        if not isinstance(document, MemoryDocument):
            raise TypeError("document must be a MemoryDocument")
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

        if not isinstance(abstract, str) or not isinstance(overview, str):
            raise TypeError("memory semantic layers must be strings")
        if not abstract.strip() or not overview.strip():
            raise ValueError("memory semantic layers must be non-empty")
        directory_path = self._existing_directory_path(directory)
        if not self._directory_exists_physical(directory):
            raise MemoryTreeIntegrityError("memory path is not a safe directory")
        overview_path = directory_path / MemoryLevel.OVERVIEW.sidecar_filename
        abstract_path = directory_path / MemoryLevel.ABSTRACT.sidecar_filename
        self._atomic_write(overview_path, overview.encode("utf-8"))
        self._atomic_write(abstract_path, abstract.encode("utf-8"))
        return abstract_path, overview_path

    def read(self, address: MemoryAddress) -> MemoryDocument:
        """读取并完整验证一个结构化 L2 文档。"""

        return self._read_visible(lambda: self._read_physical(address))

    def _read_physical(self, address: MemoryAddress) -> MemoryDocument:
        """仅供持有事务 fencing 租约的提交和恢复流程读取物理状态。"""

        return self._read_document(address)

    def read_layer(self, directory: MemoryDirectory, level: MemoryLevel) -> str:
        """读取目录 L0 或 L1 的 UTF-8 Markdown 原文。"""

        return self._read_visible(
            lambda: self._read_utf8(
                self._existing_layer_path(directory, level),
                label="memory semantic layer",
            )
        )

    def read_layer_bounded(
        self,
        directory: MemoryDirectory,
        level: MemoryLevel,
        *,
        max_bytes: int,
    ) -> str:
        """在读取前校验字节上限，避免加载损坏或异常膨胀的派生层。"""

        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        return self._read_visible(
            lambda: self._read_utf8(
                self._existing_layer_path(directory, level),
                label="memory semantic layer",
                max_bytes=max_bytes,
            )
        )

    def exists(self, address: MemoryAddress) -> bool:
        return self._read_visible(lambda: self._exists_physical(address))

    def _exists_physical(self, address: MemoryAddress) -> bool:
        try:
            return regular_file_exists(
                self._existing_document_path(address),
                artifact_root=self.root,
            )
        except DurablePathIntegrityError as exc:
            raise MemoryTreeIntegrityError("memory file cannot be inspected safely") from exc

    def layer_exists(self, directory: MemoryDirectory, level: MemoryLevel) -> bool:
        return self._read_visible(lambda: self._layer_exists_physical(directory, level))

    def _layer_exists_physical(self, directory: MemoryDirectory, level: MemoryLevel) -> bool:
        try:
            return regular_file_exists(
                self._existing_layer_path(directory, level),
                artifact_root=self.root,
            )
        except DurablePathIntegrityError as exc:
            raise MemoryTreeIntegrityError("memory semantic layer cannot be inspected safely") from exc

    def directory_exists(self, directory: MemoryDirectory) -> bool:
        return self._read_visible(lambda: self._directory_exists_physical(directory))

    def _directory_exists_physical(self, directory: MemoryDirectory) -> bool:
        try:
            return real_directory_exists(
                self._existing_directory_path(directory),
                artifact_root=self.root,
            )
        except DurablePathIntegrityError as exc:
            raise MemoryTreeIntegrityError("memory directory cannot be inspected safely") from exc

    def delete(self, address: MemoryAddress) -> bool:
        """删除一个记忆文件，并只清理其下方已经为空的动态目录。"""

        path = self._existing_document_path(address)
        try:
            deleted = durable_unlink(path, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise MemoryTreeIntegrityError("memory file cannot be deleted safely") from exc
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
            try:
                changed = durable_unlink(path, artifact_root=self.root) or changed
            except DurablePathIntegrityError as exc:
                raise MemoryTreeIntegrityError("memory semantic layer cannot be deleted safely") from exc
        return changed

    def direct_addresses(
        self,
        directory: MemoryDirectory,
        *,
        limit: int = 1_000,
    ) -> tuple[MemoryAddress, ...]:
        """有界枚举目录中直接存在的 L2 地址，不递归进入子目录。"""

        return self._read_visible(lambda: self._direct_addresses_physical(directory, limit=limit))

    def _direct_addresses_physical(
        self,
        directory: MemoryDirectory,
        *,
        limit: int,
    ) -> tuple[MemoryAddress, ...]:

        maximum = self._directory_limit(limit)
        path = self._existing_directory_path(directory)
        if not self._directory_exists_physical(directory):
            return ()
        parts = directory.parts
        addresses: tuple[MemoryAddress, ...]
        if not parts:
            children = self._content_children(path)
            allowed_directories = set(self._STATIC_DIRECTORIES)
            profile_present = False
            for child in children:
                if child.name == "profile.md" and child.is_file():
                    profile_present = True
                    continue
                if child.name in allowed_directories and child.is_dir():
                    continue
                raise MemoryTreeIntegrityError("memory root contains an unsupported entry")
            addresses = (MemoryAddress.profile(),) if profile_present else ()
        elif parts == ("preferences",):
            addresses = tuple(MemoryAddress.preference(name) for name in self._markdown_names(path))
        elif parts[0] == "entities" and len(parts) == 2:
            addresses = tuple(MemoryAddress.entity(parts[1], name) for name in self._markdown_names(path))
        elif parts == ("tools",):
            addresses = tuple(MemoryAddress.tool(name) for name in self._markdown_names(path))
        elif parts[0] == "events" and len(parts) == 4:
            event_date = date(int(parts[1]), int(parts[2]), int(parts[3]))
            addresses = tuple(MemoryAddress.event(event_date, name) for name in self._markdown_names(path))
        elif parts == ("intentions",):
            addresses = tuple(MemoryAddress.intention(name) for name in self._markdown_names(path))
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

        return self._read_visible(lambda: self._child_directories_physical(directory, limit=limit))

    def _child_directories_physical(
        self,
        directory: MemoryDirectory,
        *,
        limit: int,
    ) -> tuple[MemoryDirectory, ...]:

        maximum = self._directory_limit(limit)
        path = self._existing_directory_path(directory)
        if not self._directory_exists_physical(directory):
            return ()
        parts = directory.parts
        if not parts:
            children = tuple(MemoryDirectory((name,)) for name in self._STATIC_DIRECTORIES)
        elif parts == ("entities",):
            children = tuple(MemoryDirectory.entities(child.name) for child in self._directories(path))
        elif parts and parts[0] == "events" and len(parts) < 4:
            children = tuple(MemoryDirectory((*parts, child.name)) for child in self._directories(path))
        else:
            if any(child.is_dir() for child in self._content_children(path)):
                raise MemoryTreeIntegrityError("memory leaf directory cannot contain subdirectories")
            children = ()
        existing = tuple(child for child in children if self._directory_exists_physical(child))
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

        return self._read_visible(
            lambda: self._list_addresses_physical(kind, limit=limit, after=after)
        )

    def _list_addresses_physical(
        self,
        kind: MemoryKind | None,
        *,
        limit: int,
        after: MemoryAddress | None,
    ) -> tuple[MemoryAddress, ...]:

        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0 or limit > 10_000:
            raise ValueError("memory tree list limit must be between 1 and 10000")
        if after is not None and not isinstance(after, MemoryAddress):
            raise TypeError("memory tree list cursor must be MemoryAddress or None")
        selected_kind = None if kind is None else MemoryKind(kind)
        if after is not None and selected_kind is not None and after.kind is not selected_kind:
            raise ValueError("memory tree list cursor kind does not match the selected kind")
        after_key = None if after is None else self._address_order_key(after)
        maximum = limit
        if not self._directory_exists_physical(MemoryDirectory.root()):
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
        if address.kind is MemoryKind.TOOL:
            return Path("tools", f"{address.identity_name}.md")
        if address.kind is MemoryKind.EVENT:
            assert address.event_date is not None
            return Path(
                "events",
                f"{address.event_date.year:04d}",
                f"{address.event_date.month:02d}",
                f"{address.event_date.day:02d}",
                f"{address.identity_name}.md",
            )
        return Path("intentions", f"{address.identity_name}.md")

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
        if kind is MemoryKind.TOOL:
            yield from (MemoryAddress.tool(name) for name in self._markdown_names(self.root / "tools"))
            return
        if kind is MemoryKind.EVENT:
            yield from self._iter_events()
            return
        yield from (MemoryAddress.intention(name) for name in self._markdown_names(self.root / "intentions"))

    def _iter_events(self) -> Iterator[MemoryAddress]:
        for year_path in self._directories(self.root / "events"):
            if len(year_path.name) != 4 or not year_path.name.isdigit():
                raise MemoryTreeIntegrityError("event year directory must use YYYY")
            for month_path in self._directories(year_path):
                if len(month_path.name) != 2 or not month_path.name.isdigit():
                    raise MemoryTreeIntegrityError("event month directory must use MM")
                for day_path in self._directories(month_path):
                    if len(day_path.name) != 2 or not day_path.name.isdigit():
                        raise MemoryTreeIntegrityError("event day directory must use DD")
                    try:
                        event_date = date(int(year_path.name), int(month_path.name), int(day_path.name))
                    except ValueError as exc:
                        raise MemoryTreeIntegrityError("event directory contains an invalid calendar date") from exc
                    for name in self._markdown_names(day_path):
                        yield MemoryAddress.event(event_date, name)

    def _markdown_names(self, directory: Path) -> tuple[str, ...]:
        names: list[str] = []
        for child in self._content_children(directory):
            child_path = Path(child.name)
            if not child.is_file() or child_path.suffix != ".md" or not child_path.stem:
                raise MemoryTreeIntegrityError("memory leaf directory may contain only Markdown files")
            names.append(child_path.stem)
        return tuple(names)

    def _directories(self, directory: Path) -> tuple[Path, ...]:
        children = self._content_children(directory)
        if any(not child.is_dir() for child in children):
            raise MemoryTreeIntegrityError("memory branch may contain only directories")
        return tuple(directory / child.name for child in children)

    def _content_children(self, directory: Path) -> tuple[DurableDirectoryEntry, ...]:
        content: list[DurableDirectoryEntry] = []
        semantic_names = {
            MemoryLevel.ABSTRACT.sidecar_filename,
            MemoryLevel.OVERVIEW.sidecar_filename,
        }
        for child in self._children(directory):
            if child.name in semantic_names:
                if not child.is_file():
                    raise MemoryTreeIntegrityError("memory semantic layer is not a regular file")
                continue
            if child.name.startswith("."):
                if child.is_file() and atomic_temporary_destination(child.name) is not None:
                    continue
                raise MemoryTreeIntegrityError("memory directory contains an unsupported hidden entry")
            content.append(child)
        return tuple(content)

    def _children(self, directory: Path) -> tuple[DurableDirectoryEntry, ...]:
        try:
            return list_real_directory(
                directory,
                artifact_root=self.root,
                max_entries=self.tree_config.max_children_per_directory,
            )
        except DurablePathIntegrityError as exc:
            raise MemoryTreeIntegrityError("memory directory cannot be enumerated safely") from exc

    def _ensure_directory(self, directory: Path) -> None:
        self._require_inside_root(directory)
        try:
            ensure_real_directory(directory, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise MemoryTreeIntegrityError("memory directory cannot be created safely") from exc

    def _require_inside_root(self, path: Path) -> None:
        candidate = path.resolve(strict=False)
        if candidate != self.root and self.root not in candidate.parents:
            raise MemoryTreeIntegrityError("memory path cannot be used safely outside its tree root")

    def _read_utf8(
        self,
        path: Path,
        *,
        label: str,
        max_bytes: int | None = None,
    ) -> str:
        maximum = self.document_config.max_encoded_bytes if max_bytes is None else max_bytes
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
            raise ValueError("max_bytes must be a positive integer")
        try:
            payload = read_regular_bytes(
                path,
                artifact_root=self.root,
                max_bytes=maximum,
            )
            return payload.decode("utf-8")
        except DurablePathIntegrityError as exc:
            if "read bound" in str(exc):
                raise MemoryTreeIntegrityError(f"{label} exceeds its configured read bound") from exc
            raise MemoryTreeIntegrityError(f"{label} cannot be read safely") from exc
        except UnicodeDecodeError as exc:
            raise MemoryTreeIntegrityError(f"{label} is not valid UTF-8") from exc

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

    def _read_visible(self, load: Callable[[], _VisibleValue]) -> _VisibleValue:
        journal = self.visibility_journal
        if journal is None:
            return load()
        for _attempt in range(16):
            generation_before = journal.visibility_generation()
            pending_before = journal.pending()
            if pending_before:
                raise MemoryTreeConsistencyError("memory tree has a prepared multi-document transaction")
            value = load()
            pending_after = journal.pending()
            generation_after = journal.visibility_generation()
            if generation_before == generation_after and pending_before == pending_after and not pending_after:
                return value
        raise MemoryTreeConsistencyError("memory transactions changed continuously during a public tree read")

    def _atomic_write(self, path: Path, payload: bytes) -> None:
        try:
            atomic_replace_bytes(path, payload, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise MemoryTreeIntegrityError("memory file cannot be written safely") from exc

    def _prune_dynamic_directories(self, address: MemoryAddress, parent: Path) -> None:
        stop = {
            MemoryKind.ENTITY: self.root / "entities",
            MemoryKind.EVENT: self.root / "events",
        }.get(address.kind)
        if stop is None:
            return
        current = parent
        while current != stop:
            if self._content_children(current):
                break
            for filename in (
                MemoryLevel.ABSTRACT.sidecar_filename,
                MemoryLevel.OVERVIEW.sidecar_filename,
            ):
                sidecar = current / filename
                try:
                    durable_unlink(sidecar, artifact_root=self.root)
                except DurablePathIntegrityError as exc:
                    raise MemoryTreeIntegrityError("memory semantic layer cannot be pruned safely") from exc
            try:
                deleted = durable_rmdir(current, artifact_root=self.root)
            except DurablePathIntegrityError as exc:
                raise MemoryTreeIntegrityError("memory directory cannot be pruned safely") from exc
            except OSError:
                break
            if not deleted:
                break
            current = current.parent

    @staticmethod
    def _directory_limit(limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0 or limit > 10_000:
            raise ValueError("memory directory limit must be between 1 and 10000")
        return limit


__all__ = ["MemoryTree", "MemoryTreeConsistencyError", "MemoryTreeIntegrityError"]
