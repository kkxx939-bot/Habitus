"""严格限定于已确认目录结构的行为语义 Markdown 树。"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

from behavior.document import (
    BehaviorDocument,
    BehaviorDocumentCodec,
    BehaviorDocumentConfig,
    BehaviorDocumentIntegrityError,
    BehaviorDocumentLimitError,
)
from behavior.model import BehaviorAddress, BehaviorDirectory, BehaviorKind, BehaviorLevel, is_ascii_digits
from behavior.tree.config import BehaviorTreeConfig
from behavior.uri import BehaviorURI, BehaviorURINodeType
from foundation.ids import canonical_path_identity
from infrastructure.store.filesystem import (
    DurableDirectoryEntry,
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_replace_bytes,
    atomic_temporary_destination,
    ensure_real_directory,
    list_real_directory,
    read_regular_bytes,
    real_directory_exists,
    regular_file_exists,
)


class BehaviorTreeIntegrityError(ValueError):
    """行为树包含路径逃逸、符号链接或不符合已确认结构的条目。"""


class BehaviorTreeConflictError(RuntimeError):
    """只允许创建的 L2 地址已经绑定到不同内容。"""


class BehaviorTree:
    """安全持久化规范 L2 文档和可重建目录语义层。"""

    _STATIC_DIRECTORIES = (
        ("behaviors",),
        ("behaviors", "events"),
        ("behaviors", "outcomes"),
        ("episodes",),
    )

    def __init__(
        self,
        root: str | Path,
        *,
        document_codec: BehaviorDocumentCodec | None = None,
        document_config: BehaviorDocumentConfig | None = None,
        tree_config: BehaviorTreeConfig | None = None,
    ) -> None:
        requested = Path(root).expanduser().absolute()
        if requested.is_symlink():
            raise BehaviorTreeIntegrityError("behavior tree root cannot be a symbolic link")
        self.root = requested.resolve(strict=False)
        if document_codec is None:
            from behavior.schema import BehaviorSchemaRegistry

            document_codec = BehaviorDocumentCodec(BehaviorSchemaRegistry.load_default())
        if not isinstance(document_codec, BehaviorDocumentCodec):
            raise TypeError("document_codec must be a BehaviorDocumentCodec")
        if document_config is not None and not isinstance(document_config, BehaviorDocumentConfig):
            raise TypeError("document_config must be BehaviorDocumentConfig")
        if tree_config is not None and not isinstance(tree_config, BehaviorTreeConfig):
            raise TypeError("tree_config must be BehaviorTreeConfig")
        self._document_codec = document_codec
        self.document_config = document_config or BehaviorDocumentConfig()
        self.tree_config = tree_config or BehaviorTreeConfig()

    @property
    def document_codec(self) -> BehaviorDocumentCodec:
        return self._document_codec

    def initialize(self) -> Path:
        self._ensure_directory(self.root)
        for parts in self._STATIC_DIRECTORIES:
            self._ensure_directory(self.root.joinpath(*parts))
        return self.root

    def path_for(self, address: BehaviorAddress) -> Path:
        if not isinstance(address, BehaviorAddress):
            raise TypeError("address must be a BehaviorAddress")
        candidate = self.root / self._relative_path(address)
        self._require_inside_root(candidate)
        return candidate

    def directory_path(self, directory: BehaviorDirectory) -> Path:
        if not isinstance(directory, BehaviorDirectory):
            raise TypeError("directory must be a BehaviorDirectory")
        candidate = self.root.joinpath(*directory.identity_parts)
        self._require_inside_root(candidate)
        return candidate

    def layer_path(self, directory: BehaviorDirectory, level: BehaviorLevel) -> Path:
        normalized = BehaviorLevel(level)
        return self.directory_path(directory) / normalized.sidecar_filename

    def path_for_uri(self, uri: BehaviorURI | str) -> Path:
        parsed = BehaviorURI.parse(uri)
        if parsed.node_type is BehaviorURINodeType.DOCUMENT:
            return self.path_for(parsed.to_address())
        if parsed.node_type is BehaviorURINodeType.DIRECTORY:
            return self.directory_path(parsed.to_directory())
        directory, level = parsed.to_layer()
        return self.layer_path(directory, level)

    def write(self, document: BehaviorDocument) -> BehaviorDocument:
        """受信任低层原语；普通领域调用方必须经 Writer/事务写入 L2。"""

        if not isinstance(document, BehaviorDocument):
            raise TypeError("document must be a BehaviorDocument")
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
        self._require_child_capacity(path.parent, (path.name,))
        self._atomic_write(path, encoded)
        return document

    def create(self, document: BehaviorDocument) -> BehaviorDocument:
        """原子创建 add-only L2；目标已绑定到不同内容时拒绝覆盖。"""

        if not isinstance(document, BehaviorDocument):
            raise TypeError("document must be a BehaviorDocument")
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
        self._require_child_capacity(path.parent, (path.name,))
        try:
            atomic_create_bytes(path, encoded, artifact_root=self.root)
        except ImmutableArtifactConflictError as exc:
            raise BehaviorTreeConflictError("behavior L2 address already contains different content") from exc
        return document

    def read(self, address: BehaviorAddress) -> BehaviorDocument:
        if not isinstance(address, BehaviorAddress):
            raise TypeError("address must be a BehaviorAddress")
        raw = self._read_utf8(
            self._existing_document_path(address),
            label="behavior document",
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
        except (BehaviorDocumentIntegrityError, BehaviorDocumentLimitError) as exc:
            raise BehaviorTreeIntegrityError("behavior L2 document failed integrity validation") from exc

    def exists(self, address: BehaviorAddress) -> bool:
        if not isinstance(address, BehaviorAddress):
            raise TypeError("address must be a BehaviorAddress")
        try:
            return regular_file_exists(
                self._existing_document_path(address),
                artifact_root=self.root,
            )
        except DurablePathIntegrityError as exc:
            raise BehaviorTreeIntegrityError("behavior file cannot be inspected safely") from exc

    def write_layers(
        self,
        directory: BehaviorDirectory,
        *,
        abstract: str,
        overview: str,
    ) -> tuple[Path, Path]:
        """写入可重建 L1/L0；失败补齐和有界重试由后续 Behavior 语义刷新 Job 编排。"""

        if not isinstance(abstract, str) or not isinstance(overview, str):
            raise TypeError("behavior semantic layers must be strings")
        if not abstract.strip() or not overview.strip():
            raise ValueError("behavior semantic layers must be non-empty")
        encoded_abstract = abstract.encode("utf-8")
        encoded_overview = overview.encode("utf-8")
        self.document_config.validate_semantic_layer(encoded_abstract)
        self.document_config.validate_semantic_layer(encoded_overview)
        directory_path = self._existing_directory_path(directory)
        if not self.directory_exists(directory):
            raise BehaviorTreeIntegrityError("behavior path is not a safe directory")
        overview_path = directory_path / BehaviorLevel.OVERVIEW.sidecar_filename
        abstract_path = directory_path / BehaviorLevel.ABSTRACT.sidecar_filename
        self._require_child_capacity(directory_path, (overview_path.name, abstract_path.name))
        self._atomic_write(overview_path, encoded_overview)
        self._atomic_write(abstract_path, encoded_abstract)
        return abstract_path, overview_path

    def read_layer(self, directory: BehaviorDirectory, level: BehaviorLevel) -> str:
        return self._read_utf8(
            self._existing_layer_path(directory, level),
            label="behavior semantic layer",
        )

    def layer_exists(self, directory: BehaviorDirectory, level: BehaviorLevel) -> bool:
        try:
            return regular_file_exists(
                self._existing_layer_path(directory, level),
                artifact_root=self.root,
            )
        except DurablePathIntegrityError as exc:
            raise BehaviorTreeIntegrityError("behavior semantic layer cannot be inspected safely") from exc

    def directory_exists(self, directory: BehaviorDirectory) -> bool:
        try:
            return real_directory_exists(
                self._existing_directory_path(directory),
                artifact_root=self.root,
            )
        except DurablePathIntegrityError as exc:
            raise BehaviorTreeIntegrityError("behavior directory cannot be inspected safely") from exc

    def list_addresses(
        self,
        kind: BehaviorKind | None = None,
        *,
        limit: int = 256,
        after: BehaviorAddress | None = None,
    ) -> tuple[BehaviorAddress, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("behavior tree list limit must be between 1 and 10000")
        if after is not None and not isinstance(after, BehaviorAddress):
            raise TypeError("behavior tree list cursor must be BehaviorAddress or None")
        selected_kind = None if kind is None else BehaviorKind(kind)
        if after is not None and selected_kind is not None and after.kind is not selected_kind:
            raise ValueError("behavior tree list cursor kind does not match the selected kind")
        if not self.directory_exists(BehaviorDirectory.root()):
            return ()
        after_key = None if after is None else self._address_order_key(after)
        result: list[BehaviorAddress] = []
        kinds = (selected_kind,) if selected_kind is not None else tuple(BehaviorKind)
        for resolved_kind in kinds:
            for address in self._iter_kind(resolved_kind):
                if after_key is not None and self._address_order_key(address) <= after_key:
                    continue
                result.append(address)
                if len(result) == limit:
                    return tuple(result)
        return tuple(result)

    @staticmethod
    def _relative_path(address: BehaviorAddress) -> Path:
        return Path(*BehaviorURI.from_address(address).segments)

    @classmethod
    def _address_order_key(cls, address: BehaviorAddress) -> tuple[int, str]:
        return tuple(BehaviorKind).index(address.kind), cls._relative_path(address).as_posix()

    def _iter_kind(self, kind: BehaviorKind) -> Iterator[BehaviorAddress]:
        root = {
            BehaviorKind.EVENT: self.root / "behaviors" / "events",
            BehaviorKind.OUTCOME: self.root / "behaviors" / "outcomes",
            BehaviorKind.EPISODE: self.root / "episodes",
        }[kind]
        for year_path in self._directories(root):
            if len(year_path.name) != 4 or not is_ascii_digits(year_path.name):
                raise BehaviorTreeIntegrityError("behavior year directory must use YYYY")
            for month_path in self._directories(year_path):
                if len(month_path.name) != 2 or not is_ascii_digits(month_path.name):
                    raise BehaviorTreeIntegrityError("behavior month directory must use MM")
                for day_path in self._directories(month_path):
                    if len(day_path.name) != 2 or not is_ascii_digits(day_path.name):
                        raise BehaviorTreeIntegrityError("behavior day directory must use DD")
                    try:
                        occurred_on = date(
                            int(year_path.name),
                            int(month_path.name),
                            int(day_path.name),
                        )
                    except ValueError as exc:
                        raise BehaviorTreeIntegrityError(
                            "behavior directory contains an invalid calendar date"
                        ) from exc
                    for name in self._markdown_names(day_path):
                        try:
                            yield BehaviorAddress.from_identity(kind, occurred_on, name)
                        except (TypeError, ValueError) as exc:
                            raise BehaviorTreeIntegrityError(
                                "behavior leaf contains an invalid timestamp identity"
                            ) from exc

    def _existing_document_path(self, address: BehaviorAddress) -> Path:
        return self._existing_relative_path(self._relative_path(address), leaf_is_file=True)

    def _existing_directory_path(self, directory: BehaviorDirectory) -> Path:
        return self._existing_relative_path(Path(*directory.identity_parts), leaf_is_file=False)

    def _existing_layer_path(self, directory: BehaviorDirectory, level: BehaviorLevel) -> Path:
        normalized = BehaviorLevel(level)
        relative = Path(*directory.identity_parts, normalized.sidecar_filename)
        return self._existing_relative_path(relative, leaf_is_file=True)

    def _existing_relative_path(self, relative: Path, *, leaf_is_file: bool) -> Path:
        current = self.root
        parts = relative.parts
        for index, part in enumerate(parts):
            entries = self._children(current)
            desired = canonical_path_identity(part, "behavior path segment")
            matches: list[DurableDirectoryEntry] = []
            for entry in entries:
                try:
                    identity = canonical_path_identity(entry.name, "behavior path entry")
                except ValueError:
                    continue
                if identity == desired:
                    matches.append(entry)
            if len(matches) > 1:
                raise BehaviorTreeIntegrityError("behavior tree contains multiple physical aliases for one identity")
            if not matches:
                return current.joinpath(*parts[index:])
            selected = matches[0]
            is_leaf = index == len(parts) - 1
            if (is_leaf and leaf_is_file and not selected.is_file()) or (
                (not is_leaf or not leaf_is_file) and not selected.is_dir()
            ):
                raise BehaviorTreeIntegrityError("behavior identity path has an incompatible physical entry")
            current = current / selected.name
        return current

    def _markdown_names(self, directory: Path) -> tuple[str, ...]:
        names_by_identity: dict[str, str] = {}
        for child in self._content_children(directory):
            child_path = Path(child.name)
            if not child.is_file() or child_path.suffix != ".md" or not child_path.stem:
                raise BehaviorTreeIntegrityError("behavior leaf directory may contain only Markdown files")
            try:
                identity = canonical_path_identity(child_path.stem, "behavior document name")
            except ValueError as exc:
                raise BehaviorTreeIntegrityError("behavior leaf contains an invalid document name") from exc
            if identity in names_by_identity:
                raise BehaviorTreeIntegrityError("behavior leaf contains multiple physical aliases for one identity")
            names_by_identity[identity] = child_path.stem
        return tuple(names_by_identity[identity] for identity in sorted(names_by_identity))

    def _directories(self, directory: Path) -> tuple[Path, ...]:
        children = self._content_children(directory)
        if any(not child.is_dir() for child in children):
            raise BehaviorTreeIntegrityError("behavior branch may contain only directories")
        return tuple(directory / child.name for child in children)

    def _content_children(self, directory: Path) -> tuple[DurableDirectoryEntry, ...]:
        semantic_names = {
            BehaviorLevel.ABSTRACT.sidecar_filename,
            BehaviorLevel.OVERVIEW.sidecar_filename,
        }
        content: list[DurableDirectoryEntry] = []
        for child in self._children(directory):
            if child.name.startswith(".") and child.is_file() and atomic_temporary_destination(child.name) is not None:
                continue
            if child.name in semantic_names:
                if not child.is_file():
                    raise BehaviorTreeIntegrityError("behavior semantic layer is not a regular file")
                continue
            if child.name.startswith("."):
                raise BehaviorTreeIntegrityError("behavior directory contains an unsupported hidden entry")
            content.append(child)
        return tuple(content)

    def _children(self, directory: Path) -> tuple[DurableDirectoryEntry, ...]:
        try:
            entries = list_real_directory(
                directory,
                artifact_root=self.root,
                max_entries=self.tree_config.max_children_per_directory + 1,
            )
        except DurablePathIntegrityError as exc:
            raise BehaviorTreeIntegrityError("behavior directory cannot be enumerated safely") from exc
        persistent_count = sum(
            not (
                entry.name.startswith(".") and entry.is_file() and atomic_temporary_destination(entry.name) is not None
            )
            for entry in entries
        )
        if persistent_count > self.tree_config.max_children_per_directory:
            raise BehaviorTreeIntegrityError("behavior directory exceeds its persistent child capacity")
        return entries

    def _ensure_directory(self, directory: Path) -> None:
        self._require_inside_root(directory)
        if directory != self.root:
            self._ensure_directory(directory.parent)
            relative = directory.relative_to(self.root)
            directory = self._existing_relative_path(relative, leaf_is_file=False)
            try:
                exists = real_directory_exists(directory, artifact_root=self.root)
            except DurablePathIntegrityError as exc:
                raise BehaviorTreeIntegrityError("behavior directory cannot be inspected safely") from exc
            if not exists:
                self._require_child_capacity(directory.parent, (directory.name,))
        try:
            ensure_real_directory(directory, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise BehaviorTreeIntegrityError("behavior directory cannot be created safely") from exc

    def _require_child_capacity(self, directory: Path, child_names: tuple[str, ...]) -> None:
        entries = tuple(
            entry
            for entry in self._children(directory)
            if not (
                entry.name.startswith(".") and entry.is_file() and atomic_temporary_destination(entry.name) is not None
            )
        )
        existing_identities: set[str] = set()
        for entry in entries:
            try:
                existing_identities.add(canonical_path_identity(entry.name, "behavior path entry"))
            except ValueError:
                continue
        requested_identities = {canonical_path_identity(name, "behavior path child") for name in child_names}
        additional = len(requested_identities - existing_identities)
        if len(entries) + additional > self.tree_config.max_children_per_directory:
            raise BehaviorTreeIntegrityError("behavior directory has no remaining child capacity")

    def _require_inside_root(self, path: Path) -> None:
        candidate = path.resolve(strict=False)
        if candidate != self.root and self.root not in candidate.parents:
            raise BehaviorTreeIntegrityError("behavior path cannot be used safely outside its tree root")

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
                raise BehaviorTreeIntegrityError(f"{label} exceeds its configured read bound") from exc
            raise BehaviorTreeIntegrityError(f"{label} cannot be read safely") from exc
        except UnicodeDecodeError as exc:
            raise BehaviorTreeIntegrityError(f"{label} is not valid UTF-8") from exc

    def _atomic_write(self, path: Path, payload: bytes) -> None:
        try:
            atomic_replace_bytes(path, payload, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise BehaviorTreeIntegrityError("behavior file cannot be written safely") from exc


__all__ = ["BehaviorTree", "BehaviorTreeConflictError", "BehaviorTreeIntegrityError"]
