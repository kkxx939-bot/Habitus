"""严格限定于三种不可变样本分支的预测 Markdown 树。"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import date
from pathlib import Path

from foundation.ids import canonical_path_identity
from foundation.integrity import canonical_digest
from infrastructure.store.contracts.path_lock import PathLock
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
from infrastructure.store.sqlite.lock_store import SQLiteLockStore
from prediction.document import (
    PredictionDocument,
    PredictionDocumentCodec,
    PredictionDocumentConfig,
    PredictionDocumentIntegrityError,
    PredictionDocumentLimitError,
)
from prediction.model import (
    PredictionAddress,
    PredictionDirectory,
    PredictionKind,
    PredictionLevel,
    PredictionPatternAddress,
    PredictionPatternKind,
    is_ascii_digits,
    prediction_sample_id,
    prediction_shard,
)
from prediction.pattern.document import (
    PredictionPatternDocument,
    PredictionPatternDocumentCodec,
    PredictionPatternDocumentIntegrityError,
)
from prediction.schema import PredictionSchemaRegistry
from prediction.tree.config import PredictionTreeConfig
from prediction.uri import PredictionURI, PredictionURINodeType


class PredictionTreeIntegrityError(ValueError):
    """预测树包含路径逃逸、符号链接或非规范条目。"""


class PredictionTreeConflictError(RuntimeError):
    """不可变样本地址已经绑定到不同内容。"""


class PredictionTree:
    """安全持久化不可变预测样本和可重建目录语义层。"""

    _STATIC_DIRECTORIES = (
        ("samples",),
        *(("samples", kind.branch_name) for kind in PredictionKind),
        ("patterns",),
        ("patterns", "states"),
        ("patterns", "branches"),
    )

    def __init__(
        self,
        root: str | Path,
        *,
        document_codec: PredictionDocumentCodec | None = None,
        pattern_codec: PredictionPatternDocumentCodec | None = None,
        document_config: PredictionDocumentConfig | None = None,
        tree_config: PredictionTreeConfig | None = None,
        path_lock: PathLock | None = None,
    ) -> None:
        requested = Path(root).expanduser().absolute()
        if requested.is_symlink():
            raise PredictionTreeIntegrityError("prediction tree root cannot be a symbolic link")
        self.root = requested.resolve(strict=False)
        if document_codec is None:
            document_codec = PredictionDocumentCodec(PredictionSchemaRegistry.load_default())
        if not isinstance(document_codec, PredictionDocumentCodec):
            raise TypeError("document_codec must be a PredictionDocumentCodec")
        if pattern_codec is not None and not isinstance(pattern_codec, PredictionPatternDocumentCodec):
            raise TypeError("pattern_codec must be PredictionPatternDocumentCodec")
        if document_config is not None and not isinstance(document_config, PredictionDocumentConfig):
            raise TypeError("document_config must be PredictionDocumentConfig")
        if tree_config is not None and not isinstance(tree_config, PredictionTreeConfig):
            raise TypeError("tree_config must be PredictionTreeConfig")
        if path_lock is not None and not isinstance(path_lock, PathLock):
            raise TypeError("path_lock must be PathLock")
        self._document_codec = document_codec
        self._pattern_codec = pattern_codec or PredictionPatternDocumentCodec()
        self.document_config = document_config or PredictionDocumentConfig()
        self.tree_config = tree_config or PredictionTreeConfig()
        lock_path = self.root.parent / f".{self.root.name}.prediction-locks.sqlite3"
        self._path_lock = path_lock or PathLock(SQLiteLockStore(lock_path, initialize=False))
        self._write_lock_key = f"prediction-tree:{canonical_digest({'root': str(self.root)})}"

    @property
    def document_codec(self) -> PredictionDocumentCodec:
        return self._document_codec

    @property
    def registry(self) -> PredictionSchemaRegistry:
        """树与其写入方必须共用同一个 Schema 注册表，这里是唯一的取用入口。"""

        return self._document_codec.registry

    @property
    def pattern_codec(self) -> PredictionPatternDocumentCodec:
        return self._pattern_codec

    def initialize(self) -> Path:
        self._ensure_directory(self.root)
        for parts in self._STATIC_DIRECTORIES:
            self._ensure_directory(self.root.joinpath(*parts))
        return self.root

    def path_for(self, address: PredictionAddress) -> Path:
        if not isinstance(address, PredictionAddress):
            raise TypeError("address must be a PredictionAddress")
        candidate = self.root / self._relative_path(address)
        self._require_inside_root(candidate)
        return candidate

    def path_for_pattern(self, address: PredictionPatternAddress) -> Path:
        if not isinstance(address, PredictionPatternAddress):
            raise TypeError("address must be a PredictionPatternAddress")
        candidate = self.root.joinpath(*PredictionURI.from_pattern_address(address).segments)
        self._require_inside_root(candidate)
        return candidate

    def directory_path(self, directory: PredictionDirectory) -> Path:
        if not isinstance(directory, PredictionDirectory):
            raise TypeError("directory must be a PredictionDirectory")
        candidate = self.root.joinpath(*directory.identity_parts)
        self._require_inside_root(candidate)
        return candidate

    def layer_path(self, directory: PredictionDirectory, level: PredictionLevel) -> Path:
        normalized = PredictionLevel(level)
        return self.directory_path(directory) / normalized.sidecar_filename

    def path_for_uri(self, uri: PredictionURI | str) -> Path:
        parsed = PredictionURI.parse(uri)
        if parsed.node_type is PredictionURINodeType.DOCUMENT:
            if parsed.is_pattern_document:
                return self.path_for_pattern(parsed.to_pattern_address())
            return self.path_for(parsed.to_address())
        if parsed.node_type is PredictionURINodeType.DIRECTORY:
            return self.directory_path(parsed.to_directory())
        directory, level = parsed.to_layer()
        return self.layer_path(directory, level)

    def create(self, document: PredictionDocument) -> PredictionDocument:
        """幂等原子创建不可变样本；相同字节重放不产生变更。"""

        if not isinstance(document, PredictionDocument):
            raise TypeError("document must be a PredictionDocument")
        return self.create_many((document,))[0]

    def create_many(self, documents: Sequence[PredictionDocument]) -> tuple[PredictionDocument, ...]:
        """在一个目录 fencing 域中预检整批容量，再幂等创建不可变样本。"""

        if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
            raise TypeError("documents must be a sequence of PredictionDocument values")
        prepared: list[tuple[PredictionDocument, bytes]] = []
        encoded_by_address: dict[PredictionAddress, bytes] = {}
        for document in documents:
            if not isinstance(document, PredictionDocument):
                raise TypeError("documents must contain PredictionDocument values")
            encoded = self._document_codec.encode(document).encode("utf-8")
            self.document_config.validate_body(document.markdown_body)
            self.document_config.validate_encoded(encoded)
            previous = encoded_by_address.get(document.address)
            if previous is not None:
                if previous != encoded:
                    raise PredictionTreeConflictError(
                        "one prediction batch binds one sample identity to different content"
                    )
                continue
            encoded_by_address[document.address] = encoded
            prepared.append((document, encoded))
        if not prepared:
            return ()
        with self._write_fence():
            self.initialize()
            paths: list[tuple[PredictionDocument, bytes, Path]] = []
            children_by_parent: dict[Path, list[str]] = {}
            for document, encoded in prepared:
                path = self._existing_document_path(document.address)
                self._ensure_directory(path.parent)
                paths.append((document, encoded, path))
                children_by_parent.setdefault(path.parent, []).append(path.name)
            for parent, child_names in children_by_parent.items():
                self._require_child_capacity(parent, tuple(child_names))
            for _document, encoded, path in paths:
                try:
                    exists = regular_file_exists(path, artifact_root=self.root)
                    if exists:
                        current = read_regular_bytes(
                            path,
                            artifact_root=self.root,
                            max_bytes=self.document_config.max_encoded_bytes,
                        )
                        if current != encoded:
                            raise PredictionTreeConflictError(
                                "prediction sample identity conflicts with different content"
                            )
                except DurablePathIntegrityError as exc:
                    raise PredictionTreeIntegrityError(
                        "prediction sample cannot be preflighted safely"
                    ) from exc
            for _document, encoded, path in paths:
                try:
                    atomic_create_bytes(path, encoded, artifact_root=self.root)
                except ImmutableArtifactConflictError as exc:
                    raise PredictionTreeConflictError(
                        "prediction sample identity conflicts with different content"
                    ) from exc
        return tuple(document for document, _encoded in prepared)

    def read(self, address: PredictionAddress) -> PredictionDocument:
        if not isinstance(address, PredictionAddress):
            raise TypeError("address must be a PredictionAddress")
        raw = self._read_utf8(
            self._existing_document_path(address),
            label="prediction document",
            max_bytes=self.document_config.max_encoded_bytes,
        )
        try:
            document = self._document_codec.decode(raw, expected_address=address)
            self.document_config.validate_body(document.markdown_body)
            return document
        except (PredictionDocumentIntegrityError, PredictionDocumentLimitError) as exc:
            raise PredictionTreeIntegrityError("prediction sample failed integrity validation") from exc

    def create_pattern(self, document: PredictionPatternDocument) -> PredictionPatternDocument:
        """幂等创建一个不可变 StatePattern 或 BranchPattern。"""

        if not isinstance(document, PredictionPatternDocument):
            raise TypeError("document must be a PredictionPatternDocument")
        encoded = self._pattern_codec.encode(document).encode("utf-8")
        self.document_config.validate_body(document.markdown_body)
        self.document_config.validate_encoded(encoded)
        with self._write_fence():
            self.initialize()
            for sample_ref in document.fields["sample_refs"]:
                sample_address = PredictionURI.parse(sample_ref).to_address()
                if not self.exists(sample_address):
                    raise PredictionTreeIntegrityError(
                        "prediction pattern references a missing PredictionSample"
                    )
                sample = self.read(sample_address)
                if (
                    sample.fields["provenance"]["projection_version"]
                    != document.fields["projection_version"]
                ):
                    raise PredictionTreeIntegrityError(
                        "prediction pattern and sample use different projection versions"
                    )
            if document.kind is PredictionPatternKind.BRANCH:
                state_address = PredictionPatternAddress(
                    PredictionPatternKind.STATE,
                    document.address.state_key,
                )
                if not self.pattern_exists(state_address):
                    raise PredictionTreeIntegrityError(
                        "BranchPattern requires its parent StatePattern to exist"
                    )
                state = self.read_pattern(state_address)
                if (
                    state.fields["pattern_generation"] != document.fields["pattern_generation"]
                    or state.fields["logical_state_key"]
                    != document.fields["identity_material"]["logical_state_key"]
                ):
                    raise PredictionTreeIntegrityError(
                        "BranchPattern parent identity does not match its StatePattern"
                    )
            path = self._existing_pattern_path(document.address)
            self._ensure_directory(path.parent)
            self._require_child_capacity(path.parent, (path.name,))
            try:
                atomic_create_bytes(path, encoded, artifact_root=self.root)
            except ImmutableArtifactConflictError as exc:
                raise PredictionTreeConflictError("prediction pattern identity conflicts with different content") from exc
        return document

    def read_pattern(self, address: PredictionPatternAddress) -> PredictionPatternDocument:
        if not isinstance(address, PredictionPatternAddress):
            raise TypeError("address must be a PredictionPatternAddress")
        raw = self._read_utf8(
            self._existing_pattern_path(address),
            label="prediction pattern document",
            max_bytes=self.document_config.max_encoded_bytes,
        )
        try:
            document = self._pattern_codec.decode(raw, expected_address=address)
            self.document_config.validate_body(document.markdown_body)
            return document
        except (PredictionPatternDocumentIntegrityError, PredictionDocumentLimitError) as exc:
            raise PredictionTreeIntegrityError("prediction pattern failed integrity validation") from exc

    def exists(self, address: PredictionAddress) -> bool:
        if not isinstance(address, PredictionAddress):
            raise TypeError("address must be a PredictionAddress")
        try:
            return regular_file_exists(self._existing_document_path(address), artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise PredictionTreeIntegrityError("prediction sample cannot be inspected safely") from exc

    def pattern_exists(self, address: PredictionPatternAddress) -> bool:
        if not isinstance(address, PredictionPatternAddress):
            raise TypeError("address must be a PredictionPatternAddress")
        try:
            return regular_file_exists(self._existing_pattern_path(address), artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise PredictionTreeIntegrityError("prediction pattern cannot be inspected safely") from exc

    def write_layers(
        self,
        directory: PredictionDirectory,
        *,
        abstract: str,
        overview: str,
    ) -> tuple[Path, Path]:
        """写入可重建 L1/L0；它们只用于候选检索，不是监督标签。"""

        if not isinstance(abstract, str) or not isinstance(overview, str):
            raise TypeError("prediction semantic layers must be strings")
        if not abstract.strip() or not overview.strip():
            raise ValueError("prediction semantic layers must be non-empty")
        encoded_abstract = abstract.encode("utf-8")
        encoded_overview = overview.encode("utf-8")
        self.document_config.validate_semantic_layer(encoded_abstract)
        self.document_config.validate_semantic_layer(encoded_overview)
        with self._write_fence():
            directory_path = self._existing_directory_path(directory)
            if not self.directory_exists(directory):
                raise PredictionTreeIntegrityError("prediction path is not a safe directory")
            overview_path = directory_path / PredictionLevel.OVERVIEW.sidecar_filename
            abstract_path = directory_path / PredictionLevel.ABSTRACT.sidecar_filename
            self._require_child_capacity(directory_path, (overview_path.name, abstract_path.name))
            self._atomic_write(overview_path, encoded_overview)
            self._atomic_write(abstract_path, encoded_abstract)
        return abstract_path, overview_path

    def read_layer(self, directory: PredictionDirectory, level: PredictionLevel) -> str:
        return self._read_utf8(
            self._existing_layer_path(directory, level),
            label="prediction semantic layer",
        )

    def layer_exists(self, directory: PredictionDirectory, level: PredictionLevel) -> bool:
        try:
            return regular_file_exists(
                self._existing_layer_path(directory, level),
                artifact_root=self.root,
            )
        except DurablePathIntegrityError as exc:
            raise PredictionTreeIntegrityError("prediction semantic layer cannot be inspected safely") from exc

    def directory_exists(self, directory: PredictionDirectory) -> bool:
        try:
            return real_directory_exists(
                self._existing_directory_path(directory),
                artifact_root=self.root,
            )
        except DurablePathIntegrityError as exc:
            raise PredictionTreeIntegrityError("prediction directory cannot be inspected safely") from exc

    def list_addresses(
        self,
        kind: PredictionKind | None = None,
        *,
        limit: int = 256,
        after: PredictionAddress | None = None,
    ) -> tuple[PredictionAddress, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("prediction tree list limit must be between 1 and 10000")
        if after is not None and not isinstance(after, PredictionAddress):
            raise TypeError("prediction tree list cursor must be PredictionAddress or None")
        selected_kind = None if kind is None else PredictionKind(kind)
        if after is not None and selected_kind is not None and after.kind is not selected_kind:
            raise ValueError("prediction tree list cursor kind does not match selected kind")
        if not self.directory_exists(PredictionDirectory.root()):
            return ()
        after_key = None if after is None else self._address_order_key(after)
        result: list[PredictionAddress] = []
        kinds = (selected_kind,) if selected_kind is not None else tuple(PredictionKind)
        for resolved_kind in kinds:
            for address in self._iter_kind(resolved_kind):
                if after_key is not None and self._address_order_key(address) <= after_key:
                    continue
                result.append(address)
                if len(result) == limit:
                    return tuple(result)
        return tuple(result)

    def list_pattern_addresses(
        self,
        kind: PredictionPatternKind | None = None,
        *,
        limit: int = 256,
        after: PredictionPatternAddress | None = None,
    ) -> tuple[PredictionPatternAddress, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("prediction pattern list limit must be between 1 and 10000")
        selected = None if kind is None else PredictionPatternKind(kind)
        if after is not None and not isinstance(after, PredictionPatternAddress):
            raise TypeError("prediction pattern list cursor must be PredictionPatternAddress or None")
        if after is not None and selected is not None and after.kind is not selected:
            raise ValueError("prediction pattern list cursor kind does not match selected kind")
        if not self.directory_exists(PredictionDirectory.root()):
            return ()
        after_key = None if after is None else self._pattern_address_order_key(after)
        result: list[PredictionPatternAddress] = []
        if selected in {None, PredictionPatternKind.STATE}:
            states = PredictionDirectory(("patterns", "states"))
            if self.directory_exists(states):
                for shard_path in self._directories(self._existing_directory_path(states)):
                    for state_key in self._markdown_names(shard_path):
                        address = PredictionPatternAddress(PredictionPatternKind.STATE, state_key)
                        if after_key is not None and self._pattern_address_order_key(address) <= after_key:
                            continue
                        result.append(address)
                        if len(result) == limit:
                            return tuple(result)
            elif selected is PredictionPatternKind.STATE:
                return ()
        if selected in {None, PredictionPatternKind.BRANCH}:
            branches = PredictionDirectory(("patterns", "branches"))
            if not self.directory_exists(branches):
                return tuple(result)
            for shard_path in self._directories(self._existing_directory_path(branches)):
                for state_path in self._directories(shard_path):
                    try:
                        state_key = prediction_sample_id(state_path.name)
                    except ValueError as exc:
                        raise PredictionTreeIntegrityError("BranchPattern directory has an invalid state key") from exc
                    for branch_key in self._markdown_names(state_path):
                        address = PredictionPatternAddress(
                            PredictionPatternKind.BRANCH,
                            state_key,
                            branch_key,
                        )
                        if after_key is not None and self._pattern_address_order_key(address) <= after_key:
                            continue
                        result.append(address)
                        if len(result) == limit:
                            return tuple(result)
        return tuple(result)

    def list_branch_addresses(
        self,
        state_key: str,
        *,
        limit: int = 256,
        after: PredictionPatternAddress | None = None,
    ) -> tuple[PredictionPatternAddress, ...]:
        """直接枚举一个 StatePattern 的分支目录，不扫描其他状态。"""

        normalized_state_key = prediction_sample_id(state_key)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("prediction branch list limit must be between 1 and 10000")
        if after is not None and not isinstance(after, PredictionPatternAddress):
            raise TypeError("prediction branch list cursor must be PredictionPatternAddress or None")
        if after is not None and (
            after.kind is not PredictionPatternKind.BRANCH
            or after.state_key != normalized_state_key
        ):
            raise ValueError("prediction branch list cursor must belong to the selected StatePattern")
        directory = PredictionDirectory(
            ("patterns", "branches", prediction_shard(normalized_state_key), normalized_state_key)
        )
        if not self.directory_exists(directory):
            return ()
        names = self._markdown_names(self._existing_directory_path(directory))
        addresses = tuple(
            PredictionPatternAddress(PredictionPatternKind.BRANCH, normalized_state_key, branch_key)
            for branch_key in names
        )
        if after is not None:
            after_key = self._pattern_address_order_key(after)
            addresses = tuple(
                address
                for address in addresses
                if self._pattern_address_order_key(address) > after_key
            )
        return addresses[:limit]

    @staticmethod
    def _relative_path(address: PredictionAddress) -> Path:
        return Path(*PredictionURI.from_address(address).segments)

    @classmethod
    def _address_order_key(cls, address: PredictionAddress) -> tuple[int, str]:
        return tuple(PredictionKind).index(address.kind), cls._relative_path(address).as_posix()

    @staticmethod
    def _pattern_address_order_key(address: PredictionPatternAddress) -> tuple[int, str, str]:
        return (
            tuple(PredictionPatternKind).index(address.kind),
            address.state_key,
            address.branch_key or "",
        )

    def _iter_kind(self, kind: PredictionKind) -> Iterator[PredictionAddress]:
        root = self.root / "samples" / kind.branch_name
        for year_path in self._directories(root):
            if len(year_path.name) != 4 or not is_ascii_digits(year_path.name):
                raise PredictionTreeIntegrityError("prediction year directory must use YYYY")
            for month_path in self._directories(year_path):
                if len(month_path.name) != 2 or not is_ascii_digits(month_path.name):
                    raise PredictionTreeIntegrityError("prediction month directory must use MM")
                for day_path in self._directories(month_path):
                    if len(day_path.name) != 2 or not is_ascii_digits(day_path.name):
                        raise PredictionTreeIntegrityError("prediction day directory must use DD")
                    try:
                        sample_date = date(int(year_path.name), int(month_path.name), int(day_path.name))
                    except ValueError as exc:
                        raise PredictionTreeIntegrityError("prediction directory contains an invalid date") from exc
                    for shard_path in self._directories(day_path):
                        for materialization_id in self._markdown_names(shard_path):
                            try:
                                yield PredictionAddress(kind, sample_date, materialization_id)
                            except (TypeError, ValueError) as exc:
                                raise PredictionTreeIntegrityError(
                                    "prediction leaf contains an invalid sample ID"
                                ) from exc

    def _existing_document_path(self, address: PredictionAddress) -> Path:
        return self._existing_relative_path(self._relative_path(address), leaf_is_file=True)

    def _existing_pattern_path(self, address: PredictionPatternAddress) -> Path:
        relative = Path(*PredictionURI.from_pattern_address(address).segments)
        return self._existing_relative_path(relative, leaf_is_file=True)

    def _existing_directory_path(self, directory: PredictionDirectory) -> Path:
        return self._existing_relative_path(Path(*directory.identity_parts), leaf_is_file=False)

    def _existing_layer_path(self, directory: PredictionDirectory, level: PredictionLevel) -> Path:
        normalized = PredictionLevel(level)
        relative = Path(*directory.identity_parts, normalized.sidecar_filename)
        return self._existing_relative_path(relative, leaf_is_file=True)

    def _existing_relative_path(self, relative: Path, *, leaf_is_file: bool) -> Path:
        current = self.root
        parts = relative.parts
        for index, part in enumerate(parts):
            entries = self._children(current)
            desired = canonical_path_identity(part, "prediction path segment")
            matches: list[DurableDirectoryEntry] = []
            for entry in entries:
                try:
                    identity = canonical_path_identity(entry.name, "prediction path entry")
                except ValueError:
                    continue
                if identity == desired:
                    matches.append(entry)
            if len(matches) > 1:
                raise PredictionTreeIntegrityError("prediction tree contains physical aliases for one identity")
            if not matches:
                return current.joinpath(*parts[index:])
            selected = matches[0]
            is_leaf = index == len(parts) - 1
            if (is_leaf and leaf_is_file and not selected.is_file()) or (
                (not is_leaf or not leaf_is_file) and not selected.is_dir()
            ):
                raise PredictionTreeIntegrityError("prediction identity path has an incompatible entry")
            current = current / selected.name
        return current

    def _markdown_names(self, directory: Path) -> tuple[str, ...]:
        names_by_identity: dict[str, str] = {}
        for child in self._content_children(directory):
            child_path = Path(child.name)
            if not child.is_file() or child_path.suffix != ".md" or not child_path.stem:
                raise PredictionTreeIntegrityError("prediction leaf directory may contain only Markdown files")
            identity = canonical_path_identity(child_path.stem, "prediction sample ID")
            if identity in names_by_identity:
                raise PredictionTreeIntegrityError("prediction leaf contains physical aliases for one identity")
            names_by_identity[identity] = child_path.stem
        return tuple(names_by_identity[identity] for identity in sorted(names_by_identity))

    def _directories(self, directory: Path) -> tuple[Path, ...]:
        children = self._content_children(directory)
        if any(not child.is_dir() for child in children):
            raise PredictionTreeIntegrityError("prediction branch may contain only directories")
        return tuple(directory / child.name for child in children)

    def _content_children(self, directory: Path) -> tuple[DurableDirectoryEntry, ...]:
        semantic_names = {
            PredictionLevel.ABSTRACT.sidecar_filename,
            PredictionLevel.OVERVIEW.sidecar_filename,
        }
        content: list[DurableDirectoryEntry] = []
        for child in self._children(directory):
            if child.name.startswith(".") and child.is_file() and atomic_temporary_destination(child.name) is not None:
                continue
            if child.name in semantic_names:
                if not child.is_file():
                    raise PredictionTreeIntegrityError("prediction semantic layer is not a regular file")
                continue
            if child.name.startswith("."):
                raise PredictionTreeIntegrityError("prediction directory contains an unsupported hidden entry")
            content.append(child)
        return tuple(content)

    def _children(self, directory: Path) -> tuple[DurableDirectoryEntry, ...]:
        try:
            entries = list_real_directory(
                directory,
                artifact_root=self.root,
                max_entries=(
                    self.tree_config.max_children_per_directory
                    + self.tree_config.max_temporary_files_per_directory
                    + 1
                ),
            )
        except DurablePathIntegrityError as exc:
            raise PredictionTreeIntegrityError("prediction directory cannot be enumerated safely") from exc
        temporary_count = sum(
            (
                entry.name.startswith(".")
                and entry.is_file()
                and atomic_temporary_destination(entry.name) is not None
            )
            for entry in entries
        )
        if temporary_count > self.tree_config.max_temporary_files_per_directory:
            raise PredictionTreeIntegrityError("prediction directory exceeds its temporary file capacity")
        persistent_count = len(entries) - temporary_count
        if persistent_count > self.tree_config.max_children_per_directory:
            raise PredictionTreeIntegrityError("prediction directory exceeds its persistent child capacity")
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
                raise PredictionTreeIntegrityError("prediction directory cannot be inspected safely") from exc
            if not exists:
                self._require_child_capacity(directory.parent, (directory.name,))
        try:
            ensure_real_directory(directory, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise PredictionTreeIntegrityError("prediction directory cannot be created safely") from exc

    def _require_child_capacity(self, directory: Path, child_names: tuple[str, ...]) -> None:
        entries = tuple(
            entry
            for entry in self._children(directory)
            if not (
                entry.name.startswith(".")
                and entry.is_file()
                and atomic_temporary_destination(entry.name) is not None
            )
        )
        existing = {
            canonical_path_identity(entry.name, "prediction path entry")
            for entry in entries
        }
        requested = {canonical_path_identity(name, "prediction path child") for name in child_names}
        if len(entries) + len(requested - existing) > self.tree_config.max_children_per_directory:
            raise PredictionTreeIntegrityError("prediction directory has no remaining child capacity")

    def _require_inside_root(self, path: Path) -> None:
        candidate = path.resolve(strict=False)
        if candidate != self.root and self.root not in candidate.parents:
            raise PredictionTreeIntegrityError("prediction path cannot escape its tree root")

    def _read_utf8(self, path: Path, *, label: str, max_bytes: int | None = None) -> str:
        maximum = self.document_config.max_encoded_bytes if max_bytes is None else max_bytes
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
            raise ValueError("max_bytes must be a positive integer")
        try:
            payload = read_regular_bytes(path, artifact_root=self.root, max_bytes=maximum)
            return payload.decode("utf-8")
        except DurablePathIntegrityError as exc:
            if "read bound" in str(exc):
                raise PredictionTreeIntegrityError(f"{label} exceeds its configured read bound") from exc
            raise PredictionTreeIntegrityError(f"{label} cannot be read safely") from exc
        except UnicodeDecodeError as exc:
            raise PredictionTreeIntegrityError(f"{label} is not valid UTF-8") from exc

    def _atomic_write(self, path: Path, payload: bytes) -> None:
        try:
            atomic_replace_bytes(path, payload, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise PredictionTreeIntegrityError("prediction file cannot be written safely") from exc

    @contextmanager
    def _write_fence(self) -> Iterator[None]:
        with self._path_lock.acquire(
            self._write_lock_key,
            ttl_seconds=30,
            wait_timeout_seconds=5.0,
        ) as guard:
            with guard.fenced():
                yield


__all__ = ["PredictionTree", "PredictionTreeConflictError", "PredictionTreeIntegrityError"]
