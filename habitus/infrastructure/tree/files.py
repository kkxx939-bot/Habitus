"""一棵树里的文件操作：路径解析（大小写与 Unicode 别名算冲突）、有界枚举与每目录子项上限、安全读、原子替换与原子新建。

记忆的树和行为的每一支视图都建在它上面，各自给出自己的报错类型和上限；它不知道路径表示什么。
"""

from __future__ import annotations

from pathlib import Path

from habitus.foundation.ids import canonical_path_identity
from habitus.infrastructure.store.filesystem import (
    DurableDirectoryEntry,
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
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
from habitus.infrastructure.tree.layout import SIDECAR_FILENAMES


def _is_temporary(entry: DurableDirectoryEntry) -> bool:
    return entry.name.startswith(".") and entry.is_file() and atomic_temporary_destination(entry.name) is not None


class TreeFiles:
    """``root`` 下的安全文件操作。``error`` 是完整性问题抛的类型，``conflict`` 是只增文件撞上不同内容时抛的类型。"""

    def __init__(
        self,
        root: str | Path,
        *,
        max_children: int,
        error: type[ValueError],
        conflict: type[Exception],
    ) -> None:
        requested = Path(root).expanduser().absolute()
        if requested.is_symlink() or requested.is_junction():
            raise error("tree root cannot be a symbolic link")
        self.root = requested.resolve(strict=False)
        self.max_children = max_children
        self.error = error
        self.conflict = conflict

    # ── 路径 ────────────────────────────────────────────────────────────────

    def require_inside(self, path: Path) -> Path:
        candidate = path.resolve(strict=False)
        if candidate != self.root and self.root not in candidate.parents:
            raise self.error("tree path cannot be used safely outside its tree root")
        return path

    def resolve(self, relative: Path, *, leaf_is_file: bool) -> Path:
        """逐级把逻辑路径解析到真实物理条目：每一级按规范身份匹配，命中多个就是同一身份有别名，拒绝；
        没命中则原样拼出尚未创建的剩余路径。"""

        current = self.root
        parts = relative.parts
        for index, part in enumerate(parts):
            desired = canonical_path_identity(part, "tree path segment")
            matches: list[DurableDirectoryEntry] = []
            for entry in self.children(current):
                try:
                    identity = canonical_path_identity(entry.name, "tree path entry")
                except ValueError:
                    continue
                if identity == desired:
                    matches.append(entry)
            if len(matches) > 1:
                raise self.error("tree contains multiple physical aliases for one identity")
            if not matches:
                return current.joinpath(*parts[index:])
            selected = matches[0]
            is_leaf = index == len(parts) - 1
            if (is_leaf and leaf_is_file and not selected.is_file()) or (
                (not is_leaf or not leaf_is_file) and not selected.is_dir()
            ):
                raise self.error("tree identity path has an incompatible physical entry")
            current = current / selected.name
        return current

    # ── 枚举 ────────────────────────────────────────────────────────────────

    def children(self, directory: Path) -> tuple[DurableDirectoryEntry, ...]:
        """直接条目（含原子写临时文件）；持久条目超过上限整目录拒读。"""

        try:
            entries = list_real_directory(directory, artifact_root=self.root, max_entries=self.max_children + 1)
        except DurablePathIntegrityError as exc:
            raise self.error("tree directory cannot be enumerated safely") from exc
        if sum(not _is_temporary(entry) for entry in entries) > self.max_children:
            raise self.error("tree directory exceeds its persistent child capacity")
        return entries

    def content_children(self, directory: Path) -> tuple[DurableDirectoryEntry, ...]:
        """去掉原子写临时文件和两个侧车文件之后的条目；别的隐藏条目报错。"""

        content: list[DurableDirectoryEntry] = []
        for child in self.children(directory):
            if _is_temporary(child):
                continue
            if child.name in SIDECAR_FILENAMES:
                if not child.is_file():
                    raise self.error("tree semantic layer is not a regular file")
                continue
            if child.name.startswith("."):
                raise self.error("tree directory contains an unsupported hidden entry")
            content.append(child)
        return tuple(content)

    def markdown_names(self, directory: Path) -> tuple[str, ...]:
        """叶目录里的 Markdown 文件名（去掉 ``.md``），按规范身份排序；同一身份两个物理名报错。"""

        names_by_identity: dict[str, str] = {}
        for child in self.content_children(directory):
            child_path = Path(child.name)
            if not child.is_file() or child_path.suffix != ".md" or not child_path.stem:
                raise self.error("tree leaf directory may contain only Markdown files")
            try:
                identity = canonical_path_identity(child_path.stem, "tree document name")
            except ValueError as exc:
                raise self.error("tree leaf contains an invalid document name") from exc
            if identity in names_by_identity:
                raise self.error("tree leaf contains multiple physical aliases for one identity")
            names_by_identity[identity] = child_path.stem
        return tuple(names_by_identity[identity] for identity in sorted(names_by_identity))

    def directories(self, directory: Path) -> tuple[Path, ...]:
        children = self.content_children(directory)
        if any(not child.is_dir() for child in children):
            raise self.error("tree branch may contain only directories")
        return tuple(directory / child.name for child in children)

    def require_child_capacity(self, directory: Path, child_names: tuple[str, ...]) -> None:
        """再放进 ``child_names`` 之后，持久条目不超过上限（同身份的已有条目不重复计）。"""

        entries = tuple(entry for entry in self.children(directory) if not _is_temporary(entry))
        existing: set[str] = set()
        for entry in entries:
            try:
                existing.add(canonical_path_identity(entry.name, "tree path entry"))
            except ValueError:
                continue
        requested = {canonical_path_identity(name, "tree path child") for name in child_names}
        if len(entries) + len(requested - existing) > self.max_children:
            raise self.error("tree directory has no remaining child capacity")

    # ── 状态 ────────────────────────────────────────────────────────────────

    def file_exists(self, path: Path) -> bool:
        try:
            return regular_file_exists(path, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise self.error("tree file cannot be inspected safely") from exc

    def directory_exists(self, path: Path) -> bool:
        try:
            return real_directory_exists(path, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise self.error("tree directory cannot be inspected safely") from exc

    # ── 读写 ────────────────────────────────────────────────────────────────

    def ensure_directory(self, directory: Path, *, check_capacity: bool) -> None:
        """建目录（含缺的上级）。``check_capacity`` 为真时，新建前先查上级还放得下。"""

        self.require_inside(directory)
        if directory != self.root:
            self.ensure_directory(directory.parent, check_capacity=check_capacity)
            directory = self.resolve(directory.relative_to(self.root), leaf_is_file=False)
            if check_capacity and not self.directory_exists(directory):
                self.require_child_capacity(directory.parent, (directory.name,))
        try:
            ensure_real_directory(directory, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise self.error("tree directory cannot be created safely") from exc

    def read_utf8(self, path: Path, *, label: str, max_bytes: int) -> str:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        try:
            return read_regular_bytes(path, artifact_root=self.root, max_bytes=max_bytes).decode("utf-8")
        except DurablePathIntegrityError as exc:
            if "read bound" in str(exc):
                raise self.error(f"{label} exceeds its configured read bound") from exc
            raise self.error(f"{label} cannot be read safely") from exc
        except UnicodeDecodeError as exc:
            raise self.error(f"{label} is not valid UTF-8") from exc

    def replace(self, path: Path, payload: bytes) -> None:
        """原子替换（可改的卡片与侧车文件）。"""

        try:
            atomic_replace_bytes(path, payload, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise self.error("tree file cannot be written safely") from exc

    def create(self, path: Path, payload: bytes) -> None:
        """原子新建（只增的事件与空白）：同内容的重放算成功，已绑定到不同内容抛 ``conflict``。"""

        try:
            atomic_create_bytes(path, payload, artifact_root=self.root)
        except ImmutableArtifactConflictError as exc:
            raise self.conflict("tree address already contains different content") from exc

    def unlink(self, path: Path) -> bool:
        try:
            return durable_unlink(path, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise self.error("tree file cannot be deleted safely") from exc

    def rmdir(self, path: Path) -> bool:
        """删空目录；目录不空时照 ``os.rmdir`` 抛 ``OSError``。"""

        try:
            return durable_rmdir(path, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise self.error("tree directory cannot be removed safely") from exc


__all__ = ["TreeFiles"]
