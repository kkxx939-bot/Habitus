"""语义树各支存储共用的底座：根校验、容量预检、durable_io 原语的错误归一。

各支（概念、命中，以及接下来的关系表）都落在同一个 ``scene`` 根下，守同一套纪律：

- 根本身是符号链接就拒，不静默跟过去；
- 路径**只校验、返回未解析的形式**——返回 ``resolve()`` 之后的路径等于替它把符号链接洗白了
  （实测过：``kinds/打球 → kinds/real`` 会让两个候选的历史被合并）；
- 目录条目有上限，**写之前**预检——写成一个列不出来的目录比拒绝写更糟（那个目录从此只能写不能读）；
- 每次写都回读比对；
- 崩溃遗留的原子写临时文件与 ``.DS_Store`` 这类点文件是噪音：读时跳过，重做时清掉，
  不当记录解码、也不让它们把一天锁死。

每支只需要给出自己的错误类型，剩下的这里管。
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from habitus.foundation.ids import canonical_path_identity
from habitus.infrastructure.store.filesystem import (
    DurableDirectoryEntry,
    DurablePathIntegrityError,
    atomic_replace_bytes,
    atomic_temporary_destination,
    durable_unlink,
    ensure_real_directory,
    list_real_directory,
    read_regular_bytes,
    real_directory_exists,
    regular_file_exists,
)

#: 与行为树 ``BehaviorTreeConfig.max_children_per_directory`` 的默认值一致；语义树镜像行为树的日目录，
#: 不能比它紧。
DEFAULT_MAX_DIRECTORY_ENTRIES = 10_000


class SceneStore:
    """子类只需设 ``error_type``。所有底层失败都归一成那一种错误。"""

    error_type: ClassVar[type[ValueError]] = ValueError

    def __init__(self, scene_root: str | Path, *, max_directory_entries: int = DEFAULT_MAX_DIRECTORY_ENTRIES) -> None:
        requested = Path(scene_root)
        if requested.is_symlink():
            raise self.error_type("scene root cannot be a symbolic link")
        if (
            isinstance(max_directory_entries, bool)
            or not isinstance(max_directory_entries, int)
            or max_directory_entries <= 0
        ):
            raise self.error_type("max_directory_entries must be a positive integer")
        self.root = requested.resolve(strict=False)
        self.max_directory_entries = max_directory_entries

    # ── 路径 ────────────────────────────────────────────────────────────────

    def _inside(self, relative: Path) -> Path:
        """只校验，返回未解析的路径。"""

        candidate = (self.root / relative).resolve(strict=False)
        if candidate != self.root and self.root not in candidate.parents:
            raise self.error_type("scene path cannot be used outside its tree root")
        return self.root / relative

    # ── 文件系统 ────────────────────────────────────────────────────────────

    def _atomic_write(self, path: Path, payload: bytes, *, maximum: int) -> None:
        """原子写 + 回读逐字节比对；写之前预检目录容量。"""

        if len(payload) > maximum:
            raise self.error_type("scene record exceeds its byte budget")
        self._require_capacity(path.parent, path.name)
        try:
            atomic_replace_bytes(path, payload, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise self.error_type("scene file cannot be written safely") from exc
        if self._read_bytes(path, maximum) != payload:
            raise self.error_type("scene file failed read-back verification")

    def _read_bytes(self, path: Path, maximum: int) -> bytes | None:
        try:
            if not regular_file_exists(path, artifact_root=self.root):
                return None
            return read_regular_bytes(path, artifact_root=self.root, max_bytes=maximum)
        except FileNotFoundError:
            return None
        except DurablePathIntegrityError as exc:
            raise self.error_type("scene file cannot be read safely") from exc

    def _file_exists(self, path: Path) -> bool:
        try:
            return regular_file_exists(path, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise self.error_type("scene file cannot be inspected safely") from exc

    def _discard(self, path: Path) -> None:
        try:
            durable_unlink(path, artifact_root=self.root)
        except FileNotFoundError:
            return
        except DurablePathIntegrityError as exc:
            raise self.error_type("scene leftover cannot be removed safely") from exc

    def _ensure_directory(self, directory: Path) -> None:
        self._inside(directory.relative_to(self.root))
        try:
            ensure_real_directory(directory, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise self.error_type("scene directory cannot be created safely") from exc

    def _children(self, directory: Path) -> tuple[DurableDirectoryEntry, ...]:
        try:
            if not real_directory_exists(directory, artifact_root=self.root):
                return ()
            return tuple(
                list_real_directory(directory, artifact_root=self.root, max_entries=self.max_directory_entries)
            )
        except DurablePathIntegrityError as exc:
            raise self.error_type("scene directory cannot be listed safely") from exc

    def _directories(self, directory: Path) -> tuple[Path, ...]:
        return tuple(directory / entry.name for entry in self._children(directory) if entry.is_dir())

    def _content_files(
        self, directory: Path
    ) -> tuple[tuple[DurableDirectoryEntry, ...], tuple[DurableDirectoryEntry, ...]]:
        """把目录里的文件分成 (记录, 噪音)。噪音 = 点文件与原子写临时文件；子目录在这里报错。"""

        records: list[DurableDirectoryEntry] = []
        noise: list[DurableDirectoryEntry] = []
        for entry in self._children(directory):
            if not entry.is_file():
                raise self.error_type(f"unexpected directory inside a scene record directory: {entry.name!r}")
            if entry.name.startswith(".") or atomic_temporary_destination(entry.name) is not None:
                noise.append(entry)
            else:
                records.append(entry)
        return tuple(records), tuple(noise)

    def _discard_noise(self, directory: Path, *, keep: frozenset[str] = frozenset()) -> None:
        """清掉噪音文件（``keep`` 里的点文件除外，例如完成标记）。"""

        _records, noise = self._content_files(directory)
        for entry in noise:
            if entry.name not in keep:
                self._discard(directory / entry.name)

    def _require_capacity(self, directory: Path, name: str) -> None:
        """与行为树同一条：算规范身份，同名覆写不占新位。"""

        entries = tuple(
            entry
            for entry in self._children(directory)
            if not (entry.is_file() and atomic_temporary_destination(entry.name) is not None)
        )
        existing: set[str] = set()
        for entry in entries:
            try:
                existing.add(canonical_path_identity(entry.name, "scene path entry"))
            except ValueError:
                continue
        additional = 0 if canonical_path_identity(name, "scene path child") in existing else 1
        if len(entries) + additional > self.max_directory_entries:
            raise self.error_type("scene directory has no remaining child capacity")


__all__ = ["DEFAULT_MAX_DIRECTORY_ENTRIES", "SceneStore"]
