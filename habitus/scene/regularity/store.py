"""规律级的存储：L2 关联记录按地址写、L0 / L1 侧车按候选覆写。

与按天情景树最大的不同是**没有"代"**。那边一天一代、一个指针，是因为归组每次产出整天的划分，
要么整批换掉、要么整批回滚。这边不是：一次关联只新增**一条**记录（这个候选的这一次），历史
那些条一个字都不动。所以

- **L2 add-only**：一次发生一份，写完不改（人为重跑除外，那时按地址覆写）；
- **L0 / L1 侧车原子覆写**：每次关联更新，先写 L1 再写由它派生的 L0（与记忆树同序——L0 是
  L1 的摘要，反过来写会出现"摘要比它总结的东西还新"的一瞬）。

``days_for`` 是待关联清单的生产实现（``scene.backlog.AssociatedDays``）：按**完成标记**回答
"这天关联过没有"。

标记不能省，也不能拿"日目录在不在"代替。审查用真 SIGKILL 实测过：目录在第一条记录落盘**之前**
就建好了（而且 ``atomic_replace_bytes`` 自己也会建父目录，把 ``_ensure_directory`` 挪到写之后
没用），中断后树上留下空目录 + 临时文件；下一轮 ``days_for`` 说这天做完了、``read_day`` 返回空，
差集把它永久排除——那条关联再也不会被写出来。写失败（磁盘满）与"一天两次只写成一条"同样中招。

"不需要代"与"不需要提交点"是两件事，我一度混成一件。按天情景树的两阶段发布解决的不只是
"换哪一代"，还有"这一天算不算做完了"；这里 add-only 确实不需要代，但仍然需要那个提交点。
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

from habitus.behavior.model import is_ascii_digits
from habitus.foundation.integrity import canonical_json
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
from habitus.scene.model import (
    KINDS_SEGMENT,
    AssociationAddress,
    KindDirectory,
    RegularityLevel,
    _kind_identity,
    regularity_static_directories,
)
from habitus.scene.regularity.document import AssociationDocument, decode, encode

DONE_FILENAME = ".done.json"
MAX_RECORD_BYTES = 256 * 1024
MAX_LAYER_BYTES = 256 * 1024
# 一个候选一天最多几条关联、一棵树最多几个候选：有界列目录是这套存储的既定纪律，
# 越界说明上游出了事，该报出来而不是静默截断。
MAX_DIRECTORY_ENTRIES = 4096


class RegularityTreeError(ValueError):
    """规律级的路径逃逸、符号链接，或不符合目录结构的条目。"""


class RegularityTree:
    """``kinds/<候选>/…`` 这棵树：关联记录 + 两层侧车。"""

    def __init__(self, root: str | Path) -> None:
        requested = Path(root)
        # 与 ``SceneTree`` / ``MemoryTree`` 同一条：根本身是符号链接就拒，不能静默跟过去。
        if requested.is_symlink():
            raise RegularityTreeError("regularity tree root cannot be a symbolic link")
        self.root = requested.resolve(strict=False)

    def initialize(self) -> Path:
        self._ensure_directory(self.root)
        for parts in regularity_static_directories():
            self._ensure_directory(self._inside(Path(*parts)))
        return self.root

    # ── 写 ──────────────────────────────────────────────────────────────────

    def write(self, document: AssociationDocument) -> AssociationDocument:
        """原子写一条关联记录；写完立刻回读比对，不让半截字节留在树上。"""

        if not isinstance(document, AssociationDocument):
            raise TypeError("document must be an AssociationDocument")
        encoded = encode(document).encode("utf-8")
        if len(encoded) > MAX_RECORD_BYTES:
            raise RegularityTreeError("association record exceeds its byte budget")
        self.initialize()
        path = self._record_path(document.address)
        # 不预建日目录：``atomic_replace_bytes`` 自己会建全部父目录，而提前建只会让
        # "这天做完了"这个事实比它描述的工作更早成立。完成与否由 ``complete_day`` 的标记说了算。
        self._atomic_write(path, encoded)
        if self._read_bytes(path, MAX_RECORD_BYTES) != encoded:
            raise RegularityTreeError("association record failed read-back verification")
        return document

    def write_layers(self, kind_token: str, *, overview: str, abstract: str) -> tuple[Path, Path]:
        """先写 L1、再写由它派生的 L0。

        顺序是纪律不是偏好：L0 是 L1 的摘要，先写 L0 会留下"摘要比它总结的东西还新"的一瞬。
        """

        for name, value in (("overview", overview), ("abstract", abstract)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"regularity {name} layer must be non-empty text")
            if len(value.encode("utf-8")) > MAX_LAYER_BYTES:
                raise RegularityTreeError(f"regularity {name} layer exceeds its byte budget")
        directory = self._kind_path(kind_token)
        self.initialize()
        self._ensure_directory(directory)
        overview_path = directory / RegularityLevel.OVERVIEW.sidecar_filename
        abstract_path = directory / RegularityLevel.ABSTRACT.sidecar_filename
        self._atomic_write(overview_path, overview.encode("utf-8"))
        self._atomic_write(abstract_path, abstract.encode("utf-8"))
        return abstract_path, overview_path

    # ── 读 ──────────────────────────────────────────────────────────────────

    def read(self, address: AssociationAddress) -> AssociationDocument:
        path = self._record_path(address)
        payload = self._read_bytes(path, MAX_RECORD_BYTES)
        if payload is None:
            raise RegularityTreeError("association record does not exist")
        return decode(_utf8(payload, "association record"), expected_address=address)

    def exists(self, address: AssociationAddress) -> bool:
        return self._file_exists(self._record_path(address))

    def read_layer(self, kind_token: str, level: RegularityLevel) -> str:
        resolved = RegularityLevel(level)
        if resolved is RegularityLevel.DETAIL:
            raise RegularityTreeError("L2 is an addressed record, not a semantic layer")
        path = self._kind_path(kind_token) / resolved.sidecar_filename
        payload = self._read_bytes(path, MAX_LAYER_BYTES)
        if payload is None:
            raise RegularityTreeError("regularity semantic layer does not exist")
        return _utf8(payload, "regularity semantic layer")

    def layer_exists(self, kind_token: str, level: RegularityLevel) -> bool:
        return self._file_exists(self._kind_path(kind_token) / RegularityLevel(level).sidecar_filename)

    def read_day(self, kind_token: str, day: date) -> tuple[AssociationDocument, ...]:
        """这个候选在这一天的全部关联记录，按时刻升序。

        一天做了两次就是两条——关联与 occurrence 一一对应，不按天合并。
        """

        directory = self._day_path(kind_token, day)
        documents: list[AssociationDocument] = []
        for entry in self._children(directory):
            if not entry.is_file():
                raise RegularityTreeError("a regularity day may contain only association records")
            if entry.name == DONE_FILENAME:
                continue
            if atomic_temporary_destination(entry.name) is not None:
                # 崩溃遗留的原子写临时文件：按本模块自己的命名规则认出来、跳过，
                # 既不当成记录、也不报成损坏。
                continue
            if not entry.name.endswith(".md"):
                raise RegularityTreeError("a regularity day may contain only Markdown records")
            try:
                address = AssociationAddress.from_identity(kind_token, day, entry.name[:-3])
            except (TypeError, ValueError) as exc:
                # 叶名不规范（时间戳非规范、日期与目录不符）——报成本层的完整性错误，
                # 不让裸 ValueError 从一个把错误归一成 RegularityTreeError 的层里漏出去。
                raise RegularityTreeError("a regularity record has a non-canonical leaf name") from exc
            documents.append(self.read(address))
        # 按**时刻**排，不按文件名：同一本地日、不同 UTC 偏移时两者会分叉（09:30+0100 与
        # 15:30+0900 名字序与时刻序相反）。
        return tuple(sorted(documents, key=lambda item: item.address.started_at))

    def complete_day(
        self, kind_token: str, day: date, *, records: int, completed_at: datetime, version: str = ""
    ) -> Path:
        """把 (候选, 这一天) 标记为关联完成。

        **最后一步**：全部记录与两个侧车都落盘之后才调它。标记里记下当天应有几条，读回来对不上
        就说明中间掉了东西——这是一次关联作业的提交点。
        """

        if isinstance(records, bool) or not isinstance(records, int) or records <= 0:
            raise ValueError("a completed day must carry at least one record")
        if not isinstance(completed_at, datetime) or completed_at.utcoffset() is None:
            raise ValueError("completed_at must be a timezone-aware datetime")
        found = len(self.read_day(kind_token, day))
        if found != records:
            raise RegularityTreeError(f"day claims {records} association records but {found} are readable")
        directory = self._day_path(kind_token, day)
        # 提交前顺手清掉崩溃遗留的原子写临时文件：``read_day`` 认得出它们、会跳过，但没人清的话
        # 它们会一直堆着，而且**计入** MAX_DIRECTORY_ENTRIES，迟早把这一天的目录顶到列不出来。
        for entry in self._children(directory):
            if entry.is_file() and atomic_temporary_destination(entry.name) is not None:
                self._discard(directory / entry.name)
        path = directory / DONE_FILENAME
        payload = canonical_json(
            {
                "kind": _kind_identity(kind_token),
                "day": day.isoformat(),
                "records": records,
                "completed_at": completed_at.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z"),
                # 换提示词版本要全量重做，靠的就是这个字段：不记的话，换了版本已完成的天仍然
                # 算"做完了"，一天也不会重做。
                "association_version": version,
            }
        ).encode("utf-8")
        self._atomic_write(path, payload)
        return path

    def retain_only(self, kind_token: str, day: date, keep: frozenset[str]) -> tuple[str, ...]:
        """把这一天里不属于 ``keep``（叶名集合）的记录删掉，返回删了哪几个。

        重做一天时必须先清：``write`` 按地址覆写，本轮产出比上轮少时，上轮多出来的记录会永远留在
        目录里，让 ``complete_day`` 的条数核对再也过不去——那一天从此写不出完成标记。
        """

        directory = self._day_path(kind_token, day)
        removed: list[str] = []
        for entry in self._children(directory):
            if not entry.is_file() or entry.name in {DONE_FILENAME} or not entry.name.endswith(".md"):
                continue
            if entry.name[:-3] in keep:
                continue
            self._discard(directory / entry.name)
            removed.append(entry.name[:-3])
        return tuple(sorted(removed))

    def discard_completion(self, kind_token: str) -> tuple[date, ...]:
        """撤掉这个候选全部日子的完成标记，返回撤了哪几天。记录本身不动——重做会覆写它们。"""

        days: list[date] = []
        for day in self.day_directories(kind_token):
            marker = self._day_path(kind_token, day) / DONE_FILENAME
            if self._file_exists(marker):
                self._discard(marker)
                days.append(day)
        return tuple(days)

    def discard_layers(self, kind_token: str) -> None:
        """删掉这个候选的 L1 与 L0。重放要先清零，否则旧版本的情形会与新的混在一起累积。"""

        directory = self._kind_path(kind_token)
        for level in (RegularityLevel.OVERVIEW, RegularityLevel.ABSTRACT):
            self._discard(directory / level.sidecar_filename)

    def day_directories(self, kind_token: str) -> tuple[date, ...]:
        """这个候选下有哪些日目录（不看完成标记）。前提投影按它取材——记录先落盘、标记最后写。"""

        directory = self._kind_path(kind_token)
        days: list[date] = []
        for year in self._directories(directory):
            if not _is_digits(year.name, 4):
                continue
            for month in self._directories(year):
                if not _is_digits(month.name, 2):
                    continue
                for day in self._directories(month):
                    if not _is_digits(day.name, 2):
                        continue
                    try:
                        days.append(date(int(year.name), int(month.name), int(day.name)))
                    except ValueError:
                        continue
        return tuple(sorted(days))

    def days_for(self, kind_token: str, *, version: str | None = None) -> frozenset[date]:
        """这个候选有哪些天**关联完成了**。

        给了 ``version`` 就只算那个版本做的——换提示词版本之后要全量重做时，靠的就是这一条：
        标记里不记版本的话，换了版本已完成的天仍然算"做完了"，一天也不会重做。
        """

        """这个候选已经关联**完成**的日期——``scene.backlog.AssociatedDays`` 的生产实现。

        只认完成标记。日目录在工作开始前就存在，拿它当判据会让中断过的那一天被永久跳过。
        """

        directory = self._kind_path(kind_token)
        days: set[date] = set()
        for year in self._directories(directory):
            if not _is_digits(year.name, 4):
                continue
            for month in self._directories(year):
                if not _is_digits(month.name, 2):
                    continue
                for day in self._directories(month):
                    if not _is_digits(day.name, 2):
                        continue
                    try:
                        value = date(int(year.name), int(month.name), int(day.name))
                    except ValueError:
                        continue
                    marker = self._read_bytes(day / DONE_FILENAME, MAX_LAYER_BYTES)
                    if marker is None:
                        continue
                    if version is not None and _marker_version(marker) != version:
                        continue
                    days.add(value)
        return frozenset(days)

    def list_kinds(self) -> tuple[str, ...]:
        return tuple(sorted(entry.name for entry in self._directories(self._inside(Path(KINDS_SEGMENT)))))

    # ── 路径 ────────────────────────────────────────────────────────────────

    def _kind_path(self, kind_token: str) -> Path:
        return self._inside(Path(*KindDirectory.for_kind(kind_token).parts))

    def _day_path(self, kind_token: str, day: date) -> Path:
        return self._inside(Path(*KindDirectory.for_day(kind_token, day).parts))

    def _record_path(self, address: AssociationAddress) -> Path:
        if not isinstance(address, AssociationAddress):
            raise TypeError("address must be an AssociationAddress")
        return self._day_path(address.kind_token, address.occurred_on) / f"{address.identity_name}.md"

    def _inside(self, relative: Path) -> Path:
        """只**校验**，返回未解析的路径。

        ``durable_io`` 的安全检查是**词法**的（逐段看 ``is_symlink()``）。返回 ``resolve()`` 之后的
        路径等于替它把符号链接洗白了——每一段都成了真目录，检查恒通过，写入会静默落到链接指向
        的地方（实测：``kinds/打球 → kinds/real`` 会让两个候选的历史被合并）。``SceneTree`` 的
        同名函数正是只校验不返回，所以它拦得住。
        """

        candidate = (self.root / relative).resolve(strict=False)
        if candidate != self.root and self.root not in candidate.parents:
            raise RegularityTreeError("regularity path cannot be used outside its tree root")
        return self.root / relative

    # ── 文件系统 ────────────────────────────────────────────────────────────

    def _discard(self, path: Path) -> None:
        try:
            durable_unlink(path, artifact_root=self.root)
        except FileNotFoundError:
            return
        except DurablePathIntegrityError as exc:
            raise RegularityTreeError("regularity leftover cannot be removed safely") from exc

    def _atomic_write(self, path: Path, payload: bytes) -> None:
        try:
            atomic_replace_bytes(path, payload, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise RegularityTreeError("regularity file cannot be written safely") from exc

    def _read_bytes(self, path: Path, maximum: int) -> bytes | None:
        try:
            if not regular_file_exists(path, artifact_root=self.root):
                return None
            return read_regular_bytes(path, artifact_root=self.root, max_bytes=maximum)
        except FileNotFoundError:
            return None
        except DurablePathIntegrityError as exc:
            raise RegularityTreeError("regularity file cannot be read safely") from exc

    def _file_exists(self, path: Path) -> bool:
        try:
            return regular_file_exists(path, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise RegularityTreeError("regularity file cannot be inspected safely") from exc

    def _children(self, directory: Path) -> tuple[DurableDirectoryEntry, ...]:
        try:
            if not real_directory_exists(directory, artifact_root=self.root):
                return ()
            return tuple(list_real_directory(directory, artifact_root=self.root, max_entries=MAX_DIRECTORY_ENTRIES))
        except DurablePathIntegrityError as exc:
            raise RegularityTreeError("regularity directory cannot be listed safely") from exc

    def _directories(self, directory: Path) -> tuple[Path, ...]:
        return tuple(directory / entry.name for entry in self._children(directory) if entry.is_dir())

    def _ensure_directory(self, directory: Path) -> None:
        self._inside(directory)
        try:
            ensure_real_directory(directory, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise RegularityTreeError("regularity directory cannot be created safely") from exc


def _marker_version(payload: bytes) -> str | None:
    """完成标记里记的关联版本；老标记（没有这个字段）返回空串，不是 None。"""

    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return str(raw.get("association_version", "")) if isinstance(raw, dict) else None


def _is_digits(value: str, width: int) -> bool:
    return len(value) == width and is_ascii_digits(value)


def _utf8(payload: bytes, label: str) -> str:
    """损坏的字节要归一成本层的错误，不能漏出裸 ``UnicodeDecodeError``。"""

    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RegularityTreeError(f"{label} is not valid UTF-8") from exc


__all__ = [
    "DONE_FILENAME",
    "MAX_DIRECTORY_ENTRIES",
    "MAX_LAYER_BYTES",
    "MAX_RECORD_BYTES",
    "RegularityTree",
    "RegularityTreeError",
]
