"""预测层各 store 共用的 JSON 小文件读写：账本、单元历史的每刻记录、出口的传输状态都是"一个目录一批规范 JSON 小文件"，
读写与坏文件的规矩只在这里写一遍；每代的派生物（机会表、单元历史快照）按代整份落盘，列代、读字节也走这里的目录口。

住在组合根：``foresight`` 不许 import infrastructure，形状与编解码在各自的领域包，这里只管目录与字节。

- 记录文件：目录里以 ``.json`` 结尾、不以 ``.`` / ``_`` 开头的普通文件（``_`` 开头的是标记与水位，``.`` 开头的是写到一半
  的临时文件）；别的名字不是记录，跳过不报。
- 坏文件：一个坏文件不许让整层停。按目录读回时读不出来的（字节读不了、不是 JSON 对象、形状或版本不对）记下路径、跳过，
  调用方把条数带进观测（``take_unreadable``）；只想知道"自己这一次读到了哪些坏文件"的调用方传 ``sink=`` 自己的列表，
  不经共享的那一份。点读（``read``）同一个口径：读不出来记下、当没有。盘本身读不了（读到一半的 IO 错）不是坏文件，
  ``OSError`` 照旧抛给调用方，下一拍再来。
- 写一次不改（``create``）：同名同内容幂等；同名不同内容是损坏、抛——只有盘上那一份本身读不出来时例外：那是坏文件，
  记下、不覆盖、不抛（重写一份读得出的会把坏文件里原来是什么永远盖掉）。
- 账本的按目录读（``LedgerFiles.cached_decoded``）：每次现列目录、现取每个文件的身份，身份没变的不再打开，用内存里解码好的
  那一份；缓存只省读盘，读到的永远是盘上此刻的样子。
"""

from __future__ import annotations

import json
import os
import stat
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from habitus.config.foresight import ForesightStoreConfig
from habitus.foresight import ForesightError
from habitus.infrastructure.store.filesystem import (
    DurableDirectoryEntry,
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_replace_bytes,
    durable_unlink,
    ensure_real_directory,
    list_real_directory,
    read_regular_bytes,
    real_directory_exists,
)


class JsonFiles:
    """一棵目录树下的 JSON 小文件。``label`` 只进报错与观测（"ledger"、"unit moment"、"outlet"）。"""

    def __init__(self, root: str | Path, *, label: str, max_bytes: int, max_entries: int, max_unreadable: int) -> None:
        for name, value in (("max_bytes", max_bytes), ("max_entries", max_entries), ("max_unreadable", max_unreadable)):
            if value < 1:
                raise ForesightError(f"{label} {name} must be a positive integer")
        self.root = Path(root).expanduser().resolve(strict=False)
        self.label = label
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.max_unreadable = max_unreadable
        self._unreadable: list[str] = []
        #: 换代（线程里跑）与预测拍共用同一个实例：共享的坏文件清单加锁。
        self._lock = threading.Lock()

    # ── 目录 ────────────────────────────────────────────────────────────────

    def entries(self, directory: Path) -> tuple[DurableDirectoryEntry, ...]:
        """目录的直接条目（不在就是空）；目录本身不安全或条目超过上限是整层的错，抛。"""

        try:
            if not real_directory_exists(directory, artifact_root=self.root):
                return ()
            return tuple(list_real_directory(directory, artifact_root=self.root, max_entries=self.max_entries))
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"{self.label} directory is invalid: {exc}") from exc

    def names(self, directory: Path) -> tuple[str, ...]:
        """目录里的记录文件名（排好序）。"""

        return tuple(sorted(entry.name for entry in self.entries(directory) if is_record(entry)))

    # ── 读 ──────────────────────────────────────────────────────────────────

    def load(self, path: Path) -> dict[str, Any]:
        """一份记录的 JSON 对象；读不出来、不是对象抛 ``ForesightError``（坏文件），读到一半的 IO 错照旧抛 ``OSError``。"""

        try:
            raw = json.loads(read_regular_bytes(path, artifact_root=self.root, max_bytes=self.max_bytes))
        except (DurablePathIntegrityError, ValueError) as exc:
            raise ForesightError(f"{self.label} record {path.name} cannot be read: {exc}") from exc
        if not isinstance(raw, dict):
            raise ForesightError(f"{self.label} record {path.name} is not an object")
        return raw

    def read[T](self, path: Path, decode: Callable[[dict[str, Any]], T], sink: list[str] | None = None) -> T | None:
        """点读一份：不在是 None；坏的记下路径、当没有。"""

        if not path.is_file():
            return None
        try:
            return decode(self.load(path))
        except ForesightError:
            self.note(path, sink)
            return None

    def decoded[T](
        self, directory: Path, decode: Callable[[dict[str, Any]], T], sink: list[str] | None = None
    ) -> Iterator[tuple[str, T]]:
        """一个目录里的记录逐条解码（按文件名排）：（文件名去掉 ``.json``, 解码结果）；坏的记下路径、跳过。"""

        for name in self.names(directory):
            path = directory / name
            try:
                yield name.removesuffix(".json"), decode(self.load(path))
            except ForesightError:
                self.note(path, sink)

    def readable[T](self, path: Path, decode: Callable[[dict[str, Any]], T]) -> bool:
        """盘上这一份读得出来（不记坏文件：只给写的一方判"同名不同内容"是损坏还是坏文件）。"""

        try:
            decode(self.load(path))
        except ForesightError:
            return False
        return True

    # ── 写 ──────────────────────────────────────────────────────────────────

    def encode(self, payload: Mapping[str, Any], identity: str) -> bytes:
        """规范编码并查大小：写入就查，超限的不落盘。"""

        encoded = encode_json(payload)
        if len(encoded) > self.max_bytes:
            raise ForesightError(
                f"{self.label} record {identity} exceeds {self.max_bytes} bytes ({len(encoded)}); not written"
            )
        return encoded

    def create[T](self, path: Path, payload: Mapping[str, Any], decode: Callable[[dict[str, Any]], T]) -> bool:
        """写一次不改：新写返回 True，同内容重写返回 False；盘上同名的一份读不出来是坏文件——记下、不覆盖、返回 False；
        读得出来而内容不同是损坏，抛。"""

        identity = path.stem
        encoded = self.encode(payload, identity)
        try:
            return atomic_create_bytes(path, encoded, artifact_root=self.root)
        except ImmutableArtifactConflictError as exc:
            if path.is_file() and not self.readable(path, decode):
                self.note(path, None)
                return False
            raise ForesightError(f"{self.label} record {identity} already exists with different content") from exc
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"{self.label} path is not writable: {exc}") from exc

    def replace(self, path: Path, payload: Mapping[str, Any]) -> None:
        """整份替换（可变的小文件：水位、传输状态、每刻记录）。"""

        encoded = self.encode(payload, path.stem)
        try:
            ensure_real_directory(path.parent, artifact_root=self.root)
            atomic_replace_bytes(path, encoded, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"{self.label} path is not writable: {exc}") from exc

    def remove(self, path: Path) -> bool:
        try:
            return durable_unlink(path, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"{self.label} path is not removable: {exc}") from exc

    # ── 坏文件 ──────────────────────────────────────────────────────────────

    def note(self, path: Path, sink: list[str] | None) -> None:
        """记下一份坏文件（同一路径只记一次，最多 ``max_unreadable`` 条）。"""

        target = sink
        with self._lock:
            if target is None:
                target = self._unreadable
            if str(path) not in target and len(target) < self.max_unreadable:
                target.append(str(path))

    def take_unreadable(self) -> tuple[str, ...]:
        """上次 ``take`` 以来跳过的坏文件（路径），取走即清。"""

        with self._lock:
            found = tuple(self._unreadable)
            self._unreadable.clear()
        return found


#: 文件身份：（inode, 大小, 修改时刻纳秒, 状态变更时刻纳秒）。原子替换换 inode，就地改写换大小或修改时刻；删掉后同名重写
#: 可能重用 inode、修改时刻也能被改回去，状态变更时刻再挡一道。多重读一次只影响快慢，读到旧内容才是错。
_FileIdentity = tuple[int, int, int, int]


class LedgerFiles(JsonFiles):
    """账本的记录文件：写一次不改（``create``），所以文件身份没变，解码结果就没变。

    缓存的键是（路径, 解码函数），同一个文件被不同解码函数读时互不相干；交出去的都是不可变的账本记录（``foresight.ledger.model``
    的 ``frozen`` 数据类，字段只有标量、元组与日期时间），调用方共享同一份也改不动它。读不出来的不进缓存：每次都重读、都记下。
    预测拍与出口线程共用一个实例：缓存的增删查加锁，读文件在锁外。"""

    def __init__(self, root: str | Path, limits: ForesightStoreConfig) -> None:
        super().__init__(
            root,
            label="ledger",
            max_bytes=limits.record_max_bytes,
            max_entries=limits.directory_max_entries,
            max_unreadable=limits.unreadable_max_paths,
        )
        self._cache_capacity = limits.read_cache_max_records
        self._cache: OrderedDict[tuple[str, Callable[[dict[str, Any]], Any]], tuple[_FileIdentity, Any]] = OrderedDict()

    def cached_decoded[T](
        self, directory: Path, decode: Callable[[dict[str, Any]], T], sink: list[str] | None = None
    ) -> Iterator[tuple[str, T]]:
        """同 ``decoded``（按文件名排，坏的记下路径、跳过），身份没变的文件不再打开。"""

        for name in self.names(directory):
            path = directory / name
            key = (str(path), decode)
            identity = _file_identity(path)
            held = self._recall(key, identity)
            if held is not None:
                yield name.removesuffix(".json"), held[1]
                continue
            try:
                item = decode(self.load(path))
            except ForesightError:
                self.note(path, sink)
                continue
            if identity is not None:
                self._remember(key, identity, item)
            yield name.removesuffix(".json"), item

    def _recall(
        self, key: tuple[str, Callable[[dict[str, Any]], Any]], identity: _FileIdentity | None
    ) -> tuple[_FileIdentity, Any] | None:
        """身份相同的那一份（顺手排到最近用过）；身份变了的旧内容当场丢掉。"""

        with self._lock:
            held = self._cache.get(key)
            if held is None:
                return None
            if held[0] != identity:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return held

    def _remember(
        self, key: tuple[str, Callable[[dict[str, Any]], Any]], identity: _FileIdentity, item: object
    ) -> None:
        with self._lock:
            self._cache[key] = (identity, item)
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_capacity:
                self._cache.popitem(last=False)


def _file_identity(path: Path) -> _FileIdentity | None:
    """当下这个目录项的身份（不跟随符号链接）；不在或不是普通文件是 None：不用缓存，交给 ``load`` 按原来的规矩报。"""

    try:
        found = os.lstat(path)
    except OSError:
        return None
    if not stat.S_ISREG(found.st_mode):
        return None
    return (found.st_ino, found.st_size, found.st_mtime_ns, found.st_ctime_ns)


def is_record(entry: DurableDirectoryEntry) -> bool:
    """记录文件：``*.json``、不以 ``.`` / ``_`` 开头的普通文件。"""

    return entry.is_file() and entry.name.endswith(".json") and not entry.name.startswith((".", "_"))


def encode_json(payload: Mapping[str, Any]) -> bytes:
    """规范 JSON（键排序、紧凑、UTF-8、末尾换行）。"""

    return (json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


__all__ = ["JsonFiles", "LedgerFiles", "encode_json", "is_record"]
