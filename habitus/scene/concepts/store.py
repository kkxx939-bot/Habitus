"""``scene/concepts/`` 这一支的存储：一个概念一份 ``<身份>.md``，按身份覆写。

没有完成标记、没有代：概念不是按天产出的作业，写一份就是一份。另有一个点文件（列目录时归为噪音跳过）：
``.vocabulary-version`` 记"同步词表同步到了哪一版"，换代据此判断这一版是否已同步过。
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from habitus.scene.codec import SceneRecordError
from habitus.scene.concepts.document import decode, encode
from habitus.scene.concepts.model import ConceptDefinition, ConceptError, ConceptSet, concept_identity
from habitus.scene.storage import SceneStore

CONCEPTS_SEGMENT = "concepts"
MAX_RECORD_BYTES = 64 * 1024
SYNCED_FILENAME = ".vocabulary-version"


class ConceptStoreError(ValueError):
    """概念存储的路径逃逸、损坏记录，或违反层级约束的写入。"""


class ConceptStore(SceneStore):
    error_type: ClassVar[type[ValueError]] = ConceptStoreError

    def __init__(self, scene_root: str | Path, **options: int) -> None:
        super().__init__(scene_root, **options)
        self.directory = self._inside(Path(CONCEPTS_SEGMENT))

    def initialize(self) -> Path:
        self._ensure_directory(self.directory)
        return self.directory

    # ── 写 ──────────────────────────────────────────────────────────────────

    def write(self, definition: ConceptDefinition) -> Path:
        """原子写、回读比对。同身份覆写（同步词表改类名、改判据时就是这样更新行为概念的）。"""

        self.initialize()
        path = self.path_for(definition.identity)
        self._atomic_write(path, encode(definition).encode("utf-8"), maximum=MAX_RECORD_BYTES)
        return path

    def write_synced_version(self, version: int) -> None:
        """同步词表做完之后记下同步到的版本（最后一步：概念写完才记）。"""

        if version < 0:
            raise ConceptStoreError("vocabulary version must be a non-negative integer")
        self.initialize()
        self._atomic_write(self.directory / SYNCED_FILENAME, f"{version}\n".encode(), maximum=64)

    # ── 读 ──────────────────────────────────────────────────────────────────

    def synced_version(self) -> int:
        """上次同步到的词表版本；从没同步过是 0。"""

        payload = self._read_bytes(self.directory / SYNCED_FILENAME, 64)
        if payload is None:
            return 0
        text = payload.decode("utf-8").strip()
        if not text.isdigit():
            raise ConceptStoreError(f"{SYNCED_FILENAME} is corrupt: {text!r}")
        return int(text)

    def read(self, name: str) -> ConceptDefinition:
        identity = self._identity(name)
        payload = self._read_bytes(self.path_for(identity), MAX_RECORD_BYTES)
        if payload is None:
            raise ConceptStoreError(f"concept {name!r} does not exist")
        return self._decode(payload, identity)

    def exists(self, name: str) -> bool:
        return self._file_exists(self.path_for(self._identity(name)))

    def identities(self) -> tuple[str, ...]:
        """盘上的概念身份。叶名必须已是规范身份——否则在大小写不敏感的文件系统上会静默互认。"""

        names: list[str] = []
        records, _noise = self._content_files(self.directory)
        for entry in records:
            if not entry.name.endswith(".md"):
                raise ConceptStoreError(f"unexpected entry in the concept directory: {entry.name!r}")
            leaf = entry.name[:-3]
            if self._identity(leaf) != leaf:
                raise ConceptStoreError(f"concept file {entry.name!r} does not carry a canonical identity")
            names.append(leaf)
        return tuple(sorted(names))

    def read_all(self) -> ConceptSet:
        """整支读成一个自洽的概念集；不自洽（撞名、一个类两个行为概念）在这里报出来。"""

        try:
            return ConceptSet(self.read(identity) for identity in self.identities())
        except ConceptError as exc:
            raise ConceptStoreError(f"stored concepts are not a consistent set: {exc}") from exc

    def path_for(self, name: str) -> Path:
        return self._inside(Path(CONCEPTS_SEGMENT, f"{self._identity(name)}.md"))

    # ── 内部 ────────────────────────────────────────────────────────────────

    @staticmethod
    def _identity(name: object) -> str:
        try:
            return concept_identity(name)
        except ConceptError as exc:
            raise ConceptStoreError(str(exc)) from exc

    def _decode(self, payload: bytes, identity: str) -> ConceptDefinition:
        try:
            return decode(payload.decode("utf-8"), expected_identity=identity)
        except (UnicodeDecodeError, SceneRecordError) as exc:
            raise ConceptStoreError(f"concept record {identity!r} is corrupt: {exc}") from exc


__all__ = [
    "CONCEPTS_SEGMENT",
    "MAX_RECORD_BYTES",
    "SYNCED_FILENAME",
    "ConceptStore",
    "ConceptStoreError",
]
