"""``scene/concepts/`` 这一支的存储：一个概念一份 ``<身份>.md``，按身份覆写。

概念是被引用的词汇（假设、命中都指着它），所以这里守两条结构约束：**上级必须已经存在**（先写父再写子；
引用一个不存在的名字等于把词汇层的洞留给下游发现）、**不成环**（覆写一个概念的上级时，沿新上级往上走
不能走回自己）。层级最深几层按设计还是待定，这里不设上限。

没有完成标记、没有代：概念不是按天产出的作业，写一份就是一份。旁册 ``.vectors.json`` 与概念文件同目录，
作为点文件在列目录时归为噪音跳过。
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from habitus.scene.codec import SceneRecordError
from habitus.scene.concepts.document import decode, encode
from habitus.scene.concepts.model import ConceptDefinition, ConceptError, ConceptSet, concept_identity
from habitus.scene.storage import SceneStore

CONCEPTS_SEGMENT = "concepts"
CONCEPT_VECTORS_FILENAME = ".vectors.json"
MAX_RECORD_BYTES = 64 * 1024
#: 沿上级往上走的步数上限：磁盘上被手改出环时，这一条让 ``write`` 停下来报错而不是转圈。
_MAX_ANCESTOR_WALK = 64


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
        """原子写、回读比对；上级必须已存在且不构成环。同身份覆写。"""

        if not isinstance(definition, ConceptDefinition):
            raise TypeError("definition must be a ConceptDefinition")
        parent = definition.parent_identity
        if parent is not None:
            if not self.exists(parent):
                raise ConceptStoreError(f"parent concept {definition.parent!r} must be written before its child")
            self._require_no_cycle(definition.identity, parent)
        self.initialize()
        path = self.path_for(definition.identity)
        self._atomic_write(path, encode(definition).encode("utf-8"), maximum=MAX_RECORD_BYTES)
        return path

    # ── 读 ──────────────────────────────────────────────────────────────────

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
        """整支读成一个自洽的概念集；上级缺失或成环在这里报出来。"""

        try:
            return ConceptSet(self.read(identity) for identity in self.identities())
        except ConceptError as exc:
            raise ConceptStoreError(f"stored concepts are not a consistent set: {exc}") from exc

    def path_for(self, name: str) -> Path:
        return self._inside(Path(CONCEPTS_SEGMENT, f"{self._identity(name)}.md"))

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _require_no_cycle(self, identity: str, parent: str) -> None:
        current: str | None = parent
        for _ in range(_MAX_ANCESTOR_WALK):
            if current is None:
                return
            if current == identity:
                raise ConceptStoreError("concept hierarchy would form a cycle")
            current = self.read(current).parent_identity
        raise ConceptStoreError("concept hierarchy is deeper than the store is willing to walk")

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


__all__ = ["CONCEPT_VECTORS_FILENAME", "CONCEPTS_SEGMENT", "MAX_RECORD_BYTES", "ConceptStore", "ConceptStoreError"]
