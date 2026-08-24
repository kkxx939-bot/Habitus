"""已发布的树代：物化、校验、翻指针、留代。

发布是**两阶段**的：先把这一代整棵写进自己的目录并回读校验，再原子翻转 ``current.json``
指针。任一阶段失败都不动指针，旧代继续服务——旧实现按状态逐个成代、独立激活，读侧因此拿到
过混代数据并静默失真（见 ``TODO(PRED-DOWNSTREAM-001)`` 的发布协议一节）。

读侧的对偶纪律是**一次查询钉住一代**：``load`` 返回的是一棵完整的树，不提供"按需取一个格子"
的接口，也就不存在半路换代的机会。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from foundation.integrity import canonical_json, text_digest
from infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_replace_bytes,
    durable_rmdir,
    durable_unlink,
    ensure_real_directory,
    list_real_directory,
    read_regular_bytes,
    regular_file_exists,
)
from prediction import codec
from prediction.errors import PredictionTreeError, PredictionTreeStoreError
from prediction.model import PredictionTree

GENERATIONS_DIRECTORY = "generations"
POINTER_FILENAME = "current.json"
TREE_FILENAME = "tree.json"
MAX_TREE_BYTES = 64 * 1024 * 1024
MAX_POINTER_BYTES = 4096
_MAX_GENERATION_ENTRIES = 4096


@dataclass(frozen=True)
class PublishedGeneration:
    """指针的内容：哪一代在服务，以及验证它所需的一切。

    ``active()`` 会把磁盘上任意内容塞进这里，所以字段校验必须在构造时做——损坏的指针里一个
    非字符串的 ``generation`` 否则会一路带到路径拼接才以 ``TypeError`` 炸出来。
    """

    generation: str
    digest: str
    config_digest: str
    published_at: datetime

    def __post_init__(self) -> None:
        for name in ("generation", "digest", "config_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise PredictionTreeStoreError(f"published generation {name} must be non-empty text")
        if not isinstance(self.published_at, datetime) or self.published_at.utcoffset() is None:
            raise PredictionTreeStoreError("published generation timestamp must be timezone-aware")


class PredictionTreeStore:
    """一个目录下的多代树；同一时刻只有一代对读侧可见。"""

    def __init__(self, root: str | Path, *, retained_generations: int) -> None:
        if isinstance(retained_generations, bool) or not isinstance(retained_generations, int):
            raise PredictionTreeError("retained_generations must be an integer")
        if retained_generations < 1:
            raise PredictionTreeError("at least one generation must be retained")
        requested = Path(root).expanduser().absolute()
        if requested.is_symlink():
            raise PredictionTreeStoreError("prediction tree root cannot be a symbolic link")
        self.root = requested.resolve(strict=False)
        self.retained_generations = retained_generations

    def initialize(self) -> Path:
        self._ensure_directory(self.root)
        self._ensure_directory(self.root / GENERATIONS_DIRECTORY)
        return self.root

    # --- 发布 -----------------------------------------------------------------------------

    def publish(self, tree: PredictionTree) -> PublishedGeneration:
        """物化并激活一代；返回现在正在服务的那一代。"""

        payload = canonical_json(codec.encode(tree)).encode("utf-8")
        if len(payload) > MAX_TREE_BYTES:
            raise PredictionTreeStoreError("prediction tree exceeds the publishable size bound")
        digest = text_digest(payload.decode("utf-8"))
        generation = self._generation_name(tree, digest)
        self.initialize()
        self._materialize(generation, payload, digest)
        published = PublishedGeneration(
            generation=generation,
            digest=digest,
            config_digest=tree.config_digest,
            published_at=tree.built_at.astimezone(timezone.utc),
        )
        self._activate(published)
        self._prune(keep=generation)
        return published

    def _materialize(self, generation: str, payload: bytes, digest: str) -> None:
        """第一阶段：写这一代的字节，并**回读校验**——校验不过就当这一代没发生过。"""

        directory = self._generation_directory(generation)
        self._ensure_directory(directory)
        target = directory / TREE_FILENAME
        try:
            atomic_create_bytes(target, payload, artifact_root=self.root)
        except ImmutableArtifactConflictError as exc:
            # 同名代已经绑定了不同内容：代名含内容摘要，撞上说明存储被外部改过。
            raise PredictionTreeStoreError("prediction tree generation is already bound") from exc
        except DurablePathIntegrityError as exc:
            raise PredictionTreeStoreError("prediction tree generation cannot be written") from exc
        if text_digest(self._read(target, MAX_TREE_BYTES).decode("utf-8")) != digest:
            raise PredictionTreeStoreError("prediction tree generation failed read-back verification")

    def _activate(self, published: PublishedGeneration) -> None:
        """第二阶段：原子翻指针。到这里为止旧代一直在服务。"""

        pointer = canonical_json(
            {
                "generation": published.generation,
                "digest": published.digest,
                "config_digest": published.config_digest,
                "published_at": published.published_at.isoformat(),
            }
        ).encode("utf-8")
        try:
            atomic_replace_bytes(self.root / POINTER_FILENAME, pointer, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise PredictionTreeStoreError("prediction tree pointer cannot be replaced") from exc

    # --- 读取 -----------------------------------------------------------------------------

    def active(self) -> PublishedGeneration | None:
        """当前指针；还没发布过任何一代时为 None。"""

        path = self.root / POINTER_FILENAME
        if not self._exists(path):
            return None
        try:
            raw = json.loads(self._read(path, MAX_POINTER_BYTES).decode("utf-8"))
            if not isinstance(raw, dict):
                raise PredictionTreeStoreError("prediction tree pointer is malformed")
            return PublishedGeneration(
                generation=raw["generation"],
                digest=raw["digest"],
                config_digest=raw["config_digest"],
                published_at=datetime.fromisoformat(raw["published_at"]).astimezone(timezone.utc),
            )
        except (KeyError, TypeError, ValueError) as exc:
            # json.loads 也在这里面：损坏的字节必须统一成本层的错误，不能漏出 JSONDecodeError。
            raise PredictionTreeStoreError("prediction tree pointer is malformed") from exc

    def load(self) -> PredictionTree | None:
        """读出正在服务的那一代整棵树；没有发布过则为 None。"""

        published = self.active()
        if published is None:
            return None
        return self.load_generation(published.generation, expected_digest=published.digest)

    def load_generation(self, generation: str, *, expected_digest: str | None = None) -> PredictionTree:
        path = self._generation_directory(generation) / TREE_FILENAME
        try:
            raw = self._read(path, MAX_TREE_BYTES).decode("utf-8")
        except FileNotFoundError as exc:
            raise PredictionTreeStoreError("prediction tree generation is missing") from exc
        if expected_digest is not None and text_digest(raw) != expected_digest:
            raise PredictionTreeStoreError("prediction tree generation does not match its pointer")
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise PredictionTreeStoreError("prediction tree generation is not valid JSON") from exc
        return codec.decode(payload)

    def generations(self) -> tuple[str, ...]:
        """已物化的代，按代名升序即时间升序。"""

        try:
            entries = list_real_directory(
                self.root / GENERATIONS_DIRECTORY,
                artifact_root=self.root,
                max_entries=_MAX_GENERATION_ENTRIES,
            )
        except DurablePathIntegrityError as exc:
            raise PredictionTreeStoreError("prediction tree generations cannot be listed") from exc
        return tuple(entry.name for entry in entries if entry.is_dir())

    # --- 留代 -----------------------------------------------------------------------------

    def _prune(self, *, keep: str) -> bool:
        """只在指针已经翻过之后删旧代，且永不删正在服务的那一代。

        保护的是**指针当前指着的那一代**，不只是本次发布的那一代：两个进程并发发布时，
        B 走完全程之后 A 才翻指针，只保 ``keep`` 会让 B 把 A 正要激活的那一代删掉，读侧从此
        每次 ``load()`` 都撞上悬空指针（实测复现过）。

        **失败被吞掉**并返回 False：到这一步新代已经在服务了，垃圾回收没做干净是运维问题，
        不是发布失败。让它抛出去会把一次成功的发布记成 FAILURE，并在健康面留下永久的
        ``last_error``（夜批一天一拍，那条错误会挂到进程重启）。
        """

        try:
            active = self.active()
            protected = {keep} if active is None else {keep, active.generation}
            existing = self.generations()
            surplus = existing[: max(0, len(existing) - self.retained_generations)]
            for generation in surplus:
                if generation in protected:
                    continue
                directory = self._generation_directory(generation)
                durable_unlink(directory / TREE_FILENAME, artifact_root=self.root)
                durable_rmdir(directory, artifact_root=self.root)
        except (PredictionTreeStoreError, DurablePathIntegrityError, OSError):
            return False
        return True

    # --- 路径与 IO ------------------------------------------------------------------------

    @staticmethod
    def _generation_name(tree: PredictionTree, digest: str) -> str:
        """代名 = 构建时刻（UTC）+ 内容摘要：字典序即时间序，同时把内容身份钉在名字里。

        这里必须自己 ``astimezone(utc)``：``PredictionTree`` 只要求带时区，不要求是 UTC
        （UTC 化发生在 builder 里）。一棵从别处解码回来、还带着 +08:00 的树若直接格式化，
        名字上挂着 ``Z`` 却不是 UTC，字典序就不再等于时间序，留代会删错。
        """

        stamp = tree.built_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        return f"{stamp}Z-{digest[:12]}"

    def _generation_directory(self, generation: str) -> Path:
        if not generation or "/" in generation or generation in {".", ".."}:
            raise PredictionTreeError("generation name is not a single path segment")
        return self.root / GENERATIONS_DIRECTORY / generation

    def _ensure_directory(self, path: Path) -> None:
        try:
            ensure_real_directory(path, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise PredictionTreeStoreError("prediction tree directory cannot be created safely") from exc

    def _exists(self, path: Path) -> bool:
        try:
            return regular_file_exists(path, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise PredictionTreeStoreError("prediction tree file cannot be inspected safely") from exc

    def _read(self, path: Path, max_bytes: int) -> bytes:
        try:
            return read_regular_bytes(path, artifact_root=self.root, max_bytes=max_bytes)
        except DurablePathIntegrityError as exc:
            raise PredictionTreeStoreError("prediction tree file cannot be read safely") from exc


__all__ = [
    "GENERATIONS_DIRECTORY",
    "MAX_TREE_BYTES",
    "POINTER_FILENAME",
    "TREE_FILENAME",
    "PredictionTreeStore",
    "PublishedGeneration",
]
