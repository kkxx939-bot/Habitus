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
from datetime import UTC, datetime
from pathlib import Path

from foundation.integrity import bytes_digest, canonical_json
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
# TODO(PRED-STORE-002): 一代树整棵进出内存，``MAX_TREE_BYTES`` 这个上限的真实含义是读一次
# 约 0.5 GiB 常驻。
# - 现状：``publish`` 与 ``load_generation`` 都把整棵树当**一个** JSON 文档处理，中途必须同时
#   持有字节、文本与解析出的对象图。实测（4.67 MiB 的合成 payload，形状按七天真实树的游程
#   曲线造，tracemalloc）：读一代峰值 42.5 MiB = **9.1× payload**，其中约 8× 是解析出来的
#   dict/list/float 对象图本身，常驻到这棵树被丢弃为止。
# - **已经试过、无效的方向**（记在这里免得有人再试一遍）：把摘要与解析都留在字节上。
#   ``json.loads`` 原生吃 bytes，但它内部照样解出一份 str，于是 bytes 与 str 同时在场，
#   峰值反而从 9.12× 升到 10.12×。摘要走字节只在**不解析**的那一步真省——发布时的回读校验
#   因此从 4.96× 降到 1.00×，但那一步不是 ``publish`` 的峰值（峰值在 ``codec.encode`` +
#   ``canonical_json`` 那一段）。**峰值只有分片或流式解析能动。**
# - 具体场景：WP4 折叠粒度（约 350–420 条/天）下一年约 33 MiB，读一次就是约 0.3 GiB；若折叠
#   粒度回退到 v15 那种"任何可命名的动作"（约 1,300 条/天），195 天就撞满 64 MiB，读一次
#   约 0.58 GiB。夜批与查询进程同时各持一棵就翻倍。
# - 影响大小：中。当前真实树才 4.66 MiB（读一次约 42 MiB），离危险还很远；这是随历史增长的
#   欠账，不是现在的故障。
# - 改造方案：一代目录内**按周几分片 + 惰性加载**——
#     generations/<代名>/manifest.json（配置指纹、schema 版本、动作表、每片摘要）
#                        weekday-0.json … weekday-6.json（曲线与格子，占体积九成）
#                        edges.json / parallels.json / recurrence.json / baselines.json
#   指针只存 manifest 的摘要，两阶段发布不变（先写全部分片与 manifest 并逐片回读校验，再翻
#   指针）。查询模式正好对得上：``query.slot_outlook`` 与 ``marginal_at`` 都只要一个周几那
#   一片，一次查询加载 1/7；边与复发跨周几，各自单独一片。"一次查询钉住一代"不破——代名含
#   内容摘要、分片不可覆写，惰性句柄只从这一代目录取片，中途翻指针不影响已打开的句柄。
# - 代价：``PredictionTree`` 现在是全内存的 dataclass，``query`` 的 17 个函数签名全是
#   ``(tree, ...)``；惰性化要引入一个懒视图并扫过整个查询层与它的测试。这是本条推迟的主因。
# - 时机：**等预测算法定稿、真实消费者（语义关联层）接上来之后再做**（用户裁定 2026-09-01）。
#   到那时才知道查询模式是不是真的按周几取；现在按猜的键分片，等于把一个没验证过的假设
#   焊进存储格式。生命周期（BHV-LIFECYCLE-001）落地后树本身可能就小一截，届时一并重估。
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
        digest = bytes_digest(payload)
        generation = self._generation_name(tree, digest)
        self.initialize()
        self._materialize(generation, payload, digest)
        published = PublishedGeneration(
            generation=generation,
            digest=digest,
            config_digest=tree.config_digest,
            published_at=tree.built_at.astimezone(UTC),
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
        if bytes_digest(self._read(target, MAX_TREE_BYTES)) != digest:
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
                published_at=datetime.fromisoformat(raw["published_at"]).astimezone(UTC),
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
            raw = self._read(path, MAX_TREE_BYTES)
        except FileNotFoundError as exc:
            raise PredictionTreeStoreError("prediction tree generation is missing") from exc
        if expected_digest is not None and bytes_digest(raw) != expected_digest:
            raise PredictionTreeStoreError("prediction tree generation does not match its pointer")
        # 解码后**立刻放掉字节**再解析。这一步不改变峰值（峰值由解析出的对象图决定，见
        # TODO(PRED-STORE-002) 的实测），但少一份整棵树大小的临时拷贝；``json.loads`` 直接
        # 吃 bytes 反而更差——它内部照样解出一份 str，于是 bytes 与 str 同时在场。
        text = raw.decode("utf-8")
        del raw
        try:
            payload = json.loads(text)
        except ValueError as exc:
            raise PredictionTreeStoreError("prediction tree generation is not valid JSON") from exc
        try:
            return codec.decode(payload)
        except PredictionTreeError as exc:
            # 其余四条错误路径都归一成了本层的类型，这一条不能例外——否则照着
            # ``except PredictionTreeStoreError`` 写读侧的人，会恰好漏掉最常发生的那条
            # （旧代的字节形状与当前 schema 不兼容）。
            raise PredictionTreeStoreError(
                "prediction tree generation cannot be decoded"
            ) from exc

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

        stamp = tree.built_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%f")
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
