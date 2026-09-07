"""按天多代发布的情景树存储。

## 天是重建单位，逻辑地址不含代

物理布局：

    <root>/scenes/YYYY/MM/DD/current.json                 该日指针：哪一代在服务
    <root>/scenes/YYYY/MM/DD/generations/<代>/{leaf}.md   一代 = 那一天的全部情景文档

逻辑 URI ``scene://scenes/YYYY/MM/DD/{leaf}.md`` 经该日指针解析到当前代——重建一天不改任何
地址，引用它的下游（上下文视图、结算账本）不随重建断链。

## 两阶段发布，沿预测树的纪律

先把整代写进新目录并**逐篇回读、逐字节核对**，再原子翻该日指针；任一步失败指针不动、旧代继续
服务，写了一半的代目录被尽力清掉（清不掉也只是垃圾，不影响正确性）。读侧每次都用当前代的
原始字节重算内容摘要与指针核对——同址的另一版文档也是合法文档，只有摘要能识破它。

"处理过但一条情景都没有"是合法产出（模型全填 null）——它体现为一个零文档的代加一份指针，
与"还没处理"（无指针）可区分。

写入方只有一个（归约 sweep 收尾里的刷新器，持 behavior-root 级租约），本模块不再加锁；
并发发布同一天在设计上不发生，若发生，代名含内容摘要会让同名不同内容以冲突现形。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from habitus.behavior.model import is_ascii_digits
from habitus.foundation.ids import canonical_path_identity
from habitus.foundation.integrity import bytes_digest, canonical_digest, canonical_json
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
from habitus.scene.document import (
    SceneDocument,
    SceneDocumentCodec,
    SceneDocumentConfig,
    SceneDocumentIntegrityError,
    SceneDocumentLimitError,
)
from habitus.scene.model import SceneAddress, SceneDirectory, scene_static_directories
from habitus.scene.schema import SceneSchemaRegistry
from habitus.scene.tree.config import SceneTreeConfig

POINTER_FILENAME = "current.json"
GENERATIONS_DIRECTORY = "generations"
_MAX_POINTER_BYTES = 4096
_MAX_GENERATION_ENTRIES = 4096
_POINTER_KEYS = {"day", "generation", "digest", "document_count", "published_at", "source_digest", "scene_version"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
# 代名由本模块生成：UTC 时间戳 + 内容摘要前缀；读侧按同一形状硬校验，指针里出现别的东西即损坏。
_GENERATION_NAME = re.compile(r"^[0-9]{8}T[0-9]{12}Z-[0-9a-f]{12}$")


class SceneTreeIntegrityError(ValueError):
    """情景树包含路径逃逸、符号链接、不符合已确认结构的条目，或指针与内容不一致。"""


class SceneTreeConflictError(RuntimeError):
    """同名代已绑定到不同内容，或文档集合与目标日期/指针输入不符。"""


@dataclass(frozen=True)
class SceneDayGeneration:
    """某一天当前生效的那一代：指针的内容。"""

    day: date
    generation: str
    digest: str
    document_count: int
    published_at: datetime
    source_digest: str
    scene_version: str

    def __post_init__(self) -> None:
        if isinstance(self.day, datetime) or not isinstance(self.day, date):
            raise SceneTreeIntegrityError("scene day generation day must be a date")
        for name in ("generation", "digest", "source_digest", "scene_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise SceneTreeIntegrityError(f"scene day generation {name} must be non-empty text")
        if not _GENERATION_NAME.fullmatch(self.generation):
            raise SceneTreeIntegrityError("scene generation name does not have the published shape")
        if not _SHA256.fullmatch(self.digest) or not self.generation.endswith(self.digest[:12]):
            raise SceneTreeIntegrityError("scene generation name does not carry its content digest")
        if not _SHA256.fullmatch(self.source_digest):
            raise SceneTreeIntegrityError("scene day generation source_digest must be lowercase SHA-256 text")
        if isinstance(self.document_count, bool) or not isinstance(self.document_count, int) or self.document_count < 0:
            raise SceneTreeIntegrityError("scene day generation document_count must be a non-negative integer")
        if not isinstance(self.published_at, datetime) or self.published_at.utcoffset() is None:
            raise SceneTreeIntegrityError("scene day generation published_at must be timezone-aware")
        object.__setattr__(self, "published_at", self.published_at.astimezone(UTC))

    def to_dict(self) -> dict[str, object]:
        return {
            "day": self.day.isoformat(),
            "generation": self.generation,
            "digest": self.digest,
            "document_count": self.document_count,
            "published_at": self.published_at.isoformat(),
            "source_digest": self.source_digest,
            "scene_version": self.scene_version,
        }


def generation_digest(encoded: dict[str, bytes]) -> str:
    """一代的内容身份：每篇的叶名与字节摘要的有序集合。"""

    return canonical_digest({"documents": sorted((name, bytes_digest(payload)) for name, payload in encoded.items())})


class SceneTree:
    """安全持久化按天多代的情景 L2 文档。"""

    _STATIC_DIRECTORIES = scene_static_directories()

    def __init__(
        self,
        root: str | Path,
        *,
        document_codec: SceneDocumentCodec | None = None,
        document_config: SceneDocumentConfig | None = None,
        tree_config: SceneTreeConfig | None = None,
    ) -> None:
        requested = Path(root).expanduser().absolute()
        if requested.is_symlink():
            raise SceneTreeIntegrityError("scene tree root cannot be a symbolic link")
        self.root = requested.resolve(strict=False)
        if document_codec is None:
            document_codec = SceneDocumentCodec(SceneSchemaRegistry.load_default())
        if not isinstance(document_codec, SceneDocumentCodec):
            raise TypeError("document_codec must be a SceneDocumentCodec")
        if document_config is not None and not isinstance(document_config, SceneDocumentConfig):
            raise TypeError("document_config must be SceneDocumentConfig")
        if tree_config is not None and not isinstance(tree_config, SceneTreeConfig):
            raise TypeError("tree_config must be SceneTreeConfig")
        self._document_codec = document_codec
        self.document_config = document_config or SceneDocumentConfig()
        self.tree_config = tree_config or SceneTreeConfig()

    @property
    def document_codec(self) -> SceneDocumentCodec:
        return self._document_codec

    @property
    def registry(self) -> SceneSchemaRegistry:
        return self._document_codec.registry

    def initialize(self) -> Path:
        self._ensure_directory(self.root)
        for parts in self._STATIC_DIRECTORIES:
            self._ensure_directory(self.root.joinpath(*parts))
        return self.root

    # ── 发布 ─────────────────────────────────────────────────────────────────────

    def publish_day(
        self,
        day: date,
        documents: Sequence[SceneDocument],
        *,
        published_at: datetime,
        source_digest: str,
        scene_version: str,
    ) -> SceneDayGeneration:
        """物化并激活某一天的一代；返回现在正在服务的那一代。

        文档必须全部属于 ``day``、地址互不相同，且每篇的 ``source_digest`` / ``scene_version``
        与指针输入一致（同一天的溯源只有一个答案）；零文档是合法的。逐字节相同的整代重放幂等成功；
        同名代不同内容即冲突。任何校验都在写入之前完成。
        """

        if isinstance(day, datetime) or not isinstance(day, date):
            raise TypeError("day must be a date without a time")
        if isinstance(documents, str | bytes) or not isinstance(documents, Sequence):
            raise TypeError("documents must be a sequence of SceneDocument")
        if not isinstance(published_at, datetime) or published_at.utcoffset() is None:
            raise ValueError("published_at must be a timezone-aware datetime")
        if not isinstance(source_digest, str) or not _SHA256.fullmatch(source_digest):
            raise ValueError("source_digest must be lowercase SHA-256 text")
        if not isinstance(scene_version, str) or not scene_version:
            raise ValueError("scene_version must be non-empty text")
        encoded = self._encode_generation(day, documents, source_digest=source_digest, scene_version=scene_version)
        digest = generation_digest(encoded)
        stamp = published_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%f")
        published = SceneDayGeneration(
            day=day,
            generation=f"{stamp}Z-{digest[:12]}",
            digest=digest,
            document_count=len(encoded),
            published_at=published_at,
            source_digest=source_digest,
            scene_version=scene_version,
        )
        self.initialize()
        day_directory = self._day_path(day)
        generation_directory = day_directory / GENERATIONS_DIRECTORY / published.generation
        self._ensure_directory(generation_directory)
        try:
            for name, payload in encoded.items():
                try:
                    atomic_create_bytes(generation_directory / name, payload, artifact_root=self.root)
                except ImmutableArtifactConflictError as exc:
                    raise SceneTreeConflictError("scene generation is already bound to different content") from exc
                except DurablePathIntegrityError as exc:
                    raise SceneTreeIntegrityError("scene document cannot be written safely") from exc
            # 回读校验：整代逐篇解码、逐字节与预期一致，才允许翻指针。
            written, raw = self._read_generation(day, published.generation)
            if raw != encoded or len(written) != len(encoded):
                raise SceneTreeIntegrityError("scene generation failed read-back verification")
            try:
                atomic_replace_bytes(
                    day_directory / POINTER_FILENAME,
                    canonical_json(published.to_dict()).encode("utf-8"),
                    artifact_root=self.root,
                )
            except DurablePathIntegrityError as exc:
                raise SceneTreeIntegrityError("scene day pointer cannot be replaced") from exc
        except SceneTreeConflictError:
            # 同名代撞上了不同内容：那一代是别人的，不能动。
            raise
        except SceneTreeIntegrityError:
            # 指针没翻，这一代没人引用：尽力清掉，免得它占用留代名额。
            self._discard_generation(day, published.generation)
            raise
        self._prune(day, keep=published.generation)
        return published

    def _encode_generation(
        self, day: date, documents: Sequence[SceneDocument], *, source_digest: str, scene_version: str
    ) -> dict[str, bytes]:
        encoded: dict[str, bytes] = {}
        identities: set[str] = set()
        for document in documents:
            if not isinstance(document, SceneDocument):
                raise TypeError("documents must contain SceneDocument values")
            if document.address.occurred_on != day:
                raise SceneTreeConflictError("scene document does not belong to the published day")
            if document.fields["source_digest"] != source_digest or document.fields["scene_version"] != scene_version:
                raise SceneTreeConflictError("scene document provenance disagrees with the day pointer")
            identity = canonical_path_identity(document.address.identity_name, "scene document name")
            if identity in identities:
                raise SceneTreeConflictError("scene day contains two documents with the same identity")
            identities.add(identity)
            payload = self._document_codec.encode(document).encode("utf-8")
            self.document_config.validate_body(document.markdown_body)
            self.document_config.validate_relations(links=len(document.links), members=len(document.fields["members"]))
            self.document_config.validate_encoded(payload)
            encoded[f"{document.address.identity_name}.md"] = payload
        if len(encoded) > self.tree_config.max_children_per_directory:
            raise SceneTreeIntegrityError("scene day exceeds the directory child capacity")
        return encoded

    # ── 读取 ─────────────────────────────────────────────────────────────────────

    def day_state(self, day: date) -> SceneDayGeneration | None:
        """该日当前生效的那一代；还没处理过这一天时为 None。"""

        if isinstance(day, datetime) or not isinstance(day, date):
            raise TypeError("day must be a date without a time")
        path = self._day_path(day) / POINTER_FILENAME
        try:
            if not regular_file_exists(path, artifact_root=self.root):
                return None
            raw = json.loads(read_regular_bytes(path, artifact_root=self.root, max_bytes=_MAX_POINTER_BYTES).decode("utf-8"))
        except FileNotFoundError:
            return None
        except DurablePathIntegrityError as exc:
            raise SceneTreeIntegrityError("scene day pointer cannot be read safely") from exc
        except (UnicodeDecodeError, ValueError) as exc:
            raise SceneTreeIntegrityError("scene day pointer is malformed") from exc
        if not isinstance(raw, dict) or set(raw) != _POINTER_KEYS:
            raise SceneTreeIntegrityError("scene day pointer has an invalid shape")
        try:
            pointer = SceneDayGeneration(
                day=date.fromisoformat(str(raw["day"])),
                generation=str(raw["generation"]),
                digest=str(raw["digest"]),
                document_count=raw["document_count"],
                published_at=datetime.fromisoformat(str(raw["published_at"])),
                source_digest=str(raw["source_digest"]),
                scene_version=str(raw["scene_version"]),
            )
        except (TypeError, ValueError) as exc:
            raise SceneTreeIntegrityError("scene day pointer is malformed") from exc
        if pointer.day != day:
            raise SceneTreeIntegrityError("scene day pointer belongs to another day")
        return pointer

    def read_day(self, day: date) -> tuple[SceneDocument, ...]:
        """该日当前代的全部情景，按开始瞬时排序；未处理过的日子返回空。

        用当前代的原始字节重算内容摘要并与指针核对：同址的另一版文档也是合法文档，只有摘要能识破。
        """

        pointer = self.day_state(day)
        if pointer is None:
            return ()
        documents, raw = self._read_generation(day, pointer.generation)
        if len(documents) != pointer.document_count or generation_digest(raw) != pointer.digest:
            raise SceneTreeIntegrityError("scene generation does not match its pointer")
        return tuple(sorted(documents, key=lambda item: (item.address.started_at.astimezone(UTC), item.address.identity_name)))

    def read(self, address: SceneAddress) -> SceneDocument:
        """当前代里这个地址的文档；不在当前代时报 FileNotFoundError（完整性问题另抛，不混淆为缺席）。"""

        if not isinstance(address, SceneAddress):
            raise TypeError("address must be a SceneAddress")
        for document in self.read_day(address.occurred_on):
            if document.address == address:
                return document
        raise FileNotFoundError(f"scene document is not in the active generation: {address.identity_name}")

    def exists(self, address: SceneAddress) -> bool:
        if not isinstance(address, SceneAddress):
            raise TypeError("address must be a SceneAddress")
        return any(document.address == address for document in self.read_day(address.occurred_on))

    def list_days(self) -> tuple[date, ...]:
        """已处理过（有指针）的日子，升序。"""

        base = self.root.joinpath(*SceneDirectory.scenes().identity_parts)
        if not self._directory_exists(base):
            return ()
        days: list[date] = []
        for year in self._directories(base):
            if len(year.name) != 4 or not is_ascii_digits(year.name):
                raise SceneTreeIntegrityError("scene year directory must use YYYY")
            for month in self._directories(year):
                if len(month.name) != 2 or not is_ascii_digits(month.name):
                    raise SceneTreeIntegrityError("scene month directory must use MM")
                for day_path in self._directories(month):
                    if len(day_path.name) != 2 or not is_ascii_digits(day_path.name):
                        raise SceneTreeIntegrityError("scene day directory must use DD")
                    try:
                        day = date(int(year.name), int(month.name), int(day_path.name))
                    except ValueError as exc:
                        raise SceneTreeIntegrityError("scene directory contains an invalid calendar date") from exc
                    if self.day_state(day) is not None:
                        days.append(day)
        return tuple(sorted(days))

    def generations_of(self, day: date) -> tuple[str, ...]:
        """该日已物化的代，按代名升序即时间升序（含尚未激活或已作废的代）。"""

        path = self._day_path(day) / GENERATIONS_DIRECTORY
        if not self._directory_exists(path):
            return ()
        return tuple(entry.name for entry in self._directories(path))

    def path_for(self, address: SceneAddress) -> Path:
        """当前代里这篇文档的物理路径（运维用；读取一律走 ``read``）；该日未处理时报 FileNotFoundError。"""

        pointer = self.day_state(address.occurred_on)
        if pointer is None:
            raise FileNotFoundError("scene day has no active generation")
        return self._day_path(address.occurred_on) / GENERATIONS_DIRECTORY / pointer.generation / f"{address.identity_name}.md"

    # ── 留代 ─────────────────────────────────────────────────────────────────────

    def _prune(self, day: date, *, keep: str) -> bool:
        """指针翻过之后删该日的旧代；永不删正在服务的那一代。失败吞掉——新代已在服务。"""

        try:
            active = self.day_state(day)
            protected = {keep} if active is None else {keep, active.generation}
            existing = self.generations_of(day)
            surplus = existing[: max(0, len(existing) - self.tree_config.retained_generations)]
            for generation in surplus:
                if generation in protected:
                    continue
                self._discard_generation(day, generation)
        except (SceneTreeIntegrityError, DurablePathIntegrityError, OSError):
            return False
        return True

    def _discard_generation(self, day: date, generation: str) -> None:
        """删掉一代目录（含本模块自己遗留的临时文件）；只用于从未激活或已退出留代的代。"""

        directory = self._day_path(day) / GENERATIONS_DIRECTORY / generation
        try:
            for entry in self._children(directory):
                if entry.is_file():
                    durable_unlink(directory / entry.name, artifact_root=self.root)
            durable_rmdir(directory, artifact_root=self.root)
        except (SceneTreeIntegrityError, DurablePathIntegrityError, OSError):
            return

    # ── 机械件 ───────────────────────────────────────────────────────────────────

    def _read_generation(self, day: date, generation: str) -> tuple[tuple[SceneDocument, ...], dict[str, bytes]]:
        """读一代的全部文档，同时返回原始字节（摘要核对用）。"""

        if not _GENERATION_NAME.fullmatch(generation):
            raise SceneTreeIntegrityError("scene generation name does not have the published shape")
        directory = self._day_path(day) / GENERATIONS_DIRECTORY / generation
        if not self._directory_exists(directory):
            raise SceneTreeIntegrityError("scene generation directory is missing")
        documents: list[SceneDocument] = []
        raw_bytes: dict[str, bytes] = {}
        seen: set[str] = set()
        for entry in self._content_children(directory):
            if not entry.is_file() or not entry.name.endswith(".md") or len(entry.name) <= 3:
                raise SceneTreeIntegrityError("scene generation may contain only Markdown documents")
            stem = entry.name[:-3]
            try:
                identity = canonical_path_identity(stem, "scene document name")
                address = SceneAddress.from_identity(day, stem)
            except (TypeError, ValueError) as exc:
                raise SceneTreeIntegrityError("scene leaf contains an invalid identity") from exc
            if identity in seen:
                raise SceneTreeIntegrityError("scene generation contains multiple physical aliases for one identity")
            seen.add(identity)
            try:
                raw = read_regular_bytes(
                    directory / entry.name, artifact_root=self.root, max_bytes=self.document_config.max_encoded_bytes
                )
                text = raw.decode("utf-8")
            except FileNotFoundError as exc:
                raise SceneTreeIntegrityError("scene generation changed while being read") from exc
            except DurablePathIntegrityError as exc:
                raise SceneTreeIntegrityError("scene document cannot be read safely") from exc
            except UnicodeDecodeError as exc:
                raise SceneTreeIntegrityError("scene document is not valid UTF-8") from exc
            try:
                document = self._document_codec.decode(text, expected_address=address)
                self.document_config.validate_body(document.markdown_body)
                self.document_config.validate_relations(links=len(document.links), members=len(document.fields["members"]))
            except (SceneDocumentIntegrityError, SceneDocumentLimitError) as exc:
                raise SceneTreeIntegrityError("scene L2 document failed integrity validation") from exc
            documents.append(document)
            raw_bytes[entry.name] = raw
        return tuple(documents), raw_bytes

    def _day_path(self, day: date) -> Path:
        candidate = self.root.joinpath(*SceneDirectory.for_day(day).identity_parts)
        self._require_inside_root(candidate)
        return candidate

    def _content_children(self, directory: Path) -> tuple[DurableDirectoryEntry, ...]:
        content: list[DurableDirectoryEntry] = []
        for child in self._children(directory):
            if child.name.startswith(".") and child.is_file() and atomic_temporary_destination(child.name) is not None:
                continue
            if child.name.startswith("."):
                raise SceneTreeIntegrityError("scene directory contains an unsupported hidden entry")
            content.append(child)
        return tuple(content)

    def _directories(self, directory: Path) -> tuple[Path, ...]:
        """分支目录只允许子目录（日目录上的指针文件除外）。"""

        children = self._content_children(directory)
        for child in children:
            if child.is_dir():
                continue
            if child.name == POINTER_FILENAME and directory.name.isdigit() and len(directory.name) == 2:
                continue
            raise SceneTreeIntegrityError("scene branch may contain only directories")
        return tuple(sorted(directory / child.name for child in children if child.is_dir()))

    def _children(self, directory: Path) -> tuple[DurableDirectoryEntry, ...]:
        try:
            entries = list_real_directory(
                directory, artifact_root=self.root, max_entries=max(self.tree_config.max_children_per_directory, _MAX_GENERATION_ENTRIES) + 1
            )
        except DurablePathIntegrityError as exc:
            raise SceneTreeIntegrityError("scene directory cannot be enumerated safely") from exc
        persistent = sum(
            not (entry.name.startswith(".") and entry.is_file() and atomic_temporary_destination(entry.name) is not None)
            for entry in entries
        )
        if persistent > max(self.tree_config.max_children_per_directory, _MAX_GENERATION_ENTRIES):
            raise SceneTreeIntegrityError("scene directory exceeds its persistent child capacity")
        return entries

    def _directory_exists(self, directory: Path) -> bool:
        self._require_inside_root(directory)
        try:
            return real_directory_exists(directory, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise SceneTreeIntegrityError("scene directory cannot be inspected safely") from exc

    def _ensure_directory(self, directory: Path) -> None:
        self._require_inside_root(directory)
        try:
            ensure_real_directory(directory, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise SceneTreeIntegrityError("scene directory cannot be created safely") from exc

    def _require_inside_root(self, path: Path) -> None:
        candidate = path.resolve(strict=False)
        if candidate != self.root and self.root not in candidate.parents:
            raise SceneTreeIntegrityError("scene path cannot be used safely outside its tree root")


__all__ = [
    "GENERATIONS_DIRECTORY",
    "POINTER_FILENAME",
    "SceneDayGeneration",
    "SceneTree",
    "SceneTreeConflictError",
    "SceneTreeIntegrityError",
    "generation_digest",
]
