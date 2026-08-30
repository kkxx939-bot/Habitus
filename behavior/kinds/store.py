"""词表的单文件耐久存储：可读正文 + 末尾结构字段，读回必须逐字节自洽。

物理格式沿行为 L2 的同一哲学：正文是结构字段的确定性渲染，读取时重编码比对，任何不一致都是
完整性错误而不是"尽力解析"。

已并入 ``behavior://`` 地址空间：``behavior_root`` 传**行为树根**，文件即树根下的单文件节点
``kinds.md``（URI ``behavior://kinds.md``，见 ``BehaviorURI.kinds()``；性质同 memory 树
``profile.md`` 单文件直读）。树的日期枚举只走 occurrences/gaps 前缀，词表与之互不干扰；
地址叶名不可能与它撞车（semantic_name 拒绝 .md 后缀，gap 叶名是受控枚举）。

格式 ``behavior_kinds_v2``（BHV-KINDS-002）：条目带 label 与命中账。v1 文件（只有正名+别名）
不做读取兼容——按仓库纪律无双轨；旧根用 ``kinds/rebuild`` 从树上补齐重建。

并发说明：写入方只有归约写入层一个，且归约 ``run_once`` 全程持 behavior-root 级 fenced 锁
（词表 CAS 发生在锁内、stage 之前）；``expected_revision`` CAS 只防误用，不承担并发互斥。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from behavior.kinds.config import BehaviorKindConfig
from behavior.kinds.model import (
    BehaviorKindEntry,
    BehaviorKindError,
    BehaviorKindLimitError,
    BehaviorKindRegistry,
)
from behavior.model import KINDS_REGISTRY_FILENAME
from infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    atomic_replace_bytes,
    read_regular_bytes,
)

KINDS_SCHEMA_VERSION = "behavior_kinds_v2"
_MARKER = "\n<!-- HABITUS_BEHAVIOR_KINDS\n"
_FOOTER = "\n-->\n"
_METADATA_KEYS = {"schema_version", "revision", "updated_at", "kinds"}
_ENTRY_KEYS = {"token", "label", "aliases", "created_on", "hit_days", "hit_days_total", "hit_count"}


class BehaviorKindStoreError(ValueError):
    """词表文件的物理内容与登记约束或规范编码不一致。"""


class BehaviorKindConflictError(RuntimeError):
    """替换词表时使用了过期 revision。"""


@dataclass(frozen=True)
class BehaviorKindSnapshot:
    """一次读取看到的完整词表状态；``revision == 0`` 表示词表尚未建立。"""

    registry: BehaviorKindRegistry
    revision: int
    updated_at: datetime | None


class BehaviorKindStore:
    """在 behavior-root 下以单文件保存词表。"""

    def __init__(
        self, behavior_root: str | Path, *, config: BehaviorKindConfig | None = None
    ) -> None:
        resolved = config or BehaviorKindConfig()
        if not isinstance(resolved, BehaviorKindConfig):
            raise TypeError("config must be BehaviorKindConfig")
        self.root = Path(behavior_root).expanduser().resolve(strict=False)
        self.config = resolved
        self.path = self.root / KINDS_REGISTRY_FILENAME

    def read(self) -> BehaviorKindSnapshot:
        try:
            encoded = read_regular_bytes(
                self.path, artifact_root=self.root, max_bytes=self.config.max_encoded_bytes
            )
        except FileNotFoundError:
            return BehaviorKindSnapshot(BehaviorKindRegistry(), 0, None)
        except DurablePathIntegrityError as exc:
            raise BehaviorKindStoreError("behavior kind registry cannot be read safely") from exc
        return self._decode(encoded)

    def replace(
        self,
        registry: BehaviorKindRegistry,
        *,
        expected_revision: int,
        timestamp: datetime,
    ) -> BehaviorKindSnapshot:
        """CAS 替换整份词表；修改语义由不可变 Registry 表达，这里只负责耐久。"""

        if not isinstance(registry, BehaviorKindRegistry):
            raise TypeError("registry must be BehaviorKindRegistry")
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
            raise BehaviorKindError("expected_revision must be an integer")
        if expected_revision < 0:
            raise BehaviorKindError("expected_revision must not be negative")
        if (
            not isinstance(timestamp, datetime)
            or timestamp.tzinfo is None
            or timestamp.utcoffset() is None
        ):
            raise BehaviorKindError("timestamp must be a timezone-aware datetime")
        self._require_bounds(registry)
        current = self.read()
        if current.revision != expected_revision:
            raise BehaviorKindConflictError(
                f"behavior kind registry revision changed: expected {expected_revision}, "
                f"found {current.revision}"
            )
        updated_at = timestamp.astimezone(timezone.utc)
        snapshot = BehaviorKindSnapshot(registry, expected_revision + 1, updated_at)
        encoded = self._encode(snapshot)
        if len(encoded) > self.config.max_encoded_bytes:
            raise BehaviorKindLimitError("behavior kind registry exceeds its encoded byte bound")
        try:
            atomic_replace_bytes(self.path, encoded, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise BehaviorKindStoreError("behavior kind registry cannot be written safely") from exc
        return snapshot

    def _require_bounds(self, registry: BehaviorKindRegistry) -> None:
        if registry.kind_count > self.config.max_kinds:
            raise BehaviorKindLimitError("behavior kind registry exceeds its kind bound")
        for token in registry.tokens:
            if len(registry.aliases_of(token)) > self.config.max_aliases_per_kind:
                raise BehaviorKindLimitError(
                    f"behavior kind has more aliases than allowed: {token}"
                )

    # ── 编码 ──────────────────────────────────────────────────────────────────────

    def _encode(self, snapshot: BehaviorKindSnapshot) -> bytes:
        assert snapshot.updated_at is not None
        lines = ["# 行为类型词表", ""]
        for token in snapshot.registry.tokens:
            entry = snapshot.registry.entry_of(token)
            head = f"- {entry.label}〔{entry.token}〕"
            if entry.hit_count:
                head += f" ×{entry.hit_count}"
            if entry.last_hit_day is not None:
                head += f" 最近 {entry.last_hit_day.isoformat()}"
            lines.append(f"{head}：{'、'.join(entry.aliases)}" if entry.aliases else head)
        body = "\n".join(lines)
        metadata = {
            "schema_version": KINDS_SCHEMA_VERSION,
            "revision": snapshot.revision,
            "updated_at": snapshot.updated_at.isoformat(timespec="microseconds").replace(
                "+00:00", "Z"
            ),
            "kinds": [
                _entry_payload(snapshot.registry.entry_of(token)) for token in snapshot.registry.tokens
            ],
        }
        rendered = json.dumps(
            metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return f"{body}{_MARKER}{rendered}{_FOOTER}".encode()

    def _decode(self, encoded: bytes) -> BehaviorKindSnapshot:
        try:
            text = encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BehaviorKindStoreError("behavior kind registry is not valid UTF-8") from exc
        if text.count(_MARKER) != 1 or not text.endswith(_FOOTER):
            raise BehaviorKindStoreError("behavior kind registry metadata block is malformed")
        _body, _separator, tail = text.partition(_MARKER)
        try:
            metadata = json.loads(tail[: -len(_FOOTER)])
        except (json.JSONDecodeError, RecursionError) as exc:
            raise BehaviorKindStoreError("behavior kind registry metadata is corrupt") from exc
        if not isinstance(metadata, dict) or set(metadata) != _METADATA_KEYS:
            raise BehaviorKindStoreError("behavior kind registry metadata shape is invalid")
        if metadata["schema_version"] != KINDS_SCHEMA_VERSION:
            raise BehaviorKindStoreError("behavior kind registry schema version is unsupported")
        revision = metadata["revision"]
        if isinstance(revision, bool) or not isinstance(revision, int) or revision <= 0:
            raise BehaviorKindStoreError("behavior kind registry revision must be positive")
        raw_updated = metadata["updated_at"]
        if not isinstance(raw_updated, str):
            raise BehaviorKindStoreError("behavior kind registry updated_at must be text")
        try:
            updated_at = datetime.fromisoformat(raw_updated.replace("Z", "+00:00"))
        except ValueError as exc:
            raise BehaviorKindStoreError("behavior kind registry updated_at is invalid") from exc
        if updated_at.utcoffset() != timezone.utc.utcoffset(None):
            raise BehaviorKindStoreError("behavior kind registry updated_at must be UTC")
        entries_raw = metadata["kinds"]
        if not isinstance(entries_raw, list):
            raise BehaviorKindStoreError("behavior kind registry kinds must be a list")
        entries: dict[str, BehaviorKindEntry] = {}
        for raw in entries_raw:
            entry = _entry_from_payload(raw)
            if entry.token in entries:
                raise BehaviorKindStoreError("behavior kind token is repeated")
            entries[entry.token] = entry
        try:
            registry = BehaviorKindRegistry(entries)
        except BehaviorKindError as exc:
            raise BehaviorKindStoreError("behavior kind registry content is invalid") from exc
        self._require_bounds(registry)
        snapshot = BehaviorKindSnapshot(registry, revision, updated_at)
        if self._encode(snapshot) != encoded:
            raise BehaviorKindStoreError("behavior kind registry is not canonically encoded")
        return snapshot


def _entry_payload(entry: BehaviorKindEntry) -> dict[str, object]:
    return {
        "token": entry.token,
        "label": entry.label,
        "aliases": list(entry.aliases),
        "created_on": None if entry.created_on is None else entry.created_on.isoformat(),
        "hit_days": [day.isoformat() for day in entry.hit_days],
        "hit_days_total": entry.hit_days_total,
        "hit_count": entry.hit_count,
    }


def _entry_from_payload(raw: object) -> BehaviorKindEntry:
    if not isinstance(raw, dict) or set(raw) != _ENTRY_KEYS:
        raise BehaviorKindStoreError("behavior kind entry shape is invalid")
    aliases = raw["aliases"]
    if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
        raise BehaviorKindStoreError("behavior kind aliases must be a list of text")
    hit_days = raw["hit_days"]
    if not isinstance(hit_days, list) or any(not isinstance(day, str) for day in hit_days):
        raise BehaviorKindStoreError("behavior kind hit_days must be a list of dates")
    if not isinstance(raw["token"], str) or not isinstance(raw["label"], str):
        raise BehaviorKindStoreError("behavior kind token and label must be text")
    created_raw = raw["created_on"]
    if created_raw is not None and not isinstance(created_raw, str):
        raise BehaviorKindStoreError("behavior kind created_on must be a date or null")
    try:
        entry = BehaviorKindEntry(
            token=raw["token"],
            label=raw["label"],
            aliases=tuple(aliases),
            created_on=None if created_raw is None else _parse_day(created_raw),
            hit_days=tuple(_parse_day(day) for day in hit_days),
            hit_days_total=raw["hit_days_total"],
            hit_count=raw["hit_count"],
        )
    except (BehaviorKindError, ValueError, TypeError) as exc:
        raise BehaviorKindStoreError("behavior kind entry content is invalid") from exc
    return entry


def _parse_day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise BehaviorKindStoreError("behavior kind day is not an ISO date") from exc


__all__ = [
    "KINDS_SCHEMA_VERSION",
    "BehaviorKindConflictError",
    "BehaviorKindSnapshot",
    "BehaviorKindStore",
    "BehaviorKindStoreError",
]
