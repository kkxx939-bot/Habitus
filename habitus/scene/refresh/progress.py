"""刷新器的耐久进度：待刷新日集合、失败记账（封锁）、未发布的归组检查点。

这些都是可重建的进度，与情景树分家（住在 ``progress_root``，不在树根下）。三样东西各是一个小文件
或一个目录，读坏了抛 ``SceneRefreshError``，由编排层决定怎么降级。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from habitus.foundation.integrity import canonical_json
from habitus.infrastructure.store.filesystem import (
    atomic_replace_bytes,
    durable_unlink,
    ensure_real_directory,
    read_regular_bytes,
)
from habitus.scene.grouping.model import DraftRelation, GroupingAssembly, SceneDraft
from habitus.scene.model import SceneLinkType, SceneRole

_PENDING_NAME = "refresh_pending.json"
_BLOCKED_NAME = "blocked.json"
_CHECKPOINT_DIRECTORY = "checkpoints"
_MAX_PENDING_BYTES = 1_048_576
_MAX_BLOCKED_BYTES = 4_194_304
_MAX_CHECKPOINT_BYTES = 16_777_216


class SceneRefreshError(RuntimeError):
    """刷新器自己的耐久文件不可用（待刷新集合 / 封锁记录 / 检查点损坏）。"""


@dataclass(frozen=True)
class FailureRecord:
    """某一天最近一次失败的记账：针对哪个输入、连续几次、是不是已封锁。"""

    source_digest: str
    attempts: int
    error: str
    blocked: bool

    def to_dict(self) -> dict[str, Any]:
        return {"source_digest": self.source_digest, "attempts": self.attempts, "error": self.error, "blocked": self.blocked}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FailureRecord:
        return cls(
            source_digest=str(raw.get("source_digest", "")),
            attempts=int(raw.get("attempts", 0)),
            error=str(raw.get("error", "")),
            blocked=bool(raw.get("blocked", False)),
        )


class RefreshProgress:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().absolute()

    # ── 待刷新日 ─────────────────────────────────────────────────────────

    @property
    def _pending_path(self) -> Path:
        return self.root / _PENDING_NAME

    def pending_days(self) -> set[date]:
        try:
            encoded = read_regular_bytes(self._pending_path, artifact_root=self.root, max_bytes=_MAX_PENDING_BYTES)
        except FileNotFoundError:
            return set()
        try:
            raw = json.loads(encoded.decode("utf-8"))
            if not isinstance(raw, list):
                raise ValueError("not a list")
            return {date.fromisoformat(item) for item in raw}
        except (UnicodeDecodeError, ValueError, TypeError) as exc:
            raise SceneRefreshError("scene refresh_pending file is not decodable") from exc

    def write_pending_days(self, days: set[date]) -> None:
        ensure_real_directory(self.root, artifact_root=self.root)
        if not days:
            durable_unlink(self._pending_path, artifact_root=self.root)
            return
        encoded = json.dumps(sorted(day.isoformat() for day in days)).encode("utf-8")
        atomic_replace_bytes(self._pending_path, encoded, artifact_root=self.root)

    # ── 失败记账 ─────────────────────────────────────────────────────────

    @property
    def _blocked_path(self) -> Path:
        return self.root / _BLOCKED_NAME

    def failures(self) -> dict[date, FailureRecord]:
        try:
            encoded = read_regular_bytes(self._blocked_path, artifact_root=self.root, max_bytes=_MAX_BLOCKED_BYTES)
        except FileNotFoundError:
            return {}
        try:
            raw = json.loads(encoded.decode("utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("not an object")
            return {date.fromisoformat(key): FailureRecord.from_dict(value) for key, value in raw.items()}
        except (UnicodeDecodeError, ValueError, TypeError, AttributeError) as exc:
            raise SceneRefreshError("scene blocked file is not decodable") from exc

    def _write_failures(self, failures: Mapping[date, FailureRecord]) -> None:
        ensure_real_directory(self.root, artifact_root=self.root)
        if not failures:
            durable_unlink(self._blocked_path, artifact_root=self.root)
            return
        body = {day.isoformat(): record.to_dict() for day, record in sorted(failures.items())}
        atomic_replace_bytes(self._blocked_path, canonical_json(body).encode("utf-8"), artifact_root=self.root)

    def record_failure(self, day: date, digest: str, error: str, *, deterministic: bool, max_attempts: int) -> FailureRecord:
        """记一次失败；同一输入连续到 ``max_attempts`` 次、或属确定性失败即封锁。输入变了计数重来。"""

        failures = self.failures()
        previous = failures.get(day)
        attempts = previous.attempts + 1 if previous is not None and previous.source_digest == digest else 1
        record = FailureRecord(digest, attempts, error, deterministic or attempts >= max_attempts)
        failures[day] = record
        self._write_failures(failures)
        return record

    def clear_failure(self, day: date) -> None:
        failures = self.failures()
        if day in failures:
            failures.pop(day)
            self._write_failures(failures)

    # ── 检查点 ───────────────────────────────────────────────────────────

    def _checkpoint_path(self, day: date) -> Path:
        return self.root / _CHECKPOINT_DIRECTORY / f"{day.isoformat()}.json"

    def write_checkpoint(self, day: date, digest: str, version: str, assembly: GroupingAssembly) -> None:
        ensure_real_directory(self.root / _CHECKPOINT_DIRECTORY, artifact_root=self.root)
        body = {"day": day.isoformat(), "source_digest": digest, "scene_version": version, "assembly": assembly_to_json(assembly)}
        atomic_replace_bytes(self._checkpoint_path(day), canonical_json(body).encode("utf-8"), artifact_root=self.root)

    def load_checkpoint(self, day: date, digest: str) -> GroupingAssembly | None:
        """只认与当前输入指纹一致的检查点；不一致的当不存在（会被下一次成功覆盖或清掉）。"""

        try:
            encoded = read_regular_bytes(self._checkpoint_path(day), artifact_root=self.root, max_bytes=_MAX_CHECKPOINT_BYTES)
        except FileNotFoundError:
            return None
        try:
            raw = json.loads(encoded.decode("utf-8"))
            if not isinstance(raw, Mapping) or raw.get("source_digest") != digest:
                return None
            return assembly_from_json(raw["assembly"])
        except (UnicodeDecodeError, ValueError, TypeError, KeyError) as exc:
            raise SceneRefreshError(f"scene checkpoint for {day.isoformat()} is not decodable") from exc

    def clear_checkpoint(self, day: date) -> None:
        durable_unlink(self._checkpoint_path(day), artifact_root=self.root)


# ── 装配产物的 JSON 形状（检查点用） ──────────────────────────────────────────


def assembly_to_json(assembly: GroupingAssembly) -> dict[str, Any]:
    return {
        "scenes": [
            {
                "label": draft.label,
                "members": [[no, role.value] for no, role in draft.members],
                "effects": list(draft.effects),
                "pending_effects": [[text, no] for text, no in draft.pending_effects],
                "relations": [
                    {
                        "kind": relation.kind.value,
                        "occurrence_no": relation.occurrence_no,
                        "scene_index": relation.scene_index,
                        "reference_no": relation.reference_no,
                        "pending_no": relation.pending_no,
                    }
                    for relation in draft.relations
                ],
            }
            for draft in assembly.scenes
        ],
        "unassigned": list(assembly.unassigned),
        "signals": list(assembly.signals),
    }


def assembly_from_json(raw: object) -> GroupingAssembly:
    if not isinstance(raw, Mapping):
        raise ValueError("assembly must be an object")
    scenes = tuple(
        SceneDraft(
            label=str(item["label"]),
            members=tuple((int(no), SceneRole(role)) for no, role in item["members"]),
            effects=tuple(str(text) for text in item["effects"]),
            pending_effects=tuple((str(text), int(no)) for text, no in item["pending_effects"]),
            relations=tuple(
                DraftRelation(
                    kind=SceneLinkType(relation["kind"]),
                    occurrence_no=relation.get("occurrence_no"),
                    scene_index=relation.get("scene_index"),
                    reference_no=relation.get("reference_no"),
                    pending_no=relation.get("pending_no"),
                )
                for relation in item["relations"]
            ),
        )
        for item in raw["scenes"]
    )
    return GroupingAssembly(
        scenes=scenes,
        unassigned=tuple(int(no) for no in raw["unassigned"]),
        signals=tuple(str(note) for note in raw["signals"]),
    )


__all__ = ["FailureRecord", "RefreshProgress", "SceneRefreshError", "assembly_from_json", "assembly_to_json"]
