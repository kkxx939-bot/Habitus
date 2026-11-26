"""归约检查点文件：stage 定格的最终 payload 落在这里，崩溃重试从这里逐字节重放。

只管文件本身（编码、上界、写、清、读回重放、运维门）；把检查点逐字落树的发布动作在 ``runner``，由调用方交进来。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

from habitus.behavior.reduction.errors import BehaviorReductionError
from habitus.infrastructure.store.contracts.path_lock import LeaseGuard
from habitus.infrastructure.store.filesystem import atomic_replace_bytes, read_regular_bytes

_MAX_CHECKPOINT_BYTES = 67_108_864
_CHECKPOINT_NAME = "staged.json"


class CheckpointStore:
    """``{ledger.root}/staged.json`` 的读写；写入与重放共用同一字节上界。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / _CHECKPOINT_NAME

    def exists(self) -> bool:
        return self.path.exists()

    @staticmethod
    def encode(checkpoint: Mapping[str, Any]) -> bytes:
        return json.dumps(checkpoint, ensure_ascii=False, sort_keys=True).encode("utf-8")

    @staticmethod
    def fits(encoded: bytes) -> bool:
        """写入与重放共用同一上界：写得进读不回的检查点 = 永久卡死。"""

        return len(encoded) <= _MAX_CHECKPOINT_BYTES

    def write(self, encoded: bytes, guard: LeaseGuard) -> None:
        with guard.fenced():
            atomic_replace_bytes(self.path, encoded, artifact_root=self.root)

    def clear(self, guard: LeaseGuard) -> None:
        with guard.fenced():
            self.path.unlink(missing_ok=True)

    def replay(self, publish: Callable[[Any], None]) -> tuple[int, set[date]]:
        """重放遗留检查点（``publish`` 把它逐字落树）；清除在 ``_run_locked`` 发布完成后进行（语义刷新另有待刷新集合耐久）。"""

        try:
            encoded = read_regular_bytes(
                self.path,
                artifact_root=self.root,
                max_bytes=_MAX_CHECKPOINT_BYTES,
            )
        except FileNotFoundError:
            return 0, set()
        try:
            checkpoint = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BehaviorReductionError("reduction checkpoint is not decodable") from exc
        publish(checkpoint)
        documents = checkpoint.get("documents")
        if not isinstance(documents, list):
            return 0, set()
        days = document_days(documents)
        raw_days = checkpoint.get("touched_days")
        if isinstance(raw_days, list):
            days |= {date.fromisoformat(str(item)) for item in raw_days}
        replayed = sum(1 for item in documents if isinstance(item, Mapping) and item.get("kind") != "ledger-only")
        return replayed, days


def document_days(documents: list) -> set[date]:
    """检查点文档触及的本地日历日；ledger-only 条目不改树、不触发刷新。"""

    days: set[date] = set()
    for item in documents:
        if not isinstance(item, Mapping) or item.get("kind") == "ledger-only":
            continue
        payload = item.get("payload")
        if isinstance(payload, Mapping) and isinstance(payload.get("occurred_on"), str):
            days.add(date.fromisoformat(payload["occurred_on"]))
    return days
