"""提醒出口的传输状态落盘（方案 32 第 3.2 条）：此刻可取的请求清单、取走记录、问过的提问、宿主活性。这些是传输状态，
不进账本；账本只记事实（请求发出了、宿主说了，``runtime/foresight_ledger.py``）。

住在组合根、只管目录与字节：形状与规则在 ``runtime/foresight_outlet.py``。这里不认识账本的形状（写账本的只有账本门面）。

    <foresight 根>/outlet/open.json           此刻可取的请求（递送那一拍每拍重写、出口认出提问后删行）
    <foresight 根>/outlet/claims/<请求>.json   一条请求的取走记录（现任持有者、历任持有者、放回次数），到最晚时限删
    <foresight 根>/outlet/asked/<摘要>.json    问出口了、作答还没来的提问（Codex Desktop 的作答在下一份转写里才来），按调用身份存
    <foresight 根>/outlet/hosts/<宿主>.json    这个宿主最近一次来取的时刻（活性）

小文件、整份替换；一个坏文件不许让整层停：读不出来的当没有。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from habitus.foresight.errors import ForesightError
from habitus.infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    atomic_replace_bytes,
    durable_unlink,
    list_real_directory,
    read_regular_bytes,
)

_MAX_BYTES = 1_000_000
_MAX_ENTRIES = 10_000
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class OutletStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        self._open = self.root / "open.json"
        self._claims = self.root / "claims"
        self._asked = self.root / "asked"
        self._hosts = self.root / "hosts"

    # ── 可取的清单 ──────────────────────────────────────────────────────────

    def read_open(self) -> tuple[dict[str, Any], ...]:
        raw = self._load(self._open)
        rows = raw.get("rows") if raw is not None else None
        if not isinstance(rows, list):
            return ()
        return tuple(row for row in rows if isinstance(row, dict))

    def write_open(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self._write(self._open, {"record": "open", "rows": [dict(row) for row in rows]})

    # ── 取走记录（按请求） ─────────────────────────────────────────────────

    def read_claim(self, request_id: str) -> dict[str, Any] | None:
        return self._load(self._claims / f"{_safe(request_id, 'request_id')}.json")

    def write_claim(self, request_id: str, payload: Mapping[str, Any]) -> None:
        self._write(self._claims / f"{_safe(request_id, 'request_id')}.json", dict(payload))

    def remove_claim(self, request_id: str) -> bool:
        return self._remove(self._claims / f"{_safe(request_id, 'request_id')}.json")

    def claims(self) -> dict[str, dict[str, Any]]:
        return self._all(self._claims)

    # ── 问过的提问（按调用身份） ───────────────────────────────────────────

    @staticmethod
    def asked_key(call_id: str) -> str:
        """调用身份 → 文件名（call_id 由宿主给，形状不可控，取摘要）。"""

        if not isinstance(call_id, str) or not call_id:
            raise ForesightError("call_id must be non-empty text")
        return sha256(call_id.encode("utf-8")).hexdigest()[:24]

    def read_asked(self, key: str) -> dict[str, Any] | None:
        return self._load(self._asked / f"{_safe(key, 'asked key')}.json")

    def write_asked(self, key: str, payload: Mapping[str, Any]) -> None:
        self._write(self._asked / f"{_safe(key, 'asked key')}.json", dict(payload))

    def remove_asked(self, key: str) -> bool:
        return self._remove(self._asked / f"{_safe(key, 'asked key')}.json")

    def asked(self) -> dict[str, dict[str, Any]]:
        return self._all(self._asked)

    # ── 宿主活性 ────────────────────────────────────────────────────────────

    def touch_host(self, host: str, at: datetime) -> None:
        self._write(
            self._hosts / f"{_safe(host, 'host')}.json", {"record": "host", "host": host, "seen_at": at.isoformat()}
        )

    def hosts(self) -> dict[str, datetime]:
        found: dict[str, datetime] = {}
        for name, raw in self._all(self._hosts).items():
            try:
                found[str(raw.get("host") or name)] = datetime.fromisoformat(str(raw["seen_at"]))
            except (KeyError, ValueError):
                continue
        return found

    # ── 目录与字节 ──────────────────────────────────────────────────────────

    def _all(self, directory: Path) -> dict[str, dict[str, Any]]:
        if not directory.is_dir():
            return {}
        try:
            entries = list_real_directory(directory, artifact_root=self.root, max_entries=_MAX_ENTRIES)
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"outlet directory is invalid: {exc}") from exc
        found: dict[str, dict[str, Any]] = {}
        for entry in sorted(entries, key=lambda item: item.name):
            if not entry.is_file() or not entry.name.endswith(".json") or entry.name.startswith("."):
                continue
            raw = self._load(directory / entry.name)
            if raw is not None:
                found[entry.name.removesuffix(".json")] = raw
        return found

    def _load(self, path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            raw = json.loads(read_regular_bytes(path, artifact_root=self.root, max_bytes=_MAX_BYTES))
        except (DurablePathIntegrityError, ValueError):
            return None
        return raw if isinstance(raw, dict) else None

    def _write(self, path: Path, payload: dict[str, Any]) -> None:
        encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        if len(encoded) > _MAX_BYTES:
            raise ForesightError(f"outlet record {path.name} exceeds {_MAX_BYTES} bytes; not written")
        try:
            atomic_replace_bytes(path, encoded, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"outlet path is not writable: {exc}") from exc

    def _remove(self, path: Path) -> bool:
        try:
            return durable_unlink(path, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"outlet path is not removable: {exc}") from exc


def _safe(value: object, label: str) -> str:
    if not isinstance(value, str) or _SAFE_NAME.fullmatch(value) is None:
        raise ForesightError(f"{label} is not a safe file name: {value!r}")
    return value


__all__ = ["OutletStore"]
