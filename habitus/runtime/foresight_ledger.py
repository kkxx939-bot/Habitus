"""结算账本的落盘：承诺与结算各按日一目录、一条一文件、写一次不改。

住在组合根：``foresight`` 不许 import infrastructure，形状与编解码在 ``foresight.ledger``，这里只管目录与字节。
账本是派生物——承诺可以从判断存储重放，结算可以从承诺 + 行为树重放——所以文件只增不改、同身份重写幂等。

    storage.root/foresight/claims/YYYY/MM/DD/<claim_id>.json
    storage.root/foresight/settlements/YYYY/MM/DD/<claim_id>.json
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from habitus.foresight import ForesightError
from habitus.foresight.ledger import (
    Claim,
    Settlement,
    decode_claim,
    decode_settlement,
    encode_claim,
    encode_settlement,
)
from habitus.infrastructure.store.filesystem import (
    DurableDirectoryEntry,
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    list_real_directory,
    read_regular_bytes,
)

_MAX_ENTRIES = 100_000
_MAX_BYTES = 1_000_000


class ForesightLedgerStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        self._claims = self.root / "claims"
        self._settlements = self.root / "settlements"

    # ── 承诺 ────────────────────────────────────────────────────────────────

    def record_claim(self, claim: Claim) -> bool:
        """写一条；已存在且内容相同返回 False，内容不同是账本损坏。"""

        return self._create(self._claims, claim.day, claim.claim_id, encode_claim(claim))

    def claims_on(self, day: date) -> tuple[Claim, ...]:
        return tuple(decode_claim(raw) for raw in self._records(self._claims, day))

    def unsettled_claims(self, day: date) -> tuple[Claim, ...]:
        settled = {item.claim_id for item in self.settlements_on(day)}
        return tuple(claim for claim in self.claims_on(day) if claim.claim_id not in settled)

    def days_with_claims(self) -> tuple[date, ...]:
        return self._days(self._claims)

    # ── 结算 ────────────────────────────────────────────────────────────────

    def record_settlement(self, item: Settlement) -> bool:
        return self._create(self._settlements, item.day, item.claim_id, encode_settlement(item))

    def settlements_on(self, day: date) -> tuple[Settlement, ...]:
        return tuple(decode_settlement(raw) for raw in self._records(self._settlements, day))

    def settlements(self) -> tuple[Settlement, ...]:
        """全部结算，按日升序。门是读时投影，现阶段全量读（体量问题记在 FORESIGHT-PERF-001 一类）。"""

        items: list[Settlement] = []
        for day in self._days(self._settlements):
            items.extend(self.settlements_on(day))
        return tuple(items)

    # ── 目录与字节 ──────────────────────────────────────────────────────────

    def _create(self, root: Path, day: date, identity: str, payload: dict[str, object]) -> bool:
        encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        path = self._day_dir(root, day) / f"{identity}.json"
        try:
            return atomic_create_bytes(path, encoded, artifact_root=self.root)
        except ImmutableArtifactConflictError as exc:
            raise ForesightError(f"ledger record {identity} already exists with different content") from exc
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"ledger path is not writable: {exc}") from exc

    def _records(self, root: Path, day: date) -> Iterable[dict[str, object]]:
        directory = self._day_dir(root, day)
        for entry in self._entries(directory):
            if not entry.is_file() or not entry.name.endswith(".json") or entry.name.startswith("."):
                continue
            try:
                encoded = read_regular_bytes(directory / entry.name, artifact_root=self.root, max_bytes=_MAX_BYTES)
                raw = json.loads(encoded)
            except (DurablePathIntegrityError, ValueError) as exc:
                raise ForesightError(f"ledger record {entry.name} cannot be read: {exc}") from exc
            if not isinstance(raw, dict):
                raise ForesightError(f"ledger record {entry.name} is not an object")
            yield raw

    def _days(self, root: Path) -> tuple[date, ...]:
        days: list[date] = []
        for year in self._entries(root):
            for month in self._entries(root / year.name):
                for day in self._entries(root / year.name / month.name):
                    if not day.is_dir():
                        continue
                    try:
                        days.append(date(int(year.name), int(month.name), int(day.name)))
                    except ValueError:
                        continue
        return tuple(sorted(days))

    def _entries(self, directory: Path) -> tuple[DurableDirectoryEntry, ...]:
        if not directory.is_dir():
            return ()
        try:
            return tuple(list_real_directory(directory, artifact_root=self.root, max_entries=_MAX_ENTRIES))
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"ledger directory is invalid: {exc}") from exc

    @staticmethod
    def _day_dir(root: Path, day: date) -> Path:
        return root / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"


__all__ = ["ForesightLedgerStore"]
