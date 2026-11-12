"""每一代的封口记录：``<scene_root>/generation/seals/<日子>.json``，换代的语义树一步封口一做完就写，随完成标记一起留存与清理。

裁定 97 第 3 条的先后是"按并入之前的概念集封口 → 再并入待生效的概念"。并入之后、完成标记之前被杀，下一眼从头做这一代时，
概念集里已经有这一代刚并入的概念；再封一次就是按并入之后的概念集封口，新概念提前一代进了情境（阶段 3 补测）。所以封口做完
先落这一份：有它就是"这一代已经封过口"，接着做时不再封，直接用记下的结果——和不中断时一样，这一代只封一次。

封口没做成（尽力而为的那一步抛了）也记一份，标成没做成：不中断时这一代就没封口，接着做也不再封。文件坏了明确报错，
不当"没有"——那样会按并入之后的概念集再封一次。
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from habitus.infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    atomic_replace_bytes,
    ensure_real_directory,
    read_regular_bytes,
)
from habitus.runtime.generation_trace import prune_dated
from habitus.runtime.scene_night_seal import SealReport

_SCHEMA = "generation_seal_v1"
_MAX_BYTES = 1_048_576
_KEYS = frozenset({"schema", "day", "failed", "sealed", "rewritten", "unready_days", "unready_records", "readers"})


class SealFileError(ValueError):
    """这一代的封口记录坏了或读写不了。"""


class SealFiles:
    def __init__(self, root: str | Path, *, artifact_root: str | Path) -> None:
        self.root = Path(root)
        self.artifact_root = Path(artifact_root)

    def path(self, day: date) -> Path:
        return self.root / f"{day.isoformat()}.json"

    def read(self, day: date) -> tuple[SealReport, bool] | None:
        """第 ``day`` 代的封口结果与"没做成"；这一代还没封过口是 None，文件坏了抛 ``SealFileError``。"""

        try:
            raw = read_regular_bytes(self.path(day), artifact_root=self.artifact_root, max_bytes=_MAX_BYTES)
        except FileNotFoundError:
            return None
        except (OSError, DurablePathIntegrityError) as exc:
            raise SealFileError(f"seal record of {day} is unreadable: {type(exc).__name__}") from exc
        return decode_seal(raw, day=day)

    def write(self, day: date, report: SealReport, *, failed: bool) -> None:
        encoded = encode_seal(day, report, failed=failed)
        try:
            ensure_real_directory(self.root, artifact_root=self.artifact_root)
            atomic_replace_bytes(self.path(day), encoded, artifact_root=self.artifact_root)
        except DurablePathIntegrityError as exc:
            raise SealFileError(f"seal record of {day} cannot be written: {exc}") from exc

    def prune(self, newest: date, kept_days: int) -> int:
        return prune_dated(self.root, self.artifact_root, newest - timedelta(days=kept_days), suffix=".json")


def encode_seal(day: date, report: SealReport, *, failed: bool) -> bytes:
    raw = {
        "schema": _SCHEMA,
        "day": day.isoformat(),
        "failed": failed,
        "sealed": [item.isoformat() for item in report.sealed],
        "rewritten": report.rewritten,
        "unready_days": report.unready_days,
        "unready_records": report.unready_records,
        "readers": report.readers,
    }
    encoded = json.dumps(raw, ensure_ascii=False, sort_keys=True, indent=1).encode("utf-8")
    if len(encoded) > _MAX_BYTES:
        raise SealFileError("the seal record of a generation exceeds its byte bound")
    return encoded


def decode_seal(raw: bytes, *, day: date) -> tuple[SealReport, bool]:
    try:
        data = json.loads(raw.decode("utf-8"))
        if set(data) != _KEYS:
            raise SealFileError("a seal record has unexpected fields")
        if data["schema"] != _SCHEMA or date.fromisoformat(data["day"]) != day:
            raise SealFileError("a seal record belongs to another schema or day")
        counts = [data[key] for key in ("rewritten", "unready_days", "unready_records", "readers")]
        if type(data["failed"]) is not bool or any(type(value) is not int or value < 0 for value in counts):
            raise SealFileError("a seal record has a malformed flag or count")
        report = SealReport(
            sealed=tuple(date.fromisoformat(item) for item in data["sealed"]),
            rewritten=counts[0],
            unready_days=counts[1],
            unready_records=counts[2],
            readers=counts[3],
        )
        return report, data["failed"]
    except SealFileError:
        raise
    except (UnicodeDecodeError, AttributeError, KeyError, TypeError, ValueError) as exc:  # 含 JSON 与日期的校验错
        raise SealFileError(f"seal record of {day} is corrupt: {exc}") from exc


def prune_seals(scene: object, newest: date, kept_days: int) -> None:
    """随完成标记一起清这棵语义树的封口记录（没接就没有可清的）。"""

    files = getattr(scene, "seal_files", None)
    if files is not None:
        files.prune(newest, kept_days)


__all__ = ["SealFileError", "SealFiles", "decode_seal", "encode_seal", "prune_seals"]
