"""每一代的节律文件（实时化阶段 2「映射按槽」）：``<scene_root>/generation/rhythms/<日子>.json``，换代的语义树一步写、
模型队列的映射读，随完成标记一起留存与清理。

常态按峰分份用的节律是历史的部分：一条记录的材料取它那一天的那一代的，所以每一代把自己算出的节律冻住一份——之后再映射
那一天的记录（补映射、重判）读的还是这一份，不随后来的序列变。只存按峰分份要的东西（概念、各个峰）；节律的密度、锚日是给
写概念用的，那边每次现算，不读这里。

没有这份文件的日子（阶段 2 之前的历史、那一代的语义树一步没做成）由调用方按截止到那一天的序列现算。文件坏了明确报错，不当
"没有"——那样会悄悄换成另一份节律。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path

from habitus.infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    atomic_replace_bytes,
    ensure_real_directory,
    read_regular_bytes,
)
from habitus.runtime.generation_trace import prune_dated
from habitus.scene.concepts.rhythm import Rhythm, RhythmPeak

_SCHEMA = "generation_rhythms_v1"
_MAX_BYTES = 8 * 1_048_576


class RhythmFileError(ValueError):
    """这一代的节律文件坏了或读写不了。"""


class RhythmFiles:
    def __init__(self, root: str | Path, *, artifact_root: str | Path) -> None:
        self.root = Path(root)
        self.artifact_root = Path(artifact_root)

    def path(self, day: date) -> Path:
        return self.root / f"{day.isoformat()}.json"

    def read(self, day: date) -> Mapping[str, Rhythm] | None:
        """第 ``day`` 代的节律（概念 → 节律，只带各个峰）；没有这份文件是 None，坏了抛 ``RhythmFileError``。"""

        try:
            raw = read_regular_bytes(self.path(day), artifact_root=self.artifact_root, max_bytes=_MAX_BYTES)
        except FileNotFoundError:
            return None
        except (OSError, DurablePathIntegrityError) as exc:
            raise RhythmFileError(f"rhythms of {day} are unreadable: {type(exc).__name__}") from exc
        return decode_rhythms(raw, day=day)

    def write(self, day: date, rhythms: Mapping[str, Rhythm]) -> None:
        encoded = encode_rhythms(day, rhythms)
        try:
            ensure_real_directory(self.root, artifact_root=self.artifact_root)
            atomic_replace_bytes(self.path(day), encoded, artifact_root=self.artifact_root)
        except DurablePathIntegrityError as exc:
            raise RhythmFileError(f"rhythms of {day} cannot be written: {exc}") from exc

    def prune(self, newest: date, kept_days: int) -> int:
        return prune_dated(self.root, self.artifact_root, newest - timedelta(days=kept_days), suffix=".json")


def encode_rhythms(day: date, rhythms: Mapping[str, Rhythm]) -> bytes:
    raw = {
        "schema": _SCHEMA,
        "day": day.isoformat(),
        "rhythms": {
            name: [[peak.ordinal, peak.start_minute, peak.end_minute, peak.share] for peak in rhythm.peaks]
            for name, rhythm in sorted(rhythms.items())
            if rhythm.peaks
        },
    }
    encoded = json.dumps(raw, ensure_ascii=False, sort_keys=True, indent=1).encode("utf-8")
    if len(encoded) > _MAX_BYTES:
        raise RhythmFileError("the rhythms of a generation exceed their byte bound")
    return encoded


def decode_rhythms(raw: bytes, *, day: date) -> Mapping[str, Rhythm]:
    try:
        data = json.loads(raw.decode("utf-8"))
        if set(data) != {"schema", "day", "rhythms"}:
            raise RhythmFileError("a rhythm file has unexpected fields")
        if data["schema"] != _SCHEMA or date.fromisoformat(data["day"]) != day:
            raise RhythmFileError("a rhythm file belongs to another schema or day")
        return {
            str(name): Rhythm(
                concept=str(name),
                peaks=tuple(
                    RhythmPeak(ordinal=int(ordinal), start_minute=int(start), end_minute=int(end), share=float(share))
                    for ordinal, start, end, share in peaks
                ),
            )
            for name, peaks in data["rhythms"].items()
        }
    except RhythmFileError:
        raise
    except (UnicodeDecodeError, AttributeError, KeyError, TypeError, ValueError) as exc:  # 含 JSON 与概念的校验错
        raise RhythmFileError(f"rhythms of {day} are corrupt: {exc}") from exc


def prune_rhythms(scene: object, newest: date, kept_days: int) -> None:
    """随完成标记一起清这棵语义树的节律文件（没接就没有可清的）。"""

    files = getattr(scene, "rhythm_files", None)
    if files is not None:
        files.prune(newest, kept_days)


__all__ = ["RhythmFileError", "RhythmFiles", "decode_rhythms", "encode_rhythms", "prune_rhythms"]
