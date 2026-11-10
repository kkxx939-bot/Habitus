"""换代的完成标记（实时化方案第 3 节）：``<scene_root>/generation/<日子>.json``，一天一份，原子写入。取代夜线的 Job 账。

标记存在、读得出来就是"这一天的换代做完了"：同一代不再做（中途被杀的那一代没有标记，下一眼从头做）。标记坏了明确报错，
不当"做过"（换代把它当没做、重做时整份重写；健康面报出）。内容只放给健康面与报告看的东西：
日子、完成时刻、这一代用的序列截止日 / 词表版本 / 各 lane 关系表那一晚 / 已盖章天数、各步的结局、各个退路的计数。

**"这天处理了没有"**（``finished``，结算的后补截止读它，三值照旧）：有标记 True；晚于留存里第一个标记、没有标记 False；更早
（或一个标记都没有）None——认不出，调用方各按各的保守口径。问到的那一天的标记要读出、校验，坏了抛错。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from habitus.infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    atomic_replace_bytes,
    atomic_temporary_destination,
    durable_unlink,
    ensure_real_directory,
    list_real_directory,
    read_regular_bytes,
    real_directory_exists,
)

_SCHEMA = "generation_mark_v1"
_MAX_BYTES = 1_048_576
_MAX_FILES = 8_192
_KEYS = frozenset(
    {
        "schema",
        "day",
        "finished_at",
        "series_cutoff",
        "vocabulary_version",
        "relation_nights",
        "stamped_days",
        "steps",
        "fallbacks",
        "problems",
    }
)


class GenerationMarkError(ValueError):
    """完成标记坏了或读不了（格式不对、越界、文件系统出错）：明确报出，不当成"没做"或"做过"。"""


@dataclass(frozen=True)
class GenerationMark:
    """一天的完成标记。``steps`` 是每一步的结局（``done`` / ``degraded`` / ``skipped``）与原因，``fallbacks`` 是各个退路的计数，
    ``problems`` 是要人看的（重试收尾之后仍「错误·看不到」的计数这类）。"""

    day: date
    finished_at: datetime
    series_cutoff: date | None = None
    vocabulary_version: int | None = None
    relation_nights: Mapping[str, date | None] = field(default_factory=dict)
    stamped_days: int | None = None
    steps: Mapping[str, Mapping[str, str | None]] = field(default_factory=dict)
    fallbacks: Mapping[str, int] = field(default_factory=dict)
    problems: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.finished_at.utcoffset() is None:
            raise GenerationMarkError("a generation mark needs a timezone-aware finish time")


class GenerationMarks:
    """完成标记的落盘：一天一份文件（目录里的子目录——报告、trace——不是标记）。"""

    def __init__(self, root: str | Path, *, artifact_root: str | Path) -> None:
        self.root = Path(root)
        self.artifact_root = Path(artifact_root)

    def path(self, day: date) -> Path:
        return self.root / f"{day.isoformat()}.json"

    def days(self) -> tuple[date, ...]:
        """有标记的日子（升序）。目录读不了抛 ``GenerationMarkError``。"""

        try:
            if not real_directory_exists(self.root, artifact_root=self.artifact_root):
                return ()
            entries = list_real_directory(self.root, artifact_root=self.artifact_root, max_entries=_MAX_FILES)
        except (OSError, DurablePathIntegrityError) as exc:
            raise GenerationMarkError(f"generation marks are unreadable: {type(exc).__name__}") from exc
        found: list[date] = []
        for entry in entries:
            name = entry.name
            if (
                not entry.is_file()
                or name.startswith(".")
                or atomic_temporary_destination(name) is not None
                or not name.endswith(".json")
            ):
                continue
            try:
                day = date.fromisoformat(name[:-5])
            except ValueError:
                continue  # 不是按日子命名的文件不是标记
            if name == f"{day.isoformat()}.json":
                found.append(day)
        return tuple(sorted(found))

    def exists(self, day: date) -> bool:
        """这一天有没有完成标记；有就读出、校验——坏的抛 ``GenerationMarkError``（不吞成"做过"，F6）。"""

        return day in self.days() and self.read(day) is not None

    def finished(self, day: date) -> bool | None:
        """这一天的换代做完了没有（见模块文档串的三值口径）。目录读不了、这一天的标记坏了抛 ``GenerationMarkError``——不吞成
        "做过"或"没做"：那会让结算把它认作没有这一代的那天、截断后补，或让预测拍拿没完成的那一代建预期。"""

        days = self.days()
        if day in days:
            return self.exists(day)
        if not days or day < days[0]:
            return None
        return False

    def read(self, day: date) -> GenerationMark | None:
        """这一天的标记；没有是 None，坏了抛 ``GenerationMarkError``。"""

        try:
            raw = read_regular_bytes(self.path(day), artifact_root=self.artifact_root, max_bytes=_MAX_BYTES)
        except FileNotFoundError:
            return None
        except (OSError, DurablePathIntegrityError) as exc:
            raise GenerationMarkError(f"generation mark of {day} is unreadable: {type(exc).__name__}") from exc
        return decode_mark(raw, day=day)

    def latest(self) -> GenerationMark | None:
        days = self.days()
        return None if not days else self.read(days[-1])

    def write(self, mark: GenerationMark) -> None:
        encoded = encode_mark(mark)
        try:
            ensure_real_directory(self.root, artifact_root=self.artifact_root)
            atomic_replace_bytes(self.path(mark.day), encoded, artifact_root=self.artifact_root)
        except DurablePathIntegrityError as exc:
            raise GenerationMarkError(f"generation mark of {mark.day} cannot be written: {exc}") from exc
        if self.read(mark.day) != mark:
            raise GenerationMarkError(f"generation mark of {mark.day} failed read-back verification")

    def prune(self, newest: date, kept_days: int) -> int:
        """比 ``newest`` 早 ``kept_days`` 天以上的标记删掉。"""

        floor = newest - timedelta(days=kept_days)
        removed = 0
        for day in self.days():
            if day < floor and durable_unlink(self.path(day), artifact_root=self.artifact_root):
                removed += 1
        return removed


def encode_mark(mark: GenerationMark) -> bytes:
    raw = {
        "schema": _SCHEMA,
        "day": mark.day.isoformat(),
        "finished_at": mark.finished_at.isoformat(),
        "series_cutoff": None if mark.series_cutoff is None else mark.series_cutoff.isoformat(),
        "vocabulary_version": mark.vocabulary_version,
        "relation_nights": {
            lane: None if night is None else night.isoformat() for lane, night in sorted(mark.relation_nights.items())
        },
        "stamped_days": mark.stamped_days,
        "steps": {step: dict(value) for step, value in mark.steps.items()},
        "fallbacks": dict(sorted(mark.fallbacks.items())),
        "problems": list(mark.problems),
    }
    encoded = json.dumps(raw, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    if len(encoded) > _MAX_BYTES:
        raise GenerationMarkError("generation mark exceeds its byte bound")
    return encoded


def decode_mark(raw: bytes, *, day: date) -> GenerationMark:
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_keys)
        if not isinstance(data, dict) or set(data) != _KEYS:
            raise GenerationMarkError("generation mark has unexpected fields")
        if data["schema"] != _SCHEMA or date.fromisoformat(data["day"]) != day:
            raise GenerationMarkError("generation mark belongs to another schema or day")
        nights, steps, fallbacks, problems = data["relation_nights"], data["steps"], data["fallbacks"], data["problems"]
        if not isinstance(nights, dict) or not isinstance(steps, dict) or not isinstance(fallbacks, dict):
            raise GenerationMarkError("generation mark sections must be objects")
        if not isinstance(problems, list) or not all(isinstance(item, str) for item in problems):
            raise GenerationMarkError("generation mark problems must be a list of text")
        version, stamped = data["vocabulary_version"], data["stamped_days"]
        for number in (version, stamped):
            if number is not None and (isinstance(number, bool) or not isinstance(number, int)):
                raise GenerationMarkError("generation mark counters must be integers")
        decoded_steps: dict[str, dict[str, str | None]] = {}
        for step, value in steps.items():
            if not isinstance(value, dict) or not all(
                isinstance(key, str) and (item is None or isinstance(item, str)) for key, item in value.items()
            ):
                raise GenerationMarkError("generation mark steps must map text to text")
            decoded_steps[str(step)] = dict(value)
        decoded_fallbacks: dict[str, int] = {}
        for kind, count in fallbacks.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise GenerationMarkError("generation mark fallbacks must be non-negative integers")
            decoded_fallbacks[str(kind)] = count
        cutoff = data["series_cutoff"]
        return GenerationMark(
            day=day,
            finished_at=datetime.fromisoformat(data["finished_at"]),
            series_cutoff=None if cutoff is None else date.fromisoformat(cutoff),
            vocabulary_version=version,
            relation_nights={
                str(lane): None if night is None else date.fromisoformat(str(night)) for lane, night in nights.items()
            },
            stamped_days=stamped,
            steps=decoded_steps,
            fallbacks=decoded_fallbacks,
            problems=tuple(problems),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, GenerationMarkError):
            raise
        raise GenerationMarkError(f"generation mark of {day} is corrupt: {exc}") from exc


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """JSON 对象里同一个键写了两次：不收（不让后一个悄悄盖掉前一个，F7）。"""

    found: dict[str, object] = {}
    for key, value in pairs:
        if key in found:
            raise GenerationMarkError(f"generation mark repeats the key {key!r}")
        found[key] = value
    return found


__all__ = ["GenerationMark", "GenerationMarkError", "GenerationMarks", "decode_mark", "encode_mark"]
