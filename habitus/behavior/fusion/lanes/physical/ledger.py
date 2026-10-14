"""物理 lane 收过什么：上游的段编号 → 当时收下的内容；以及每块设备哪些时间收过行为、哪些时间记成了没在看。

两件事要一直成立，而凭据观测在落树之后就释放、覆盖记录过了窗口也会清，靠它们只认得出最近的：

- **同一段再送一遍不重复写、上游改判不收**：按"哪台设备的哪一段"记下当时内容的摘要
  （``fusion/physical/segments/<编号摘要>.json``）。摘要只由上游给的字段算，不含本机的时区、主体名、代码里的措辞。
- **后来的交付不许和收下的矛盾**：行为树只增不改，一条行为落在已经记成"没在看"的时段里，下游就读不了这棵树了。
  所以按设备、按日另记收过的时段（``fusion/physical/spans/<设备摘要>/<UTC 日>/<单位摘要>.json``），新交付进来时对一遍。

两份都只增不减、一个单位一个小文件，没有行为内容。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import Enum
from pathlib import Path

from habitus.behavior.fusion.errors import BehaviorFusionError
from habitus.infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    list_real_directory,
    read_regular_bytes,
)

LEDGER_SCHEMA_VERSION = "behavior_physical_segment_ledger_v2"
_MAX_ENTRY_BYTES = 4_096
_MAX_DAY_ENTRIES = 100_000


class SpanKind(str, Enum):
    """收下的一个时段是什么。"""

    SEGMENT = "segment"  # 一段行为
    UNOBSERVED = "unobserved"  # 一段没在看


@dataclass(frozen=True)
class AcceptedSpan:
    """收下的一个时段（UTC）。"""

    kind: SpanKind
    started_at: datetime
    ended_at: datetime


class PhysicalSegmentLedger:
    """按"哪台设备的哪一个单位"记下收过的内容摘要与时段。"""

    def __init__(self, behavior_root: str | Path) -> None:
        self.root = Path(behavior_root).expanduser().resolve(strict=False)
        self.segment_root = self.root / "fusion" / "physical" / "segments"
        self.span_root = self.root / "fusion" / "physical" / "spans"

    def accepted(self, device: str, unit: str) -> str | None:
        """这个单位收过没有；收过就返回当时内容的摘要。"""

        try:
            encoded = read_regular_bytes(
                self._unit_path(device, unit), artifact_root=self.root, max_bytes=_MAX_ENTRY_BYTES
            )
        except FileNotFoundError:
            return None
        digest = _decode(encoded).get("content_digest")
        if not isinstance(digest, str):
            raise BehaviorFusionError("physical segment ledger entry has no content digest")
        return digest

    def accept(self, device: str, unit: str, content_digest: str, span: AcceptedSpan) -> None:
        """记下这个单位收下了；同样内容重复记是幂等的。先记时段、后记编号：编号在，时段一定在。"""

        started, ended = _utc(span.started_at), _utc(span.ended_at)
        record = json.dumps(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "kind": span.kind.value,
                "started_at": started.isoformat(timespec="microseconds"),
                "ended_at": ended.isoformat(timespec="microseconds"),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        day = started.date()
        while day <= ended.date():
            self._create(self._span_path(device, day, unit), record)
            day += timedelta(days=1)
        self._create(
            self._unit_path(device, unit),
            json.dumps(
                {"schema_version": LEDGER_SCHEMA_VERSION, "content_digest": content_digest},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8"),
        )

    def spans(self, device: str, started_at: datetime, ended_at: datetime) -> tuple[AcceptedSpan, ...]:
        """这台设备收过的、与 ``[started_at, ended_at]`` 所在的那些 UTC 日有关的时段（去重，按开始时刻排好）。"""

        found: dict[tuple[str, datetime, datetime], AcceptedSpan] = {}
        day = _utc(started_at).date()
        while day <= _utc(ended_at).date():
            directory = self._span_directory(device, day)
            try:
                entries = list_real_directory(directory, artifact_root=self.root, max_entries=_MAX_DAY_ENTRIES)
            except FileNotFoundError:
                entries = ()
            except DurablePathIntegrityError as exc:
                raise BehaviorFusionError("physical span directory is invalid or exceeds its bound") from exc
            for entry in entries:
                if not entry.is_file() or not entry.name.endswith(".json"):
                    continue
                payload = _decode(
                    read_regular_bytes(directory / entry.name, artifact_root=self.root, max_bytes=_MAX_ENTRY_BYTES)
                )
                kind, started, ended = payload.get("kind"), payload.get("started_at"), payload.get("ended_at")
                if not isinstance(kind, str) or not isinstance(started, str) or not isinstance(ended, str):
                    raise BehaviorFusionError("physical span ledger entry is not decodable")
                try:
                    span = AcceptedSpan(SpanKind(kind), datetime.fromisoformat(started), datetime.fromisoformat(ended))
                except ValueError as exc:
                    raise BehaviorFusionError("physical span ledger entry is not decodable") from exc
                found[(span.kind.value, span.started_at, span.ended_at)] = span
            day += timedelta(days=1)
        return tuple(sorted(found.values(), key=lambda item: (item.started_at, item.ended_at, item.kind.value)))

    def _create(self, path: Path, encoded: bytes) -> None:
        try:
            atomic_create_bytes(path, encoded, artifact_root=self.root)
        except ImmutableArtifactConflictError as exc:
            existing = read_regular_bytes(path, artifact_root=self.root, max_bytes=_MAX_ENTRY_BYTES)
            if existing != encoded:
                raise BehaviorFusionError("physical ledger entry was already written with different content") from exc

    def _unit_path(self, device: str, unit: str) -> Path:
        return self.segment_root / f"{_digest(f'{device}#{unit}')}.json"

    def _span_directory(self, device: str, day: date) -> Path:
        return self.span_root / _digest(device) / day.isoformat()

    def _span_path(self, device: str, day: date, unit: str) -> Path:
        return self._span_directory(device, day) / f"{_digest(unit)}.json"


def _decode(encoded: bytes) -> dict[str, object]:
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BehaviorFusionError("physical ledger entry is not decodable") from exc
    if not isinstance(payload, dict):
        raise BehaviorFusionError("physical ledger entry must be an object")
    return payload


def _utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = ["LEDGER_SCHEMA_VERSION", "AcceptedSpan", "PhysicalSegmentLedger", "SpanKind"]
