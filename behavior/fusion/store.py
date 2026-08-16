"""行为判断的只创建不可变存储。

判断按内容身份命名，所以同一条判断重复落盘天然幂等——不需要撞车比对，因为身份就是内容。

## 不能走 canonical_json

``foundation.integrity.canonicalize`` 会把 datetime 折成 UTC，而 ``started_at`` /
``last_observed_at`` 带的是**本地偏移**：东八区凌晨 00:30 折成 UTC 就掉到前一天，归约层按"人的
一天"做的任何聚合都会错。所以这里用固定键序 + 保留偏移的时间文本自行序列化——字节仍然确定
（``sort_keys`` + 固定分隔符），只是不折 UTC。

``judged_at`` 与 ``evidence_ready_at`` 相反，它们是"我们什么时候知道的"，没有本地日历含义，
一律归一到 UTC。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from behavior.fusion.config import BehaviorFusionConfig
from behavior.fusion.derivation import DurableJudgement, judgement_payload
from behavior.fusion.errors import BehaviorFusionError, BehaviorFusionLimitError
from infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_temporary_destination,
    list_real_directory,
    read_regular_bytes,
)

JUDGEMENT_SCHEMA_VERSION = "behavior_judgement_v1"

# 读路径按名字取这些键，所以写路径必须逐个要求它们存在——否则一条缺键的记录写得进、读不回，
# 而它会让 ``list()`` 抛裸 KeyError（连错误谱系都不在），且本层没有删除接口，枚举就此永久报废。
_JUDGEMENT_KEYS = frozenset(
    {
        "schema_version",
        "judgement_id",
        "judged_at",
        "evidence_ready_at",
        "started_at",
        "last_observed_at",
        "observation_ids",
        "source_refs",
        "subjects",
        "behavior",
        "goal",
        "summary",
        "basis",
        "status",
        "status_basis",
        "relations",
        "fusion_version",
        "prompt_version",
    }
)
_JUDGEMENT_FILE = re.compile(r"^(?P<judgement_id>[0-9a-f]{64})\.json$")

# TODO(BHV-LIFECYCLE-001): 与观测存储、回执存储同一个生命周期缺口——``list()`` 是全量扁平枚举，
# 越过 ``max_judgement_files`` 即永久抛错，而写入仍然一直成功。改造随三棵树的生命周期一并做：
# 按本地日历日分区、提供有界的时间窗读取、设定保留期。判断的释放门槛要比观测更晚：观测可在
# "已形成判断"后释放，判断必须活到归约层把它变成树内文档之后。


class BehaviorJudgementStore:
    """在 behavior-root 下按 ``judgement_id`` 保存不可变判断。"""

    def __init__(
        self, behavior_root: str | Path, *, config: BehaviorFusionConfig | None = None
    ) -> None:
        self.root = Path(behavior_root).expanduser().resolve(strict=False)
        resolved = config or BehaviorFusionConfig()
        if not isinstance(resolved, BehaviorFusionConfig):
            raise TypeError("config must be BehaviorFusionConfig")
        self.config = resolved
        self.judgement_root = self.root / "fusion" / "judgements"

    def put(self, judgement: DurableJudgement) -> None:
        """落盘一条判断；内容身份保证重复写入幂等。"""

        if not isinstance(judgement, DurableJudgement):
            raise TypeError("judgement must be DurableJudgement")
        self.put_payload(judgement_payload(judgement))

    def put_payload(self, payload: Mapping[str, Any]) -> None:
        """落盘一份已经派生好的判断记录。

        作业在检查点处暂存的就是这个形状，重放时直接从它落盘——不需要（也不能）重新派生，
        重新派生要再调一次模型。
        """

        if not isinstance(payload, Mapping) or "judgement_id" not in payload:
            raise BehaviorFusionError("a judgement payload must carry its judgement_id")
        identity = payload["judgement_id"]
        encoded = self._encode_record({"schema_version": JUDGEMENT_SCHEMA_VERSION, **dict(payload)})
        self._require_decodable(encoded, identity)
        try:
            atomic_create_bytes(self._path(identity), encoded, artifact_root=self.root)
        except ImmutableArtifactConflictError as exc:
            existing = self._read_bytes(identity)
            if existing is None or existing != encoded:
                raise BehaviorFusionError(
                    "judgement identity collides with different stored content"
                ) from exc

    def read(self, judgement_id: str) -> dict[str, Any] | None:
        encoded = self._read_bytes(judgement_id)
        if encoded is None:
            return None
        record = self._decode(encoded)
        if record["judgement_id"] != judgement_id:
            raise BehaviorFusionError("judgement filename does not match its judgement_id")
        return record

    def list(self) -> tuple[dict[str, Any], ...]:
        try:
            entries = list_real_directory(
                self.judgement_root,
                artifact_root=self.root,
                max_entries=self.config.max_judgement_files,
            )
        except DurablePathIntegrityError as exc:
            raise BehaviorFusionError(
                "judgement directory is invalid or exceeds its bound"
            ) from exc
        identities: list[str] = []
        for entry in entries:
            temporary = atomic_temporary_destination(entry.name)
            if entry.is_file() and temporary is not None and _JUDGEMENT_FILE.fullmatch(temporary):
                continue
            match = _JUDGEMENT_FILE.fullmatch(entry.name)
            if not entry.is_file() or match is None:
                # 不匹配本存储命名规则的条目一定不是本存储写的（.DS_Store、同步副本）。为它们
                # 整库硬失败，等于让一次偶发污染永久瘫痪枚举。
                continue
            identities.append(match.group("judgement_id"))
        records = [self._required_read(identity) for identity in sorted(identities)]
        # 排序键的存在性由 ``_decode`` 的形状校验保证，这里不会再抛 KeyError。
        return tuple(
            sorted(records, key=lambda item: (item["evidence_ready_at"], item["judgement_id"]))
        )

    def recent_before(
        self, moment: datetime, *, limit: int, lookback_seconds: float
    ) -> tuple[dict[str, Any], ...]:
        """取 ``moment`` 之前最近的若干条判断，按行为时间升序返回。

        截断用 ``evidence_ready_at``（这条判断何时成立）而不是 ``started_at``（行为何时发生）：
        融合一段观测时能看见的，只能是**在这段观测的证据齐备之前就已经成立**的判断。用行为时间
        截断会把"后来才知道的事"喂回给更早的那一段，那是标签泄漏。

        选取必须确定性：按 ``(evidence_ready_at, judgement_id)`` 取尾部，不按相关性排序——否则
        同一批输入会得到不同上下文，融合就不可重放。
        """

        if not isinstance(moment, datetime) or moment.utcoffset() is None:
            raise BehaviorFusionError("moment must be a timezone-aware datetime")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise BehaviorFusionError("limit must be a positive integer")
        if isinstance(lookback_seconds, bool) or not isinstance(lookback_seconds, (int, float)):
            raise BehaviorFusionError("lookback_seconds must be a number")
        if lookback_seconds <= 0:
            raise BehaviorFusionError("lookback_seconds must be positive")
        cutoff = moment.astimezone(timezone.utc)
        earliest = cutoff - timedelta(seconds=float(lookback_seconds))
        selected = [
            record
            for record in self.list()
            if earliest <= _parse_instant(record["evidence_ready_at"]) < cutoff
        ]
        tail = selected[-limit:]
        # 按**时刻**排，不能按 ``started_at`` 的字符串排：它刻意保留本地偏移，所以字符串序不是
        # 时间序。跨时区（出行）或 DST 切换时，1 小时的偏移变化就足以让两条判断的先后颠倒，
        # 而这个顺序直接决定模型看到的 C1..Cn 编号。
        return tuple(
            sorted(tail, key=lambda item: (_parse_instant(item["started_at"]), item["judgement_id"]))
        )

    def _required_read(self, judgement_id: str) -> dict[str, Any]:
        record = self.read(judgement_id)
        if record is None:
            raise BehaviorFusionError("judgement disappeared during enumeration")
        return record

    def _read_bytes(self, judgement_id: str) -> bytes | None:
        try:
            return read_regular_bytes(
                self._path(judgement_id),
                artifact_root=self.root,
                max_bytes=self.config.max_judgement_file_bytes,
            )
        except FileNotFoundError:
            return None
        except DurablePathIntegrityError as exc:
            raise BehaviorFusionError("judgement record cannot be read") from exc

    def _decode(self, encoded: bytes) -> dict[str, Any]:
        try:
            record = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise BehaviorFusionError("judgement record is corrupt") from exc
        if not isinstance(record, dict) or record.get("schema_version") != JUDGEMENT_SCHEMA_VERSION:
            raise BehaviorFusionError("judgement record schema is invalid")
        if set(record) != _JUDGEMENT_KEYS:
            missing = sorted(_JUDGEMENT_KEYS - set(record))
            unknown = sorted(set(record) - _JUDGEMENT_KEYS)
            raise BehaviorFusionError(
                f"judgement record shape is invalid: missing={missing} unknown={unknown}"
            )
        if self._encode_record(record) != encoded:
            raise BehaviorFusionError("judgement record is not canonically encoded")
        return record

    def _require_decodable(self, encoded: bytes, judgement_id: str) -> None:
        record = self._decode(encoded)
        if record["judgement_id"] != judgement_id:
            raise BehaviorFusionError("judgement record does not carry its own identity")

    def _path(self, judgement_id: str) -> Path:
        if _JUDGEMENT_FILE.fullmatch(f"{judgement_id}.json") is None:
            raise BehaviorFusionError("judgement_id must be lowercase SHA-256 text")
        return self.judgement_root / f"{judgement_id}.json"

    def _encode_record(self, record: dict[str, Any]) -> bytes:
        encoded = (
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        if len(encoded) > self.config.max_judgement_file_bytes:
            raise BehaviorFusionLimitError("judgement exceeds its configured file bound")
        return encoded


def _parse_instant(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


__all__ = ["JUDGEMENT_SCHEMA_VERSION", "BehaviorJudgementStore"]
