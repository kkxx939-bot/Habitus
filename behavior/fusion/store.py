"""行为判断存储：内容身份命名、只创建不覆盖，**归约发布后释放**。

判断按内容身份命名，所以同一条判断重复落盘天然幂等——不需要撞车比对，因为身份就是内容。

## 生命周期：发布即删

判断是原料不是数据——真正的数据只在行为树上。一条判断被归约链发布成 occurrence/gap、消费账本
写完之后，代码里再没有任何读者（融合上下文只回看封口窗口内的、归约只读未消费的），于是由归约
在同一轮里 ``discard``。本存储因此只保有**尚未封口的最近一个窗口**，全量枚举本来就该是这个规模；
"当日实况"的读者已封口部分读树、未封口部分读这里，不得假设这里有全天。

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
from datetime import UTC, datetime, timedelta
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
    durable_unlink,
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

# BHV-LIFECYCLE-001（判断部分已落地，2026-08-30）：释放门槛不是保留期而是**消费**——归约发布
# 后 ``discard``；"当日实况"改为已封口读树、未封口读这里，双消费者门槛随之消失。观测同理在链
# 发布后释放（``behavior/reduction/runner.py``），"已释放的观测均有覆盖记录"由融合覆盖索引
# （``behavior/fusion/coverage.py``，写在回执落盘同一步）保证。容量悬崖的其余两处（kinds 词表、
# 树单日目录）见 ``behavior/fusion/__init__.py`` 的 TODO(BHV-REALDATA-001)。


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
        self,
        moment: datetime,
        *,
        limit: int,
        lookback_seconds: float,
        judged_before: datetime | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """本段之前已经成立的最近 ``limit`` 条判断（按 started_at 升序）。

        "之前"按**事件时间**判：判断的 ``last_observed_at <= moment``（本段最早观测的发生时刻）。
        事件时间上晚于本段的判断绝不可见——那是把后来才知道的事喂回给更早的一段（标签泄漏，
        补发的旧段尤其如此）。不再按送达时刻（``evidence_ready_at``）严格小于截断：真实数据里视觉/
        转写混合、2 秒抽样，前一段末条的送达常晚于本段首条，按送达截断会系统性丢掉"被切段拦腰
        切断的前半截"（BHV-REALDATA-001 审计复现）。``judged_before`` 给了再加一道"已落盘"（融合串行，
        本作业开始前落盘的才算成立）。

        回看窗口同样按事件时间，且与截断用**同一个字段**：``last_observed_at >= moment - lookback``
        （"这条判断在过去 lookback 内还有观测"）。不能用 ``started_at`` 作下界——做饭、看电影这类
        超过 lookback 的长行为，其后续段会看不到自己的前半截、无法 continues（审计 NEW-4 复现）。选取必须确定性：按
        ``(last_observed_at, started_at, judgement_id)`` 取尾部——并列时不能退化成按哈希随机抽样。
        """

        if not isinstance(moment, datetime) or moment.utcoffset() is None:
            raise BehaviorFusionError("moment must be a timezone-aware datetime")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise BehaviorFusionError("limit must be a positive integer")
        if isinstance(lookback_seconds, bool) or not isinstance(lookback_seconds, (int, float)):
            raise BehaviorFusionError("lookback_seconds must be a number")
        if lookback_seconds <= 0:
            raise BehaviorFusionError("lookback_seconds must be positive")
        if judged_before is not None and (
            not isinstance(judged_before, datetime) or judged_before.utcoffset() is None
        ):
            raise BehaviorFusionError("judged_before must be a timezone-aware datetime")
        cutoff = moment.astimezone(UTC)
        earliest = cutoff - timedelta(seconds=float(lookback_seconds))
        settled = None if judged_before is None else judged_before.astimezone(UTC)

        def eligible(record: Mapping[str, Any]) -> bool:
            if _parse_instant(record["last_observed_at"]) > cutoff:
                return False
            if _parse_instant(record["last_observed_at"]) < earliest:
                return False
            return settled is None or _parse_instant(record["judged_at"]) <= settled

        selected = sorted(
            (record for record in self.list() if eligible(record)),
            key=lambda item: (
                _parse_instant(item["last_observed_at"]),
                _parse_instant(item["started_at"]),
                item["judgement_id"],
            ),
        )
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

    def discard(self, judgement_id: str) -> bool:
        """释放一条已被归约消费的判断；不存在时幂等返回 ``False``。"""

        return durable_unlink(self._path(judgement_id), artifact_root=self.root)

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
