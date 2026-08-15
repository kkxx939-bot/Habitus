"""行为观测交付的只创建不可变存储。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from behavior.observation.config import BehaviorObservationConfig
from behavior.observation.errors import BehaviorObservationError, BehaviorObservationLimitError
from behavior.observation.model import BehaviorObservationEnvelope, require_sha256
from foundation.integrity import canonical_json
from infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_temporary_destination,
    list_real_directory,
    read_regular_bytes,
)

_SOURCE_FILE = re.compile(r"^(?P<source_id>[0-9a-f]{64})\.json$")

# TODO(BHV-FUSION-003): 跨交付的观测级去重归融合层。
# - 现状：幂等只到交付粒度（``source_id`` 由 ``observer_id`` 与 ``delivery_id`` 派生）。同一条
#   观测（``observation_id`` 完全相同）换一个 ``delivery_id`` 再送一次，就会落成两份记录。
# - 具体场景：滑窗推送是上游常态——每 30 秒推送"最近 60 秒的观测"即有 50% 重叠；agent 重启后
#   用新 ``delivery_id`` 重发；多实例同时上报。实测两个 delivery 携带同一条观测，存储里该事实
#   出现 2 次而 ``observation_id`` 只有 1 个。
# - 影响大小：中。不影响本层正确性，但融合若不归并会把同一份证据数两遍，导致同一份证据
#   被数两遍：判断的证据链里出现重复观测，或一次真实行为被切成两条判断，再往下变成预测的计数偏差。
# - 改造方案：融合层读取观测时按 ``observation_id`` 归并，同一编号只采信一次；证据引用保留全部
#   来源交付以便审计。本层不建观测级索引——那需要先知道融合的读取形态，现在设计很可能白做。
# - 时机：与事件融合同批实现。

# TODO(BHV-LIFECYCLE-001): 三棵树的生命周期尚未设计，观测存储的枚举上限是它最先暴露的症状。
# - 现状：``list()`` 越过 ``max_files`` 即永久抛错而不是返回前 N 条；``put`` 仍然一直成功；本类
#   只有 ``put`` / ``list`` / ``read``，没有任何删除、归档或分区接口。实测把上限设为 3 时，第 4 次
#   交付后 ``list()`` 永久失败，而写入不受影响。
# - 具体场景：默认 ``max_files=100_000``，按每分钟一次交付计算约 69 天后 ``list()`` 必然失效。这
#   不是攻击，是日程表。此外即使未到上限，``list()`` 是全量扁平枚举并逐个解码校验，而融合的第一
#   步就是"取某个时间窗的观测"——缺少按时间的分区或索引。
# - 影响大小：高（必然发生且模块内无法恢复），但不应在本层单独修：``behavior/`` 行为树与
#   ``prediction/`` 预测树同样没有任何保留期、归档或清理设施（两者 grep 均无 lifecycle/retention/
#   prune 实现），而 ``memory/`` 已经有 ``MemoryLifecycleManager`` 与 ``LifecycleWorker`` 的完整先例。
# - 改造方案：按 memory 已验证的形态统一设计三棵树的生命周期——保留期、跨层压缩、释放门槛与独立
#   维护 Worker；观测的释放门槛是"该段观测已确证形成判断"，行为树与预测树各有自己的门槛。同时
#   把枚举改为按时间分区（与行为树 ``events/YYYY/MM/DD`` 对齐）并提供有界的时间窗读取。
# - 时机：与 behavior 接入 Runtime 组合根（``TODO(BHV-RUNTIME-001)``）同批；在此之前观测一律保留，
#   因为在融合能证明某段观测已形成 Event 之前删除它们，会造成无法重建的事实缺口。


class BehaviorObservationStore:
    """在 behavior-root 下按 ``source_id`` 保存不可变观测交付。

    同一观测者用同一 ``delivery_id`` 重投会得到同一 ``source_id``：撞车时比对内容摘要，
    一致就复用既有文件，因此重复交付天然幂等，不同内容复用同一交付身份则明确失败。
    """

    def __init__(self, behavior_root: str | Path, *, config: BehaviorObservationConfig | None = None) -> None:
        self.root = Path(behavior_root).expanduser().resolve(strict=False)
        resolved = config or BehaviorObservationConfig()
        if not isinstance(resolved, BehaviorObservationConfig):
            raise TypeError("config must be BehaviorObservationConfig")
        self.config = resolved
        self.envelope_root = self.root / "observations" / "envelopes"

    def put(self, envelope: BehaviorObservationEnvelope) -> BehaviorObservationEnvelope:
        if not isinstance(envelope, BehaviorObservationEnvelope):
            raise TypeError("envelope must be BehaviorObservationEnvelope")
        encoded = self._encode(envelope)
        # 先解码自校验再落盘：写路径若比读路径宽松，任何"写得进读不回"的记录都会永久留在
        # 磁盘上，占死该身份并让整个枚举失效——而本层没有删除接口。宁可在落盘前失败。
        self._require_decodable(encoded)
        try:
            atomic_create_bytes(self._path(envelope.source_id), encoded, artifact_root=self.root)
        except ImmutableArtifactConflictError as exc:
            try:
                current = self.read(envelope.source_id)
            except Exception as read_error:
                raise BehaviorObservationError(
                    "observation delivery collides with an unreadable artifact"
                ) from read_error
            if current is None or current.payload_digest != envelope.payload_digest:
                raise BehaviorObservationError(
                    "delivery_id conflicts with a different observation payload"
                ) from exc
            return current
        stored = self.read(envelope.source_id)
        if stored is None or stored.record_digest != envelope.record_digest:
            raise BehaviorObservationError("observation delivery was not durably read back")
        return stored

    def read(self, source_id: str) -> BehaviorObservationEnvelope | None:
        try:
            encoded = read_regular_bytes(
                self._path(source_id), artifact_root=self.root, max_bytes=self.config.max_file_bytes
            )
        except FileNotFoundError:
            return None
        try:
            envelope = BehaviorObservationEnvelope.from_dict(json.loads(encoded), config=self.config)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, BehaviorObservationError) as exc:
            raise BehaviorObservationError("observation delivery record is corrupt") from exc
        if encoded != self._encode(envelope):
            raise BehaviorObservationError("observation delivery record is not canonically encoded")
        if envelope.source_id != source_id:
            raise BehaviorObservationError("observation delivery filename does not match source_id")
        return envelope

    def list(self) -> tuple[BehaviorObservationEnvelope, ...]:
        try:
            entries = list_real_directory(
                self.envelope_root, artifact_root=self.root, max_entries=self.config.max_files
            )
        except DurablePathIntegrityError as exc:
            raise BehaviorObservationError(
                "observation envelope directory is invalid or exceeds its bound"
            ) from exc
        source_ids: list[str] = []
        for entry in entries:
            temporary = atomic_temporary_destination(entry.name)
            if entry.is_file() and temporary is not None and _SOURCE_FILE.fullmatch(temporary) is not None:
                continue
            match = _SOURCE_FILE.fullmatch(entry.name)
            if not entry.is_file() or match is None:
                # 不匹配本存储命名规则的条目一定不是本存储写的（.DS_Store、备份工具残留、
                # 云同步副本）。为它们整库硬失败，等于让一次偶发污染永久瘫痪枚举；本存储
                # 自己的文件若损坏仍然明确失败，不在这里被掩盖。
                continue
            source_ids.append(match.group("source_id"))
        envelopes = tuple(self._required_read(source_id) for source_id in sorted(source_ids))
        return tuple(
            sorted(envelopes, key=lambda item: (item.batch.started_at, item.recorded_at, item.source_id))
        )

    def _required_read(self, source_id: str) -> BehaviorObservationEnvelope:
        envelope = self.read(source_id)
        if envelope is None:
            raise BehaviorObservationError("observation delivery disappeared during enumeration")
        return envelope

    def _path(self, source_id: str) -> Path:
        require_sha256(source_id, "source_id")
        return self.envelope_root / f"{source_id}.json"

    def _require_decodable(self, encoded: bytes) -> None:
        """确认这份字节能被读路径原样接受；不能就在落盘前失败。"""

        try:
            restored = BehaviorObservationEnvelope.from_dict(json.loads(encoded), config=self.config)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, BehaviorObservationError) as exc:
            raise BehaviorObservationError(
                "observation delivery cannot be read back by its own reader"
            ) from exc
        if self._encode(restored) != encoded:
            raise BehaviorObservationError("observation delivery is not canonically encoded")

    def _encode(self, envelope: BehaviorObservationEnvelope) -> bytes:
        if len(envelope.batch.observations) > self.config.max_observations_per_batch:
            raise BehaviorObservationLimitError("observation batch exceeds its configured item limit")
        encoded = (canonical_json(envelope.to_dict()) + "\n").encode("utf-8")
        if len(encoded) > self.config.max_file_bytes:
            raise BehaviorObservationLimitError("observation delivery exceeds its configured file bound")
        return encoded


__all__ = ["BehaviorObservationStore"]
