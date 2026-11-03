"""行为观测交付的只创建不可变存储。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from habitus.behavior.observation.config import BehaviorObservationConfig
from habitus.behavior.observation.errors import BehaviorObservationError, BehaviorObservationLimitError
from habitus.behavior.observation.model import BehaviorObservationEnvelope, require_sha256
from habitus.foundation.integrity import canonical_json
from habitus.infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_temporary_destination,
    durable_unlink,
    list_real_directory,
    read_regular_bytes,
)

_SOURCE_FILE = re.compile(r"^(?P<source_id>[0-9a-f]{64})\.json$")

# 生命周期（26 号清单 L2、6.2）：观测在所属判断发布到行为树、且被回执覆盖之后由归约释放（``discard``），不按保留期堆积；
# ``list()`` 每次都列目录、按身份缓存解码好的信封（文件只创建、不改写），释放时一起淘汰。``max_files`` 是保护闸。
# 本存储自己的一份读不出来（坏了、不是规范编码、文件名对不上）：数据坏了（裁定 94 错误分类）——把坏的那一份挪进
# ``observations/isolated/``、记下身份（``take_isolated``，归约留信号报出），枚举照常往下走：它钉不住封口视界、挡不住定稿。
# 读不出它的时间，所以树上落不成「错误」空白；这一点只能报出、等人看。


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
        self.isolated_root = self.root / "observations" / "isolated"
        #: 这一次 ``take_isolated`` 以来挪进隔离区的身份（坏了挪不动的也在里面：照样跳过、每次都报）。
        self._isolated: list[str] = []
        #: 交付文件只创建、不改写：按身份缓存解码好的信封（26 号清单 6.2），释放时一起淘汰。列目录照旧每次都列。
        self._decoded: dict[str, BehaviorObservationEnvelope] = {}

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
                raise BehaviorObservationError("delivery_id conflicts with a different observation payload") from exc
            return current
        stored = self.read(envelope.source_id)
        if stored is None or stored.record_digest != envelope.record_digest:
            raise BehaviorObservationError("observation delivery was not durably read back")
        return stored

    def read(self, source_id: str) -> BehaviorObservationEnvelope | None:
        cached = self._decoded.get(source_id)
        if cached is not None:
            if self._path(source_id).is_file():
                return cached
            self._decoded.pop(source_id, None)  # 别的实例释放了它
            return None
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
        self._decoded[source_id] = envelope
        return envelope

    def list(self) -> tuple[BehaviorObservationEnvelope, ...]:
        try:
            entries = list_real_directory(
                self.envelope_root, artifact_root=self.root, max_entries=self.config.max_files
            )
        except DurablePathIntegrityError as exc:
            raise BehaviorObservationError("observation envelope directory is invalid or exceeds its bound") from exc
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
        envelopes: list[BehaviorObservationEnvelope] = []
        for source_id in sorted(source_ids):
            try:
                envelopes.append(self._required_read(source_id))
            except (BehaviorObservationError, DurablePathIntegrityError):
                if self._path(source_id).is_file():
                    self._isolate(source_id)  # 坏了：放一边，不挡后面（裁定 94 错误分类）
                    continue
                raise
        return tuple(sorted(envelopes, key=lambda item: (item.batch.started_at, item.recorded_at, item.source_id)))

    def take_isolated(self) -> tuple[str, ...]:
        """上次取走以来因为读不出而放一边的交付身份，取走即清。"""

        found, self._isolated = tuple(self._isolated), []
        return found

    def _isolate(self, source_id: str) -> None:
        """原字节复制进隔离区再删掉原件；读不出原字节（超限、不是普通文件）就留在原处，只是每次都跳过、都报。"""

        self._isolated.append(source_id)
        self._decoded.pop(source_id, None)
        try:
            raw = read_regular_bytes(
                self._path(source_id), artifact_root=self.root, max_bytes=self.config.max_file_bytes
            )
            atomic_create_bytes(self.isolated_root / f"{source_id}.json", raw, artifact_root=self.root)
        except ImmutableArtifactConflictError:
            pass  # 上一次挪了一半：副本已在
        except (OSError, DurablePathIntegrityError):
            return
        durable_unlink(self._path(source_id), artifact_root=self.root)

    def discard(self, source_id: str) -> bool:
        """释放一份交付：它的全部观测都已融合且所属判断都已发布到树。不存在时幂等返回 ``False``。"""

        self._decoded.pop(source_id, None)
        return durable_unlink(self._path(source_id), artifact_root=self.root)

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
            raise BehaviorObservationError("observation delivery cannot be read back by its own reader") from exc
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
