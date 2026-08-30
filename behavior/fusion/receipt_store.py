"""融合回执的只创建不可变存储。

结构与 ``behavior/observation/store.py`` 一致——先解码自校验再落盘、枚举跳过非本存储命名的
条目、写后读回验证——只有撞车语义不同：

观测交付的内容是输入的纯函数，同一 ``source_id`` 撞车时内容必须逐字节相同，否则说明有人复用
了交付身份，必须失败。**融合不是纯函数**：同一批观测重跑，模型完全可能给出措辞不同的判断。
所以这里撞车时复用先到的那份，并把它返回给调用方，而不是拿新结果去覆盖或比对——重跑不该让一次
崩溃恢复变成一次硬失败，而先到的那份才是与已落盘判断对应的那份。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from behavior.fusion.config import BehaviorFusionConfig
from behavior.fusion.errors import BehaviorFusionError, BehaviorFusionLimitError
from behavior.fusion.receipt import BehaviorFusionReceipt
from foundation.integrity import canonical_json
from infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_temporary_destination,
    durable_unlink,
    list_real_directory,
    read_regular_bytes,
)

_RECEIPT_FILE = re.compile(r"^(?P<receipt_id>[0-9a-f]{64})\.json$")

# BHV-LIFECYCLE-001（回执部分已落地，2026-08-30）：覆盖信息移到 ``behavior/fusion/coverage.py``
# （按日分区、窗口过期），入队与封口不再枚举本目录；回执本身只服务作业期幂等，由归约按同一窗口
# ``expire``。下面的原始记录保留作历史：
# 回执与观测存储共用同一个生命周期缺口——``list()`` 是全量扁平枚举，
# 越过 ``max_receipt_files`` 即永久抛错，而 ``put`` 仍然一直成功。按每分钟一次融合计算，默认
# 上限约 69 天后枚举必然失效。**而重写之后它已经进了排队的关键路径**（``enqueue`` 每次扫描都要
# 全量解码回执），所以越界不再只是审计问题，是整条融合流水线停摆。改造随生命周期一并做：按本地
# 日历日分区、提供有界的时间窗读取、设定保留期。回执的释放门槛：它必须活得比它记录的判断更久，
# 否则那批判断的来源与同批读不懂的片段就再也对不上账。


class BehaviorFusionReceiptStore:
    """在 behavior-root 下按 ``receipt_id`` 保存不可变融合回执。"""

    def __init__(
        self, behavior_root: str | Path, *, config: BehaviorFusionConfig | None = None
    ) -> None:
        self.root = Path(behavior_root).expanduser().resolve(strict=False)
        resolved = config or BehaviorFusionConfig()
        if not isinstance(resolved, BehaviorFusionConfig):
            raise TypeError("config must be BehaviorFusionConfig")
        self.config = resolved
        self.receipt_root = self.root / "fusion" / "receipts"

    def put(self, receipt: BehaviorFusionReceipt) -> BehaviorFusionReceipt:
        """落盘一份回执；同一片段同版本已有回执时复用既有的那份。"""

        if not isinstance(receipt, BehaviorFusionReceipt):
            raise TypeError("receipt must be BehaviorFusionReceipt")
        encoded = self._encode(receipt)
        # 先解码自校验再落盘：写路径若比读路径宽松，"写得进读不回"的记录会永久占死该身份并让
        # 整个枚举失效，而本层没有删除接口。
        self._require_decodable(encoded)
        try:
            atomic_create_bytes(self._path(receipt.receipt_id), encoded, artifact_root=self.root)
        except ImmutableArtifactConflictError as exc:
            existing = self._existing(receipt.receipt_id, exc)
            return existing
        stored = self.read(receipt.receipt_id)
        if stored is None or stored.record_digest != receipt.record_digest:
            raise BehaviorFusionError("fusion receipt was not durably read back")
        return stored

    def read(self, receipt_id: str) -> BehaviorFusionReceipt | None:
        try:
            encoded = read_regular_bytes(
                self._path(receipt_id),
                artifact_root=self.root,
                max_bytes=self.config.max_receipt_file_bytes,
            )
        except FileNotFoundError:
            return None
        try:
            receipt = BehaviorFusionReceipt.from_dict(json.loads(encoded))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, BehaviorFusionError) as exc:
            raise BehaviorFusionError("fusion receipt record is corrupt") from exc
        if encoded != self._encode(receipt):
            raise BehaviorFusionError("fusion receipt record is not canonically encoded")
        if receipt.receipt_id != receipt_id:
            raise BehaviorFusionError("fusion receipt filename does not match receipt_id")
        return receipt

    def list(self) -> tuple[BehaviorFusionReceipt, ...]:
        try:
            entries = list_real_directory(
                self.receipt_root, artifact_root=self.root, max_entries=self.config.max_receipt_files
            )
        except DurablePathIntegrityError as exc:
            raise BehaviorFusionError(
                "fusion receipt directory is invalid or exceeds its bound"
            ) from exc
        receipt_ids: list[str] = []
        for entry in entries:
            temporary = atomic_temporary_destination(entry.name)
            if entry.is_file() and temporary is not None and _RECEIPT_FILE.fullmatch(temporary):
                continue
            match = _RECEIPT_FILE.fullmatch(entry.name)
            if not entry.is_file() or match is None:
                # 不匹配本存储命名规则的条目一定不是本存储写的；为它们整库硬失败，等于让一次
                # 偶发污染永久瘫痪枚举。本存储自己的文件若损坏仍然明确失败。
                continue
            receipt_ids.append(match.group("receipt_id"))
        receipts = tuple(self._required_read(receipt_id) for receipt_id in sorted(receipt_ids))
        return tuple(sorted(receipts, key=lambda item: (item.judged_at, item.receipt_id)))

    def discard(self, receipt_id: str) -> bool:
        """删除一份回执；不存在时幂等返回 ``False``。"""

        return durable_unlink(self._path(receipt_id), artifact_root=self.root)

    def expire(self, before: datetime) -> int:
        """删掉 ``judged_at`` 早于 ``before`` 的回执；覆盖信息另有索引，回执本身只服务作业期幂等。"""

        if not isinstance(before, datetime) or before.utcoffset() is None:
            raise TypeError("before must be a timezone-aware datetime")
        removed = 0
        for receipt in self.list():
            if receipt.judged_at < before and self.discard(receipt.receipt_id):
                removed += 1
        return removed

    def _existing(
        self, receipt_id: str, conflict: ImmutableArtifactConflictError
    ) -> BehaviorFusionReceipt:
        try:
            current = self.read(receipt_id)
        except Exception as read_error:
            raise BehaviorFusionError(
                "fusion receipt collides with an unreadable artifact"
            ) from read_error
        if current is None:
            raise BehaviorFusionError(
                "fusion receipt collides with a vanished artifact"
            ) from conflict
        return current

    def _required_read(self, receipt_id: str) -> BehaviorFusionReceipt:
        receipt = self.read(receipt_id)
        if receipt is None:
            raise BehaviorFusionError("fusion receipt disappeared during enumeration")
        return receipt

    def _path(self, receipt_id: str) -> Path:
        if not isinstance(receipt_id, str) or _RECEIPT_FILE.fullmatch(f"{receipt_id}.json") is None:
            raise BehaviorFusionError("receipt_id must be lowercase SHA-256 text")
        return self.receipt_root / f"{receipt_id}.json"

    def _require_decodable(self, encoded: bytes) -> None:
        try:
            restored = BehaviorFusionReceipt.from_dict(json.loads(encoded))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, BehaviorFusionError) as exc:
            raise BehaviorFusionError(
                "fusion receipt cannot be read back by its own reader"
            ) from exc
        if self._encode(restored) != encoded:
            raise BehaviorFusionError("fusion receipt is not canonically encoded")

    def _encode(self, receipt: BehaviorFusionReceipt) -> bytes:
        encoded = (canonical_json(receipt.to_dict()) + "\n").encode("utf-8")
        if len(encoded) > self.config.max_receipt_file_bytes:
            raise BehaviorFusionLimitError("fusion receipt exceeds its configured file bound")
        return encoded


__all__ = ["BehaviorFusionReceiptStore"]
