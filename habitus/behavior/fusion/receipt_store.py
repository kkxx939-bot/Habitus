"""融合回执的只创建不可变存储。

结构与 ``behavior/observation/store.py`` 一致——先解码自校验再落盘、枚举跳过非本存储命名的
条目、写后读回验证——只有撞车语义不同：

观测交付的内容是输入的纯函数，同一 ``source_id`` 撞车时内容必须逐字节相同，否则说明有人复用
了交付身份，必须失败。**融合不是纯函数**：同一批观测重跑，模型完全可能给出措辞不同的判断。
所以这里撞车时复用先到的那份，并把它返回给调用方，而不是拿新结果去覆盖或比对——重跑不该让一次
崩溃恢复变成一次硬失败，而先到的那份才是与已落盘判断对应的那份。

## 文件名带判定时刻

文件名是 ``<judged_at 的 UTC 时刻>_<receipt_id>.json``（如 ``20261022T013000123456Z_<64 位>.json``）。
过期只看文件名里的时刻、不解码正文——回执改版本时，盘上留着的旧版本回执不会让每一轮归约的过期卡住。
同一 ``receipt_id`` 重跑时判定时刻可能不同（夜里重试把"没读懂"的一轮重新描述），所以撞车按
``receipt_id`` 在目录里找，而不是靠路径相同。不合这个命名的条目不是本存储写的，枚举与过期都跳过。

"先找、再创建"两步之间不是原子的：同一 ``receipt_id`` 的两个写者（白天的 lane、夜里重试、会话源补救、物理
lane 重发）各在自己的时刻路径上创建，就会留下两份。所以 ``put`` 按 ``receipt_id`` 取路径锁（与行为树写入同一个
``LockStore``），找与建在同一段租约里做完。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from habitus.behavior.fusion.config import BehaviorFusionConfig
from habitus.behavior.fusion.errors import BehaviorFusionError, BehaviorFusionLimitError
from habitus.behavior.fusion.receipt import BehaviorFusionReceipt
from habitus.foundation.integrity import canonical_json
from habitus.infrastructure.store.contracts.lock import LockStore
from habitus.infrastructure.store.contracts.path_lock import PathLock
from habitus.infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_temporary_destination,
    durable_unlink,
    list_real_directory,
    read_regular_bytes,
)

_RECEIPT_FILE = re.compile(r"^(?P<moment>\d{8}T\d{12}Z)_(?P<receipt_id>[0-9a-f]{64})\.json$")
_MOMENT_FORMAT = "%Y%m%dT%H%M%S%fZ"
# 临界区只是列一次目录、写一个小文件；另一个写者拿着锁时等它做完，而不是立刻失败。
_LOCK_TTL_SECONDS = 30
_LOCK_WAIT_SECONDS = 30.0

# 生命周期：覆盖信息在 ``behavior/fusion/coverage.py``（按日分区、按每条的释放条件删），入队与封口不再枚举本目录；
# 回执本身只服务作业期幂等，由归约按覆盖窗口 ``expire``。常态下回执比它记录的判断活得久（判断发布即释放），但没有
# 强制：判断要是滞留在存储里（隔离、没归约成），它会比回执活得久——对账时以判断为准。


class BehaviorFusionReceiptStore:
    """在 behavior-root 下按判定时刻与 ``receipt_id`` 保存不可变融合回执。"""

    def __init__(
        self,
        behavior_root: str | Path,
        *,
        lock_store: LockStore,
        config: BehaviorFusionConfig | None = None,
    ) -> None:
        self.root = Path(behavior_root).expanduser().resolve(strict=False)
        resolved = config or BehaviorFusionConfig()
        if not isinstance(resolved, BehaviorFusionConfig):
            raise TypeError("config must be BehaviorFusionConfig")
        required = ("acquire", "renew", "fenced", "release", "discard_idle")
        if any(not callable(getattr(lock_store, name, None)) for name in required):
            raise TypeError("lock_store must implement the LockStore contract")
        self.config = resolved
        self.receipt_root = self.root / "fusion" / "receipts"
        self.lock_store = lock_store
        self._path_lock = PathLock(lock_store)
        self._lock_prefix = "behavior-fusion-receipt:" + hashlib.sha256(str(self.root).encode("utf-8")).hexdigest()[:24]

    def put(self, receipt: BehaviorFusionReceipt) -> BehaviorFusionReceipt:
        """落盘一份回执；同一片段同版本已有回执时复用既有的那份（不论它是哪个时刻判的）。"""

        if not isinstance(receipt, BehaviorFusionReceipt):
            raise TypeError("receipt must be BehaviorFusionReceipt")
        encoded = self._encode(receipt)
        # 先解码自校验再落盘：写路径若比读路径宽松，"写得进读不回"的记录会永久占死该身份并让
        # 整个枚举失效，而本层没有删除接口。
        self._require_decodable(encoded)
        with self._fenced(receipt.receipt_id):
            if self._located(receipt.receipt_id) is not None:
                return self._existing(receipt.receipt_id, None)
            name = self._name(receipt.judged_at, receipt.receipt_id)
            try:
                atomic_create_bytes(self.receipt_root / name, encoded, artifact_root=self.root)
            except ImmutableArtifactConflictError as exc:
                return self._existing(receipt.receipt_id, exc)
        stored = self._read_named(receipt.judged_at, receipt.receipt_id, name)
        if stored is None or stored.record_digest != receipt.record_digest:
            raise BehaviorFusionError("fusion receipt was not durably read back")
        return stored

    def read(self, receipt_id: str) -> BehaviorFusionReceipt | None:
        located = self._located(receipt_id)
        if located is None:
            return None
        return self._read_named(*located)

    def list(self) -> tuple[BehaviorFusionReceipt, ...]:
        receipts = (self._required_read(*entry) for entry in self._entries())
        return tuple(sorted(receipts, key=lambda item: (item.judged_at, item.receipt_id)))

    def _read_named(self, moment: datetime, receipt_id: str, name: str) -> BehaviorFusionReceipt | None:
        try:
            encoded = read_regular_bytes(
                self.receipt_root / name,
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
        if receipt.judged_at != moment:
            raise BehaviorFusionError("fusion receipt filename does not match judged_at")
        return receipt

    def discard(self, receipt_id: str) -> bool:
        """删除一份回执；不存在时幂等返回 ``False``。"""

        located = self._located(receipt_id)
        if located is None:
            return False
        return durable_unlink(self.receipt_root / located[2], artifact_root=self.root)

    def expire(self, before: datetime) -> int:
        """删掉文件名里判定时刻早于 ``before`` 的回执，不解码正文；覆盖信息另有索引，回执本身只服务作业期幂等。"""

        if not isinstance(before, datetime) or before.utcoffset() is None:
            raise TypeError("before must be a timezone-aware datetime")
        removed = 0
        for moment, _receipt_id, name in self._entries():
            if moment < before and durable_unlink(self.receipt_root / name, artifact_root=self.root):
                removed += 1
        return removed

    @contextmanager
    def _fenced(self, receipt_id: str) -> Iterator[None]:
        _require_identity(receipt_id)
        key = f"{self._lock_prefix}:{receipt_id}"
        with self._path_lock.acquire(
            key, ttl_seconds=_LOCK_TTL_SECONDS, wait_timeout_seconds=_LOCK_WAIT_SECONDS
        ) as guard:
            with self._path_lock.fenced((guard,)):
                yield
        # 回执只写一次：锁放了之后这一行再也用不上，空闲的删掉；别人正持有的不动
        self.lock_store.discard_idle(key)

    def _entries(self) -> tuple[tuple[datetime, str, str], ...]:
        """目录里本存储写的回执：(判定时刻, receipt_id, 文件名)，只看文件名。"""

        try:
            entries = list_real_directory(
                self.receipt_root, artifact_root=self.root, max_entries=self.config.max_receipt_files
            )
        except DurablePathIntegrityError as exc:
            raise BehaviorFusionError("fusion receipt directory is invalid or exceeds its bound") from exc
        found: dict[str, tuple[datetime, str]] = {}
        for entry in entries:
            temporary = atomic_temporary_destination(entry.name)
            if entry.is_file() and temporary is not None and _RECEIPT_FILE.fullmatch(temporary):
                continue
            match = _RECEIPT_FILE.fullmatch(entry.name)
            if not entry.is_file() or match is None:
                # 不匹配本存储命名规则的条目一定不是本存储写的；为它们整库硬失败，等于让一次
                # 偶发污染永久瘫痪枚举。本存储自己的文件若损坏仍然明确失败。
                continue
            try:
                moment = datetime.strptime(match.group("moment"), _MOMENT_FORMAT).replace(tzinfo=UTC)
            except ValueError as exc:
                raise BehaviorFusionError(f"fusion receipt filename carries an invalid moment: {entry.name}") from exc
            found[entry.name] = (moment, match.group("receipt_id"))
        return tuple((moment, identity, name) for name, (moment, identity) in sorted(found.items()))

    def _located(self, receipt_id: str) -> tuple[datetime, str, str] | None:
        _require_identity(receipt_id)
        matches = [entry for entry in self._entries() if entry[1] == receipt_id]
        if len(matches) > 1:
            raise BehaviorFusionError(f"fusion receipt {receipt_id} is stored more than once")
        return matches[0] if matches else None

    def _existing(self, receipt_id: str, conflict: ImmutableArtifactConflictError | None) -> BehaviorFusionReceipt:
        try:
            current = self.read(receipt_id)
        except Exception as read_error:
            raise BehaviorFusionError("fusion receipt collides with an unreadable artifact") from read_error
        if current is None:
            raise BehaviorFusionError("fusion receipt collides with a vanished artifact") from conflict
        return current

    def _required_read(self, moment: datetime, receipt_id: str, name: str) -> BehaviorFusionReceipt:
        receipt = self._read_named(moment, receipt_id, name)
        if receipt is None:
            raise BehaviorFusionError("fusion receipt disappeared during enumeration")
        return receipt

    @staticmethod
    def _name(judged_at: datetime, receipt_id: str) -> str:
        _require_identity(receipt_id)
        return f"{judged_at.astimezone(UTC).strftime(_MOMENT_FORMAT)}_{receipt_id}.json"

    def _require_decodable(self, encoded: bytes) -> None:
        try:
            restored = BehaviorFusionReceipt.from_dict(json.loads(encoded))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, BehaviorFusionError) as exc:
            raise BehaviorFusionError("fusion receipt cannot be read back by its own reader") from exc
        if self._encode(restored) != encoded:
            raise BehaviorFusionError("fusion receipt is not canonically encoded")

    def _encode(self, receipt: BehaviorFusionReceipt) -> bytes:
        encoded = (canonical_json(receipt.to_dict()) + "\n").encode("utf-8")
        if len(encoded) > self.config.max_receipt_file_bytes:
            raise BehaviorFusionLimitError("fusion receipt exceeds its configured file bound")
        return encoded


def _require_identity(receipt_id: object) -> None:
    if not isinstance(receipt_id, str) or re.fullmatch(r"[0-9a-f]{64}", receipt_id) is None:
        raise BehaviorFusionError("receipt_id must be lowercase SHA-256 text")


__all__ = ["BehaviorFusionReceiptStore"]
