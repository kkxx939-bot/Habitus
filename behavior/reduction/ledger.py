"""归约消费账本：哪些判断已经被物化成了哪个树节点。

一物三用：
1. **幂等重放**——账本条目按链身份 add-only 落盘（字节相同幂等成功），崩溃重试不重复归约；
2. **释放门槛**——判断存储的生命周期（``TODO(BHV-LIFECYCLE-001)`` 双消费者门槛）以"已被归约
   消费"为其中一半依据；
3. **链接解析**——晚封口的链引用早已落盘的目标时，从这里把判断身份换成树 URI。

账本只存关联不存语义（同"结算账本"纪律）；条目内容与 staged 检查点逐字节一致，重放路径上
没有任何重新计算。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from behavior.reduction.errors import BehaviorReductionError
from infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    durable_unlink,
    read_regular_bytes,
)

_MAX_ENTRY_BYTES = 1_048_576

_ENTRY_KEYS = {
    "chain_digest",
    "kind",
    "uri",
    "judgement_ids",
    "staged_at",
    "reduction_version",
}
_KINDS = {"occurrence", "gap"}


@dataclass(frozen=True)
class BehaviorReductionEntry:
    """一条链（或一段空白）的消费记录。"""

    chain_digest: str
    kind: str
    uri: str
    judgement_ids: tuple[str, ...]
    staged_at: str
    reduction_version: str

    def __post_init__(self) -> None:
        for name in ("chain_digest", "uri", "staged_at", "reduction_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise BehaviorReductionError(f"ledger entry field {name} must be non-empty text")
        if self.kind not in _KINDS:
            raise BehaviorReductionError("ledger entry kind must be occurrence or gap")
        if not self.judgement_ids or any(
            not isinstance(item, str) or not item for item in self.judgement_ids
        ):
            raise BehaviorReductionError("ledger entry must consume at least one judgement")

    def to_bytes(self) -> bytes:
        payload = {
            "chain_digest": self.chain_digest,
            "kind": self.kind,
            "uri": self.uri,
            "judgement_ids": list(self.judgement_ids),
            "staged_at": self.staged_at,
            "reduction_version": self.reduction_version,
        }
        return (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> BehaviorReductionEntry:
        if not isinstance(payload, Mapping) or set(payload) != _ENTRY_KEYS:
            raise BehaviorReductionError("ledger entry has an unexpected shape")
        judgement_ids = payload["judgement_ids"]
        if isinstance(judgement_ids, str | bytes) or not isinstance(judgement_ids, list):
            raise BehaviorReductionError("ledger entry judgement_ids must be an array")
        return cls(
            chain_digest=str(payload["chain_digest"]),
            kind=str(payload["kind"]),
            uri=str(payload["uri"]),
            judgement_ids=tuple(str(item) for item in judgement_ids),
            staged_at=str(payload["staged_at"]),
            reduction_version=str(payload["reduction_version"]),
        )


class BehaviorReductionLedger:
    """按链身份 add-only 的消费账本。"""

    def __init__(self, root: str | Path) -> None:
        resolved = Path(root).expanduser().absolute()
        self.root = resolved
        self._entries_dir = resolved / "consumed"

    def append(self, entry: BehaviorReductionEntry) -> None:
        """落一条消费记录；逐字节相同的重放幂等成功，同链不同内容立即失败。"""

        if not isinstance(entry, BehaviorReductionEntry):
            raise TypeError("entry must be BehaviorReductionEntry")
        path = self._entries_dir / f"{entry.chain_digest}.json"
        try:
            atomic_create_bytes(path, entry.to_bytes(), artifact_root=self.root)
        except (ImmutableArtifactConflictError, DurablePathIntegrityError) as exc:
            raise BehaviorReductionError(
                f"reduction ledger entry for chain {entry.chain_digest} conflicts with an "
                f"existing record"
            ) from exc

    def has(self, chain_digest: str) -> bool:
        """这条链是否已记账——发布时记命中账的幂等依据（重放同一检查点不重复记）。"""

        return (self._entries_dir / f"{chain_digest}.json").is_file()

    def load(self) -> tuple[BehaviorReductionEntry, ...]:
        """读全部消费记录，按链身份排序（确定性）。"""

        if not self._entries_dir.is_dir():
            return ()
        entries: list[BehaviorReductionEntry] = []
        for path in sorted(self._entries_dir.iterdir()):
            if not path.name.endswith(".json"):
                continue
            try:
                encoded = read_regular_bytes(
                    path, artifact_root=self.root, max_bytes=_MAX_ENTRY_BYTES
                )
            except FileNotFoundError as exc:
                raise BehaviorReductionError(
                    f"reduction ledger entry vanished: {path.name}"
                ) from exc
            try:
                payload = json.loads(encoded.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BehaviorReductionError(
                    f"reduction ledger entry is not decodable: {path.name}"
                ) from exc
            entry = BehaviorReductionEntry.from_mapping(payload)
            if f"{entry.chain_digest}.json" != path.name:
                raise BehaviorReductionError(
                    f"reduction ledger entry does not match its filename: {path.name}"
                )
            entries.append(entry)
        return tuple(entries)

    def expire(self, before: datetime) -> int:
        """删掉 ``staged_at`` 早于 ``before`` 的消费记录。

        账本三个用途都只在封口窗口附近有效：幂等重放看当轮检查点；释放门槛在判断被 discard 后
        自然成立；链接解析只指向回看窗口内的目标。窗口之外的条目只是堆积。
        """

        if not isinstance(before, datetime) or before.utcoffset() is None:
            raise TypeError("before must be a timezone-aware datetime")
        removed = 0
        for entry in self.load():
            staged_at = datetime.fromisoformat(entry.staged_at)
            if staged_at.utcoffset() is None:
                continue
            if staged_at < before and durable_unlink(
                self._entries_dir / f"{entry.chain_digest}.json", artifact_root=self.root
            ):
                removed += 1
        return removed

    def consumed_judgement_ids(self) -> frozenset[str]:
        return frozenset(
            judgement_id for entry in self.load() for judgement_id in entry.judgement_ids
        )


__all__ = ["BehaviorReductionEntry", "BehaviorReductionLedger"]
