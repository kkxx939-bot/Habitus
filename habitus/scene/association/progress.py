"""关联的耐久进度：失败记账与未发布的检查点。

形态沿归组那份刷新进度（已随归组删掉），但有两处**故意不一样**。

**没有"待办文件"。** 归组的工作集来自"已定稿的日子"，要自己记住还欠哪些；关联的待办是
``backlog()`` 每轮从树上现算的差集——出处日减去已完成的日期减去被挡住的日期——本身就是幂等的，
持久化一份反而会和树说两个话。

**检查点存已经解析完的产物，不存带编号的草稿。** 情境的编号是按输入顺序临时给的：存一份
``situation_no: 2`` 的草稿，下一轮情境列表变了、重新编号，重放就会把这条记录挂到另一种情境上，
而且没有任何地方会报错。所以这里存的是**编码后的记录本身**，重放变成纯 I/O。

键是 ``(候选, 日期)``。失败记账按输入指纹计次、指纹变了自动解封，这一条与归组相同；它同时充当
``backlog`` 的跳过源，让一件永久失败的任务不要把那个候选的配额永久吃掉。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from habitus.foundation.integrity import canonical_json
from habitus.infrastructure.store.filesystem import (
    atomic_replace_bytes,
    durable_unlink,
    ensure_real_directory,
    read_regular_bytes,
)
from habitus.scene.model import AssociationAddress, _kind_identity
from habitus.scene.regularity.document import AssociationDocument, decode, encode

_BLOCKED_NAME = "blocked.json"
_CHECKPOINT_DIRECTORY = "checkpoints"
_MAX_BLOCKED_BYTES = 4_194_304
_MAX_CHECKPOINT_BYTES = 16_777_216


class AssociationProgressError(RuntimeError):
    """耐久进度自身不可解读——这不是模型的问题，是我们写下的东西坏了。"""


@dataclass(frozen=True)
class TaskKey:
    """一件待关联的事的身份：哪个候选、哪一天。

    ``kind_token`` 归一成规范身份之后再存，与 ``AssociationAddress`` 同一条纪律：人写法不参与
    相等。不归一的话，``Gym`` 建的键与从盘上读回来的 ``gym`` 键不相等——失败次数永远停在 1、
    ``clear_failure`` 永远 pop 不中，而 ``days_for`` 却按规范身份比，于是封锁生效、解封失效。
    """

    kind_token: str
    day: date

    def __post_init__(self) -> None:
        if isinstance(self.day, datetime) or not isinstance(self.day, date):
            raise TypeError("association task day must be a date without a time")
        object.__setattr__(self, "kind_token", _kind_identity(self.kind_token))

    @property
    def identity(self) -> str:
        return f"{self.kind_token}/{self.day.isoformat()}"

    @classmethod
    def parse(cls, raw: object) -> TaskKey:
        if not isinstance(raw, str) or raw.count("/") != 1:
            raise AssociationProgressError("association task key has an invalid shape")
        kind, _slash, day = raw.partition("/")
        try:
            return cls(kind_token=kind, day=date.fromisoformat(day))
        except (TypeError, ValueError) as exc:
            raise AssociationProgressError("association task key has an invalid shape") from exc


@dataclass(frozen=True)
class FailureRecord:
    source_digest: str
    attempts: int
    error: str
    blocked: bool
    association_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_digest": self.source_digest,
            "attempts": self.attempts,
            "error": self.error,
            "blocked": self.blocked,
            "association_version": self.association_version,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FailureRecord:
        return cls(
            source_digest=str(raw.get("source_digest", "")),
            attempts=int(raw.get("attempts", 0)),
            error=str(raw.get("error", "")),
            blocked=bool(raw.get("blocked", False)),
            association_version=str(raw.get("association_version", "")),
        )


@dataclass(frozen=True)
class Checkpoint:
    """一次成功的模型调用之后、落盘之前的产物。``unanswered`` 决定要不要写完成标记。"""

    records: tuple[AssociationDocument, ...]
    unanswered: tuple[int, ...] = ()


class AssociationProgress:
    """``progress_root`` 下的失败记账与检查点。零并发假设：编排层整轮持一把租约。

    ``version`` 是当前的提示词 + schema 版本。封锁**按版本作废**：换了版本，旧版本下封住的那些
    自动重新进入待办。这不是"过期"——它是"封锁的那个理由（同一份输入、同一套提示词答不出来）
    已经不成立了"。而按输入指纹解封在规定的接线下根本够不着：被封的任务压根不会被生成成任务，
    那条分支永远跑不到。真实的解封触发就是三样：换提示词版本、调大上限、人工清账。
    """

    def __init__(self, root: str | Path, *, version: str = "") -> None:
        self.root = Path(root).expanduser().absolute()
        self.version = version

    # ── 失败记账 ─────────────────────────────────────────────────────────

    @property
    def _blocked_path(self) -> Path:
        return self.root / _BLOCKED_NAME

    def failures(self) -> dict[TaskKey, FailureRecord]:
        try:
            encoded = read_regular_bytes(self._blocked_path, artifact_root=self.root, max_bytes=_MAX_BLOCKED_BYTES)
        except FileNotFoundError:
            return {}
        try:
            raw = json.loads(encoded.decode("utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("not an object")
            return {TaskKey.parse(key): FailureRecord.from_dict(value) for key, value in raw.items()}
        except (UnicodeDecodeError, ValueError, TypeError, AttributeError) as exc:
            raise AssociationProgressError("association blocked file is not decodable") from exc

    def record_failure(
        self, task: TaskKey, digest: str, error: str, *, deterministic: bool, max_attempts: int
    ) -> FailureRecord:
        """同一输入连续失败到上限就挡住；确定性失败（超限）一次就挡住。指纹变了计数归一。"""

        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")
        failures = self.failures()
        previous = failures.get(task)
        attempts = previous.attempts + 1 if previous is not None and previous.source_digest == digest else 1
        record = FailureRecord(
            digest, attempts, error, deterministic or attempts >= max_attempts, self.version
        )
        failures[task] = record
        self._write_failures(failures)
        return record

    def clear_failure(self, task: TaskKey) -> None:
        failures = self.failures()
        if failures.pop(task, None) is not None:
            self._write_failures(failures)

    def days_for(self, kind_token: str) -> frozenset[date]:
        """``backlog`` 的跳过源：这个候选有哪些天这一轮不要再取了。"""

        wanted = _kind_identity(kind_token)
        return frozenset(
            task.day
            for task, record in self.failures().items()
            if record.blocked and record.association_version == self.version and task.kind_token == wanted
        )

    def _write_failures(self, failures: Mapping[TaskKey, FailureRecord]) -> None:
        ensure_real_directory(self.root, artifact_root=self.root)
        if not failures:
            durable_unlink(self._blocked_path, artifact_root=self.root)
            return
        body = {task.identity: record.to_dict() for task, record in sorted(failures.items(), key=_order)}
        atomic_replace_bytes(self._blocked_path, canonical_json(body).encode("utf-8"), artifact_root=self.root)

    # ── 检查点 ───────────────────────────────────────────────────────────

    def _checkpoint_path(self, task: TaskKey) -> Path:
        return self.root / _CHECKPOINT_DIRECTORY / _kind_identity(task.kind_token) / f"{task.day.isoformat()}.json"

    def write_checkpoint(self, task: TaskKey, digest: str, version: str, checkpoint: Checkpoint) -> None:
        if not isinstance(checkpoint, Checkpoint):
            raise TypeError("checkpoint must be a Checkpoint")
        path = self._checkpoint_path(task)
        ensure_real_directory(path.parent, artifact_root=self.root)
        body = {
            "task": task.identity,
            "source_digest": digest,
            "association_version": version,
            "unanswered": list(checkpoint.unanswered),
            # 叶名显式存下来，不从正文标题里反解——地址是回读比对的一端，不该靠解析另一端得到。
            "records": [
                {"identity_name": document.address.identity_name, "raw": encode(document)}
                for document in checkpoint.records
            ],
        }
        atomic_replace_bytes(path, canonical_json(body).encode("utf-8"), artifact_root=self.root)

    def load_checkpoint(self, task: TaskKey, digest: str) -> Checkpoint | None:
        """指纹不符就当没有——输入变了，上次那份产物说的是另一件事。"""

        try:
            encoded = read_regular_bytes(
                self._checkpoint_path(task), artifact_root=self.root, max_bytes=_MAX_CHECKPOINT_BYTES
            )
        except FileNotFoundError:
            return None
        try:
            raw = json.loads(encoded.decode("utf-8"))
            if not isinstance(raw, Mapping) or raw.get("source_digest") != digest:
                return None
            records = tuple(_record(task, item) for item in raw["records"])
            return Checkpoint(records=records, unanswered=tuple(int(no) for no in raw["unanswered"]))
        except (UnicodeDecodeError, ValueError, TypeError, KeyError) as exc:
            raise AssociationProgressError(f"association checkpoint for {task.identity} is not decodable") from exc

    def clear_checkpoint(self, task: TaskKey) -> None:
        durable_unlink(self._checkpoint_path(task), artifact_root=self.root)

    def unblock(self, kind_token: str | None = None) -> tuple[str, ...]:
        """运维正门：清掉封锁记账。不给候选就清全部。返回清了哪几件。"""

        failures = self.failures()
        wanted = None if kind_token is None else _kind_identity(kind_token)
        cleared = [task for task in failures if wanted is None or task.kind_token == wanted]
        for task in cleared:
            failures.pop(task)
        if cleared:
            self._write_failures(failures)
        return tuple(sorted(task.identity for task in cleared))


def _record(task: TaskKey, item: object) -> AssociationDocument:
    """地址从任务与叶名重建，再让 ``decode`` 回读比对——重放走的是与落盘完全相同的那条校验。"""

    if not isinstance(item, Mapping):
        raise ValueError("a checkpointed record must be an object")
    raw, identity_name = item["raw"], item["identity_name"]
    if not isinstance(raw, str) or not isinstance(identity_name, str):
        raise ValueError("a checkpointed record must carry its text and its leaf name")
    return decode(raw, expected_address=AssociationAddress.from_identity(task.kind_token, task.day, identity_name))


def _order(item: tuple[TaskKey, FailureRecord]) -> tuple[date, str]:
    return item[0].day, item[0].identity


__all__ = [
    "AssociationProgress",
    "AssociationProgressError",
    "Checkpoint",
    "FailureRecord",
    "TaskKey",
]
