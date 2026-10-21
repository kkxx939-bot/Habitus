"""重试堆的落盘（裁定 67、68）：一项一个文件，``<支的根>/retry/backlog/<身份>.json``；"重试到哪一天"记在同目录的
``_through.json``。

- **按天走**（裁定 68 第 3 条）：每项属于一天（主体本地日）；那一天的夜批跑过之后（``advance``），再来属于那一天的项
  一律不收（``LATE``）——当天重试、当天废弃，没有遗漏到第二天的。
- ``put``：同身份幂等（``PRESENT``）；满了（``max_items``，保护闸）新来的拒（``FULL``）——不挤旧的：旧的当晚就要试。
  拒收的由调用方计数、留信号。
- ``snapshot(through=)``：属于 ``through`` 那天及以前的项（按进来的先后）；读不出的文件也列出来，一起删。名字不是
  本堆格式的文件不认识、不碰。
- ``discard``：删给定的那几项；``snapshot`` 之后新进来的不受影响。
- 拒收的（满了 / 那天已过）累计在 ``_refused.json``，健康面读（``refused``）：堆满丢了几轮要数得到（裁定 67 第 8、12 条）。
- 列目录时撞上别处正在写的临时文件（目录在枚举中变了）就重列，几次都撞上才抛。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from habitus.behavior.retry.model import Admission, RetryError, RetryItem
from habitus.infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_replace_bytes,
    durable_unlink,
    list_real_directory,
    read_regular_bytes,
)

_DIRECTORY = ("retry", "backlog")
_THROUGH = "_through.json"
_REFUSED = "_refused.json"
_ITEM_NAME = re.compile(r"[0-9a-f]{32}\.json")
_LIST_ATTEMPTS = 5


@dataclass(frozen=True)
class RetryBatch:
    """一晚取出的那一批：读得出的项，与读不出的文件身份（一起删、计进确定性失败）。"""

    items: tuple[RetryItem, ...]
    unreadable: tuple[str, ...] = ()

    @property
    def ids(self) -> tuple[str, ...]:
        return (*(item.item_id for item in self.items), *self.unreadable)


class RetryBacklogStore:
    def __init__(self, root: str | Path, *, max_items: int, max_item_bytes: int = 1_048_576) -> None:
        for label, value in (("max_items", max_items), ("max_item_bytes", max_item_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self.root = Path(root).expanduser().absolute()
        self.directory = self.root.joinpath(*_DIRECTORY)
        self.max_items = max_items
        self.max_item_bytes = max_item_bytes

    def put(self, item: RetryItem) -> Admission:
        if not isinstance(item, RetryItem):
            raise TypeError("item must be RetryItem")
        through = self.through()
        if through is not None and item.day <= through:
            return self._refuse(Admission.LATE)
        path = self._path(item.item_id)
        if path.is_file():
            return Admission.PRESENT
        encoded = item.to_bytes()
        if len(encoded) > self.max_item_bytes or self.count() >= self.max_items:
            return self._refuse(Admission.FULL)
        try:
            atomic_create_bytes(path, encoded, artifact_root=self.root)
        except (ImmutableArtifactConflictError, DurablePathIntegrityError):
            return Admission.PRESENT if path.is_file() else Admission.FULL  # 同一项并发进来：先到的那份就是它
        return Admission.KEPT

    def refused(self) -> dict[str, int]:
        """累计拒收了几项：``{"full": …, "late": …}``。读不出当没有。"""

        raw = self._read_json(_REFUSED)
        return {key: _count(raw.get(key)) for key in (Admission.FULL.value, Admission.LATE.value)}

    def _refuse(self, admission: Admission) -> Admission:
        counts = self.refused()
        counts[admission.value] += 1
        atomic_replace_bytes(
            self.directory / _REFUSED, json.dumps(counts, sort_keys=True).encode("utf-8"), artifact_root=self.root
        )
        return admission

    def _read_json(self, name: str) -> dict[str, object]:
        try:
            encoded = read_regular_bytes(self.directory / name, artifact_root=self.root, max_bytes=4_096)
            raw = json.loads(encoded.decode("utf-8"))
        except FileNotFoundError:
            return {}
        except (ValueError, UnicodeDecodeError, DurablePathIntegrityError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def through(self) -> date | None:
        """重试到了哪一天（那一天及以前的不再收）；还没跑过为 None。读不出当没跑过。"""

        value = self._read_json(_THROUGH).get("through")
        try:
            return None if value is None else date.fromisoformat(str(value))
        except ValueError:
            return None

    def advance(self, through: date) -> None:
        """这一天（及以前）的夜批跑过了：之后属于这些天的项不再收。只往前推。"""

        current = self.through()
        if current is not None and through <= current:
            return
        payload = json.dumps({"through": through.isoformat()}).encode("utf-8")
        atomic_replace_bytes(self.directory / _THROUGH, payload, artifact_root=self.root)

    def snapshot(self, *, through: date | None = None) -> RetryBatch:
        items: list[RetryItem] = []
        unreadable: list[str] = []
        for name in self._names():
            identity = name.removesuffix(".json")
            try:
                encoded = read_regular_bytes(
                    self.directory / name, artifact_root=self.root, max_bytes=self.max_item_bytes
                )
                item = RetryItem.from_mapping(json.loads(encoded.decode("utf-8")))
            except FileNotFoundError:
                continue
            except (RetryError, UnicodeDecodeError, json.JSONDecodeError, ValueError, DurablePathIntegrityError):
                unreadable.append(identity)
                continue
            if item.item_id != identity:
                unreadable.append(identity)
                continue
            if through is None or item.day <= through:
                items.append(item)
        items.sort(key=lambda item: (item.entered_at, item.item_id))
        return RetryBatch(tuple(items), tuple(sorted(unreadable)))

    def discard(self, ids: Iterable[str]) -> int:
        return sum(1 for identity in ids if durable_unlink(self._path(identity), artifact_root=self.root))

    def count(self) -> int:
        return len(self._names())

    def _names(self) -> tuple[str, ...]:
        if not self.directory.is_dir():
            return ()
        for attempt in range(_LIST_ATTEMPTS):
            try:
                entries = list_real_directory(
                    self.directory, artifact_root=self.root, max_entries=self.max_items * 2 + 16
                )
            except DurablePathIntegrityError:
                if attempt + 1 == _LIST_ATTEMPTS:
                    raise
                continue  # 别处正在写（临时文件来了又走）：重列
            return tuple(
                sorted(entry.name for entry in entries if entry.is_file() and _ITEM_NAME.fullmatch(entry.name))
            )
        return ()

    def _path(self, identity: str) -> Path:
        if not _ITEM_NAME.fullmatch(f"{identity}.json"):
            raise RetryError("retry identity must be 32 lowercase hexadecimal characters")
        return self.directory / f"{identity}.json"


def _count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


__all__ = ["RetryBacklogStore", "RetryBatch"]
