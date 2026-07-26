"""一次记忆解析周期内使用的临时页面编号映射。"""

from __future__ import annotations

from collections.abc import Iterable

from infrastructure.editor.snapshot import SnapshotBatch
from memory.document import MemoryDocument
from memory.uri import MemoryURI

EXISTING_PAGE_ID_MIN = 1
EXISTING_PAGE_ID_MAX = 99
NEW_PAGE_ID_MIN = 100


class MemoryPageIdError(ValueError):
    """临时页面编号不满足当前解析周期的身份约束。"""


class MemoryPageIdMap:
    """保存临时 ``page_id`` 到最终 L2 URI 的解析映射。

    映射只服务一次结构化解析，不属于记忆文档，也不得跨批次复用。旧节点
    只有一个系统编号；节点合并完成后，可以有多个新编号指向同一最终 URI。
    """

    def __init__(self) -> None:
        self._page_to_uri: dict[int, str] = {}
        self._uri_to_pages: dict[str, set[int]] = {}
        self._next_existing_page_id = EXISTING_PAGE_ID_MIN

    @classmethod
    def from_snapshots(cls, snapshots: SnapshotBatch[MemoryDocument]) -> MemoryPageIdMap:
        """按快照身份的稳定顺序为所有已存在旧记忆分配编号。"""

        if not isinstance(snapshots, SnapshotBatch):
            raise TypeError("snapshots must be a SnapshotBatch")
        result = cls()
        for snapshot in snapshots.snapshots:
            if not snapshot.exists:
                continue
            if not isinstance(snapshot.value, MemoryDocument):
                raise MemoryPageIdError("existing memory snapshot has an invalid value")
            document_uri = str(MemoryURI.from_address(snapshot.value.address))
            if snapshot.identity != document_uri:
                raise MemoryPageIdError("existing memory snapshot identity does not match its document")
            result.register_existing(snapshot.identity)
        return result

    def register_existing(self, uri: MemoryURI | str) -> int:
        """注册一个完整旧记忆 URI，重复注册同一 URI 时返回原编号。"""

        normalized = self._document_uri(uri)
        existing = self._uri_to_pages.get(normalized, set())
        existing_page_ids = [page_id for page_id in existing if page_id <= EXISTING_PAGE_ID_MAX]
        if existing_page_ids:
            return min(existing_page_ids)
        if self._next_existing_page_id > EXISTING_PAGE_ID_MAX:
            raise MemoryPageIdError("existing memory page_id range is exhausted")
        page_id = self._next_existing_page_id
        self._next_existing_page_id += 1
        self._bind(page_id, normalized)
        return page_id

    def register_new(self, uri: MemoryURI | str, page_id: int) -> None:
        """在最终 URI 已确定后绑定模型声明的新节点编号。"""

        normalized_page_id = validate_new_page_id(page_id)
        normalized_uri = self._document_uri(uri)
        self._bind(normalized_page_id, normalized_uri)

    def register_resolved(self, uri: MemoryURI | str, page_id: int) -> None:
        """绑定节点决策产生的最终 URI，并保护旧编号不被重定向。"""

        normalized_page_id = validate_page_id(page_id)
        normalized_uri = self._document_uri(uri)
        if normalized_page_id <= EXISTING_PAGE_ID_MAX:
            existing_uri = self._page_to_uri.get(normalized_page_id)
            if existing_uri is None:
                raise MemoryPageIdError("existing page_id was not registered from old-memory snapshots")
            if existing_uri != normalized_uri:
                raise MemoryPageIdError("existing page_id cannot be redirected to another memory URI")
            return
        self._bind(normalized_page_id, normalized_uri)

    def copy(self) -> MemoryPageIdMap:
        """复制当前解析状态，供后续阶段无副作用地补充新节点绑定。"""

        result = MemoryPageIdMap()
        result._page_to_uri = dict(self._page_to_uri)
        result._uri_to_pages = {uri: set(page_ids) for uri, page_ids in self._uri_to_pages.items()}
        result._next_existing_page_id = self._next_existing_page_id
        return result

    def resolve(self, page_id: int) -> str | None:
        """返回编号对应的规范 L2 URI；尚未绑定时返回 ``None``。"""

        normalized = validate_page_id(page_id)
        return self._page_to_uri.get(normalized)

    def page_id_for(self, uri: MemoryURI | str) -> int | None:
        """返回规范 L2 URI 的临时编号；未注册时返回 ``None``。"""

        page_ids = self._uri_to_pages.get(self._document_uri(uri))
        return min(page_ids) if page_ids else None

    def page_ids_for(self, uri: MemoryURI | str) -> frozenset[int]:
        """返回当前指向同一最终 URI 的全部临时编号。"""

        return frozenset(self._uri_to_pages.get(self._document_uri(uri), set()))

    def existing_items(self) -> tuple[tuple[int, str], ...]:
        """按编号返回系统分配的全部旧节点映射。"""

        return tuple(
            (page_id, uri) for page_id, uri in sorted(self._page_to_uri.items()) if page_id <= EXISTING_PAGE_ID_MAX
        )

    def page_ids(self) -> frozenset[int]:
        """返回当前已绑定的全部编号。"""

        return frozenset(self._page_to_uri)

    def items(self) -> tuple[tuple[int, str], ...]:
        """按编号返回当前全部临时编号和最终 URI。"""

        return tuple(sorted(self._page_to_uri.items()))

    def _bind(self, page_id: int, uri: str) -> None:
        bound_uri = self._page_to_uri.get(page_id)
        if bound_uri is not None and bound_uri != uri:
            raise MemoryPageIdError("page_id is already bound to another memory URI")
        self._page_to_uri[page_id] = uri
        self._uri_to_pages.setdefault(uri, set()).add(page_id)

    @staticmethod
    def _document_uri(value: MemoryURI | str) -> str:
        parsed = MemoryURI.parse(value)
        parsed.to_address()
        return str(parsed)


def validate_page_id(value: object) -> int:
    """严格校验任意临时页面编号。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < EXISTING_PAGE_ID_MIN:
        raise MemoryPageIdError("page_id must be a positive integer")
    return value


def validate_new_page_id(value: object) -> int:
    """严格校验模型为新节点声明的编号。"""

    page_id = validate_page_id(value)
    if page_id < NEW_PAGE_ID_MIN:
        raise MemoryPageIdError("new memory page_id must be at least 100")
    return page_id


def validate_unique_page_ids(values: Iterable[int]) -> None:
    """拒绝同一解析输出中的重复编号。"""

    seen: set[int] = set()
    for raw_value in values:
        page_id = validate_page_id(raw_value)
        if page_id in seen:
            raise MemoryPageIdError("memory candidate batch contains a duplicate page_id")
        seen.add(page_id)


__all__ = [
    "EXISTING_PAGE_ID_MAX",
    "EXISTING_PAGE_ID_MIN",
    "MemoryPageIdError",
    "MemoryPageIdMap",
    "NEW_PAGE_ID_MIN",
    "validate_new_page_id",
    "validate_page_id",
    "validate_unique_page_ids",
]
