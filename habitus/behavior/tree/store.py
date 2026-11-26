"""habitus 树里一支的事件与空白：``occurrences/<支>/…``、``gaps/<支>/…``（文件操作见 ``infrastructure/tree``）。

每一支一个视图，只看得到自己那一支：序列、按天读树、覆盖率、"那天有没有来过东西"都默认"一棵树里只有这一支"，
空白的字段里又没有支，所以支只能由视图给（裁定 112）。同一支全程用同一个视图实例。
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path

from habitus.behavior.document import (
    BehaviorDocument,
    BehaviorDocumentCodec,
    BehaviorDocumentConfig,
    BehaviorDocumentIntegrityError,
    BehaviorDocumentLimitError,
)
from habitus.behavior.model import (
    BehaviorAddress,
    BehaviorDirectory,
    BehaviorKind,
    is_ascii_digits,
    kind_directory_prefix,
)
from habitus.behavior.schema import BehaviorSchemaRegistry
from habitus.behavior.tree.config import BehaviorTreeConfig
from habitus.behavior.uri import BehaviorURI
from habitus.foundation.ids import canonical_path_identity
from habitus.infrastructure.tree.files import TreeFiles


class BehaviorTreeIntegrityError(ValueError):
    """行为树包含路径逃逸、符号链接或不符合已确认结构的条目。"""


class BehaviorTreeConflictError(RuntimeError):
    """只允许创建的 L2 地址已经绑定到不同内容。"""


class BehaviorTree:
    """一支的只增 L2 读写视图。``root`` 是整棵 habitus 树的根，``lane`` 是这一支。"""

    def __init__(
        self,
        root: str | Path,
        *,
        lane: str,
        document_codec: BehaviorDocumentCodec | None = None,
        document_config: BehaviorDocumentConfig | None = None,
        tree_config: BehaviorTreeConfig | None = None,
    ) -> None:
        self.tree_config = tree_config or BehaviorTreeConfig()
        self._files = TreeFiles(
            root,
            max_children=self.tree_config.max_children_per_directory,
            error=BehaviorTreeIntegrityError,
            conflict=BehaviorTreeConflictError,
        )
        self.root = self._files.root
        # 校验支名：不认识的支在这里就失败，不等第一次读写
        self.lane = BehaviorURI.from_directory(BehaviorDirectory.occurrences(), lane).lane
        if document_codec is None:
            document_codec = BehaviorDocumentCodec(BehaviorSchemaRegistry.load_default())
        self._document_codec = document_codec
        self.document_config = document_config or BehaviorDocumentConfig()

    @property
    def document_codec(self) -> BehaviorDocumentCodec:
        return self._document_codec

    @property
    def registry(self) -> BehaviorSchemaRegistry:
        """树与其写入方必须共用同一个 Schema 注册表，这里是唯一的取用入口。"""

        return self._document_codec.registry

    def uri_for(self, address: BehaviorAddress) -> BehaviorURI:
        """这一支里这个地址的 ``habitus://`` 地址。"""

        return BehaviorURI.from_address(address, self.lane)

    def initialize(self) -> Path:
        """建这一支的两个目录 ``occurrences/<支>``、``gaps/<支>``（树根下其余条目归记忆检查）。"""

        for kind in BehaviorKind:
            self._ensure_directory(self._kind_root(kind))
        return self.root

    def path_for(self, address: BehaviorAddress) -> Path:
        return self._files.require_inside(self.root / self._relative_path(address))

    def directory_path(self, directory: BehaviorDirectory) -> Path:
        return self._files.require_inside(self.root.joinpath(*self._directory_parts(directory)))

    def storable_bytes(self, document: BehaviorDocument) -> bytes:
        """文档写盘时的编码；正文或编码后字节超出上限就拒（``BehaviorDocumentLimitError``）。写入与归约发布前的干跑共用这一处。"""

        encoded = self._document_codec.encode(document).encode("utf-8")
        self.document_config.validate_body(document.markdown_body)
        self.document_config.validate_encoded(encoded)
        return encoded

    def create(self, document: BehaviorDocument) -> BehaviorDocument:
        """原子创建 add-only L2；目标已绑定到不同内容时拒绝覆盖。"""

        encoded = self.storable_bytes(document)
        self.initialize()
        path = self._existing_document_path(document.address)
        self._ensure_directory(path.parent)
        self._files.require_child_capacity(path.parent, (path.name,))
        self._files.create(path, encoded)
        return document

    def read(self, address: BehaviorAddress) -> BehaviorDocument:
        return self._decode(self._existing_document_path(address), address)

    def list_day_addresses(self, kind: BehaviorKind, occurred_on: date) -> tuple[BehaviorAddress, ...]:
        """某天某类的全部地址；目录只枚举一次、不读文档。

        逐个 ``exists``/``read`` 都会重新枚举整个日目录，成本随篇数平方增长，所以"这一天有哪些地址
        已被占用"必须整天问一次。归约的撞车消歧与落盘干跑走这里。
        """

        resolved_kind = BehaviorKind(kind)
        physical = self._day_directory(resolved_kind, occurred_on)
        if not physical.is_dir():
            return ()
        return tuple(
            self._leaf_address(resolved_kind, occurred_on, stem) for stem in self._files.markdown_names(physical)
        )

    def read_day(self, kind: BehaviorKind, occurred_on: date) -> tuple[BehaviorDocument, ...]:
        """一次性读出某天某类的全部文档；目录只解析、只枚举一次。

        逐篇 ``read`` 每次都沿路径逐级做身份校验，其中日目录那一级会重新枚举整个目录，读一天的
        成本随篇数平方增长。事件序列与语义关联都是按天整块读，走这里。
        """

        resolved_kind = BehaviorKind(kind)
        physical = self._day_directory(resolved_kind, occurred_on)
        if not physical.is_dir():
            return ()
        documents = [
            self._decode(physical / f"{stem}.md", self._leaf_address(resolved_kind, occurred_on, stem))
            for stem in self._files.markdown_names(physical)
        ]
        documents.sort(key=lambda item: (item.address.started_at, item.address.identity_name))
        return tuple(documents)

    def exists(self, address: BehaviorAddress) -> bool:
        return self._files.file_exists(self._existing_document_path(address))

    def directory_exists(self, directory: BehaviorDirectory) -> bool:
        return self._files.directory_exists(
            self._files.resolve(Path(*self._directory_parts(directory)), leaf_is_file=False)
        )

    def list_addresses(
        self,
        kind: BehaviorKind | None = None,
        *,
        limit: int = 256,
        after: BehaviorAddress | None = None,
    ) -> tuple[BehaviorAddress, ...]:
        if not 1 <= limit <= 10_000:
            raise ValueError("behavior tree list limit must be between 1 and 10000")
        selected_kind = None if kind is None else BehaviorKind(kind)
        if after is not None and selected_kind is not None and after.kind is not selected_kind:
            raise ValueError("behavior tree list cursor kind does not match the selected kind")
        after_key = None if after is None else self._address_order_key(after)
        result: list[BehaviorAddress] = []
        kinds = (selected_kind,) if selected_kind is not None else tuple(BehaviorKind)
        for resolved_kind in kinds:
            if after is not None and tuple(BehaviorKind).index(resolved_kind) < tuple(BehaviorKind).index(after.kind):
                continue
            # 翻页从游标那一天续：排序键先按日期，游标之前的年、月、日按目录名直接跳过
            since = after.occurred_on if after is not None and resolved_kind is after.kind else None
            for occurred_on, day_path in self._day_paths(resolved_kind, since=since):
                for stem in self._files.markdown_names(day_path):
                    address = self._leaf_address(resolved_kind, occurred_on, stem)
                    if after_key is not None and self._address_order_key(address) <= after_key:
                        continue
                    result.append(address)
                    if len(result) == limit:
                        return tuple(result)
        return tuple(result)

    def days(self, kind: BehaviorKind) -> tuple[date, ...]:
        """这一种文档有哪些日子（只列年 / 月 / 日目录，不列叶子）：按天缓存的读侧据此找新增的日子。"""

        return tuple(day for day, _path in self._day_paths(BehaviorKind(kind)))

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _relative_path(self, address: BehaviorAddress) -> Path:
        return Path(*self.uri_for(address).segments)

    def _directory_parts(self, directory: BehaviorDirectory) -> tuple[str, ...]:
        return BehaviorURI.from_directory(directory, self.lane).segments

    def _kind_root(self, kind: BehaviorKind) -> Path:
        return self.root.joinpath(*kind_directory_prefix(kind), self.lane)

    def _address_order_key(self, address: BehaviorAddress) -> tuple[int, str]:
        # 游标序必须与枚举产出序同一口径：枚举按叶名的 canonical 身份（NFC+casefold）排序
        # （见 TreeFiles.markdown_names），而 identity_name 里的时间戳 'T' 未折叠——字面序与折叠序在
        # 特定命名下会反转，分页边界附近的条目会被静默漏掉或重复产出。
        relative = self._relative_path(address)
        folded_leaf = canonical_path_identity(relative.name.removesuffix(".md"), "behavior document name")
        return tuple(BehaviorKind).index(address.kind), relative.with_name(f"{folded_leaf}.md").as_posix()

    def _day_directory(self, kind: BehaviorKind, occurred_on: date) -> Path:
        if isinstance(occurred_on, datetime):
            raise TypeError("occurred_on must be a date without a time")
        directory = BehaviorDirectory._dated(
            kind_directory_prefix(kind), occurred_on.year, occurred_on.month, occurred_on.day
        )
        return self._files.resolve(Path(*self._directory_parts(directory)), leaf_is_file=False)

    @staticmethod
    def _leaf_address(kind: BehaviorKind, occurred_on: date, stem: str) -> BehaviorAddress:
        try:
            return BehaviorAddress.from_identity(kind, occurred_on, stem)
        except (TypeError, ValueError) as exc:
            raise BehaviorTreeIntegrityError("behavior leaf contains an invalid timestamp identity") from exc

    def _day_paths(self, kind: BehaviorKind, *, since: date | None = None) -> Iterator[tuple[date, Path]]:
        """日目录，按日期升序；``since`` 跳过它之前的年、月、日（按目录名，不进去列）。"""

        for year_path in self._files.directories(self._kind_root(kind)):
            if len(year_path.name) != 4 or not is_ascii_digits(year_path.name):
                raise BehaviorTreeIntegrityError("behavior year directory must use YYYY")
            if since is not None and int(year_path.name) < since.year:
                continue
            for month_path in self._files.directories(year_path):
                if len(month_path.name) != 2 or not is_ascii_digits(month_path.name):
                    raise BehaviorTreeIntegrityError("behavior month directory must use MM")
                if since is not None and (int(year_path.name), int(month_path.name)) < (since.year, since.month):
                    continue
                for day_path in self._files.directories(month_path):
                    if len(day_path.name) != 2 or not is_ascii_digits(day_path.name):
                        raise BehaviorTreeIntegrityError("behavior day directory must use DD")
                    try:
                        occurred_on = date(int(year_path.name), int(month_path.name), int(day_path.name))
                    except ValueError as exc:
                        raise BehaviorTreeIntegrityError(
                            "behavior directory contains an invalid calendar date"
                        ) from exc
                    if since is not None and occurred_on < since:
                        continue
                    yield occurred_on, day_path

    def _existing_document_path(self, address: BehaviorAddress) -> Path:
        return self._files.resolve(self._relative_path(address), leaf_is_file=True)

    def _ensure_directory(self, directory: Path) -> None:
        self._files.ensure_directory(directory, check_capacity=True)

    def _decode(self, path: Path, address: BehaviorAddress) -> BehaviorDocument:
        raw = self._files.read_utf8(path, label="behavior document", max_bytes=self.document_config.max_encoded_bytes)
        try:
            document = self._document_codec.decode(raw, expected_address=address)
            self.document_config.validate_body(document.markdown_body)
            return document
        except (BehaviorDocumentIntegrityError, BehaviorDocumentLimitError) as exc:
            raise BehaviorTreeIntegrityError("behavior L2 document failed integrity validation") from exc


__all__ = ["BehaviorTree", "BehaviorTreeConflictError", "BehaviorTreeIntegrityError"]
