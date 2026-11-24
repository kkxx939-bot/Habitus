"""长期记忆域共享的类型、地址和目录模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from habitus.foundation.ids import canonical_path_identity, require_safe_path_segment
from habitus.infrastructure.tree.layout import (
    ABSTRACT_FILENAME,
    ENTITIES_DIRECTORY,
    OVERVIEW_FILENAME,
    PREFERENCES_DIRECTORY,
    TOOLS_DIRECTORY,
)

#: 卡片的三个目录（档案是根下的单文件 ``profile.md``）。树根下还有事件与空白的目录，不归记忆读写。
MEMORY_DIRECTORIES = (PREFERENCES_DIRECTORY, ENTITIES_DIRECTORY, TOOLS_DIRECTORY)


class MemoryKind(str, Enum):
    PROFILE = "profile"
    PREFERENCE = "preference"
    ENTITY = "entity"
    TOOL = "tool"


class MemoryLevel(int, Enum):
    """记忆树中的语义层级。"""

    ABSTRACT = 0
    OVERVIEW = 1
    DETAIL = 2

    @property
    def sidecar_filename(self) -> str:
        """返回 L0/L1 唯一侧车文件名；L2 没有侧车文件。"""

        if self is MemoryLevel.ABSTRACT:
            return ABSTRACT_FILENAME
        if self is MemoryLevel.OVERVIEW:
            return OVERVIEW_FILENAME
        raise ValueError("L2 uses a MemoryAddress instead of a semantic sidecar")

    @classmethod
    def from_sidecar_filename(cls, filename: object) -> MemoryLevel | None:
        """把唯一侧车文件名反解为 L0/L1。"""

        if filename == ABSTRACT_FILENAME:
            return cls.ABSTRACT
        if filename == OVERVIEW_FILENAME:
            return cls.OVERVIEW
        return None


# 新写入的记忆名字（topic、category、name、tool_name）的 UTF-8 字节上限。
# 由记忆名派生的最长单段文件名是原子写的临时名 ``.{name}.md.{32 位 hex}.tmp``，比名字多 1 + 3 + 1 + 32 + 4 = 41 字节；
# category 只当目录名，目录内只有固定名的侧车，派生开销更小。单段上限 255 − 41 = 214，取整到不大于 214、
# 且不大于行为侧 200（behavior/model.py 的 _MAX_IDENTITY_NAME_UTF8_BYTES）的 200。落盘文件名是归一身份，
# casefold 可能变长（"İ" 2 字节 → "i̇" 3 字节），所以原文与归一身份两者都要在上限内。
MAX_MEMORY_NAME_UTF8_BYTES = 200


def _semantic_name(value: object, field_name: str, *, max_utf8_bytes: int | None = None) -> str:
    """校验拼进路径的记忆名字。``max_utf8_bytes`` 只在新写入时给：读回已落盘的旧文件不卡长度。"""

    name = require_safe_path_segment(value, field_name)
    if name != name.strip() or any(not character.isprintable() for character in name):
        raise ValueError(f"{field_name} must not contain surrounding whitespace or control characters")
    identity = canonical_path_identity(name, field_name)
    if max_utf8_bytes is not None:
        size = max(len(name.encode("utf-8")), len(identity.encode("utf-8")))
        if size > max_utf8_bytes:
            raise ValueError(f"{field_name} must be at most {max_utf8_bytes} UTF-8 bytes, got {size}")
    if identity.endswith(".md"):
        raise ValueError(f"{field_name} must not include the .md suffix")
    reserved_stems = {
        MemoryLevel.ABSTRACT.sidecar_filename[:-3],
        MemoryLevel.OVERVIEW.sidecar_filename[:-3],
    }
    if identity in reserved_stems:
        raise ValueError(f"{field_name} conflicts with a reserved semantic layer")
    return name


def require_new_memory_name(value: object, field_name: str) -> str:
    """模型新给出的记忆名字：在路径段校验之上再卡字节上限。已落盘的旧记忆按地址读回时不经过这里。"""

    return _semantic_name(value, field_name, max_utf8_bytes=MAX_MEMORY_NAME_UTF8_BYTES)


@dataclass(frozen=True)
class MemoryAddress:
    """唯一映射到记忆树中一个 Markdown 文件的语义地址。"""

    kind: MemoryKind
    name: str = field(default="", compare=False)
    category: str = field(default="", compare=False)
    _identity_name: str = field(init=False, repr=False, default="")
    _identity_category: str = field(init=False, repr=False, default="")

    def __post_init__(self) -> None:
        kind = MemoryKind(self.kind)
        object.__setattr__(self, "kind", kind)
        if kind is MemoryKind.PROFILE:
            if self.name or self.category:
                raise ValueError("profile memory does not accept name or category")
            return

        name = _semantic_name(self.name, f"{kind.value} name")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "_identity_name",
            canonical_path_identity(name, f"{kind.value} name"),
        )
        if kind is MemoryKind.ENTITY:
            category = _semantic_name(self.category, "entity category")
            object.__setattr__(self, "category", category)
            object.__setattr__(
                self,
                "_identity_category",
                canonical_path_identity(category, "entity category"),
            )
        elif self.category:
            raise ValueError(f"{kind.value} memory does not accept category")

    @property
    def identity_name(self) -> str:
        return self._identity_name

    @property
    def identity_category(self) -> str:
        return self._identity_category

    @classmethod
    def profile(cls) -> MemoryAddress:
        return cls(MemoryKind.PROFILE)

    @classmethod
    def preference(cls, topic: str) -> MemoryAddress:
        return cls(MemoryKind.PREFERENCE, name=topic)

    @classmethod
    def entity(cls, category: str, name: str) -> MemoryAddress:
        return cls(MemoryKind.ENTITY, name=name, category=category)

    @classmethod
    def tool(cls, tool_name: str) -> MemoryAddress:
        return cls(MemoryKind.TOOL, name=tool_name)


@dataclass(frozen=True)
class MemoryDirectory:
    """严格限定于已确认记忆树的目录地址。"""

    parts: tuple[str, ...] = field(default=(), compare=False)
    _identity_parts: tuple[str, ...] = field(init=False, repr=False, default=())

    def __post_init__(self) -> None:
        parts = tuple(self.parts)
        object.__setattr__(self, "parts", parts)
        if not parts:
            return
        root = parts[0]
        if root not in MEMORY_DIRECTORIES:
            raise ValueError("memory directory is outside the confirmed tree")
        if root in {"preferences", "tools"}:
            if len(parts) != 1:
                raise ValueError(f"{root} memory directory cannot contain subdirectories")
            object.__setattr__(self, "_identity_parts", parts)
            return
        if len(parts) > 2:
            raise ValueError("entity memory directory is deeper than category level")
        if len(parts) == 2:
            object.__setattr__(
                self,
                "_identity_parts",
                (
                    root,
                    canonical_path_identity(
                        _semantic_name(parts[1], "entity category"),
                        "entity category",
                    ),
                ),
            )
        else:
            object.__setattr__(self, "_identity_parts", parts)

    @property
    def identity_parts(self) -> tuple[str, ...]:
        return self._identity_parts

    @property
    def label(self) -> str:
        """概览标题行与 L0 里的目录名。"""

        return "/".join(self.parts) or "记忆根"

    @classmethod
    def root(cls) -> MemoryDirectory:
        return cls()

    @classmethod
    def preferences(cls) -> MemoryDirectory:
        return cls(("preferences",))

    @classmethod
    def entities(cls, category: str | None = None) -> MemoryDirectory:
        if category is None:
            return cls(("entities",))
        return cls(("entities", _semantic_name(category, "entity category")))

    @classmethod
    def tools(cls) -> MemoryDirectory:
        return cls(("tools",))

    @classmethod
    def for_address(cls, address: MemoryAddress) -> MemoryDirectory:
        if address.kind is MemoryKind.PROFILE:
            return cls.root()
        if address.kind is MemoryKind.PREFERENCE:
            return cls.preferences()
        if address.kind is MemoryKind.ENTITY:
            return cls.entities(address.category)
        return cls.tools()

    def parent(self) -> MemoryDirectory | None:
        """返回上级目录；根目录没有上级。"""

        if not self.parts:
            return None
        return MemoryDirectory(self.parts[:-1])

    def lineage(self) -> tuple[MemoryDirectory, ...]:
        """按当前目录到根目录的顺序返回完整祖先链。"""

        directories: list[MemoryDirectory] = []
        current: MemoryDirectory | None = self
        while current is not None:
            directories.append(current)
            current = current.parent()
        return tuple(directories)


__all__ = [
    "MAX_MEMORY_NAME_UTF8_BYTES",
    "MEMORY_DIRECTORIES",
    "MemoryAddress",
    "MemoryDirectory",
    "MemoryKind",
    "MemoryLevel",
    "require_new_memory_name",
]
