"""概念定义：语义树的词汇层，一个概念一份。只有两种概念（2026-11-14 用户定："节点来做行为事件来预测"）：

- **行为概念**：词表里一个行为类原样对应的概念（裁定 20 第 2 条），身份 = 类编号（``s-k0004``），显示名是类名，判据是类判据；
  同步词表时自动生成，不调模型。一条记录算哪个行为概念只看它的类编号（读时现读，不存命中）。
- **日历情境**：周几、几号、月末、日型这类外部条件（裁定 59 第 7 条），由日历机械生成（``calendar_situations``），不由模型写；
  只当前因或调节条件，不当后果。

- **显示名不许重**：给人看的是名字不是编号（裁定 21-1），同一条 lane 能一起出现的在用概念名字不许重。
- **没有版本**：定义变了就整个重算它的命中。
- **停用**：行为概念停用时 ``retired`` 置真（不再命中，历史命中留着读）；日历情境不退。来源是词表：词 20 天没被选中退出在用、
  或自己长的节点被删（词表方案二乙），下一次同步词表停用它；又在用了改回来。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from functools import cached_property
from types import MappingProxyType

from habitus.behavior.model import semantic_name
from habitus.foundation.ids import canonical_path_identity, canonical_text_identity, require_safe_path_segment
from habitus.foundation.integrity import canonical_digest
from habitus.scene.concepts.situation import SituationError, SituationRule

MAX_DEFINITION_CHARS = 400
MAX_NOTE_CHARS = 400
#: 概念身份会被拼进组合键（前因的集合、条件的集合），这几个分隔符不许出现在它里面，
#: 否则两个不同的组合能拼出同一个键（实测：概念名 ``出差中+晚睡`` 与集合 ``{出差中, 晚睡}``）。
IDENTITY_SEPARATORS = ("+", "@", "--", "#")


class ConceptError(ValueError):
    """概念定义或概念集与自己的约束矛盾。"""


class ConceptLookupError(ConceptError, KeyError):
    """按名字查不到概念——同时是 ``KeyError``，让 ``Mapping.get`` 照常返回默认值。"""

    def __str__(self) -> str:  # KeyError 会给消息加引号，这里还原成普通文本
        return ValueError.__str__(self)


class ConceptRole(str, Enum):
    """行为概念可当后件；日历情境（日型）只当前件或调节条件。"""

    BEHAVIOR = "behavior"
    DAY_TYPE = "day_type"

    @property
    def is_behavior(self) -> bool:
        return self is ConceptRole.BEHAVIOR

    @property
    def is_situation(self) -> bool:
        return not self.is_behavior


class ConceptOrigin(str, Enum):
    """概念从哪来：同步词表自动生成（行为概念），或日历机械生成（日历情境，裁定 59）。"""

    VOCABULARY = "vocabulary"
    CALENDAR = "calendar"


@dataclass(frozen=True)
class ConceptSource:
    """概念从哪来。"""

    origin: ConceptOrigin

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", ConceptOrigin(self.origin))


def concept_identity(name: object) -> str:
    """概念名的规范身份：与行为树叶名同一套做法（NFC + casefold，拒隐藏名与 ``.md`` 后缀），并拒分隔符。"""

    try:
        return canonical_path_identity(
            _identity_token(semantic_name(name, "concept name"), "concept name", None), "concept name"
        )
    except (TypeError, ValueError) as exc:
        raise ConceptError(str(exc)) from exc


@dataclass(frozen=True)
class ConceptDefinition:
    name: str
    definition: str
    role: ConceptRole
    source: ConceptSource
    created_at: datetime
    #: 行为概念：它自己那一个类；日历情境为空。
    classes: tuple[str, ...] = ()
    #: 行为概念所在的 lane；日历情境为 None（每条 lane 都看得见日历）。
    lane: str | None = None
    #: 行为概念的显示名（类名）；它的 ``name`` 是类编号，类改名身份不变。
    title: str | None = None
    #: 行为概念的类在词表里不在用了（退出在用或被删）：它不再命中，历史命中留着读。
    retired: bool = False
    #: 日历情境怎么算（日历键与成立的取值）；行为概念为 None。
    situation: SituationRule | None = None

    def __post_init__(self) -> None:
        identity = concept_identity(self.name)
        object.__setattr__(
            self, "definition", _single_line(self.definition, "concept definition", MAX_DEFINITION_CHARS)
        )
        object.__setattr__(self, "role", ConceptRole(self.role))
        if self.created_at.utcoffset() is None:
            raise ConceptError("concept created_at must be a timezone-aware datetime")
        object.__setattr__(self, "created_at", self.created_at.astimezone(UTC))
        classes = tuple(_single_line(item, "concept class", MAX_NOTE_CHARS) for item in self.classes)
        object.__setattr__(self, "classes", classes)
        if self.title is not None:
            object.__setattr__(self, "title", _single_line(self.title, "concept title", MAX_DEFINITION_CHARS))
        if self.role.is_situation:
            if classes or self.lane is not None or self.title is not None or self.retired:
                raise ConceptError("a calendar situation is not tied to vocabulary classes and never retires")
            if self.situation is None or self.source.origin is not ConceptOrigin.CALENDAR:
                raise ConceptError("a calendar situation is generated from the calendar and carries its calendar rule")
            return
        if self.situation is not None or self.source.origin is not ConceptOrigin.VOCABULARY:
            raise ConceptError("a behaviour concept is generated from its vocabulary class and carries no situation")
        object.__setattr__(self, "lane", _single_line(self.lane, "concept lane", MAX_NOTE_CHARS))
        if len(classes) != 1 or identity != concept_identity(classes[0]) or self.title is None:
            raise ConceptError("a behaviour concept is named by its one class id and titled with the class name")

    @cached_property
    def identity(self) -> str:
        """规范身份。定义不可变，算一次就记住（两两检验每晚要问几十万次，每次重新校验身份串占了折叠的大半时间）。"""

        return concept_identity(self.name)

    @property
    def label(self) -> str:
        """给人和模型看的名字：行为概念是类名，日历情境是自己的名字。"""

        return self.title if self.title is not None else self.name

    @property
    def fingerprint(self) -> str:
        """判据的内容指纹：会改变"什么算命中"的那些字段。来源、时间与显示名不在里面。"""

        return canonical_digest(
            {
                "identity": self.identity,
                "definition": self.definition,
                "role": self.role.value,
                "classes": list(self.classes),
                "lane": self.lane,
                "situation": None if self.situation is None else self.situation.payload(),
            }
        )[:16]


class ConceptSet(Mapping[str, ConceptDefinition]):
    """一组自洽的概念：身份不重、一个类最多一个行为概念、同一条 lane 能一起出现的在用概念显示名不重。键是规范身份。"""

    def __init__(self, definitions: Iterable[ConceptDefinition]) -> None:
        resolved: dict[str, ConceptDefinition] = {}
        for definition in definitions:
            if definition.identity in resolved:
                raise ConceptError(f"concept {definition.name!r} appears twice in the set")
            resolved[definition.identity] = definition
        base = {definition.classes[0]: identity for identity, definition in resolved.items() if definition.classes}
        self._items: Mapping[str, ConceptDefinition] = MappingProxyType(resolved)
        self._base: Mapping[str, str] = MappingProxyType(base)
        for lane in {item.lane for item in resolved.values() if item.lane is not None}:
            self._require_unique_labels(lane)

    def _require_unique_labels(self, lane: str) -> None:
        """给模型看的是名字不是编号（裁定 21-1）：同一条 lane 能一起出现的概念（这条 lane 的 + 不属于任何 lane 的）名字不许重。

        停用了的不算（与词表同一口径：词表只要求在用的类不重名）：它们不再给模型看，并掉的「吃饭」之后又长出一个
        「吃饭」，两个概念各自留着、只有在用的那个会被问到。
        """

        seen: dict[str, str] = {}
        for item in self._items.values():
            if item.lane not in (lane, None) or item.retired:
                continue
            key = canonical_text_identity(item.label, "concept label")
            if key in seen:
                raise ConceptError(
                    f"concepts {seen[key]!r} and {item.name!r} both show as {item.label!r} in lane {lane!r}"
                )
            seen[key] = item.name

    def __getitem__(self, key: str) -> ConceptDefinition:
        # 已是规范身份的直接命中，不再每次重新校验身份串（两两检验按名字查几十万次）；别的写法照常规范化
        found = self._items.get(key) if isinstance(key, str) else None
        if found is not None:
            return found
        try:
            identity = concept_identity(key)
        except ConceptError as exc:
            raise ConceptLookupError(str(exc)) from exc
        try:
            return self._items[identity]
        except KeyError:
            raise ConceptLookupError(f"concept {key!r} is not in the set") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str) and key in self._items:
            return True
        try:
            return concept_identity(key) in self._items
        except ConceptError:
            return False

    def behaviors(self) -> tuple[str, ...]:
        return tuple(identity for identity, item in self._items.items() if item.role.is_behavior)

    def situations(self) -> tuple[str, ...]:
        return tuple(identity for identity, item in self._items.items() if item.role.is_situation)

    def base_for(self, class_id: str) -> str | None:
        """一个类的行为概念的身份；这个类还没有行为概念时为 None。"""

        return self._base.get(class_id)

    def lane_of(self, name: str) -> str | None:
        """概念属于哪条 lane：行为概念是自己的 lane；日历情境不属于任何一条 lane（None，两边都能用）。不在集里的也给 None。"""

        return self[name].lane if name in self else None

    def relatable_to(self, consequent: str) -> tuple[str, ...]:
        """能和这个后件连成一条关系的概念（不含它自己）：同一条 lane 的行为概念 + 不属于任何 lane 的日历情境。

        关系只在一条 lane 内（裁定 20 第 1 条）：两两检验的对从这里取。
        """

        own = self[consequent].identity
        lane = self.lane_of(own)
        return tuple(
            identity for identity in sorted(self._items) if identity != own and self.lane_of(identity) in (lane, None)
        )

    def label_of(self, name: str) -> str:
        """概念 → 给人和模型看的名字；不在集里的原样返回（读侧拿它渲染，概念删了的旧记录照样印得出）。"""

        return self[name].label if name in self else name

    def class_label(self, class_id: str) -> str:
        """类编号 → 给人和模型看的类名（它的行为概念的显示名）；还没同步到的类原样给编号。"""

        base = self._base.get(class_id)
        return class_id if base is None else self._items[base].label

    def classes_of(self, name: str) -> tuple[str, ...]:
        """概念 → 它的类（行为概念是它自己那一个类）；日历情境为空。"""

        return self[name].classes


def _single_line(value: object, label: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ConceptError(f"{label} must be text")
    if not value.strip() or value != value.strip():
        raise ConceptError(f"{label} must be non-empty text without surrounding whitespace")
    if "\n" in value or "\r" in value or any(not character.isprintable() for character in value):
        raise ConceptError(f"{label} must be a single printable line")
    if len(value) > limit:
        raise ConceptError(f"{label} exceeds {limit} characters")
    return value


def _identity_token(value: object, label: str, limit: int | None) -> str:
    """会被拼进路径身份的名字：单行、路径安全、不含分隔符。"""

    text = _single_line(value, label, limit if limit is not None else MAX_DEFINITION_CHARS)
    try:
        require_safe_path_segment(text, label)
    except ValueError as exc:
        raise ConceptError(str(exc)) from exc
    if any(separator in text for separator in IDENTITY_SEPARATORS):
        raise ConceptError(f"{label} must not contain any of {IDENTITY_SEPARATORS!r}")
    return text


__all__ = [
    "IDENTITY_SEPARATORS",
    "MAX_DEFINITION_CHARS",
    "ConceptDefinition",
    "ConceptError",
    "ConceptLookupError",
    "ConceptOrigin",
    "ConceptRole",
    "ConceptSet",
    "ConceptSource",
    "SituationError",
    "SituationRule",
    "concept_identity",
]
