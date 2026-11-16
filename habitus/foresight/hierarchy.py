"""词表上下级在预测层的那一份（裁定 101 第 8 条）：预期、账、单元仍按节点的编号记，一个节点一本账；凡是问
"他做没做这件事"的地方，认这条预期的节点自己和它全部活着的下级的记录（``family``）——反过来不认：做了上级那件宽的事，不算
做到了细的。上级和下级在同一件事上只说最细的那一条（``related``、``depth``），节点不在用以后不再说它（``active``）。

foresight 不认识词表包：这一份由组合根从词表状态装出来交进来（``runtime.scene_vocabulary.kind_hierarchy``），每一拍开头取一份、
整拍用同一份。不给就是"没有上下级、全都在用"（``FLAT``）。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class KindHierarchy:
    """``below``：节点 → 它全部活着的下级（下级的下级也算，不含自己）；``in_use``：此刻在用的节点（None = 不知道，一律当在用）。"""

    below: Mapping[str, frozenset[str]] = field(default_factory=dict)
    in_use: frozenset[str] | None = None

    def __post_init__(self) -> None:
        below = {kind: frozenset(items) for kind, items in self.below.items()}
        if any(kind in items for kind, items in below.items()):
            raise ValueError("a node is not its own descendant")
        object.__setattr__(self, "below", MappingProxyType(below))

    def family(self, kind_token: str) -> frozenset[str]:
        """算"这个节点的事做了"的编号：它自己加全部下级。"""

        return frozenset({kind_token}) | self.below.get(kind_token, frozenset())

    def counted_for(self, kind_token: str) -> frozenset[str]:
        """一条挂在这个编号上的记录算"做了"给哪些节点：它自己加它的每一个上级（``family`` 的反面）。"""

        return frozenset({kind_token}) | frozenset(kind for kind, items in self.below.items() if kind_token in items)

    def done(self, kind_tokens: Iterable[str]) -> frozenset[str]:
        """这几条记录（的编号）算"做了"的全部节点：每一条自己加它的每一个上级。"""

        return frozenset(kind for token in kind_tokens for kind in self.counted_for(token))

    def related(self, first: str, second: str) -> bool:
        """一个是另一个的上级（同一个不算）。"""

        return first != second and (second in self.family(first) or first in self.family(second))

    def depth(self, kind_token: str) -> int:
        """它有几个上级（层数减一）：同一拍里细的先处理，按它从大到小排。"""

        return sum(1 for items in self.below.values() if kind_token in items)

    def active(self, kind_token: str) -> bool:
        return self.in_use is None or kind_token in self.in_use


#: 没有上下级、全都在用：没接词表时用它。
FLAT = KindHierarchy()

#: 组合根交进来的口：调一次给出此刻的一份。
KindHierarchyPort = Callable[[], KindHierarchy]


__all__ = ["FLAT", "KindHierarchy", "KindHierarchyPort"]
