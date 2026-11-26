"""词表上下级在预测层的那一份（裁定 105）：预期、账、单元按节点的编号记，一个节点一本账，攒证据只数直接选中它的记录。
他做了下级那件事的那一次，对上级是"不知道"——不算做了，也不算没做：预期到期时不记命中也不记漏做（结算按"窗里有观测空白"
那条记中性），到点要提醒上级时这一次不说（``family``）。上下级之间不互相借记录，也不互相避让；节点不在用以后不再说它（``active``）。

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
        """这个节点自己加全部下级：其中下级的记录对它是"不知道"（不算做了、也不算没做）。"""

        return frozenset({kind_token}) | self.below.get(kind_token, frozenset())

    def counted_for(self, kind_token: str) -> frozenset[str]:
        """一条挂在这个编号上的记录涉及哪些节点：它自己（做了）加它的每一个上级（不知道）——``family`` 的反面。"""

        return frozenset({kind_token}) | frozenset(kind for kind, items in self.below.items() if kind_token in items)

    def done(self, kind_tokens: Iterable[str]) -> frozenset[str]:
        """这几条记录（的编号）涉及的全部节点：每一条自己加它的每一个上级（"今天做过没有"、"表那一代之后再出现过没有"：
        做了的、说不清做没做的都不再当"没做"）。"""

        return frozenset(kind for token in kind_tokens for kind in self.counted_for(token))

    def active(self, kind_token: str) -> bool:
        return self.in_use is None or kind_token in self.in_use


#: 没有上下级、全都在用：没接词表时用它。
FLAT = KindHierarchy()

#: 组合根交进来的口：调一次给出此刻的一份。
KindHierarchyPort = Callable[[], KindHierarchy]


__all__ = ["FLAT", "KindHierarchy", "KindHierarchyPort"]
