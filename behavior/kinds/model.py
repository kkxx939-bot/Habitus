"""行为类型词表：跨次身份的唯一登记处。

## 为什么存在

融合层每次给出的行为名是自由文本，同一类行为会换着说法出现（「洗手」「洗了手」「清洁双手」）。
不归一的话，按类的一切统计（频率、时刻分布、转移、距上次）都会被名字的随机波动切碎——一个真实
的规律被拆成几条各自不显著的曲线。词表登记「正名 + 别名」，写入层据此给每条 occurrence 打稳定的
kind_token。

## 只登记名字，不承载知识

"这类行为通常何时发生"属于预测层；词表只回答"这两个名字是不是同一类"。一个人的行为类型天然
收敛，所以是单文件（沿 memory 树 ``profile.md`` 的单文件直读模式），不是每类型一个节点——这也是
"kind 节点会无限堆积"那个担忧的解法。

## 身份归属在写入侧，语义生成在融合侧

融合的 LLM 不定身份：融合提示词空间实测极其敏感（正文净增 65 字即让用例从 8/8 掉到 5/8）；词表
若进 ``FUSION_VERSION`` 会让版本随每个新类型漂移、排队作业反复改挂；全局累积状态不进只看有界
上下文的流式判断层。写入层拿词表让模型**逐字复用或新建**——与 memory 链"解析 LLM 产候选、
Editor 按 page_id 清单定身份"同构。原始名照常保留在 occurrence 上，归错可改词表重打，不动地址。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from behavior.model import semantic_name
from foundation.ids import canonical_path_identity


class BehaviorKindError(ValueError):
    """词表内容或对词表的操作违反登记约束。"""


class BehaviorKindLimitError(BehaviorKindError):
    """词表超出配置的容量边界。"""


@dataclass(frozen=True)
class BehaviorKindRegistry:
    """正名到别名集合的不可变登记表。

    全部名字（正名与别名一起）在 ``canonical_path_identity``（NFC + casefold）意义下唯一：
    命中同一身份的两个写法必属于同一类，且只能有一个登记位置。修改返回新实例，落盘由
    ``BehaviorKindStore`` 负责。
    """

    kinds: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kinds, Mapping):
            raise BehaviorKindError("kinds must map token text to alias sequences")
        identity_owner: dict[str, str] = {}
        normalized: dict[str, tuple[str, ...]] = {}
        for raw_token, raw_aliases in self.kinds.items():
            token = semantic_name(raw_token, "behavior kind token")
            token_identity = canonical_path_identity(token, "behavior kind token")
            if token_identity in identity_owner:
                raise BehaviorKindError(
                    f"behavior kind name collides with an existing entry: {token}"
                )
            identity_owner[token_identity] = token
            if isinstance(raw_aliases, str) or not isinstance(raw_aliases, tuple | list):
                raise BehaviorKindError("behavior kind aliases must be a sequence of names")
            aliases: list[str] = []
            for raw_alias in raw_aliases:
                alias = semantic_name(raw_alias, "behavior kind alias")
                alias_identity = canonical_path_identity(alias, "behavior kind alias")
                if alias_identity in identity_owner:
                    raise BehaviorKindError(
                        f"behavior kind name collides with an existing entry: {alias}"
                    )
                identity_owner[alias_identity] = alias
                aliases.append(alias)
            normalized[token] = tuple(
                sorted(aliases, key=lambda name: canonical_path_identity(name, "behavior kind alias"))
            )
        ordered = {
            token: normalized[token]
            for token in sorted(
                normalized, key=lambda name: canonical_path_identity(name, "behavior kind token")
            )
        }
        object.__setattr__(self, "kinds", MappingProxyType(ordered))

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(self.kinds)

    @property
    def kind_count(self) -> int:
        return len(self.kinds)

    def aliases_of(self, token: str) -> tuple[str, ...]:
        resolved = self._require_token(token)
        return self.kinds[resolved]

    def token_for(self, name: object) -> str | None:
        """按规范身份查这个名字属于哪一类；未登记返回 None。"""

        identity = canonical_path_identity(
            semantic_name(name, "behavior kind name"), "behavior kind name"
        )
        for token, aliases in self.kinds.items():
            if canonical_path_identity(token, "behavior kind token") == identity:
                return token
            for alias in aliases:
                if canonical_path_identity(alias, "behavior kind alias") == identity:
                    return token
        return None

    def with_new_kind(self, name: object) -> BehaviorKindRegistry:
        """把一个未登记的名字登记为新类型的正名。"""

        resolved = semantic_name(name, "behavior kind token")
        return BehaviorKindRegistry({**dict(self.kinds), resolved: ()})

    def with_alias(self, token: str, alias: object) -> BehaviorKindRegistry:
        """把一个未登记的名字并入既有类型作为别名。"""

        resolved_token = self._require_token(token)
        resolved_alias = semantic_name(alias, "behavior kind alias")
        merged = dict(self.kinds)
        merged[resolved_token] = (*merged[resolved_token], resolved_alias)
        return BehaviorKindRegistry(merged)

    def _require_token(self, token: str) -> str:
        resolved = semantic_name(token, "behavior kind token")
        if resolved not in self.kinds:
            raise BehaviorKindError(f"behavior kind token is not registered: {resolved}")
        return resolved


__all__ = [
    "BehaviorKindError",
    "BehaviorKindLimitError",
    "BehaviorKindRegistry",
]
