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

TODO(BHV-KINDS-002): 词表在真实数据上的膨胀与归一形状（2026-08-30 与用户讨论后登记，先解
折叠粒度 WP4 再动；粒度修好后本条仍要做，否则词表仍会随时间无限增长）。
【实测】EgoLife 七天：1,083 个 kind，其中 708 个只在一天出现；≥5 天复现的 108 个几乎全是微动作
（交谈 1950、使用手机 412、看 245、向右转 177…）；归一到第 1000 个名字时词表已 498 个（新建率
≈50%）；benchmark 上已出现实质归错（「与医生通话」吞掉所有打电话）。每次归一把**整本词表**渲染进
提示词、全部正名列进 schema enum——成本与干扰项随词表线性增长，归错很可能就是 1000 个候选摆在
面前时的模式匹配失手。一次 sweep 近三千个名字≈三千次串行调用（≈2 小时）。
【裁定过的原则】"一个人的行为类型天然收敛"在微动作粒度下不成立；容量撞顶已降级为原始名作 token
（reduction/runner）；批量归一/短名单是同一次提示词改动，必须按纪律拿真实名字做对照，不凭推理写。
【方案】
① 词表是**派生物**：occurrence 永久保留原始名，词表可随时从树上全部原始名重建（策略变了就重打），
   把它当身份决策的缓存而不是真相——待用户确认。
② 活跃/退休：条目记 last_hit_at / hit_count；只有活跃的进提示词与 enum，退休的仍在词表里、名字或
   别名精确命中时走 token_for 快路径直接复活（身份不断、统计不碎）。这是记忆侧"按使用定温度、久不用
   退休"的逻辑用在词表上（对原料是错的，对词表是对的）。退休天数 N 不拍：拿七天数据算 kind 隔 k 天
   再次命中的概率再定。
③ 归一不再对着整本词表：未知名字先用确定性预选（字面/字符重合；将来可用 embedding）取几十个候选
   短名单，模型只在短名单里选或 null；多个新名字一批一次调用。提示词与 enum 从 O(词表) 变 O(几十)。
   短名单走字面重合还是 embedding——待用户确认。
④ 别名吸收加保守约束（别名不得比正名更泛），解「与医生通话」这类吞并。
【验证】固化数据 benchmark/data/egolife_week：用树上 9,270 个原始名按新方案重建词表，对比 kind 数与
已知归错样本（只跑一天约 300 个名字的模型调用）；用七天数据算复现间隔分布定 N。
【影响】不做：词表随时间无限增长，每次归一的成本与归错率随之上升，预测树按 token 聚合的统计被碎片
化。做：改 behavior/kinds（model/store/resolver/config）与 reduction 的调用方式，不动树地址与字段。
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
