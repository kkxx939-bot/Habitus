"""行为类型词表：跨次身份的唯一登记处。

## 为什么存在

融合层每次给出的行为名是自由文本，同一类行为会换着说法出现（「洗手」「洗了手」「清洁双手」）。
不归一的话，按类的一切统计（频率、时刻分布、转移、距上次）都会被名字的随机波动切碎——一个真实
的规律被拆成几条各自不显著的曲线。词表登记「token + label + 别名」，写入层据此给每条 occurrence
打稳定的 kind_token。

## 只做两件事，不承载知识

词表只回答两个问题：**这个名字属于哪个 token**（让同一件事有同一个聚合键），以及**这个 token
最近还在发生吗**（命中账与存活期，让不再发生的名字退出候选）。它不判断一类行为重不重要、该提醒
还是该代劳——那是预测层从行为树统计里得出的结论；它也不被预测树读取——预测树只读行为树上的
``kind_token``。合并两个 token 是写入层改树（重打 occurrence 的 kind_token）的操作，词表只提供
"谁并入谁"。（2026-08-30 与用户逐条裁定；早先"非单位闸门""merged_into 读侧解析"两案均被否，
理由是不能把过多职责放在词表。）

## 身份归属在写入侧，语义判断只由模型做

融合的 LLM 不定身份：融合提示词空间实测极其敏感（正文净增 65 字即让用例从 8/8 掉到 5/8）；词表
若进 ``FUSION_VERSION`` 会让版本随每个新类型漂移。写入层归一时，"这个名字和某个候选是不是同一件
事"**只由模型判**——代码不用子串、字面重合、泛/具体之类的规则替它下结论（用户裁定：规则覆盖不了
日常场景）。代码只做结构校验（match 必须在给它的候选里、每个名字恰好一项）和候选准备（embedding
最近邻 ∪ 高频，只影响召回）。原始名照常保留在 occurrence 上，归错可改词表重打，不动地址。

TODO(BHV-KINDS-002): 词表在真实数据上的膨胀与归一形状——方案定稿（2026-08-30，与用户数轮讨论）。
【实测】EgoLife 七天（v15 折叠粒度）：1,083 个 kind，708 个只在一天出现；整本词表渲染进提示词、
全部正名列进 enum，一名一调用串行 3,061 次；已出现实质归错（「与医生通话」吞掉所有打电话）。
膨胀两层原因：微动作（≈1/5 kind、1/3 occurrence，WP4 已从源头砍掉）与带对象/对方/话题的变体长尾
（"与Shure交谈""吃披萨"各自成 kind，归一保守）。
【embedding 实验】（豆包 1024 维，2,956 个名字）最近邻 top-30 对别名→正名召回 95%（字面重合 85%）；
但正例与最难负例的相似度分布重叠（p50 0.675 vs 0.702；上楼/下楼 0.77、拿起/放下 0.64），没有可用
的自动合并阈值——embedding 只做召回，判定必须是模型。
【方案】
① 条目 = token（稳定 id，首见原始名）+ label（模型给的可读名）+ aliases + 命中账。
② 命中账按**行为日**记（occurrence 的 started_at 所在日，不是归约时刻）：最近 64 个不同命中日
   （有界、晚到可插入）、累计天数与次数。过期用**数据时钟**（本轮最新行为日），不用墙钟——停机
   一个月不会把词表删光。存活期限 = last_hit_day + max(base_days, multiplier × max_gap_days)：
   一次性名字基础期后删，周期行为按自己量出来的间隔续命；每次命中重算。删除只删词表条目与向量，
   树上 occurrence 不动（季节性行为淡季删、旺季同名回来 token 同串、树上照聚）。
③ 候选准备：名字的 embedding 对全部 kind 的 (token, label) 向量取最近 30 ∪ 命中天数最多 20；
   embedding 不可用退字面重合并留信号。向量是派生旁册（``kinds.vectors.json``），换模型整表重算。
④ 判定：10 个名字一批一次调用，每个名字附证据（所在判断的 summary）；输出 match（候选编号 |
   同批另一名字编号 | null）与 label；校验只守结构，违约只重问违约项，耗尽当新建并留信号。
⑤ 合并：写入层操作——词表把 a 并入 b（别名与账合并、删 a），树上 a 的 occurrence 重打为 b。
   离线整理工具成对交模型判"是否同一件事"，候选对来自向量最近邻与 label 相同者。
⑥ 重建 = 补齐（树上每个原始名保证有 token）+ 账按树重算 + 向量补算；v1 文件不兼容，靠重建迁。
【验证】DAY1 v18 融合产物重放归约直接建词表（≈24 次归一调用），看归类与 label；七天数据零调用
模拟存活期；抽 100 对人工标注做归错率基准。
【不做】自动重要性判断、非单位闸门、merged_into 读侧解析、预测树改动。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from types import MappingProxyType

from behavior.model import semantic_name
from foundation.ids import canonical_path_identity

# 每个条目保留的最近命中日数：足够从中量出周/月/季频的最长间隔（64 天里的相邻差），又让条目有界。
HIT_DAYS_KEPT = 64


class BehaviorKindError(ValueError):
    """词表内容或对词表的操作违反登记约束。"""


class BehaviorKindLimitError(BehaviorKindError):
    """词表超出配置的容量边界。"""


def _day(value: object, field_name: str) -> date:
    """只接受日历日：datetime 是 date 的子类，但带时刻的值不是"行为日"。"""

    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    raise BehaviorKindError(f"{field_name} must be a date")


@dataclass(frozen=True)
class BehaviorKindEntry:
    """一个行为类型：稳定 token、可读 label、别名集合与命中账。

    ``hit_days`` 是最近 ``HIT_DAYS_KEPT`` 个**不同**命中日（升序）；没有命中过的条目（预置、
    重建时补齐）``hit_days`` 为空，不参与过期。
    """

    token: str
    label: str
    aliases: tuple[str, ...] = ()
    created_on: date | None = None
    hit_days: tuple[date, ...] = ()
    hit_days_total: int = 0
    hit_count: int = 0

    def __post_init__(self) -> None:
        token = semantic_name(self.token, "behavior kind token")
        label = semantic_name(self.label, "behavior kind label")
        if isinstance(self.aliases, str) or not isinstance(self.aliases, Sequence):
            raise BehaviorKindError("behavior kind aliases must be a sequence of names")
        aliases = tuple(semantic_name(alias, "behavior kind alias") for alias in self.aliases)
        if len({canonical_path_identity(a, "behavior kind alias") for a in aliases}) != len(aliases):
            raise BehaviorKindError(f"behavior kind aliases repeat a name: {token}")
        if self.created_on is not None:
            _day(self.created_on, "behavior kind created_on")
        if isinstance(self.hit_days, str) or not isinstance(self.hit_days, Sequence):
            raise BehaviorKindError("behavior kind hit_days must be a sequence of dates")
        days = tuple(_day(item, "behavior kind hit day") for item in self.hit_days)
        if list(days) != sorted(set(days)):
            raise BehaviorKindError(f"behavior kind hit_days must be strictly ascending: {token}")
        if len(days) > HIT_DAYS_KEPT:
            raise BehaviorKindError(f"behavior kind keeps at most {HIT_DAYS_KEPT} hit days: {token}")
        for name, value in (("hit_days_total", self.hit_days_total), ("hit_count", self.hit_count)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise BehaviorKindError(f"behavior kind {name} must be a non-negative integer")
        if self.hit_days_total < len(days) or self.hit_count < self.hit_days_total:
            raise BehaviorKindError(f"behavior kind hit account is inconsistent: {token}")
        object.__setattr__(self, "token", token)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "hit_days", days)

    @property
    def names(self) -> tuple[str, ...]:
        return (self.token, *self.aliases)

    @property
    def last_hit_day(self) -> date | None:
        return self.hit_days[-1] if self.hit_days else None

    @property
    def max_gap_days(self) -> int:
        """相邻两个命中日的最长间隔；只命中过一天时为 0。"""

        if len(self.hit_days) < 2:
            return 0
        return max((later - earlier).days for earlier, later in zip(self.hit_days, self.hit_days[1:], strict=False))

    def with_hit(self, day: date) -> BehaviorKindEntry:
        """记一次命中：同一天多次只算一天；晚到的日子插入重排；超过保留数丢最早的。"""

        hit_day = _day(day, "behavior kind hit day")
        if hit_day in self.hit_days:
            return replace(self, hit_count=self.hit_count + 1)
        days = tuple(sorted((*self.hit_days, hit_day)))[-HIT_DAYS_KEPT:]
        return replace(
            self,
            hit_days=days,
            hit_days_total=self.hit_days_total + 1,
            hit_count=self.hit_count + 1,
            created_on=self.created_on if self.created_on is not None else hit_day,
        )

    def expires_after(self, *, base_days: int, gap_multiplier: int) -> date | None:
        """存活期限：``last_hit_day + max(base_days, multiplier × max_gap_days)``；无命中则 None。"""

        last = self.last_hit_day
        if last is None:
            return None
        return last + timedelta(days=max(base_days, gap_multiplier * self.max_gap_days))


def _entry_from(token: object, value: object) -> BehaviorKindEntry:
    """接受 ``Entry`` 或（预置/测试用的）别名序列；别名序列表示一个从未命中的登记条目。"""

    if isinstance(value, BehaviorKindEntry):
        if value.token != token:
            raise BehaviorKindError("behavior kind entry token must match its mapping key")
        return value
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise BehaviorKindError("behavior kind aliases must be a sequence of names")
    resolved = semantic_name(token, "behavior kind token")
    return BehaviorKindEntry(token=resolved, label=resolved, aliases=tuple(value))


@dataclass(frozen=True)
class BehaviorKindRegistry:
    """token 到条目的不可变登记表。

    全部名字（token 与别名一起）在 ``canonical_path_identity``（NFC + casefold）意义下唯一：
    命中同一身份的两个写法必属于同一类，且只能有一个登记位置。修改返回新实例，落盘由
    ``BehaviorKindStore`` 负责。
    """

    # 运行时也接受（预置/测试用的）别名序列作为值——表示一个从未命中的登记条目；
    # ``__post_init__`` 统一规范化成条目，所以类型上只声明条目。
    entries: Mapping[str, BehaviorKindEntry] = field(default_factory=dict)
    _index: Mapping[str, str] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.entries, Mapping):
            raise BehaviorKindError("kinds must map token text to entries")
        normalized = {
            semantic_name(token, "behavior kind token"): _entry_from(token, value)
            for token, value in self.entries.items()
        }
        index: dict[str, str] = {}
        for token, entry in normalized.items():
            for name in entry.names:
                identity = canonical_path_identity(name, "behavior kind name")
                if identity in index:
                    raise BehaviorKindError(
                        f"behavior kind name collides with an existing entry: {name}"
                    )
                index[identity] = token
        ordered = {
            token: normalized[token]
            for token in sorted(
                normalized, key=lambda name: canonical_path_identity(name, "behavior kind token")
            )
        }
        object.__setattr__(self, "entries", MappingProxyType(ordered))
        object.__setattr__(self, "_index", MappingProxyType(index))

    # ── 读 ─────────────────────────────────────────────────────────────────────────

    @property
    def kinds(self) -> Mapping[str, tuple[str, ...]]:
        """token → 别名（登记视图；不含账）。"""

        return MappingProxyType({token: entry.aliases for token, entry in self.entries.items()})

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(self.entries)

    @property
    def kind_count(self) -> int:
        return len(self.entries)

    def entry_of(self, token: str) -> BehaviorKindEntry:
        return self.entries[self._require_token(token)]

    def aliases_of(self, token: str) -> tuple[str, ...]:
        return self.entry_of(token).aliases

    def label_of(self, token: str) -> str:
        return self.entry_of(token).label

    def token_for(self, name: object) -> str | None:
        """按规范身份查这个名字属于哪一类；未登记返回 None。O(1)。"""

        identity = canonical_path_identity(
            semantic_name(name, "behavior kind name"), "behavior kind name"
        )
        return self._index.get(identity)

    def most_hit(self, limit: int) -> tuple[str, ...]:
        """命中天数最多的 token（并列按 token 身份序）。"""

        ranked = sorted(
            self.entries.values(),
            key=lambda entry: (-entry.hit_days_total, canonical_path_identity(entry.token, "behavior kind token")),
        )
        return tuple(entry.token for entry in ranked[: max(0, limit)])

    def expired(self, *, on: date, base_days: int, gap_multiplier: int) -> tuple[str, ...]:
        """到 ``on`` 这一天已过存活期限的 token（从未命中的条目不参与）。"""

        day = _day(on, "behavior kind expiry day")
        out = []
        for token, entry in self.entries.items():
            deadline = entry.expires_after(base_days=base_days, gap_multiplier=gap_multiplier)
            if deadline is not None and day > deadline:
                out.append(token)
        return tuple(out)

    # ── 写（返回新实例）────────────────────────────────────────────────────────────

    def with_new_kind(
        self, name: object, *, label: object | None = None, day: date | None = None
    ) -> BehaviorKindRegistry:
        """把一个未登记的名字登记为新类型：token = 名字本身；``day`` 给了就记首次命中。"""

        token = semantic_name(name, "behavior kind token")
        entry = BehaviorKindEntry(
            token=token, label=semantic_name(label, "behavior kind label") if label is not None else token
        )
        if day is not None:
            entry = entry.with_hit(day)
        return BehaviorKindRegistry({**dict(self.entries), token: entry})

    def with_alias(self, token: str, alias: object, *, day: date | None = None) -> BehaviorKindRegistry:
        """把一个未登记的名字并入既有类型作为别名；``day`` 给了顺带记命中。"""

        resolved = self._require_token(token)
        entry = self.entries[resolved]
        entry = replace(entry, aliases=(*entry.aliases, semantic_name(alias, "behavior kind alias")))
        if day is not None:
            entry = entry.with_hit(day)
        return self._replaced(resolved, entry)

    def with_hit(self, token: str, day: date) -> BehaviorKindRegistry:
        resolved = self._require_token(token)
        return self._replaced(resolved, self.entries[resolved].with_hit(day))

    def with_label(self, token: str, label: object) -> BehaviorKindRegistry:
        resolved = self._require_token(token)
        return self._replaced(
            resolved, replace(self.entries[resolved], label=semantic_name(label, "behavior kind label"))
        )

    def without(self, token: str) -> BehaviorKindRegistry:
        resolved = self._require_token(token)
        remaining = {key: value for key, value in self.entries.items() if key != resolved}
        return BehaviorKindRegistry(remaining)

    def merged(self, source: str, target: str) -> BehaviorKindRegistry:
        """把 ``source`` 并入 ``target``：别名与账合并进 target，source 条目删除。

        树上 source 的 occurrence 由写入层重打为 target——词表这里只提供映射与账。
        """

        src = self._require_token(source)
        dst = self._require_token(target)
        if src == dst:
            raise BehaviorKindError("cannot merge a behavior kind into itself")
        a, b = self.entries[src], self.entries[dst]
        days = tuple(sorted(set(a.hit_days) | set(b.hit_days)))[-HIT_DAYS_KEPT:]
        created = min((d for d in (a.created_on, b.created_on) if d is not None), default=None)
        merged = replace(
            b,
            aliases=(*b.aliases, a.token, *a.aliases),
            created_on=created,
            hit_days=days,
            hit_days_total=a.hit_days_total + b.hit_days_total,
            hit_count=a.hit_count + b.hit_count,
        )
        remaining = {key: value for key, value in self.entries.items() if key != src}
        remaining[dst] = merged
        return BehaviorKindRegistry(remaining)

    # ── 内部 ──────────────────────────────────────────────────────────────────────

    def _replaced(self, token: str, entry: BehaviorKindEntry) -> BehaviorKindRegistry:
        return BehaviorKindRegistry({**dict(self.entries), token: entry})

    def _require_token(self, token: str) -> str:
        resolved = semantic_name(token, "behavior kind token")
        if resolved not in self.entries:
            raise BehaviorKindError(f"behavior kind token is not registered: {resolved}")
        return resolved


__all__ = [
    "HIT_DAYS_KEPT",
    "BehaviorKindEntry",
    "BehaviorKindError",
    "BehaviorKindLimitError",
    "BehaviorKindRegistry",
]
