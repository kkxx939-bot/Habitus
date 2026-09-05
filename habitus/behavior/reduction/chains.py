"""判断链的机械组装：supersedes 就地替换、continues 并链。

零语义、纯函数：这里只解释融合层已经声明的关系边，不发明任何判断。规则（均为
``TODO(BHV-TREE-REBUILD-001)`` 的既定裁定）：

- **supersedes 就地替换**：被替换的判断退出视图（全史仍留判断存储、随链一并消费），替换者
  继承它的 continues 结构边——链的形状不因修正而断开；被替换判断的其余关系边（concurrent_with /
  results_from）随判断本身作废。同一目标被多条判断替换时，取 ``(evidence_ready_at,
  judgement_id)`` 最大者，其余的替换边作废（确定性，不选"更好"的）。
- **continues 并链**：可读判断之间的 continues 边把它们并成一条链；一条链归约成一条 occurrence。
- **没读懂段是单点**：融合层禁止没读懂判断携带关系；指向它的 supersedes 表示"后来读懂了这段"，
  该段随替换者一并消费、不再物化成 gap；指向它的其他关系边机械作废（没读懂段无法当并行/因果的
  目标）。

所有被作废的边都记进 ``dropped_edges``——机械丢弃是降级不是解释，必须留下信号。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from habitus.behavior.reduction.errors import BehaviorReductionError
from habitus.behavior.reduction.record import ReducibleJudgement
from habitus.foundation.integrity import canonical_digest

_CONTINUES = "continues"
_SUPERSEDES = "supersedes"
_CROSS_KINDS = ("concurrent_with", "results_from")


@dataclass(frozen=True)
class BehaviorChain:
    """一条闭合前的判断链：修正后视图 + 被替换的全史。"""

    view: tuple[ReducibleJudgement, ...]
    superseded: tuple[ReducibleJudgement, ...]

    def __post_init__(self) -> None:
        if not self.view:
            raise BehaviorReductionError("a behavior chain must have a non-empty view")

    @property
    def head(self) -> ReducibleJudgement:
        return self.view[0]

    @property
    def tail(self) -> ReducibleJudgement:
        return self.view[-1]

    @property
    def consumed(self) -> tuple[ReducibleJudgement, ...]:
        """本链消费的全部判断（视图 + 被替换者），按确定性顺序。"""

        return tuple(
            sorted((*self.view, *self.superseded), key=lambda item: (item.evidence_ready_at, item.judgement_id))
        )

    @property
    def newest_evidence_at(self) -> datetime:
        """链上最晚成立的判断时刻——封口判据的输入。"""

        return max(item.evidence_ready_at for item in self.consumed)

    @property
    def chain_digest(self) -> str:
        """链身份：消费的判断身份集合的内容摘要；消费账本以它为键。"""

        return canonical_digest(
            {"judgement_ids": sorted(item.judgement_id for item in self.consumed)}
        )

    @property
    def order_key(self) -> tuple[datetime, str]:
        """批内全序：链头行为时刻 + 链头身份。前向边只指向严格更小的键。"""

        return self.head.order_key

    @property
    def semantic_edges(self) -> tuple[tuple[str, str], ...]:
        """视图成员声明的跨链语义边 ``(kind, 目标判断身份)``，去重定序。

        目标可能在本批（由 ``ChainAssembly.cross_links_of`` 解析）也可能早已消费（由 runner
        拿消费账本解析成树 URI）；这里只如实枚举。
        """

        seen: dict[tuple[str, str], None] = {}
        for member in self.view:
            for kind, target in member.relations:
                if kind in _CROSS_KINDS:
                    seen.setdefault((kind, target), None)
        return tuple(seen)


@dataclass(frozen=True)
class ChainAssembly:
    """一次组装的完整产物。"""

    chains: tuple[BehaviorChain, ...]
    gaps: tuple[ReducibleJudgement, ...]
    absorbed_unreadable: tuple[ReducibleJudgement, ...]
    dropped_edges: tuple[str, ...]
    chain_of: Mapping[str, int]
    forward_links: tuple[tuple[tuple[str, int], ...], ...]
    # 被隔离（时间倒挂矛盾链、supersedes 成环）而滞留存储的判断身份：runner 据此给指向它们的
    # 边以准确的作废理由，而不是误导性的 "neither reducible nor consumed"。
    quarantined_ids: frozenset[str] = frozenset()

    def cross_links_of(self, chain_index: int) -> tuple[tuple[str, int], ...]:
        """一条链保留的批内前向语义边 ``(kind, 目标链下标)``，已在组装时规范化并去重。

        规范化规则（零语义、只处理方向）：add-only 树只存前向、读侧取对称闭包。
        concurrent_with 是**对称**关系、融合允许任一边声明——不论声明在哪一边，统一挂到
        ``order_key`` 更晚的链上指回更早的链，不丢弃；results_from 有方向（结果指向原因），
        指向更晚链的边机械作废并留信号。指回本链、指向没读懂段的边同样作废并留信号。
        """

        return self.forward_links[chain_index]


def _replacements(
    records: Sequence[ReducibleJudgement], by_id: Mapping[str, ReducibleJudgement]
) -> tuple[dict[str, str], list[str]]:
    """supersedes 的胜出映射（目标 → 替换者）与作废边。"""

    dropped: list[str] = []
    winners: dict[str, ReducibleJudgement] = {}
    for record in records:
        if not record.is_readable:
            continue
        for kind, target_id in record.relations:
            if kind != _SUPERSEDES:
                continue
            if target_id not in by_id:
                dropped.append(
                    f"supersedes from {record.judgement_id} dropped: target {target_id} is not reducible"
                )
                continue
            current = winners.get(target_id)
            if current is None or (record.evidence_ready_at, record.judgement_id) > (
                current.evidence_ready_at,
                current.judgement_id,
            ):
                if current is not None:
                    dropped.append(
                        f"supersedes from {current.judgement_id} dropped: "
                        f"{record.judgement_id} replaces {target_id} instead"
                    )
                winners[target_id] = record
            else:
                dropped.append(
                    f"supersedes from {record.judgement_id} dropped: "
                    f"{current.judgement_id} replaces {target_id} instead"
                )
    return {target: winner.judgement_id for target, winner in winners.items()}, dropped


def _follow(replaced: Mapping[str, str], judgement_id: str) -> str:
    """穿透替换链取最终身份；出现环时停在环内 id 最大者（确定性降级，不会发生于合法产物）。"""

    seen: set[str] = set()
    current = judgement_id
    while current in replaced and current not in seen:
        seen.add(current)
        current = replaced[current]
    if current in seen:
        return max(seen)
    return current


def assemble_chains(records: Sequence[ReducibleJudgement]) -> ChainAssembly:
    """把一批未消费判断组装成链与空白段；输入相同则输出逐位相同。"""

    if isinstance(records, str | bytes) or not isinstance(records, Sequence):
        raise BehaviorReductionError("records must be a sequence of ReducibleJudgement")
    ordered = tuple(sorted(records, key=lambda item: (item.evidence_ready_at, item.judgement_id)))
    by_id = {item.judgement_id: item for item in ordered}
    if len(by_id) != len(ordered):
        raise BehaviorReductionError("judgement records contain duplicate identities")

    replaced, dropped = _replacements(ordered, by_id)

    # 没读懂段：被 supersedes 认领的（后来读懂了）并入替换者链的全史、随链消费——只从 gaps
    # 里排除是不够的：替换关系随链消费后就从待归约集合里消失，下一轮它会被错当成无主空白。
    absorbed_ids = {target for target in replaced if not by_id[target].is_readable}
    gaps = tuple(
        item for item in ordered if not item.is_readable and item.judgement_id not in absorbed_ids
    )
    absorbed = tuple(by_id[target] for target in sorted(absorbed_ids))

    # 并查集：continues 结构边（穿透替换）把可读判断并成链；被替换的可读判断把自己的
    # 结构边过继给替换者。
    members = [item for item in ordered if item.is_readable and item.judgement_id not in replaced]
    parent: dict[str, str] = {item.judgement_id: item.judgement_id for item in members}

    def find(node: str) -> str:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            # 定序合并：小 id 作根，保证同输入同结构。
            if left_root > right_root:
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root

    for record in ordered:
        if not record.is_readable:
            continue
        owner = _follow(replaced, record.judgement_id)
        for kind, target_id in record.relations:
            if kind != _CONTINUES:
                continue
            target = _follow(replaced, target_id)
            if target not in by_id:
                dropped.append(
                    f"continues from {record.judgement_id} dropped: target {target_id} is not reducible"
                )
                continue
            if not by_id[target].is_readable:
                dropped.append(
                    f"continues from {record.judgement_id} dropped: target {target_id} is unreadable"
                )
                continue
            if owner == target:
                dropped.append(
                    f"continues from {record.judgement_id} dropped: it resolves to its own chain"
                )
                continue
            union(owner, target)

    groups: dict[str, list[ReducibleJudgement]] = {}
    for item in members:
        groups.setdefault(find(item.judgement_id), []).append(item)

    # "新判断即新链头"（规格字面）：替换者**严格**继承被替换者在链中的位置，不与自己的时间取
    # min、不重新排队——位置是链的结构，时间只是它自带的内容；否则时间修正会让链头易主、行为
    # 改名（评审抓过：min() 会让"改早"方向静默换头）。改早产生的"延续早于事件开始"倒挂由下方
    # 矛盾隔离接住，与改晚方向同一处置——两个方向都是同一类产物内部矛盾。
    effective_key: dict[str, tuple[datetime, str]] = {
        item.judgement_id: item.order_key for item in members
    }
    inherited: dict[str, tuple[datetime, str]] = {}
    for target_id in sorted(replaced, key=lambda tid: by_id[tid].order_key):
        final = _follow(replaced, target_id)
        if final in effective_key:
            key = by_id[target_id].order_key
            if final not in inherited or key < inherited[final]:
                inherited[final] = key
    effective_key.update(inherited)

    # 被替换的判断（可读的、以及被认领的没读懂段）都归入替换者所在链的全史，随链一并消费。
    stranded: set[str] = set()
    superseded_by_root: dict[str, list[ReducibleJudgement]] = {}
    for target_id in replaced:
        final = _follow(replaced, target_id)
        if final in parent:
            superseded_by_root.setdefault(find(final), []).append(by_id[target_id])
        else:
            # 只可能是 supersedes 成环（合法产物推不出环）：这些判断本轮不被任何链消费，
            # 每轮重扫都会再次报出——内部矛盾必须持续可见，不许静默吞掉。
            dropped.append(
                f"supersedes chain for {target_id} is cyclic; its judgements stay unconsumed"
            )
            stranded.add(target_id)

    chains: list[BehaviorChain] = []
    quarantined: set[str] = set(stranded)
    for root in sorted(groups):
        view = tuple(
            sorted(groups[root], key=lambda item: effective_key[item.judgement_id])
        )
        superseded = tuple(
            sorted(superseded_by_root.get(root, ()), key=lambda item: item.order_key)
        )
        # 矛盾链不进树（用户裁定）：修正把开始时间改到与链的位置结构倒挂——一件事的延续不可能
        # 早于它自己的开始，改晚链头、改早中段都是同一类产物内部矛盾（同 supersedes 成环）。
        # 不消费、不落树、每轮留信号，持续可见等处置。
        earliest = min(item.started_at for item in view)
        if earliest < view[0].started_at:
            dropped.append(
                f"chain headed by {view[0].judgement_id} quarantined: a correction moved its "
                f"start past one of its own continuation segments"
            )
            quarantined.update(item.judgement_id for item in (*view, *superseded))
            continue
        chains.append(BehaviorChain(view=view, superseded=superseded))
    chains.sort(key=lambda chain: chain.order_key)

    chain_of: dict[str, int] = {}
    for index, chain in enumerate(chains):
        for item in chain.consumed:
            chain_of[item.judgement_id] = index

    # 批内跨链边的规范化（cross_links_of 返回的就是这份结果）。目标既不在本批也不是没读懂段
    # 的边留给 runner——那可能是早已消费、URI 在账本里的合法目标，本函数看不到账本。
    gap_ids = {item.judgement_id for item in gaps}
    forward: list[dict[tuple[str, int], None]] = [{} for _ in chains]
    for index, chain in enumerate(chains):
        for member in chain.view:
            for kind, target_id in member.relations:
                if kind not in _CROSS_KINDS:
                    continue
                target_index = chain_of.get(target_id)
                if target_index is None:
                    if target_id in gap_ids:
                        dropped.append(
                            f"{kind} from {member.judgement_id} dropped: "
                            f"target {target_id} is an unreadable stretch"
                        )
                    continue
                if target_index == index:
                    dropped.append(
                        f"{kind} from {member.judgement_id} dropped: it points into its own chain"
                    )
                elif kind == "concurrent_with":
                    # 对称关系：统一挂到更晚的链上指回更早的链——模型声明在哪一边都不丢。
                    late, early = (
                        (index, target_index)
                        if chains[index].order_key > chains[target_index].order_key
                        else (target_index, index)
                    )
                    forward[late][(kind, early)] = None
                elif chains[target_index].order_key < chain.order_key:
                    forward[index][(kind, target_index)] = None
                else:
                    dropped.append(
                        f"{kind} from {member.judgement_id} dropped: "
                        f"a result cannot point at a chain that starts later"
                    )
        # 被替换判断的跨链边随判断本身作废——机械丢弃必须留信号。
        for member in chain.superseded:
            for kind, _target_id in member.relations:
                if kind in _CROSS_KINDS:
                    dropped.append(
                        f"{kind} from superseded {member.judgement_id} dropped: "
                        f"its judgement was replaced"
                    )

    return ChainAssembly(
        chains=tuple(chains),
        gaps=gaps,
        absorbed_unreadable=absorbed,
        dropped_edges=tuple(dropped),
        chain_of=chain_of,
        forward_links=tuple(tuple(edges) for edges in forward),
        quarantined_ids=frozenset(quarantined),
    )


__all__ = ["BehaviorChain", "ChainAssembly", "assemble_chains"]
