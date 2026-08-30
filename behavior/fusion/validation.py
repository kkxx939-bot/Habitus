"""判断批次与输入片段之间的确定性校验。

## 只检查自洽，不规定现实

约束分两类，这里**只有第一类**：

    自洽性——违反了说明我们自己的产物内部矛盾
      引用的片段编号必须落在本段范围内
      每个主体都必须在它覆盖的片段里出现过
      关系必须指向本批已声明的判断
      同一主体、同一时刻开始的同名行为只能声明一条判断（同一件事被记了两遍）

    现实形状——违反了只说明现实不长我们想的那样，不该写成硬失败
      一帧只能属于一条判断        （并行时现实就是属于两边）
      全部片段必须无重无漏地划分  （观测本来就不完整）
      一条事实的证据帧必须连续    （帧连不连续是上游数据的事实）

历史上被真实数据推翻的约束一条不落全在第二类。穷尽性由 ``frames`` 的形状承担，不在这里重复。

## 抗幻觉靠绑定，不靠语气

语义字段必须绑定片段编号，于是"编造"就变成"引用了不存在的编号"，可以确定性检测。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from behavior.fusion.errors import BehaviorFusionError
from behavior.fusion.judgement import BehaviorJudgementBatch, JudgementRelation
from behavior.observation import BehaviorObservation


def validate_judgement_batch(
    batch: BehaviorJudgementBatch,
    fragments: Sequence[BehaviorObservation],
) -> None:
    """按输入片段校验一批判断；不满足任何一条自洽性即抛出。

    ``fragments`` 按位置对应临时编号 ``1..N``，与提示词里呈现给模型的编号一致。
    """

    if not isinstance(batch, BehaviorJudgementBatch):
        raise TypeError("batch must be BehaviorJudgementBatch")
    if isinstance(fragments, (str, bytes)) or not isinstance(fragments, Sequence) or not fragments:
        raise BehaviorFusionError("fusion requires a non-empty fragment sequence")
    if any(not isinstance(item, BehaviorObservation) for item in fragments):
        raise TypeError("fragments must contain BehaviorObservation values")
    by_no = {index: fragment for index, fragment in enumerate(fragments, start=1)}

    _require_known_fragments(batch, by_no)
    _require_subject_present(batch, by_no)
    _require_one_judgement_per_start(batch, by_no)
    _require_ordering(batch)


def _require_known_fragments(
    batch: BehaviorJudgementBatch, by_no: Mapping[int, BehaviorObservation]
) -> None:
    """任何引用都必须落在本批片段编号内；编造语义会在这里现形。"""

    referenced: set[int] = set()
    for judgement in batch.judgements:
        referenced.update(judgement.covers)
        for fact in judgement.claim.basis:
            referenced.update(fact.fragment_nos)
    unknown = sorted(referenced - set(by_no))
    if unknown:
        raise BehaviorFusionError(f"judgements reference fragments outside this segment: {unknown}")


def _require_subject_present(
    batch: BehaviorJudgementBatch, by_no: Mapping[int, BehaviorObservation]
) -> None:
    """每个主体都必须在它覆盖的片段里出现过——否则这条判断说的是观测里没有的人。

    装配层拿到 ``participants_by_no`` 时已在那里降级（剔名 / 整条降为没读懂），这里是后置断言：
    经装配的批次不该再触发；直接构造的批次或未给 participants 的装配仍由它把关。
    """

    for judgement in batch.judgements:
        if not judgement.subjects:
            continue
        available = {
            participant
            for fragment_no in judgement.covers
            for participant in by_no[fragment_no].participants
        }
        unknown = sorted(set(judgement.subjects) - available)
        if unknown and len(unknown) < len(judgement.subjects):
            raise BehaviorFusionError(
                f"judgement[{judgement.judgement_no}] names subjects absent from its fragments: "
                f"{unknown}"
            )
        # 主体**全部**不在场：这是"观测里没有的人做的事"，对被跟踪主体而言是旁人之事——回执按
        # out_of_scope 记、不落盘（装配层已留 subject_absent 信号），不在这里硬拒。


def _require_one_judgement_per_start(
    batch: BehaviorJudgementBatch, by_no: Mapping[int, BehaviorObservation]
) -> None:
    """同一主体、同一时刻开始的同名行为必须是同一条判断——判重只在这一层解决。

    "同一时刻"取覆盖片段里最早的 ``occurred_at``，正是归约后行为树地址的 ``started_at``。
    两条这样的判断各自声称自己就是那件事，却互不相认——这是产物的内部矛盾，不是现实的形状：
    同一个人不可能在同一瞬间把同一件事**开始两次**，那只是同一件事被看了两遍。下游对此一律
    不再判断（写入层的序号后缀只是防卡死的保命阀，预测树与语义层拿到什么算什么），所以重复
    只能在这里、由模型合并掉。

    延续/修正链上的两条不算矛盾：continues 归约时并成同一条 occurrence，supersedes 只留一条。
    机械合并不做——挑谁的 goal/summary 存活是发明语义；抛错走反馈重试，让模型自己折叠。
    """

    linked: dict[int, set[int]] = {}
    for item in batch.judgements:
        for link in item.relations:
            if link.target_no is None or link.kind not in (
                JudgementRelation.CONTINUES,
                JudgementRelation.SUPERSEDES,
            ):
                continue
            linked.setdefault(item.judgement_no, set()).add(link.target_no)
            linked.setdefault(link.target_no, set()).add(item.judgement_no)

    component: dict[int, int] = {}
    for item in batch.judgements:
        if item.judgement_no in component:
            continue
        component[item.judgement_no] = item.judgement_no
        stack = [item.judgement_no]
        while stack:
            for neighbour in linked.get(stack.pop(), ()):
                if neighbour not in component:
                    component[neighbour] = item.judgement_no
                    stack.append(neighbour)

    seen: dict[tuple[frozenset[str], str | None, object], int] = {}
    for item in batch.judgements:
        if not item.claim.is_readable:
            continue
        started_at = min(by_no[no].occurred_at for no in item.covers)
        key = (frozenset(item.subjects), item.claim.behavior, started_at)
        earlier = seen.get(key)
        if earlier is None:
            seen[key] = item.judgement_no
        elif component[earlier] != component[item.judgement_no]:
            raise BehaviorFusionError(
                f"judgement[{item.judgement_no}] and judgement[{earlier}] both describe "
                f"'{item.claim.behavior}' starting at the same moment: that is one behaviour "
                f"seen twice — merge them into one judgement, or use supersedes if one "
                f"corrects the other"
            )


def _require_ordering(batch: BehaviorJudgementBatch) -> None:
    """判断按最早覆盖片段排列；并行时"时间顺序"因此有确定含义。

    这是装配层的归约产物，在这里复核一次是纵深防御——顺序一旦错乱，下游按顺序做的任何归约
    （合并延续链、物化 occurrence）都会跟着错。
    """

    earliest = [item.earliest_fragment_no for item in batch.judgements]
    if earliest != sorted(earliest):
        raise BehaviorFusionError("judgements must be ordered by their earliest fragment")


def unreadable_ratio(batch: BehaviorJudgementBatch, fragment_count: int) -> float:
    """读不懂的片段占比；这个数上升即上游语义在退化。"""

    if isinstance(fragment_count, bool) or not isinstance(fragment_count, int) or fragment_count <= 0:
        raise BehaviorFusionError("fragment_count must be a positive integer")
    return batch.unreadable_fragment_count / fragment_count


__all__ = ["unreadable_ratio", "validate_judgement_batch"]
