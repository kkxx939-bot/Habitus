"""把模型的逐帧标签装配成判断批次。

## 为什么不让模型直接输出分组

模型有两种输出，可靠性差一个数量级：

- **局部语义判断**——"这一帧和上一帧是不是同一件事？它服务什么目标？"实测稳定。
- **全局结构约束**——"把 N 个编号无重无漏地分成若干组"。实测三次全错，漏掉的全是位移与过渡帧。

后者是组合记账，不是语义任务，序列越长越差。所以线格式让模型逐帧回答"这一帧属于谁"，
``covers``、``basis`` 的帧集合、判断之间的顺序全部由本模块归约——模型碰不到，也就错不了。

## 归约出来的东西

    covers            某条判断的 frames 行的并集
    basis.fragment_nos 归到该 basis 条目的 frames 行
    判断顺序           按最早覆盖片段排

## 这里只检查自洽，不检查现实

会失败的都是"我们自己的产物内部矛盾"：引用了没声明的判断编号、声明了没有任何帧归属的 basis
条目、frames 行数与输入对不上。**不会**因为"一帧属于两条判断"或"某条 basis 的帧不连续"而失败
——那些是上游数据的事实，不是本层能规定的。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from behavior.fusion.config import BehaviorFusionConfig
from behavior.fusion.errors import BehaviorFusionError, BehaviorFusionLimitError
from behavior.fusion.judgement import (
    BehaviorClaim,
    BehaviorFact,
    BehaviorJudgement,
    BehaviorJudgementBatch,
    JudgementLink,
    JudgementRelation,
    JudgementStatus,
    JudgementStatusBasis,
)

_BATCH_KEYS = {"judgements", "frames"}
_JUDGEMENT_KEYS = {
    "judgement_no",
    "subjects",
    "behavior",
    "goal",
    "summary",
    "basis",
    "status",
    "status_basis",
    "relations",
}
_RELATION_KEYS = {"kind", "target", "context_target"}
_FACT_KEYS = {"basis_no", "semantics"}
_FRAME_KEYS = {"no", "assignments"}
_ASSIGNMENT_KEYS = {"judgement_no", "basis_no"}


def assemble_judgement_batch(
    value: object,
    *,
    fragment_count: int,
    context_states: Sequence[str | None] = (),
    config: BehaviorFusionConfig | None = None,
    participants_by_no: Mapping[int, Sequence[str]] | None = None,
) -> BehaviorJudgementBatch:
    """把线格式装配成判断批次；结构性错误在这里以精确坐标失败。

    ``participants_by_no`` 是各片段（按 ``1..N``）的参与者；给了它，主体不在覆盖片段里出现的
    判断在这里降级（剔名或整条降为没读懂），而不是等到校验层整批拒绝。

    ## 记账疏漏一律降级、留信号，不整批拒

    真实日尺度数据实测（BHV-REALDATA-001）：同帧重复归属、主体不在场、``continues`` 指向已完成的
    先前判断、goal 的 basis 全无帧归属，一周合计上千次；每次整批拒绝都要白烧三次模型调用再退避，
    而串行队列里一个段反复失败就封死一整天。这四类都是模型两张表之间的记账疏漏或判断习惯，
    不是语义不可用——降级后产物仍然自洽（校验层的断言照跑），损失以 ``degradations`` 逐条留痕。
    """

    resolved = config or BehaviorFusionConfig()
    if not isinstance(resolved, BehaviorFusionConfig):
        raise TypeError("config must be BehaviorFusionConfig")
    if isinstance(fragment_count, bool) or not isinstance(fragment_count, int) or fragment_count <= 0:
        raise BehaviorFusionError("fragment_count must be a positive integer")
    if isinstance(context_states, (str, bytes)) or not isinstance(context_states, Sequence):
        raise BehaviorFusionError("context_states must be a sequence")
    if participants_by_no is not None and not isinstance(participants_by_no, Mapping):
        raise BehaviorFusionError("participants_by_no must be a mapping")
    payload = _mapping(value, _BATCH_KEYS, "fusion output")
    notes: list[str] = []

    declared = _judgements(payload["judgements"], config=resolved)
    frames = _frames(payload["frames"], fragment_count=fragment_count, config=resolved, notes=notes)
    covers, fact_frames, dropped = _reduce_coverage(frames, declared)
    if not covers:
        # 全部帧都无归属、连一条判断都没有：这段观测什么都没发生也要有交代——一条没读懂判断
        # 不对（读懂了），所以这里明确失败让模型至少声明一条（哪怕只是覆盖边界的动作段）。
        raise BehaviorFusionError("frames leave every judgement without a fragment; declare what happened")

    built = [
        _build(
            number,
            declared[number],
            covers[number],
            fact_frames,
            dropped,
            participants_by_no=participants_by_no,
            notes=notes,
        )
        for number in sorted(covers, key=lambda no: (covers[no][0], no))
    ]
    built = _require_targets_declared(built, context_states=tuple(context_states), notes=notes)
    _require_distinguishable(built)
    owned = {no for item in built for no in item.covers}
    unowned = tuple(no for no in range(1, fragment_count + 1) if no not in owned)
    return BehaviorJudgementBatch(
        _close_concurrency(built), degradations=tuple(notes), unowned_fragment_nos=unowned
    )


# --- 线格式解析 -----------------------------------------------------------------------


def _judgements(value: object, *, config: BehaviorFusionConfig) -> dict[int, dict[str, Any]]:
    raw = _array(value, "judgements", max_items=config.max_judgements)
    declared: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(raw):
        label = f"judgements[{index}]"
        payload = _mapping(item, _JUDGEMENT_KEYS, label)
        number = _positive_int(payload["judgement_no"], f"{label}.judgement_no")
        if number in declared:
            raise BehaviorFusionError(f"{label}.judgement_no is declared more than once: {number}")
        declared[number] = _judgement_fields(payload, label, config=config)
    if not declared:
        raise BehaviorFusionError("judgements must declare at least one judgement")
    return declared


def _judgement_fields(
    payload: Mapping[str, Any], label: str, *, config: BehaviorFusionConfig
) -> dict[str, Any]:
    behavior = _optional_text(payload["behavior"], f"{label}.behavior", max_chars=config.max_name_chars)
    facts = _facts(payload["basis"], label, config=config)
    return {
        "subjects": _text_tuple(
            payload["subjects"], f"{label}.subjects", max_chars=config.max_name_chars
        ),
        "behavior": behavior,
        "goal": _optional_text(payload["goal"], f"{label}.goal", max_chars=config.max_text_chars),
        "summary": _optional_text(payload["summary"], f"{label}.summary", max_chars=config.max_text_chars),
        "facts": facts,
        "status": _enum_member(payload["status"], JudgementStatus, f"{label}.status", allow_none=True),
        "status_basis": _enum_member(
            payload["status_basis"], JudgementStatusBasis, f"{label}.status_basis", allow_none=True
        ),
        "relations": _relations(payload["relations"], label, config=config),
    }


def _relations(
    value: object, label: str, *, config: BehaviorFusionConfig
) -> tuple[JudgementLink, ...]:
    raw = _array(value, f"{label}.relations", max_items=config.max_judgements)
    links: list[JudgementLink] = []
    for index, item in enumerate(raw):
        item_label = f"{label}.relations[{index}]"
        payload = _mapping(item, _RELATION_KEYS, item_label)
        kind = _enum_member(payload["kind"], JudgementRelation, f"{item_label}.kind", allow_none=False)
        target = payload["target"]
        context_target = payload["context_target"]
        if (target is None) == (context_target is None):
            raise BehaviorFusionError(
                f"{item_label} must name exactly one of target / context_target"
            )
        links.append(
            JudgementLink(
                kind=kind,
                target_no=None if target is None else _positive_int(target, f"{item_label}.target"),
                context_no=(
                    None
                    if context_target is None
                    else _positive_int(context_target, f"{item_label}.context_target")
                ),
            )
        )
    return tuple(links)


def _facts(value: object, label: str, *, config: BehaviorFusionConfig) -> dict[int, str]:
    raw = _array(value, f"{label}.basis", max_items=config.max_basis_facts)
    facts: dict[int, str] = {}
    for index, item in enumerate(raw):
        item_label = f"{label}.basis[{index}]"
        payload = _mapping(item, _FACT_KEYS, item_label)
        number = _positive_int(payload["basis_no"], f"{item_label}.basis_no")
        if number in facts:
            raise BehaviorFusionError(f"{item_label}.basis_no is declared more than once: {number}")
        facts[number] = _text(payload["semantics"], f"{item_label}.semantics", max_chars=config.max_text_chars)
    return facts


def _frames(
    value: object, *, fragment_count: int, config: BehaviorFusionConfig, notes: list[str]
) -> tuple[dict[str, Any], ...]:
    """逐帧归属表；穷尽性在这里由**形状**保证，而不是事后校验。"""

    raw = _array(value, "frames", max_items=config.max_fragments_per_segment)
    if len(raw) != fragment_count:
        raise BehaviorFusionError(
            f"frames must contain exactly one row per input fragment: "
            f"expected {fragment_count}, got {len(raw)}"
        )
    frames: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        label = f"frames[{index - 1}]"
        payload = _mapping(item, _FRAME_KEYS, label)
        number = _positive_int(payload["no"], f"{label}.no")
        if number != index:
            raise BehaviorFusionError(f"{label}.no must be {index}, got {number}")
        assignments = _assignments(payload["assignments"], label, config=config, notes=notes)
        frames.append({"no": number, "assignments": assignments})
    return tuple(frames)


def _assignments(
    value: object, label: str, *, config: BehaviorFusionConfig, notes: list[str]
) -> tuple[tuple[int, int | None], ...]:
    raw = _array(value, f"{label}.assignments", max_items=config.max_judgements)
    # 空列表是显式声明："看到了、看懂了、不构成任何事"（无意识小动作、过渡帧）。行必须存在
    # ——穷尽性靠行数保证；无归属是声明不是省略。
    resolved: list[tuple[int, int | None]] = []
    for index, item in enumerate(raw):
        item_label = f"{label}.assignments[{index}]"
        payload = _mapping(item, _ASSIGNMENT_KEYS, item_label)
        judgement_no = _positive_int(payload["judgement_no"], f"{item_label}.judgement_no")
        basis_no = payload["basis_no"]
        if basis_no is not None:
            basis_no = _positive_int(basis_no, f"{item_label}.basis_no")
        if any(existing == judgement_no for existing, _ in resolved):
            # 一帧属于两条**不同**判断是现实（并行的交界帧）；一帧在同一条判断里出现两次在现实里
            # 没有对应物，只是模型把 judgement_no 写了两遍（或给了两个 basis）。取先写的那个：
            # 一帧在一条判断里只归一个步骤，信息损失可忽略；整批拒则白烧一次调用。
            notes.append(f"duplicate_assignment {label} judgement={judgement_no}")
            continue
        resolved.append((judgement_no, basis_no))
    return tuple(resolved)


# --- 归约 -----------------------------------------------------------------------------


def _reduce_coverage(
    frames: Sequence[Mapping[str, Any]], declared: Mapping[int, Mapping[str, Any]]
) -> tuple[
    dict[int, tuple[int, ...]],
    dict[tuple[int, int], tuple[int, ...]],
    frozenset[int],
]:
    """从逐帧标签归约出每条判断覆盖的片段、每个 basis 条目覆盖的片段，以及被丢弃的判断编号。"""

    covers: dict[int, list[int]] = {number: [] for number in declared}
    fact_frames: dict[tuple[int, int], list[int]] = {}
    for frame in frames:
        for judgement_no, basis_no in frame["assignments"]:
            judgement = declared.get(judgement_no)
            if judgement is None:
                raise BehaviorFusionError(
                    f"frame {frame['no']} references an undeclared judgement: {judgement_no}"
                )
            covers[judgement_no].append(frame["no"])
            if basis_no is None:
                continue
            if basis_no not in judgement["facts"]:
                # 又一处两张表之间的记账疏漏：``judgements`` 里只声明了 basis 1、2，``frames``
                # 里却把某一帧标成了 basis 3。**这一帧归给哪条判断是清楚的**，错的只是子分组，
                # 所以降级成"归给这条判断、不归任何 basis"，而不是整批拒绝白烧一次调用。
                # 实测这条在真实模型上是间歇性的（同一输入三次里犯两次），靠提示词根治不了。
                continue
            fact_frames.setdefault((judgement_no, basis_no), []).append(frame["no"])
    # 声明了却没有任何帧归属的判断、以及没有任何帧归属的 basis 条目，一律**丢弃**而不是整批拒绝。
    #
    # 它们是模型的记账疏漏——``judgements`` 与 ``frames`` 是分开写的两段，声明一条却忘了在
    # frames 里给它任何一帧非常自然，而这类组合式记账正是实测中模型做不好的部分。整批拒绝要
    # 白烧一次完整调用、吃掉一次重试预算，用尽后整个作业失败退避；而丢弃**不损失任何观测**：
    # 一条判断若没有任何帧，就没有任何片段以它为归属（有归属的帧会让 covers 非空），basis 同理。
    #
    empty = {number for number, items in covers.items() if not items}
    covers = {number: items for number, items in covers.items() if number not in empty}
    fact_frames = {key: items for key, items in fact_frames.items() if key[0] not in empty}
    return (
        {number: tuple(items) for number, items in covers.items()},
        {key: tuple(items) for key, items in fact_frames.items()},
        frozenset(empty),
    )


def _build(
    number: int,
    fields: Mapping[str, Any],
    covers: tuple[int, ...],
    fact_frames: Mapping[tuple[int, int], tuple[int, ...]],
    dropped: frozenset[int],
    *,
    participants_by_no: Mapping[int, Sequence[str]] | None,
    notes: list[str],
) -> BehaviorJudgement:
    facts: Mapping[int, str] = fields["facts"]
    # 只保留真的有帧归属的 basis 条目；声明了却没有任何帧的那些在归约时已被丢弃（见
    # ``_reduce_coverage``），它们没有任何观测支撑，留着等于把模型的记账疏漏写进产物。
    grounded = [basis_no for basis_no in facts if (number, basis_no) in fact_frames]
    # basis 条目按它最早覆盖的片段排序——顺序是归约产物，让模型再写一遍只是多一个出错的机会，
    # 而真正同时发生的两条事实其帧本来就交错，"谁在前"没有意义。
    ordered = sorted(grounded, key=lambda basis_no: (fact_frames[(number, basis_no)][0], basis_no))
    behavior = fields["behavior"]
    goal = fields["goal"]
    summary = fields["summary"]
    subjects = fields["subjects"]
    status = fields["status"]
    status_basis = fields["status_basis"]
    relations = tuple(link for link in fields["relations"] if link.target_no not in dropped)
    if behavior is not None and participants_by_no is not None:
        available = {
            participant
            for fragment_no in covers
            for participant in participants_by_no.get(fragment_no, ())
        }
        present = tuple(subject for subject in subjects if subject in available)
        absent = [subject for subject in subjects if subject not in available]
        if absent and present:
            # 模型从"她 / 大家 / 众人"推出了在场的人，而观测的 participants 里没有——只剔掉观测
            # 撑不住的名字，判断本身保留。
            notes.append(f"subject_absent judgement={number} dropped={absent}")
            subjects = present
        elif absent:
            # 没有一个主体是观测里出现过的：这条判断说的是观测里没有的人，不能作为可读判断落盘，
            # 但它覆盖的观测也不能丢——整条降为"没读懂"，与融合层"不能融合的也要留下"同构。
            # 它的 subjects/status/relations 随之清空（没读懂段不携带这些），指向它的关系由
            # ``_require_targets_declared`` 剪掉。
            notes.append(f"subject_absent judgement={number} dropped={absent} unreadable")
            behavior = goal = summary = None
            subjects = ()
            status = status_basis = None
            relations = ()
            ordered = []
    claim = BehaviorClaim(
        behavior=behavior,
        goal=goal,
        summary=summary,
        basis=tuple(
            BehaviorFact(semantics=facts[basis_no], fragment_nos=fact_frames[(number, basis_no)])
            for basis_no in ordered
        ),
    )
    return BehaviorJudgement(
        judgement_no=number,
        covers=covers,
        subjects=subjects,
        claim=claim,
        status=status,
        status_basis=status_basis,
        # 指向被丢弃判断的关系一并剪掉：目标已经不存在了，留着就是一条指向空处的连接。
        relations=relations,
    )


def _require_targets_declared(
    judgements: Sequence[BehaviorJudgement],
    *,
    context_states: tuple[str | None, ...],
    notes: list[str],
) -> list[BehaviorJudgement]:
    """关系目标必须存在且可读；``continues`` 指向已完成的目标则**剪掉这条边**并留信号。

    返回剪边之后的判断列表（判断本身不动）。剪边而不是整批拒：真实数据一周 172 次，模型看行为
    相似度不看状态（早上洗过手、中午再洗标成延续），贴在 C 行上的约束压不住；剪掉之后本段判断
    按独立的新一次行为落树、先前那条不动——只删一条自相矛盾的关系，不发明任何事实（改成
    supersedes 是替模型说它没说的话；并入已完成的链会让链无限滚长）。
    """

    known = {item.judgement_no: item for item in judgements}
    # 装配期被降为没读懂的判断（主体全不在场）：指向它的关系一律剪掉——它已不是一个行为。
    degraded_unreadable = {
        item.judgement_no
        for item in judgements
        if not item.claim.is_readable and any(
            note.startswith(f"subject_absent judgement={item.judgement_no} ") and note.endswith("unreadable")
            for note in notes
        )
    }
    rebuilt: list[BehaviorJudgement] = []
    for item in judgements:
        kept: list[JudgementLink] = []
        for link in item.relations:
            if link.context_no is not None:
                if link.context_no > len(context_states):
                    raise BehaviorFusionError(
                        f"judgement[{item.judgement_no}] relates to context judgement "
                        f"C{link.context_no}, but only {len(context_states)} were shown"
                    )
                if not _continuable(link, context_states[link.context_no - 1]):
                    notes.append(
                        f"continues_completed judgement={item.judgement_no} target=C{link.context_no}"
                    )
                    continue
                kept.append(link)
                continue
            assert link.target_no is not None  # 恰好一个非空，上面已经排除 context 分支
            target = known.get(link.target_no)
            if target is None:
                raise BehaviorFusionError(
                    f"judgement[{item.judgement_no}] relates to an undeclared judgement: "
                    f"{link.target_no}"
                )
            if link.target_no in degraded_unreadable:
                notes.append(
                    f"relation_dropped judgement={item.judgement_no} target={link.target_no} "
                    f"kind={link.kind.value} (target degraded to unreadable)"
                )
                continue
            if not target.claim.is_readable:
                # 没读懂的那段不是一个行为，谈不上与它延续、并行或被它修正。在这里以可定位的
                # 坐标拒绝，而不是等到对称补齐时由系统自己造出一条非法关系再炸——那样反馈给
                # 模型的错误落在它根本没写过的地方，它无从改起。
                raise BehaviorFusionError(
                    f"judgement[{item.judgement_no}] relates to judgement[{link.target_no}], "
                    f"which is unreadable"
                )
            if not _continuable(link, None if target.status is None else target.status.value):
                # 同批内的 continues 不剪：判重规则（validation._require_one_judgement_per_start）
                # 正是靠这条边把"同一件事被看成两条"认作一条、由归约并链；剪掉它反而让两条同名
                # 同刻判断互不相认、撞成硬拒（WP1 首次真实对照实测）。保留、留信号，归约按 continues
                # 并链，尾部状态定结局——模型明说这是同一件事，同批内并起来是最安全的解释。
                notes.append(
                    f"continues_completed judgement={item.judgement_no} target={link.target_no} kept"
                )
            kept.append(link)
        rebuilt.append(item if len(kept) == len(item.relations) else replace(item, relations=tuple(kept)))
    return rebuilt


def _require_distinguishable(judgements: Sequence[BehaviorJudgement]) -> None:
    """同一批里不能有两条内容与覆盖范围完全一样的判断。

    耐久身份由内容派生（不含临时编号），所以这样的两条会算出同一个 ``judgement_id``，落盘时
    静默合并成一条，而回执仍声称产出了两条。在这里拦是因为它是**模型的结构性错误、可以反馈重试**；
    放到落盘阶段才发现，就变成不可重试的失败并阻塞整条串行队列。
    """

    seen: dict[tuple[Any, ...], int] = {}
    for item in judgements:
        claim = item.claim
        key = (
            item.covers,
            item.subjects,
            claim.behavior,
            claim.goal,
            claim.summary,
            tuple((fact.semantics, fact.fragment_nos) for fact in claim.basis),
            item.status,
            item.status_basis,
        )
        earlier = seen.get(key)
        if earlier is not None:
            raise BehaviorFusionError(
                f"judgement[{item.judgement_no}] is indistinguishable from judgement[{earlier}]: "
                f"merge them into one, or make them differ"
            )
        seen[key] = item.judgement_no


def _continuable(link: JudgementLink, target_status: str | None) -> bool:
    """一件已经完成的事没有"后半段"。

    甲说自己已完成、乙说自己是甲的延续，这是我们自己两条产物之间的矛盾，不是现实的形状。
    模型若认为先前那条判错了（以为完成其实没完），正确表达是 ``supersedes``；它没这么说，
    我们就只剪掉这条矛盾的边（见 ``_require_targets_declared``），不替它改口。

    实测这条会被触发：早上洗一次手、四小时后又洗一次，模型三次全部标成 continues——时间跨度
    没能阻止它，判断主要看行为相似度；真实日尺度数据一周 172 次。
    """

    return link.kind is not JudgementRelation.CONTINUES or target_status != "completed"


def _close_concurrency(judgements: Sequence[BehaviorJudgement]) -> tuple[BehaviorJudgement, ...]:
    """并行取对称闭包：一边声明即可，另一边由系统补上。

    要求模型两边都写，等于让它写完后一条再回头改前一条——那是记账，不是语义判断，实测正是它
    在失败（「接电话」声明了与「做饭」并行，而「做饭」还停在独立）。并行本来就是对称的，补齐是
    确定性的，没有理由让模型来做。
    """

    extra: dict[int, list[JudgementLink]] = {}
    for item in judgements:
        for link in item.relations:
            if link.kind is not JudgementRelation.CONCURRENT_WITH or link.context_no is not None:
                # 指向先前判断的关系补不了对称——那条已经落盘、不可变。单向如实记录即可，读的
                # 一侧取对称闭包。
                continue
            target = next(other for other in judgements if other.judgement_no == link.target_no)
            declared = any(
                back.kind is JudgementRelation.CONCURRENT_WITH
                and back.target_no == item.judgement_no
                for back in target.relations
            )
            already = any(
                existing.target_no == item.judgement_no
                for existing in extra.get(target.judgement_no, ())
            )
            if not declared and not already:
                extra.setdefault(target.judgement_no, []).append(
                    JudgementLink(
                        kind=JudgementRelation.CONCURRENT_WITH,
                        target_no=item.judgement_no,
                        context_no=None,
                    )
                )
    if not extra:
        return tuple(judgements)
    return tuple(
        item
        if item.judgement_no not in extra
        else replace(item, relations=(*item.relations, *extra[item.judgement_no]))
        for item in judgements
    )


# --- 基础守卫 -------------------------------------------------------------------------


def _mapping(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise BehaviorFusionError(f"{label} must be an object with string keys")
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise BehaviorFusionError(f"{label} contains unsupported keys: {sorted(unknown)}")
    if missing:
        raise BehaviorFusionError(f"{label} is missing keys: {sorted(missing)}")
    return value


def _array(value: object, label: str, *, max_items: int) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BehaviorFusionError(f"{label} must be an array")
    if len(value) > max_items:
        raise BehaviorFusionLimitError(f"{label} exceeds its configured item limit")
    return value


def _text(value: object, label: str, *, max_chars: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BehaviorFusionError(f"{label} must be non-empty text")
    normalized = " ".join(value.split())
    if len(normalized) > max_chars:
        raise BehaviorFusionLimitError(f"{label} exceeds its configured character limit")
    return normalized


def _optional_text(value: object, label: str, *, max_chars: int) -> str | None:
    if value is None:
        return None
    return _text(value, label, max_chars=max_chars)


def _text_tuple(value: object, label: str, *, max_chars: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BehaviorFusionError(f"{label} must be an array")
    return tuple(_text(item, f"{label} item", max_chars=max_chars) for item in value)


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BehaviorFusionError(f"{label} must be a positive integer")
    return value


def _enum_member(value: object, enum: Any, label: str, *, allow_none: bool) -> Any:
    if value is None:
        if not allow_none:
            raise BehaviorFusionError(f"{label} must not be null")
        return None
    text = _text(value, label, max_chars=64)
    try:
        return enum(text)
    except ValueError as exc:
        allowed = sorted(member.value for member in enum)
        raise BehaviorFusionError(f"{label} must be one of {allowed}") from exc


__all__ = ["assemble_judgement_batch"]
