"""安慰剂前件：把前件换成一条与后件**无关**的行为，看效应还在不在。

七c ⑭ 定的那把尺子。它回答的不是"这条关系对不对"，而是**"我们这套读法的误报率有多高"**——
安慰剂本该一条都不显著；显著的那部分就是误报率的直接测量，也是"门槛能不能定 3"的判据
（基准凭常识筛过的假设里假的占几成，取决于筛子准不准；安慰剂就是筛子完全没起作用时的对照）。

**安慰剂是真的假设**：写进 ``hypotheses/``、一样开账、一样结算、一样出读数。不这么做就量不到东西——
误报率是"同一套账本口径在无关前件上会说多少次有效应"，只有让它真的走一遍才算得出。
因此它也**从写入日起攒账**，和触点③ 提的结构假设同一条纪律。

**"无关"是算法能判的三条**（不问模型：模型判"相关不相关"正是我们要检验的那把筛子，让它来选就循环了）：

1. 不是这个后件任何一条真假设的前件——那些是常识说"有关"的，选了就不是安慰剂；
2. 与后件不在同一条祖先链上——「打球 → 运动」那种重言本来就被假设层拒；
3. 不是后件自己。

**选得可重放**：按 (后件, 前件) 的稳定散列排序挑，不用随机数，而且只看本后件（不受这一批里还有哪些后件影响）。
安慰剂一旦写盘就开始攒账，每晚换一批等于每晚重置样本，那把尺子就永远量不出东西。

**身份带前缀** ``PLACEBO_MARK``（见 ``hypotheses.model``）：不与真假设争名字。

**无节律型的形状先不配**（2026-09-30 二-2）：裁定五选的 (c)（收口规则 + 双向验证）属生命周期那一份，还没建；
建好之前配出来的无节律型安慰剂读不了（只会标"未校准"）却一直堆账（探针四夜 93 条），所以先停配，生命周期做完再打开。
真假设照常开账、照样标"未校准"。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from habitus.foundation.integrity import canonical_digest
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.hypotheses.model import (
    Antecedent,
    Hypothesis,
    HypothesisError,
    HypothesisOrigin,
    HypothesisSource,
)

#: 每个后件配几条安慰剂（裁定里的"安慰剂 M"，**未定**）。2 条起步：太少量不出误报率，太多把账本撑大
#: （安慰剂和真假设一样开账）。重放时按"真假设条数 × M 的总账本体量"复核。
PLACEBO_PER_CONSEQUENT = 2
PLACEBO_NOTE = "安慰剂：前件换成与后件无关的行为，用来量这套读法的误报率（七c ⑭）"


def placebo_hypotheses(
    concepts: ConceptSet,
    real: Sequence[Hypothesis],
    *,
    now: datetime,
    per_consequent: int = PLACEBO_PER_CONSEQUENT,
) -> tuple[Hypothesis, ...]:
    """给每个已经有真假设的后件配 ``per_consequent`` 条无关前件，**每条真假设的形状各配一份**。

    尺子要和被量的东西同一把：真假设有第 1–4 次机会各一本账，安慰剂只有第 1 次的话，
    量到的只是第 1 本账的误报率，其余三本账没有对照（评审 C-2）。所以按 (方面 · 方向 · 第几次机会 · 地平线 ·
    先验类型) 分形状，同一个后件的几个形状共用同一批无关前件——这样"同一次机会"那一档里真假设与安慰剂可比。

    "已经用过"只看**本后件**：真假设的前件与本后件已配的安慰剂。跨后件累计（旧写法把整批 ``found`` 传进去）
    会让排在后面的后件把候选池挑空——探针 20 个后件应配 40 条只配出 21 条、8 个后件一条没有（评审 B-16）。
    已经存在的安慰剂由调用方过滤（传进来的 ``real`` 里带上它们即可，函数只补齐缺的那些）。
    """

    if not isinstance(concepts, ConceptSet):
        raise TypeError("concepts must be a ConceptSet")
    if isinstance(per_consequent, bool) or not isinstance(per_consequent, int) or per_consequent < 0:
        raise ValueError("per_consequent must be a non-negative integer")
    by_consequent: dict[str, list[Hypothesis]] = {}
    for item in real:
        by_consequent.setdefault(item.consequent_identity, []).append(item)
    found: list[Hypothesis] = []
    for identity, items in sorted(by_consequent.items()):
        templates = [item for item in items if not item.source.origin.is_placebo]
        if not templates:
            continue  # 这个后件只有安慰剂，没有真假设可量
        existing = [item for item in items if item.source.origin.is_placebo]
        antecedents = _antecedents_for(concepts, identity, templates, existing, per_consequent)
        known = {item.identity for item in existing}
        for template in _shapes(templates):
            if template.is_open_ended:
                continue  # 二-2（2026-09-30）：无节律型先停配——没有平时概率可比、读数只会标"未校准"，却一直堆账；生命周期做完再打开
            for antecedent in antecedents:
                built = _placebo(template, antecedent, now=now)
                if built is not None and built.identity not in known:
                    known.add(built.identity)
                    found.append(built)
    return tuple(found)


def _shapes(templates: Sequence[Hypothesis]) -> tuple[Hypothesis, ...]:
    """同一个后件的真假设按形状去重（方面 · 方向 · 第几次机会 · 地平线 · 先验类型），每种形状留一条当模板。"""

    seen: dict[tuple[object, ...], Hypothesis] = {}
    for item in sorted(templates, key=lambda h: h.identity):
        key = (item.aspect, item.direction, item.consequent_peak, item.horizon, item.type_prior)
        seen.setdefault(key, item)
    return tuple(seen.values())


def _antecedents_for(
    concepts: ConceptSet, consequent: str, templates: Sequence[Hypothesis], existing: Sequence[Hypothesis], count: int
) -> tuple[str, ...]:
    """本后件的那批无关前件：先沿用盘上已有安慰剂的前件（它们一旦写盘就在攒账），不够再按稳定散列补。"""

    chosen: list[str] = []
    for item in existing:
        for antecedent in item.antecedents:
            if antecedent.identity not in chosen:
                chosen.append(antecedent.identity)
    used = {item.identity for hypothesis in templates for item in hypothesis.antecedents} | set(chosen)
    pool = [
        identity
        for identity in concepts.behaviors()
        if identity != consequent
        and identity not in used
        and not concepts.is_ancestor(identity, consequent)
        and not concepts.is_ancestor(consequent, identity)
    ]
    pool.sort(key=lambda identity: canonical_digest({"consequent": consequent, "antecedent": identity}))
    for identity in pool:
        if len(chosen) >= count:
            break
        chosen.append(identity)
    return tuple(chosen[:count])


def _placebo(template: Hypothesis, antecedent: str, *, now: datetime) -> Hypothesis | None:
    """照着真假设的形状造一条，只换前件与来源。造不出来（档、重言之类）就跳过，不硬凑。"""

    try:
        # 安慰剂的前因不分峰（它只是随手配的参照，分峰只会把本就少的样本切碎）；后果峰与峰表抄模板的。
        return Hypothesis(
            antecedents=(Antecedent(antecedent),),
            consequent=template.consequent,
            aspect=template.aspect,
            direction=template.direction,
            note=PLACEBO_NOTE,
            source=HypothesisSource(origin=HypothesisOrigin.PLACEBO, note=f"照 {template.identity} 的形状"),
            created_at=now,
            type_prior=template.type_prior,
            consequent_peak=template.consequent_peak,
            windows={template.consequent_identity: template.windows[template.consequent_identity]} if template.consequent_peak is not None else {},
            horizon=template.horizon,
        )
    except HypothesisError:
        return None


def split_by_origin(hypotheses: Mapping[str, Hypothesis]) -> tuple[dict[str, Hypothesis], dict[str, Hypothesis]]:
    """(给人看的, 安慰剂)。投影只拿前者——安慰剂不是读数，是尺子，印进 behaviours/ 会当成真因果读。"""

    real = {identity: item for identity, item in hypotheses.items() if not item.source.origin.is_placebo}
    placebo = {identity: item for identity, item in hypotheses.items() if item.source.origin.is_placebo}
    return real, placebo


__all__ = ["PLACEBO_NOTE", "PLACEBO_PER_CONSEQUENT", "placebo_hypotheses", "split_by_origin"]
