"""behaviours/：一个行为的**两面**——什么导致它（前因）、它导致了什么（后果）。

用户 2026-09-27 定的形状：「一个行为可能是别人的前因，也可能别人的后果，而且对一个行为很多时候是有几个行为
一起决定的，而不是单一决定的。」

- **前因** = 后件是它的那些假设。``hypotheses/`` 本来就按后件分目录（"此刻要判 B，有哪些假设指向 B"），
  所以这一面的数据早就在，缺的只是一份把它和读数摆在一起的文件。
- **后果** = 前件集合里含它的那些假设。这一面要扫遍所有后件目录才拼得出来，所以非它不可。
- 前因**分组**：``单个原因`` 与 ``几个一起（多体）``，子集摆在超集上面——主因看"单个原因"那几行的大小。
- **份额**（七e：「份额不直接估，从 {A} 与 {A,B} 两条账一比得出」）不是新机制：七f-3 定的正确对照是
  **调节 {A,B} vs {A,¬B}**（同一本账切成两半，两半不重叠），而 ``stability_scan`` 已经在算这两层。
  这里只是把两层的差摆出来，真不真由"两层区间不重叠"判（就是 ``moderations``）。
  ⚠ 不要拿 ``{A,B}`` 减 ``{A}`` **全体**——那两条账样本嵌套，实测"B 其实毫无额外影响"时仍有 12% 的情况
  印出 |差| ≥ 25pp。

纯投影：一个数都不新算，全部来自 ``read_relations`` 已经算好的读数。没有写入者，整个删掉可重建。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from habitus.scene.hypotheses.model import Hypothesis
from habitus.scene.views.relations import Moderation, RelationReading


@dataclass(frozen=True)
class BehaviourSide:
    """一面里的一条：那条关系的读数 + 它在这一面里怎么读（对前因面是"谁导致的"，对后果面是"导致了谁"）。"""

    hypothesis: Hypothesis
    reading: RelationReading

    @property
    def is_combination(self) -> bool:
        """前件是一组（多体）——它要的对照是集合效应，与单前件那几行不是一个量。"""

        return len(self.hypothesis.antecedents) > 1

    @property
    def shares(self) -> tuple[Moderation, ...]:
        """这条关系按情境切开的两层：``{A,B}`` vs ``{A,¬B}``，两层的差就是那个情境的额外贡献（份额）。"""

        return self.reading.stability.layers


@dataclass(frozen=True)
class BehaviourView:
    """一个行为概念的两面。``causes`` 已按"单个原因 / 几个一起"排好（单前件在前）。"""

    concept: str
    causes: tuple[BehaviourSide, ...]
    effects: tuple[BehaviourSide, ...]

    @property
    def single_causes(self) -> tuple[BehaviourSide, ...]:
        return tuple(item for item in self.causes if not item.is_combination)

    @property
    def combined_causes(self) -> tuple[BehaviourSide, ...]:
        return tuple(item for item in self.causes if item.is_combination)


def behaviour_views(readings: Iterable[RelationReading], hypotheses: Mapping[str, Hypothesis]) -> tuple[BehaviourView, ...]:
    """把一轮读数按行为概念翻成两面。读数里没有对应假设的跳过（账在、假设删了那种）。

    一条关系在**两个**行为的文件里各出现一次：对它的后件是"前因"，对它的每个前件行为是"后果"。情境概念不给
    自己的文件——它当不了后件，"它导致了什么"这一面对它没有意义，而"什么导致它"更不该由这一层回答。
    """

    causes: dict[str, list[BehaviourSide]] = {}
    effects: dict[str, list[BehaviourSide]] = {}
    for reading in readings:
        hypothesis = hypotheses.get(reading.hypothesis_identity)
        if hypothesis is None:
            continue
        side = BehaviourSide(hypothesis, reading)
        causes.setdefault(hypothesis.consequent_identity, []).append(side)
        for item in hypothesis.antecedents:
            effects.setdefault(item.identity, []).append(side)
    return tuple(
        BehaviourView(
            concept=concept,
            causes=tuple(sorted(causes.get(concept, ()), key=_order)),
            effects=tuple(sorted(effects.get(concept, ()), key=_order)),
        )
        for concept in sorted(set(causes) | set(effects))
    )


def _order(side: BehaviourSide) -> tuple[int, str]:
    """单前件在前（主因先看），然后按身份排——顺序与模型无关、可重放。"""

    return (len(side.hypothesis.antecedents), side.hypothesis.identity)


__all__ = ["BehaviourSide", "BehaviourView", "behaviour_views"]
