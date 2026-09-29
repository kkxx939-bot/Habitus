"""一个概念命中过哪些 kind——机会口那座桥的 scene 那一半。

**为什么需要这座桥**：预测树的曲线按 `kind_token` 存（`prediction/source.py` 用的就是这个字段），也就是
上游去重之后的名字；而语义树用的是概念。要问树"这个**概念**在这个周几的哪几个时段会发生"，就得先知道
它对应哪些 kind。

**叶子概念多半一对一，聚合概念才要相加**（2026-09-28 裁定）：
- 叶子概念（45 天里出现 ≥10 次的那些 kind 各给一个）——一个概念一个 kind，"相加"退化成"就用那条曲线"；
- 聚合/上级概念（「排查问题」= 排查 CI 失败 + 排查性能 + 排查设备 + …，凑到 ≥10 次）——跨几个 kind，
  这时候才真的要把几条曲线相加。相加的依据是 B13「行为事件有先后顺序，人不能同时做两件事」：
  同一个槽上这几个 kind 近似互斥，互斥事件"至少一个发生"就是相加。

**语义不会因此丢**：叶子各自一个概念、各自一本账（因果关系分开记），粗的那一层靠 `parent` 聚合
（十 ③：祖先概念不参与命中，读时沿 parent 链聚合）。所以细层保语义、粗层保样本，读侧按七i 三
"账薄就退回上一级"选用。

**一个上限**：概念可以比 kind 粗，**不能比 kind 细**。上游把很多不同的事折进一个 kind 时，语义分辨率
就钉在那儿了，概念层级救不回来——那要改上游的折叠（属事件融合那条线）。所以这里把"跨了几个 kind、
各多少次"报出来，让人看得见分辨率够不够。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from habitus.scene.concepts.model import ConceptSet
from habitus.scene.occurrences.model import ConceptHits


@dataclass(frozen=True)
class KindSpread:
    """一个概念命中过的 kind 与各自的次数（次数多的在前，同次数按名字）。

    ``kinds`` 就是机会口要相加的那几条曲线的键。``spread`` 大于 1 时值得看一眼：可能是这个概念定宽了
    （该拆），也可能它本来就是聚合概念（排查问题 = 五种排查）。
    """

    concept: str
    counts: Mapping[str, int]

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self.counts, key=lambda kind: (-self.counts[kind], kind)))

    @property
    def occurrences(self) -> int:
        return sum(self.counts.values())

    @property
    def spread(self) -> int:
        return len(self.counts)


def concept_kinds(records: Iterable[ConceptHits], concepts: ConceptSet) -> tuple[KindSpread, ...]:
    """扫一批命中记录，数出每个概念命中过哪些 kind。

    **算的是命中本身，不沿 parent 链聚合**：聚合是读侧的事，而这里的产物要喂给机会口，机会口对
    "叶子概念"和"聚合概念"用同一套做法（各自把自己命中过的 kind 相加），所以这里只管记录上写着的命中。
    概念集里已经不存在的名字跳过——概念删了，它的历史命中不该复活。
    """

    if not isinstance(concepts, ConceptSet):
        raise TypeError("concepts must be a ConceptSet")
    tally: dict[str, dict[str, int]] = {}
    for record in records:
        for hit in record.hits:
            if hit.identity not in concepts:
                continue
            tally.setdefault(hit.identity, {})
            tally[hit.identity][record.kind_token] = tally[hit.identity].get(record.kind_token, 0) + 1
    return tuple(
        KindSpread(concept=identity, counts=MappingProxyType(dict(sorted(counts.items()))))
        for identity, counts in sorted(tally.items())
    )


def kinds_by_concept(spreads: Sequence[KindSpread]) -> Mapping[str, tuple[str, ...]]:
    """机会口要的形状：概念身份 → 要相加的那几条曲线的键。"""

    return MappingProxyType({item.concept: item.kinds for item in spreads})


__all__ = ["KindSpread", "concept_kinds", "kinds_by_concept"]
