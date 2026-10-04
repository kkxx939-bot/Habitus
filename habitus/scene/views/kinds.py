"""一个概念命中过哪些 kind——机会口那座桥的 scene 那一半。

**为什么需要这座桥**：预测树的曲线按 `kind_token` 存（`prediction/source.py` 用的就是这个字段），也就是
上游去重之后的名字；而语义树用的是概念。要问树"这个**概念**在这个周几的哪几个时段会发生"，就得先知道
它对应哪些 kind。

**叶子概念多半一对一，聚合概念才要相加**（2026-09-28 裁定）：
- 叶子概念（45 天里出现 ≥10 次的那些 kind 各给一个）——一个概念一个 kind，"相加"退化成"就用那条曲线"；
- 聚合概念（「排查问题」= 排查 CI 失败 + 排查性能 + 排查设备 + …，凑到 ≥10 次）——跨几个 kind，
  这时候才真的要把几条曲线相加。相加的依据是 B13「行为事件有先后顺序，人不能同时做两件事」：
  同一个槽上这几个 kind 近似互斥，互斥事件"至少一个发生"就是相加。

⚠ **上级概念（自己不认领 kind、只有子概念）现在拿不到曲线**：这里只数记录上写着的命中，映射只判叶子，上级概念
永远不在桥里（评审 A-2 / R3-04a：探针 12 条以上级为后件的假设、204 条承诺全无对照，还被错分成无节律型）。
"上级的 kind = 后代叶子的 kind 的并集"这一步 2026-09-30 裁定**并入词表与事件融合改造**，这里先不做。
真实数据上叶子概念也常沾多个 kind（一个概念 5–13 个），本来率因此偏高 1.5–4 倍（R3-04b，同样并入词表改造）。

**语义不会因此丢**：叶子各自一个概念、各自一本账（因果关系分开记），粗的那一层靠 `parent` 聚合
（十 ③：祖先概念不参与命中，读时沿 parent 链聚合）。所以细层保语义、粗层保样本，读侧按七i 三
"账薄就退回上一级"选用。

**对称的另一半是"一条 occurrence 命中几个概念"**（``concept_overlap``）：这一头说的是概念定宽了没有，
那一头说的是概念之间重叠了没有。2026-09-29 探针实测：16 个平级近义概念（0 个 parent）让一条 occurrence
平均命中 **4 个**，于是同一次前件命中开出 4 倍的承诺，本来清楚的因果被切成 4 条各自更薄的账。
所以这个数要报出来。

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


#: 一条 occurrence 平均命中几个叶子概念，超过这个数就报"概念集在互相稀释"。
#: 2 = "一件事偶尔同时算两个概念"还说得过去，长期高于 2 就说明该挂层级或该合并。
#: **待定值**（2026-09-29 探针上实测到 4.0），真实数据上按"命中数分布"复核。
MAX_MEAN_CONCEPT_HITS = 2.0


@dataclass(frozen=True)
class ConceptOverlap:
    """概念之间重叠了多少：一条 occurrence 平均命中几个叶子概念，以及最常一起命中的那几对。

    **只看叶子**：祖先概念本来就该跟着子概念一起成立（读侧沿 parent 链聚合），那是设计不是重叠。
    共现最高的那几对是可直接动手的线索——它们要么该并成一个概念，要么该挂到同一个 ``parent`` 下。
    """

    records: int
    hits: int
    pairs: Mapping[tuple[str, str], int]

    @property
    def mean_hits(self) -> float:
        return self.hits / self.records if self.records else 0.0

    @property
    def diluted(self) -> bool:
        """概念集在互相稀释：平均命中数偏高。"""

        return self.mean_hits > MAX_MEAN_CONCEPT_HITS

    def crowded(self, top: int = 5) -> tuple[tuple[tuple[str, str], int], ...]:
        return tuple(sorted(self.pairs.items(), key=lambda item: (-item[1], item[0]))[:top])

    def render(self) -> str:
        pairs = "；".join(f"{left} + {right} 共 {count} 次" for (left, right), count in self.crowded(3))
        return f"一条 occurrence 平均命中 {self.mean_hits:.1f} 个概念（{self.records} 条）" + (f"；最常同时命中：{pairs}" if pairs else "")


def concept_overlap(records: Iterable[ConceptHits], concepts: ConceptSet) -> ConceptOverlap:
    """数出概念之间的重叠。概念集里已经不存在的名字跳过（与 ``concept_kinds`` 同口径）。"""

    if not isinstance(concepts, ConceptSet):
        raise TypeError("concepts must be a ConceptSet")
    total = seen = 0
    pairs: dict[tuple[str, str], int] = {}
    for record in records:
        names = sorted({hit.identity for hit in record.hits if hit.identity in concepts})
        total += 1
        seen += len(names)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                pairs[(left, right)] = pairs.get((left, right), 0) + 1
    return ConceptOverlap(records=total, hits=seen, pairs=MappingProxyType(pairs))


__all__ = ["MAX_MEAN_CONCEPT_HITS", "ConceptOverlap", "KindSpread", "concept_kinds", "concept_overlap", "kinds_by_concept"]
