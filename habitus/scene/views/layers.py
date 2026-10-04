"""四种分层（情境 / 剂量 / 共享证据 / 干预）：同一个套路——按谓词切账 → ``layer_reading`` → 比区间。

**稳定性按情境分层是三态的**（2026-09-30 裁定八）：每个情境只收**判过它**的承诺（承诺从命中记录抄来 ``situations_checked``），
分"在场 / 判过且不在场"两层；两层任一不够样本 → **待定**；都够且区间分开、差达分界 → **影响大**（被它调节）；都够但差不到 → **影响小**。
没判过的承诺不进任何一层——"不知道"不是"不在场"（评审 B-11 ②）。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import partial
from types import MappingProxyType

from habitus.scene.concepts.model import ConceptSet
from habitus.scene.hypotheses.model import Direction, Hypothesis
from habitus.scene.ledger.model import Claim
from habitus.scene.ledger.store import LedgerStore
from habitus.scene.views.accounts import Account
from habitus.scene.views.config import ViewsConfig
from habitus.scene.views.fulfilment import LayerReading, fulfilment_of
from habitus.scene.views.stats import is_monotonic
from habitus.scene.views.strength import strength_of


def layer_reading(
    account: Account,
    hypothesis: Hypothesis,
    config: ViewsConfig,
    *,
    gate: tuple[int, int] | None = None,
    seed: str | None = None,
    now: datetime | None = None,
) -> LayerReading:
    """分层读数：节律型给强度（p1−p0），无节律型给兑现率。两者都有 ``.interval`` 与 ``.accumulation``，
    所以"两层比区间"那段判据一个字不用改（P-1：用户 09-27 定"四处"一起按型分派）。"""

    if hypothesis.is_open_ended:
        return fulfilment_of(account, hypothesis, config, gate=gate, now=now)
    return strength_of(account, hypothesis, config, gate=gate, seed=seed, now=now)


class Influence(str, Enum):
    """一个情境对这条关系的影响，三态（2026-09-30 裁定八 ③④）。"""

    #: 两层里有一层没攒够：不下结论——**不是"影响小"**（裁定八 ④）。
    PENDING = "pending"
    #: 两层都够、区间分开、点估计差过分界：被它调节，按它拆开、以后预测先判它。
    MAJOR = "major"
    #: 两层都够、分不开或差不到分界：不拆，只记。
    MINOR = "minor"


@dataclass(frozen=True)
class Moderation:
    situation: str
    with_situation: LayerReading
    without_situation: LayerReading

    @property
    def gap(self) -> float | None:
        """两层点估计的差（在场 − 不在场）；任一层没区间就 None。"""

        a, b = self.with_situation.interval, self.without_situation.interval
        if a is None or b is None:
            return None
        return a.point - b.point


@dataclass(frozen=True)
class StabilityReport:
    """稳定性扫的结果：每个判过的情境三态里的哪一态；``moderations`` 是其中影响大的、``tested`` 是两层都读得出的、
    ``skipped`` 是待定的。对照没按情境分（刀 5 前的诚实标注）。"""

    moderations: tuple[Moderation, ...]
    tested: tuple[str, ...]
    skipped: tuple[str, ...]
    control_split_by_situation: bool = False
    #: 每个情境的三态（给人读的概念名 → 态）。
    influence: Mapping[str, Influence] = MappingProxyType({})
    #: 每个**测过**的情境的两层（不管区间有没有分开）。份额要它：七f-3 定的"调节 {A,B} vs {A,¬B}"就是
    #: 这两层的差（"已经晚睡了，出差还额外压多少"），而 ``moderations`` 只留区间分开的那些，
    #: 所以差算得出但印不出来。两层各自有区间，差真不真由"区间不重叠"判。
    layers: tuple[Moderation, ...] = ()


def stability_scan(account: Account, hypothesis: Hypothesis, concepts: ConceptSet, config: ViewsConfig, now: datetime | None = None) -> StabilityReport:
    """对每个承诺上**判过**的情境概念分账：在场一层、判过且不在场一层（没判过的承诺不进任何一层——裁定八 ②：
    一个情境从它进来那天起量，旧承诺不重算也不算"不在场"）。两层任一没攒够 → 待定；都够、区间分开且差过分界 → 影响大；
    否则影响小（裁定八 ③④）。"""

    situations = sorted({
        concepts[name].identity
        for claim in account.claims
        for name in (*claim.situation_snapshot, *claim.situations_checked)
        if name in concepts and concepts[name].role.is_situation
    })
    moderations: list[Moderation] = []
    layers: list[Moderation] = []
    tested: list[str] = []
    skipped: list[str] = []
    influence: dict[str, Influence] = {}
    for situation in situations:
        label = concepts[situation].name  # 给人读的是概念名，不是规范身份
        with_it = account.where(partial(_has_situation, situation=situation, concepts=concepts))
        without = account.where(partial(_checked_and_lacks_situation, situation=situation, concepts=concepts))
        a = layer_reading(with_it, hypothesis, config, gate=config.split_gate, seed=f"{account.hypothesis_identity}|{situation}|with", now=now)
        b = layer_reading(without, hypothesis, config, gate=config.split_gate, seed=f"{account.hypothesis_identity}|{situation}|without", now=now)
        if a.interval is None or b.interval is None:
            skipped.append(label)
            influence[label] = Influence.PENDING
            continue
        tested.append(label)
        moderation = Moderation(label, a, b)
        layers.append(moderation)
        gap = moderation.gap
        if a.interval.disjoint_from(b.interval) and gap is not None and abs(gap) >= config.influence_gap(a.unit):
            moderations.append(moderation)
            influence[label] = Influence.MAJOR
        else:
            influence[label] = Influence.MINOR
    return StabilityReport(tuple(moderations), tuple(tested), tuple(skipped), layers=tuple(layers), influence=MappingProxyType(influence))


@dataclass(frozen=True)
class DoseReading:
    concept: str
    by_grade: tuple[tuple[str, LayerReading], ...]
    monotonic: bool


def dose_readings(account: Account, hypothesis: Hypothesis, concepts: ConceptSet, config: ViewsConfig, now: datetime | None = None) -> tuple[DoseReading, ...]:
    """前件（假设里没写死档的那些）按承诺记下的档分账，看强度是否随档单调。档按**数值下界**排，不按声明顺序。"""

    found: list[DoseReading] = []
    for item in hypothesis.antecedents:
        if item.grade is not None or item.identity not in concepts or not concepts[item.identity].grades:
            continue
        if any(grade.wraps_midnight for grade in concepts[item.identity].grades):
            continue  # 跨午夜的档在数轴上排不出顺序，见 unordered_dose_concepts
        order = [grade.name for grade in sorted(concepts[item.identity].grades, key=lambda grade: grade.lower)]
        by_grade: list[tuple[str, LayerReading]] = []
        for grade in order:
            stratum = account.where(partial(_has_grade, identity=item.identity, grade=grade))
            strength = layer_reading(stratum, hypothesis, config, gate=config.split_gate, seed=f"{account.hypothesis_identity}|{item.identity}|{grade}", now=now)
            if strength.interval is not None:
                by_grade.append((grade, strength))
        if len(by_grade) >= 2:
            points = [s.interval.point for _g, s in by_grade if s.interval is not None]
            # 单调**且与基准猜的方向同号**才算趋势：一条从 −0.68 走到 −0.08 的曲线在数值上单调，但它说的是
            # "档越重影响越小"，与"熬得越晚越不吃"相反，不该升可信度。
            trend = is_monotonic(points) and _agrees_direction(points, hypothesis.direction)
            found.append(DoseReading(item.concept, tuple(by_grade), trend))
    return tuple(found)


def unordered_dose_concepts(hypothesis: Hypothesis, concepts: ConceptSet) -> tuple[str, ...]:
    """前件里**档跨午夜**的那些概念：它们不出剂量读数，要报出来。

    剂量读的是"档越重、效应越往一边走"，所以档必须能排成一条阶梯——而阶梯是按 ``grade.lower``
    在数轴上排的。一个 22:00–02:00 的档 ``lower`` 是 1320、``upper`` 是 120，与一个 02:00–04:00 的档
    （``lower`` 120）比大小，排出来是"凌晨在深夜之前"，那条"趋势"就是假的。
    所以这一项不出数；要出，得先给绝对时刻的档定一个参照点（相对常态的档不受影响，它们不跨午夜）。
    """

    return tuple(
        item.concept
        for item in hypothesis.antecedents
        if item.grade is None and item.identity in concepts and any(grade.wraps_midnight for grade in concepts[item.identity].grades)
    )


def _agrees_direction(points: Sequence[float], direction: Direction) -> bool:
    """档越重、效应越往基准说的那一边走。"""

    if len(points) < 2:
        return False
    delta = points[-1] - points[0]
    return delta > 0 if direction is Direction.UP else delta < 0


def _has_situation(claim: Claim, situation: str, concepts: ConceptSet) -> bool:
    return any(concepts[name].identity == situation for name in claim.situation_snapshot if name in concepts)


def _checked_and_lacks_situation(claim: Claim, situation: str, concepts: ConceptSet) -> bool:
    """判过这个情境、且不在场。旧记录没有 ``situations_checked`` 这一栏的承诺不算"不在场"。"""

    if _has_situation(claim, situation, concepts):
        return False
    return any(concepts[name].identity == situation for name in claim.situations_checked if name in concepts)


def _has_grade(claim: Claim, identity: str, grade: str) -> bool:
    return any(hit.identity == identity and hit.grade == grade for hit in claim.antecedent_hits)


# ── 归因：共享证据的调节对照 ───────────────────────────────────────────────────


class EvidenceIndex:
    """出力 occurrence → 哪些假设的承诺用过它。一次扫盘建好、全部假设共用（不然每条都扫一遍整本账）。"""

    def __init__(self, ledger: LedgerStore) -> None:
        self._by_uri: dict[str, set[str]] = {}
        self._uris_by_hypothesis: dict[str, set[str]] = {}
        for identity in ledger.hypotheses():
            uris = {uri for claim in ledger.claims_for(identity) for uri in claim.antecedent_uris}
            self._uris_by_hypothesis[identity] = uris
            for uri in uris:
                self._by_uri.setdefault(uri, set()).add(identity)

    def sharing(self, hypothesis_identity: str, uris: Iterable[str]) -> tuple[str, ...]:
        others = {identity for uri in uris for identity in self._by_uri.get(uri, ()) if identity != hypothesis_identity}
        return tuple(sorted(others))

    def uris_of(self, hypothesis_identity: str) -> frozenset[str]:
        return frozenset(self._uris_by_hypothesis.get(hypothesis_identity, ()))


@dataclass(frozen=True)
class SharedEvidence:
    """与本条共用同一次证据的另一条假设 C，以及 {A,C} vs {A,¬C} 那条调节对照（七i 四）。"""

    other: str
    shared_claims: int
    with_other: LayerReading
    without_other: LayerReading

    @property
    def separable(self) -> bool | None:
        """能不能分出谁在起作用：两层都过门槛才有答案；区间不重叠 → C 在场时确实不一样。"""

        if self.with_other.interval is None or self.without_other.interval is None:
            return None
        return self.with_other.interval.disjoint_from(self.without_other.interval)


def shared_evidence(
    account: Account,
    hypothesis: Hypothesis,
    index: EvidenceIndex,
    config: ViewsConfig,
    hypotheses: Mapping[str, Hypothesis],
    now: datetime | None = None,
) -> tuple[SharedEvidence, ...]:
    """七i 四要的那条对照：**另一个原因概念**（熬夜工作）命中了同一次 occurrence，按"那次证据是否也开了它的承诺"
    分两层 = {A,C} vs {A,¬C}，看得出谁在起作用。

    排除三类不是"另一个原因"的：
    - **同一个前件集合**的兄弟假设（晚睡→起床 / 咖啡 / 就寝）——那是七i 一的多影响，它们的证据 100% 重合，
      "不在场"那层恒空，印出来只有"分不开"三个字的噪音（每条关系多印 3 行，评审 A-6/C-5）；
    - 前件集合是本条**子集或超集**的（{晚睡,出差}→早餐）——那是七i 二的多体账，与 ``stability_scan`` 按「出差中」分账重复；
    - **后件不同**的——"是睡得晚还是工作到晚在起作用"是对同一顿早饭说的。
    """

    mine = frozenset(item.identity for item in hypothesis.antecedents)
    found: list[SharedEvidence] = []
    listed: set[frozenset[str]] = set()
    for other in index.sharing(account.hypothesis_identity, account.antecedent_uris):
        peer = hypotheses.get(other)
        if peer is None or peer.consequent_identity != hypothesis.consequent_identity:
            continue
        if peer.source.origin.is_placebo:
            continue  # 安慰剂是尺子不是原因：印进真假设的详情页会被当成真关系读（评审 A-10 / B-6 / C-9）
        theirs_antecedents = frozenset(item.identity for item in peer.antecedents)
        if theirs_antecedents <= mine or mine <= theirs_antecedents:
            continue
        if theirs_antecedents in listed:
            continue  # 同一个同伴的 --1/--2/--3/--4 共用同一批证据，印一次就够（评审 C-14 ③）
        listed.add(theirs_antecedents)
        theirs = index.uris_of(other)
        with_other = account.where(partial(touches, uris=theirs))
        without_other = account.where(partial(avoids, uris=theirs))
        found.append(
            SharedEvidence(
                other=other,
                shared_claims=len(with_other.claims),
                with_other=layer_reading(with_other, hypothesis, config, gate=config.split_gate, seed=f"{account.hypothesis_identity}|{other}|with", now=now),
                without_other=layer_reading(without_other, hypothesis, config, gate=config.split_gate, seed=f"{account.hypothesis_identity}|{other}|without", now=now),
            )
        )
    return tuple(found)


def touches(claim: Claim, uris: frozenset[str]) -> bool:
    return bool(set(claim.antecedent_uris) & uris)


def avoids(claim: Claim, uris: frozenset[str]) -> bool:
    return not touches(claim, uris)


def reminded_split(account: Account, hypothesis: Hypothesis, config: ViewsConfig, now: datetime | None = None) -> tuple[LayerReading, LayerReading]:
    """干预分层：提醒过 / 没提醒各一份（各自过分层门槛，不够就是区间 None）。"""

    reminded = account.where(lambda claim: claim.ref in account.interventions)
    quiet = account.where(lambda claim: claim.ref not in account.interventions)
    return (
        layer_reading(reminded, hypothesis, config, gate=config.split_gate, seed=f"{account.hypothesis_identity}|reminded", now=now),
        layer_reading(quiet, hypothesis, config, gate=config.split_gate, seed=f"{account.hypothesis_identity}|quiet", now=now),
    )



__all__ = [
    "DoseReading",
    "EvidenceIndex",
    "Influence",
    "Moderation",
    "SharedEvidence",
    "StabilityReport",
    "avoids",
    "dose_readings",
    "layer_reading",
    "reminded_split",
    "shared_evidence",
    "stability_scan",
    "touches",
    "unordered_dose_concepts",
]
