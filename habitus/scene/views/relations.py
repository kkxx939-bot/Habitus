"""relations/：每条假设的当前读数——把账读成强度、类型、稳定性、剂量、归因，拼成一份 ``RelationReading``。

全部读时算、可重算；一个数都不落回账本。拆成几个模块各管一件事（2026-09-30 用户要求，R3-27）：
``accounts``（账与块）· ``strength``（三套强度 + 类型）· ``fulfilment``（无节律型）· ``layers``（四种分层）· 本模块（上一级指针 + 组装）。

**先过累积度门槛，再谈数**（用户 09-26 裁定：少样本不出数）。门槛双维度：进账的机会至少 ``min_count`` 次、
且落在至少 ``min_blocks`` 个不同的块里；块 = 同一个后果窗口（二-6）。门槛之下不出强度，只报攒了多少、还差多少，
并给上一级（父概念 / 少一个元素的子集）的读数当指针——预测层按七i 三退回上一级用。

**假设分两型**：``consequent_peak`` 有数的是**节律型**（窗口账）；``None`` 的是**无节律型**，只读 ``FulfilmentReading``（未校准，裁定五）。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import partial

from habitus.scene.concepts.model import ConceptSet
from habitus.scene.hypotheses.model import (
    ASPECT_SEPARATOR,
    ELEMENT_SEPARATOR,
    Antecedent,
    Aspect,
    Hypothesis,
)
from habitus.scene.ledger.store import LedgerStore
from habitus.scene.views.accounts import (
    Account,
    Accumulation,
    Pair,
    accumulation_of,
    block_hours_of,
    informative_pairs,
    load_account,
    settled_pairs,
)
from habitus.scene.views.config import ViewsConfig
from habitus.scene.views.fulfilment import FulfilmentReading, LayerReading, expected_wait_hours, fulfilment_of
from habitus.scene.views.layers import (
    DoseReading,
    EvidenceIndex,
    Influence,
    Moderation,
    SharedEvidence,
    StabilityReport,
    avoids,
    dose_readings,
    layer_reading,
    reminded_split,
    shared_evidence,
    stability_scan,
    unordered_dose_concepts,
)
from habitus.scene.views.strength import Strength, TypeReading, TypeReadout, read_type, settled_by, strength_of


def borrow_candidates(hypothesis: Hypothesis, concepts: ConceptSet) -> tuple[str, ...]:
    """账薄时可以退回去用的上一级假设身份：单前件退到不带档 / 父概念，多前件退到少一个元素的子集。"""

    found: list[str] = []
    if len(hypothesis.antecedents) == 1:
        item = hypothesis.antecedents[0]
        if item.identity in concepts and item.grade is not None:
            found.append(_identity_with(hypothesis, (Antecedent(item.concept),)))
        if item.identity in concepts:
            for parent in concepts.ancestors(item.identity):
                found.append(_identity_with(hypothesis, (Antecedent(concepts[parent].name),)))
    else:
        for index in range(len(hypothesis.antecedents)):
            subset = hypothesis.antecedents[:index] + hypothesis.antecedents[index + 1 :]
            if any(concepts[a.identity].role.is_behavior for a in subset if a.identity in concepts):
                found.append(_identity_with(hypothesis, subset))
    return tuple(dict.fromkeys(found))


def _identity_with(hypothesis: Hypothesis, antecedents: tuple[Antecedent, ...]) -> str:
    """按 ``hypotheses`` 那边的分隔符拼上一级的身份；分隔符一改这里跟着变，不手抄。

    机会段用**本条自己的**（``opportunity_token``）：上一级猜的是别的机会数时就找不到它，于是不给指针——
    那是对的，借来的应该是同一个量（A-7 已定"拿子的口径读父账读出来的不是上一级的读数"）。
    """

    leaf = (
        ELEMENT_SEPARATOR.join(item.leaf_token for item in sorted(antecedents, key=lambda a: a.identity))
        + ASPECT_SEPARATOR
        + hypothesis.aspect.value
        + ASPECT_SEPARATOR
        + hypothesis.opportunity_token
    )
    return f"{hypothesis.consequent_identity}/{leaf}"


@dataclass(frozen=True)
class FallbackReading:
    """自己没攒够时给预测层的指针：上一级哪条、留一法之后它的读数。"""

    identity: str
    strength: Strength


def fallback_reading(
    account: Account,
    hypothesis: Hypothesis,
    ledger: LedgerStore,
    concepts: ConceptSet,
    config: ViewsConfig,
    hypotheses: Mapping[str, Hypothesis],
    now: datetime | None = None,
) -> FallbackReading | None:
    """账最厚、且过了门槛的上一级；留一法 = 去掉与本条共用任何一条出力 occurrence 的承诺（七f-6）。

    **用父假设本体的口径读父账**：父猜的可能是别的窗口（父 #2、子 #1），拿子的口径读父账读出来的不是
    "上一级的读数"（父自己读 −0.13，子拿去读同一本账得 −0.88，评审 A-7）。父假设不在 ``hypotheses`` 里（只有账、
    没有假设）就不给指针——那种账没人能解释它在量什么。
    """

    own = account.antecedent_uris
    best: FallbackReading | None = None
    for identity in borrow_candidates(hypothesis, concepts):
        parent_hypothesis = hypotheses.get(identity)
        if parent_hypothesis is None or parent_hypothesis.source.origin.is_placebo:
            continue  # 安慰剂的身份带前缀，本来就拼不出来；再拦一道是防将来有人改了身份规则
        parent = load_account(ledger, identity).where(partial(avoids, uris=own))
        strength = strength_of(parent, parent_hypothesis, config, seed=identity, now=now)
        if not strength.sufficient or strength.interval is None:
            continue
        if best is None or strength.accumulation.count > best.strength.accumulation.count:
            best = FallbackReading(identity, strength)
    return best


# ── 读数 ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RelationReading:
    hypothesis_identity: str
    aspect: Aspect
    strength: Strength
    type_reading: TypeReadout | None
    stability: StabilityReport
    dose: tuple[DoseReading, ...]
    shared: tuple[SharedEvidence, ...]
    reminded: tuple[LayerReading, LayerReading]
    open_claims: int
    #: 无节律型才有：兑现率与兑现间隔（节律型是 None）。有它的时候 ``strength`` 只带计数，``type_reading`` 一定是 None。
    fulfilment: FulfilmentReading | None = None
    fallback: FallbackReading | None = None
    #: 前件里档跨午夜的概念：剂量阶梯排不出顺序，所以那几个不出剂量读数（见 ``unordered_dose_concepts``）。
    unordered_dose: tuple[str, ...] = ()
    #: 账里有按**别的**指纹开的承诺——同身份的假设改过第几次机会或方向之后，旧口径的观测还留在盘上。
    stale_fingerprints: tuple[str, ...] = ()
    #: 这本账的对照出自哪几代预测树（出处，跨天多代是正常的——树每晚重建）。
    control_generations: tuple[str, ...] = ()
    #: **同一个触发日**里出现了两代对照：那一晚的承诺开在树重建的两边，这一批不可信。
    mixed_control_generations: tuple[str, ...] = ()

    @property
    def mixed_fingerprints(self) -> bool:
        return bool(self.stale_fingerprints)

    @property
    def clean(self) -> bool:
        """这本账的口径没混：没有按旧指纹开的承诺，同一触发日里没有两代对照。"""

        return not self.stale_fingerprints and not self.mixed_control_generations

    @property
    def uncalibrated(self) -> bool:
        """无节律型的读数**未校准**（2026-09-30 裁定五）：它读的是兑现率，没有"平时概率"可比、没有"区间含不含零"，
        攒够了就必然"显著"——真假设与安慰剂读出来一个样。收口规则与双向验证（生命周期那一份）做完之前不作数。"""

        return self.fulfilment is not None

    @property
    def significant(self) -> bool:
        """**只说"这个数不等于零"**，不说它换了条件还成不成立——那是稳定性，两件事（用户 09-27 要求分开印）。
        未校准的一律为假。"""

        if self.uncalibrated:
            return False
        return self.clean and self.strength.interval is not None and not self.strength.interval.contains_zero

    @property
    def stability_untested(self) -> bool:
        """一个情境都没测过（没有可分层的情境，或每一层都太少）——"没被推翻"里最容易被误读成"已证实"的就是这一段。"""

        return not self.stability.tested

    @property
    def unrefuted(self) -> bool:
        """过了门槛、区间不含零、没被任何测过的情境调节、账没混口径。**不是"已证实"**：没测过的情境不算。

        无节律型另一套判据（没有 p1−p0 可言）：见 ``FulfilmentReading.unrefuted``。"""

        if self.uncalibrated:
            return False  # 裁定五：无节律型的读数未校准，不进"未被推翻"的榜
        return (
            self.strength.interval is not None
            and not self.strength.interval.contains_zero
            and not self.stability.moderations
            and not self.stale_fingerprints
            and not self.mixed_control_generations
        )


def read_relation(
    hypothesis: Hypothesis,
    *,
    ledger: LedgerStore,
    concepts: ConceptSet,
    config: ViewsConfig | None = None,
    index: EvidenceIndex | None = None,
    hypotheses: Mapping[str, Hypothesis] | None = None,
    now: datetime,
) -> RelationReading:
    """读一条关系。``hypotheses`` 是这一轮在读的假设集（用来查上一级本体、辨别兄弟假设）。

    ``now`` **必填**：它是"命中记录可信到哪一刻"，读侧靠它判命运已定（对称截断）与仍立着多久（无节律型分母第三项）。
    旧签名缺省 ``None`` = 不截断，夜批漏传了一次就让第二轮的修法在生产路径上关了三周（评审 A-3 / B-3 / C-3）；
    "不截断"在生产里没有合法用途。
    """

    resolved = config or ViewsConfig()
    known = dict(hypotheses or {})
    known.setdefault(hypothesis.identity, hypothesis)
    account = load_account(ledger, hypothesis.identity)
    strength = strength_of(account, hypothesis, resolved, now=now)
    # 无节律型的四处分层也照跑，只是每层读的是兑现率而不是 p1−p0（P-1，用户 09-27 定"四处"）。
    fulfilment = fulfilment_of(account, hypothesis, resolved, now=now) if hypothesis.is_open_ended else None
    fallback = (
        None
        if hypothesis.is_open_ended or strength.sufficient
        else fallback_reading(account, hypothesis, ledger, concepts, resolved, known, now)
    )
    return RelationReading(
        hypothesis_identity=hypothesis.identity,
        aspect=hypothesis.aspect,
        strength=strength,
        type_reading=read_type(strength, resolved),
        stability=stability_scan(account, hypothesis, concepts, resolved, now),
        dose=dose_readings(account, hypothesis, concepts, resolved, now),
        unordered_dose=unordered_dose_concepts(hypothesis, concepts),
        shared=shared_evidence(account, hypothesis, index or EvidenceIndex(ledger), resolved, known, now),
        reminded=reminded_split(account, hypothesis, resolved, now),
        open_claims=account.open,
        fulfilment=fulfilment,
        fallback=fallback,
        stale_fingerprints=tuple(sorted(account.fingerprints - {hypothesis.fingerprint})),
        control_generations=tuple(sorted(account.control_generations)),
        mixed_control_generations=account.generations_split_within_a_day,
    )


def read_relations(
    hypotheses: Iterable[Hypothesis],
    *,
    ledger: LedgerStore,
    concepts: ConceptSet,
    config: ViewsConfig | None = None,
    now: datetime,
) -> tuple[RelationReading, ...]:
    """一轮读全部关系：证据倒排索引与假设集只建一次，两者都给每条读数用。``now`` 必填（见 ``read_relation``）。"""

    items = tuple(hypotheses)
    known = {item.identity: item for item in items}
    index = EvidenceIndex(ledger)
    return tuple(
        read_relation(item, ledger=ledger, concepts=concepts, config=config, index=index, hypotheses=known, now=now) for item in items
    )


__all__ = [
    "Account",
    "Accumulation",
    "DoseReading",
    "EvidenceIndex",
    "FallbackReading",
    "FulfilmentReading",
    "Influence",
    "LayerReading",
    "Moderation",
    "Pair",
    "RelationReading",
    "SharedEvidence",
    "StabilityReport",
    "Strength",
    "TypeReading",
    "TypeReadout",
    "ViewsConfig",
    "accumulation_of",
    "block_hours_of",
    "borrow_candidates",
    "dose_readings",
    "expected_wait_hours",
    "fallback_reading",
    "fulfilment_of",
    "informative_pairs",
    "layer_reading",
    "load_account",
    "read_relation",
    "read_relations",
    "read_type",
    "reminded_split",
    "settled_by",
    "settled_pairs",
    "shared_evidence",
    "stability_scan",
    "strength_of",
    "unordered_dose_concepts",
]
