"""把一刻的证据装成一个包：候选、每个候选的四层数字与历史卡、此刻场景。

编排而已：数字在 ``numbers``、卡在 ``context`` 与 ``cards``、类型在 ``model``，本模块只负责"按同一个
槽把它们对齐"，并且保证**钟面的映射只有一种算法**——槽与周几由预测树自己的 ``SlotKey.of`` 按树发布
的槽宽算出来，不在这里另写一遍取模。

候选只按本周几取（``query.slot_outlook``），与算法一致：曲线按（周几，行为）发布，只在周二做过的事
周三不是候选。候选的**入场**不按"大于 N 次"：本槽、邻域、跨周几三层里至少一层有出处日（在这个时段
附近真发生过）的候选摊开卡，只有全天层有出处的只列名字与数字。一次算不算数，判断者对着数字自己判。

纯函数：不读时钟、不碰存储、不调模型。此刻是谁、树钉哪一代、缓存怎么建、未封口从哪读、规律树怎么读，
全部由组合根注入。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType

from habitus.foresight.cards import NowScene, now_scene
from habitus.foresight.context import CandidateBackground, candidate_background
from habitus.foresight.errors import ForesightError
from habitus.foresight.model import LAYER_NAMES, CandidateNumbers, Moment, Provenance, UnsealedRow
from habitus.foresight.numbers import CellIndex, candidate_numbers, provenance
from habitus.prediction.model import PredictionTree, SlotKey
from habitus.prediction.query import slot_outlook
from habitus.scene.views import AssociationGloss, DayIndexCache, Situation

#: "这个候选哪几天关联完成了"。由组合根注入——本层不认识规律树的存储，只认这个事实。
AssociatedDays = Callable[[str], frozenset[date]]
#: "这个候选在这些日子的关联记录"，键是 occurrence URI。
GlossesFor = Callable[[str, Iterable[date]], Mapping[str, AssociationGloss]]
#: "这个候选的情形列表"，按周几分成（本周几出现过的，其他周几的）。
SituationsFor = Callable[[str, int], tuple[tuple[Situation, ...], tuple[Situation, ...]]]


@dataclass(frozen=True)
class CandidateEvidence:
    """一个候选在这一刻的证据：树发布的数、四层拆解、历史卡、情形列表。

    ``expanded`` 为 False 的候选只有数字（这个时段附近从没发生过，只列名）；它的 ``background`` 是空的，
    情形也不取。
    """

    kind_token: str
    numbers: CandidateNumbers
    provenance: Provenance
    expanded: bool
    background: CandidateBackground
    situations_here: tuple[Situation, ...] = ()
    situations_elsewhere: tuple[Situation, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind_token, str) or not self.kind_token:
            raise ForesightError("candidate kind_token must be non-empty text")
        if not self.expanded and (self.background.cards or self.situations_here or self.situations_elsewhere):
            raise ForesightError("a candidate that is only named must not carry cards or situations")

    @property
    def unassociated(self) -> tuple[date, ...]:
        """四层合起来有数、却没有语义背景的日子。"""

        return self.provenance.unassociated


@dataclass(frozen=True)
class EvidencePack:
    """一刻的全部证据：钉住的一代、此刻、此刻场景、按名字排好的候选。``slot_minutes`` 是这一代树的槽宽。"""

    generation: str
    config_digest: str
    slot_minutes: int
    moment: Moment
    now: NowScene
    candidates: tuple[CandidateEvidence, ...]

    def __post_init__(self) -> None:
        for name in ("generation", "config_digest"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ForesightError(f"evidence pack {name} must be non-empty text")
        if isinstance(self.slot_minutes, bool) or not isinstance(self.slot_minutes, int) or self.slot_minutes <= 0:
            raise ForesightError("evidence pack slot_minutes must be a positive integer")
        if not 0 <= self.moment.slot < self.slots_per_day:
            raise ForesightError("the pack's moment does not fit its clock face")
        names = [item.kind_token for item in self.candidates]
        if names != sorted(set(names)):
            raise ForesightError("evidence pack candidates must be unique and sorted by name")
        if self.now.moment != self.moment:
            raise ForesightError("the now scene must be taken at the pack's own moment")

    @property
    def slots_per_day(self) -> int:
        return 1440 // self.slot_minutes

    @property
    def expanded(self) -> tuple[CandidateEvidence, ...]:
        return tuple(item for item in self.candidates if item.expanded)

    @property
    def named_only(self) -> tuple[CandidateEvidence, ...]:
        return tuple(item for item in self.candidates if not item.expanded)


def moment_at(at: datetime, *, slot_minutes: int, day_note: str | None = None) -> Moment:
    """把一个**主体本地时刻**落到钟面上。槽与周几走树的映射，不另写一份。"""

    if not isinstance(at, datetime) or at.utcoffset() is None:
        raise ForesightError("moment must be a timezone-aware local datetime")
    key = SlotKey.of(at, slot_minutes=slot_minutes)
    return Moment(at=at, slot=key.slot, weekday=key.weekday, day_note=day_note)


def assemble(
    tree: PredictionTree,
    moment: Moment,
    cache: DayIndexCache,
    *,
    generation: str,
    unsealed: Sequence[UnsealedRow],
    glosses_for: GlossesFor,
    situations_for: SituationsFor,
    associated: AssociatedDays,
    half_width: int,
    window_days: int,
    transition_window_seconds: float,
    max_days_per_layer: int,
) -> EvidencePack:
    """这一刻的证据包。

    ``associated`` 回答"这个候选这一天关联完成了没有"；``glosses_for`` 按同一把尺子读记录。数字那边算出的
    ``unassociated`` 与卡上贴到的记录必须来自同一个判据——两处用两个判据是这一整套最容易出的错。
    """

    if not isinstance(tree, PredictionTree):
        raise ForesightError("tree must be a PredictionTree")
    slot = SlotKey(weekday=moment.weekday, slot=moment.slot)
    cells = CellIndex.of(tree)
    scene = now_scene(moment, cache, unsealed=unsealed, slot_minutes=tree.slot_minutes, half_width=half_width)
    candidates = tuple(
        _candidate(
            candidate.action,
            cells,
            slot,
            moment,
            cache,
            scene,
            glosses_for=glosses_for,
            situations_for=situations_for,
            associated=associated,
            half_width=half_width,
            window_days=window_days,
            transition_window_seconds=transition_window_seconds,
            max_days_per_layer=max_days_per_layer,
        )
        for candidate in slot_outlook(tree, slot).candidates
    )
    return EvidencePack(
        generation=generation,
        config_digest=tree.config_digest,
        slot_minutes=tree.slot_minutes,
        moment=moment,
        now=scene,
        candidates=tuple(sorted(candidates, key=lambda item: item.kind_token)),
    )


def _candidate(
    kind_token: str,
    cells: CellIndex,
    slot: SlotKey,
    moment: Moment,
    cache: DayIndexCache,
    scene: NowScene,
    *,
    glosses_for: GlossesFor,
    situations_for: SituationsFor,
    associated: AssociatedDays,
    half_width: int,
    window_days: int,
    transition_window_seconds: float,
    max_days_per_layer: int,
) -> CandidateEvidence:
    """一个候选：树发布的数、四层拆解，摊开的再配历史卡与情形。"""

    tree = cells.tree
    numbers = candidate_numbers(
        tree,
        slot,
        kind_token,
        done_today=scene.done_today.get(kind_token, 0),
        elapsed_seconds=scene.elapsed_seconds(kind_token),
    )
    done = associated(kind_token)
    layers = provenance(cells, kind_token, slot, half_width=half_width, associated_on=done.__contains__)
    if not any(layer.days for layer in (layers.slot, layers.pool, layers.cross_weekday)):
        return CandidateEvidence(
            kind_token=kind_token, numbers=numbers, provenance=layers, expanded=False, background=_empty_background()
        )
    background = candidate_background(
        layers,
        kind_token,
        cache,
        glosses=glosses_for(kind_token, layers.all_day.days),
        slot_minutes=tree.slot_minutes,
        slot_index=moment.slot,
        half_width=half_width,
        window_days=window_days,
        transition_window_seconds=transition_window_seconds,
        max_days_per_layer=max_days_per_layer,
    )
    here, elsewhere = situations_for(kind_token, moment.weekday)
    return CandidateEvidence(
        kind_token=kind_token,
        numbers=numbers,
        provenance=layers,
        expanded=True,
        background=background,
        situations_here=here,
        situations_elsewhere=elsewhere,
    )


def _empty_background() -> CandidateBackground:
    return CandidateBackground(cards=(), dropped_days=MappingProxyType(dict.fromkeys(LAYER_NAMES, 0)))


__all__ = [
    "AssociatedDays",
    "CandidateEvidence",
    "EvidencePack",
    "GlossesFor",
    "SituationsFor",
    "assemble",
    "moment_at",
]
