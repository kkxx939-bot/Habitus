"""语义侧：一个候选的历史卡，按四层的出处日取，同一次发生只一张卡。

树的四层各由一批不同的日子攒成，背景就必须跟着各自那批走——拿"全部覆盖日 + 槽过滤"给所有层
配同一份历史，等于让判断者看着邻域的数字读全天的背景。日子从树来（``foresight.numbers``），
本模块只负责按层去问行为树，并且把**槽过滤**也对齐到那一层实际的算法：

===============  ====================  ==============================================
层                日子                  槽过滤
===============  ====================  ==============================================
``slot``          该格的出处日           该槽，半宽 0
``pool``          ±k 格的出处日并集       该槽，半宽 k（与树的池化同一个窗口）
``cross_weekday`` 七个周几 × ±k 的并集     该槽，半宽 k（这一层只扩周几，不动时间窗）
``all_day``       该动作全部格子的并集     不过滤（这一层本来就不看时刻）
===============  ====================  ==============================================

四层互相包含（本槽 ⊂ 邻域 ⊂ 跨周几 ⊂ 全天），同一次发生会被四层都取到。卡**只建一张**，标它落在的
最内层——判断者不该把 09-09 那次打球当四次；四层表照旧按层给计数与出处日，那是 ``numbers`` 的事，
这里一个数不动。

**不按"关联完成了没有"筛日子。**卡上的序列与视图来自行为树，语义层做没做过它一点都不影响；关联进度
由 ``day_associated`` 与 ``Provenance.unassociated`` 如实摆出来，与取到几张卡是两件事。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC
from types import MappingProxyType

from habitus.foresight.cards import HistoryCard, history_card
from habitus.foresight.errors import ForesightError
from habitus.foresight.model import LAYER_NAMES, Layer, Provenance
from habitus.scene.views import AssociationGloss, DayIndexCache, history_contexts


@dataclass(frozen=True)
class CandidateBackground:
    """一个候选的全部历史卡，按时刻升序。

    ``dropped_days`` 是每层被 ``max_days_per_layer`` 截掉的**更早**的日子数，要说出来，否则"给你看的
    这几张"会被读成"一共就这几张"。卡的张数不设上限（2026-09-16 定：方案验证之前不做截断）。
    """

    cards: tuple[HistoryCard, ...]
    dropped_days: Mapping[str, int]

    def __post_init__(self) -> None:
        if set(self.dropped_days) != set(LAYER_NAMES):
            raise ForesightError("candidate background must account for all four layers")
        uris = [card.uri for card in self.cards]
        if len(set(uris)) != len(uris):
            raise ForesightError("a candidate background must not carry the same occurrence twice")

    def in_layer(self, name: str) -> tuple[HistoryCard, ...]:
        return tuple(card for card in self.cards if card.layer == name)


def candidate_background(
    provenance: Provenance,
    kind_token: str,
    cache: DayIndexCache,
    *,
    glosses: Mapping[str, AssociationGloss],
    slot_minutes: int,
    slot_index: int,
    half_width: int,
    window_days: int,
    transition_window_seconds: float,
    max_days_per_layer: int,
) -> CandidateBackground:
    """按四层的出处日取候选的历史卡；从最内层往外取，先取到的层就是那张卡的层。"""

    if not isinstance(provenance, Provenance):
        raise ForesightError("provenance must be a Provenance")
    if isinstance(max_days_per_layer, bool) or not isinstance(max_days_per_layer, int) or max_days_per_layer <= 0:
        raise ForesightError("max_days_per_layer must be a positive integer")
    unassociated = set(provenance.unassociated)
    cards: dict[str, HistoryCard] = {}
    dropped: dict[str, int] = {}
    for layer in provenance:
        days = layer.days[-max_days_per_layer:]
        dropped[layer.name] = len(layer.days) - len(days)
        minutes, index, width = _slot_filter(layer, slot_minutes=slot_minutes, slot_index=slot_index, half_width=half_width)
        for view in history_contexts(
            kind_token,
            cache,
            days=days,
            window_days=window_days,
            slot_minutes=minutes,
            slot_index=index,
            slot_half_width=width,
            transition_window_seconds=transition_window_seconds,
        ):
            uri = view.occurrence_uri
            if uri is None or uri in cards:
                continue
            day_associated = view.day not in unassociated
            gloss = glosses.get(uri)
            if gloss is not None and not day_associated:
                # 数字那边说这天没关联、记录那边却有：两边读的不是同一个版本。硬拒，不让判断者拿到
                # 一张"表说没背景、卡上贴着背景"的自相矛盾的卡。
                raise ForesightError(
                    f"{kind_token!r} on {view.day} has an association record but is not counted as associated; "
                    "the glosses and the associated-days source disagree on the association version"
                )
            cards[uri] = history_card(
                view,
                layer.name,
                cache,
                gloss=gloss,
                day_associated=day_associated,
                slot_minutes=slot_minutes,
                half_width=half_width,
            )
    ordered = tuple(sorted(cards.values(), key=lambda card: (card.at.astimezone(UTC), card.uri)))
    return CandidateBackground(cards=ordered, dropped_days=MappingProxyType(dropped))


def _slot_filter(
    layer: Layer, *, slot_minutes: int, slot_index: int, half_width: int
) -> tuple[int | None, int | None, int]:
    """这一层该用多宽的槽过滤——与树算这一层时的窗口一一对应。"""

    if layer.name == "all_day":
        return (None, None, 0)
    # 只有本槽是单格；邻域与跨周几都用同一个 ±k 窗口——它们的差别在**日子**（本周几 / 七个
    # 周几），不在时间窗。链上的 cross 就是这么算的。
    return (slot_minutes, slot_index, 0 if layer.name == "slot" else half_width)


__all__ = ["CandidateBackground", "candidate_background"]
