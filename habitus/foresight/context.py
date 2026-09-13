"""语义侧：每一层只按**这一层的出处日**去取当时的上下文。

树的四层各由一批不同的日子攒成，背景就必须跟着各自那批走——拿"全部覆盖日 + 槽过滤"给所有层
配同一份历史，等于让判断者看着邻域的数字读全天的背景。日子从树来（``foresight.numbers``），
本模块只负责按层去问情景树，并且把**槽过滤**也对齐到那一层实际的算法：

===============  ====================  ==============================================
层                日子                  槽过滤
===============  ====================  ==============================================
``slot``          该格的出处日           该槽，半宽 0
``pool``          ±k 格的出处日并集       该槽，半宽 k（与树的池化同一个窗口）
``cross_weekday`` 七个周几 × ±k 的并集     该槽，半宽 k（这一层只扩周几，不动时间窗）
``all_day``       该动作全部格子的并集     不过滤（这一层本来就不看时刻）
===============  ====================  ==============================================

日子已经把范围收死了，槽过滤只负责在那一天里挑中对的那一次——同一天里这个 kind 可能发生过
好几回，出处日说不出是哪一回，槽说得出。
"""

from __future__ import annotations

from dataclasses import dataclass

from habitus.foresight.errors import ForesightError
from habitus.foresight.model import Layer
from habitus.scene.views import ContextView, DayIndexCache, history_contexts


@dataclass(frozen=True)
class LayerBackground:
    """一层的数字配一层的背景。

    ``dropped_days`` 是被保护闸截掉的**更早**的日子：一个跑了两年的习惯，全天那层的出处日
    有几百个，全展开既撑爆上下文也没有额外信息。截掉多少要说出来，否则判断者会把"给他看的
    这 40 天"读成"一共就这 40 天"。
    """

    layer: Layer
    views: tuple[ContextView, ...]
    dropped_days: int


def layer_background(
    layer: Layer,
    kind_token: str,
    cache: DayIndexCache,
    *,
    slot_minutes: int,
    slot_index: int,
    half_width: int,
    window_days: int,
    transition_window_seconds: float,
    max_days: int,
) -> LayerBackground:
    """按 ``layer`` 自己的出处日取这一层的历史上下文，最近的 ``max_days`` 天优先。

    只问归组过的日子（``layer.grouped``）；没归组的那几天已经在 ``layer.ungrouped`` 里如实
    摆着，不在这里再默默丢一次。
    """

    if not isinstance(layer, Layer):
        raise ForesightError("layer must be a Layer")
    if isinstance(max_days, bool) or not isinstance(max_days, int) or max_days <= 0:
        raise ForesightError("max_days must be a positive integer")
    available = layer.grouped
    days = available[-max_days:]
    minutes, index, width = _slot_filter(layer, slot_minutes=slot_minutes, slot_index=slot_index, half_width=half_width)
    views = history_contexts(
        kind_token,
        cache,
        days=days,
        window_days=window_days,
        slot_minutes=minutes,
        slot_index=index,
        slot_half_width=width,
        transition_window_seconds=transition_window_seconds,
    )
    return LayerBackground(layer=layer, views=views, dropped_days=len(available) - len(days))


def _slot_filter(
    layer: Layer, *, slot_minutes: int, slot_index: int, half_width: int
) -> tuple[int | None, int | None, int]:
    """这一层该用多宽的槽过滤——与树算这一层时的窗口一一对应。"""

    if layer.name == "all_day":
        return (None, None, 0)
    # 只有本槽是单格；邻域与跨周几都用同一个 ±k 窗口——它们的差别在**日子**（本周几 / 七个
    # 周几），不在时间窗。链上的 cross 就是这么算的。
    return (slot_minutes, slot_index, 0 if layer.name == "slot" else half_width)


__all__ = ["LayerBackground", "layer_background"]
