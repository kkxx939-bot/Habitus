"""历史卡与此刻场景：判断者要比的两样东西，同一种材料。

一张历史卡是候选的**一次发生**：那次发生前后 ±k 槽内的原子行为序列（之前、这次、之后），加行为树
投影出来的视图（起因、并行、上一次、和谁、紧邻上下条），加语义树对那次的关联记录（当时是什么情况、
属于哪种情形、前因、用掉与留下的前提）。此刻场景是同一种材料的前半截：今天到此刻、当前槽 ±k 内已经
发生的原子行为，加还没封口的判断、今天的观测空白、今天已经做过什么。

本模块认识 scene（序列与视图从那里来），对预测树零知识——卡属于哪一层由 ``context`` 标，数字在
``numbers``。行为文档的字段名不出 scene：这里只用 ``DayIndex.rows`` 给的 ``FlowRow``。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType

from habitus.foresight.errors import ForesightError
from habitus.foresight.model import LAYER_NAMES, Moment, UnsealedRow
from habitus.foundation.integrity import canonical_digest
from habitus.scene.views import (
    AssociationGloss,
    ContextView,
    DayIndexCache,
    FlowRow,
    ObservationGap,
    slot_neighbourhood,
    slot_neighbourhood_until,
    slot_window,
)

_UNREADABLE_GAP_KIND = "没读懂"


@dataclass(frozen=True)
class HistoryCard:
    """候选的一次历史发生。``layer`` 是它落在的最内层（本槽 ⊂ 邻域 ⊂ 跨周几 ⊂ 全天）。

    ``flow[own_index]`` 就是这次；之前的行在它前面，之后的行在它后面。``gloss`` 为 None 有两种情形，
    由 ``day_associated`` 分开：那天关联还没做完（有数、没背景），或做完了但这一次没有留下记录。
    有记录就一定是已关联的那天——两边用的是同一个版本，不允许一张卡自相矛盾。
    """

    layer: str
    uri: str
    at: datetime
    flow: tuple[FlowRow, ...]
    own_index: int
    view: ContextView
    gloss: AssociationGloss | None
    day_associated: bool

    def __post_init__(self) -> None:
        if self.layer not in LAYER_NAMES:
            raise ForesightError(f"unknown shrinkage layer: {self.layer!r}")
        if not 0 <= self.own_index < len(self.flow) or self.flow[self.own_index].uri != self.uri:
            raise ForesightError("a history card must find its own occurrence inside its flow")
        if self.gloss is not None and self.gloss.occurrence_uri != self.uri:
            raise ForesightError("a history card's gloss must describe its own occurrence")
        if self.gloss is not None and not self.day_associated:
            raise ForesightError("a history card cannot carry a gloss for a day that is not associated")

    @property
    def own(self) -> FlowRow:
        return self.flow[self.own_index]

    @property
    def before(self) -> tuple[FlowRow, ...]:
        return self.flow[: self.own_index]

    @property
    def after(self) -> tuple[FlowRow, ...]:
        return self.flow[self.own_index + 1 :]


@dataclass(frozen=True)
class NowScene:
    """今天到此刻、当前槽 ±k 内的场景。

    ``since`` 是窗口起点（此刻所在槽往前 k 槽的墙钟起点）；``flow`` 是行为树上已封口的行；``unsealed`` 是
    判断存储里还没归约、读得懂、落在窗口内的判断（没有 URI）；``gaps`` 是今天到此刻为止树上的观测空白，
    ``unsealed_gaps`` 是判断存储里还没归约的今天的"没读懂"段；``done_today`` 是今天到此刻（不限邻域）每个
    kind 已发生的次数，未封口且落到 kind 的也计入（不限窗口）；``last_today`` 是这些 kind 今天最后一次的
    开始时刻，复发的"距上次"从它算。
    """

    moment: Moment
    since: datetime
    flow: tuple[FlowRow, ...]
    unsealed: tuple[UnsealedRow, ...]
    gaps: tuple[ObservationGap, ...]
    unsealed_gaps: tuple[ObservationGap, ...]
    done_today: Mapping[str, int]
    last_today: Mapping[str, datetime]

    def __post_init__(self) -> None:
        if self.since.astimezone(UTC) > self.moment.at.astimezone(UTC):
            raise ForesightError("the now scene's window must start no later than its moment")
        if set(self.done_today) != set(self.last_today):
            raise ForesightError("done_today and last_today must name the same kinds")
        if any(not row.readable for row in self.unsealed):
            raise ForesightError("unreadable unsealed rows belong in unsealed_gaps, not in unsealed")

    @property
    def fingerprint(self) -> str:
        """场景内容的摘要：同一槽内它没变，判断就不必再做一次。此刻的时分不算在内——那是钟在走，不是场景在变。"""

        return canonical_digest(
            {
                "flow": [(row.uri, _iso(row.at), _iso(row.last_observed_at)) for row in self.flow],
                "unsealed": [
                    (row.name, row.kind_token, _iso(row.started_at), _iso(row.last_observed_at), row.summary)
                    for row in self.unsealed
                ],
                "gaps": [(_iso(gap.started_at), _iso(gap.ended_at), gap.kind) for gap in self.gaps],
                "unsealed_gaps": [(_iso(gap.started_at), _iso(gap.ended_at), gap.kind) for gap in self.unsealed_gaps],
                "done_today": dict(self.done_today),
            }
        )

    def elapsed_seconds(self, kind_token: str) -> float | None:
        """今天这件事最后一次开始到此刻的秒数；今天没做过就是 None。"""

        last = self.last_today.get(kind_token)
        if last is None:
            return None
        return (self.moment.at.astimezone(UTC) - last.astimezone(UTC)).total_seconds()


def history_card(
    view: ContextView,
    layer: str,
    cache: DayIndexCache,
    *,
    gloss: AssociationGloss | None,
    day_associated: bool,
    slot_minutes: int,
    half_width: int,
) -> HistoryCard:
    """把一条投影视图铺成一张卡：补上它前后 ±k 槽的序列与那次的关联记录。"""

    if view.occurrence_uri is None:
        raise ForesightError("a history card needs a view of a real occurrence")
    flow = slot_neighbourhood(view.occurrence_uri, cache, slot_minutes=slot_minutes, half_width=half_width)
    own_index = next(index for index, row in enumerate(flow) if row.uri == view.occurrence_uri)
    return HistoryCard(
        layer=layer,
        uri=view.occurrence_uri,
        at=view.at,
        flow=flow,
        own_index=own_index,
        view=view,
        gloss=gloss,
        day_associated=day_associated,
    )


def now_scene(
    moment: Moment,
    cache: DayIndexCache,
    *,
    unsealed: Sequence[UnsealedRow],
    slot_minutes: int,
    half_width: int,
) -> NowScene:
    """此刻场景。

    未封口的行按与树上同一口径处理：今天到此刻开始的都算"今天做过"，落在当前窗口内的进流；没读懂的
    是今天的空白。已经在树上出现的同一条（同名同时刻）不重复计。调用方给的 ``unsealed`` 应当覆盖今天
    从零点起到此刻——只给窗口内那一截，"今天做过"就会漏掉早上那些还没归约的。
    """

    flow = slot_neighbourhood_until(moment.at, cache, slot_minutes=slot_minutes, half_width=half_width)
    start, _ = slot_window(moment.at, slot_minutes=slot_minutes, half_width=half_width)
    begin, finish = start.astimezone(UTC), moment.at.astimezone(UTC)
    day_begin = moment.at.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)
    today = cache.day(moment.day)
    sealed = {(row.name, row.at.astimezone(UTC)) for row in today.rows}
    today_rows = sorted(
        (row for row in unsealed if day_begin <= row.started_at.astimezone(UTC) <= finish),
        key=lambda row: (row.started_at.astimezone(UTC), row.name or ""),
    )
    fresh = [row for row in today_rows if row.readable and (row.name, row.started_at.astimezone(UTC)) not in sealed]
    pending = tuple(row for row in fresh if row.started_at.astimezone(UTC) >= begin)
    unsealed_gaps = tuple(
        ObservationGap(started_at=row.started_at, ended_at=row.last_observed_at, kind=_UNREADABLE_GAP_KIND)
        for row in today_rows
        if not row.readable
    )
    gaps = tuple(gap for gap in today.gaps if gap.started_at.astimezone(UTC) <= finish)
    done: Counter[str] = Counter()
    last: dict[str, datetime] = {}
    for row in today.rows:
        if row.at.astimezone(UTC) <= finish:
            _count(done, last, row.kind_token, row.at)
    for judgement in fresh:
        if judgement.kind_token is not None:
            _count(done, last, judgement.kind_token, judgement.started_at)
    return NowScene(
        moment=moment,
        since=start,
        flow=flow,
        unsealed=pending,
        gaps=gaps,
        unsealed_gaps=unsealed_gaps,
        done_today=MappingProxyType(dict(sorted(done.items()))),
        last_today=MappingProxyType(dict(sorted(last.items()))),
    )


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _count(done: Counter[str], last: dict[str, datetime], kind_token: str, started: datetime) -> None:
    done[kind_token] += 1
    previous = last.get(kind_token)
    if previous is None or started.astimezone(UTC) > previous.astimezone(UTC):
        last[kind_token] = started


__all__ = ["HistoryCard", "NowScene", "history_card", "now_scene"]
