"""行为单元在组合根的那一半（裁定 56 第 12 条，方案第九节 6g）：把这一拍到了时窗的预期按行为拢成单元、读夜里重标过的历史
（``runtime.foresight_history``，裁定 55 新版第 2 条）、算单元读数与档；给可见面读单元视图。算法本身在 ``foresight.units``
（纯函数），这里只管拢组与读历史。

**拢组**：这一拍到了时窗、还没决定过的预期按类分组；同一类里说中的范围（起算 → 最晚时限）和它们连成一片、还没到时窗的那几条
一起拉进来——它们是同一刻的另外几个输入，单元替它们一并决定（记「单元已选」），不再等各自时窗开了再各说各的。依据只从到了时窗的
那几条里选（``units.basis`` 的 ``due``）。
**历史**：读这条 lane 盘上最晚一晚的单元历史（每晚重算一次），按晚缓存；白天不扫账本。
**档**：单元的档按行为记（``UnitLevel``），滞回读这个行为单元上一次的档，不读哪一本账的级；单元的提醒反馈与设置也按行为
（裁定 61：开口只看单元）。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime

from habitus.foresight import Evidence, Standing
from habitus.foresight.edges import PastMoment
from habitus.foresight.history import UnitHistory
from habitus.foresight.ledger import Expectation
from habitus.foresight.opportunities import OpportunityTable, TreeReading
from habitus.foresight.reliability import Reliability
from habitus.foresight.units import (
    Check,
    Input,
    UnitReading,
    basis,
    connected,
    self_check,
    tree_reading,
    unit_level,
    unit_reading,
)
from habitus.runtime.foresight_history_store import HistoryStore


@dataclass(frozen=True)
class Unit:
    """这一拍的一个单元：说话的那条（依据）、一并决定的另外几条、全部输入、单元读数与档、依据那本账换上单元档之后的位置。"""

    basis: Expectation
    members: tuple[Expectation, ...]
    inputs: tuple[Input, ...]
    reading: UnitReading
    level: int
    standing: Standing

    @property
    def kind_token(self) -> str:
        return self.basis.kind_token


@dataclass(frozen=True)
class UnitView:
    """可见面读的一个单元：行为、每个输入（账、语义树读数、两个数对不对得上）。"""

    kind_token: str
    inputs: tuple[tuple[str, Standing, Check], ...]


class UnitPlanner:
    def __init__(
        self, history: HistoryStore | None, *, evidence: Evidence, prior_share: float, lane: str = "session"
    ) -> None:
        if history is not None and not isinstance(history, HistoryStore):
            raise TypeError("history must be a HistoryStore or None")
        if not isinstance(lane, str) or not lane:
            raise ValueError("lane must be non-empty text")
        if not isinstance(evidence, Evidence):
            raise TypeError("evidence must be an Evidence")
        if isinstance(prior_share, bool) or not isinstance(prior_share, int | float) or not 0.0 < prior_share < 1.0:
            raise ValueError("prior_share must lie in (0, 1)")
        self.history = history
        self.evidence = evidence
        self.prior_share = float(prior_share)
        #: 没有机会表时按哪条 lane 读单元历史（账本是一条 lane 一本）。
        self.lane = lane
        #: （lane, 那一晚）→ 读回的单元历史：每晚只重算一次，白天每拍不再读盘。
        self._cached: dict[str, tuple[date, UnitHistory]] = {}

    def plan(
        self,
        opened: Sequence[Expectation],
        waiting: Sequence[Expectation],
        *,
        standing_of: Callable[[Expectation], Standing],
        previous_level: Callable[[str], int | None],
        feedback_of: Callable[[str], Reliability | None],
        setting_of: Callable[[str], str | None],
        table: OpportunityTable | None,
        now: datetime,
    ) -> tuple[Unit, ...]:
        """``opened`` 是这一拍到了时窗、没决定过的；``waiting`` 是还没到时窗、没决定过的（同一刻的另外几个输入从这里拉）。
        ``previous_level`` 按行为类给这个单元上一次的档；``feedback_of`` / ``setting_of`` 按行为类给单元的提醒反馈读数与生效的设置。"""

        history = self.history_of(table)
        return tuple(
            self.unit_for(
                group,
                due=due,
                standing_of=standing_of,
                previous_level=previous_level,
                feedback_of=feedback_of,
                setting_of=setting_of,
                table=table,
                now=now,
                history=history,
            )
            for group, due in _groups(opened, waiting)
        )

    def unit_for(
        self,
        group: Sequence[Expectation],
        *,
        due: Iterable[str],
        standing_of: Callable[[Expectation], Standing],
        previous_level: Callable[[str], int | None],
        feedback_of: Callable[[str], Reliability | None],
        setting_of: Callable[[str], str | None],
        table: OpportunityTable | None,
        now: datetime,
        history: UnitHistory | None = None,
    ) -> Unit:
        """一组输入（同一个行为、同一刻）的单元：时窗开时按 ``plan`` 拢的组，时窗末按依据与它当时一并决定的那几条重算一次。"""

        if history is None:
            history = self.history_of(table)
        inputs = tuple(Input(item, standing_of(item), *tree_reading(table, item)) for item in group)
        kind = group[0].kind_token
        past: tuple[PastMoment, ...] = history.of(kind) if history is not None else ()
        coincidental = frozenset(item.key for item in table.relations if item.coincidental) if table is not None else ()
        reading = unit_reading(
            inputs, past, now=now, evidence=self.evidence, prior_share=self.prior_share, coincidental=coincidental
        )
        chosen = basis(inputs, due=due)
        level = unit_level(
            chosen, reading, previous=previous_level(kind), feedback=feedback_of(kind), setting=setting_of(kind)
        )
        return Unit(
            basis=chosen.expectation,
            members=tuple(item for item in group if item.expectation_id != chosen.expectation.expectation_id),
            inputs=inputs,
            reading=reading,
            level=level,
            standing=replace(chosen.standing, level=level),
        )

    @staticmethod
    def views(inputs_by_kind: Mapping[str, Sequence[tuple[str, Standing, TreeReading | None]]]) -> tuple[UnitView, ...]:
        """可见面：每个行为的输入与自检（调用方给出每个输入的账身份、此刻的位置与语义树读数）。"""

        return tuple(
            UnitView(
                kind, tuple((account, standing, self_check(standing, reading)) for account, standing, reading in inputs)
            )
            for kind, inputs in sorted(inputs_by_kind.items())
        )

    def history_of(self, table: OpportunityTable | None) -> UnitHistory | None:
        """这条 lane 盘上最晚一晚的单元历史（每晚只读一次）；边的读数从它来。"""

        if self.history is None:
            return None
        lane = table.lane if table is not None else self.lane
        night = self.history.latest_night(lane)
        if night is None:
            return None
        cached = self._cached.get(lane)
        if cached is None or cached[0] != night:
            cached = (night, self.history.read(lane, night))
            self._cached[lane] = cached
        return cached[1]


def _groups(
    opened: Sequence[Expectation], waiting: Iterable[Expectation]
) -> list[tuple[list[Expectation], tuple[str, ...]]]:
    """按类、按说中的范围连成一片拢组（与历史上"一刻"同一把尺 ``units.connected``）；同一刻还没到时窗的那几条也拉进来。
    返回（这一组，其中到了时窗的那几条的身份）。"""

    by_kind: dict[str, list[Expectation]] = {}
    for item in opened:
        by_kind.setdefault(item.kind_token, []).append(item)
    pending: dict[str, list[Expectation]] = {}
    for item in waiting:
        pending.setdefault(item.kind_token, []).append(item)
    found: list[tuple[list[Expectation], tuple[str, ...]]] = []
    for kind in sorted(by_kind):
        for group in connected(by_kind[kind]):
            start = min(item.observed_from for item in group)
            end = max(item.deadline for item in group)
            due = tuple(item.expectation_id for item in group)
            group += [item for item in pending.get(kind, ()) if item.observed_from < end and item.deadline > start]
            found.append((sorted(group, key=lambda item: (item.window[0], item.expectation_id)), due))
    return found


__all__ = ["Unit", "UnitPlanner", "UnitView"]
