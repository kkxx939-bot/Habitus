"""递送那一拍读边与单元的数：边此刻的位置、单元的提醒反馈与生效的设置，档变了才记一条。可见面用同一个读法。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime

from habitus.foresight import Evidence, Standing, Tolerance, climb
from habitus.foresight.feedback import feedback_reading
from habitus.foresight.ladder import effective_setting
from habitus.foresight.ledger import AccountLevel, Expectation, UnitLevel
from habitus.foresight.opportunities import OpportunityTable
from habitus.foresight.reliability import Reliability
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_units import Unit, UnitPlanner


class StandingReader:
    def __init__(
        self, ledger: ForesightLedgerStore, units: UnitPlanner, *, tolerance: Tolerance, evidence: Evidence
    ) -> None:
        self.ledger = ledger
        self.units = units
        self.tolerance = tolerance
        self.evidence = evidence

    def standing(self, account: str, kind: str, now: datetime, table: OpportunityTable | None) -> Standing:
        """一条边此刻的位置：从这个行为的单元历史（换代重标过的最新一代）里签名含它的那几刻读。"""

        history = self.units.history_of(table)
        return climb(
            history.of(kind) if history is not None else (),
            account,
            now=now,
            tolerance=self.tolerance,
            evidence=self.evidence,
            setting=self.ledger.setting_of(kind),
            previous=self.ledger.level_of(account),
        )

    def feedback(self, kind: str, table: OpportunityTable | None) -> Reliability | None:
        """单元的提醒反馈读数：反馈那几笔，加上单元历史里他没做的那几刻推时钟（``feedback.feedback_reading``）。"""

        facts = self.ledger.feedback_of(kind)
        if not facts:
            return None
        history = self.units.history_of(table)
        return feedback_reading(
            facts,
            moments=history.labelled(kind) if history is not None else (),
            alpha=self.evidence.alpha,
            half_life=self.evidence.half_life,
            window=self.evidence.window,
        )

    def setting(self, kind: str, now: datetime) -> str | None:
        return effective_setting(self.ledger.setting_of(kind), now)

    def for_tick(self, now: datetime, table: OpportunityTable | None) -> TickReadings:
        return TickReadings(self, now, table)

    def record_levels(self, accounts: Mapping[str, Standing], units: Iterable[Unit], now: datetime) -> int:
        """档变了才记一条：账的级按这本账自己从事实折出的档（滞回要知道上一次在哪一档；可见面要知道哪天升的），
        行为单元的档另记（``UnitLevel``，按行为）——两者分开，账自己折档时的滞回不读单元的档。依据不能开口的单元不记单元的档。"""

        changes = 0
        for account, standing in accounts.items():
            previous = self.ledger.level_of(account)
            if (previous.level if previous is not None else 0) != standing.level:
                if self.ledger.record_level(AccountLevel(account, now, standing.level)):
                    changes += 1
        latest: dict[str, int] = {}
        for unit in units:
            # 依据不能开口（只有候选关系、巧合到了时窗）的档是强制的 0，不是单元读数定的：不写单元的档，滞回读的上一次不被它冲掉
            # （同久别回来只记影子、不写单元的档）；成员接手时按同一组输入照常算档
            if not unit.voiced:
                continue
            latest[unit.kind_token] = unit.level
        for kind, level in latest.items():
            previous_unit = self.ledger.unit_level_of(kind)
            if (previous_unit.level if previous_unit is not None else 0) != level:
                if self.ledger.record_unit_level(UnitLevel(kind, now, level)):
                    changes += 1
        return changes


class TickReadings:
    """一拍里的读数：每本账的位置、每个行为的反馈与设置只读一次；单元按这一拍的读数拢。"""

    def __init__(self, reader: StandingReader, now: datetime, table: OpportunityTable | None) -> None:
        self.reader, self.now, self.table = reader, now, table
        #: 这一拍读过的账 → 位置（拍末按它记账的级）。
        self.accounts: dict[str, Standing] = {}
        self._feedbacks: dict[str, Reliability | None] = {}
        self._settings: dict[str, str | None] = {}

    def standing_of(self, item: Expectation) -> Standing:
        if item.account not in self.accounts:
            self.accounts[item.account] = self.reader.standing(item.account, item.kind_token, self.now, self.table)
        return self.accounts[item.account]

    def previous_level(self, kind: str) -> int | None:
        level = self.reader.ledger.unit_level_of(kind)
        return level.level if level is not None else None

    def feedback_of(self, kind: str) -> Reliability | None:
        if kind not in self._feedbacks:
            self._feedbacks[kind] = self.reader.feedback(kind, self.table)
        return self._feedbacks[kind]

    def setting_of(self, kind: str) -> str | None:
        if kind not in self._settings:
            self._settings[kind] = self.reader.setting(kind, self.now)
        return self._settings[kind]

    def plan(self, opened: Sequence[Expectation], waiting: Sequence[Expectation]) -> tuple[Unit, ...]:
        return self.reader.units.plan(
            opened,
            waiting,
            standing_of=self.standing_of,
            previous_level=self.previous_level,
            feedback_of=self.feedback_of,
            setting_of=self.setting_of,
            table=self.table,
            now=self.now,
        )

    def unit_of(self, group: Sequence[Expectation], due: Iterable[str]) -> Unit:
        return self.reader.units.unit_for(
            group,
            due=due,
            standing_of=self.standing_of,
            previous_level=self.previous_level,
            feedback_of=self.feedback_of,
            setting_of=self.setting_of,
            table=self.table,
            now=self.now,
        )


__all__ = ["StandingReader", "TickReadings"]
