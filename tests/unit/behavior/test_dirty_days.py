"""脏日日志（``behavior/tree/dirty_days``）：写与读同一个字节上限，写得进就读得回。"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from habitus.behavior.tree import dirty_days
from habitus.behavior.tree.dirty_days import DirtyDayLog

_FIRST = date(2026, 10, 1)


def test_a_log_that_would_outgrow_its_bound_folds_into_one_entry_and_always_reads_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E9：记到超出上限时把已有各号并成一号再记——文件始终在上限之内、读得回；水位落在被并的几号之间时，
    读侧拿到的是超集（多重读几天），不会漏掉水位之后变过的日子。"""

    monkeypatch.setattr(dirty_days, "_MAX_BYTES", 400)
    log = DirtyDayLog(tmp_path / "reduction")
    days = [_FIRST + timedelta(days=offset) for offset in range(20)]

    numbers = [log.record([day]) for day in days]

    assert numbers == list(range(1, 21))  # 序号照旧一号一号往后数
    assert log.path.stat().st_size <= 400
    assert log.after(0) == (20, frozenset(days))
    last, since_three = log.after(3)
    assert last == 20 and set(days[3:]) <= since_three
    assert log.after(20) == (20, frozenset())
