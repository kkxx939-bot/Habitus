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


def test_two_readers_keep_their_own_watermarks_and_the_log_is_cut_at_the_lowest(tmp_path: Path) -> None:
    """实时化缺陷 5（E11）：读者 A 记下水位 → 某天变了 → 读者 B 读到并截断 → A 再读，A 也读到变过的那一天。水位按读者名落盘，
    日志只截到全部读者里最小的那个；换一个实例（进程重启）水位还在。"""

    log = DirtyDayLog(tmp_path / "reduction")
    log.advance("a", log.last())
    log.advance("b", log.last())
    changed = log.record([_FIRST])
    _last, seen_by_b = log.after(log.watermark("b") or 0)
    log.advance("b", changed)
    assert seen_by_b == frozenset({_FIRST})
    reopened = DirtyDayLog(tmp_path / "reduction")
    assert reopened.watermark("a") == 0 and reopened.watermark("b") == changed
    assert reopened.after(reopened.watermark("a") or 0) == (changed, frozenset({_FIRST}))  # A 也读得到
    reopened.advance("a", changed)
    assert reopened.after(0) == (changed, frozenset())  # 两个读者都读过了才截掉


def test_a_reader_watermark_file_with_a_repeated_key_is_refused(tmp_path: Path) -> None:
    """F7：读者水位文件里同一个读者名写了两次，不让后一个悄悄盖掉前一个——报错，同截断、多字段。"""

    from habitus.behavior.tree.dirty_days import DirtyDayLogError

    log = DirtyDayLog(tmp_path / "reduction")
    log.advance("a", 0)
    log.readers_path.write_text('{"a": 5, "a": 0}', encoding="utf-8")
    with pytest.raises(DirtyDayLogError):
        log.watermark("a")
