"""账本按账压实的编排（26 号清单 4.8；裁定 61 第 7 条）：窗外的文件删掉、摘要留下；写完摘要、删文件之前崩溃也不变；
没结完的日子挡住压实；级的变化只留诊断够得着的那一段与它之前最近的一条。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from habitus.foresight import Evidence, account_key, ranked_account
from habitus.foresight.ledger import AccountFact, AccountLevel, Expectation
from habitus.runtime.foresight_compaction import LedgerCompaction
from habitus.runtime.foresight_ledger import ForesightLedgerStore

CST = timezone(timedelta(hours=8))
START = datetime(2026, 1, 1, 9, 0, tzinfo=CST)
NOW = START + timedelta(days=200)
ACCOUNT = account_key("s-k0002", "日子", "rel")
EVIDENCE = Evidence(alpha=0.05, half_life=8.0)


def seeded(ledger: ForesightLedgerStore, days: int = 120) -> None:
    for day in range(days):
        at = START + timedelta(days=day)
        if day % 9 == 8:
            ledger.append_fact(AccountFact(ACCOUNT, at, False, 1.0, "过期待决", f"d{day:03d}"))
        else:
            minutes = 0 if day % 3 else 40
            kind = "准时" if minutes == 0 else "晚了但做了"
            ledger.append_fact(AccountFact(ACCOUNT, at, True, 1.0, kind, f"d{day:03d}", offset_minutes=minutes))


def reading(ledger: ForesightLedgerStore):  # type: ignore[no-untyped-def]
    """压实之外读得到的东西：窗内的事实与压实段的摘要（偏差、间隔）。"""

    account = ranked_account(ledger.facts_of(ACCOUNT), visible_at=NOW, base=ledger.compaction_of(ACCOUNT))
    base = account.base
    return (account.prediction.keys[-35:], None if base is None else (base.offsets, base.gap_count, base.longest_gap))


def compaction(ledger: ForesightLedgerStore) -> LedgerCompaction:
    return LedgerCompaction(ledger, evidence=EVIDENCE, clock=lambda: NOW)


def test_compaction_keeps_the_window_and_the_summary_and_drops_the_files(tmp_path) -> None:
    ledger = ForesightLedgerStore(tmp_path)
    seeded(ledger)
    keys, _summary = reading(ledger)
    report = compaction(ledger).run(night=NOW.date(), table=None)
    assert (report.accounts, report.compacted, report.failed) == (1, 1, {})
    base = ledger.compaction_of(ACCOUNT)
    assert base is not None and base.through == (START + timedelta(days=84), "d084")  # 120 次里留最近 35 次
    assert report.discarded == 85
    assert len(list((ledger._account_dir(ACCOUNT) / "facts").iterdir())) == 120 - report.discarded
    after_keys, summary = reading(ledger)
    assert after_keys == keys and summary is not None
    offsets, gaps, longest = summary
    assert (
        dict(offsets) == {0: 47, 40: 29} and gaps == 84 and longest == 1.0
    )  # 前 85 天：窗内 47、晚 40 分钟 29、没做 9
    # 第二晚：没有新机会就不再压；之后的事实照常接着 fold
    assert compaction(ledger).run(night=NOW.date(), table=None).compacted == 0


def test_a_crash_after_writing_the_summary_and_before_deleting_the_files_changes_nothing(tmp_path) -> None:
    ledger = ForesightLedgerStore(tmp_path)
    seeded(ledger)
    keys, _summary = reading(ledger)
    original = ledger.discard_compacted
    ledger.discard_compacted = lambda account: 0  # type: ignore[method-assign]
    compaction(ledger).run(night=NOW.date(), table=None)
    assert len(list((ledger._account_dir(ACCOUNT) / "facts").iterdir())) == 120  # 文件都还在
    assert len(ledger.facts_of(ACCOUNT)) == 35  # 读的一方只看压实以外的
    assert reading(ledger)[0] == keys
    ledger.discard_compacted = original  # type: ignore[method-assign]
    assert compaction(ledger).run(night=NOW.date(), table=None).discarded == 85  # 下一晚把残留删掉
    assert reading(ledger)[0] == keys


def test_an_unfinished_day_holds_compaction_back(tmp_path) -> None:
    ledger = ForesightLedgerStore(tmp_path)
    seeded(ledger)
    formed = START + timedelta(days=10)
    ledger.record_expectation(
        Expectation(
            expectation_id="open",
            kind_token="s-k0002",
            day=formed.date(),
            formed_at=formed,
            source="日子",
            trigger="rel",
            trigger_instance="behavior://open.md",
            window=(formed, formed + timedelta(minutes=30)),
            deadline=formed + timedelta(hours=2),
            basis=(),
            opportunities=3,
            hits=3,
            observed_from=formed,
        )
    )
    compaction(ledger).run(night=NOW.date(), table=None)
    base = ledger.compaction_of(ACCOUNT)
    assert base is not None and base.through == (START + timedelta(days=9), "d009")  # 第 10 天还没结完


def test_only_the_reach_of_diagnosis_and_the_level_just_before_it_are_kept(tmp_path) -> None:
    ledger = ForesightLedgerStore(tmp_path)
    seeded(ledger, days=40)
    for offset, level in ((150, 1), (120, 0), (90, 1), (30, 0), (5, 1)):
        ledger.record_level(AccountLevel(ACCOUNT, NOW - timedelta(days=offset), level))
    report = compaction(ledger).run(night=NOW.date(), table=None)
    assert report.trimmed == 2
    assert [item.at for item in ledger.levels_of(ACCOUNT)] == [
        NOW - timedelta(days=90),
        NOW - timedelta(days=30),
        NOW - timedelta(days=5),
    ]
    assert date(2026, 1, 1) < NOW.date()
