"""每晚新增的召回（裁定 60 第 2 条）：按字符二元组相似挑旧条目；一条旧条目只进一组；没有重合的不召回；每条有上限。"""

from __future__ import annotations

from datetime import date

import pytest

from habitus.behavior.kinds.codec import pending_from, pending_payload
from habitus.behavior.kinds.ids import Lane
from habitus.behavior.kinds.nightly.recall import recall
from habitus.behavior.kinds.pending import PendingEntry


def entry(key: str, name: str, content: str, *, grouped: bool = True) -> PendingEntry:
    day = date(2026, 3, 1)
    return PendingEntry(
        f"behavior://{key}", Lane.SESSION, day, name, content, entered_on=day, grouped_on=day if grouped else None
    )


def test_similar_old_entries_are_recalled_once_each_within_the_cap() -> None:
    old = [
        entry("o1", "订去杭州的机票", "订 10 月去杭州出差的机票"),
        entry("o2", "订杭州机票", "订杭州的机票"),
        entry("o3", "问孩子作业", "问孩子为什么不写作业"),
        entry("o4", "Ｑ", "＃"),
    ]
    first = entry("n1", "订杭州出差的机票", "给下个月杭州出差订机票", grouped=False)
    second = entry("n2", "订去杭州的机票", "订去杭州的机票", grouped=False)
    found = recall([first, second], old, per_entry=1)
    assert [item.occurrence for item in found] == ["behavior://o1", "behavior://o2"]  # 第二条拿不到已被第一条挑走的
    assert recall([first], old, per_entry=5)[-1].occurrence != "behavior://o4"  # 没有重合的不召回
    assert all(item.occurrence != "behavior://o3" for item in recall([first], old, per_entry=2))
    assert recall([first], old, per_entry=0) == ()
    with pytest.raises(ValueError):
        recall([first], old, per_entry=-1)


def test_the_pool_codec_keeps_the_dates_and_reads_an_older_pool_as_never_grouped() -> None:
    dated = entry("o1", "订机票", "订机票")
    assert pending_from(pending_payload(dated)) == dated
    legacy = {key: value for key, value in pending_payload(dated).items() if key not in {"entered_on", "grouped_on"}}
    restored = pending_from(legacy)
    assert restored.fresh and restored.entered_on is None
