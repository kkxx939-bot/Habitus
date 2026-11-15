"""第二步甲：未封口的那座桥与模型队列（共享说明 3.2、3.3；预期表 C19、C19b、D4）。

判断上已经带着选词结果的（会话）直接用、这一拍不调模型、不产生 ``unsealed`` 欠账；没带的（物理）照现在这样提前选、按节点
身份缓存，全新盘面（在用状态修订号 0）下也照样选。桥由组合根的 ``runtime.foresight.unsealed_bridge`` 从一支装出来。
旧代码上应当失败：词表版本为 0 时桥一条都不归。
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from habitus.runtime.foresight import unsealed_bridge
from tests.unit.word_pick.site import CST, Site, Word, at
from tests.unit.word_pick.test_physical_pick import delivery

DAY = date(2026, 4, 13)


@pytest.fixture
def site(tmp_path: Path):  # type: ignore[no-untyped-def]
    built = Site(tmp_path)
    yield built
    built.close()


def test_c19_d4_an_unsealed_session_judgement_already_carries_its_word(site: Site) -> None:
    site.router.picks.append(Word("s-k0002"))
    site.consume(site.turn("conv-u", at(DAY, 9), "帮我写一份本周周报"))  # 写了判断，还没归约
    picks = site.router.count("pick")
    bridge = unsealed_bridge(site.session)
    assert bridge.unclassified() == 0  # 会话的判断不欠 unsealed 的账
    asyncio.run(bridge.prepare(until=at(DAY, 23)))
    assert site.router.count("pick") == picks  # 这一拍选词 0 次
    (row,) = bridge.rows(since=at(DAY, 0), until=at(DAY, 23))
    assert row.kind_token == "s-k0002"


def test_c19b_an_unsealed_physical_judgement_is_picked_once_then_read_from_the_cache_on_a_fresh_disk(
    site: Site,
) -> None:
    site.router.picks.extend([Word("p-k0005"), Word("p-k0003")])
    asyncio.run(site.runtime.deliver_physical(delivery("换衣服")))  # 还没归约：判断在判断存储里等封口
    bridge = unsealed_bridge(site.physical)
    now = datetime.now(tz=CST)
    asyncio.run(bridge.prepare(until=now))
    assert site.router.count("pick") == 1
    asyncio.run(bridge.prepare(until=now + timedelta(minutes=15)))
    assert site.router.count("pick") == 1  # 下一拍读缓存
    rows = [row for row in bridge.rows(since=now - timedelta(hours=6), until=now) if row.name is not None]
    assert [row.kind_token for row in rows] == ["p-k0005"]


def test_d4_the_model_queue_has_no_vocabulary_category() -> None:
    from habitus.runtime import model_queue_state

    assert set(model_queue_state.CATEGORIES) == {"retry", "unsealed", "advice", "sync"}
    assert not hasattr(model_queue_state, "VOCABULARY")
