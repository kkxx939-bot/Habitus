"""第二步甲：物理 lane 的选词留在归约那一步（共享说明 3.2；预期表 D1、D2、C20、C21）。

物理的判断在同步的写盘流程里产生、不带选词结果，归约给它选：一条记录一次调用，提示词是会话那份去掉 ``none``、``continue``、
【最近的记录】、【之前的几轮】，开头说明这是摄像头、手表看到的现实活动。全部经组合根的正门（``runtime.deliver_physical``）
与这一支的归约走。旧代码上应当失败：归约调的是白天归类（``behavior_kind_classify``），没有【词表】这一段。
"""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from habitus.behavior.model import BehaviorKind
from tests.unit.word_pick.site import CST, NONE, PHYSICAL_WORDS, Pick, Site, Word, vocabulary_block


def delivery(*behaviors: str, delivery_id: str = "delivery-pick") -> dict[str, object]:
    """手表三小时前的一份交付：每个行为一段，前后错开；整段都在看。时刻都早于封口视界，送到就能封口。"""

    base = (datetime.now(tz=CST) - timedelta(hours=3)).replace(microsecond=0)

    def at(seconds: int) -> str:
        return (base + timedelta(seconds=seconds)).isoformat()

    return {
        "source": "physical/s12",
        "device": "watch-01",
        "delivery_id": delivery_id,
        "sent_at": at(1_900),
        "segments": [
            {
                "id": f"seg-{index}",
                "behavior": name,
                "started_at": at(60 + 300 * index),
                "ended_at": at(200 + 300 * index),
            }
            for index, name in enumerate(behaviors)
        ],
        "coverage": [{"started_at": at(0), "ended_at": at(1_800), "state": "observing"}],
    }


@pytest.fixture
def site(tmp_path: Path):  # type: ignore[no-untyped-def]
    built = Site(tmp_path)
    yield built
    built.close()


def physical_tokens(site: Site) -> dict[str, str]:
    tree = site.physical.tree
    return {
        str(tree.read(address).fields["name"]): str(tree.read(address).fields["kind_token"])
        for address in tree.list_addresses(BehaviorKind.OCCURRENCE)
    }


def test_c20_d2_each_physical_record_is_picked_once_in_the_reduction(site: Site) -> None:
    site.router.picks.extend([Word("p-k0005"), Word("p-k0003")])
    asyncio.run(site.runtime.deliver_physical(delivery("换衣服", "起床")))
    assert site.router.count("pick") == 0  # 物理的判断不带选词结果
    site.reduce(site.physical)
    assert site.router.count("pick") == 2  # 一条记录一次调用
    assert site.router.count("classify") == 0
    assert physical_tokens(site) == {"换衣服": "p-k0005", "起床": "p-k0003"}


def test_c20_the_physical_prompt_has_no_none_continue_or_conversation_sections(site: Site) -> None:
    site.router.picks.append(Word("p-k0005"))
    asyncio.run(site.runtime.deliver_physical(delivery("换衣服")))
    site.reduce(site.physical)
    (text,) = site.router.texts("pick")
    assert '"none"' not in text and '"continue"' not in text
    assert "【最近的记录】" not in text and "【这场对话里之前的几轮】" not in text
    assert '"missing"' in text
    assert "摄像头" in text or "手表" in text  # 开头说明这是看到的现实活动


def test_c21_none_or_continue_are_not_valid_physical_answers_and_are_asked_again(site: Site) -> None:
    site.router.picks.extend([NONE, Pick("continue", 1), Word("p-k0005")])
    asyncio.run(site.runtime.deliver_physical(delivery("换衣服")))
    site.reduce(site.physical)
    assert site.router.count("pick") == 3
    assert physical_tokens(site) == {"换衣服": "p-k0005"}


def test_d1_the_physical_vocabulary_block_is_the_initial_223_word_table(site: Site) -> None:
    site.router.picks.append(Word("p-k0005"))
    asyncio.run(site.runtime.deliver_physical(delivery("换衣服")))
    site.reduce(site.physical)
    (text,) = site.router.texts("pick")
    block = vocabulary_block(text)
    assert block.unparsed == ()
    assert len(block.words) == 223
    levels = block.levels()
    assert Counter(levels.values()) == Counter({1: 92, 2: 119, 3: 12})
    ids = [word.id for word in block.words]
    assert len(set(ids)) == 223
    assert ids == [f"p-k{number:04d}" for number in range(1, 224)] or ids == [str(n) for n in range(1, 224)]
    seen: dict[str, int] = {}
    for word in block.words:
        level = levels[word.id]
        if level == 1:
            assert word.parent_id is None
        else:
            assert word.parent_id in seen and seen[word.parent_id] == level - 1
        seen[word.id] = level
    names = {word.id: word.name for word in block.words}
    for token, name in PHYSICAL_WORDS.items():
        number = token if token in names else str(int(token.removeprefix("p-k")))
        assert names[number] == name
