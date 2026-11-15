"""第二步甲：词的"在用"状态与换代（共享说明 3.1、3.3；预期表 D1 在用 0 个、D5、D6、D8b、C14–C18）。

一个词在两个不同的本地日被直接选中，下一次换代起在用；换代在读序列之后、语义树一步之前数一遍（纯计数、不调模型）；语义树
每个在用的词一个行为概念（身份 = 编号、显示名 = 词名、定义 = 词的定义），不在用的词没有概念。全部经组合根走：会话轮由
lane 描述 + 选词、归约落树，换代用组合根装好的 ``Generation.run``，概念从语义树的概念存储读。
旧代码上应当失败：没有选词，归约里的白天归类把这些轮归成「非事件」，语义树没有行为概念。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from habitus.scene.concepts.store import ConceptStore
from tests.unit.word_pick.config_keys import added_keys
from tests.unit.word_pick.site import EXAMPLE_CONFIG, Continue, Site, Word, at

D13, D14, D15, D16, D17 = (date(2026, 4, day) for day in (13, 14, 15, 16, 17))
OLD_STEPS = ("migrate", "calendar", "retry", "settle", "series", "scene", "opportunities", "days", "mark")


def behaviours(site: Site) -> dict[str, Any]:
    concepts = ConceptStore(site.runtime.config.scene_root).read_all()
    return {identity: concepts[identity] for identity in concepts.behaviors()}


@pytest.fixture
def site(tmp_path: Path):  # type: ignore[no-untyped-def]
    built = Site(tmp_path)
    yield built
    built.close()


def test_c15_d6_a_word_picked_on_two_different_days_is_in_use_from_the_next_generation(site: Site) -> None:
    site.record("conv-a", at(D13, 9), "帮我写一份本周周报", Word("s-k0002"))
    site.record("conv-b", at(D14, 9), "把这个季度的工作写个总结", Word("s-k0003"))
    site.record("conv-c", at(D15, 9), "这周的周报也帮我写一下", Word("s-k0002"))

    site.generate(D15)
    assert "s-k0002" not in behaviours(site)  # 04-15 当天的换代只看到 04-13 那一天

    site.generate(D16)
    concepts = behaviours(site)
    assert set(concepts) == {"s-k0002"}  # 行为概念 = 在用的词；只选过一天的「写总结」没有概念
    concept = concepts["s-k0002"]
    assert (concept.title, concept.definition, concept.lane) == (
        "写周报",
        "写周期性的工作汇报，含日报、月报",
        "session",
    )


def test_c14_two_picks_on_the_same_day_do_not_make_a_word_in_use(site: Site) -> None:
    site.record("conv-a", at(D13, 9), "帮我写一份本周周报", Word("s-k0002"))
    site.record("conv-a", at(D13, 15), "周报再改一版", Word("s-k0002"))
    # 对照：「写总结」两天各一次
    site.record("conv-b", at(D13, 10), "把这个季度的工作写个总结", Word("s-k0003"))
    site.record("conv-b", at(D14, 10), "再写一个项目总结", Word("s-k0003"))

    site.generate(D15)
    assert set(behaviours(site)) == {"s-k0003"}


def test_c16_a_continuation_on_another_day_does_not_count_as_picking_the_word(site: Site) -> None:
    site.record("conv-a", at(D13, 9), "帮我写一份项目方案", Word("s-k0004"))
    site.record("conv-a", at(D14, 9), "方案写得怎么样了", Continue(said="帮我写一份项目方案"))
    site.record("conv-b", at(D13, 10), "把这个季度的工作写个总结", Word("s-k0003"))
    site.record("conv-b", at(D14, 10), "再写一个项目总结", Word("s-k0003"))

    site.generate(D15)
    assert set(behaviours(site)) == {"s-k0003"}


def test_c17_a_second_level_word_is_in_use_without_its_parent(site: Site) -> None:
    site.record("conv-a", at(D13, 9), "帮我写一份本周周报", Word("s-k0002"))
    site.record("conv-a", at(D14, 9), "这周的周报也帮我写一下", Word("s-k0002"))

    site.generate(D15)
    concepts = behaviours(site)
    assert "s-k0002" in concepts and "s-k0001" not in concepts


def _cap_two(raw: dict[str, Any]) -> None:
    """在用上限设成 2：example.yaml 里新增的、默认值是 100 的那个键（"在用上限 100"，键名由实现定）。"""

    capped = [item for item, value in added_keys(EXAMPLE_CONFIG).items() if value == "100"]
    assert len(capped) == 1, f"expected exactly one new key whose example value is 100, found {capped}"
    ((section, key),) = capped
    raw.setdefault(section, {})[key] = 2


def test_c18_with_a_cap_of_two_the_first_two_to_qualify_are_in_use_and_the_third_waits(tmp_path: Path) -> None:
    site = Site(tmp_path, edit=_cap_two)
    try:
        # 够条件的先后按"第二个日子"：写论文 04-14、写报告 04-15、写总结 04-16；编号顺序正好反过来
        site.record("conv-a", at(D13, 9), "帮我写一篇投稿论文", Word("s-k0006"))
        site.record("conv-a", at(D14, 9), "论文再写一节", Word("s-k0006"))
        site.record("conv-b", at(D13, 10), "帮我写一份调研报告", Word("s-k0005"))
        site.record("conv-b", at(D15, 10), "再写一份测试报告", Word("s-k0005"))
        site.record("conv-c", at(D14, 11), "把这个季度的工作写个总结", Word("s-k0003"))
        site.record("conv-c", at(D16, 11), "再写一个项目总结", Word("s-k0003"))

        site.generate(D17)
        assert set(behaviours(site)) == {"s-k0006", "s-k0005"}
    finally:
        site.close()


def test_d5_the_generation_counts_words_in_use_after_the_series_and_before_the_scene_and_never_migrates(
    site: Site,
) -> None:
    site.record("conv-a", at(D13, 9), "帮我写一份本周周报", Word("s-k0002"))
    site.record("conv-a", at(D14, 9), "这周的周报也帮我写一下", Word("s-k0002"))
    calls = len(site.router.calls)
    result = site.generate(D15)
    names = [step.step for step in result.steps]
    assert "migrate" not in names
    counting = names[names.index("series") + 1]
    assert counting not in OLD_STEPS and names[names.index(counting) + 1] == "scene"
    assert {step.step: step.outcome for step in result.steps}[counting] == "done"
    asked = [call.kind for call in site.router.calls[calls:]]
    assert "pick" not in asked and "classify" not in asked  # 纯计数，不调模型


def test_d8b_the_vocabulary_version_is_the_in_use_revision_and_stays_when_the_set_does_not_change(site: Site) -> None:
    site.record("conv-a", at(D13, 9), "帮我写一份本周周报", Word("s-k0002"))
    site.record("conv-a", at(D14, 9), "这周的周报也帮我写一下", Word("s-k0002"))
    generation = site.runtime.components.generation.generation
    concepts = ConceptStore(site.runtime.config.scene_root)

    site.generate(D15)
    assert set(behaviours(site)) == {"s-k0002"}
    assert concepts.synced_version() == 1  # 在用集合从空变成一个词：修订号 0 → 1
    first = generation.marks.read(D15)
    assert first is not None and first.vocabulary_version == 1

    site.generate(D16)  # 中间没有新选的词：在用集合没变
    assert concepts.synced_version() == 1
    second = generation.marks.read(D16)
    assert second is not None and second.vocabulary_version == 1


def test_d1_a_fresh_disk_has_no_word_in_use(site: Site) -> None:
    """全新盘面在用的词 0 个：只选过一天的词换代后也没有概念（旧代码上同样没有行为概念，这一条在旧代码上也过）。"""

    site.record("conv-a", at(D13, 9), "帮我写一份本周周报", Word("s-k0002"))
    site.generate(D14)
    assert behaviours(site) == {}
