"""第二步乙：新增的四个配置（预期表 D4；共享说明 3.4）。旧代码上应当失败。

键名还没定：照第二步甲的做法，拿改动前 ``example.yaml`` 的原样拷贝（``baseline_example.yaml.txt``，HEAD ``17ebc64``）比出
新增的键，按示例值认：名字最长 6、每个上级候选位置 5、退出天数 20、总层数 3。前三个另外各改一个值，看它真的接进了主链。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from tests.unit.word_grow.site import HERE, WRITE_DOC, Grow, GrowSite, at, grown, names, node
from tests.unit.word_pick.config_keys import listed_keys
from tests.unit.word_pick.site import EXAMPLE_CONFIG, config

BASELINE = HERE / "baseline_example.yaml.txt"
NAME_CHARS, SLOTS, EXIT_DAYS, LEVELS = "6", "5", "20", "3"


def added() -> dict[tuple[str, str], str]:
    before = listed_keys(BASELINE)
    return {item: value for item, value in listed_keys(EXAMPLE_CONFIG).items() if item not in before}


def _key(value: str) -> tuple[str, str]:
    found = [item for item, listed in added().items() if listed == value]
    assert len(found) == 1, f"expected exactly one new example key whose value is {value}, found {found}"
    return found[0]


def _setting(value: str, to: int) -> Any:
    section, key = _key(value)

    def edit(raw: dict[str, Any]) -> None:
        raw.setdefault(section, {})[key] = to

    return edit


def test_d4_the_example_lists_exactly_four_new_keys_with_their_defaults() -> None:
    assert sorted(added().values(), key=int) == sorted([NAME_CHARS, SLOTS, EXIT_DAYS, LEVELS], key=int)


def test_d4_the_new_keys_load_at_their_example_values(tmp_path: Path) -> None:
    keys = {value: _key(value) for value in (NAME_CHARS, SLOTS, EXIT_DAYS, LEVELS)}

    def edit(raw: dict[str, Any]) -> None:
        for value, (section, key) in keys.items():
            raw.setdefault(section, {})[key] = int(value)

    config(tmp_path, edit=edit)


@pytest.fixture
def make(tmp_path: Path):  # type: ignore[no-untyped-def]
    built: list[GrowSite] = []

    def factory(value: str, to: int) -> GrowSite:
        site = GrowSite(tmp_path / f"site-{len(built)}", edit=_setting(value, to))
        built.append(site)
        return site

    yield factory
    for site in built:
        site.close()


D13 = date(2026, 4, 13)


def test_d4_the_name_length_limit_comes_from_the_config(make: Any) -> None:
    site = make(NAME_CHARS, 3)
    site.record("conv", at(D13, 9), "帮我写一份产品需求文档", Grow("写文档", node("写需求书")))
    assert site.token_at(at(D13, 9)) == WRITE_DOC
    site.record("conv", at(D13, 10), "帮我写一份规格", Grow("写文档", node("写规格")))
    assert site.token_at(at(D13, 10)) == grown(1)


def test_d4_the_candidate_places_come_from_the_config(make: Any) -> None:
    site = make(SLOTS, 2)
    for index, name in enumerate(("写需求书", "写设计书", "写测试书")):
        site.record(f"conv-{index}", at(D13, 9 + index), f"帮我{name}", Grow("写文档", node(name)))
    block = names(site.probe(D13))
    assert "写需求书" not in block and {"写设计书", "写测试书"} <= set(block)


def test_d4_the_exit_days_come_from_the_config(make: Any) -> None:
    site = make(EXIT_DAYS, 5)
    site.record("conv", at(D13, 9), "帮我写一份产品需求文档", Grow("写文档", node("写需求书")))
    site.generate(date(2026, 4, 17))
    assert "写需求书" in names(site.probe(date(2026, 4, 17)))
    site.generate(date(2026, 4, 18))
    assert "写需求书" not in names(site.probe(date(2026, 4, 18)))
