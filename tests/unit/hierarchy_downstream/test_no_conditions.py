"""丙一删"候选条件"那次模型调用（预期表 D2、D3、D5、D6 与场景 C28、C29；共享说明 3.7）。

C28 / D2 / D3 用基线数据（``site.plant_baseline``）：第一代语义树 → 模型队列做到没有欠账（``site.drain``，替身照着提示词答：
先验都答 unsure、候选条件提名单里第一个类）→ 第二代。C29 走组合根装好的整套 Runtime（``tests.unit.word_grow.site.GrowSite``）：
启动、换代、模型队列，哪一步报出来都算。改动前（``780e298``）都有候选条件这一问与提议簿，这几条在旧代码上应当失败。
"""

from __future__ import annotations

import asyncio
import re
from datetime import date
from pathlib import Path

import pytest

from habitus.config import ConfigError, HabitusConfig
from tests.unit.hierarchy_downstream.site import BASELINE_NIGHTS, answering_world, drain, plant_baseline
from tests.unit.hierarchy_downstream.test_unchanged import C30_CONDITIONS
from tests.unit.runtime.concept_layer_site import SUBJECT, example_mapping
from tests.unit.word_grow.site import GrowSite, at
from tests.unit.word_pick.site import Word

REPOSITORY = Path(__file__).resolve().parents[3]
CONDITION_CALL = "scene_relation_conditions"
#: 改动前的代码在基线数据上写下的那份提议簿（``advice/conditions/session.json``）。
OLD_BOOK = (
    '{"asked":{"s-k0002":"22aae31308c483a7f886679f00df161cb383896ffa6c0aadb88400b478f5a0f4"},"proposals":'
    '[{"antecedent":"s-k0002","condition":"timeline:s-k0004","model":"fake-1","proposed_on":"2026-04-12","raw":{},'
    '"refused":null,"version":"scene_relation_conditions_prompt_v4+promptb8e3817a+schemac3c2051e9668+llm:fake-1",'
    '"why":"之前做过"}]}'
)


def test_c28_d2_d3_a_generation_and_the_model_queue_never_ask_for_conditions_nor_write_a_proposal_book(
    tmp_path: Path,
) -> None:
    world, provider = answering_world(tmp_path)
    plant_baseline(world)
    world.generate(BASELINE_NIGHTS[0])
    report = drain(world, BASELINE_NIGHTS[0])
    world.generate(BASELINE_NIGHTS[1])
    asked = [item["name"] for item in provider.requests]
    kinds = [item.detail.get("kind") for item in report.done]
    book = sorted(path.name for path in (world.config.scene_root / "advice" / "conditions").glob("*"))
    assert (CONDITION_CALL in asked, "conditions" in kinds, book) == (False, False, [])
    assert set(kinds) == {"priors"}  # advice 一类只剩先验
    state = world.scene.relations.read("session", BASELINE_NIGHTS[1])
    assert sorted({relation.key.condition for relation in state.relations}) == C30_CONDITIONS


def test_c29_an_old_proposal_book_on_disk_is_refused_naming_the_file(tmp_path: Path) -> None:
    book = tmp_path / "data" / "scene" / "advice" / "conditions" / "session.json"
    book.parent.mkdir(parents=True)
    book.write_text(OLD_BOOK, encoding="utf-8")
    with pytest.raises(Exception, match=r"conditions[/\\]session\.json"):  # noqa: PT011 - 报错类型由实现定，只要点出文件
        site = GrowSite(tmp_path)
        try:
            site.record("conv-w", at(date(2026, 4, 13), 9), "帮我写一份本周周报", Word("s-k0002"))
            site.record("conv-w", at(date(2026, 4, 14), 9), "再写一份周报", Word("s-k0002"))
            site.generate(date(2026, 4, 15))
            queue = site.runtime.components.model_queue
            assert queue is not None
            asyncio.run(queue.drain())
        finally:
            site.close()


def test_d5_the_condition_question_setting_is_refused_by_the_config(tmp_path: Path) -> None:
    payload = example_mapping(tmp_path)
    payload["behavior"] = {"primary_subject": SUBJECT}
    payload["scene"] = {"enabled": True, "relations": {"condition_ask_min_antecedents": 40}}
    with pytest.raises(ConfigError, match="condition_ask_min_antecedents"):
        HabitusConfig.from_mapping(payload)


def test_d6_the_proposal_names_are_gone_from_the_package() -> None:
    pattern = re.compile(r"\bPROPOSABLE\b|\bcompile_proposal\b|\bcandidates\.proposed\b")
    found = sorted(
        f"{path.relative_to(REPOSITORY).as_posix()}:{number}"
        for path in (REPOSITORY / "habitus").rglob("*.py")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if pattern.search(line)
    )
    assert found == []
