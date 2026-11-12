"""先验混合问法的归题（实时化阶段 2 · C+D，立项第 8、9、12 条；预期 CD-9、CD-10）。

一对（前因, 后果）只归一题，按两个概念谁先出现分：后果比前因晚出现 → 旧问法（固定后果、列前因）；否则 → 新问法（固定前因、
列后果）。还没进概念集的（将要生成的基础概念、待生效的概念）算比所有已有的都晚、彼此同一批。模型用照着提示词答的替身。
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import replace
from datetime import date, timedelta

from habitus.runtime.scene_advice import RelationAdvice
from habitus.scene.advice.pairing import assign, consequent_is_later
from habitus.scene.advice.prior import PriorAdvisor, PriorConfig
from habitus.scene.advice.store import AdviceStore
from habitus.scene.concepts import ConceptSet
from tests.unit.scene.advice_fixtures import answering_client
from tests.unit.scene.concept_fixtures import CREATED, base
from tests.unit.scene.relation_fixtures import CONCEPTS, TITLES

LANE = "session"
OLD = frozenset(CONCEPTS)


def advice_on(root):  # type: ignore[no-untyped-def]
    client, provider = answering_client()
    advice = RelationAdvice(
        store=AdviceStore(root / "scene"),
        prior=PriorAdvisor(client, config=PriorConfig(transient_retry_delay_seconds=0.0)),
        conditions=None,
    )
    return advice, provider


def old_questions(prompts: list[str]) -> list[str]:
    """旧问法（固定后果）那几条提示词的后果名。"""

    return [prompt.split("## 后果\n")[1].split("：")[0] for prompt in prompts if prompt.startswith("## 后果\n")]


def new_questions(prompts: list[str]) -> list[str]:
    """新问法（固定前因）那几条提示词的前因名。"""

    return [prompt.split("## 前因\n")[1].split("：")[0] for prompt in prompts if prompt.startswith("## 前因\n")]


def listed(prompt: str) -> set[str]:
    return {line[2:].split("：")[0] for line in prompt.splitlines() if line.startswith("- ")}


# ── 纯规则 ──────────────────────────────────────────────────────────────


def test_a_pair_goes_to_the_later_consequent_otherwise_to_the_antecedent() -> None:
    """后果晚出现 → 按后果；前因晚出现、同一天、或两个都还没进概念集 → 按前因；只有后果还没进概念集 → 按后果。"""

    early, late = date(2026, 9, 1), date(2026, 9, 2)
    assert consequent_is_later(early, late) and consequent_is_later(early, None)
    assert not consequent_is_later(late, early) and not consequent_is_later(early, early)
    assert not consequent_is_later(None, early) and not consequent_is_later(None, None)


def test_cd_9_a_new_concept_brings_exactly_two_questions() -> None:
    """CD-9：已有的概念之间都有答案，来了一个还没进概念集的新概念——恰好两题：它当前因的新问法一题，列出现不晚于它的后果
    （全部已有的）；它当后果的旧问法一题，列比它早出现的前因（全部已有的）。每一对只归一题。"""

    names = ("a", "b", "c", "new")
    seen = {"a": date(2026, 9, 1), "b": date(2026, 9, 1), "c": date(2026, 9, 3), "new": None}
    questions = assign(
        names,
        comparable=lambda x, y: x != y,
        seen=seen.__getitem__,
        answered=lambda x, y: "new" not in (x, y),
    )
    assert questions.count == 2
    assert questions.by_consequent == {"new": ("a", "b", "c")}
    assert questions.by_antecedent == {"new": ("a", "b", "c")}


def test_cd_9_two_new_concepts_of_one_batch_ask_each_other_the_new_way() -> None:
    """CD-9 后半：同一批的两个新概念之间取新问法——各自那道新问法的题里列着对方，旧问法的题里不列。"""

    names = ("a", "x", "y")
    seen = {"a": date(2026, 9, 1), "x": None, "y": None}
    questions = assign(names, comparable=lambda p, q: p != q, seen=seen.__getitem__, answered=lambda p, q: False)
    assert questions.by_antecedent == {"x": ("a", "y"), "y": ("a", "x")}
    assert questions.by_consequent == {"x": ("a",), "y": ("a",)}
    pairs = [(c, a) for c, found in questions.by_consequent.items() for a in found]
    pairs += [(c, a) for a, found in questions.by_antecedent.items() for c in found]
    assert len(pairs) == len(set(pairs)) == 6  # 每一对只归一题：3 个概念两两有序 6 对


# ── 经运行时的先验（两份缓存、真的提示词）──────────────────────────────


def test_cd_9_the_queue_asks_a_new_concept_twice_three_rounds_each_and_caches_both_halves(tmp_path) -> None:
    """CD-9 走 ``RelationAdvice.ask_priors``：已有五个概念的先验都答过，再给一个还没进概念集的新概念（``settled`` 不含它）——
    只问两题、各三遍乱序：旧问法那题的后果是它、列五个前因；新问法那题的前因是它、列五个后果。之后两份缓存都齐，不再问。"""

    advice, provider = advice_on(tmp_path)
    asyncio.run(advice.ask_priors(CONCEPTS, LANE, [], OLD))
    before = len(provider.prompts)
    grown = ConceptSet([*CONCEPTS.values(), base("提问咨询")])
    assert advice.priors_owed(grown, LANE, OLD) == 2
    asyncio.run(advice.ask_priors(grown, LANE, [], OLD))
    later = provider.prompts[before:]
    assert len(later) == 6
    assert Counter(old_questions(later)) == {"提问咨询": 3} and Counter(new_questions(later)) == {"提问咨询": 3}
    assert all(listed(prompt) == set(TITLES) for prompt in later)
    assert advice.priors_owed(grown, LANE, OLD) == 0
    assert len(advice.priors(grown, LANE, OLD).levels()) == len(TITLES) * (len(TITLES) + 1)


def test_cd_10_asking_the_same_new_concept_a_day_later_gives_the_same_prompts(tmp_path) -> None:
    """CD-10：同一个新概念，当天（还待生效，不在概念集里）问与隔一天（已由换代并入、创建时刻是那一代）再问——两题的提示词
    逐字相同：清单只由出现先后决定，不看队列做到哪。"""

    def asked(root, concepts: ConceptSet, settled: frozenset[str]) -> list[str]:  # type: ignore[no-untyped-def]
        advice, provider = advice_on(root)
        asyncio.run(advice.ask_priors(CONCEPTS, LANE, [], OLD))
        before = len(provider.prompts)
        asyncio.run(advice.ask_priors(concepts, LANE, [], settled))
        return provider.prompts[before:]

    new = base("提问咨询")
    pending = ConceptSet([*CONCEPTS.values(), new])
    same_day = asked(tmp_path / "same_day", pending, OLD)
    merged = replace(new, created_at=CREATED + timedelta(days=1))  # 下一次换代并入时才定创建时刻
    settled = ConceptSet([*CONCEPTS.values(), merged])
    next_day = asked(tmp_path / "next_day", settled, frozenset(settled))
    assert len(same_day) == 6 and sorted(same_day) == sorted(next_day)


def test_cd_10_a_concept_that_arrives_later_does_not_reshape_the_earlier_ones_questions(tmp_path) -> None:
    """CD-10 的另一面：新概念 A 的题还没问，又来了更晚的 B（A 已并入，B 还待生效）——A 的两题清单与 B 没来时相同（A→B、B→A
    都归 B 的那两题），B 只多它自己的两题。"""

    first = base("提问咨询")
    later = base("整理笔记")
    merged = replace(first, created_at=CREATED + timedelta(days=1))
    alone = ConceptSet([*CONCEPTS.values(), merged])
    both = ConceptSet([*CONCEPTS.values(), merged, later])

    advice, provider = advice_on(tmp_path / "alone")
    asyncio.run(advice.ask_priors(CONCEPTS, LANE, [], OLD))
    before = len(provider.prompts)
    asyncio.run(advice.ask_priors(alone, LANE, [], frozenset(alone)))
    without_b = provider.prompts[before:]

    advice, provider = advice_on(tmp_path / "both")
    asyncio.run(advice.ask_priors(CONCEPTS, LANE, [], OLD))
    before = len(provider.prompts)
    asyncio.run(advice.ask_priors(both, LANE, [], frozenset(alone)))
    with_b = provider.prompts[before:]

    about_a = [prompt for prompt in with_b if prompt.split("\n")[1].startswith("提问咨询：")]
    assert sorted(about_a) == sorted(without_b)
    assert Counter(old_questions(with_b))["整理笔记"] == 3 and Counter(new_questions(with_b))["整理笔记"] == 3
    assert len(with_b) == 12


def test_cd_12_rule_a_renamed_concept_is_another_concept_and_is_asked_again(tmp_path) -> None:
    """立项第 12 条：概念改了判据算另一个概念——涉及它的对子按出现先后重新归题、重问；没改的那些对子照用缓存。"""

    advice, provider = advice_on(tmp_path)
    asyncio.run(advice.ask_priors(CONCEPTS, LANE, [], OLD))
    before = len(provider.prompts)
    target = next(name for name in CONCEPTS if CONCEPTS.label_of(name) == "调研")
    changed = ConceptSet(
        [
            replace(CONCEPTS[target], definition="查资料、读文档") if name == target else CONCEPTS[name]
            for name in CONCEPTS
        ]
    )
    assert advice.priors_owed(changed, LANE, OLD) >= 1
    asyncio.run(advice.ask_priors(changed, LANE, [], OLD))
    later = provider.prompts[before:]
    assert later and all("调研" in prompt for prompt in later)
    assert advice.priors_owed(changed, LANE, OLD) == 0
