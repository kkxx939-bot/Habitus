"""夜批：五步按顺序跑一遍真链子（含守门），以及守门不过时停下来报。"""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta

import pytest

from habitus.model_client import ChatClient, ModelResponse
from habitus.model_client.structured import StructuredChatClient
from habitus.runtime.scene_coverage import TreeCoverage
from habitus.runtime.scene_night import SceneNightConfig, SceneNightError, SceneNightlyRun
from habitus.scene.calendar import NominalCalendar
from habitus.scene.concepts import ConceptSet, ConceptStore, ConceptVectorIndex, embedding_text
from habitus.scene.hypotheses import HypothesisStore
from habitus.scene.ledger import LedgerConfig, LedgerStore
from habitus.scene.occurrences import ConceptHit, ConceptHitStore
from habitus.scene.occurrences.mapper import ConceptMapper, MapperConfig
from habitus.scene.views import ViewsConfig, ViewsStore
from tests.unit.foresight.scripted_model import ScriptedProvider, model_config
from tests.unit.runtime.prediction_tree_fixtures import curve, tree
from tests.unit.scene.concept_fixtures import (
    ALL_CONCEPTS,
    BALL,
    BREAKFAST,
    EXERCISE,
    SLEEP_LATE,
    TableEmbedder,
)
from tests.unit.scene.fixtures import CST, DAY1, SUBJECT, Site, at, publish, publish_gap
from tests.unit.scene.ledger_fixtures import BEDTIME, LATE_TO_BREAKFAST, MAPPER, record

TONIGHT = DAY1 + timedelta(days=3)  # 2026-08-18，周二
NOW = at(TONIGHT + timedelta(days=1), 3, 0)
CONCEPTS = ConceptSet((*ALL_CONCEPTS, BEDTIME))
VECTORS = {
    embedding_text(SLEEP_LATE): (1.0, 0.0, 0.0, 0.0),
    embedding_text(BREAKFAST): (0.0, 1.0, 0.0, 0.0),
    embedding_text(EXERCISE): (0.0, 0.0, 1.0, 0.0),
    embedding_text(BALL): (0.0, 0.0, 0.9, 0.1),
    embedding_text(BEDTIME): (0.8, 0.0, 0.0, 0.2),
    "就寝\n上床睡觉": (0.9, 0.1, 0.0, 0.0),
    "吃了碗面\n坐下吃了碗面": (0.1, 0.9, 0.0, 0.0),
}


class AnswerByCandidates(ScriptedProvider):
    """按"哪条行为 / 哪个候选"作答：``yes`` 里写 ``行为名/概念名``，其余一律 ``no``。

    比"按调用次序回放"稳：一天里几条 occurrence 的顺序由行为树的目录枚举定，换个名字就变，
    而这个测试要验的是夜批的顺序，不是行为树的枚举顺序。按行为名分开答也更真实——一碗 07:30 的面
    比常态就寝晚 8 小时，机械规则会放它过，"它不是入睡"只有模型答得了。
    """

    def __init__(self, yes: frozenset[str]) -> None:
        super().__init__([])
        self.yes = yes

    async def complete_async(self, request):  # type: ignore[no-untyped-def, override]
        prompt = request.request.messages[-1].content or ""
        self.prompts.append(prompt)
        self.calls += 1
        behaviour = next(line.removeprefix("行为：") for line in prompt.splitlines() if line.startswith("行为："))
        section = prompt.split("## 候选概念")[-1]
        names = [line.split("：", 1)[0].removeprefix("- ") for line in section.splitlines() if line.startswith("- ")]
        body = {
            "verdicts": [{"concept": name, "verdict": "yes" if f"{behaviour}/{name}" in self.yes else "no"} for name in names]
        }
        return ModelResponse(content=json.dumps(body, ensure_ascii=False), model=self.model, provider=self.provider_name, finish_reason="stop")


def site_with_three_bedtimes(tmp_path) -> tuple[Site, ConceptHitStore]:
    """前三天各一条就寝（够常态的三个样本），直接写进命中盘——历史不必再调模型。"""

    site = Site(tmp_path, now=NOW)
    hits = ConceptHitStore(tmp_path / "scene")
    for offset, (hour, minute) in enumerate(((23, 30), (23, 40), (23, 20))):
        day = DAY1 + timedelta(days=offset)
        hits.write(record(day, "睡觉", hour, minute, ConceptHit("就寝")))
        hits.complete_day(day, records=1, completed_at=at(day, 23, 59), mapper=MAPPER)
    return site, hits


def nightly(tmp_path, hits: ConceptHitStore, site: Site, *, yes: frozenset[str]) -> tuple[SceneNightlyRun, AnswerByCandidates]:
    concepts = ConceptStore(tmp_path / "scene")
    for definition in (EXERCISE, BALL, SLEEP_LATE, BREAKFAST, BEDTIME, *[item for item in ALL_CONCEPTS if item.role.is_situation]):
        concepts.write(definition)
    hypotheses = HypothesisStore(tmp_path / "scene")
    hypotheses.write(LATE_TO_BREAKFAST, CONCEPTS)
    provider = AnswerByCandidates(yes)
    client = StructuredChatClient(ChatClient(model_config(), provider), validation_retries=1)

    async def mapper_for(concept_set: ConceptSet) -> ConceptMapper:
        index = ConceptVectorIndex("fake-embed", 4)
        for identity in concept_set.behavior_leaves():
            index = index.with_vector(concept_set[identity], VECTORS[embedding_text(concept_set[identity])])
        return ConceptMapper(
            client,
            TableEmbedder(VECTORS),
            concept_set,
            index,
            config=MapperConfig(recall_limit=2, transient_retry_delay_seconds=0.0),
            subject=SUBJECT,
            clock=lambda: NOW,
        )

    run = SceneNightlyRun(
        behavior_tree=site.behavior_tree,
        concepts=concepts,
        hypotheses=hypotheses,
        hits=hits,
        ledger=LedgerStore(tmp_path / "scene"),
        views=ViewsStore(tmp_path / "scene"),
        coverage=TreeCoverage(site.behavior_tree),
        calendar=NominalCalendar(),
        timezone=CST,
        mapper_for=mapper_for,
        subject=SUBJECT,
        config=SceneNightConfig(
            ledger=LedgerConfig(),
            views=ViewsConfig(min_count=3, min_blocks=2),
            placebos=0,  # 其余测试量的是五步本身；安慰剂那条自己打开
        ),
    )
    return run, provider


def prediction_tree():
    """早餐（kind 吃饭）每天 07:00–08:30 有个峰；就寝（kind 睡觉）夜里有个峰。"""

    curves = {}
    for weekday in range(7):
        curves[(weekday, "吃饭")] = curve({14: 0.2, 15: 0.5, 16: 0.1})
        curves[(weekday, "睡觉")] = curve({46: 0.4, 47: 0.5})
    return tree(curves)


def test_the_five_steps_run_in_order_on_a_real_chain(tmp_path) -> None:
    """就寝 02:10（比常态 23:30 晚 160 分钟 → 晚睡）→ 开一条「晚睡→早餐」的承诺 → 早餐来了就结 → 落盘。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    publish(site.behavior_tree, TONIGHT, "就寝", 2, 10, summary="上床睡觉", kind="睡觉", lasts_minutes=30)
    publish(site.behavior_tree, TONIGHT, "吃了碗面", 7, 30, summary="坐下吃了碗面", kind="吃饭")
    publish_gap(site.behavior_tree, TONIGHT, (12, 0), (13, 0))
    run, provider = nightly(tmp_path, hits, site, yes=frozenset({"就寝/晚睡", "吃了碗面/早餐"}))

    report = asyncio.run(run.run(TONIGHT, tree=prediction_tree(), now=NOW))
    assert report.mapped == 2 and report.unresolved == 0 and provider.calls == 2
    assert report.concepts == 7 and report.hypotheses == 1  # 5 个行为概念 + 出差中 + 周末
    assert report.opened == 1 and report.without_control == 0  # 机会口从树上取到了对照
    assert report.settled == 1 and report.pending == 0  # 早餐在第 1 次机会上来了
    assert report.relations == 1 and report.views_written >= 5
    assert report.generation.startswith("20260818T")
    assert "映射 2" in report.summary() and "开承诺 1" in report.summary()
    # 投影真的落了盘，而且完成标记在（半截的投影没有标记）
    views = tmp_path / "scene" / "views"
    assert ViewsStore(tmp_path / "scene").is_complete()
    assert (views / "relations" / f"{LATE_TO_BREAKFAST.identity}.md").exists()
    assert (views / "behaviours" / "晚睡.md").exists()


def test_a_missing_baseline_is_reported_and_the_candidate_goes_unresolved(tmp_path) -> None:
    """常态样本不够 → 那个键不给 → 引用它的「晚睡」记未决（不是"没命中"），而且夜批把这件事报出来。"""

    site = Site(tmp_path, now=NOW)
    hits = ConceptHitStore(tmp_path / "scene")  # 一条历史都没有
    publish(site.behavior_tree, TONIGHT, "就寝", 2, 10, summary="上床睡觉", kind="睡觉", lasts_minutes=30)
    run, _provider = nightly(tmp_path, hits, site, yes=frozenset({"就寝/晚睡"}))

    report = asyncio.run(run.run(TONIGHT, tree=prediction_tree(), now=NOW))
    assert report.mapped == 1 and report.unresolved == 1 and report.opened == 0
    assert any("就寝:usual_start:recent 样本不够" in signal for signal in report.signals)


def test_the_gate_stops_the_night_when_the_ledger_has_orphan_settlements(tmp_path) -> None:
    """结算在、承诺没了：继续跑会把一条没有承诺的结算算进分母，所以停下来报。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    run, _provider = nightly(tmp_path, hits, site, yes=frozenset({"就寝/晚睡"}))
    ledger = LedgerStore(tmp_path / "scene")
    publish(site.behavior_tree, TONIGHT, "就寝", 2, 10, summary="上床睡觉", kind="睡觉", lasts_minutes=30)
    # 先正常跑一夜开出承诺，再把承诺文件删掉，留下孤儿结算
    asyncio.run(run.run(TONIGHT, tree=prediction_tree(), now=NOW))
    publish(site.behavior_tree, TONIGHT + timedelta(days=1), "吃了碗面", 7, 30, summary="坐下吃了碗面", kind="吃饭")
    run_two, _ = nightly(tmp_path, hits, site, yes=frozenset({"吃了碗面/早餐"}))
    asyncio.run(run_two.run(TONIGHT + timedelta(days=1), tree=prediction_tree(), now=NOW + timedelta(days=1)))
    settlements = ledger.settlements_for(LATE_TO_BREAKFAST.identity)
    assert settlements, "先要有一条结算才谈得上孤儿"
    ledger.claim_path(settlements[0].ref).unlink()
    with pytest.raises(SceneNightError, match="settlements whose claims are gone"):
        asyncio.run(run_two.run(TONIGHT + timedelta(days=2), tree=prediction_tree(), now=NOW + timedelta(days=2)))


def test_the_nightly_run_tops_up_placebos_and_keeps_them_out_of_the_views(tmp_path) -> None:
    """B6 的前半：安慰剂由算法补齐、一样开账，但**不进投影**——它是尺子不是读数。"""

    from habitus.scene.hypotheses.placebo import split_by_origin

    site, hits = site_with_three_bedtimes(tmp_path)
    publish(site.behavior_tree, TONIGHT, "就寝", 2, 10, summary="上床睡觉", kind="睡觉", lasts_minutes=30)
    publish(site.behavior_tree, TONIGHT, "吃了碗面", 7, 30, summary="坐下吃了碗面", kind="吃饭")
    run, _provider = nightly(tmp_path, hits, site, yes=frozenset({"就寝/晚睡", "吃了碗面/早餐"}))
    object.__setattr__(run.config, "placebos", 1)

    report = asyncio.run(run.run(TONIGHT, tree=prediction_tree(), now=NOW))
    store = HypothesisStore(tmp_path / "scene")
    real, placebo = split_by_origin({item.identity: item for item in store.read_all()})
    assert len(placebo) == 1 and len(real) == 1  # 一条真假设配一条安慰剂
    assert any("新配" in signal for signal in report.signals)
    # 安慰剂进了账本（它要攒样本），但没进投影
    assert report.placebo is not None
    relations = list((tmp_path / "scene" / "views" / "relations").rglob("*.md"))
    assert len(relations) == report.relations == 1  # 只有真假设那一条
    assert not any(next(iter(placebo)).split("/")[-1] in path.name for path in relations)


def test_the_closure_step_is_skipped_when_no_touchpoint_is_injected(tmp_path) -> None:
    """闭环要调模型，而重放与离线读数不该被模型可用性卡住：没注入触点③ 就整拍跳过，前五步照跑。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    publish(site.behavior_tree, TONIGHT, "就寝", 2, 10, summary="上床睡觉", kind="睡觉", lasts_minutes=30)
    run, _provider = nightly(tmp_path, hits, site, yes=frozenset({"就寝/晚睡"}))
    report = asyncio.run(run.run(TONIGHT, tree=prediction_tree(), now=NOW))
    assert report.closure is None and run.closure is None
    assert "闭环" not in report.summary()


def test_r3_03_the_night_reads_relations_at_the_trust_boundary_not_untruncated(tmp_path) -> None:
    """夜批读数传 ``until``（命中记录可信到哪一刻）：承诺命运未定的要截掉。旧写法没传，对称截断在生产路径上关了三周（评审 A-3/B-3/C-3）。

    这一夜 02:10 晚睡开了承诺、07:30 早餐把它结掉（落在自己那个早餐窗里）。**对称截断**要等窗口过完才收它——不然
    "来了的当下就进 n、没来的要等窗口过完才进 n"把 p1 系统性读高（第二轮 C-4）。窗口账（2026-10-01）的收口点就是窗口末尾：
    到 ``until``（次日零点）窗口已过完 → 进 n；在窗口里（08:00）读 → 它进 ``immature``、不进 n。
    """

    site, hits = site_with_three_bedtimes(tmp_path)
    publish(site.behavior_tree, TONIGHT, "就寝", 2, 10, summary="上床睡觉", kind="睡觉", lasts_minutes=30)
    publish(site.behavior_tree, TONIGHT, "吃了碗面", 7, 30, summary="坐下吃了碗面", kind="吃饭")
    run, _provider = nightly(tmp_path, hits, site, yes=frozenset({"就寝/晚睡", "吃了碗面/早餐"}))
    seen: dict[str, object] = {}
    original = run._project

    def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        return original(*args, **kwargs)

    run._project = spy  # type: ignore[method-assign]
    report = asyncio.run(run.run(TONIGHT, tree=prediction_tree(), now=NOW))
    # 传给读数的是可信边界（最后一个已映射日的次日零点，主体时区），不是墙钟 NOW
    assert seen["until"] == at(TONIGHT + timedelta(days=1), 0, 0) and seen["now"] == NOW
    assert report.opened == 1 and report.settled == 1 and report.pending == 0
    from habitus.scene.views import read_relations

    (early,) = read_relations((LATE_TO_BREAKFAST,), ledger=run.ledger, concepts=run.concepts.read_all(), config=run.config.views, now=at(TONIGHT, 8, 0))
    assert early.strength.immature == 1 and early.strength.accumulation.count == 0  # 窗口还没过完：命运未定 → 截掉
    (reading,) = read_relations((LATE_TO_BREAKFAST,), ledger=run.ledger, concepts=run.concepts.read_all(), config=run.config.views, now=seen["until"])  # type: ignore[arg-type]
    assert reading.strength.immature == 0 and reading.strength.accumulation.count == 1 and reading.strength.events == 1  # 窗口过完了才进 n


def test_r3_18_remapping_a_day_voids_its_ledger_before_reopening(tmp_path) -> None:
    """概念集变了 → 同一天重映射 → 命中变了 → 那天的承诺与靶那天记录结的账先撤再开（评审 A-12 / B-5）。"""

    site, hits = site_with_three_bedtimes(tmp_path)
    publish(site.behavior_tree, TONIGHT, "就寝", 2, 10, summary="上床睡觉", kind="睡觉", lasts_minutes=30)
    publish(site.behavior_tree, TONIGHT, "吃了碗面", 7, 30, summary="坐下吃了碗面", kind="吃饭")
    run, _provider = nightly(tmp_path, hits, site, yes=frozenset({"就寝/晚睡", "吃了碗面/早餐"}))
    first = asyncio.run(run.run(TONIGHT, tree=prediction_tree(), now=NOW))
    assert first.opened == 1 and first.settled == 1
    assert len(run.ledger.claims_for(LATE_TO_BREAKFAST.identity)) == 1
    # 口径变了（概念集加一个概念 → 映射器版本变）→ 重跑同一天：模型这次不再判「晚睡」
    from tests.unit.scene.concept_fixtures import concept

    VECTORS["夜宵：睡前吃东西"] = (0.0, 0.5, 0.5, 0.0)
    run.concepts.write(concept("夜宵", "睡前吃东西"))
    run2, _p = nightly(tmp_path, hits, site, yes=frozenset({"吃了碗面/早餐"}))
    second = asyncio.run(run2.run(TONIGHT, tree=prediction_tree(), now=NOW + timedelta(hours=1)))
    assert second.mapped == 2 and any(signal.startswith("remap:") for signal in second.signals)
    # 旧的那条承诺（前件「晚睡」）被撤了，新的命中里没有晚睡 → 一条都不该开
    assert run2.ledger.claims_for(LATE_TO_BREAKFAST.identity) == () and second.opened == 0
