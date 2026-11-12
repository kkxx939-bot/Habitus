"""生效时点（实时化阶段 2 · C+D，立项第 1、4–7 条；预期 CD-12、CD-13、CD-14）。

模型队列里"昨天及以前"的写概念、先验与候选条件的补问都在今天这一代完成之后才做；写概念的结果待生效，下一次换代的语义树
一步并入概念集、这时定创建时刻；先验对着"概念集 + 将要生成的基础概念 + 待生效的概念"问，概念进概念集的那一代先验已经备好。
用真的语义树、真的队列、照着提示词答的替身模型。
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

from habitus.model_client import ModelTransportError
from habitus.runtime import health as health_module
from habitus.runtime.generation_marks import GenerationMark, GenerationMarks
from habitus.runtime.model_queue import ADVICE, CONCEPTS, DONE, QueueSettings
from habitus.series.reader import read_series
from tests.unit.runtime.test_model_queue_acceptance import TODAY, disk, done, queue_on, scene_on
from tests.unit.runtime.test_scene_night import NOW, publish_tonight, site_with_three_bedtimes

EARLY = "吃得早"
AUTHORED = {
    "concepts": [
        {
            "name": EARLY,
            "definition": "七点以前吃的早餐",
            "role": "behavior",
            "kind": "refinement",
            "classes": ["早餐"],
            "context": "occurrence",
            "baseline_keys": [],
            "rule": None,
            "grades": [],
            "situation": None,
            "why": "提醒句一样，只是时间早",
        }
    ]
}
TOMORROW = TODAY + timedelta(days=1)


def world(root: Path):  # type: ignore[no-untyped-def]
    """三天就寝 + 今晚封口的两条；今天这一代的语义树一步已经做完（概念同步好、昨天盖了章）；接上概念作者与先验。"""

    from habitus.runtime.scene_authoring import ConceptAuthoring
    from habitus.scene.concepts.author import AuthorConfig, ConceptAuthor
    from tests.unit.foresight.scripted_model import recording_client

    site, _hits = site_with_three_bedtimes(root)
    publish_tonight(site)
    scene, provider, advisor = scene_on(root, site, advice=True)
    asyncio.run(scene.generate(series=read_series(site.behavior_tree, cutoff=TODAY), now=NOW))
    client, author = recording_client([AUTHORED])
    scene.authoring = ConceptAuthoring(
        ConceptAuthor(client, config=AuthorConfig(transient_retry_delay_seconds=0.0), clock=lambda: NOW)
    )
    return site, scene, provider, advisor, author


def labels(scene) -> set[str]:  # type: ignore[no-untyped-def]
    concepts = scene.concepts.read_all()
    return {concepts.label_of(name) for name in concepts}


def next_generation(site, scene) -> None:  # type: ignore[no-untyped-def]
    """明天那一代的语义树一步（没有模型调用）。"""

    asyncio.run(scene.generate(series=read_series(site.behavior_tree, cutoff=TOMORROW), now=NOW + timedelta(days=1)))


def test_cd_12_an_authored_concept_waits_for_the_next_generation_with_its_prior_ready(tmp_path) -> None:
    """CD-12：队列白天写出的细分概念当天不在概念集里（存成待生效）；先验当天就对着它问了；下一次换代并入，创建时刻是那一代；
    并入时它的先验已经齐了——换代折叠不缺先验、不调模型。撤回即失败：写概念的结果直接进概念集，第一段的断言变红。"""

    site, scene, _provider, advisor, author = world(tmp_path)
    report = asyncio.run(queue_on(tmp_path, scene, site).drain())
    assert (CONCEPTS, DONE) in done(report) and author.calls == 1
    assert done(report).index((CONCEPTS, DONE)) < done(report).index((ADVICE, DONE))  # 先写概念、再问先验
    assert EARLY not in labels(scene)
    assert [item.label for item in scene.concepts.pending()] == [EARLY]
    asked = [prompt for prompt in advisor.prompts if EARLY in prompt]
    assert asked  # 先验当天就对着待生效的它问了

    calls = advisor.calls
    next_generation(site, scene)
    concepts = scene.concepts.read_all()
    merged = [name for name in concepts if concepts.label_of(name) == EARLY]
    assert merged and concepts[merged[0]].created_at == NOW + timedelta(days=1)
    assert scene.concepts.pending() == ()
    lane = concepts.lane_of(merged[0])
    assert lane is not None and scene.advice.priors(concepts, lane).missing() == 0  # type: ignore[union-attr]
    assert advisor.calls == calls  # 换代里没有模型调用


def test_cd_12_nothing_authored_or_asked_before_todays_generation_is_done(tmp_path) -> None:
    """立项第 1 条：今天这一代还没做完时，写概念与补问都不算欠着、不做；完成标记写上之后才做。"""

    site, scene, _provider, advisor, author = world(tmp_path)
    marks = GenerationMarks(tmp_path / "scene" / "generation", artifact_root=tmp_path / "scene")
    queue = queue_on(tmp_path, scene, site, marks=marks)
    before = asyncio.run(queue.drain())
    assert author.calls == 0 and advisor.calls == 0
    assert before.owed[CONCEPTS] == 0 and before.owed[ADVICE] == 0
    marks.write(GenerationMark(day=TODAY, finished_at=NOW))
    after = asyncio.run(queue.drain())
    assert (CONCEPTS, DONE) in done(after) and (ADVICE, DONE) in done(after) and author.calls == 1


def outage_on(*fakes):  # type: ignore[no-untyped-def]
    """让这几个替身模型一调就抛"调不通"；返回开关。"""

    down = [True]
    for fake in fakes:
        good = fake.complete_async

        async def answer(request, _fake=fake, _good=good):  # type: ignore[no-untyped-def]
            if down[0]:
                _fake.calls += 1
                raise ModelTransportError("connection refused")
            return await _good(request)

        fake.complete_async = answer
    return down


def test_cd_13_an_outage_of_a_few_hours_on_the_same_day_asks_the_same_prompts_and_leaves_the_same_disk(
    tmp_path,
) -> None:
    """CD-13：模型断了几小时、当天恢复——队列补做的写概念与先验的提示词与准时做的逐字相同；下一次换代之后的盘面与没断过的
    相同（除完成章上的盖章时刻）。"""

    from tests.unit.runtime.test_model_queue_acceptance import masked

    site, scene, _provider, advisor, author = world(tmp_path / "on_time")
    assert asyncio.run(queue_on(tmp_path / "on_time", scene, site).drain()).clear
    next_generation(site, scene)
    on_time_prompts = (list(author.prompts), list(advisor.prompts))
    expected = masked(disk(tmp_path / "on_time"))

    root = tmp_path / "outage"
    site, scene, provider, advisor, author = world(root)
    down = outage_on(provider, advisor, author)
    clock = [NOW]
    queue = queue_on(root, scene, site, clock=lambda: clock[0], settings=QueueSettings(unreachable_after_failures=1))
    for hours in (0, 1, 2, 3):
        clock[0] = NOW + timedelta(hours=hours)
        report = asyncio.run(queue.drain())
        assert not report.clear
    assert scene.concepts.pending() == ()  # 断着的时候什么都没写成
    down[0] = False
    clock[0] = NOW + timedelta(hours=6)
    assert asyncio.run(queue.drain()).clear
    next_generation(site, scene)
    assert (list(author.prompts), list(advisor.prompts)) == on_time_prompts
    assert masked(disk(root)) == expected


def test_cd_14_an_outage_across_a_generation_loses_nothing_is_reported_and_is_made_up_afterwards(tmp_path) -> None:
    """CD-14：模型断到跨了一次换代——欠着的写概念与补问一项不丢；健康面报"有活欠着跨过了换代"；恢复后自动补做，写出的概念
    待生效、再下一次换代并入（比没断过的晚一代——差异是预期之内的，不要求相同）。"""

    root = tmp_path
    site, scene, provider, advisor, author = world(root)
    marks = GenerationMarks(root / "scene" / "generation", artifact_root=root / "scene")
    marks.write(GenerationMark(day=TODAY, finished_at=NOW))
    down = outage_on(provider, advisor, author)
    clock = [NOW + timedelta(hours=1)]
    sealed = [TODAY]
    queue = queue_on(
        root,
        scene,
        site,
        clock=lambda: clock[0],
        sealed=lambda: sealed[0],
        marks=marks,
        settings=QueueSettings(unreachable_after_failures=1, warning_items=1, warning_wait_seconds=3_600.0),
    )
    first = asyncio.run(queue.drain())
    assert not first.clear and first.owed[CONCEPTS] == 1

    # 明天那一代照做（换代不调模型）：写概念还欠着，概念集里没有它
    next_generation(site, scene)
    marks.write(GenerationMark(day=TOMORROW, finished_at=NOW + timedelta(days=1)))
    clock[0], sealed[0] = NOW + timedelta(days=1, hours=1), TOMORROW
    asyncio.run(queue.drain())
    problems = dict(health_module.queue_problems(queue.state, queue.settings, clock[0]))
    assert "concepts×1" in problems["model_queue_crossed"]
    assert EARLY not in labels(scene)

    down[0] = False
    clock[0] = NOW + timedelta(days=1, hours=3)
    report = asyncio.run(queue.drain())
    assert report.clear and (CONCEPTS, DONE) in done(report) and author.calls >= 1
    assert [item.label for item in scene.concepts.pending()] == [EARLY]
    assert dict(health_module.queue_problems(queue.state, queue.settings, clock[0])) == {}
