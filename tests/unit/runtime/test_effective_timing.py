"""生效时点（实时化阶段 2 · C+D，立项第 1、4–7 条；预期 CD-12、CD-13、CD-14）。

模型队列里"昨天及以前"的先验与候选条件的补问都在今天这一代完成之后才做；先验对着"概念集 + 同步词表将要写的概念"问，
新类的行为概念进概念集的那一代先验已经备好。用真的语义树、真的队列、照着提示词答的替身模型。

2026-11-14 概念层只留词表节点与日历之后没有"写概念"这一类：原来 CD-12 那一条（写出的概念待生效、下一代并入）随之删掉，
这里留先验那一半。
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

from habitus.model_client import ModelTransportError
from habitus.runtime import health as health_module
from habitus.runtime.generation_marks import GenerationMark, GenerationMarks
from habitus.runtime.model_queue import ADVICE, DONE, ModelQueueState, QueueSettings
from tests.series_helpers import read_series
from tests.unit.kind_ids import kind_id
from tests.unit.runtime.test_model_queue_acceptance import TODAY, chat_breaker, disk, done, queue_on, scene_on
from tests.unit.runtime.test_scene_night import NOW, publish_tonight, site_with_three_bedtimes

LUNCH = kind_id("午饭")
TOMORROW = TODAY + timedelta(days=1)


def world(root: Path):  # type: ignore[no-untyped-def]
    """三天就寝 + 今晚封口的两条；今天这一代的语义树一步已经做完（概念同步好、昨天盖了章）；之后词表新长出「午饭」这个类
    （它的行为概念下一代才进概念集）；接上先验。"""

    site, _hits = site_with_three_bedtimes(root)
    publish_tonight(site)
    scene, provider, advisor = scene_on(root, site, advice=True)
    asyncio.run(
        scene.generate(
            series=read_series(site.behavior_tree, cutoff=TODAY), words=scene.syncing.catalog.current(), now=NOW
        )
    )
    scene.syncing.catalog.active[LUNCH] = ("午饭", "中午坐下吃一顿")
    return site, scene, provider, advisor


def next_generation(site, scene) -> None:  # type: ignore[no-untyped-def]
    """明天那一代的语义树一步（没有模型调用）。"""

    asyncio.run(
        scene.generate(
            series=read_series(site.behavior_tree, cutoff=TOMORROW),
            words=scene.syncing.catalog.current(),
            now=NOW + timedelta(days=1),
        )
    )


def test_cd_12_a_new_class_has_its_prior_ready_when_it_enters_the_concept_set(tmp_path) -> None:
    """CD-12 的先验那一半：词表新长出的类当天还不在概念集里，先验当天就对着它问了；下一次换代它进概念集时先验已经齐了——
    换代折叠不缺先验、不调模型。撤回即失败：先验只对着盘上的概念集问，第一段的 ``asked`` 是空。"""

    site, scene, _provider, advisor = world(tmp_path)
    report = asyncio.run(queue_on(tmp_path, scene, site).drain())
    assert (ADVICE, DONE) in done(report)
    assert LUNCH not in scene.concepts.read_all()
    assert [prompt for prompt in advisor.prompts if "午饭" in prompt]  # 先验当天就对着它问了

    calls = advisor.calls
    next_generation(site, scene)
    concepts = scene.concepts.read_all()
    assert LUNCH in concepts
    lane = concepts.lane_of(LUNCH)
    assert lane is not None and scene.advice.priors(concepts, lane).missing() == 0  # type: ignore[union-attr]
    assert advisor.calls == calls  # 换代里没有模型调用


def test_cd_12_nothing_asked_before_todays_generation_is_done(tmp_path) -> None:
    """立项第 1 条：今天这一代还没做完时，补问不算欠着、不做；完成标记写上之后才做。"""

    site, scene, _provider, advisor = world(tmp_path)
    marks = GenerationMarks(tmp_path / "scene" / "generation", artifact_root=tmp_path / "scene")
    queue = queue_on(tmp_path, scene, site, marks=marks)
    before = asyncio.run(queue.drain())
    assert advisor.calls == 0 and before.owed[ADVICE] == 0
    marks.write(GenerationMark(day=TODAY, finished_at=NOW))
    after = asyncio.run(queue.drain())
    assert (ADVICE, DONE) in done(after) and advisor.calls > 0


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
    """CD-13：模型断了几小时、当天恢复——队列补做的先验的提示词与准时做的逐字相同；下一次换代之后的盘面与没断过的相同
    （除完成章上的盖章时刻）。"""

    from tests.unit.runtime.test_model_queue_acceptance import masked

    site, scene, _provider, advisor = world(tmp_path / "on_time")
    assert asyncio.run(queue_on(tmp_path / "on_time", scene, site).drain()).clear
    next_generation(site, scene)
    on_time_prompts = list(advisor.prompts)
    expected = masked(disk(tmp_path / "on_time"))

    root = tmp_path / "outage"
    site, scene, _provider, advisor = world(root)
    down = outage_on(advisor)
    clock = [NOW]
    queue = queue_on(root, scene, site, clock=lambda: clock[0], state=ModelQueueState(chat_breaker(scene, clock)))
    for hours in (0, 1, 2, 3):
        clock[0] = NOW + timedelta(hours=hours)
        report = asyncio.run(queue.drain())
        assert not report.clear
    down[0] = False
    clock[0] = NOW + timedelta(hours=6)
    assert asyncio.run(queue.drain()).clear
    next_generation(site, scene)
    assert list(advisor.prompts) == on_time_prompts
    assert masked(disk(root)) == expected


def test_cd_14_an_outage_across_a_generation_loses_nothing_is_reported_and_is_made_up_afterwards(tmp_path) -> None:
    """CD-14：模型断到跨了一次换代——欠着的补问一项不丢；健康面报"有活欠着跨过了换代"；恢复后自动补做。"""

    root = tmp_path
    site, scene, _provider, advisor = world(root)
    marks = GenerationMarks(root / "scene" / "generation", artifact_root=root / "scene")
    marks.write(GenerationMark(day=TODAY, finished_at=NOW))
    down = outage_on(advisor)
    clock = [NOW + timedelta(hours=1)]
    sealed = [TODAY]
    queue = queue_on(
        root,
        scene,
        site,
        clock=lambda: clock[0],
        sealed=lambda: sealed[0],
        marks=marks,
        state=ModelQueueState(chat_breaker(scene, clock)),
        settings=QueueSettings(warning_items=1, warning_wait_seconds=3_600.0),
    )
    first = asyncio.run(queue.drain())
    assert not first.clear and first.owed[ADVICE] >= 1

    # 明天那一代照做（换代不调模型）：补问还欠着
    next_generation(site, scene)
    marks.write(GenerationMark(day=TOMORROW, finished_at=NOW + timedelta(days=1)))
    clock[0], sealed[0] = NOW + timedelta(days=1, hours=1), TOMORROW
    asyncio.run(queue.drain())
    problems = dict(health_module.queue_problems(queue.state, queue.settings, clock[0]))
    assert "advice×1" in problems["model_queue_crossed"]

    down[0] = False
    clock[0] = NOW + timedelta(days=1, hours=3)
    report = asyncio.run(queue.drain())
    assert report.clear and (ADVICE, DONE) in done(report)
    assert dict(health_module.queue_problems(queue.state, queue.settings, clock[0])) == {}
