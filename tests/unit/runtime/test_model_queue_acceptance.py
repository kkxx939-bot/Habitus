"""模型队列的验收（实时化阶段 2 · A1；测试工程师写，独立于实现者的 ``test_model_queue.py``）。

对着 ``立项-第二部分.md`` 第 4 节的预期表，一行至少一条（A1-5 … A1-12），用确定性替身模型；另加三处回放证明不了、要单元测试
守着"换代不调它们"的：探模型、迁移牵到的重映射、重试收尾；以及对抗（坏的重试项、口径不对的命中、不规范的先验缓存、两个队列
挂同一个根、一项做到一半被取消）。

每条的文档串写：预期出处、这条测试撤掉哪一处会变红。
"""

from __future__ import annotations

import asyncio
import inspect
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from habitus.behavior.retry import (
    Attempt,
    Reduction,
    RetryBranch,
    RetryCause,
    RetryItem,
    RetryKind,
    RetryStage,
    retry_id,
)
from habitus.behavior.retry.store import RetryBacklogStore
from habitus.behavior.tree.writer import BehaviorDocumentWriter
from habitus.behavior.uri import BehaviorURI
from habitus.config import ConfigError
from habitus.config.scene import SceneConfig
from habitus.foundation.lifecycle import BackoffPolicy
from habitus.infrastructure.lifecycle import RetryLedger
from habitus.model_client import ModelResponse, ModelTransportError
from habitus.runtime import health as health_module
from habitus.runtime.foresight import build_foresight_nightly
from habitus.runtime.foresight_days import DayReport
from habitus.runtime.foresight_forming import Formed
from habitus.runtime.generation import Generation
from habitus.runtime.generation_lease import Lease
from habitus.runtime.generation_marks import GenerationMark, GenerationMarks
from habitus.runtime.generation_result import (
    DAYS,
    DEGRADED,
    FALLBACK_RELATIONS,
    FALLBACK_UNMAPPED,
    MARK,
    OPPORTUNITIES,
    SCENE,
    STEPS,
)
from habitus.runtime.generation_result import (
    DONE as STEP_DONE,
)
from habitus.runtime.generation_result import (
    RETRY as RETRY_STEP,
)
from habitus.runtime.generation_trace import TraceFiles
from habitus.runtime.model_queue import (
    ADVICE,
    DONE,
    MAPPING,
    RETRY,
    SYNC,
    UNFINISHED,
    UNREACHABLE,
    ModelQueue,
    QueueJournal,
    QueueSettings,
)
from habitus.runtime.model_queue_worker import ModelQueueWorker
from habitus.runtime.scene_advice import RelationAdvice
from habitus.scene.advice.prior import PriorAdvisor, PriorConfig
from habitus.scene.advice.store import AdviceStore
from habitus.scene.occurrences import ConceptHitStore
from habitus.series.reader import SeriesReader, read_series
from tests.lock_store import ProcessLocalLockStore
from tests.unit.runtime.test_foresight_nightly import components as foresight_components
from tests.unit.runtime.test_generation import StubSettlement, assembled_with, generate, sealed_through
from tests.unit.runtime.test_scene_night import (
    BREAKFAST,
    MEAL,
    NOW,
    TONIGHT,
    FakeCatalog,
    nightly,
    publish_tonight,
    site_with_three_bedtimes,
)
from tests.unit.scene.advice_fixtures import answering_client
from tests.unit.scene.fixtures import CST

TODAY = NOW.date()
YES = frozenset({"吃了碗面/一起床就吃"})


# ── 现场 ────────────────────────────────────────────────────────────────


def scene_on(root: Path, site, *, yes=YES, catalog=None, advice: bool = False):  # type: ignore[no-untyped-def]
    """在 ``root`` 这份盘上重新开一套语义树对象（存储、映射器、替身模型都是新的）：相当于进程重起后看到的那一份。"""

    scene, provider = nightly(root, ConceptHitStore(root / "scene"), site, yes=yes, catalog=catalog)
    advisor = None
    if advice:
        client, advisor = answering_client()
        scene.advice = RelationAdvice(
            store=AdviceStore(root / "scene"),
            prior=PriorAdvisor(client, config=PriorConfig(transient_retry_delay_seconds=0.0)),
            conditions=None,
        )
    return scene, provider, advisor


def fresh(root: Path, **options):  # type: ignore[no-untyped-def]
    """三天就寝的历史 + 今晚封口的两条记录；返回（树的现场, 语义树, 映射的替身模型, 先验的替身模型）。"""

    site, _hits = site_with_three_bedtimes(root)
    publish_tonight(site)
    scene, provider, advisor = scene_on(root, site, **options)
    return site, scene, provider, advisor


def queue_on(root: Path, scene, site, *, clock=lambda: NOW, sealed=lambda: TODAY, **options) -> ModelQueue:  # type: ignore[no-untyped-def]
    """一个全新的队列对象（没有任何内存状态）；重试账、流水都落在 ``root`` 这份盘上。"""

    values = dict(
        scene=scene,
        series=SeriesReader(site.behavior_tree, name="model_queue"),
        today=lambda: clock().astimezone(CST).date(),
        sealed_day=sealed,
        vocabulary=lambda: (1, False),
        retries=RetryLedger(
            root / "scene",
            owner="model-queue",
            policy=BackoffPolicy(base_seconds=60.0, cap_seconds=3600.0),
            max_attempts=3,
            clock=clock,
        ),
        lease=Lease(ProcessLocalLockStore(), "scene:model-queue:acceptance", ttl_seconds=60.0, heartbeat_seconds=20.0),
        journal=QueueJournal(root / "scene" / "model_queue", artifact_root=root / "scene", zone=CST),
        clock=clock,
    )
    values.update(options)
    return ModelQueue(**values)  # type: ignore[arg-type]


def disk(root: Path) -> dict[str, bytes]:
    """语义树的产物（概念、命中与完成章、先验缓存与提议簿、关系表）；不含队列自己的流水、换代自己的账与各本重试账。"""

    found = {}
    for path in sorted((root / "scene").rglob("*")):
        relative = path.relative_to(root / "scene").as_posix()
        if not path.is_file() or relative.startswith(("model_queue/", "generation/", "lifecycle/")):
            continue
        if "retry" in relative:
            continue
        found[relative] = path.read_bytes()
    return found


def done(report) -> list[tuple[str, str]]:  # type: ignore[no-untyped-def]
    return [(item.category, item.outcome) for item in report.done]


def mapped(scene, day) -> bool:  # type: ignore[no-untyped-def]
    """这一天序列里的每条记录都映射好了、没有模型没答成的。映射按槽（阶段 2 第三小步）之后队列只映射、不盖日完成章——章由
    换代封口时盖——所以原来这份文件里"队列做完 = 那一天盖了章"的断言都改成读这个。"""

    from habitus.scene.occurrences.model import UnresolvedReason

    series = read_series(scene.behavior_tree, cutoff=day + timedelta(days=1))
    found = scene.hits.read_day(day)
    failed = [
        reason for item in found for reason in item.unresolved.values() if reason is UnresolvedReason.MODEL_FAILED
    ]
    return len(found) == len(series.on(day)) and not failed


def seal(scene) -> None:  # type: ignore[no-untyped-def]
    """换代的语义树一步里封口的那一段：重算情境、核条数、盖日完成章（映射按槽之后章只在这里盖）。"""

    series = read_series(scene.behavior_tree, cutoff=TODAY)
    concepts = scene.concepts.read_all()
    scene.sealing.seal(series, concepts, scene.mapper_for(concepts), NOW, [])


def others(report) -> list[tuple[str, str]]:  # type: ignore[no-untyped-def]
    """这一趟里映射之外的项。映射按槽之后映射不等封口：原来"封口之前队列只做出错重试"的断言里，映射那几项另算。"""

    return [item for item in done(report) if item[0] != MAPPING]


def straight(root: Path, **options) -> dict[str, bytes]:  # type: ignore[no-untyped-def]
    """不中断的那一份：队列做到没有欠账后的盘面。"""

    site, scene, _provider, _advisor = fresh(root, **options)
    report = asyncio.run(queue_on(root, scene, site).drain())
    assert report.clear
    return disk(root)


class StubDays:
    def __init__(self) -> None:
        self.ran = 0

    async def run_once(self):  # type: ignore[no-untyped-def]
        self.ran += 1
        return DayReport(datetime.now().astimezone(), Formed())


def generation_on(root: Path, site, scene, **options) -> tuple[Generation, StubDays, object]:  # type: ignore[no-untyped-def]
    """真的语义树 + 真的机会表（预测层的那一拍）+ 日子预期的替身，装成一个换代。"""

    built = foresight_components(root, site)
    foresight = build_foresight_nightly(
        built, relations=scene.relations, relation_config=scene.relation_config, clock=lambda: NOW
    )
    run = assembled_with(root, site, scene, foresight=foresight, **options)
    days = StubDays()
    run.days = days  # type: ignore[assignment]
    return run, days, built


def explode(provider) -> list[str]:  # type: ignore[no-untyped-def]
    """让这个替身模型一调就抛，返回记调用的清单。"""

    attempts: list[str] = []

    async def exploding(request):  # type: ignore[no-untyped-def]
        attempts.append("call")
        raise RuntimeError("the model must not be called here")

    provider.complete_async = exploding
    return attempts


# ── A1-5：换代时模型客户端一调就抛 ──────────────────────────────────────


def test_a1_5_a_generation_writes_the_table_the_day_expectations_and_the_mark_while_the_model_throws(tmp_path) -> None:
    """A1-5（共享说明第 3 节"换代那一刻模型客户端一调就抛：这一代照样写出完成标记"）：队列一项都没做、模型一调就抛，这一代
    仍 complete——机会表、日子预期、完成标记都写出，模型一次都没被调到；没映射的记录计进 ``unmapped_records``。

    撤回即失败：把语义树那一步里任一处模型调用放回去（映射、写概念、补问），``attempts`` 不再是空。"""

    site, scene, provider, advisor = fresh(tmp_path, advice=True)
    attempts = explode(provider)
    advisor_attempts = explode(advisor)
    queue = queue_on(tmp_path, scene, site)
    run, days, built = generation_on(
        tmp_path, site, scene, models_down=queue.state.down, queue_owed=queue.state.owed_now
    )
    result = generate(run)
    assert result.complete and result.failed is None
    assert run.marks.finished(TODAY) is True
    assert days.ran == 1 and result.step(DAYS).outcome == STEP_DONE  # type: ignore[union-attr]
    table = built.opportunities.read("session", TODAY)  # type: ignore[attr-defined]
    assert table.night == TODAY
    assert attempts == [] and advisor_attempts == []
    assert result.fallbacks[FALLBACK_UNMAPPED] == 2  # 今晚封口的两条没映射：按"不知道"算、数出来
    assert not mapped(scene, TONIGHT)
    assert [item.step for item in result.steps] == list(STEPS) and "probe" not in STEPS and "vocabulary" not in STEPS


# ── 回放证明不了的三处：探模型、迁移牵到的重映射、重试收尾 ──────────────


def test_neither_the_generation_nor_the_queue_probes_the_model(tmp_path) -> None:
    """探模型（这份回放数据上本来就是 0 次调用）：换代没有探针这个口，做一代时一次模型都不碰；队列也不接探针——模型调用
    失败只看连着几项没成（按通用规则改过：原来守的是"探的是队列"）。

    撤回即失败：把探模型放回换代（``Generation(probe=…)`` 或在步骤里调探针），断言变红。"""

    assert "probe" not in inspect.signature(Generation.__init__).parameters
    assert "probe" not in inspect.signature(ModelQueue.__init__).parameters
    site, scene, provider, _advisor = fresh(tmp_path)
    queue = queue_on(tmp_path, scene, site)
    run, _days, _built = generation_on(
        tmp_path, site, scene, models_down=queue.state.down, queue_owed=queue.state.owed_now
    )
    assert generate(run).complete and provider.calls == 0
    assert asyncio.run(queue.drain()).clear


def moved_breakfast(root: Path):  # type: ignore[no-untyped-def]
    """TONIGHT 那碗面已经映射过；之后迁移把它从「早餐」改到「吃饭」（树上的编号改了、变更流里有这一条）。

    C+D 改过：概念只由换代的同步词表写，映射之前先做一次今天这一代的语义树一步（同步概念），映射之后再做一次（封口、
    盖章）——原来队列自己同步概念、封口由别处做。"""

    site, _hits = site_with_three_bedtimes(root)
    _bed, meal = publish_tonight(site)
    catalog = FakeCatalog()
    scene, provider, _advisor = scene_on(root, site, catalog=catalog)
    series = read_series(site.behavior_tree, cutoff=TODAY)
    asyncio.run(scene.generate(series=series, now=NOW))
    assert asyncio.run(queue_on(root, scene, site).drain()).clear
    asyncio.run(scene.generate(series=series, now=NOW))
    address = BehaviorURI.parse(meal).to_address()
    BehaviorDocumentWriter(site.behavior_tree, ProcessLocalLockStore(), clock=lambda: NOW).restamp_kind_token(
        address, MEAL
    )
    catalog.move(meal, BREAKFAST, MEAL)
    return site, scene, provider, address


def test_the_generation_leaves_the_remap_of_moved_days_to_the_queue(tmp_path) -> None:
    """迁移牵到的重映射（回放数据上 0 次调用）：换代的语义树一步碰到迁移改了编号的已映射日子，不调映射器、不记"同步到的
    版本"（记了队列就读不到那份迁移清单）；队列随后把那一天重映射，结果与"队列先做"的相同。

    撤回即失败：换代的同步词表那一段带上 ``remap``，``calls`` 不再是 0。"""

    later = NOW + timedelta(days=1)

    def clock() -> datetime:
        return later

    # 队列先做（换代之前就重映射完）
    site, scene, provider, address = moved_breakfast(tmp_path / "queue_first")
    assert asyncio.run(
        queue_on(tmp_path / "queue_first", scene, site, clock=clock, sealed=lambda: later.date()).drain()
    )
    first = disk(tmp_path / "queue_first")
    assert scene.hits.read(address).kind_token == MEAL

    # 换代先到
    site, scene, provider, address = moved_breakfast(tmp_path / "generation_first")
    before, synced = provider.calls, scene.concepts.synced_version()
    series = read_series(site.behavior_tree, cutoff=later.date())
    asyncio.run(scene.generate(series=series, now=later))
    assert provider.calls == before  # 换代里没有模型调用
    assert scene.hits.read(address).kind_token == BREAKFAST  # 还没重映射：留给队列
    assert scene.concepts.synced_version() == synced  # 版本没记：迁移清单还读得到
    queue = queue_on(tmp_path / "generation_first", scene, site, clock=clock, sealed=lambda: later.date())
    report = asyncio.run(queue.drain())
    assert (SYNC, DONE) in done(report) and report.clear
    assert scene.hits.read(address).kind_token == MEAL
    second = disk(tmp_path / "generation_first")
    second = {name: body for name, body in second.items() if not name.startswith("relations/")}
    first = {name: body for name, body in first.items() if not name.startswith("relations/")}
    assert second == first


class CountingChains:
    """链处理器的替身：数被试了几次、撤回了谁。"""

    def __init__(self, outcome: Attempt = Attempt.RESTORED, good: bool = True) -> None:
        self.outcome = outcome
        self.good = good
        self.attempted: list[str] = []
        self.abandoned: list[str] = []

    async def attempt(self, item: RetryItem) -> Attempt:
        self.attempted.append(item.item_id)
        return self.outcome

    def recovered(self, item: RetryItem) -> bool:
        return self.good

    def abandon(self, item: RetryItem) -> None:
        self.abandoned.append(item.item_id)


def chain_item(name: str, day: date, cause: RetryCause = RetryCause.UNAVAILABLE) -> RetryItem:
    return RetryItem(
        retry_id(RetryKind.CHAIN, name),
        RetryKind.CHAIN,
        cause,
        day,
        datetime(2026, 8, 18, 12, 0, tzinfo=UTC),
        {"records": [{"name": name}]},
    )


def retry_stage(root: Path, chains: CountingChains, reductions: list[str] | None = None) -> RetryStage:
    backlog = RetryBacklogStore(root / "backlog", max_items=10)

    async def reduce() -> Reduction:
        if reductions is not None:
            reductions.append("reduce")
        return Reduction.DONE

    return RetryStage([RetryBranch("session", backlog, {RetryKind.CHAIN: chains}, reduce)])  # type: ignore[dict-item]


def test_the_generations_retry_closing_never_tries_an_item(tmp_path) -> None:
    """重试收尾（回放数据上 0 次调用）：昨天的可重试项到换代时还在堆里——换代不试它（处理器的 ``attempt`` 一次都没被调、
    归约没跑），只丢弃、撤回、计数，这一代照常写成。

    撤回即失败：把换代的重试收尾改回 ``run_once``，``attempted`` 不再是空。"""

    site, scene, _provider, _advisor = fresh(tmp_path)
    chains = CountingChains()
    reductions: list[str] = []
    stage = retry_stage(tmp_path, chains, reductions)
    item = chain_item("昨天没补回的链", TODAY - timedelta(days=1))
    stage.branches[0].backlog.put(item)
    run, _days, _built = generation_on(tmp_path, site, scene, retry=stage)
    result = generate(run)
    assert result.complete
    assert chains.attempted == [] and reductions == []
    assert chains.abandoned == [item.item_id] and stage.branches[0].backlog.count() == 0
    step = result.step(RETRY_STEP)
    assert step is not None and step.outcome == STEP_DONE
    assert step.report["retry"]["unavailable"] == 1  # type: ignore[index]


# ── A1-6：队列做到一半杀进程，重起 ──────────────────────────────────────


@pytest.mark.parametrize("killed_in", [SYNC, MAPPING, ADVICE])
def test_a1_6_a_queue_killed_in_the_middle_of_an_item_is_resumed_from_disk(tmp_path, killed_in: str) -> None:
    """A1-6（共享说明第 3 节"队列做到一半进程被杀：重起后不读任何做到哪了的账，从盘上重新认欠账，接着做，结果相同"）：
    在一项做到一半时取消（同步词表写完概念之后、映射写了一条命中之后、先验问了一个后果之后），然后**整套对象重开**（新的存储、
    新的队列，没有任何内存状态），做到没有欠账——盘面与不中断的逐字节相同。

    撤回即失败：队列重起后只做内存里记的欠账，重开的那一个什么都不做，盘面不同。

    C+D 改过 ``sync`` 那一格：概念只由换代的同步词表写，队列的同步词表一项只剩"迁移牵到的重判 → 记下同步到的版本"，原来
    切在"写完概念之后"的那一刀落不到队列里了。改成现场"换代已把概念同步好、词表另记着一条迁移"
    （``test_model_queue.site_and_scene``），切在"重判做完、同步到的版本还没记"那一刻。"""

    if killed_in == SYNC:
        from tests.unit.runtime.test_model_queue import site_and_scene

        def sync_site(where: Path):  # type: ignore[no-untyped-def]
            where.mkdir(parents=True)
            site, scene, _provider = site_and_scene(where, yes=YES)
            return site, scene

        site, scene = sync_site(tmp_path / "straight")
        assert asyncio.run(queue_on(tmp_path / "straight", scene, site).drain()).clear
        expected = disk(tmp_path / "straight")
        root = tmp_path / "killed"
        site, scene = sync_site(root)
        queue = queue_on(root, scene, site)

        def cut_version(version):  # type: ignore[no-untyped-def]
            raise asyncio.CancelledError

        scene.concepts.write_synced_version = cut_version  # type: ignore[method-assign]
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(queue.drain())
        assert disk(root) != expected  # 真的停在了半路
        reopened, _provider = nightly(root, ConceptHitStore(root / "scene"), site, yes=YES, catalog=scene.catalog)
        report = asyncio.run(queue_on(root, reopened, site).drain())
        assert report.clear and (SYNC, DONE) in done(report)
        assert disk(root) == expected
        return

    expected = straight(tmp_path / "straight", advice=True)

    root = tmp_path / "killed"
    site, scene, provider, advisor = fresh(root, advice=True)
    queue = queue_on(root, scene, site)
    if killed_in == MAPPING:
        real_write = scene.hits.write

        def cut_write(record):  # type: ignore[no-untyped-def]
            real_write(record)
            raise asyncio.CancelledError

        scene.hits.write = cut_write  # type: ignore[method-assign]
    else:
        store = scene.advice.store  # type: ignore[union-attr]
        real_prior = store.write_prior

        def cut_prior(*args, **kwargs):  # type: ignore[no-untyped-def]
            real_prior(*args, **kwargs)
            raise asyncio.CancelledError

        store.write_prior = cut_prior  # type: ignore[method-assign]
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(queue.drain())
    assert disk(root) != expected  # 真的停在了半路

    reopened, _provider, _advisor = scene_on(root, site, advice=True)
    report = asyncio.run(queue_on(root, reopened, site).drain())
    assert report.clear and report.done
    assert disk(root) == expected


# ── A1-7：一次模型调用没答成 / 超时 ─────────────────────────────────────


def test_a1_7_an_item_whose_model_call_fails_backs_off_on_the_ledger_and_the_queue_moves_on(tmp_path) -> None:
    """A1-7 的前半（"这一项按重试账退避；队列接着做下一项"）：映射里模型这一次没答成（答复不合格，不是整个调不通）——这一天
    不盖章、记进队列的重试账，队列不停，后面的类照做；退避期内这一项不重排，项还在（欠账数不为 0）；退避过了、模型
    好了，这一项补上，盘面与没出过事的相同（除完成章上的时刻）。

    撤回即失败：一项没做成就把它当成"做完"放过，``owed[MAPPING]`` 归零、退避过后不补。

    C+D 改过：补问排到了映射前面（今天这一代之后做，先后见 ``model_queue_state.CATEGORIES``），映射是一轮里最后一类，
    "后面的类照做"在这里看不出来，改成"同一趟里别的项照做"；"一项没做成不挡后面的类"由 H3 与复验 A 两条守（出错重试是
    第一类）。"""

    clock = [NOW]
    site, scene, provider, advisor = fresh(tmp_path, advice=True)
    good = provider.complete_async
    broken = [True]

    async def answer(request):  # type: ignore[no-untyped-def]
        if broken[0]:
            provider.calls += 1
            return ModelResponse(content="not json", model=provider.model, provider="fake", finish_reason="stop")
        return await good(request)

    provider.complete_async = answer
    # 收口删了离线一次做完那条路与它的按天映射重试账（``scene-mapper``）：退避走队列自己那本（``queue_on`` 的钟跟着这条测试走）
    queue = queue_on(tmp_path, scene, site, clock=lambda: clock[0])
    report = asyncio.run(queue.drain())
    assert (MAPPING, UNFINISHED) in done(report)
    assert (ADVICE, DONE) in done(report) and (MAPPING, DONE) in done(report)  # 同一趟里别的项照做
    assert not mapped(scene, TONIGHT) and report.owed[MAPPING] == 1 and not report.clear
    assert not queue.state.down()  # 一次没答成不是整个调不通：不整体退避
    asked = provider.calls
    broken[0] = False
    again = asyncio.run(queue.drain())
    assert provider.calls == asked and again.owed[MAPPING] == 1  # 退避期内不重问，项还在
    clock[0] = NOW + timedelta(hours=2)
    final = asyncio.run(queue.drain())
    assert (MAPPING, DONE) in done(final) and final.clear and mapped(scene, TONIGHT)


def timing_out_mapper(tmp_path):  # type: ignore[no-untyped-def]
    """映射的模型调用一律超时（``TimeoutError``），走真的 ``ChatClient`` 与真的 ``ModelCallTracer``；别的触点照答。"""

    from habitus.model_client import ChatClient
    from habitus.model_client.structured import StructuredChatClient
    from habitus.runtime.generation_trace import ModelCallTracer
    from tests.unit.foresight.scripted_model import model_config

    site, scene, provider, advisor = fresh(tmp_path, advice=True)

    async def timing_out(request):  # type: ignore[no-untyped-def]
        provider.calls += 1
        raise TimeoutError("read timed out")

    provider.complete_async = timing_out
    client = StructuredChatClient(
        ChatClient(model_config(), provider, observer=ModelCallTracer()), validation_retries=1
    )
    mapper_for = scene.mapper_for

    def traced(concepts):  # type: ignore[no-untyped-def]
        mapper = mapper_for(concepts)
        mapper.client = client
        return mapper

    scene.mapper_for = traced  # type: ignore[method-assign]
    traces = TraceFiles(tmp_path / "scene" / "generation" / "traces", artifact_root=tmp_path / "scene")
    return site, scene, provider, advisor, traces


def test_a1_7_a_timed_out_model_call_backs_off_only_its_item_and_the_queue_moves_on(tmp_path) -> None:
    """A1-7 / G2（"一次模型调用超时：这一项退避；队列接着做下一项"；方案第 4 节"一条卡住不挡后面的"）：映射的调用一直
    超时——这一项在进程里退避（模型调用失败不记落盘的重试账、不会被放弃），队列不进整体退避，后面的先验补问照做；退避期内
    再 ``drain`` 不重问。没有探针（通用规则：只看连着几项没成）。

    撤回即失败：一项报调不通就整体退避，``state.down()`` 为真、``backing_off_until`` 不是 None。

    C+D 改过：补问排到了映射前面，原来"映射之后的补问照做"的先后断言改成"同一趟里补问、另一条记录的映射照做"。"""

    site, scene, provider, advisor, traces = timing_out_mapper(tmp_path)
    queue = queue_on(tmp_path, scene, site, traces=traces)
    report = asyncio.run(queue.drain())
    assert provider.calls >= 1
    assert not queue.state.down() and report.backing_off_until is None
    assert (MAPPING, UNREACHABLE) in done(report)
    assert (ADVICE, DONE) in done(report) and (MAPPING, DONE) in done(report) and advisor.calls > 0
    assert report.owed[MAPPING] == 1 and not mapped(scene, TONIGHT)  # 项没丢
    assert queue.retries.size() == 0  # 模型失败不进落盘的重试账
    asked = provider.calls
    again = asyncio.run(queue.drain())
    assert provider.calls == asked and again.owed[MAPPING] == 1 and not queue.state.down()  # 退避期内不重问


def test_a1_7_a_timed_out_model_call_is_counted_on_the_trace(tmp_path) -> None:
    """A1-7 的后半 / G3：队列一项里模型调用超时，这一项的 ``model_queue.item`` span 上没成的调用数（``model_calls_failed``，
    不分错误种类）至少是 1；没有失败调用的项上没有这个属性。"""

    site, scene, provider, _advisor, traces = timing_out_mapper(tmp_path)
    asyncio.run(queue_on(tmp_path, scene, site, traces=traces).drain())
    assert provider.calls >= 1
    lines = [json.loads(line) for line in traces.path(TODAY).read_text(encoding="utf-8").splitlines()]
    items = [line for line in lines if line["name"] == "model_queue.item" and line["attributes"]["category"] == MAPPING]
    assert items and sum(int(line["attributes"].get("model_calls_failed", 0)) for line in items) >= 1
    others = [
        line for line in lines if line["name"] == "model_queue.item" and line["attributes"]["category"] != MAPPING
    ]
    assert all("model_calls_failed" not in line["attributes"] for line in others)


# ── A1-8：模型整个调不通 ────────────────────────────────────────────────


def masked(files: dict[str, bytes]) -> dict[str, bytes]:
    """去掉完成章上的盖章时刻（中断过的那一份是在恢复的那一刻盖的章）。"""

    found = {}
    for name, body in files.items():
        if name.endswith(".done.json"):
            marker = json.loads(body)
            marker.pop("completed_at")
            body = json.dumps(marker, sort_keys=True, ensure_ascii=False).encode("utf-8")
        found[name] = body
    return found


def test_a1_8_an_unreachable_model_backs_the_whole_queue_off_and_nothing_is_lost(tmp_path) -> None:
    """A1-8（方案第 4 节"模型整个调不通时队列整体退避，不逐条等超时"）：做到一半模型断了（第一次调模型起整个调不通）——
    连着配置的项数都失败（这里配成 1 项，线上默认 3），队列整体退避、这一趟后面的项不再试；退避期内再 ``drain`` 一次模型都
    不碰；到点接着做、头一项又失败——退避加倍、封顶；欠着的项一个不少。模型恢复后做完，盘面与没断过的相同（除完成章上的
    盖章时刻）。

    撤回即失败：模型整个调不通时不整体退避（当成普通的一项失败接着做），"退避期内不碰模型"与"这一趟后面的项不再试"变红。

    C+D 改过：补问排到了映射前面，一轮里头一项是先验。断的是同一个模型：映射与先验的替身都断（原来只断映射的那一个，
    头一项是映射）；"后面的项不再试"看映射一项都没试、映射的替身一次没被调。"""

    expected = masked(straight(tmp_path / "straight", advice=True))

    clock = [NOW]
    root = tmp_path / "outage"
    site, scene, provider, advisor = fresh(root, advice=True)
    armed, down = [True], [False]

    def outage(fake):  # type: ignore[no-untyped-def]
        good = fake.complete_async

        async def answer(request):  # type: ignore[no-untyped-def]
            down[0] = down[0] or armed[0]  # 第一次调模型起就断了
            if down[0]:
                fake.calls += 1
                raise ModelTransportError("connection refused")
            return await good(request)

        fake.complete_async = answer

    outage(provider)
    outage(advisor)
    queue = queue_on(
        root,
        scene,
        site,
        clock=lambda: clock[0],
        settings=QueueSettings(backoff_base_seconds=60.0, backoff_max_seconds=100.0, unreachable_after_failures=1),
    )
    first = asyncio.run(queue.drain())
    assert done(first) == [(ADVICE, UNREACHABLE)] and queue.state.down()
    assert provider.calls == 0  # 后面的项（映射）不再试
    assert first.backing_off_until == NOW + timedelta(seconds=60)
    assert first.owed[MAPPING] >= 1 and not first.clear and not mapped(scene, TONIGHT)
    asked, asked_advice = provider.calls, advisor.calls

    clock[0] = NOW + timedelta(seconds=30)
    during = asyncio.run(queue.drain())
    assert during.done == () and provider.calls == asked and advisor.calls == asked_advice  # 退避期内一次都不碰模型
    assert during.backing_off_until == NOW + timedelta(seconds=60) and during.owed[MAPPING] >= 1

    clock[0] = NOW + timedelta(seconds=61)
    second = asyncio.run(queue.drain())
    assert done(second) == [(ADVICE, UNREACHABLE)] and advisor.calls > asked_advice  # 到点接着做：头一项又失败
    assert second.backing_off_until == clock[0] + timedelta(seconds=100)  # 加倍（120）被封顶压到 100
    assert queue.state.failures == 2 and queue.state.backoffs == 2 and second.owed[MAPPING] >= 1
    names = [name for name, _detail in health_module.queue_problems(queue.state, queue.settings, clock[0])]
    assert "model_queue_backoff" in names

    armed[0] = down[0] = False
    clock[0] = NOW + timedelta(seconds=200)
    final = asyncio.run(queue.drain())
    assert final.clear and not queue.state.down() and queue.state.failures == 0 and final.backing_off_until is None
    assert masked(disk(root)) == expected
    assert health_module.queue_problems(queue.state, queue.settings, clock[0]) == []


# ── A1-9：换代时队列在整体退避 ──────────────────────────────────────────


def test_a1_9_a_generation_during_the_backoff_writes_no_relation_table_and_uses_the_previous_one(tmp_path) -> None:
    """A1-9（裁定 96、立项第二部分第 2 节末）：第一天队列做完、换代写出关系表；第二天模型整个调不通，队列进了整体退避——
    第二天这一代照样 complete，但不写这一代的关系表，机会表用上一代的那一晚，``relations_previous`` 计 1。退避状态是真的队列
    自己跑出来的（不是手填的布尔值）：靠"连着几项模型调用失败"进整体退避，这里配成 1 项（线上默认 3）。"""

    site, scene, provider, _advisor = fresh(tmp_path)
    clock = [NOW]
    queue = queue_on(
        tmp_path,
        scene,
        site,
        clock=lambda: clock[0],
        sealed=lambda: clock[0].date(),
        settings=QueueSettings(unreachable_after_failures=1),
    )
    assert asyncio.run(queue.drain()).clear
    built = foresight_components(tmp_path, site)
    foresight = build_foresight_nightly(
        built, relations=scene.relations, relation_config=scene.relation_config, clock=lambda: clock[0]
    )

    def generation() -> Generation:
        run = assembled_with(
            tmp_path,
            site,
            scene,
            foresight=foresight,
            clock=lambda: clock[0],
            closed_days=sealed_through(clock[0].date() - timedelta(days=1)),
            models_down=queue.state.down,
            queue_owed=queue.state.owed_now,
        )
        run.days = StubDays()  # type: ignore[assignment]
        return run

    first = generate(generation())
    assert first.complete and scene.relations.nights("session") == (TODAY,)
    assert first.fallbacks[FALLBACK_RELATIONS] == 0

    # 第二天：又封口一天的记录，模型整个调不通
    tomorrow = TODAY + timedelta(days=1)
    publish_tonight(site, day=TODAY)
    clock[0] = NOW + timedelta(days=1)

    async def unreachable(request):  # type: ignore[no-untyped-def]
        raise ModelTransportError("connection refused")

    provider.complete_async = unreachable
    report = asyncio.run(queue.drain())
    assert queue.state.down() and not report.clear
    second = generate(generation())
    assert second.complete and second.day == tomorrow
    assert scene.relations.nights("session") == (TODAY,)  # 这一代的关系表没写
    assert second.fallbacks[FALLBACK_RELATIONS] == 1
    assert second.relation_nights == {"session": TODAY}  # 机会表用的是上一代那一晚的关系表
    assert built.opportunities.read("session", tomorrow).relation_night == TODAY
    step = second.step(SCENE)
    assert step is not None and step.outcome == DEGRADED and "模型调不通" in str(step.report["relations_held"])
    assert second.step(OPPORTUNITIES).outcome == STEP_DONE and second.step(MARK).outcome == STEP_DONE  # type: ignore[union-attr]
    assert second.fallbacks[FALLBACK_UNMAPPED] == 2  # 昨天那两条没映射：换代自己数得出来
    # H2：整体退避让这一轮中途返回，排在后面的类也从盘上认过——换代报告记下的欠账是全的
    assert second.queue_owed is not None and second.queue_owed[MAPPING] >= 1


# ── A1-10：重试堆里有今天的可重试项 ─────────────────────────────────────


def test_a1_10_todays_retryable_item_is_tried_the_same_day_kept_while_the_model_is_down_and_dropped_at_sealing(
    tmp_path,
) -> None:
    """A1-10（方案第 4 节"出错重试就近""模型此刻用不了不算试过，留到下一次；那天封口时还没成的才丢"）：

    - 今天进堆的可重试项，队列当天就试（不等那天封口），补回了就从堆里删掉；
    - 试的时候模型用不了：这一次不算试过，项还在堆里、在进程里等着（按通用规则改过：单独一项失败不再整体退避）；
    - 那天封口后，换代的重试收尾把还没成的丢掉、撤回、计一项没补回——换代自己不试。

    撤回即失败：队列不试今天的项（只留给换代），第一段 ``attempted`` 是空。"""

    # 当天就试、补回
    site, scene, _provider, _advisor = fresh(tmp_path / "recovered")
    chains = CountingChains()
    stage = retry_stage(tmp_path / "recovered", chains)
    backlog = stage.branches[0].backlog
    item = chain_item("今天出错的链", TODAY)
    backlog.put(item)
    queue = queue_on(
        tmp_path / "recovered", scene, site, retry=stage, sealed=lambda: None
    )  # 还没到能做已封口的活的时候
    report = asyncio.run(queue.drain())
    assert others(report) == [(RETRY, DONE)] and chains.attempted == [item.item_id]
    assert backlog.count() == 0 and backlog.through() is None  # 就近试不推"重试到哪一天"

    # 模型用不了的那一次：项还在
    site, scene, _provider, _advisor = fresh(tmp_path / "down")
    chains = CountingChains(outcome=Attempt.UNAVAILABLE)
    stage = retry_stage(tmp_path / "down", chains)
    backlog = stage.branches[0].backlog
    backlog.put(item)
    clock = [NOW]
    queue = queue_on(tmp_path / "down", scene, site, retry=stage, clock=lambda: clock[0], sealed=lambda: None)
    report = asyncio.run(queue.drain())
    assert others(report) == [(RETRY, UNREACHABLE)] and not queue.state.down()
    assert asyncio.run(queue.drain()).done == ()  # 这一项在等：不每一趟都重问
    assert backlog.count() == 1 and report.owed[RETRY] == 1 and chains.abandoned == []
    tried = len(chains.attempted)
    assert tried == 1

    # 那天封口了：换代的重试收尾丢掉没成的，自己不试
    clock[0] = NOW + timedelta(days=1)
    run, _days, _built = generation_on(
        tmp_path / "down",
        site,
        scene,
        retry=stage,
        clock=lambda: clock[0],
        closed_days=sealed_through(TODAY),
        models_down=queue.state.down,
    )
    result = generate(run)
    assert result.complete and result.day == TODAY + timedelta(days=1)
    assert backlog.count() == 0 and chains.abandoned == [item.item_id] and len(chains.attempted) == tried
    step = result.step(RETRY_STEP)
    assert step is not None and step.report["retry"]["unavailable"] == 1  # type: ignore[index]
    clock[0] = NOW + timedelta(days=1, hours=2)
    assert asyncio.run(queue.drain()).owed[RETRY] == 0  # 封口之后队列不再认它欠着


# ── A1-11：健康面 ───────────────────────────────────────────────────────


def test_a1_11_health_degrades_on_a_backlog_with_counts_and_the_oldest_wait_and_clears_by_itself(tmp_path) -> None:
    """A1-11：欠账到了阈值，健康面多一项 non-critical 的降级，带各类条数与最老一项等了多久；清完自己消掉。读的是运行时
    健康面真的会调的那个函数（``runtime.health._model_queue_checks``）。

    C+D 改过：现场换成"换代已把概念同步好、词表另记着一条迁移"（``test_model_queue.site_and_scene``）——概念只由换代写之后，
    原来的现场在换代之前队列不欠同步词表。"""

    from tests.unit.runtime.test_model_queue import site_and_scene

    clock = [NOW]
    site, scene, _provider = site_and_scene(tmp_path)
    queue = queue_on(
        tmp_path,
        scene,
        site,
        clock=lambda: clock[0],
        settings=QueueSettings(warning_items=1, warning_wait_seconds=3_600.0),
    )
    worker = ModelQueueWorker(queue, shutdown_timeout_seconds=1.0)
    assert health_module._model_queue_checks(worker) == []

    real = scene.syncing.run

    async def broken(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise OSError("disk")

    scene.syncing.run = broken  # type: ignore[method-assign]
    report = asyncio.run(queue.drain())
    assert report.owed[SYNC] == 1
    checks = {check.name: check for check in health_module._model_queue_checks(worker)}
    backlog = checks["model_queue_backlog"]
    assert backlog.status.value == "degraded" and backlog.critical is False
    assert "sync=1" in backlog.detail and "oldest waited" in backlog.detail and "owed 1" in backlog.detail

    scene.syncing.run = real  # type: ignore[method-assign]
    clock[0] = NOW + timedelta(hours=2)
    assert asyncio.run(queue.drain()).clear
    assert health_module._model_queue_checks(worker) == []


# ── A1-12：配置 ─────────────────────────────────────────────────────────

QUEUE_FIELDS = (
    "model_queue_poll_seconds",
    "model_queue_backoff_base_seconds",
    "model_queue_backoff_max_seconds",
    "model_queue_shutdown_timeout_seconds",
    "model_queue_backlog_warning_items",
    "model_queue_backlog_warning_seconds",
)


def test_a1_12_the_queue_settings_are_in_the_example_yaml_with_their_defaults_and_bad_ones_are_refused() -> None:
    """A1-12：新字段逐个列在示例 YAML 里、写的值就是代码里的默认值；未知字段、越界、类型不对都被拒；合法值进得来。"""

    example = (Path(__file__).resolve().parents[3] / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8")
    listed = {}
    for line in example.splitlines():
        text = line.strip().removeprefix("#").strip()
        name, _colon, rest = text.partition(":")
        if name in QUEUE_FIELDS:
            listed[name] = float(rest.split("#")[0].strip())
    defaults = SceneConfig()
    assert set(listed) == set(QUEUE_FIELDS)
    assert listed == {name: float(getattr(defaults, name)) for name in QUEUE_FIELDS}

    accepted = SceneConfig.from_mapping(
        {"enabled": True, "model_queue_poll_seconds": 5, "model_queue_backlog_warning_items": 7}
    )
    assert accepted.model_queue_poll_seconds == 5 and accepted.model_queue_backlog_warning_items == 7
    for name, bad in (
        ("model_queue_poll_seconds", 0),
        ("model_queue_poll_seconds", True),
        ("model_queue_poll_seconds", "60"),
        ("model_queue_poll_seconds", 86_401),
        ("model_queue_backoff_base_seconds", 0.5),
        ("model_queue_backoff_base_seconds", 3_601.0),  # 起点超过封顶
        ("model_queue_backoff_max_seconds", 86_401.0),
        ("model_queue_shutdown_timeout_seconds", 0),
        ("model_queue_shutdown_timeout_seconds", 3_601),
        ("model_queue_backlog_warning_items", 0),
        ("model_queue_backlog_warning_items", 2.5),
        ("model_queue_backlog_warning_items", "50"),
        ("model_queue_backlog_warning_seconds", 59),
        ("model_queue_backlog_warning_seconds", None),
    ):
        with pytest.raises(ConfigError):
            SceneConfig.from_mapping({name: bad})
    for unknown in ("model_queue_poll", "model_queue_backoff_seconds", "model_queue_enabled"):
        with pytest.raises(ConfigError):
            SceneConfig.from_mapping({unknown: 60})


# ── 对抗 ────────────────────────────────────────────────────────────────


def test_a_corrupt_retry_item_neither_stalls_the_queue_nor_vanishes_uncounted(tmp_path) -> None:
    """重试堆里一项文件损坏：读得了的那一项照试；坏的那一项不卡队列，被丢弃时按"我们自己处理不了的"计数（不静默消失）——
    与好的项一起时就近丢、只有它自己时留到那天封口由重试收尾丢。"""

    site, scene, _provider, _advisor = fresh(tmp_path / "mixed")
    chains = CountingChains()
    stage = retry_stage(tmp_path / "mixed", chains)
    backlog = stage.branches[0].backlog
    good, bad = chain_item("好的链", TODAY), chain_item("坏的链", TODAY)
    backlog.put(good)
    backlog.put(bad)
    backlog._path(bad.item_id).write_bytes(b"{not json")
    report = asyncio.run(queue_on(tmp_path / "mixed", scene, site, retry=stage, sealed=lambda: None).drain())
    assert others(report) == [(RETRY, DONE)] and chains.attempted == [good.item_id] and backlog.count() == 0
    closed = stage.close(through=TODAY)
    assert (closed.recovered, closed.deterministic) == (1, 1)  # 坏的那一项计进了那天的报告

    site, scene, _provider, _advisor = fresh(tmp_path / "alone")
    chains = CountingChains()
    stage = retry_stage(tmp_path / "alone", chains)
    backlog = stage.branches[0].backlog
    backlog.put(bad)
    backlog._path(bad.item_id).write_bytes(b"{not json")
    queue = queue_on(tmp_path / "alone", scene, site, retry=stage)
    report = asyncio.run(queue.drain())
    assert report.clear and RETRY not in [category for category, _outcome in done(report)]  # 不卡队列，后面的类照做
    assert (MAPPING, DONE) in done(report) and chains.attempted == []
    closed = stage.close(through=TODAY)
    assert closed.deterministic == 1 and backlog.count() == 0


def test_hits_stamped_under_another_mapper_version_are_recognised_and_mapped_again(tmp_path) -> None:
    """命中文件的口径不是现在的口径（换过提示词 / 模型之后盘上留着的旧命中与旧完成章）：队列从盘上认出这一天还欠着、
    重新映射，盘面回到与一直用现在的口径做出来的相同——不拿旧口径的命中当现成的。"""

    from dataclasses import replace

    site, scene, _provider, _advisor = fresh(tmp_path / "straight")
    assert asyncio.run(queue_on(tmp_path / "straight", scene, site).drain()).clear
    seal(scene)  # 映射按槽之后章由换代封口时盖：两边都封口再比
    expected = disk(tmp_path / "straight")
    root = tmp_path / "stale"
    site, scene, provider, _advisor = fresh(root)
    assert asyncio.run(queue_on(root, scene, site).drain()).clear
    seal(scene)
    marker = scene.hits.read_marker(TONIGHT)
    assert marker is not None
    old = "scene_concept_mapper_prompt_v0+old"
    for record in scene.hits.read_day(TONIGHT):
        scene.hits.write(replace(record, mapper=old))
    scene.hits.complete_day(TONIGHT, records=marker.records, completed_at=marker.completed_at, mapper=old)
    assert disk(root) != expected

    reopened, provider, _advisor = scene_on(root, site)
    queue = queue_on(root, reopened, site)
    report = asyncio.run(queue.drain())
    assert (MAPPING, DONE) in done(report) and report.clear and provider.calls > 0
    assert reopened.hits.read_marker(TONIGHT) is None  # 重判过的那一天旧章撤了
    seal(reopened)
    assert reopened.hits.read_marker(TONIGHT).mapper == marker.mapper  # type: ignore[union-attr]
    assert disk(root) == expected


def test_a_prior_cache_that_is_not_canonical_json_fails_loudly_and_is_never_used(tmp_path) -> None:
    """先验缓存不是规范 JSON（被人手改过）：队列这一趟明确失败（抛 ``AdviceStoreError``，常驻循环记成这一趟出错、健康面
    报出）、不拿它当答案、不改写它；换代照样 complete，但不拿读不了的缓存判出一张关系表（这一代沿用上一代的）。文件修好
    之后队列自己恢复。"""

    from habitus.scene.advice.store import AdviceStoreError

    root = tmp_path
    site, scene, _provider, advisor = fresh(root, advice=True)
    assert asyncio.run(queue_on(root, scene, site).drain()).clear
    victim = sorted((root / "scene" / "advice" / "priors").glob("*.json"))[0]
    canonical = victim.read_bytes()
    victim.write_text(json.dumps(json.loads(canonical), indent=2), encoding="utf-8")  # 合法 JSON，但不是规范形式
    tampered = victim.read_bytes()
    asked = advisor.calls

    queue = queue_on(root, scene, site)
    for _ in range(2):
        with pytest.raises(AdviceStoreError):
            asyncio.run(queue.drain())
    assert advisor.calls == asked and victim.read_bytes() == tampered
    run, _days, _built = generation_on(root, site, scene, models_down=queue.state.down)
    result = generate(run)
    assert result.complete and scene.relations.nights("session") == ()
    assert result.fallbacks[FALLBACK_RELATIONS] == 1 and result.step(SCENE).outcome == DEGRADED  # type: ignore[union-attr]

    victim.write_bytes(canonical)
    assert asyncio.run(queue.drain()).clear


def test_two_queues_on_one_root_draining_together_map_each_record_once(tmp_path) -> None:
    """两个 ``ModelQueue`` 挂同一个根、同时 ``drain``（各有各的存储对象与替身模型，共用一把队列租约）：只有一个在做，另一个
    这一趟什么都不做、也不说"没有欠账"；同一条记录只被映射一次、同一个后果只被问一次，盘面与单个队列做出来的相同。"""

    expected = straight(tmp_path / "straight", advice=True)
    root = tmp_path / "shared"
    site, scene, provider, advisor = fresh(root, advice=True)
    other_scene, other_provider, other_advisor = scene_on(root, site, advice=True)
    locks = ProcessLocalLockStore()

    def lease() -> Lease:
        return Lease(locks, "scene:model-queue:shared", ttl_seconds=60.0, heartbeat_seconds=20.0)

    first = queue_on(root, scene, site, lease=lease())
    second = queue_on(root, other_scene, site, lease=lease())

    async def both():  # type: ignore[no-untyped-def]
        return await asyncio.gather(first.drain(), second.drain())

    one, two = asyncio.run(both())
    worked, idle = (one, two) if one.done else (two, one)
    assert worked.clear and idle.done == () and idle.held_elsewhere and not idle.clear
    single = straight_calls(tmp_path / "count")
    assert (provider.calls + other_provider.calls, advisor.calls + other_advisor.calls) == single  # type: ignore[union-attr]
    assert disk(root) == expected
    after = asyncio.run(second.drain())  # 租约放开之后另一个再看：从盘上认出已经没有欠账
    assert after.clear and after.done == () and other_provider.calls + provider.calls == single[0]


def straight_calls(root: Path) -> tuple[int, int]:
    site, scene, provider, advisor = fresh(root, advice=True)
    assert asyncio.run(queue_on(root, scene, site).drain()).clear
    return provider.calls, advisor.calls  # type: ignore[union-attr]


def test_an_item_cancelled_halfway_is_picked_up_again_by_the_same_queue(tmp_path) -> None:
    """队列一项做到一半被取消（停机超时）后，同一个队列对象重新 ``drain``：租约已经放开（不会一直 ``held_elsewhere``）、
    取消的那一项没记成失败（不进退避）、从盘上重新认出来做完，盘面与不中断的相同。"""

    expected = straight(tmp_path / "straight", advice=True)
    root = tmp_path / "cancelled"
    site, scene, _provider, _advisor = fresh(root, advice=True)
    queue = queue_on(root, scene, site)
    real_write = scene.hits.write
    armed = [True]

    def cut(record):  # type: ignore[no-untyped-def]
        real_write(record)
        if armed[0]:
            armed[0] = False
            raise asyncio.CancelledError

    scene.hits.write = cut  # type: ignore[method-assign]
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(queue.drain())
    report = asyncio.run(queue.drain())
    assert not report.held_elsewhere and report.clear
    assert all(outcome == DONE for _category, outcome in done(report)) and (MAPPING, DONE) in done(report)
    assert disk(root) == expected


# ── 修复轮（04-修复清单-A1）G1、G4、G5、G6、G7 ───────────────────────────


def test_g1_a_vocabulary_job_stuck_on_the_model_does_not_hold_todays_generation(tmp_path) -> None:
    """G1（方案"模型多慢都不影响今天这一代"）：队列的词表一项等模型时整段持着 sweep 锁（真的 ``BehaviorReductionRunner``、
    真的锁）。这时换代看一眼：没有没做完的词表迁移，换代不去拿这把锁——这一代 complete，完成标记、日子预期都写出。有没做完
    的迁移时照旧：续做迁移要这把锁，拿不到是还没准备好，这一眼到此为止。

    撤回即失败：续做迁移那一步不管有没有迁移都去拿 sweep 锁，第一段的 ``held`` 是 ``migrate``、不 complete。"""

    from habitus.behavior.reduction.kinds_jobs import JobsOutcome
    from habitus.runtime.model_queue import VocabularyBranch
    from tests.unit.behavior.test_reduction_runner import Harness

    site, scene, _provider, _advisor = fresh(tmp_path / "scene_side")
    harness = Harness(tmp_path / "behavior_side")
    migrating = [False]

    async def scenario():  # type: ignore[no-untyped-def]
        entered, release = asyncio.Event(), asyncio.Event()
        due = [True]

        async def nightly(now, *, night, checkpoint):  # type: ignore[no-untyped-def]
            entered.set()
            await release.wait()  # 词表整理活里那一次模型调用：卡住
            due[0] = False
            return JobsOutcome(model_calls=1)

        harness.vocabulary_jobs.nightly = nightly  # type: ignore[method-assign]
        branch = VocabularyBranch("session", lambda _night: due[0], harness.runner.nightly_vocabulary)
        queue = queue_on(tmp_path / "scene_side", scene, site, vocabulary_branches=(branch,))
        draining = asyncio.create_task(queue.drain())
        await entered.wait()
        # 有没做完的迁移：照旧要锁
        run, days, _built = generation_on(
            tmp_path / "scene_side",
            site,
            scene,
            migration_jobs=(harness.runner.vocabulary.resume,),
            vocabulary=lambda: (1, migrating[0]),
        )
        migrating[0] = True
        held = await run.run(TODAY)
        migrating[0] = False
        # 没有没做完的迁移：不拿锁，这一代写成
        result = await run.run(TODAY)
        stuck = not draining.done()
        release.set()
        report = await draining
        return held, result, stuck, days, run, report

    held, result, stuck, days, run, report = asyncio.run(scenario())
    assert held.held == "migrate" and not held.complete
    assert stuck  # 换代写成的时候那次模型调用还卡着
    assert result.complete and result.held is None and run.marks.finished(TODAY) is True and days.ran == 1
    assert done(report)[0] == ("vocabulary", DONE)


def test_g4_what_is_owed_keeps_its_age_and_its_generations_across_midnight(tmp_path) -> None:
    """G4（用户 11-11：跨夜的欠账要在健康面看得见，清完才消）：一类活一直欠着，中间过了零点（今天还不能做已封口的活的那一段）
    ——"从什么时候起欠着"不变、告警不消；之后又写成一代"跨了几代"是 1，再一代是 2；清完才都归零。

    C+D 改过（实时化阶段 2「生效时点」）：同步词表（迁移牵到的重判）现在是今天这一代完成之后才算欠着的活——现场换成"换代
    已把概念同步好、词表另记着一条迁移"（``test_model_queue.site_and_scene``），今天这一代的完成标记先写上；跨代的数从
    欠账开始之后写成的那几代数起，所以是 1、2（原来第一份标记写在欠账开始之后，是 1、2 的前一格）。"""

    from tests.unit.runtime.test_model_queue import site_and_scene

    site, scene, _provider = site_and_scene(tmp_path)
    marks = GenerationMarks(tmp_path / "scene" / "generation", artifact_root=tmp_path / "scene")
    marks.write(GenerationMark(day=TODAY, finished_at=NOW - timedelta(minutes=5)))
    clock: list[datetime] = [NOW]
    sealed: list[date | None] = [TODAY]
    queue = queue_on(
        tmp_path,
        scene,
        site,
        clock=lambda: clock[0],
        sealed=lambda: sealed[0],
        marks=marks,
        settings=QueueSettings(warning_items=1, warning_wait_seconds=3_600.0),
    )
    real = scene.syncing.run

    async def broken(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise OSError("disk")

    scene.syncing.run = broken  # type: ignore[method-assign]

    def problems() -> dict[str, str]:
        asyncio.run(queue.drain())
        return dict(health_module.queue_problems(queue.state, queue.settings, clock[0]))

    found = problems()
    assert "model_queue_backlog" in found and "model_queue_crossed" not in found and queue.state.since[SYNC] == NOW

    # 过零点：今天还不能做已封口的活
    clock[0], sealed[0] = NOW + timedelta(hours=21, minutes=10), None
    found = problems()
    assert queue.state.owed_now()[SYNC] == 1 and queue.state.since[SYNC] == NOW
    assert "sync=1" in found["model_queue_backlog"]

    # 第二天封口、这一代还没做完（同步词表这时不看，原来的数留着）；这一代写成之后跨了一代
    clock[0], sealed[0] = NOW + timedelta(hours=22), TODAY + timedelta(days=1)
    assert queue.state.since[SYNC] == NOW and "model_queue_backlog" in problems()
    marks.write(GenerationMark(day=TODAY + timedelta(days=1), finished_at=NOW + timedelta(hours=22, minutes=5)))
    clock[0] = NOW + timedelta(hours=34)
    assert problems()["model_queue_crossed"].endswith("sync×1")

    marks.write(GenerationMark(day=TODAY + timedelta(days=2), finished_at=NOW + timedelta(hours=46, minutes=5)))
    clock[0], sealed[0] = NOW + timedelta(hours=47), TODAY + timedelta(days=2)
    found = problems()
    assert found["model_queue_crossed"].endswith("sync×2") and queue.state.since[SYNC] == NOW
    assert "oldest waited 169200s" in found["model_queue_backlog"]  # 47 小时

    scene.syncing.run = real  # type: ignore[method-assign]
    clock[0] = NOW + timedelta(hours=50)
    assert problems() == {} and SYNC not in queue.state.since and not queue.state.crossed


def test_g5_a_category_whose_backlog_cannot_be_read_does_not_stop_the_ones_after_it(tmp_path) -> None:
    """G5（"一条卡住不挡后面的"对读盘出错同样成立；出错不能静默）：词表一类认欠账时读盘抛了——这一类记失败，后面的同步词表、
    映射、补问这一趟照做完；健康面报出哪一类、原因码；这一趟做完后 ``drain()`` 仍把那个读盘的错抛出来。坏的地方修好后，下一趟
    自己恢复、健康面那一项消掉。

    撤回即失败：认欠账抛了就中断这一趟，后面的映射没做（今晚那天没盖章）。"""

    from habitus.runtime.model_queue import VocabularyBranch

    site, scene, _provider, advisor = fresh(tmp_path, advice=True)
    broken = [True]

    def due(_night):  # type: ignore[no-untyped-def]
        if broken[0]:
            raise ValueError("pending pool is corrupt")
        return False

    async def never(_night):  # type: ignore[no-untyped-def]
        raise AssertionError("not due")

    queue = queue_on(tmp_path, scene, site, vocabulary_branches=(VocabularyBranch("session", due, never),))
    worker = ModelQueueWorker(queue, shutdown_timeout_seconds=1.0)
    with pytest.raises(ValueError, match="pending pool is corrupt"):
        asyncio.run(queue.drain())
    assert mapped(scene, TONIGHT) and advisor.calls > 0  # 后面的类做完了
    assert queue.state.errors == {"vocabulary": "ValueError"}
    checks = {check.name: check for check in health_module._model_queue_checks(worker)}
    assert "vocabulary: ValueError" in checks["model_queue_scan"].detail
    assert checks["model_queue_scan"].status.value == "degraded" and checks["model_queue_scan"].critical is False

    broken[0] = False
    assert asyncio.run(queue.drain()).clear and queue.state.errors == {}
    assert health_module._model_queue_checks(worker) == []


def test_g5_a_corrupt_completion_marker_is_reported_per_category_and_the_earlier_ones_are_done(tmp_path) -> None:
    """G5 用真的坏文件：一天的完成章坏了，映射与补问两类都读不了命中——各记各的失败（不是只报第一类），排在前面的同步词表
    照做；``drain()`` 抛的是第一个读盘的错；不改写坏文件。"""

    from habitus.scene.occurrences.store import ConceptHitStoreError

    site, scene, _provider, _advisor = fresh(tmp_path, advice=True)
    marker = tmp_path / "scene" / "occurrences" / f"{TONIGHT:%Y/%m/%d}" / ".done.json"
    asyncio.run(queue_on(tmp_path, scene, site).drain())
    seal(scene)  # 映射按槽之后章由换代封口时盖
    good = marker.read_bytes()
    publish_tonight(site, day=TODAY)  # 第二天又封口一天，同步词表有活（新的日历取值）
    marker.write_bytes(b"{broken")
    later = NOW + timedelta(days=1)
    reopened, _provider, _advisor = scene_on(tmp_path, site, advice=True)
    queue = queue_on(tmp_path, reopened, site, clock=lambda: later, sealed=lambda: later.date())
    with pytest.raises(ConceptHitStoreError):
        asyncio.run(queue.drain())
    assert queue.state.errors == {MAPPING: "ConceptHitStoreError", ADVICE: "ConceptHitStoreError"}
    assert marker.read_bytes() == b"{broken"
    marker.write_bytes(good)
    assert asyncio.run(queue.drain()).clear and queue.state.errors == {}


class SlowReduce:
    """一支的归约替身：第一次按 ``first`` 的做法（等放行 / 锁被占），之后做成；``landed`` 记归约落树时那一天结算过没有。"""

    def __init__(self, first: str, settled: list[str], landed: list[bool]) -> None:
        self.first = first
        self.runs = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.settled = settled
        self.landed = landed

    async def __call__(self) -> Reduction:
        self.runs += 1
        if self.runs == 1 and self.first == "busy":
            return Reduction.BUSY
        if self.runs == 1:
            self.entered.set()
            await self.release.wait()
        self.landed.append(bool(self.settled))  # 这一刻写回的判断落树
        return Reduction.DONE


@pytest.mark.parametrize("first", ["in-flight", "busy"])
def test_g6_a_written_back_item_still_waiting_for_its_reduction_defers_the_settlement(tmp_path, first: str) -> None:
    """G6（裁定 68 第 3 条：写回的判断要在那天的结算之前落树）：队列就近重试写回了昨天的一项、还在等归约——归约正做到一半
    （``in-flight``），或归约的锁被占、项留着（``busy``）——这时换代来了：

    - 重试收尾不动那一项（不撤回、不丢弃、不推"重试到哪一天"）；
    - 这一代的结算留到下次（``settlement_deferred`` 1、结算一次都没跑），其余各步照做、照写完成标记；
    - 之后归约把那条判断落了树，落树的那一刻昨天还没结算；
    - 昨天另一项没试成的仍会被就近试（日子没推），补回的判断同样落在结算之前；
    - 下一代：重试收尾推了日子，留下的结算补上。

    撤回即失败：重试收尾不管有没有在等归约的项都照丢、照结算，``abandoned`` 不空、结算在落树之前跑了。"""

    yesterday = TODAY - timedelta(days=1)
    site, scene, _provider, _advisor = fresh(tmp_path)
    chains = CountingChains()
    settled: list[str] = []
    landed: list[bool] = []
    reduce = SlowReduce(first, settled, landed)
    backlog = RetryBacklogStore(tmp_path / "backlog", max_items=10)
    stage = RetryStage([RetryBranch("session", backlog, {RetryKind.CHAIN: chains}, reduce)])  # type: ignore[dict-item]
    # 映射按槽之后映射不等封口：先让它做完，下面这一趟里就只有出错重试这一类有活（场景不变：项留着的时候换代来了）
    asyncio.run(queue_on(tmp_path, scene, site, sealed=lambda: None).drain())
    item = chain_item("昨天写回了的链", yesterday)
    backlog.put(item)
    clock = [NOW]
    queue = queue_on(tmp_path, scene, site, retry=stage, clock=lambda: clock[0], sealed=lambda: None)

    def generation() -> Generation:
        run, _days, _built = generation_on(
            tmp_path,
            site,
            scene,
            retry=stage,
            clock=lambda: clock[0],
            closed_days=sealed_through(clock[0].date() - timedelta(days=1)),
            models_down=queue.state.down,
        )
        run.settlement = StubSettlement(settled)  # type: ignore[assignment]
        return run

    async def first_day():  # type: ignore[no-untyped-def]
        draining = asyncio.create_task(queue.drain())
        if first == "in-flight":
            await reduce.entered.wait()
        else:
            await draining
        result = await generation().run(TODAY)
        snapshot = (list(chains.abandoned), backlog.count(), backlog.through(), list(settled), list(landed))
        late = chain_item("昨天另一条没试成的链", yesterday)
        backlog.put(late)  # 日子没推：昨天的项照收
        reduce.release.set()
        await draining
        await queue.drain()
        return result, snapshot, late

    result, snapshot, late = asyncio.run(first_day())
    abandoned, count, through, settled_then, landed_then = snapshot
    assert result.complete and result.held is None and generation().marks.finished(TODAY) is True
    assert abandoned == [] and count == 1 and through is None  # 重试收尾没动
    assert settled_then == [] and landed_then == []  # 这一代没结算；那条判断还没落树
    assert result.fallbacks["settlement_deferred"] == 1
    retry_step, settle_step = result.step(RETRY_STEP), result.step("settle")
    assert retry_step is not None and retry_step.outcome == DEGRADED and "等归约" in str(retry_step.reason)
    assert settle_step is not None and settle_step.outcome == "skipped"
    # 之后归约落了树：两项都在结算之前落的
    assert landed and not any(landed) and settled == []
    assert chains.attempted.count(item.item_id) == 1  # 写回的那一项没有再问一次模型
    assert late.item_id in chains.attempted and backlog.count() == 0 and chains.abandoned == []

    # 下一代：推日子、把留下的结算补上
    clock[0] = NOW + timedelta(days=1)
    following = generate(generation())
    assert following.complete and settled == ["结算"] and following.fallbacks["settlement_deferred"] == 0
    assert backlog.through() == TODAY
    assert following.step(RETRY_STEP).report["retry"]["recovered"] == 2  # type: ignore[index, union-attr]


def test_g7_items_left_by_a_kill_are_tried_again_after_a_restart_not_dropped_as_leftovers(tmp_path) -> None:
    """G7：就近试到一半进程被杀（一项正在试），重起后（新的重试编排对象，同一个堆）这一项照常再试、补回；到封口时计成补回，
    不是"上一次剩下的"直接丢。"""

    site, scene, _provider, _advisor = fresh(tmp_path)
    backlog_root = tmp_path

    class Killed(CountingChains):
        async def attempt(self, item: RetryItem) -> Attempt:
            self.attempted.append(item.item_id)
            raise asyncio.CancelledError

    item = chain_item("试到一半被杀的链", TODAY)
    stage = retry_stage(backlog_root, Killed())
    stage.branches[0].backlog.put(item)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(queue_on(tmp_path, scene, site, retry=stage, sealed=lambda: None).drain())
    assert stage.branches[0].backlog.count() == 1

    chains = CountingChains()
    restarted = retry_stage(backlog_root, chains)
    report = asyncio.run(queue_on(tmp_path, scene, site, retry=restarted, sealed=lambda: None).drain())
    assert others(report) == [(RETRY, DONE)] and chains.attempted == [item.item_id] and chains.abandoned == []
    closed = restarted.close(through=TODAY)
    assert (closed.recovered, closed.leftover, closed.unavailable) == (1, 0, 0)


# ── 第二轮修复 H1、H2、H3 ───────────────────────────────────────────────


def test_h2_a_round_cut_short_by_the_backoff_still_records_what_every_category_owes(tmp_path) -> None:
    """H2：整体退避让一轮中途到此为止（这里第一项模型调用失败、配成 1 项就整体退避）——排在后面的各类"还欠几项"仍从盘上认
    一遍再记：健康面与换代报告读到映射欠着，不是上一次的旧数。（原来的触发方式是这一轮开头探针不通；探针已去掉。）"""

    from habitus.runtime.model_queue import VocabularyBranch

    async def failing(_night):  # type: ignore[no-untyped-def]
        raise ModelTransportError("connection refused")

    site, scene, _provider, _advisor = fresh(tmp_path, advice=True)
    queue = queue_on(
        tmp_path,
        scene,
        site,
        vocabulary_branches=(VocabularyBranch("session", lambda _night: True, failing),),
        settings=QueueSettings(unreachable_after_failures=1),
    )
    report = asyncio.run(queue.drain())
    assert done(report) == [("vocabulary", UNREACHABLE)] and queue.state.down()
    assert report.owed[MAPPING] == 2  # 映射按条数：两条记录
    # C+D 改过：补问排在词表之后（长新类 → 写概念 → 先验 → 候选条件），词表还欠着时它不看、不算欠（材料还没定）；同步词表的
    # 概念只由换代写，这份现场换代之前队列不欠它。原来这里断言 sync == 1、advice >= 1
    assert report.owed[SYNC] == 0 and report.owed[ADVICE] == 0
    assert not mapped(scene, TONIGHT)  # 只认、不做


class PerItemChains(CountingChains):
    """链处理器的替身：按项的名字给不同的结局。"""

    def __init__(self, outcomes: dict[str, Attempt]) -> None:
        super().__init__()
        self.outcomes = outcomes

    async def attempt(self, item: RetryItem) -> Attempt:
        self.attempted.append(item.item_id)
        return self.outcomes[item.payload["records"][0]["name"]]


def test_h3_a_retry_item_stuck_on_timeouts_is_kept_and_does_not_stop_the_others_or_the_later_categories(
    tmp_path,
) -> None:
    """H3：出错重试里一项一直卡在"模型用不了"上（不分起因）——这一项留在堆里、不算试过、在进程里退避（不进落盘的重试账）；
    同一趟里映射、补问照做，堆里别的项也照试；退避期内不重问那一项。

    撤回即失败：出错重试一报"模型用不了"就整体退避，映射这一趟做不到。"""

    site, scene, _provider, advisor = fresh(tmp_path / "timeout", advice=True)
    chains = PerItemChains({"一直超时的链": Attempt.UNAVAILABLE, "好的链": Attempt.RESTORED})
    stage = retry_stage(tmp_path / "timeout", chains)
    backlog = stage.branches[0].backlog
    stuck, good = chain_item("一直超时的链", TODAY), chain_item("好的链", TODAY)
    backlog.put(stuck)
    backlog.put(good)
    clock = [NOW]
    queue = queue_on(tmp_path / "timeout", scene, site, retry=stage, clock=lambda: clock[0])
    report = asyncio.run(queue.drain())
    assert not queue.state.down() and report.backing_off_until is None
    assert (RETRY, UNREACHABLE) in done(report) and (MAPPING, DONE) in done(report) and (ADVICE, DONE) in done(report)
    assert mapped(scene, TONIGHT) and advisor.calls > 0  # 后面的类这一趟照做
    assert good.item_id in chains.attempted and not backlog._path(good.item_id).exists()  # 别的项照试、补回了
    names = [path.name for path in backlog.directory.iterdir() if path.name.startswith(stuck.item_id)]
    assert names and backlog.count() == 1 and chains.abandoned == []  # 卡住的那一项留在堆里
    assert report.owed[RETRY] == 1 and queue.retries.size() == 0
    tried = chains.attempted.count(stuck.item_id)
    again = asyncio.run(queue.drain())
    assert chains.attempted.count(stuck.item_id) == tried and again.done == ()  # 退避期内不重问
    clock[0] = NOW + timedelta(hours=2)
    chains.outcomes["一直超时的链"] = Attempt.RESTORED
    final = asyncio.run(queue.drain())
    assert done(final) == [(RETRY, DONE)] and backlog.count() == 0 and final.clear
    assert stage.close(through=TODAY).recovered == 2  # 那一项不算试坏过：补回计一次


# ── 通用规则（第二次复验确认的两处）：模型调用失败不磨项，整体退避只看连着几项没成 ──────────────


def test_a_classification_timeout_swallowed_by_the_reduction_does_not_back_the_whole_queue_off(tmp_path) -> None:
    """复验 A：出错重试里放回的链，归约给它归类时超时——归约不抛，只在报告里记"模型用不了"。按一次模型调用失败算：
    只卡住的那一项在进程里退避，队列不进整体退避；140 分钟里一趟一趟来，映射与补问第一趟就做完，那一项一直留在堆里、
    没有被放弃。

    撤回即失败：归约报出的"模型用不了"直接让队列整体退避，映射与补问一项都做不到。"""

    from types import SimpleNamespace

    from habitus.behavior.kinds.ids import Lane
    from habitus.behavior.reduction import BehaviorReductionReport
    from habitus.runtime.behavior import _reduce

    reductions: list[str] = []

    async def run_once() -> BehaviorReductionReport:
        reductions.append("reduce")
        return BehaviorReductionReport(0, 0, 0, 0, (), unavailable=1)  # 归类超时被归约吞了

    site, scene, _provider, advisor = fresh(tmp_path, advice=True)
    chains = CountingChains(good=False)  # 归类没成：放回的链没有落树
    backlog = RetryBacklogStore(tmp_path / "backlog", max_items=10)
    branch = SimpleNamespace(lane=Lane.SESSION, reduction_runner=SimpleNamespace(run_once=run_once))
    stage = RetryStage([RetryBranch("session", backlog, {RetryKind.CHAIN: chains}, _reduce(branch))])  # type: ignore[arg-type,dict-item]
    item = chain_item("归类时超时的链", TODAY)
    backlog.put(item)
    clock = [NOW]
    queue = queue_on(tmp_path, scene, site, retry=stage, clock=lambda: clock[0])
    first = asyncio.run(queue.drain())
    assert (RETRY, UNREACHABLE) in done(first) and not queue.state.down() and first.backing_off_until is None
    assert (MAPPING, DONE) in done(first) and (ADVICE, DONE) in done(first)  # 后面的类第一趟就做了
    assert mapped(scene, TONIGHT) and advisor.calls > 0
    for minutes in range(10, 141, 10):
        clock[0] = NOW + timedelta(minutes=minutes)
        report = asyncio.run(queue.drain())
        assert not queue.state.down() and report.backing_off_until is None, minutes
        assert report.owed[RETRY] == 1 and report.owed[MAPPING] == 0 and report.owed[ADVICE] == 0, minutes
    assert backlog.count() == 1 and chains.abandoned == [] and queue.retries.size() == 0  # 留着，没有被放弃
    assert 1 < len(reductions) < 14  # 退避逐次加倍：不是每一趟都重问


@pytest.mark.parametrize("kind", ["transport", "quota"])
def test_twenty_minutes_of_failing_calls_give_nothing_up_and_everything_is_done_afterwards(tmp_path, kind: str) -> None:
    """复验 B，按那一条通用规则：所有模型调用都失败（不分错误种类——429 / 503 那一类与额度那一类各跑一遍）持续二十分钟，
    每分钟看一次——没有一项被记到放弃（两本重试账都是空的），连着几项没成之后队列整体退避（不逐条去撞），恢复后那一天的
    映射与那一晚的补问都做完，盘面与没出过事的相同。

    撤回即失败：模型调用失败按重试账记这一项，次数用完后映射那一天与补问被放弃，恢复后不补。"""

    from habitus.model_client import ModelQuotaError

    def failure() -> Exception:
        return ModelTransportError("503 service unavailable") if kind == "transport" else ModelQuotaError("no balance")

    expected = masked(straight(tmp_path / "straight", advice=True))
    root = tmp_path / "failing"
    clock = [NOW]
    site, scene, provider, advisor = fresh(root, advice=True)
    good_map, good_advice = provider.complete_async, advisor.complete_async
    out = [True]

    async def map_answer(request):  # type: ignore[no-untyped-def]
        if out[0]:
            provider.calls += 1
            raise failure()
        return await good_map(request)

    async def advice_answer(request):  # type: ignore[no-untyped-def]
        if out[0]:
            advisor.calls += 1
            raise failure()
        return await good_advice(request)

    provider.complete_async, advisor.complete_async = map_answer, advice_answer
    queue = queue_on(root, scene, site, clock=lambda: clock[0], settings=QueueSettings(unreachable_after_failures=2))
    backed_off = False
    tries = 0
    for minute in range(0, 21):
        clock[0] = NOW + timedelta(minutes=minute)
        report = asyncio.run(queue.drain())
        tries += len(report.done)
        backed_off = backed_off or queue.state.down()
        assert report.owed[MAPPING] == 1 and report.owed[ADVICE] >= 1 and not report.clear, minute  # 一直欠着、看得见
        assert queue.retries.size() == 0, minute  # 模型调用失败不进落盘的重试账
    assert backed_off and not mapped(scene, TONIGHT)
    # 整体退避 + 各项退避：不是每分钟每项都去撞一次。C+D 改过：原来数的是模型调用次数（< 42）；先验一项里一题要问三遍、
    # 新旧两种问法各有题，补问又排到映射前面先撞，一次尝试的调用数变了——改数"试了几项"，这才是这条要守的
    assert tries < 21 * 2

    out[0] = False
    clock[0] = NOW + timedelta(hours=2)
    final = asyncio.run(queue.drain())
    assert final.clear and not queue.state.down() and mapped(scene, TONIGHT)
    assert queue.retries.given_up() == ()
    assert masked(disk(root)) == expected
