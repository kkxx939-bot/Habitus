"""预测层的组合根接线：可选启用、钉住一代、此刻按主体时区落钟面、窗口只有一处出处、每槽一拍与同槽复用。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

from habitus.behavior import BehaviorDocumentWriter
from habitus.behavior.model import BehaviorKind
from habitus.behavior.tree import BehaviorTree
from habitus.config import HabitusConfig
from habitus.config.loader import ConfigError
from habitus.foresight import ForesightError, UnsealedRow, moment_at
from habitus.foresight.judge import Judgement
from habitus.foundation.observability import ObservationEvent
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.runtime.foresight import ForesightWorker, build_foresight_components
from habitus.runtime.prediction import build_prediction_components
from habitus.scene import AssociationLedger
from habitus.scene.regularity import RegularityTree
from tests.integration.test_runtime_assembly import REPOSITORY_ROOT
from tests.unit.behavior.tree_payloads import occurrence_payload
from tests.unit.foresight.fixtures import Ledger, ScriptedJudge
from tests.unit.runtime.fixtures import STARTUP_PARAMETERS
from tests.unit.runtime.test_behavior_pipeline import SUBJECT

CST = timezone(timedelta(hours=8))
FIRST = datetime(2026, 8, 3, tzinfo=CST)  # 周一
EVENING = FIRST + timedelta(days=21, hours=19, minutes=5)


def raw_config(tmp_path: Path, *, scene: bool = True, foresight: bool = True, **overrides: Any) -> dict[str, Any]:
    """example.yaml 之上把路由换成 fake，并把行为 / 预测 / 情景 / 预测层四组填满。"""

    raw = yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))
    raw["storage"]["root"] = str(tmp_path / "data")
    for route, adapter in (
        (raw["models"]["chat"]["route"], "fake_chat"),
        (raw["models"]["embedding"]["route"], "fake_embedding"),
        (raw["models"]["rerank"]["route"], "fake_rerank"),
        (raw["memory"]["vector_store"]["route"], "fake_vector"),
        (raw["conversation"]["summary_vector_store"]["route"], "fake_vector"),
    ):
        route.update(provider="fake", adapter=adapter, credential_ref="")
    raw["behavior"] = {"primary_subject": SUBJECT}
    raw["prediction"] = dict(STARTUP_PARAMETERS)
    raw["scene"] = {"enabled": scene}
    raw["foresight"] = {"enabled": foresight, **overrides}
    return raw


def trees(config: HabitusConfig, *, weeks: int = 4) -> tuple[BehaviorTree, RegularityTree, AssociationLedger]:
    """一棵有"每周一晚上打球"的行为树，外加一棵（尚未关联的）规律树与按版本读它的事实源。"""

    behavior_tree = BehaviorTree(config.behavior_root / "tree")
    writer = BehaviorDocumentWriter(
        behavior_tree, ProcessLocalLockStore(), clock=lambda: FIRST + timedelta(days=7 * weeks)
    )
    for week in range(weeks):
        started = FIRST + timedelta(days=7 * week, hours=19)
        writer.publish(
            BehaviorKind.OCCURRENCE,
            occurrence_payload(
                occurred_on=started.date(),
                name="打球",
                kind_token="打球",
                started_at=started,
                last_observed_at=started + timedelta(minutes=60),
                onset_available_at=started + timedelta(seconds=2),
                basis=(),
                goal=None,
            ),
        )
    regularity_tree = RegularityTree(config.scene_root / "regularity")
    regularity_tree.initialize()
    return behavior_tree, regularity_tree, Ledger(regularity_tree, "test_association_v1")


def scripted_judge() -> ScriptedJudge:
    """不调模型的判断者：每包一条空判断，记下收到的包。"""

    script = Judgement(
        judged_at=EVENING.astimezone(UTC),
        generation="script",
        moment=moment_at(EVENING, slot_minutes=15),
        verdicts=(),
        day_state="正常",
        day_note=None,
        judge_version="scripted-judge",
    )
    return ScriptedJudge(script)


def assembled(tmp_path: Path, *, now: datetime = EVENING, judge: ScriptedJudge | None = None):
    """走完真实顺序：行为树 → 发布一代预测树 → 组装预测层（判断者是脚本化的）。"""

    config = HabitusConfig.from_mapping(raw_config(tmp_path))
    behavior_tree, regularity_tree, associated = trees(config)
    prediction = build_prediction_components(config, behavior_tree=behavior_tree, clock=lambda: now)
    assert prediction is not None
    asyncio.run(prediction.worker.run_once())
    components = build_foresight_components(
        config,
        behavior_tree=behavior_tree,
        regularity_tree=regularity_tree,
        associated=associated,
        store=prediction.store,
        judge=judge if judge is not None else scripted_judge(),
        clock=lambda: now,
    )
    assert components is not None
    return config, components


def test_foresight_is_absent_until_it_is_switched_on(tmp_path) -> None:
    config = HabitusConfig.from_mapping(raw_config(tmp_path, foresight=False))
    behavior_tree, regularity_tree, associated = trees(config)
    assert config.foresight.enabled is False
    assert (
        build_foresight_components(
            config,
            behavior_tree=behavior_tree,
            regularity_tree=regularity_tree,
            associated=associated,
            store=None,
            judge=scripted_judge(),
        )
        is None
    )


def test_an_enabled_foresight_layer_must_have_a_judge(tmp_path) -> None:
    """判断者二选一（注入的 judge 或结构化客户端）；两个都不给不是"暂时不判"，是接线漏了。"""

    config = HabitusConfig.from_mapping(raw_config(tmp_path))
    behavior_tree, regularity_tree, associated = trees(config)
    prediction = build_prediction_components(config, behavior_tree=behavior_tree)
    assert prediction is not None
    with pytest.raises(ValueError, match="needs a judge"):
        build_foresight_components(
            config,
            behavior_tree=behavior_tree,
            regularity_tree=regularity_tree,
            associated=associated,
            store=prediction.store,
        )


def test_enabling_foresight_without_the_semantic_layer_is_refused(tmp_path) -> None:
    """配置自相矛盾要在启动时炸：数字取自预测树、与之对应的历史取自语义层，缺一边装配不出证据。

    放过去的下场是启动一切正常、健康面报 foresight 已启用、什么都没发生、无处可查。
    """

    with pytest.raises(ConfigError, match="config.foresight is enabled"):
        HabitusConfig.from_mapping(raw_config(tmp_path, scene=False))


def test_the_window_has_one_source_the_prediction_tree(tmp_path) -> None:
    """窗口宽度与转移窗只从预测树的参数来，预测层这一组只有保护闸。

    配两份的下场是数字按 ±3 槽算、背景按另一个宽度取，两边都对不上却都不报错。
    """

    config, components = assembled(tmp_path)
    assembler = components.assembler
    assert assembler.half_width == config.prediction.pool_half_width == 3
    assert assembler.transition_window_seconds == config.prediction.transition_window_seconds
    assert assembler.max_days_per_layer == config.foresight.max_days_per_layer
    assert assembler.window_days == config.foresight.window_days


def test_the_moment_lands_on_the_clock_face_in_the_subject_timezone(tmp_path) -> None:
    """传 UTC 进来也要先换回主体本地时刻再落钟面——错一个时区就整体错一天、错一个槽。"""

    _config, components = assembled(tmp_path)
    pack = components.assembler.assemble(now=EVENING.astimezone(UTC))
    moment = pack.moment
    assert moment.at.utcoffset() == timedelta(hours=8)
    assert (moment.weekday, moment.slot) == (0, 76)  # 周一 19:00–19:15
    assert moment.day_note is None  # 没有当地日历数据时是显式的空，不是漏了字段
    assert pack.generation and pack.now.moment == moment
    (candidate,) = pack.expanded
    assert candidate.kind_token == "打球"
    # 规律树一天都没关联，但**卡照样是满的**：序列与视图来自行为树，语义层做没做过不影响。
    # 没关联的日子由 unassociated 如实摆出来，卡上关联那一栏说"那天还没关联"。
    assert candidate.unassociated == candidate.provenance.all_day.days
    cards = candidate.background.cards
    assert [card.at.date() for card in cards] == list(candidate.provenance.slot.days)
    assert all(card.gloss is None and not card.day_associated for card in cards)
    # 今天 19:00 那次已经在树上，此刻 19:05 的场景里有它；没注入判断存储时未封口明说没补，不是漏了。
    assert [row.name for row in pack.now.flow] == ["打球"] and pack.now.unsealed == ()
    assert pack.now.done_today == {"打球": 1}


def test_a_generation_built_with_other_parameters_is_refused(tmp_path) -> None:
    """跨参数混读永不允许：组合出来的数字互不一致，而且看不出来。"""

    _config, components = assembled(tmp_path)
    components.assembler.expected_digest = "另一套参数"
    with pytest.raises(ForesightError, match="different estimation"):
        components.assembler.assemble(now=EVENING)


def test_without_a_published_generation_it_says_so_instead_of_answering_zero(tmp_path) -> None:
    """"还没算过"与"什么都不会发生"必须分得清——后者会让上层安心闭嘴。"""

    config = HabitusConfig.from_mapping(raw_config(tmp_path))
    behavior_tree, regularity_tree, associated = trees(config)
    prediction = build_prediction_components(config, behavior_tree=behavior_tree)
    assert prediction is not None
    components = build_foresight_components(
        config,
        behavior_tree=behavior_tree,
        regularity_tree=regularity_tree,
        associated=associated,
        store=prediction.store,
        judge=scripted_judge(),
    )
    assert components is not None
    with pytest.raises(ForesightError, match="no prediction generation"):
        components.assembler.assemble(now=EVENING)
    # 一拍失败留在 worker 的 last_error 与观测里，下一槽照常；不吞掉。
    events: list[ObservationEvent] = []
    components.worker.observer = _Recorder(events)  # type: ignore[assignment]
    with pytest.raises(ForesightError):
        asyncio.run(components.worker.run_once())
    assert isinstance(components.worker.last_error, ForesightError)
    assert [(event.operation, event.status.value) for event in events] == [("judgement", "failure")]


class _Recorder:
    def __init__(self, events: list[ObservationEvent]) -> None:
        self.events = events

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)


class _Rows:
    """脚本化的未封口读口。"""

    def __init__(self, *rows: UnsealedRow) -> None:
        self._rows = rows

    def rows(self, *, since: datetime, until: datetime) -> tuple[UnsealedRow, ...]:
        return self._rows


def test_the_same_slot_with_an_unchanged_scene_reuses_the_judgement(tmp_path) -> None:
    """同一槽内此刻场景没变就不再调模型；场景一变、槽一换，都重新判。"""

    judge = scripted_judge()
    _config, components = assembled(tmp_path, judge=judge)
    runner = components.runner

    first = asyncio.run(runner.run_once(now=EVENING))
    assert not first.reused and len(judge.packs) == 1
    # 两分钟后，同槽、树上没有新行、判断存储没有新链：复用，判断对象就是上一次那个。
    again = asyncio.run(runner.run_once(now=EVENING + timedelta(minutes=2)))
    assert again.reused and again.judgement is first.judgement and len(judge.packs) == 1
    assert runner.last is again
    # 判断存储里多了一条未封口的行：场景变了，重新判。
    components.assembler.unsealed = _Rows(
        UnsealedRow(name="换鞋", kind_token=None, started_at=EVENING + timedelta(minutes=1), last_observed_at=EVENING + timedelta(minutes=3), summary="换了鞋")
    )
    changed = asyncio.run(runner.run_once(now=EVENING + timedelta(minutes=4)))
    assert not changed.reused and len(judge.packs) == 2
    assert [row.name for row in changed.pack.now.unsealed] == ["换鞋"]
    # 下一个槽：候选可能不同，重新判。
    next_slot = asyncio.run(runner.run_once(now=EVENING + timedelta(minutes=12)))
    assert not next_slot.reused and next_slot.pack.moment.slot == 77 and len(judge.packs) == 3


def test_the_worker_ticks_on_slot_boundaries_in_the_subject_timezone(tmp_path) -> None:
    """节奏就是树的槽宽：19:05 距下一槽（19:15）600 秒；正好在边界上算下一个整槽。"""

    _config, components = assembled(tmp_path)
    worker = components.worker
    assert isinstance(worker, ForesightWorker) and worker.slot_minutes == 15
    assert worker.seconds_until_next_slot() == 600.0
    worker._clock = lambda: EVENING.replace(minute=15)  # noqa: SLF001 - 测试拨钟
    assert worker.seconds_until_next_slot() == 900.0
    worker._clock = lambda: EVENING.astimezone(UTC)  # 传 UTC 进来也按主体本地时区算边界
    assert worker.seconds_until_next_slot() == 600.0
    # 边界从**这一拍开始**的时刻算：19:13 起拍、跑了两分钟到 19:15，下一拍是 19:15 那一槽（等 0 秒），不是 19:30。
    assert worker.seconds_until_next_slot(at=EVENING.replace(minute=13)) == 120.0

    events: list[ObservationEvent] = []
    worker.observer = _Recorder(events)  # type: ignore[assignment]
    run = asyncio.run(worker.run_once())
    assert not run.reused and run.judgement.generation == run.pack.generation
    (event,) = events
    assert (event.category, event.operation, event.status.value) == ("foresight", "judgement", "success")
    assert event.attributes["reused"] is False and event.attributes["expanded"] == 1 and event.attributes["slot"] == 76


def test_the_builder_refuses_two_sources_for_the_same_part(tmp_path) -> None:
    """判断者与未封口读口都是二选一：两个都给不是"多一份保险"，是接线错误。"""

    config = HabitusConfig.from_mapping(raw_config(tmp_path))
    behavior_tree, regularity_tree, associated = trees(config)
    prediction = build_prediction_components(config, behavior_tree=behavior_tree)
    assert prediction is not None
    common = dict(behavior_tree=behavior_tree, regularity_tree=regularity_tree, associated=associated, store=prediction.store)
    with pytest.raises(ValueError, match="not both"):
        build_foresight_components(config, judge=scripted_judge(), structured_chat=object(), **common)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not both"):
        build_foresight_components(config, judge=scripted_judge(), unsealed=_Rows(), judgements=object(), **common)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="needs the judgement store"):
        build_foresight_components(config, judge=scripted_judge(), judgements=object(), **common)  # type: ignore[arg-type]


def test_the_worker_loop_judges_on_start_and_stops_cleanly(tmp_path) -> None:
    judge = scripted_judge()
    _config, components = assembled(tmp_path, judge=judge)

    async def scenario() -> None:
        await components.worker.start()
        # 第一拍在线程里装配整包（解码一代树、读行为树），整条 CI 并跑时不止几十毫秒；等它到，或等它报错。
        for _ in range(1_000):
            if judge.packs or components.worker.last_error is not None:
                break
            await asyncio.sleep(0.01)
        assert components.worker.last_error is None
        assert components.worker.running and len(judge.packs) == 1
        await components.worker.stop()
        assert not components.worker.running and components.worker.last_error is None

    asyncio.run(scenario())


def test_a_naive_now_is_refused_instead_of_being_read_in_the_process_time_zone(tmp_path) -> None:
    _config, components = assembled(tmp_path)
    with pytest.raises(ForesightError, match="timezone-aware"):
        components.assembler.assemble(now=EVENING.replace(tzinfo=None))


def test_records_are_read_with_the_same_version_the_ledger_judges_by(tmp_path) -> None:
    """规律树上按 v1 关联了周一那次：事实源按 v1 读 → 卡上有记录；事实源按 v2 读 → 那天没关联、卡上没记录。
    两边只有一个版本旋钮，不可能一边有一边没有。"""

    from tests.unit.scene.fixtures import associate

    config = HabitusConfig.from_mapping(raw_config(tmp_path))
    behavior_tree, regularity_tree, _ledger = trees(config)
    first = "behavior://occurrences/2026/08/03/打球--20260803T190000000000%2B0800.md"
    associate(regularity_tree, first, kind="打球", context="第一周", version="v1")
    prediction = build_prediction_components(config, behavior_tree=behavior_tree, clock=lambda: EVENING)
    assert prediction is not None
    asyncio.run(prediction.worker.run_once())

    def pack_with(version: str):
        components = build_foresight_components(
            config,
            behavior_tree=behavior_tree,
            regularity_tree=regularity_tree,
            associated=Ledger(regularity_tree, version),
            store=prediction.store,
            judge=scripted_judge(),
            clock=lambda: EVENING,
        )
        assert components is not None
        return components.assembler.assemble(now=EVENING)

    (v1,) = pack_with("v1").expanded
    (v2,) = pack_with("v2").expanded
    assert v1.background.cards[0].gloss is not None and v1.background.cards[0].day_associated
    assert v2.background.cards[0].gloss is None and not v2.background.cards[0].day_associated
    assert v1.unassociated != v2.unassociated
