"""概念层只留"词表节点 + 日历"（2026-11-14）的新行为测试（预期表 D1、D4、D5、D9、D10 与场景 C1、C2、C5、C6）。

这些测试在改动前的代码（``9f23a13``）上应当失败，改动后应当通过。

几条测试在 ``scene/concepts-pending/`` 下放一份旧盘面留下的"待生效概念"（模型写的）：改动前换代会把它并进概念集、映射会为它问
模型；改动后概念集只由词表与日历生成，没有任何一步读这个目录。旧格式的盘面文件取自 04-28 那一代的真实盘面
（``tests/fixtures/scene_legacy``，来源 ``~/Desktop/Habitus/results/2026-04-27-c55f7da``）：概念文件、命中记录、日完成章、关系表、
先验缓存、机会表。改动后它们一律是旧版本，读到要明确报错（``ValueError`` 或子类，信息里点得出是哪个文件）。
"""

from __future__ import annotations

import re
import shutil
from datetime import date, datetime, time
from pathlib import Path

import pytest

from habitus.config import ConfigError, HabitusConfig
from habitus.runtime.foresight_opportunities import OpportunityStore
from habitus.runtime.generation_summary import scene_summary
from habitus.runtime.model_queue_health import queue_pending
from habitus.runtime.model_queue_state import CATEGORIES, ModelQueueState
from habitus.scene.advice.store import AdviceStore
from habitus.scene.concepts.calendar_situations import calendar_key
from habitus.scene.concepts.store import ConceptStore
from habitus.scene.occurrences.store import ConceptHitStore
from habitus.scene.relations.store import RelationStore
from tests.series_helpers import read_series
from tests.unit.runtime.concept_layer_site import (
    CLASSES,
    CST,
    EXPENSE,
    LEGACY,
    RUN,
    SUBJECT,
    ModelMustNotBeCalled,
    example_mapping,
    generate,
    hits_on,
    scene_run,
)
from tests.unit.scene.fixtures import Site, publish

WEDNESDAY = date(2026, 4, 22)
NIGHT = date(2026, 4, 23)


def at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=CST)


def scene_root(tmp_path: Path) -> Path:
    return tmp_path / "data" / "scene"


def leave_pending(tmp_path: Path, name: str) -> None:
    """旧盘面留下的一份待生效概念（模型写的）。"""

    target = scene_root(tmp_path) / "concepts-pending"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(LEGACY / "concepts" / f"{name}.md", target / f"{name}.md")


def a_wednesday(tmp_path: Path) -> Site:
    site = Site(tmp_path, now=at(NIGHT, 1))
    publish(site.behavior_tree, WEDNESDAY, "晨跑", 6, 10, kind=RUN, lasts_minutes=40)
    publish(site.behavior_tree, WEDNESDAY, "补记一笔支出", 21, 28, kind=EXPENSE)
    return site


# ── D1 / C1：概念集只有"每个词表类一个行为概念"与日历情境 ─────────────────────────


def test_d1_c1_after_a_generation_the_concept_set_is_only_class_concepts_and_calendar(tmp_path: Path) -> None:
    site = a_wednesday(tmp_path)
    leave_pending(tmp_path, "居住空间场景")
    run = scene_run(tmp_path, site, ModelMustNotBeCalled())
    generate(run, NIGHT, at(NIGHT, 1))
    concepts = run.concepts.read_all()
    classes = {identity for identity in concepts if concepts.base_for(identity) == identity}
    calendar = {identity for identity in concepts if calendar_key(concepts, identity) is not None}
    assert classes == set(CLASSES)
    assert {concepts[identity].name for identity in calendar} == {"周三", "22号"}
    assert set(concepts) == classes | calendar, f"other concepts: {sorted(set(concepts) - classes - calendar)}"


# ── D4 / C2：换代把一天封口、建好命中记录，对话模型调用 0 次 ─────────────────────


def test_d4_c2_a_generation_seals_a_day_with_hit_records_and_never_calls_the_dialogue_model(tmp_path: Path) -> None:
    site = a_wednesday(tmp_path)
    leave_pending(tmp_path, "补记支出")  # 旧盘面里挂在 s-k0016 上、要模型判的细分概念
    client = ModelMustNotBeCalled()
    run = scene_run(tmp_path, site, client)
    report = generate(run, NIGHT, at(NIGHT, 1))
    assert client.names == []
    assert run.hits.days_done() == frozenset({WEDNESDAY})
    assert report.sealed_days == 1
    expected = {record.uri for record in read_series(run.behavior_tree, cutoff=NIGHT).on(WEDNESDAY)}
    assert {record.occurrence_uri for record in hits_on(run, WEDNESDAY)} == expected
    assert len(expected) == 2


# ── D9：换代的语义树一步里没有"并入待生效的概念"、不写这一代的节律 ─────────────────


def test_d9_a_generation_does_not_merge_pending_concepts(tmp_path: Path) -> None:
    site = a_wednesday(tmp_path)
    leave_pending(tmp_path, "居住空间场景")
    run = scene_run(tmp_path, site, ModelMustNotBeCalled())
    report = generate(run, NIGHT, at(NIGHT, 1))
    assert "居住空间场景" not in run.concepts.read_all()
    assert "并入" not in report.summary()
    assert not any("待生效" in signal for signal in report.signals)


def test_d9_the_generation_report_has_no_merged_concepts(tmp_path: Path) -> None:
    site = a_wednesday(tmp_path)
    leave_pending(tmp_path, "居住空间场景")
    run = scene_run(tmp_path, site, ModelMustNotBeCalled())
    report = generate(run, NIGHT, at(NIGHT, 1))
    # 换代报告文件里语义树那一段（``generation/reports``）就是这一份
    assert "merged_concepts" not in scene_summary(report)


def test_d9_a_generation_does_not_write_rhythms(tmp_path: Path) -> None:
    site = a_wednesday(tmp_path)
    run = scene_run(tmp_path, site, ModelMustNotBeCalled())
    generate(run, NIGHT, at(NIGHT, 1))
    assert not (scene_root(tmp_path) / "generation" / "rhythms").exists()


# ── D5 / C6：模型队列没有"写概念"与"映射"这两类 ─────────────────────────────────


@pytest.mark.parametrize("category", ["concepts", "mapping"])
def test_d5_c6_the_model_queue_has_no_concepts_or_mapping_category(category: str) -> None:
    assert category not in CATEGORIES
    assert category not in ModelQueueState().owed_now()


def test_d5_the_health_reading_has_no_pending_concepts() -> None:
    state = ModelQueueState()
    state.pending.update({"revisions": 1, "concepts": 1, "priors_by_consequent": 1, "priors_by_antecedent": 1})
    assert "concepts" not in queue_pending(state)


# ── D5：配置里不再有映射余量、常态容差与"判不了"的占比 ─────────────────────────────


@pytest.mark.parametrize(
    ("scene", "field"),
    [
        ({"mapping_margin_slots": 2}, "mapping_margin_slots"),
        ({"baseline_slack_slots": 3}, "baseline_slack_slots"),
        ({"relations": {"max_blind_share": 0.5}}, "max_blind_share"),
    ],
)
def test_d5_the_retired_scene_settings_are_refused_by_the_config(
    tmp_path: Path, scene: dict[str, object], field: str
) -> None:
    payload = example_mapping(tmp_path)
    payload["behavior"] = {"primary_subject": SUBJECT}
    payload["scene"] = {"enabled": True, **scene}
    with pytest.raises(ConfigError, match=field):
        HabitusConfig.from_mapping(payload)


# ── D10 / C5：旧形状的概念文件、命中记录读取时报错 ──────────────────────────────

LEGACY_CONCEPTS = (
    "晨跑训练",  # 细分概念 + 数值规则（按开始时刻切）
    "晚间生活缴费",  # 细分概念 + 规则 + 档
    "补记支出",  # 细分概念，要模型判
    "食品采购",  # 汇总概念
    "连续运动打卡",  # 派生情境（连续）
    "前日已跑步",  # 派生情境（昨天）
    "居住空间场景",  # 地点情境
)


@pytest.mark.parametrize("name", LEGACY_CONCEPTS)
def test_d10_an_old_shaped_concept_file_is_refused_naming_the_file(tmp_path: Path, name: str) -> None:
    store = ConceptStore(tmp_path / "scene")
    store.initialize()
    shutil.copyfile(LEGACY / "concepts" / f"{name}.md", store.directory / f"{name}.md")
    with pytest.raises(ValueError, match=name):
        store.read(name)


def test_d10_c5_reading_a_concept_set_holding_the_morning_run_refinement_fails_naming_it(tmp_path: Path) -> None:
    store = ConceptStore(tmp_path / "scene")
    store.initialize()
    shutil.copyfile(LEGACY / "concepts" / "晨跑训练.md", store.directory / "晨跑训练.md")
    with pytest.raises(ValueError, match="晨跑训练"):
        store.read_all()


LEGACY_HITS = (
    (date(2026, 4, 13), "下单日式便当外送--20260413T114700000000+0800"),  # 带细分命中（hits）与常态快照字段
    (date(2026, 4, 14), "补记一笔支出--20260414T212800000000+0800"),  # 带未决（unresolved）
)


@pytest.mark.parametrize(("day", "leaf"), LEGACY_HITS)
def test_d10_an_old_shaped_hit_record_is_refused_naming_the_file(tmp_path: Path, day: date, leaf: str) -> None:
    store = ConceptHitStore(tmp_path / "scene", lane="session")
    folder = tmp_path / "scene" / "occurrences" / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}"
    folder.mkdir(parents=True)
    shutil.copyfile(LEGACY / "occurrences" / f"{leaf}.md", folder / f"{leaf}.md")
    with pytest.raises(ValueError, match=re.escape(leaf)):
        store.read_day(day)


# ── D10：日完成章、关系表、先验缓存、机会表的旧版本读取时报错 ─────────────────────────


def test_d10_an_old_version_day_seal_is_refused_naming_the_file(tmp_path: Path) -> None:
    store = ConceptHitStore(tmp_path / "scene", lane="session")
    folder = tmp_path / "scene" / "occurrences" / "2026" / "04" / "20"
    folder.mkdir(parents=True)
    shutil.copyfile(LEGACY / "seals" / "2026" / "04" / "20" / ".done.json", folder / ".done.json")
    with pytest.raises(ValueError, match=r"\.done\.json"):
        store.days_done()


def test_d10_an_old_version_relation_table_is_refused_naming_the_file(tmp_path: Path) -> None:
    store = RelationStore(tmp_path / "scene")
    folder = tmp_path / "scene" / "relations" / "session"
    folder.mkdir(parents=True)
    shutil.copyfile(LEGACY / "relations" / "session" / "2026-04-27.json", folder / "2026-04-27.json")
    with pytest.raises(ValueError, match="2026-04-27"):
        store.read("session", date(2026, 4, 27))


PRIOR = "53bd11a8783f564fa8a8fe2d1b9a830f13aa306966d32fa79e74775ca087e52c"
PRIOR_BY_ANTECEDENT = "4bbb63f5421a487d1521e9bbaa8af665ef5f68d4bcc0fd3151391d6dcaf4ddbe"


def test_d10_an_old_version_prior_cache_is_refused_naming_the_file(tmp_path: Path) -> None:
    store = AdviceStore(tmp_path / "scene")
    folder = tmp_path / "scene" / "advice" / "priors"
    folder.mkdir(parents=True)
    shutil.copyfile(LEGACY / "advice" / "priors" / f"{PRIOR}.json", folder / f"{PRIOR}.json")
    with pytest.raises(ValueError, match=PRIOR):
        store.prior(PRIOR)


def test_d10_an_old_version_prior_cache_by_antecedent_is_refused_naming_the_file(tmp_path: Path) -> None:
    store = AdviceStore(tmp_path / "scene")
    folder = tmp_path / "scene" / "advice" / "priors-by-antecedent"
    folder.mkdir(parents=True)
    name = f"{PRIOR_BY_ANTECEDENT}.json"
    shutil.copyfile(LEGACY / "advice" / "priors-by-antecedent" / name, folder / name)
    with pytest.raises(ValueError, match=PRIOR_BY_ANTECEDENT):
        store.prior_by_antecedent(PRIOR_BY_ANTECEDENT)


def test_d10_an_old_version_opportunity_table_is_refused_naming_the_file(tmp_path: Path) -> None:
    store = OpportunityStore(tmp_path / "foresight")
    folder = tmp_path / "foresight" / "opportunities" / "session"
    folder.mkdir(parents=True)
    shutil.copyfile(LEGACY / "opportunities" / "session" / "2026-04-26.json", folder / "2026-04-26.json")
    with pytest.raises(ValueError, match="2026-04-26"):
        store.read("session", date(2026, 4, 26))
