"""概念层只留"词表节点 + 日历"（2026-11-14）的对抗测试：盘上的概念文件、命中记录、日完成章被改坏或带着旧字段时一律拒收。

每条都从一份**新代码自己写出的合法文件**出发，多数只改尾部 JSON 的一个字段，再读：报错必须是 ``ValueError``（或子类），信息里
点得出是哪个文件。只改一个字段，是为了让"只拒某一个旧字段、放过别的"的实现也守不住（旧盘面的真实文件往往同时带好几个旧字段，
拿它们测不出这一点）。

守的是"读不出来、点得出文件"这个行为，不是哪一道检查拒的：正文不动，所以字段集合检查与"按字段重新渲染、逐字节比对"两道都会拒；
只放开其中一道的实现照样拒收（行为仍对），两道一起放开才会让这里变红。另有几条按名字就不是"一处"：真实的旧日历情境文件
（整份旧形状）、同一个类第二个行为概念（另一个文件 + 改名）、叶名不是规范身份（换掉整个目录的文件名）。
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import pytest

from habitus.foundation.integrity import canonical_json
from habitus.scene.concepts.store import ConceptStore
from habitus.scene.occurrences.store import DONE_FILENAME, ConceptHitStore
from tests.unit.runtime.concept_layer_site import CST, EXPENSE, LEGACY, RUN, ModelMustNotBeCalled, generate, scene_run
from tests.unit.scene.fixtures import Site, publish

WEDNESDAY = date(2026, 4, 22)
NIGHT = date(2026, 4, 23)
CONCEPT_MARKER = "\n<!-- HABITUS_CONCEPT\n"
HITS_MARKER = "\n<!-- HABITUS_CONCEPT_HITS\n"
FOOTER = "\n-->\n"


def at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=CST)


def sealed_wednesday(tmp_path: Path) -> Path:
    """一次换代把周三（两条记录）封口，返回语义树根：概念、命中记录、日完成章都是新代码自己写的。"""

    site = Site(tmp_path, now=at(NIGHT, 1))
    publish(site.behavior_tree, WEDNESDAY, "晨跑", 6, 10, kind=RUN, lasts_minutes=40)
    publish(site.behavior_tree, WEDNESDAY, "补记一笔支出", 21, 28, kind=EXPENSE)
    run = scene_run(tmp_path, site, ModelMustNotBeCalled())
    generate(run, NIGHT, at(NIGHT, 1))
    assert run.hits.days_done() == frozenset({WEDNESDAY})
    return tmp_path / "data" / "scene"


def edit_footer(path: Path, marker: str, edit: Callable[[dict[str, Any]], None]) -> None:
    """改尾部规范 JSON 里的一处，正文不动、JSON 仍是规范形式（只让"字段不对"这一处出错）。"""

    text = path.read_text(encoding="utf-8")
    index = text.rfind(marker)
    payload = json.loads(text[index + len(marker) : -len(FOOTER)])
    edit(payload)
    path.write_text(text[: index + len(marker)] + canonical_json(payload) + FOOTER, encoding="utf-8")


# ── 概念目录 ──────────────────────────────────────────────────────────────────


def concept_path(scene: Path, identity: str) -> Path:
    return scene / "concepts" / f"{identity}.md"


def refused_concept(scene: Path, identity: str) -> None:
    store = ConceptStore(scene)
    with pytest.raises(ValueError, match=re.escape(identity)):
        store.read(identity)
    with pytest.raises(ValueError, match=re.escape(identity)):
        store.read_all()


def test_a_fresh_concept_directory_reads_back(tmp_path: Path) -> None:
    """对照：没改过的盘面读得回来（下面每条只改了一处）。"""

    scene = sealed_wednesday(tmp_path)
    concepts = ConceptStore(scene).read_all()
    assert RUN in concepts and "周三" in concepts


def test_a_truncated_concept_file_is_refused(tmp_path: Path) -> None:
    scene = sealed_wednesday(tmp_path)
    path = concept_path(scene, RUN)
    text = path.read_text(encoding="utf-8")
    path.write_text(text[: len(text) // 2], encoding="utf-8")
    refused_concept(scene, RUN)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("colour", "红"),  # 一个谁都不认识的字段
        ("kind", "basic"),  # 旧的种类字段，取看似无害的值
        ("kind", None),
        ("rule", None),  # 只带 rule（空的），别的都像词表类概念
        ("rule", {"field": "started_at", "after": "05:45", "before": "07:45"}),
        ("grades", []),  # 只带 grades（空的）
        ("grades", [{"name": "大额", "at_least": 100}]),
        ("baseline_keys", ["amount"]),  # 非空的常态键
        ("baseline_keys", []),
        ("context", "occurrence"),
    ],
)
def test_a_behaviour_concept_with_one_extra_old_or_unknown_field_is_refused(
    tmp_path: Path, key: str, value: object
) -> None:
    scene = sealed_wednesday(tmp_path)
    edit_footer(concept_path(scene, RUN), CONCEPT_MARKER, lambda payload: payload.__setitem__(key, value))
    refused_concept(scene, RUN)


@pytest.mark.parametrize("role", ["state", "object", "derived", "situation", "Behavior"])
def test_a_concept_with_an_unknown_role_is_refused(tmp_path: Path, role: str) -> None:
    scene = sealed_wednesday(tmp_path)
    edit_footer(concept_path(scene, RUN), CONCEPT_MARKER, lambda payload: payload.__setitem__("role", role))
    refused_concept(scene, RUN)


@pytest.mark.parametrize(
    "situation",
    [
        {"basis": "calendar", "value": "周几", "values": ["周三"]},  # 旧的日历情境：带 basis
        {"basis": "subject", "value": "同在的人", "values": ["同事"]},  # 情境族是"同在的人"
        {"basis": "place", "value": "地点", "values": ["公司"]},
        {"value": "周几"},  # 少了 values
    ],
)
def test_a_calendar_situation_in_another_shape_is_refused(tmp_path: Path, situation: dict[str, object]) -> None:
    scene = sealed_wednesday(tmp_path)
    edit_footer(
        concept_path(scene, "周三"), CONCEPT_MARKER, lambda payload: payload.__setitem__("situation", situation)
    )
    refused_concept(scene, "周三")


def test_a_real_old_calendar_situation_file_is_refused(tmp_path: Path) -> None:
    """旧盘面上的日历情境「周三」（``situation`` 里带 ``basis``，另带 kind / rule / grades / baseline_keys / context）。"""

    scene = sealed_wednesday(tmp_path)
    shutil.copyfile(LEGACY / "concepts" / "周三.md", concept_path(scene, "周三"))
    refused_concept(scene, "周三")


def test_a_second_behaviour_concept_for_the_same_class_is_refused(tmp_path: Path) -> None:
    """同一个类第二个行为概念：放在另一个文件里、名字不是类编号。"""

    scene = sealed_wednesday(tmp_path)
    second = scene / "concepts" / "跑步训练.md"
    shutil.copyfile(concept_path(scene, RUN), second)
    edit_footer(second, CONCEPT_MARKER, lambda payload: payload.__setitem__("name", "跑步训练"))
    store = ConceptStore(scene)
    with pytest.raises(ValueError, match="跑步训练"):
        store.read_all()


def test_a_copy_of_a_behaviour_concept_under_another_leaf_is_refused(tmp_path: Path) -> None:
    """同一个类的行为概念原样复制到另一个叶名下：身份与地址对不上。"""

    scene = sealed_wednesday(tmp_path)
    shutil.copyfile(concept_path(scene, RUN), scene / "concepts" / "跑步训练.md")
    with pytest.raises(ValueError, match="跑步训练"):
        ConceptStore(scene).read_all()


def test_a_concept_leaf_that_is_not_the_canonical_identity_is_refused(tmp_path: Path) -> None:
    scene = sealed_wednesday(tmp_path)
    folder = scene / "concepts"
    text = concept_path(scene, RUN).read_bytes()
    for entry in folder.iterdir():
        if entry.name.endswith(".md"):
            entry.unlink()
    (folder / "S-K0023.md").write_bytes(text)  # 大小写不是规范身份（大小写不敏感的文件系统上会与 s-k0023 静默互认）
    with pytest.raises(ValueError, match="S-K0023"):
        ConceptStore(scene).read_all()


# ── 命中记录目录 ────────────────────────────────────────────────────────────────


def hit_paths(scene: Path) -> list[Path]:
    folder = scene / "occurrences" / f"{WEDNESDAY:%Y}" / f"{WEDNESDAY:%m}" / f"{WEDNESDAY:%d}"
    return sorted(path for path in folder.iterdir() if path.name.endswith(".md"))


def test_a_fresh_hit_day_reads_back(tmp_path: Path) -> None:
    scene = sealed_wednesday(tmp_path)
    assert len(ConceptHitStore(scene).read_day(WEDNESDAY)) == 2


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("hits", []),
        ("hits", ["晨跑训练"]),
        ("unresolved", []),
        ("unresolved", [{"concept": "补记支出", "reason": "model_failed"}]),
        ("baseline_snapshot", {}),
        ("baseline_snapshot", {"amount": {"median": 30.0}}),
        ("mapper", "scene_mapper_v7+llm:gpt"),
        ("signals", []),
        ("recent_digest", "0123456789abcdef"),
        ("mapped_at", "2026-04-23T01:00:00.000000Z"),
    ],
)
def test_a_hit_record_with_one_old_field_is_refused_naming_the_file(tmp_path: Path, key: str, value: object) -> None:
    scene = sealed_wednesday(tmp_path)
    path = hit_paths(scene)[0]
    edit_footer(path, HITS_MARKER, lambda payload: payload.__setitem__(key, value))
    with pytest.raises(ValueError, match=re.escape(path.name.removesuffix(".md"))):
        ConceptHitStore(scene).read_day(WEDNESDAY)


@pytest.mark.parametrize("key", ["hits_version", "sealed_at", "situations_checked", "classified"])
def test_a_hit_record_missing_a_field_is_refused_naming_the_file(tmp_path: Path, key: str) -> None:
    scene = sealed_wednesday(tmp_path)
    path = hit_paths(scene)[0]
    edit_footer(path, HITS_MARKER, lambda payload: payload.pop(key))
    with pytest.raises(ValueError, match=re.escape(path.name.removesuffix(".md"))):
        ConceptHitStore(scene).read_day(WEDNESDAY)


# ── 日完成章 ──────────────────────────────────────────────────────────────────


def marker_path(scene: Path) -> Path:
    return scene / "occurrences" / f"{WEDNESDAY:%Y}" / f"{WEDNESDAY:%m}" / f"{WEDNESDAY:%d}" / DONE_FILENAME


def edit_marker(scene: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    path = marker_path(scene)
    payload = json.loads(path.read_text(encoding="utf-8"))
    edit(payload)
    path.write_text(canonical_json(payload), encoding="utf-8")


@pytest.mark.parametrize(
    ("key", "value"),
    [("hits_version", "scene_hits_v1"), ("unresolved", 0), ("mapper", "scene_mapper_v7"), ("colour", "红")],
)
def test_a_day_seal_with_one_extra_key_is_refused_naming_the_file(tmp_path: Path, key: str, value: object) -> None:
    scene = sealed_wednesday(tmp_path)
    edit_marker(scene, lambda payload: payload.__setitem__(key, value))
    store = ConceptHitStore(scene)
    with pytest.raises(ValueError, match=re.escape(DONE_FILENAME)):
        store.read_marker(WEDNESDAY)
    with pytest.raises(ValueError, match=re.escape(DONE_FILENAME)):
        store.days_done()


@pytest.mark.parametrize("key", ["day", "records", "concepts", "digest", "completed_at"])
def test_a_day_seal_missing_a_key_is_refused_naming_the_file(tmp_path: Path, key: str) -> None:
    scene = sealed_wednesday(tmp_path)
    edit_marker(scene, lambda payload: payload.pop(key))
    with pytest.raises(ValueError, match=re.escape(DONE_FILENAME)):
        ConceptHitStore(scene).read_marker(WEDNESDAY)


# ── D2：从空盘面换代，不产生待生效的概念目录与写概念的记号 ─────────────────────────


def test_d2_a_generation_from_an_empty_disk_leaves_no_pending_concepts_or_authoring_mark(tmp_path: Path) -> None:
    scene = sealed_wednesday(tmp_path)
    assert not (scene / "concepts-pending").exists()
    assert not (scene / "concepts" / ".authored.json").exists()
    assert not (scene / "generation" / "rhythms").exists()
