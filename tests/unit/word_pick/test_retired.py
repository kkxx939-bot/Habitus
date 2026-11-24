"""第二步甲删掉的东西（共享说明第 1 节、3.3；预期表 D3、D7、D8、D9）。

- D3：白天归类、每晚新增、拆改（含分组）这几个模型触点在 ``habitus/`` 里不存在了（多出来的每轮一次选词由
  ``test_session_pick`` / ``test_physical_pick`` 按调用次数验）。
- D7：待定池、每晚新增、拆改、分组、锚点、合并、冷却那批 ``kinds_*`` 键写进配置就加载报错；新增的键
  （在用需要的日子数 2、在用上限 100、选词重问次数 2、最近记录条数 8、同会话条数 3、之前轮数 3、每轮截断 600）
  在 example.yaml 里都列出来了，并且按示例值能加载。
- D8：词表目录（行为树根）里有旧文件时启动明确报错，点出文件。
- D9：第 1 节列的名字在 ``habitus/`` 里 grep 不到（同名无关的单独限定范围）。
旧代码上这些都应当失败。
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from habitus.config import HabitusConfig
from tests.unit.word_pick.config_keys import RETIRED_BEHAVIOR_KEYS, added_keys
from tests.unit.word_pick.site import EXAMPLE_CONFIG, REPOSITORY_ROOT, config, example_mapping, runtime_of

HABITUS = REPOSITORY_ROOT / "habitus"


def python_sources(root: Path = HABITUS) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def mentions(pattern: str, sources: list[Path]) -> list[str]:
    compiled = re.compile(pattern)
    found = []
    for path in sources:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if compiled.search(line):
                found.append(f"{path.relative_to(REPOSITORY_ROOT)}:{number}: {line.strip()}")
    return found


# ── D3 ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "touch",
    ["behavior_kind_classify", "behavior_kind_nightly", "behavior_kind_revision", "behavior_kind_grouping"],
)
def test_d3_the_retired_model_touchpoints_are_gone(touch: str) -> None:
    assert mentions(rf"[\"']{touch}[\"']", python_sources()) == []


# ── D7 ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("key", "value"), sorted(RETIRED_BEHAVIOR_KEYS.items()))
def test_d7_a_retired_kinds_key_is_refused_when_loading(tmp_path: Path, key: str, value: object) -> None:
    raw = example_mapping(tmp_path)
    raw["behavior"] = {"primary_subject": "家庭成员A", key: value}
    with pytest.raises(Exception) as caught:
        HabitusConfig.from_mapping(raw)
    assert key in str(caught.value)


def test_d7_the_new_keys_are_listed_in_the_example_with_their_defaults() -> None:
    added = added_keys(EXAMPLE_CONFIG)
    values = Counter(added.values())
    wanted = Counter({"2": 2, "100": 1, "8": 1, "3": 2, "600": 1})
    assert values & wanted == wanted, f"new keys in example.yaml: {added}"
    assert not any(key in RETIRED_BEHAVIOR_KEYS for _, key in added)


def test_d7_the_new_keys_load_at_their_example_values(tmp_path: Path) -> None:
    added = added_keys(EXAMPLE_CONFIG)
    assert added, "example.yaml lists no new keys"

    def edit(raw: dict[str, Any]) -> None:
        for (section, key), value in added.items():
            raw.setdefault(section, {})[key] = int(value) if value.isdigit() else value

    config(tmp_path, edit=edit)


# ── D8 ──────────────────────────────────────────────────────────────────────

LEGACY_FILES = (
    "kinds.changes.jsonl",
    "kinds.pending.json",
    "kinds.snapshot.json",
    "kinds.jobs.json",
    "kinds.migration.json",
    "kinds.revision.json",
)


@pytest.mark.parametrize("lane", ["session", "physical"])
@pytest.mark.parametrize("name", LEGACY_FILES)
def test_d8_a_legacy_vocabulary_file_on_disk_refuses_the_start_and_names_the_file(
    tmp_path: Path, name: str, lane: str
) -> None:
    habitus_config = config(tmp_path)
    directory = habitus_config.behavior_root / lane  # 词表文件在这一支树外的根（裁定 112），旧词表各文件也查这里
    directory.mkdir(parents=True)
    (directory / name).write_text("{}\n", encoding="utf-8")

    def boot() -> None:
        runtime, _router = runtime_of(habitus_config)
        try:
            asyncio.run(runtime.start())
            asyncio.run(runtime.stop())
        finally:
            asyncio.run(runtime.close())

    with pytest.raises(Exception) as caught:
        boot()
    chain = []
    error: BaseException | None = caught.value
    while error is not None:
        chain.append(str(error))
        error = error.__cause__ or error.__context__
    assert any(name in text for text in chain), chain


# ── D9 ──────────────────────────────────────────────────────────────────────

RETIRED_NAMES = (
    "PendingPool",
    "PendingEntry",
    "PendingNote",
    "kind_pending",
    "NightlyGrower",
    "PendingRecheck",
    "Reviser",
    "AnchorGate",
    "JobState",
    "revision_due",
    "VocabularyJobs",
    "KindMigrator",
    "kinds_migration",
    "Branch",
    "RetireClass",
    "ReviseClass",
    "merge_operations",
    "split_operations",
    "Move",
    "descendants",
    "current_ids",
    "changes_since",
    "CatalogChanges",
    "MovedOccurrence",
    "read_migration",
    "read_revision",
    "vocabulary_migrating",
    "held_by_migration",
    "run_vocabulary_job",
    "DaytimeClassifier",
    "sentence_key",
    "VOCABULARY_MOVED",
    "MIGRATING",
)


@pytest.mark.parametrize("name", RETIRED_NAMES)
def test_d9_a_retired_name_is_gone_from_habitus(name: str) -> None:
    assert mentions(rf"\b{name}\b", python_sources()) == []


def test_d9_class_origins_promoted_and_split_are_gone() -> None:
    sources = python_sources(HABITUS / "behavior")
    assert mentions(r"\bPROMOTED\b|\bSPLIT\b", sources) == []


def test_d9_the_model_queue_has_no_vocabulary_category_name() -> None:
    sources = [path for path in python_sources(HABITUS / "runtime") if path.name.startswith("model_queue")]
    assert mentions(r"\bVOCABULARY\b", sources) == []
