"""第二步乙阶段 C 补的测试：状态文件的对抗读、两个写者并发、预测层按天读树的换算（C35）。

- 对抗：状态文件里每一种不自洽都要明确报错并点出文件（共享说明 3.2、第 4 节；``Vocabulary.__post_init__`` 的不变量）。
- 并发：两条线程同时在同一个上级下面建不同名字的节点，同时换代在数词表，状态文件不丢任何一方的修改（重复跑）。
- C35：预测层按天读树（形成预期、结算、重结、诊断共用 ``day_cache``）读到挂在已删节点上的记录，读成最近的活着的上级，
  和序列读出来的一致；``runtime`` 里的按天索引只经 ``day_cache`` 一处建（四个调用方都过同一个换算口）。
"""

from __future__ import annotations

import ast
import json
import threading
import time
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

from habitus.behavior.kinds.active import count_active
from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.kinds.grow import Proposal, settle
from habitus.behavior.kinds.ids import ClassId, Lane
from habitus.behavior.kinds.model import Vocabulary
from habitus.behavior.kinds.store import BehaviorKindStore, BehaviorKindStoreError
from habitus.infrastructure.store.sqlite import SQLiteLockStore
from habitus.runtime.foresight_tree_rows import day_cache
from tests.unit.word_grow.site import WRITE_DOC, Grow, GrowSite, at, grown, node
from tests.unit.word_pick.site import CST

ROOT = Path(__file__).resolve().parents[3]
CONFIG = BehaviorKindConfig()
DOC = ClassId.parse(WRITE_DOC)
QUERY = ClassId.parse("s-k0098")
MOMENT = datetime(2026, 4, 13, 9, tzinfo=CST)


def _grown_state(tmp_path: Path) -> tuple[BehaviorKindStore, dict[str, Any]]:
    """一份合法的状态：写文档下面 X（s-k0280）、X 下面 Y（s-k0281）、写文档下面 Z（s-k0282，随后删掉），查数据在用。"""

    store = BehaviorKindStore(
        tmp_path,
        lane=Lane.SESSION,
        lock_store=SQLiteLockStore(tmp_path / "locks.sqlite3"),
        lock_ttl_seconds=30,
        config=CONFIG,
    )

    def build(vocabulary: Vocabulary) -> Vocabulary:
        vocabulary, _ = settle(
            vocabulary, DOC, (Proposal("写需求书", "写需求书这一种"), Proposal("写验收段", "写验收段这一种")),
            at=MOMENT, config=CONFIG,
        )  # fmt: skip
        vocabulary, z = settle(vocabulary, DOC, (Proposal("写设计书", "写设计书这一种"),), at=MOMENT, config=CONFIG)
        vocabulary = vocabulary.without(z, on=date(2026, 4, 14))
        return vocabulary.with_active({QUERY: date(2026, 4, 15)})

    store.update(build)
    raw = json.loads(store.state_path.read_text(encoding="utf-8"))
    assert [item["id"] for item in raw["grown"]] == [grown(1), grown(2)]
    assert [item["id"] for item in raw["deleted"]] == [grown(3)]
    return store, raw


def _grown_entry(raw: dict[str, Any], word_id: str) -> dict[str, Any]:
    (found,) = [item for item in raw["grown"] if item["id"] == word_id]
    return found


def _parent_unknown(raw: dict[str, Any]) -> None:
    _grown_entry(raw, grown(1))["parent"] = "s-k0999"


def _level_mismatch(raw: dict[str, Any]) -> None:
    _grown_entry(raw, grown(1))["level"] = 3


def _same_name_as_live(raw: dict[str, Any]) -> None:
    _grown_entry(raw, grown(2))["name"] = "写周报"


def _id_clashes_with_initial(raw: dict[str, Any]) -> None:
    _grown_entry(raw, grown(2))["id"] = "s-k0005"


def _deleted_parent_unknown(raw: dict[str, Any]) -> None:
    raw["deleted"][0]["parent"] = "s-k0999"


def _deleted_in_use(raw: dict[str, Any]) -> None:
    raw["active"].append({"id": grown(3), "since": "2026-04-15"})


def _sequence_missing(raw: dict[str, Any]) -> None:
    del raw["sequence"]


def _old_version(raw: dict[str, Any]) -> None:
    raw["schema_version"] = "behavior_kinds_state_v1"


@pytest.mark.parametrize(
    "break_it",
    [
        _parent_unknown,
        _level_mismatch,
        _same_name_as_live,
        _id_clashes_with_initial,
        _deleted_parent_unknown,
        _deleted_in_use,
        _sequence_missing,
        _old_version,
    ],
)
def test_a_state_file_that_does_not_hold_together_is_refused_and_names_the_file(
    tmp_path: Path, break_it: Callable[[dict[str, Any]], None]
) -> None:
    store, raw = _grown_state(tmp_path)
    break_it(raw)
    store.state_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(BehaviorKindStoreError) as caught:
        store.read()
    assert str(store.state_path) in str(caught.value)
    with pytest.raises(BehaviorKindStoreError):  # 写者也读不过去，不在坏状态上接着写
        store.update(lambda vocabulary: vocabulary.with_active({}))


def _grow(store: BehaviorKindStore, name: str, start: threading.Barrier) -> Callable[[], None]:
    def run() -> None:
        start.wait(timeout=5)

        def change(vocabulary: Vocabulary) -> Vocabulary:
            settled, _ = settle(vocabulary, DOC, (Proposal(name, f"{name}这一种"),), at=MOMENT, config=CONFIG)
            time.sleep(0.005)  # 拉长"读 → 改"之间的窗口，让三个写者真的撞上
            return settled

        store.update(change)

    return run


def _count(store: BehaviorKindStore, start: threading.Barrier) -> Callable[[], None]:
    picks = (("s-k0098", date(2026, 4, 13)), ("s-k0098", date(2026, 4, 14)))

    def run() -> None:
        start.wait(timeout=5)

        def change(vocabulary: Vocabulary) -> Vocabulary:
            counted = count_active(
                vocabulary, picks, today=date(2026, 4, 15), min_days=2, cap=100, retire_days=CONFIG.retire_days
            )
            time.sleep(0.005)
            return counted.vocabulary

        store.update(change)

    return run


@pytest.mark.parametrize("round_", range(15))
def test_two_growing_turns_and_a_generation_writing_at_once_lose_nothing(tmp_path: Path, round_: int) -> None:
    store = BehaviorKindStore(
        tmp_path,
        lane=Lane.SESSION,
        lock_store=SQLiteLockStore(tmp_path / "locks.sqlite3"),
        lock_ttl_seconds=30,
        config=CONFIG,
    )
    start = threading.Barrier(3)
    failures: list[BaseException] = []

    def guarded(run: Callable[[], None]) -> Callable[[], None]:
        def target() -> None:
            try:
                run()
            except BaseException as exc:  # pragma: no cover - 失败在主线程报
                failures.append(exc)

        return target

    workers = [
        threading.Thread(target=guarded(_grow(store, "写需求书", start))),
        threading.Thread(target=guarded(_grow(store, "写设计书", start))),
        threading.Thread(target=guarded(_count(store, start))),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
    assert not failures
    final = store.read()
    names = {word.name: word.id for word in final.words if word.grown}
    assert set(names) == {"写需求书", "写设计书"}
    assert len(set(names.values())) == 2 and set(names.values()) == {ClassId.parse(grown(1)), ClassId.parse(grown(2))}
    assert set(final.active) == {QUERY} and final.revision == 1
    raw = json.loads(store.state_path.read_text(encoding="utf-8"))
    assert raw["sequence"] == 3  # 三次写各写了一次


def _with_foresight(raw: dict[str, Any]) -> None:
    raw["foresight"] = {"enabled": True}


def test_c35_the_day_index_reads_a_deleted_nodes_record_as_its_parent_like_the_series(tmp_path: Path) -> None:
    site = GrowSite(tmp_path, edit=_with_foresight)
    try:
        site.record("conv-x", at(date(2026, 4, 13), 9), "帮我写一份产品需求文档", Grow("写文档", node("写需求书")))
        foresight = site.runtime.components.foresight
        assert foresight is not None
        before = day_cache(foresight.read_side).day(date(2026, 4, 13))
        assert [row.kind_token for row in before.rows] == [grown(1)]
        site.generate(date(2026, 5, 3))  # 20 天没被选中：删掉
        rows = day_cache(foresight.read_side).day(date(2026, 4, 13)).rows
        assert [row.kind_token for row in rows] == [WRITE_DOC]
        series = site.series_tokens(date(2026, 5, 4))
        assert [series[row.at] for row in rows] == [WRITE_DOC]
    finally:
        site.close()


def test_c35_every_day_index_in_the_runtime_is_built_through_the_one_resolving_helper() -> None:
    """``runtime`` 里建 ``DayIndexCache`` 的只有 ``foresight_tree_rows.day_cache``（它带换算口）：形成预期、结算、重结、
    诊断四个调用方都经它，没有哪个绕过换算自己建。"""

    builders: list[str] = []
    for path in sorted((ROOT / "habitus" / "runtime").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in ast.walk(tree):
            if (
                isinstance(call, ast.Call)
                and getattr(call.func, "id", getattr(call.func, "attr", None)) == "DayIndexCache"
            ):
                builders.append(path.name)
    assert builders == ["foresight_tree_rows.py"]
    users = {
        path.name
        for path in (ROOT / "habitus" / "runtime").glob("foresight_*.py")
        if "day_cache(self.read_side)" in path.read_text(encoding="utf-8")
    }
    assert users == {"foresight_forming.py", "foresight_settlement.py", "foresight_settlement_diagnosis.py"}
