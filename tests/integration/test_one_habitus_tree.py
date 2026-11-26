"""一棵 habitus 树（裁定 112）：两支互相看不见、地址只差前缀和支、卡片这半边不受事件目录影响、旧盘面与旧配置在启动时报错。"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from habitus.behavior.kinds.ids import ClassId, Lane
from habitus.behavior.model import BehaviorAddress, BehaviorKind
from habitus.behavior.tree import BehaviorTree
from habitus.behavior.tree.writer import BehaviorDocumentWriter, BehaviorPublishConflictError
from habitus.behavior.uri import BehaviorURI, BehaviorURIError
from habitus.config.loader import ConfigError
from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.tree.overview import Overview, OverviewChild
from habitus.memory.editor.retrieval.retriever import memory_directory_listing
from habitus.memory.editor.retrieval.search import memory_content_search
from habitus.memory.model import MemoryAddress, MemoryDirectory, MemoryKind, MemoryLevel
from habitus.memory.uri import MemoryURI
from habitus.runtime import build_runtime
from habitus.runtime.foresight import foresight_ledger
from habitus.runtime.foresight_tree_rows import tree_day
from habitus.runtime.runtime_lifecycle import TwoTreeLayoutError
from habitus.runtime.scene import habit_cards, relation_cards
from habitus.scene.habits import HabitNumbers
from habitus.scene.relations.engine import RelationKey
from habitus.scene.relations.spans import Segment
from habitus.scene.relations.state import LaneState, Relation, Status
from habitus.scene.views.index import DayIndex
from habitus.series.reader import admitted, read_series
from tests.helpers import document
from tests.lock_store import ProcessLocalLockStore
from tests.unit.behavior.kinds_fixtures import named_words
from tests.unit.behavior.tree_payloads import DAY, gap_payload, local, occurrence_payload
from tests.unit.runtime.test_behavior_pipeline import (
    PICK_BODY,
    behavior_enabled_config,
    physical_delivery,
    scripted_dependencies,
)
from tests.unit.runtime.test_foresight_outlet import awaiting, expectation

CST = timezone(timedelta(hours=8))


def _views(tmp_path):  # type: ignore[no-untyped-def]
    root = tmp_path / "habitus"
    store = ProcessLocalLockStore()
    session, physical = BehaviorTree(root, lane="session"), BehaviorTree(root, lane="physical")
    writers = {
        lane: BehaviorDocumentWriter(tree, store, clock=lambda: local(23, 0))
        for lane, tree in (("session", session), ("physical", physical))
    }
    return root, session, physical, writers


def test_each_lane_view_sees_only_its_own_occurrences_and_gaps(tmp_path) -> None:  # type: ignore[no-untyped-def]
    root, session, physical, writers = _views(tmp_path)
    occurrence = writers["physical"].publish(BehaviorKind.OCCURRENCE, occurrence_payload())
    gap = writers["physical"].publish(BehaviorKind.GAP, gap_payload())
    # 落在物理一支的目录下
    assert physical.path_for(occurrence.address) == root / "occurrences" / "physical" / "2026" / "08" / "16" / (
        f"{occurrence.address.identity_name}.md"
    )
    assert physical.path_for(gap.address).relative_to(root).parts[:2] == ("gaps", "physical")
    # 会话一支的视图什么都看不到：天、按天读、枚举、存在性、序列里的空白（覆盖率读它）、"那天有没有来过东西"
    for kind in BehaviorKind:
        assert session.days(kind) == ()
        assert session.read_day(kind, DAY) == ()
        assert session.list_addresses(kind) == ()
    assert not session.exists(occurrence.address) and physical.exists(occurrence.address)
    assert read_series(session, cutoff=DAY + timedelta(days=1)).gaps == ()
    assert len(read_series(physical, cutoff=DAY + timedelta(days=1)).gaps) == 1
    quiet = DayIndex(session, DAY, subject="家庭成员A", admits=admitted, read_gaps=lambda day: ())
    assert quiet.silent
    assert not DayIndex(physical, DAY, subject="家庭成员A", admits=admitted, read_gaps=lambda day: ()).silent
    # 同一个地址在两支是两个文件；同一支里同内容重放算成功，不同内容报冲突
    writers["session"].publish(BehaviorKind.OCCURRENCE, occurrence_payload())
    assert session.uri_for(occurrence.address) != physical.uri_for(occurrence.address)
    writers["physical"].publish(BehaviorKind.OCCURRENCE, occurrence_payload())
    with pytest.raises(BehaviorPublishConflictError):
        writers["physical"].publish(BehaviorKind.OCCURRENCE, occurrence_payload(summary="换了一句概要"))


#: 合并之前（4b3ab8d）的行为地址：叶名与百分号编码必须逐字节相同，只换掉协议前缀并插入支。
_BEFORE = {
    "看 日程": "occurrences/2026/03/02/看%20日程--20260302T090507123456%2B0800.md",
    "写周报@周五": "occurrences/2026/03/02/写周报%40周五--20260302T090507123456%2B0800.md",
    "100% 完成": "occurrences/2026/03/02/100%25%20完成--20260302T090507123456%2B0800.md",
    "Ünïcode Ñame": "occurrences/2026/03/02/ünïcode%20ñame--20260302T090507123456%2B0800.md",
    "热身--拉伸": "occurrences/2026/03/02/热身--拉伸--20260302T090507123456%2B0800.md",
}


@pytest.mark.parametrize(("name", "before"), sorted(_BEFORE.items()))
def test_leaf_names_and_percent_encoding_are_byte_identical_to_before(name: str, before: str) -> None:
    started = datetime(2026, 3, 2, 9, 5, 7, 123456, tzinfo=CST)
    uri = BehaviorURI.from_address(BehaviorAddress.occurrence(started.date(), name, started), "session")
    kind, rest = before.split("/", 1)
    assert str(uri) == f"habitus://{kind}/session/{rest}"
    gap = BehaviorURI.from_address(BehaviorAddress.gap(started.date(), "没读懂", started), "physical")
    assert str(gap) == "habitus://gaps/physical/2026/03/02/没读懂--20260302T090507123456%2B0800.md"
    # 卡片地址同一套编码
    assert str(MemoryURI.from_address(MemoryAddress.preference("回答 风格@x"))) == "habitus://preferences/回答%20风格%40x.md"
    assert str(MemoryURI.from_address(MemoryAddress.entity("项目", "100% 完成"))) == "habitus://entities/项目/100%25%20完成.md"


def test_tree_lookups_parse_scheme_bearing_sources_strictly_and_skip_the_rest() -> None:
    """原来静默当"找不到"的取行：带 ``://`` 的出处必须是一支里的事件地址，旧写法与别的地址报错；不是树上的出处才跳过。"""

    good = "habitus://occurrences/session/2026/08/16/洗了手--20260816T193018000000%2B0800.md"
    assert tree_day(good) == date(2026, 8, 16)
    assert tree_day("unsealed:2026-08-16T19:30:18+08:00") is None
    assert tree_day("2026-08-16") is None
    old = "behavior:" + "//occurrences/2026/08/16/洗了手--20260816T193018000000%2B0800.md"
    gap = "habitus://gaps/session/2026/08/16/没读懂--20260816T193018000000%2B0800.md"  # 空白不是出处
    habit, reminder = "habitus://habits/session/s-k0001.md", "habitus://reminders/session/2026/08/16/e1.md"  # 只做记录，裁定 113
    relation = "habitus://relations/session/s-k0001/s-k0002--chain.md"  # 只做记录，裁定 115
    for bad in (old, "habitus://occurrences/2026/08/16/洗了手--20260816T193018000000%2B0800.md", "habitus://x.md", gap, habit, reminder, relation):
        with pytest.raises(BehaviorURIError):
            tree_day(bad)


def test_cards_ignore_the_occurrence_and_gap_directories_at_the_root(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """根下有事件与空白的目录：记忆树初始化、写卡片、刷概览、冷热枚举、检索清单、提取时列目录与搜内容照常，都不把它们算进去。"""

    runtime = build_runtime(
        behavior_enabled_config(tmp_path), providers=scripted_dependencies([PICK_BODY]), path_lock=PathLock(ProcessLocalLockStore())
    )
    runtime.initialize()
    branches = runtime.components.behavior
    assert branches is not None
    asyncio.run(runtime.deliver_physical(physical_delivery()))
    report = asyncio.run(branches.physical.reduction_runner.run_once())
    assert (report.published_occurrences, report.published_gaps) == (1, 1)
    # 根下再放一张习惯卡片、一条提醒记录（裁定 113）：记忆这半边同样看不到它们
    config = behavior_enabled_config(tmp_path)
    habit_cards(config, "physical").sync(
        named_words(("喝水",), lane=Lane.PHYSICAL).with_active({ClassId(Lane.PHYSICAL, 1): DAY}),
        {"p-k0001": HabitNumbers(DAY, DAY, 1, 0, 0, 0, 0, None)},
        now=datetime.now(UTC),
    )
    awaiting(foresight_ledger(config, "session"), expectation())
    # 再放一张关系卡片（裁定 115）：同样看不到
    key = RelationKey("physical", "p-k0001", "p-k0002", Segment.CHAIN)
    relation_cards(config, "physical").sync(
        LaneState("physical", DAY, 1, (Relation(key, Status.RETIRED, True, retired_on=DAY),)),
        {"p-k0001": "喝水", "p-k0002": "周报"},
        now=datetime.now(UTC),
    )
    memory = runtime.components.memory
    tree = memory.tree
    assert {path.name for path in tree.root.iterdir()} >= {
        "occurrences",
        "gaps",
        "habits",
        "reminders",
        "relations",
        "preferences",
        "entities",
        "tools",
    }
    assert list((tree.root / "habits").rglob("*.md")) and list((tree.root / "reminders").rglob("*.md"))
    assert list((tree.root / "relations").rglob("*.md"))

    card = document(MemoryKind.PREFERENCE)
    tree.initialize()
    tree.write(card)
    memory.semantic_refresher.rebuild()
    root_overview = tree.read_layer(MemoryDirectory.root(), MemoryLevel.OVERVIEW)
    assert "occurrences" not in root_overview and "gaps" not in root_overview
    assert tree.list_addresses() == (card.address,)
    assert tree.direct_addresses(MemoryDirectory.root()) == ()
    assert tuple(entry.uri for entry in memory_directory_listing(tree, MemoryDirectory.preferences())) == (
        MemoryURI.from_address(card.address),
    )
    word = next(iter(str(value) for value in card.fields.values() if isinstance(value, str) and value))[:2]
    hits = memory_content_search(tree, None, keywords=(word,), directory=None, limit=10)
    assert {hit.uri for hit in hits} <= {MemoryURI.from_address(card.address)}
    catalog = memory.search.catalog.read(kinds=tuple(MemoryKind), now=datetime.now(UTC), on_recall_failure=lambda error: None)
    assert [item.uri for item in (*catalog.cards, *catalog.cold, *catalog.hidden)] == [MemoryURI.from_address(card.address)]
    assert catalog.notes == ()
    for directory in (
        "habitus://occurrences/physical",
        "habitus://habits/physical",
        "habitus://reminders/session",
        "habitus://relations/physical",
    ):
        with pytest.raises(ValueError):
            MemoryURI(directory)  # 浏览、提取的读口都只认卡片地址
    for scoped in (MemoryDirectory.root(), None):
        hits = memory_content_search(tree, None, keywords=("喝水", "周报"), directory=scoped, limit=10)
        assert all("habits" not in str(hit.uri) and "relations" not in str(hit.uri) for hit in hits)
    # 清单的范围是显式限定的，不靠根概览刚好没列：根概览里就算出现事件目录，清单也不往里走、不失败
    listed = Overview.parse("记忆根", root_overview)
    widened = Overview("记忆根", listed.cards, (*listed.children, OverviewChild("occurrences", 1)), listed.omitted_cards)
    tree.write_layers(MemoryDirectory.root(), abstract=widened.abstract(), overview=widened.render())
    again = memory.search.catalog.read(kinds=tuple(MemoryKind), now=datetime.now(UTC), on_recall_failure=lambda error: None)
    assert [item.uri for item in (*again.cards, *again.cold, *again.hidden)] == [MemoryURI.from_address(card.address)]


def test_without_the_behavior_side_the_tree_has_no_occurrence_or_gap_directories(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from tests.integration.test_runtime_assembly import runtime_config, runtime_dependencies

    config = runtime_config(tmp_path)
    runtime = build_runtime(config, providers=runtime_dependencies(), path_lock=PathLock(ProcessLocalLockStore()))
    assert runtime.components.behavior is None
    runtime.initialize()
    names = {path.name for path in config.tree_root.iterdir()}
    assert names == {"preferences", "entities", "tools"}


def test_without_the_semantic_layer_or_foresight_there_are_no_habit_or_reminder_directories(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """习惯卡片与关系卡片只在语义树开着时写、提醒记录只在预测开着时写，目录都是第一次写时才建（裁定 113、115）：行为侧开着、两者关着时
    树里没有 ``habits``、``relations``、``reminders``。"""

    config = behavior_enabled_config(tmp_path)
    runtime = build_runtime(config, providers=scripted_dependencies([PICK_BODY]), path_lock=PathLock(ProcessLocalLockStore()))
    assert runtime.components.pipelines == () and not config.foresight.enabled
    runtime.initialize()
    asyncio.run(runtime.deliver_physical(physical_delivery()))
    asyncio.run(runtime.components.behavior.physical.reduction_runner.run_once())  # type: ignore[union-attr]
    assert {path.name for path in config.tree_root.iterdir()} == {"preferences", "entities", "tools", "occurrences", "gaps"}


def test_turning_the_behavior_side_off_keeps_its_directories_untouched_and_unread(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """先开着行为侧写过事件，再关掉它在同一个存储根上启动：照常启动，事件与空白原样留着、不读（关掉不该要求删历史，
    与合并前"行为树留在盘上不读"一样；裁定 112 第 1 条）。"""

    from tests.integration.test_runtime_assembly import runtime_dependencies

    on = behavior_enabled_config(tmp_path)
    runtime = build_runtime(on, providers=scripted_dependencies([PICK_BODY]), path_lock=PathLock(ProcessLocalLockStore()))
    runtime.initialize()
    assert runtime.components.behavior is not None
    asyncio.run(runtime.deliver_physical(physical_delivery()))
    asyncio.run(runtime.components.behavior.physical.reduction_runner.run_once())
    written = sorted(path.relative_to(on.tree_root) for path in (on.tree_root / "occurrences").rglob("*.md"))
    assert written

    import yaml

    from habitus.config import HabitusConfig
    from tests.integration.test_runtime_assembly import REPOSITORY_ROOT

    payload = yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(on.storage_root)
    payload["observability"]["tracing"]["enabled"] = False
    payload["models"]["chat"]["route"].update(provider="fake", adapter="fake_chat", credential_ref="")
    off = HabitusConfig.from_mapping(payload)  # 同一个存储根，行为侧关着（没有 primary_subject）
    restarted = build_runtime(off, providers=runtime_dependencies(), path_lock=PathLock(ProcessLocalLockStore()))
    assert restarted.components.behavior is None
    restarted.initialize()
    tree = restarted.components.memory.tree
    assert tree.list_addresses() == ()
    assert sorted(path.relative_to(off.tree_root) for path in (off.tree_root / "occurrences").rglob("*.md")) == written


@pytest.mark.parametrize("old", ["memory", "behavior/session/tree", "behavior/physical/tree"])
def test_an_old_two_tree_layout_refuses_the_start_before_anything_runs(tmp_path, old: str) -> None:  # type: ignore[no-untyped-def]
    config = behavior_enabled_config(tmp_path)
    (config.storage_root / old).mkdir(parents=True)
    runtime = build_runtime(config, providers=scripted_dependencies([PICK_BODY]), path_lock=PathLock(ProcessLocalLockStore()))
    with pytest.raises(TwoTreeLayoutError, match=re.escape(str(config.storage_root / old))):
        runtime.initialize()
    # 报错发生在初始化最前面：树、行为的存储都还没建，归约更没起来
    assert not config.tree_root.exists()
    assert not (config.behavior_root / "session" / "reduction").exists()


def test_the_removed_behavior_overview_settings_are_named_when_loading(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ConfigError, match="semantic_max_overview_chars"):
        behavior_enabled_config(tmp_path, semantic_max_overview_chars=64000)
