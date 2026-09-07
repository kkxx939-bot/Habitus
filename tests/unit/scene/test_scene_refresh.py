"""定稿日刷新：每天归组一次、已有一代不重算、强制重算走 force、检查点重放、失败预算与封锁、调用预算、
读时派生的待用前提清单、自带租约。"""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from habitus.scene import SceneRefreshBusyError, SceneTreeConflictError, pending_before
from habitus.scene.grouping import SceneGroupingLimitError
from habitus.scene.refresh import SceneRefreshConfig, SceneRefresher, build_grouping_input
from habitus.scene.uri import SceneURI
from tests.unit.scene.fixtures import DAY1, DAY2, DAY3, SUBJECT, Site, at, publish
from tests.unit.scene.scene_payloads import CST


def test_closed_days_publish_documents_with_cross_day_links_and_the_ledger_follows(tmp_path) -> None:
    site = Site(tmp_path, now=at(DAY3, 12, 0))
    uris = site.seed()

    report = site.refresh(DAY1, DAY2)

    assert report.published == (DAY1, DAY2)
    assert report.failed == () and report.deferred == () and report.blocked == ()
    assert report.model_calls == 2
    shopping = site.scene_tree.read_day(DAY1)
    assert [document.address.label for document in shopping] == ["去超市采购"]
    assert shopping[0].fields["pending_effects"] == ({"text": "家里有今晚要用的食材", "producer_uri": uris["buy"]},)
    dinner = site.scene_tree.read_day(DAY2)[0]
    assert dinner.address.label == "准备晚饭"
    assert dinner.fields["started_at"].startswith("2026-08-16T19:12:00")
    assert dinner.fields["ended_at"].startswith("2026-08-16T20:25:00")
    links = {(link.link_type.value, str(link.to_uri)) for link in dinner.links}
    assert links == {("needs", uris["buy"]), ("results_from", str(SceneURI.from_address(shopping[0].address)))}
    assert all(link.lag_seconds > 0 for link in dinner.links)
    # 归组输入带了三类参照
    day2_input = site.grouper.payloads[-1]
    assert [item.label for item in day2_input.scene_references] == ["去超市采购"]
    assert [item.text for item in day2_input.pending_references] == ["家里有今晚要用的食材"]
    assert [(item.kind_token, item.days_ago, item.scene_label, item.today_nos) for item in day2_input.last_occurrences] == [
        ("看手机", 1, None, (3,))
    ]
    assert len(day2_input.gaps) == 1
    # 清单：DAY2 开始前有一项；DAY2 的 needs 边兑现之后 DAY3 看不到它
    assert [item.text for item in pending_before(site.scene_tree, DAY2, expiry_days=30)] == ["家里有今晚要用的食材"]
    assert pending_before(site.scene_tree, DAY3, expiry_days=30) == ()
    assert any("scenes=1 unassigned=1" in note for note in report.signals)
    assert any("role_degraded: scripted" in note for note in report.signals)


def test_a_grouped_day_is_final_until_forced(tmp_path) -> None:
    """历史不变：定稿日归组一次；之后补发进树也不重算，重算只由 force 触发。"""

    site = Site(tmp_path, now=at(DAY3, 12, 0))
    site.seed()
    site.refresh(DAY1, DAY2)
    calls = len(site.grouper.payloads)
    generations = site.scene_tree.generations_of(DAY1)

    publish(site.behavior_tree, DAY1, "取快递", 18, 0)  # 定稿后的补发
    again = site.refresh(DAY1, DAY2)

    assert again.unchanged == (DAY1, DAY2) and again.published == ()
    assert len(site.grouper.payloads) == calls
    assert site.scene_tree.generations_of(DAY1) == generations

    forced = site.refresh(DAY1, force=True)

    assert forced.published == (DAY1,) and forced.model_calls == 1
    assert site.scene_tree.generations_of(DAY1) != generations
    assert len(site.scene_tree.read_day(DAY1)[0].fields["members"]) == 2
    # 输入与版本都没变时 force 也零调用
    assert site.refresh(DAY1, force=True).unchanged == (DAY1,)
    assert site.refresher.pending_days() == ()


def test_a_failed_publish_keeps_the_day_pending_and_replays_the_checkpoint(tmp_path, monkeypatch) -> None:
    site = Site(tmp_path, now=at(DAY3, 12, 0))
    site.seed()
    original = site.scene_tree.publish_day

    def explode(*args, **kwargs):
        raise SceneTreeConflictError("simulated")

    monkeypatch.setattr(site.scene_tree, "publish_day", explode)
    report = site.refresh(DAY1)
    assert report.failed == (DAY1,)
    assert site.refresher.pending_days() == (DAY1,)
    assert (tmp_path / "scene" / "refresh" / "checkpoints" / "2026-08-15.json").exists()
    assert site.refresher.blocked_days() == {}  # 记了一次失败，还没到封锁
    calls = len(site.grouper.payloads)

    monkeypatch.setattr(site.scene_tree, "publish_day", original)
    replay = site.refresh()

    assert replay.published == (DAY1,)
    assert len(site.grouper.payloads) == calls  # 检查点重放，零模型调用
    assert any("checkpoint replayed" in note for note in replay.signals)
    assert not (tmp_path / "scene" / "refresh" / "checkpoints" / "2026-08-15.json").exists()


def test_model_call_budget_defers_the_rest_to_the_next_run(tmp_path) -> None:
    site = Site(tmp_path, now=at(DAY3, 12, 0), max_model_calls_per_run=1)
    site.seed()
    first = site.refresh(DAY1, DAY2)
    assert first.published == (DAY1,) and first.deferred == (DAY2,) and first.model_calls == 1
    assert site.refresher.pending_days() == (DAY2,)
    second = site.refresh()
    assert second.published == (DAY2,) and site.refresher.pending_days() == ()


def test_repeated_failures_on_the_same_input_block_the_day_until_the_input_changes(tmp_path) -> None:
    site = Site(tmp_path, now=at(DAY3, 12, 0))
    site.seed()
    site.grouper.explode = RuntimeError("model keeps failing")

    outcomes = [site.refresh(DAY1) for _ in range(3)]

    assert [report.failed for report in outcomes[:2]] == [(DAY1,), (DAY1,)]
    assert outcomes[2].blocked == (DAY1,) and outcomes[2].failed == ()
    assert site.refresher.pending_days() == ()
    assert DAY1 in site.refresher.blocked_days()
    # 输入没变：交回来也不再烧调用
    calls = len(site.grouper.payloads)
    again = site.refresh(DAY1)
    assert again.blocked == (DAY1,) and len(site.grouper.payloads) == calls
    # 输入变了：自动解封、重新归组
    site.grouper.explode = None
    publish(site.behavior_tree, DAY1, "取快递", 18, 0)
    assert site.refresh(DAY1).published == (DAY1,)
    assert site.refresher.blocked_days() == {}


def test_a_deterministic_limit_error_blocks_immediately(tmp_path) -> None:
    site = Site(tmp_path, now=at(DAY3, 12, 0))
    site.seed()
    site.grouper.explode = SceneGroupingLimitError("too many occurrences")
    report = site.refresh(DAY1)
    assert report.blocked == (DAY1,) and site.refresher.pending_days() == ()
    assert any("deterministic failure" in note for note in report.signals)


def test_a_scene_that_cannot_be_materialized_is_dropped_without_failing_the_day(tmp_path) -> None:
    site = Site(tmp_path, now=at(DAY3, 12, 0))
    site.seed()
    site.grouper.duplicate_label = True

    report = site.refresh(DAY1, DAY2)

    assert report.published == (DAY1, DAY2)
    assert [document.address.label for document in site.scene_tree.read_day(DAY2)] == ["准备晚饭"]
    assert any("collides with an earlier scene" in note for note in report.signals)


def test_an_empty_day_publishes_an_empty_generation(tmp_path) -> None:
    site = Site(tmp_path, now=at(DAY3, 12, 0))
    site.behavior_tree.initialize()

    report = site.refresh(DAY1)

    assert report.published == (DAY1,)
    assert site.scene_tree.read_day(DAY1) == ()
    assert site.scene_tree.day_state(DAY1) is not None
    assert site.grouper.payloads == []
    assert site.refresh(DAY1).unchanged == (DAY1,)


def test_stale_days_are_closed_days_without_a_current_generation(tmp_path) -> None:
    site = Site(tmp_path, now=at(DAY3, 12, 0))
    site.seed()
    assert site.refresher.stale_days((DAY1, DAY2)) == (DAY1, DAY2)
    site.refresh(DAY1)
    assert site.refresher.stale_days((DAY1, DAY2)) == (DAY2,)
    assert site.refresher.stale_days(()) == ()
    site.grouper.version = "scripted_grouping_v2+schema000000000000"  # type: ignore[misc]
    assert site.refresher.stale_days((DAY1, DAY2)) == (DAY1, DAY2)  # 改版：夜批会按当前版本重算


def test_pending_items_expire_and_disambiguated_duplicates_are_skipped(tmp_path) -> None:
    site = Site(tmp_path, now=at(DAY3, 12, 0))
    uris = site.seed()
    publish(site.behavior_tree, DAY2, "洗菜 (2)", 19, 43, kind="洗菜", original_name="洗菜")
    site.refresh(DAY1)

    assert pending_before(site.scene_tree, DAY3, expiry_days=1) == ()  # 15 日建立，到 17 日已超过 1 天
    assert [item.producer_uri for item in pending_before(site.scene_tree, DAY3, expiry_days=2)] == [uris["buy"]]
    payload = build_grouping_input(
        DAY2, subject=SUBJECT, behavior_tree=site.behavior_tree, scene_tree=site.scene_tree, lookback_days=3, pending_expiry_days=30
    )
    assert payload is not None
    assert [row.name for row in payload.occurrences] == ["商量晚餐", "洗菜", "看手机"]
    assert build_grouping_input(DAY3, subject=SUBJECT, behavior_tree=site.behavior_tree, scene_tree=site.scene_tree, lookback_days=3, pending_expiry_days=30) is None
    with pytest.raises(ValueError):
        pending_before(site.scene_tree, DAY2, expiry_days=0)


def test_refresh_holds_its_own_lease(tmp_path) -> None:
    site = Site(tmp_path, now=at(DAY3, 12, 0))
    site.seed()
    token = site.lock_store.acquire(site.refresher._lock_key, ttl_seconds=30)
    try:
        with pytest.raises(SceneRefreshBusyError):
            site.refresh(DAY1)
    finally:
        site.lock_store.release(token)
    assert site.refresh(DAY1).published == (DAY1,)


def test_constructor_guards(tmp_path) -> None:
    site = Site(tmp_path, now=at(DAY3, 12, 0))
    with pytest.raises(ValueError, match="overlap"):
        SceneRefresher(
            behavior_tree=site.behavior_tree,
            scene_tree=site.scene_tree,
            grouper=site.grouper,
            subject=SUBJECT,
            progress_root=site.scene_tree.root / "refresh",
            lock_store=site.lock_store,
        )
    with pytest.raises(TypeError, match="grouper"):
        SceneRefresher(behavior_tree=site.behavior_tree, scene_tree=site.scene_tree, grouper=object(), subject=SUBJECT, progress_root=tmp_path / "p", lock_store=site.lock_store)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="lock_store"):
        SceneRefresher(behavior_tree=site.behavior_tree, scene_tree=site.scene_tree, grouper=site.grouper, subject=SUBJECT, progress_root=tmp_path / "p", lock_store=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SceneRefreshConfig(lookback_days=0)
    with pytest.raises(TypeError):
        asyncio.run(site.refresher.refresh_days((datetime(2026, 8, 15, tzinfo=CST),)))  # type: ignore[arg-type]


def test_grouping_input_builder_is_injectable(tmp_path) -> None:
    """参照怎么选是可替换的：注入的装配器说"今天没有可归组的输入"，刷新器就发布空的一代、不调模型。"""

    site = Site(tmp_path, now=at(DAY3, 12, 0))
    site.seed()
    seen: list = []

    def builder(day, *, subject, behavior_tree, scene_tree, lookback_days, pending_expiry_days):
        seen.append((day, subject, lookback_days, pending_expiry_days))
        return None

    refresher = SceneRefresher(
        behavior_tree=site.behavior_tree,
        scene_tree=site.scene_tree,
        grouper=site.grouper,
        subject=SUBJECT,
        progress_root=tmp_path / "scene" / "refresh-2",
        lock_store=site.lock_store,
        config=SceneRefreshConfig(lookback_days=3, pending_expiry_days=30),
        clock=lambda: site.now,
        input_builder=builder,
    )
    report = asyncio.run(refresher.refresh_days((DAY1,)))
    assert report.published == (DAY1,) and site.grouper.payloads == []
    assert seen == [(DAY1, SUBJECT, 3, 30)]
    assert site.scene_tree.read_day(DAY1) == ()
    assert refresher.input_builder is builder and SceneRefresher.__init__.__kwdefaults__["input_builder"] is build_grouping_input
    with pytest.raises(TypeError, match="input_builder"):
        SceneRefresher(behavior_tree=site.behavior_tree, scene_tree=site.scene_tree, grouper=site.grouper, subject=SUBJECT, progress_root=tmp_path / "p", lock_store=site.lock_store, input_builder=object())  # type: ignore[arg-type]
