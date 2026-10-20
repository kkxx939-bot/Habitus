"""行为侧、语义树、预测层的生命周期参与者（26 号清单 6.14）：词表日志能截到哪一版；账本哪些天能整天放、哪些只放索引；
语义树还要读词表的哪些版本。"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta, timezone

from habitus.behavior.kinds.changes import AddClass, ChangeReason, VersionRecord
from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.kinds.store import BehaviorKindStore
from habitus.foresight.ledger import AccountFact, Expectation
from habitus.foresight.ledger.compaction import AccountCompaction
from habitus.runtime.foresight_history_store import HistoryStore
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.lifecycle_behavior import vocabulary_floor
from habitus.runtime.lifecycle_foresight import ForesightParticipant
from tests.unit.behavior.kinds_fixtures import named_classes

CST = timezone(timedelta(hours=8))


class _Context:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def assert_owned(self) -> None:
        return None

    def fence_token(self):  # type: ignore[no-untyped-def]
        raise AssertionError("not used")


def test_the_vocabulary_floor_is_the_oldest_version_anything_still_reads(tmp_path) -> None:
    store = BehaviorKindStore(tmp_path, config=BehaviorKindConfig(cooldown_days=14))
    classes = named_classes(("一", "二", "三", "四"))
    now = datetime(2026, 10, 30, tzinfo=UTC)
    for version, (item, days_ago) in enumerate(zip(classes, (60, 40, 10, 2), strict=True), start=1):
        store.append(
            VersionRecord(version, now - timedelta(days=days_ago), ChangeReason.NIGHTLY, (AddClass(item),)),
            expected_version=version - 1,
        )
    assert vocabulary_floor(store, now=now, derived=()) == 2  # 冷却期里最老的是第 3 版：留第 3 版之后的
    assert vocabulary_floor(store, now=now, derived=(1, 4)) == 1  # 语义树有一晚还记着第 1 版
    assert vocabulary_floor(store, now=now + timedelta(days=30), derived=()) == 4  # 冷却期里没有：到当前


def _expectation(day: date, identity: str) -> Expectation:
    formed = datetime(day.year, day.month, day.day, 9, 0, tzinfo=CST)
    return Expectation(
        expectation_id=identity,
        kind_token="s-k0002",
        day=day,
        formed_at=formed,
        source="日子",
        trigger="rel",
        trigger_instance=f"behavior://{identity}.md",
        window=(formed, formed + timedelta(minutes=30)),
        deadline=formed + timedelta(hours=3),
        basis=(),
        opportunities=3,
        hits=3,
        observed_from=formed,
    )


def test_a_day_is_released_only_when_final_out_of_reach_recorded_and_compacted(tmp_path) -> None:
    ledger = ForesightLedgerStore(tmp_path / "ledger")
    history = HistoryStore(tmp_path / "ledger")
    old, recent = date(2026, 6, 1), date(2026, 9, 25)
    now = datetime(2026, 10, 10, 12, 0, tzinfo=CST)
    for day, identity in ((old, "old"), (recent, "recent")):
        item = _expectation(day, identity)
        ledger.record_expectation(item)
        ledger.append_fact(AccountFact(item.account, item.formed_at, True, 1.0, "准时", identity))
        ledger.mark_final(day, at=item.deadline)
    participant = ForesightParticipant(ledger, history, lane="session", clock=lambda: now)
    asyncio.run(participant.maintain(_Context(now)))  # type: ignore[arg-type]
    assert ledger.expectations_on(old) and ledger.day_of("old") is None  # 没记进单元历史、没压实：只放索引
    assert ledger.day_of("recent") is None  # 最晚时限 + 8 天也过了：放索引
    history.set_recorded_days("session", {old, recent})
    account = _expectation(old, "old").account
    ledger.write_compaction(
        AccountCompaction(
            account=account,
            through=(_expectation(old, "old").formed_at, "old"),
            offsets=((0, 1),),
            gap_count=0,
            longest_gap=0.0,
            last_day=old,
            day_types=(None,),
        )
    )
    report = asyncio.run(participant.maintain(_Context(now)))  # type: ignore[arg-type]
    assert report.released >= 1
    assert ledger.expectations_on(old) == () and old not in ledger.days_with_expectations()
    assert ledger.expectations_on(recent)  # 还在诊断够得着的那一段里：整天留着


def test_the_scene_reports_the_vocabulary_versions_it_still_reads(tmp_path) -> None:
    from habitus.runtime.lifecycle_scene import SceneParticipant, scene_vocabulary_versions
    from habitus.scene.concepts.store import ConceptStore
    from habitus.scene.relations.store import RelationStore

    relations, concepts = RelationStore(tmp_path / "scene"), ConceptStore(tmp_path / "scene")
    assert scene_vocabulary_versions(relations, concepts)() == (0,)  # 从没同步过：第 0 版之后的都要
    report = asyncio.run(SceneParticipant(relations).maintain(_Context(datetime.now(UTC))))  # type: ignore[arg-type]
    assert report.released == 0


def test_a_day_killed_halfway_through_its_release_is_released_on_a_later_round(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """整天释放删到结算目录（含结完标记）之后被杀，接着夜批跑了一次单元历史：这一天在结完低水位以下，照样算结完，单元
    历史不把它从"记过的日子"里拿掉，之后的生命周期轮次把它放完（L10.4；原先成了既不 final 又在水位以下的僵尸日）。"""

    import pytest

    from habitus.foresight import Evidence
    from habitus.runtime.foresight_history import HistoryStage

    ledger = ForesightLedgerStore(tmp_path / "ledger")
    history = HistoryStore(tmp_path / "ledger")
    old, newer = date(2026, 6, 1), date(2026, 6, 2)
    now = datetime(2026, 10, 10, 12, 0, tzinfo=CST)
    for day, identity in ((old, "old"), (newer, "newer")):
        item = _expectation(day, identity)
        ledger.record_expectation(item)
        ledger.append_fact(AccountFact(item.account, item.formed_at, True, 1.0, "准时", identity))
        ledger.mark_final(day, at=item.deadline)
    history.set_recorded_days("session", {old, newer})
    item = _expectation(newer, "newer")
    ledger.write_compaction(
        AccountCompaction(
            account=item.account,
            through=(item.formed_at, "newer"),
            offsets=((0, 2),),
            gap_count=0,
            longest_gap=0.0,
            last_day=newer,
            day_types=(None,),
        )
    )
    real = ledger._discard_day_dir

    def killed_after_settlements(root, day):  # type: ignore[no-untyped-def]
        real(root, day)
        if root == ledger._settlements:
            raise KeyboardInterrupt("killed")

    ledger._discard_day_dir = killed_after_settlements  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        ForesightParticipant(ledger, history, lane="session", clock=lambda: now)._maintain_blocking(_Context(now))

    restarted = ForesightLedgerStore(tmp_path / "ledger")
    stage = HistoryStage(restarted, history, evidence=Evidence(alpha=0.05, half_life=10.0), statuses=("成立",))
    stage._record("session", date(2026, 10, 10))
    participant = ForesightParticipant(restarted, history, lane="session", clock=lambda: now)
    participant._maintain_blocking(_Context(now))

    assert restarted.is_final(old)
    assert restarted.expectations_on(old) == ()


def test_reopen_killed_halfway_never_hides_the_day(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """撤销结完删到一半被杀：这一天要么还算结完、要么回到 ``open_days``，不许两边都不在（L10.4）。"""

    import pytest

    ledger = ForesightLedgerStore(tmp_path / "ledger")
    first = date(2026, 6, 1)
    days = [first + timedelta(days=offset) for offset in range(3)]
    for index, day in enumerate(days):
        item = _expectation(day, f"e{index}")
        ledger.record_expectation(item)
        ledger.mark_final(day, at=item.deadline)
    assert ledger.final_through() == days[-1]

    def killed(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise KeyboardInterrupt("killed")

    ledger._write_watermark = killed  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        ledger.reopen(days[1])

    restarted = ForesightLedgerStore(tmp_path / "ledger")
    assert restarted.is_final(days[1]) or days[1] in restarted.open_days()


def test_an_unreadable_relation_night_does_not_block_the_vocabulary_floor(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """一晚关系表读不了：词表下限跳过它（折叠本来就不接它），其余各晚照常给版本；原先整份解码抛错，行为侧参与者整轮
    失败、物理一支的过期也跟着停（L10.7）。只读词表版本这一个字段。"""

    from habitus.runtime.lifecycle_scene import scene_vocabulary_versions
    from habitus.scene.concepts.store import ConceptStore
    from habitus.scene.relations.store import RelationStore
    from tests.unit.scene.test_relation_transitions_rotation import _nights, _with_a_transition

    store = RelationStore(tmp_path / "scene")
    nights = _nights(2)
    for state, transitions in nights:
        store.write(state, _with_a_transition(state, transitions))
    older, newer = (state.night for state, _ in nights)
    (tmp_path / "scene" / "relations" / "session" / f"{older.isoformat()}.json").write_text("{not json")

    versions = scene_vocabulary_versions(store, ConceptStore(tmp_path / "scene"))()

    assert store.vocabulary_of("session", older) is None
    assert store.vocabulary_of("session", newer) == store.read("session", newer).vocabulary
    assert store.vocabulary_of("session", newer) in versions


def test_housekeeping_purges_what_was_abandoned_long_ago_and_reports_both_stores(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """隔离区与重试账原先只进不出（L10.10）：放弃超过统一寿命的清掉，没放弃的、还在重试的不动；两样都报容量。"""

    from habitus.foundation.lifecycle import BackoffPolicy, DeterministicFailure
    from habitus.infrastructure.lifecycle import Quarantine, RetryLedger
    from habitus.runtime.lifecycle_housekeeping import LifecycleHousekeepingParticipant

    class _Broken(DeterministicFailure, ValueError):
        pass

    long_ago = datetime(2026, 8, 1, tzinfo=UTC)
    now = long_ago + timedelta(days=45)
    clock = [long_ago]
    quarantine = Quarantine(tmp_path, owner="q", clock=lambda: clock[0])
    quarantine.put("old-abandoned", b"x", reason="bad")
    quarantine.put("old-kept", b"y", reason="bad")
    quarantine.abandon("old-abandoned")
    retries = RetryLedger(
        tmp_path,
        owner="r",
        policy=BackoffPolicy(base_seconds=60.0, cap_seconds=60.0),
        max_attempts=3,
        clock=lambda: clock[0],
    )
    retries.record_failure("given-up", _Broken("bad"))
    retries.record_failure("retrying", ConnectionError("down"))
    clock[0] = now
    retries.record_failure("given-up-recently", _Broken("bad"))

    report = LifecycleHousekeepingParticipant((quarantine,), (retries,), keep_days=30)._maintain_blocking(_Context(now))

    assert report.released == 2
    assert [item.item_id for item in quarantine.items()] == ["old-kept"]
    assert retries.entry("given-up") is None
    assert retries.entry("retrying") is not None and retries.entry("given-up-recently") is not None
    assert {reading.name: reading.count for reading in report.readings} == {"quarantine_q": 1, "retry_r": 2}


def test_an_index_entry_waits_for_the_latest_deadline_plus_the_grace_before_it_goes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """6.11：结完的日子，过了那天最晚时限 + 8 天才删预期的索引项——早删了，还可能被触发的同一身份会被重新建出来
    （L10.14 补测：原先这道闸没有测试守着）。"""

    from habitus.runtime.lifecycle_foresight import INDEX_GRACE_DAYS

    ledger = ForesightLedgerStore(tmp_path / "ledger")
    history = HistoryStore(tmp_path / "ledger")
    day = date(2026, 9, 1)
    item = _expectation(day, "indexed")
    ledger.record_expectation(item)
    ledger.mark_final(day, at=item.deadline)
    index = ledger._index / "indexed.json"

    early = item.deadline + timedelta(days=INDEX_GRACE_DAYS - 1)
    ForesightParticipant(ledger, history, lane="session", clock=lambda: early)._maintain_blocking(_Context(early))
    assert index.exists()

    late = item.deadline + timedelta(days=INDEX_GRACE_DAYS, hours=1)
    ForesightParticipant(ledger, history, lane="session", clock=lambda: late)._maintain_blocking(_Context(late))
    assert not index.exists()
    assert ledger.expectations_on(day)  # 整天还没放（单元历史没记过）：只放了索引


def test_releasing_a_day_drops_it_from_the_in_memory_finals(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """6.13：整天放掉的日子也从"结完了"的进程内缓存里拿掉，缓存不随历史一直长（L10.14 补测）。"""

    ledger = ForesightLedgerStore(tmp_path / "ledger")
    day = date(2026, 6, 1)
    item = _expectation(day, "gone")
    ledger.record_expectation(item)
    ledger.mark_final(day, at=item.deadline)
    assert day in ledger._finals

    ledger.discard_day(day)

    assert day not in ledger._finals
