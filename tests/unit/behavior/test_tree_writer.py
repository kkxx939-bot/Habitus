"""重建后的行为树：add-only 发布、前向链接约束与树的枚举。"""

from __future__ import annotations

from datetime import UTC, timedelta

import pytest

from behavior import (
    BehaviorDocumentWriter,
    BehaviorKind,
    BehaviorLinkType,
    BehaviorPublishConflictError,
    BehaviorTree,
    BehaviorURI,
)
from infrastructure.store.locks import ProcessLocalLockStore
from tests.unit.behavior.tree_payloads import (
    action_segment_payload,
    gap_payload,
    local,
    occurrence_payload,
)


def build_writer(tmp_path) -> tuple[BehaviorDocumentWriter, BehaviorTree]:
    tree = BehaviorTree(tmp_path / "behavior-tree")
    writer = BehaviorDocumentWriter(tree, ProcessLocalLockStore(), clock=lambda: local(23, 0))
    return writer, tree


def test_publish_occurrence_and_read_back(tmp_path) -> None:
    writer, tree = build_writer(tmp_path)
    document = writer.publish(BehaviorKind.OCCURRENCE, occurrence_payload())
    assert tree.exists(document.address)
    loaded = tree.read(document.address)
    assert loaded == document
    assert loaded.fields["kind_token"] == "洗手"
    assert loaded.fields["started_at"].endswith("+08:00")
    assert loaded.metadata.revision == 1


def test_replay_semantics_three_ways(tmp_path) -> None:
    """死规则⑤的落盘侧契约：同字节幂等、异内容冲突、异时钟即异字节也冲突。"""

    writer, tree = build_writer(tmp_path)
    first = writer.publish(BehaviorKind.OCCURRENCE, occurrence_payload())

    # 维度一：同 payload + 同 clock → 逐字节相同 → 幂等成功（崩溃重试不许卡死队列）
    replayed = writer.publish(BehaviorKind.OCCURRENCE, occurrence_payload())
    assert replayed == first
    assert len(tree.list_addresses(BehaviorKind.OCCURRENCE)) == 1

    # 维度二：同地址 + 不同内容 → 冲突
    with pytest.raises(BehaviorPublishConflictError, match="different content"):
        writer.publish(BehaviorKind.OCCURRENCE, occurrence_payload(summary="换了一句话"))

    # 维度三：同 payload + 不同 clock → metadata 字节不同 → 冲突。
    # 这钉死了归约层的义务：重试必须复用 stage 时定格的时间戳，不能现取时钟。
    other_clock = BehaviorDocumentWriter(
        tree, ProcessLocalLockStore(), clock=lambda: local(23, 30)
    )
    with pytest.raises(BehaviorPublishConflictError, match="different content"):
        other_clock.publish(BehaviorKind.OCCURRENCE, occurrence_payload())


def test_forward_link_requires_existing_target(tmp_path) -> None:
    writer, tree = build_writer(tmp_path)
    cause = writer.publish(BehaviorKind.OCCURRENCE, occurrence_payload())
    cause_uri = BehaviorURI.from_address(cause.address)

    follow = writer.publish(
        BehaviorKind.OCCURRENCE,
        action_segment_payload(
            started_at=local(19, 40),
            last_observed_at=local(19, 41),
            onset_available_at=local(19, 40, 2),
        ),
        links=((BehaviorLinkType.RESULTS_FROM, cause_uri),),
    )
    loaded = tree.read(follow.address)
    assert len(loaded.links) == 1
    assert str(loaded.links[0].to_uri) == str(cause_uri)
    assert loaded.links[0].link_type is BehaviorLinkType.RESULTS_FROM
    assert not hasattr(loaded, "backlinks")  # 只存前向，读侧取闭包；backlinks 槽位已整体退役

    missing = str(cause_uri).replace("洗了手", "不存在的行为")
    with pytest.raises(BehaviorPublishConflictError, match="does not exist"):
        writer.publish(
            BehaviorKind.OCCURRENCE,
            action_segment_payload(
                started_at=local(21, 0),
                last_observed_at=local(21, 1),
                onset_available_at=local(21, 0, 2),
            ),
            links=((BehaviorLinkType.CONCURRENT_WITH, missing),),
        )


def test_gap_publish_and_no_links(tmp_path) -> None:
    writer, tree = build_writer(tmp_path)
    document = writer.publish(BehaviorKind.GAP, gap_payload())
    assert tree.exists(document.address)
    occurrence = writer.publish(BehaviorKind.OCCURRENCE, occurrence_payload())
    with pytest.raises(ValueError, match="only occurrences carry forward links"):
        writer.publish(
            BehaviorKind.GAP,
            gap_payload(started_at=local(21, 10), ended_at=local(21, 20)),
            links=(
                (
                    BehaviorLinkType.CONCURRENT_WITH,
                    BehaviorURI.from_address(occurrence.address),
                ),
            ),
        )


def test_timeline_enumeration_interleaves_kinds_by_day(tmp_path) -> None:
    """一天的完整图景 = occurrences + gaps 两次范围读，同一套日期目录。"""

    writer, tree = build_writer(tmp_path)
    writer.publish(BehaviorKind.OCCURRENCE, occurrence_payload())
    writer.publish(
        BehaviorKind.OCCURRENCE,
        action_segment_payload(
            started_at=local(20, 50),
            last_observed_at=local(20, 51),
            onset_available_at=local(20, 50, 2),
        ),
    )
    writer.publish(BehaviorKind.GAP, gap_payload())

    occurrences = tree.list_addresses(BehaviorKind.OCCURRENCE)
    gaps = tree.list_addresses(BehaviorKind.GAP)
    assert [address.name for address in occurrences] == ["洗了手", "起身走开"]
    assert [address.name for address in gaps] == ["没读懂"]

    timeline = sorted(
        (*occurrences, *gaps),
        key=lambda address: address.started_at.astimezone(tz=None),
    )
    assert [address.name for address in timeline] == ["洗了手", "没读懂", "起身走开"]


def test_cross_midnight_occurrence_lands_on_its_local_day(tmp_path) -> None:
    """判别性输入：东八区 00:30 的 UTC 日期是前一天——折 UTC 的旧坑一复活此测试即红。"""

    writer, tree = build_writer(tmp_path)
    small_hours = occurrence_payload(
        occurred_on=local(0, 30).date(),
        started_at=local(0, 30),
        last_observed_at=local(0, 39),
        onset_available_at=local(0, 31),
        basis=(),
        goal=None,
        name="起夜",
        kind_token="起夜",
        summary="起夜上了个厕所",
    )
    document = writer.publish(BehaviorKind.OCCURRENCE, small_hours)
    uri = BehaviorURI.from_address(document.address)
    assert uri.segments[1:4] == ("2026", "08", "16")           # 本地日
    utc_day = document.address.started_at.astimezone(UTC).date()
    assert utc_day.isoformat() == "2026-08-15"                 # UTC 日确实不同 → 输入有判别力
    assert document.address.started_at.utcoffset() == timedelta(hours=8)


def test_directory_capacity_holds_under_concurrent_publishes(tmp_path) -> None:
    """容量不变量跨文档，目录锁键让并发发布串行——不许合谋越界后把整目录变成不可读。"""

    import threading
    import time

    from behavior import BehaviorTreeConfig, BehaviorTreeIntegrityError

    tree = BehaviorTree(
        tmp_path / "behavior-tree",
        tree_config=BehaviorTreeConfig(max_children_per_directory=2),
    )
    lock_store = ProcessLocalLockStore()
    writer = BehaviorDocumentWriter(tree, lock_store, clock=lambda: local(23, 0))

    outcomes: list[str] = []
    barrier = threading.Barrier(3)

    def publish(name: str) -> None:
        barrier.wait()
        payload = occurrence_payload(
            name=name, kind_token=name, goal=None, basis=(), summary=name
        )
        # ProcessLocalLockStore 争用即拒（TimeoutError），不排队——按仓库惯例有界重试。
        for _attempt in range(50):
            try:
                writer.publish(BehaviorKind.OCCURRENCE, payload)
                outcomes.append("ok")
                return
            except TimeoutError:
                time.sleep(0.005)
            except BehaviorTreeIntegrityError:
                outcomes.append("capacity")
                return
        outcomes.append("lock_starved")

    threads = [threading.Thread(target=publish, args=(f"动作{i}",)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["capacity", "ok", "ok"]
    # 关键断言：目录没有被合谋越界，读侧枚举完整可用
    assert len(tree.list_addresses(BehaviorKind.OCCURRENCE)) == 2
