"""③ 概念命中的模型与存储：与行为树同构的地址、按天完成标记及其纪律（重做先撤、口径一致）、按时刻窗读。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from habitus.behavior.model import BehaviorAddress
from habitus.behavior.uri import BehaviorURI
from habitus.scene.codec import SceneRecordError
from habitus.scene.occurrences import ConceptHit, ConceptHits, ConceptHitsError, ConceptHitStore, ConceptHitStoreError
from habitus.scene.occurrences.document import decode, encode
from tests.unit.scene.fixtures import CST, DAY1, DAY2, DAY3, Site, at, publish

NOW = datetime(2026, 8, 18, 3, 0, tzinfo=UTC)
MAPPER = "scene_concept_mapper_prompt_v2+schemaabc+emb:fake+llm:fake+concepts:0123"
OTHER_MAPPER = MAPPER.replace("0123", "4567")


def hits_for(
    uri: str,
    *hits: ConceptHit,
    situations: tuple[ConceptHit, ...] = (),
    unresolved: tuple[str, ...] = (),
    baseline: dict[str, str] | None = None,
    mapper: str = MAPPER,
    signals: tuple[str, ...] = (),
    lasts_minutes: int = 10,
) -> ConceptHits:
    started = BehaviorURI.parse(uri).to_address().started_at
    return ConceptHits(
        occurrence_uri=uri,
        kind_token="就寝",
        last_observed_at=started + timedelta(minutes=lasts_minutes),
        hits=hits,
        situation_hits=situations,
        unresolved=unresolved,
        baseline_snapshot=baseline or {},
        mapper=mapper,
        mapped_at=NOW,
        signals=signals,
    )


def test_a_record_must_point_at_an_occurrence_and_stay_self_consistent(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "就寝", 2, 10)
    record = hits_for(uri, ConceptHit("晚睡", "重"), ConceptHit("熬夜工作"), situations=(ConceptHit("出差中"),), unresolved=("早餐",))
    assert record.graded_hits == {"晚睡": "重", "熬夜工作": None} and record.graded_situations == {"出差中": None}
    assert record.unresolved_identities == {"早餐"} and record.started_at == at(DAY1, 2, 10)
    # 账本判前件命中用 ``matches``：不给档任何档都算，给了档要相同——「晚睡@重」不能被一个轻档夜晚命中。
    assert record.matches("晚睡") and record.matches("晚睡", "重") and not record.matches("晚睡", "轻")
    assert record.matches("出差中") and not record.matches("早餐")

    gap_uri = str(BehaviorURI.from_address(BehaviorAddress.gap(DAY1, "没读懂", at(DAY1, 20, 0))))
    with pytest.raises(ConceptHitsError, match="occurrence document"):
        hits_for(gap_uri)
    with pytest.raises(ConceptHitsError, match="repeats"):
        hits_for(uri, ConceptHit("晚睡"), ConceptHit("晚睡", "重"))
    with pytest.raises(ConceptHitsError, match="both"):
        hits_for(uri, ConceptHit("晚睡"), situations=(ConceptHit("晚睡"),))
    with pytest.raises(ConceptHitsError, match="unresolved concept cannot also be a hit"):
        hits_for(uri, ConceptHit("晚睡"), unresolved=("晚睡",))
    with pytest.raises(ConceptHitsError, match="precede"):
        hits_for(uri, lasts_minutes=-5)  # 上游不自洽不静默抹平
    with pytest.raises(ConceptHitsError, match="single line"):
        hits_for(uri, baseline={"就寝 常态": "23:30\n"})
    with pytest.raises(ConceptHitsError):
        ConceptHit("")


def test_a_record_round_trips_and_a_tampered_body_is_rejected(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "就寝", 2, 10)
    record = hits_for(
        uri, ConceptHit("晚睡", "重"), unresolved=("早餐",), baseline={"就寝 常态时刻": "23:30"}, signals=("structured: answered on attempt 2",)
    )
    text = encode(record)
    assert "类别：就寝 · 最后所见 02:20" in text and "命中：晚睡（重）" in text and "情境：（无）" in text
    assert "未决：早餐" in text and "常态：就寝 常态时刻 = 23:30" in text and "信号：structured: answered on attempt 2" in text
    assert decode(text) == record
    with pytest.raises(SceneRecordError, match="canonical rendering"):
        decode(text.replace("命中：晚睡（重）", "命中：晚睡"))
    with pytest.raises(SceneRecordError, match="occurrence of its address"):
        decode(text, expected_uri=publish(site.behavior_tree, DAY1, "起床", 9, 40))


def test_store_mirrors_the_behaviour_tree_address_and_reads_a_day_in_time_order(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    late = publish(site.behavior_tree, DAY1, "就寝", 23, 50)
    early = publish(site.behavior_tree, DAY1, "早餐", 8, 0)
    store = ConceptHitStore(tmp_path / "scene")
    path = store.write(hits_for(late, ConceptHit("晚睡", "轻")))
    store.write(hits_for(early, ConceptHit("早餐")))

    leaf = BehaviorURI.parse(late).to_address().identity_name
    assert path == tmp_path / "scene" / "occurrences" / "2026" / "08" / "15" / f"{leaf}.md"
    assert path.relative_to(tmp_path / "scene") == site.behavior_tree.path_for(BehaviorURI.parse(late).to_address()).relative_to(
        site.behavior_tree.root
    )
    assert [record.occurrence_uri for record in store.read_day(DAY1)] == [early, late]
    assert store.exists(BehaviorURI.parse(late).to_address())
    assert not store.exists(BehaviorAddress.occurrence(DAY2, "就寝", at(DAY2, 23, 50)))
    with pytest.raises(ConceptHitStoreError, match="occurrences only"):
        store.path_for(BehaviorAddress.gap(DAY1, "没读懂", at(DAY1, 20, 0)))


def test_a_window_is_read_by_moment_across_day_directories(tmp_path) -> None:
    """窗是时刻，目录是本地日：横跨日界的窗要能完整取到、按时刻排，账本不用自己拼日子。"""

    site = Site(tmp_path, now=NOW)
    store = ConceptHitStore(tmp_path / "scene")
    d1_late = publish(site.behavior_tree, DAY1, "就寝", 23, 50)
    d2_early = publish(site.behavior_tree, DAY2, "早餐", 7, 30)
    d2_lunch = publish(site.behavior_tree, DAY2, "午饭", 12, 0)
    d3_early = publish(site.behavior_tree, DAY3, "早餐", 7, 30)
    for uri in (d1_late, d2_early, d2_lunch, d3_early):
        store.write(hits_for(uri))

    window = store.read_window(at(DAY1, 23, 0), at(DAY2, 11, 0))  # 短窗 12h
    assert [record.occurrence_uri for record in window] == [d1_late, d2_early]
    assert store.read_window(at(DAY2, 11, 0), at(DAY2, 11, 0)) == ()
    # 起点带别的偏移也不漏也不多：UTC 23:31 = 东八区次日 07:31，07:30 的早餐刚好在窗前，窗内只剩 DAY2 12:00 那条。
    assert [record.occurrence_uri for record in store.read_window(datetime(2026, 8, 15, 23, 31, tzinfo=UTC), at(DAY2, 13, 0))] == [d2_lunch]
    assert [record.occurrence_uri for record in store.read_window(datetime(2026, 8, 15, 23, 30, tzinfo=UTC), at(DAY2, 13, 0))] == [d2_early, d2_lunch]
    assert window[0].started_at.tzinfo == CST


def test_completion_marker_counts_checks_the_mapper_and_summarises_the_day(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "就寝", 23, 50)
    other = publish(site.behavior_tree, DAY1, "早餐", 8, 0)
    store = ConceptHitStore(tmp_path / "scene")
    store.write(hits_for(uri, ConceptHit("晚睡", "轻"), situations=(ConceptHit("周末"),)))
    store.write(hits_for(other, unresolved=("早餐",)))

    assert store.days_done() == frozenset() and store.read_marker(DAY1) is None  # 记录落了、标记没落，这天不算做完
    with pytest.raises(ConceptHitStoreError, match="claims 3"):
        store.complete_day(DAY1, records=3, completed_at=NOW, mapper=MAPPER)
    with pytest.raises(ConceptHitStoreError, match="another mapper"):
        store.complete_day(DAY1, records=2, completed_at=NOW, mapper=OTHER_MAPPER)  # 两代口径不许混着盖章
    marker = store.complete_day(DAY1, records=2, completed_at=NOW, mapper=MAPPER)
    assert (marker.records, marker.unresolved, marker.concepts) == (2, 1, {"晚睡", "周末"})
    assert store.read_marker(DAY1) == marker
    # 一天没有任何 occurrence 也要有标记，否则夜批永远重扫它。
    store.complete_day(DAY3, records=0, completed_at=NOW, mapper=MAPPER)
    assert store.days_done() == {DAY1, DAY3}
    assert store.days_done(mapper=MAPPER) == {DAY1, DAY3} and store.days_done(mapper=OTHER_MAPPER) == frozenset()
    assert store.discard_completion(DAY1) and not store.discard_completion(DAY1)
    assert store.days_done() == {DAY3}


def test_redoing_a_day_withdraws_the_marker_in_the_same_step_and_clears_noise(tmp_path) -> None:
    """删记录还留着"这天做完了"，崩溃后留下的就是"做完了但内容不对"——``retain_only`` 同一步撤标记。"""

    site = Site(tmp_path, now=NOW)
    kept = publish(site.behavior_tree, DAY1, "就寝", 23, 50)
    gone = publish(site.behavior_tree, DAY1, "看手机", 22, 0)
    store = ConceptHitStore(tmp_path / "scene")
    store.write(hits_for(kept, ConceptHit("晚睡", "轻")))
    store.write(hits_for(gone))
    store.complete_day(DAY1, records=2, completed_at=NOW, mapper=MAPPER)
    day_dir = store.path_for(BehaviorURI.parse(kept).to_address()).parent
    leftover = day_dir / f".{BehaviorURI.parse(gone).to_address().identity_name}.md.{'0' * 32}.tmp"
    leftover.write_bytes(b"half")
    (day_dir / ".DS_Store").write_bytes(b"\x00")  # 用 Finder 打开一次就会出现；不能把一天锁死

    assert [r.occurrence_uri for r in store.read_day(DAY1)] == [gone, kept]  # 噪音读时跳过
    removed = store.retain_only(DAY1, frozenset({BehaviorURI.parse(kept).to_address().identity_name}))
    assert removed == (BehaviorURI.parse(gone).to_address().identity_name,)
    assert store.days_done() == frozenset()  # 标记随删除一起撤了
    assert not leftover.exists() and not (day_dir / ".DS_Store").exists()
    assert [record.occurrence_uri for record in store.read_day(DAY1)] == [kept]
    store.complete_day(DAY1, records=1, completed_at=NOW + timedelta(minutes=1), mapper=MAPPER)
    (day_dir / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ConceptHitStoreError, match="Markdown"):
        store.read_day(DAY1)


def test_store_enforces_capacity_and_canonical_leaf_names(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    store = ConceptHitStore(tmp_path / "scene", max_directory_entries=2)
    a = publish(site.behavior_tree, DAY1, "就寝", 23, 50)
    b = publish(site.behavior_tree, DAY1, "早餐", 8, 0)
    c = publish(site.behavior_tree, DAY1, "午饭", 12, 0)
    store.write(hits_for(a))
    store.write(hits_for(b))
    store.write(hits_for(a, ConceptHit("晚睡")))  # 覆写不占新位
    with pytest.raises(ConceptHitStoreError, match="capacity"):
        store.write(hits_for(c))
    day_dir = store.path_for(BehaviorURI.parse(a).to_address()).parent
    canonical = store.path_for(BehaviorURI.parse(b).to_address())
    canonical.rename(day_dir / canonical.name.replace("早餐", "早餐 "))  # 非规范叶名
    with pytest.raises(ConceptHitStoreError, match="canonical"):
        store.read_day(DAY1)
