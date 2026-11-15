"""③ 命中记录的模型与存储：行为概念从类编号现读、「待定」没有类、记录带 lane 与判过的日历情境、与行为树同构的地址、
按天完成标记及其纪律（重做先撤、口径一致、字段严格）、按时刻窗读。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from habitus.behavior.model import BehaviorAddress
from habitus.behavior.uri import BehaviorURI
from habitus.scene.codec import SceneRecordError
from habitus.scene.occurrences import ConceptHit, ConceptHits, ConceptHitsError, ConceptHitStore, ConceptHitStoreError
from habitus.scene.occurrences.document import decode, encode
from tests.unit.kind_ids import kind_id
from tests.unit.scene.fixtures import DAY1, DAY2, DAY3, Site, at, publish

NOW = datetime(2026, 8, 18, 3, 0, tzinfo=UTC)
HITS_VERSION = "scene_hits_v1"
OTHER_HITS_VERSION = "scene_hits_v0"
DIGEST = "0123456789abcdef"
BEDTIME = kind_id("就寝")


def hits_for(
    uri: str,
    *,
    situations: tuple[ConceptHit, ...] = (),
    checked: tuple[str, ...] = (),
    hits_version: str = HITS_VERSION,
    lasts_minutes: int = 10,
    kind_token: str = BEDTIME,
    classified: bool = True,
    lane: str = "session",
) -> ConceptHits:
    started = BehaviorURI.parse(uri).to_address().started_at
    return ConceptHits(
        occurrence_uri=uri,
        kind_token=kind_token,
        classified=classified,
        last_observed_at=started + timedelta(minutes=lasts_minutes),
        situation_hits=situations,
        situations_checked=checked,
        hits_version=hits_version,
        sealed_at=NOW,
        lane=lane,
    )


def test_a_record_must_point_at_an_occurrence_and_stay_self_consistent(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "就寝", 2, 10)
    record = hits_for(uri, situations=(ConceptHit("周末"),), checked=("周五",))
    # 行为概念不存，从编号现读：这一条是就寝，它算的就是就寝那个类的行为概念
    assert record.base_concept == BEDTIME and record.started_at == at(DAY1, 2, 10)
    # 成立的情境必然判过：没列进 checked 的补进去
    assert record.situations_checked == ("周五", "周末")

    gap_uri = str(BehaviorURI.from_address(BehaviorAddress.gap(DAY1, "没读懂", at(DAY1, 20, 0))))
    with pytest.raises(ConceptHitsError, match="occurrence document"):
        hits_for(gap_uri)
    with pytest.raises(ConceptHitsError, match="repeats"):
        hits_for(uri, situations=(ConceptHit("周末"), ConceptHit("周末")))
    with pytest.raises(ConceptHitsError, match="precede"):
        hits_for(uri, lasts_minutes=-5)  # 上游不自洽不静默抹平
    with pytest.raises(ConceptHitsError):
        ConceptHit("")
    with pytest.raises(ConceptHitsError, match="lane must be non-empty"):
        hits_for(uri, lane=" ")
    assert record.lane == "session"


def test_a_pending_record_has_no_class(tmp_path) -> None:
    """「待定」：没有类，不算任何行为概念；日历情境照常记。"""

    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "讨论提示词", 15, 17)
    pending = hits_for(uri, situations=(ConceptHit("周末"),), kind_token="s-待定", classified=False)
    assert pending.base_concept is None
    assert decode(encode(pending)) == pending and "待定 · 最后所见 15:27" in encode(pending)


def test_a_record_round_trips_and_a_tampered_body_is_rejected(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "就寝", 2, 10)
    record = hits_for(uri, situations=(ConceptHit("周末"),), checked=("周五",))
    text = encode(record)
    # 编号不进正文（类改名不该让盘上记录对不上规范渲染）
    assert "已归类 · 最后所见 02:20" in text and BEDTIME not in text.split("<!--")[0]
    assert "情境：周末（判过 周五、周末）" in text
    assert decode(text) == record
    with pytest.raises(SceneRecordError, match="canonical rendering"):
        decode(text.replace("情境：周末", "情境：周日"))
    with pytest.raises(SceneRecordError, match="occurrence of its address"):
        decode(text, expected_uri=publish(site.behavior_tree, DAY1, "起床", 9, 40))
    with pytest.raises(SceneRecordError, match="must be a boolean"):
        decode(text.replace('"classified":true', '"classified":"yes"'))
    with pytest.raises(SceneRecordError, match="must be text"):
        decode(text.replace('"situation_hits":["周末"]', '"situation_hits":[7]'))
    with pytest.raises(SceneRecordError, match="lane"):
        decode(text.replace('"lane":"session"', '"lane":3'))


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('"classified":true', '"classified":true,"hits":[]'),
        ('"situations_checked":[]', '"situations_checked":[],"unresolved":{}'),
        ('"classified":true', '"baseline_snapshot":{},"classified":true'),
        ('"record_type"', '"recent_digest":null,"record_type"'),
        ('"occurrence_uri"', '"mapper":"x","occurrence_uri"'),
        ('"hits_version":"scene_hits_v1",', ""),
        ('"situation_hits"', '"signals":[],"situation_hits"'),
    ],
)
def test_a_record_with_extra_or_missing_fields_is_refused(tmp_path, old: str, new: str) -> None:
    """命中记录的字段集合是严格的：旧盘面的细分命中、未决、常态快照、材料摘要这些键一出现就拒（不静默忽略），少了也拒。"""

    site = Site(tmp_path, now=NOW)
    text = encode(hits_for(publish(site.behavior_tree, DAY1, "就寝", 2, 10)))
    assert old in text
    with pytest.raises(SceneRecordError, match="unexpected or missing fields"):
        decode(text.replace(old, new))


def test_store_mirrors_the_behaviour_tree_address_and_reads_a_day_in_time_order(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    late = publish(site.behavior_tree, DAY1, "就寝", 23, 50)
    early = publish(site.behavior_tree, DAY1, "早餐", 8, 0)
    store = ConceptHitStore(tmp_path / "scene")
    path = store.write(hits_for(late))
    store.write(hits_for(early, kind_token=kind_id("早餐")))

    leaf = BehaviorURI.parse(late).to_address().identity_name
    assert path == tmp_path / "scene" / "occurrences" / "2026" / "08" / "15" / f"{leaf}.md"
    assert path.relative_to(tmp_path / "scene") == site.behavior_tree.path_for(
        BehaviorURI.parse(late).to_address()
    ).relative_to(site.behavior_tree.root)
    assert [record.occurrence_uri for record in store.read_day(DAY1)] == [early, late]
    assert store.exists(BehaviorURI.parse(late).to_address())
    assert not store.exists(BehaviorAddress.occurrence(DAY2, "就寝", at(DAY2, 23, 50)))
    with pytest.raises(ConceptHitStoreError, match="occurrences only"):
        store.path_for(BehaviorAddress.gap(DAY1, "没读懂", at(DAY1, 20, 0)))


def test_completion_marker_counts_checks_the_hits_version_and_summarises_the_day(tmp_path) -> None:
    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "就寝", 23, 50)
    other = publish(site.behavior_tree, DAY1, "早餐", 8, 0)
    store = ConceptHitStore(tmp_path / "scene")
    store.write(hits_for(uri, situations=(ConceptHit("周末"),)))
    store.write(hits_for(other, kind_token="s-待定", classified=False))

    assert store.days_done() == frozenset() and store.read_marker(DAY1) is None  # 记录落了、标记没落，这天不算做完
    with pytest.raises(ConceptHitStoreError, match="claims 3"):
        store.complete_day(DAY1, records=3, completed_at=NOW, hits_version=HITS_VERSION, digest=DIGEST)
    with pytest.raises(ConceptHitStoreError, match="another hits version"):
        store.complete_day(DAY1, records=2, completed_at=NOW, hits_version=OTHER_HITS_VERSION, digest=DIGEST)  # 两代口径不许混着盖章
    marker = store.complete_day(DAY1, records=2, completed_at=NOW, hits_version=HITS_VERSION, digest=DIGEST)
    # 当天出现过的概念：行为概念（编号；「待定」没有）+ 情境
    assert (marker.records, marker.concepts, marker.digest) == (2, {BEDTIME, "周末"}, DIGEST)
    assert store.read_marker(DAY1) == marker
    # 一天没有任何 occurrence 也要有标记，否则换代永远重扫它。
    store.complete_day(DAY3, records=0, completed_at=NOW, hits_version=HITS_VERSION, digest=DIGEST)
    assert store.days_done() == {DAY1, DAY3}
    assert store.discard_completion(DAY1) and not store.discard_completion(DAY1)
    assert store.days_done() == {DAY3}


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('"digest":"0123456789abcdef"', '"digest":"0123456789abcdef","unresolved":0', "unexpected or missing fields"),
        ('"digest":"0123456789abcdef"', '"mapper":"x"', "unexpected or missing fields"),
        ('"records":1', '"records":"1"', "wrong type"),
        ('"day":"2026-08-15"', '"day":"2026-08-16"', "another day"),
    ],
)
def test_a_completion_marker_of_another_shape_is_refused_naming_the_file(tmp_path, old: str, new: str, message: str) -> None:
    """完成章的字段集合是严格的：旧盘面的未决条数、映射口径一出现就拒，信息里带文件名。"""

    site = Site(tmp_path, now=NOW)
    store = ConceptHitStore(tmp_path / "scene")
    store.write(hits_for(publish(site.behavior_tree, DAY1, "就寝", 23, 50)))
    store.complete_day(DAY1, records=1, completed_at=NOW, hits_version=HITS_VERSION, digest=DIGEST)
    marker = store.path_for(BehaviorAddress.occurrence(DAY1, "就寝", at(DAY1, 23, 50))).parent / ".done.json"
    text = marker.read_text(encoding="utf-8")
    assert old in text
    marker.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(ConceptHitStoreError, match=rf"\.done\.json.*{message}"):
        store.read_marker(DAY1)


def test_redoing_a_day_withdraws_the_marker_in_the_same_step_and_clears_noise(tmp_path) -> None:
    """删记录还留着"这天做完了"，崩溃后留下的就是"做完了但内容不对"——``retain_only`` 同一步撤标记。"""

    site = Site(tmp_path, now=NOW)
    kept = publish(site.behavior_tree, DAY1, "就寝", 23, 50)
    gone = publish(site.behavior_tree, DAY1, "看手机", 22, 0)
    store = ConceptHitStore(tmp_path / "scene")
    store.write(hits_for(kept))
    store.write(hits_for(gone))
    store.complete_day(DAY1, records=2, completed_at=NOW, hits_version=HITS_VERSION, digest=DIGEST)
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
    store.complete_day(DAY1, records=1, completed_at=NOW + timedelta(minutes=1), hits_version=HITS_VERSION, digest=DIGEST)
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
    store.write(hits_for(a, lasts_minutes=20))  # 覆写不占新位
    with pytest.raises(ConceptHitStoreError, match="capacity"):
        store.write(hits_for(c))
    day_dir = store.path_for(BehaviorURI.parse(a).to_address()).parent
    canonical = store.path_for(BehaviorURI.parse(b).to_address())
    canonical.rename(day_dir / canonical.name.replace("早餐", "早餐 "))  # 非规范叶名
    with pytest.raises(ConceptHitStoreError, match="canonical"):
        store.read_day(DAY1)


def test_a_finished_day_is_decoded_once_and_read_again_only_when_its_marker_changes(tmp_path) -> None:
    """26 号清单 6.10：做完的日子按（日子, 完成标记）缓存解码好的命中，封口与关系检验共用；重做（标记变了）重读；
    没做完的不缓存。"""

    site = Site(tmp_path, now=NOW)
    uri = publish(site.behavior_tree, DAY1, "就寝", 23, 50)
    store = ConceptHitStore(tmp_path / "scene")
    store.write(hits_for(uri))
    decoded: list[object] = []
    original = store._decode

    def counting(payload, address):  # type: ignore[no-untyped-def]
        decoded.append(address)
        return original(payload, address)

    store._decode = counting  # type: ignore[method-assign]
    store.read_day(DAY1)
    store.read_day(DAY1)
    assert len(decoded) == 2  # 没做完：每次都读盘
    store.complete_day(DAY1, records=1, completed_at=NOW, hits_version=HITS_VERSION, digest=DIGEST)
    decoded.clear()
    first = store.read_day(DAY1)
    assert store.read_day(DAY1) == first and len(decoded) == 1  # 做完了：只解码一次
    store.retain_only(DAY1, frozenset({BehaviorURI.parse(uri).to_address().identity_name}))
    store.write(hits_for(uri, situations=(ConceptHit("周末"),)))
    store.complete_day(DAY1, records=1, completed_at=NOW + timedelta(days=1), hits_version=HITS_VERSION, digest=DIGEST)
    (redone,) = store.read_day(DAY1)
    assert redone.situation_hits == (ConceptHit("周末"),)  # 重做过：标记变了、重读
