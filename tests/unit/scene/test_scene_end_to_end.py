"""语义树整条链：概念命中 → 开承诺 → 结算 → 读数 → 投影落盘；以及崩溃留下半截文件时各层怎么表现。"""

from __future__ import annotations

from datetime import timedelta

import pytest

from habitus.scene.ledger import (
    LedgerConfig,
    LedgerStore,
    LedgerStoreError,
    Outcome,
    open_claims_for_day,
    settle_due_all,
)
from habitus.scene.occurrences import ConceptHit, ConceptHitStore
from habitus.scene.views import (
    ViewsConfig,
    ViewsStore,
    behaviour_views,
    entity_slices,
    materialize_views,
    profile_view,
    read_relations,
    residue_candidates,
    standing_intentions,
)
from tests.unit.scene.fixtures import DAY1, at
from tests.unit.scene.ledger_fixtures import (
    CONCEPTS,
    LATE_TO_BREAKFAST,
    LATE_TO_WAKE,
    NOW,
    FixedCoverage,
    TableOpportunities,
    record,
)

CONFIG = LedgerConfig(snapshot_opportunities=16, censor_after=3)
VIEWS = ViewsConfig(min_count=6, min_blocks=3, split_min=3, split_min_blocks=2)


def test_the_chain_runs_from_hits_to_materialized_views(tmp_path) -> None:
    """七h 走一遍：八个晚睡的周五，其中三个早上吃了早饭、五个没吃 → 读出抑制；一条还没结的承诺出现在 intentions。"""

    hits = ConceptHitStore(tmp_path / "scene")
    ledger = LedgerStore(tmp_path / "scene")
    days = [DAY1 + timedelta(days=7 * i) for i in range(8)]
    for i, day in enumerate(days):
        hits.write(record(day, "就寝", 2, 10, ConceptHit("晚睡", "轻"), situations=("周末",) if i % 2 else ()))
        if i in (0, 3, 6):
            hits.write(record(day, "吃了碗面", 7, 30, "早餐"))
        hits.write(record(day, "起床", 9, 30 + 5 * (i % 4), "起床"))  # 09:30 / 09:35 / 09:40 / 09:45：真实时刻有抖动
        hits.write(record(day, "看手机", 21, 0, kind="操作手机"))
    for day in days:
        report = open_claims_for_day(day, hypotheses=(LATE_TO_BREAKFAST, LATE_TO_WAKE), concepts=CONCEPTS, hits=hits, ledger=ledger, opportunities=TableOpportunities(), now=NOW, config=CONFIG)
        assert report.opened == 2

    # 最后一个周五之后一天结算：前七个周五的早餐账都能结（来了 / 等过 3 个看清的机会），第八个还在等。
    later = at(days[-1] + timedelta(days=1), 12, 0)
    reports = settle_due_all((LATE_TO_BREAKFAST, LATE_TO_WAKE), concepts=CONCEPTS, hits=hits, ledger=ledger, coverage=FixedCoverage(1.0), now=later, until=later, config=CONFIG)
    by_identity = {report.hypothesis_identity: report for report in reports}
    assert (by_identity[LATE_TO_BREAKFAST.identity].settled, by_identity[LATE_TO_BREAKFAST.identity].pending) == (7, 1)
    assert by_identity[LATE_TO_WAKE.identity].settled == 8  # 起床每次都在第 1 次机会上来了
    outcomes = [s.outcome for s in ledger.settlements_for(LATE_TO_BREAKFAST.identity)]
    assert outcomes.count(Outcome.OCCURRED) == 3 and outcomes.count(Outcome.CENSORED) == 4

    readings = read_relations((LATE_TO_BREAKFAST, LATE_TO_WAKE), ledger=ledger, concepts=CONCEPTS, config=VIEWS)
    breakfast, wake = readings
    assert breakfast.strength.sufficient and breakfast.strength.p1 == pytest.approx(3 / 7) and breakfast.strength.control == pytest.approx(0.88)
    assert breakfast.strength.interval is not None and breakfast.strength.interval.high < 0 and breakfast.type_reading is not None
    assert breakfast.stability.tested == ("周末",) and breakfast.open_claims == 1
    assert wake.strength.interval is not None and wake.strength.interval.point == pytest.approx(87.5)  # 中位 09:37:30 − 08:10

    intentions = standing_intentions(ledger, {LATE_TO_BREAKFAST.identity: LATE_TO_BREAKFAST, LATE_TO_WAKE.identity: LATE_TO_WAKE}, now=later)
    assert len(intentions) == 1 and intentions[0].consequent == "早餐" and intentions[0].opportunities_passed == 2
    residue = residue_candidates(hits, days, k=5, d=3, claimed=CONCEPTS.claimed_kinds())
    assert [(c.kind_token, c.occurrences, c.ready) for c in residue] == [("操作手机", 8, True)]

    known = {LATE_TO_BREAKFAST.identity: LATE_TO_BREAKFAST, LATE_TO_WAKE.identity: LATE_TO_WAKE}
    store = ViewsStore(tmp_path / "scene")
    written = materialize_views(
        store,
        hypotheses=known,
        readings=readings,
        behaviours=behaviour_views(readings, known),
        intentions=intentions,
        residue=residue,
        profile=profile_view(readings),
        entities=entity_slices(readings, CONCEPTS),
        concepts=CONCEPTS,
        now=later,
        k=5,
        d=3,
    )
    # 多出三份 behaviours/（晚睡的后果面、早餐与起床的前因面）。
    assert len(written) == 8 and store.is_complete()
    late = (tmp_path / "scene" / "views" / "behaviours" / "晚睡.md").read_text(encoding="utf-8")
    assert "## 它导致了什么（后果）" in late and late.count("- → ") == 2
    text = (tmp_path / "scene" / "views" / "relations" / f"{LATE_TO_BREAKFAST.identity}.md").read_text(encoding="utf-8")
    assert "实际 43%（3 次到、4 次右删失） vs 本来 88%" in text and "抑制" in text and "未结算：1 条" in text
    assert "已过 2 次机会" in (tmp_path / "scene" / "views" / "intentions" / "index.md").read_text(encoding="utf-8")


def test_half_written_files_are_noise_truncated_records_are_reported_and_views_stay_incomplete(tmp_path) -> None:
    hits = ConceptHitStore(tmp_path / "scene")
    ledger = LedgerStore(tmp_path / "scene")
    hits.write(record(DAY1, "就寝", 2, 10, ConceptHit("晚睡", "轻")))
    open_claims_for_day(DAY1, hypotheses=(LATE_TO_BREAKFAST,), concepts=CONCEPTS, hits=hits, ledger=ledger, opportunities=TableOpportunities(), now=NOW, config=CONFIG)
    (claim,) = ledger.claims_for(LATE_TO_BREAKFAST.identity)
    day_dir = ledger.claim_path(claim.ref).parent
    # 原子写崩在半路：留下 .tmp——是噪音，扫账时**跳过但不删**（读路径删会杀掉并发写者正在 link 的临时文件），
    # 下一次往这个目录写记录时顺手清掉。
    leftover = day_dir / f".{claim.ref.leaf}.json.{'f' * 32}.tmp"
    leftover.write_bytes(b'{"half":')
    assert ledger.claims_for(LATE_TO_BREAKFAST.identity) == (claim,) and leftover.exists()
    ledger.write_claim(claim)  # 同内容重放：写路径清噪音、记录本身是空操作
    assert not leftover.exists()
    # 记录本身被截断：报"损坏"，不静默当成没有。
    path = ledger.claim_path(claim.ref)
    path.write_bytes(path.read_bytes()[:-5])
    with pytest.raises(LedgerStoreError, match="corrupt"):
        ledger.claims_for(LATE_TO_BREAKFAST.identity)
    # 投影：还没走完 replace_all 就没有标记；读侧据此知道这份不完整。
    store = ViewsStore(tmp_path / "scene")
    assert not store.is_complete()
    (tmp_path / "scene" / "views").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scene" / "views" / "profile.md").write_text("# 半截", encoding="utf-8")
    assert not store.is_complete()
