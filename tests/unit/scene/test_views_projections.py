"""⑤ residue / profile / entities 三个投影与落盘（intentions/ 2026-09-30 删：前因与后果是两个行为，没结的账不是"前提"）。"""

from __future__ import annotations

from datetime import timedelta

from habitus.scene.concepts import ConceptRole, ConceptSet
from habitus.scene.hypotheses import Direction, TypePrior
from habitus.scene.ledger import Claim, LedgerStore, OpportunityPass, Outcome, Settlement
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
)
from tests.unit.scene.concept_fixtures import ALL_CONCEPTS, concept
from tests.unit.scene.fixtures import DAY1, DAY2, DAY3, at
from tests.unit.scene.ledger_fixtures import (
    BEDTIME,
    BOOKING,
    BOOKING_TO_BALL,
    COFFEE,
    CONCEPTS,
    LATE_TO_BREAKFAST,
    NOW,
    WAKE,
    daily_snapshot,
    hypothesis,
    record,
    uri_for,
)

WITH_A = concept("和A一起", "A 在场", role=ConceptRole.OBJECT)
CONCEPTS_WITH_A = ConceptSet((*ALL_CONCEPTS, BOOKING, WAKE, COFFEE, BEDTIME, WITH_A))


def test_residue_counts_unmapped_kinds_and_says_how_far_from_upgrading(tmp_path) -> None:
    hits = ConceptHitStore(tmp_path / "scene")
    hits.write(record(DAY1, "看手机", 21, 0, kind="操作手机"))
    hits.write(record(DAY1, "刷手机", 22, 0, kind="操作手机"))
    hits.write(record(DAY2, "看手机", 21, 0, kind="操作手机"))
    hits.write(record(DAY2, "就寝", 2, 10, ConceptHit("晚睡", "轻")))  # 命中了：不是残差
    unresolved = record(DAY3, "吃了碗面", 7, 0, kind="吃饭")
    hits.write(Claim.__new__(Claim) and unresolved.__class__(**{**unresolved.__dict__, "unresolved": ("早餐",)}))  # 未决：材料没给到，不是残差
    hits.write(record(DAY3, "发呆", 15, 0, kind="发呆"))
    found = residue_candidates(hits, (DAY1, DAY2, DAY3), k=5, d=3)
    assert [(c.kind_token, c.occurrences, c.days, c.short_by_occurrences, c.short_by_days) for c in found] == [("操作手机", 3, 2, 2, 1), ("发呆", 1, 1, 4, 2)]
    assert not found[0].ready and residue_candidates(hits, (DAY1, DAY2, DAY3), k=3, d=2)[0].ready
    # 升级之后：那个 kind 被新概念认领，历史记录不回填也不再当候选。
    assert [c.kind_token for c in residue_candidates(hits, (DAY1, DAY2, DAY3), k=3, d=2, claimed=frozenset({"操作手机"}))] == ["发呆"]


def test_profile_ranks_stable_relations_and_entities_slice_by_object_concepts(tmp_path) -> None:
    ledger = LedgerStore(tmp_path / "scene")
    with_a = hypothesis("晚睡", consequent="咖啡", aspect=LATE_TO_BREAKFAST.aspect, direction=Direction.UP, type_prior=TypePrior.PROMOTING, note="x")
    for i in range(12):
        day = DAY1 + timedelta(days=7 * i)  # 一周一次：块长 7 天，12 条跨 12 块
        for hyp, outcome, situations in (
            (LATE_TO_BREAKFAST, Outcome.ABSENT, ()),
            (with_a, Outcome.OCCURRED if i < 6 else Outcome.ABSENT, ("和A一起",) if i < 6 else ()),
        ):
            trigger = uri_for(day, "就寝", 2, 10)
            snapshot = daily_snapshot(hyp.consequent, at(day, 2, 10), 8)
            claim = Claim(
                hypothesis_identity=hyp.identity,
                hypothesis_fingerprint=hyp.fingerprint,
                aspect=hyp.aspect,
                trigger_uri=trigger,
                antecedent_hits=(ConceptHit("晚睡", "轻"),),
                antecedent_uris=(trigger,),
                situation_snapshot=situations,
                situations_checked=("和A一起",),
                control=snapshot,
                created_at=NOW,
            )
            ledger.write_claim(claim)
            first = snapshot.at(1)
            assert first is not None
            settlement = (
                Settlement(claim.ref, outcome, NOW, observed_at=first.at, fulfilling_uri=uri_for(day, "x", first.at.hour, first.at.minute), latency_hours=(first.at - at(day, 2, 10)).total_seconds() / 3600, opportunity_index=1)
                if outcome is Outcome.OCCURRED
                else Settlement(claim.ref, outcome, NOW, opportunity_index=1, passes=(OpportunityPass(first.at, True),))
            )
            ledger.write_settlement(settlement)
    readings = read_relations((LATE_TO_BREAKFAST, with_a), ledger=ledger, concepts=CONCEPTS_WITH_A, config=ViewsConfig(min_count=6, min_blocks=3, split_min=3), now=NOW + timedelta(days=365))
    profile = profile_view(readings, top_n=5)
    assert [r.hypothesis_identity for r in profile.stable_relations] == [LATE_TO_BREAKFAST.identity]  # 另一条被「和A一起」调节，不算未被推翻
    slices = entity_slices(readings, CONCEPTS_WITH_A)
    assert [s.concept for s in slices] == ["和A一起"] and slices[0].moderations[0][0] == with_a.identity

    known = {LATE_TO_BREAKFAST.identity: LATE_TO_BREAKFAST, with_a.identity: with_a}
    store = ViewsStore(tmp_path / "scene")
    (tmp_path / "scene" / "views" / "relations").mkdir(parents=True)
    stale = tmp_path / "scene" / "views" / "relations" / "旧的.md"
    stale.write_text("x", encoding="utf-8")
    written = materialize_views(
        store,
        hypotheses=known,
        readings=readings,
        behaviours=behaviour_views(readings, known),
        residue=(),
        profile=profile,
        entities=slices,
        concepts=CONCEPTS_WITH_A,
        now=NOW,
        k=5,
        d=3,
    )
    names = {str(path.relative_to(tmp_path / "scene" / "views")) for path in written}
    assert names == {
        "behaviours/咖啡.md", "behaviours/晚睡.md", "behaviours/早餐.md",
        "entities/和a一起.md", "profile.md",
        f"relations/{LATE_TO_BREAKFAST.identity}.md", f"relations/{with_a.identity}.md", "residue/index.md",
    }
    assert not stale.exists()  # 整个重写：不在集合里的旧文件删掉
    assert store.is_complete() and (tmp_path / "scene" / "views" / ".done.json").exists()
    relation_text = (tmp_path / "scene" / "views" / "relations" / f"{with_a.identity}.md").read_text(encoding="utf-8")
    assert "## 情境" in relation_text and "- 被「和A一起」调节：有 " in relation_text
    assert "实际 0%（0 次到、12 次没来） vs 本来 88%" in (tmp_path / "scene" / "views" / "relations" / f"{LATE_TO_BREAKFAST.identity}.md").read_text(encoding="utf-8")
    # behaviours/：一个行为的两面。「晚睡」是两条假设的前件 → 后果面两行；前因面空（基准没为它写过）。
    late = (tmp_path / "scene" / "views" / "behaviours" / "晚睡.md").read_text(encoding="utf-8")
    assert "## 什么导致它（前因）" in late and "（基准还没有为它写过假设）" in late
    assert "## 它导致了什么（后果）" in late and late.count("- → ") == 2
    # 「咖啡」这一面反过来：它是后件，所以前因面有一行，而那一行下面挂着份额（「和A一起」的额外贡献）。
    coffee = (tmp_path / "scene" / "views" / "behaviours" / "咖啡.md").read_text(encoding="utf-8")
    # 份额：两层区间分开了（+24..+50 vs −50..−24），所以额外贡献这一行不带"看不出"。
    assert "**单个原因**" in coffee and "↳ 「和A一起」在场 " in coffee
    assert "→ 额外 +100pp" in coffee and "看不出" not in coffee
    assert "未被推翻 1 条" in (tmp_path / "scene" / "views" / "profile.md").read_text(encoding="utf-8")
    (tmp_path / "scene" / "views" / "orphan").mkdir()
    materialize_views(store, hypotheses={}, readings=(), behaviours=(), residue=(), profile=profile_view(()), entities=(), concepts=CONCEPTS_WITH_A, now=NOW, k=5, d=3)
    assert not (tmp_path / "scene" / "views" / "orphan").exists() and not (tmp_path / "scene" / "views" / "relations").exists()  # 空目录回收
    assert CONCEPTS is not None
