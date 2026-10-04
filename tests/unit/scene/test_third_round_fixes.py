"""第三轮评审（2026-09-29/30）第一批修复的回归：这些都是当时评审用复现脚本跑出来、单测没抓到的。

每条测试名里带 R3 编号，对着 `重构交接_2026-09-27/评审报告/第三轮_2026-09-29/汇总_交叉核对.md` 销账。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from habitus.scene.concepts import ConceptRole, ConceptSet
from habitus.scene.concepts.author import KindBrief, assemble_concepts
from habitus.scene.hypotheses import Antecedent, Direction, HypothesisOrigin, HypothesisSource, TypePrior
from habitus.scene.hypotheses.model import PLACEBO_MARK, Hypothesis
from habitus.scene.hypotheses.placebo import placebo_hypotheses
from habitus.scene.ledger import LedgerStore, Outcome, Settlement, open_claims_for_day
from habitus.scene.ledger.settlement import mapped_until, missing_days
from habitus.scene.occurrences import ConceptHit, ConceptHitStore
from tests.unit.scene.concept_fixtures import concept
from tests.unit.scene.fixtures import CST, DAY1, DAY2, DAY3, at
from tests.unit.scene.ledger_fixtures import (
    BOOKING_TO_BALL,
    CONCEPTS,
    LATE_TO_BREAKFAST,
    MAPPER,
    NOW,
    TableOpportunities,
    hypothesis,
    record,
)


def stores(tmp_path) -> tuple[ConceptHitStore, LedgerStore]:
    return ConceptHitStore(tmp_path / "scene"), LedgerStore(tmp_path / "scene")


# ── R3-17 · 开账去重 ─────────────────────────────────────────────────────


def test_r3_17_open_ended_claims_dedupe_by_trigger_only(tmp_path) -> None:
    """周一约球、周二又约：两条前提，都要开（FIFO 与释放建立在两条都开着之上）。旧写法在树上有峰时把第二条吞掉（评审 A-6）。"""

    hits, ledger = stores(tmp_path)
    hits.write(record(DAY1, "约了球", 10, 0, "约球"))
    hits.write(record(DAY1, "又约了一场", 16, 0, "约球"))
    report = open_claims_for_day(DAY1, hypotheses=(BOOKING_TO_BALL,), concepts=CONCEPTS, hits=hits, ledger=ledger, opportunities=TableOpportunities(), now=NOW)
    assert (report.opened, report.overlapping) == (2, 0)


def test_r3_17_reruns_after_settlement_do_not_reopen(tmp_path) -> None:
    """结算之后重跑同一天，不该多开一条（评审 B-4）。二-6 之后去重只按触发：同一触发一条，重跑幂等。"""

    hits, ledger = stores(tmp_path)
    hits.write(record(DAY1, "就寝", 2, 10, ConceptHit("晚睡", "轻")))
    run = lambda: open_claims_for_day(DAY1, hypotheses=(LATE_TO_BREAKFAST,), concepts=CONCEPTS, hits=hits, ledger=ledger, opportunities=TableOpportunities(), now=NOW)
    assert run().opened == 1
    (claim,) = ledger.claims_for(LATE_TO_BREAKFAST.identity)
    ledger.write_settlement(Settlement(claim.ref, Outcome.OCCURRED, NOW, observed_at=at(DAY1, 7, 30), fulfilling_uri=record(DAY1, "早餐", 7, 30, "早餐").occurrence_uri, latency_hours=5.3, opportunity_index=1))
    again = run()
    assert (again.opened, again.already_open) == (0, 1) and len(ledger.claims_for(LATE_TO_BREAKFAST.identity)) == 1


# ── 裁定四 · 不回填 ─────────────────────────────────────────────────────


def test_ruling_4_closure_and_placebo_hypotheses_never_open_claims_before_they_were_written(tmp_path) -> None:
    """闭环与安慰剂读过账，写入时刻之前的触发不算它的证据；基准凭常识写、可以回看历史。同一夜跑两遍也不多开。"""

    hits, ledger = stores(tmp_path)
    hits.write(record(DAY1, "就寝", 2, 10, ConceptHit("晚睡", "轻")))
    hits.write(record(DAY1, "打球", 19, 0, "打球"))
    hits.write(record(DAY2, "就寝", 1, 0, ConceptHit("晚睡", "轻")))
    hits.write(record(DAY2, "打球", 19, 0, "打球"))
    written_at = at(DAY1, 23, 0)  # 闭环假设是 DAY1 晚上写下的
    moderation = Hypothesis(
        antecedents=(Antecedent("打球"),), consequent="早餐", aspect="probability", direction=Direction.UP, type_prior=TypePrior.PROMOTING,
        note="闭环提的：打完球饿", source=HypothesisSource(HypothesisOrigin.MODERATION), created_at=written_at, windows=LATE_TO_BREAKFAST.windows,
    )
    baseline = hypothesis("晚睡")  # created_at 在 9 月，比数据晚——基准凭常识写，可以回看
    for day in (DAY1, DAY2):
        report = open_claims_for_day(day, hypotheses=(moderation, baseline), concepts=CONCEPTS, hits=hits, ledger=ledger, opportunities=TableOpportunities(), now=NOW)
        assert report.backfill_refused == (1 if day == DAY1 else 0)
    assert [c.anchor for c in ledger.claims_for(moderation.identity)] == [at(DAY2, 19, 0)]  # DAY1 19:00 早于写入时刻，没开
    assert len(ledger.claims_for(baseline.identity)) == 2  # 基准两条都开
    again = open_claims_for_day(DAY1, hypotheses=(moderation,), concepts=CONCEPTS, hits=hits, ledger=ledger, opportunities=TableOpportunities(), now=NOW)
    assert again.opened == 0 and again.backfill_refused == 1


# ── R3-15 / R3-12 · 安慰剂 ───────────────────────────────────────────────


def test_r3_15_placebos_match_every_real_shape_and_pool_per_consequent() -> None:
    """真假设有咖啡第 1–3 个峰各一条 → 安慰剂也各一条（同一把尺子，峰表照抄）；前件按本后件挑，不受这一批里别的后件影响；身份带前缀。"""

    real = [hypothesis("晚睡", consequent="咖啡", direction=Direction.UP, type_prior=TypePrior.PROMOTING, consequent_peak=k) for k in (1, 2, 3)]
    made = placebo_hypotheses(CONCEPTS, real, now=NOW, per_consequent=2)
    assert sorted({item.consequent_peak for item in made}) == [1, 2, 3] and len(made) == 6
    assert all(item.windows["咖啡"] == real[0].windows["咖啡"] for item in made)
    antecedents = {item.antecedents[0].identity for item in made}
    assert len(antecedents) == 2 and "晚睡" not in antecedents and "咖啡" not in antecedents
    assert all(item.leaf.startswith(PLACEBO_MARK) for item in made) and all(item.source.origin.is_placebo for item in made)
    # 同一后件挑中的前件不随批里有没有别的后件而变
    other = hypothesis("晚睡", consequent="起床", aspect="timing", direction=Direction.UP, type_prior=None)
    with_other = placebo_hypotheses(CONCEPTS, [*real, other], now=NOW, per_consequent=2)
    assert {item.antecedents[0].identity for item in with_other if item.consequent == "咖啡"} == antecedents
    # 补差额：已有的不动
    assert placebo_hypotheses(CONCEPTS, [*real, *made], now=NOW, per_consequent=2) == ()
    # 真假设想写同前件的那条时不撞名：身份不同
    twin = Hypothesis(
        antecedents=(Antecedent(sorted(antecedents)[0]),), consequent="咖啡", aspect="probability", direction=Direction.UP,
        type_prior=TypePrior.PROMOTING, note="闭环提的", source=HypothesisSource(HypothesisOrigin.MODERATION), created_at=NOW, windows=real[0].windows,
    )
    assert twin.identity not in {item.identity for item in made}


# ── R3-11 · 可信边界 ─────────────────────────────────────────────────────


def test_r3_11_trust_stops_before_a_missing_day_and_uses_the_subject_timezone(tmp_path) -> None:
    hits, _ledger = stores(tmp_path)
    for day in (DAY1, DAY3):  # DAY2 那一夜没跑
        hits.complete_day(day, records=0, completed_at=at(day, 23, 59), mapper=MAPPER)
    now_utc = datetime(2026, 8, 17, 19, 0, tzinfo=UTC)  # = 本地 08-18 03:00
    assert mapped_until(hits, now_utc, timezone=CST) == datetime(2026, 8, 16, 0, 0, tzinfo=CST)  # 停在漏掉的 DAY2 之前
    assert missing_days(hits, since=DAY1, through=DAY3) == (DAY2,)
    hits.complete_day(DAY2, records=0, completed_at=at(DAY2, 23, 59), mapper=MAPPER)
    assert mapped_until(hits, now_utc, timezone=CST) == datetime(2026, 8, 18, 0, 0, tzinfo=CST)  # 连续到 DAY3 → 次日零点
    # 日界按主体时区：不传 timezone 时退到 now 的（UTC）——那会越过日界 8 小时，所以夜批一律传
    assert mapped_until(hits, now_utc) == datetime(2026, 8, 17, 19, 0, tzinfo=UTC)


# ── R3-24 · 触点① 连带丢弃 ───────────────────────────────────────────────


def test_r3_24_dropping_a_concept_cascades_with_both_reasons_instead_of_rejecting_the_batch() -> None:
    """「收尾验收」凑 8 次被算法丢 → 盯着它的「收尾期」也丢、理由写明；情境概念挂行为上级 → 丢那一个。合格的照常写。"""

    briefs = (KindBrief("修改代码", 30, 20), KindBrief("收口验收", 8, 6))
    answer = {
        "concepts": [
            {"name": "写代码", "definition": "改动代码的行为", "role": "behavior", "parent": None, "claims": ["修改代码"], "context": "occurrence", "baseline_keys": [], "rule": None, "grades": [], "situation": None, "why": None},
            {"name": "收尾验收", "definition": "收口验收", "role": "behavior", "parent": None, "claims": ["收口验收"], "context": "occurrence", "baseline_keys": [], "rule": None, "grades": [], "situation": None, "why": None},
            {"name": "收尾期", "definition": "连着三天在收尾", "role": "derived", "parent": None, "claims": [], "context": "day", "baseline_keys": [], "rule": None, "grades": [],
             "situation": {"basis": "streak", "weekdays": [], "value": None, "concept": "收尾验收", "grade": None, "days": 3}, "why": None},
            {"name": "写代码期", "definition": "连着三天写代码", "role": "derived", "parent": "写代码", "claims": [], "context": "day", "baseline_keys": [], "rule": None, "grades": [],
             "situation": {"basis": "streak", "weekdays": [], "value": None, "concept": "写代码", "grade": None, "days": 3}, "why": None},
        ]
    }
    definitions, claims, dropped = assemble_concepts(answer, briefs, ConceptSet(()), now=datetime(2026, 9, 30, tzinfo=UTC))
    assert [item.name for item in definitions] == ["写代码"] and claims == {"写代码": ("修改代码",)}
    assert any("收尾验收（凑出来只有 8 次" in line for line in dropped)
    assert any("收尾期（它的情境说明盯着「收尾验收」被丢了，连带丢掉）" in line for line in dropped)
    assert any("写代码期（情境概念不能挂在行为概念「写代码」下面" in line for line in dropped)


def test_a_situation_concept_may_not_hang_under_a_behaviour_parent() -> None:
    """概念集层面也拦（R3-24 ④）：情境概念挂行为上级会让那个行为叶子再也不被映射。"""

    from habitus.scene.concepts.situation import SituationBasis, SituationRule

    period = concept("持续写代码期", "连着三天打球", role=ConceptRole.DERIVED, parent="打球", situation=SituationRule(SituationBasis.STREAK, concept="打球", days=3))
    with pytest.raises(Exception, match="both be behaviours or both be situations"):
        ConceptSet((*CONCEPTS.values(), period))


# ── 第三批冒烟（10-01）· 容差不越过邻峰中点 ─────────────────────────────────


def test_tolerance_never_makes_neighbouring_windows_overlap() -> None:
    """咖啡 08:30–09:15 与 09:30–10:15 只隔 15 分钟：各展 45 分钟容差会让两个窗口重叠——一条 09:20 的咖啡归两本账，
    快照也不合法（冒烟第 3 次第二夜就塌在这里）。容差最多展到与邻峰的中点；最后一个峰的"后邻"是次日第一个。"""

    from habitus.scene.concepts.model import widen_windows
    from habitus.scene.hypotheses import PeakWindow
    from habitus.scene.hypotheses.model import peak_index_of, widened_spans

    table = (PeakWindow(1, 510, 555), PeakWindow(2, 570, 615), PeakWindow(3, 1200, 1245))
    spans = widened_spans(table, 45)
    assert spans[0] == (465, 562) and spans[1] == (562, 660)  # 中点 562 处分开，不重叠
    assert spans[2] == (1155, 1290)  # 后邻是次日 08:30（1950），展满 45 分钟
    assert spans[0][0] == 465  # 前邻是前一天 20:00–20:45（−195），展满
    assert peak_index_of(table, 9 * 60 + 20, slack_minutes=45) == 1 and peak_index_of(table, 9 * 60 + 25, slack_minutes=45) == 2
    assert peak_index_of(table, 12 * 60, slack_minutes=45) == 0
    # 只有一个峰：邻居是它自己隔一天，容差展满
    assert widen_windows([(420, 480)], 45) == ((375, 525),)
    # 跨午夜的峰（23:00–01:00）加容差、次日 00:40 的到来归它
    late = (PeakWindow(1, 1380, 1500),)
    assert peak_index_of(late, 40, slack_minutes=30) == 1 and peak_index_of(late, 22 * 60 + 40, slack_minutes=30) == 1
