"""证据包里"已成立的关系"：只摆已成立、不带条件、短跨度、前因此刻在场的那几条（语义树新方案 ``13`` 第 9 步）。

在场与检验同一把尺（裁定 27 第 1 条、E8）：同一条链从前因**做完**起量，按槽算。
现场是一棵真实的行为树：周三 10:20，今天 08:00 写文档、10:00 讨论方案（各做 10 分钟），昨天 15:00 调研，五天前提交推送。
"""

from __future__ import annotations

from datetime import date, timedelta

from habitus.foresight import UnsealedRow, moment_at, present_relations, render_pack
from habitus.scene.concepts import ConceptSet
from habitus.scene.relations import Relation, RelationKey, RelationTest, Segment, Status, Verdict
from habitus.scene.relations.engine import Subset
from habitus.scene.relations.stats import Effect
from tests.unit.foresight.fixtures import SLOT_MINUTES, Ground
from tests.unit.kind_ids import kind_id
from tests.unit.scene.concept_fixtures import base, cid, group
from tests.unit.scene.fixtures import at

TODAY = date(2026, 8, 12)  # 周三
NOW = at(TODAY, 10, 20)
WINDOW_SLOTS = 3
MINIMUM = 5
CONCEPTS = ConceptSet(
    [
        *(base(title) for title in ("讨论方案", "修改代码", "调研", "写文档", "提交推送", "验证测试")),
        group("沟通", "和人对方案", "讨论方案", "调研"),
    ]
)


def ground(tmp_path):  # type: ignore[no-untyped-def]
    site = Ground(tmp_path, now=NOW)
    site.record(TODAY, "写文档", 8, 0)
    site.record(TODAY, "讨论方案", 10, 0)
    site.record(TODAY - timedelta(days=1), "调研", 15, 0)
    site.record(TODAY - timedelta(days=5), "提交推送", 11, 0)
    return site


def reading(count: int = 12, treated: float = 0.5) -> Subset:
    return Subset(
        antecedents=count,
        blocks=6,
        effect=Effect(antecedents=count, treated_rate=treated, control_rate=0.1),
        p_value=0.001,
        interval=(0.2, 0.6),
        relative_interval=None,
    )


def relation(
    antecedent: str,
    consequent: str,
    segment: Segment = Segment.CHAIN,
    *,
    status: Status = Status.ESTABLISHED,
    condition: str = "",
    maintenance: Subset | None = None,
    tonight: RelationTest | None = None,
) -> Relation:
    return Relation(
        key=RelationKey("session", cid(antecedent), cid(consequent), segment, condition),
        status=status,
        upward=True,
        maintenance=maintenance if maintenance is not None else reading(),
        tonight=tonight,
    )


def present(site, relations, *, unsealed=(), now=NOW):  # type: ignore[no-untyped-def]
    moment = moment_at(now, slot_minutes=SLOT_MINUTES)
    return present_relations(
        relations,
        CONCEPTS,
        moment,
        site.cache(),
        unsealed,
        slot_minutes=SLOT_MINUTES,
        window_slots=WINDOW_SLOTS,
        minimum_antecedents=MINIMUM,
    )


def test_each_short_span_is_present_by_its_own_window(tmp_path) -> None:
    site = ground(tmp_path)
    notes = present(
        site,
        [
            relation("讨论方案", "修改代码"),  # 10:10 做完，此刻在 3 槽之内
            relation("写文档", "修改代码", Segment.REST_OF_DAY),  # 08:10 做完，早已过了 3 槽
            relation("调研", "写文档", Segment.NEXT_DAY),  # 昨天
            relation("提交推送", "验证测试", Segment.DAYS_2_7),  # 五天前
        ],
    )
    code = notes[kind_id("修改代码")]
    assert [(note.antecedent, note.segment, note.seen_at) for note in code] == [
        ("写文档", Segment.REST_OF_DAY, at(TODAY, 8, 10)),
        ("讨论方案", Segment.CHAIN, at(TODAY, 10, 10)),
    ]
    assert [note.segment for note in notes[kind_id("写文档")]] == [Segment.NEXT_DAY]
    assert [note.segment for note in notes[kind_id("验证测试")]] == [Segment.DAYS_2_7]
    assert code[1].key == f"session|{kind_id('讨论方案')}|{kind_id('修改代码')}|chain"
    assert code[1].source.startswith("behavior://occurrences/")  # 已封口的前因：那条记录的 URI
    assert (code[1].treated_rate, code[1].control_rate, code[1].interval, code[1].antecedents) == (
        0.5,
        0.1,
        (0.2, 0.6),
        12,
    )


def test_an_antecedent_outside_its_span_is_absent(tmp_path) -> None:
    site = ground(tmp_path)
    notes = present(
        site,
        [
            relation("写文档", "修改代码"),  # 08:10 做完，早出了链的窗口
            relation("讨论方案", "修改代码", Segment.REST_OF_DAY),  # 10:10 做完还在窗口内，算链不算"当天之后"
            relation("讨论方案", "修改代码", Segment.NEXT_DAY),  # 昨天没讨论
            relation("调研", "修改代码", Segment.DAYS_2_7),  # 调研是昨天，不在 2–7 天
        ],
    )
    assert dict(notes) == {}


def test_the_chain_window_counts_from_when_the_antecedent_was_done(tmp_path) -> None:
    """同一条链从做完起量（与检验同一把尺）：10:10 做完在第 40 槽，10:50（第 43 槽）还在 3 槽之内，11:00（第 44 槽）不在。"""

    site = ground(tmp_path)
    chain = [relation("讨论方案", "修改代码")]
    assert kind_id("修改代码") in present(site, chain, now=at(TODAY, 10, 50))
    assert dict(present(site, chain, now=at(TODAY, 11, 0))) == {}


def test_a_note_is_dropped_once_the_consequent_has_already_started(tmp_path) -> None:
    """讨论方案 10:10 做完，修改代码 10:15 已经开始：那段窗口说的事已经发生了，不再摆（不让模型把它算第二遍）。"""

    site = ground(tmp_path)
    site.record(TODAY, "修改代码", 10, 15)
    assert dict(present(site, [relation("讨论方案", "修改代码")])) == {}


def test_a_small_maintenance_window_falls_back_to_the_full_reading(tmp_path) -> None:
    """维持检验这一段只攒了 2 次（读出 100%）：不够前向验证的门槛，改用全部数据的读数，并写出用了几次（E7：样本少不出数）。"""

    site = ground(tmp_path)
    full = RelationTest(
        key=RelationKey("session", cid("讨论方案"), cid("修改代码"), Segment.CHAIN),
        verdict=Verdict.TESTED,
        antecedents=22,
        blocks=15,
        block_days=1.0,
        effect=Effect(antecedents=22, treated_rate=0.68, control_rate=0.25),
        p_value=0.001,
        minimum_p=1e-6,
        upward=True,
        interval=(0.27, 0.60),
    )
    notes = present(site, [relation("讨论方案", "修改代码", maintenance=reading(count=2, treated=1.0), tonight=full)])
    (note,) = notes[kind_id("修改代码")]
    assert (note.treated_rate, note.antecedents, note.interval) == (0.68, 22, (0.27, 0.60))


def test_only_established_unconditioned_short_relations_between_classes_are_shown(tmp_path) -> None:
    site = ground(tmp_path)
    notes = present(
        site,
        [
            relation("讨论方案", "修改代码", status=Status.CANDIDATE),
            relation("讨论方案", "修改代码", status=Status.EXPIRED),
            relation("讨论方案", "修改代码", condition="同一目标"),
            relation("讨论方案", "修改代码", Segment.FIRST_SEEN),
            relation("讨论方案", "沟通"),  # 后果是汇总概念：候选是类，对不上
        ],
    )
    assert dict(notes) == {}


def test_a_group_antecedent_gives_way_to_its_present_member(tmp_path) -> None:
    site = ground(tmp_path)
    notes = present(
        site,
        [
            relation("沟通", "修改代码"),
            relation("讨论方案", "修改代码"),
            relation("沟通", "写文档", Segment.NEXT_DAY),  # 成员（调研）没有同一段的关系：汇总的留着
        ],
    )
    assert [note.antecedent for note in notes[kind_id("修改代码")]] == ["讨论方案"]
    assert [note.antecedent for note in notes[kind_id("写文档")]] == ["沟通"]


def test_an_unsealed_judgement_counts_for_the_chain(tmp_path) -> None:
    site = ground(tmp_path)
    row = UnsealedRow(
        name="跑单测",
        kind_token=kind_id("验证测试"),
        started_at=at(TODAY, 10, 5),
        last_observed_at=at(TODAY, 10, 12),
        summary=None,
    )
    relations = [relation("验证测试", "提交推送")]
    assert dict(present(site, relations)) == {}
    notes = present(site, relations, unsealed=[row])
    (note,) = notes[kind_id("提交推送")]
    assert note.seen_at == at(TODAY, 10, 12)
    # 前因靠现场归类读出来的：出处记成未封口，承诺上照记（E19）
    assert note.source == f"unsealed:{at(TODAY, 10, 5).isoformat()}"


def test_the_pack_renders_the_section_under_its_candidate(tmp_path) -> None:
    site = ground(tmp_path)
    for day in (TODAY - timedelta(days=7), TODAY - timedelta(days=14)):
        site.record(day, "讨论方案", 10, 0)
        site.record(day, "修改代码", 10, 30)
    notes = present(site, [relation("讨论方案", "修改代码")])
    pack = site.pack(NOW, relations=notes)
    text = render_pack(pack)
    assert "### 已成立的关系（前因此刻在场）" in text
    # 只挂在后果那个候选下面：讨论方案自己那一节没有
    discuss = text.split("## 讨论方案")[1]
    assert "已成立的关系" not in discuss
    assert (
        "- 讨论方案 之后（同一条链：做完之后 45 分钟内，与转移边重叠）出现这一类的比例 50%（12 次），"
        "对照（同一天别的事做完之后）10%，差 +20 到 +60 个百分点；此刻：讨论方案 2026-08-12 周三 10:10 做完"
    ) in text
    assert "靠先验" not in text
