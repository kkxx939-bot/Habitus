"""装配层与校验层的契约：形状保证什么、校验守什么、什么刻意不管。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from behavior.fusion import (
    BehaviorFusionConfig,
    JudgementRelation,
    assemble_judgement_batch,
    fusion_json_schema,
    render_context_judgements,
    render_fragments,
    unreadable_ratio,
    validate_judgement_batch,
)
from behavior.fusion.errors import BehaviorFusionError, BehaviorFusionLimitError
from behavior.fusion.prompt import FUSION_SYSTEM_PROMPT as FUSION_SYSTEM_PROMPT_TEXT
from behavior.fusion.schema import JUDGEMENT_FUSION_JSON_SCHEMA
from behavior.observation import BehaviorObservation, BehaviorObservationConfig
from tests.unit.behavior.fusion_wire import SUBJECT, judgement, unreadable, wire

TZ8 = timezone(timedelta(hours=8))
NOW = datetime(2026, 8, 14, 20, 0, tzinfo=TZ8)
OBSERVATION_CONFIG = BehaviorObservationConfig()


def fragment(
    offset: int, semantics: str, *, participants: list[str] | None = None
) -> BehaviorObservation:
    at = NOW + timedelta(seconds=offset)
    return BehaviorObservation.create(
        observer_id="home-a/hall",
        occurred_at=at,
        available_at=at + timedelta(milliseconds=800),
        modality="vision",
        semantics=semantics,
        participants=participants or [SUBJECT],
        knowledge_state="observed",
        confidence=0.9,
        evidence_refs=[f"cam:{offset}"],
        config=OBSERVATION_CONFIG,
    )


FRAGMENTS = [
    fragment(0, "人走到水池边"),
    fragment(4, "手伸向水龙头"),
    fragment(8, "水流出"),
    fragment(33, "人打了个哈欠"),
    fragment(37, "画面模糊，看不清"),
]


def body() -> dict[str, Any]:
    """一件有目标的事（1-3）、一个动作（4）、一段没读懂（5）。"""

    return wire(
        [
            judgement(
                1,
                behavior="洗手",
                goal="清洁双手",
                basis=["走到水池边打开水龙头", "冲洗双手"],
            ),
            judgement(2, behavior="打哈欠"),
            unreadable(3),
        ],
        [[(1, 1)], [(1, 1)], [(1, 2)], [(2, None)], [(3, None)]],
    )


def assemble(raw: dict[str, Any] | None = None, *, fragments: list[BehaviorObservation] | None = None):
    resolved = FRAGMENTS if fragments is None else fragments
    return assemble_judgement_batch(raw or body(), fragment_count=len(resolved))


def checked(raw: dict[str, Any] | None = None, *, fragments: list[BehaviorObservation] | None = None):
    resolved = FRAGMENTS if fragments is None else fragments
    batch = assemble(raw, fragments=resolved)
    validate_judgement_batch(batch, resolved)
    return batch


# --- 三层递减由两个字段决定 -------------------------------------------------------------


def test_the_three_levels_are_two_fields_not_three_structures() -> None:
    batch = checked()
    first, second, third = batch.judgements
    assert (first.claim.is_readable, first.claim.is_goal_directed) == (True, True)
    assert (second.claim.is_readable, second.claim.is_goal_directed) == (True, False)
    assert (third.claim.is_readable, third.claim.is_goal_directed) == (False, False)
    assert unreadable_ratio(batch, len(FRAGMENTS)) == pytest.approx(0.2)


def test_a_goal_directed_claim_must_record_the_facts_that_constitute_it() -> None:
    """归约层要靠 basis 判断行为有没有达成；只给原始帧等于让它重判一遍。"""

    raw = wire([judgement(1, behavior="洗手", goal="清洁双手", basis=[])], [[(1, None)]] * 5)
    with pytest.raises(BehaviorFusionError, match="must record the facts"):
        assemble(raw)


def test_a_claim_without_a_goal_must_not_decompose() -> None:
    """没有目标的判断本身就在操作那一层，里面没有东西可再分解。"""

    raw = wire([judgement(1, behavior="打哈欠", goal=None, basis=["张嘴"])], [[(1, 1)]] * 5)
    with pytest.raises(BehaviorFusionError, match="must not decompose"):
        assemble(raw)


def test_an_unreadable_judgement_must_stay_empty() -> None:
    raw = body()
    raw["judgements"][2]["goal"] = "某个目标"
    with pytest.raises(BehaviorFusionError, match="must not carry goal"):
        assemble(raw)


# --- 穷尽性由形状承担 ------------------------------------------------------------------


def test_a_short_frame_table_fails_with_its_own_length() -> None:
    """反馈是机械可修的（"少了一行"），不需要模型再判断一次该把哪一帧塞到哪。"""

    raw = body()
    raw["frames"] = raw["frames"][:4]
    with pytest.raises(BehaviorFusionError, match="expected 5, got 4"):
        assemble(raw)


def test_frame_rows_must_line_up_with_their_position() -> None:
    raw = body()
    raw["frames"][2]["no"] = 9
    with pytest.raises(BehaviorFusionError, match=r"frames\[2\].no must be 3, got 9"):
        assemble(raw)


def test_a_frame_cannot_be_left_without_an_owner() -> None:
    """读不懂的帧要归给一条 behavior 为空的判断，而不是空着——缺失要记录，不能沉默。"""

    raw = body()
    raw["frames"][4]["assignments"] = []
    with pytest.raises(BehaviorFusionError, match="assignments must not be empty"):
        assemble(raw)


def test_the_schema_pins_the_frame_table_to_the_input_length() -> None:
    for section in ("judgements", "frames"):
        item = JUDGEMENT_FUSION_JSON_SCHEMA["properties"][section]["items"]
        assert item["additionalProperties"] is False
        assert set(item["required"]) == set(item["properties"])
    pinned = fusion_json_schema(7)["properties"]["frames"]
    assert (pinned["minItems"], pinned["maxItems"]) == (7, 7)


# --- 归约产物：模型碰不到也就错不了 ------------------------------------------------------


def test_coverage_and_basis_frames_are_reduced_from_the_frame_table() -> None:
    batch = checked()
    washing = batch.judgements[0]
    assert washing.covers == (1, 2, 3)
    assert [item.fragment_nos for item in washing.claim.basis] == [(1, 2), (3,)]


def test_basis_order_is_derived_from_the_earliest_frame() -> None:
    """声明顺序颠倒也会被归约成正确的时间顺序；顺序不该让模型再写一遍。"""

    raw = wire(
        [judgement(1, behavior="洗手", goal="清洁双手", basis=["冲洗双手", "走到水池边"])],
        [[(1, 2)], [(1, 2)], [(1, 1)], [(1, 1)], [(1, 1)]],
    )
    batch = checked(raw)
    assert [item.semantics for item in batch.judgements[0].claim.basis] == ["走到水池边", "冲洗双手"]


def test_judgements_are_ordered_by_their_earliest_fragment() -> None:
    raw = wire(
        [
            judgement(1, behavior="看电视", goal="观看节目", basis=["坐下看电视"]),
            judgement(2, behavior="洗手", goal="清洁双手", basis=["冲洗双手"]),
        ],
        [[(2, 1)], [(2, 1)], [(2, 1)], [(1, 1)], [(1, 1)]],
    )
    batch = checked(raw)
    assert [item.claim.behavior for item in batch.judgements] == ["洗手", "看电视"]


def test_concurrency_is_closed_symmetrically_by_the_system() -> None:
    """并行只需一边声明。要求两边都写，等于让模型写完后一条再回头改前一条——那是记账。"""

    raw = wire(
        [
            judgement(1, behavior="吃饭", goal="吃完这顿饭", basis=["进食"]),
            judgement(
                2,
                behavior="看手机",
                goal="查看内容",
                basis=["注视屏幕"],
                relations=[("concurrent_with", 1)],
            ),
        ],
        [[(1, 1)], [(1, 1), (2, 1)], [(2, 1)], [(2, 1)], [(1, 1)]],
    )
    batch = checked(raw)
    eating, phone = batch.judgements
    assert [(link.kind, link.target_no) for link in phone.relations] == [
        (JudgementRelation.CONCURRENT_WITH, 1)
    ]
    assert [(link.kind, link.target_no) for link in eating.relations] == [
        (JudgementRelation.CONCURRENT_WITH, 2)
    ]


def test_a_judgement_may_carry_several_relations() -> None:
    """一条判断可以既延续前一段又与另一条并行；单值字段撑不住这种情况。"""

    raw = wire(
        [
            judgement(1, behavior="做饭", goal="准备餐食", basis=["翻炒"], status="ongoing"),
            judgement(2, behavior="接电话", goal="通话", basis=["通话"]),
            judgement(
                3,
                behavior="继续做饭",
                goal="准备餐食",
                basis=["盛盘"],
                relations=[("continues", 1), ("concurrent_with", 2)],
            ),
        ],
        [[(1, 1)], [(1, 1)], [(2, 1)], [(2, 1)], [(3, 1)]],
    )
    batch = checked(raw)
    third = batch.judgements[2]
    assert {link.kind for link in third.relations} == {
        JudgementRelation.CONTINUES,
        JudgementRelation.CONCURRENT_WITH,
    }


# --- 自洽性守卫 -------------------------------------------------------------------------


def test_relations_must_point_at_a_declared_judgement() -> None:
    raw = body()
    raw["judgements"][1]["relations"] = [{"kind": "continues", "target": 99, "context_target": None}]
    with pytest.raises(BehaviorFusionError, match="undeclared judgement: 99"):
        assemble(raw)


def test_a_judgement_cannot_relate_to_itself() -> None:
    raw = body()
    raw["judgements"][1]["relations"] = [{"kind": "continues", "target": 2, "context_target": None}]
    with pytest.raises(BehaviorFusionError, match="cannot relate to itself"):
        assemble(raw)


def test_frames_cannot_reference_an_undeclared_judgement() -> None:
    raw = body()
    raw["frames"][0]["assignments"] = [{"judgement_no": 99, "basis_no": None}]
    with pytest.raises(BehaviorFusionError, match="undeclared judgement: 99"):
        assemble(raw)


def test_frames_cannot_reference_an_undeclared_basis() -> None:
    raw = body()
    raw["frames"][0]["assignments"] = [{"judgement_no": 1, "basis_no": 9}]
    with pytest.raises(BehaviorFusionError, match="undeclared basis 9"):
        assemble(raw)


def test_a_declared_basis_no_fragment_belongs_to_is_rejected() -> None:
    """声明了事实却没有任何帧支撑它，等于凭空多出一段行为。"""

    raw = body()
    raw["judgements"][0]["basis"].append({"basis_no": 3, "semantics": "并不存在的事实"})
    with pytest.raises(BehaviorFusionError, match="no fragment belongs to"):
        assemble(raw)


def test_a_judgement_covering_nothing_is_rejected() -> None:
    raw = body()
    raw["judgements"].append(judgement(4, behavior="幽灵行为", goal="无", basis=["无"]))
    with pytest.raises(BehaviorFusionError, match="cover no fragment"):
        assemble(raw)


def test_the_subject_must_appear_in_the_covered_fragments() -> None:
    raw = body()
    raw["judgements"][0]["subjects"] = ["陌生人"]
    with pytest.raises(BehaviorFusionError, match="names subjects absent"):
        validate_judgement_batch(assemble(raw), FRAGMENTS)


# --- 刻意不管的：现实的形状 --------------------------------------------------------------


def test_one_frame_may_belong_to_two_judgements() -> None:
    """并行交错时边界帧本来就同时属于两边；要求互斥是替现实立法。"""

    raw = wire(
        [
            judgement(1, behavior="吃饭", goal="吃完这顿饭", basis=["进食"]),
            judgement(
                2,
                behavior="看手机",
                goal="查看内容",
                basis=["注视屏幕"],
                relations=[("concurrent_with", 1)],
            ),
        ],
        [[(1, 1)], [(1, 1), (2, 1)], [(2, 1)], [(2, 1)], [(1, 1)]],
    )
    batch = checked(raw)
    assert batch.judgements[0].covers == (1, 2, 5)
    assert batch.judgements[1].covers == (2, 3, 4)


def test_a_basis_fact_may_cover_discontinuous_frames() -> None:
    """帧连不连续是上游数据的事实，不是本层能规定的策略。"""

    raw = wire(
        [
            judgement(1, behavior="做饭", goal="准备餐食", basis=["翻炒"]),
            judgement(2, behavior="接电话", goal="通话", basis=["通话"]),
        ],
        [[(1, 1)], [(2, 1)], [(2, 1)], [(1, 1)], [(1, 1)]],
    )
    batch = checked(raw)
    assert batch.judgements[0].claim.basis[0].fragment_nos == (1, 4, 5)


# --- 形状守卫 ---------------------------------------------------------------------------


def test_unknown_or_missing_fields_are_rejected() -> None:
    raw = body()
    raw["judgements"][0]["extra"] = 1
    with pytest.raises(BehaviorFusionError, match="unsupported keys"):
        assemble(raw)

    raw = body()
    del raw["judgements"][0]["summary"]
    with pytest.raises(BehaviorFusionError, match="missing keys"):
        assemble(raw)


def test_enum_values_report_what_is_allowed() -> None:
    raw = body()
    raw["judgements"][0]["status"] = "done"
    with pytest.raises(BehaviorFusionError, match="must be one of"):
        assemble(raw)


def test_capacity_bounds_raise_inside_the_fusion_error_family() -> None:
    raw = body()
    raw["judgements"][0]["basis"] = [
        {"basis_no": index, "semantics": f"事实{index}"} for index in range(1, 9)
    ]
    with pytest.raises(BehaviorFusionLimitError):
        assemble_judgement_batch(
            raw, fragment_count=len(FRAGMENTS), config=BehaviorFusionConfig(max_basis_facts=4)
        )
    assert issubclass(BehaviorFusionLimitError, BehaviorFusionError)


# --- 渲染 -------------------------------------------------------------------------------


def test_fragments_render_with_temporary_numbers_and_relative_offsets() -> None:
    """只给相对偏移不给绝对时间：偏移足以看出连续性与观测空白，绝对时间会诱使模型输出时间字段。"""

    rendered = render_fragments(FRAGMENTS)
    lines = rendered.splitlines()
    assert lines[0].startswith("#1 (+0s) [vision/observed]")
    assert lines[4].startswith("#5 (+37s)")
    assert "2026" not in rendered


def test_context_judgements_render_with_their_own_reference_numbers() -> None:
    """C1..Cn 就是 ``context_target`` 要填的数字——模型写不对 64 位的内容身份。"""

    assert render_context_judgements((), NOW) == "（无）"
    rendered = render_context_judgements(
        (
            {
                "started_at": (NOW - timedelta(minutes=20)).isoformat(),
                "behavior": "洗手",
                "goal": "清洁双手",
                "status": "completed",
                "status_basis": "observed",
            },
        ),
        NOW,
    )
    assert rendered.startswith("C1  1200秒前开始：洗手，目标：清洁双手，completed/observed")
    assert "2026" not in rendered


# --- 提示词与代码的契约 -----------------------------------------------------------------


def _example_one_fragments() -> list[BehaviorObservation]:
    # 第 7 帧是旁人经过，画面里出现的是另一个人——与提示词示例一一致。
    return [
        fragment(
            index * 4,
            f"示例一片段{index}",
            participants=["家庭成员B"] if index == 7 else None,
        )
        for index in range(1, 14)
    ]


def _example_two_fragments() -> list[BehaviorObservation]:
    return [fragment(index * 5, f"示例二片段{index}") for index in range(1, 10)]


def test_the_prompt_examples_are_accepted_by_the_assembly_layer() -> None:
    """把提示词里两个 worked example 逐字转成线格式，必须能通过装配与校验。

    这条测试存在的唯一理由：这个项目栽过两次"要求只存在于代码里"——一次是 status 的合法取值
    提示词没写，一次是 subject/summary 必填但示例里根本没有这两个字段。示例是模型最会照抄的
    部分，示例本身通不过，模型就没有任何途径写对。
    """

    one = wire(
        [
            judgement(
                1,
                behavior="洗手",
                goal="清洁双手",
                summary="回家后洗了手",
                basis=["走到水池边打开水龙头", "打肥皂搓手", "冲水关龙头甩手"],
            ),
            judgement(
                2,
                behavior="端着水杯走过",
                summary="另一个人端着水杯经过",
                subjects=["家庭成员B"],
            ),
            judgement(3, behavior="打哈欠", summary="打了一个哈欠"),
            unreadable(4),
            judgement(
                5, behavior="开空调", goal="调节室温", summary="打开了空调", basis=["拿起遥控器开机"]
            ),
        ],
        [
            [(1, 1)], [(1, 1)], [(1, 1)],
            [(1, 2)], [(1, 2)], [(1, 2)],
            [(2, None)],
            [(1, 3)], [(1, 3)],
            [(3, None)],
            [(4, None)],
            [(5, 1)], [(5, 1)],
        ],
    )
    fragments = _example_one_fragments()
    batch = assemble_judgement_batch(one, fragment_count=len(fragments))
    validate_judgement_batch(batch, fragments)
    assert [item.claim.behavior for item in batch.judgements] == [
        "洗手",
        "端着水杯走过",
        "打哈欠",
        None,
        "开空调",
    ]
    # 旁人那条只覆盖第 7 帧，且洗手那条不含它——示例本身就示范了"不要吸收旁人"。
    assert batch.judgements[1].covers == (7,)
    assert batch.judgements[0].covers == (1, 2, 3, 4, 5, 6, 8, 9)

    two = wire(
        [
            judgement(
                1,
                behavior="吃饭",
                goal="吃完这顿饭",
                summary="在餐桌前吃饭，中途离开",
                basis=["在餐桌前进食"],
                status="ongoing",
                status_basis="observation_lost",
            ),
            judgement(
                2,
                behavior="看手机",
                goal="查看手机内容",
                summary="边吃饭边看手机",
                basis=["拿起手机注视屏幕"],
                relations=[("concurrent_with", 1)],
            ),
            judgement(3, behavior="离开餐桌", summary="放下筷子起身走开"),
            judgement(
                4,
                behavior="继续吃饭",
                goal="吃完这顿饭",
                summary="回到餐桌继续吃",
                basis=["回到餐桌继续进食"],
                status="ongoing",
                status_basis="observation_lost",
                relations=[("continues", 1)],
            ),
        ],
        [
            [(1, 1)], [(1, 1)],
            [(2, 1)], [(1, 1), (2, 1)], [(2, 1)],
            [(1, 1)],
            [(3, None)],
            [(4, 1)], [(4, 1)],
        ],
    )
    fragments = _example_two_fragments()
    batch = assemble_judgement_batch(two, fragment_count=len(fragments))
    validate_judgement_batch(batch, fragments)
    assert batch.judgements[0].covers == (1, 2, 4, 6)


def test_the_prompt_never_names_a_relation_the_code_does_not_have() -> None:
    """"independent"曾经写在示例的要点里，而代码里根本没有这个取值。"""

    from behavior.fusion.prompt import FUSION_SYSTEM_PROMPT

    assert "independent" not in FUSION_SYSTEM_PROMPT
    for member in JudgementRelation:
        assert member.value in FUSION_SYSTEM_PROMPT


def test_the_prompt_explains_every_hard_required_field_before_the_examples() -> None:
    """字段必须在**说明部分**讲过，不能只在示例里出现过一次。

    只查"整篇提示词里有没有这个词"是没有区分力的——``subject=家庭成员A`` 出现在示例里就
    算命中，而 A1 的本质恰恰是说明部分只字未提、模型只能靠猜。
    """

    guidance, separator, _examples = FUSION_SYSTEM_PROMPT_TEXT.partition("## 示例一")
    assert separator, "提示词的示例分界标题变了，这条测试需要同步更新"
    for field in (
        "subject",
        "summary",
        "behavior",
        "goal",
        "basis",
        "status",
        "status_basis",
        "relations",
    ):
        assert field in guidance, field


def test_an_unreadable_judgement_cannot_be_a_relation_target() -> None:
    """没读懂的那段不是一个行为，谈不上与它并行——而且对称补齐会由系统造出非法状态。"""

    raw = wire(
        [
            judgement(1, behavior="吃饭", goal="吃完这顿饭", basis=["进食"], relations=[("concurrent_with", 2)]),
            unreadable(2),
        ],
        [[(1, 1)], [(1, 1)], [(1, 1)], [(1, 1)], [(2, None)]],
    )
    with pytest.raises(BehaviorFusionError, match="which is unreadable"):
        assemble(raw)


def test_the_unreadable_ratio_is_a_set_union_not_a_sum() -> None:
    """一帧可属多条判断，求和会把同一帧数好几遍，算出大于 1 的"比例"。"""

    # 两条不可读判断覆盖范围重叠但不相同（完全相同会被"不可区分"守卫拦下）。
    raw = wire(
        [unreadable(1), unreadable(2)],
        [[(1, None)], [(1, None)], [(1, None), (2, None)], [(2, None)], [(2, None)]],
    )
    batch = assemble(raw)
    assert sum(len(item.covers) for item in batch.judgements) == 6  # 求和会得到 6/5 = 1.2
    assert unreadable_ratio(batch, 5) == 1.0

    # 一帧同时被读懂和没读懂认领时，它已经被读懂了，不该计入。
    raw = wire(
        [judgement(1, behavior="洗手", goal="清洁双手", basis=["冲洗"]), unreadable(2)],
        [[(1, 1)], [(1, 1)], [(1, 1)], [(1, 1)], [(1, 1), (2, None)]],
    )
    assert unreadable_ratio(assemble(raw), 5) == 0.0


def test_two_indistinguishable_judgements_are_refused() -> None:
    """耐久身份由内容派生，所以这样的两条会算出同一个 judgement_id、落盘时静默合并成一条。

    在装配层拦是因为它是**模型的结构性错误、可以反馈重试**；等到落盘阶段才发现，就变成不可
    重试的失败并阻塞整条串行队列。
    """

    raw = wire(
        [
            judgement(1, behavior="洗手", goal="清洁双手", basis=["冲洗"]),
            judgement(2, behavior="洗手", goal="清洁双手", basis=["冲洗"]),
        ],
        [[(1, 1), (2, 1)]] * 5,
    )
    with pytest.raises(BehaviorFusionError, match="indistinguishable from judgement\\[1\\]"):
        assemble(raw)


def test_a_judgement_may_name_several_subjects() -> None:
    """两个人一起抬桌子是一件事，不是两件，也不是"只记一个人"。

    单值 subject 撑不住：实测模型一半把两人拼成一个字符串（随后被校验拒绝）、一半只留一个人。
    """

    fragments = [
        fragment(index * 4, f"两人抬桌子{index}", participants=[SUBJECT, "家庭成员B"])
        for index in range(5)
    ]
    raw = wire(
        [
            judgement(
                1,
                behavior="搬桌子",
                goal="把桌子移到墙边",
                basis=["两人一起抬起并搬运桌子"],
                subjects=[SUBJECT, "家庭成员B"],
            )
        ],
        [[(1, 1)]] * 5,
    )
    batch = checked(raw, fragments=fragments)
    assert batch.judgements[0].subjects == (SUBJECT, "家庭成员B")


def test_every_named_subject_must_appear_in_the_covered_fragments() -> None:
    raw = wire(
        [judgement(1, behavior="洗手", goal="清洁双手", basis=["冲洗"], subjects=[SUBJECT, "陌生人"])],
        [[(1, 1)]] * 5,
    )
    with pytest.raises(BehaviorFusionError, match=r"names subjects absent from its fragments: \['陌生人'\]"):
        validate_judgement_batch(assemble(raw), FRAGMENTS)


def test_results_from_points_backwards_like_every_other_relation() -> None:
    """结果指回原因，不是原因指向结果——先前的判断已经落盘且不可变，只有后来的能指回去。"""

    raw = wire(
        [
            judgement(1, behavior="开空调", goal="调节室温", basis=["按下遥控器"]),
            judgement(
                2,
                behavior="脱外套",
                summary="室温升高后脱下外套",
                relations=[("results_from", 1)],
            ),
        ],
        [[(1, 1)], [(1, 1)], [(1, 1)], [(2, None)], [(2, None)]],
    )
    batch = checked(raw)
    assert [(link.kind, link.target_no) for link in batch.judgements[1].relations] == [
        (JudgementRelation.RESULTS_FROM, 1)
    ]
    # 对称补齐只针对并行；因果是有向的，不能反过来给「开空调」补一条。
    assert batch.judgements[0].relations == ()


def test_continues_cannot_point_at_a_finished_judgement() -> None:
    """一件已完成的事没有"后半段"——同样的行为再做一次是新的一件事。

    实测触发过：早上洗一次手、四小时后又洗一次，模型三次全部标成 continues，时间跨度没能阻止它。
    先前那条若判错了（写了 completed 其实没完），正确的表达是 supersedes。
    """

    raw = wire(
        [
            judgement(1, behavior="洗手", goal="清洁双手", basis=["冲洗"], status="completed"),
            judgement(
                2,
                behavior="洗手",
                goal="清洁双手",
                basis=["再次冲洗"],
                relations=[("continues", 1)],
            ),
        ],
        [[(1, 1)], [(1, 1)], [(2, 1)], [(2, 1)], [(2, 1)]],
    )
    with pytest.raises(BehaviorFusionError, match="already completed"):
        assemble(raw)


def test_supersedes_may_point_at_a_finished_judgement() -> None:
    """修正先前的判断正是为"以为完成其实没完"准备的，不该被上面那条守卫误伤。"""

    raw = wire(
        [
            judgement(1, behavior="洗手", goal="清洁双手", basis=["冲洗"], status="completed"),
            judgement(
                2,
                behavior="洗手",
                goal="清洁双手",
                basis=["其实还在冲洗"],
                status="ongoing",
                relations=[("supersedes", 1)],
            ),
        ],
        [[(1, 1)], [(1, 1)], [(2, 1)], [(2, 1)], [(2, 1)]],
    )
    assert len(checked(raw).judgements) == 2
