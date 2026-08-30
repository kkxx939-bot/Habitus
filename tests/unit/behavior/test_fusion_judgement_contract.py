"""装配层与校验层的契约：形状保证什么、校验守什么、什么刻意不管。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from behavior.fusion import (
    BehaviorFusionConfig,
    JudgementRelation,
    JudgementStatus,
    JudgementStatusBasis,
    assemble_judgement_batch,
    fusion_json_schema,
    render_context_judgements,
    render_fragments,
    unreadable_ratio,
    validate_judgement_batch,
)
from behavior.fusion.errors import BehaviorFusionError, BehaviorFusionLimitError
from behavior.fusion.judgement import BehaviorClaim, BehaviorFact, BehaviorJudgement
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


def test_a_goal_without_grounded_facts_keeps_the_goal_and_drops_nothing_else() -> None:
    """goal 与 basis 互不约束：说得出目标却分不出步骤（一次交谈）是现实的形状，不是产物矛盾。"""

    raw = wire([judgement(1, behavior="洗手", goal="清洁双手", basis=[])], [[(1, None)]] * 5)
    batch = checked(raw)
    (only,) = batch.judgements
    assert only.claim.is_readable and only.claim.goal == "清洁双手"
    assert only.claim.basis == () and batch.degradations == ()


def test_declared_but_ungrounded_facts_are_dropped_and_the_goal_survives() -> None:
    """模型写了两条步骤、frames 里一帧都没分给它们——步骤被丢（记账疏漏），目标不受牵连。"""

    raw = wire(
        [judgement(1, behavior="洗手", goal="清洁双手", basis=["打开水龙头", "冲洗双手"])],
        [[(1, None)]] * 5,
    )
    batch = checked(raw)
    (only,) = batch.judgements
    assert only.claim.goal == "清洁双手" and only.claim.basis == ()


def test_a_unit_without_a_goal_may_still_have_steps() -> None:
    """锁门、喝水说不出目标却有步骤：一条 occurrence 是"可提醒或可代劳的单位"，goal 只是可读字段。"""

    raw = wire([judgement(1, behavior="喝水", goal=None, basis=["拿起杯子", "喝水"])], [[(1, 1)], [(1, 1)], [(1, 2)], [(1, 2)], [(1, 2)]])
    batch = checked(raw)
    (only,) = batch.judgements
    assert only.claim.goal is None and [fact.semantics for fact in only.claim.basis] == ["拿起杯子", "喝水"]


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


def test_a_frame_may_be_left_unowned_and_is_reported() -> None:
    """无意识小动作、过渡帧填 []：看到了、看懂了、不构成任何事——树上不写，但批次要报出来。

    读不懂的帧仍归给一条 behavior 为空的判断；两者口径各自干净。
    """

    raw = body()
    raw["frames"][3]["assignments"] = []  # 打哈欠那一帧
    batch = checked(raw)
    assert batch.unowned_fragment_nos == (4,)
    assert [item.claim.behavior for item in batch.judgements] == ["洗手", None]
    # 行不能少：穷尽性靠行数保证，无归属是声明不是省略
    short = body()
    short["frames"] = short["frames"][:-1]
    with pytest.raises(BehaviorFusionError, match="exactly one row"):
        assemble(short)


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


def test_a_frame_pointing_at_an_undeclared_basis_falls_back_to_the_judgement() -> None:
    """``frames`` 标了一个 ``judgements`` 里不存在的 basis 编号——降级，不要整批拒绝。

    这是两张表之间的记账疏漏：判断里只声明了 basis 1、2，frames 里却标成 basis 3。**这一帧
    归给哪条判断是清楚的**，错的只是子分组，所以把它按"归给这条判断、不归任何 basis"处理。
    实测这条在真实模型上是间歇性的（同一输入三次里犯两次），属于组合式记账而不是语义判断，
    靠提示词根治不了；整批拒绝的代价是白烧一次完整调用并吃掉一次重试预算。
    """

    raw = body()
    raw["frames"][0]["assignments"] = [{"judgement_no": 1, "basis_no": 9}]

    batch = assemble(raw)

    first = next(item for item in batch.judgements if item.judgement_no == 1)
    assert 1 in first.covers  # 帧仍然归给这条判断
    assert all(1 not in fact.fragment_nos for fact in first.claim.basis)  # 但不进任何 basis
    validate_judgement_batch(batch, FRAGMENTS)


def test_a_declared_basis_no_fragment_belongs_to_is_dropped() -> None:
    """声明了事实却没有任何帧支撑它——丢掉那一条，不要整批拒绝。

    ``judgements`` 与 ``frames`` 是分开写的两段，多声明一条 basis 却忘了在 frames 里给它任何
    一帧，是模型的记账疏漏而不是语义错误。整批拒绝要白烧一次完整调用并吃掉一次重试预算；
    而丢掉它不损失任何观测——没有任何帧以它为归属。
    """

    raw = body()
    raw["judgements"][0]["basis"].append({"basis_no": 3, "semantics": "并不存在的事实"})

    batch = assemble(raw)

    assert "并不存在的事实" not in [
        fact.semantics for item in batch.judgements for fact in item.claim.basis
    ]
    validate_judgement_batch(batch, FRAGMENTS)


def test_a_judgement_covering_nothing_is_dropped() -> None:
    """一条判断若没有任何帧归属，它就没有任何观测支撑——丢掉，指向它的关系一并剪掉。"""

    raw = body()
    raw["judgements"].append(judgement(4, behavior="幽灵行为", goal="无", basis=["无"]))
    raw["judgements"][0]["relations"] = [
        {"kind": "concurrent_with", "target": 4, "context_target": None}
    ]

    batch = assemble(raw)

    assert 4 not in [item.judgement_no for item in batch.judgements]
    assert "幽灵行为" not in [item.claim.behavior for item in batch.judgements]
    # 关系的目标已经不存在，留着就是一条指向空处的连接。
    assert all(
        link.target_no != 4 for item in batch.judgements for link in item.relations
    )
    validate_judgement_batch(batch, FRAGMENTS)


def participants_of(fragments: list[BehaviorObservation]) -> dict[int, tuple[str, ...]]:
    return {index: item.participants for index, item in enumerate(fragments, start=1)}


def test_the_subject_must_appear_in_the_covered_fragments() -> None:
    """校验层的后置断言仍在：不经装配期降级（不给 participants）时，陌生主体被整批拒。"""

    raw = body()
    raw["judgements"][0]["subjects"] = ["陌生人"]
    with pytest.raises(BehaviorFusionError, match="names subjects absent"):
        validate_judgement_batch(assemble(raw), FRAGMENTS)


def test_a_judgement_whose_subjects_are_all_absent_is_degraded_to_unreadable() -> None:
    """主体没有一个在覆盖片段里出现过：这段观测不能丢，但也不能作为可读判断落盘——整条降为没读懂。

    真实数据一周 371 次（模型从"她 / 大家"推出在场者，上游 participants 里没有），整批拒会让
    同一段反复失败直至封死队列。
    """

    raw = body()
    raw["judgements"][0]["subjects"] = ["陌生人"]
    batch = assemble_judgement_batch(
        raw, fragment_count=len(FRAGMENTS), participants_by_no=participants_of(FRAGMENTS)
    )
    validate_judgement_batch(batch, FRAGMENTS)  # 后置断言通过：产物自洽
    first = batch.judgements[0]
    assert not first.claim.is_readable
    assert first.subjects == () and first.status is None and first.relations == ()
    assert first.covers == (1, 2, 3)  # 覆盖的观测原样保留，归约层会把它物化成 gap
    assert batch.degradations == ("subject_absent judgement=1 dropped=['陌生人'] unreadable",)


def test_relations_pointing_at_a_degraded_judgement_are_pruned() -> None:
    """指向被降为没读懂的判断的关系一并剪掉——它已不是一个行为，谈不上并行或延续。"""

    raw = wire(
        [
            judgement(1, behavior="洗手", goal="清洁双手", basis=["冲洗"], subjects=["陌生人"]),
            judgement(2, behavior="打哈欠", relations=[("concurrent_with", 1)]),
            unreadable(3),
        ],
        [[(1, 1)], [(1, 1)], [(1, 1)], [(2, None)], [(3, None)]],
    )
    batch = assemble_judgement_batch(
        raw, fragment_count=len(FRAGMENTS), participants_by_no=participants_of(FRAGMENTS)
    )
    validate_judgement_batch(batch, FRAGMENTS)
    assert batch.judgements[1].relations == ()
    assert batch.degradations == (
        "subject_absent judgement=1 dropped=['陌生人'] unreadable",
        "relation_dropped judgement=2 target=1 kind=concurrent_with (target degraded to unreadable)",
    )


def duplicated_wash(*, second_relations: list[tuple[str, int]] | None = None) -> dict[str, Any]:
    """两条「洗手」共享最早一帧——同一主体、同一时刻开始的同名行为被声明成两条。"""

    return wire(
        [
            judgement(1, behavior="洗手", goal="清洁双手", basis=["打开水龙头冲手"]),
            judgement(
                2,
                behavior="洗手",
                goal="清洁双手",
                basis=["打肥皂搓手"],
                relations=second_relations,
            ),
            judgement(3, behavior="打哈欠"),
            unreadable(4),
        ],
        [[(1, 1), (2, 1)], [(1, 1)], [(2, 1)], [(3, None)], [(4, None)]],
    )


def test_the_same_behaviour_cannot_start_twice_at_the_same_moment() -> None:
    """判重只在融合层解决（用户裁定）：同一件事被看成两条，反馈重试让模型自己合并。"""

    with pytest.raises(BehaviorFusionError, match="one behaviour seen twice"):
        checked(duplicated_wash())


def test_concurrency_does_not_excuse_a_duplicated_start() -> None:
    """标成并行也不行——同一个人不可能在同一瞬间把同一件事开始两次。"""

    with pytest.raises(BehaviorFusionError, match="one behaviour seen twice"):
        checked(duplicated_wash(second_relations=[("concurrent_with", 1)]))


def test_supersedes_excuses_a_duplicated_start() -> None:
    """修正链上的两条不是重复：supersedes 归约时只留一条，不存在两个同址产物。"""

    checked(duplicated_wash(second_relations=[("supersedes", 1)]))


def test_the_same_behaviour_at_a_later_moment_is_a_new_event() -> None:
    """同名但开始时刻不同——再做一次是新的一件事，照常接受。"""

    raw = wire(
        [
            judgement(1, behavior="洗手", goal="清洁双手", basis=["打开水龙头冲手"]),
            judgement(2, behavior="洗手", goal="清洁双手", basis=["再次冲洗双手"]),
            judgement(3, behavior="打哈欠"),
            unreadable(4),
        ],
        [[(1, 1)], [(1, 1)], [(2, 1)], [(3, None)], [(4, None)]],
    )
    checked(raw)


def test_two_people_may_start_the_same_behaviour_at_the_same_moment() -> None:
    """主体不同就是两件事：旁人同刻做同名的事照常另起一条，不触发判重。"""

    shared = [
        fragment(0, "两人走到水池边", participants=[SUBJECT, "家庭成员B"]),
        fragment(4, "家庭成员A在搓手"),
        fragment(8, "家庭成员B在旁边冲手", participants=["家庭成员B"]),
    ]
    raw = wire(
        [
            judgement(1, behavior="洗手", goal="清洁双手", basis=["走到水池边搓手"]),
            judgement(
                2,
                behavior="洗手",
                goal="清洁双手",
                subjects=["家庭成员B"],
                basis=["走到水池边冲手"],
            ),
        ],
        [[(1, 1), (2, 1)], [(1, 1)], [(2, 1)]],
    )
    checked(raw, fragments=shared)


def test_two_fragments_at_the_same_second_still_count_as_the_same_start() -> None:
    """判重比的是最早观测的时刻，不是"共享同一帧"——两条不同片段同秒到达也算同一时刻。"""

    same_second = [
        fragment(0, "人在水池边搓手"),
        fragment(0, "水声，人在冲洗"),
        fragment(4, "人关水龙头"),
    ]
    raw = wire(
        [
            judgement(1, behavior="洗手", goal="清洁双手", basis=["搓手"]),
            judgement(2, behavior="洗手", goal="清洁双手", basis=["冲洗"]),
        ],
        [[(1, 1)], [(2, 1)], [(1, 1), (2, 1)]],
    )
    with pytest.raises(BehaviorFusionError, match="one behaviour seen twice"):
        checked(raw, fragments=same_second)


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
    head, first = rendered.split("\n", 1)
    # 头一行是贴在 C 行块上的约束（参照不是模板），C 行从第二行起。
    assert head.startswith("（只用来判 continues / supersedes")
    assert first.startswith("C1  1200秒前开始：洗手，目标：清洁双手，completed/observed")
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
    """字段必须在**说明部分**讲过，而且枚举的每个取值都要写出来。

    这条测试守的是本项目栽过两次的那道缝：要求只存在于代码里。一次是 status 的合法取值提示词
    根本没写，一次是 subject/summary 被代码强制必填、示例里却连出现都没出现。

    只查"字段名在不在指引里"是**没有区分力**的：曾经把 3962 字的指引整段换成一行关键词罗列，
    三层递减、折叠判据、status 与 status_basis 的区分、continues 的限制全部消失，这条测试
    照样全绿。所以这里改成两件事一起查——

      1. 每个枚举的**全部取值**都必须出现在指引里，取值直接从代码里的枚举取，
         这样代码新增一个取值而提示词没跟上时，这条测试会红。
      2. 每个枚举字段都必须有自己的**说明小节**，光在别处被提一嘴不算。
    """

    guidance, separator, _examples = FUSION_SYSTEM_PROMPT_TEXT.partition("## 示例一")
    assert separator, "提示词的示例分界标题变了，这条测试需要同步更新"

    for field in ("subject", "summary", "behavior", "goal", "basis", "status", "relations"):
        assert field in guidance, f"指引部分没有讲 {field}"

    for heading, enum in (
        ("## status：", JudgementStatus),
        ("## status_basis：", JudgementStatusBasis),
        ("## relations：", JudgementRelation),
        ("## basis：", None),
    ):
        assert heading in guidance, f"指引部分缺少 {heading} 小节"
        if enum is None:
            continue
        section = guidance.split(heading, 1)[1].split("\n## ", 1)[0]
        missing = [member.value for member in enum if member.value not in section]
        assert not missing, f"{heading} 小节没有写出这些取值：{missing}"


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


def test_an_absent_subject_is_dropped_when_another_is_present() -> None:
    """模型多写了一个观测里没有的人：只剔掉那个名字，判断保留、语义不动。"""

    raw = wire(
        [judgement(1, behavior="洗手", goal="清洁双手", basis=["冲洗"], subjects=[SUBJECT, "陌生人"])],
        [[(1, 1)]] * 5,
    )
    batch = assemble_judgement_batch(
        raw, fragment_count=len(FRAGMENTS), participants_by_no=participants_of(FRAGMENTS)
    )
    validate_judgement_batch(batch, FRAGMENTS)
    (only,) = batch.judgements
    assert only.subjects == (SUBJECT,)
    assert only.claim.goal == "清洁双手"
    assert batch.degradations == ("subject_absent judgement=1 dropped=['陌生人']",)


def test_a_duplicated_assignment_keeps_the_first_and_leaves_a_signal() -> None:
    """一帧在同一条判断里出现两次在现实里没有对应物——取先写的，不整批拒。

    一帧属于两条**不同**判断仍然合法（并行的交界帧，见
    ``test_one_frame_may_belong_to_two_judgements``）。
    """

    raw = wire(
        [judgement(1, behavior="洗手", goal="清洁双手", basis=["打开水龙头", "冲洗双手"])],
        [[(1, 1)], [(1, 1), (1, 2)], [(1, 2)], [(1, 2)], [(1, 2)]],
    )
    batch = checked(raw)
    (only,) = batch.judgements
    assert [fact.fragment_nos for fact in only.claim.basis] == [(1, 2), (3, 4, 5)]
    assert batch.degradations == ("duplicate_assignment frames[1] judgement=1",)


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
    # 同批内保留这条边、留信号：归约按 continues 并链，尾部状态定结局。剪掉反而会让两条同名
    # 同刻的判断互不相认、撞上判重硬拒（真实对照实测）。真正要剪的是指向先前上下文的那种。
    batch = checked(raw)
    assert [len(item.relations) for item in batch.judgements] == [0, 1]
    assert batch.degradations == ("continues_completed judgement=2 target=1 kept",)


def test_a_behaviour_seen_twice_still_passes_when_the_model_links_the_halves() -> None:
    """同主体同刻同名的两条判断本是判重硬拒；模型用 continues 把它们连起来就算同一件事——即使
    被指的那条已写成 completed，这条边也不能被剪掉，否则判重规则失去认亲的依据。"""

    raw = wire(
        [
            judgement(1, behavior="看手机", status="completed"),
            judgement(2, behavior="看手机", relations=[("continues", 1)]),
        ],
        [[(1, None), (2, None)], [(1, None)], [(2, None)], [(2, None)], [(2, None)]],
    )
    batch = checked(raw)  # 同一最早帧 → 同刻开始；有 continues 边 → 判重放行
    assert batch.degradations == ("continues_completed judgement=2 target=1 kept",)


def test_continues_into_a_finished_context_judgement_is_pruned_the_same_way() -> None:
    raw = wire(
        [judgement(1, behavior="洗手", goal="清洁双手", basis=["冲洗"], context_relations=[("continues", 1)])],
        [[(1, 1)]] * 5,
    )
    batch = assemble_judgement_batch(raw, fragment_count=5, context_states=("completed",))
    assert batch.judgements[0].relations == ()
    assert batch.degradations == ("continues_completed judgement=1 target=C1",)
    # 先前那条还没做完时，延续是合法的，不动。
    intact = assemble_judgement_batch(raw, fragment_count=5, context_states=("ongoing",))
    assert len(intact.judgements[0].relations) == 1 and intact.degradations == ()


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


# --- 纵深防御：这些守卫从装配层走不到，但改坏了必须有人知道 ---------------------------------
#
# 它们此前全部没有测试。用变异测试逐条打靶时，把 ``_require_ordering`` / ``_require_known_fragments``
# 整个关掉、把 basis 越界与重复关系的检查删掉、允许同一帧重复归给同一判断——8873 条测试一条不红。
# 纵深防御没人守着，就等于没有。


def _batch(judgements, frames, count):
    return assemble_judgement_batch(wire(judgements, frames), fragment_count=count)


def test_validation_rejects_judgements_out_of_order() -> None:
    """判断必须按最早覆盖片段排列；顺序错乱会让下游按顺序做的每一次归约跟着错。"""

    batch = _batch(
        [judgement(1, behavior="洗手", goal="清洁双手", basis=["冲手"]), judgement(2, behavior="打哈欠")],
        [[(1, 1)], [(2, None)]],
        2,
    )
    fragments = [fragment(0, "人在洗手"), fragment(4, "人打哈欠")]
    validate_judgement_batch(batch, fragments)  # 正序通过

    scrambled = replace(batch, judgements=tuple(reversed(batch.judgements)))
    with pytest.raises(BehaviorFusionError, match="ordered by their earliest fragment"):
        validate_judgement_batch(scrambled, fragments)


def test_validation_rejects_references_to_fragments_outside_the_segment() -> None:
    """引用必须落在本批片段编号内——编造出来的语义会在这里现形。"""

    batch = _batch([judgement(1, behavior="洗手", goal="清洁双手", basis=["冲手"])], [[(1, 1)], [(1, 1)]], 2)
    fragments = [fragment(0, "人在洗手"), fragment(4, "人在冲手")]
    validate_judgement_batch(batch, fragments)

    # 只给一条片段，而判断覆盖了两条：多出来的那条不属于本段。
    with pytest.raises(BehaviorFusionError, match="outside this segment"):
        validate_judgement_batch(batch, fragments[:1])


def test_a_basis_cannot_reference_a_fragment_its_judgement_does_not_cover() -> None:
    """basis 的证据必须在这条判断自己覆盖的帧里——这是我们产物是否自洽，不是在规定现实。

    装配层从 ``frames`` 反推 ``covers``，所以走装配这条路永远违反不了它；这条守卫是纯粹的
    纵深防御，只能直接构造对象来打。它守的是"别的构造路径（归约层、迁移脚本）绕过装配时，
    自相矛盾的判断不能悄悄成立"。
    """

    with pytest.raises(BehaviorFusionError, match="does not cover"):
        BehaviorJudgement(
            judgement_no=1,
            covers=(1,),
            subjects=(SUBJECT,),
            claim=BehaviorClaim(
                behavior="洗手",
                goal="清洁双手",
                summary="洗了手",
                basis=(BehaviorFact(semantics="冲手", fragment_nos=(1, 2)),),
            ),
            status=JudgementStatus.COMPLETED,
            status_basis=JudgementStatusBasis.OBSERVED,
            relations=(),
        )


def test_a_judgement_cannot_declare_the_same_relation_twice() -> None:
    """同一条关系写两遍是我们自己的产物内部重复，不是现实的形状。"""

    with pytest.raises(BehaviorFusionError, match="same relation twice"):
        _batch(
            [
                judgement(1, behavior="吃饭", goal="吃完这顿饭", basis=["进食"]),
                judgement(
                    2,
                    behavior="看手机",
                    goal="查看内容",
                    basis=["看屏幕"],
                    relations=[("concurrent_with", 1), ("concurrent_with", 1)],
                ),
            ],
            [[(1, 1)], [(2, 1)]],
            2,
        )


def test_a_fragment_assigned_to_the_same_judgement_twice_keeps_the_first_assignment() -> None:
    """一帧可以属于多条判断（并行的交界帧），在同一条判断里重复出现则取先写的、留信号。"""

    batch = _batch(
        [judgement(1, behavior="洗手", goal="清洁双手", basis=["冲手", "擦干"])],
        [[(1, 1), (1, 2), (1, 1)]],
        1,
    )
    (only,) = batch.judgements
    assert [(fact.semantics, fact.fragment_nos) for fact in only.claim.basis] == [("冲手", (1,))]
    assert batch.degradations == (
        "duplicate_assignment frames[0] judgement=1",
        "duplicate_assignment frames[0] judgement=1",
    )


def test_a_completed_context_judgement_is_marked_as_not_continuable() -> None:
    """已经做完的先前判断，在 C 行上就标出来"不能被 continues 指"。

    约束必须贴在它作用的那一行上。提示词正文里本来就有一整段讲这条规则，模型仍然 **5/5**
    先写 ``continues`` 指向一条 completed 的判断、被守卫打回，每次多烧两次模型调用（早上洗一次
    手、中午又洗一次，模型把第二次当成第一次的后半段）。把同一句话加长写进示例要点，仍然 5/5
    无改善；改成在 C 行上就地标注之后，这条用例的结构重试归零，而真该延续的正例不受影响。

    这条测试守的是那个标注本身——它一旦被"清理"掉，退化不会有任何别的地方报错。
    """

    def rendered(status: str) -> str:
        return render_context_judgements(
            (
                {
                    "started_at": (NOW - timedelta(minutes=20)).isoformat(),
                    "behavior": "洗手",
                    "goal": "清洁双手",
                    "status": status,
                    "status_basis": "observed",
                },
            ),
            NOW,
        )

    assert "不能被 continues 指" in rendered("completed")
    # 还没做完的那些是合法的延续目标，不能一起标上——那会把真正的跨窗口延续压死。
    for status in ("ongoing", "interrupted", "abandoned"):
        assert "不能被 continues 指" not in rendered(status)


def test_every_hard_failure_the_model_can_trigger_is_stated_where_it_fills_the_field() -> None:
    """代码强制的东西必须讲给模型，而且要讲在**它填这个字段的地方**——schema 的字段描述里。

    这个项目栽过三次同一条缝：``status`` 的合法取值只在代码里；``subject``/``summary`` 被强制
    必填而示例里根本没出现过；``basis`` 写成"只有 goal 非 null 时才填"——那是**必要条件**的说法，
    读起来像"允许不填"，而代码当成充要，等于把话说反了。

    为什么放 schema 而不是加进系统提示词：实测把这几条契约写进提示词正文（净增 65 字），
    ``status-abandoned`` 从 8/8 掉到 5/8（同样本量头对头），而写进 schema 描述**不占提示词空间**，
    又紧挨着模型填值的位置。这是"三种落点效力递增"那条经验的直接应用。
    """

    properties = JUDGEMENT_FUSION_JSON_SCHEMA["properties"]["judgements"]["items"]["properties"]

    basis = properties["basis"]["description"]
    assert "步骤" in basis and "不单独" in basis or "不是独立" in basis, basis
    assert "只有" not in basis, "别写成必要条件的说法，那读起来像'允许不填'"

    # 目标的合法性贴在 ``target`` 字段上，**不能**贴在 relations 数组的描述里：那里是模型决定
    # "要不要发关系"的位置，摆一句限制性从句会把发关系这件事本身一起压住——实测把它写在数组
    # 描述上，`concurrent-cook-and-call` 从 8/8 掉到 7/8（两次全量里更是 3/3 掉到 1/3），
    # 而模型漏标的正是它本来判得出的那条并行。
    target = JUDGEMENT_FUSION_JSON_SCHEMA["properties"]["judgements"]["items"]["properties"][
        "relations"
    ]["items"]["properties"]["target"]["description"]
    assert "另一条" in target and "读得懂" in target, target
    relations = properties["relations"]["description"]
    assert "但" not in relations, "别在'要不要发关系'的位置上摆限制性从句"

    assert "两遍" in properties["subjects"]["description"]

    # interrupted 与 abandoned 的区别在"有没有别的事打断"，不在"做没做完"。这两档的措辞必须与
    # 提示词一致；schema 留着旧的模糊说法，等于在模型填值的位置给一个更弱的定义。
    status = properties["status"]["description"]
    assert "被别的事打断" in status and "没人打断" in status, status


def test_a_behaviour_name_unusable_as_an_address_is_rejected_with_feedback() -> None:
    """behavior 名将来就是树地址：坏名字在这里打回让模型换说法，而不是封口后卡死归约。

    空白类脏名（首尾空格、内嵌换行）由装配层的文本归一顺手清洗掉，不必烧重试；这里拒的是
    清洗救不回来的：路径不安全字符、保留后缀、超出地址字节预算（含消歧后缀预留）。
    """

    for bad in ("a/b", "洗手.md", "洗" * 60):
        raw = body()
        raw["judgements"][0]["behavior"] = bad
        with pytest.raises(BehaviorFusionError, match="tree address"):
            assemble(raw)

    cleansed = body()
    cleansed["judgements"][0]["behavior"] = " 洗\n手"
    assert assemble(cleansed).judgements[0].claim.behavior == "洗 手"
