"""关联的输入约束、渲染、装配降级与服务边界。

装配的判据只有一条：**产出里的每个引用都能在输入里找回去，而且指的是更早的事**。找不回去、
或者指向比这次还晚的，那一项降级，不整批拒；只有一条草稿都没剩下才整批不可用。
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta

import pytest

from habitus.model_client import ChatClient, ModelStructuredOutputError, ModelTransportError, StructuredChatClient
from habitus.scene.association import (
    ASSOCIATION_JSON_SCHEMA,
    ASSOCIATION_SYSTEM_PROMPT,
    ASSOCIATION_VERSION,
    AssociationAssembly,
    AssociationAssemblyError,
    AssociationConfig,
    AssociationDraft,
    AssociationLimitError,
    Associator,
    DayFacts,
    LLMAssociator,
    assemble_association,
    association_json_schema,
    build_request,
    render_causes,
    render_facts,
    render_pending,
    render_situations,
)
from habitus.scene.association.assembly import MAX_PREMISES_PER_OCCURRENCE
from habitus.scene.association.model import CauseRow, SituationRow
from tests.unit.scene.association_payloads import (
    DAY,
    ScriptedProvider,
    association_input,
    entry,
    instant,
    model_config,
    occurrence,
    output,
    recording_client,
    two_target_input,
)


def _replace(row, **changes):  # type: ignore[no-untyped-def]
    return type(row)(**{**vars(row), **changes})


# ── 输入的自洽 ───────────────────────────────────────────────────────────────


def test_a_target_must_be_an_occurrence_of_this_candidate() -> None:
    with pytest.raises(ValueError, match="occurrence of this candidate"):
        association_input(targets=(2,))


def test_a_target_must_exist_in_the_day() -> None:
    with pytest.raises(ValueError, match="one of the day's occurrences"):
        association_input(targets=(9,))


def test_the_day_stream_must_be_ordered_by_instant() -> None:
    rows = (occurrence(1, 19, "打球", kind="打球"), occurrence(2, 8, "吃早饭"))
    with pytest.raises(ValueError, match="ordered by started_at"):
        association_input(targets=(1,), occurrences=rows)


def test_rows_must_be_numbered_from_one_in_order() -> None:
    rows = (occurrence(1, 8, "吃早饭"), occurrence(3, 19, "打球", kind="打球"))
    with pytest.raises(ValueError, match="numbered 1..N"):
        association_input(targets=(3,), occurrences=rows)


def test_a_row_number_that_is_not_a_plain_integer_is_refused() -> None:
    """口径要与下游取值一致：``int(1.5)`` 混过去的行，渲染出的编号是 #4.5，谁也引用不到。"""

    rows = (occurrence(1, 8, "吃早饭"), _replace(occurrence(2, 19, "打球", kind="打球"), no=2.0))
    with pytest.raises(ValueError, match="must be a positive integer"):
        association_input(targets=(2,), occurrences=rows)


def test_a_day_with_a_time_is_rejected() -> None:
    with pytest.raises(TypeError, match="without a time"):
        association_input(day=datetime(2026, 9, 11, 8, 0))


@pytest.mark.parametrize("facts", [DayFacts(weekday=1, month=9), DayFacts(weekday=4, month=8)])
def test_day_facts_must_describe_the_same_day(facts: DayFacts) -> None:
    """上游一个 off-by-one 会让模型写下"周五晚上"，而那是一句通顺的话，装配层看不出来。"""

    with pytest.raises(ValueError, match="same day"):
        association_input(facts=facts)


def test_a_naive_instant_is_refused() -> None:
    naive = (
        occurrence(1, 8, "吃早饭"),
        _replace(occurrence(2, 19, "打球", kind="打球"), started_at=datetime(2026, 9, 11, 19)),
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        association_input(targets=(2,), occurrences=naive)


def test_a_naive_observation_gap_is_refused() -> None:
    facts = DayFacts(weekday=4, month=9, observed_gaps=((datetime(2026, 9, 11, 13), instant(14)),))
    with pytest.raises(ValueError, match="timezone-aware"):
        association_input(facts=facts)


def test_an_empty_kind_token_is_refused() -> None:
    with pytest.raises(ValueError, match="kind_token"):
        association_input(kind_token="")


def test_a_day_with_no_occurrences_is_refused() -> None:
    with pytest.raises(ValueError, match="the day's occurrences"):
        association_input(occurrences=())


def test_a_call_without_targets_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one target"):
        association_input(targets=())


def test_repeated_targets_are_refused() -> None:
    with pytest.raises(ValueError, match="must not repeat"):
        two_target_input(targets=(2, 2))


def test_causes_are_numbered_after_the_day_stream_in_one_space() -> None:
    """一套编号：当天三条占 1..3，前因候选从 4 起——模型只需记住"引用你看到的那个 #n"。"""

    payload = association_input()
    assert sorted(payload.citable) == [1, 2, 3, 4]
    assert payload.cause_no(4) == 1
    assert payload.cause_no(3) is None
    assert payload.cause_no(5) is None


def test_every_citable_number_has_an_instant_and_a_row_accessor() -> None:
    payload = association_input()
    assert set(payload.instants) == payload.citable
    assert payload.instants[4] == payload.causes[0].started_at
    assert payload.row(3).name == "打球"
    with pytest.raises(KeyError):
        payload.row(4)


# ── 渲染 ─────────────────────────────────────────────────────────────────────


def test_the_target_is_marked_inside_the_day_stream() -> None:
    content = build_request(association_input()).messages[1].content or ""
    assert "#3 19:00  打球  目标=活动一下  打球的一句话 ←这次" in content
    assert "#1 08:00  吃早饭  吃早饭的一句话\n" in content


def test_deterministic_facts_are_rendered_not_asked_for() -> None:
    rendered = render_facts(association_input())
    assert "2026-09-11 周五（工作日）" in rendered
    assert "13:00–14:00" in rendered


def test_the_month_is_not_restated_next_to_the_iso_date() -> None:
    assert "9 月" not in render_facts(association_input())


def test_prediction_numbers_do_not_reach_the_model() -> None:
    """罕见度与转移计数是预测树的统计。摆进提示词会让模型拿计数倒推语义前因，统计被洗成语义
    再喂回预测层——语义层的维度必须是语义的表现。"""

    rendered = render_causes(association_input())
    assert rendered == "  #4  2026-09-06 18:00  和朋友通电话  约这周打一次球"
    assert not hasattr(association_input().causes[0], "rarity")


def test_situations_carry_a_prefix_and_a_bounded_date_list() -> None:
    """同屏还有 #n 与 Pn；而一种跑了半年的情境会把真正要比对的那句话淹没。"""

    many = SituationRow(
        no=1, text="周五下班后自己去", days=tuple(date(2026, 6, 1) + timedelta(days=7 * n) for n in range(9))
    )
    rendered = render_situations(association_input(situations=(many,)))
    assert rendered.startswith("  S1  周五下班后自己去（共 9 次，最近 ")
    assert rendered.count("2026-") == 3


def test_a_short_situation_lists_its_days_without_a_count() -> None:
    rendered = render_situations(association_input())
    assert rendered == "  S1  周五下班后自己去（2026-09-04）"


def test_absent_sections_say_so_rather_than_vanish() -> None:
    """空着的小节要留一句"（无）"：整段消失会让模型以为这次没给它这份材料。"""

    payload = association_input(situations=(), causes=(), pending=())
    assert render_causes(payload) == "（无）"
    assert render_pending(payload) == "（无）"
    assert "还没有" in render_situations(payload)


def test_the_origin_section_appears_only_when_there_is_one() -> None:
    assert "这个行为是怎么开始的" not in (build_request(association_input()).messages[1].content or "")
    with_origin = build_request(association_input(origin="第一次是同事带着去的")).messages[1].content or ""
    assert "这个行为是怎么开始的\n第一次是同事带着去的" in with_origin


def test_long_summaries_are_trimmed_in_the_prompt_only() -> None:
    rows = (occurrence(1, 8, "吃早饭"), _replace(occurrence(2, 19, "打球", kind="打球"), summary="长" * 400))
    payload = association_input(targets=(2,), occurrences=rows)
    content = build_request(payload).messages[1].content or ""
    assert "长" * 160 + "…" in content
    assert payload.occurrences[1].summary == "长" * 400


def test_the_worked_example_uses_the_numbering_the_renderer_produces() -> None:
    """示例是模型最爱照抄的东西。它教的编号方案必须就是渲染器产出的那一套。"""

    rendered = build_request(two_target_input()).messages[1].content or ""
    assert "  #5  2026-09-06 18:00" in rendered  # 当天 4 行，前因第一条就是 #5
    assert "    ## 当天的流\n      #1 08:00" in ASSOCIATION_SYSTEM_PROMPT
    assert "      #5  2026-09-06 18:00" in ASSOCIATION_SYSTEM_PROMPT
    assert "←这次" in ASSOCIATION_SYSTEM_PROMPT
    assert "      S1  周五下班后自己去（2026-09-04）" in ASSOCIATION_SYSTEM_PROMPT


# ── schema ───────────────────────────────────────────────────────────────────


def test_the_entry_count_is_pinned_to_the_target_count() -> None:
    entries = association_json_schema(3)["properties"]["entries"]
    assert (entries["minItems"], entries["maxItems"]) == (3, 3)
    assert "maxItems" not in ASSOCIATION_JSON_SCHEMA["properties"]["entries"]


def test_every_object_in_the_schema_is_closed() -> None:
    def closed(node: object) -> bool:
        if isinstance(node, list):
            return all(closed(item) for item in node)
        if not isinstance(node, dict):
            return True
        if node.get("type") == "object" and node.get("additionalProperties") is not False:
            return False
        return all(closed(value) for value in node.values())

    assert closed(ASSOCIATION_JSON_SCHEMA)
    assert not closed({"a": [{"type": "object", "properties": {}}]})


@pytest.mark.parametrize("count", [0, -1, True])
def test_a_non_positive_target_count_is_refused(count: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        association_json_schema(count)  # type: ignore[arg-type]


# ── 装配：穷尽性 ─────────────────────────────────────────────────────────────


def test_a_target_nobody_spoke_about_makes_the_call_unusable() -> None:
    with pytest.raises(AssociationAssemblyError, match="#2"):
        assemble_association(output(entry(occurrence_no=4)), two_target_input())


def test_an_entry_about_an_untargeted_occurrence_is_dropped() -> None:
    result = assemble_association(output(entry(), entry(occurrence_no=1)), association_input())
    assert [draft.occurrence_no for draft in result.drafts] == [3]
    assert any("not one of this call's targets" in note for note in result.signals)


def test_a_repeated_target_keeps_the_first_entry() -> None:
    later = entry(context="后来的这一句")
    result = assemble_association(output(entry(), later), association_input())
    assert result.drafts[0].context == "周五晚上，下午刚和朋友通过电话说好一起打"
    assert any("more than once" in note for note in result.signals)


def test_drafts_follow_the_target_order_not_the_output_order() -> None:
    payload = two_target_input(targets=(4, 2))
    result = assemble_association(output(entry(occurrence_no=2), entry(occurrence_no=4)), payload)
    assert [draft.occurrence_no for draft in result.drafts] == [4, 2]


@pytest.mark.parametrize("parsed", [[], {"entries": {}}, "entries"])
def test_a_shapeless_output_is_refused(parsed: object) -> None:
    with pytest.raises(AssociationAssemblyError):
        assemble_association(parsed, association_input())


def test_a_malformed_entry_is_dropped_before_anything_else() -> None:
    with pytest.raises(AssociationAssemblyError, match="#3"):
        assemble_association({"entries": ["not an object"]}, association_input())


@pytest.mark.parametrize("no", [True, "3", None, 3.5])
def test_an_occurrence_no_that_is_not_an_index_is_reported_as_such(no: object) -> None:
    """bool 是 int 的子类。这三处护栏是"降级不变成整批失败"的唯一屏障。"""

    payload = two_target_input()
    result = assemble_association(
        output(entry(occurrence_no=2), entry(occurrence_no=no), entry(occurrence_no=4)), payload
    )
    assert [draft.occurrence_no for draft in result.drafts] == [2, 4]
    assert any("is not an integer" in note for note in result.signals)


# ── 装配：逐项降级 ───────────────────────────────────────────────────────────


def test_a_citation_that_is_not_in_the_input_is_dropped() -> None:
    result = assemble_association(output(entry(cites=[2, 9])), association_input())
    assert result.drafts[0].cites == (2,)
    assert any("not available here" in note for note in result.signals)


def test_an_entry_citing_nothing_real_is_dropped_and_named() -> None:
    """依据是硬要求：一句指不回任何记录的话无从核对，留着就是留一句编出来的。"""

    payload = two_target_input()
    result = assemble_association(output(entry(occurrence_no=4), entry(occurrence_no=2, cites=[77])), payload)
    assert [draft.occurrence_no for draft in result.drafts] == [4]
    assert result.unanswered == (2,)
    assert any("cites nothing that exists" in note for note in result.signals)


def test_every_target_losing_its_entry_makes_the_call_unusable() -> None:
    with pytest.raises(AssociationAssemblyError, match="without a usable entry"):
        assemble_association(output(entry(cites=[77])), association_input())


def test_a_cause_outside_the_cited_rows_is_dropped() -> None:
    result = assemble_association(output(entry(cites=[2], causes=[2, 4])), association_input())
    assert result.drafts[0].causes == (2,)
    assert any("not an earlier cited row" in note for note in result.signals)


def test_a_cause_later_than_this_occurrence_is_dropped() -> None:
    """10 点那次打球的前因不可能是当晚 18 点的通话。编号在不在输入里，不等于它更早。"""

    payload = two_target_input()
    body = output(entry(occurrence_no=2, cites=[1, 2, 3, 4], causes=[3, 4]), entry(occurrence_no=4, cites=[3]))
    result = assemble_association(body, payload)
    assert result.drafts[0].causes == ()
    assert any("#2 causes names 3" in note for note in result.signals)


def test_an_occurrence_cannot_be_its_own_cause() -> None:
    result = assemble_association(output(entry(cites=[3], causes=[3])), association_input())
    assert result.drafts[0].causes == ()
    assert any("#3 causes names 3" in note for note in result.signals)


def test_an_earlier_day_row_is_a_legitimate_cause() -> None:
    result = assemble_association(output(entry(cites=[2, 4], causes=[4])), association_input())
    assert result.drafts[0].causes == (4,)


def test_citations_are_deduplicated_and_the_repeat_is_reported() -> None:
    result = assemble_association(output(entry(cites=[4, 2, 2])), association_input())
    assert result.drafts[0].cites == (2, 4)
    assert any("repeats #2" in note for note in result.signals)


def test_an_unknown_situation_degrades_to_the_new_one() -> None:
    result = assemble_association(output(entry(situation_no=7)), association_input())
    assert (result.drafts[0].situation_no, result.drafts[0].new_situation) == (None, "和朋友约好之后一起去打")
    assert any("unknown situation 7" in note for note in result.signals)


def test_naming_both_an_existing_and_a_new_situation_keeps_the_existing_one() -> None:
    result = assemble_association(output(entry(situation_no=1)), association_input())
    assert (result.drafts[0].situation_no, result.drafts[0].new_situation) == (1, None)
    assert any("kept 1" in note for note in result.signals)


def test_naming_an_existing_situation_alone_is_the_ordinary_path() -> None:
    result = assemble_association(output(entry(situation_no=1, new_situation=None)), association_input())
    assert (result.drafts[0].situation_no, result.drafts[0].new_situation) == (1, None)
    assert result.signals == ()


def test_a_new_situation_that_restates_an_existing_one_is_merged_into_it() -> None:
    """规范身份相同就是同一个地址上的同一件事——这是产物自洽，不是"像不像"。"""

    result = assemble_association(output(entry(new_situation="周五下班后自己去 ")), association_input())
    assert (result.drafts[0].situation_no, result.drafts[0].new_situation) == (1, None)
    assert any("restated existing situation 1" in note for note in result.signals)


def test_belonging_to_no_situation_is_a_degradation_not_a_drop() -> None:
    """少说一点也是产出：不归情境的一次仍然留下它的上下文。"""

    result = assemble_association(output(entry(situation_no=None, new_situation="   ")), association_input())
    assert (result.drafts[0].situation_no, result.drafts[0].new_situation) == (None, None)
    assert result.drafts[0].context
    assert any("belongs to no situation" in note for note in result.signals)


def test_a_boolean_situation_no_degrades_instead_of_raising() -> None:
    result = assemble_association(output(entry(situation_no=True)), association_input())
    assert result.drafts[0].situation_no is None
    assert any("unknown situation True" in note for note in result.signals)


def test_consuming_a_premise_that_does_not_exist_is_dropped() -> None:
    result = assemble_association(output(entry(consumed=[1, 4])), association_input())
    assert result.drafts[0].consumed == (1,)
    assert any("not an unspent premise" in note for note in result.signals)


def test_a_premise_can_only_be_consumed_once_in_a_batch() -> None:
    """一条"和朋友约好打一次球"不能被同一天两次打球各兑现一遍。"""

    payload = two_target_input()
    body = output(entry(occurrence_no=2, cites=[1], consumed=[1]), entry(occurrence_no=4, cites=[3], consumed=[1]))
    result = assemble_association(body, payload)
    assert [draft.consumed for draft in result.drafts] == [(1,), ()]
    assert any("#4 consumed names 1" in note for note in result.signals)


def test_a_left_premise_waiting_for_nothing_is_dropped() -> None:
    left = [{"text": "买回了明天的食材", "consumed_by": "做饭"}, {"text": "洗完了碗", "consumed_by": "  "}]
    result = assemble_association(output(entry(left=left)), association_input())
    assert result.drafts[0].left == (("买回了明天的食材", "做饭"),)
    assert any("waits for nothing" in note for note in result.signals)


def test_an_unseen_kind_may_still_be_what_a_premise_waits_for() -> None:
    """不拿词表核对 ``consumed_by``：一个还没在树上出现过的种类是完全可能的。"""

    left = [{"text": "挂了下周的号", "consumed_by": "从没见过的行为"}]
    result = assemble_association(output(entry(left=left)), association_input())
    assert result.drafts[0].left == (("挂了下周的号", "从没见过的行为"),)


def test_a_consumed_by_too_long_to_store_is_dropped() -> None:
    """这约束的是我们自己存不存得下、渲不渲染得出，不是现实该长什么样。"""

    left = [{"text": "留下了什么", "consumed_by": "长" * 100}]
    result = assemble_association(output(entry(left=left)), association_input())
    assert result.drafts[0].left == ()
    assert any("too long to store" in note for note in result.signals)


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("cites", "2,4", "cites is not an array"),
        ("left", "一条", "left is not an array"),
    ],
)
def test_a_field_that_is_not_an_array_says_so(field: str, value: object, fragment: str) -> None:
    payload = two_target_input()
    body = output(entry(occurrence_no=2, cites=[1]), entry(occurrence_no=4, **{field: value}))
    result = assemble_association(body, payload)
    assert any(fragment in note for note in result.signals)


def test_a_malformed_left_item_is_dropped() -> None:
    result = assemble_association(output(entry(left=["不是对象"])), association_input())
    assert result.drafts[0].left == ()
    assert any("malformed pending effect" in note for note in result.signals)


def test_a_repeated_left_premise_is_dropped() -> None:
    item = {"text": "买回了明天的食材", "consumed_by": "做饭"}
    result = assemble_association(output(entry(left=[item, dict(item)])), association_input())
    assert result.drafts[0].left == (("买回了明天的食材", "做饭"),)
    assert any("repeats pending effect" in note for note in result.signals)


def test_integral_floats_are_taken_as_the_index_they_obviously_mean() -> None:
    """JSON 修复路径够得着 ``2.0``。模型指对了两条真实的行，不该因为写法整批报废。"""

    result = assemble_association(output(entry(cites=[2.0, 4.0], causes=[4.0], consumed=[1.0])), association_input())
    assert (result.drafts[0].cites, result.drafts[0].causes, result.drafts[0].consumed) == ((2, 4), (4,), (1,))
    assert sum("index_coerced" in note for note in result.signals) == 4


def test_a_boolean_inside_a_reference_array_degrades_instead_of_raising() -> None:
    result = assemble_association(output(entry(cites=[True, 2])), association_input())
    assert result.drafts[0].cites == (2,)
    assert any("names True" in note for note in result.signals)


def test_a_null_reference_array_is_treated_as_empty() -> None:
    result = assemble_association(output(entry(causes=None, consumed=None, left=None)), association_input())
    assert (result.drafts[0].causes, result.drafts[0].consumed, result.drafts[0].left) == ((), (), ())


def test_invisible_characters_are_cleaned_rather_than_failing_the_call() -> None:
    result = assemble_association(output(entry(context="周五​晚上  刚打完电话")), association_input())
    assert result.drafts[0].context == "周五 晚上 刚打完电话"


def test_a_non_text_field_is_treated_as_empty_rather_than_raising() -> None:
    result = assemble_association(output(entry(left=[{"text": 5, "consumed_by": "做饭"}])), association_input())
    assert result.drafts[0].left == ()
    assert any("says nothing" in note for note in result.signals)


def test_an_empty_context_drops_that_entry() -> None:
    payload = two_target_input()
    result = assemble_association(output(entry(occurrence_no=4), entry(occurrence_no=2, context="​")), payload)
    assert [draft.occurrence_no for draft in result.drafts] == [4]
    assert result.unanswered == (2,)
    assert any("no usable context" in note for note in result.signals)


# ── 草稿与装配结果自己的自洽 ─────────────────────────────────────────────────


def test_a_draft_cannot_cite_nothing() -> None:
    with pytest.raises(ValueError, match="cite at least one"):
        AssociationDraft(occurrence_no=1, context="一句话", cites=())


def test_a_draft_cannot_claim_a_cause_it_did_not_cite() -> None:
    with pytest.raises(ValueError, match="subset of its cites"):
        AssociationDraft(occurrence_no=1, context="一句话", cites=(2,), causes=(3,))


def test_a_draft_cannot_name_two_situations_at_once() -> None:
    with pytest.raises(ValueError, match="existing situation and a new one"):
        AssociationDraft(occurrence_no=1, context="一句话", cites=(2,), situation_no=1, new_situation="新的")


def test_a_draft_context_must_be_a_single_line() -> None:
    with pytest.raises(ValueError, match="single non-empty line"):
        AssociationDraft(occurrence_no=1, context="上\n下", cites=(2,))


@pytest.mark.parametrize("cites", [(2, 1), (1, 1)])
def test_draft_references_must_be_strictly_ascending(cites: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="strictly ascending"):
        AssociationDraft(occurrence_no=1, context="一句话", cites=cites)


def test_an_assembly_cannot_answer_the_same_occurrence_twice() -> None:
    draft = AssociationDraft(occurrence_no=1, context="一句话", cites=(1,))
    with pytest.raises(ValueError, match="more than once"):
        AssociationAssembly(drafts=(draft, draft))


def test_an_assembly_cannot_both_answer_and_skip_an_occurrence() -> None:
    draft = AssociationDraft(occurrence_no=1, context="一句话", cites=(1,))
    with pytest.raises(ValueError, match="both answer and skip"):
        AssociationAssembly(drafts=(draft,), unanswered=(1,))


# ── 服务 ─────────────────────────────────────────────────────────────────────


def _associator(bodies: list[dict[str, object]], **config: object):  # type: ignore[no-untyped-def]
    client, provider = recording_client(bodies)
    return LLMAssociator(client, config=AssociationConfig(**config)), client, provider  # type: ignore[arg-type]


def test_the_protocol_is_satisfied_by_the_production_implementation() -> None:
    associator, _client, _provider = _associator([output()])
    checked: Associator = associator
    assert checked.version == ASSOCIATION_VERSION


def test_the_version_pins_both_the_prompt_and_the_schema() -> None:
    associator, _client, _provider = _associator([output()])
    assert associator.version == ASSOCIATION_VERSION
    assert ASSOCIATION_VERSION.startswith("scene_association_prompt_v1+schema")


def test_the_service_returns_the_assembled_drafts_and_pins_the_entry_count() -> None:
    associator, client, provider = _associator([output()])

    result = asyncio.run(associator.associate(association_input()))

    assert [draft.occurrence_no for draft in result.drafts] == [3]
    assert provider.calls == 1
    assert client.schemas[0]["properties"]["entries"]["maxItems"] == 1
    assert client.names == ["scene_association"]
    assert "#3 19:00  打球" in provider.prompts[0]


def test_the_entry_count_follows_the_batch_size() -> None:
    body = output(entry(occurrence_no=2, cites=[1]), entry(occurrence_no=4, cites=[3]))
    associator, client, _provider = _associator([body])

    asyncio.run(associator.associate(two_target_input()))

    assert client.schemas[0]["properties"]["entries"]["minItems"] == 2


def test_the_client_must_be_a_structured_chat_client() -> None:
    with pytest.raises(TypeError, match="StructuredChatClient"):
        LLMAssociator(object())  # type: ignore[arg-type]


def test_the_payload_must_be_an_association_input() -> None:
    associator, _client, _provider = _associator([output()])
    with pytest.raises(TypeError, match="AssociationInput"):
        asyncio.run(associator.associate(object()))  # type: ignore[arg-type]


def test_an_unusable_answer_becomes_a_structured_output_error_after_correction() -> None:
    """生产路径上装配是结构层的 validator：整批不可用触发纠正重试，对外是结构化输出错误。"""

    associator, _client, provider = _associator([output(entry(cites=[77]))])
    with pytest.raises(ModelStructuredOutputError):
        asyncio.run(associator.associate(association_input()))
    assert provider.calls == 2  # validation_retries=1：原问一次、纠正一次


def test_a_transient_transport_error_is_retried_once() -> None:
    class Flaky(StructuredChatClient):
        failures = 1
        attempts = 0

        async def complete_json_async(self, request, **kwargs):  # type: ignore[no-untyped-def, override]
            Flaky.attempts += 1
            if Flaky.attempts <= Flaky.failures:
                raise ModelTransportError("boom")
            return await super().complete_json_async(request, **kwargs)

    provider = ScriptedProvider([output()])
    client = Flaky(ChatClient(model_config(), provider), validation_retries=1)
    associator = LLMAssociator(client, config=AssociationConfig(transient_retry_delay_seconds=0))

    assert asyncio.run(associator.associate(association_input())).drafts
    assert Flaky.attempts == 2

    Flaky.attempts = 0
    Flaky.failures = 5
    exhausted = LLMAssociator(client, config=AssociationConfig(transient_retries=1, transient_retry_delay_seconds=0))
    with pytest.raises(ModelTransportError):
        asyncio.run(exhausted.associate(association_input()))
    assert Flaky.attempts == 2
    Flaky.failures = 1


def test_too_many_targets_skip_the_batch_rather_than_splitting_it() -> None:
    """切块会破坏"同一天几次一起看"这个前提，所以超限就跳过留信号。"""

    associator, _client, provider = _associator([output()], max_targets_per_call=1)
    with pytest.raises(AssociationLimitError, match="max_targets_per_call"):
        asyncio.run(associator.associate(two_target_input()))
    assert provider.calls == 0


def test_an_oversized_prompt_skips_the_batch() -> None:
    associator, _client, provider = _associator([output()], max_prompt_chars=10)
    with pytest.raises(AssociationLimitError, match="max_prompt_chars"):
        asyncio.run(associator.associate(association_input()))
    assert provider.calls == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_targets_per_call", 0),
        ("max_prompt_chars", 0),
        ("transient_retries", -1),
        ("transient_retry_delay_seconds", -1.0),
    ],
)
def test_the_config_refuses_impossible_numbers(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        AssociationConfig(**{field: value})  # type: ignore[arg-type]


def test_the_fixture_day_is_the_one_the_targets_happened_on() -> None:
    payload = association_input()
    assert payload.day == DAY == payload.occurrences[2].started_at.date()
    assert payload.causes[0].started_at.date() < payload.day


def test_a_cause_row_number_is_offset_by_the_whole_day_stream() -> None:
    two = association_input(causes=(association_input().causes[0], _replace(association_input().causes[0], no=2)))
    assert sorted(two.citable) == [1, 2, 3, 4, 5]
    assert two.cause_no(5) == 2


def test_a_cause_row_must_carry_its_own_instant() -> None:
    bad = CauseRow(
        no=1, uri="behavior://x", name="通话", started_at=datetime(2026, 9, 6, 18), summary="约球"
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        association_input(causes=(bad,))


# ── 存得下、渲得出：三道属于我们自己的闸 ─────────────────────────────────────


def test_a_premise_too_long_to_store_is_dropped() -> None:
    """记录整体有字节硬闸，但撞上它是整条写不进去、那一天永远完不成，而调用方完全不知道原因。
    降级要发生在这里。"""

    left = [{"text": "长" * 1_000, "consumed_by": "做饭"}]
    result = assemble_association(output(entry(left=left)), association_input())
    assert result.drafts[0].left == ()
    assert any("too long to store" in note for note in result.signals)


def test_a_flood_of_premises_from_one_occurrence_is_capped() -> None:
    left = [{"text": f"第 {n} 条前提", "consumed_by": "做饭"} for n in range(MAX_PREMISES_PER_OCCURRENCE + 3)]
    result = assemble_association(output(entry(left=left)), association_input())
    assert len(result.drafts[0].left) == MAX_PREMISES_PER_OCCURRENCE
    assert any("already left" in note for note in result.signals)


def test_the_situation_list_is_capped_and_says_how_many_it_left_out() -> None:
    """L1 只增不减。没有这道闸，情形攒多了会撞提示词字符上限，而超限被判成确定性失败、
    第一次就把那个候选永久挡住。"""

    many = tuple(
        SituationRow(no=n, text=f"第 {n} 种情形", days=(date(2026, 9, 4),) * min(n, 1)) for n in range(1, 20)
    )
    rendered = render_situations(association_input(situations=many))

    assert rendered.count("  S") == 12
    assert "另有 7 种更少见的情形没有列出" in rendered
