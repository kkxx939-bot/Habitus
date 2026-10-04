"""② 假设：身份不含版本、集合无序、先验只给方向、第几次机会只用来读、对着概念集核对、同身份改内容要显式。"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from habitus.scene.codec import SceneRecordError
from habitus.scene.concepts import ConceptRole, ConceptSet
from habitus.scene.hypotheses import (
    Antecedent,
    Aspect,
    Direction,
    Hypothesis,
    HypothesisError,
    HypothesisOrigin,
    HypothesisSource,
    HypothesisStore,
    HypothesisStoreError,
    PeakWindow,
    TypePrior,
)
from habitus.scene.hypotheses.document import decode, encode
from tests.unit.scene.concept_fixtures import ALL_CONCEPTS, concept, concept_set

CREATED = datetime(2026, 9, 26, 1, 0, tzinfo=UTC)
BASELINE = HypothesisSource(HypothesisOrigin.BASELINE)
#: 测试用的峰表：早餐一天一个峰；咖啡三个。窗口从节律口抄，这里写死。
WINDOWS = {
    "早餐": (PeakWindow(1, 375, 555),),
    "起床": (PeakWindow(1, 400, 580),),
    "咖啡": (PeakWindow(1, 465, 555), PeakWindow(2, 795, 885), PeakWindow(3, 1155, 1245)),
    "晚睡": (PeakWindow(1, 1380, 1560),),
}


def hypothesis(
    *antecedents: Antecedent | str,
    consequent: str = "早餐",
    aspect: Aspect = Aspect.PROBABILITY,
    direction: Direction = Direction.DOWN,
    type_prior: TypePrior | None = TypePrior.INHIBITING,
    consequent_peak: int | None = 1,
    horizon: int = 1,
    split_by: tuple[str, ...] = (),
    released_by: tuple[str, ...] = (),
    note: str = "睡得晚起得晚，早饭常跳过",
    windows: dict[str, tuple[PeakWindow, ...]] | None = None,
) -> Hypothesis:
    items = tuple(item if isinstance(item, Antecedent) else Antecedent(item) for item in antecedents) or (Antecedent("晚睡"),)
    if windows is None:
        windows = {name: table for name, table in WINDOWS.items() if (name == consequent and consequent_peak is not None) or any(a.concept == name and a.peak is not None for a in items)}
    return Hypothesis(
        antecedents=items,
        consequent=consequent,
        aspect=aspect,
        direction=direction,
        type_prior=type_prior,
        consequent_peak=consequent_peak,
        windows=windows,
        horizon=horizon,
        split_by=split_by,
        released_by=released_by,
        note=note,
        source=BASELINE,
        created_at=CREATED,
    )


def enabling(*antecedents: str, consequent: str = "打球", released_by: tuple[str, ...] = ()) -> Hypothesis:
    """无节律型：概率方面、后件没有峰（不数峰、没有时效）；兑现率与兑现间隔是它的读数。"""

    return hypothesis(
        *antecedents, consequent=consequent, direction=Direction.UP, type_prior=TypePrior.ENABLING, consequent_peak=None, released_by=released_by, note="约了就去"
    )


# ── 身份 ──────────────────────────────────────────────────────────────────────


def test_identity_is_antecedent_set_consequent_and_aspect_with_no_version() -> None:
    late_travel = hypothesis(Antecedent("晚睡", "重"), "出差中")
    assert late_travel.identity == "早餐/出差中+晚睡@重--probability--1"
    assert late_travel.label() == "{出差中, 晚睡·重} → 早餐 · 概率"
    assert hypothesis("出差中", Antecedent("晚睡", "重")).identity == late_travel.identity  # 集合无序
    assert hypothesis("晚睡", aspect=Aspect.TIMING, direction=Direction.UP, type_prior=None).identity == "早餐/晚睡--timing--1"
    # 后果峰号进身份：节律型是数字，无节律型是 open —— 一天多峰的后件才挂得下逐峰三条（账按钟面峰记，2026-10-01）。
    assert enabling("约球").identity == "打球/约球--probability--open"
    assert enabling("约球").opportunity_label == "无节律型：不数峰，直到后件到来"
    coffee = {k: hypothesis("晚睡", consequent="咖啡", direction=Direction.UP, type_prior=TypePrior.PROMOTING, consequent_peak=k) for k in (1, 2, 3)}
    assert {item.identity for item in coffee.values()} == {"咖啡/晚睡--probability--1", "咖啡/晚睡--probability--2", "咖啡/晚睡--probability--3"}
    assert coffee[2].consequent_window == PeakWindow(2, 795, 885) and coffee[2].opportunity_label == "后件的第 2 个峰（13:15–14:45）"
    # 前因也按峰分：峰号跟在概念后面（#0 = 峰外）。
    split = hypothesis(Antecedent("晚睡", peak=1), consequent="咖啡", direction=Direction.UP, type_prior=TypePrior.PROMOTING, consequent_peak=2)
    assert split.identity == "咖啡/晚睡#1--probability--2" and split.label() == "{晚睡·第1峰} → 咖啡 · 概率"
    assert hypothesis(Antecedent("晚睡", peak=0), consequent="咖啡", direction=Direction.UP, type_prior=TypePrior.PROMOTING).identity == "咖啡/晚睡#0--probability--1"
    assert enabling("约球").is_open_ended and not hypothesis("晚睡").is_open_ended
    # 释放条件进指纹（它改变"这本账在量什么"：写了它之后，被作废的承诺不再等在那里）。
    released = enabling("约球", released_by=("约球",))
    assert released.opportunity_label == "无节律型：不数峰，直到后件到来或 约球"
    assert released.fingerprint != enabling("约球").fingerprint and released.identity == enabling("约球").identity
    assert hypothesis("晚睡", consequent="咖啡", direction=Direction.UP, type_prior=TypePrior.PROMOTING, aspect=Aspect.COUNT, horizon=3).opportunity_label == "后件的第 1–3 个峰（07:45–09:15）"
    # 指纹只看"账在量什么"：改理由不变，改峰号或峰表就变。
    assert hypothesis("晚睡", note="另一句").fingerprint == hypothesis("晚睡").fingerprint
    assert coffee[2].fingerprint != coffee[1].fingerprint
    assert hypothesis("晚睡", windows={"早餐": (PeakWindow(1, 360, 540),)}).fingerprint != hypothesis("晚睡").fingerprint


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(consequent="晚睡"), "own antecedent"),
        (dict(consequent_peak=0), "1st"),
        (dict(consequent_peak=31), "1st"),
        (dict(consequent_peak=2), "no peak #2"),  # 早餐只有一个峰
        (dict(antecedents=(Antecedent("晚睡", peak=2),)), "no peak #2"),
        (dict(windows={}), "no peak #1"),  # 有峰号就必须有峰表
        (dict(windows={"早餐": (PeakWindow(2, 1, 2),)}), "numbered 1..n"),
        (dict(aspect=Aspect.TIMING, type_prior=None, direction=Direction.UP, consequent_peak=None), "name the consequent peak"),  # 时刻要知道量哪个峰
        (dict(horizon=3), "only count hypotheses"),
        (dict(aspect=Aspect.COUNT, type_prior=None, direction=Direction.UP, horizon=0), "horizon counts"),
        (dict(type_prior=None), "carry a type prior"),
        (dict(type_prior=TypePrior.INHIBITING, direction=Direction.UP), "agree with the type prior"),
        (dict(type_prior=TypePrior.PROMOTING, direction=Direction.DOWN), "agree with the type prior"),
        (dict(note="两行\n理由"), "single printable line"),
        (dict(split_by=("在家", "在家")), "once"),
    ],
)
def test_a_hypothesis_is_self_consistent(kwargs, message) -> None:
    antecedents = kwargs.pop("antecedents", ())
    with pytest.raises(HypothesisError, match=message):
        hypothesis(*antecedents, **kwargs)


def test_timing_and_count_aspects_may_leave_the_type_prior_blank() -> None:
    """"起床推后"不是抑制也不是促进：非概率方面允许不标。"""

    item = hypothesis("晚睡", consequent="起床", aspect=Aspect.TIMING, direction=Direction.UP, type_prior=None, note="睡得晚起得晚")
    assert item.type_prior is None and "先验类型：（不标）" in encode(item)
    assert decode(encode(item)) == item


def test_antecedent_sets_are_bounded_and_name_each_concept_once() -> None:
    with pytest.raises(HypothesisError, match="each concept once"):
        hypothesis("晚睡", Antecedent("晚睡", "重"))
    with pytest.raises(HypothesisError, match="1–4"):
        hypothesis("a", "b", "c", "d", "e")
    with pytest.raises(HypothesisError):
        Antecedent("晚睡", " 重")
    with pytest.raises(HypothesisError):
        Antecedent(".hidden")
    # 分隔符：概念名与档名里都不许有，否则 {晚睡@重+出差中} 与 {晚睡@"重+出差中"} 落同一个文件。
    with pytest.raises(HypothesisError):
        Antecedent("晚睡", "重+出差中")
    with pytest.raises(HypothesisError):
        Antecedent("出差中+晚睡")


# ── 对着概念集核对 ─────────────────────────────────────────────────────────────


def test_validation_against_the_concept_set() -> None:
    concepts = concept_set()
    hypothesis(Antecedent("晚睡", "重"), "出差中", split_by=("周末",)).validate_against(concepts)
    with pytest.raises(HypothesisError, match="behaviour concept"):
        hypothesis("晚睡", consequent="出差中", windows={"出差中": WINDOWS["早餐"]}).validate_against(concepts)
    with pytest.raises(HypothesisError, match="not in the concept set"):
        hypothesis("熬夜工作").validate_against(concepts)
    with pytest.raises(HypothesisError, match="no grade"):
        hypothesis(Antecedent("晚睡", "极重")).validate_against(concepts)
    with pytest.raises(HypothesisError, match="anchor"):
        hypothesis("出差中", "周末").validate_against(concepts)  # 只有情境概念，没有事件可锚窗
    with pytest.raises(HypothesisError, match="situation concept"):
        hypothesis("晚睡", split_by=("打球",)).validate_against(concepts)
    # 释放条件是机械匹配的，只能是行为概念；且只有无节律型能写，不能指向后件自己。
    open_ended = dict(consequent="早餐", direction=Direction.UP, type_prior=TypePrior.ENABLING, consequent_peak=None)
    hypothesis("打球", released_by=("运动",), **open_ended).validate_against(concepts)
    with pytest.raises(HypothesisError, match="released_by .* must be a behaviour concept"):
        hypothesis("打球", released_by=("出差中",), **open_ended).validate_against(concepts)
    with pytest.raises(HypothesisError, match="not in the concept set|behaviour concept"):
        hypothesis("打球", released_by=("熬夜工作",), **open_ended).validate_against(concepts)
    with pytest.raises(HypothesisError, match="released_by cannot name the consequent"):
        hypothesis("打球", released_by=("早餐",), **open_ended)
    with pytest.raises(HypothesisError, match="released_by belongs to open-ended"):
        hypothesis("打球", consequent="早餐", released_by=("运动",), direction=Direction.UP, type_prior=TypePrior.ENABLING)
    with pytest.raises(HypothesisError, match="released_by names each concept once"):
        hypothesis("打球", released_by=("运动", "运动"), **open_ended)
    # 「打球 → 运动」是重言：读侧沿 parent 链聚合后 100% 命中。两个方向都拒。
    with pytest.raises(HypothesisError, match="tautology"):
        hypothesis("打球", consequent="运动", direction=Direction.UP, type_prior=TypePrior.PROMOTING, note="x", windows={"运动": WINDOWS["早餐"]}).validate_against(concepts)
    with pytest.raises(HypothesisError, match="tautology"):
        hypothesis("运动", consequent="打球", direction=Direction.UP, type_prior=TypePrior.PROMOTING, note="x", windows={"打球": WINDOWS["早餐"]}).validate_against(concepts)


# ── 落盘格式 ──────────────────────────────────────────────────────────────────


def test_a_hypothesis_document_round_trips_and_rejects_a_tampered_body() -> None:
    item = hypothesis(Antecedent("晚睡", "重"), "出差中", split_by=("周末",))
    text = encode(item)
    assert "# {出差中, 晚睡·重} → 早餐 · 概率" in text
    assert "方向：↓ 更不可能发生" in text and "落在：后件的第 1 个峰（06:15–09:15）" in text and "分账建议：周末" in text
    assert f"指纹：{item.fingerprint}" in text and "强度" not in text  # 基准不写强度
    assert decode(text) == item
    open_text = encode(enabling("约球"))
    assert "· 概率" in open_text and "落在：无节律型：不数峰，直到后件到来" in open_text and decode(open_text) == enabling("约球")
    # 释放条件那一行只印给无节律型；节律型印出来恒为空。
    assert "释放条件：（无，一直立着）" in open_text and "释放条件" not in text
    released_text = encode(enabling("约球", released_by=("约球",)))
    assert "释放条件：约球" in released_text and decode(released_text) == enabling("约球", released_by=("约球",))
    with pytest.raises(SceneRecordError, match="canonical rendering"):
        decode(text.replace("方向：↓", "方向：↑"))
    with pytest.raises(SceneRecordError, match="identity"):
        decode(text, expected_identity="早餐/晚睡--probability--1")
    with pytest.raises(SceneRecordError):
        decode(text.replace('"consequent_peak":1', '"consequent_peak":"1"'))  # 峰号是整数


# ── 存储 ──────────────────────────────────────────────────────────────────────


def test_store_files_by_consequent_and_checks_against_the_concept_set(tmp_path) -> None:
    concepts = concept_set()
    store = HypothesisStore(tmp_path / "scene")
    late = hypothesis("晚睡")
    late_travel = hypothesis(Antecedent("晚睡", "重"), "出差中")
    timing = hypothesis("晚睡", aspect=Aspect.TIMING, direction=Direction.UP, type_prior=None)
    ball = hypothesis("打球", consequent="早餐", direction=Direction.UP, type_prior=TypePrior.PROMOTING, note="打完球饿")

    path = store.write(late, concepts)
    assert path == tmp_path / "scene" / "hypotheses" / "早餐" / "晚睡--probability--1.md"
    for item in (late_travel, timing, ball):
        store.write(item, concepts)
    store.write(hypothesis("晚睡", consequent="打球", note="累了不去", windows={"打球": WINDOWS["早餐"]}), concepts)

    assert store.consequents() == ("打球", "早餐")
    assert [item.identity for item in store.for_consequent("早餐")] == [
        "早餐/出差中+晚睡@重--probability--1",
        "早餐/打球--probability--1",
        "早餐/晚睡--probability--1",
        "早餐/晚睡--timing--1",
    ]
    assert store.read(late.identity) == late and store.exists(late.identity) and not store.exists("早餐/咖啡--count--1")
    assert len(store.read_all()) == 5 and store.for_consequent("咖啡") == ()
    (tmp_path / "scene" / "hypotheses" / "早餐" / ".DS_Store").write_bytes(b"\x00")
    assert len(store.for_consequent("早餐")) == 4  # 点文件是噪音

    with pytest.raises(HypothesisStoreError, match="no grade"):
        store.write(hypothesis(Antecedent("晚睡", "极重")), concepts)
    with pytest.raises(HypothesisStoreError, match="'<consequent>/<leaf>'"):
        store.read("晚睡--probability--1")


def test_rewriting_a_hypothesis_with_another_fingerprint_must_be_explicit(tmp_path) -> None:
    """同身份改内容（换方向）会让一本账混两种量：默认拒写，``supersede=True`` 表示调用方会处理那本账。

    换"第几次机会"已经不是这一类了——它进了身份，所以是另一条假设、另一本账，两条并存。
    """

    concepts = concept_set()
    store = HypothesisStore(tmp_path / "scene")
    store.write(hypothesis("晚睡"), concepts)
    store.write(hypothesis("晚睡", note="另一句理由"), concepts)  # 同指纹，随便改
    assert store.read("早餐/晚睡--probability--1").note == "另一句理由"
    # 换后果峰号是**另一条假设、另一本账**（峰号进身份了），所以两条并存、不需要 supersede。
    two_peaks = {"早餐": (*WINDOWS["早餐"], PeakWindow(2, 700, 800))}
    store.write(hypothesis("晚睡", consequent_peak=2, windows=two_peaks), concepts)
    assert {h.identity for h in store.for_consequent("早餐")} == {"早餐/晚睡--probability--1", "早餐/晚睡--probability--2"}
    # 换峰表（同一个峰号、窗口挪了）不动身份、只动指纹：这本账量的时段变了，要显式。
    with pytest.raises(HypothesisStoreError, match="supersede=True"):
        store.write(hypothesis("晚睡", windows={"早餐": (PeakWindow(1, 360, 540),)}), concepts)
    # 同身份改内容仍要显式：换方向（抑制→促进）不动身份、只动指纹。
    with pytest.raises(HypothesisStoreError, match="supersede=True"):
        store.write(hypothesis("晚睡", direction=Direction.UP, type_prior=TypePrior.PROMOTING), concepts)
    store.write(hypothesis("晚睡", direction=Direction.UP, type_prior=TypePrior.PROMOTING), concepts, supersede=True)
    assert store.read("早餐/晚睡--probability--1").direction is Direction.UP


def test_store_reports_corrupt_records_and_non_canonical_directories(tmp_path) -> None:
    store = HypothesisStore(tmp_path / "scene")
    item = hypothesis("晚睡")
    path = store.write(item, concept_set())
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(HypothesisStoreError, match="corrupt"):
        store.read(item.identity)
    (tmp_path / "scene" / "hypotheses" / "Breakfast").mkdir()
    with pytest.raises(HypothesisStoreError, match="canonical identity"):
        store.consequents()
    # 概念集变了之后旧假设仍在盘上：读得出来，但再写就过不了核对。
    smaller = ConceptSet([c for c in ALL_CONCEPTS if c.name != "晚睡"] + [concept("晚睡", "另一句", role=ConceptRole.STATE)])
    with pytest.raises(HypothesisStoreError, match="anchor"):
        store.write(hypothesis("晚睡"), smaller)
