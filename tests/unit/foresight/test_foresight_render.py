"""证据的 Markdown 渲染：数字连着出处、没说的说出来、卡分三段、保护闸从远到近砍。"""

from __future__ import annotations

from datetime import timedelta

from habitus.foresight import CandidateEvidence, CandidateNumbers, Layer, Provenance, render_candidate
from habitus.foresight.assemble import _empty_background
from habitus.foresight.render import render_moment
from tests.unit.foresight.fixtures import MONDAY, Ground, at

NOW = MONDAY + timedelta(days=28)
PRIOR_ONLY = CandidateNumbers(
    marginal=0.04,
    hazard=0.0,
    cumulative=0.0,
    lift_all_day=1.0,
    lift_weekday=0.0,
    count=0.0,
    n_eff=0.0,
    trend=None,
    trend_n_eff=0.0,
    recurrence=None,
    done_today=0,
)


def evidence_for(tmp_path, *, name: str = "site", gap: bool = False, max_days: int = 40) -> CandidateEvidence:
    """三个周一都在 19:00 打球一小时；第一周之后洗澡、第三周之前先喝了水，第二周之前断过档；三周都关联了。

    每次新建现场都用自己的目录：同一棵行为树发两遍同一批 occurrence 会造出重复。
    """

    ground = Ground(tmp_path / name, now=at(NOW, 23, 0))
    for week in range(3):
        uri = ground.record(MONDAY + timedelta(days=7 * week), "打球", 19, 0, kind="打球", lasts_minutes=60)
        ground.associate(uri, kind="打球", context=f"第 {week + 1} 周的周一晚上去打的", situation="周一晚上自己去")
    ground.record(MONDAY, "洗澡", 20, 10, kind="洗澡")
    ground.record(MONDAY + timedelta(days=14), "喝水", 18, 50, kind="喝水")
    if gap:
        # 空白段里**不能**有任何行为的起点：「没读懂」段被段内读出的行为证伪之后整段作废
        # （行为树与预测树同一条规则），那样就测不到删失了。
        ground.gap(MONDAY + timedelta(days=7), 18, 0, 18, 50)
    pack = ground.pack(at(NOW, 19, 5), max_days=max_days)
    return next(item for item in pack.candidates if item.kind_token == "打球")


def test_the_table_shows_the_raw_ratio_for_the_three_chain_layers(tmp_path) -> None:
    """链上三层写成"分子/分母 = 比值"，只有全天是已发布的率——判断者要看出这个数薄不薄。"""

    text = render_candidate(evidence_for(tmp_path))
    assert "| 本槽 | 3.00/3.00 = 1.000 | 3 天 | — |" in text
    lines = {line.split("|")[1].strip(): line for line in text.splitlines() if line.startswith("| ")}
    assert "=" in lines["邻域"] and "=" in lines["跨周几"]
    assert "=" not in lines["全天"]


def test_the_moment_line_names_the_slot_and_the_calendar(tmp_path) -> None:
    ground = Ground(tmp_path, now=at(NOW, 23, 0))
    ground.record(MONDAY, "打球", 19, 0, kind="打球")
    moment = ground.pack(at(NOW, 19, 5)).moment
    assert render_moment(moment) == "此刻：2026-08-31 周一 19:05（第 76 槽）"


def test_a_slot_that_was_never_observed_is_not_a_measured_zero(tmp_path) -> None:
    """分母为零是"这一格我们根本没看过"，不是"看了没发生"。

    观测跨度还没覆盖到那个周几时，曝光连键都没有。渲染成 0.00/0.00 = 0.000 会让判断者把它
    读成一个有把握的实测零——而这一列本来就是为了让他看出数薄不薄。
    """

    ground = Ground(tmp_path / "unobserved", now=at(MONDAY + timedelta(days=2), 23, 0))
    ground.record(MONDAY, "打球", 19, 0, kind="打球")
    # 观测跨度只有周一到周三，周五那一整列一次都没看过；周五没有曲线，打球不是候选，直接造证据。
    friday_pack = ground.pack(at(MONDAY + timedelta(days=4), 19, 5))
    assert friday_pack.candidates == ()
    from habitus.foresight import CellIndex, provenance
    from habitus.prediction.model import SlotKey

    layers = provenance(CellIndex.of(ground.tree()), "打球", SlotKey(weekday=4, slot=76), half_width=3, associated_on=lambda _day: True)
    evidence = CandidateEvidence(
        kind_token="打球", numbers=PRIOR_ONLY, provenance=layers, expanded=False, background=_empty_background()
    )
    assert layers.slot.exposure == 0.0
    text = render_candidate(evidence)
    assert "| 本槽 | 从没看过这一格 |" in text
    assert not any("0.000" in line for line in text.splitlines() if line.startswith("| "))  # 表里没有冒充实测的零
    assert "只列名字与数字" in text


def test_cards_are_told_in_three_parts_with_their_gloss(tmp_path) -> None:
    text = render_candidate(evidence_for(tmp_path))
    assert "### 历史 · 3 次发生，每次一张卡（按 # 编号引用）" in text
    assert "- #1 2026-08-03 周一 19:00–20:00 打球 〔本槽〕（周一晚上自己去）" in text
    assert "  之前：（无）" in text and "  之后：20:10 洗澡(1)" in text
    assert "  之前：18:50 喝水(1)" in text
    assert "  关联：第 1 周的周一晚上去打的" in text
    assert "### 情形" in text and "本周几出现过：周一晚上自己去（3 天）" in text


def test_what_is_missing_is_said_out_loud(tmp_path) -> None:
    """没关联的日子与被保护闸截掉的日子都要写出来，否则"给你看的这几条"会被读成"一共就这几条"。"""

    text = render_candidate(evidence_for(tmp_path, name="capped", max_days=1))
    assert "更早的日子没有展开（本槽 2、邻域 2、跨周几 2、全天 2 天）" in text
    full = render_candidate(evidence_for(tmp_path, name="full"))
    assert "还没关联" not in full  # 这个现场三天都关联了，就不该无中生有地报缺


def test_an_unassociated_day_is_named_on_its_card(tmp_path) -> None:
    ground = Ground(tmp_path, now=at(NOW, 23, 0))
    ground.record(MONDAY, "打球", 19, 0, kind="打球")
    (candidate,) = ground.pack(at(NOW, 19, 5)).expanded
    text = render_candidate(candidate)
    assert "1 天有数、语义层还没关联" in text
    assert "  关联：那天还没关联" in text


def test_the_three_valued_neighbour_is_told_as_it_is(tmp_path) -> None:
    """删失不能说成 ∅：那是在观测最差的地方下最确凿的结论，树那边专门有一条规则防它。"""

    text = render_candidate(evidence_for(tmp_path, gap=True))
    assert "紧邻上一条：那段没看清（删失）" in text
    assert "紧邻上一条：没有（∅）" in text  # 另外两天窗口内确实什么都没有
    assert "紧邻下一条：洗澡" in text


def test_every_card_is_rendered_no_matter_which_layer(tmp_path) -> None:
    """不按字数砍（2026-09-16 定）：包里有什么就渲什么，四层的卡都在。"""

    ground = Ground(tmp_path, now=at(NOW, 23, 0))
    for week in range(3):
        ground.record(MONDAY + timedelta(days=7 * week), "打球", 19, 0, kind="打球")
    ground.record(MONDAY + timedelta(days=9), "打球", 8, 0, kind="打球")  # 周三早上，全天层
    (evidence,) = ground.pack(at(NOW, 19, 5)).expanded
    text = render_candidate(evidence)
    assert "〔全天〕" in text and "〔本槽〕" in text


def test_a_rate_with_no_days_behind_it_is_marked_as_prior_only() -> None:
    """一层一天都没发生过、率却不是 0，那是 Laplace 先验在说话；不标出来会被当成实测结论。"""

    empty = Layer(name="slot", value=0.0, days=(), unassociated=(), hits=0.0, exposure=4.0)
    layers = Provenance(
        slot=empty,
        pool=Layer(name="pool", value=0.0, days=(), unassociated=(), hits=0.0, exposure=20.0),
        cross_weekday=Layer(name="cross_weekday", value=0.0, days=(), unassociated=(), hits=0.0, exposure=140.0),
        all_day=Layer(name="all_day", value=0.04, days=(), unassociated=()),
    )
    text = render_candidate(
        CandidateEvidence(kind_token="吃药", numbers=PRIOR_ONLY, provenance=layers, expanded=False, background=_empty_background())
    )
    assert "| 全天 | 0.0400（只有先验，没有证据） | 0 天 | — |" in text
    assert "复发：没有间隔样本 · 今天还没做" in text
