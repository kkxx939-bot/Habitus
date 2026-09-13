"""证据的 Markdown 渲染：数字连着出处、没说的说出来、保护闸从远到近砍。"""

from __future__ import annotations

from datetime import timedelta

import pytest

from habitus.foresight import CellIndex, ForesightError, Layer, LayerBackground, render_candidate
from habitus.foresight.assemble import CandidateEvidence, candidate_evidence, moment_at
from tests.unit.foresight.fixtures import Ground, at
from tests.unit.foresight.test_foresight_provenance import MONDAY

NOW = MONDAY + timedelta(days=28)


def evidence_for(tmp_path, *, name: str = "site", gap: bool = False, max_days: int = 40):
    """三个周一都在 19:00 打球；第三周之前先喝了水，第二周之前断过档。

    每次新建现场都用自己的目录：同一棵行为树发两遍同一批 occurrence 会造出重复，而已归组的
    日子再 refresh 一次什么都不会发布。
    """

    ground = Ground(tmp_path / name, now=at(NOW, 23, 0))
    for week in range(3):
        ground.record(MONDAY + timedelta(days=7 * week), "打球", 19, 0, kind="打球")
    ground.record(MONDAY, "洗澡", 20, 10, kind="洗澡")
    ground.record(MONDAY + timedelta(days=14), "喝水", 18, 50, kind="喝水")
    if gap:
        # 空白段里**不能**有任何行为的起点：「没读懂」段被段内读出的行为证伪之后整段作废
        # （行为树与预测树同一条规则），那样就测不到删失了。
        ground.gap(MONDAY + timedelta(days=7), 18, 0, 18, 50)
    days = [MONDAY + timedelta(days=7 * week) for week in range(3)]
    ground.group(*days)
    moment = moment_at(at(NOW, 19, 5), slot_minutes=15)
    return candidate_evidence(
        CellIndex.of(ground.tree()),
        "打球",
        moment,
        ground.cache(),
        half_width=3,
        window_days=30,
        transition_window_seconds=7_200.0,
        max_days_per_layer=max_days,
    )


def test_the_table_shows_the_raw_ratio_for_the_three_chain_layers(tmp_path) -> None:
    """链上三层写成"分子/分母 = 比值"，只有全天是已发布的率——判断者要看出这个数薄不薄。"""

    text = render_candidate(evidence_for(tmp_path))
    assert "| 本槽 | 3.00/3.00 = 1.000 | 3 天 | — |" in text
    lines = {line.split("|")[1].strip(): line for line in text.splitlines() if line.startswith("| ")}
    assert "=" in lines["邻域"] and "=" in lines["跨周几"]
    assert "=" not in lines["全天"]
    assert "此刻：" in text and "周一 19:05（第 76 槽）" in text


def test_a_slot_that_was_never_observed_is_not_a_measured_zero(tmp_path) -> None:
    """分母为零是"这一格我们根本没看过"，不是"看了没发生"。

    观测跨度还没覆盖到那个周几时，曝光连键都没有。渲染成 0.00/0.00 = 0.000 会让判断者把它
    读成一个有把握的实测零——而这一列本来就是为了让他看出数薄不薄。
    """

    ground = Ground(tmp_path / "unobserved", now=at(MONDAY + timedelta(days=2), 23, 0))
    ground.record(MONDAY, "打球", 19, 0, kind="打球")
    ground.group(MONDAY)
    # 观测跨度只有周一到周三，周五那一整列一次都没看过。
    friday = moment_at(at(MONDAY + timedelta(days=4), 19, 5), slot_minutes=15)
    evidence = candidate_evidence(
        CellIndex.of(ground.tree()), "打球", friday, ground.cache(),
        half_width=3, window_days=30, transition_window_seconds=7_200.0, max_days_per_layer=40,
    )
    assert evidence.layers[0].layer.exposure == 0.0
    text = render_candidate(evidence)
    assert "| 本槽 | 从没看过这一格 |" in text
    assert "0.000" not in text.split("### ")[0]  # 表里没有任何冒充实测的零


def test_what_is_missing_is_said_out_loud(tmp_path) -> None:
    """没归组的日子与被保护闸截掉的日子都要写出来，否则"给你看的这几条"会被读成"一共这几条"。"""

    text = render_candidate(evidence_for(tmp_path, name="capped", max_days=1))
    assert "更早的 2 天没有展开" in text
    full = render_candidate(evidence_for(tmp_path, name="full"))
    assert "还没归组" not in full  # 这个现场三天都归了组，就不该无中生有地报缺


def test_the_three_valued_neighbour_is_told_as_it_is(tmp_path) -> None:
    """删失不能说成 ∅：那是在观测最差的地方下最确凿的结论，树那边专门有一条规则防它。"""

    text = render_candidate(evidence_for(tmp_path, gap=True))
    assert "紧邻上一条：那段没看清（删失）" in text
    assert "紧邻上一条：没有（∅）" in text  # 另外两天窗口内确实什么都没有
    assert "紧邻下一条：洗澡" in text


def test_the_protective_limit_trims_from_the_far_layers_first(tmp_path) -> None:
    """从全天往本槽砍：离此刻越远的层，少看几条损失越小；四层的数字与出处一个都不砍。"""

    evidence = evidence_for(tmp_path)
    full = render_candidate(evidence)
    cap = len(full) // 2
    trimmed = render_candidate(evidence, max_chars=cap)
    # 断言必须对着 max_chars，不是对着完整文本：只查"没变长"的话，砍序写错、甚至只砍一层，
    # 测试照样全绿（变异实证）。
    assert len(trimmed) <= cap
    assert "| 本槽 |" in trimmed and "| 全天 |" in trimmed  # 表永远在
    assert "因篇幅未展开" in trimmed
    head = trimmed.index("### 本槽")
    assert trimmed.count("- 08-", head, trimmed.index("### 邻域")) > 0  # 本槽最后才砍
    with pytest.raises(ForesightError):
        render_candidate(evidence, max_chars=0)


def test_a_rate_with_no_days_behind_it_is_marked_as_prior_only(tmp_path) -> None:
    """率不为零但一天都没发生过，说的只能是平滑用的先验，不是实测。

    链上三层改成裸比值之后，这种组合在正常数据上已经到不了了（分子为零则比值为零）——这条
    是**护栏**：只要还有任何一层给的是已发布的平滑率，它就可能出现，而一个 0.04 冒充实测的
    代价太大。所以直接构造那个形状来钉住渲染规则本身。
    """

    evidence = evidence_for(tmp_path, name="prior")
    forged = CandidateEvidence(
        kind_token=evidence.kind_token,
        moment=evidence.moment,
        layers=(
            *evidence.layers[:3],
            LayerBackground(
                layer=Layer(name="all_day", value=0.0425, days=(), ungrouped=()), views=(), dropped_days=0
            ),
        ),
    )
    text = render_candidate(forged)
    assert "0.0425（只有先验，没有证据）" in text
    assert "没有出处日" in text and "这一层一天都没发生过，没有历史可看" in text
