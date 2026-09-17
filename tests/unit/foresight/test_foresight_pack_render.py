"""整包的渲染：此刻场景在前、候选各一节（数字在前、卡在后）、只列名的收尾。不按字数砍。"""

from __future__ import annotations

from datetime import timedelta

from habitus.foresight import UnsealedRow, render_pack
from tests.unit.foresight.fixtures import MONDAY, Ground, at

NOW = MONDAY + timedelta(days=28)


def ground_for(tmp_path) -> Ground:
    """周一 19:00 打球四周、18:40 收拾球包；早饭每天 8:00（19:05 时只列名）；今天有未封口与空白。"""

    ground = Ground(tmp_path, now=at(NOW, 23, 0))
    for week in range(4):
        day = MONDAY + timedelta(days=7 * week)
        ground.record(day, "收拾球包", 18, 40, kind="收拾球包")
        play = ground.record(day, "打球", 19, 0, kind="打球", lasts_minutes=60)
        ground.record(day, "吃早饭", 8, 0, kind="吃早饭")
        if week < 2:
            ground.associate(play, kind="打球", context=f"第 {week + 1} 周", situation="周一下班后自己去")
    ground.record(MONDAY + timedelta(days=2), "打球", 19, 0, kind="打球")  # 周三，跨周几层
    ground.gap(NOW, 12, 0, 13, 0)
    return ground


def test_the_pack_reads_now_first_then_each_expanded_candidate_then_the_named(tmp_path) -> None:
    ground = ground_for(tmp_path)
    unsealed = (
        UnsealedRow(name="换鞋", kind_token=None, started_at=at(NOW, 18, 50), last_observed_at=at(NOW, 18, 52), summary="换了球鞋"),
        UnsealedRow(name=None, kind_token=None, started_at=at(NOW, 18, 20), last_observed_at=at(NOW, 18, 30), summary=None),
    )
    text = render_pack(ground.pack(at(NOW, 19, 5), unsealed=unsealed))

    assert text.startswith("# 证据包 · 一代 test-generation\n\n此刻：2026-08-31 周一 19:05（第 76 槽）")
    assert "钟面：槽宽 15 分钟，一天 96 槽，第 0 槽从 00:00 起" in text
    now = text.index("## 此刻场景（18:15–19:05）")
    play = text.index("## 打球")
    pack = text.index("## 收拾球包")
    named = text.index("## 只列名的候选")
    assert now < play < pack < named
    # 今天是第五个周一，树上还没有今天的行：此刻场景里只有那条未封口的，今天做过什么也是空的。
    assert "到此刻已发生：18:50 换鞋（未封口、未归类）" in text
    assert "今天做过：（无）" in text
    assert "观测空白：12:00–13:00（没读懂） ｜ 18:20–18:30（没读懂）（未封口）" in text
    assert "- 吃早饭：边际" in text
    assert "  之前：18:40 收拾球包(1)" in text  # 卡上每行带那天的次数


def test_sealed_and_unsealed_rows_share_one_timeline_in_time_order(tmp_path) -> None:
    """长链未封口、短链已封口时未封口的会更早：合成一条线按时刻排，不是先封口的后未封口的。"""

    ground = ground_for(tmp_path)
    ground.record(NOW, "关灯", 18, 50, kind="关灯")
    unsealed = (
        UnsealedRow(name="看电视", kind_token="看电视", started_at=at(NOW, 18, 40), last_observed_at=at(NOW, 19, 0), summary="看"),
    )
    text = render_pack(ground.pack(at(NOW, 19, 5), unsealed=unsealed))
    assert "到此刻已发生：18:40 看电视（未封口） ｜ 18:50 关灯(1)" in text


def test_numbers_come_before_cards_in_every_candidate_section(tmp_path) -> None:
    """先候选再上下文（2026-09-16 定）：一节里四层表、发布的率在前，历史卡在后。"""

    text = render_pack(ground_for(tmp_path).pack(at(NOW, 19, 5)))
    section = text[text.index("## 打球") : text.index("## 收拾球包")]
    assert section.index("| 本槽 |") < section.index("发布的率：") < section.index("### 历史")


def test_cards_are_numbered_per_candidate_from_one(tmp_path) -> None:
    """判断者的 basis 引用的是候选自己的卡号，所以每个候选都从 #1 编；装配按同一序换回 URI。"""

    text = render_pack(ground_for(tmp_path).pack(at(NOW, 19, 5)))
    play = text[text.index("## 打球") : text.index("## 收拾球包")]
    pack = text[text.index("## 收拾球包") : text.index("## 只列名的候选")]
    assert "- #1 " in play and "- #5 " in play and "- #6 " not in play
    assert "- #1 " in pack and "- #4 " in pack and "- #5 " not in pack
