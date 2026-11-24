"""上下级在递送里不互相避让（裁定 105 第 4 条）：上级和下级各凭自己的账开口——同一拍都到时窗末各说各的，下级早上说过不挡下午
的上级，三层同一拍各说各的；只有"他这回做了下级"对上级是"不知道"，这一次不说（记「已发生」，裁定 105 第 2 条）。

入口是 ``DeliveryStage.run_once``，每 15 分钟一拍；摆法见 ``delivery_ticks``：上级 P、下级 C 两本账都在提醒档，宿主活着。
原来（裁定 101 第 8 条、裁定 104 第 3 条）上级记「上下级已说」、或等下级先决定，这里的前三条在旧代码上失败。
"""

from __future__ import annotations

from pathlib import Path

from habitus.foresight import Seen
from habitus.foresight.hierarchy import KindHierarchy
from tests.unit.runtime.delivery_ticks import (
    CHILD,
    PARENT,
    book,
    child_item,
    clock,
    listed,
    made,
    parent_item,
    site,
    walk,
)

GRAND = "s-k0107"
THREE_LAYERS = KindHierarchy({GRAND: frozenset({PARENT, CHILD}), PARENT: frozenset({CHILD})})


def test_a_parent_and_its_child_closing_in_the_same_tick_are_both_said(tmp_path: Path) -> None:
    run, ledger, outlet = site(tmp_path, parent_item(), child_item(window=(clock(9, 10), clock(9, 40))))
    walk(run, clock(9), clock(9, 45))
    assert book(ledger, "p", "c") == {
        "p": [("等时窗末", "09:15"), ("待取", "09:45")],
        "c": [("等时窗末", "09:15"), ("待取", "09:45")],
    }
    assert sorted(listed(outlet)) == ["c", "p"]


def test_the_parent_does_not_wait_for_a_child_whose_window_has_not_opened(tmp_path: Path) -> None:
    run, ledger, _outlet = site(tmp_path, parent_item(), child_item())
    walk(run, clock(9), clock(11, 45))
    assert book(ledger, "p", "c") == {
        "p": [("等时窗末", "09:15"), ("待取", "09:45")],
        "c": [("等时窗末", "11:00"), ("待取", "11:30")],
    }


def test_three_layers_closing_in_the_same_tick_are_each_said(tmp_path: Path) -> None:
    grand = made("g", GRAND, (clock(9, 10), clock(9, 40)))
    child = child_item(window=(clock(9, 10), clock(9, 40)))
    run, ledger, _outlet = site(tmp_path, grand, parent_item(), child, daily_cap=3)
    walk(run, clock(9), clock(9, 45), kinds=THREE_LAYERS)
    assert {identity: reasons[-1][0] for identity, reasons in book(ledger, "g", "p", "c").items()} == {
        "g": "待取",
        "p": "待取",
        "c": "待取",
    }


def test_when_he_did_the_child_the_parent_is_not_said_this_time(tmp_path: Path) -> None:
    """他 09:20 做了下级：对上级是"不知道"，这一次不说（「已发生」）；下级自己也做了。"""

    child = child_item(window=(clock(9, 10), clock(9, 40)))
    run, ledger, outlet = site(tmp_path, parent_item(), child)
    rows = (Seen(clock(9, 20), clock(9, 25), CHILD, "occ:done.md"),)
    walk(run, clock(9), clock(9, 45), rows=rows)
    assert book(ledger, "p", "c") == {
        "p": [("等时窗末", "09:15"), ("已发生", "09:45")],
        "c": [("等时窗末", "09:15"), ("已发生", "09:45")],
    }
    assert listed(outlet) == []


def test_when_he_did_the_parent_the_child_is_still_said(tmp_path: Path) -> None:
    """守门：做了上级那件宽的事，不算做了下级（旧代码上本来就成立）。"""

    child = child_item(window=(clock(9, 10), clock(9, 40)))
    run, ledger, _outlet = site(tmp_path, child)
    rows = (Seen(clock(9, 20), clock(9, 25), PARENT, "occ:done.md"),)
    walk(run, clock(9), clock(9, 45), rows=rows)
    assert book(ledger, "c")["c"][-1][0] == "待取"
