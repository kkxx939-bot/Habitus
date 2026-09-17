"""规律树的读口：关联记录按 ``occurrence_uri`` 取出、只认有完成标记的日子；情形按周几分两组。"""

from __future__ import annotations

from datetime import date

import pytest

from habitus.scene.regularity.overview import Situation
from habitus.scene.views import association_glosses, situations_of
from tests.unit.scene.fixtures import DAY1, DAY2, DAY3, Site, associate, at, publish, regularity_tree

# DAY1 = 2026-08-15 周六，DAY2 周日，DAY3 周一
SATURDAY, SUNDAY, MONDAY = 5, 6, 0


def site(tmp_path) -> Site:
    return Site(tmp_path, now=at(DAY3, 23, 0))


def test_glosses_are_keyed_by_occurrence_and_carry_the_five_things(tmp_path) -> None:
    ground = site(tmp_path)
    store = regularity_tree(tmp_path)
    call = publish(ground.behavior_tree, DAY1, "与朋友通电话", 15, 50, kind="通电话")
    own = publish(ground.behavior_tree, DAY1, "打球", 16, 40)
    associate(
        store,
        own,
        kind="打球",
        context="周六下午和朋友约好后去打的",
        situation="和朋友约好之后一起去",
        causes=(call,),
        consumed=((call, "约好这周打球"),),
        left=(("球拍胶皮该换了", "买胶皮"),),
    )

    glosses = association_glosses(store, "打球", (DAY1,))

    assert set(glosses) == {own}
    gloss = glosses[own]
    assert gloss.context == "周六下午和朋友约好后去打的"
    assert gloss.situation == "和朋友约好之后一起去"
    assert gloss.causes == ((call, "与朋友通电话"),)
    assert gloss.consumed == ((call, "约好这周打球"),)
    assert gloss.left == (("球拍胶皮该换了", "买胶皮"),)


def test_only_days_with_a_completion_marker_are_read(tmp_path) -> None:
    """一天的记录是逐条写、最后打标记的：没标记的那天一条都不取，别的天照取。"""

    ground = site(tmp_path)
    store = regularity_tree(tmp_path)
    done = publish(ground.behavior_tree, DAY1, "打球", 16, 40)
    half = publish(ground.behavior_tree, DAY2, "打球", 16, 30)
    associate(store, done, kind="打球", context="约好后去打")
    associate(store, half, kind="打球", context="写了一半", complete=False)

    glosses = association_glosses(store, "打球", (DAY1, DAY2, DAY3))

    assert set(glosses) == {done}


def test_the_version_is_the_same_yardstick_the_refresher_uses(tmp_path) -> None:
    """按版本读：旧版本关联的那天在新版本眼里还没关联，读口也不能把旧记录当背景贴上去。"""

    ground = site(tmp_path)
    store = regularity_tree(tmp_path)
    own = publish(ground.behavior_tree, DAY1, "打球", 16, 40)
    associate(store, own, kind="打球", context="旧版本写的", version="old_prompt")

    assert set(association_glosses(store, "打球", (DAY1,))) == {own}
    assert association_glosses(store, "打球", (DAY1,), version="new_prompt") == {}


def test_glosses_are_empty_for_a_candidate_never_associated(tmp_path) -> None:
    store = regularity_tree(tmp_path)
    assert association_glosses(store, "打球", (DAY1,)) == {}


def test_situations_split_into_this_weekday_and_the_rest(tmp_path) -> None:
    """周三判断打球时先看周三出现过的情形；同时出现在本周几与别的周几的算本周几那一组。"""

    ground = site(tmp_path)
    store = regularity_tree(tmp_path)
    saturday = publish(ground.behavior_tree, DAY1, "打球", 16, 40)
    sunday = publish(ground.behavior_tree, DAY2, "打球", 10, 0)
    monday = publish(ground.behavior_tree, DAY3, "打球", 16, 30)
    associate(store, saturday, kind="打球", context="周六", situation="和朋友约好之后一起去")
    associate(store, sunday, kind="打球", context="周日", situation="周末上午自己去")
    associate(store, monday, kind="打球", context="周一", situation="和朋友约好之后一起去")

    here, elsewhere = situations_of(store, "打球", weekday=MONDAY)

    assert here == (Situation("和朋友约好之后一起去", (DAY1, DAY3)),)
    assert elsewhere == (Situation("周末上午自己去", (DAY2,)),)
    assert situations_of(store, "打球", weekday=SUNDAY) == (
        (Situation("周末上午自己去", (DAY2,)),),
        (Situation("和朋友约好之后一起去", (DAY1, DAY3)),),
    )


def test_a_candidate_without_an_overview_has_no_situations(tmp_path) -> None:
    store = regularity_tree(tmp_path)
    assert situations_of(store, "打球", weekday=SATURDAY) == ((), ())


@pytest.mark.parametrize("weekday", [-1, 7, True])
def test_the_weekday_must_be_a_clock_face_weekday(tmp_path, weekday: object) -> None:
    store = regularity_tree(tmp_path)
    with pytest.raises(ValueError, match="between 0 and 6"):
        situations_of(store, "打球", weekday=weekday)  # type: ignore[arg-type]


def test_the_read_side_refuses_things_that_are_not_a_regularity_tree(tmp_path) -> None:
    with pytest.raises(TypeError, match="RegularityTree"):
        association_glosses(object(), "打球", (date(2026, 8, 15),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="RegularityTree"):
        situations_of(object(), "打球", weekday=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kind_token"):
        association_glosses(regularity_tree(tmp_path), "", (date(2026, 8, 15),))
