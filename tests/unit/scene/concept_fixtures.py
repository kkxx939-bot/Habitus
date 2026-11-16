"""概念层测试共用的材料：一小组自洽的概念（词表类的行为概念 + 日历情境）。

- 行为概念：词表一个类一个，身份 = 类编号（测试里用 ``kind_id(类名)`` 配一个稳定编号），显示名 = 类名。
  测试里按类名写、用 ``cid`` 换成身份：``cid("早餐")`` = 早餐那个类的编号；日历情境的身份就是名字。
- 日历情境：「周末」（日历.周几 ∈ {周六, 周日}）。
"""

from __future__ import annotations

from datetime import UTC, datetime

from habitus.scene.concepts import (
    ConceptDefinition,
    ConceptOrigin,
    ConceptRole,
    ConceptSet,
    ConceptSource,
)
from habitus.scene.concepts.situation import SituationRule
from habitus.series import EventSeries
from tests.unit.kind_ids import kind_id

CREATED = datetime(2026, 9, 26, 0, 0, tzinfo=UTC)
VOCABULARY = ConceptSource(ConceptOrigin.VOCABULARY)
LANE = "session"
#: 测试里出现过的类名；``cid`` 只把这些换成编号，别的名字（日历情境）原样是身份。
CLASS_TITLES: set[str] = set()


def cid(name: str) -> str:
    """测试里的概念名 → 概念身份：类名换成那个类的编号，其余原样。"""

    return kind_id(name) if name in CLASS_TITLES else name


def base(title: str, definition: str | None = None, *, retired: bool = False) -> ConceptDefinition:
    """一个类的行为概念（同步词表自动生成的那种）。"""

    CLASS_TITLES.add(title)
    class_id = kind_id(title)
    return ConceptDefinition(
        name=class_id,
        definition=definition or f"{title}这件事",
        role=ConceptRole.BEHAVIOR,
        source=VOCABULARY,
        created_at=CREATED,
        classes=(class_id,),
        lane=LANE,
        title=title,
        retired=retired,
    )


def calendar(name: str, key: str, *values: str) -> ConceptDefinition:
    """日历情境（裁定 59）：外部条件的日历键取某几个值，机械生成、不由模型写。"""

    return ConceptDefinition(
        name=name,
        definition=f"当地日历上的{name}",
        role=ConceptRole.DAY_TYPE,
        source=ConceptSource(ConceptOrigin.CALENDAR),
        created_at=CREATED,
        situation=SituationRule(value=key, values=values),
    )


BEDTIME = base("就寝", "上床睡觉")
BREAKFAST = base("早餐", "早上的第一顿饭")
CODING = base("修改代码", "改动代码文件")
BALL = base("打球", "参与一场球类运动")
RUNNING = base("跑步", "跑步锻炼")
#: 「周末」是纯日历的，算法算得出：周六/周日。
WEEKEND = calendar("周末", "日历.周几", "周六", "周日")

ALL_CONCEPTS = (BEDTIME, BREAKFAST, CODING, BALL, RUNNING, WEEKEND)


def concept_set() -> ConceptSet:
    return ConceptSet(ALL_CONCEPTS)


def every_kind(series: EventSeries) -> ConceptSet:
    """序列里每个已转正的编号各一个在用的行为概念（名字就是编号）：只关心复现、不关心词表的测试拿它当"全都在用"。"""

    lanes = {record.kind_token: record.lane for record in series.records if record.classified}
    return ConceptSet(
        ConceptDefinition(
            name=token,
            definition=f"{token}这件事",
            role=ConceptRole.BEHAVIOR,
            source=VOCABULARY,
            created_at=CREATED,
            classes=(token,),
            lane=lane,
            title=token,
        )
        for token, lane in sorted(lanes.items())
    )


__all__ = [
    "ALL_CONCEPTS",
    "BALL",
    "BEDTIME",
    "BREAKFAST",
    "CLASS_TITLES",
    "CODING",
    "CREATED",
    "LANE",
    "RUNNING",
    "VOCABULARY",
    "WEEKEND",
    "base",
    "calendar",
    "cid",
    "concept_set",
    "every_kind",
]
