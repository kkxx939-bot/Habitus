"""概念层测试共用的材料：一小组自洽的概念（裁定 20 的四种）。

- 基础概念：词表一个类一个，身份 = 类编号（测试里用 ``kind_id(类名)`` 配一个稳定编号），显示名 = 类名。
  测试里按类名写、用 ``cid`` 换成身份：``cid("早餐")`` = 早餐那个类的编号；细分 / 汇总 / 情境概念的身份就是名字。
- 细分概念：「晚睡」= 就寝 + 比近期常态晚两小时以上（数值，算法判）；「起床就吃」= 早餐 + 起床后半小时内
  （要当天时间线，模型判）；「返工」= 修改代码 + 改的是前两天刚改过的同一处（要近几天同类记录，模型判）。
- 汇总概念：「运动」= 打球 + 跑步。
- 情境概念：「出差中」（没有情境说明，永不命中）、「周末」（日历算得出）。
"""

from __future__ import annotations

from datetime import UTC, datetime

from habitus.scene.concepts import (
    ConceptDefinition,
    ConceptGrade,
    ConceptKind,
    ConceptOrigin,
    ConceptRole,
    ConceptSet,
    ConceptSource,
    ContextScope,
    GradeMeasure,
    MechanicalRule,
)
from habitus.scene.concepts.situation import SituationBasis, SituationRule
from tests.unit.kind_ids import kind_id

CREATED = datetime(2026, 9, 26, 0, 0, tzinfo=UTC)
VOCABULARY = ConceptSource(ConceptOrigin.VOCABULARY)
AUTHORED = ConceptSource(ConceptOrigin.AUTHOR)
LANE = "session"
#: 测试里出现过的类名；``cid`` 只把这些换成编号，别的名字（细分 / 汇总 / 情境概念）原样是身份。
CLASS_TITLES: set[str] = set()


def cid(name: str) -> str:
    """测试里的概念名 → 概念身份：类名换成那个类的编号，其余原样。"""

    return kind_id(name) if name in CLASS_TITLES else name


def base(title: str, definition: str | None = None, *, retired: bool = False) -> ConceptDefinition:
    """一个类的基础概念（同步词表自动生成的那种）。"""

    CLASS_TITLES.add(title)
    class_id = kind_id(title)
    return ConceptDefinition(
        name=class_id,
        definition=definition or f"{title}这件事",
        role=ConceptRole.BEHAVIOR,
        source=VOCABULARY,
        created_at=CREATED,
        kind=ConceptKind.BASE,
        classes=(class_id,),
        lane=LANE,
        title=title,
        retired=retired,
    )


def refinement(
    name: str,
    definition: str,
    of: str,
    *,
    rule: MechanicalRule | None = None,
    grades: tuple[ConceptGrade, ...] = (),
    context: ContextScope = ContextScope.OCCURRENCE,
    baseline_keys: tuple[str, ...] = (),
) -> ConceptDefinition:
    """细分概念：源类（类名）+ 区别。"""

    CLASS_TITLES.add(of)
    return ConceptDefinition(
        name=name,
        definition=definition,
        role=ConceptRole.BEHAVIOR,
        source=AUTHORED,
        created_at=CREATED,
        kind=ConceptKind.REFINEMENT,
        classes=(kind_id(of),),
        lane=LANE,
        grades=grades,
        rule=rule,
        context=context,
        baseline_keys=baseline_keys,
    )


def group(name: str, definition: str, *members: str) -> ConceptDefinition:
    """汇总概念：成员类（类名）。"""

    CLASS_TITLES.update(members)
    return ConceptDefinition(
        name=name,
        definition=definition,
        role=ConceptRole.BEHAVIOR,
        source=AUTHORED,
        created_at=CREATED,
        kind=ConceptKind.GROUP,
        classes=tuple(kind_id(member) for member in members),
        lane=LANE,
    )


def situation(
    name: str,
    definition: str,
    *,
    role: ConceptRole = ConceptRole.STATE,
    rule: SituationRule | None = None,
    grades: tuple[ConceptGrade, ...] = (),
    baseline_keys: tuple[str, ...] = (),
) -> ConceptDefinition:
    return ConceptDefinition(
        name=name,
        definition=definition,
        role=role,
        source=AUTHORED,
        created_at=CREATED,
        grades=grades,
        baseline_keys=baseline_keys,
        situation=rule,
    )


BEDTIME = base("就寝", "上床睡觉")
BEDTIME_KEY = f"{BEDTIME.name}:usual_start:recent"  # 键的第三段是窗；判据一律比近期常态（2026-09-27 裁定）
BEDTIME_KEY_ALL = f"{BEDTIME.name}:usual_start:all"  # 历来常态只用来读漂移，不能当判据

#: 晚睡 = 入睡比常态晚两小时以上；档按偏移分：轻 2–4 小时，重 4–12 小时。
LATE_RULE = MechanicalRule(GradeMeasure.START_MINUTE_OF_DAY, lower=120, upper=None, relative_to=BEDTIME_KEY)
LATE_GRADES = (
    ConceptGrade("轻", GradeMeasure.START_MINUTE_OF_DAY, 120, 240, relative=True),
    ConceptGrade("重", GradeMeasure.START_MINUTE_OF_DAY, 240, 720, relative=True),
)

SLEEP_LATE = refinement("晚睡", "开始时刻比近期常态晚两小时以上", "就寝", rule=LATE_RULE, grades=LATE_GRADES)
BREAKFAST = base("早餐", "早上的第一顿饭")
EAT_ON_WAKING = refinement("起床就吃", "起床后半小时内就吃了", "早餐", context=ContextScope.DAY)
CODING = base("修改代码", "改动代码文件")
REWORK = refinement("返工", "改的是前两天刚改过的同一处", "修改代码", context=ContextScope.RECENT)
BALL = base("打球", "参与一场球类运动")
RUNNING = base("跑步", "跑步锻炼")
EXERCISE = group("运动", "以活动身体为目的的行为", "打球", "跑步")
#: 「出差中」**故意不带情境说明**：这类状态要等事实门（``scene/facts.py``）接上真实数据源才算得出，
#: 在那之前它进得了概念集、也能被假设引用，只是永远不命中、分不出层。
TRAVELLING = situation("出差中", "人在常住地之外过夜")
#: 「周末」是纯日历的，算法算得出：名义上的周六/周日。
WEEKEND = situation(
    "周末",
    "当地日历上的周六或周日",
    role=ConceptRole.DAY_TYPE,
    rule=SituationRule(SituationBasis.WEEKDAYS, weekdays=(5, 6)),
)

ALL_CONCEPTS = (
    BEDTIME,
    SLEEP_LATE,
    BREAKFAST,
    EAT_ON_WAKING,
    CODING,
    REWORK,
    BALL,
    RUNNING,
    EXERCISE,
    TRAVELLING,
    WEEKEND,
)


def concept_set() -> ConceptSet:
    return ConceptSet(ALL_CONCEPTS)


__all__ = [
    "ALL_CONCEPTS",
    "AUTHORED",
    "BALL",
    "BEDTIME",
    "BEDTIME_KEY",
    "BEDTIME_KEY_ALL",
    "BREAKFAST",
    "CLASS_TITLES",
    "CODING",
    "CREATED",
    "EAT_ON_WAKING",
    "EXERCISE",
    "LANE",
    "LATE_GRADES",
    "LATE_RULE",
    "REWORK",
    "RUNNING",
    "SLEEP_LATE",
    "TRAVELLING",
    "VOCABULARY",
    "WEEKEND",
    "base",
    "cid",
    "concept_set",
    "group",
    "refinement",
    "situation",
]
