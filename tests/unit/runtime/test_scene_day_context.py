"""映射那两个参数的生产者：情境三族从真材料里算出来、派生情境按事件锚定并能看到今天凌晨那条、常态按天一份。"""

from __future__ import annotations

from datetime import timedelta

from habitus.runtime.scene_situations import SceneDayContext
from habitus.scene.concepts import ConceptRole, ConceptSet
from habitus.scene.concepts.situation import SituationBasis as B
from habitus.scene.concepts.situation import SituationRule
from habitus.scene.occurrences import ConceptHit, ConceptHitStore
from tests.unit.scene.concept_fixtures import BEDTIME_KEY, situation
from tests.unit.scene.fixtures import CST, DAY1, SUBJECT, Site, at, publish
from tests.unit.scene.hit_fixtures import CONCEPTS, MAPPER, record

SATURDAY = DAY1  # 2026-08-15，周六
LAST_NIGHT = situation(
    "昨晚晚睡",
    "这条行为之前 24 小时内的就寝命中了晚睡",
    role=ConceptRole.DERIVED,
    rule=SituationRule(B.YESTERDAY, concept="晚睡"),
)
AT_OFFICE = situation("在公司", "地点是公司", role=ConceptRole.OBJECT, rule=SituationRule(B.PLACE, value="公司"))
CONCEPT_SET = ConceptSet((*CONCEPTS.values(), LAST_NIGHT, AT_OFFICE))


def test_the_two_callables_are_built_from_real_material(tmp_path) -> None:
    """一天一装：日历那句话、这条 occurrence 的地点、之前 24 小时内的命中（含今天凌晨那条）、以及常态表。"""

    site = Site(tmp_path, now=at(SATURDAY, 23, 0))
    hits = ConceptHitStore(tmp_path / "scene")
    # 前几天：三条就寝（够常态的三个样本），都没晚睡；**今天**凌晨 02:10 那条才是晚睡——它落在今天的目录里，
    # 按日历日取"昨天"会漏掉它（评审 A-8 / B-11 / C-8）。
    for days, hour, minute in ((1, 23, 40), (2, 0, 20), (3, 23, 50)):
        day = SATURDAY - timedelta(days=days)
        hits.write(record(day, "睡觉", hour, minute, ConceptHit("就寝")))
        hits.complete_day(day, records=1, completed_at=at(day, 23, 59), mapper=MAPPER)
    hits.write(record(SATURDAY, "睡觉", 2, 10, ConceptHit("就寝"), ConceptHit("晚睡", "重")))

    class Calendar:
        def describe(self, day):
            return "补班日，按周一上班" if day == SATURDAY else None

    context = SceneDayContext(
        SATURDAY, concepts=CONCEPT_SET, hits=hits, calendar=Calendar(), timezone=CST, subject=SUBJECT
    )
    uri = publish(site.behavior_tree, SATURDAY, "写代码", 10, 0, place="公司")
    document = site.behavior_tree.read(_address(uri))
    outcome = context.situation_for(document)
    assert {hit.concept for hit in outcome.hits} == {"周末", "昨晚晚睡", "在公司"}
    assert set(outcome.checked) == {"周末", "昨晚晚睡", "在公司"}
    # 凌晨那条自己不算"之前"：01:00 的行为看不到 02:10 的晚睡
    early = site.behavior_tree.read(_address(publish(site.behavior_tree, SATURDAY, "看手机", 1, 0)))
    assert "昨晚晚睡" not in {hit.concept for hit in context.situation_for(early).hits}
    # 常态：三条就寝的环形中位数（23:40 / 00:20 / 23:50），不含今天，按天一份、与是哪条 occurrence 无关
    assert context.baseline_for(document) == {BEDTIME_KEY: "23:50"}
    assert context.baselines.samples[BEDTIME_KEY] == 3


def test_without_history_only_the_calendar_situations_hold(tmp_path) -> None:
    """什么都没映射过时派生情境判不了：不当成立，也不当"判过不成立"——不知道不等于成立。"""

    hits = ConceptHitStore(tmp_path / "scene")
    context = SceneDayContext(SATURDAY, concepts=CONCEPT_SET, hits=hits, calendar=_Silent(), timezone=CST)
    site = Site(tmp_path, now=at(SATURDAY, 23, 0))
    uri = publish(site.behavior_tree, SATURDAY, "写代码", 10, 0)
    document = site.behavior_tree.read(_address(uri))
    outcome = context.situation_for(document)
    assert {hit.concept for hit in outcome.hits} == {"周末"}  # 只有纯日历那一条
    assert "昨晚晚睡" in outcome.checked  # 历史读了（是空的），所以判过、不成立
    assert context.baseline_for(document) == {} and context.baselines.missing == (BEDTIME_KEY,)


class _Silent:
    def describe(self, day):
        return None


def _address(uri: str):
    from habitus.behavior.uri import BehaviorURI

    return BehaviorURI.parse(uri).to_address()
