"""映射那两个参数的生产者：情境四族从真材料里算出来、常态按天一份、开放承诺按"那天的起点"算。"""

from __future__ import annotations

from datetime import timedelta

from habitus.runtime.scene_situations import SceneDayContext
from habitus.scene.concepts import ConceptRole, ConceptSet
from habitus.scene.concepts.situation import SituationBasis as B
from habitus.scene.concepts.situation import SituationRule
from habitus.scene.ledger import LedgerConfig, LedgerStore, open_claims_for_day
from habitus.scene.occurrences import ConceptHit, ConceptHitStore
from habitus.scene.views.intentions import open_consequents_at
from tests.unit.scene.concept_fixtures import BEDTIME_KEY, concept
from tests.unit.scene.fixtures import CST, DAY1, SUBJECT, Site, at, publish
from tests.unit.scene.ledger_fixtures import CONCEPTS, MAPPER, NOW, TableOpportunities, hypothesis, record

SATURDAY = DAY1  # 2026-08-15，周六
FRIDAY = SATURDAY - timedelta(days=1)
BOOKED = concept("约了球还没打", "约球的承诺还没结", role=ConceptRole.STATE, situation=SituationRule(B.OPEN_CLAIM, concept="打球"))
LAST_NIGHT = concept("昨晚晚睡", "昨天的就寝命中了晚睡", role=ConceptRole.DERIVED, situation=SituationRule(B.YESTERDAY, concept="晚睡"))
AT_OFFICE = concept("在公司", "地点是公司", role=ConceptRole.OBJECT, situation=SituationRule(B.PLACE, value="公司"))
CONCEPT_SET = ConceptSet((*CONCEPTS.values(), BOOKED, LAST_NIGHT, AT_OFFICE))
BOOKING_TO_BALL = hypothesis("约球", consequent="打球", direction="up", type_prior="promoting", expected_at=None)
CONFIG = LedgerConfig(snapshot_opportunities=16, censor_after=3)


def test_the_two_callables_are_built_from_real_material(tmp_path) -> None:
    """一天一装：日历那句话、这条 occurrence 的地点、昨天的命中、那天起点的开放承诺、以及常态表。"""

    site = Site(tmp_path, now=at(SATURDAY, 23, 0))
    hits = ConceptHitStore(tmp_path / "scene")
    ledger = LedgerStore(tmp_path / "scene")
    # 前几天：三条就寝（够常态的三个样本），其中昨天那条命中「晚睡」；再加一条约球开出一条无节律承诺。
    for days, hour, minute in ((1, 23, 40), (2, 0, 20), (3, 23, 50)):
        day = SATURDAY - timedelta(days=days)
        extra = (ConceptHit("晚睡", "重"),) if days == 1 else ()
        hits.write(record(day, "睡觉", hour, minute, ConceptHit("就寝"), *extra))
        hits.complete_day(day, records=1, completed_at=at(day, 23, 59), mapper=MAPPER)
    hits.write(record(FRIDAY, "约了球", 12, 0, ConceptHit("约球")))
    open_claims_for_day(
        FRIDAY,
        hypotheses=(BOOKING_TO_BALL,),
        concepts=CONCEPT_SET,
        hits=hits,
        ledger=ledger,
        opportunities=TableOpportunities(),
        now=NOW,
        config=CONFIG,
    )
    assert open_consequents_at(ledger, {BOOKING_TO_BALL.identity: BOOKING_TO_BALL}, moment=at(SATURDAY, 0, 0)) == frozenset({"打球"})

    class Calendar:
        def describe(self, day):
            return "补班日，按周一上班" if day == SATURDAY else None

    context = SceneDayContext(
        SATURDAY,
        concepts=CONCEPT_SET,
        hits=hits,
        calendar=Calendar(),
        timezone=CST,
        ledger=ledger,
        hypotheses={BOOKING_TO_BALL.identity: BOOKING_TO_BALL},
        subject=SUBJECT,
    )
    uri = publish(site.behavior_tree, SATURDAY, "写代码", 10, 0, place="公司")
    document = site.behavior_tree.read(_address(uri))
    assert {hit.concept for hit in context.situation_for(document)} == {"周末", "昨晚晚睡", "约了球还没打", "在公司"}
    # 常态：三条就寝的环形中位数（23:40 / 00:20 / 23:50），按天一份、与是哪条 occurrence 无关
    assert context.baseline_for(document) == {BEDTIME_KEY: "23:50"}
    assert context.baselines.samples[BEDTIME_KEY] == 3


def test_without_a_ledger_no_open_claim_situation_holds(tmp_path) -> None:
    """账没接上时"约了球还没打"就是算不出，不当成立——不知道不等于成立。"""

    hits = ConceptHitStore(tmp_path / "scene")
    context = SceneDayContext(SATURDAY, concepts=CONCEPT_SET, hits=hits, calendar=_Silent(), timezone=CST)
    site = Site(tmp_path, now=at(SATURDAY, 23, 0))
    uri = publish(site.behavior_tree, SATURDAY, "写代码", 10, 0)
    document = site.behavior_tree.read(_address(uri))
    assert {hit.concept for hit in context.situation_for(document)} == {"周末"}  # 只有纯日历那一条
    assert context.baseline_for(document) == {} and context.baselines.missing == (BEDTIME_KEY,)


class _Silent:
    def describe(self, day):
        return None


def _address(uri: str):
    from habitus.behavior.uri import BehaviorURI

    return BehaviorURI.parse(uri).to_address()
