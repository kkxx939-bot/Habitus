"""常态、情境、夜批测试共用的材料：不经映射器直接造概念命中记录。"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from habitus.behavior.kinds.ids import Lane, pending_token
from habitus.behavior.model import BehaviorAddress
from habitus.behavior.uri import BehaviorURI
from habitus.scene.concepts import ConceptKind, ConceptSet
from habitus.scene.occurrences import ConceptHit, ConceptHits
from tests.unit.kind_ids import kind_id
from tests.unit.scene.concept_fixtures import ALL_CONCEPTS, LANE, base, cid
from tests.unit.scene.fixtures import at

NOW = datetime(2026, 8, 18, 3, 0, tzinfo=UTC)
MAPPER = "scene_concept_mapper_prompt_v2+schemaabc+emb:fake+llm:fake+concepts:0123"
CONCEPTS = ConceptSet((*ALL_CONCEPTS, base("起床", "起床离开床"), base("咖啡", "喝了一杯咖啡")))


def uri_for(day: date, name: str, hour: int, minute: int) -> str:
    return str(BehaviorURI.from_address(BehaviorAddress.occurrence(day, name, at(day, hour, minute))))


def record(
    day: date,
    name: str,
    hour: int,
    minute: int,
    *hits: ConceptHit | str,
    situations: tuple[ConceptHit | str, ...] = (),
    checked: tuple[str, ...] = (),
    kind: str | None = None,
    lasts_minutes: int = 10,
    pending: bool = False,
) -> ConceptHits:
    """测试里用类名写：命中里的基础概念（类名）就是这条的类，读成 ``kind_token``、不存成命中；没给就用 ``kind`` 或行为名。

    ``checked`` 是这条记录判过（但不一定在场）的情境（裁定八 ①）；在场的自动算判过。``pending`` = 这条在「待定」里。
    """

    behaviour = [_hit(item) for item in hits]
    bases = [hit for hit in behaviour if hit.identity in CONCEPTS and CONCEPTS[hit.identity].kind is ConceptKind.BASE]
    token = pending_token(Lane.SESSION) if pending else (bases[0].identity if bases else kind_id(kind or name))
    return ConceptHits(
        occurrence_uri=uri_for(day, name, hour, minute),
        kind_token=token,
        last_observed_at=at(day, hour, minute) + timedelta(minutes=lasts_minutes),
        hits=tuple(hit for hit in behaviour if hit not in bases),
        situation_hits=tuple(_hit(item) for item in situations),
        situations_checked=checked,
        unresolved={},
        baseline_snapshot={},
        mapper=MAPPER,
        mapped_at=NOW,
        lane=LANE,
        classified=not pending,
    )


def _hit(item: ConceptHit | str) -> ConceptHit:
    hit = item if isinstance(item, ConceptHit) else ConceptHit(item)
    return ConceptHit(cid(hit.concept), hit.grade)
