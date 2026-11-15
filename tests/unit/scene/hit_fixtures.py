"""情境、换代测试共用的材料：不经封口直接造命中记录。"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from habitus.behavior.kinds.ids import Lane, pending_token
from habitus.behavior.model import BehaviorAddress
from habitus.behavior.uri import BehaviorURI
from habitus.scene.concepts import ConceptSet
from habitus.scene.occurrences import ConceptHit, ConceptHits
from habitus.scene.occurrences.daily import HITS_VERSION
from tests.unit.kind_ids import kind_id
from tests.unit.scene.concept_fixtures import ALL_CONCEPTS, LANE, base, cid
from tests.unit.scene.fixtures import at

NOW = datetime(2026, 8, 18, 3, 0, tzinfo=UTC)
CONCEPTS = ConceptSet((*ALL_CONCEPTS, base("起床", "起床离开床"), base("咖啡", "喝了一杯咖啡")))


def uri_for(day: date, name: str, hour: int, minute: int) -> str:
    return str(BehaviorURI.from_address(BehaviorAddress.occurrence(day, name, at(day, hour, minute))))


def record(
    day: date,
    name: str,
    hour: int,
    minute: int,
    kind_title: str | None = None,
    *,
    situations: tuple[ConceptHit | str, ...] = (),
    checked: tuple[str, ...] = (),
    kind: str | None = None,
    lasts_minutes: int = 10,
    pending: bool = False,
) -> ConceptHits:
    """测试里用类名写：``kind_title``（类名）就是这条的类，读成 ``kind_token``；没给就用 ``kind`` 或行为名。

    ``checked`` 是这条记录判过（但不一定在场）的情境（裁定八 ①）；在场的自动算判过。``pending`` = 这条在「待定」里。
    """

    token = pending_token(Lane.SESSION) if pending else kind_id(kind_title or kind or name)
    return ConceptHits(
        occurrence_uri=uri_for(day, name, hour, minute),
        kind_token=token,
        last_observed_at=at(day, hour, minute) + timedelta(minutes=lasts_minutes),
        situation_hits=tuple(item if isinstance(item, ConceptHit) else ConceptHit(cid(item)) for item in situations),
        situations_checked=checked,
        hits_version=HITS_VERSION,
        sealed_at=NOW,
        lane=LANE,
        classified=not pending,
    )
