"""段的量法版本：判据改了版本的那一段，带着的行清掉；其余段原样（26 号清单 3.5）。"""

from __future__ import annotations

from datetime import date

from habitus.scene.relations import LaneState, Status
from habitus.scene.relations.longspan import MEASURE_VERSIONS
from habitus.scene.relations.spans import Segment
from habitus.scene.relations.state import Relation, _drop_stale_measures
from tests.unit.scene.relation_fixtures import CODE, DISCUSS, key

NIGHT = date(2026, 10, 18)


def _relation(segment: Segment) -> Relation:
    return Relation(
        key=key(DISCUSS, CODE, segment) if segment is not Segment.CHAIN else key(DISCUSS, CODE),
        status=Status.CANDIDATE,
        upward=True,
        discovered_on=NIGHT,
    )


def test_rows_of_a_segment_with_an_older_measure_are_dropped() -> None:
    previous = LaneState(
        lane="session", night=NIGHT, family=2, relations=(_relation(Segment.RETURN), _relation(Segment.CHAIN))
    )
    upgraded = _drop_stale_measures(previous)
    assert [relation.key.segment for relation in upgraded.relations] == [Segment.CHAIN]
    assert upgraded.measure_versions[Segment.RETURN.value] == MEASURE_VERSIONS[Segment.RETURN.value]


def test_current_versions_keep_everything() -> None:
    previous = LaneState(
        lane="session",
        night=NIGHT,
        family=2,
        relations=(_relation(Segment.RETURN),),
        measure_versions=dict(MEASURE_VERSIONS),
    )
    assert _drop_stale_measures(previous) is previous
