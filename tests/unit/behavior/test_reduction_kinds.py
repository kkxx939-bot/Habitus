"""归约里的词表一步（``reduction.kinds_step``）：判断上带着选词结果的直接落；没带的一条一次选词，没问出结论的放一边
（词表方案 2026-11-14）。"""

from __future__ import annotations

import asyncio
from typing import Any

from habitus.behavior.kinds.calls import CallFailure
from habitus.behavior.kinds.ids import Lane
from habitus.behavior.reduction.kinds_step import KindStamping
from habitus.behavior.reduction.record import parse_judgement_record
from habitus.model_client import ModelContentSafetyError
from tests.unit.behavior.kinds_fixtures import NameMatchingProvider, ScriptedProvider, kind_components
from tests.unit.behavior.reduction_fixtures import at, judgement_record


def _record(seed: str, behavior: str, *, kind_token: str | None = None) -> Any:
    return parse_judgement_record(
        judgement_record(
            seed,
            behavior=behavior,
            started_at=at(0),
            last_observed_at=at(60),
            evidence_ready_at=at(70),
            observation_ids=(f"obs-{seed}",),
            source_refs=(f"src-{seed}",),
            kind_token=kind_token,
        )
    )


def _stamp(stamping: KindStamping, records: list[Any], beats: list[int]) -> Any:
    return asyncio.run(stamping.classify(records, checkpoint=lambda: beats.append(1)))


def test_a_carried_pick_lands_as_is_and_the_rest_are_picked_one_call_each() -> None:
    provider = NameMatchingProvider()
    stamping = kind_components(names=("修改代码", "调研"), provider=provider)
    beats: list[int] = []
    carried, edit, other = _record("a", "随便", kind_token="s-续话"), _record("b", "修改代码"), _record("c", "做饭")
    stamps = _stamp(stamping, [carried, edit, other], beats)
    assert stamps.tokens == {
        carried.chain_digest: "s-续话",
        edit.chain_digest: "s-k0001",
        other.chain_digest: "s-表外",
    }
    assert provider.calls == stamps.model_calls == 2 and beats == [1, 1]  # 每问一条续一次租约


def test_a_refused_pick_is_set_aside_and_a_lane_without_a_picker_never_asks() -> None:
    stamping = kind_components(names=("修改代码",), provider=ScriptedProvider([ModelContentSafetyError("blocked")]))
    record = _record("a", "修改代码")
    stamps = _stamp(stamping, [record], [])
    assert stamps.tokens == {} and stamps.set_aside == {record.chain_digest: CallFailure.REFUSED}
    unpicked = KindStamping(None, lane=Lane.SESSION)
    stamps = _stamp(unpicked, [record, _record("b", "调研", kind_token="s-k0002")], [])
    assert stamps.set_aside == {record.chain_digest: CallFailure.BUG} and stamps.model_calls == 0
