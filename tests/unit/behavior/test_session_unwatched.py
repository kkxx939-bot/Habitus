"""会话一支的「未观测」（实时化方案第 6 节）：系统没在跑的那一段写成一条空判断（依据 ``unobserved``、两条凭据、回执照常），
身份由起止派生、重写幂等，归约认成「未观测」空白。"""

from __future__ import annotations

from pathlib import Path

import pytest

from habitus.behavior.fusion.lanes.session import SessionUnwatchedRecorder
from habitus.behavior.reduction.ledger import BehaviorReductionLedger
from habitus.behavior.reduction.payloads import UNOBSERVED_GAP_KIND, gap_kind_of
from habitus.behavior.reduction.pending import pending_judgements
from tests.unit.behavior.session_fixtures import SUBJECT, ZONE, Stores, at


def test_an_unwatched_span_is_one_empty_judgement_the_reduction_reads_as_unobserved(tmp_path: Path) -> None:
    stores = Stores(tmp_path)
    recorder = SessionUnwatchedRecorder(
        observations=stores.observations,
        judgements=stores.judgements,
        receipts=stores.receipts,
        coverage=stores.coverage,
        primary_subject=SUBJECT,
        zone=ZONE,
        clock=lambda: stores.now,
    )
    first = recorder.record(at(0), at(240))
    assert first is not None
    (payload,) = stores.judgements.list()
    assert payload["behavior"] is None and payload["status_basis"] == "unobserved"
    assert len(payload["observation_ids"]) == 2  # 这一段的开头与结尾
    assert recorder.record(at(0), at(240)) is None and len(stores.judgements.list()) == 1  # 写完过：不再写
    (gap,) = pending_judgements(stores.judgements, BehaviorReductionLedger(tmp_path / "reduction")).gaps
    assert gap_kind_of(gap) == UNOBSERVED_GAP_KIND and gap.started_at == at(0) and gap.last_observed_at == at(240)
    with pytest.raises(ValueError):
        recorder.record(at(10), at(10))
