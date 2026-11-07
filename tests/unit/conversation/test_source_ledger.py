"""生命周期账的三样小存储：标记、还没终态、地址水位。"""

from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path

import pytest

from habitus.conversation.source import ConversationSourceError, SourceAddress
from habitus.infrastructure.lifecycle import ParticipantReport
from habitus.runtime.lifecycle_sources import ConversationSourceParticipant
from tests.platform_helpers import long_path
from tests.unit.conversation.source_v2_helpers import source, source_ledger
from tests.unit.conversation.test_source_release import _deliver, _setup

ADDRESS = SourceAddress("conversation-a", date(2026, 8, 8))


def test_watermark_only_moves_forward(tmp_path: Path) -> None:
    watermarks = source_ledger(tmp_path).watermarks
    assert watermarks.released_through(ADDRESS) is None
    watermarks.advance(ADDRESS, 5)
    watermarks.advance(ADDRESS, 5)
    assert watermarks.covers(ADDRESS, 5)
    assert not watermarks.covers(ADDRESS, 6)
    with pytest.raises(ConversationSourceError, match="backwards"):
        watermarks.advance(ADDRESS, 4)


@pytest.mark.parametrize("bad", [True, "5", 5.0, None, -1])
def test_a_watermark_read_back_with_a_non_integer_value_is_invalid(tmp_path: Path, bad: object) -> None:
    """盘上水位不是非负整数（``true``、字符串、浮点、空、负数）：读回时就拒。"""

    watermarks = source_ledger(tmp_path).watermarks
    watermarks.advance(ADDRESS, 5)
    path = long_path(watermarks._path(ADDRESS))  # noqa: SLF001 - 直接改盘上的字节
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["released_through"] = bad
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConversationSourceError, match="source watermark value is invalid"):
        watermarks.released_through(ADDRESS)


def test_marking_is_idempotent_and_ordered(tmp_path: Path) -> None:
    ledger = source_ledger(tmp_path)
    later, earlier = source(sequence=3, delivery_seed="b"), source(sequence=0, delivery_seed="a")
    for envelope in (later, earlier, later):
        ledger.admit(envelope)
    markers = ledger.index.markers(SourceAddress.of(earlier))
    assert [marker.source_id for marker in markers] == [earlier.source_id, later.source_id]
    assert [marker.source_id for marker in ledger.earlier(later)] == [earlier.source_id]
    assert {source_id for source_id, _ in ledger.pending.listing().sources} == {earlier.source_id, later.source_id}


def test_index_rejects_unknown_entries_but_skips_temporary_files(tmp_path: Path) -> None:
    ledger = source_ledger(tmp_path)
    envelope = source()
    ledger.admit(envelope)
    directory = tmp_path / "source" / "index" / SourceAddress.of(envelope).digest
    long_path(directory / f".{next(directory.iterdir()).name}.{'0' * 32}.tmp").write_bytes(b"partial")
    assert len(ledger.index.markers(SourceAddress.of(envelope))) == 1
    (directory / "stray.txt").write_text("x")
    with pytest.raises(ConversationSourceError, match="unsupported"):
        ledger.index.markers(SourceAddress.of(envelope))


def test_address_round_trips_and_rejects_bad_values() -> None:
    assert SourceAddress.from_dict(ADDRESS.to_dict()) == ADDRESS
    assert len(ADDRESS.digest) == 32
    with pytest.raises(ConversationSourceError):
        SourceAddress.from_dict({"conversation_id": "x"})
    with pytest.raises(ConversationSourceError):
        SourceAddress("", date(2026, 8, 8))


def test_participant_releases_and_reports_capacity(tmp_path: Path) -> None:
    async def scenario() -> ParticipantReport:
        service, _memory, _behavior, releaser, ctx = _setup(tmp_path)
        for n in range(3):
            await _deliver(service, source(sequence=n, delivery_seed=f"d{n}"))
        return await ConversationSourceParticipant(releaser, batch_addresses=1_000).maintain(ctx)

    report = asyncio.run(scenario())
    assert report.released == 2
    readings = {reading.name: reading for reading in report.readings}
    assert readings["index_addresses"].count == 1
    assert readings["pending_sources"].count == 0


def test_marks_beyond_the_bounds_are_refused_at_write_time(tmp_path: Path) -> None:
    """枚举按上限读、超了就抛：写进去的不能比它多。新地址、同地址的新标记、新的"还没终态"到了上限就拒收，已有的
    重复标记照常幂等；拒收之后枚举与容量读数都还读得出来（L10.6；原先写入不查、读端一超就整支卡死）。"""

    from habitus.conversation.source import ConversationSourceCapacityError, PendingSources, SourceIndex

    index = SourceIndex(tmp_path, max_addresses=2, max_markers_per_address=2)
    pending = PendingSources(tmp_path, max_entries=3)
    first, second = source(conversation_id="a"), source(conversation_id="b")
    for envelope in (first, second):
        index.mark(envelope)
        pending.mark(envelope)
    index.mark(first)  # 重复：幂等，不算新的
    with pytest.raises(ConversationSourceCapacityError):
        index.mark(source(conversation_id="c"))
    index.mark(source(conversation_id="a", sequence=1, delivery_seed="a1"))
    with pytest.raises(ConversationSourceCapacityError):
        index.mark(source(conversation_id="a", sequence=2, delivery_seed="a2"))
    pending.mark(source(conversation_id="a", sequence=1, delivery_seed="a1"))
    with pytest.raises(ConversationSourceCapacityError):
        pending.mark(source(conversation_id="d"))

    assert len(index.address_directories()) == 2
    assert pending.count() == 3


def test_release_walks_the_addresses_in_batches_with_a_cursor(tmp_path: Path) -> None:
    """一轮最多看 ``batch_addresses`` 个地址，下一轮从游标往后接着看，到头了从头再来（L10.6）。"""

    async def scenario() -> None:
        service, _memory, _behavior, releaser, ctx = _setup(tmp_path)
        for name in ("a", "b", "c"):
            for sequence in (0, 1):
                await _deliver(
                    service, source(conversation_id=name, sequence=sequence, delivery_seed=f"{name}{sequence}")
                )
        participant = ConversationSourceParticipant(releaser, batch_addresses=2)

        first = await participant.maintain(ctx)
        second = await participant.maintain(ctx)
        third = await participant.maintain(ctx)

        assert (first.released, second.released, third.released) == (2, 1, 0)

    asyncio.run(scenario())
