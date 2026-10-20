"""隔离区：副本可取回；同一项不覆盖；只清放弃了的。"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from habitus.infrastructure.lifecycle import Quarantine


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 19, 9, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def _quarantine(root: Path, clock: _Clock) -> Quarantine:
    return Quarantine(root, owner="behavior-reduction-session", clock=clock)


def test_put_then_payload_round_trips(tmp_path: Path) -> None:
    quarantine = _quarantine(tmp_path, _Clock())
    assert quarantine.put("judgement/abc", b"payload", reason="no readable name", meta={"lane": "session"})
    assert quarantine.payload("judgement/abc") == b"payload"
    (item,) = quarantine.items()
    assert item.item_id == "judgement/abc"
    assert item.reason == "no readable name"
    assert item.meta == {"lane": "session"}


def test_same_item_is_not_overwritten(tmp_path: Path) -> None:
    quarantine = _quarantine(tmp_path, _Clock())
    assert quarantine.put("x", b"first", reason="r")
    assert not quarantine.put("x", b"second", reason="r")
    assert quarantine.payload("x") == b"first"


def test_purge_only_removes_abandoned_items(tmp_path: Path) -> None:
    clock = _Clock()
    quarantine = _quarantine(tmp_path, clock)
    quarantine.put("kept", b"a", reason="r")
    quarantine.put("dropped", b"b", reason="r")
    assert quarantine.abandon("dropped")
    assert not quarantine.abandon("dropped")
    clock.now += timedelta(days=31)
    assert quarantine.purge(clock.now - timedelta(days=30)) == 1
    assert [item.item_id for item in quarantine.items()] == ["kept"]


def test_discard_after_replay(tmp_path: Path) -> None:
    quarantine = _quarantine(tmp_path, _Clock())
    quarantine.put("x", b"a", reason="r")
    assert quarantine.discard("x")
    assert quarantine.items() == ()
    assert not quarantine.discard("x")


def test_payload_size_is_bounded(tmp_path: Path) -> None:
    quarantine = Quarantine(tmp_path, owner="o", clock=_Clock(), max_payload_bytes=4)
    with pytest.raises(ValueError, match="size bound"):
        quarantine.put("x", b"12345", reason="r")


@pytest.mark.parametrize(("payload", "reason", "error"), [("text", "r", TypeError), (b"a", " ", ValueError)])
def test_put_validates_inputs(tmp_path: Path, payload: object, reason: str, error: type[Exception]) -> None:
    with pytest.raises(error):
        _quarantine(tmp_path, _Clock()).put("x", payload, reason=reason)  # type: ignore[arg-type]


def test_put_repairs_a_missing_record_after_a_crash(tmp_path: Path) -> None:
    quarantine = _quarantine(tmp_path, _Clock())
    quarantine.put("x", b"a", reason="r")
    (folder,) = [path for path in (tmp_path / "lifecycle" / "quarantine" / quarantine.owner).iterdir()]
    (folder / "item.json").unlink()
    assert quarantine.items() == ()
    assert not quarantine.put("x", b"a", reason="r")
    assert [item.item_id for item in quarantine.items()] == ["x"]
