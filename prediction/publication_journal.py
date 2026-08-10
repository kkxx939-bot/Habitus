"""PredictionSample 批量发布的可恢复提交记录。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from foundation.integrity import canonical_json, canonicalize
from infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_replace_bytes,
    ensure_real_directory,
    read_regular_bytes,
)

_LOWER_HEX = frozenset("0123456789abcdef")


class PredictionPublicationState(str, Enum):
    PREPARED = "prepared"
    COMMITTED = "committed"


@dataclass(frozen=True)
class PredictionPublicationRecord:
    job_id: str
    documents: tuple[tuple[str, str], ...]
    state: PredictionPublicationState


class PredictionPublicationJournalError(RuntimeError):
    """批量发布记录损坏或与重放内容不一致。"""


class PredictionPublicationJournal:
    """先记录完整预期集合；部分发布后可用同一批内容幂等补齐。"""

    def __init__(self, root: str | Path) -> None:
        requested = Path(root).expanduser().absolute()
        if requested.is_symlink():
            raise PredictionPublicationJournalError("prediction publication journal root cannot be a symbolic link")
        self.root = requested.resolve(strict=False)

    def prepare(self, record: PredictionPublicationRecord) -> PredictionPublicationRecord:
        if record.state is not PredictionPublicationState.PREPARED:
            raise ValueError("new prediction publication must start PREPARED")
        payload = self._encode(record)
        self._initialize()
        try:
            atomic_create_bytes(self._path(record.job_id), payload, artifact_root=self.root)
        except ImmutableArtifactConflictError as exc:
            current = self.read(record.job_id)
            if current.documents != record.documents:
                raise PredictionPublicationJournalError("prediction publication job identity conflicts") from exc
            return current
        return self.read(record.job_id)

    def commit(self, job_id: str) -> PredictionPublicationRecord:
        current = self.read(job_id)
        committed = PredictionPublicationRecord(current.job_id, current.documents, PredictionPublicationState.COMMITTED)
        try:
            atomic_replace_bytes(self._path(job_id), self._encode(committed), artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise PredictionPublicationJournalError("prediction publication job cannot be committed safely") from exc
        return self.read(job_id)

    def read(self, job_id: str) -> PredictionPublicationRecord:
        try:
            raw = read_regular_bytes(self._path(job_id), artifact_root=self.root, max_bytes=2_000_000)
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=self._unique_object)
        except (DurablePathIntegrityError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PredictionPublicationJournalError("prediction publication journal cannot be read safely") from exc
        return self._decode(value, expected_job_id=job_id)

    def _initialize(self) -> None:
        try:
            ensure_real_directory(self.root, artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise PredictionPublicationJournalError("prediction publication journal cannot be initialized") from exc

    def _path(self, job_id: str) -> Path:
        if not isinstance(job_id, str) or len(job_id) != 64 or any(character not in "0123456789abcdef" for character in job_id):
            raise ValueError("prediction publication job ID must be a lowercase SHA-256 digest")
        return self.root / f"{job_id}.json"

    @staticmethod
    def _encode(record: PredictionPublicationRecord) -> bytes:
        value = canonicalize(
            {
                "schema": "m2bos-prediction-publication-v1",
                "job_id": record.job_id,
                "documents": [{"uri": uri, "digest": digest} for uri, digest in record.documents],
                "state": record.state.value,
            }
        )
        return (canonical_json(value) + "\n").encode("utf-8")

    @staticmethod
    def _decode(value: Any, *, expected_job_id: str) -> PredictionPublicationRecord:
        if not isinstance(value, Mapping) or set(value) != {"schema", "job_id", "documents", "state"}:
            raise PredictionPublicationJournalError("prediction publication journal has an invalid shape")
        if value["schema"] != "m2bos-prediction-publication-v1" or value["job_id"] != expected_job_id:
            raise PredictionPublicationJournalError("prediction publication journal identity is invalid")
        documents_value = value["documents"]
        if not isinstance(documents_value, list):
            raise PredictionPublicationJournalError("prediction publication documents must be an array")
        documents: list[tuple[str, str]] = []
        for item in documents_value:
            if not isinstance(item, Mapping) or set(item) != {"uri", "digest"}:
                raise PredictionPublicationJournalError("prediction publication document entry is invalid")
            uri, digest = item["uri"], item["digest"]
            if (
                not isinstance(uri, str)
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in _LOWER_HEX for character in digest)
            ):
                raise PredictionPublicationJournalError("prediction publication document identity is invalid")
            documents.append((uri, digest))
        if tuple(documents) != tuple(sorted(documents)) or len(documents) != len({uri for uri, _ in documents}):
            raise PredictionPublicationJournalError("prediction publication documents must be unique and sorted")
        try:
            state = PredictionPublicationState(value["state"])
        except (TypeError, ValueError) as exc:
            raise PredictionPublicationJournalError("prediction publication state is invalid") from exc
        return PredictionPublicationRecord(expected_job_id, tuple(documents), state)

    @staticmethod
    def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PredictionPublicationJournalError("prediction publication journal has duplicate keys")
            result[key] = value
        return result


__all__ = [
    "PredictionPublicationJournal",
    "PredictionPublicationJournalError",
    "PredictionPublicationRecord",
    "PredictionPublicationState",
]
