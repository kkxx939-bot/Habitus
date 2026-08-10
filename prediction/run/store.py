"""不可变 PredictionRun 的安全持久化账本。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, NoReturn

from foundation.integrity import canonical_json
from infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    ensure_real_directory,
    read_regular_bytes,
)
from prediction.model import prediction_sample_id
from prediction.run.model import PredictionRun


class PredictionRunStoreError(RuntimeError):
    """PredictionRun 账本路径、内容或不可变身份无效。"""


class PredictionRunStore:
    """按预测发生日保存运行记录；相同内容重放幂等。"""

    def __init__(self, root: str | Path, *, max_record_bytes: int = 2_000_000) -> None:
        requested = Path(root).expanduser().absolute()
        if requested.is_symlink():
            raise PredictionRunStoreError("prediction run root cannot be a symbolic link")
        if isinstance(max_record_bytes, bool) or not isinstance(max_record_bytes, int) or max_record_bytes <= 0:
            raise ValueError("max_record_bytes must be a positive integer")
        self.root = requested.resolve(strict=False)
        self.max_record_bytes = max_record_bytes

    def create(self, run: PredictionRun) -> PredictionRun:
        if not isinstance(run, PredictionRun):
            raise TypeError("run must be PredictionRun")
        encoded = (canonical_json(run.to_dict()) + "\n").encode("utf-8")
        if len(encoded) > self.max_record_bytes:
            raise PredictionRunStoreError("prediction run exceeds its configured size limit")
        path = self.path_for(run.predicted_at.date(), run.run_id)
        try:
            ensure_real_directory(path.parent, artifact_root=self.root)
            atomic_create_bytes(path, encoded, artifact_root=self.root)
        except (DurablePathIntegrityError, ImmutableArtifactConflictError) as exc:
            raise PredictionRunStoreError("prediction run cannot be created safely") from exc
        return self.read(run.predicted_at.date(), run.run_id)

    def read(self, predicted_on: date, run_id: str) -> PredictionRun:
        path = self.path_for(predicted_on, run_id)
        try:
            encoded = read_regular_bytes(path, artifact_root=self.root, max_bytes=self.max_record_bytes)
            value = json.loads(
                encoded.decode("utf-8"),
                object_pairs_hook=self._unique_object,
                parse_constant=self._reject_constant,
            )
            run = PredictionRun.from_dict(value)
        except (
            OSError,
            DurablePathIntegrityError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            raise PredictionRunStoreError("prediction run cannot be read safely") from exc
        if run.run_id != run_id or run.predicted_at.date() != predicted_on:
            raise PredictionRunStoreError("prediction run content does not match its physical address")
        if (canonical_json(run.to_dict()) + "\n").encode("utf-8") != encoded:
            raise PredictionRunStoreError("prediction run is not canonically serialized")
        return run

    def path_for(self, predicted_on: date, run_id: str) -> Path:
        if not isinstance(predicted_on, date):
            raise TypeError("predicted_on must be a date")
        normalized_id = prediction_sample_id(run_id)
        return self.root / f"{predicted_on.year:04d}" / f"{predicted_on.month:02d}" / f"{predicted_on.day:02d}" / f"{normalized_id}.json"

    @staticmethod
    def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PredictionRunStoreError("prediction run contains duplicate JSON keys")
            result[key] = value
        return result

    @staticmethod
    def _reject_constant(value: str) -> NoReturn:
        raise PredictionRunStoreError(f"prediction run contains invalid JSON constant: {value}")


__all__ = ["PredictionRunStore", "PredictionRunStoreError"]
