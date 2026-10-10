"""``scene/advice/`` 的存储：模型输出按输入摘要缓存，回放与回测只读缓存、不重新问模型（语义树新方案 ``13`` ②）。

```
advice/priors/<缓存键>.json        一个后果的先验（缓存键 = 输入摘要 + 先验触点的版本与模型）
advice/conditions/<lane>.json      这条 lane 的提议簿：收下的条件、已拒清单、每个前因上次问的概念集指纹
```

规范 JSON，读回逐字节比对；原子写、回读比对（``SceneStore``）。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any, ClassVar

from habitus.foundation.ids import canonical_path_identity
from habitus.foundation.integrity import CanonicalSerializationError, canonical_json
from habitus.scene.advice.model import PriorLevel, Proposal, ProposalBook
from habitus.scene.storage import SceneStore

ADVICE_SEGMENT = "advice"
MAX_RECORD_BYTES = 4 * 1024 * 1024


class AdviceStoreError(ValueError):
    """建议缓存的路径逃逸或记录损坏。"""


class AdviceStore(SceneStore):
    error_type: ClassVar[type[ValueError]] = AdviceStoreError

    def __init__(self, scene_root: str | Path, **options: int) -> None:
        super().__init__(scene_root, **options)
        self.directory = self._inside(Path(ADVICE_SEGMENT))

    # ── 先验 ────────────────────────────────────────────────────────────────

    def prior(self, key: str) -> Mapping[str, PriorLevel] | None:
        payload = self._read(self._prior_path(key))
        if payload is None:
            return None
        try:
            return {str(name): PriorLevel(level) for name, level in payload["levels"].items()}
        except (KeyError, AttributeError, ValueError) as exc:
            raise AdviceStoreError(f"prior cache {key!r} is corrupt: {exc}") from exc

    def write_prior(
        self,
        key: str,
        *,
        lane: str,
        consequent: str,
        levels: Mapping[str, PriorLevel],
        version: str,
        models: tuple[str, ...],
    ) -> None:
        self._write(
            self._prior_path(key),
            {
                "lane": lane,
                "consequent": consequent,
                "levels": {name: level.value for name, level in levels.items()},
                "version": version,
                "models": list(models),
            },
        )

    # ── 提议簿 ──────────────────────────────────────────────────────────────

    def book(self, lane: str) -> ProposalBook:
        payload = self._read(self._book_path(lane))
        if payload is None:
            return ProposalBook(lane=lane)
        try:
            return ProposalBook(
                lane=lane,
                proposals=tuple(_proposal(item) for item in payload["proposals"]),
                asked={str(name): str(digest) for name, digest in payload["asked"].items()},
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AdviceStoreError(f"proposal book of lane {lane!r} is corrupt: {exc}") from exc

    def write_book(self, book: ProposalBook) -> None:
        self._write(
            self._book_path(book.lane),
            {
                "proposals": [
                    {
                        "antecedent": item.antecedent,
                        "proposed_on": item.proposed_on.isoformat(),
                        "why": item.why,
                        "condition": item.condition,
                        "raw": dict(item.raw),
                        "refused": item.refused,
                        "version": item.version,
                        "model": item.model,
                    }
                    for item in book.proposals
                ],
                "asked": dict(book.asked),
            },
        )

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _prior_path(self, key: str) -> Path:
        if not key.isalnum():
            raise AdviceStoreError("a prior cache key must be a digest")
        return self._inside(Path(ADVICE_SEGMENT, "priors", f"{key}.json"))

    def _book_path(self, lane: str) -> Path:
        try:
            name = canonical_path_identity(lane, "advice lane")
        except ValueError as exc:
            raise AdviceStoreError(str(exc)) from exc
        if name != lane:
            raise AdviceStoreError(f"lane {lane!r} is not a canonical file name")
        return self._inside(Path(ADVICE_SEGMENT, "conditions", f"{lane}.json"))

    def _write(self, path: Path, payload: Mapping[str, Any]) -> None:
        try:
            text = canonical_json(payload)
        except CanonicalSerializationError as exc:
            raise AdviceStoreError("advice record is not canonically serializable") from exc
        self._ensure_directory(path.parent)
        self._atomic_write(path, text.encode("utf-8"), maximum=MAX_RECORD_BYTES)

    def _read(self, path: Path) -> dict[str, Any] | None:
        payload = self._read_bytes(path, MAX_RECORD_BYTES)
        if payload is None:
            return None
        try:
            text = payload.decode("utf-8")
            parsed = json.loads(text)
        except (UnicodeDecodeError, ValueError) as exc:
            raise AdviceStoreError(f"advice record {path.name!r} is not JSON") from exc
        if not isinstance(parsed, dict) or canonical_json(parsed) != text:
            raise AdviceStoreError(f"advice record {path.name!r} is not in canonical form")
        return parsed


def _proposal(body: Mapping[str, Any]) -> Proposal:
    return Proposal(
        antecedent=str(body["antecedent"]),
        proposed_on=date.fromisoformat(body["proposed_on"]),
        why=str(body["why"]),
        condition=body["condition"],
        raw=dict(body["raw"]),
        refused=body["refused"],
        version=str(body["version"]),
        model=str(body["model"]),
    )


__all__ = ["ADVICE_SEGMENT", "AdviceStore", "AdviceStoreError"]
