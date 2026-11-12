"""``scene/advice/`` 的存储：模型输出按输入摘要缓存，回放与回测只读缓存、不重新问模型（语义树新方案 ``13`` ②）。

```
advice/priors/<缓存键>.json        一个后果的先验（旧问法：固定后果、列前因；缓存键 = 输入摘要 + 先验触点的版本与模型）
advice/priors-by-antecedent/<缓存键>.json  一个前因的先验（新问法：固定前因、列后果；实时化阶段 2「先验混合问法」）
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
#: 新问法的先验缓存目录（按前因存）。
BY_ANTECEDENT = "priors-by-antecedent"
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

    def prior_keys(self, lane: str) -> tuple[str, ...]:
        """这条 lane 的全部先验缓存键（给清孤儿用：后果改名、改判据、换提示词或模型都会换键，旧文件没人再读）。"""

        directory = self._inside(Path(ADVICE_SEGMENT, "priors"))
        records, _noise = self._content_files(directory)
        keys = []
        for entry in records:
            key = entry.name.removesuffix(".json")
            payload = self._read(self._prior_path(key))
            if payload is not None and payload.get("lane") == lane:
                keys.append(key)
        return tuple(sorted(keys))

    def discard_prior(self, key: str) -> None:
        self._discard(self._prior_path(key))

    # ── 先验（新问法：按前因存）────────────────────────────────────────────

    def prior_by_antecedent(self, key: str) -> Mapping[str, PriorLevel] | None:
        payload = self._read(self._prior_path(key, BY_ANTECEDENT))
        if payload is None:
            return None
        try:
            return {str(name): PriorLevel(level) for name, level in payload["levels"].items()}
        except (KeyError, AttributeError, ValueError) as exc:
            raise AdviceStoreError(f"prior cache {key!r} is corrupt: {exc}") from exc

    def write_prior_by_antecedent(
        self,
        key: str,
        *,
        lane: str,
        antecedent: str,
        levels: Mapping[str, PriorLevel],
        version: str,
        models: tuple[str, ...],
    ) -> None:
        self._write(
            self._prior_path(key, BY_ANTECEDENT),
            {
                "lane": lane,
                "antecedent": antecedent,
                "levels": {name: level.value for name, level in levels.items()},
                "version": version,
                "models": list(models),
            },
        )

    def prior_by_antecedent_keys(self, lane: str) -> tuple[str, ...]:
        """这条 lane 按前因存的全部先验缓存键（清孤儿用）。"""

        directory = self._inside(Path(ADVICE_SEGMENT, BY_ANTECEDENT))
        records, _noise = self._content_files(directory)
        keys = []
        for entry in records:
            key = entry.name.removesuffix(".json")
            payload = self._read(self._prior_path(key, BY_ANTECEDENT))
            if payload is not None and payload.get("lane") == lane:
                keys.append(key)
        return tuple(sorted(keys))

    def discard_prior_by_antecedent(self, key: str) -> None:
        self._discard(self._prior_path(key, BY_ANTECEDENT))

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

    def _prior_path(self, key: str, directory: str = "priors") -> Path:
        if not key.isalnum():
            raise AdviceStoreError("a prior cache key must be a digest")
        return self._inside(Path(ADVICE_SEGMENT, directory, f"{key}.json"))

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
    condition, refused = body["condition"], body["refused"]
    if condition is not None and not isinstance(condition, str):
        raise ValueError("proposal condition must be text or null")
    if refused is not None and not isinstance(refused, str):
        raise ValueError("proposal refused reason must be text or null")
    return Proposal(
        antecedent=str(body["antecedent"]),
        proposed_on=date.fromisoformat(body["proposed_on"]),
        why=str(body["why"]),
        condition=condition,
        raw=dict(body["raw"]),
        refused=refused,
        version=str(body["version"]),
        model=str(body["model"]),
    )


__all__ = ["ADVICE_SEGMENT", "AdviceStore", "AdviceStoreError"]
