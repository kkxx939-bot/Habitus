"""``scene/advice/`` 的存储：模型输出按输入摘要缓存，回放与回测只读缓存、不重新问模型（语义树新方案 ``13`` ②）。

```
advice/priors/<缓存键>.json        一个后果的先验（旧问法：固定后果、列前因；缓存键 = 输入摘要 + 先验触点的版本与模型）
advice/priors-by-antecedent/<缓存键>.json  一个前因的先验（新问法：固定前因、列后果；实时化阶段 2「先验混合问法」）
```

``advice/conditions/`` 是已删的提议簿（问模型提候选条件那一路，2026-11-15 删）：盘上还有就明确报错、点出文件，不迁移
（``refuse_proposal_books``）。

规范 JSON，读回逐字节比对；原子写、回读比对（``SceneStore``）。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from habitus.foundation.integrity import CanonicalSerializationError, canonical_json
from habitus.scene.advice.model import PriorLevel
from habitus.scene.storage import SceneStore

ADVICE_SEGMENT = "advice"
#: 先验缓存的格式版本：语义树只剩词表类与日历之后（2026-11-14）升了一版，旧版本的缓存读到明确报错、不迁移。
PRIOR_SCHEMA = "habitus.scene.advice.prior/v2"
#: 新问法的先验缓存目录（按前因存）。
BY_ANTECEDENT = "priors-by-antecedent"
MAX_RECORD_BYTES = 4 * 1024 * 1024
#: 已删的提议簿目录。
RETIRED_BOOKS = "conditions"


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
        _require_schema(payload, key)
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
                "schema": PRIOR_SCHEMA,
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
        _require_schema(payload, key)
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
                "schema": PRIOR_SCHEMA,
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

    def refuse_proposal_books(self) -> None:
        """盘上还留着已删的提议簿（``advice/conditions/`` 下有任何文件）就报错、点出文件；读先验之前调。"""

        entries = self._children(self.directory / RETIRED_BOOKS)
        if entries:
            found = ", ".join(sorted(f"{ADVICE_SEGMENT}/{RETIRED_BOOKS}/{entry.name}" for entry in entries))
            raise AdviceStoreError(
                f"retired proposal books are still on disk ({found}): candidate conditions were removed, "
                "delete these files (they are not migrated)"
            )

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _prior_path(self, key: str, directory: str = "priors") -> Path:
        if not key.isalnum():
            raise AdviceStoreError("a prior cache key must be a digest")
        return self._inside(Path(ADVICE_SEGMENT, directory, f"{key}.json"))

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


def _require_schema(payload: Mapping[str, Any], key: str) -> None:
    if payload.get("schema") != PRIOR_SCHEMA:
        raise AdviceStoreError(
            f"prior cache {key!r} has schema {payload.get('schema')!r}, this build reads {PRIOR_SCHEMA!r}"
        )


__all__ = ["ADVICE_SEGMENT", "AdviceStore", "AdviceStoreError"]
