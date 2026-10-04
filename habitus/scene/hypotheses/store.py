"""``scene/hypotheses/`` 这一支的存储：按后件分目录，一条假设一份。

``hypotheses/<后件身份>/<前件集合--方面>.md``。按后件分是为预测层查的方式定的："此刻要判 B，有哪些假设指向 B"。
写入时对着概念集核对（后件是行为概念、前件都在、档是定义过的、分账项是情境概念）——假设引用词汇层，
词汇层的洞不能留给账本去发现。

**同身份改内容要显式**：盘上已有同身份、指纹不同的一条时，默认拒写。换窗、换方向会让挂在这条假设下的账
混两种量，事后分不开；``supersede=True`` 表示调用方知道这回事、会一并处理那本账。改理由与来源不算改内容。
没有代、没有完成标记。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import ClassVar

from habitus.scene.codec import SceneRecordError
from habitus.scene.concepts.model import ConceptError, ConceptSet, concept_identity
from habitus.scene.hypotheses.document import decode, encode
from habitus.scene.hypotheses.model import Hypothesis, HypothesisError
from habitus.scene.storage import SceneStore

HYPOTHESES_SEGMENT = "hypotheses"
#: 闭环"问过了"的旁册文件名（点开头：不是假设文件，``read_all`` 与 ``consequents`` 只看目录）。
CLOSURE_ASKED_FILE = ".closure-asked.json"
MAX_RECORD_BYTES = 64 * 1024


class HypothesisStoreError(ValueError):
    """假设存储的路径逃逸、损坏记录，或对着概念集不成立的写入。"""


class HypothesisStore(SceneStore):
    error_type: ClassVar[type[ValueError]] = HypothesisStoreError

    def __init__(self, scene_root: str | Path, **options: int) -> None:
        super().__init__(scene_root, **options)
        self.directory = self._inside(Path(HYPOTHESES_SEGMENT))

    def initialize(self) -> Path:
        self._ensure_directory(self.directory)
        return self.directory

    # ── 写 ──────────────────────────────────────────────────────────────────

    def write(self, hypothesis: Hypothesis, concepts: ConceptSet, *, supersede: bool = False) -> Path:
        """对着概念集核对后原子写、回读比对；同身份、指纹不同的改写要 ``supersede=True``。"""

        if not isinstance(hypothesis, Hypothesis):
            raise TypeError("hypothesis must be a Hypothesis")
        if not isinstance(concepts, ConceptSet):
            raise TypeError("concepts must be a ConceptSet")
        try:
            hypothesis.validate_against(concepts)
        except HypothesisError as exc:
            raise HypothesisStoreError(str(exc)) from exc
        path = self.path_for(hypothesis.identity)
        current = self._read_bytes(path, MAX_RECORD_BYTES)
        if current is not None and not supersede:
            existing = self._decode(current, hypothesis.identity)
            if existing.fingerprint != hypothesis.fingerprint:
                raise HypothesisStoreError(
                    f"hypothesis {hypothesis.identity!r} already exists with fingerprint {existing.fingerprint}; "
                    "pass supersede=True to replace it (and deal with its ledger)"
                )
        self.initialize()
        self._atomic_write(path, encode(hypothesis).encode("utf-8"), maximum=MAX_RECORD_BYTES)
        return path

    # ── 闭环问过的事实 ──────────────────────────────────────────────────────

    def closure_asked(self) -> Mapping[str, str]:
        """闭环已经问过触点③ 的那些"不一样"（事实键 → 问的那天）。

        "不一样"是从读数里现扫的，账没变它每晚都在；不记下来就每晚重问同三条、第 4 条起永远轮不到（评审 A-5 / B-7 / C-10）。
        它不是假设、不是账，只是一份"问过了"的备忘，所以放在 hypotheses/ 目录下的一个 JSON 旁册里。
        """

        payload = self._read_bytes(self._asked_path(), MAX_RECORD_BYTES)
        if payload is None:
            return {}
        try:
            data = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HypothesisStoreError("the closure-asked sidecar is corrupt") from exc
        if not isinstance(data, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in data.items()):
            raise HypothesisStoreError("the closure-asked sidecar has the wrong shape")
        return dict(data)

    def record_closure_asked(self, key: str, day: date) -> None:
        if not isinstance(key, str) or not key.strip():
            raise HypothesisStoreError("a closure fact key must be non-empty text")
        table = dict(self.closure_asked())
        table[key] = day.isoformat()
        self.initialize()
        payload = json.dumps(dict(sorted(table.items())), ensure_ascii=False, indent=1).encode("utf-8")
        self._atomic_write(self._asked_path(), payload, maximum=MAX_RECORD_BYTES)

    def _asked_path(self) -> Path:
        return self._inside(Path(HYPOTHESES_SEGMENT) / CLOSURE_ASKED_FILE)

    # ── 读 ──────────────────────────────────────────────────────────────────

    def read(self, identity: str) -> Hypothesis:
        payload = self._read_bytes(self.path_for(identity), MAX_RECORD_BYTES)
        if payload is None:
            raise HypothesisStoreError(f"hypothesis {identity!r} does not exist")
        return self._decode(payload, identity)

    def exists(self, identity: str) -> bool:
        return self._file_exists(self.path_for(identity))

    def consequents(self) -> tuple[str, ...]:
        """有假设指向的后件身份。目录名必须已是规范身份。"""

        names: list[str] = []
        for entry in self._children(self.directory):
            if not entry.is_dir():
                if entry.name.startswith("."):
                    continue
                raise HypothesisStoreError(f"unexpected file in the hypotheses directory: {entry.name!r}")
            if self._identity(entry.name) != entry.name:
                raise HypothesisStoreError(f"hypotheses directory {entry.name!r} does not carry a canonical identity")
            names.append(entry.name)
        return tuple(sorted(names))

    def for_consequent(self, name: str) -> tuple[Hypothesis, ...]:
        """指向这个后件的全部假设，按身份排序——预测层的查法。"""

        consequent = self._identity(name)
        directory = self._inside(Path(HYPOTHESES_SEGMENT, consequent))
        records, _noise = self._content_files(directory)
        found: list[Hypothesis] = []
        for entry in records:
            if not entry.name.endswith(".md"):
                raise HypothesisStoreError(f"unexpected entry under hypotheses/{consequent}: {entry.name!r}")
            found.append(self.read(f"{consequent}/{entry.name[:-3]}"))
        return tuple(sorted(found, key=lambda item: item.identity))

    def read_all(self) -> tuple[Hypothesis, ...]:
        return tuple(item for consequent in self.consequents() for item in self.for_consequent(consequent))

    def path_for(self, identity: str) -> Path:
        if not isinstance(identity, str):
            raise TypeError("identity must be a string")
        consequent, separator, leaf = identity.partition("/")
        if not separator or not leaf or "/" in leaf:
            raise HypothesisStoreError("hypothesis identity is '<consequent>/<leaf>'")
        return self._inside(Path(HYPOTHESES_SEGMENT, self._identity(consequent), f"{leaf}.md"))

    # ── 内部 ────────────────────────────────────────────────────────────────

    @staticmethod
    def _identity(name: object) -> str:
        try:
            return concept_identity(name)
        except ConceptError as exc:
            raise HypothesisStoreError(str(exc)) from exc

    def _decode(self, payload: bytes, identity: str) -> Hypothesis:
        try:
            return decode(payload.decode("utf-8"), expected_identity=identity)
        except (UnicodeDecodeError, SceneRecordError) as exc:
            raise HypothesisStoreError(f"hypothesis record {identity!r} is corrupt: {exc}") from exc


__all__ = ["HYPOTHESES_SEGMENT", "MAX_RECORD_BYTES", "HypothesisStore", "HypothesisStoreError"]
