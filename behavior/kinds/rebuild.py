"""从行为树重建词表：补齐 + 账按树重算 + 向量补算（BHV-KINDS-002 方案⑥）。

词表是派生物，真正的数据只在树上：每条 occurrence 都带着 ``kind_token``（聚合键）、``name`` /
``original_name``（融合原话）与 ``occurred_on``（行为日）。所以重建**零模型调用**：
- **补齐**：树上出现的每个 kind_token 都保证登记；原始名与 token 不同时并作它的别名。已登记条目的
  label 与别名保留（那是模型判过的东西），只有账被重算。
- **账按树重算**：命中账从空开始，按 occurrence 的行为日逐条 ``with_hit``——账因此可自愈。
- **向量补算**：有 embedder 时给缺向量的 token/label 补上；没有就跳过（候选退字面重合）。

用途：v1 格式词表不兼容（无双轨），旧根靠这条路迁；账出错、旁册丢失也走这里。
不做的：合并（方案⑤，成对交模型判"是否同一件事"+ 树上重打 token）——另立。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from behavior.kinds.model import BehaviorKindEntry, BehaviorKindError, BehaviorKindRegistry
from behavior.kinds.store import BehaviorKindStore, BehaviorKindStoreError
from behavior.kinds.vectors import BehaviorKindVectorStore, names_missing_vectors
from behavior.model import BehaviorKind
from behavior.tree import BehaviorTree


class _Vector(Protocol):
    @property
    def values(self) -> Sequence[float]: ...


class _Embedder(Protocol):
    """只用到 ``embed_documents``；不 import ModelClient——模型触点收敛在 resolver。"""

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[_Vector]: ...


@dataclass(frozen=True)
class BehaviorKindRebuildReport:
    """一次重建的可观测结果。"""

    occurrences: int
    kinds: int
    aliases_added: int
    vectors_added: int
    signals: tuple[str, ...]


async def rebuild_registry(
    tree: BehaviorTree,
    store: BehaviorKindStore,
    *,
    now: datetime,
    vectors: BehaviorKindVectorStore | None = None,
    embedder: _Embedder | None = None,
) -> BehaviorKindRebuildReport:
    """按树重建词表并落盘；返回报告。旧词表可读则保留其 label/别名，账一律重算。"""

    signals: list[str] = []
    try:
        snapshot = store.read()
        base = snapshot.registry
        revision = snapshot.revision
    except BehaviorKindStoreError as exc:
        # v1 或损坏：从空建；旧文件的修订号读不出，按"不存在"CAS——文件仍在则 replace 会因修订号冲突失败，
        # 所以先删掉它（派生物，可重建）。
        signals.append(f"kind_registry_unreadable {exc}; rebuilding from tree")
        store.path.unlink(missing_ok=True)
        base = BehaviorKindRegistry()
        revision = 0

    # 保留 label/别名，账清零
    entries = {
        token: BehaviorKindEntry(token=token, label=entry.label, aliases=entry.aliases)
        for token, entry in base.entries.items()
    }
    registry = BehaviorKindRegistry(entries)
    occurrences = 0
    aliases_added = 0
    for document in tree.iter_documents(BehaviorKind.OCCURRENCE):
        occurrences += 1
        fields = document.fields
        if fields.get("original_name") is not None:
            continue  # 撞车消歧记录 = 已知重复：预测树、语义层、命中账一律不计（死规则②）
        token = str(fields.get("kind_token") or "")
        name = str(fields.get("original_name") or fields.get("name") or "")
        if not token:
            signals.append(f"kind_rebuild_skipped occurrence without kind_token at {document.address}")
            continue
        try:
            if registry.token_for(token) is None:
                registry = registry.with_new_kind(token)
            owner = registry.token_for(token)
            assert owner is not None
            if name and registry.token_for(name) is None:
                registry = registry.with_alias(owner, name)
                aliases_added += 1
            registry = registry.with_hit(owner, document.address.occurred_on)
        except BehaviorKindError as exc:
            signals.append(f"kind_rebuild_skipped {token!r}/{name!r}: {exc}")

    written = store.replace(registry, expected_revision=revision, timestamp=now)

    vectors_added = 0
    if vectors is not None and embedder is not None:
        index = vectors.read()
        missing = names_missing_vectors(written.registry, index)
        if missing:
            embedded = await embedder.embed_documents(list(missing))
            index = index.with_vectors(
                {name: tuple(vector.values) for name, vector in zip(missing, embedded, strict=True)}
            )
            vectors_added = len(missing)
        vectors.replace(index.retain(written.registry.names_in_use()))
    return BehaviorKindRebuildReport(
        occurrences=occurrences,
        kinds=written.registry.kind_count,
        aliases_added=aliases_added,
        vectors_added=vectors_added,
        signals=tuple(signals),
    )


__all__ = ["BehaviorKindRebuildReport", "rebuild_registry"]
