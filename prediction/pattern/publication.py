"""完整 Pattern generation 的验证、发布与激活。

TODO(PRED-STORE-001): 发布批次原子化,修复继承合成跨 generation 混用。

- 现状:``publish`` 按 State 分组逐个成代、独立激活,跨状态没有原子性;
  manifest 的 ``learning_identity`` 只含超参与词表版本,没有批次身份,
  预测端无从校验两个状态是否出自同一次学习。
- 问题场景:稀疏状态继承合成(``predictor._with_inherited_candidates``)
  一次预测要读两个状态——匹配态(序列/时段)与同域根状态。重学习后
  再发布时状态按 key 排序逐个激活,或发布中途崩溃后,匹配态还指旧批、
  根状态已指新批。例:晚间时段态(旧批,本地只有"倒一杯水" p=0.55)
  并联根状态(新批,新增"拉上窗帘" p=0.30、"倒一杯水"重估为 0.20),
  重叠质量 S 按新父层算得 0.20,escape=(1−0.55)/(1−0.20)≈0.5625,
  "拉上窗帘"被合成为 ≈0.169;若同批旧父层里它本应只有 ≈0.08,本地
  分支与继承分支的排名可能翻转,边际门与 top-1 在不一致的数字上决策。
- 影响大小:中危。不违反数学不变量(总概率仍 <1,不崩溃),run 的
  双 pattern binding 可事后审计;但每次重学习发布必然经过该窗口,
  窗口内预测静默失真;发布中途崩溃则混合态持续到下次成功发布。
- 改造方案(方向已确认):两阶段发布——第一阶段对本批全部状态
  prepare + 物化校验,第二阶段统一翻转 active 指针,任一失败整批不
  激活(旧代继续服务);``learning_identity`` 增补 ``learned_at`` 批次
  身份;预测端继承合成处加同批断言,不同批直接抛错拒绝该状态预测,
  不静默退化。翻转阶段逐文件写、非严格原子,剩余毫秒级窗口由断言
  兜底,不再引入批级单点记录。
- 时机:behavior 剩余模块落地、prediction 接 Runtime 主链
  (TODO(PRED-RUNTIME-001))时一并实现,避免现在改激活协议约束
  后续设计。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from foundation.integrity import text_digest
from prediction.model import PredictionPatternKind
from prediction.pattern.contracts import PredictionPatternRepository
from prediction.pattern.document import PredictionPatternDocument
from prediction.pattern.generation import (
    PredictionPatternGenerationEntry,
    PredictionPatternGenerationManifest,
    PredictionPatternGenerationRecord,
    PredictionPatternGenerationStore,
    default_generation_store_root,
)
from prediction.uri import PredictionURI


class PredictionPatternPublicationError(RuntimeError):
    """完整 generation 无法被规范物化或回读。"""


class PredictionPatternGenerationPublisher:
    """只有完整、可回读的 generation 才能成为 active。"""

    def __init__(
        self,
        tree: PredictionPatternRepository,
        *,
        store: PredictionPatternGenerationStore | None = None,
    ) -> None:
        if not isinstance(tree, PredictionPatternRepository):
            raise TypeError("tree must satisfy PredictionPatternRepository")
        self.tree = tree
        self.store = store or PredictionPatternGenerationStore(
            default_generation_store_root(tree.root)
        )

    def publish(
        self,
        documents: Sequence[PredictionPatternDocument],
        *,
        created_at: datetime | None = None,
        learning_identity: Mapping[str, Any] | None = None,
    ) -> tuple[PredictionPatternGenerationRecord, ...]:
        """按 State 分组发布；每个 State 及其 Branch 独立成代、独立激活。

        ``learning_identity`` 是学习器的匹配关键参数快照（时段参数、词表
        版本），随 manifest 落盘并进入 manifest 摘要，供预测端构造期握手，
        防止学习/预测两侧配置漂移时静默命中语义错误的状态。
        """

        ordered = self._validated_documents(documents)
        timestamp = created_at or datetime.now(timezone.utc)
        grouped: dict[str, list[PredictionPatternDocument]] = defaultdict(list)
        for document in ordered:
            grouped[document.address.state_key].append(document)
        return tuple(
            self._publish_state(
                grouped[state_key], created_at=timestamp, learning_identity=learning_identity
            )
            for state_key in sorted(grouped)
        )

    def _publish_state(
        self,
        documents: Sequence[PredictionPatternDocument],
        *,
        created_at: datetime,
        learning_identity: Mapping[str, Any] | None,
    ) -> PredictionPatternGenerationRecord:
        state = next(item for item in documents if item.kind is PredictionPatternKind.STATE)
        entries = tuple(
            sorted(
                PredictionPatternGenerationEntry(
                    str(PredictionURI.from_pattern_address(document.address)),
                    text_digest(self.tree.pattern_codec.encode(document)),
                )
                for document in documents
            )
        )
        manifest = PredictionPatternGenerationManifest(
            generation_id=state.fields["pattern_generation"],
            projection_version=state.fields["projection_version"],
            created_at=created_at,
            logical_state_key=state.fields["logical_state_key"],
            entries=entries,
            learning_identity=learning_identity,
        )
        prepared = self.store.prepare(manifest)
        manifest = prepared.manifest

        for document in documents:
            self.tree.create_pattern(document)
        self._verify_materialized(manifest)
        ready = self.store._mark_ready(manifest.logical_state_key, manifest.generation_id)
        return self.store._activate(ready.manifest.logical_state_key, ready.manifest.generation_id)

    def _validated_documents(
        self,
        documents: Sequence[PredictionPatternDocument],
    ) -> tuple[PredictionPatternDocument, ...]:
        if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
            raise TypeError("documents must be a sequence of PredictionPatternDocument values")
        by_uri: dict[str, PredictionPatternDocument] = {}
        for document in documents:
            if not isinstance(document, PredictionPatternDocument):
                raise TypeError("documents must contain PredictionPatternDocument values")
            uri = str(PredictionURI.from_pattern_address(document.address))
            existing = by_uri.get(uri)
            if existing is not None and existing != document:
                raise ValueError("one generation cannot bind one Pattern URI to different content")
            by_uri[uri] = document
        if not by_uri:
            raise ValueError("pattern generation cannot be empty")
        ordered = tuple(
            sorted(
                by_uri.values(),
                key=lambda document: (
                    0 if document.kind is PredictionPatternKind.STATE else 1,
                    str(PredictionURI.from_pattern_address(document.address)),
                ),
            )
        )
        generations = {document.fields["pattern_generation"] for document in ordered}
        versions = {document.fields["projection_version"] for document in ordered}
        if len(generations) != 1 or len(versions) != 1:
            raise ValueError("one Pattern generation must use one generation ID and projection version")
        states = {
            document.address.state_key: document
            for document in ordered
            if document.kind is PredictionPatternKind.STATE
        }
        if not states:
            raise ValueError("pattern generation requires at least one StatePattern")
        branches_by_state: dict[str, list[PredictionPatternDocument]] = defaultdict(list)
        for document in ordered:
            if document.kind is PredictionPatternKind.STATE:
                continue
            if document.address.state_key not in states:
                raise ValueError("Pattern generation Branch requires its State in the same manifest")
            branches_by_state[document.address.state_key].append(document)
        for branches in branches_by_state.values():
            if sum(branch.fields["conditional_probability"] for branch in branches) > 1.000000001:
                raise ValueError("Pattern generation outgoing branch probabilities cannot exceed one")
        return ordered

    def _verify_materialized(self, manifest: PredictionPatternGenerationManifest) -> None:
        for entry in manifest.entries:
            address = PredictionURI.parse(entry.uri).to_pattern_address()
            document = self.tree.read_pattern(address)
            if document.fields["pattern_generation"] != manifest.generation_id:
                raise PredictionPatternPublicationError(
                    "Pattern document belongs to another generation"
                )
            if document.fields["projection_version"] != manifest.projection_version:
                raise PredictionPatternPublicationError(
                    "Pattern document uses another projection version"
                )
            if text_digest(self.tree.pattern_codec.encode(document)) != entry.digest:
                raise PredictionPatternPublicationError(
                    "Pattern document digest does not match its manifest"
                )


__all__ = ["PredictionPatternGenerationPublisher", "PredictionPatternPublicationError"]
