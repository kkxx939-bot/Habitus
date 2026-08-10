"""Behavior Semantic Tree 到 Prediction Tree 样本的确定性单向投影。"""

from __future__ import annotations

from behavior import BehaviorKind, BehaviorSchemaRegistry, BehaviorTree, BehaviorURI
from prediction.document import PredictionDocument, PredictionDocumentCodec
from prediction.factory import PredictionSampleFactory
from prediction.projection._behavior_source import BehaviorProjectionSource
from prediction.projection._consequence_samples import outcome_consequence_samples
from prediction.projection._contract import PROJECTION_VERSION, PROJECTOR_DIGEST
from prediction.projection._trajectory_samples import episode_trajectory_samples
from prediction.projection._transition_samples import (
    episode_transition_samples,
    event_transition_samples,
)
from prediction.schema import PredictionSchemaRegistry


class BehaviorPredictionProjector:
    """只读 Behavior L2，生成不带写入副作用的规范 Prediction 文档。"""

    def __init__(
        self,
        tree: BehaviorTree,
        *,
        behavior_registry: BehaviorSchemaRegistry | None = None,
        prediction_codec: PredictionDocumentCodec | None = None,
    ) -> None:
        if not isinstance(tree, BehaviorTree):
            raise TypeError("tree must be a BehaviorTree")
        if prediction_codec is None:
            prediction_codec = PredictionDocumentCodec(PredictionSchemaRegistry.load_default())
        if not isinstance(prediction_codec, PredictionDocumentCodec):
            raise TypeError("prediction_codec must be a PredictionDocumentCodec")
        self.source = BehaviorProjectionSource(tree, registry=behavior_registry)
        self.factory = PredictionSampleFactory(prediction_codec)

    def project(self, uri: BehaviorURI | str) -> tuple[PredictionDocument, ...]:
        """按 Behavior 文档类型分派到唯一确定性投影规则。"""

        parsed = BehaviorURI.parse(uri)
        kind = parsed.to_address().kind
        if kind is BehaviorKind.EVENT:
            return self.project_event(parsed)
        if kind is BehaviorKind.EPISODE:
            return self.project_episode(parsed)
        return self.project_outcome(parsed)

    def project_event(self, uri: BehaviorURI | str) -> tuple[PredictionDocument, ...]:
        """从一个 Event 生成每个 Action 前缀以及 Event 终止的 TransitionSample。"""

        return event_transition_samples(self.source, self.factory, uri)

    def project_episode(self, uri: BehaviorURI | str) -> tuple[PredictionDocument, ...]:
        """从 Episode 同时生成事件级 Transition 和 Phase 级 Trajectory。"""

        episode = self.source.read(uri, expected_kind=BehaviorKind.EPISODE)
        events = tuple(
            self.source.read(event_uri, expected_kind=BehaviorKind.EVENT)
            for event_uri in episode.fields["ordered_event_uris"]
        )
        return (
            *episode_transition_samples(self.source, self.factory, episode, events),
            *episode_trajectory_samples(self.source, self.factory, episode, events),
        )

    def project_outcome(self, uri: BehaviorURI | str) -> tuple[PredictionDocument, ...]:
        """从当前 Outcome revision 为每条结果生成一个版本绑定的 ConsequenceSample。"""

        return outcome_consequence_samples(self.source, self.factory, uri)


__all__ = ["BehaviorPredictionProjector", "PROJECTION_VERSION", "PROJECTOR_DIGEST"]
