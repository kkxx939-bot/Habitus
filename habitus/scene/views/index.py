"""按日读一次、多次查：一天的 occurrence、它们的短程关系、所属的事、未兑现的待用前提。

缓存的生命周期是**一次查询**（一次预测里同一天只读一次）：只有已定稿且已归组的日子（情景树有当前
一代）才进缓存，今天与未归组的日子每次重读——这是"今天必须新鲜、历史不会变"的机械保证。
"""

from __future__ import annotations

from datetime import UTC, date

from habitus.behavior.document import BehaviorDocument
from habitus.behavior.document.link import BehaviorLinkType
from habitus.behavior.model import BehaviorKind
from habitus.behavior.tree import BehaviorTree
from habitus.behavior.uri import BehaviorURI
from habitus.scene.document import SceneDocument
from habitus.scene.ledger import PendingItem, pending_before
from habitus.scene.tree import SceneTree
from habitus.scene.uri import SceneURI
from habitus.scene.views.model import ActionRef


class DayIndex:
    """一天的只读索引。行为侧跳过撞车消歧的重复（与预测夜批同口径）；情景侧只读当前一代。"""

    def __init__(self, behavior_tree: BehaviorTree, scene_tree: SceneTree, day: date, *, subject: str) -> None:
        self.day = day
        self.subject = subject
        self.occurrences: dict[str, BehaviorDocument] = {}
        for document in behavior_tree.read_day(BehaviorKind.OCCURRENCE, day):
            if document.fields.get("original_name") is not None:
                continue
            self.occurrences[str(BehaviorURI.from_address(document.address))] = document
        self.ordered: tuple[str, ...] = tuple(
            sorted(self.occurrences, key=lambda uri: (self.occurrences[uri].address.started_at.astimezone(UTC), uri))
        )
        self.covered = scene_tree.day_state(day) is not None
        self.scenes: dict[str, SceneDocument] = {}
        self.scene_of: dict[str, str] = {}
        if self.covered:
            for scene in scene_tree.read_day(day):
                scene_uri = str(SceneURI.from_address(scene.address))
                self.scenes[scene_uri] = scene
                for member in scene.fields["members"]:
                    self.scene_of[str(member["uri"])] = scene_uri
        # 行为树的短程关系（只存前向、目标可能在相邻的一天）：按类型分开留原始目标 URI，投影时经 cache
        # 解析并对 concurrent_with 取对称闭包
        self.concurrent_targets: dict[str, tuple[str, ...]] = {}
        self.results_from_targets: dict[str, tuple[str, ...]] = {}
        for uri, document in self.occurrences.items():
            concurrent = [str(link.to_uri) for link in document.links if link.link_type is BehaviorLinkType.CONCURRENT_WITH]
            results = [str(link.to_uri) for link in document.links if link.link_type is BehaviorLinkType.RESULTS_FROM]
            self.concurrent_targets[uri] = tuple(concurrent)
            self.results_from_targets[uri] = tuple(results)

    def ref(self, uri: str) -> ActionRef:
        document = self.occurrences[uri]
        return ActionRef(uri=uri, name=str(document.fields["name"]), kind_token=str(document.fields["kind_token"]))

    def others(self, uri: str) -> tuple[str, ...]:
        """"和谁"：subjects 里主体之外的人（主体总在里面，留着它这一槽就永远对上）。"""

        return tuple(str(item) for item in self.occurrences[uri].fields["subjects"] if str(item) != self.subject)


class DayIndexCache:
    """一次查询里同一天只读一次；未定稿/未归组的日子不缓存。"""

    def __init__(self, behavior_tree: BehaviorTree, scene_tree: SceneTree, *, subject: str) -> None:
        if not isinstance(behavior_tree, BehaviorTree):
            raise TypeError("behavior_tree must be a BehaviorTree")
        if not isinstance(scene_tree, SceneTree):
            raise TypeError("scene_tree must be a SceneTree")
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError("subject must be non-empty text")
        self.behavior_tree = behavior_tree
        self.scene_tree = scene_tree
        self.subject = subject
        self._days: dict[date, DayIndex] = {}
        self._pending: dict[tuple[date, int], tuple[PendingItem, ...]] = {}

    def day(self, day: date) -> DayIndex:
        cached = self._days.get(day)
        if cached is not None:
            return cached
        index = DayIndex(self.behavior_tree, self.scene_tree, day, subject=self.subject)
        if index.covered:
            self._days[day] = index
        return index

    def pending(self, today: date, expiry_days: int) -> tuple[PendingItem, ...]:
        """今天开始前未兑现的待用前提；与候选无关，一次查询里只算一次。"""

        key = (today, expiry_days)
        if key not in self._pending:
            self._pending[key] = pending_before(self.scene_tree, today, expiry_days=expiry_days)
        return self._pending[key]


__all__ = ["DayIndex", "DayIndexCache"]
