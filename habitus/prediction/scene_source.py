"""从情景树读取一次重建可能用到的情景流、成员与关系实例。

这是本层**唯一**碰情景树的读取入口（与 ``source.py`` 之于行为树同构），也是
``prediction → scene`` 这条依赖的唯一落点。产物类型住在 ``model.py``（纯数据，不沾 scene）。

第一期只定读接口、不做任何计数：情景层计数与跨层边是否进预测树，由回测决定（用户裁定）。

读取纪律：

- 只读不写；只读每天**当前生效的那一代**；
- 成员的 kind_token 不在情景文档上（那里只有 occurrence URI），从行为树按天整块读一次解析；
  解析不到的成员（被跳过的消歧重复、已不在树上）计数跳过，不硬失败；
- 角色与边类型原样以字符串交出，不在本层解释含义；
- ``covered_days`` 是情景树处理过的全部日子（含零情景日），供判决核对"语义层覆盖到哪一天"。
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from habitus.behavior.document import BehaviorDocument
from habitus.behavior.tree import BehaviorKind, BehaviorTree
from habitus.behavior.uri import BehaviorURI
from habitus.prediction.errors import PredictionTreeError
from habitus.prediction.model import ObservedScene, ObservedSceneMember, ObservedSceneRelation, SceneSnapshot
from habitus.scene.tree import SceneTree
from habitus.scene.uri import SceneURI


def read(scene_tree: SceneTree, behavior_tree: BehaviorTree) -> SceneSnapshot:
    """读出全部已处理日子的情景、成员（含 kind_token 与相对情景开始的偏移）与关系。"""

    if not isinstance(scene_tree, SceneTree):
        raise PredictionTreeError("scene_tree must be a SceneTree")
    if not isinstance(behavior_tree, BehaviorTree):
        raise PredictionTreeError("behavior_tree must be a BehaviorTree")
    scenes: list[ObservedScene] = []
    members: list[ObservedSceneMember] = []
    relations: list[ObservedSceneRelation] = []
    unresolved = 0
    skipped_duplicates = 0
    occurrences_by_day: dict[date, dict[str, BehaviorDocument]] = {}
    covered_days = scene_tree.list_days()

    def occurrence(uri: str) -> BehaviorDocument | None:
        parsed = BehaviorURI.parse(uri)
        day = parsed.to_address().occurred_on
        index = occurrences_by_day.get(day)
        if index is None:
            index = {}
            for document in behavior_tree.read_day(BehaviorKind.OCCURRENCE, day):
                index[str(BehaviorURI.from_address(document.address))] = document
            occurrences_by_day[day] = index
        return index.get(str(parsed))

    for day in covered_days:
        for document in scene_tree.read_day(day):
            uri = str(SceneURI.from_address(document.address))
            started_at = document.address.started_at
            scene_index = len(scenes)
            scenes.append(
                ObservedScene(
                    label=document.address.label,
                    started_at=started_at,
                    ended_at=datetime.fromisoformat(str(document.fields["ended_at"])),
                    day=day,
                    uri=uri,
                )
            )
            for member in document.fields["members"]:
                resolved = occurrence(member["uri"])
                if resolved is None:
                    unresolved += 1
                    continue
                if resolved.fields.get("original_name") is not None:
                    # 撞车消歧的已知重复：与预测夜批同一口径，机械跳过、只计数。
                    skipped_duplicates += 1
                    continue
                offset = (resolved.address.started_at.astimezone(UTC) - started_at.astimezone(UTC)).total_seconds()
                members.append(
                    ObservedSceneMember(
                        scene_index=scene_index,
                        action=str(resolved.fields["kind_token"]),
                        role=str(member["role"]),
                        offset_seconds=offset,
                    )
                )
            relations.extend(
                ObservedSceneRelation(
                    kind=link.link_type.value,
                    from_uri=uri,
                    to_uri=str(link.to_uri),
                    lag_seconds=float(link.lag_seconds),
                )
                for link in document.links
            )
    # 日期目录是本地日历日，不是瞬时边界：跨偏移的两天在"日序"上可以与"瞬时序"相反，所以像
    # ``source.read`` 一样全局按瞬时排序，再把成员的下标重映射到排序后的位置。
    # 情景目标必须在其日当前一代里：某天被重建而后续日尚未级联重算时，旧地址会悬空——读侧丢弃只计数，
    # 不把一条指向已不存在的情景的边交给预测。
    known = {scene.uri for scene in scenes}
    kept_relations = [
        relation for relation in relations if not relation.to_uri.startswith("scene://") or relation.to_uri in known
    ]
    dangling = len(relations) - len(kept_relations)
    order = sorted(range(len(scenes)), key=lambda index: (scenes[index].started_at.astimezone(UTC), scenes[index].uri))
    rank = {index: position for position, index in enumerate(order)}
    return SceneSnapshot(
        scenes=tuple(scenes[index] for index in order),
        members=tuple(
            ObservedSceneMember(
                scene_index=rank[member.scene_index],
                action=member.action,
                role=member.role,
                offset_seconds=member.offset_seconds,
            )
            for member in sorted(members, key=lambda item: (rank[item.scene_index], item.offset_seconds, item.action))
        ),
        relations=tuple(kept_relations),
        covered_days=covered_days,
        unresolved_members=unresolved,
        skipped_duplicates=skipped_duplicates,
        dangling_relations=dangling,
    )


__all__ = ["read"]
