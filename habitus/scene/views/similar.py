"""从今天的场景反查历史上"像今天这样"的事：成员 kind 与今天近期 kind 重叠的情景，及它们接下来发生了什么。

对应预测树的转移维度，但方向相反：树从候选出发问"它之前紧挨着什么"，这里从今天已经发生的几步出发
问"历史上这几步属于哪件事、那件事后面接的是什么"——候选层可能漏掉、语义上却强关联的下一步从这里
来。全部机械：重叠按 kind_token 集合交集算；与投影同口径，只有 essential / optional 成员参与重叠与
"接下来"（irrelevant 是"发生在这件事期间"，不是这件事的一步），完整成员序列带角色原样给出；排序按
重叠数、再按日期新近、再按 URI，结果确定；不打分。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date

from habitus.scene.model import SceneRole
from habitus.scene.views.index import DayIndex, DayIndexCache
from habitus.scene.views.model import ActionRef

_STEP_ROLES = frozenset({SceneRole.ESSENTIAL.value, SceneRole.OPTIONAL.value})


@dataclass(frozen=True)
class SceneMember:
    """情景成员按开始时刻排好：行为引用 + 角色。"""

    action: ActionRef
    role: str

    @property
    def is_step(self) -> bool:
        return self.role in _STEP_ROLES


@dataclass(frozen=True)
class SimilarScene:
    """一件与今天场景重叠的历史事：重叠了哪些 kind、完整成员序列、以及**最后一个重叠的步骤之后**的步骤。

    ``following`` 就是"那天走到这一步之后接着做了什么"——"这一步"取今天的 kind 集合在那件事里最后一次
    出现的位置（同一 kind 在一件事里首尾都出现时，按末尾算），判断者拿它对今天的下一步。
    """

    uri: str
    label: str
    day: date
    overlap: tuple[str, ...]
    members: tuple[SceneMember, ...]
    following: tuple[SceneMember, ...]
    # 当地日历对那一天的说法；"上次也是补班日"这种对照要靠它，但它不参与重叠与排序。
    day_note: str | None = None


def similar_scenes(
    cache: DayIndexCache,
    *,
    days: Iterable[date],
    kinds: frozenset[str],
    limit: int,
) -> tuple[SimilarScene, ...]:
    """在情景树覆盖过的 ``days`` 里找步骤 kind 与 ``kinds`` 有交集的情景，按重叠数降序取前 ``limit`` 件。

    ``kinds`` 为空即没有可比的今天，返回空——不用"全部情景"冒充相似。覆盖与否先看指针，不为没覆盖的
    日子解码行为树。"""

    if not isinstance(kinds, frozenset) or any(not isinstance(kind, str) or not kind for kind in kinds):
        raise TypeError("kinds must be a frozenset of non-empty kind tokens")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if not kinds:
        return ()
    found: list[SimilarScene] = []
    for day in sorted(set(days)):
        if not cache.covered(day):
            continue
        index = cache.day(day)
        for scene_uri, scene in index.scenes.items():
            members = _members(index, scene.fields["members"])
            hits = [position for position, member in enumerate(members) if member.is_step and member.action.kind_token in kinds]
            if not hits:
                continue
            found.append(
                SimilarScene(
                    uri=scene_uri,
                    label=scene.address.label,
                    day=day,
                    overlap=tuple(sorted({members[position].action.kind_token for position in hits})),
                    members=members,
                    following=tuple(member for member in members[hits[-1] + 1 :] if member.is_step),
                    day_note=index.day_note,
                )
            )
    found.sort(key=lambda item: (-len(item.overlap), -item.day.toordinal(), item.uri))
    return tuple(found[:limit])


def _members(index: DayIndex, raw_members: Iterable[object]) -> tuple[SceneMember, ...]:
    """成员按 ``index.ordered`` 的全序排（开始瞬时、再 URI），不在树上的（消歧重复）跳过。"""

    roles = {str(member["uri"]): str(member["role"]) for member in raw_members if isinstance(member, Mapping)}
    return tuple(SceneMember(index.ref(uri), roles[uri]) for uri in index.ordered if uri in roles)


__all__ = ["SceneMember", "SimilarScene", "similar_scenes"]
