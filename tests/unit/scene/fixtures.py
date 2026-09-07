"""scene 测试共用的现场：两棵树 + 脚本化归组器 + 发布行为的助手 + 脚本化结构化客户端。"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from habitus.behavior import BehaviorDocumentWriter
from habitus.behavior.model import BehaviorKind
from habitus.behavior.tree import BehaviorTree
from habitus.behavior.uri import BehaviorURI
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.model_client import ChatClient, ChatModelConfig, ProviderConfig, StructuredChatClient
from habitus.scene import SceneTree
from habitus.scene.grouping import DraftRelation, GroupingAssembly, SceneDraft, SceneGroupingInput
from habitus.scene.model import SceneLinkType, SceneRole
from habitus.scene.refresh import SceneRefreshConfig, SceneRefresher
from tests.unit.behavior.test_kinds import ScriptedProvider
from tests.unit.behavior.tree_payloads import gap_payload, occurrence_payload
from tests.unit.scene.scene_payloads import CST

DAY1 = date(2026, 8, 15)
DAY2 = date(2026, 8, 16)
DAY3 = date(2026, 8, 17)
SUBJECT = "家庭成员A"


def at(day: date, hour: int, minute: int) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=CST)


def publish(tree: BehaviorTree, day: date, name: str, hour: int, minute: int, *, kind: str | None = None, **overrides: Any) -> str:
    """往行为树发布一条 occurrence，返回它的 URI。"""

    started = at(day, hour, minute)
    writer = BehaviorDocumentWriter(tree, ProcessLocalLockStore(), clock=lambda: started + timedelta(hours=3))
    payload = occurrence_payload(
        occurred_on=day,
        name=name,
        kind_token=kind or name,
        started_at=started,
        last_observed_at=started + timedelta(minutes=10),
        onset_available_at=started + timedelta(seconds=2),
        basis=(),
        goal=None,
        **overrides,
    )
    document = writer.publish(BehaviorKind.OCCURRENCE, payload)
    return str(BehaviorURI.from_address(document.address))


class ScriptedGrouper:
    """按输入的日期决定归组结果：DAY1 一件"采购"（留待用前提），DAY2 一件"准备晚饭"指回它。"""

    version = "scripted_grouping_v1+schema000000000000"

    def __init__(self) -> None:
        self.payloads: list[SceneGroupingInput] = []
        self.explode: Exception | None = None
        self.duplicate_label = False

    async def group(self, payload: SceneGroupingInput) -> GroupingAssembly:
        self.payloads.append(payload)
        if self.explode is not None:
            raise self.explode
        by_name = {row.name: row.no for row in payload.occurrences}
        if payload.day == DAY1:
            members = tuple((no, SceneRole.ESSENTIAL) for name, no in by_name.items() if name != "看手机")
            return GroupingAssembly(
                scenes=(
                    SceneDraft(
                        label="去超市采购",
                        members=members,
                        effects=("买到了晚饭食材", f"共 {len(members)} 步"),
                        pending_effects=(("家里有今晚要用的食材", by_name["去超市买菜"]),),
                    ),
                ),
                unassigned=tuple(no for name, no in by_name.items() if name == "看手机"),
                signals=(),
            )
        relations: list[DraftRelation] = []
        if payload.pending_references:
            relations.append(DraftRelation(SceneLinkType.NEEDS, pending_no=1))
        if payload.scene_references:
            relations.append(DraftRelation(SceneLinkType.RESULTS_FROM, reference_no=1))
        dinner = SceneDraft(
            label="准备晚饭",
            members=tuple((no, SceneRole.ESSENTIAL) for no in by_name.values()),
            relations=tuple(relations),
        )
        scenes: tuple[SceneDraft, ...] = (dinner,)
        if self.duplicate_label:
            # 同地址的第二篇：装配层按规范身份合并，脚本化 grouper 绕过装配层，用来验证文档级降级
            scenes = (dinner, SceneDraft(label="准备晚饭", members=dinner.members[:1]))
        return GroupingAssembly(scenes=scenes, unassigned=(), signals=("role_degraded: scripted",))


def scripted_by_day(by_day):
    """按天给定装配结果的归组函数（``by_day[day](by_name) -> GroupingAssembly``）。"""

    async def group(payload):
        by_name = {row.name: row.no for row in payload.occurrences}
        return by_day[payload.day](by_name)

    return group


class Site:
    """一套真实的两棵树 + 可拨时钟 + 脚本化归组器的刷新现场。"""

    def __init__(self, tmp_path: Path, *, now: datetime, **config: Any) -> None:
        self.behavior_tree = BehaviorTree(tmp_path / "behavior" / "tree")
        self.scene_tree = SceneTree(tmp_path / "scene" / "tree")
        self.grouper = ScriptedGrouper()
        self.now = now
        self.lock_store = ProcessLocalLockStore()
        self.refresher = SceneRefresher(
            behavior_tree=self.behavior_tree,
            scene_tree=self.scene_tree,
            grouper=self.grouper,
            subject=SUBJECT,
            progress_root=tmp_path / "scene" / "refresh",
            lock_store=self.lock_store,
            config=SceneRefreshConfig(lookback_days=3, pending_expiry_days=30, **config),
            clock=lambda: self.now,
        )

    def seed(self) -> dict[str, str]:
        uris = {
            "buy": publish(self.behavior_tree, DAY1, "去超市买菜", 15, 20, kind="买菜"),
            "phone1": publish(self.behavior_tree, DAY1, "看手机", 21, 0),
            "discuss": publish(self.behavior_tree, DAY2, "商量晚餐", 19, 12, kind="交谈"),
            "wash": publish(self.behavior_tree, DAY2, "洗菜", 19, 43),
            "phone2": publish(self.behavior_tree, DAY2, "看手机", 20, 15),
        }
        writer = BehaviorDocumentWriter(self.behavior_tree, ProcessLocalLockStore(), clock=lambda: self.now)
        writer.publish(BehaviorKind.GAP, gap_payload(occurred_on=DAY2, started_at=at(DAY2, 20, 30), ended_at=at(DAY2, 20, 50)))
        return uris

    def refresh(self, *days: date, force: bool = False):
        import asyncio

        return asyncio.run(self.refresher.refresh_days(days, force=force))


def structured_client(provider: ScriptedProvider) -> StructuredChatClient:
    """脚本化 Provider 上的结构化客户端（与 kinds 测试同一配置）。"""

    model_config = ChatModelConfig(
        route=ProviderConfig(provider="fake", adapter="openai_compatible_chat", model="fake-1", base_url="https://example.invalid", credential_ref="FAKE_KEY"),
        context_window_tokens=128_000,
        max_output_tokens=8_000,
        structured_output_mode="json_schema",
    )
    return StructuredChatClient(ChatClient(model_config, provider), validation_retries=1)


__all__ = ["DAY1", "DAY2", "DAY3", "SUBJECT", "ScriptedGrouper", "Site", "at", "publish", "scripted_by_day", "structured_client"]
