"""预测层测试的现场：一棵真实的行为树喂出预测树，卡的序列与视图从同一棵行为树读。

四层出处的全部意义就是"数字与背景来自同一批日子"，所以夹具不能一边造假树一边造假卡——
预测树必须从行为树来，与夜批的真实顺序一致（行为树封口 → 预测树重建）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path

from habitus.behavior import BehaviorDocumentWriter
from habitus.behavior.model import BehaviorKind
from habitus.foresight import EvidencePack, RelationNote, UnsealedRow, assemble, moment_at
from habitus.foresight.judge import Judgement
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.prediction import builder
from habitus.prediction.config import PredictionTreeConfig
from habitus.prediction.model import PredictionTree
from habitus.scene.views import DayIndexCache
from habitus.series.reader import admitted
from tests.unit.behavior.tree_payloads import gap_payload
from tests.unit.kind_ids import kind_names
from tests.unit.prediction.prediction_fixtures import snapshot_from_tree
from tests.unit.scene.fixtures import SUBJECT, Site, at, publish

SLOT_MINUTES = 15


def unchanged(token: str) -> tuple[str, ...]:
    """词表没拆改过：编号现在对应的就是它自己（结算、计数的 ``current``）。"""

    return (token,)


def settled() -> bool:
    """词表没有迁移在进行（结算的 ``migrating``）。"""

    return False


def config(**overrides) -> PredictionTreeConfig:
    """几乎关掉收缩的一组参数：四层的裸比值要能直接手算。"""

    values = dict(
        slot_minutes=SLOT_MINUTES,
        decay_half_life_days=3_650.0,
        recent_half_life_days=14.0,
        recurrence_half_life_days=3_650.0,
        pool_half_width=2,
        shrink_slot_to_pool=0.001,
        shrink_pool_to_weekday=0.001,
        shrink_weekday_to_all_day=0.001,
        laplace_epsilon=0.001,
        transition_window_slots=8,
        shrink_edge=0.001,
        recurrence_window_days=90.0,
        rebuild_interval_seconds=86_400.0,
        published_generations=3,
    )
    values.update(overrides)
    return PredictionTreeConfig(**values)


def slot_of(hour: int, minute: int = 0) -> int:
    return (hour * 60 + minute) // SLOT_MINUTES


class Ground:
    """一棵行为树 + 由它派生的预测树。"""

    def __init__(self, tmp_path: Path, *, now: datetime) -> None:
        self.site = Site(tmp_path, now=now)

    def record(
        self, day: date, name: str, hour: int, minute: int = 0, *, kind: str | None = None, lasts_minutes: int = 10
    ) -> str:
        return publish(self.site.behavior_tree, day, name, hour, minute, kind=kind, lasts_minutes=lasts_minutes)

    def gap(
        self, day: date, start_hour: int, start_minute: int, end_hour: int, end_minute: int, *, kind: str = "没读懂"
    ) -> None:
        """一段观测空白：删失与曝光都靠它，没有它测不出"那段没看清"。"""

        writer = BehaviorDocumentWriter(self.site.behavior_tree, ProcessLocalLockStore(), clock=lambda: at(day, 23, 59))
        writer.publish(
            BehaviorKind.GAP,
            gap_payload(
                occurred_on=day,
                started_at=at(day, start_hour, start_minute),
                ended_at=at(day, end_hour, end_minute),
                gap_kind=kind,
            ),
        )

    def tree(self, **overrides) -> PredictionTree:
        snapshot = snapshot_from_tree(self.site.behavior_tree)
        latest = snapshot.latest_day
        assert latest is not None
        return builder.build(
            snapshot,
            config=config(**overrides),
            reference=latest,
            built_at=datetime(2026, 12, 31, tzinfo=UTC),
        )

    def cache(self) -> DayIndexCache:
        return DayIndexCache(self.site.behavior_tree, subject=SUBJECT, admits=admitted)

    def pack(
        self,
        now: datetime,
        *,
        unsealed: Sequence[UnsealedRow] = (),
        half_width: int = 3,
        window_days: int = 30,
        max_days: int = 40,
        tree: PredictionTree | None = None,
        relations: Mapping[str, tuple[RelationNote, ...]] | None = None,
    ) -> EvidencePack:
        """走真实读口装一包。"""

        resolved = tree if tree is not None else self.tree()
        moment = moment_at(now, slot_minutes=resolved.slot_minutes)
        return assemble(
            resolved,
            moment,
            self.cache(),
            generation="test-generation",
            unsealed=unsealed,
            half_width=half_width,
            window_days=window_days,
            transition_window_seconds=7_200.0,
            max_days_per_layer=max_days,
            labels=kind_names(),
            relations=relations,
        )


class ScriptedJudge:
    """脚本化的判断：不调模型，按脚本回放 ``Judgement``（判断本身不改，只把包的一代与此刻换成收到的那一包的）。

    记下收到的每一包，节奏层的"同槽复用不再判"要靠它数次数。
    """

    version = "scripted-judge"

    def __init__(self, script: Judgement) -> None:
        self.script = script
        self.packs: list[EvidencePack] = []

    async def judge(self, pack: EvidencePack) -> Judgement:
        self.packs.append(pack)
        return Judgement(
            judged_at=self.script.judged_at,
            generation=pack.generation,
            moment=pack.moment,
            verdicts=self.script.verdicts,
            day_state=self.script.day_state,
            day_note=self.script.day_note,
            judge_version=self.version,
            signals=self.script.signals,
        )


#: 各现场共用的锚点周一。现场常量属于夹具模块，不能挂在某个测试文件上让别人去 import。
MONDAY = date(2026, 8, 3)

__all__ = ["MONDAY", "SLOT_MINUTES", "Ground", "ScriptedJudge", "at", "config", "slot_of"]
