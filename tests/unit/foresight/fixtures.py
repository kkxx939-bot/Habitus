"""预测层测试的现场：一棵真实的行为树同时喂出**预测树**与**规律树**，两边是同一批 occurrence。

四层出处的全部意义就是"数字与背景来自同一批日子"，所以夹具不能一边造假树一边造假记录——
两棵派生树必须从同一棵行为树来，与夜批的真实顺序一致（行为树封口 → 预测树重建 → 关联）。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

from habitus.behavior import BehaviorDocumentWriter
from habitus.behavior.model import BehaviorKind
from habitus.foresight import AssociatedDays, EvidencePack, UnsealedRow, assemble, moment_at
from habitus.foresight.judge import Judgement
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.prediction import builder, source
from habitus.prediction.config import PredictionTreeConfig
from habitus.prediction.model import PredictionTree
from habitus.scene.regularity import RegularityTree
from habitus.scene.views import DayIndexCache, association_glosses, situations_of
from tests.unit.behavior.tree_payloads import gap_payload
from tests.unit.scene.fixtures import ASSOCIATION_VERSION, SUBJECT, Site, associate, at, publish, regularity_tree

SLOT_MINUTES = 15


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
        transition_window_seconds=7_200.0,
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
    """一棵行为树 + 由它派生的规律树与预测树。"""

    def __init__(self, tmp_path: Path, *, now: datetime) -> None:
        self.site = Site(tmp_path, now=now)
        self.regularity = regularity_tree(tmp_path)

    def record(
        self, day: date, name: str, hour: int, minute: int = 0, *, kind: str | None = None, lasts_minutes: int = 10
    ) -> str:
        return publish(self.site.behavior_tree, day, name, hour, minute, kind=kind, lasts_minutes=lasts_minutes)

    def gap(
        self, day: date, start_hour: int, start_minute: int, end_hour: int, end_minute: int, *, kind: str = "没读懂"
    ) -> None:
        """一段观测空白：删失与曝光都靠它，没有它测不出"那段没看清"。"""

        writer = BehaviorDocumentWriter(
            self.site.behavior_tree, ProcessLocalLockStore(), clock=lambda: at(day, 23, 59)
        )
        writer.publish(
            BehaviorKind.GAP,
            gap_payload(
                occurred_on=day,
                started_at=at(day, start_hour, start_minute),
                ended_at=at(day, end_hour, end_minute),
                gap_kind=kind,
            ),
        )

    def associate(
        self,
        uri: str,
        *,
        kind: str,
        context: str,
        situation: str | None = None,
        causes: tuple[str, ...] = (),
        consumed: tuple[tuple[str, str], ...] = (),
        left: tuple[tuple[str, str], ...] = (),
    ) -> None:
        """往规律树写这次发生的关联记录，并给那一天打完成标记。"""

        associate(
            self.regularity,
            uri,
            kind=kind,
            context=context,
            situation=situation,
            causes=causes,
            consumed=consumed,
            left=left,
        )

    def associated(self, *days: date) -> AssociatedDays:
        """脚本化的事实源：不管规律树上有什么，直接说这几天关联完成了。"""

        done = frozenset(days)
        return lambda _kind: done

    def associated_days(self) -> AssociatedDays:
        """真实的事实源：规律树上有完成标记的日子（按当前关联版本）。"""

        return Ledger(self.regularity, ASSOCIATION_VERSION).days_for

    def tree(self, **overrides) -> PredictionTree:
        snapshot = source.read(self.site.behavior_tree)
        latest = snapshot.latest_day
        assert latest is not None
        return builder.build(
            snapshot,
            config=config(**overrides),
            reference=latest,
            built_at=datetime(2026, 12, 31, tzinfo=UTC),
        )

    def cache(self) -> DayIndexCache:
        return DayIndexCache(self.site.behavior_tree, subject=SUBJECT)

    def pack(
        self,
        now: datetime,
        *,
        unsealed: Sequence[UnsealedRow] = (),
        half_width: int = 3,
        window_days: int = 30,
        max_days: int = 40,
        tree: PredictionTree | None = None,
    ) -> EvidencePack:
        """走真实读口装一包：规律树上的记录按关联版本读，与 associated_days 同一把尺子。"""

        resolved = tree if tree is not None else self.tree()
        moment = moment_at(now, slot_minutes=resolved.slot_minutes)
        return assemble(
            resolved,
            moment,
            self.cache(),
            generation="test-generation",
            unsealed=unsealed,
            glosses_for=lambda kind, days: association_glosses(self.regularity, kind, days, version=ASSOCIATION_VERSION),
            situations_for=lambda kind, weekday: situations_of(self.regularity, kind, weekday=weekday),
            associated=self.associated_days(),
            half_width=half_width,
            window_days=window_days,
            transition_window_seconds=7_200.0,
            max_days_per_layer=max_days,
        )


class Ledger:
    """规律树上的完成标记按版本读——与刷新器的 ``associated_days`` 同形状（``AssociationLedger``）。"""

    def __init__(self, tree: RegularityTree, version: str) -> None:
        self.tree = tree
        self.version = version

    def days_for(self, kind_token: str) -> frozenset[date]:
        return self.tree.days_for(kind_token, version=self.version)


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

__all__ = ["MONDAY", "SLOT_MINUTES", "Ground", "Ledger", "ScriptedJudge", "at", "config", "slot_of"]
