"""第 N 晚在组合根的串法：行为树只读一次，语义树与机会表用同一份序列（语义树新方案 ``13`` 第四节；预测树删了，裁定 59）。

```
重试   = RetryStage.run_once(through=昨天)                 ← 白天读不懂的按天重试一次、落树、试完就删（裁定 67、68）；出错只留观测
结算   = SettlementStage.run_once()                       ← 预测层：已到最晚时限、那天定稿的预期对着行为树结；没接时跳过
今晚 N = min(主体所在地的"今天", 最后一个已封口日子的次日)   ← 封口由归约说了算（与结算同一个口径）
序列 = series.reader.SeriesReader.read(截止日 N)         ← 行为树每晚只拼这一次（按天缓存，只重读变过的日子）；一条记录都没有就这一晚不跑
语义树 = SceneNightlyRun.run(序列)                        ← B1 同步词表 → B2 写概念 → B3 映射 → B4 关系；复现与节律由 temporal 算
机会表 = ForesightNightlyRun.run(这一晚的概念集与时间线)     ← 预测层 N1，零模型；找原因跟在它后面；没接时跳过
```

**结算排在最前**：找原因只给已经结出来的落空找，所以同一晚里结算必须在它之前——排在后面的话，今晚结出的落空要到
明晚才找原因，第二天白天核"上次的原因今天又在"就晚了一天（年假第二天照样开口）。结算只读账本、行为树与上一晚的机会表、
不读这一晚的语义树，排在最前不缺材料；这一晚其余几步因为没封口、词表迁移跳过时它照样跑（它自己也看词表迁移）。
它失败自己留观测、被这里吞掉，不许拖着后面一夜不跑。

语义树与预测层对同一条记录的读法一致，由序列保证（CI 钉住）。序列里还没有一条可用记录时语义树这一晚不跑（没有可量的东西；
原来这道闸是"预测树发布了没有"，清单 5.2）。

词表迁移做到一半时不跑（第四轮评审 E3）：迁移先改树上的行、最后才写新版本，做到一半读出来的序列一半新编号一半旧编号；
读序列前后各看一次词表状态（版本、有没有未完成的迁移），不一致就这一晚不跑语义树，下一次再来。

截止日不能只取"今天"（第四轮评审 E1）：零点刚过时昨天那条链可能还在融合静默期、没归约，夜批把昨天当成完整的一天映射、
盖完成章，之后补发进来的记录就永远不映射了。所以只用到最后一个已封口的日子为止。"今天"按主体所在地的时区算
（``zone``），不看进程默认时区。节拍由 ``runtime.nightly_worker.NightlyWorker`` 管。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from habitus.behavior.retry import RetryReport, RetryStage
from habitus.behavior.tree import BehaviorTree
from habitus.runtime.foresight_nightly import ForesightNightlyRun, OpportunityReport
from habitus.runtime.foresight_settlement import SettlementReport, SettlementStage
from habitus.runtime.scene_night import NightReport, SceneNightlyRun
from habitus.series import EventSeries
from habitus.series.reader import SeriesReader


@dataclass(frozen=True)
class NightOutcome:
    night: date
    series: EventSeries
    scene: NightReport | None
    #: 这一晚为什么没跑（没跑是 None）。
    skipped: str | None = None
    #: 预测层的机会表（没接预测层、或语义树这一晚没跑到就是 None）。
    opportunities: OpportunityReport | None = None
    #: 这一晚的结算（没接预测层、或结算失败就是 None）。
    settled: SettlementReport | None = None
    #: 结算抛了什么（异常类型名；没失败是 None）。整晚照跑，常驻节拍据它在健康面上留痕（2026-10-18 四方评审）。
    settlement_failed: str | None = None
    #: 这一晚开头的重试（没接是 None）与它抛了什么。
    retried: RetryReport | None = None
    retry_failed: str | None = None


class Nightly:
    def __init__(
        self,
        *,
        behavior_tree: BehaviorTree,
        scene: SceneNightlyRun,
        zone: ZoneInfo,
        closed_days: Callable[[], Iterable[date]],
        vocabulary: Callable[[], tuple[int, bool]],
        clock: Callable[[], datetime] | None = None,
        foresight: ForesightNightlyRun | None = None,
        settlement: SettlementStage | None = None,
        refresh_calendar: Callable[[], object] | None = None,
        series: SeriesReader | None = None,
        retry: RetryStage | None = None,
    ) -> None:
        if not isinstance(behavior_tree, BehaviorTree):
            raise TypeError("behavior_tree must be a BehaviorTree")
        if scene.behavior_tree is not behavior_tree:
            raise ValueError("the scene run must read the same behaviour tree")
        if not isinstance(zone, ZoneInfo):
            raise TypeError("zone must be a ZoneInfo")
        self.behavior_tree = behavior_tree
        if series is not None and (not isinstance(series, SeriesReader) or series.tree is not behavior_tree):
            raise ValueError("the series reader must read the same behaviour tree")
        #: 常驻的序列读者：按天缓存行为树文档，跨晚复用（26 号清单 6.7）。
        self.series = series if series is not None else SeriesReader(behavior_tree)
        self.scene = scene
        self.zone = zone
        if not callable(closed_days):
            raise TypeError("closed_days must be callable")
        self.closed_days = closed_days
        if not callable(vocabulary):
            raise TypeError("vocabulary must be callable (version, migrating)")
        #: 词表此刻的（版本, 有没有未完成的迁移）
        self.vocabulary = vocabulary
        if retry is not None and not isinstance(retry, RetryStage):
            raise TypeError("retry must be a RetryStage or None")
        self.retry = retry
        if foresight is not None and not isinstance(foresight, ForesightNightlyRun):
            raise TypeError("foresight must be a ForesightNightlyRun or None")
        self.foresight = foresight
        if settlement is not None and not isinstance(settlement, SettlementStage):
            raise TypeError("settlement must be a SettlementStage or None")
        self.settlement = settlement
        if refresh_calendar is not None and not callable(refresh_calendar):
            raise TypeError("refresh_calendar must be callable or None")
        #: 每晚开头重读一遍放假数据（读不了的年份由日历自己跳过并挂出问题，不抛）。
        self.refresh_calendar = refresh_calendar
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    def tonight(self) -> date:
        """主体所在地的"今天"（``zone``）。按进程默认时区算会差一天，每个数字看起来都还是对的。"""

        return self._clock().astimezone(self.zone).date()

    async def run_once(self) -> NightOutcome:
        if self.refresh_calendar is not None:
            await asyncio.to_thread(self.refresh_calendar)
        # 重试排在结算之前：补回来的当晚就按真实情况结、进序列（裁定 67 第 2 条）
        retried, retry_failed = await self._retry()
        settled, failed = await self._settle()
        closed = sorted(self.closed_days())
        night = self.tonight()
        if not closed:
            # 一天都还没封口（刚装上）：没有完整的一天可算，这一晚不跑
            return NightOutcome(
                night,
                EventSeries(cutoff=night),
                None,
                "一天都还没封口",
                settled=settled,
                settlement_failed=failed,
                retried=retried,
                retry_failed=retry_failed,
            )
        night = min(night, closed[-1] + timedelta(days=1))
        before = self.vocabulary()
        if before[1]:
            return NightOutcome(
                night,
                EventSeries(cutoff=night),
                None,
                "词表迁移还没做完",
                settled=settled,
                settlement_failed=failed,
                retried=retried,
                retry_failed=retry_failed,
            )
        # 全量扫行为树与写机会表是同步 IO：下沉到线程，不卡白天那一拍与同一个循环上的在线请求（2026-10-18 四方评审）
        series = await asyncio.to_thread(self.series.read, cutoff=night)
        if self.vocabulary() != before:
            return NightOutcome(
                night,
                EventSeries(cutoff=night),
                None,
                "读序列时词表变了",
                settled=settled,
                settlement_failed=failed,
                retried=retried,
                retry_failed=retry_failed,
            )
        if not series.records:
            # 只有观测空白、一条记录都没有（上游刚接入）：没有可量的东西
            return NightOutcome(
                night,
                series,
                None,
                "序列里还没有记录",
                settled=settled,
                settlement_failed=failed,
                retried=retried,
                retry_failed=retry_failed,
            )
        report = await self.scene.run(series=series, now=self._clock())
        opportunities = None
        if self.foresight is not None and report.materials is not None:
            opportunities = await asyncio.to_thread(
                self.foresight.run,
                night=series.cutoff,
                concepts=report.materials.concepts,
                timelines=report.materials.timelines,
                recurrences=report.materials.recurrences,
                coverage=report.materials.coverage,
            )
        return NightOutcome(
            night=night,
            series=series,
            scene=report,
            opportunities=opportunities,
            settled=settled,
            settlement_failed=failed,
            retried=retried,
            retry_failed=retry_failed,
        )

    def clean_leftovers(self) -> int:
        """常驻节拍一起来就调：上一晚中途被杀剩下的重试项撤回、删掉（裁定 68 第 3 条）。"""

        return 0 if self.retry is None else self.retry.clean_leftovers()

    async def _retry(self) -> tuple[RetryReport | None, str | None]:
        """重试与它抛了什么：出错不许拖着后面一夜不跑（裁定 67 第 4 条），但要交出去给健康面。"""

        if self.retry is None:
            return None, None
        try:
            # 04:00 那一趟处理前一天及更早的；零点之后新进、属于今天的等今天那一趟（裁定 68 第 3 条）
            return await self.retry.run_once(through=self.tonight() - timedelta(days=1)), None
        except Exception as exc:  # noqa: BLE001 - 不许拖着后面一夜不跑
            return None, type(exc).__name__

    async def _settle(self) -> tuple[SettlementReport | None, str | None]:
        """结算与它抛了什么。失败不许拖着后面一夜不跑，但要交出去：只吞掉的话健康面一直报好（结算自己已记观测）。"""

        if self.settlement is None:
            return None, None
        try:
            return await asyncio.to_thread(self.settlement.run_once), None
        except Exception as exc:  # noqa: BLE001 - 结算自己已观测；不许拖着后面一夜不跑
            return None, type(exc).__name__


__all__ = ["NightOutcome", "Nightly"]
