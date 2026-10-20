"""节律：一个行为概念做的那天几点做、多久一次（裁定 59 第 6 条第 4、5 步）。

**读它的两处**：触点①写概念时看这个类的作息（"一天三次"就不该定成一个概念）；常态按峰分份（``occurrences.baselines``）。

**从哪来**：``temporal`` 按"做的那天几点做"算日内分布与峰，按观测覆盖算密度；这里只把一个概念（基础概念是它那一个类，
汇总概念是成员类合起来的发生）交给它、再翻成给人看的形状。原来从预测树的周几曲线取，预测树删了（裁定 59）。
周几、工作日这类"锚在哪种日子上"不在这里猜：读语义树上已经成立的日历关系（``anchors``，由调用方给）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from habitus.scene.concepts.model import MINUTES_PER_DAY, ConceptError, ConceptSet, concept_identity
from habitus.series.model import EventSeries, SeriesRecord
from habitus.temporal import Rulers, coverage_of, kind_reading

#: 一天最多认几个峰。一天十几个峰的行为（操作手机）多半是概念定得太宽，不是真有十几次机会——保护闸，超了就截断。
MAX_RHYTHM_PEAKS = 12
#: 节律那一行的写法版本：措辞或口径改了就升，随写概念的版本一起进缓存键（提示词指纹只摘系统提示词，不含材料）。
RHYTHM_RENDER_VERSION = "rhythm_render_v2"


@dataclass(frozen=True)
class RhythmPeak:
    """做的那天里的一个高峰：第几个、几点到几点、做的那几次里有多大比例落在这里。"""

    ordinal: int
    start_minute: int
    end_minute: int
    share: float

    def __post_init__(self) -> None:
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 1:
            raise ConceptError("a rhythm peak's ordinal counts from 1")
        for label in ("start_minute", "end_minute"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConceptError(f"rhythm peak {label} must be an integer minute of day")
        # 跨午夜的峰（就寝 23:00–01:00）起点在当天、终点过 24:00：终点最多到次日同一刻。
        if (
            not 0 <= self.start_minute < MINUTES_PER_DAY
            or not self.start_minute < self.end_minute <= self.start_minute + MINUTES_PER_DAY
        ):
            raise ConceptError("a rhythm peak starts inside one day and ends within 24 hours of its start")
        if isinstance(self.share, bool) or not isinstance(self.share, int | float) or not 0.0 < self.share <= 1.0:
            raise ConceptError("rhythm peak share lies in (0, 1]")

    @property
    def wraps_midnight(self) -> bool:
        return self.end_minute > MINUTES_PER_DAY

    @property
    def label(self) -> str:
        tail = "（次日）" if self.wraps_midnight else ""
        return f"{_clock(self.start_minute)}–{_clock(self.end_minute)}{tail}"

    def render(self) -> str:
        return f"{self.label}（占 {self.share:.0%}）"


@dataclass(frozen=True)
class Rhythm:
    """一个行为概念的作息：做的那天的几个峰、平均多久一次、发生日占比、锚在哪种日子上（已成立的日历关系）。

    ``peaks`` 空表示看不出一天里固定在几点（太少、或时刻散）。密度两个数取不到就是 None。
    """

    concept: str
    peaks: tuple[RhythmPeak, ...] = ()
    days_per_occurrence: float | None = None
    occurrence_day_share: float | None = None
    anchors: tuple[str, ...] = ()
    #: 给模型看的名字（基础概念的 ``concept`` 是类编号，显示用类名）；不给就用 ``concept``。
    label: str | None = None

    def __post_init__(self) -> None:
        concept_identity(self.concept)
        peaks = tuple(self.peaks)
        if any(not isinstance(peak, RhythmPeak) for peak in peaks):
            raise ConceptError("rhythm peaks must be RhythmPeak values")
        if len(peaks) > MAX_RHYTHM_PEAKS:
            raise ConceptError(f"a rhythm carries at most {MAX_RHYTHM_PEAKS} peaks")
        if [peak.ordinal for peak in peaks] != list(range(1, len(peaks) + 1)):
            raise ConceptError("rhythm peak ordinals are 1..n in clock order")
        for earlier, later in zip(peaks, peaks[1:], strict=False):
            if later.start_minute < earlier.end_minute:
                raise ConceptError("rhythm peaks must not overlap")
        object.__setattr__(self, "peaks", peaks)
        if (self.days_per_occurrence is None) != (self.occurrence_day_share is None):
            raise ConceptError("the two density numbers come together")
        if self.days_per_occurrence is not None and (
            self.days_per_occurrence <= 0.0
            or self.occurrence_day_share is None
            or not 0.0 < self.occurrence_day_share <= 1.0
        ):
            raise ConceptError("density numbers must be positive and the day share at most 1")
        object.__setattr__(self, "anchors", tuple(self.anchors))

    def render(self) -> str:
        """给提示词的一行：做的那天几点做（不是"随便哪天"），多久一次，锚在哪种日子上。"""

        name = self.label or self.concept
        parts = [
            f"做的那天多在 {'、'.join(peak.render() for peak in self.peaks)}"
            if self.peaks
            else "看不出一天里固定在几点"
        ]
        if self.days_per_occurrence is not None and self.occurrence_day_share is not None:
            parts.append(
                f"平均 {self.days_per_occurrence:.1f} 天一次，在看的日子里 {self.occurrence_day_share:.0%} 有它"
            )
        if self.anchors:
            parts.append(f"锚在{'、'.join(self.anchors)}")
        return f"{name}：{'；'.join(parts)}"


#: 算节律用的尺子，由组合根从 ``config.temporal`` 抄过来（语义树不读配置）；就是 ``temporal`` 那一份。
RhythmRulers = Rulers


def rhythms_of(
    series: EventSeries,
    concepts: ConceptSet,
    rulers: RhythmRulers,
    *,
    anchors: Mapping[str, Sequence[str]] | None = None,
) -> Mapping[str, Rhythm]:
    """每个行为概念的节律。概念的发生 = 它那几个类（基础 / 细分是自己的类，汇总是成员类）的记录合在一起再扎堆；
    一次都没有的概念不在结果里（调用方按"看不出"处理）。``anchors``：概念 → 已成立的日历关系说的那几种日子。"""

    coverage = {}
    found: dict[str, Rhythm] = {}
    for identity in concepts.behaviors():
        classes = set(concepts.classes_of(identity))
        records: list[SeriesRecord] = [
            record for record in series.records if record.classified and record.kind_token in classes
        ]
        if not records:
            continue
        lane = records[0].lane
        if lane not in coverage:
            coverage[lane] = coverage_of(series, lane, slot_minutes=rulers.slot_minutes)
        reading = kind_reading(records, coverage[lane], rulers)
        profile, density = reading.profile, reading.density
        peaks: tuple[RhythmPeak, ...] = ()
        if profile is not None:
            peaks = tuple(
                RhythmPeak(
                    ordinal=ordinal,
                    start_minute=peak.first_slot * rulers.slot_minutes,
                    end_minute=(peak.last_slot + 1) * rulers.slot_minutes,
                    share=min(1.0, peak.share),
                )
                for ordinal, peak in enumerate(profile.peaks[:MAX_RHYTHM_PEAKS], start=1)
            )
        found[identity] = Rhythm(
            concept=identity,
            peaks=peaks,
            days_per_occurrence=None if density is None else density.days_per_occurrence,
            occurrence_day_share=None if density is None else density.occurrence_day_share,
            anchors=tuple((anchors or {}).get(identity, ())),
            label=concepts[identity].label,
        )
    return MappingProxyType(found)


def _clock(minute_of_day: int) -> str:
    hour, minute = divmod(minute_of_day % MINUTES_PER_DAY, 60)
    return f"{hour:02d}:{minute:02d}"


__all__ = [
    "MAX_RHYTHM_PEAKS",
    "RHYTHM_RENDER_VERSION",
    "Rhythm",
    "RhythmPeak",
    "RhythmRulers",
    "rhythms_of",
]
