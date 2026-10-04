"""把读数落成 ``scene/views/`` 下给人看的 Markdown。无写入者、不是权威：每次整个重写，旧文件不在新集合里就删。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

from habitus.foundation.integrity import canonical_json
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.hypotheses.model import ASPECT_LABELS, Aspect, Hypothesis, TypePrior
from habitus.scene.storage import SceneStore
from habitus.scene.views.behaviours import BehaviourSide, BehaviourView
from habitus.scene.views.people import EntitySlice, ProfileView
from habitus.scene.views.relations import FulfilmentReading, LayerReading, RelationReading, Strength, TypeReading
from habitus.scene.views.residue import ResidueCandidate

VIEWS_SEGMENT = "views"
DONE_FILENAME = ".done.json"
MAX_VIEW_BYTES = 1024 * 1024
_TYPE_LABELS = {TypeReading.NONE: "无", TypeReading.INHIBITING: "抑制", TypeReading.ENABLING: "使能", TypeReading.PROMOTING: "促进"}
#: 基准猜的类型也印中文：这份文件给人读、也给判决 LLM 读，同一行里读数是"抑制"而先验是 "inhibiting"，
#: 读的人得自己翻一遍（评审 C-17 的另一半）。
_PRIOR_LABELS = {TypePrior.ENABLING: "使能", TypePrior.PROMOTING: "促进", TypePrior.INHIBITING: "抑制"}


class ViewsStoreError(ValueError):
    """投影目录的路径逃逸或写入失败。"""


class ViewsStore(SceneStore):
    error_type: ClassVar[type[ValueError]] = ViewsStoreError

    def __init__(self, scene_root: str | Path, **options: int) -> None:
        super().__init__(scene_root, **options)
        self.directory = self._inside(Path(VIEWS_SEGMENT))

    def replace_all(self, files: Mapping[str, str], *, now: datetime) -> tuple[Path, ...]:
        """写全部视图文件（相对 ``views/`` 的路径 → 文本），删掉不在集合里的旧 ``.md`` 与空目录，最后落完成标记。

        标记先撤再落：中途崩掉留下的是"没做完"（没有标记，读侧知道这份不完整），而不是"一部分新一部分旧"却盖着章。
        """

        marker = self._inside(Path(VIEWS_SEGMENT, DONE_FILENAME))
        self._discard(marker)
        written: list[Path] = []
        for relative, text in sorted(files.items()):
            path = self._inside(Path(VIEWS_SEGMENT, relative))
            self._ensure_directory(path.parent)
            self._atomic_write(path, text.encode("utf-8"), maximum=MAX_VIEW_BYTES)
            written.append(path)
        keep = {path for path in written}
        for existing in self._markdown_files(self.directory):
            if existing not in keep:
                self._discard(existing)
        self._prune_empty(self.directory)
        payload = canonical_json({"files": len(written), "completed_at": now.astimezone(UTC)}).encode("utf-8")
        self._ensure_directory(self.directory)
        self._atomic_write(marker, payload, maximum=MAX_VIEW_BYTES)
        return tuple(written)

    def is_complete(self) -> bool:
        """上一次 ``replace_all`` 有没有走完。"""

        return self._file_exists(self._inside(Path(VIEWS_SEGMENT, DONE_FILENAME)))

    def _prune_empty(self, directory: Path) -> None:
        for entry in self._children(directory):
            if entry.is_dir():
                child = directory / entry.name
                self._prune_empty(child)
                if not self._children(child):
                    child.rmdir()

    def _markdown_files(self, directory: Path) -> list[Path]:
        found: list[Path] = []
        for entry in self._children(directory):
            if entry.is_dir():
                found.extend(self._markdown_files(directory / entry.name))
            elif entry.is_file() and entry.name.endswith(".md"):
                found.append(directory / entry.name)
        return found


def render_relation(reading: RelationReading, hypothesis: Hypothesis) -> str:
    kind = "无节律型" if hypothesis.is_open_ended else "节律型"
    lines = [
        f"# {hypothesis.label()}",
        "",
        f"假设：`{reading.hypothesis_identity}` · 方面 {ASPECT_LABELS[reading.aspect]} · {kind} · 落在 {hypothesis.opportunity_label}",
        "",
    ]
    if reading.fulfilment is not None:
        lines.append(f"兑现：{_fulfilment(reading.fulfilment)}")
        lines.append("（无节律型只读兑现：后件没节律，「比平时多几成」与使能/促进的分法都无从谈起，对照只有「它本来多久来一次」）")
    else:
        lines.append(f"强度：{_strength(reading.strength)}")
    if reading.fulfilment is None and reading.strength.interval is None and reading.fallback is not None:
        lines.append(f"退回上一级：`{reading.fallback.identity}`（留一法后）{_strength(reading.fallback.strength)}")
    if reading.strength.interval is not None or (reading.fulfilment is not None and reading.fulfilment.interval is not None):
        # R-3（用户 09-27 裁定"区间承认不修天间相关，渲染上标出"）：块 bootstrap 的块 = max(1 天, 锚间隔 p50)，
        # 一天一次的前件每条自成一块，成簇（AR(1) ρ=0.6）时实测覆盖 64% vs 名义 90%。修法等真实数据看自相关再定。
        lines.append("注：区间未修天与天之间的相关（前件成簇时它偏窄），成簇修正等真实数据看自相关再定")
    if reading.type_reading is not None:
        readout = reading.type_reading
        numbers = "" if readout.pn is None or readout.ps is None else f"，PN={readout.pn:.2f} PS={readout.ps:.2f}（外生 + 单调下的界）"
        lines.append(f"类型：{_TYPE_LABELS[readout.reading]}（读自账{numbers}；基准猜的是 {_PRIOR_LABELS[hypothesis.type_prior] if hypothesis.type_prior else '—'}）")
    if reading.strength.agrees_with_prior is not None:
        lines.append(f"与基准方向：{'一致' if reading.strength.agrees_with_prior else '相反'}")
    # 「显著」与「稳定」是两件事，分开印（用户 09-27）：显著 = 这个数不等于零；稳定 = 换了条件还成不成立。
    if reading.uncalibrated:
        lines.append("显著：未校准（无节律型没有平时概率可比，攒够就必然\"显著\"；收口规则与双向验证做完前不作数，2026-09-30 裁定五）")
    else:
        lines.append(f"显著：{'是' if reading.significant else '否'}")
    if reading.strength.total_passes:
        share = reading.strength.skipped_passes / reading.strength.total_passes
        caveat = "（只剔\"没来\"的机会，来了的从不剔，占比高时实际率偏高）" if share >= 0.1 else ""
        lines.append(f"没看清而剔掉的机会：{reading.strength.skipped_passes}/{reading.strength.total_passes}（{share:.0%}）{caveat}")
    lines.append(f"稳定性：{_stability(reading)}")
    lines.append(f"未结算：{reading.open_claims} 条")
    if reading.stale_fingerprints:
        lines.append(f"⚠ 账里有按旧口径（指纹 {', '.join(reading.stale_fingerprints)}）开的承诺：这条假设改过第几次机会或方向，读数不可信")
    if reading.mixed_control_generations:
        lines.append(f"⚠ 对照跨了两代预测树（{', '.join(reading.mixed_control_generations)}）：这一批承诺开在树重建的两边")
    stability = reading.stability
    if stability.tested or stability.skipped:
        lines += ["", "## 情境（对照未按情境分，刀 5 前的诚实标注）"]
        for moderation in stability.moderations:
            lines.append(f"- 被「{moderation.situation}」调节：有 {_reading(moderation.with_situation)} ｜ 无 {_reading(moderation.without_situation)}")
        untouched = [name for name in stability.tested if name not in {m.situation for m in stability.moderations}]
        if untouched:
            lines.append(f"- 测过、没分开：{' / '.join(untouched)}")
        if stability.skipped:
            lines.append(f"- 太少不下结论：{' / '.join(stability.skipped)}")
    if reading.dose or reading.unordered_dose:
        lines += ["", "## 剂量"]
        for dose in reading.dose:
            grades = " · ".join(f"{grade} {_reading(strength)}" for grade, strength in dose.by_grade)
            trend = "单调，可信度升一档" if dose.monotonic else ("档不够三个，不谈趋势" if len(dose.by_grade) < 3 else "不单调")
            lines.append(f"- {dose.concept}：{grades} → {trend}")
        for concept in reading.unordered_dose:
            # 不静默：档跨午夜时阶梯排出来是"凌晨在深夜之前"，那条趋势是假的，所以不出数、说明为什么。
            lines.append(f"- {concept}：档跨午夜（22:00–02:00 这一类），阶梯排不出顺序，这一项不出数")
    reminded, quiet = reading.reminded
    # 从没提醒过就不印这一节（评审 C-17）：否则每条关系都多一行"提醒过：还没攒够 0 次"。
    if reminded.accumulation.count:
        lines += ["", "## 干预分层", f"- 提醒过：{_reading(reminded)} ｜ 没提醒：{_reading(quiet)}"]
    if reading.shared:
        lines += ["", "## 共享证据（不是独立确认）"]
        for shared in reading.shared:
            verdict = "分不开（有一层没攒够）" if shared.separable is None else ("在场时确实不一样" if shared.separable else "在不在场差不多")
            lines.append(f"- `{shared.other}` 共用 {shared.shared_claims} 次证据：它在场 {_reading(shared.with_other)} ｜ 不在场 {_reading(shared.without_other)} → {verdict}")
    return "\n".join(lines) + "\n"


def _reading(reading: LayerReading) -> str:
    """一层的读数渲染：按类型分派。无节律型那一层是兑现率，节律型是强度。"""

    return _fulfilment(reading) if isinstance(reading, FulfilmentReading) else _strength(reading)


def _fulfilment(reading: FulfilmentReading) -> str:
    """无节律型的一行：兑现率（带区间）、到达者中位、对照、最久立了多久、分母的构成。"""

    acc = reading.accumulation
    parts: list[str] = []
    if reading.interval is None or reading.rate is None:
        if acc.sufficient:
            parts.append(f"攒够了（{acc.count} 次 · 跨 {acc.blocks} 块）但读不出数：兑现率没有区间")
        else:
            parts.append(f"还没攒够：{acc.count} 次 · 跨 {acc.blocks} 块（块长 {acc.block_hours / 24:.1f} 天），还差 {acc.short_by_count} 次 / {acc.short_by_blocks} 块")
    else:
        parts.append(f"{reading.rate:.0%} [{reading.interval.low:.0%}, {reading.interval.high:.0%}] · n={acc.count} · 跨 {acc.blocks} 块")
    parts.append(f"兑现 {reading.fulfilled} · 释放 {reading.released} · 关掉 {reading.closed} · 仍立着 {reading.standing_total}（其中 {reading.standing} 已过对照间隔）")
    if reading.median_hours is not None:
        parts.append(f"到达者中位 {reading.median_hours:.0f} 小时")
    parts.append("无对照（快照给不出「本来多久一次」）" if reading.control_hours is None else f"对照：本来约 {reading.control_hours:.0f} 小时才发生一次")
    if reading.longest_standing_hours is not None and reading.standing_total:
        parts.append(f"最久已立 {reading.longest_standing_hours:.0f} 小时")
    return " · ".join(parts)


def _strength(strength: Strength) -> str:
    acc = strength.accumulation
    tail = f" · 无对照 {strength.without_control}" if strength.without_control else ""
    if strength.interval is None:
        if acc.sufficient and strength.degenerate_point is not None:
            # 观测值全相同：区间算不出宽度，但点估计是实打实的（评审 A-14：六次都喝 3 杯不是"还没攒够"）
            unit = {"minutes": "分", "times": "次"}.get(strength.unit, strength.unit)
            digits = 1 if strength.unit == "times" else 0
            return f"{strength.degenerate_point:+.{digits}f} {unit}（{acc.count} 次全相同，区间算不出宽度）· n={acc.count} · 跨 {acc.blocks} 块{tail}"
        if acc.sufficient:
            # 过了门槛却没有区间：不是没攒够，是读不出（第 k 次机会没有风险集……），原因印出来（评审 C-14 ②）
            why = "没有一条承诺带平时概率" if strength.aspect is Aspect.PROBABILITY and strength.p1 is None else "区间算不出宽度"
            return f"攒够了（{acc.count} 次 · 跨 {acc.blocks} 块）但读不出数：{why}{tail}"
        return f"还没攒够：{acc.count} 次 · 跨 {acc.blocks} 块（块长 {acc.block_hours / 24:.1f} 天），还差 {acc.short_by_count} 次 / {acc.short_by_blocks} 块{tail}"
    interval = strength.interval
    if strength.aspect is Aspect.PROBABILITY:
        p1 = "" if strength.p1 is None or strength.control is None else f"实际 {strength.p1:.0%}（{strength.events} 次到、{strength.absent} 次没来） vs 本来 {strength.control:.0%} → "
        censored = f" · 没看清 {strength.censored}" if strength.censored else ""
        return f"{p1}{interval.point * 100:+.0f}pp [{interval.low * 100:+.0f}, {interval.high * 100:+.0f}] · n={acc.count} · 跨 {acc.blocks} 块{censored}{_latency(strength)}{tail}"
    unit = {"minutes": "分", "times": "次"}.get(strength.unit, strength.unit)
    absent = f" · 缺席 {strength.absent}" if strength.absent else ""
    # 次数留一位小数：对照是几个峰的概率之和（0.5+0.4+0.3），"+0.8 次"按整数印成 "+1 次" 就把差额说反了（评审 C-9）。
    digits = 1 if strength.unit == "times" else 0
    body = f"{interval.point:+.{digits}f} {unit} [{interval.low:+.{digits}f}, {interval.high:+.{digits}f}]"
    return f"{body} · n={acc.count} · 跨 {acc.blocks} 块{absent}{tail}"


def _stability(reading: RelationReading) -> str:
    """稳定性是“换了条件还成不成立”（ICP 那条），不是“这个数不等于零”。一个情境都没测过就明说未测。"""

    if reading.stability.moderations:
        return "被「" + "」「".join(item.situation for item in reading.stability.moderations) + "」调节"
    if reading.stability_untested:
        skipped = f"（{len(reading.stability.skipped)} 个情境太少不下结论）" if reading.stability.skipped else "（没有可分层的情境）"
    else:
        return f"测过 {len(reading.stability.tested)} 个情境，都没分开"
    return f"未测{skipped}"


def _latency(strength: Strength) -> str:
    """来了那些次从锚到后件的中位间隔。"是不做还是往后推"不在这一行答——由同一组前因的几本峰账摆在一起看（behaviours 那一面）。"""

    if strength.median_hours is None:
        return ""
    return f" · 到达者中位 {strength.median_hours:.0f} 小时"


def render_behaviour(view: BehaviourView, concepts: ConceptSet) -> str:
    """一个行为的两面：什么导致它 / 它导致了什么。前因面分组，并把份额（情境的额外贡献）摆在它下面。"""

    name = concepts[view.concept].name if view.concept in concepts else view.concept
    lines = [f"# {name}", "", "## 什么导致它（前因）", ""]
    if not view.causes:
        lines.append("（基准还没有为它写过假设）")
    if view.single_causes:
        lines.append("**单个原因**")
        lines += [line for side in view.single_causes for line in _cause_lines(side)]
    if view.combined_causes:
        lines += ["", "**几个一起（多体）**"]
        lines += [line for side in view.combined_causes for line in _cause_lines(side)]
        lines.append("注：多体那几行要的是「集合效应」对照（两件事都没发生时的本来率），而现在对照取的是树上那个峰，这一条还没做。")
    lines += ["", "## 它导致了什么（后果）", ""]
    if not view.effects:
        lines.append("（基准还没有为它写过假设）")
    for side in view.effects:
        body = _fulfilment(side.reading.fulfilment) if side.reading.fulfilment is not None else _strength(side.reading.strength)
        lines.append(f"- → {side.hypothesis.consequent} · {ASPECT_LABELS[side.hypothesis.aspect]} {side.hypothesis.opportunity_label}   {body}")
    return "\n".join(lines) + "\n"


def _compact(reading: LayerReading) -> str:
    """分层那一行用的短式：只要点估计 + 区间 + n。完整式（逐次、兑现、无对照…）属于那条关系自己的详情页。"""

    interval, count = reading.interval, reading.accumulation.count
    if interval is None:
        return f"还没攒够 n={count}"
    unit, scale = _unit_of(reading)
    return f"{interval.point * scale:+.0f}{unit} [{interval.low * scale:+.0f}, {interval.high * scale:+.0f}] n={count}"


def _unit_of(reading: LayerReading) -> tuple[str, float]:
    """这个读数的单位与倍数：概率与兑现率都是百分点，时刻是分钟，次数是次。"""

    if isinstance(reading, FulfilmentReading):
        return "pp", 100.0
    return {"probability": ("pp", 100.0), "minutes": ("分", 1.0), "times": ("次", 1.0)}.get(reading.unit, (reading.unit, 1.0))


def _cause_lines(side: BehaviourSide) -> list[str]:
    body = _fulfilment(side.reading.fulfilment) if side.reading.fulfilment is not None else _strength(side.reading.strength)
    lines = [f"- {side.hypothesis.label()} {side.hypothesis.opportunity_label}   {body}"]
    for share in side.shares:
        # 份额 = 同一本账切成两半的差（七f-3 的「调节 {A,B} vs {A,¬B}」）；
        # 真不真看两层区间有没有分开，没分开就明说看不出。
        a, b = share.with_situation, share.without_situation
        if a.interval is None or b.interval is None:
            continue
        unit, scale = _unit_of(a)
        gap = (a.interval.point - b.interval.point) * scale
        verdict = "" if a.interval.disjoint_from(b.interval) else "（区间还重叠，看不出）"
        lines.append(f"    ↳ 「{share.situation}」在场 {_compact(a)}／不在场 {_compact(b)} → 额外 {gap:+.0f}{unit}{verdict}")
    return lines


def render_residue(items: Iterable[ResidueCandidate], *, k: int, d: int) -> str:
    lines = [f"# 残差候选 · 升级判据 ≥{k} 次且跨 ≥{d} 天", ""]
    rows = list(items)
    if not rows:
        lines.append("（无）")
    for item in rows:
        status = "可升级" if item.ready else f"还差 {item.short_by_occurrences} 次 / {item.short_by_days} 天"
        lines.append(f"- {item.kind_token}：{item.occurrences} 次 · {item.days} 天（{item.first_seen} → {item.last_seen}）· {status}")
    return "\n".join(lines) + "\n"


def render_profile(profile: ProfileView, hypotheses: Mapping[str, Hypothesis]) -> str:
    lines = ["# 这个人", "", f"关系 {profile.total_relations} 条，其中未被推翻 {len(profile.stable_relations)} 条；未结算的承诺 {profile.unsettled_open_claims} 条", ""]
    lines.append("## 未被推翻的关系（过了门槛、区间不含零、没被测过的情境调节）")
    if not profile.stable_relations:
        lines.append("（还没有）")
    for reading in profile.stable_relations:
        label = hypotheses[reading.hypothesis_identity].label() if reading.hypothesis_identity in hypotheses else reading.hypothesis_identity
        body = _fulfilment(reading.fulfilment) if reading.fulfilment is not None else _strength(reading.strength)
        lines.append(f"- {label}：{body}")
    lines += ["", "注：上面的区间未修天与天之间的相关（前件成簇时它偏窄）。"]
    lines += ["作息骨架与被约束的行为清单要读预测树的曝光与率分布，随组合根接线补。"]
    return "\n".join(lines) + "\n"


def render_entity(item: EntitySlice, hypotheses: Mapping[str, Hypothesis]) -> str:
    lines = [f"# {item.concept}", ""]
    for identity, moderation in item.moderations:
        label = hypotheses[identity].label() if identity in hypotheses else identity
        lines.append(f"- {label}：在场 {_reading(moderation.with_situation)} ｜ 不在 {_reading(moderation.without_situation)}")
    return "\n".join(lines) + "\n"


def materialize_views(
    store: ViewsStore,
    *,
    hypotheses: Mapping[str, Hypothesis],
    readings: Iterable[RelationReading],
    behaviours: Iterable[BehaviourView],
    residue: Iterable[ResidueCandidate],
    profile: ProfileView,
    entities: Iterable[EntitySlice],
    concepts: ConceptSet,
    now: datetime,
    k: int,
    d: int,
) -> tuple[Path, ...]:
    files: dict[str, str] = {}
    for reading in readings:
        hypothesis = hypotheses.get(reading.hypothesis_identity)
        if hypothesis is None:
            continue
        files[f"relations/{reading.hypothesis_identity}.md"] = render_relation(reading, hypothesis)
    for view in behaviours:
        files[f"behaviours/{view.concept}.md"] = render_behaviour(view, concepts)
    files["residue/index.md"] = render_residue(residue, k=k, d=d)
    files["profile.md"] = render_profile(profile, hypotheses)
    for item in entities:
        files[f"entities/{concepts[item.concept].identity}.md"] = render_entity(item, hypotheses)
    return store.replace_all(files, now=now)


__all__ = [
    "DONE_FILENAME",
    "VIEWS_SEGMENT",
    "ViewsStore",
    "ViewsStoreError",
    "materialize_views",
    "render_behaviour",
    "render_entity",
    "render_profile",
    "render_relation",
    "render_residue",
]
