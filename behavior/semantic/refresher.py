"""按日 → 月 → 年自下而上刷新行为树的 L0/L1。

与 memory 语义刷新器同构，但有一处结构性简化：行为树是**单写入方**（归约 runner，且本刷新器
只在归约 sweep 的 behavior-root fenced 锁内被调用），目录在生成期间不可能变化——memory 那套
"快照 → 锁外生成 → 锁内比对 digest → 变了重来"的稳定快照重试整体省掉。调用方不持锁属于误用。

分工（用户裁定）：

- **occurrences 日目录的 overview 就是"这一天"的正典摘要**：快照除本目录的行为文档外，并入
  同日 gaps 目录的空白段——空白如实写进当日叙述；月/年从各日 L0 上卷，由 LLM 生成叙述。
- **gaps 层级每级也自带摘要**，但内容是空白段的枚举，无语义判断——确定性渲染，零模型调用。

成本控制：overview 尾部嵌一行来源 digest 注释；来源未变即 UNCHANGED、零模型调用、零写入。
刷新失败不阻塞归约主流程（摘要是可重建派生物），由调用方把异常降级成信号。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, datetime, timezone

from behavior.model import BehaviorAddress, BehaviorDirectory, BehaviorKind, BehaviorLevel
from behavior.semantic.config import BehaviorSemanticConfig
from behavior.semantic.generator import BehaviorOverviewGenerator
from behavior.semantic.model import (
    BehaviorDirectorySnapshot,
    BehaviorSemanticEntry,
    BehaviorSemanticEntryKind,
    BehaviorSemanticRefreshResult,
    BehaviorSemanticRefreshStatus,
)
from behavior.tree import BehaviorTree
from foundation.integrity import canonical_digest

_SOURCE_MARKER = re.compile(r"<!-- habitus-semantic-source: ([0-9a-f]{64}) -->")
_PAGE_LIMIT = 10_000


class BehaviorSemanticRefreshError(RuntimeError):
    """生成结果无法安全发布。"""


class BehaviorSemanticRefresher:
    """读取 L2 与子目录 L0，生成 L1 后确定性派生 L0。"""

    def __init__(
        self,
        tree: BehaviorTree,
        generator: BehaviorOverviewGenerator,
        *,
        config: BehaviorSemanticConfig | None = None,
    ) -> None:
        if not isinstance(tree, BehaviorTree):
            raise TypeError("tree must be a BehaviorTree")
        if not callable(getattr(generator, "generate", None)):
            raise TypeError("generator must implement generate(snapshot)")
        resolved = config or BehaviorSemanticConfig()
        if not isinstance(resolved, BehaviorSemanticConfig):
            raise TypeError("config must be BehaviorSemanticConfig")
        self.tree = tree
        self.generator = generator
        self.config = resolved

    async def refresh_days(
        self, days: Iterable[date]
    ) -> tuple[BehaviorSemanticRefreshResult, ...]:
        """刷新受影响的日目录及其月/年祖先；自下而上，同一目录一轮只刷一次。"""

        affected = sorted(set(days))
        if any(not isinstance(day, date) for day in affected):
            raise TypeError("days must contain date values")
        if not affected:
            return ()
        occurrences = self._addresses_by_date(BehaviorKind.OCCURRENCE)
        gaps = self._addresses_by_date(BehaviorKind.GAP)

        results: list[BehaviorSemanticRefreshResult] = []
        for day in affected:
            results.append(await self._refresh_occurrence_day(day, occurrences, gaps))
            results.append(self._refresh_gap_day(day, gaps))
        for year, month in sorted({(day.year, day.month) for day in affected}):
            occ_days = sorted({d for d in occurrences if (d.year, d.month) == (year, month)})
            gap_days = sorted({d for d in gaps if (d.year, d.month) == (year, month)})
            results.append(
                await self._refresh_rollup(
                    BehaviorDirectory.occurrences(year, month),
                    tuple(
                        BehaviorDirectory.occurrences(d.year, d.month, d.day) for d in occ_days
                    ),
                )
            )
            results.append(
                self._refresh_gap_rollup(
                    BehaviorDirectory.gaps(year, month),
                    tuple(BehaviorDirectory.gaps(d.year, d.month, d.day) for d in gap_days),
                )
            )
        for year in sorted({day.year for day in affected}):
            occ_months = sorted({(d.year, d.month) for d in occurrences if d.year == year})
            gap_months = sorted({(d.year, d.month) for d in gaps if d.year == year})
            results.append(
                await self._refresh_rollup(
                    BehaviorDirectory.occurrences(year),
                    tuple(BehaviorDirectory.occurrences(y, m) for y, m in occ_months),
                )
            )
            results.append(
                self._refresh_gap_rollup(
                    BehaviorDirectory.gaps(year),
                    tuple(BehaviorDirectory.gaps(y, m) for y, m in gap_months),
                )
            )
        return tuple(results)

    # ── occurrences 层级：LLM 叙述 ───────────────────────────────────────────────────

    async def _refresh_occurrence_day(
        self,
        day: date,
        occurrences: dict[date, list[BehaviorAddress]],
        gaps: dict[date, list[BehaviorAddress]],
    ) -> BehaviorSemanticRefreshResult:
        directory = BehaviorDirectory.occurrences(day.year, day.month, day.day)
        if not self.tree.directory_exists(directory):
            return BehaviorSemanticRefreshResult(directory, BehaviorSemanticRefreshStatus.MISSING)
        # 当日叙述的输入 = 行为 + 同日空白，按开始瞬时排成一条时间轴（叙述要按时间讲）。
        timeline = sorted(
            (
                *(
                    (address, BehaviorSemanticEntryKind.OCCURRENCE)
                    for address in occurrences.get(day, ())
                ),
                *((address, BehaviorSemanticEntryKind.GAP) for address in gaps.get(day, ())),
            ),
            key=lambda pair: (_instant(pair[0].started_at), pair[0].identity_name),
        )
        collected: list[BehaviorSemanticEntry] = []
        for address, kind in timeline:
            document = self.tree.read(address)
            if (
                kind is BehaviorSemanticEntryKind.OCCURRENCE
                and document.fields.get("original_name") is not None
            ):
                # 撞车消歧的重复记录（original_name 非空 = 已知重复）：死规则②裁定语义层
                # 一律不计入——机械认标记跳过，不让同一行为在当日叙述里出现两遍。
                continue
            collected.append(
                BehaviorSemanticEntry(
                    name=f"{address.identity_name}.md",
                    kind=kind,
                    content=document.markdown_body,
                )
            )
        entries = tuple(collected)
        if not entries:
            return BehaviorSemanticRefreshResult(directory, BehaviorSemanticRefreshStatus.EMPTY)
        return await self._publish_generated(directory, entries)

    async def _refresh_rollup(
        self, directory: BehaviorDirectory, children: tuple[BehaviorDirectory, ...]
    ) -> BehaviorSemanticRefreshResult:
        """月/年概览：输入是各子目录的 L0 摘要。"""

        if not self.tree.directory_exists(directory):
            return BehaviorSemanticRefreshResult(directory, BehaviorSemanticRefreshStatus.MISSING)
        entries = tuple(
            BehaviorSemanticEntry(
                name=child.identity_parts[-1],
                kind=BehaviorSemanticEntryKind.DIRECTORY,
                content=self.tree.read_layer(child, BehaviorLevel.ABSTRACT),
            )
            for child in children
            if self.tree.layer_exists(child, BehaviorLevel.ABSTRACT)
        )
        if not entries:
            return BehaviorSemanticRefreshResult(directory, BehaviorSemanticRefreshStatus.EMPTY)
        return await self._publish_generated(directory, entries)

    async def _publish_generated(
        self, directory: BehaviorDirectory, entries: tuple[BehaviorSemanticEntry, ...]
    ) -> BehaviorSemanticRefreshResult:
        snapshot = BehaviorDirectorySnapshot(directory=directory, entries=entries)
        if self._source_digest(directory) == snapshot.digest:
            return BehaviorSemanticRefreshResult(
                directory, BehaviorSemanticRefreshStatus.UNCHANGED, snapshot.digest
            )
        overview = await self.generator.generate(snapshot)
        if not isinstance(overview, str) or not overview.strip():
            raise BehaviorSemanticRefreshError("behavior overview generator returned empty text")
        return self._write(directory, overview, snapshot.digest)

    # ── gaps 层级：确定性枚举，零模型调用 ─────────────────────────────────────────────

    def _refresh_gap_day(
        self, day: date, gaps: dict[date, list[BehaviorAddress]]
    ) -> BehaviorSemanticRefreshResult:
        directory = BehaviorDirectory.gaps(day.year, day.month, day.day)
        if not self.tree.directory_exists(directory):
            return BehaviorSemanticRefreshResult(directory, BehaviorSemanticRefreshStatus.MISSING)
        day_gaps = sorted(
            gaps.get(day, ()),
            key=lambda address: (_instant(address.started_at), address.identity_name),
        )
        if not day_gaps:
            return BehaviorSemanticRefreshResult(directory, BehaviorSemanticRefreshStatus.EMPTY)
        lines = []
        for address in day_gaps:
            fields = self.tree.read(address).fields
            lines.append(f"- {fields['started_at']} — {fields['ended_at']}（{address.name}）")
        overview = (
            f"# {day.isoformat()} 观测空白\n\n本日记录 {len(lines)} 段观测空白：\n\n"
            + "\n".join(lines)
            + "\n"
        )
        return self._publish_deterministic(directory, overview)

    def _refresh_gap_rollup(
        self, directory: BehaviorDirectory, children: tuple[BehaviorDirectory, ...]
    ) -> BehaviorSemanticRefreshResult:
        if not self.tree.directory_exists(directory):
            return BehaviorSemanticRefreshResult(directory, BehaviorSemanticRefreshStatus.MISSING)
        rows = [
            (child.identity_parts[-1], self.tree.read_layer(child, BehaviorLevel.ABSTRACT))
            for child in children
            if self.tree.layer_exists(child, BehaviorLevel.ABSTRACT)
        ]
        if not rows:
            return BehaviorSemanticRefreshResult(directory, BehaviorSemanticRefreshStatus.EMPTY)
        title = "-".join(directory.identity_parts[1:])
        body = "\n".join(f"- {name}：{abstract.strip()}" for name, abstract in rows)
        return self._publish_deterministic(directory, f"# {title} 观测空白\n\n{body}\n")

    def _publish_deterministic(
        self, directory: BehaviorDirectory, overview: str
    ) -> BehaviorSemanticRefreshResult:
        digest = canonical_digest({"overview": overview})
        if self._source_digest(directory) == digest:
            return BehaviorSemanticRefreshResult(
                directory, BehaviorSemanticRefreshStatus.UNCHANGED, digest
            )
        return self._write(directory, overview, digest)

    # ── 机械件 ───────────────────────────────────────────────────────────────────────

    def _write(
        self, directory: BehaviorDirectory, overview: str, digest: str
    ) -> BehaviorSemanticRefreshResult:
        normalized = overview.strip() + "\n"
        if len(normalized) > self.config.max_overview_chars:
            raise BehaviorSemanticRefreshError("behavior overview exceeds its configured bound")
        stamped = f"{normalized}\n<!-- habitus-semantic-source: {digest} -->\n"
        self.tree.write_layers(
            directory, abstract=self._abstract_from_overview(normalized), overview=stamped
        )
        return BehaviorSemanticRefreshResult(
            directory, BehaviorSemanticRefreshStatus.WRITTEN, digest
        )

    def _source_digest(self, directory: BehaviorDirectory) -> str | None:
        """读上次写入的来源 digest：**只认最后一个非空行**。

        narrative 是模型产出、其输入是不受信的观测语义——若正文中被诱导出现同形注释，
        用 search 取首个匹配会被污染（每轮误判"变了"白烧模型，或伪造 UNCHANGED 冻结摘要）。
        真 footer 由 ``_write`` 固定拼在末行。
        """

        if not self.tree.layer_exists(directory, BehaviorLevel.OVERVIEW):
            return None
        for line in reversed(
            self.tree.read_layer(directory, BehaviorLevel.OVERVIEW).splitlines()
        ):
            stripped = line.strip()
            if not stripped:
                continue
            match = _SOURCE_MARKER.fullmatch(stripped)
            return match.group(1) if match else None
        return None

    def _abstract_from_overview(self, overview: str) -> str:
        """L0 = L1 首段的确定性压缩（沿 memory 同一规则）。"""

        content: list[str] = []
        started = False
        for line in overview.splitlines():
            stripped = line.strip()
            if not started:
                if not stripped or stripped.startswith("#"):
                    continue
                started = True
            if stripped.startswith("##"):
                break
            if stripped:
                content.append(stripped)
        compact = " ".join(content)
        if not compact:
            raise BehaviorSemanticRefreshError("behavior overview has no leading paragraph")
        if len(compact) > self.config.max_abstract_chars:
            compact = compact[: self.config.max_abstract_chars - 1].rstrip() + "…"
        return compact + "\n"

    def _addresses_by_date(self, kind: BehaviorKind) -> dict[date, list[BehaviorAddress]]:
        """全树按日分组；全量扫描是 BHV-LIFECYCLE-001 的已知欠账，随时间窗读取一并改。"""

        grouped: dict[date, list[BehaviorAddress]] = {}
        cursor: BehaviorAddress | None = None
        while True:
            page = self.tree.list_addresses(kind, limit=_PAGE_LIMIT, after=cursor)
            for address in page:
                grouped.setdefault(address.occurred_on, []).append(address)
            if len(page) < _PAGE_LIMIT:
                return grouped
            cursor = page[-1]


def _instant(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


__all__ = ["BehaviorSemanticRefreshError", "BehaviorSemanticRefresher"]
