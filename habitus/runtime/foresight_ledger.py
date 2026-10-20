"""账本的落盘：按预期的六种记录各一个目录、按预期所在的日子分目录；按账的记录在 ``accounts/<账>/`` 下（裁定 48）；
按行为单元的记录在 ``units/<行为>/`` 下（裁定 61：提醒反馈、设置、档按行为）。一条一文件、写一次不改。

住在组合根：``foresight`` 不许 import infrastructure，形状与编解码在 ``foresight.ledger``，这里只管目录与字节。
同身份重写幂等（内容相同返回 False），内容不同是账本损坏。

    <root>/expectations/YYYY/MM/DD/<expectation_id>.json
    <root>/deliveries/YYYY/MM/DD/<expectation_id>--<decided_at>.json
    <root>/responses/YYYY/MM/DD/<expectation_id>--<responded_at>.json
    <root>/settlements/YYYY/MM/DD/<expectation_id>.json
    <root>/rulings/YYYY/MM/DD/<expectation_id>--<ruled_at>.json
    <root>/omissions/YYYY/MM/DD/<记录身份>.json，另有 _scanned.json 标这一天已查过漏报
    <root>/settlements/YYYY/MM/DD/_final.json        这一天结完了（全部结算、过期待决的裁决再也不会变），之后不再读
    <root>/index/expectations/<expectation_id>.json   身份 → 它在哪一天（同一身份全账本只建一条，跨天的触发靠它查重）
    <root>/accounts/<账哈希>/account.json              这本账是什么（类、触发种类、触发身份）
    <root>/accounts/<账哈希>/facts/<预期>--<种类>--<时刻>.json
    <root>/accounts/<账哈希>/levels/<at>.json
    <root>/units/<行为类哈希>/levels/<at>.json          行为单元的档的变化（按行为，与账的级分开）
    <root>/units/<行为类哈希>/settings/<at>.json        人对这个行为的设置
    <root>/units/<行为类哈希>/feedback/<预期>--<时刻>.json  说出口之后他做没做（提醒反馈，一个行为一份）

账是流水（权重每晚从单元历史数，裁定 61）；开不开口看单元：单元历史、单元的提醒反馈、单元的设置。

递送、回话、裁决都挂在它们那条预期所在的日子下，读一条预期的全部往来只开一个目录。
一个坏文件不许让整层停：按日子读回时读不出来（解析不了、形状或版本不对）的那一份记下路径、跳过，调用方把条数带进观测
（``take_unreadable``）；只想知道"自己这一次读到了哪些坏文件"的调用方传 ``unreadable=`` 自己的列表，不经共享的那一份。

"还要看的日子"（``open_days``）= 有预期、还没标结完的日子：结算、递送、提醒标记只读这些，账本再长每拍也只读最近一截；
最远看到哪天由账本自己说（时窗跨多少天、过期待决等多久裁决），不写死回看天数。
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

from habitus.foresight import ForesightError
from habitus.foresight.ledger import (
    FACT_KINDS,
    AccountFact,
    AccountLevel,
    Delivery,
    Expectation,
    Omission,
    Response,
    Ruling,
    Settlement,
    UnitFeedback,
    UnitLevel,
    UnitSetting,
    decode_delivery,
    decode_expectation,
    decode_fact,
    decode_feedback,
    decode_level,
    decode_omission,
    decode_response,
    decode_ruling,
    decode_setting,
    decode_settlement,
    decode_unit_level,
    encode_delivery,
    encode_expectation,
    encode_fact,
    encode_feedback,
    encode_level,
    encode_omission,
    encode_response,
    encode_ruling,
    encode_setting,
    encode_settlement,
    encode_unit_level,
    parse_account,
)
from habitus.foresight.ledger.compaction import AccountCompaction, decode_compaction, encode_compaction
from habitus.foundation.lifecycle import Watermark, mark
from habitus.infrastructure.store.filesystem import (
    DurableDirectoryEntry,
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_replace_bytes,
    durable_unlink,
    list_real_directory,
    read_regular_bytes,
)

_MAX_ENTRIES = 100_000
_MAX_BYTES = 1_000_000
_SCANNED = "_scanned"
_FINAL = "_final"
_MAX_UNREADABLE = 1_000
#: 一本账压实了的那一段（整份替换）。
_COMPACTION = "compaction.json"
#: 结完低水位、查过漏报的日子（26 号清单 6.11）。
_FINAL_WATERMARK = "final_through.json"
_SCAN_WATERMARK = "omissions_scanned.json"


def _fitting(payload: dict[str, Any], identity: str) -> bytes:
    """编码并查大小：写入就查（以前只在读回时查，超限的文件照样落盘，按目录读时被当坏文件跳过、结算永远结不到它，
    点读时直接抛、一条记录卡住整晚结算；26 号清单 2.5.4）。"""

    encoded = _encoded(payload)
    if len(encoded) > _MAX_BYTES:
        raise ForesightError(f"ledger record {identity} exceeds {_MAX_BYTES} bytes ({len(encoded)}); not written")
    return encoded


class ForesightLedgerStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        self._expectations = self.root / "expectations"
        self._deliveries = self.root / "deliveries"
        self._responses = self.root / "responses"
        self._settlements = self.root / "settlements"
        self._rulings = self.root / "rulings"
        self._omissions = self.root / "omissions"
        self._index = self.root / "index" / "expectations"
        self._accounts = self.root / "accounts"
        self._units = self.root / "units"
        self._unreadable: list[str] = []
        #: 标过结完的日子（标记只增不撤，记住了就不用再看盘）。
        self._finals: set[date] = set()
        #: 夜批（结算在线程里跑）与白天那一拍共用这一个实例：这两份内存状态的改动加锁（2026-10-17 四方评审）。
        self._lock = threading.Lock()

    # ── 预期 ────────────────────────────────────────────────────────────────

    def record_expectation(self, item: Expectation) -> bool:
        """同一身份全账本只写一次：索引先占位（身份 → 日子），另一天再来同一身份就是重复、不写。"""

        known = self.day_of(item.expectation_id)
        if known is not None and known != item.day:
            return False
        payload = encode_expectation(item)
        _fitting(payload, item.expectation_id)  # 超限的连索引都不占：否则留下一条指向不存在预期的索引
        # 先占索引再写预期：中间断了，重来时索引指着同一天、预期照写；反过来的顺序会让第二天按身份查不到、再建一份
        self._index_create(item.expectation_id, item.day)
        return self._create(self._expectations, item.day, item.expectation_id, payload)

    def day_of(self, expectation_id: str) -> date | None:
        """这个身份的预期在哪一天；没建过是 None。"""

        path = self._index / f"{expectation_id}.json"
        if not path.is_file():
            return None
        raw = self._load(path)
        try:
            return date.fromisoformat(str(raw["day"]))
        except (KeyError, ValueError) as exc:
            raise ForesightError(f"ledger index entry {expectation_id} is corrupt") from exc

    def has_expectation(self, expectation_id: str) -> bool:
        """这条预期真写下了：索引占了位、而且那一天的预期文件在（2026-10-17 四方评审：只看索引的话，"索引写了、预期文件还没写
        就断了"的那条，之后每拍都被当成已有、``record_expectation`` 补写那一步永远走不到，这次机会就丢了）。"""

        day = self.day_of(expectation_id)
        return day is not None and (self._day_dir(self._expectations, day) / f"{expectation_id}.json").is_file()

    def expectations_on(self, day: date, *, unreadable: list[str] | None = None) -> tuple[Expectation, ...]:
        return tuple(self._records(self._expectations, day, decode_expectation, unreadable))

    def expectation(self, day: date, expectation_id: str) -> Expectation | None:
        raw = self._read(self._expectations, day, expectation_id)
        return None if raw is None else decode_expectation(raw)

    def days_with_expectations(self, *, since: date | None = None) -> tuple[date, ...]:
        return tuple(day for day in self._days(self._expectations) if since is None or day >= since)

    def open_days(self) -> tuple[date, ...]:
        """有预期、还没标结完的日子（升序）：只看"结完低水位"以上的（26 号清单 6.11）。"""

        through = self.final_through()
        since = None if through is None else through + timedelta(days=1)
        return tuple(day for day in self.days_with_expectations(since=since) if not self.is_final(day))

    def final_through(self) -> date | None:
        """结完低水位：这一天及以前有预期的日子全部结完了（没有为 None）。"""

        raw = self._read_watermark(_FINAL_WATERMARK)
        return None if raw is None else date.fromisoformat(str(raw["through"]))

    def _advance_final_through(self) -> None:
        through = self.final_through()
        since = None if through is None else through + timedelta(days=1)
        latest = through
        for day in self.days_with_expectations(since=since):
            if not self.is_final(day):
                break
            latest = day
        if latest != through and latest is not None:
            self._write_watermark(_FINAL_WATERMARK, {"through": latest.isoformat()})

    def mark_final(self, day: date, *, at: datetime) -> bool:
        """这一天结完了：全部结算过、过期待决的裁决再也不会变。之后结算、递送、提醒标记都不再读它。"""

        payload: dict[str, Any] = {"record": "final", "day": day.isoformat(), "at": at.isoformat()}
        created = self._create(self._settlements, day, _FINAL, payload)
        with self._lock:
            self._finals.add(day)
        self._advance_final_through()
        return created

    def reopen(self, day: date) -> None:
        """撤销"这一天结完了"：重结把这一天的某条预期改回了过期待决，之后每晚还要看它有没有补做。

        先降水位、再删标记：水位以下的日子一律算结完（``is_final``），反过来删到一半被杀，这一天既不 final 又在水位以下，
        从 ``open_days`` 里永久消失（L10.4）。"""

        through = self.final_through()
        if through is not None and day <= through:
            self._write_watermark(_FINAL_WATERMARK, {"through": (day - timedelta(days=1)).isoformat()})
        durable_unlink(self._day_dir(self._settlements, day) / f"{_FINAL}.json", artifact_root=self.root)
        with self._lock:
            self._finals.discard(day)

    def is_final(self, day: date) -> bool:
        """结完了：有标记，或在结完低水位以下（水位只在那一段全部结完时推进；整天释放删到一半、标记先没了的日子
        照样算结完，不会变成"既不 final 又在水位以下"的僵尸日，L10.4）。"""

        with self._lock:
            if day in self._finals:
                return True
        through = self.final_through()
        if through is not None and day <= through:
            return True
        if (self._day_dir(self._settlements, day) / f"{_FINAL}.json").is_file():
            with self._lock:
                self._finals.add(day)
            return True
        return False

    def unsettled_expectations(self, day: date) -> tuple[Expectation, ...]:
        settled = {item.expectation_id for item in self.settlements_on(day)}
        return tuple(item for item in self.expectations_on(day) if item.expectation_id not in settled)

    # ── 递送与回话 ──────────────────────────────────────────────────────────

    def record_delivery(self, day: date, item: Delivery) -> bool:
        return self._create(
            self._deliveries, day, _stamped(item.expectation_id, item.decided_at), encode_delivery(item)
        )

    def deliveries_on(self, day: date, *, unreadable: list[str] | None = None) -> tuple[Delivery, ...]:
        return tuple(self._records(self._deliveries, day, decode_delivery, unreadable))

    def delivery_names(self, day: date) -> frozenset[str]:
        """这一天递送目录里的文件名（只增，名字集合没变 = 内容没变；读口的缓存拿它判要不要重读）。"""

        return frozenset(entry.name for entry in self._entries(self._day_dir(self._deliveries, day)))

    def record_response(self, day: date, item: Response) -> bool:
        return self._create(
            self._responses, day, _stamped(item.expectation_id, item.responded_at), encode_response(item)
        )

    def responses_on(self, day: date) -> tuple[Response, ...]:
        return tuple(self._records(self._responses, day, decode_response))

    # ── 结算与裁决 ──────────────────────────────────────────────────────────

    def record_settlement(self, item: Settlement) -> bool:
        return self._create(self._settlements, item.day, item.expectation_id, encode_settlement(item))

    def settlements_on(self, day: date) -> tuple[Settlement, ...]:
        return tuple(self._records(self._settlements, day, decode_settlement))

    def record_ruling(self, day: date, item: Ruling) -> bool:
        return self._create(self._rulings, day, _stamped(item.expectation_id, item.ruled_at), encode_ruling(item))

    def rulings_on(self, day: date) -> tuple[Ruling, ...]:
        return tuple(self._records(self._rulings, day, decode_ruling))

    # ── 漏报 ────────────────────────────────────────────────────────────────

    def record_omission(self, item: Omission) -> bool:
        """一条记录的漏报只记一次：已经记过（同一条 occurrence）就不再写——重放时内容里的 ``noticed_at`` 不同，按内容比会被当成
        账本损坏，整晚结算每晚都失败、日子永远结不完（2026-10-17 四方评审）。"""

        identity = sha256(item.occurrence_uri.encode("utf-8")).hexdigest()[:24]
        if (self._day_dir(self._omissions, item.day) / f"{identity}.json").is_file():
            return False
        return self._create(self._omissions, item.day, identity, encode_omission(item))

    def omissions_on(self, day: date) -> tuple[Omission, ...]:
        return tuple(self._records(self._omissions, day, decode_omission))

    def scan_watermark(self, *, start: date) -> Watermark[date]:
        """查过漏报的日子（26 号清单 6.11）：连续前缀 + 零散查过的；没有时从 ``start`` 的前一天起。以前每天一个
        ``_scanned.json`` 的旧标记第一次读时并进来、再删掉。"""

        raw = self._read_watermark(_SCAN_WATERMARK)
        if raw is None:
            watermark: Watermark[date] = Watermark(through=start - timedelta(days=1))
        else:
            watermark = Watermark(
                through=date.fromisoformat(str(raw["through"])),
                above=frozenset(date.fromisoformat(str(day)) for day in raw["above"]),
            )
        legacy = [day for day in self._days(self._omissions) if self._read(self._omissions, day, _SCANNED) is not None]
        if legacy:
            for day in legacy:
                watermark = mark(watermark, day, successor=_next_day) if watermark.through < day else watermark
            self._save_scan_watermark(watermark)
            for day in legacy:
                durable_unlink(self._day_dir(self._omissions, day) / f"{_SCANNED}.json", artifact_root=self.root)
        return watermark

    def mark_scanned(self, day: date, *, start: date) -> Watermark[date]:
        """这一天查过漏报了（同一天不再查第二遍）。"""

        watermark = mark(self.scan_watermark(start=start), day, successor=_next_day)
        self._save_scan_watermark(watermark)
        return watermark

    def discard_omissions_before(self, day: date) -> int:
        """漏报只给可见面看最近一段（``DeliveryStage.omission_views`` 默认 30 天）：更早的日目录整个删掉；返回删了几天。"""

        removed = 0
        for found in self._days(self._omissions):
            if found < day:
                self._discard_day_dir(self._omissions, found)
                removed += 1
        return removed

    # ── 按天释放（26 号清单 6.11；哪些天能放由 ``runtime.lifecycle_foresight`` 判） ───────────────

    def discard_index(self, day: date) -> int:
        """这一天的预期的索引项删掉（索引平铺、从不列，只能经预期拿身份）；返回删了几条。要在删预期之前删：
        反过来的话，``record_expectation`` 会按"索引在、预期不在"把预期重建出来。"""

        removed = 0
        for item in self.expectations_on(day):
            removed += durable_unlink(self._index / f"{item.expectation_id}.json", artifact_root=self.root)
        return removed

    def discard_day(self, day: date) -> None:
        """这一天的预期、递送、回话、结算、追加裁决整天删掉（先删索引）。"""

        self.discard_index(day)
        for root in (self._deliveries, self._responses, self._rulings, self._settlements, self._expectations):
            self._discard_day_dir(root, day)
        with self._lock:
            self._finals.discard(day)

    def trim_unit_levels(self, kind_token: str, *, before: datetime) -> int:
        """单元的档只留 ``before`` 之后的，外加它之前最近的一条；返回删了几个。"""

        directory = self._unit_dir(kind_token) / "levels"
        dated: list[tuple[datetime, Path]] = []
        for entry in self._entries(directory):
            if entry.is_file() and entry.name.endswith(".json"):
                path = directory / entry.name
                try:
                    dated.append((decode_unit_level(self._load(path)).at, path))
                except ForesightError:
                    continue
        older = sorted((item for item in dated if item[0] < before), key=lambda item: item[0])
        return sum(1 for _at, path in older[:-1] if durable_unlink(path, artifact_root=self.root))

    def unit_kinds(self) -> tuple[str, ...]:
        """记过单元档的行为（目录名是摘要，类编号从档里读）。"""

        found: set[str] = set()
        for entry in self._entries(self._units):
            for record in self._decoded(self._units / entry.name / "levels", decode_unit_level, None):
                found.add(record.kind_token)
                break
        return tuple(sorted(found))

    def _discard_day_dir(self, root: Path, day: date) -> None:
        directory = self._day_dir(root, day)
        for entry in self._entries(directory):
            if entry.is_file():
                durable_unlink(directory / entry.name, artifact_root=self.root)
        for path in (directory, directory.parent, directory.parent.parent):
            try:
                path.rmdir()
            except OSError:
                break

    def _save_scan_watermark(self, watermark: Watermark[date]) -> None:
        self._write_watermark(
            _SCAN_WATERMARK,
            {"through": watermark.through.isoformat(), "above": sorted(day.isoformat() for day in watermark.above)},
        )

    def _read_watermark(self, name: str) -> dict[str, Any] | None:
        path = self.root / name
        return self._load(path) if path.is_file() else None

    def _write_watermark(self, name: str, payload: dict[str, Any]) -> None:
        try:
            atomic_replace_bytes(self.root / name, _encoded(payload), artifact_root=self.root)
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"ledger watermark {name} is not writable: {exc}") from exc

    # ── 按账 ────────────────────────────────────────────────────────────────

    def append_fact(self, item: AccountFact) -> bool:
        identity = f"{item.expectation_id}--k{FACT_KINDS.index(item.kind):02d}--{_stamp(item.precedence)}"
        return self._account_create(item.account, "facts", identity, encode_fact(item))

    def facts_of(self, account: str) -> tuple[AccountFact, ...]:
        """这本账压实以外的事实；压实那一段里残留的文件（写完累计值、删文件之前崩溃）一律忽略。"""

        base = self.compaction_of(account)
        found = self._account_records(account, "facts", decode_fact)
        return tuple(fact for fact in found if base is None or not base.covers(fact.at, fact.expectation_id))

    # ── 压实（26 号清单 4.8；策略在 ``foresight.compaction``，这里只管字节）──────────────────

    def compaction_of(self, account: str) -> AccountCompaction | None:
        path = self._account_dir(account) / _COMPACTION
        if not path.is_file():
            return None
        found = decode_compaction(self._load(path))
        if found.account != account:
            raise ForesightError("account compaction does not match its account")
        return found

    def write_compaction(self, item: AccountCompaction) -> None:
        """先写累计值（整份替换），再删它盖住的事实文件（``discard_compacted``）：中途崩溃时读的一方忽略残留，不会数两遍。"""

        directory = self._account_dir(item.account)
        meta = directory / "account.json"
        if not meta.is_file():
            self._write(meta, {"record": "account", "account": item.account})
        try:
            atomic_replace_bytes(
                directory / _COMPACTION, _fitting(encode_compaction(item), _COMPACTION), artifact_root=self.root
            )
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"ledger path is not writable: {exc}") from exc

    def discard_compacted(self, account: str) -> int:
        """删掉压实那一段盖住的事实文件（同一条预期后写的改判一起删）；返回删了几个。没有压实就什么也不删。"""

        base = self.compaction_of(account)
        if base is None:
            return 0
        directory = self._account_dir(account) / "facts"
        removed = 0
        for entry in self._entries(directory):
            if not entry.is_file() or not entry.name.endswith(".json"):
                continue
            path = directory / entry.name
            try:
                fact = decode_fact(self._load(path))
            except ForesightError:
                continue
            if base.covers(fact.at, fact.expectation_id) and durable_unlink(path, artifact_root=self.root):
                removed += 1
        return removed

    def trim_levels(self, account: str, *, before: datetime) -> int:
        """级的变化只留 ``before`` 之后的，外加 ``before`` 之前最近的那一条（滞回与诊断要知道那时在哪一级）；返回删了几个。"""

        directory = self._account_dir(account) / "levels"
        dated: list[tuple[datetime, Path]] = []
        for entry in self._entries(directory):
            if entry.is_file() and entry.name.endswith(".json"):
                path = directory / entry.name
                try:
                    dated.append((decode_level(self._load(path)).at, path))
                except ForesightError:
                    continue
        older = sorted((item for item in dated if item[0] < before), key=lambda item: item[0])
        return sum(1 for _at, path in older[:-1] if durable_unlink(path, artifact_root=self.root))

    def record_level(self, item: AccountLevel) -> bool:
        return self._account_create(item.account, "levels", _stamp(item.at), encode_level(item))

    def level_of(self, account: str) -> AccountLevel | None:
        """最近一条级的变化（没有就是从没变过：影子）。"""

        found = list(self._account_records(account, "levels", decode_level))
        return max(found, key=lambda item: item.at) if found else None

    def levels_of(self, account: str) -> tuple[AccountLevel, ...]:
        return tuple(sorted(self._account_records(account, "levels", decode_level), key=lambda i: i.at))

    # ── 单元的档、设置、提醒反馈（按行为，与账的级分开；裁定 61） ──────────────────────────

    def record_unit_level(self, item: UnitLevel) -> bool:
        return self._write(
            self._unit_dir(item.kind_token) / "levels" / f"{_stamp(item.at)}.json", encode_unit_level(item)
        )

    def record_setting(self, item: UnitSetting) -> bool:
        return self._write(
            self._unit_dir(item.kind_token) / "settings" / f"{_stamp(item.at)}.json", encode_setting(item)
        )

    def setting_of(self, kind_token: str) -> UnitSetting | None:
        """这个行为最新的一条设置（没有就是没设过）。"""

        found = list(self._decoded(self._unit_dir(kind_token) / "settings", decode_setting, None))
        return max(found, key=lambda item: item.at) if found else None

    def settings_of(self, kind_token: str) -> tuple[UnitSetting, ...]:
        return tuple(
            sorted(self._decoded(self._unit_dir(kind_token) / "settings", decode_setting, None), key=lambda i: i.at)
        )

    def record_feedback(self, item: UnitFeedback) -> bool:
        """说出口的那条预期结算时写的一笔提醒反馈（同一条预期一笔；重写幂等）。"""

        return self._write(
            self._unit_dir(item.kind_token) / "feedback" / f"{item.expectation_id}--{_stamp(item.at)}.json",
            encode_feedback(item),
        )

    def feedback_of(self, kind_token: str) -> tuple[UnitFeedback, ...]:
        """这个行为的全部提醒反馈（按说出口的那一刻排）。"""

        found = self._decoded(self._unit_dir(kind_token) / "feedback", decode_feedback, None)
        return tuple(sorted(found, key=lambda item: (item.at, item.expectation_id)))

    def unit_level_of(self, kind_token: str) -> UnitLevel | None:
        """这个行为单元最近一条档的变化（没有就是从没变过：影子）。"""

        found = list(self._decoded(self._unit_dir(kind_token) / "levels", decode_unit_level, None))
        return max(found, key=lambda item: item.at) if found else None

    def _unit_dir(self, kind_token: str) -> Path:
        return self._units / sha256(kind_token.encode("utf-8")).hexdigest()[:24]

    def accounts(self) -> tuple[str, ...]:
        """账本里有过记录的全部账（可见面用）。"""

        found: list[str] = []
        for entry in self._entries(self._accounts):
            if not entry.is_dir():
                continue
            path = self._accounts / entry.name / "account.json"
            if not path.is_file():
                continue
            try:
                found.append(parse_account_text(self._load(path)))
            except ForesightError:
                self._note_unreadable(path, None)
        return tuple(sorted(found))

    def _account_dir(self, account: str) -> Path:
        return self._accounts / sha256(account.encode("utf-8")).hexdigest()[:24]

    def _account_create(self, account: str, kind: str, identity: str, payload: dict[str, Any]) -> bool:
        directory = self._account_dir(account)
        meta = directory / "account.json"
        if not meta.is_file():
            self._write(meta, {"record": "account", "account": account})
        return self._write(directory / kind / f"{identity}.json", payload)

    def _account_records[T](self, account: str, kind: str, decode: Callable[[dict[str, Any]], T]) -> Iterable[T]:
        return self._decoded(self._account_dir(account) / kind, decode, None)

    # ── 坏文件 ──────────────────────────────────────────────────────────────

    def take_unreadable(self) -> tuple[str, ...]:
        """上次 ``take`` 以来按日子读回时跳过的坏文件（路径），取走即清。"""

        with self._lock:
            found = tuple(self._unreadable)
            self._unreadable.clear()
        return found

    def _note_unreadable(self, path: Path, sink: list[str] | None) -> None:
        if sink is not None:
            if str(path) not in sink and len(sink) < _MAX_UNREADABLE:
                sink.append(str(path))
            return
        with self._lock:
            if str(path) not in self._unreadable and len(self._unreadable) < _MAX_UNREADABLE:
                self._unreadable.append(str(path))

    # ── 目录与字节 ──────────────────────────────────────────────────────────

    def _index_create(self, expectation_id: str, day: date) -> None:
        payload: dict[str, Any] = {"record": "index", "expectation_id": expectation_id, "day": day.isoformat()}
        try:
            atomic_create_bytes(self._index / f"{expectation_id}.json", _encoded(payload), artifact_root=self.root)
        except ImmutableArtifactConflictError as exc:
            raise ForesightError(f"ledger index entry {expectation_id} already points at another day") from exc
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"ledger index is not writable: {exc}") from exc

    def _create(self, root: Path, day: date, identity: str, payload: dict[str, Any]) -> bool:
        return self._write(self._day_dir(root, day) / f"{identity}.json", payload)

    def _write(self, path: Path, payload: dict[str, Any]) -> bool:
        identity = path.stem
        encoded = _fitting(payload, identity)
        try:
            return atomic_create_bytes(path, encoded, artifact_root=self.root)
        except ImmutableArtifactConflictError as exc:
            raise ForesightError(f"ledger record {identity} already exists with different content") from exc
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"ledger path is not writable: {exc}") from exc

    def _read(self, root: Path, day: date, identity: str) -> dict[str, Any] | None:
        path = self._day_dir(root, day) / f"{identity}.json"
        if not path.is_file():
            return None
        return self._load(path)

    def _records[T](
        self, root: Path, day: date, decode: Callable[[dict[str, Any]], T], sink: list[str] | None = None
    ) -> Iterable[T]:
        return self._decoded(self._day_dir(root, day), decode, sink)

    def _decoded[T](
        self, directory: Path, decode: Callable[[dict[str, Any]], T], sink: list[str] | None
    ) -> Iterable[T]:
        """一个目录里的记录逐条解码；读不出来、形状或版本不对的（解码在 try 里）记下路径、跳过。"""

        for entry in sorted(self._entries(directory), key=lambda item: item.name):
            if not entry.is_file() or not entry.name.endswith(".json") or entry.name.startswith((".", "_")):
                continue
            path = directory / entry.name
            try:
                yield decode(self._load(path))
            except ForesightError:
                self._note_unreadable(path, sink)

    def _load(self, path: Path) -> dict[str, Any]:
        try:
            encoded = read_regular_bytes(path, artifact_root=self.root, max_bytes=_MAX_BYTES)
            raw = json.loads(encoded)
        except (DurablePathIntegrityError, ValueError) as exc:
            raise ForesightError(f"ledger record {path.name} cannot be read: {exc}") from exc
        if not isinstance(raw, dict):
            raise ForesightError(f"ledger record {path.name} is not an object")
        return raw

    def _days(self, root: Path) -> tuple[date, ...]:
        days: list[date] = []
        for year in self._entries(root):
            for month in self._entries(root / year.name):
                for day in self._entries(root / year.name / month.name):
                    if not day.is_dir():
                        continue
                    try:
                        days.append(date(int(year.name), int(month.name), int(day.name)))
                    except ValueError:
                        continue
        return tuple(sorted(days))

    def _entries(self, directory: Path) -> tuple[DurableDirectoryEntry, ...]:
        if not directory.is_dir():
            return ()
        try:
            return tuple(list_real_directory(directory, artifact_root=self.root, max_entries=_MAX_ENTRIES))
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"ledger directory is invalid: {exc}") from exc

    @staticmethod
    def _day_dir(root: Path, day: date) -> Path:
        return root / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"


def _next_day(day: date) -> date:
    return day + timedelta(days=1)


def _encoded(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _stamp(at: datetime) -> str:
    return at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%f")


def _stamped(expectation_id: str, at: datetime) -> str:
    return f"{expectation_id}--{_stamp(at)}"


def parse_account_text(raw: dict[str, Any]) -> str:
    """``account.json`` → 账身份（校验它能拆成 类 / 触发种类 / 触发身份）。"""

    account = raw.get("account")
    if raw.get("record") != "account" or not isinstance(account, str):
        raise ForesightError("account meta is malformed")
    parse_account(account)
    return account


__all__ = ["ForesightLedgerStore"]
