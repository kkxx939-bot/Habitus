"""账本的落盘：按预期的六种记录各一个目录、按预期所在的日子分目录；按账的记录在 ``accounts/<账>/`` 下（裁定 48）；
按行为单元的记录在 ``units/<行为>/`` 下（裁定 61：提醒反馈、设置、档按行为）。一条一文件、写一次不改。

住在组合根：``foresight`` 不许 import infrastructure，形状与编解码在 ``foresight.ledger``，这里只管目录与字节。
同身份重写幂等（内容相同返回 False），内容不同是账本损坏。

    <root>/expectations/YYYY/MM/DD/<expectation_id>.json
    <root>/deliveries/YYYY/MM/DD/<expectation_id>--<decided_at>.json
    <root>/responses/YYYY/MM/DD/<expectation_id>--<responded_at>--<轮的摘要>.json
    <root>/settlements/YYYY/MM/DD/<expectation_id>.json
    <root>/rulings/YYYY/MM/DD/<expectation_id>--<ruled_at>.json
    <root>/omissions/YYYY/MM/DD/<记录身份>.json       查过漏报的日子记在 omissions_scanned.json 水位里
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
一个坏文件不许让整层停（读写的规矩在 ``runtime.foresight_files``，各 store 同一份）：读不出来（解析不了、形状或版本不对）的
那一份记下路径、跳过，调用方把条数带进观测（``take_unreadable``）；只想知道"自己这一次读到了哪些坏文件"的调用方传
``unreadable=`` 自己的列表，不经共享的那一份。点读同一个口径：索引、水位读不出来记下、当没有（水位没有就按逐天的结完标记认）；
结算、事实、提醒反馈这些写一次的记录，盘上同名的一份读不出来时不重写、不抛——这一条当已经写过（结算文件在就算结过），
后面的日子照常结。盘读到一半的 IO 错不是坏文件，照旧抛。

"还要看的日子"（``open_days``）= 有预期、还没标结完的日子：结算、递送、提醒标记只读这些，账本再长每拍也只读最近一截；
最远看到哪天由账本自己说（时窗跨多少天、过期待决等多久裁决），不写死回看天数。
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

from habitus.config.foresight import ForesightStoreConfig
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
from habitus.foresight.ledger.compaction import (
    STALE_COMPACTION_SCHEMA,
    AccountCompaction,
    decode_compaction,
    encode_compaction,
)
from habitus.foundation.lifecycle import Watermark, mark
from habitus.infrastructure.store.filesystem import (
    DurableDirectoryEntry,
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    durable_unlink,
)
from habitus.runtime.foresight_files import JsonFiles

_FINAL = "_final"
#: 一本账压实了的那一段（整份替换）。
_COMPACTION = "compaction.json"
#: 结完低水位、查过漏报的日子（26 号清单 6.11）。
_FINAL_WATERMARK = "final_through.json"
_SCAN_WATERMARK = "omissions_scanned.json"


def _watermark(raw: dict[str, Any]) -> dict[str, Any]:
    """水位文件的形状：``through`` 是日期。"""

    try:
        date.fromisoformat(str(raw["through"]))
    except (KeyError, ValueError) as exc:
        raise ForesightError("ledger watermark is malformed") from exc
    return raw


def _index_day(raw: dict[str, Any]) -> date:
    try:
        return date.fromisoformat(str(raw["day"]))
    except (KeyError, ValueError) as exc:
        raise ForesightError("ledger index entry is corrupt") from exc


class ForesightLedgerStore:
    def __init__(self, root: str | Path, *, limits: ForesightStoreConfig | None = None) -> None:
        limits = limits if limits is not None else ForesightStoreConfig()
        #: 写一次的记录写入就查大小（超限的不落盘：按目录读时会被当坏文件跳过、结算永远结不到它，26 号清单 2.5.4）。
        self._files = JsonFiles(
            root,
            label="ledger",
            max_bytes=limits.record_max_bytes,
            max_entries=limits.directory_max_entries,
            max_unreadable=limits.unreadable_max_paths,
        )
        self.root = self._files.root
        self._expectations = self.root / "expectations"
        self._deliveries = self.root / "deliveries"
        self._responses = self.root / "responses"
        self._settlements = self.root / "settlements"
        self._rulings = self.root / "rulings"
        self._omissions = self.root / "omissions"
        self._index = self.root / "index" / "expectations"
        self._accounts = self.root / "accounts"
        self._units = self.root / "units"
        #: 标过结完的日子（标记只增不撤，记住了就不用再看盘）。
        self._finals: set[date] = set()
        #: 夜批（结算在线程里跑）与白天那一拍共用这一个实例：这两份内存状态的改动加锁（2026-10-17 四方评审）。
        self._lock = threading.Lock()

    # ── 预期 ────────────────────────────────────────────────────────────────

    def record_expectation(self, item: Expectation) -> bool:
        """同一身份全账本只写一次：索引先占位（身份 → 日子），另一天再来同一身份就是重复、不写。"""

        placed, known = self._index_entry(item.expectation_id)
        if placed and known != item.day:
            # 另一天的同一身份是重复；索引坏了（记下坏文件）：这个身份占过位、不知道在哪一天，不另起一份
            return False
        payload = encode_expectation(item)
        self._files.encode(payload, item.expectation_id)  # 超限的连索引都不占：否则留下一条指向不存在预期的索引
        # 先占索引再写预期：中间断了，重来时索引指着同一天、预期照写；反过来的顺序会让第二天按身份查不到、再建一份
        self._index_create(item.expectation_id, item.day)
        return self._create(self._expectations, item.day, item.expectation_id, payload, decode_expectation)

    def day_of(self, expectation_id: str) -> date | None:
        """这个身份的预期在哪一天；没建过、索引读不出来（记下坏文件）是 None。"""

        return self._files.read(self._index / f"{expectation_id}.json", _index_day)

    def has_expectation(self, expectation_id: str) -> bool:
        """这条预期真写下了：索引占了位、而且那一天的预期文件在（2026-10-17 四方评审：只看索引的话，"索引写了、预期文件还没写
        就断了"的那条，之后每拍都被当成已有、``record_expectation`` 补写那一步永远走不到，这次机会就丢了）。索引坏了的
        算写过：不知道它在哪一天，不另起一份。"""

        placed, day = self._index_entry(expectation_id)
        if not placed:
            return False
        return day is None or (self._day_dir(self._expectations, day) / f"{expectation_id}.json").is_file()

    def _index_entry(self, expectation_id: str) -> tuple[bool, date | None]:
        """（索引占没占位, 指的那一天）；占了位却读不出来（记下坏文件）是 ``(True, None)``。"""

        path = self._index / f"{expectation_id}.json"
        if not path.is_file():
            return False, None
        return True, self._files.read(path, _index_day)

    def expectations_on(self, day: date, *, unreadable: list[str] | None = None) -> tuple[Expectation, ...]:
        return tuple(self._records(self._expectations, day, decode_expectation, unreadable))

    def expectation(self, day: date, expectation_id: str) -> Expectation | None:
        return self._files.read(self._day_dir(self._expectations, day) / f"{expectation_id}.json", decode_expectation)

    def days_with_expectations(self, *, since: date | None = None) -> tuple[date, ...]:
        return tuple(day for day in self._days(self._expectations) if since is None or day >= since)

    def open_days(self) -> tuple[date, ...]:
        """有预期、还没标结完的日子（升序）：只看"结完低水位"以上的（26 号清单 6.11）。"""

        through = self.final_through()
        since = None if through is None else through + timedelta(days=1)
        return tuple(day for day in self.days_with_expectations(since=since) if not self.is_final(day))

    def final_through(self) -> date | None:
        """结完低水位：这一天及以前有预期的日子全部结完了（没有为 None；读不出来也是 None，记下坏文件——逐天的结完标记照样认，
        下一次推进水位时整份重写）。"""

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
        created = self._create(self._settlements, day, _FINAL, payload, _final_marker)
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
        """这一天还没结算的预期。结算文件在就算结过——读不出来的那一份也是（记下坏文件）：当成没结、再结一遍，结出的内容
        （结算时刻不同）必然和盘上的不一样，写不进去；坏文件那一条不重结，后面的日子照常结。"""

        directory = self._day_dir(self._settlements, day)
        settled = {item.expectation_id for item in self.settlements_on(day)}
        settled |= {name.removesuffix(".json") for name in self._files.names(directory)}
        return tuple(item for item in self.expectations_on(day) if item.expectation_id not in settled)

    # ── 递送与回话 ──────────────────────────────────────────────────────────

    def record_delivery(self, day: date, item: Delivery) -> bool:
        return self._create(
            self._deliveries,
            day,
            _stamped(item.expectation_id, item.decided_at),
            encode_delivery(item),
            decode_delivery,
        )

    def deliveries_on(self, day: date, *, unreadable: list[str] | None = None) -> tuple[Delivery, ...]:
        return tuple(self._records(self._deliveries, day, decode_delivery, unreadable))

    def delivery_names(self, day: date) -> frozenset[str]:
        """这一天递送目录里的文件名（只增，名字集合没变 = 内容没变；读口的缓存拿它判要不要重读）。"""

        return frozenset(entry.name for entry in self._entries(self._day_dir(self._deliveries, day)))

    def record_response(self, day: date, item: Response) -> bool:
        """同一刻对同一条提醒可能有两条回话：出口按他点的固定选项写一条（轮 = call_id）、会话 lane 的桥按模型判的写一条
        （轮 = 作答那一轮的身份，时刻同一条消息）——文件名带上轮的摘要，两条都落得下；同一轮重放仍幂等。"""

        identity = (
            f"{_stamped(item.expectation_id, item.responded_at)}--{sha256(item.turn.encode('utf-8')).hexdigest()[:12]}"
        )
        return self._create(self._responses, day, identity, encode_response(item), decode_response)

    def responses_on(self, day: date) -> tuple[Response, ...]:
        return tuple(self._records(self._responses, day, decode_response))

    # ── 结算与裁决 ──────────────────────────────────────────────────────────

    def record_settlement(self, item: Settlement) -> bool:
        return self._create(
            self._settlements, item.day, item.expectation_id, encode_settlement(item), decode_settlement
        )

    def settlements_on(self, day: date) -> tuple[Settlement, ...]:
        return tuple(self._records(self._settlements, day, decode_settlement))

    def record_ruling(self, day: date, item: Ruling) -> bool:
        return self._create(
            self._rulings, day, _stamped(item.expectation_id, item.ruled_at), encode_ruling(item), decode_ruling
        )

    def rulings_on(self, day: date) -> tuple[Ruling, ...]:
        return tuple(self._records(self._rulings, day, decode_ruling))

    # ── 漏报 ────────────────────────────────────────────────────────────────

    def record_omission(self, item: Omission) -> bool:
        """一条记录的漏报只记一次：已经记过（同一条 occurrence）就不再写——重放时内容里的 ``noticed_at`` 不同，按内容比会被当成
        账本损坏，整晚结算每晚都失败、日子永远结不完（2026-10-17 四方评审）。"""

        identity = sha256(item.occurrence_uri.encode("utf-8")).hexdigest()[:24]
        if (self._day_dir(self._omissions, item.day) / f"{identity}.json").is_file():
            return False
        return self._create(self._omissions, item.day, identity, encode_omission(item), decode_omission)

    def omissions_on(self, day: date) -> tuple[Omission, ...]:
        return tuple(self._records(self._omissions, day, decode_omission))

    def scan_watermark(self, *, start: date) -> Watermark[date]:
        """查过漏报的日子（26 号清单 6.11）：连续前缀 + 零散查过的；没有时从 ``start`` 的前一天起。"""

        raw = self._read_watermark(_SCAN_WATERMARK, _scan_watermark)
        if raw is None:
            return Watermark(through=start - timedelta(days=1))
        return Watermark(
            through=date.fromisoformat(str(raw["through"])),
            above=frozenset(date.fromisoformat(str(day)) for day in raw["above"]),
        )

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

        return self._trim_dated(self._unit_dir(kind_token) / "levels", lambda raw: decode_unit_level(raw).at, before)

    def _trim_dated(self, directory: Path, moment: Callable[[dict[str, Any]], datetime], before: datetime) -> int:
        """档的变化只留 ``before`` 之后的，外加 ``before`` 之前最近的那一条（滞回与诊断要知道那时在哪一档）；读不出来的
        档文件记下坏文件、不删（不知道它是哪一刻的）。返回删了几个。"""

        dated = [(at, directory / f"{stem}.json") for stem, at in self._files.decoded(directory, moment)]
        older = sorted((item for item in dated if item[0] < before), key=lambda item: item[0])
        return sum(1 for _at, path in older[:-1] if durable_unlink(path, artifact_root=self.root))

    def unit_kinds(self) -> tuple[str, ...]:
        """记过单元档的行为（目录名是摘要，类编号从档里读）。"""

        found: set[str] = set()
        for entry in self._entries(self._units):
            for record in self._decoded(self._units / entry.name / "levels", decode_unit_level):
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

    def _read_watermark(
        self, name: str, decode: Callable[[dict[str, Any]], dict[str, Any]] = _watermark
    ) -> dict[str, Any] | None:
        """水位（读不出来的记下坏文件、当没有）。"""

        return self._files.read(self.root / name, decode)

    def _write_watermark(self, name: str, payload: dict[str, Any]) -> None:
        self._files.replace(self.root / name, payload)

    # ── 按账 ────────────────────────────────────────────────────────────────

    def append_fact(self, item: AccountFact) -> bool:
        identity = f"{item.expectation_id}--k{FACT_KINDS.index(item.kind):02d}--{_stamp(item.precedence)}"
        return self._account_create(item.account, "facts", identity, encode_fact(item), decode_fact)

    def facts_of(self, account: str) -> tuple[AccountFact, ...]:
        """这本账压实以外的事实；压实那一段里残留的文件（写完累计值、删文件之前崩溃）一律忽略。"""

        base = self.compaction_of(account)
        found = self._decoded(self._account_dir(account) / "facts", decode_fact)
        return tuple(fact for fact in found if base is None or not base.covers(fact.at, fact.expectation_id))

    # ── 压实（26 号清单 4.8；策略在 ``foresight.compaction``，这里只管字节）──────────────────

    def compaction_of(self, account: str) -> AccountCompaction | None:
        path = self._account_dir(account) / _COMPACTION
        if not path.is_file():
            return None
        raw = self._files.load(path)
        if raw.get("schema_version") == STALE_COMPACTION_SCHEMA:
            raise ForesightError(
                f"这是旧格式的压实摘要（{STALE_COMPACTION_SCHEMA}，只存间隔个数与最长的、算不出久别要的典型间隔），新版本不迁移；"
                f"删除 {path} 后重启即可。会丢掉的只有这本账压实那一段的摘要（自然命中的偏差、机会间隔），窗口内的事实不受影响"
            )
        found = decode_compaction(raw)
        if found.account != account:
            raise ForesightError("account compaction does not match its account")
        return found

    def write_compaction(self, item: AccountCompaction) -> None:
        """先写累计值（整份替换），再删它盖住的事实文件（``discard_compacted``）：中途崩溃时读的一方忽略残留，不会数两遍。"""

        directory = self._account_dir(item.account)
        meta = directory / "account.json"
        if not meta.is_file():
            self._write(meta, {"record": "account", "account": item.account}, parse_account_text)
        self._files.replace(directory / _COMPACTION, encode_compaction(item))

    def discard_compacted(self, account: str) -> int:
        """删掉压实那一段盖住的事实文件（同一条预期后写的改判一起删）；返回删了几个。没有压实就什么也不删。"""

        base = self.compaction_of(account)
        if base is None:
            return 0
        directory = self._account_dir(account) / "facts"
        removed = 0
        for stem, fact in self._files.decoded(directory, decode_fact):
            if base.covers(fact.at, fact.expectation_id) and durable_unlink(
                directory / f"{stem}.json", artifact_root=self.root
            ):
                removed += 1
        return removed

    def trim_levels(self, account: str, *, before: datetime) -> int:
        """账的级只留 ``before`` 之后的，外加它之前最近的一条；返回删了几个。"""

        return self._trim_dated(self._account_dir(account) / "levels", lambda raw: decode_level(raw).at, before)

    def record_level(self, item: AccountLevel) -> bool:
        return self._account_create(item.account, "levels", _stamp(item.at), encode_level(item), decode_level)

    def level_of(self, account: str) -> AccountLevel | None:
        """最近一条级的变化（没有就是从没变过：影子）。"""

        found = list(self._decoded(self._account_dir(account) / "levels", decode_level))
        return max(found, key=lambda item: item.at) if found else None

    def levels_of(self, account: str) -> tuple[AccountLevel, ...]:
        return tuple(sorted(self._decoded(self._account_dir(account) / "levels", decode_level), key=lambda i: i.at))

    # ── 单元的档、设置、提醒反馈（按行为，与账的级分开；裁定 61） ──────────────────────────

    def record_unit_level(self, item: UnitLevel) -> bool:
        return self._write(
            self._unit_dir(item.kind_token) / "levels" / f"{_stamp(item.at)}.json",
            encode_unit_level(item),
            decode_unit_level,
        )

    def record_setting(self, item: UnitSetting) -> bool:
        return self._write(
            self._unit_dir(item.kind_token) / "settings" / f"{_stamp(item.at)}.json",
            encode_setting(item),
            decode_setting,
        )

    def setting_of(self, kind_token: str) -> UnitSetting | None:
        """这个行为最新的一条设置（没有就是没设过）。"""

        found = list(self._decoded(self._unit_dir(kind_token) / "settings", decode_setting))
        return max(found, key=lambda item: item.at) if found else None

    def settings_of(self, kind_token: str) -> tuple[UnitSetting, ...]:
        return tuple(sorted(self._decoded(self._unit_dir(kind_token) / "settings", decode_setting), key=lambda i: i.at))

    def record_feedback(self, item: UnitFeedback) -> bool:
        """说出口的那条预期结算时写的一笔提醒反馈（同一条预期一笔；重写幂等）。"""

        return self._write(
            self._unit_dir(item.kind_token) / "feedback" / f"{item.expectation_id}--{_stamp(item.at)}.json",
            encode_feedback(item),
            decode_feedback,
        )

    def feedback_of(self, kind_token: str) -> tuple[UnitFeedback, ...]:
        """这个行为的全部提醒反馈（按说出口的那一刻排）。"""

        found = self._decoded(self._unit_dir(kind_token) / "feedback", decode_feedback)
        return tuple(sorted(found, key=lambda item: (item.at, item.expectation_id)))

    def trim_feedback(self, kind_token: str, *, keep: int) -> int:
        """这个行为的提醒反馈按说出口的先后只留最近 ``keep`` 条；更早的、它那条预期所在那天已经整天释放的删掉（那天还在的
        留着：结算的"写过就不再写"守卫要看它）。返回删了几条（裁定 66）。"""

        directory = self._unit_dir(kind_token) / "feedback"
        found = [(item, directory / f"{stem}.json") for stem, item in self._files.decoded(directory, decode_feedback)]
        found.sort(key=lambda pair: (pair[0].at, pair[0].expectation_id))
        older = found[: max(0, len(found) - keep)]
        return sum(
            1
            for item, path in older
            if self._released(item.expectation_id, item.at) and durable_unlink(path, artifact_root=self.root)
        )

    def _released(self, expectation_id: str, formed_at: datetime) -> bool:
        """这条预期所在那天已经整天释放：索引还在就看它指的那天，索引先放了就看形成时刻前后各一天（预期的日子是主体本地日，
        形成时刻可能是 UTC 写法）里都没有它的预期文件。"""

        day = self.day_of(expectation_id)
        candidates = (
            (day,) if day is not None else tuple(formed_at.date() + timedelta(days=shift) for shift in (-1, 0, 1))
        )
        return not any(
            (self._day_dir(self._expectations, found) / f"{expectation_id}.json").is_file() for found in candidates
        )

    def unit_level_of(self, kind_token: str) -> UnitLevel | None:
        """这个行为单元最近一条档的变化（没有就是从没变过：影子）。"""

        found = list(self._decoded(self._unit_dir(kind_token) / "levels", decode_unit_level))
        return max(found, key=lambda item: item.at) if found else None

    def _unit_dir(self, kind_token: str) -> Path:
        return self._units / sha256(kind_token.encode("utf-8")).hexdigest()[:24]

    def accounts(self) -> tuple[str, ...]:
        """账本里有过记录的全部账（可见面用）。"""

        found: list[str] = []
        for entry in self._entries(self._accounts):
            if not entry.is_dir():
                continue
            account = self._files.read(self._accounts / entry.name / "account.json", parse_account_text)
            if account is not None:
                found.append(account)
        return tuple(sorted(found))

    def _account_dir(self, account: str) -> Path:
        return self._accounts / sha256(account.encode("utf-8")).hexdigest()[:24]

    def _account_create(
        self, account: str, kind: str, identity: str, payload: dict[str, Any], decode: Callable[[dict[str, Any]], Any]
    ) -> bool:
        directory = self._account_dir(account)
        meta = directory / "account.json"
        if not meta.is_file():
            self._write(meta, {"record": "account", "account": account}, parse_account_text)
        return self._write(directory / kind / f"{identity}.json", payload, decode)

    # ── 坏文件 ──────────────────────────────────────────────────────────────

    def take_unreadable(self) -> tuple[str, ...]:
        """上次 ``take`` 以来读回时跳过的坏文件（路径），取走即清。"""

        return self._files.take_unreadable()

    # ── 目录与字节 ──────────────────────────────────────────────────────────

    def _index_create(self, expectation_id: str, day: date) -> None:
        payload: dict[str, Any] = {"record": "index", "expectation_id": expectation_id, "day": day.isoformat()}
        try:
            atomic_create_bytes(
                self._index / f"{expectation_id}.json",
                self._files.encode(payload, expectation_id),
                artifact_root=self.root,
            )
        except ImmutableArtifactConflictError as exc:
            raise ForesightError(f"ledger index entry {expectation_id} already points at another day") from exc
        except DurablePathIntegrityError as exc:
            raise ForesightError(f"ledger index is not writable: {exc}") from exc

    def _create(
        self, root: Path, day: date, identity: str, payload: dict[str, Any], decode: Callable[[dict[str, Any]], Any]
    ) -> bool:
        return self._write(self._day_dir(root, day) / f"{identity}.json", payload, decode)

    def _write(self, path: Path, payload: dict[str, Any], decode: Callable[[dict[str, Any]], Any]) -> bool:
        """写一次不改（``JsonFiles.create``）：盘上同名的一份读不出来是坏文件，记下、不重写、不抛。"""

        return self._files.create(path, payload, decode)

    def _records[T](
        self, root: Path, day: date, decode: Callable[[dict[str, Any]], T], sink: list[str] | None = None
    ) -> Iterable[T]:
        return self._decoded(self._day_dir(root, day), decode, sink)

    def _decoded[T](
        self, directory: Path, decode: Callable[[dict[str, Any]], T], sink: list[str] | None = None
    ) -> Iterable[T]:
        """一个目录里的记录逐条解码；读不出来、形状或版本不对的记下路径、跳过。"""

        return (item for _stem, item in self._files.decoded(directory, decode, sink))

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
                        continue  # 目录名拼不成日期：不是账本的日目录，不算哪一天
        return tuple(sorted(days))

    def _entries(self, directory: Path) -> tuple[DurableDirectoryEntry, ...]:
        return self._files.entries(directory)

    @staticmethod
    def _day_dir(root: Path, day: date) -> Path:
        return root / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"


def _next_day(day: date) -> date:
    return day + timedelta(days=1)


def _scan_watermark(raw: dict[str, Any]) -> dict[str, Any]:
    """查过漏报的水位：``through`` 加零散查过的 ``above``。"""

    _watermark(raw)
    above = raw.get("above")
    try:
        if not isinstance(above, list):
            raise ValueError("above must be a list")
        for day in above:
            date.fromisoformat(str(day))
    except ValueError as exc:
        raise ForesightError("ledger scan watermark is malformed") from exc
    return raw


def _final_marker(raw: dict[str, Any]) -> dict[str, Any]:
    if raw.get("record") != "final":
        raise ForesightError("ledger final marker is malformed")
    return raw


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
