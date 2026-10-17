"""账本的落盘：按预期的六种记录各一个目录、按预期所在的日子分目录；按账的四种记录在 ``accounts/<账>/`` 下（裁定 48）。
一条一文件、写一次不改。

住在组合根：``foresight`` 不许 import infrastructure，形状与编解码在 ``foresight.ledger``，这里只管目录与字节。
同身份重写幂等（内容相同返回 False），内容不同是账本损坏。

    <root>/expectations/YYYY/MM/DD/<expectation_id>.json
    <root>/deliveries/YYYY/MM/DD/<expectation_id>--<decided_at>.json
    <root>/responses/YYYY/MM/DD/<expectation_id>--<responded_at>.json
    <root>/settlements/YYYY/MM/DD/<expectation_id>.json
    <root>/rulings/YYYY/MM/DD/<expectation_id>--<ruled_at>.json
    <root>/omissions/YYYY/MM/DD/<记录身份>.json，另有 _scanned.json 标这一天已查过漏报
    <root>/settlements/YYYY/MM/DD/_final.json        这一天结完了（全部结算、过期待决的裁决再也不会变），之后不再读
    <root>/settlements/YYYY/MM/DD/_explained.json    这一天的落空都找过原因了（裁定 55），机会表那一拍不再读
    <root>/index/expectations/<expectation_id>.json   身份 → 它在哪一天（同一身份全账本只建一条，跨天的触发靠它查重）
    <root>/accounts/<账哈希>/account.json              这本账是什么（类、触发种类、触发身份）
    <root>/accounts/<账哈希>/facts/<预期>--<轴>--<时刻>.json
    <root>/accounts/<账哈希>/severity/<judged_at>.json
    <root>/accounts/<账哈希>/settings/<at>.json
    <root>/accounts/<账哈希>/levels/<at>.json

按账的那一套回答"这件事可预测吗、该不该开口"：递法只读它，一拍只开到了时窗那几本账的目录，不扫整本账。

递送、回话、裁决都挂在它们那条预期所在的日子下，读一条预期的全部往来只开一个目录。
一个坏文件不许让整层停：按日子读回时读不出来（解析不了、形状或版本不对）的那一份记下路径、跳过，调用方把条数带进观测
（``take_unreadable``）；只想知道"自己这一次读到了哪些坏文件"的调用方传 ``unreadable=`` 自己的列表，不经共享的那一份。

"还要看的日子"（``open_days``）= 有预期、还没标结完的日子：结算、递送、提醒标记只读这些，账本再长每拍也只读最近一截；
最远看到哪天由账本自己说（时窗跨多少天、过期待决等多久裁决），不写死回看天数。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from habitus.foresight import ForesightError
from habitus.foresight.ledger import (
    FACT_KINDS,
    AccountFact,
    AccountLevel,
    AccountSetting,
    AccountSeverity,
    Delivery,
    Expectation,
    Omission,
    Response,
    Ruling,
    Settlement,
    decode_delivery,
    decode_expectation,
    decode_fact,
    decode_level,
    decode_omission,
    decode_response,
    decode_ruling,
    decode_setting,
    decode_settlement,
    decode_severity,
    encode_delivery,
    encode_expectation,
    encode_fact,
    encode_level,
    encode_omission,
    encode_response,
    encode_ruling,
    encode_setting,
    encode_settlement,
    encode_severity,
    parse_account,
)
from habitus.infrastructure.store.filesystem import (
    DurableDirectoryEntry,
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    list_real_directory,
    read_regular_bytes,
)

_MAX_ENTRIES = 100_000
_MAX_BYTES = 1_000_000
_SCANNED = "_scanned"
_FINAL = "_final"
#: 事实文件身份里的轴码（只许往后加：身份进了文件名）。
_AXIS_CODES = {"预测": "p", "提醒组": "r", "扣发组": "h"}
_EXPLAINED = "_explained"
_MAX_UNREADABLE = 1_000


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
        self._unreadable: list[str] = []
        #: 标过结完的日子（标记只增不撤，记住了就不用再看盘）。
        self._finals: set[date] = set()

    # ── 预期 ────────────────────────────────────────────────────────────────

    def record_expectation(self, item: Expectation) -> bool:
        """同一身份全账本只写一次：索引先占位（身份 → 日子），另一天再来同一身份就是重复、不写。"""

        known = self.day_of(item.expectation_id)
        if known is not None and known != item.day:
            return False
        # 先占索引再写预期：中间断了，重来时索引指着同一天、预期照写；反过来的顺序会让第二天按身份查不到、再建一份
        self._index_create(item.expectation_id, item.day)
        return self._create(self._expectations, item.day, item.expectation_id, encode_expectation(item))

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
        return self.day_of(expectation_id) is not None

    def expectations_on(self, day: date, *, unreadable: list[str] | None = None) -> tuple[Expectation, ...]:
        return tuple(self._records(self._expectations, day, decode_expectation, unreadable))

    def expectation(self, day: date, expectation_id: str) -> Expectation | None:
        raw = self._read(self._expectations, day, expectation_id)
        return None if raw is None else decode_expectation(raw)

    def days_with_expectations(self) -> tuple[date, ...]:
        return self._days(self._expectations)

    def open_days(self) -> tuple[date, ...]:
        """有预期、还没标结完的日子（升序）。"""

        return tuple(day for day in self.days_with_expectations() if not self.is_final(day))

    def mark_final(self, day: date, *, at: datetime) -> bool:
        """这一天结完了：全部结算过、过期待决的裁决再也不会变。之后结算、递送、提醒标记都不再读它。"""

        payload: dict[str, Any] = {"record": "final", "day": day.isoformat(), "at": at.isoformat()}
        created = self._create(self._settlements, day, _FINAL, payload)
        self._finals.add(day)
        return created

    def reopen(self, day: date) -> None:
        """撤销"这一天结完了"：重结把这一天的某条预期改回了过期待决，之后每晚还要看它有没有补做。"""

        (self._day_dir(self._settlements, day) / f"{_FINAL}.json").unlink(missing_ok=True)
        (self._day_dir(self._settlements, day) / f"{_EXPLAINED}.json").unlink(missing_ok=True)
        self._finals.discard(day)

    def mark_explained(self, day: date, *, at: datetime) -> bool:
        """这一天的落空都找过原因了（裁定 55）：之后机会表那一拍不再读它。重结打开这一天时一起撤销。"""

        payload: dict[str, Any] = {"record": "explained", "day": day.isoformat(), "at": at.isoformat()}
        return self._create(self._settlements, day, _EXPLAINED, payload)

    def forget_explained(self, day: date) -> None:
        (self._day_dir(self._settlements, day) / f"{_EXPLAINED}.json").unlink(missing_ok=True)

    def explained(self, day: date) -> bool:
        return (self._day_dir(self._settlements, day) / f"{_EXPLAINED}.json").is_file()

    def is_final(self, day: date) -> bool:
        if day in self._finals:
            return True
        if (self._day_dir(self._settlements, day) / f"{_FINAL}.json").is_file():
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
        identity = sha256(item.occurrence_uri.encode("utf-8")).hexdigest()[:24]
        return self._create(self._omissions, item.day, identity, encode_omission(item))

    def omissions_on(self, day: date) -> tuple[Omission, ...]:
        return tuple(self._records(self._omissions, day, decode_omission))

    def mark_scanned(self, day: date, *, scanned_at: datetime, found: int) -> bool:
        """这一天查过漏报了（同一天不再查第二遍）。"""

        payload: dict[str, Any] = {
            "record": "scanned",
            "day": day.isoformat(),
            "scanned_at": scanned_at.isoformat(),
            "found": found,
        }
        return self._create(self._omissions, day, _SCANNED, payload)

    def scanned(self, day: date) -> bool:
        return self._read(self._omissions, day, _SCANNED) is not None

    # ── 按账 ────────────────────────────────────────────────────────────────

    def append_fact(self, item: AccountFact) -> bool:
        axis = _AXIS_CODES[item.axis]
        identity = f"{item.expectation_id}--{axis}{FACT_KINDS.index(item.kind):02d}--{_stamp(item.precedence)}"
        return self._account_create(item.account, "facts", identity, encode_fact(item))

    def facts_of(self, account: str) -> tuple[AccountFact, ...]:
        return tuple(self._account_records(account, "facts", decode_fact))

    def record_severity(self, item: AccountSeverity) -> bool:
        return self._account_create(item.account, "severity", _stamp(item.judged_at), encode_severity(item))

    def severity_of(self, account: str) -> AccountSeverity | None:
        """最近一次判定（过没过期由调用方按时刻看）。"""

        found = list(self._account_records(account, "severity", decode_severity))
        return max(found, key=lambda item: item.judged_at) if found else None

    def severities_of(self, account: str) -> tuple[AccountSeverity, ...]:
        return tuple(self._account_records(account, "severity", decode_severity))

    def record_setting(self, item: AccountSetting) -> bool:
        return self._account_create(item.account, "settings", _stamp(item.at), encode_setting(item))

    def setting_of(self, account: str) -> AccountSetting | None:
        found = list(self._account_records(account, "settings", decode_setting))
        return max(found, key=lambda item: item.at) if found else None

    def settings_of(self, account: str) -> tuple[AccountSetting, ...]:
        return tuple(self._account_records(account, "settings", decode_setting))

    def record_level(self, item: AccountLevel) -> bool:
        return self._account_create(item.account, "levels", _stamp(item.at), encode_level(item))

    def level_of(self, account: str) -> AccountLevel | None:
        """最近一条级的变化（没有就是从没变过：影子）。"""

        found = list(self._account_records(account, "levels", decode_level))
        return max(found, key=lambda item: item.at) if found else None

    def levels_of(self, account: str) -> tuple[AccountLevel, ...]:
        return tuple(sorted(self._account_records(account, "levels", decode_level), key=lambda i: i.at))

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

        found = tuple(self._unreadable)
        self._unreadable.clear()
        return found

    def _note_unreadable(self, path: Path, sink: list[str] | None) -> None:
        target = self._unreadable if sink is None else sink
        if str(path) not in target and len(target) < _MAX_UNREADABLE:
            target.append(str(path))

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
        try:
            return atomic_create_bytes(path, _encoded(payload), artifact_root=self.root)
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

    def _decoded[T](self, directory: Path, decode: Callable[[dict[str, Any]], T], sink: list[str] | None) -> Iterable[T]:
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
