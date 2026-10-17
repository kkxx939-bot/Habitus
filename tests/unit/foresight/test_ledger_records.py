"""账本记录（按预期六种、按账四种）：形状校验、编解码往返、身份不含槽号与时刻。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from habitus.foresight import ForesightError
from habitus.foresight.ledger import (
    AccountFact,
    AccountLevel,
    AccountSetting,
    AccountSeverity,
    Antecedent,
    Delivery,
    Expectation,
    Omission,
    Response,
    Ruling,
    Settlement,
    account_key,
    anchor_uri,
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
    expectation_identity,
    parse_account,
)

CST = timezone(timedelta(hours=8))
FORMED = datetime(2026, 12, 10, 14, 7, tzinfo=CST)
DAY = FORMED.date()


def expectation(**overrides: object) -> Expectation:
    values: dict[str, object] = dict(
        expectation_id="abc",
        kind_token="s-k0002",
        day=DAY,
        formed_at=FORMED,
        source="事件",
        trigger="session|s-k0001|s-k0002|chain",
        trigger_instance="behavior://occurrences/2026/12/10/订机票--x.md",
        window=(FORMED + timedelta(minutes=4), FORMED + timedelta(minutes=51)),
        deadline=FORMED + timedelta(hours=3),
        basis=("behavior://a.md", "behavior://b.md"),
        opportunities=13,
        hits=10,
        observed_from=FORMED - timedelta(minutes=30),
        antecedents=(Antecedent("session|s-k0001|s-k0002|chain", True, True, "behavior://x.md"),),
        level=1,
        generation="gen-1",
        night=DAY - timedelta(days=1),
        version="v1",
    )
    values.update(overrides)
    return Expectation(**values)  # type: ignore[arg-type]


def test_identity_ignores_the_slot_the_moment_and_the_day() -> None:
    first = expectation_identity("s-k0002", "事件", "rel", "behavior://x.md")
    assert first == expectation_identity("s-k0002", "事件", "rel", "behavior://x.md")
    assert first != expectation_identity("s-k0002", "事件", "rel", "behavior://y.md")
    assert first != expectation_identity("s-k0002", "日子", "rel", "behavior://x.md")
    assert len(first) == 24
    # 离上次够久的实例带着这一簇的第一天；别的实例就是出处本身
    assert anchor_uri("behavior://x.md@2026-10-12") == "behavior://x.md"
    assert anchor_uri("behavior://x.md") == "behavior://x.md" and anchor_uri("2026-10-12") == "2026-10-12"


def test_every_record_round_trips_through_its_codec() -> None:
    item = expectation()
    assert decode_expectation(encode_expectation(item)) == item
    delivery = Delivery("abc", FORMED, 1, True, "说了", 1.0, "上次订完机票你接着订了酒店，这次要吗？")
    assert decode_delivery(encode_delivery(delivery)) == delivery
    shadow = Delivery("abc", FORMED, 0, False, "影子", 0.0)
    assert decode_delivery(encode_delivery(shadow)) == shadow
    response = Response("abc", FORMED + timedelta(minutes=1), "这次不用", "住朋友家")
    assert decode_response(encode_response(response)) == response
    settlement = Settlement("abc", "s-k0002", DAY, FORMED + timedelta(days=1), "晚了但做了", "behavior://c.md", 12)
    assert decode_settlement(encode_settlement(settlement)) == settlement
    ruling = Ruling("abc", FORMED + timedelta(days=2), "后补", "behavior://d.md", 400)
    assert decode_ruling(encode_ruling(ruling)) == ruling
    omission = Omission("behavior://e.md", "s-k0003", DAY, FORMED + timedelta(days=1), False)
    assert decode_omission(encode_omission(omission)) == omission
    account = item.account
    fact = AccountFact(account, FORMED, "预测", True, 0.4, "晚了但做了", "abc", offset_minutes=12)
    assert decode_fact(encode_fact(fact)) == fact
    severity = AccountSeverity(account, FORMED, 4.0, "上次漏了要补办", FORMED + timedelta(days=60))
    assert decode_severity(encode_severity(severity)) == severity
    setting = AccountSetting(account, FORMED, "以后别提了")
    assert decode_setting(encode_setting(setting)) == setting
    level = AccountLevel(account, FORMED, 1)
    assert decode_level(encode_level(level)) == level


def test_records_refuse_shapes_that_cannot_be_settled() -> None:
    with pytest.raises(ForesightError, match="window"):
        expectation(window=(FORMED + timedelta(hours=1), FORMED))
    with pytest.raises(ForesightError, match="deadline"):
        expectation(deadline=FORMED)
    # 时窗可以早于形成时刻（到了钟点才建，形成时他可能已经做了：裁定 49）
    assert expectation(window=(FORMED - timedelta(minutes=1), FORMED + timedelta(hours=1))).window[0] < FORMED
    with pytest.raises(ForesightError, match="no later than when it was formed"):
        expectation(observed_from=FORMED + timedelta(minutes=1))
    with pytest.raises(ForesightError, match="after its deadline"):
        expectation(
            window=(FORMED - timedelta(hours=2), FORMED - timedelta(hours=1)), deadline=FORMED - timedelta(minutes=1)
        )
    with pytest.raises(ForesightError, match="hits cannot exceed"):
        expectation(hits=14)
    with pytest.raises(ForesightError, match="basis lists more"):
        expectation(hits=1)
    with pytest.raises(ForesightError, match="unknown expectation source"):
        expectation(source="猜的")
    with pytest.raises(ForesightError, match="timezone-aware"):
        expectation(formed_at=FORMED.replace(tzinfo=None))
    with pytest.raises(ForesightError, match="absent antecedent"):
        Antecedent("rel", False, True, "behavior://x.md")
    with pytest.raises(ForesightError, match="说了"):
        Delivery("abc", FORMED, 1, True, "影子", 1.0)
    with pytest.raises(ForesightError, match="unspoken delivery carries no text"):
        Delivery("abc", FORMED, 0, False, "影子", 0.0, "话")
    assert Delivery("abc", FORMED, 1, False, "等时窗末", 0.0, "话").text == "话"  # 话先备好、到时窗末再说
    with pytest.raises(ForesightError, match="unknown delivery level"):
        Delivery("abc", FORMED, 2, True, "说了", 1.0)  # 只到提醒（裁定 48）
    with pytest.raises(ForesightError, match="names the occurrence exactly"):
        Settlement("abc", "s-k0002", DAY, FORMED, "准时")
    with pytest.raises(ForesightError, match="offset exactly"):
        Settlement("abc", "s-k0002", DAY, FORMED, "准时", "behavior://c.md", 3)
    with pytest.raises(ForesightError, match="后补"):
        Ruling("abc", FORMED, "后补")
    with pytest.raises(ForesightError, match="后补"):
        Ruling("abc", FORMED, "用户说忘了", "behavior://d.md", 5)
    with pytest.raises(ForesightError, match="unknown settlement outcome"):
        Settlement("abc", "s-k0002", DAY, FORMED, "多余")  # 每个机会都建，已做的照常结（裁定 49）
    with pytest.raises(ForesightError, match="unknown response"):
        Response("abc", FORMED, "接受")  # 人能说的只剩三句（裁定 50）
    account = expectation().account
    with pytest.raises(ForesightError, match="strength"):
        AccountFact(account, FORMED, "预测", True, 1.5, "准时", "abc")
    with pytest.raises(ForesightError, match="unknown fact axis"):
        AccountFact(account, FORMED, "代劳", True, 1.0, "准时", "abc")
    with pytest.raises(ForesightError, match="malformed account key"):
        AccountFact("s-k0002", FORMED, "预测", True, 1.0, "准时", "abc")
    with pytest.raises(ForesightError, match="expire after"):
        AccountSeverity(account, FORMED, 4.0, "", FORMED)
    with pytest.raises(ForesightError, match="cost_ratio"):
        AccountSeverity(account, FORMED, 40.0, "", FORMED + timedelta(days=1))  # 代价比超出保护闸
    with pytest.raises(ForesightError, match="unknown setting"):
        AccountSetting(account, FORMED, "代劳")
    with pytest.raises(ForesightError, match="unknown account level"):
        AccountLevel(account, FORMED, 3)
    assert parse_account(account_key("s-k0002", "钟面", "weekday=1")) == ("s-k0002", "钟面", "weekday=1")


def test_decoding_refuses_other_schemas_and_other_records() -> None:
    raw = encode_expectation(expectation())
    with pytest.raises(ForesightError, match="expected delivery"):
        decode_delivery(raw)
    raw["schema_version"] = "foresight_ledger_v5"
    with pytest.raises(ForesightError, match="schema"):
        decode_expectation(raw)
    broken = encode_expectation(expectation())
    del broken["window"]
    with pytest.raises(ForesightError, match="malformed"):
        decode_expectation(broken)


def test_covers_is_after_observed_from_and_up_to_the_deadline() -> None:
    item = expectation()
    assert not item.covers(item.observed_from)
    assert item.covers(FORMED - timedelta(minutes=1))  # 形成之前、起算之后做的也算（裁定 49）
    assert item.covers(item.deadline)
    assert not item.covers(item.deadline + timedelta(seconds=1))
    assert item.account == account_key("s-k0002", "事件", "session|s-k0001|s-k0002|chain")
