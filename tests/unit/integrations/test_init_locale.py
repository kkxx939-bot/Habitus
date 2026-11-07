"""``habitus-server init`` 检测系统时区、交互时请用户确认，检测不到才回退新加坡（W3）。

系统时区一律用替身代替 ``tzlocal`` 的返回值（``locale_setup._tzlocal_name``），测试结果不随跑测试的机器所在地变化。
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import available_timezones

import pytest
import yaml

from habitus.config import HabitusConfig
from habitus.infrastructure.store.filesystem import restrict_file_to_owner
from habitus.integrations.local_service import cli, locale_setup
from habitus.integrations.local_service import initialization as initialization_module
from habitus.integrations.local_service.cloud_setup import default_cloud_selection
from habitus.integrations.local_service.initialization import initialize_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_CONFIG = REPOSITORY_ROOT / "habitus" / "config" / "example.yaml"


@pytest.fixture(autouse=True)
def _isolate_default_plugin_connection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        initialization_module,
        "DEFAULT_PLUGIN_CONNECTION_PATH",
        tmp_path / "default-home" / ".habitus" / "agent-plugin" / "connection.json",
    )


def _system_zone(monkeypatch: pytest.MonkeyPatch, reader: Callable[[], object]) -> None:
    monkeypatch.setattr(locale_setup, "_tzlocal_name", reader)


def _returns(name: object) -> Callable[[], object]:
    return lambda: name


def _raises() -> object:
    raise LookupError("tzlocal could not work out the local zone")


def _locale_of(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload["locale"]


def _interactive_init(monkeypatch: pytest.MonkeyPatch, destination: Path, answers: Iterable[str], *extra: str) -> Mock:
    """走真实的 ``_prompt_text`` / ``_prompt_required_text``：只替换 ``input`` 的回答。"""

    typed = Mock(side_effect=list(answers))
    monkeypatch.setattr(cli, "_interactive", Mock(return_value=True))
    monkeypatch.setattr(cli, "_prompt_cloud_setup", Mock(return_value=default_cloud_selection()))
    monkeypatch.setattr(cli, "_configure_missing_credentials", Mock())
    monkeypatch.setattr(cli, "_confirm", Mock(return_value=False))
    monkeypatch.setattr("builtins.input", typed)
    cli.main(["init", "--config", str(destination), "--skip-doctor", "--no-start", *extra])
    return typed


def _noninteractive_init(destination: Path, *extra: str) -> None:
    cli.main(["init", "--config", str(destination), "--non-interactive", "--skip-doctor", "--no-start", *extra])


# ---------- 检测 ----------


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        ("Asia/Singapore", "Asia/Singapore"),
        ("Asia/Shanghai", "Asia/Shanghai"),
        ("Europe/Berlin", "Europe/Berlin"),
        ("UTC", None),
        ("Etc/UTC", None),
        ("Etc/GMT-8", None),
        ("GMT", None),
        ("Universal", None),
        ("Nowhere/Nothing", None),
        ("asia/singapore", None),  # 检测结果不做大小写猜测：不是规范名就算检测不到
        ("", None),
        (None, None),
    ],
)
def test_system_time_zone_detection_only_accepts_a_real_place(
    monkeypatch: pytest.MonkeyPatch, reported: object, expected: str | None
) -> None:
    _system_zone(monkeypatch, _returns(reported))
    assert locale_setup.detect_system_timezone() == expected


def test_a_failing_tzlocal_counts_as_not_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    _system_zone(monkeypatch, _raises)
    assert locale_setup.detect_system_timezone() is None


@pytest.mark.parametrize(
    ("zone", "region"),
    [
        ("Asia/Singapore", "SG"),
        ("Singapore", "SG"),
        ("Asia/Shanghai", "CN"),
        ("Asia/Chongqing", "CN"),
        ("Asia/Harbin", "CN"),
        ("PRC", "CN"),
        ("Asia/Chungking", "CN"),
        ("Asia/Urumqi", "CN"),
        ("Asia/Kashgar", "CN"),
        ("Asia/Kuala_Lumpur", None),  # 和新加坡同一份时区规则，但不是新加坡的放假安排
        ("Asia/Hong_Kong", None),  # 香港不是中国大陆的放假安排
        ("Europe/Berlin", None),
        ("UTC", None),
    ],
)
def test_region_is_only_derived_for_regions_with_bundled_holiday_data(zone: str, region: str | None) -> None:
    assert locale_setup.region_for_timezone(zone) == region


# ---------- init 交互 ----------


def test_interactive_init_accepts_the_detected_singapore_zone_on_enter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "sg" / "config.yaml"
    _system_zone(monkeypatch, _returns("Asia/Singapore"))

    _interactive_init(monkeypatch, destination, [""])

    assert "Detected system time zone: Asia/Singapore (region: SG)" in capsys.readouterr().out
    assert _locale_of(destination) == {"timezone": "Asia/Singapore", "region": "SG"}
    config = HabitusConfig.from_file(destination)
    assert (config.locale.timezone, config.locale.region) == ("Asia/Singapore", "SG")


def test_interactive_init_on_a_shanghai_machine_keeps_shanghai_and_cn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "cn" / "config.yaml"
    _system_zone(monkeypatch, _returns("Asia/Shanghai"))

    _interactive_init(monkeypatch, destination, [""])

    config = HabitusConfig.from_file(destination)
    assert (config.locale.timezone, config.locale.region) == ("Asia/Shanghai", "CN")


def test_interactive_init_in_berlin_asks_for_the_region(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "de" / "config.yaml"
    _system_zone(monkeypatch, _returns("Europe/Berlin"))

    # 回车接受时区；地区必填：先空着回车（要求必填）、再输错形状、最后输对
    typed = _interactive_init(monkeypatch, destination, ["", "", "Germany", "DE"])

    assert typed.call_count == 4
    output = capsys.readouterr().out
    assert "Detected system time zone: Europe/Berlin" in output
    config = HabitusConfig.from_file(destination)
    assert (config.locale.timezone, config.locale.region) == ("Europe/Berlin", "DE")


def test_interactive_init_writes_the_exact_tzdata_spelling_and_an_upper_case_region(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 用户已确认的两条：时区名大小写不对换成 tzdata 的写法（否则 macOS 上写得进、搬到 Linux 读不出）；地区码转大写
    destination = tmp_path / "spelling" / "config.yaml"
    _system_zone(monkeypatch, _returns("Asia/Singapore"))

    _interactive_init(monkeypatch, destination, ["europe/BERLIN", "de-by"])

    assert _locale_of(destination) == {"timezone": "Europe/Berlin", "region": "DE-BY"}
    config = HabitusConfig.from_file(destination)
    assert (config.locale.timezone, config.locale.region) == ("Europe/Berlin", "DE-BY")


def test_interactive_init_can_change_the_zone_to_another_valid_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "changed" / "config.yaml"
    _system_zone(monkeypatch, _returns("Asia/Singapore"))

    _interactive_init(monkeypatch, destination, ["Asia/Shanghai"])

    config = HabitusConfig.from_file(destination)
    assert (config.locale.timezone, config.locale.region) == ("Asia/Shanghai", "CN")


def test_interactive_init_asks_again_after_an_unknown_zone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "retry" / "config.yaml"
    _system_zone(monkeypatch, _returns("Asia/Singapore"))

    typed = _interactive_init(monkeypatch, destination, ["Asia/Singapur", "Asia/Shanghai"])

    assert typed.call_count == 2
    assert "Unknown time zone 'Asia/Singapur'" in capsys.readouterr().out
    assert _locale_of(destination) == {"timezone": "Asia/Shanghai", "region": "CN"}


def test_interactive_init_without_a_detected_zone_offers_singapore(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "utc" / "config.yaml"
    _system_zone(monkeypatch, _returns("UTC"))

    typed = _interactive_init(monkeypatch, destination, [""])

    assert "No system time zone detected; defaulting to Asia/Singapore" in capsys.readouterr().out
    assert "[Asia/Singapore]" in typed.call_args_list[0].args[0]
    assert _locale_of(destination) == {"timezone": "Asia/Singapore", "region": "SG"}


def test_interactive_init_end_of_input_at_the_zone_prompt_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "eof" / "config.yaml"
    _system_zone(monkeypatch, _returns("Asia/Singapore"))
    monkeypatch.setattr(cli, "_interactive", Mock(return_value=True))
    monkeypatch.setattr(cli, "_prompt_cloud_setup", Mock(return_value=default_cloud_selection()))
    monkeypatch.setattr("builtins.input", Mock(side_effect=EOFError))

    cli.main(["init", "--config", str(destination), "--skip-doctor", "--no-start"])

    assert not destination.exists()


# ---------- init 非交互 ----------


def test_noninteractive_init_uses_the_detected_zone_and_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "cn" / "config.yaml"
    _system_zone(monkeypatch, _returns("Asia/Shanghai"))

    _noninteractive_init(destination)

    output = capsys.readouterr().out
    assert "Locale: time zone Asia/Shanghai (detected from the system), region CN (from the time zone)" in output
    config = HabitusConfig.from_file(destination)
    assert (config.locale.timezone, config.locale.region) == ("Asia/Shanghai", "CN")


def test_noninteractive_init_in_a_utc_container_writes_singapore_and_says_it_is_a_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "utc" / "config.yaml"
    _system_zone(monkeypatch, _returns("UTC"))

    _noninteractive_init(destination)

    output = capsys.readouterr().out
    assert "Locale: time zone Asia/Singapore (default; no system time zone detected), region SG (default)" in output
    assert "locale.timezone" in output and "locale.region" in output
    config = HabitusConfig.from_file(destination)
    assert (config.locale.timezone, config.locale.region) == ("Asia/Singapore", "SG")


@pytest.mark.parametrize("reader", [_returns("Nowhere/Nothing"), _raises])
def test_noninteractive_init_without_a_usable_zone_falls_back_to_singapore(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reader: Callable[[], object]
) -> None:
    destination = tmp_path / "fallback" / "config.yaml"
    _system_zone(monkeypatch, reader)

    _noninteractive_init(destination)

    assert _locale_of(destination) == {"timezone": "Asia/Singapore", "region": "SG"}


def test_noninteractive_init_in_berlin_writes_sg_and_tells_the_user_to_change_the_region(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "de" / "config.yaml"
    _system_zone(monkeypatch, _returns("Europe/Berlin"))

    _noninteractive_init(destination)

    output = capsys.readouterr().out
    assert "Locale: time zone Europe/Berlin (detected from the system), region SG (default)" in output
    assert "If you are not in Singapore, set locale.region" in output
    assert _locale_of(destination) == {"timezone": "Europe/Berlin", "region": "SG"}


def test_noninteractive_force_rebuild_writes_the_detected_locale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "force" / "config.yaml"
    _system_zone(monkeypatch, _returns("Asia/Singapore"))
    _noninteractive_init(destination)
    _system_zone(monkeypatch, _returns("Asia/Shanghai"))

    _noninteractive_init(destination, "--force")

    assert _locale_of(destination) == {"timezone": "Asia/Shanghai", "region": "CN"}


# ---------- 已有配置不动 ----------


def _existing_config_in_berlin(destination: Path) -> bytes:
    initialize_config(destination, source=EXAMPLE_CONFIG)
    text = destination.read_text(encoding="utf-8")
    original = (text + "\nlocale:\n  timezone: Europe/Berlin\n  region: DE\n").encode("utf-8")
    destination.write_bytes(original)
    restrict_file_to_owner(destination)
    return original


def test_rerunning_init_without_force_leaves_an_existing_locale_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "existing" / "config.yaml"
    original = _existing_config_in_berlin(destination)
    _system_zone(monkeypatch, _returns("Asia/Singapore"))

    _noninteractive_init(destination)

    assert destination.read_bytes() == original
    assert HabitusConfig.from_file(destination).locale.timezone == "Europe/Berlin"


def test_interactive_rerun_that_declines_the_cloud_update_leaves_the_locale_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "existing" / "config.yaml"
    original = _existing_config_in_berlin(destination)
    _system_zone(monkeypatch, _returns("Asia/Singapore"))
    typed = Mock(side_effect=AssertionError("no prompt expected"))
    monkeypatch.setattr(cli, "_interactive", Mock(return_value=True))
    monkeypatch.setattr(cli, "_configure_missing_credentials", Mock())
    monkeypatch.setattr(cli, "_confirm", Mock(return_value=False))
    monkeypatch.setattr("builtins.input", typed)

    cli.main(["init", "--config", str(destination), "--skip-doctor", "--no-start"])

    assert destination.read_bytes() == original


def test_interactive_cloud_route_update_without_force_keeps_the_existing_locale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "existing" / "config.yaml"
    _existing_config_in_berlin(destination)
    _system_zone(monkeypatch, _returns("Asia/Singapore"))
    confirmations = iter((True, False))
    monkeypatch.setattr(cli, "_interactive", Mock(return_value=True))
    monkeypatch.setattr(cli, "_prompt_cloud_setup", Mock(return_value=default_cloud_selection()))
    monkeypatch.setattr(cli, "_configure_missing_credentials", Mock())
    monkeypatch.setattr(cli, "_confirm", lambda *_args, **_kwargs: next(confirmations))
    monkeypatch.setattr("builtins.input", Mock(side_effect=AssertionError("no locale prompt expected")))

    cli.main(["init", "--config", str(destination), "--skip-doctor", "--no-start"])

    config = HabitusConfig.from_file(destination)
    assert (config.locale.timezone, config.locale.region) == ("Europe/Berlin", "DE")


def test_init_from_an_existing_yaml_does_not_touch_its_locale(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "source" / "config.yaml"
    _existing_config_in_berlin(source)
    destination = tmp_path / "copy" / "config.yaml"
    _system_zone(monkeypatch, _returns("Asia/Singapore"))

    _noninteractive_init(destination, "--from-config", str(source))

    assert _locale_of(destination) == {"timezone": "Europe/Berlin", "region": "DE"}


# ---------- --force 重建时旧的日历目录（用户 2026-11-06 定：地区变了就清掉） ----------


def _existing_berlin_config_with_calendar(destination: Path) -> None:
    initialize_config(destination, source=EXAMPLE_CONFIG)
    text = destination.read_text(encoding="utf-8")
    calendar = destination.parent / "de-holidays"
    destination.write_text(
        text
        + f"\nlocale:\n  timezone: Europe/Berlin\n  region: DE\n  calendar_path: {calendar}\n"
        + "  nominal_weekend: [6]\n",
        encoding="utf-8",
    )
    restrict_file_to_owner(destination)


def test_interactive_force_rebuild_that_changes_the_region_drops_the_old_calendar_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "rebuild" / "config.yaml"
    _existing_berlin_config_with_calendar(destination)
    _system_zone(monkeypatch, _returns("Asia/Singapore"))

    _interactive_init(monkeypatch, destination, ["Asia/Singapore"], "--force")

    assert _locale_of(destination) == {"timezone": "Asia/Singapore", "region": "SG", "nominal_weekend": [6]}
    output = capsys.readouterr().out
    assert "Removed locale.calendar_path" in output and "region DE" in output and "now SG" in output
    assert HabitusConfig.from_file(destination).locale.calendar_path is None


def test_interactive_force_rebuild_that_keeps_the_region_keeps_the_calendar_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "rebuild" / "config.yaml"
    _existing_berlin_config_with_calendar(destination)
    _system_zone(monkeypatch, _returns("Europe/Berlin"))

    _interactive_init(monkeypatch, destination, ["", "DE"], "--force")

    locale = _locale_of(destination)
    assert locale["region"] == "DE" and locale["calendar_path"] == str(destination.parent / "de-holidays")
    assert "Removed locale.calendar_path" not in capsys.readouterr().out


# ---------- --force 重建：原配置里的所在地是默认值，换了要说（用户 2026-11-06 定） ----------


def test_interactive_force_rebuild_keeps_the_current_locale_on_enter_even_when_the_system_says_otherwise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "travel" / "config.yaml"
    _existing_berlin_config_with_calendar(destination)
    _system_zone(monkeypatch, _returns("Asia/Singapore"))

    typed = _interactive_init(monkeypatch, destination, [""], "--force")

    assert typed.call_count == 1  # 时区回车保留，地区跟着原配置，不再问
    locale = _locale_of(destination)
    assert (locale["timezone"], locale["region"]) == ("Europe/Berlin", "DE")
    assert locale["calendar_path"] == str(destination.parent / "de-holidays")
    output = capsys.readouterr().out
    assert "Current config: time zone Europe/Berlin, region DE." in output
    assert "Detected system time zone: Asia/Singapore (region: SG)." in output
    assert "Locale changed" not in output


def test_interactive_force_rebuild_that_changes_the_zone_says_what_changed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "moved" / "config.yaml"
    _existing_config_in_berlin(destination)
    _system_zone(monkeypatch, _returns("Asia/Singapore"))

    _interactive_init(monkeypatch, destination, ["Asia/Singapore"], "--force")

    assert _locale_of(destination) == {"timezone": "Asia/Singapore", "region": "SG"}
    assert "Locale changed: Europe/Berlin / DE -> Asia/Singapore / SG." in capsys.readouterr().out


def test_noninteractive_force_rebuild_says_when_the_detected_locale_differs_from_the_old_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "server" / "config.yaml"
    _existing_config_in_berlin(destination)
    _system_zone(monkeypatch, _returns("Asia/Singapore"))

    _noninteractive_init(destination, "--force")

    assert _locale_of(destination) == {"timezone": "Asia/Singapore", "region": "SG"}
    assert "Locale changed: Europe/Berlin / DE -> Asia/Singapore / SG." in capsys.readouterr().out


def test_noninteractive_force_rebuild_of_an_unreadable_config_still_rebuilds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # --force 正是修坏配置的出口：读不出旧所在地不能挡重建
    destination = tmp_path / "broken" / "config.yaml"
    destination.parent.mkdir(parents=True)
    destination.write_text("locale: [unclosed\n", encoding="utf-8")
    restrict_file_to_owner(destination)
    _system_zone(monkeypatch, _returns("Asia/Singapore"))

    _noninteractive_init(destination, "--force")

    assert _locale_of(destination) == {"timezone": "Asia/Singapore", "region": "SG"}
    assert "Locale changed" not in capsys.readouterr().out


def test_interactive_force_rebuild_of_a_config_without_locale_offers_the_detected_zone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 旧版 init 生成的配置没有 locale 段：等于从没选过，默认值用检测到的
    destination = tmp_path / "old" / "config.yaml"
    initialize_config(destination, source=EXAMPLE_CONFIG)
    _system_zone(monkeypatch, _returns("Asia/Shanghai"))

    _interactive_init(monkeypatch, destination, [""], "--force")

    assert _locale_of(destination) == {"timezone": "Asia/Shanghai", "region": "CN"}
    output = capsys.readouterr().out
    assert "Current config" not in output and "Locale changed" not in output


# ---------- 换所在地的影响说明（只写 11-06 读码核实过的后果） ----------


def _notice(previous: tuple[str, str], current: tuple[str, str]) -> str:
    return locale_setup.locale_change_notice(locale_setup.LocaleChoice(*previous), locale_setup.LocaleChoice(*current))


def test_moving_time_zone_says_history_is_kept_and_statistics_join_on_the_local_clock() -> None:
    text = _notice(("Europe/Berlin", "DE"), ("Europe/Berlin", "DE"))
    assert text.startswith("Locale changed: Europe/Berlin / DE -> Europe/Berlin / DE.")
    moved = _notice(("Asia/Shanghai", "CN"), ("Europe/Berlin", "CN"))
    assert "nothing already recorded is rewritten" in moved
    assert "keep the dates and times they were recorded with; new ones use Europe/Berlin" in moved
    assert "the earlier history stays on the old clock" in moved
    assert "longer or shorter than 24 hours" in moved
    assert "Holiday and workday labels" not in moved  # 地区没变


def test_shanghai_and_singapore_share_their_offset_so_times_are_unaffected() -> None:
    text = _notice(("Asia/Shanghai", "CN"), ("Asia/Singapore", "SG"))
    assert "Asia/Shanghai and Asia/Singapore have had the same UTC offset; dates and times are unaffected." in text
    assert "longer or shorter than 24 hours" not in text
    # 地区跟着从 CN 变成 SG：过去的日子按新加坡的放假重新打日型
    assert "recomputed with SG's holidays" in text
    assert "No holiday data ships" not in text


def test_a_region_without_bundled_holidays_says_day_types_stay_unknown() -> None:
    text = _notice(("Asia/Singapore", "SG"), ("Europe/Berlin", "DE"))
    assert "recomputed with DE's holidays" in text
    assert "No holiday data ships for DE (bundled: CN, SG)" in text
    assert "locale.calendar_path" in text


def test_daylight_saving_zones_do_not_count_as_the_same_offset() -> None:
    # 冬天柏林与伦敦差 1 小时、夏天也差 1 小时，但伦敦与里斯本全年一致
    assert "same UTC offset" not in _notice(("Europe/Berlin", "DE"), ("Europe/London", "DE"))
    assert "same UTC offset" in _notice(("Europe/London", "GB"), ("Europe/Lisbon", "GB"))


@pytest.mark.skipif(sys.platform != "win32", reason="Windows 上 tzlocal 走注册表换算 IANA 名，POSIX 走 /etc/localtime")
def test_windows_registry_time_zone_maps_to_a_known_iana_name() -> None:
    """阶段 3：W3 报告里没实测的那一环——真 Windows 上 tzlocal 从注册表拿到的名字必须在 tzdata 的名字表里。"""

    name = locale_setup._tzlocal_name()
    assert isinstance(name, str)
    assert name in available_timezones()


@pytest.mark.parametrize(
    ("locale", "expected"),
    [
        ({"timezone": 8}, None),
        ({"timezone": ["Asia/Shanghai"]}, None),
        ("Asia/Shanghai", None),
        (
            {"timezone": "Asia/Shanghai", "region": 86},
            locale_setup.LocaleChoice("Asia/Shanghai", locale_setup.DEFAULT_REGION),
        ),
    ],
)
def test_recorded_locale_reads_only_text_from_the_existing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, locale: object, expected: object
) -> None:
    """重建前配置里的 ``locale`` 类型不对：时区不是文本就当没选过，地区不是文本按默认值（W16：类型在读入处查）。"""

    existing = tmp_path / "config.yaml"
    existing.write_text("placeholder: true\n", encoding="utf-8")
    monkeypatch.setattr(initialization_module, "load_initialization_mapping", lambda _path: {"locale": locale})
    assert locale_setup.recorded_locale(existing) == expected
