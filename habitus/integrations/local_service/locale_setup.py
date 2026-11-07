"""``habitus-server init`` 定主体所在地（``locale.timezone``、``locale.region``）：检测系统时区，交互时请用户确认（W3）。

为什么 init 要问：时区决定"今天是哪一天"——记忆抽取按它定 Event 落在哪一天，行为侧与预测层按它定本地日界与槽位；
地区决定读哪一份当地放假安排。默认值（新加坡）只是兜底，装在别处的人不该悄悄按新加坡算。

- 检测用 ``tzlocal``：macOS、Linux 读 ``TZ`` 与 ``/etc/localtime``，Windows 把注册表里的时区名换成 IANA 名，三个平台一套
  代码。拿到的名字必须是 tzdata 名字表里有的（逐字、区分大小写；链接名如 ``Singapore``、``PRC`` 也算）；``UTC``、``Etc/*`` 这类不代表真实所在地的名字，与拿不到、抛异常一样算检测不到。
- 地区码只从"随包带了放假数据的地区"推（``holidays/<地区码>/``：SG、CN）；别的时区推不出地区，交互时请用户填，非交互时
  写 SG 并在输出里说明要去改。
- 只在新建或 ``--force`` 重建配置时调用；已有配置的 locale 由用户自己维护，init 不碰。
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import TextIO
from zoneinfo import ZoneInfo, available_timezones

from habitus.config.locale import LocaleConfig

DEFAULT_TIMEZONE = "Asia/Singapore"
DEFAULT_REGION = "SG"

# 不代表真实所在地的时区名：容器、CI、服务器常设成它们；当成"检测到"会把一个人的日子按格林尼治切。
_PLACELESS = frozenset({"UTC", "UCT", "GMT", "GMT0", "GMT+0", "GMT-0", "Greenwich", "Universal", "Zulu", "Factory"})

# 随包带了放假数据的地区，按时区推地区码。中国大陆：Asia/Chongqing、Asia/Chungking、Asia/Harbin、PRC 在 tzdata 里是
# Asia/Shanghai 的别名（同一份规则）；Asia/Urumqi（及其别名 Asia/Kashgar）是新疆的独立时区，不是别名，但同属中国大陆、
# 用同一份国务院放假安排。香港、澳门、台湾不是这份安排，不在表里。
_REGION_BY_TIMEZONE = {
    "Asia/Singapore": "SG",
    "Singapore": "SG",
    "Asia/Shanghai": "CN",
    "Asia/Chongqing": "CN",
    "Asia/Chungking": "CN",
    "Asia/Harbin": "CN",
    "PRC": "CN",
    "Asia/Urumqi": "CN",
    "Asia/Kashgar": "CN",
}


@dataclass(frozen=True)
class LocaleChoice:
    """init 定下来的所在地。"""

    timezone: str
    region: str


@cache
def _known_timezones() -> frozenset[str]:
    return frozenset(available_timezones())


def _tzlocal_name() -> object:
    from tzlocal import get_localzone_name

    return get_localzone_name()


def detect_system_timezone() -> str | None:
    """系统时区的 IANA 名；拿不到、不在 tzdata 名字表里（逐字比对）、或是不代表所在地的名字，都是 None。"""

    try:
        name = _tzlocal_name()
    except Exception:  # tzlocal 在各平台抛的异常不一（找不到、配置冲突、注册表读不了），都算检测不到
        return None
    if not isinstance(name, str) or name not in _known_timezones():
        return None
    if name in _PLACELESS or name.startswith("Etc/"):
        return None
    return name


def region_for_timezone(name: str) -> str | None:
    """能从时区确定的地区码（只限随包带了放假数据的地区）；推不出是 None。"""

    return _REGION_BY_TIMEZONE.get(name)


def canonical_timezone(text: str) -> str | None:
    """用户输入的时区名对应的规范名：完全一致直接用；只差大小写时换成规范写法（大小写不敏感的文件系统上
    ``ZoneInfo("asia/singapore")`` 也读得出，写进配置换到 Linux 就读不出了）；不认识是 None。"""

    name = text.strip()
    known = _known_timezones()
    if name in known:
        return name
    matches = [item for item in known if item.casefold() == name.casefold()]
    return matches[0] if len(matches) == 1 else None


def choose_locale(*, interactive: bool, output: TextIO, previous: LocaleChoice | None = None) -> LocaleChoice:
    """``previous`` 是重建前配置里写着的所在地（没写过就是 None）。交互时它是默认值：回车保留，要换得自己输入；
    检测结果只作参考显示，不会因为人在外地跑了一次 ``--force`` 就被悄悄换掉（用户 2026-11-06 定）。"""

    detected = detect_system_timezone()
    if not interactive:
        return _noninteractive_choice(detected, output)
    if previous is not None:
        output.write(f"Current config: time zone {previous.timezone}, region {previous.region}.\n")
    if detected is None:
        output.write(f"No system time zone detected; defaulting to {DEFAULT_TIMEZONE}.\n")
    else:
        output.write(f"Detected system time zone: {detected} (region: {region_for_timezone(detected) or 'unknown'}).\n")
    default = previous.timezone if previous is not None else detected or DEFAULT_TIMEZONE
    timezone = _ask_timezone(default, output, keep=previous is not None)
    if previous is not None and timezone == previous.timezone:
        region = previous.region
    else:
        region = region_for_timezone(timezone) or _ask_region(timezone, output)
    output.write(f"Locale: time zone {timezone}, region {region}.\n")
    return LocaleChoice(timezone, region)


def _noninteractive_choice(detected: str | None, output: TextIO) -> LocaleChoice:
    if detected is None:
        output.write(
            f"Locale: time zone {DEFAULT_TIMEZONE} (default; no system time zone detected), region {DEFAULT_REGION} "
            "(default). If you are not in Singapore, set locale.timezone and locale.region in the config file.\n"
        )
        return LocaleChoice(DEFAULT_TIMEZONE, DEFAULT_REGION)
    region = region_for_timezone(detected)
    if region is None:
        output.write(
            f"Locale: time zone {detected} (detected from the system), region {DEFAULT_REGION} (default). "
            "If you are not in Singapore, set locale.region in the config file to your ISO 3166 code (e.g. DE, US-CA).\n"
        )
        return LocaleChoice(detected, DEFAULT_REGION)
    output.write(f"Locale: time zone {detected} (detected from the system), region {region} (from the time zone).\n")
    return LocaleChoice(detected, region)


def _ask_timezone(default: str, output: TextIO, *, keep: bool = False) -> str:
    prompt_text = _prompts()[0]
    label = (
        "Time zone (IANA name; press Enter to keep the current one)"
        if keep
        else "Time zone (IANA name; press Enter to accept)"
    )
    while True:
        answer = prompt_text(label, default)
        timezone = canonical_timezone(answer)
        if timezone is not None:
            return timezone
        output.write(f"Unknown time zone {answer!r}; enter an IANA name such as Asia/Singapore or Europe/Berlin.\n")


def _ask_region(timezone: str, output: TextIO) -> str:
    prompt_required_text = _prompts()[1]
    while True:
        answer = prompt_required_text(f"Region code for {timezone} (ISO 3166, e.g. DE or US-CA)").strip().upper()
        try:
            LocaleConfig(timezone=timezone, region=answer)
        except ValueError:
            output.write("Region code must be an ISO 3166 code such as DE, SG or US-CA.\n")
            continue
        return answer


def _prompts() -> tuple[Callable[[str, str], str], Callable[[str], str]]:
    """沿用 init 那一套输入函数（输入流结束统一变成 ``WizardCancelled``，配置不写）；用时再取，测试替换得到。"""

    from habitus.integrations.local_service import cli

    return cli._prompt_text, cli._prompt_required_text


def with_locale(payload: Mapping[str, object], choice: LocaleChoice, output: TextIO | None = None) -> dict[str, object]:
    """把定下来的时区与地区写进待写配置；``locale`` 里别的项（名义周末等）原样留着。

    例外是 ``calendar_path``：它指向的放假数据是某一个地区的，地区换了还留着，读日历时会因地区对不上而整年未知、
    且没有提示。所以地区变了就清掉它、改读随包数据，并在输出里说一声；地区没变就保留（用户 2026-11-06 定）。
    """

    existing = payload.get("locale")
    locale = dict(existing) if isinstance(existing, Mapping) else {}
    previous_region = locale.get("region") or DEFAULT_REGION
    locale["timezone"] = choice.timezone
    locale["region"] = choice.region
    if locale.get("calendar_path") is not None and previous_region != choice.region:
        stale = locale.pop("calendar_path")
        if output is not None:
            output.write(
                f"Removed locale.calendar_path {stale}: it held holiday data for region {previous_region}, "
                f"and the region is now {choice.region}; the bundled holiday data is used instead.\n"
            )
    return {**payload, "locale": locale}


def settle_locale(
    payload: Mapping[str, object],
    *,
    interactive: bool,
    existing: Path | None = None,
    output: TextIO | None = None,
) -> dict[str, object]:
    """init 新建或重建配置时调用：定所在地并写进待写配置。``existing`` 是配置文件路径（新建时文件还不存在）；
    重建前后所在地变了，输出里写清从什么换成什么、会影响什么。"""

    stream = output or sys.stdout
    previous = recorded_locale(existing)
    choice = choose_locale(interactive=interactive, output=stream, previous=previous)
    if previous is not None and choice != previous:
        stream.write(locale_change_notice(previous, choice))
    return with_locale(payload, choice, stream)


def recorded_locale(existing: Path | None) -> LocaleChoice | None:
    """重建前配置里写着的所在地。没写 ``locale.timezone`` 就是 None（从没选过）；只写了时区时地区按默认值算。
    读不出来（文件坏了，正是 ``--force`` 要修的情形）也是 None，不挡重建。"""

    if existing is None or not existing.is_file():
        return None
    from habitus.integrations.local_service.initialization import load_initialization_mapping

    try:
        locale = load_initialization_mapping(existing).get("locale")
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(locale, Mapping):
        return None
    timezone, region = locale.get("timezone"), locale.get("region")
    if not isinstance(timezone, str):
        return None
    return LocaleChoice(timezone, region if isinstance(region, str) else DEFAULT_REGION)


def locale_change_notice(previous: LocaleChoice, choice: LocaleChoice) -> str:
    """所在地换了：写清换了什么、会影响什么（11-06 读码核实过的后果，只写核实过的）。

    - 已存的行为 occurrence、记忆 Event、观测都在写入时把日期与偏移定死，换了不改写（``behavior/model.py``、
      ``memory/model.py`` 的地址带本地日期）；
    - 夜批每晚从行为树全量重读算统计：节律与槽位按记录自带的偏移（``temporal/profile.py``），所以换时区前后
      是两段墙钟接在一起；
    - 日型不存盘、读时现算，"那天人在哪个地区"还没接（``runtime/conditions.py`` 模块说明的待办），所以换地区后
      过去的日子按新地区重新打日型。
    """

    lines = [f"Locale changed: {previous.timezone} / {previous.region} -> {choice.timezone} / {choice.region}."]
    lines.append("What this changes (nothing already recorded is rewritten):")
    if choice.timezone != previous.timezone:
        lines.extend(_time_zone_effects(previous.timezone, choice.timezone))
    if choice.region != previous.region:
        lines.extend(_region_effects(choice.region))
    return "".join(f"{line}\n" for line in lines)


def _time_zone_effects(previous: str, current: str) -> list[str]:
    if _same_offsets(previous, current):
        return [f"  - {previous} and {current} have had the same UTC offset; dates and times are unaffected."]
    return [
        f"  - Recorded behaviours and memory events keep the dates and times they were recorded with; new ones use {current}.",
        "  - Habit statistics join the history before and after the change on the local clock: right if you moved and keep",
        "    the same local routine; if you only corrected a wrong setting, the earlier history stays on the old clock.",
        "  - The day of the change is longer or shorter than 24 hours; reminders and missed-habit judgements around it",
        "    may be off.",
    ]


def _region_effects(region: str) -> list[str]:
    effects = [
        f"  - Holiday and workday labels of past days are recomputed with {region}'s holidays (where you were on each",
        "    day is not recorded); patterns that depend on holidays or weekends will be learned again.",
    ]
    if not _has_bundled_holidays(region):
        effects.append(
            f"  - No holiday data ships for {region} (bundled: {', '.join(_bundled_regions())}); day types stay unknown"
            f" until locale.calendar_path points to a directory of {region} holiday files."
        )
    return effects


def _same_offsets(first: str, second: str) -> bool:
    """近两年逐小时比偏移（覆盖一个完整的夏令时周期）；都一样才算时间上无影响。"""

    one, other = ZoneInfo(first), ZoneInfo(second)
    instant = datetime.now(UTC) - timedelta(days=730)
    end = datetime.now(UTC) + timedelta(days=365)
    while instant <= end:
        if instant.astimezone(one).utcoffset() != instant.astimezone(other).utcoffset():
            return False
        instant += timedelta(hours=1)
    return True


def _has_bundled_holidays(region: str) -> bool:
    from habitus.conditions.calendar_data import bundled_holiday_directory

    return bundled_holiday_directory(region) is not None


def _bundled_regions() -> tuple[str, ...]:
    return tuple(region for region in sorted(set(_REGION_BY_TIMEZONE.values())) if _has_bundled_holidays(region))


__all__ = [
    "DEFAULT_REGION",
    "DEFAULT_TIMEZONE",
    "LocaleChoice",
    "canonical_timezone",
    "choose_locale",
    "detect_system_timezone",
    "region_for_timezone",
    "settle_locale",
    "with_locale",
]
