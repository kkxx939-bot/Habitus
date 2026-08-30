"""行为语义树共享的节点、地址与目录值对象。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum

from foundation.ids import canonical_path_identity, require_safe_path_segment

_IDENTITY_SEPARATOR = "--"
_IDENTITY_TIMESTAMP = re.compile(
    r"^(?P<year>[0-9]{4})(?P<month>[0-9]{2})(?P<day>[0-9]{2})T"
    r"(?P<hour>[0-9]{2})(?P<minute>[0-9]{2})(?P<second>[0-9]{2})"
    r"(?P<microsecond>[0-9]{6})(?P<sign>[+-])(?P<offset_hour>[0-9]{2})(?P<offset_minute>[0-9]{2})$"
)
_MAX_IDENTITY_NAME_UTF8_BYTES = 200
_IDENTITY_SUFFIX_BYTES = len(b"--20000101T000000000000+0000")
_MAX_SEMANTIC_NAME_UTF8_BYTES = _MAX_IDENTITY_NAME_UTF8_BYTES - _IDENTITY_SUFFIX_BYTES

# 融合层对 behavior 名的字节预算：在语义名预算上再预留撞车消歧后缀（"-2".."-9999" 级别）的
# 空间——否则贴着预算上限的合法判断名，一旦触发消歧就会在归约期撑爆地址身份、卡死整条队列。
MAX_BEHAVIOR_NAME_UTF8_BYTES = _MAX_SEMANTIC_NAME_UTF8_BYTES - 8

# 树根的单文件词表节点（behavior://kinds.md，性质同 memory 树 profile.md 单文件直读）。
# 地址叶名不可能与它撞车：semantic_name 拒绝 .md 后缀，gap 叶名是受控枚举。
KINDS_REGISTRY_FILENAME = "kinds.md"


class BehaviorKind(str, Enum):
    """行为语义树中拥有独立 L2 文档的节点类型。

    ``OCCURRENCE`` 是一次被观测到的行为单位（可提醒或可代劳的那种；``goal`` 可空，空非空不
    决定它算不算一条）；``GAP`` 是一段观测空白（没读懂 / 未观测），与行为共享同一条时间轴。
    Episode（情景实例）仍在设计悬置中，届时再加入，不留占位成员。
    """

    OCCURRENCE = "occurrence"
    GAP = "gap"


class BehaviorLevel(int, Enum):
    """行为语义树的目录语义层级。"""

    ABSTRACT = 0
    OVERVIEW = 1
    DETAIL = 2

    @property
    def sidecar_filename(self) -> str:
        if self is BehaviorLevel.ABSTRACT:
            return ".abstract.md"
        if self is BehaviorLevel.OVERVIEW:
            return ".overview.md"
        raise ValueError("L2 uses a BehaviorAddress instead of a semantic sidecar")

    @classmethod
    def from_sidecar_filename(cls, filename: object) -> BehaviorLevel | None:
        if filename == ".abstract.md":
            return cls.ABSTRACT
        if filename == ".overview.md":
            return cls.OVERVIEW
        return None


OCCURRENCES_SEGMENT = "occurrences"
GAPS_SEGMENT = "gaps"

_KIND_DIRECTORY_PREFIXES: dict[BehaviorKind, tuple[str, ...]] = {
    BehaviorKind.OCCURRENCE: (OCCURRENCES_SEGMENT,),
    BehaviorKind.GAP: (GAPS_SEGMENT,),
}


def kind_directory_prefix(kind: BehaviorKind) -> tuple[str, ...]:
    """返回该文档类型的固定目录前缀；这里是行为树路径结构的唯一真相来源。"""

    return _KIND_DIRECTORY_PREFIXES[BehaviorKind(kind)]


def kind_for_directory_prefix(segments: tuple[str, ...]) -> BehaviorKind | None:
    """把路径开头映射回文档类型；没有任何前缀匹配时返回 None。"""

    for kind, prefix in _KIND_DIRECTORY_PREFIXES.items():
        if segments[: len(prefix)] == prefix:
            return kind
    return None


def behavior_static_directories() -> tuple[tuple[str, ...], ...]:
    """初始化必须存在的固定目录；父目录一定排在子目录之前。"""

    directories: list[tuple[str, ...]] = []
    for prefix in _KIND_DIRECTORY_PREFIXES.values():
        for depth in range(1, len(prefix) + 1):
            candidate = prefix[:depth]
            if candidate not in directories:
                directories.append(candidate)
    return tuple(sorted(directories, key=lambda parts: (len(parts), parts)))


def semantic_name(value: object, field_name: str) -> str:
    """校验可读语义名称，同时生成路径身份所需的稳定输入。"""

    name = require_safe_path_segment(value, field_name)
    if name != name.strip() or any(not character.isprintable() for character in name):
        raise ValueError(f"{field_name} must not contain surrounding whitespace or control characters")
    identity = canonical_path_identity(name, field_name)
    if identity.startswith("."):
        raise ValueError(f"{field_name} must not use a hidden-file name")
    if len(identity.encode("utf-8")) > _MAX_SEMANTIC_NAME_UTF8_BYTES:
        raise ValueError(f"{field_name} exceeds the portable path length budget")
    if identity.endswith(".md"):
        raise ValueError(f"{field_name} must not include the .md suffix")
    reserved = {
        BehaviorLevel.ABSTRACT.sidecar_filename[:-3],
        BehaviorLevel.OVERVIEW.sidecar_filename[:-3],
    }
    if identity in reserved:
        raise ValueError(f"{field_name} conflicts with a reserved semantic layer")
    return name


def behavior_local_timestamp(value: object, field_name: str) -> datetime:
    """校验用于行为事实和地址身份的带时区本地时间。"""

    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise TypeError(f"{field_name} must be a timezone-aware datetime")
    offset = value.utcoffset()
    assert offset is not None
    if offset.microseconds or offset.total_seconds() % 60:
        raise ValueError(f"{field_name} UTC offset must use whole minutes")
    # 树上不存在 UTC 时间（用户裁定）：行为时间一律是实际发生地的本地时刻+偏移。零偏移在本系统里
    # 不是"用户住在零时区"，而是上游把时间折成了 UTC 的事故信号（canonicalize 折 UTC 的坑真实
    # 踩过）——放进来会把东八区凌晨的行为错到前一天，硬拒让事故当场现形。
    if not offset:
        raise ValueError(f"{field_name} must carry a non-zero local UTC offset (a +00:00 stamp here means something upstream folded to UTC)")
    return value


def behavior_identity_name(name: object, started_at: object, field_name: str) -> str:
    """从可读语义名称和事实开始时间生成唯一的地址叶名。

    往返只保证还原到**规范身份**（NFC+casefold 后的名字）——原始写法保存在文档的 ``name``
    字段里，不从叶名还原。地址相等性按规范身份比较，与此一致。
    """

    resolved_name = semantic_name(name, field_name)
    resolved_started_at = behavior_local_timestamp(started_at, f"{field_name} started_at")
    identity = (
        f"{canonical_path_identity(resolved_name, field_name)}"
        f"{_IDENTITY_SEPARATOR}{_format_identity_timestamp(resolved_started_at)}"
    )
    if len(identity.encode("utf-8")) > _MAX_IDENTITY_NAME_UTF8_BYTES:
        raise ValueError(f"{field_name} identity exceeds the portable path length budget")
    return identity


def split_behavior_identity(value: object, field_name: str) -> tuple[str, datetime]:
    """把 URI 叶名还原为规范语义名称和用于身份的本地开始时间。"""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} identity must be a string")
    name, separator, timestamp = value.rpartition(_IDENTITY_SEPARATOR)
    if not separator or not name or not timestamp:
        raise ValueError(f"{field_name} identity must contain a semantic name and timestamp")
    resolved_name = semantic_name(name, field_name)
    started_at = _parse_identity_timestamp(timestamp, field_name)
    return resolved_name, started_at


def _format_identity_timestamp(value: datetime) -> str:
    offset = value.utcoffset()
    assert offset is not None
    offset_minutes = int(offset.total_seconds() // 60)
    sign = "+" if offset_minutes >= 0 else "-"
    absolute_offset = abs(offset_minutes)
    offset_hour, offset_minute = divmod(absolute_offset, 60)
    return (
        f"{value.year:04d}{value.month:02d}{value.day:02d}T"
        f"{value.hour:02d}{value.minute:02d}{value.second:02d}{value.microsecond:06d}"
        f"{sign}{offset_hour:02d}{offset_minute:02d}"
    )


def _parse_identity_timestamp(value: str, field_name: str) -> datetime:
    match = _IDENTITY_TIMESTAMP.fullmatch(value)
    if match is None:
        raise ValueError(f"{field_name} identity contains an invalid timestamp")
    parts = {name: int(raw) for name, raw in match.groupdict().items() if name != "sign"}
    offset_minutes = parts["offset_hour"] * 60 + parts["offset_minute"]
    if match.group("sign") == "-":
        offset_minutes = -offset_minutes
    try:
        resolved = datetime(
            parts["year"],
            parts["month"],
            parts["day"],
            parts["hour"],
            parts["minute"],
            parts["second"],
            parts["microsecond"],
            tzinfo=timezone(timedelta(minutes=offset_minutes)),
        )
    except ValueError as exc:
        raise ValueError(f"{field_name} identity contains an invalid timestamp") from exc
    if _format_identity_timestamp(resolved) != value:
        raise ValueError(f"{field_name} identity timestamp is not canonical")
    return resolved


def is_ascii_digits(value: object) -> bool:
    """只接受行为树日期目录使用的 ASCII 十进制数字。"""

    return isinstance(value, str) and value.isascii() and value.isdigit()


# TODO(BHV-TREE-REBUILD-001): 行为树按已定稿的新节点设计整体重建；本 TODO 是唯一规格来源。
# 【状态】树核心（本文件/uri/schema 五件套/两份 yaml/document/writer/kinds）已落地并经三路评审
# 修正；归约写入层已落地于 ``behavior/reduction/``（五条死规则+staged 检查点+封口扫描+消费账本；
# 判重守卫落融合校验（fusion_v2）、没读懂判断入库（fusion_v3））；kinds 已并入地址空间
# （behavior://kinds.md，树根单文件节点）；L0/L1 生成器已落地于 ``behavior/semantic/``
# （日叙述并入同日空白、月/年上卷、digest 短路；结构落地，提示词调优子项遗留——见 generator.py）；
# Runtime 组合根接线已完成（BHV-RUNTIME-001 关闭：Config.behavior 标量组可选启用、
# Runtime/behavior.py 组装 + 融合/归约两个常驻 Worker + 观测投递正门 + 健康面，端到端接缝
# 测试在 tests/unit/runtime/test_behavior_pipeline.py）。**进程内链路**至此全部走完；明确的
# 剩余缺口：① 观测投递的 HTTP 端点未做——云侧行为 agent 够不着进程内方法，随上游契约接入时
# 与认证/协议一起定（正门已在 ``Runtime.deliver_behavior_observations``）；② BHV-FUSION-003
# 只闭合了窗口注入一半，容量类配置随 BHV-LIFECYCLE-001 同批；③ 各处点名的实测调优
# （融合/kinds/语义三处提示词）与生命周期欠账。
# 【裁定】episodes 零占位维持（用户确认，见下方 BHV-EPISODE-002 的新定位）；links 归信封层
# （用户确认，沿 memory 树先例）；basis 步骤时间窗校验删除（用户确认，见 validators.py）。
# 与用户数十轮讨论逐条裁定，吸收并关闭旧 TODO(BHV-TREE-TIMEBASE-001)（其地址时间基问题由下述
# 地址设计解决）。按"定位 → 树形状 → occurrence → gap → 写入层 → 边界与缺口 → 已关闭支线 →
# 待定"组织。
#
# ── 定位与三根柱子 ─────────────────────────────────────────────────────────────────────
# - 行为树是行为数据的**唯一汇聚点**：融合的判断在归约层被处理成规整的行为数据，供给两个方向——
#   时间预测树要数字，预测层语义关联要语义；两边的数据源都只有它（判断存储会释放，语义不搬即丢）。
# - 三根柱子互证：**纯 add-only**（可信：过去不变；封口后引用在机械上不可能到达，可变性没有生产者，
#   Outcome/追加旁路整体取消）；**单一写入口**（可溯：树的每个字都走 观测→融合→归约 正门，预测层
#   永不回写；拒绝提醒也是行为、从正门进）；**双面字段**（够用：数字面是写入时刻定格的派生值、事后
#   不可复现，语义面是从将释放的判断存储搬运的自足语义）。
# - 双面不拆节点：memory 的多节点按"事实种类"分，而两面是**同一事实的两个投影**（同身份同生命周期
#   同溯源）——memory 自己的答案就是"正文与结构字段属于同一个原子文件"。分组在 schema 字段角色上
#   显式化，不付存储层代价。
#
# ── 树形状 ────────────────────────────────────────────────────────────────────────────
#     behavior://
#       occurrences/YYYY/MM/DD/{原始名}--{ts}.md   行为（add-only，双面）
#       gaps/       YYYY/MM/DD/{类型}--{ts}.md     观测空白（add-only；类型=没读懂|未观测）
#       episodes/   （代码零占位，用户裁定）        预测闭环成熟后再设计（BHV-EPISODE-002）
#       kinds.md                                    类型词表（已并入本地址空间：树根单文件节点，
#                                                   URI behavior://kinds.md（REGISTRY 节点类型）；
#                                                   性质同 memory profile.md 单文件直读）
#     各级日期目录带 .abstract.md/.overview.md（L0/L1 由 behavior/semantic/ 生成与刷新；调优
#     子项见 generator.py 的 TODO(BHV-SEMANTIC-004·调优)）。
# - 目录归一：旧 behaviors/events + behaviors/outcomes + episodes 三处 → 上述形状；BehaviorKind
#   改为 OCCURRENCE|GAP（EPISODE 成员随 BHV-EPISODE-002 定形态时再加）。机制层（地址值对象、
#   URI、codec、锁/读回、目录容量）全部保留（CAS 随追加通道一并退役），只换词表与节点。
# - gap 节点使行为与空白在同一条时间轴上（一次范围读得一天完整图景，L0/L1 日摘要可如实含空白），
#   并与"缺失必须记录不能解释"纪律同构；上游没接覆盖信号时树里自然没有"未观测"节点 → 曝光分母
#   自然数满全部时段，退化假设零特例代码。gap 字段极薄：起止时间 + 溯源（没读懂→judgement_ids/
#   observation_ids；未观测→信号来源标识，等上游契约）。
#
# ── occurrence 字段（schema 按角色分面：address / numeric / semantic / system）──────────────
# - 时间纪律（用户裁定，推翻我最初的 UTC 方案）：**occurrence 上全部时间统一为本地时间+显式偏移**，
#   available_at 也不例外——带偏移的本地时间就是完整瞬时，混用两种约定才是"取错时间"的事故源；
#   换算只在归约写入时发生一次，树内与读侧零换算。UTC 归一保持在观测/判断层的入口纪律，不进树。
#   时序比较一律先解析为瞬时；地址文本不承诺时间序。
# - 地址：occurred_on（本地日历日，即目录）+ name（**原始名**，融合层原话，永不变；kind_token 可修
#   而地址不动）+ started_at（本地+偏移）+ utc_offset_minutes（显式成分，自校验 occurred_on ==
#   started_at 的本地日；canonicalize 折 UTC 的旧坑由"序列化一律保留偏移的字符串"挡住）。
# - 数字面（时间预测树夜批读；机器类型白名单、全部必填）：
#     kind_token         归一 token ← kinds 词表（写入层 LLM 复用/新建；融合层不定身份——提示词
#                        空间实测极敏感、词表进 FUSION_VERSION 会天天漂、全局状态不进流式判断层）
#     status             ongoing|completed|interrupted|abandoned  ← 链尾判断逐字，零发明
#     status_basis       observed|inferred|observation_lost       ← 同上；interrupted 与
#                        observation_lost 之分是删失纪律的根
#     last_observed_at   最后支持观测的时刻（≠ 结束时刻）
#     onset_available_at 链头判断的 evidence_ready_at——防标签泄漏的切点（终审补回的字段）
#     reminded           开始前窗口内被提醒过与否 ← 干预账本（干预混淆）
# - 语义面（判断存储会释放，此处必须自足）：goal（可空；一条 occurrence 是"可提醒或可代劳的行为
#   单位"而不是"任何动作"——2026-08-30 裁定，无意识小动作与过渡帧在融合层归属为空、不进树，
#   曝光分母靠"这段在观测"而非动作段落树）、summary（链内有序拼接）、
#   subjects（多值）、place（可空，等上游）、original_name（可空，仅撞车消歧记录非空）、
#   basis[{semantics, observation_ids, 起止, available_at}]（每步时间在写入时从观测物化）。
# - links 在文档**信封层**（用户裁定，沿 memory 树同一机制：links 是文档间的结构关系，与
#   created_at 同级的元数据，不是本篇的语义内容）：**只有前向**，concurrent_with / results_from
#   → 先前 occurrence URI；add-only 下不维护 backlinks，读侧取对称闭包——沿融合层同一纪律。
# - 溯源（system 角色，不渲染进正文）：judgement_ids / observation_ids / source_refs /
#   fusion_version / reduction_version。
# - 旧 schema 整体退役：confidence、conflicts、trigger、constraints、closure_reason、actions 的
#   全部 14 个子字段（actor/method/parameters/target_refs/…）——无源或 coding-agent 词表；
#   vocabulary.py 的 EVENT_STATUSES 等任务态词表随之替换为融合层四值/三值。
# - schema 机制小改：BehaviorFieldRole 加 NUMERIC/SEMANTIC（CONTENT 随 episodes 退出，零占位）；
#   numeric 角色强制机器类型白名单+必填；system 不渲染；renderer 保持手写但加守卫测试
#   "每个非 system 字段必须出现在渲染结果里"。
#
# ── 写入层（归约；零发明）────────────────────────────────────────────────────────────────
# - 封口是机械推论不是语义判断：融合的一切跨窗口引用只能指向 recent_before 的上下文
#   （FUSION_CONTEXT_LIMIT=8 / FUSION_CONTEXT_LOOKBACK_SECONDS=3600s，用户裁定 1 小时，
#   唯一出处 behavior/fusion/config.py，归约层只准 import 不准自带数），链尾 evidence_ready_at
#   老出窗口后任何判断都不可能再引用它 → 链闭合、信息已定、物化。判据须同时对墙钟与融合队列
#   前沿（oldest_uncommitted 段的最早 available_at）出窗——停机补算的积压段 cutoff 是历史时刻，
#   只看墙钟会把还能被引用的链提前封口。晚到的修正/结果在机械上不存在（取消追加旁路的依据）。
# - 五条死规则（用户一口气裁定）：
#   ① 链头取值：occurrence 的 name 与 started_at 取链头**第一条未被替换**判断的值；supersedes
#     替换链头时新判断即新链头——替换者继承被替换者在链中的**位置**（不按自己的时间重排，
#     时间修正不会让行为改名）。修正把开始时间改到晚于链上延续段 = 产物内部矛盾（延续不可能
#     早于事件自己的开始），该链隔离不落树、每轮留信号（用户裁定：矛盾产物不进树，同 supersedes
#     成环一类处置）。
#   ② 撞车消歧：同批出现同地址（同名判断共享最早观测）时，后序一条名字加确定性序号后缀，
#     原始名照存语义面 original_name 字段（已落 schema）——地址是身份不是语义。（后续裁定：
#     判重只在融合层解决——同主体同刻开始的同名行为在融合校验里打回让模型合并（fusion_v2）；
#     本条后缀只是防卡死的保命阀 + 融合判重率的探测信号，出现率不为零就拿真实样本去修融合。
#     消歧记录即已知重复：original_name 非空的 occurrence **预测树夜批与语义层一律不计入**——
#     标记在写入时机械打好（地址撞车这一事实），下游只认标记、不做判断。）
#   ③ others_present（后续裁定：撤销）——系统只有一个观测对象、一个主体，"旁人在场"不属于本系统
#     的记录范围；被分流的旁人判断保持只分流不归约，语义面不设 others_present 字段。
#   ④ 两个"日"分界：树只认日历日——行为按**实际发生时刻**落目录，没有别的选项：时间预测树
#     直接按钟面槽位对实际时刻计数，凌晨的行为就该落在凌晨的槽里。"这属于前一晚"式的归属是
#     语义解释，归语义关联层，树与预测树零处理、不为它建任何结构。
#   ⑤ staged 幂等（抄融合 StagedFusion 的检查点形状）：kind 归一、reminded 查询这些会随时间漂移
#     的输入全部发生在 stage 之前；stage 之后只有确定性落盘——否则崩溃重试间词表/账本已变，
#     同地址不同字节，add-only 撞车卡死串行队列。
# - 其余规则：continues 合并、supersedes 在缓冲内就地替换（全史留在判断存储，树存修正后视图）；
#   summary/basis/goal 确定性拼接不再调模型（goal 为链内非空值按序去重拼接——用户裁定原封不动
#   保留，语义关联层要用，不许替模型二选一或置空丢失）；跨链 links 只指向**已封口**目标，目标未封则本链推迟封口
#   （封口按依赖拓扑排）；kind 归一是写入层唯一的 LLM 触点（对齐名字，不发明判断——与 memory 链
#   "解析 LLM 产候选、Editor 按 page_id 定身份"同构）；behavior=空的判断不进树，但在释放前转成
#   带起止的"没读懂"gap 节点（语义关联要用：「没发生」与「发生了但没读懂」是两种空白）。
#
# ── 日循环边界与上游缺口 ─────────────────────────────────────────────────────────────────
# - 树滞后约一个融合回看窗（现 1h，唯一出处 behavior/fusion/config.py）：当日实况由读者两段拼接
#   ——已封口部分读树、未封口的最近一个窗口读判断存储（2026-08-30 改定：原料**消费即释放**，
#   判断在发布后即删、判断存储里没有全天；释放门槛因此只剩"已发布"一个，不再是双消费者）。
#   夜批读到的"昨天"缺傍晚尾巴的处理（推迟批时点 or 连未封缓冲一起读）见待定。
# - 上游契约缺口三项（按现状写入设计，接入时再具体化——用户确认）：place 字段；观测 modality 加
#   交互模态（数字拒绝走正门所需）；覆盖信号走运维通道（观测层明确拒收心跳、指向的运维通道尚不
#   存在；信号到来前"未观测"gap 无生产者，退化=视作全程在看）。
#
# ── 已关闭的支线 ──────────────────────────────────────────────────────────────────────
# - Outcome/amendments 追加通道：封口窗口使晚到引用机械上不可能，取消；结果由结果方 occurrence 的
#   前向 results_from 表达。
# - 旁侧账本：被 gap 树节点取代（时间轴统一、纪律同构、退化自动成立）。
# - kind 归一放融合层：被三条硬约束否决（提示词敏感、版本漂移、全局状态），维持写入层。
# - 数字面/语义面拆成两类节点：两面是同一事实的投影，拆分复活跨文档事务与镜像地址的全部代价；
#   分组在 schema 角色层显式化。
#
# ── 待定与悬置 ────────────────────────────────────────────────────────────────────────
# - TODO(BHV-EPISODE-002)（用户裁定：留到预测执行层成形之后，代码零占位）：episode 记载的是
#   **走完预测闭环的成熟行为**——一个长期事件/动作经历了"预测→主动提醒→主体认可→（最终）代劳"
#   的完整生命周期后，这段历程与它的上下文需要一个地方记下来（例：'每晚 22:30 锁门'从被预测、
#   被提醒、被主体确认，到系统可代劳，episode 记录这一形成过程及其情景上下文）。字段、写入者、
#   以及预测层产出如何过"观测→融合→归约"单一写入口的张力都未设计——用户明说"还没想好怎么写"，
#   等预测算法与两级执行模型（提醒确认→稳定后直接执行）落地后再定形态；届时若预测层需要回写，
#   必须重审单一写入口纪律，不得旁路。
# - L0/L1 生成器已落地（遗留调优子项：提示词须按纪律实测调优，见 behavior/semantic/generator.py）；
#   生命周期（BHV-LIFECYCLE-001，判断释放改双消费者门槛）；
#   夜批傍晚滞后的取舍；覆盖信号收件箱形态（随上游契约）。
# - 实施顺序建议：schema 角色扩展 + occurrences/gaps yaml + 新词表（一个提交）→ 写入层（staged +
#   五条死规则）→ kinds 并入地址空间 → 旧三类 schema 与镜像机制退役。
# - 观察项：无目标动作段对 kinds 词表的膨胀压力（监控支持度=1 的条目占比）。


@dataclass(frozen=True)
class BehaviorAddress:
    """唯一映射到一个行为语义 L2 Markdown 文档的地址。"""

    kind: BehaviorKind
    occurred_on: date
    name: str = field(compare=False)
    started_at: datetime = field(compare=False)
    _identity_name: str = field(init=False, repr=False, compare=True)

    def __post_init__(self) -> None:
        kind = BehaviorKind(self.kind)
        if isinstance(self.occurred_on, datetime) or not isinstance(self.occurred_on, date):
            raise TypeError("behavior address occurred_on must be a date without a time")
        name = semantic_name(self.name, f"{kind.value} name")
        started_at = behavior_local_timestamp(self.started_at, f"{kind.value} started_at")
        if self.occurred_on != started_at.date():
            raise ValueError("behavior address date must match the local started_at date")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(
            self,
            "_identity_name",
            behavior_identity_name(name, started_at, f"{kind.value} name"),
        )

    @property
    def identity_name(self) -> str:
        return self._identity_name

    @classmethod
    def occurrence(cls, occurred_on: date, name: str, started_at: datetime) -> BehaviorAddress:
        return cls(BehaviorKind.OCCURRENCE, occurred_on, name, started_at)

    @classmethod
    def gap(cls, occurred_on: date, gap_kind: str, started_at: datetime) -> BehaviorAddress:
        """空白节点的叶名以空白类型（没读懂/未观测）为语义名。"""

        return cls(BehaviorKind.GAP, occurred_on, gap_kind, started_at)

    @classmethod
    def from_identity(
        cls,
        kind: BehaviorKind,
        occurred_on: date,
        identity_name: str,
    ) -> BehaviorAddress:
        """从 URI 或物理文件叶名恢复完整地址值对象。"""

        normalized_kind = BehaviorKind(kind)
        name, started_at = split_behavior_identity(identity_name, f"{normalized_kind.value} name")
        return cls(normalized_kind, occurred_on, name, started_at)


@dataclass(frozen=True)
class BehaviorDirectory:
    """严格限定于已确认行为语义树的目录地址。"""

    parts: tuple[str, ...] = field(default=(), compare=False)
    _identity_parts: tuple[str, ...] = field(init=False, repr=False, default=())

    def __post_init__(self) -> None:
        if isinstance(self.parts, str) or not isinstance(self.parts, tuple):
            raise TypeError("behavior directory parts must be a tuple of strings")
        parts = tuple(self.parts)
        object.__setattr__(self, "parts", parts)
        self._validate(parts)
        object.__setattr__(self, "_identity_parts", parts)

    @property
    def identity_parts(self) -> tuple[str, ...]:
        return self._identity_parts

    @classmethod
    def _validate(cls, parts: tuple[str, ...]) -> None:
        if not parts:
            return
        kind = kind_for_directory_prefix(parts)
        if kind is not None:
            cls._validate_dated_branch(parts, prefix_length=len(kind_directory_prefix(kind)))
            return
        if parts in behavior_static_directories():
            return
        raise ValueError("behavior directory is outside the confirmed tree")

    @staticmethod
    def _validate_dated_branch(parts: tuple[str, ...], *, prefix_length: int) -> None:
        values = parts[prefix_length:]
        if len(values) > 3:
            raise ValueError("behavior dated directory is deeper than day level")
        widths = (4, 2, 2)
        labels = ("year", "month", "day")
        for value, width, label in zip(values, widths, labels, strict=False):
            if not isinstance(value, str) or len(value) != width or not is_ascii_digits(value):
                raise ValueError(f"behavior {label} directory has an invalid format")
        if values and not 1 <= int(values[0]) <= 9999:
            raise ValueError("behavior year directory is outside the calendar range")
        if len(values) >= 2 and not 1 <= int(values[1]) <= 12:
            raise ValueError("behavior month directory is outside the calendar range")
        if len(values) == 3:
            try:
                date(int(values[0]), int(values[1]), int(values[2]))
            except ValueError as exc:
                raise ValueError("behavior day directory is not a valid calendar date") from exc

    @classmethod
    def root(cls) -> BehaviorDirectory:
        return cls()

    @classmethod
    def occurrences(
        cls,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
    ) -> BehaviorDirectory:
        return cls._dated(kind_directory_prefix(BehaviorKind.OCCURRENCE), year, month, day)

    @classmethod
    def gaps(
        cls,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
    ) -> BehaviorDirectory:
        return cls._dated(kind_directory_prefix(BehaviorKind.GAP), year, month, day)

    @classmethod
    def _dated(
        cls,
        prefix: tuple[str, ...],
        year: int | None,
        month: int | None,
        day: int | None,
    ) -> BehaviorDirectory:
        if year is None:
            if month is not None or day is not None:
                raise ValueError("behavior month or day requires a year")
            return cls(prefix)
        if isinstance(year, bool) or not isinstance(year, int):
            raise TypeError("behavior directory year must be an integer")
        parts = [*prefix, f"{year:04d}"]
        if month is None:
            if day is not None:
                raise ValueError("behavior day requires a month")
            return cls(tuple(parts))
        if isinstance(month, bool) or not isinstance(month, int):
            raise TypeError("behavior directory month must be an integer")
        parts.append(f"{month:02d}")
        if day is None:
            return cls(tuple(parts))
        if isinstance(day, bool) or not isinstance(day, int):
            raise TypeError("behavior directory day must be an integer")
        parts.append(f"{day:02d}")
        return cls(tuple(parts))

    @classmethod
    def for_address(cls, address: BehaviorAddress) -> BehaviorDirectory:
        if not isinstance(address, BehaviorAddress):
            raise TypeError("address must be a BehaviorAddress")
        occurred_on = address.occurred_on
        return cls._dated(
            kind_directory_prefix(address.kind),
            occurred_on.year,
            occurred_on.month,
            occurred_on.day,
        )

    def parent(self) -> BehaviorDirectory | None:
        if not self.parts:
            return None
        return BehaviorDirectory(self.parts[:-1])

    def lineage(self) -> tuple[BehaviorDirectory, ...]:
        directories: list[BehaviorDirectory] = []
        current: BehaviorDirectory | None = self
        while current is not None:
            directories.append(current)
            current = current.parent()
        return tuple(directories)


__all__ = [
    "GAPS_SEGMENT",
    "KINDS_REGISTRY_FILENAME",
    "MAX_BEHAVIOR_NAME_UTF8_BYTES",
    "OCCURRENCES_SEGMENT",
    "BehaviorAddress",
    "BehaviorDirectory",
    "BehaviorKind",
    "BehaviorLevel",
    "behavior_identity_name",
    "behavior_local_timestamp",
    "behavior_static_directories",
    "is_ascii_digits",
    "kind_directory_prefix",
    "kind_for_directory_prefix",
    "split_behavior_identity",
]
