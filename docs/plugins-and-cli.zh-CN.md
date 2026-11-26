# 插件与命令行

[English](plugins-and-cli.md) | 中文 · [← README](../README.zh-CN.md)

## 插件怎么工作

### 每个宿主事件做什么

命令行 hook 类宿主（Codex、Claude Code、ZCode、DeepSeek Harness）的每个 hook 脚本只调一个共享的分发器；需要联网的慢活交给脱离出去的后台进程，hook 本身尽快返回：

| 事件 | hook 前台做的 | 后台进程做的 |
|---|---|---|
| SessionStart | 检查服务兼容、建会话状态 | 补送这个会话和同一宿主其他会话卡住的转写；ZCode 与 DeepSeek Harness 没有 SessionEnd，这里还让宿主那边已经没有的会话的状态与转写镜像退场 |
| UserPromptSubmit | 召回记忆、取提醒，拼成注入块交给宿主 | — |
| Stop | 读这一轮的转写增量，放进本地 outbox | 把 outbox 逐条送服务 |
| SessionEnd | 同 Stop | 送完后让服务把这段对话封段，再清理过期的会话状态 |
| PreCompact | 前台直接读增量、送完并封段（压缩前不留尾巴） | — |

OpenCode 与 pi 是进程内插件：它们在各自的事件上（OpenCode 的 `chat.message`、`session.idle`、`experimental.session.compacting`、`session.deleted`；pi 的 `before_agent_start`、`agent_settled`、`session_before_compact`、`session_shutdown`）调用同一套逻辑。

一段增量进 outbox 之前，消息与工具结果里的图片等媒体载荷（base64、`data:` URL）换成一个小占位，只留类型、媒体类型、去掉的载荷的字节数与 sha256，文字一个字不动；本地服务对只有媒体的工具结果就按占位记原载荷的大小与 sha256，与不经插件直接收到原载荷时记的逐值相同。一批超过服务的请求上限（`/api/v1/capabilities` 里的 `max_request_bytes`，老服务不给就按 16 MiB）时，在轮的边界拆成几个请求依次送，一轮不拆开。单独一轮仍超限时，本地服务有 `remember_parts_v1` 特性就把这一批的请求体无损切成几块（每块都在上限以内）按顺序上传，本地服务收齐、核对摘要后拼回完整的一轮，再与一次送来的完全一样处理；一轮在语义上不拆。任何一块失败，这一批都留在 outbox 里，下次从第一块整批重送（本地服务重启丢了暂存也一样），同一个投递号不会被记两遍。拼回的一轮与一次送来的一样受会话源的单条上限约束（`conversation.source.max_envelope_bytes`，默认 64 MiB），超了同样标 blocked。本地服务没有 `remember_parts_v1` 时这一轮不送：留在 outbox 里标 blocked，`habitus doctor` 写明是哪个会话、这一批多大、超了多少。

### 召回与提醒的注入

你说一句话时，插件先确认服务兼容（API 版本 `1.0`、支持这个会话协议、具备 `conversation_cursor`、`flush`、`recall`、`remember`、`remember_idempotency_v1` 五个特性）并且服务的召回已就绪（`/ready` 结果里的 `recall_ready`：只是记忆写入侧停住，例如有 Job 失败或在等模型恢复时，它仍为 true，召回照常），再调召回；服务有 `reminders_v1` 特性（只在预测层开着时公布）时，再取到期的提醒（一次最多 3 条，可配；插件要的条数超过本地服务的 `foresight.outlet_claim_limit` 时，本地服务按自己的上限给）。子代理的"一句话"是主 agent 写的任务，它的结果已经回到主会话的转写里：子代理的会话不召回、不取提醒，也不送进记忆（Codex 的子线程从 rollout 开头的 `session_meta` 认，Claude Code 认 `agent_id`，OpenCode 与 DeepSeek Harness 认父会话）。非交互会话同样跳过：Claude Code 的 `claude -p` 与 Agent SDK 会话（hook 进程环境里 `CLAUDE_CODE_ENTRYPOINT` 以 `sdk-` 开头）、Codex 的 `codex exec` 会话（rollout 开头 `session_meta.source` 为 `exec`），不召回、不取提醒、不送转写，也不留会话状态。

这一路有一个总预算（默认 20 秒，`HABITUS_PLUGIN_PROMPT_BUDGET_MS`），给召回留出足够时间、也保证整个 hook 在宿主给的 30 秒内结束；服务没有响应时这一轮什么都不注入，你的话照常交给 agent。取提醒出错只丢掉提醒，召回到的记忆照样注入。

注入块的格式：

```
<habitus-memory-context receipt="<32 位十六进制随机数>">
{"format":"habitus_memory_context_v1","content":"…"}
</habitus-memory-context>
```

内容是 JSON 字符串，其中的 `<` 与 `>` 被转义，块不会被内容提前闭合。插件把每个块的随机数与摘要记在会话状态里（留最近 32 条；读转写一直失败期间注入的一条不丢，读成之后再收回）；下次送转写时按这份记录把注入块从转写里剥掉，只剩注入块的消息整条丢弃，所以注入的内容不会被当成你说的话送回服务。

提醒用同一种包装（标签 `habitus-reminder`，里面同样是 `{"format":"habitus_memory_context_v1","content":"…"}`），`content` 是一段给 agent 的指令；下面是 `content` 字段里的文字：

```
Habitus 的提醒，不是用户说的话。在做任何别的事之前，先用 <宿主的提问工具> 问用户下面这件事：
题目原样用这句话、给下面列出的全部选项、允许他自己填别的话。问完照常做他让你做的事；他怎么答你都不用另外处理，系统自己会记。
1. 『<提醒的话>』
   选项：现在就做 / 这次不用 / 知道了 / 已经做过了
```

agent 问没问出口、你怎么答的，服务都从送回的转写里认出来。

### 送转写：切批、outbox 与重送

插件只在宿主标出的"一轮结束"处切批：Codex 是每轮最终答复（`phase = final_answer`）且没有挂着的工具调用；OpenCode / ZCode 是完成的、`finish` 不是 `tool-calls` 的答复；pi 是 `stopReason` 不是 `toolUse` / `pending`、也没有被宿主撤回重试的答复；DeepSeek Harness 是 `turn/end` 事件。做到一半的那一轮留在宿主的转写里，等它做完再一起送。

每一批先写进本地 outbox（一个会话一个目录、文件名按转写位置排序），再按顺序送服务；每批带上服务给的起始序号与一个确定的投递号，同一批重送不会被记两遍。服务答"不可重试"的错误时这一批标为 blocked，之后的批不再越过它（`habitus doctor` 会报出来）；可重试的错误放回队列，下次再送。

### 失败时的表现

插件出任何错都不拦宿主：写一条操作日志，照常给宿主一个空的成功输出。召回、取提醒因为连不上、不兼容、未就绪、超出这一轮的时间预算或服务回错而没拿到东西，也记一条、带原因；同一会话里接连是同一个原因时只记第一次，过一小时仍是它再记一次。`HABITUS_PLUGIN_DEBUG=1` 时 hook 与 OpenCode / pi 插件还把堆栈打到 stderr；送转写的后台进程没有 stderr，它的错误只在操作日志里。插件配置本身读不出来（例如 `HABITUS_URL` 不是本机地址）时往 stderr 写一行原因。

### 插件的环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `HABITUS_MEMORY_ENABLED` | `true` | 关掉后插件什么都不做 |
| `HABITUS_URL` | 读 `connection.json`，否则 `http://127.0.0.1:8787` | 服务地址，只接受本机回环地址，不跟随重定向 |
| `HABITUS_PLUGIN_STATE_DIR` | 设了 `HABITUS_CONFIG_FILE` 时是配置文件旁的 `agent-plugin/`，否则 `~/.habitus/agent-plugin` | 插件状态目录。`habitus-server init` 用自定义配置路径时，会在那份配置旁写 `agent-plugin/connection.json`，同时改写默认的 `~/.habitus/agent-plugin/connection.json`（设了 `HABITUS_PLUGIN_STATE_DIR` 时是那里的一份），不带 `HABITUS_CONFIG_FILE` 的宿主也连这个服务；`habitus-server` 启动时重写的是同几份 |
| `HABITUS_PLUGIN_TIMEOUT_MS` | `15000` | 后台送转写等请求的超时；`habitus doctor` / `habitus plugin status` 问 Codex hook 批没批也用它（到点没答就报查不到，Codex 进程随即停掉，过 1 秒还在就强杀） |
| `HABITUS_PLUGIN_PROMPT_BUDGET_MS` | `20000` | UserPromptSubmit 这一路的总预算 |
| `HABITUS_PLUGIN_REMINDER_LIMIT` | `3` | 一次最多取几条提醒（1–20；本地服务最多给它的 `foresight.outlet_claim_limit` 条） |
| `HABITUS_PLUGIN_REMINDER_TIMEOUT_MS` | `2000` | 取提醒的超时 |
| `HABITUS_PLUGIN_MAX_TRANSCRIPT_BYTES` | 64 MiB | 一次从游标往后读多少转写，按媒体载荷换成占位之后的大小算（剩下的紧接着再读）。剥之前单独一行就超过它的，不管在文件的什么位置、是什么记录，一律跳过：这一行的内容不进记忆，它所在那一轮的其余记录照常送，跳过记进操作日志，`habitus doctor` 报提示、写明是哪个会话的第几行。只剩文字仍超过它的一轮读不出来：这个会话停采，直到调大这个值，`habitus doctor` 会列出是哪个会话 |
| `HABITUS_PLUGIN_SESSION_GRACE_HOURS` | `24` | 会话状态退场的宽限期 |
| `HABITUS_PLUGIN_LOG_MAX_BYTES` / `HABITUS_PLUGIN_LOG_KEEP` | 5 MiB / 3 | 操作日志轮转 |
| `HABITUS_PLUGIN_DEBUG` | 关 | 出错时往 stderr 打堆栈（送转写的后台进程打不出，只写操作日志） |

## 命令参考

### `habitus`（npm）

| 命令 | 作用 |
|---|---|
| `habitus plugin install [--harness ID …] [--prepare-only] [--json]` | 把插件暂存到 `~/.habitus/plugin-marketplace` 并在宿主里注册。省略 `--harness`（或写 `all`）时装本机有的全部宿主；某个宿主装不了（例如有 `~/.dsh` 但 PATH 上没有 `dsh`）时跳过它并写明原因。`~/.habitus/plugin-marketplace` 已存在、非空又不是 Habitus 的 marketplace 时拒绝替换。`--prepare-only` 只暂存、不注册 |
| `habitus plugin update [--harness ID …] [--prepare-only]` | npm 包升级后重新暂存并重新注册（不带 `--harness` 时只重新注册已经注册过的宿主） |
| `habitus plugin remove [--harness ID …]` | 从宿主里注销；暂存目录在没有宿主还指着它时删除 |
| `habitus plugin status [--harness ID …]` | 每个宿主：已注册 / 未注册 / 本机没有 / 入口指向的插件已不在；暂存副本是否比 npm 包旧；Codex 的 hook 有几个还没批准 |
| `habitus harnesses [--json]` | 支持的宿主，以及本机有没有（命令在 PATH 上，或宿主的配置目录在） |
| `habitus doctor [--json]` | 体检：Node 版本、插件包完整、各宿主注册、宿主的 hook 特性是否开着、Codex hook 是否批准、服务健康 / 就绪 / 兼容 / 提醒特性（提醒没开时报 warn 并写出怎么开；记忆写入只是在等模型恢复时就绪一项报 warn）、outbox 有没有卡住、有没有读不出转写的会话（因超过读取额度被跳过的转写行报提示）。有失败项时退出码 1 |
| `habitus status [--json]` | 服务健康、就绪（未就绪时列出不健康的项）、版本与特性；未就绪时退出码 1 |
| `habitus recall <问题> [--limit N] [--json]` | 搜记忆 |
| `habitus memory ls [地址] [--json]` / `tree [地址] [-L N] [--json]` / `read <地址> [--json]` | 浏览记忆树（地址可省略 `habitus://` 前缀；`tree` 默认 3 层，最多 6 层） |
| `habitus reminders [--day YYYY-MM-DD] [--json]` | 某天（默认服务时区的今天）每条提醒走到哪一步：待取 / 取走了 / 说了 / 宿主没取 / 已发生（被问之前你已经做了），以及回话与结算 |
| `habitus --version` | 打印 npm 包版本 |

每个子命令都接受 `--help`。命令行与插件找服务的方式相同：`HABITUS_URL`，否则 `connection.json`，否则 `http://127.0.0.1:8787`。需要服务新特性的命令（`memory` 要 `memory_tree_v1`、`reminders` 要 `reminders_report_v1`）在服务太老时会说明。

### `habitus-server`（pip）

| 命令 | 作用 |
|---|---|
| `habitus-server [--config PATH] [serve]` | 起服务（不写子命令时就是 `serve`）。先做启动前检查（依赖、凭据、端口），不通过就一行说明原因退出；存储目录还是旧版本的布局（有 `vector/` 目录、单独的 `memory/` 根或行为树）时拒绝启动并点出目录，不迁移；同一个存储目录只能起一个实例 |
| `habitus-server init [--config PATH] [--from-config PATH] [--force] [--non-interactive] [--doctor \| --skip-doctor] [--deep] [--probe-timeout S] [--harness ID …] [--all-harnesses] [--start \| --no-start]` | 写配置与连接文件；可选跑体检、替你执行 `habitus plugin install`、起服务 |
| `habitus-server doctor [--config PATH] [--json] [--skip-port] [--deep] [--probe-timeout S]` | 本地服务体检：依赖（缺了说装哪个分组）、配置权限、凭据、`network`（调模型用的证书库、额外 CA、代理从哪来，代理去掉密码；CA 文件或证书库加载不了时 FAIL）、端口（被占时提示可能已有服务在跑）、插件的 `connection.json` 指的是不是配置的地址（不是时报提示）、`--deep` 时实际探测对话模型（证书不被信任、代理要认证时写出怎么配） |
| `habitus-server repair-source-outputs <source_id> --consumer memory\|behavior_session [--dry-run]` | 运维：清理一份会话源多出来的孤儿输出（需要先停服务） |

配置文件取 `--config`，否则 `HABITUS_CONFIG_FILE`，否则 `~/.habitus/config.yaml`。
