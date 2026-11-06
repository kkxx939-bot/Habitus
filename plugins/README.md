# habitus

中文 | [English](#english)

Habitus 的命令行：把 Habitus 的记忆与提醒插件装进编程 agent，检查、查询本机的 Habitus 服务。

完整文档（原理、宿主支持、HTTP API、数据与隐私）见仓库 README：[中文](https://github.com/kkxx939-bot/Habitus/blob/main/README.zh-CN.md) | [English](https://github.com/kkxx939-bot/Habitus/blob/main/README.md)。

```sh
npm install -g habitus
```

需要 Node.js ≥ 18（ZCode 要 ≥ 22.5，DeepSeek Harness 的压缩会话日志要 ≥ 22.15）。

## 先装 Habitus 服务

插件把对话送给本机的 Habitus 服务、从它取记忆和提醒。服务是 Python 写的（Python ≥ 3.12），现在从仓库装：

```sh
pip install "habitus[server] @ git+https://github.com/kkxx939-bot/Habitus"   # server 含 http、model；向量库默认是本地文件，不用账号
habitus-server init        # 写配置（默认要 DeepSeek、Embedding 服务两份凭据；Embedding 的地址与模型自己填），最后问要不要体检
habitus-server serve       # 起服务（前台运行），默认 http://127.0.0.1:8787；下面的命令另开一个终端
```

`init` 新建配置时检测系统时区，交互时请你确认或改填（检测不到默认 `Asia/Singapore`），地区码从时区推（新加坡 `SG`、中国大陆 `CN`，别的时区请你填）；`--non-interactive` 时输出里有一行说明写了什么。记忆的 Event 日期也按这个时区算。

`init` 写出的配置只开记忆：插件会召回记忆，但没有提醒（`habitus doctor` 报 warn "reminders are off"）。要提醒，在配置里补上 `behavior.primary_subject`、`scene.enabled: true`、`foresight.enabled: true`，重启服务；见仓库 README 的[快速开始](https://github.com/kkxx939-bot/Habitus/blob/main/README.zh-CN.md#快速开始)与[配置](https://github.com/kkxx939-bot/Habitus/blob/main/README.zh-CN.md#配置)。

## 装插件

```sh
habitus harnesses                         # 支持哪些宿主、本机装了哪些
habitus plugin install --harness codex    # 可重复 --harness；省略就是本机装了的全部
habitus doctor                            # 服务、宿主注册、hook 开关
```

| `--harness` | 宿主 |
|---|---|
| `codex` | OpenAI Codex |
| `claude-code` | Claude Code |
| `zcode` | ZCode |
| `opencode` | OpenCode |
| `pi` | pi |
| `dsh` | DeepSeek Harness |

插件升级（`npm update -g habitus`）之后跑 `habitus plugin update`；卸载用 `habitus plugin remove`。

子代理会话与非交互会话（Claude Code 的 `claude -p` / Agent SDK、Codex 的 `codex exec`）不召回、不取提醒、也不送进记忆。各宿主的已知局限与插件的工作方式见仓库 README 的[功能与宿主支持](https://github.com/kkxx939-bot/Habitus/blob/main/README.zh-CN.md#功能与宿主支持)与[插件怎么工作](https://github.com/kkxx939-bot/Habitus/blob/main/README.zh-CN.md#插件怎么工作)。

## 其他命令

```sh
habitus status                 # 服务健康、版本与特性
habitus recall <问题>           # 搜记忆
habitus memory tree            # 记忆树一览；memory ls / memory read <memory:// 地址>；都接受 --json
habitus reminders [--day D]    # 某天每条提醒走到哪一步
habitus --version              # npm 包版本
```

完整的命令参考见仓库 README 的[命令参考](https://github.com/kkxx939-bot/Habitus/blob/main/README.zh-CN.md#命令参考)。

服务地址取 `HABITUS_URL`，或 `habitus-server init` 写的 `~/.habitus/agent-plugin/connection.json`，默认 `http://127.0.0.1:8787`。

## 许可证

Apache-2.0。DeepSeek Harness（MIT）的插件只按它公开的 hook 桥与会话日志格式对接，没有带它的代码。

---

## English

The Habitus command line: installs Habitus's memory and reminder plugins into coding agents, and checks and queries the local Habitus service. Its help and messages are currently in Chinese.

Full documentation (how it works, supported hosts, HTTP API, data and privacy) is in the repository README: [English](https://github.com/kkxx939-bot/Habitus/blob/main/README.md) | [中文](https://github.com/kkxx939-bot/Habitus/blob/main/README.zh-CN.md).

```sh
npm install -g habitus
```

Requires Node.js ≥ 18 (ZCode needs ≥ 22.5, DeepSeek Harness's compressed session logs need ≥ 22.15).

### Install the Habitus service first

The plugins send conversations to the local Habitus service and fetch memories and reminders from it. The service is written in Python (≥ 3.12) and is currently installed from the repository:

```sh
pip install "habitus[server] @ git+https://github.com/kkxx939-bot/Habitus"   # server = http + model; the vector store defaults to local files, no account needed
habitus-server init        # writes the configuration (the default needs DeepSeek and embedding-service credentials; you fill in the embedding URL and model), then offers a health check
habitus-server serve       # starts the service (in the foreground), default http://127.0.0.1:8787; run the commands below in another terminal
```

When `init` creates a configuration it detects the system time zone and asks you to confirm or change it (falling back to `Asia/Singapore` when nothing is detected); the region code follows from the zone (`SG` for Singapore, `CN` for mainland China, otherwise you are asked). With `--non-interactive`, one output line says what was written. Memory dates its Events by this time zone too.

The configuration written by `init` enables memory only: the plugins recall memories but there are no reminders (`habitus doctor` warns "reminders are off"). To get reminders, add `behavior.primary_subject`, `scene.enabled: true` and `foresight.enabled: true` to the configuration and restart the service; see [Quick start](https://github.com/kkxx939-bot/Habitus/blob/main/README.md#quick-start) and [Configuration](https://github.com/kkxx939-bot/Habitus/blob/main/README.md#configuration) in the repository README.

### Install the plugins

```sh
habitus harnesses                         # which hosts are supported and which are present
habitus plugin install --harness codex    # repeat --harness; omit it for every host present
habitus doctor                            # service, host registration, hook switches
```

`--harness` values: `codex` (OpenAI Codex), `claude-code` (Claude Code), `zcode` (ZCode), `opencode` (OpenCode), `pi` (pi), `dsh` (DeepSeek Harness).

After upgrading the package (`npm update -g habitus`) run `habitus plugin update`; uninstall with `habitus plugin remove`.

Sub-agent sessions and non-interactive sessions (Claude Code `claude -p` / Agent SDK, Codex `codex exec`) get no recall and no reminders and are not sent to memory. Per-host limitations and how the plugins work are under [Features and supported hosts](https://github.com/kkxx939-bot/Habitus/blob/main/README.md#features-and-supported-hosts) and [How the plugins work](https://github.com/kkxx939-bot/Habitus/blob/main/README.md#how-the-plugins-work) in the repository README.

### Other commands

```sh
habitus status                 # service health, version and features
habitus recall <question>      # search memory
habitus memory tree            # overview of the memory tree; memory ls / memory read <memory:// address>; all accept --json
habitus reminders [--day D]    # how far each reminder of a day got
habitus --version              # npm package version
```

The full command reference is under [Command reference](https://github.com/kkxx939-bot/Habitus/blob/main/README.md#command-reference) in the repository README.

The service address comes from `HABITUS_URL`, or `~/.habitus/agent-plugin/connection.json` written by `habitus-server init`, defaulting to `http://127.0.0.1:8787`.

### License

Apache-2.0. The DeepSeek Harness (MIT) plugin only talks to its public hook bridge and session log format and contains none of its code.
