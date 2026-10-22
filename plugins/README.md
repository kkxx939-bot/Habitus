# habitus

Habitus 的命令行：把 Habitus 的记忆与提醒插件装进编程 agent，检查、查询本机的 Habitus 服务。

```sh
npm install -g habitus
```

需要 Node.js ≥ 18（ZCode 要 ≥ 22.5，DeepSeek Harness 的压缩会话日志要 ≥ 22.15）。

## 先装 Habitus 服务

插件把对话送给本机的 Habitus 服务、从它取记忆和提醒。服务是 Python 写的（Python ≥ 3.13），现在从仓库装：

```sh
pip install "git+https://github.com/kkxx939-bot/Habitus"
habitus-server init        # 写配置、跑 Doctor
habitus-server serve       # 起服务，默认 http://127.0.0.1:8787
```

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

## 其他命令

```sh
habitus status                 # 服务健康、版本与特性
habitus recall <问题>           # 搜记忆
habitus memory tree            # 记忆树一览；memory ls / memory read <memory:// 地址>
habitus reminders [--day D]    # 某天每条提醒走到哪一步
```

服务地址取 `HABITUS_URL`，或 `habitus-server init` 写的 `~/.habitus/agent-plugin/connection.json`，默认 `http://127.0.0.1:8787`。

## 许可证

Apache-2.0。DeepSeek Harness（MIT）的插件只按它公开的 hook 桥与会话日志格式对接，没有带它的代码。
