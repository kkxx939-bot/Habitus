# HTTP API

[English](http-api.md) | 中文 · [← README](../README.zh-CN.md)

服务只接受本机请求：只能绑定回环地址；`Host` 头必须是回环地址；带 `Origin` 头时它也必须是本机的 http/https 地址（`null`、跨站、带用户信息的一律拒绝）。请求体默认上限 16 MiB（`http.max_request_bytes`，1 KiB–64 MiB）。分块上传的各块暂存在本地服务的内存里：总量上限 `http.remember_parts_max_bytes`（默认 64 MiB，不小于请求体上限），一次上传超过 `http.remember_parts_ttl_seconds`（默认 600 秒）没有新块的，在下一次有任何上传的块到来时清掉（过期的不占总量；一直没有块到来时它留在内存里）；本地服务重启后暂存为空，插件从第一块重送。

成功响应 `{"status":"ok","result":…,"request_id":…}`；错误响应 `{"status":"error","error":{"code","message","retryable","retry_after_seconds"?,"details"?},"request_id":…}`；每个响应都带 `X-Request-ID` 头。

| 方法与路径 | 作用 |
|---|---|
| `GET /api/v1/capabilities` | API 版本、服务版本、支持的会话协议、特性、请求体上限（`max_request_bytes`） |
| `POST /api/v1/memory/remember` | 送一批会话转写（插件用），幂等 |
| `POST /api/v1/memory/remember/parts` | 单独一轮超过请求体上限时分块上传的一块（插件用）：`delivery_id`、`part_index`、`part_count`、`body_bytes`、`body_sha256`、`data`（这一批 remember 请求体字节的一段，标准 base64）；收齐前回 `remember: null`，收齐那一块回拼回的一批按 `/memory/remember` 处理的结果；接不上已暂存的部分回可重试的 `ABORTED`（从第 0 块重送），暂存满了回可重试的 `RESOURCE_EXHAUSTED`，整批超过暂存上限回不可重试的 `REQUEST_TOO_LARGE` |
| `GET /api/v1/memory/conversations/cursor` | 这段对话的下一个序号 |
| `POST /api/v1/memory/flush` | 把这段对话已送的部分封段 |
| `POST /api/v1/memory/recall` | 召回记忆 |
| `GET /api/v1/memory/tree`、`GET /api/v1/memory/document` | 浏览记忆树、读一条记忆 |
| `GET /api/v1/reminders` | 某天的提醒进度 |
| `POST /api/v1/reminders/claim` | 宿主取到期的提醒（只给有提问工具的宿主） |
| `GET /api/v1/memory/jobs`、`/memory/jobs/blocked`、`/memory/jobs/{序号}` | 记忆写入任务的状态；`/memory/jobs` 与 `/memory/jobs/{序号}` 必带查询参数 `conversation_id`（1–240 个 UTF-8 字节，所有带 `conversation_id` 的接口同一上限，超了回 400 `INVALID_ARGUMENT`）与 `started_on`（日期），缺了回 400 `INVALID_ARGUMENT` |
| `POST /api/v1/operations/memory/jobs/{序号}/retry` | 运维：重试失败的写入任务；必带查询参数 `conversation_id`、`started_on`，请求体 `{"expected_version": <version>}`，四个值都取自 `GET /memory/jobs/blocked` 返回的 `job` |
| `GET /api/v1/operations/audit` | 运维：最近的审计事件 |
| `GET /health`、`GET /ready`、`GET /metrics` | 健康、就绪（未就绪时 503，正文照样是健康读数；有记忆 Job 失败或在等模型恢复时也是 503，这时结果里的 `recall_ready` 仍为 true，插件照常召回）、Prometheus 指标 |

特性表（`features`）：`conversation_cursor`、`flush`、`memory_jobs`、`memory_tree_v1`、`recall`、`remember`、`remember_idempotency_v1`、`remember_parts_v1`、`reminders_v1`、`reminders_report_v1`。客户端按它协商：缺了必需特性的服务插件不连，缺了可选特性的只少那一项功能。

错误码与 HTTP 状态：`INVALID_ARGUMENT` 400、`NOT_FOUND` 404、`METHOD_NOT_ALLOWED` 405、`CONFLICT` / `ABORTED` 409、`FAILED_PRECONDITION` 412、`REQUEST_TOO_LARGE` 413、`RESOURCE_EXHAUSTED` 429、`UNAVAILABLE` 503、`DEADLINE_EXCEEDED` 504、`INTERNAL` 500。请求校验失败统一是 400，`details.validation_errors` 列出每个字段的问题。
