适用：改 `habitus/model_client/**`，或任何调用模型、处理结构化输出的代码时读。

# 模型客户端

## 边界

- 模型能力统一由 `habitus/model_client/` 承载，只有对话（chat）一种能力（向量与重排已删，裁定 110）。
- 不得绑定厂商品牌 SDK，不得保存或发送图片、音频、视频。旧 `LLMClient/`、LiteLLM/OpenAI-compatible 实现及兼容导入必须退出，旧 `infrastructure/model` 路径不得恢复。
- Chat 保持同构的 `ChatClient -> ChatProvider` 边界；流式请求的重试语义由 Chat 运行层独立处理。

## 厂商与协议

- 公共 `ProviderConfig` 分离厂商身份 `provider` 与调用协议 `adapter`，保存模型、无凭据地址、凭据引用（`credential_ref`）和运行边界。
- `ProviderFactory` 只按 `(capability, adapter)` 解析显式注册的 builder：不枚举模型名、不按厂商猜协议、不静默回退。用已有 adapter 的新厂商只能通过配置接入；只有出现全新的请求/响应协议才新增 adapter 代码。
- `openai_compatible_chat`（Chat Completions）与 `anthropic_messages`（Anthropic Messages，用户 2026-11-04 定）是平级的协议适配器：只负责消息、工具、结构化输出、推理参数、普通响应和 SSE 事件的协议转换以及单次 HTTP 调用；不得内置重试、并发调度或按厂商名称分支。同协议的厂商、网关只换配置。其他协议（如 Responses）等后续逐项实现。
- 默认配置与 init 向导不预置任何 Chat 厂商（用户 2026-11-06 定），只写占位模板；doctor 与启动前检查遇到占位值直接报错。DeepSeek 只是向导里可选的 Chat Completions 预置，不是默认。

## 失败处理（裁定 108）

- 模型调用失败只在 `model_client/failures.py` 一处，按协议错误代码字段、HTTP 状态码与传输异常归入 `model_client/errors.py` 的 13 类；不按厂商、模型名或报错文字判断。
- 每类按三条规则之一处理："等一等可能好"、"要人处理"、"这一条输入有问题"。
- Chat 客户端运行层是唯一的重试层，带一个熔断。模型层之外只用 `model_failure` / `model_failure_rule` 或按规则分好的组分派，不点名具体异常类型。

## 结构化输出

- 结构化输出能力必须显式声明为 `none`、`json_object` 或 `json_schema`，不能用布尔值掩盖协议差异。`anthropic_messages` 只有 `json_schema`（Schema 由适配器换成协议子集，完整约束仍由运行层在客户端校验）与 `none`，协议没有 `json_object`。
- 分层容错：可以从代码块或外围文本提取 JSON，可以修复尾逗号、单引号等纯语法问题；修复只针对提取出的 JSON 候选，修复结果仍必须通过完整 JSON Schema 和领域模型校验。
- 禁止：把 list/dict/number 自动转成 string，把 string/dict 自动包成 list，把非法枚举替换成默认值，忽略未知字段，补造必填值，猜测业务内容。
- 解析、JSON Schema、领域校验失败必须区分阶段，把精确错误反馈给模型做有界重试；重试提示要求保留已经合法的值，不得用有损类型转换规避校验。超过重试次数返回明确失败，不生成结构化结果。
