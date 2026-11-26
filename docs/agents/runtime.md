适用：改 `habitus/runtime/**` 时读。换代的步骤、预测拍的水位与「未观测」见 `behavior-scene-foresight.md`。

# runtime

## 组合根

- `habitus/runtime/` 是唯一组合根：消费已校验的配置，构造并共享 `ModelClient`、Conversation、Agent SearchService、Memory Editor、语义刷新、MemoryJob Runner；组装行为侧两支（会话、物理）、语义树换代与预测层；管理 MemoryWorker、LifecycleWorker、行为归约、模型队列 Worker 与预测 Worker（换代在它里面）的初始化、启动、停止和关闭。
- runtime 里另外只住三类东西：
  - 跨包的桥：会话源 ↔ 会话 lane、未封口判断、开着的提醒与提醒出口、预测拍写会话一支「未观测」的口、闲置封存与它对会话源的释放。
  - 预测层的落盘与编排：`foresight` 包不碰 infrastructure、不读 `series.reader`；账本、单元历史、机会表、出口状态的读写与"取材料 → 调纯规则 → 落账"在 `runtime/foresight_*`。
  - 语义树各步的编排（`runtime/scene_*`），换代的触发、先后、完成标记、预测拍水位、三把锁和观测（`runtime/generation*`、`runtime/scene_lock.py`），模型队列（`runtime/model_queue*`）。
- 判法、规则、算法一律放在各自领域包的纯函数里（预测层的在 `habitus/foresight`），runtime 只取材料、调用、落盘。`model_client/`、`pre/`、`memory/`、`infrastructure/`、`foundation/` 不得反向导入 `runtime`（架构测试 `test_domain_packages_never_import_top_level_runtime` 守着）。

## 构造、启动与关闭

- 不用进程全局单例、Service Locator、延迟 setter 注入、`atexit`，构造函数里不做存储或网络副作用。
- 启动 Worker 前必须完成本地锁与 habitus 树初始化：卡片目录总是建；两支事件与空白目录只在行为侧开着时建。存储根下还有旧的 `vector/` 目录、合并前的记忆根或行为树时，在这一步明确报错。
- 初始化遇到旧盘面（`workflow/transactions/`、`workflow/receipts/`、旧格式任务文件）明确报错，不迁移（裁定 111）。
- 关闭时必须关闭模型客户端。
- 启动顺序：行为归约 → 模型队列 Worker → 预测 Worker；停止反过来。LifecycleWorker 在 MemoryWorker 之后启动、在它之前停止。

## MemoryWorker

- 同一 memory-root 上固定单并发。认领必须用耐久的 `claim_id + claim_generation + worker_id + lease_expires_at` fencing；续约、完成和失败结算都必须匹配同一租约；过期租约只能通过递增 generation 接管；陈旧 Worker 不得继续发布或改变 Job。
- 重试时间以 `next_attempt_at` 耐久保存、有界退避；自动重试期间挡住后续序号，用完判 FAILED 终态，后续序号照常推进（裁定 111）。

## LifecycleWorker

- 首次自动维护等一个完整配置周期（闲置封存等一拍）。
- 每轮以 Conversation 根派生的全局耐久 lease 保证同一根只有一个维护者；续约失败必须取消本轮。
- 按有界批量轮转全部 Conversation，隔离单会话失败。轮转位置只以日期和 `conversation_id` 写入 `workflow/lifecycle/state.json`，不存任何会话正文；Runtime 重启或维护者切换后不得总从最早目录重新开始。
- 停止时先停新的生命周期周期和新 Job 认领，在各自心跳继续运行的情况下有界等待当前操作；超时才取消，由下一次安全恢复或维护继续。

## 换代与模型队列的进程结构（裁定 96、97、98、102）

- 取代了原来的夜线 Job Worker 与它的 Job 账。换代与模型队列复用领域无关的锁、原子文件与退避原语，不复用 MemoryJob 本体。
- 语义树与预测层按 lane 各装一份：`<scene_root>/<lane>/`、`<foresight_root>/<lane>/`；锁、租约、完成标记、模型队列状态各支一份。预测 Worker 与模型队列 Worker 各一个，每一槽依次做两支（会话在前），一支的失败只算这一支的。提醒的出口共用一个，账按支记。
- 换代没有自己的 Worker，也没有换代租约。预测 Worker 每个槽边界持预测根派生的 SQLite 租约（进程内另加一把锁），先看要不要换代、要就先做这一代，再做这一槽的预测拍。"同一根同一天只换一次"由这把租约加完成标记保证。
- 租约在独立线程续：一次续不上不取消，到期前仍续不上才取消这一段。进程起来第一次看换代时，先撤回上一次中途被杀剩下的重试项。
- 调模型的活（出错重试就近试、物理一支未封口判断的选词、先验混合问法）都在模型队列 Worker 里一项一项做。"还欠什么"每一轮从盘上现算，不记账；先验在今天这一代完成之后才算欠着。
- 模型调用失败按模型层统一的三条规则处理（裁定 108）：重试只在模型客户端一层；"等一等可能好"有界重试后交回，这一项留到下一轮再试；"要人处理"直接打开熔断、降到试探节奏；"这一条输入有问题"同一份输入不重发。队列不另做"连着几项失败就退避"，统一读对话熔断（`models.call_failures`），队列、换代、记忆任务与健康面都读它。

## 锁的先后

- 拿锁先后固定为：模型队列租约 → 预测根租约 → sweep 锁 → 语义树写锁。
- 语义树写锁由语义树根派生：换代的语义树一步整步持有；模型队列每次落盘时持有、调模型时放开；换代在做时，队列在两项之间停下。

## 观测

- 每代一个 span、每步一个子 span，模型队列每项一个 span；写指标与 INFO 日志。span 属性只放计数、固定枚举与受控原因码。
- 每代写报告文件与本地 trace 文件，没写成的也写。报告含关系表双向变化、换代时各类还欠几项、结算那一节的 loss；按行为的明细只留本地，不进 span。
- 健康面读：换代的最近结果、卡住的日子、日期回退、下一次重试时刻与最新完成标记；模型队列的循环存活、积压与跨过换代的欠账；三种模型能力的熔断（开着时写出给人看的原因）；长期记忆跳过的段；系统时区与配置时区的偏移。
