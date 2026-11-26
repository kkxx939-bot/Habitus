"""在仓库最外层组装完整记忆主链。"""

# TODO(PLUGIN-NO-SQLITE-001): 插件化交付下 Python 侧仍有三处 SQLite（召回冷热、Summary 使用回执、跨进程租约锁），去向待定。

from __future__ import annotations

from collections.abc import Mapping

from habitus.config import HabitusConfig
from habitus.foundation.observability import CompositeObserver, Observer
from habitus.infrastructure.observability import ManagedObservability, MetricRegistry
from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.store.sqlite import SQLiteLockStore
from habitus.model_client import ChatClient, ProviderFactory, StructuredChatClient
from habitus.pre.conversation import ConversationAdapterRegistry
from habitus.runtime.assembly_shared import AssemblyShared
from habitus.runtime.behavior import (
    behavior_domain_configs,
    build_behavior_components,
)
from habitus.runtime.components import RuntimeComponents, RuntimeInfrastructure, RuntimeModels
from habitus.runtime.conversation_assembly import (
    build_conversation,
    build_source_delivery,
    build_sources,
    runtime_conversation,
    source_settled,
)
from habitus.runtime.memory_assembly import build_memory, build_memory_search
from habitus.runtime.pipeline_assembly import assemble_downstream
from habitus.runtime.runtime import Runtime
from habitus.runtime.scene import model_tracer
from habitus.runtime.scene_config import relation_thresholds
from habitus.runtime.workflow_assembly import build_lifecycle, build_memory_workflow


def validate_domain_configs(config: HabitusConfig) -> None:
    """覆盖项的值域只由领域配置校验（配置层只查类型、不 import 领域）：把写了的覆盖项构造成领域配置走一遍。

    不论那一侧开没开都构造：写错的值以带配置键的 ``ConfigError`` 拒，不被静默收下、也不拖到运行中才炸。
    ``build_runtime`` 开头、本地服务的 doctor 与 init 读完配置都调它。
    """

    behavior_domain_configs(config.behavior)
    relation_thresholds(config)


def build_runtime(
    config: HabitusConfig,
    *,
    providers: ProviderFactory | None = None,
    conversation_adapters: ConversationAdapterRegistry | None = None,
    path_lock: PathLock | None = None,
    observer: Observer | None = None,
) -> Runtime:
    """无存储写入、无模型请求地完成一次显式依赖组装。各段按依赖先后组装：基础 → 长期记忆、会话 → 记忆任务 → 会话源 →
    行为侧 → 检索、下游 → 会话源交付 → 生命周期。"""

    validate_domain_configs(config)
    infrastructure, models, shared = _foundation(config, providers, conversation_adapters, path_lock, observer)
    memory = build_memory(config, shared.path_lock, shared.structured_chat)
    conversation = build_conversation(config, shared.path_lock, shared.structured_chat)
    workflow = build_memory_workflow(config, memory, conversation, shared)
    stores = build_sources(config, conversation.journal, workflow.enqueuer)
    # 行为组件要先于会话源的交付组装：会话 lane 是会话源的一个消费者，写的是行为侧的判断存储。
    behavior = build_behavior_components(
        config,
        structured_chat=shared.structured_chat,
        lock_store=shared.path_lock.lock_store,
        observer=shared.observer,
        span_controller=shared.managed_observability,
        source_settled=source_settled(stores),
    )
    search = build_memory_search(config, memory, conversation, behavior, shared.structured_chat, shared.observer)
    downstream = assemble_downstream(config, behavior, shared)
    delivery = build_source_delivery(config, stores, conversation.journal, behavior, downstream, shared)
    components = RuntimeComponents(
        infrastructure=infrastructure,
        models=models,
        conversation=runtime_conversation(conversation, stores, delivery),
        memory=search,
        workflow=build_lifecycle(config, memory, workflow, stores, delivery, downstream, behavior, shared),
        behavior=behavior,
        pipelines=downstream.pipelines,
        slot_worker=downstream.slot_worker,
        model_queue=downstream.model_queue,
        outlet=downstream.outlet,
    )
    return Runtime(config, components, conversation_adapters=shared.adapters)


def _foundation(
    config: HabitusConfig,
    providers: ProviderFactory | None,
    conversation_adapters: ConversationAdapterRegistry | None,
    path_lock: PathLock | None,
    observer: Observer | None,
) -> tuple[RuntimeInfrastructure, RuntimeModels, AssemblyShared]:
    """观测、锁与模型客户端：后面各段都共用这几样。"""

    metrics_config = config.observability.metrics
    observability = MetricRegistry(
        enabled=metrics_config.enabled,
        namespace=metrics_config.namespace,
        max_recent_events=metrics_config.max_recent_events,
        duration_buckets=metrics_config.duration_buckets_seconds,
    )
    managed_observability = ManagedObservability(
        config.observability,
        workflow_root=config.workflow_root,
        tracing_headers=config.credentials.resolve(config.observability.tracing.credential_ref),
    )
    observers: list[Observer] = [observability, managed_observability, model_tracer(observability)]
    if observer is not None:
        observers.append(observer)
    operation_observer: Observer = CompositeObserver(*observers)
    resolved_lock = path_lock or PathLock(
        SQLiteLockStore(config.workflow_root / "locks.sqlite3", config=config.storage.sqlite_lock, initialize=False),
        observer=operation_observer,
    )
    resolved_providers = providers or _builtin_provider_factory()
    chat = _chat_client(resolved_providers, config, operation_observer)
    structured_chat = StructuredChatClient(
        chat,
        allow_json_repair=config.models.structured_output.allow_json_repair,
        validation_retries=config.models.structured_output.validation_retries,
    )
    shared = AssemblyShared(
        path_lock=resolved_lock,
        observer=operation_observer,
        managed_observability=managed_observability,
        structured_chat=structured_chat,
        adapters=(
            conversation_adapters if conversation_adapters is not None else ConversationAdapterRegistry.with_builtins()
        ),
    )
    infrastructure = RuntimeInfrastructure(
        path_lock=resolved_lock,
        observability=observability,
        observer=operation_observer,
        managed_observability=managed_observability,
    )
    return (
        infrastructure,
        RuntimeModels(providers=resolved_providers, chat=chat, structured_chat=structured_chat),
        shared,
    )


def _builtin_provider_factory() -> ProviderFactory:
    """延迟导入可选 HTTP 依赖，并只注册当前内置协议适配器。"""

    from habitus.model_client.adapters import register_builtin_adapters

    providers = ProviderFactory()
    register_builtin_adapters(providers)
    return providers


def _chat_client(providers: ProviderFactory, config: HabitusConfig, observer: Observer) -> ChatClient:
    """对话能力的客户端，带自己的熔断，用 ``models.call_failures`` 这组数（裁定 108）。"""

    models = config.models
    return providers.create_chat_client(
        models.chat,
        credentials=_model_credentials(config, models.chat.route.credential_ref),
        observer=observer,
        outbound=config.network,
        failures=models.call_failures,
    )


def _model_credentials(config: HabitusConfig, reference: str) -> Mapping[str, str]:
    return config.credentials.resolve(reference)


__all__ = ["build_runtime"]
