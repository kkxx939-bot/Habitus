"""同一 Source Consumer 出现多个孤儿 Output 时的显式人工修复入口。

交付路径本身产生不了第二个 Output：执行栅栏把同一 (source, consumer) 串行化，
后到者在锁内会看到既有孤儿并直接采用，不会再写一份。因此"多个孤儿"只可能来自
外力——手工拷贝、备份还原、半完成的迁移，或两侧误用了作用域不同的锁后端。

正因为它意味着存储被外力动过，交付层保持 fail-closed、不做静默仲裁：静默挑一个
会把真实的存储问题伪装成正常运行。修复只在这里发生，且必须由人显式发起。

选择规则是确定性的，不是任意的：只保留与**当前** Processor Fingerprint 吻合的那份，
其余判为陈旧。没有任何一份吻合时拒绝修复——那说明当前代码从未在这个来源上产出过
结果，该保留哪一份不是本模块能决定的。由于被保留的那份随后会被交付层直接采用，
修复不会触发 Consumer 重新执行，也就不会重放任何已经落盘的写入副作用。
"""

from __future__ import annotations

from dataclasses import dataclass

from habitus.conversation.source.delivery import ConversationConsumerDelivery
from habitus.conversation.source.model import ConversationSourceEnvelope, ConversationSourceError
from habitus.conversation.source.receipt import ConversationSourceConsumer


class ConversationSourceRepairError(ConversationSourceError):
    """当前状态不满足显式修复的前提。"""


@dataclass(frozen=True)
class ConversationSourceRepairResult:
    """一次修复保留和删除了哪些 Output；供运维审计。"""

    source_id: str
    consumer: ConversationSourceConsumer
    retained_output_id: str
    removed_output_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "consumer", ConversationSourceConsumer(self.consumer))
        if not self.removed_output_ids:
            raise ValueError("repair result must record at least one removed output")
        if self.retained_output_id in self.removed_output_ids:
            raise ValueError("retained output cannot also be removed")


class ConversationSourceOutputRepair:
    """只处理"无 Outcome 且存在多个 Output"这一种确定性可修复的损坏。"""

    def __init__(self, delivery: ConversationConsumerDelivery) -> None:
        if not isinstance(delivery, ConversationConsumerDelivery):
            raise TypeError("delivery must be ConversationConsumerDelivery")
        self.delivery = delivery

    def repair(
        self,
        source: ConversationSourceEnvelope,
        consumer: ConversationSourceConsumer,
    ) -> ConversationSourceRepairResult:
        """按当前 Processor 身份保留唯一吻合的 Output，删除其余陈旧 Output。

        前提由本方法自己从存储重新判定，不解析交付层的错误文本——错误文本
        不是契约，按它分支会在措辞变化时静默失效。
        """

        resolved = ConversationSourceConsumer(consumer)
        implementation = self.delivery.consumers[resolved]
        outputs = implementation.output_store
        if self.delivery.outcomes.read(source.source_id, resolved) is not None:
            raise ConversationSourceRepairError(
                "repair only applies before a durable outcome exists"
            )
        existing = outputs.list(source)
        if len(existing) < 2:
            raise ConversationSourceRepairError(
                "repair requires more than one orphan output"
            )
        expected = outputs.expected_output_id(source, implementation.processor_fingerprint)
        retained = tuple(
            output for output in existing if outputs.ref(output).output_id == expected
        )
        if len(retained) != 1:
            raise ConversationSourceRepairError(
                "no orphan output matches the current processor fingerprint; "
                "resolve it manually before retrying delivery"
            )
        removed: list[str] = []
        for output in existing:
            output_id = outputs.ref(output).output_id
            if output_id == expected:
                continue
            outputs.remove(source, output_id)
            removed.append(output_id)
        return ConversationSourceRepairResult(
            source_id=source.source_id,
            consumer=resolved,
            retained_output_id=expected,
            removed_output_ids=tuple(sorted(removed)),
        )


__all__ = [
    "ConversationSourceOutputRepair",
    "ConversationSourceRepairError",
    "ConversationSourceRepairResult",
]
