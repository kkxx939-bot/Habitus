"""会话源的生命周期账：按地址的标记、还没终态的源、地址水位三样合在一个门面里。

交付（有序检查、已释放的短路）、调度（put 之前打标记）、恢复（只看没终态的）、释放都只认这一个对象，
不各自拼路径。三样各是一个只做一件事的小存储。
"""

from __future__ import annotations

from habitus.conversation.source.address import SourceAddress
from habitus.conversation.source.index import SourceIndex, SourceMarker
from habitus.conversation.source.model import ConversationSourceEnvelope
from habitus.conversation.source.pending import PendingSources
from habitus.conversation.source.watermark import AddressWatermarks


class ConversationSourceLedger:
    def __init__(self, index: SourceIndex, pending: PendingSources, watermarks: AddressWatermarks) -> None:
        if not (index.root == pending.root == watermarks.root):
            raise ValueError("source ledger parts must share one conversation root")
        self.index = index
        self.pending = pending
        self.watermarks = watermarks

    def admit(self, envelope: ConversationSourceEnvelope) -> SourceMarker:
        """在信封落盘之前打两样标记；重复调用幂等（put 撞车返回已有信封时也会补建）。"""

        marker = self.index.mark(envelope)
        self.pending.mark(envelope)
        return marker

    def earlier(self, envelope: ConversationSourceEnvelope) -> tuple[SourceMarker, ...]:
        """同一地址里排在它前面的源。"""

        current = SourceMarker.of(envelope).order
        return tuple(marker for marker in self.index.markers(SourceAddress.of(envelope)) if marker.order < current)

    def released(self, address: SourceAddress, end_sequence: int) -> bool:
        return self.watermarks.covers(address, end_sequence)


__all__ = ["ConversationSourceLedger"]
