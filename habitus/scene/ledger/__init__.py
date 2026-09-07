"""待用前提清单：由情景树在读取时派生，不落盘。

一件事在建立时留下的 ``pending_effects``（预约了理发、买回了食材）进入清单；此后某件事以
``needs`` 边指回**建立它的那条行为**（或那件事）即为兑现；超过 ``expiry_days`` 未兑现的项过期。
清单是情景树的纯派生物：任何一天重建，读出来的清单自动跟着变——不存在一份会漂的副本。
过期期限第一期是运维参数（用户裁定初值 90 天），最终由统一的生命周期算法按 needs 边的
滞后分布管理。
"""

from habitus.scene.ledger.pending import PendingItem, pending_before

__all__ = ["PendingItem", "pending_before"]
