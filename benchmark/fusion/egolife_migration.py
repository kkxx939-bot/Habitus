"""把 v15 时代的冻结 EgoLife 产物迁移到 WP4 之后的读取契约。

固化的七天真实数据集（``benchmark/data/egolife_week/run_root/behavior``，不进版本控制）是行为层
的标准输入集：改归约、行为树、预测树、语义关联都靠**重放已落盘的融合判断**验证，不必再调模型。
但那批产物产自 2026-08-29，WP4 之后 main 已经读不动它，两处硬失败都会让重放静默作废——

1. **回执缺 ``unowned_observation_ids``**（WP4 为"读得懂但不属于任何事"的帧新增的字段）。回执
   的读路径是严格键集，缺键直接抛 ``fusion receipt record is corrupt``，归约第一步就停。v15
   那次运行里"无归属"这个概念还不存在，所以补空集是**忠实**的而不是编数据；``record_digest``
   覆盖全部字段，必须跟着重算。
2. **没有覆盖索引**（``fusion/coverage/``，WP4 从"枚举回执目录"改过来的）。封口视界取"尚未被
   任何回执覆盖的最早观测的 available_at"，索引为空等于"全部观测都没融合过"，视界永远停在数据
   起点。实测：不重建索引跑 8 轮归约，9,270 条链**一条都封不了口**，全部 pending。

迁移**只作用于拷贝**，冻结数据集本身不动。典型用法：

    cp -R benchmark/data/egolife_week/run_root/behavior /tmp/replay
    rm -rf /tmp/replay/tree /tmp/replay/reduction
    python -m benchmark.fusion.egolife_migration /tmp/replay
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from behavior.fusion.coverage import BehaviorCoverageIndex

# ``_record`` 是私有名，这里刻意用它：迁移要从已落盘的 JSON 反推 ``record_digest``，而正常的
# 构造路径 ``build_fusion_receipt`` 需要原始片段与判断对象——重放时它们已经不在手边。摘要口径
# 只有一份，抄一份出来迟早会漂。``receipt.py`` 若改了记录形状，本文件要跟着改。
from behavior.fusion.receipt import BehaviorFusionReceipt, _record
from behavior.fusion.receipt_store import BehaviorFusionReceiptStore
from foundation.integrity import canonical_digest

# 冻结数据集的位置：它不进版本控制，毁了无法重建。
_FROZEN_DATASET = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class MigrationReport:
    """迁移了什么；数字直接印出来供重放日志留痕。"""

    receipts_total: int
    receipts_migrated: int
    coverage_records: int


def migrate(behavior_root: str | Path) -> MigrationReport:
    """就地迁移一个 behavior 根；已经是新形状的回执原样跳过（幂等）。"""

    root = Path(behavior_root).expanduser().resolve(strict=False)
    # 按**内容**认根而不是按目录名：拷贝出来的重放根叫什么都行，但它必须真的装着回执，
    # 否则一个手滑的路径会在别处建出一棵空的 fusion/coverage。
    if not (root / "fusion" / "receipts").is_dir():
        raise ValueError(f"{root} does not look like a behavior root (no fusion/receipts)")
    # 迁移是**先删后写、非原子**的（回执只创建不覆盖，只能先 unlink 再走正门落盘），中途失败
    # 会留下一个既不是旧形状也不是新形状的根。冻结数据集是行为层唯一的标准输入集，毁了就没了，
    # 所以指向它本身时直接拒绝——docstring 里那句"只作用于拷贝"要有东西兜着。
    if _FROZEN_DATASET in root.parents or root == _FROZEN_DATASET:
        raise ValueError(
            "refusing to migrate the frozen dataset in place; copy the run root first"
        )
    store = BehaviorFusionReceiptStore(root)
    paths = sorted((root / "fusion" / "receipts").rglob("*.json"))
    migrated = 0
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if "unowned_observation_ids" in raw:
            continue
        values = _values_with_unowned(raw)
        receipt = BehaviorFusionReceipt(
            **values, record_digest=canonical_digest(_record(values))
        )
        # 回执是只创建不覆盖的：先删掉旧字节，再走正门落盘（正门会回读自校验）。
        path.unlink()
        store.put(receipt)
        migrated += 1
    coverage = BehaviorCoverageIndex(root)
    recorded = 0
    for receipt in store.list():
        coverage.record(receipt)
        recorded += 1
    return MigrationReport(
        receipts_total=len(paths), receipts_migrated=migrated, coverage_records=recorded
    )


def _values_with_unowned(raw: dict[str, Any]) -> dict[str, Any]:
    """旧记录的全部字段，外加空的无归属集合。"""

    return {
        "receipt_id": raw["receipt_id"],
        "segment_digest": raw["segment_digest"],
        "observation_ids": tuple(raw["observation_ids"]),
        "source_refs": tuple(raw["source_refs"]),
        "fusion_version": raw["fusion_version"],
        "prompt_version": raw["prompt_version"],
        "validation_attempts": raw["validation_attempts"],
        "judged_at": datetime.fromisoformat(
            raw["judged_at"].replace("Z", "+00:00")
        ).astimezone(UTC),
        "judgement_ids": tuple(raw["judgement_ids"]),
        "unreadable_observation_ids": tuple(raw["unreadable_observation_ids"]),
        "out_of_scope_observation_ids": tuple(raw["out_of_scope_observation_ids"]),
        # v15 那次运行里"无归属"还不存在，空集是忠实记录。
        "unowned_observation_ids": (),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    report = migrate(argv[0])
    print(
        f"receipts total={report.receipts_total} migrated={report.receipts_migrated} "
        f"coverage_records={report.coverage_records}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - 运维脚本入口
    raise SystemExit(main(sys.argv[1:]))


__all__ = ["MigrationReport", "migrate"]
