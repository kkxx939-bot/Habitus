"""Job 文件里的时间固定写满六位微秒并以 Z 结尾（S1 把格式化并到 ``foundation.timestamps.format_utc_z``）。"""

import json
import re
from datetime import date
from pathlib import Path

from habitus.memory.conversation import ConversationAddress
from tests.unit.workflow.test_job_store import Clock, source, store

_FIXED_MICROSECONDS_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")


def test_job_file_times_keep_fixed_width_microseconds(tmp_path: Path) -> None:
    clock = Clock()
    clock.now = clock.now.replace(microsecond=0)
    jobs = store(tmp_path, clock)
    jobs.stage(ConversationAddress("conversation-a", date(2026, 7, 1)), source("conversation-a", 0))
    job_path = next(path for path in jobs.jobs_root.glob("*.json") if path.name != "state.json")
    payload = json.loads(job_path.read_text(encoding="utf-8"))

    assert _FIXED_MICROSECONDS_Z.fullmatch(payload["created_at"]), payload["created_at"]
    assert _FIXED_MICROSECONDS_Z.fullmatch(payload["updated_at"]), payload["updated_at"]
