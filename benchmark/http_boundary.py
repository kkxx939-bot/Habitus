"""从外部 HTTP 调用方视角测量 m2bOS 远程服务边界。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from itertools import count
from pathlib import Path
from urllib.parse import urlparse

import httpx

from benchmark.boundary import (
    BoundaryPolicy,
    BoundaryProfile,
    environment_metadata,
    write_boundary_outputs,
)
from benchmark.isolation import require_empty_directory
from benchmark.metrics import evidence_recall, latency_distribution
from benchmark.model import BenchmarkDataset, BenchmarkMessage, BenchmarkSession


class HTTPBoundaryBenchmark:
    """在真实监听地址上执行预热、稳态并发、写入排空和长尾统计。"""

    def __init__(
        self,
        dataset: BenchmarkDataset,
        *,
        profile: BoundaryProfile,
        policy: BoundaryPolicy,
        server_url: str,
        server_identity: str,
        api_key: str,
        output_directory: str | Path,
        top_k: int = 10,
        allow_writes: bool = False,
        seed_dataset: bool = False,
        request_timeout_seconds: float = 120.0,
        drain_timeout_seconds: float = 1_800.0,
    ) -> None:
        if not isinstance(dataset, BenchmarkDataset):
            raise TypeError("HTTP boundary benchmark requires a dataset")
        if not isinstance(profile, BoundaryProfile) or not isinstance(policy, BoundaryPolicy):
            raise TypeError("HTTP boundary benchmark requires a profile and policy")
        parsed = urlparse(server_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("server_url must be an absolute HTTP(S) URL without credentials")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be non-empty text")
        if (
            not isinstance(server_identity, str)
            or not server_identity.strip()
            or server_identity != server_identity.strip()
        ):
            raise ValueError("server_identity must be normalized non-empty text")
        if seed_dataset and not allow_writes:
            raise ValueError("seed_dataset requires allow_writes")
        if any(value > 0 for value in profile.write_fractions) and not allow_writes:
            raise ValueError("profile contains writes; pass allow_writes or override write_fractions to zero")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        for name, value in (
            ("request_timeout_seconds", request_timeout_seconds),
            ("drain_timeout_seconds", drain_timeout_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float) or not 1 <= float(value) <= 86_400:
                raise ValueError(f"{name} must be between one and 86400")
        self.dataset = dataset
        self.profile = profile
        self.policy = policy
        self.server_url = server_url.rstrip("/")
        self.server_identity = server_identity
        self.api_key = api_key
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.top_k = top_k
        self.allow_writes = allow_writes
        self.seed_dataset = seed_dataset
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.drain_timeout_seconds = float(drain_timeout_seconds)

    async def run(self) -> Mapping[str, object]:
        require_empty_directory(self.output_directory, label="HTTP boundary output directory")
        raw_directory = self.output_directory / "raw"
        raw_directory.mkdir()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(
            base_url=self.server_url,
            headers=headers,
            timeout=self.request_timeout_seconds,
        ) as client:
            await _require_ready(client)
            if self.seed_dataset:
                await _seed(client, self.dataset, timeout_seconds=self.drain_timeout_seconds)
            points: list[Mapping[str, object]] = []
            for write_fraction in self.profile.write_fractions:
                for concurrency in self.profile.concurrency_levels:
                    for repetition in range(1, self.profile.repetitions + 1):
                        await _warmup(
                            client,
                            self.dataset,
                            concurrency=concurrency,
                            duration_seconds=self.profile.warmup_seconds,
                            top_k=self.top_k,
                        )
                        events, queue_samples, elapsed = await _run_phase(
                            client,
                            self.dataset,
                            concurrency=concurrency,
                            write_fraction=write_fraction,
                            duration_seconds=self.profile.phase_seconds,
                            top_k=self.top_k,
                            sample_interval_seconds=self.profile.sample_interval_seconds,
                        )
                        drain_started = time.perf_counter()
                        drained, failed_jobs = await _wait_jobs(
                            client,
                            events,
                            timeout_seconds=self.drain_timeout_seconds,
                        )
                        drain_seconds = time.perf_counter() - drain_started
                        point = _point(
                            events,
                            queue_samples,
                            phase_seconds=elapsed,
                            concurrency=concurrency,
                            write_fraction=write_fraction,
                            repetition=repetition,
                            drain_seconds=drain_seconds,
                            drained=drained,
                            failed_jobs=failed_jobs,
                        )
                        points.append(point)
                        point_id = f"w{round(write_fraction * 100):03d}-c{concurrency:03d}-r{repetition:02d}"
                        (raw_directory / f"{point_id}.json").write_text(
                            json.dumps(point, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                            encoding="utf-8",
                        )
        metadata = environment_metadata(
            Path(__file__).resolve().parents[1],
            config_digest=hashlib.sha256(f"{self.server_url}\0{self.server_identity}".encode()).hexdigest(),
            dataset_digest=_dataset_digest(self.dataset),
        )
        metadata.update(
            {
                "dataset": self.dataset.name.value,
                "server_origin": self.server_url,
                "server_identity": self.server_identity,
                "top_k": self.top_k,
                "seed_dataset": self.seed_dataset,
                "writes_are_persistent": self.allow_writes,
            }
        )
        return write_boundary_outputs(
            self.output_directory,
            lane="http",
            profile=self.profile,
            points=points,
            policy=self.policy,
            metadata=metadata,
        )


async def _require_ready(client: httpx.AsyncClient) -> None:
    response = await client.get("/ready")
    if response.status_code != 200:
        raise RuntimeError(f"m2bOS HTTP readiness returned {response.status_code}")


async def _seed(client: httpx.AsyncClient, dataset: BenchmarkDataset, *, timeout_seconds: float) -> None:
    for sample in dataset.samples:
        for session_index, session in enumerate(sample.sessions):
            conversation_id = f"boundary-seed-{sample.sample_id}-{session_index:06d}"
            payload = _remember_payload(
                conversation_id,
                session,
                operation_id=session_index,
                wait_timeout_seconds=timeout_seconds,
            )
            response = await client.post("/api/v1/memory/remember", json=payload)
            if response.status_code != 200:
                raise RuntimeError(f"HTTP benchmark seed failed with status {response.status_code}")


async def _warmup(
    client: httpx.AsyncClient,
    dataset: BenchmarkDataset,
    *,
    concurrency: int,
    duration_seconds: float,
    top_k: int,
) -> None:
    if duration_seconds <= 0:
        return
    questions = tuple(question for sample in dataset.samples for question in sample.questions)
    deadline = time.monotonic() + duration_seconds

    async def worker(offset: int) -> None:
        index = offset
        while time.monotonic() < deadline:
            response = await client.post(
                "/api/v1/memory/recall",
                json={"query": questions[index % len(questions)].question, "limit": top_k},
            )
            if response.status_code != 200:
                raise RuntimeError(f"HTTP benchmark warmup failed with status {response.status_code}")
            index += concurrency

    await asyncio.gather(*(worker(index) for index in range(concurrency)))


async def _run_phase(
    client: httpx.AsyncClient,
    dataset: BenchmarkDataset,
    *,
    concurrency: int,
    write_fraction: float,
    duration_seconds: float,
    top_k: int,
    sample_interval_seconds: float,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...], float]:
    questions = tuple(question for sample in dataset.samples for question in sample.questions)
    sessions = tuple(session for sample in dataset.samples for session in sample.sessions)
    events: list[dict[str, object]] = []
    queue_samples: list[dict[str, object]] = []
    operation_ids = count()
    stop = asyncio.Event()
    ready = asyncio.Event()

    async def execute_recall(worker_id: int, operation_id: int) -> None:
        question = questions[operation_id % len(questions)]
        started = time.perf_counter()
        try:
            response = await client.post(
                "/api/v1/memory/recall",
                json={"query": question.question, "limit": top_k},
            )
            result = _success_result(response)
            context = result.get("context")
            quality = evidence_recall(context, question.evidence_texts) if isinstance(context, str) else None
            degradations = _mapping_array(result.get("degradations"), "HTTP recall degradations")
            events.append(
                {
                    "operation": "recall",
                    "worker_id": worker_id,
                    "latency_ms": (time.perf_counter() - started) * 1_000,
                    "success": True,
                    "status_code": response.status_code,
                    "recall_at_k": quality,
                    "degradation_count": len(degradations),
                    "jobs": [],
                    "error_type": "",
                }
            )
        except Exception as exc:
            events.append(
                {
                    "operation": "recall",
                    "worker_id": worker_id,
                    "latency_ms": (time.perf_counter() - started) * 1_000,
                    "success": False,
                    "status_code": getattr(getattr(exc, "response", None), "status_code", None),
                    "recall_at_k": None,
                    "degradation_count": 0,
                    "jobs": [],
                    "error_type": type(exc).__name__,
                }
            )

    async def execute_remember(worker_id: int, operation_id: int) -> None:
        session = sessions[operation_id % len(sessions)]
        conversation_id = f"http-boundary-{worker_id:03d}-{operation_id:012d}"
        payload = _remember_payload(conversation_id, session, operation_id=operation_id)
        started = time.perf_counter()
        try:
            response = await client.post("/api/v1/memory/remember", json=payload)
            result = _success_result(response)
            jobs = _job_identities(result.get("jobs"), conversation_id, session.started_at.date().isoformat())
            if not jobs:
                raise ValueError("HTTP remember operation produced no MemoryJob")
            events.append(
                {
                    "operation": "remember",
                    "worker_id": worker_id,
                    "latency_ms": (time.perf_counter() - started) * 1_000,
                    "success": True,
                    "status_code": response.status_code,
                    "recall_at_k": None,
                    "degradation_count": 0,
                    "jobs": jobs,
                    "error_type": "",
                }
            )
        except Exception as exc:
            events.append(
                {
                    "operation": "remember",
                    "worker_id": worker_id,
                    "latency_ms": (time.perf_counter() - started) * 1_000,
                    "success": False,
                    "status_code": getattr(getattr(exc, "response", None), "status_code", None),
                    "recall_at_k": None,
                    "degradation_count": 0,
                    "jobs": [],
                    "error_type": type(exc).__name__,
                }
            )

    async def operation_worker(worker_id: int) -> None:
        await ready.wait()
        while not stop.is_set():
            operation_id = next(operation_ids)
            if _is_write_operation(operation_id, write_fraction):
                await execute_remember(worker_id, operation_id)
            else:
                await execute_recall(worker_id, operation_id)

    async def sample_queue() -> None:
        await ready.wait()
        while not stop.is_set():
            try:
                snapshot = await _queue_metrics(client)
                queue_samples.append({"success": True, **snapshot})
            except Exception as exc:
                queue_samples.append({"success": False, "error_type": type(exc).__name__})
            try:
                await asyncio.wait_for(stop.wait(), timeout=sample_interval_seconds)
            except TimeoutError:
                pass

    tasks = [asyncio.create_task(operation_worker(index)) for index in range(concurrency)]
    started = time.perf_counter()
    sampler = asyncio.create_task(sample_queue())
    ready.set()
    await asyncio.sleep(duration_seconds)
    stop.set()
    await asyncio.gather(*tasks)
    await sampler
    return tuple(events), tuple(queue_samples), time.perf_counter() - started


async def _queue_metrics(client: httpx.AsyncClient) -> dict[str, float]:
    response = await client.get("/metrics")
    response.raise_for_status()
    values = {
        name: _prometheus_gauge(response.text, suffix)
        for name, suffix in (
            ("staged", "_memory_jobs_staged"),
            ("queued", "_memory_jobs_queued"),
            ("running", "_memory_jobs_running"),
            ("failed", "_memory_jobs_failed"),
            ("oldest_age_seconds", "_memory_job_oldest_age_seconds"),
        )
    }
    values["depth"] = values["staged"] + values["queued"] + values["running"]
    return values


def _prometheus_gauge(text: str, suffix: str) -> float:
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric, separator, raw_value = line.rpartition(" ")
        if separator and metric.split("{", 1)[0].endswith(suffix):
            return _number(float(raw_value), f"Prometheus gauge {suffix}")
    raise ValueError(f"Prometheus gauge is missing: {suffix}")


async def _wait_jobs(
    client: httpx.AsyncClient,
    events: Sequence[Mapping[str, object]],
    *,
    timeout_seconds: float,
) -> tuple[bool, int]:
    pending = {
        (
            _positive_integer(job.get("memory_sequence"), "memory sequence"),
            _non_empty_text(job.get("conversation_id"), "conversation id"),
            _non_empty_text(job.get("started_on"), "started on"),
        )
        for event in events
        for job in _mapping_array(event.get("jobs"), "HTTP job identities")
    }
    if not pending:
        return True, 0
    deadline = time.monotonic() + timeout_seconds
    failed = 0
    while pending and time.monotonic() < deadline:
        completed: set[tuple[int, str, str]] = set()
        for sequence, conversation_id, started_on in tuple(pending):
            response = await client.get(
                f"/api/v1/memory/jobs/{sequence}",
                params={"conversation_id": conversation_id, "started_on": started_on},
            )
            result = _success_result(response)
            status = result.get("job_status")
            if status == "committed":
                completed.add((sequence, conversation_id, started_on))
            elif status == "failed":
                completed.add((sequence, conversation_id, started_on))
                failed += 1
        pending -= completed
        if pending:
            await asyncio.sleep(0.1)
    return not pending and failed == 0, failed


def _point(
    events: Sequence[Mapping[str, object]],
    queue_samples: Sequence[Mapping[str, object]],
    *,
    phase_seconds: float,
    concurrency: int,
    write_fraction: float,
    repetition: int,
    drain_seconds: float,
    drained: bool,
    failed_jobs: int,
) -> Mapping[str, object]:
    latencies = [_number(item.get("latency_ms"), "HTTP operation latency") for item in events]
    distribution = latency_distribution(latencies)
    undrained_failure = int(not drained and failed_jobs == 0)
    failures = sum(item.get("success") is not True for item in events) + failed_jobs + undrained_failure
    denominator = len(events) + failed_jobs + undrained_failure
    recalls = [
        value
        for item in events
        if item.get("success") is True and (value := _optional_number(item.get("recall_at_k"))) is not None
    ]
    write_count = sum(item.get("operation") == "remember" for item in events)
    read_count = len(events) - write_count
    degraded_read_count = sum(
        item.get("operation") == "recall" and _non_negative_integer(item.get("degradation_count", 0)) > 0
        for item in events
    )
    successful_queue_samples = [item for item in queue_samples if item.get("success") is True]
    queue_depths = [_number(item.get("depth"), "HTTP queue depth") for item in successful_queue_samples]
    queue_ages = [_number(item.get("oldest_age_seconds"), "HTTP queue age") for item in successful_queue_samples]
    return {
        "schema_version": "m2bos_boundary_point_v1",
        "lane": "http",
        "scenario": "recall" if write_fraction == 0 else "mixed",
        "scale_name": "server_existing_state",
        "scale_value": 1,
        "concurrency": concurrency,
        "write_fraction": write_fraction,
        "repetition": repetition,
        "drained": drained,
        "failed_jobs": failed_jobs,
        "metrics": {
            "operation_count": len(events),
            "read_operation_count": read_count,
            "write_operation_count": write_count,
            "achieved_write_fraction": write_count / max(len(events), 1),
            "degraded_read_rate": degraded_read_count / max(read_count, 1),
            "throughput_per_second": len(events) / max(phase_seconds, 1e-9),
            "error_rate": failures / max(denominator, 1),
            "p50_ms": distribution["p50"],
            "p95_ms": distribution["p95"],
            "p99_ms": distribution["p99"],
            "recall_at_k": sum(recalls) / len(recalls) if recalls else None,
            "queue_depth": max(queue_depths) if queue_depths else None,
            "queue_oldest_age_seconds": max(queue_ages) if queue_ages else None,
            "metrics_observation_error_count": sum(item.get("success") is not True for item in queue_samples),
            "drain_seconds": drain_seconds,
        },
    }


def _remember_payload(
    conversation_id: str,
    session: BenchmarkSession,
    *,
    operation_id: int,
    wait_timeout_seconds: float | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "conversation_id": conversation_id,
        "started_on": session.started_at.date().isoformat(),
        "protocol": "openai_chat_completions",
        "payload": {"messages": _openai_messages(session.messages, operation_id=operation_id)},
        "start_sequence": 0,
        "occurred_at": session.started_at.isoformat(),
        "after_turn": True,
    }
    if wait_timeout_seconds is not None:
        payload["wait_timeout_seconds"] = wait_timeout_seconds
    return payload


def _openai_messages(messages: Sequence[BenchmarkMessage], *, operation_id: int) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    call_ids: dict[str, str] = {}
    for index, message in enumerate(messages):
        if message.role == "prompt":
            result.append({"role": "user", "content": message.content})
        elif message.role == "completion":
            result.append({"role": "assistant", "content": message.content})
        elif message.role == "tool_call":
            assert message.tool_call_id is not None
            generated_call_id = f"{message.tool_call_id}-{operation_id}-{index}"
            call_ids[message.tool_call_id] = generated_call_id
            result.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": generated_call_id,
                            "type": "function",
                            "function": {
                                "name": message.tool_name,
                                "arguments": json.dumps(message.content, ensure_ascii=False),
                            },
                        }
                    ],
                }
            )
        else:
            assert message.tool_call_id is not None
            tool_result_call_id = call_ids.get(message.tool_call_id)
            if tool_result_call_id is None:
                raise ValueError("tool result has no preceding tool call in benchmark session")
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_result_call_id,
                    "content": json.dumps(message.content, ensure_ascii=False),
                    "status": message.tool_status,
                }
            )
    return result


def _success_result(response: httpx.Response) -> Mapping[str, object]:
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, Mapping) or not isinstance(value.get("result"), Mapping):
        raise ValueError("m2bOS HTTP success response has an invalid envelope")
    return value["result"]


def _job_identities(value: object, conversation_id: str, started_on: str) -> list[dict[str, object]]:
    jobs = _mapping_array(value, "remember jobs")
    identities: list[dict[str, object]] = []
    for item in jobs:
        returned_conversation = _non_empty_text(item.get("conversation_id"), "conversation id")
        returned_started_on = _non_empty_text(item.get("started_on"), "started on")
        if returned_conversation != conversation_id or returned_started_on != started_on:
            raise ValueError("HTTP remember returned a MemoryJob for another conversation")
        identities.append(
            {
                "memory_sequence": _positive_integer(item.get("memory_sequence"), "memory sequence"),
                "conversation_id": returned_conversation,
                "started_on": returned_started_on,
            }
        )
    return identities


def _mapping_array(value: object, label: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{label} must be an object array")
    return list(value)


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _non_negative_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError("degradation count must be a non-negative integer")
    return value


def _non_empty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _optional_number(value: object) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _is_write_operation(operation_id: int, write_fraction: float) -> bool:
    if write_fraction <= 0:
        return False
    if write_fraction >= 1:
        return True
    return int((operation_id + 1) * write_fraction) > int(operation_id * write_fraction)


def _dataset_digest(dataset: BenchmarkDataset) -> str:
    payload = "\0".join((dataset.name.value, dataset.source_path, *(sample.digest for sample in dataset.samples)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["HTTPBoundaryBenchmark"]
