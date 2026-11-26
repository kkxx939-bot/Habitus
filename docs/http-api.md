# HTTP API

English | [中文](http-api.zh-CN.md) · [← README](../README.md)

The service only accepts local requests: it can only bind to a loopback address; the `Host` header must be a loopback address; an `Origin` header, if present, must be a local http/https origin (`null`, cross-site, or with user info is rejected). The request body limit is 16 MiB by default (`http.max_request_bytes`, 1 KiB–64 MiB). Parts of a split upload are staged in the local service's memory: at most `http.remember_parts_max_bytes` in total (64 MiB by default, never below the request limit), and an upload that has received no new part for `http.remember_parts_ttl_seconds` (600 seconds by default) is dropped when the next part of any upload arrives (expired uploads never count against the total; while no part arrives they stay in memory); after a local service restart the staging is empty and the plugin resends from the first part.

Success responses are `{"status":"ok","result":…,"request_id":…}`; errors are `{"status":"error","error":{"code","message","retryable","retry_after_seconds"?,"details"?},"request_id":…}`. Every response carries an `X-Request-ID` header.

| Method and path | Purpose |
|---|---|
| `GET /api/v1/capabilities` | API version, service version, supported conversation protocols, features, request body limit (`max_request_bytes`) |
| `POST /api/v1/memory/remember` | Send a batch of transcript (used by the plugins); idempotent |
| `POST /api/v1/memory/remember/parts` | One part of a batch whose single turn is over the request limit (used by the plugins): `delivery_id`, `part_index`, `part_count`, `body_bytes`, `body_sha256`, `data` (a slice of the batch's remember request body bytes, standard base64); answers `remember: null` until the last part, and the last part answers with the result of processing the reassembled batch as `/memory/remember`; a part that does not follow the staged ones gets a retryable `ABORTED` (resend from part 0), full staging a retryable `RESOURCE_EXHAUSTED`, and a batch over the staging limit a non-retryable `REQUEST_TOO_LARGE` |
| `GET /api/v1/memory/conversations/cursor` | Next sequence number for a conversation |
| `POST /api/v1/memory/flush` | Seal what has been sent for a conversation |
| `POST /api/v1/memory/recall` | Recall memories |
| `GET /api/v1/memory/tree`, `GET /api/v1/memory/document` | Browse the memory tree, read one memory |
| `GET /api/v1/reminders` | Reminder progress for a day |
| `POST /api/v1/reminders/claim` | A host fetches due reminders (only hosts that have a question tool) |
| `GET /api/v1/memory/jobs`, `/memory/jobs/blocked`, `/memory/jobs/{sequence}` | Status of memory write jobs; `/memory/jobs` and `/memory/jobs/{sequence}` require the query parameters `conversation_id` (1–240 UTF-8 bytes, the same limit on every endpoint that takes one; longer is a 400 `INVALID_ARGUMENT`) and `started_on` (a date), otherwise 400 `INVALID_ARGUMENT` |
| `POST /api/v1/operations/memory/jobs/{sequence}/retry` | Operations: retry a failed write job; requires the query parameters `conversation_id` and `started_on` and the body `{"expected_version": <version>}`, all four values taken from the `job` returned by `GET /memory/jobs/blocked` |
| `GET /api/v1/operations/audit` | Operations: recent audit events |
| `GET /health`, `GET /ready`, `GET /metrics` | Health, readiness (503 when not ready, with the health reading in the body; also 503 while a memory job has failed or is waiting for the model, in which case `recall_ready` in the result stays true and the plugins keep recalling), Prometheus metrics |

Features: `conversation_cursor`, `flush`, `memory_jobs`, `memory_tree_v1`, `recall`, `remember`, `remember_idempotency_v1`, `remember_parts_v1`, `reminders_v1`, `reminders_report_v1`. Clients negotiate on these: the plugins refuse a service missing a required feature; a missing optional feature only disables that function.

Error codes and HTTP status: `INVALID_ARGUMENT` 400, `NOT_FOUND` 404, `METHOD_NOT_ALLOWED` 405, `CONFLICT` / `ABORTED` 409, `FAILED_PRECONDITION` 412, `REQUEST_TOO_LARGE` 413, `RESOURCE_EXHAUSTED` 429, `UNAVAILABLE` 503, `DEADLINE_EXCEEDED` 504, `INTERNAL` 500. Request validation failures are always 400, with `details.validation_errors` listing each field.
