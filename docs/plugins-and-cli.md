# Plugins and command line

English | [中文](plugins-and-cli.zh-CN.md) · [← README](../README.md)

## How the plugins work

### What happens on each host event

For command-hook hosts (Codex, Claude Code, ZCode, DeepSeek Harness) every hook script calls one shared dispatcher. Slow network work is handed to a detached background process so the hook returns quickly:

| Event | Hook (foreground) | Background process |
|---|---|---|
| SessionStart | Check service compatibility, create session state | Resend anything stuck for this session and other sessions of the same host; on ZCode and DeepSeek Harness, which have no SessionEnd, also retire stale session state and transcript mirrors of sessions the host no longer has |
| UserPromptSubmit | Recall memories and fetch reminders, return the injection block to the host | — |
| Stop | Read this turn's transcript delta into the local outbox | Send the outbox to the service, in order |
| SessionEnd | Same as Stop | After sending, ask the service to seal the conversation, then retire stale session state |
| PreCompact | Read the delta, send it and seal, all in the foreground (nothing is left behind before compaction) | — |

OpenCode and pi are in-process plugins; they call the same logic from their own events (OpenCode: `chat.message`, `session.idle`, `experimental.session.compacting`, `session.deleted`; pi: `before_agent_start`, `agent_settled`, `session_before_compact`, `session_shutdown`).

Before a delta goes into the outbox, image and other media payloads (base64 and `data:` URLs in messages and tool results) are replaced by a small placeholder that keeps only the type, the media type and the byte count and sha256 of the payload removed; text is never touched. For a tool result that holds only media, the local service records the original payload's size and sha256 from the placeholder, value for value the same as when it receives the payload itself. A batch larger than the service's request limit (`max_request_bytes` from `/api/v1/capabilities`, 16 MiB when an older service does not say) is sent as several requests cut at turn boundaries, never inside a turn. When a single turn is still over the limit and the local service has the `remember_parts_v1` feature, the batch's request body is cut losslessly into parts (each under the limit) and uploaded in order; the local service collects them, checks the digest, puts the whole turn back together and processes it exactly as if it had arrived in one request, so a turn is never split in meaning. If any part fails the batch stays in the outbox and is resent from the first part next time (also when a local service restart lost the staged parts), and the delivery id keeps it from being recorded twice. The reassembled turn is bound, like a single request, by the conversation source's per-record limit (`conversation.source.max_envelope_bytes`, 64 MiB by default) and is marked blocked if it exceeds it. Against a local service without `remember_parts_v1` such a turn is not sent: it stays in the outbox as blocked and `habitus doctor` names the session, the batch size and how far it is over.

### Injecting recall and reminders

When you say something, the plugin first checks that the service is compatible (API version `1.0`, supports this conversation protocol, and has the five features `conversation_cursor`, `flush`, `recall`, `remember`, `remember_idempotency_v1`) and ready for recall (`recall_ready` in the `/ready` result: it stays true when only the memory write side is stopped, for example a failed job or a job waiting for the model, so recall goes on), then recalls. If the service has the `reminders_v1` feature (advertised only while the prediction layer is on) it also fetches due reminders (up to 3 at a time, configurable; when the plugin asks for more than the local service's `foresight.outlet_claim_limit`, the service hands out up to its own limit). A subagent's "prompt" is a task written by the main agent and its result already comes back into the main session's transcript, so subagent sessions get no recall and no reminders and are never sent to memory (Codex subagent threads are recognised from the `session_meta` at the top of their rollout, Claude Code ones from `agent_id`, OpenCode and DeepSeek Harness ones from their parent session). Non-interactive sessions are skipped the same way: Claude Code `claude -p` and Agent SDK sessions (`CLAUDE_CODE_ENTRYPOINT` starting with `sdk-` in the hook's environment) and Codex `codex exec` sessions (`session_meta.source` is `exec` at the top of the rollout) get no recall and no reminders, send no transcript and leave no session state.

This path has a total budget (20 s by default, `HABITUS_PLUGIN_PROMPT_BUDGET_MS`) that leaves recall enough time while keeping the whole hook within the 30 s the host allows. If the service does not respond, nothing is injected for that turn and your prompt goes to the agent as usual. A failure while fetching reminders only drops the reminders; recalled memories are still injected.

The injection block looks like this:

```
<habitus-memory-context receipt="<32 hex digits>">
{"format":"habitus_memory_context_v1","content":"…"}
</habitus-memory-context>
```

The content is a JSON string with `<` and `>` escaped, so it cannot close the block early. The plugin records each block's nonce and digest in the session state (the latest 32; while the transcript keeps failing to read, every block injected meanwhile is kept until it is read again). When it later sends the transcript it strips these blocks out, and drops messages that contained nothing but an injection block, so injected content is never sent back as something you said.

Reminders use the same wrapper (tag `habitus-reminder`, again `{"format":"habitus_memory_context_v1","content":"…"}`) whose `content` is an instruction to the agent; below is the text of the `content` field:

```
Habitus 的提醒，不是用户说的话。在做任何别的事之前，先用 <host question tool> 问用户下面这件事：
题目原样用这句话、给下面列出的全部选项、允许他自己填别的话。问完照常做他让你做的事；他怎么答你都不用另外处理，系统自己会记。
1. 『<reminder text>』
   选项：现在就做 / 这次不用 / 知道了 / 已经做过了
```

(The instruction is in Chinese: "This is a Habitus reminder, not something the user said. Before doing anything else, ask the user the following with <tool>; use this sentence verbatim as the question, offer all the options listed (do it now / not this time / noted / already done), and allow a free-form answer. Then carry on with what they asked; you don't need to handle the answer, the system records it.") Whether the agent asked, and what you answered, is recognized by the service from the transcript that comes back.

### Sending transcripts: cutting, the outbox, and resending

The plugin only cuts a batch where the host marks the end of a turn: for Codex, the turn's final answer (`phase = final_answer`) with no pending tool call; for OpenCode / ZCode, a completed reply whose `finish` is not `tool-calls`; for pi, a reply whose `stopReason` is neither `toolUse` nor `pending` and that the host did not withdraw for a retry; for DeepSeek Harness, the `turn/end` event. A turn still in progress stays in the host's transcript and is sent once it finishes.

Each batch is first written to a local outbox (one directory per session, files ordered by transcript position) and then sent in order. Every batch carries the starting sequence number the service handed out and a deterministic delivery id, so resending a batch never records it twice. If the service answers with a non-retryable error the batch is marked blocked and later batches do not overtake it (`habitus doctor` reports it); retryable errors put the batch back for the next attempt.

### When something fails

The plugin never blocks the host. On any error it writes an entry to its operation log and returns an empty success to the host. When recall or claiming reminders comes back empty because the service is unreachable, incompatible, not ready, out of the prompt time budget or answers with an error, that is logged too, with the reason; while the same reason keeps recurring in a session it is logged once, and again after an hour. With `HABITUS_PLUGIN_DEBUG=1` the hooks and the OpenCode / pi plugins also print the stack to stderr; the background process that sends transcripts has no stderr, so its errors are only in the operation log. If the plugin's own configuration cannot be read (for example `HABITUS_URL` is not a loopback address), it writes one line explaining why to stderr.

### Plugin environment variables

| Variable | Default | Purpose |
|---|---|---|
| `HABITUS_MEMORY_ENABLED` | `true` | Turn the plugin off entirely |
| `HABITUS_URL` | from `connection.json`, else `http://127.0.0.1:8787` | Service address; loopback only, and redirects are not followed |
| `HABITUS_PLUGIN_STATE_DIR` | `agent-plugin/` next to the config file when `HABITUS_CONFIG_FILE` is set, else `~/.habitus/agent-plugin` | Plugin state directory. `habitus-server init` with a custom config path writes `agent-plugin/connection.json` next to that config and also rewrites the default `~/.habitus/agent-plugin/connection.json` (or the one under `HABITUS_PLUGIN_STATE_DIR`), so hosts that do not pass `HABITUS_CONFIG_FILE` find that service too; `habitus-server` rewrites the same files when it starts |
| `HABITUS_PLUGIN_TIMEOUT_MS` | `15000` | Timeout for background requests such as sending transcripts, and for `habitus doctor` / `habitus plugin status` asking Codex which hooks are approved (if Codex has not answered by then the check reports that it could not ask; the Codex process is stopped, and killed if it is still running a second later) |
| `HABITUS_PLUGIN_PROMPT_BUDGET_MS` | `20000` | Total budget for the UserPromptSubmit path |
| `HABITUS_PLUGIN_REMINDER_LIMIT` | `3` | Reminders fetched at a time (1–20; the local service hands out at most its `foresight.outlet_claim_limit`) |
| `HABITUS_PLUGIN_REMINDER_TIMEOUT_MS` | `2000` | Timeout for fetching reminders |
| `HABITUS_PLUGIN_MAX_TRANSCRIPT_BYTES` | 64 MiB | How much of a transcript is read in one go, from the cursor onward, counted after media payloads are replaced by placeholders (the rest is read right after). A single transcript line over this before stripping is skipped wherever it sits in the file and whatever record it is: its content does not reach memory, the rest of its turn is captured as usual, the skip goes into the operation log, and `habitus doctor` warns with the session and line. A turn whose text alone is over this cannot be read: that session stops being captured until the limit is raised, and `habitus doctor` names it |
| `HABITUS_PLUGIN_SESSION_GRACE_HOURS` | `24` | Grace period before stale session state is retired |
| `HABITUS_PLUGIN_LOG_MAX_BYTES` / `HABITUS_PLUGIN_LOG_KEEP` | 5 MiB / 3 | Operation log rotation |
| `HABITUS_PLUGIN_DEBUG` | off | Print stacks to stderr on error (not from the background sender, which only writes the operation log) |

## Command reference

The help and messages of both command-line tools (`habitus`, `habitus-server`) are currently in Chinese.

### `habitus` (npm)

| Command | What it does |
|---|---|
| `habitus plugin install [--harness ID …] [--prepare-only] [--json]` | Stage the plugins under `~/.habitus/plugin-marketplace` and register them with the host. Without `--harness` (or with `all`) it installs into every host present; a host that cannot be installed (for example `~/.dsh` exists but `dsh` is not on PATH) is skipped with the reason. Refuses to replace `~/.habitus/plugin-marketplace` if it is not empty and is not a Habitus marketplace. `--prepare-only` stages without registering |
| `habitus plugin update [--harness ID …] [--prepare-only]` | Restage and re-register after upgrading the npm package (without `--harness`: only the hosts already registered) |
| `habitus plugin remove [--harness ID …]` | Unregister from the host; the staging directory is deleted once no host points at it |
| `habitus plugin status [--harness ID …]` | Per host: registered / not registered / not on this machine / entry points at a missing plugin; whether the staged copy is older than the npm package; how many Codex hooks are still unapproved |
| `habitus harnesses [--json]` | Supported hosts and whether each is present (command on PATH, or the host's configuration directory exists) |
| `habitus doctor [--json]` | Checks: Node version, plugin bundle integrity, host registration, host hook features enabled, Codex hook approval, service health / readiness / compatibility / reminder feature (a warning that says how to turn reminders on when they are off; readiness only warns while memory writes are waiting for the model), stuck outbox, sessions whose transcript could not be read (and, as a warning, transcript lines skipped for being over the read limit). Exit code 1 on any failure |
| `habitus status [--json]` | Service health, readiness (listing unhealthy checks when not ready), version and features; exit code 1 when not ready |
| `habitus recall <question> [--limit N] [--json]` | Search memory |
| `habitus memory ls [uri] [--json]` / `tree [uri] [-L N] [--json]` / `read <uri> [--json]` | Browse the memory tree (the `habitus://` prefix may be omitted; `tree` shows 3 levels by default, up to 6) |
| `habitus reminders [--day YYYY-MM-DD] [--json]` | For a day (default: today in the service's time zone), how far each reminder got — awaiting pickup / picked up / asked / host never picked up / already happened (you did it before being asked) — plus replies and settlement |
| `habitus --version` | Print the npm package version |

Every subcommand accepts `--help`. The CLI finds the service the same way the plugins do: `HABITUS_URL`, else `connection.json`, else `http://127.0.0.1:8787`. Commands that need newer service features (`memory` needs `memory_tree_v1`, `reminders` needs `reminders_report_v1`) say so when the service is too old.

### `habitus-server` (pip)

| Command | What it does |
|---|---|
| `habitus-server [--config PATH] [serve]` | Run the service (`serve` is the default). A startup check runs first (dependencies, credentials, port); if it fails the service exits with a one-line reason. A storage directory still laid out by an older version (a `vector/` directory, a separate `memory/` root or behavior tree) is refused with the directory named; there is no migration. Only one instance per storage directory |
| `habitus-server init [--config PATH] [--from-config PATH] [--force] [--non-interactive] [--doctor \| --skip-doctor] [--deep] [--probe-timeout S] [--harness ID …] [--all-harnesses] [--start \| --no-start]` | Write the configuration and connection file; optionally run the health check, run `habitus plugin install` for you, and start the service |
| `habitus-server doctor [--config PATH] [--json] [--skip-port] [--deep] [--probe-timeout S]` | Local service checks: dependencies (naming the group to install), config file permissions, credentials, `network` (the trust store, extra CA and which proxy model calls use, with any password removed; FAIL when the CA file or the trust store cannot be loaded), port (suggesting a running service when it is taken), whether the plugins' `connection.json` points at the configured address (a warning when not), and with `--deep` a live probe of the chat model (an untrusted certificate or a proxy asking for authentication is reported with what to configure) |
| `habitus-server repair-source-outputs <source_id> --consumer memory\|behavior_session [--dry-run]` | Operations: clean up surplus orphan outputs of one conversation source (the service must be stopped) |

The configuration file is `--config`, else `HABITUS_CONFIG_FILE`, else `~/.habitus/config.yaml`.
