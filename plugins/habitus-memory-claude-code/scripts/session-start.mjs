import { runHook } from "./shared/hook-runner.mjs";
import { hostAdapter } from "./host-adapter.mjs";
await runHook("session-start", hostAdapter);
