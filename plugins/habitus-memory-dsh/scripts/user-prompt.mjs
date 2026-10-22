import { runHook } from "./shared/hook-runner.mjs";
import { hostAdapter } from "./host-adapter.mjs";
await runHook("user-prompt", hostAdapter);
