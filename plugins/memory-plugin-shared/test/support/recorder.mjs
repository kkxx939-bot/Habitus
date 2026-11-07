// Windows 上的记录器（POSIX 用 isolated-env.mjs 里只靠 sh 内建的那份）：假宿主 <命令>.cmd 调它，
// 把（命令、HOME、cwd、参数）记进 FAKE_CODEX_LOG，格式与 POSIX 记录器相同，然后以 127 失败
import { appendFileSync } from "node:fs";

const [command, ...argv] = process.argv.slice(2);
appendFileSync(process.env.FAKE_CODEX_LOG, `${command}\t${process.env.HOME}\t${process.cwd()}\t${argv.join(" ")}\n`);
process.exitCode = 127;
