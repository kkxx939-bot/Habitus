// habitus reminders（方案 36 第 2 步）：某天进了提醒出口的每条提醒走到了哪一步——待取 / 取走了 / 说了 / 宿主没取，
// 他怎么回的、结算结果；顶上一行是 R8 要看的数（问出口的比例、宿主没取、放回次数）。只读，经 GET /api/v1/reminders。

import { call, connect, requireFeature } from "./service.mjs";

const USAGE = "usage: habitus reminders [--day YYYY-MM-DD] [--json]";

export async function reminders(args, { stdout, env }) {
  const values = [...args];
  if (values.includes("-h") || values.includes("--help")) { stdout.write(`${USAGE}\n`); return 0; }
  const json = take(values, "--json");
  const day = value(values, "--day");
  if (values.length > 0) throw new Error(`unknown argument: ${values[0]}\n${USAGE}`);
  if (day != null && !/^\d{4}-\d{2}-\d{2}$/.test(day)) throw new Error("--day must look like 2026-10-21");
  const connection = connect(env);
  await requireFeature(connection, "reminders_report_v1");
  const report = await call(connection, "reminders", `/api/v1/reminders${day ? `?day=${day}` : ""}`);
  if (json) { stdout.write(`${JSON.stringify(report, null, 2)}\n`); return 0; }
  const total = report.reminders.length;
  if (total === 0) { stdout.write(`${report.day}  没有进出口的提醒。\n`); return 0; }
  const putBack = report.reminders.reduce((sum, item) => sum + (item.attempts ?? 0), 0);
  stdout.write(
    `${report.day}  ${total} 条：说了 ${report.spoken} · 取走了 ${report.taken} · 待取 ${report.awaiting}`
    + ` · 宿主没取 ${report.never_claimed} · 回话 ${report.responded}`
    + `  问出口 ${report.spoken}/${total}${putBack > 0 ? `  放回 ${putBack} 次` : ""}\n\n`,
  );
  for (const item of report.reminders) {
    stdout.write(`${clock(item.awaiting_at)}  ${item.kind}  ${progress(item)}\n`);
    stdout.write(`       ${item.text}\n`);
  }
  return 0;
}

function progress(item) {
  const steps = [`待取（时限 ${clock(item.deadline)}）`];
  const holder = item.handoffs.find((handoff) => handoff.ended == null);
  if (item.stage === "取走了" && holder) steps.push(`${holder.host} 取走 ${clock(holder.claimed_at)}`);
  if (item.stage === "说了") steps.push(`说了（${item.spoken_via ?? "?"} ${clock(item.spoken_at)}）`);
  if (item.stage === "宿主没取") steps.push("宿主没取");
  if (item.response) steps.push(`回话 ${item.response} ${clock(item.responded_at)}`);
  if (item.outcome) steps.push(`结果 ${item.outcome}`);
  if (item.attempts) steps.push(`放回 ${item.attempts} 次`);
  return steps.join(" → ");
}

// 账本里的时刻带着写入时的时区偏移，直接取那一刻的钟点
function clock(iso) {
  const match = typeof iso === "string" ? iso.match(/T(\d{2}:\d{2})/) : null;
  return match ? match[1] : "?";
}

function take(args, flag) {
  const index = args.indexOf(flag);
  if (index < 0) return false;
  args.splice(index, 1);
  return true;
}

function value(args, flag) {
  const index = args.indexOf(flag);
  if (index < 0) return null;
  const result = args[index + 1];
  if (result == null) throw new Error(`${flag} requires a value`);
  args.splice(index, 2);
  return result;
}
