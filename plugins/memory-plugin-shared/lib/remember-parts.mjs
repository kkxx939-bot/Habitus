// 单独一轮的文字本身就超过本地服务请求体上限时的分块上传（remember_parts_v1，裁定 72 第 2 条）。
// 分的是这一批 remember 请求体序列化后的字节，不是轮：本地服务收齐、核对摘要后拼回原请求体，
// 再按 /memory/remember 原样处理（一轮在语义上不拆，裁定 69）。每块带同一个投递号、块序号、总块数、整批的字节数与 SHA-256，
// 字节段用标准 base64 装进 JSON（base64 字符在 JSON 里不需要转义，块的大小可以事先算准）。
// 任何一块失败都整批从第 0 块重送：本地服务见到第 0 块就作废这个投递号旧的暂存，重送幂等。
import { createHash } from "node:crypto";

const MAX_PART_COUNT = 100_000;

export function rememberPartBody(part) {
  return JSON.stringify({
    delivery_id: part.delivery_id,
    part_index: part.part_index,
    part_count: part.part_count,
    body_bytes: part.body_bytes,
    body_sha256: part.body_sha256,
    data: part.data,
  });
}

// 除去 data 的部分按最长的写法量（块序号、总块数、字节数都取最大），切出来的每块都不会超过上限
function envelopeBytes(deliveryId) {
  return Buffer.byteLength(rememberPartBody({
    delivery_id: deliveryId,
    part_index: MAX_PART_COUNT - 1,
    part_count: MAX_PART_COUNT,
    body_bytes: Number.MAX_SAFE_INTEGER,
    body_sha256: "0".repeat(64),
    data: "",
  }));
}

// 把一批 remember 请求体无损切成几块，每块的请求体都不超过 limit
export function splitRememberBody(deliveryId, body, limit) {
  const bytes = Buffer.from(body, "utf8");
  const chunkBytes = Math.floor((limit - envelopeBytes(deliveryId)) / 4) * 3;
  if (chunkBytes <= 0) throw new Error(`the local service request limit of ${limit} bytes leaves no room for a remember part`);
  const count = Math.ceil(bytes.length / chunkBytes);
  if (count < 2 || count > MAX_PART_COUNT) throw new Error(`a ${bytes.length}-byte batch cannot be uploaded in ${count} part(s)`);
  const digest = createHash("sha256").update(bytes).digest("hex");
  return Array.from({ length: count }, (_, index) => ({
    delivery_id: deliveryId,
    part_index: index,
    part_count: count,
    body_bytes: bytes.length,
    body_sha256: digest,
    data: bytes.subarray(index * chunkBytes, (index + 1) * chunkBytes).toString("base64"),
  }));
}

// 按顺序送完各块；送成功的结果与一次 remember 的响应同形（result 是拼回那一批的 RememberResult），失败原样交回
export async function sendRememberParts(service, parts) {
  for (const part of parts) {
    const response = await service.rememberPart(part);
    if (!response.ok) return response;
    if (part.part_index < part.part_count - 1) continue;
    const remembered = response.result?.remember;
    if (remembered == null || typeof remembered !== "object") {
      return { ok: false, status: response.status, retryable: true, error: "the local service did not assemble the remember parts" };
    }
    return { ok: true, status: response.status, result: remembered };
  }
  return { ok: false, status: 0, retryable: true, error: "no remember parts to send" };
}
