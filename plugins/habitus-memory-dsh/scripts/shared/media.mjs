// GENERATED FROM plugins/memory-plugin-shared/lib. DO NOT EDIT.
// 转写里的媒体载荷（截图、读图、附件的 base64 / data: URL）在送给服务之前换成小占位：服务端各协议适配器只从带 text 的段取文字，
// 媒体本来就不存原始载荷（AGENTS.md：媒体只存描述、大小）；留着它们会把一批撑过服务的请求上限。占位只留类型、媒体类型（有的话）、
// 去掉的载荷字节数 omitted_bytes 与 sha256 omitted_sha256，不带 text 字段，文字一个字不动。字节数与 sha256 都按载荷文本本身
// （base64 文本或整条 data: URL）的 UTF-8 字节算，不解码 base64：两侧不必各写一个宽严不一的解码器，本地服务不经插件时按同一口径
// 算出逐值相同的 original_size_bytes / original_sha256（habitus/pre/conversation/media.py describe_media_only）。
// 认的形状（各宿主真实转写里见到的）：
// - Codex：`{"type":"input_image","image_url":"data:…"}`（工具结果、消息）；MCP 工具结果 `{"type":"image","data":"<base64>","mimeType":…}`
// - Claude Code：`{"type":"image","source":{"type":"base64","media_type":…,"data":…}}`（消息、tool_result；document 同形）；
//   记录上的 `toolUseResult`：读图 `{"type":"image","file":{"base64":…,"type":…}}`，读 PDF `{"type":"pdf","file":{"base64":…}}`
// - pi：`{"type":"image","data":"<base64>","mimeType":…}`
// - OpenCode / ZCode：file 段与工具附件 `{"mime":…,"url":"data:…"}`
// 字符串里的内容（比如工具把图片 JSON 当文字输出）是文字，不碰。

import { createHash } from "node:crypto";

const MEDIA_TYPES = new Set(["image", "input_image", "image_url", "document", "file", "input_file", "audio", "input_audio"]);
// 带 file.base64 的那种（Claude Code 记录上的 toolUseResult）另认 pdf：只在这个形状上认，别的形状的 pdf 不当媒体
const FILE_RESULT_TYPES = new Set([...MEDIA_TYPES, "pdf"]);

function isDataUrl(value) {
  return typeof value === "string" && value.startsWith("data:");
}

function dataUrlMediaType(value) {
  const match = /^data:([^;,]+)[;,]/.exec(value);
  return match ? match[1] : undefined;
}

// 这个对象若是带载荷的媒体，给出 {payload, mediaType}；不是给 null
function mediaPayload(value) {
  if (value.source?.type === "base64" && typeof value.source.data === "string") {
    return { payload: value.source.data, mediaType: value.source.media_type };
  }
  if (isDataUrl(value.image_url)) return { payload: value.image_url, mediaType: dataUrlMediaType(value.image_url) };
  if (isDataUrl(value.image_url?.url)) return { payload: value.image_url.url, mediaType: dataUrlMediaType(value.image_url.url) };
  if (isDataUrl(value.url) && (MEDIA_TYPES.has(value.type) || typeof value.mime === "string")) {
    return { payload: value.url, mediaType: value.mime ?? dataUrlMediaType(value.url) };
  }
  if (MEDIA_TYPES.has(value.type) && typeof value.data === "string") {
    return { payload: value.data, mediaType: value.mimeType ?? value.media_type ?? value.mime };
  }
  if (FILE_RESULT_TYPES.has(value.type) && typeof value.file?.base64 === "string") {
    return { payload: value.file.base64, mediaType: value.file.type };
  }
  return null;
}

function placeholder(value, { payload, mediaType }) {
  const result = { type: typeof value.type === "string" ? value.type : "file" };
  if (typeof mediaType === "string" && mediaType) result.media_type = mediaType;
  result.omitted_bytes = Buffer.byteLength(payload, "utf8");
  result.omitted_sha256 = createHash("sha256").update(payload, "utf8").digest("hex");
  return result;
}

// 返回去掉媒体载荷后的值（没有媒体的部分原样共享，不复制）
export function stripMedia(value) {
  if (Array.isArray(value)) {
    let changed = false;
    const items = value.map((item) => {
      const stripped = stripMedia(item);
      if (stripped !== item) changed = true;
      return stripped;
    });
    return changed ? items : value;
  }
  if (value === null || typeof value !== "object") return value;
  const media = mediaPayload(value);
  if (media) return placeholder(value, media);
  let changed = false;
  const result = {};
  for (const [key, item] of Object.entries(value)) {
    const stripped = stripMedia(item);
    if (stripped !== item) changed = true;
    result[key] = stripped;
  }
  return changed ? result : value;
}
