// 插件与本地服务的兼容口径只有这一份：hook（plugin-core）、doctor 与命令行（bin/service.mjs）都从这里取。

export const API_VERSION = "1.0";
export const REQUIRED_FEATURES = Object.freeze(["conversation_cursor", "flush", "recall", "remember", "remember_idempotency_v1"]);
// 提醒出口不是必需的特性：服务没有它就只做 memory，老服务照常工作（方案 32 第四节）。
export const REMINDERS_FEATURE = "reminders_v1";
// 分块上传不是必需的特性：本地服务没有它时，单独一轮就超过请求体上限的那一批标 blocked、由 doctor 报出
export const REMEMBER_PARTS_FEATURE = "remember_parts_v1";
// 服务的请求体上限：capabilities 里的 max_request_bytes；老服务不给这个字段时按服务的默认上限算
export const DEFAULT_MAX_REQUEST_BYTES = 16 * 1024 * 1024;

export function maxRequestBytes(capabilities) {
  const value = capabilities?.max_request_bytes;
  return Number.isSafeInteger(value) && value > 0 ? value : DEFAULT_MAX_REQUEST_BYTES;
}

function featureSet(capabilities) {
  return new Set(Array.isArray(capabilities?.features) ? capabilities.features : []);
}

export function missingFeatures(capabilities, features) {
  const available = featureSet(capabilities);
  return features.filter((feature) => !available.has(feature));
}

export function hasFeature(capabilities, feature) {
  return featureSet(capabilities).has(feature);
}

// 服务对这些会话协议缺什么：API 版本对不对、缺哪些协议、缺哪些必需特性
export function serviceGaps(capabilities, protocols) {
  const available = new Set(Array.isArray(capabilities?.protocols) ? capabilities.protocols : []);
  return {
    apiVersion: capabilities?.api_version === API_VERSION,
    missingProtocols: protocols.filter((protocol) => !available.has(protocol)),
    missingFeatures: missingFeatures(capabilities, REQUIRED_FEATURES),
  };
}

export function requireCompatibleService(capabilities, protocol) {
  const gaps = serviceGaps(capabilities, [protocol]);
  if (!gaps.apiVersion) throw new Error(`unsupported Habitus API version: ${capabilities?.api_version}`);
  if (gaps.missingProtocols.length > 0) throw new Error(`Habitus service does not support protocol: ${protocol}`);
  if (gaps.missingFeatures.length > 0) throw new Error(`Habitus service is missing features: ${gaps.missingFeatures.join(", ")}`);
}
