// 命令行连本地 Habitus 服务的公共部分：取连接配置、发请求、失败时说清是连不上、服务太老，还是请求本身被拒。

import { loadPluginConfig } from "../memory-plugin-shared/lib/config.mjs";
import { HabitusServiceClient } from "../memory-plugin-shared/lib/service-client.mjs";

export function connect(env) {
  const config = loadPluginConfig(env);
  return { config, service: new HabitusServiceClient(config) };
}

// 子命令要的服务特性（方案 36 风险 2）：没有就报"服务太老"，不去撞一个不存在的接口
export async function requireFeature({ config, service }, feature) {
  const capabilities = await service.capabilities();
  if (!capabilities.ok) throw new Error(await failureMessage({ config, service }, capabilities, "capabilities"));
  if (!(capabilities.result?.features || []).includes(feature)) {
    throw new Error(`Habitus 服务太老，不支持这个命令（缺 ${feature}）；升级 habitus-server 后重试`);
  }
}

export async function call(connection, label, path, init = {}) {
  const response = await connection.service.request(path, init);
  if (!response.ok) throw new Error(await failureMessage(connection, response, label));
  return response.result;
}

async function failureMessage({ config, service }, response, label) {
  if (!(await service.health()).ok) return `连不上 Habitus 服务（${config.baseUrl}），先启动：habitus-server`;
  const message = response.error?.message;
  return `${label} failed: ${typeof message === "string" ? message : JSON.stringify(response.error)}`;
}
