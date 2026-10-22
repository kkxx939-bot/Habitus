import { mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { HARNESS_REGISTRY } from "../harnesses.mjs";

// 共享库复制进每个宿主插件的 scripts/shared（宿主只加载插件自己目录里的文件）。目标取自宿主注册表，
// 源里删掉或改名的文件，副本里也删掉，免得旧文件跟着包发出去。
const PLUGINS = join(dirname(fileURLToPath(import.meta.url)), "..");
const SOURCE = join(PLUGINS, "memory-plugin-shared", "lib");
export const HEADER = "// GENERATED FROM plugins/memory-plugin-shared/lib. DO NOT EDIT.\n";

export function sharedTargets(registry = HARNESS_REGISTRY) {
  return [...new Set(registry.list().map((definition) => definition.pluginDirectory))]
    .map((directory) => join(PLUGINS, directory, "scripts", "shared"));
}

export async function sharedSources() {
  return (await readdir(SOURCE)).filter((name) => name.endsWith(".mjs")).sort();
}

async function sync() {
  const sources = await sharedSources();
  for (const target of sharedTargets()) {
    await mkdir(target, { recursive: true });
    for (const stale of (await readdir(target)).filter((name) => !sources.includes(name))) {
      await rm(join(target, stale), { force: true });
    }
    for (const filename of sources) {
      await writeFile(join(target, filename), HEADER + await readFile(join(SOURCE, filename), "utf8"), "utf8");
    }
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) await sync();
