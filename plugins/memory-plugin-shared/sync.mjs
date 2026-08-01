import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const PLUGINS = join(dirname(fileURLToPath(import.meta.url)), "..");
const SOURCE = join(PLUGINS, "memory-plugin-shared", "lib");
const TARGETS = [
  join(PLUGINS, "m2bos-memory", "scripts", "shared"),
  join(PLUGINS, "m2bos-memory-claude-code", "scripts", "shared"),
];
const HEADER = "// GENERATED FROM plugins/memory-plugin-shared/lib. DO NOT EDIT.\n";

for (const target of TARGETS) {
  await mkdir(target, { recursive: true });
  for (const filename of (await readdir(SOURCE)).filter((name) => name.endsWith(".mjs")).sort()) {
    await writeFile(join(target, filename), HEADER + await readFile(join(SOURCE, filename), "utf8"), "utf8");
  }
}
