import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { HARNESS_REGISTRY } from "../../harnesses.mjs";
import { HEADER, sharedSources, sharedTargets } from "../sync.mjs";

const PLUGINS = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const LIB = join(PLUGINS, "memory-plugin-shared", "lib");

test("every plugin's shared copy matches the library exactly, with no leftovers", async () => {
  const sources = await sharedSources();
  for (const target of sharedTargets()) {
    assert.deepEqual((await readdir(target)).sort(), sources, `${target} has stale or missing files; run npm run sync`);
    for (const filename of sources) {
      assert.equal(
        await readFile(join(target, filename), "utf8"),
        HEADER + await readFile(join(LIB, filename), "utf8"),
        `${target}/${filename} is stale; run npm run sync`,
      );
    }
  }
});

test("the shared library only imports node builtins and its own siblings (it is copied into each plugin)", async () => {
  for (const filename of await sharedSources()) {
    const source = await readFile(join(LIB, filename), "utf8");
    for (const [, specifier] of source.matchAll(/(?:from|import)\s*\(?\s*["']([^"']+)["']/g)) {
      assert.ok(
        specifier.startsWith("node:") || /^\.\/[^/]+\.mjs$/.test(specifier),
        `${filename} imports ${specifier}`,
      );
    }
  }
});

test("the npm package ships every registered plugin directory", async () => {
  const manifest = JSON.parse(await readFile(join(PLUGINS, "package.json"), "utf8"));
  const shipped = manifest.files.map((entry) => entry.replace(/\/$/, ""));
  for (const definition of HARNESS_REGISTRY.list()) {
    assert.ok(shipped.includes(definition.pluginDirectory), `package.json files misses ${definition.pluginDirectory}`);
  }
});
