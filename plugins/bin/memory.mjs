// habitus memory ls / tree / read（方案 36 第 2 步）：按 memory:// 地址浏览 Habitus 记下的东西。只读，经本地服务的
// GET /api/v1/memory/tree 与 /api/v1/memory/document；服务不给文件路径，这里也只认地址。

import { call, connect, requireFeature } from "./service.mjs";

const USAGE = `usage: habitus memory ls [地址] [--json]
       habitus memory tree [地址] [-L 层数] [--json]
       habitus memory read <地址> [--json]
地址形如 memory://preferences、memory://entities/项目/habitus.md；省略 memory:// 也行，省略地址是根。`;
const DEFAULT_DEPTH = 3;
const MAX_DEPTH = 6;

export async function memory(args, { stdout, env }) {
  const action = args.shift();
  if (!action || action === "-h" || action === "--help") { stdout.write(`${USAGE}\n`); return 0; }
  const json = take(args, "--json");
  if (action === "ls") return ls(args, { stdout, env, json });
  if (action === "tree") return tree(args, { stdout, env, json });
  if (action === "read") return read(args, { stdout, env, json });
  throw new Error(USAGE);
}

async function ls(args, { stdout, env, json }) {
  const uri = address(single(args, { optional: true }));
  const connection = await ready(env);
  const listing = await list(connection, uri);
  if (json) return write(stdout, listing);
  if (listing.abstract) stdout.write(`${listing.abstract.trim()}\n\n`);
  for (const item of listing.directories) stdout.write(`${name(item)}/\n`);
  for (const item of listing.documents) stdout.write(`${name(item)}\n`);
  if (listing.directories.length + listing.documents.length === 0) stdout.write("（空）\n");
  return 0;
}

async function tree(args, { stdout, env, json }) {
  const depthValue = value(args, "-L");
  const depth = depthValue == null ? DEFAULT_DEPTH : Number(depthValue);
  if (!Number.isSafeInteger(depth) || depth < 1 || depth > MAX_DEPTH) throw new Error(`-L must be an integer between 1 and ${MAX_DEPTH}`);
  const uri = address(single(args, { optional: true }));
  const connection = await ready(env);
  const root = await walk(connection, uri, depth);
  if (json) return write(stdout, root);
  stdout.write(`${root.uri}\n`);
  render(stdout, root, "");
  return 0;
}

async function read(args, { stdout, env, json }) {
  const target = single(args, { optional: false });
  const connection = await ready(env);
  const result = await call(connection, "memory read", `/api/v1/memory/document?uri=${encodeURIComponent(address(target))}`);
  if (json) return write(stdout, result);
  if (result.node === "layer") { stdout.write(`${result.text.trim()}\n`); return 0; }
  stdout.write(`${result.uri}\n${result.kind} · 第 ${result.revision} 版 · 更新于 ${result.updated_at}\n\n`);
  // 正文就是由字段渲染出来的，有正文只印正文；结构化字段看 --json
  if (result.text?.trim()) stdout.write(`${result.text.trim()}\n`);
  else {
    for (const [key, item] of Object.entries(result.fields || {})) {
      stdout.write(`${key}: ${typeof item === "string" ? item : JSON.stringify(item)}\n`);
    }
  }
  const links = [...(result.links || []), ...(result.backlinks || [])];
  if (links.length > 0) {
    stdout.write("\n关联：\n");
    for (const link of result.links || []) stdout.write(`  → ${link.to_uri}（${link.link_type}）\n`);
    for (const link of result.backlinks || []) stdout.write(`  ← ${link.from_uri}（${link.link_type}）\n`);
  }
  return 0;
}

async function ready(env) {
  const connection = connect(env);
  await requireFeature(connection, "memory_tree_v1");
  return connection;
}

function list(connection, uri) {
  const query = uri === "memory://" ? "" : `?uri=${encodeURIComponent(uri)}`;
  return call(connection, "memory ls", `/api/v1/memory/tree${query}`);
}

// 一层一层往下列；文档是叶子，到层数就停（停下的目录标出还有多少没展开）
async function walk(connection, uri, depth) {
  const listing = await list(connection, uri);
  const node = { uri: listing.uri, documents: listing.documents, directories: [], collapsed: false };
  for (const child of listing.directories) {
    if (depth > 1) node.directories.push(await walk(connection, child, depth - 1));
    else node.directories.push({ uri: child, documents: [], directories: [], collapsed: true });
  }
  return node;
}

function render(stdout, node, indent) {
  const entries = [
    ...node.directories.map((child) => ({ label: `${name(child.uri)}/${child.collapsed ? " …" : ""}`, child })),
    ...node.documents.map((uri) => ({ label: name(uri), child: null })),
  ];
  entries.forEach(({ label, child }, index) => {
    const last = index === entries.length - 1;
    stdout.write(`${indent}${last ? "└── " : "├── "}${label}\n`);
    if (child) render(stdout, child, `${indent}${last ? "    " : "│   "}`);
  });
}

function address(value) {
  if (value == null || value === "" || value === "/") return "memory://";
  const text = String(value).trim();
  return text.startsWith("memory://") ? text.replace(/\/+$/, "") || "memory://" : `memory://${text.replace(/^\/+|\/+$/g, "")}`;
}

function name(uri) {
  const path = decodeURIComponent(String(uri).slice("memory://".length));
  return path.split("/").at(-1) || "memory://";
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

function single(args, { optional }) {
  if (args.length > 1) throw new Error(`unknown argument: ${args[1]}`);
  if (args.length === 0 && !optional) throw new Error(USAGE);
  if (args[0]?.startsWith("-")) throw new Error(`unknown argument: ${args[0]}`);
  return args[0];
}

function write(stdout, result) {
  stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  return 0;
}
