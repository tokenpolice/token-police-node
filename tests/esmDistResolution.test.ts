/**
 * ESM dist resolution — end-to-end against the BUILT bundle.
 *
 * In dist/index.js esbuild rewrites bare `require` into a `__require` shim
 * that THROWS ("Dynamic require of ... is not supported") under `import`.
 * Before the app-first resolver, the enforcer's whole auto-discovery path ran
 * through that shim, so an ESM app's provider SDK copy was silently never
 * wrapped (empirically: same CJS `openai` fixture — ESM entry wrapped=false,
 * CJS entry wrapped=true). These tests spawn real node children against
 * dist/index.js to pin:
 * 1. an app-level CJS `openai` package IS enforcer-wrapped from an ESM entry,
 *    with no zero-wrap warning,
 * 2. when the wrap cannot land (shape mismatch), the zero-wrap warning
 *    appears on stderr — and the app still exits cleanly (golden rule),
 * 3. chdir-after-import: the child starts in a NEUTRAL cwd, loads the dist
 *    bundle, then process.chdir()s into the app root before init() — the app
 *    copy must still be wrapped (the anchor is evaluated per resolution
 *    attempt, not frozen at SDK load time) and no zero-wrap warning fires.
 *
 * REQUIRES dist/ — run `npm run build` first. The suite skips (loudly, not
 * fails) when dist/index.js is missing, since vitest may run before a build.
 */
import { describe, test, expect, afterAll } from "vitest";
import { spawnSync } from "node:child_process";
import {
  mkdtempSync,
  mkdirSync,
  writeFileSync,
  rmSync,
  existsSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const sdkRoot = resolve(fileURLToPath(new URL(".", import.meta.url)), "..");
const distEsm = join(sdkRoot, "dist", "index.js");
const hasDist = existsSync(distEsm);

// Fake CJS `openai` package matching the registry objectPath
// ["OpenAI","Chat","Completions","prototype"], method "create".
const VALID_OPENAI_SRC = `
function create() { return Promise.resolve({}); }
function Completions() {}
Completions.prototype.create = create;
const Chat = { Completions };
function OpenAI() {}
OpenAI.Chat = Chat;
module.exports = { OpenAI, default: OpenAI };
`;

// Resolvable but unwrappable: the module loads fine, the objectPath doesn't.
const MISMATCHED_OPENAI_SRC = `module.exports = {};`;

// ESM entry: capture the app copy's method identity before/after init() of
// the BUILT ESM bundle. serverless deployment → no SSE stream to hang on;
// explicit exit(0) so lingering fail-open timers can't hold the process.
const ENTRY_SRC = `
import { createRequire } from "node:module";
import { join } from "node:path";
const req = createRequire(join(process.cwd(), "node_modules"));
const before = req("openai").OpenAI?.Chat?.Completions?.prototype?.create;
const { init } = await import(process.env.TP_DIST);
init({
  apiKey: "tp_sk_test",
  deployment: "serverless",
  firewall: "dry_run",
  baseUrl: "http://127.0.0.1:9",
});
const after = req("openai").OpenAI?.Chat?.Completions?.prototype?.create;
console.log("TP_WRAPPED=" + (before !== undefined && after !== undefined && after !== before));
process.exit(0);
`;

// Same probe as ENTRY_SRC, but for a systemd/pm2-style launch: the process
// starts in a NEUTRAL cwd, imports the built bundle FIRST (so any load-time
// anchor freezing would capture the WRONG cwd), then chdir()s into the app
// root before init(). Because cwd changes mid-run, the app's `openai` copy is
// resolved relative to the FIXTURE dir explicitly (TP_FIXTURE), never cwd.
const CHDIR_ENTRY_SRC = `
import { createRequire } from "node:module";
import { join } from "node:path";
const fixtureDir = process.env.TP_FIXTURE;
const { init } = await import(process.env.TP_DIST);
process.chdir(fixtureDir);
const req = createRequire(join(fixtureDir, "node_modules"));
const before = req("openai").OpenAI?.Chat?.Completions?.prototype?.create;
init({
  apiKey: "tp_sk_test",
  deployment: "serverless",
  firewall: "dry_run",
  baseUrl: "http://127.0.0.1:9",
});
const after = req("openai").OpenAI?.Chat?.Completions?.prototype?.create;
console.log("TP_WRAPPED=" + (before !== undefined && after !== undefined && after !== before));
process.exit(0);
`;

function makeFixture(openaiSrc: string): string {
  const dir = mkdtempSync(join(tmpdir(), "tp-esm-dist-"));
  const pkgDir = join(dir, "node_modules", "openai");
  mkdirSync(pkgDir, { recursive: true });
  writeFileSync(
    join(pkgDir, "package.json"),
    JSON.stringify({ name: "openai", version: "6.0.0", main: "index.js" }),
  );
  writeFileSync(join(pkgDir, "index.js"), openaiSrc);
  writeFileSync(join(dir, "entry.mjs"), ENTRY_SRC);
  writeFileSync(join(dir, "entry-chdir.mjs"), CHDIR_ENTRY_SRC);
  return dir;
}

function runFixture(
  dir: string,
  opts: { entry?: string; cwd?: string; env?: Record<string, string> } = {},
): {
  status: number | null;
  stdout: string;
  stderr: string;
} {
  const out = spawnSync(
    process.execPath,
    [join(dir, opts.entry ?? "entry.mjs")],
    {
      cwd: opts.cwd ?? dir,
      env: { ...process.env, TP_DIST: distEsm, ...opts.env },
      encoding: "utf8",
      timeout: 8000,
    },
  );
  return { status: out.status, stdout: out.stdout ?? "", stderr: out.stderr ?? "" };
}

const fixtures: string[] = [];

afterAll(() => {
  for (const dir of fixtures) {
    try {
      rmSync(dir, { recursive: true, force: true });
    } catch {
      /* best-effort cleanup */
    }
  }
});

describe.skipIf(!hasDist)("ESM dist — app-copy enforcement (build required)", () => {
  test("app-level CJS openai copy gets enforcer-wrapped from an ESM entry", () => {
    const dir = makeFixture(VALID_OPENAI_SRC);
    fixtures.push(dir);
    const out = runFixture(dir);
    expect(out.status).toBe(0);
    expect(out.stdout).toContain("TP_WRAPPED=true");
    // The wrap landed, so the zero-wrap warning must NOT fire.
    expect(out.stderr).not.toContain("could not attach enforcement");
  });

  test("resolvable-but-unwrappable openai → zero-wrap warning on stderr", () => {
    const dir = makeFixture(MISMATCHED_OPENAI_SRC);
    fixtures.push(dir);
    const out = runFixture(dir);
    // Golden rule: even in the broken state the app exits cleanly.
    expect(out.status).toBe(0);
    expect(out.stdout).toContain("TP_WRAPPED=false");
    expect(out.stderr).toContain("[TokenPolice Warning]");
    expect(out.stderr).toContain("could not attach enforcement");
    expect(out.stderr).toContain("openai");
    expect(out.stderr).toContain("instrumentModules");
  });

  test("chdir-after-import: app copy is wrapped from the FINAL cwd, not the load-time cwd", () => {
    const fixtureDir = makeFixture(VALID_OPENAI_SRC);
    fixtures.push(fixtureDir);
    // Neutral start dir with NO node_modules — a load-time-frozen app anchor
    // would capture this dir, miss the app's openai copy, fall back to the
    // SDK-relative anchor (pre-fix wrong copy), and ALSO suppress the
    // zero-wrap warning (same stale anchor in _appCanResolve).
    const neutralDir = mkdtempSync(join(tmpdir(), "tp-neutral-"));
    fixtures.push(neutralDir);
    const out = runFixture(fixtureDir, {
      entry: "entry-chdir.mjs",
      cwd: neutralDir,
      env: { TP_FIXTURE: fixtureDir },
    });
    expect(out.status).toBe(0);
    expect(out.stdout).toContain("TP_WRAPPED=true");
    expect(out.stderr).not.toContain("could not attach enforcement");
  });
});

if (!hasDist) {
  // Loud skip marker so a missing build is visible in the run output.
  describe("ESM dist — app-copy enforcement", () => {
    test.skip("SKIPPED: dist/index.js missing — run `npm run build` first", () => {});
  });
}
