/**
 * Enforcer app-first module resolution (resolveProviderModule).
 *
 * The enforcer and telemetry must attach to the SAME provider-SDK module copy.
 * Telemetry resolves app-first (createRequire(process.cwd()/node_modules) →
 * SDK-relative); the enforcer's old bare `require(...)` diverged from that on
 * file:/pnpm/monorepo topologies (resolved the SDK's own devDependency copies)
 * and threw outright in the ESM dist (esbuild's `__require` shim), so the
 * app's copy was silently never wrapped — pre-flight /check never ran, calls
 * were mis-attributed, and manual-telemetry providers lost all cost data.
 *
 * These tests pin the replacement resolver:
 * 1. app anchor wins over the SDK anchor,
 * 2. SDK-relative fallback when the app anchor misses,
 * 3. successful resolutions are cached — the SAME object comes back, and
 *    uninstrument() restores onto exactly the module object that was wrapped,
 * 4. total resolution failure → undefined, no throw (silent-skip preserved),
 * 5. hostile anchors / modules can never throw out of the resolver or out of
 *    autoInstrument (golden rule).
 */
import { describe, test, expect, afterEach, afterAll, vi } from "vitest";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, realpathSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { autoInstrument, uninstrument, __test__ } from "../src/enforcer";
import { setClient } from "../src/state";

const {
  _resolveWithAnchors,
  resolveProviderModule,
  _resolvedModules,
  _getAppRequire,
} = __test__ as any;

// A fake `openai` module record shaped like the real package root, matching
// the registry objectPath ["OpenAI","Chat","Completions","prototype"].
function makeFakeOpenAIModule(): any {
  function create(this: any) { return Promise.resolve({}); }
  function Completions(this: any) {}
  Completions.prototype.create = create;
  const Chat = { Completions };
  function OpenAI(this: any) {}
  (OpenAI as any).Chat = Chat;
  return { OpenAI, default: OpenAI };
}

afterEach(() => {
  try { uninstrument(); } catch { /* ignore */ }
  try { setClient(undefined as any); } catch { /* ignore */ }
  // The resolution cache is intentionally process-lifetime; scrub test seeds
  // so suites in this file can't contaminate each other.
  try { _resolvedModules.delete("openai"); } catch { /* ignore */ }
  vi.restoreAllMocks();
});

describe("_resolveWithAnchors — anchor ordering", () => {
  test("app anchor wins over the SDK anchor", () => {
    const appMod = { src: "app" };
    const sdkMod = { src: "sdk" };
    const appReq: any = vi.fn(() => appMod);
    const sdkReq: any = vi.fn(() => sdkMod);
    const out = _resolveWithAnchors("openai", [appReq, sdkReq], new Map());
    expect(out).toBe(appMod);
    expect(sdkReq).not.toHaveBeenCalled();
  });

  test("falls back to the SDK anchor when the app anchor misses", () => {
    const sdkMod = { src: "sdk" };
    const appReq: any = vi.fn(() => { throw new Error("Cannot find module"); });
    const sdkReq: any = vi.fn(() => sdkMod);
    expect(_resolveWithAnchors("openai", [appReq, sdkReq], new Map())).toBe(sdkMod);
  });

  test("undefined anchors are skipped (fail-open construction)", () => {
    const sdkMod = { src: "sdk" };
    const sdkReq: any = vi.fn(() => sdkMod);
    expect(_resolveWithAnchors("openai", [undefined, sdkReq], new Map())).toBe(sdkMod);
  });

  test("total failure returns undefined without throwing", () => {
    const boom: any = () => { throw new Error("Dynamic require of \"openai\" is not supported"); };
    let out: any = "sentinel";
    expect(() => {
      out = _resolveWithAnchors("openai", [boom, boom], new Map());
    }).not.toThrow();
    expect(out).toBeUndefined();
  });

  test("no anchors at all returns undefined without throwing", () => {
    expect(_resolveWithAnchors("openai", [undefined, undefined], new Map())).toBeUndefined();
  });
});

describe("_resolveWithAnchors — cache identity", () => {
  test("second resolution returns the IDENTICAL object without re-requiring", () => {
    const cache = new Map<string, any>();
    const mod = { src: "app" };
    const appReq: any = vi.fn(() => mod);
    const first = _resolveWithAnchors("openai", [appReq], cache);
    const second = _resolveWithAnchors("openai", [appReq], cache);
    expect(first).toBe(mod);
    expect(second).toBe(first);
    expect(appReq).toHaveBeenCalledTimes(1);
  });

  test("a failed resolution is NOT negatively cached — a later anchor hit works", () => {
    const cache = new Map<string, any>();
    const boom: any = () => { throw new Error("not installed yet"); };
    expect(_resolveWithAnchors("openai", [boom], cache)).toBeUndefined();
    const mod = { src: "app" };
    expect(_resolveWithAnchors("openai", [() => mod], cache)).toBe(mod);
  });
});

describe("resolveProviderModule → wrap → uninstrument round trip", () => {
  test("uninstrument restores the original method on the SAME module object that was wrapped", () => {
    const fake = makeFakeOpenAIModule();
    const original = fake.OpenAI.Chat.Completions.prototype.create;
    // Seed the process-level cache so auto-discovery (no instrumentModules)
    // resolves OUR module record — exactly how the cache guarantees identity
    // between the wrap and the restore.
    _resolvedModules.set("openai", fake);

    autoInstrument();
    expect(fake.OpenAI.Chat.Completions.prototype.create).not.toBe(original);

    uninstrument();
    expect(fake.OpenAI.Chat.Completions.prototype.create).toBe(original);
  });

  test("cached copy is what wrap sees — resolveProviderModule returns it verbatim", () => {
    const fake = makeFakeOpenAIModule();
    _resolvedModules.set("openai", fake);
    expect(resolveProviderModule("openai")).toBe(fake);
  });
});

describe("_getAppRequire — lazy per-call anchor (chdir-after-import contract)", () => {
  // The anchor must be constructed from the CURRENT process.cwd() at each
  // call, never frozen at SDK-module-load time. process.chdir() is unsafe in
  // vitest workers (shared process, races other test files), so cwd is
  // MOCKED via vi.spyOn instead — same observable input to _getAppRequire.
  // The full end-to-end chdir repro lives in esmDistResolution.test.ts
  // (spawned child, real process.chdir against the built dist).

  const anchorDirs: string[] = [];

  function makeAppDir(): string {
    const dir = mkdtempSync(join(tmpdir(), "tp-anchor-"));
    anchorDirs.push(dir);
    const pkgDir = join(dir, "node_modules", "tp-anchor-probe");
    mkdirSync(pkgDir, { recursive: true });
    writeFileSync(
      join(pkgDir, "package.json"),
      JSON.stringify({ name: "tp-anchor-probe", version: "1.0.0", main: "index.js" }),
    );
    writeFileSync(join(pkgDir, "index.js"), "module.exports = {};");
    // require.resolve realpath-resolves its result, and on macOS os.tmpdir()
    // sits behind a /var → /private/var symlink — return the realpath so the
    // test's join(dir, ...) expectations compare like with like.
    return realpathSync(dir);
  }

  afterAll(() => {
    for (const dir of anchorDirs) {
      try { rmSync(dir, { recursive: true, force: true }); } catch { /* best-effort */ }
    }
  });

  test("a fresh call reflects a CHANGED cwd (anchor is not frozen)", () => {
    const dirA = makeAppDir();
    const dirB = makeAppDir();
    const cwdSpy = vi.spyOn(process, "cwd");

    cwdSpy.mockReturnValue(dirA);
    const resolvedA = _getAppRequire()?.resolve("tp-anchor-probe");
    expect(resolvedA).toBe(join(dirA, "node_modules", "tp-anchor-probe", "index.js"));

    cwdSpy.mockReturnValue(dirB);
    const resolvedB = _getAppRequire()?.resolve("tp-anchor-probe");
    expect(resolvedB).toBe(join(dirB, "node_modules", "tp-anchor-probe", "index.js"));
  });

  test("a throwing process.cwd cannot escape _getAppRequire (golden rule)", () => {
    vi.spyOn(process, "cwd").mockImplementation(() => {
      throw new Error("boom: cwd deleted");
    });
    expect(() => _getAppRequire()).not.toThrow();
  });
});

describe("golden rule — hostile modules cannot escape", () => {
  test("a module whose property access throws is skipped silently by autoInstrument", () => {
    const hostile = new Proxy({}, {
      get() { throw new Error("boom: hostile provider module"); },
    });
    _resolvedModules.set("openai", hostile);
    expect(() => autoInstrument()).not.toThrow();
  });

  test("resolveProviderModule never throws for an unresolvable name", () => {
    let out: any = "sentinel";
    expect(() => {
      out = resolveProviderModule("definitely-not-a-real-package-tokenpolice");
    }).not.toThrow();
    expect(out).toBeUndefined();
  });
});
