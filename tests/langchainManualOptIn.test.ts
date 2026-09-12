/**
 * B2 — LangChain instrumentation must be opt-in under instrumentModules mode.
 *
 * The raw `@traceloop/instrumentation-langchain` can be resolvable purely
 * transitively (e.g. via @traceloop/node-server-sdk). Constructing it globally
 * patches LangChain's CallbackManager (first-patcher-wins), so in manual
 * (instrumentModules) mode we must NOT even load it unless the integrator
 * opted in — either by installing the `token-police-langchain` companion or by
 * passing a `langChain` key (matched case-insensitively).
 *
 * Auto-discovery mode keeps the documented raw-instrumentor back-compat path.
 *
 * How this is tested: `_applyLangChainInstrumentation` builds its resolver from
 * `process.cwd()` at call time, so each case chdir's into a temp fixture app
 * whose node_modules holds fake packages. The fixtures live in os.tmpdir(), i.e.
 * outside the repo tree, so Node's upward resolution can't escape into a real
 * copy of either package. (Requires a pool where process.chdir exists — vitest's
 * default `forks` pool.)
 */
import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach, vi } from "vitest";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { trace, context, propagation } from "@opentelemetry/api";
import { setupOpenTelemetry, unsetupOpenTelemetry } from "../src/telemetry";

const RAW_FLAG = "__tpFakeRawLangChainConstructed";
const COMPANION_FLAG = "__tpFakeCompanionLangChainConstructed";

/**
 * Writes a fake instrumentor package exporting `LangChainInstrumentation` —
 * the export name the SDK instantiates (`pkg.LangChainInstrumentation ||
 * pkg.default`). Its CONSTRUCTOR bumps a global counter, which is what the
 * real package's global CallbackManager patch rides on.
 */
function writeFakeInstrumentor(appDir: string, pkgName: string, flag: string): void {
  const dir = join(appDir, "node_modules", ...pkgName.split("/"));
  mkdirSync(dir, { recursive: true });
  writeFileSync(
    join(dir, "package.json"),
    JSON.stringify({ name: pkgName, version: "0.0.0", main: "index.js" }),
  );
  writeFileSync(
    join(dir, "index.js"),
    `const FLAG = ${JSON.stringify(flag)};
class LangChainInstrumentation {
  constructor() {
    globalThis[FLAG] = (globalThis[FLAG] || 0) + 1;
  }
  setTracerProvider() {}
  enable() {}
  disable() {}
  manuallyInstrument() {}
}
module.exports = { LangChainInstrumentation };
`,
  );
}

let tmpRoot: string;
let rawOnlyApp: string;
let companionApp: string;
let originalCwd: string;
// vi.spyOn's return type is generic over the spied signature; `any` keeps the
// assertion helpers below readable without fighting it.
let warnSpy: any;

beforeAll(() => {
  originalCwd = process.cwd();
  tmpRoot = mkdtempSync(join(tmpdir(), "tp-lc-optin-"));

  // App A: only the raw third-party instrumentor is resolvable.
  rawOnlyApp = join(tmpRoot, "raw-only-app");
  mkdirSync(rawOnlyApp, { recursive: true });
  writeFakeInstrumentor(rawOnlyApp, "@traceloop/instrumentation-langchain", RAW_FLAG);

  // App B: the TokenPolice companion is installed (that IS the opt-in), with
  // the raw instrumentor also present so we can prove which one is used.
  companionApp = join(tmpRoot, "companion-app");
  mkdirSync(companionApp, { recursive: true });
  writeFakeInstrumentor(companionApp, "@traceloop/instrumentation-langchain", RAW_FLAG);
  writeFakeInstrumentor(companionApp, "token-police-langchain", COMPANION_FLAG);
});

afterAll(() => {
  try {
    process.chdir(originalCwd);
  } catch {
    /* ignore */
  }
  try {
    rmSync(tmpRoot, { recursive: true, force: true });
  } catch {
    /* ignore */
  }
});

function resetState(): void {
  try {
    unsetupOpenTelemetry();
  } catch {
    /* ignore */
  }
  try {
    trace.disable();
  } catch {
    /* ignore */
  }
  try {
    context.disable();
  } catch {
    /* ignore */
  }
  try {
    propagation.disable();
  } catch {
    /* ignore */
  }
  delete (globalThis as any)[RAW_FLAG];
  delete (globalThis as any)[COMPANION_FLAG];
  try {
    process.chdir(originalCwd);
  } catch {
    /* ignore */
  }
}

beforeEach(() => {
  resetState();
  warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
});

afterEach(() => {
  resetState();
  vi.restoreAllMocks();
});

function warnings(): string {
  return warnSpy.mock.calls.map((c) => c.join(" ")).join("\n");
}

describe("LangChain instrumentation opt-in under instrumentModules mode (B2)", () => {
  it("manual mode without a langChain key: raw instrumentor is NOT constructed, and warns", () => {
    process.chdir(rawOnlyApp);
    // `cohere` is a real InstrumentModules key with no INSTRUMENTOR_REGISTRY
    // entry, so manual mode is active without pulling in other instrumentors.
    setupOpenTelemetry({ cohere: {} } as any);

    expect((globalThis as any)[RAW_FLAG]).toBeUndefined();
    expect(warnings()).toContain("requires explicit opt-in for LangChain");
    expect(warnings()).toContain("@traceloop/instrumentation-langchain");
  });

  it("manual mode WITH langChain key: raw instrumentor IS constructed", () => {
    process.chdir(rawOnlyApp);
    setupOpenTelemetry({ cohere: {}, langChain: {} } as any);

    expect((globalThis as any)[RAW_FLAG]).toBe(1);
    expect(warnings()).not.toContain("requires explicit opt-in for LangChain");
  });

  it("manual mode with lowercase `langchain` key: still opted in (case-insensitive)", () => {
    process.chdir(rawOnlyApp);
    setupOpenTelemetry({ cohere: {}, langchain: {} } as any);

    expect((globalThis as any)[RAW_FLAG]).toBe(1);
    expect(warnings()).not.toContain("requires explicit opt-in for LangChain");
  });

  it("auto-discovery mode: raw instrumentor back-compat path is preserved", () => {
    process.chdir(rawOnlyApp);
    setupOpenTelemetry();

    expect((globalThis as any)[RAW_FLAG]).toBe(1);
    expect(warnings()).not.toContain("requires explicit opt-in for LangChain");
  });

  it("manual mode with the companion installed: companion is used, raw is not", () => {
    process.chdir(companionApp);
    setupOpenTelemetry({ cohere: {} } as any);

    expect((globalThis as any)[COMPANION_FLAG]).toBe(1);
    expect((globalThis as any)[RAW_FLAG]).toBeUndefined();
    expect(warnings()).not.toContain("requires explicit opt-in for LangChain");
  });
});
