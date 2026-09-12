/**
 * Zero-wrap LOUD warning (init()-time enforcement audit).
 *
 * Historically the enforcer's failure to attach to a provider SDK was
 * completely silent: the wrap-path `catch { return; }` logged nothing and the
 * only diagnostics were logErrors-gated. The consequence set is severe —
 * pre-flight /check never runs (firewall inert), calls get logged under the
 * wrong provider (baseURL attribution lives in the enforcer), and for
 * manual-telemetry providers ALL telemetry/cost is lost.
 *
 * The audit: after autoInstrument, every distinct _TARGET_METHODS module name
 * that resolves from the APP's node_modules but received zero wraps is named
 * in ONE deduped `[TokenPolice Warning]` console.warn, in every firewall
 * mode. Providers supplied via instrumentModules wrap normally and stay
 * silent. The audit itself is fail-open: any internal throw must never reach
 * the customer's init() (golden rule).
 */
import { describe, test, expect, afterEach, vi } from "vitest";
import {
  autoInstrument,
  uninstrument,
  getUnwrappedResolvableProviders,
} from "../src/enforcer";
import * as enforcer from "../src/enforcer";
import { init, TokenPolice } from "../src/client";
import { setClient } from "../src/state";

const WARN_PREFIX = "[TokenPolice Warning]";

// Same fake-namespace helpers as initNeverThrows.test.ts.
function makeValidOpenAI(): any {
  function create(this: any) { return Promise.resolve({}); }
  function Completions(this: any) {}
  Completions.prototype.create = create;
  const Chat = { Completions };
  function OpenAI(this: any) {}
  (OpenAI as any).Chat = Chat;
  return OpenAI;
}

afterEach(() => {
  try { uninstrument(); } catch { /* ignore */ }
  try { setClient(undefined as any); } catch { /* ignore */ }
  vi.restoreAllMocks();
});

describe("getUnwrappedResolvableProviders — audit core", () => {
  test("resolvable + zero wraps → named; deduped across the registry's repeated moduleNames", () => {
    // Nothing instrumented yet; claim everything is app-resolvable.
    const names = getUnwrappedResolvableProviders(() => true);
    expect(names).toContain("openai");
    expect(names).toContain("@google/genai");
    // _TARGET_METHODS lists openai many times — the audit reports it once.
    expect(names.filter((n) => n === "openai")).toHaveLength(1);
  });

  test("a provider wrapped via instrumentModules is NOT reported", () => {
    autoInstrument({ openai: makeValidOpenAI() } as any);
    const names = getUnwrappedResolvableProviders(() => true);
    expect(names).not.toContain("openai");
    // Sibling with zero wraps in this run is still reported.
    expect(names).toContain("@cerebras/cerebras_cloud_sdk");
  });

  test("not resolvable from the app → not reported (SDK-only copies don't count)", () => {
    expect(getUnwrappedResolvableProviders(() => false)).toEqual([]);
  });

  test("uninstrument resets the counts so a re-audit sees zero wraps again", () => {
    autoInstrument({ openai: makeValidOpenAI() } as any);
    expect(getUnwrappedResolvableProviders(() => true)).not.toContain("openai");
    uninstrument();
    expect(getUnwrappedResolvableProviders(() => true)).toContain("openai");
  });

  test("golden rule: a throwing canResolve returns [] instead of escaping", () => {
    let out: any;
    expect(() => {
      out = getUnwrappedResolvableProviders(() => { throw new Error("boom: hostile resolve"); });
    }).not.toThrow();
    expect(out).toEqual([]);
  });
});

describe("init() warning emission", () => {
  test("zero-wrap provider → exactly one warning naming the package and instrumentModules", () => {
    vi.spyOn(enforcer, "getUnwrappedResolvableProviders").mockReturnValue([
      "@cerebras/cerebras_cloud_sdk",
    ]);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    init({ apiKey: "tp_sk_test", deployment: "serverless", firewall: "dry_run" } as any);

    const zeroWrapWarns = warn.mock.calls
      .map((c) => String(c[0]))
      .filter((m) => m.startsWith(WARN_PREFIX) && m.includes("could not attach enforcement"));
    expect(zeroWrapWarns).toHaveLength(1);
    expect(zeroWrapWarns[0]).toContain("@cerebras/cerebras_cloud_sdk");
    expect(zeroWrapWarns[0]).toContain("instrumentModules");
    // Names all three consequences.
    expect(zeroWrapWarns[0]).toContain("pre-flight checks will not run");
    expect(zeroWrapWarns[0]).toContain("wrong provider");
    expect(zeroWrapWarns[0]).toContain("cost data may be missing");
  });

  test("deduped per process: a second init() does not re-warn for the same provider", () => {
    // NOTE: the dedupe set is intentionally process-level (client.ts
    // _zeroWrapWarned), so this test must use a provider name no other test
    // in this file has already warned about.
    vi.spyOn(enforcer, "getUnwrappedResolvableProviders").mockReturnValue([
      "groq-sdk",
    ]);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    init({ apiKey: "tp_sk_test", deployment: "serverless", firewall: "dry_run" } as any);
    init({ apiKey: "tp_sk_test", deployment: "serverless", firewall: "dry_run" } as any);

    const zeroWrapWarns = warn.mock.calls
      .map((c) => String(c[0]))
      .filter((m) => m.includes("could not attach enforcement") && m.includes("groq-sdk"));
    expect(zeroWrapWarns).toHaveLength(1);
  });

  test("no zero-wrap providers → no warning", () => {
    vi.spyOn(enforcer, "getUnwrappedResolvableProviders").mockReturnValue([]);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    init({ apiKey: "tp_sk_test", deployment: "serverless", firewall: "off" } as any);

    const zeroWrapWarns = warn.mock.calls
      .map((c) => String(c[0]))
      .filter((m) => m.includes("could not attach enforcement"));
    expect(zeroWrapWarns).toHaveLength(0);
  });

  test("golden rule: a throwing audit never escapes init()", () => {
    vi.spyOn(enforcer, "getUnwrappedResolvableProviders").mockImplementation(() => {
      throw new Error("boom: audit internals");
    });
    vi.spyOn(console, "warn").mockImplementation(() => {});

    let client: any;
    expect(() => {
      client = init({ apiKey: "tp_sk_test", deployment: "serverless", firewall: "enforce" } as any);
    }).not.toThrow();
    expect(client).toBeInstanceOf(TokenPolice);
  });
});
