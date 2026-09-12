/**
 * Init() must NEVER throw into customer code at setup time.
 *
 * A malformed / partially-installed instrumented SDK (a throwing property
 * getter, a throwing module body, broken OTel global state) must degrade to
 * reduced telemetry, not crash the customer app at boot. These tests pin the
 * four Node guard sites:
 * 1. per-prep-entry guard inside autoInstrument's instrumentModules loop,
 * 2. per-target guard around each _wrapMethod call,
 * 3. setupOpenTelemetry call-site guard in init(),
 * 4. autoInstrument call-site guard in init() (defense-in-depth).
 *
 * Mirrors token-police-python/tests/test_init_never_throws.py (same intent;
 * the prep-loop + autoInstrument call-site guards are Node-only by design —
 * Python's auto_instrument has no pre-loop prep body).
 */
import { describe, test, expect, afterEach, vi } from "vitest";
import { autoInstrument, uninstrument } from "../src/enforcer";
import { init, TokenPolice } from "../src/client";
import { setClient } from "../src/state";
import * as telemetry from "../src/telemetry";

// A fake `openai` namespace value: _resolvePath walks the registry path
// ["OpenAI","Chat","Completions","prototype"] then wraps method `create`.
function makeValidOpenAI(): any {
  function create(this: any) { return Promise.resolve({}); }
  function Completions(this: any) {}
  Completions.prototype.create = create;
  const Chat = { Completions };
  function OpenAI(this: any) {}
  (OpenAI as any).Chat = Chat;
  return OpenAI;
}

// A fake `anthropic` namespace value: path ["Anthropic","Messages","prototype"].
function makeValidAnthropic(): any {
  function create(this: any) { return Promise.resolve({}); }
  function Messages(this: any) {}
  Messages.prototype.create = create;
  function Anthropic(this: any) {}
  (Anthropic as any).Messages = Messages;
  return Anthropic;
}

// A value that throws on ANY property access. Survives the prep loop: the
// "openai"/"anthropic" branches now read properties via _pickClassExport, but
// that helper's own try/catch swallows the hostile getters and returns the value
// verbatim. It then throws inside the per-target loop when _resolvePath reads a
// property — exercising the LOOP guard.
function throwingProxy(): any {
  return new Proxy({}, { get() { throw new Error("boom: throwing property access"); } });
}

afterEach(() => {
  try { uninstrument(); } catch { /* ignore */ }
  try { setClient(undefined as any); } catch { /* ignore */ }
  vi.restoreAllMocks();
});

describe("Per-target loop guard", () => {
  // Assertion 2/4
  test("throwing target does not propagate", () => {
    expect(() => autoInstrument({ openai: throwingProxy() } as any)).not.toThrow();
  });

  // Assertion 7/8 — per-target isolation: a throwing target (openai, processed
  // FIRST) must not prevent a later valid target (anthropic) from instrumenting.
  test("per-target isolation: a throwing target does not skip a later valid one", () => {
    const Anthropic = makeValidAnthropic();
    const original = Anthropic.Messages.prototype.create;
    expect(() =>
      autoInstrument({ openai: throwingProxy(), anthropic: Anthropic } as any),
    ).not.toThrow();
    expect(Anthropic.Messages.prototype.create).not.toBe(original);
  });
});

describe("Prep-phase guard (Node-only isolation mechanism)", () => {
  // Assertion 21 — a throwing property getter under "langchain" throws DURING
  // the prep loop (before the per-target loop). The per-prep-entry guard must
  // log-and-skip it and CONTINUE building moduleMap so the valid "openai" entry
  // still instruments. Fails under a call-site-only guard (which would abort the
  // whole autoInstrument invocation → openai never wrapped).
  test("prep-phase throwing module is isolated and a valid entry still instruments", () => {
    const throwingLangchain = {
      get chatModelsModule(): any { throw new Error("boom: prep getter"); },
    };
    const OpenAI = makeValidOpenAI();
    const original = OpenAI.Chat.Completions.prototype.create;
    expect(() =>
      autoInstrument({ langchain: throwingLangchain, openai: OpenAI } as any),
    ).not.toThrow();
    expect(OpenAI.Chat.Completions.prototype.create).not.toBe(original);
  });

  // Same isolation for the llamaindex getter site (enforcer.ts:6787-6789).
  test("prep-phase throwing llamaindex getter is isolated and openai still instruments", () => {
    const throwingLlamaIndex = {
      get openaiModule(): any { throw new Error("boom: llamaindex getter"); },
    };
    const OpenAI = makeValidOpenAI();
    const original = OpenAI.Chat.Completions.prototype.create;
    expect(() =>
      autoInstrument({ llamaindex: throwingLlamaIndex, openai: OpenAI } as any),
    ).not.toThrow();
    expect(OpenAI.Chat.Completions.prototype.create).not.toBe(original);
  });
});

describe("SetupOpenTelemetry / autoInstrument call-site guards", () => {
  // Assertion 6 — setupOpenTelemetry throwing must not escape init().
  test("setup failure does not propagate from init()", () => {
    const spy = vi
      .spyOn(telemetry, "setupOpenTelemetry")
      .mockImplementation(() => { throw new Error("boom: otel provider construction"); });
    expect(() =>
      init({ apiKey: "tp_sk_test", deployment: "serverless", firewall: "off" } as any),
    ).not.toThrow();
    expect(spy).toHaveBeenCalled();
  });

  // Assertion 4 — init() with a throwing module entry completes + returns.
  test("init() never throws when a module entry throws during setup", () => {
    let client: any;
    expect(() => {
      client = init({
        apiKey: "tp_sk_test",
        deployment: "serverless",
        firewall: "dry_run",
        instrumentModules: { openai: throwingProxy() },
      } as any);
    }).not.toThrow();
    expect(client).toBeInstanceOf(TokenPolice);
  });
});

describe("Happy path / off mode (no regression)", () => {
  // Assertion 9 — positive control: a healthy entry IS wrapped (identity changes).
  test("healthy autoInstrument wraps a valid target", () => {
    const OpenAI = makeValidOpenAI();
    const original = OpenAI.Chat.Completions.prototype.create;
    autoInstrument({ openai: OpenAI } as any);
    expect(OpenAI.Chat.Completions.prototype.create).not.toBe(original);
  });

  // Assertion 10 — off mode DOES install the tap ("off = telemetry only"
  // must deliver full telemetry). The tap is log-only in off: the enforcer choke
  // point (_runAsyncCheck) short-circuits on firewall==="off" before any /check,
  // block, or reroute — so installing the wrapper is safe and required for the
  // manual-tap providers to emit /log.
  test("off mode installs the tap (log-only)", () => {
    const OpenAI = makeValidOpenAI();
    const original = OpenAI.Chat.Completions.prototype.create;
    init({
      apiKey: "tp_sk_test",
      deployment: "serverless",
      firewall: "off",
      instrumentModules: { openai: OpenAI },
    } as any);
    expect(OpenAI.Chat.Completions.prototype.create).not.toBe(original);
  });
});

describe("Failure-path logging is silent by default", () => {
  // Assertion 14 — no console.warn when logErrors unset; the gated diagnostic
  // fires only when logErrors:true.
  test("setup failure is silent unless logErrors", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    // No client / logErrors unset → silent.
    autoInstrument({ openai: throwingProxy() } as any);
    expect(warnSpy).not.toHaveBeenCalled();
    uninstrument();

    // logErrors:true → the gated diagnostic fires.
    setClient(new TokenPolice({ apiKey: "tp_sk_test", deployment: "serverless", logErrors: true }));
    autoInstrument({ openai: throwingProxy() } as any);
    expect(warnSpy).toHaveBeenCalled();
  });
});
