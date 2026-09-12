/**
 * Node-4: the per-module `logger.debug` in client.ts and enforcer.ts were hard
 * no-ops (`(msg) => {}`), so:
 * - flushSync()'s pending-count diagnostic never printed (its only action).
 * - uninstrument() restore-thunk failures were permanently swallowed.
 * Fix: wire both loggers to the SAME gate telemetry.ts already uses
 * (`getClient()?.logErrors` → console.log), and make `debug` accept a
 * `() => string` callback so callers can defer building an expensive message
 * until debug is actually on. §4-minor-e applies that laziness to the hot
 * span-end line in telemetry.ts:onEnd.
 *
 * These tests assert:
 * 1. client (via flushSync) + enforcer (via autoInstrument) debug lines reach
 * the console ONLY when logErrors:true; silent by default.
 * 2. flushSync never throws (empty / closed client), returns undefined, and
 * emits the real pending-count diagnostic (from `_pendingPromises.size`)
 * when debug is enabled.
 * 3. span-end: with debug OFF the attribute-key enumeration in the debug
 * message is NOT executed (proven with an ownKeys-recording Proxy).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { TokenPolice } from "../src/client";
import { setClient, getClient } from "../src/state";
import { autoInstrument, uninstrument } from "../src/enforcer";
import { TokenPoliceSpanProcessor } from "../src/telemetry";

// ── console.log interceptor (records only our debug lines) ───────────
let debugLines: string[] = [];
let origLog: typeof console.log;

beforeEach(() => {
  debugLines = [];
  origLog = console.log;
  console.log = (...a: unknown[]) => {
    const s = a.map(String).join(" ");
    if (s.includes("[TokenPolice Debug]")) debugLines.push(s);
    // swallow otherwise to keep test output clean
  };
});

afterEach(() => {
  console.log = origLog;
  setClient(null as any);
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

/** Stub global fetch so `log()` produces a settled fire-and-forget promise. */
function stubFetch(): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () =>
      new Response(JSON.stringify({ allowed: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    ),
  );
}

function makeClient(logErrors: boolean): TokenPolice {
  const c = new TokenPolice({
    apiKey: "tp_sk_test123",
    baseUrl: "http://localhost:13099",
    timeout: 0.5,
    deployment: "serverless",
    logErrors,
  });
  setClient(c); // the module loggers resolve logErrors via getClient()
  return c;
}

// A ReadableSpan stub whose `attributes` is a Proxy recording ownKeys traps —
// `Object.keys(attrs)` (the only enumeration in the debug message) trips it.
function fakeSpanWithKeyProbe(
  attrs: Record<string, unknown>,
  name: string,
  onEnumerate: () => void,
) {
  const proxied = new Proxy(attrs, {
    ownKeys(target) {
      onEnumerate();
      return Reflect.ownKeys(target);
    },
  });
  return {
    attributes: proxied,
    name,
    startTime: [0, 0] as [number, number],
    endTime: [1, 0] as [number, number],
    spanContext: () => ({
      traceId: "0000000000000000000000000000abcd",
      spanId: "0000000000001234",
    }),
    parentSpanId: undefined,
  } as any;
}

describe("Node-4: client logger.debug wired via flushSync diagnostic", () => {
  it("logErrors:false (default) → flushSync emits 0 debug lines with pending logs", async () => {
    stubFetch();
    const client = makeClient(false);
    client.log("u", "free", "default", "", "gpt-4o", "openai", 100, 50, 0);
    // pending log present, but debug is gated off
    let result: unknown = "sentinel";
    expect(() => {
      result = client.flushSync();
    }).not.toThrow();
    expect(result).toBeUndefined();
    expect(debugLines.length).toBe(0);
    await client.flush();
  });

  it("logErrors:true → flushSync emits the real pending-count diagnostic", async () => {
    stubFetch();
    const client = makeClient(true);
    client.log("u", "free", "default", "", "gpt-4o", "openai", 100, 50, 0);
    let result: unknown = "sentinel";
    expect(() => {
      result = client.flushSync();
    }).not.toThrow();
    expect(result).toBeUndefined();
    expect(debugLines.length).toBeGreaterThanOrEqual(1);
    // The diagnostic reports the actual pending count (1) — proves it reads
    // real state (_pendingPromises.size), not a canned string.
    expect(debugLines.some((l) => /\b1 pending log/.test(l))).toBe(true);
    await client.flush();
  });

  it("flushSync never throws on an empty client and emits nothing (size 0)", () => {
    const client = makeClient(true);
    let result: unknown = "sentinel";
    expect(() => {
      result = client.flushSync();
    }).not.toThrow();
    expect(result).toBeUndefined();
    // No pending logs ⇒ no diagnostic even with debug on.
    expect(debugLines.length).toBe(0);
  });

  it("flushSync never throws on a closed client", async () => {
    stubFetch();
    const client = makeClient(true);
    await client.close();
    expect(() => client.flushSync()).not.toThrow();
  });
});

describe("Node-4: enforcer logger.debug wired via autoInstrument", () => {
  afterEach(() => {
    try {
      uninstrument();
    } catch {
      /* ignore */
    }
  });

  it("logErrors:true → autoInstrument emits its debug line", () => {
    makeClient(true);
    autoInstrument();
    expect(
      debugLines.some((l) => l.includes("Pre-flight enforcement hooks applied.")),
    ).toBe(true);
  });

  it("logErrors:false (default) → autoInstrument is silent", () => {
    makeClient(false);
    autoInstrument();
    expect(debugLines.length).toBe(0);
  });
});

describe("Node-4 §4-minor-e: span-end debug message is lazy (no enumeration when off)", () => {
  it("debug OFF → attribute-key enumeration for the debug line is NOT run", () => {
    stubFetch(); // onEnd's fire-and-forget /log must not hit the network
    makeClient(false); // logErrors off
    const proc = new TokenPoliceSpanProcessor();
    let offEnumerations = 0;
    proc.onEnd(
      fakeSpanWithKeyProbe(
        { "gen_ai.system": "openai", "gen_ai.usage.input_tokens": 3 },
        "openai.chat",
        () => {
          offEnumerations++;
        },
      ),
    );
    const offCount = offEnumerations;

    makeClient(true); // logErrors on
    let onEnumerations = 0;
    proc.onEnd(
      fakeSpanWithKeyProbe(
        { "gen_ai.system": "openai", "gen_ai.usage.input_tokens": 3 },
        "openai.chat",
        () => {
          onEnumerations++;
        },
      ),
    );
    const onCount = onEnumerations;

    // The debug-on run performs at least one MORE ownKeys enumeration than the
    // debug-off run — that extra one is the deferred `Object.keys(attrs)` inside
    // the lazy debug callback, which is skipped entirely when debug is off.
    expect(onCount).toBeGreaterThan(offCount);
    // And the debug line really did print when on.
    expect(debugLines.some((l) => l.includes("Span ended: openai.chat"))).toBe(
      true,
    );
  });
});
