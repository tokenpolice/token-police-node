/**
 * Regression: the shared `logger.debug` in telemetry.ts must be GATED on
 * the client `logErrors` flag (default false), mirroring frameworkTools.ts:33-42
 * `_debug`. Pre-fix it printed `[TokenPolice Debug] …` to stdout on every
 * instrumented span end (hot path), spamming production logs.
 *
 * RED/GREEN: pre-fix the `logger` literal at telemetry.ts:54-57 was
 * `debug: (msg) => console.log(...)` (ungated) — the default-off case below
 * emits ≥1 `[TokenPolice Debug]` line and FAILS. Post-fix (gate added) it
 * emits 0 and PASSES.
 *
 * Contract: the security regression contract .md (AGREED, 9 assertions).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  TokenPoliceSpanProcessor,
  setupOpenTelemetry,
  unsetupOpenTelemetry,
} from "../src/telemetry";
import { setClient } from "../src/state";
import * as state from "../src/state";

// ── console.log interceptor ──────────────────────────────────────────
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
  try {
    unsetupOpenTelemetry();
  } catch {
    /* ignore */
  }
  vi.restoreAllMocks();
});

// A minimal ReadableSpan good enough for onEnd.
function fakeSpan(attrs: Record<string, unknown>, name: string) {
  return {
    attributes: attrs,
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

describe("Telemetry logger.debug is gated on logErrors", () => {
  // Assertion 3 — default config (no logErrors) → ZERO debug lines.
  // (Pre-fix this fails: the ungated logger.debug at:765 prints ≥1 line.)
  it("default config (logErrors unset): span end emits 0 [TokenPolice Debug] lines", () => {
    setClient({ log: () => {} } as any); // no logErrors ⇒ undefined/falsey
    const proc = new TokenPoliceSpanProcessor();
    proc.onEnd(fakeSpan({ "tp.kind": "agent" }, "audit_wf"));
    expect(debugLines.length).toBe(0);
  });

  // Assertion 2 & 4 — logErrors:true restores the:765 span-end debug line
  // (proves the gate is a gate, not a permanent no-op).
  it("logErrors:true: span end emits ≥1 [TokenPolice Debug] line", () => {
    setClient({ logErrors: true, log: () => {} } as any);
    const proc = new TokenPoliceSpanProcessor();
    proc.onEnd(fakeSpan({ "tp.kind": "agent" }, "audit_wf"));
    expect(debugLines.length).toBeGreaterThanOrEqual(1);
    expect(debugLines.some((l) => l.includes("Span ended: audit_wf"))).toBe(
      true,
    );
  });

  // Assertion 6 (INV-2) — with logErrors:true, gated debug lines carry ONLY
  // {span.name, attribute KEYS, token counts} — never attribute VALUES,
  // prompt/completion content, or an API key.
  it("INV-2: enabled debug output contains no content values or api keys", () => {
    const PROMPT_SENTINEL = "SUPER_SECRET_PROMPT_DO_NOT_LOG";
    const KEY_SENTINEL = "tp_sk_PLANTED_KEY_DO_NOT_LOG";
    setClient({ logErrors: true, log: () => {} } as any);
    const proc = new TokenPoliceSpanProcessor();
    proc.onEnd(
      fakeSpan(
        {
          "gen_ai.system": "openai",
          "gen_ai.request.model": "gpt-4o",
          "gen_ai.prompt.0.content": PROMPT_SENTINEL, // VALUE must not leak
          "gen_ai.completion.0.content": PROMPT_SENTINEL,
          "tp.user_id": KEY_SENTINEL, // VALUE must not leak
          "gen_ai.usage.input_tokens": 11,
          "gen_ai.usage.output_tokens": 7,
        },
        "openai.chat",
      ),
    );
    // Something WAS emitted (the gate is on) …
    expect(debugLines.length).toBeGreaterThanOrEqual(1);
    const joined = debugLines.join("\n");
    // … but only KEYS / name / counts, never the planted VALUES or a key.
    expect(joined).not.toContain(PROMPT_SENTINEL);
    expect(joined).not.toContain(KEY_SENTINEL);
    expect(joined).not.toContain("tp_sk_");
    // Positive: the attribute KEYS and token counts ARE what appears.
    expect(joined).toContain("openai.chat");
    expect(joined).toContain("gen_ai.prompt.0.content"); // the KEY only
    expect(joined).toContain("11 input and 7 output tokens");
  });

  // Assertion 5 — the gate's OWN try/catch is the sole
  // protection here. We force getClient() to THROW and reach logger.debug via
  // setupOpenTelemetry (telemetry.ts:1539) — a call site OUTSIDE onEnd's
  // swallowing try/catch (:759–). So if the gate's try/catch were removed, the
  // throw would propagate out of setupOpenTelemetry and this test would FAIL.
  // (Falsification demonstrated at build stage by deleting only the try/catch.)
  it("GetClient() throwing does not escape the gate (no throw, no output)", () => {
    unsetupOpenTelemetry(); // ensure _isSetup=false so setup reaches:1539
    const spy = vi
      .spyOn(state, "getClient")
      .mockImplementation(() => {
        throw new Error("boom: getClient exploded");
      });
    try {
      // Must NOT throw — the gate's try/catch swallows getClient()'s throw.
      expect(() => setupOpenTelemetry()).not.toThrow();
      // And nothing was printed on the throwing path.
      expect(debugLines.length).toBe(0);
      // Sanity: the gate really did invoke the throwing getClient.
      expect(spy).toHaveBeenCalled();
    } finally {
      spy.mockRestore();
    }
  });
});
