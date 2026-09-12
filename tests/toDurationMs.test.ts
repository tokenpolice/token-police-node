/**
 * ToDurationMs: positive sub-ms deltas report 1, true-zero stays 0.
 * Multi-ms keeps Node Math.round. Wired through tool/structural span emitters.
 * safeMonoNow: monotonic sub-ms clock for residual Date.now() same-ms tools.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { toDurationMs, safeMonoNow } from "../src/_classify";
import { TokenPoliceSpanProcessor } from "../src/telemetry";
import { setClient } from "../src/state";

const EXTRAS_ARG = 13;

describe("toDurationMs", () => {
  it("positive sub-ms → 1", () => {
    expect(toDurationMs(0.3)).toBe(1);
    expect(toDurationMs(0.001)).toBe(1);
    expect(toDurationMs(0.49)).toBe(1); // was 0 under bare Math.round
  });

  it("true-zero / negative / non-finite → 0", () => {
    expect(toDurationMs(0)).toBe(0);
    expect(toDurationMs(-1)).toBe(0);
    expect(toDurationMs(Number.NaN)).toBe(0);
    expect(toDurationMs(Number.POSITIVE_INFINITY)).toBe(0);
  });

  it("multi-ms keeps Math.round", () => {
    expect(toDurationMs(5)).toBe(5);
    expect(toDurationMs(5.2)).toBe(5);
    expect(toDurationMs(1.5)).toBe(2);
    expect(toDurationMs(1500)).toBe(1500);
  });
});

describe("_logToolSpan duration_ms", () => {
  let logged: any[];

  beforeEach(() => {
    logged = [];
    setClient({ log: (...args: any[]) => logged.push(args) } as any);
  });
  afterEach(() => setClient(null as any));

  function fakeToolSpan(start: [number, number], end: [number, number]) {
    return {
      name: "lookup_tool.tool",
      attributes: {
        "gen_ai.tool.name": "lookup_tool",
        "gen_ai.tool.type": "function",
        "tp.user_id": "u-b09",
        "tp.paid_plan": "free",
        "tp.workflow_name": "wf",
      },
      startTime: start,
      endTime: end,
      spanContext: () => ({
        traceId: "a".repeat(32),
        spanId: "b".repeat(16),
        traceFlags: 1,
      }),
      parentSpanContext: undefined,
      status: { code: 0 },
    } as any;
  }

  it("0.3ms OTel delta → duration_ms === 1", () => {
    // HrTime [s, ns]; 300_000 ns = 0.3 ms
    const span = fakeToolSpan([1, 0], [1, 300_000]);
    const attrs = span.attributes;
    (new TokenPoliceSpanProcessor() as any)._logToolSpan(
      span,
      attrs,
      "lookup_tool.tool",
    );
    expect(logged).toHaveLength(1);
    expect(logged[0][EXTRAS_ARG].call_outcome.duration_ms).toBe(1);
  });

  it("zero-delta span → duration_ms === 0", () => {
    const span = fakeToolSpan([1, 0], [1, 0]);
    (new TokenPoliceSpanProcessor() as any)._logToolSpan(
      span,
      span.attributes,
      "lookup_tool.tool",
    );
    expect(logged[0][EXTRAS_ARG].call_outcome.duration_ms).toBe(0);
  });

  it("5ms delta unchanged", () => {
    // 5_000_000 ns = 5 ms
    const span = fakeToolSpan([0, 0], [0, 5_000_000]);
    (new TokenPoliceSpanProcessor() as any)._logToolSpan(
      span,
      span.attributes,
      "lookup_tool.tool",
    );
    expect(logged[0][EXTRAS_ARG].call_outcome.duration_ms).toBe(5);
  });
});

describe("safeMonoNow", () => {
  it("returns a finite number and advances between calls", () => {
    const a = safeMonoNow();
    const b = safeMonoNow();
    expect(Number.isFinite(a)).toBe(true);
    expect(Number.isFinite(b)).toBe(true);
    // Monotonic non-decreasing; typically advances even for back-to-back reads.
    expect(b).toBeGreaterThanOrEqual(a);
  });

  it("prefers performance.now when available", () => {
    const orig = globalThis.performance;
    let calls = 0;
    Object.defineProperty(globalThis, "performance", {
      configurable: true,
      value: {
        now: () => {
          calls += 1;
          return 12.5 + calls * 0.1;
        },
      },
    });
    try {
      const a = safeMonoNow();
      const b = safeMonoNow();
      expect(a).toBe(12.6);
      expect(b).toBe(12.7);
      expect(calls).toBe(2);
      expect(toDurationMs(b - a)).toBe(1); // 0.1ms → 1
    } finally {
      Object.defineProperty(globalThis, "performance", {
        configurable: true,
        value: orig,
      });
    }
  });

  it("falls back to Date.now when performance.now is missing", () => {
    const orig = globalThis.performance;
    Object.defineProperty(globalThis, "performance", {
      configurable: true,
      value: undefined,
    });
    try {
      const n = safeMonoNow();
      expect(Number.isFinite(n)).toBe(true);
      // Date.now() is near epoch ms, not uptime-like small values
      expect(n).toBeGreaterThan(1e12);
    } finally {
      Object.defineProperty(globalThis, "performance", {
        configurable: true,
        value: orig,
      });
    }
  });
});
