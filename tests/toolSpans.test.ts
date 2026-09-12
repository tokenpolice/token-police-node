/**
 * Tool-span capture tests — the manual tp.tool / tp.toolSpan API and its
 * fail-open guarantees. Mirrors the Python tests in
 * token-police-python/tests/{test_core,test_never_fail}.py.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { setClient } from "../src/state";
import { tool, toolSpan } from "../src/context";

// tp.log positional signature (see client.ts):
// (userId, paidPlan, workflowName, sessionId, model, provider,
// inputTokens, outputTokens, cachedTokens, metadata, span,
// promptComposition, responseComposition, extras)
// → span is args[10], extras is args[13].
const SPAN_ARG = 10;
const EXTRAS_ARG = 13;

describe("tool spans", () => {
  let logged: any[];
  beforeEach(() => {
    logged = [];
    setClient({ log: (...args: any[]) => logged.push(args) } as any);
  });
  afterEach(() => {
    setClient(null as any);
  });

  it("toolSpan emits a tool row with metadata only (no raw content)", () => {
    const out = toolSpan(
      { name: "web_search", callId: "call_1", args: "my secret query" },
      () => "secret result text",
    );
    expect(out).toBe("secret result text");
    expect(logged).toHaveLength(1);
    const span = logged[0][SPAN_ARG];
    const extras = logged[0][EXTRAS_ARG];
    expect(span.span_kind).toBe("tool");
    expect(span.span_name).toBe("web_search");
    expect(extras.tool.name).toBe("web_search");
    expect(extras.tool.call_id).toBe("call_1");
    expect(extras.tool.param_length).toBe("my secret query".length);
    expect(extras.tool.result_length).toBe("secret result text".length);
    expect(extras.call_outcome.status).toBe("success");
    // Privacy: raw args/results never leave — only hash + length.
    const blob = JSON.stringify(logged[0]);
    expect(blob).not.toContain("my secret query");
    expect(blob).not.toContain("secret result text");
  });

  it("tool() wrapper records failure and re-throws the original error", () => {
    const boom = tool({ name: "boom" }, () => {
      throw new Error("kaboom");
    });
    expect(() => boom()).toThrow("kaboom");
    expect(logged).toHaveLength(1);
    expect(logged[0][EXTRAS_ARG].call_outcome.status).toBe("failed");
  });

  it("toolSpan awaits an async fn and records a tool row", async () => {
    const out = await toolSpan({ name: "fetch" }, async () => 42);
    expect(out).toBe(42);
    expect(logged).toHaveLength(1);
    expect(logged[0][SPAN_ARG].span_kind).toBe("tool");
    expect(logged[0][EXTRAS_ARG].call_outcome.status).toBe("success");
  });

  // Residual: Date.now() same-ms tools used to report duration_ms=0.
  // Mono measurement + toDurationMs must report >= 1 for real (incl. sub-ms) work.
  it("toolSpan no-op → duration_ms >= 1", () => {
    toolSpan({ name: "noop" }, () => undefined);
    expect(logged).toHaveLength(1);
    expect(logged[0][EXTRAS_ARG].call_outcome.duration_ms).toBeGreaterThanOrEqual(1);
  });

  // Multi-ms work must report a proportionate duration, not the sub-ms floor of 1.
  // Deliberately NOT `await setTimeout(r, 5)`: libuv derives a timer's deadline from
  // the event loop's *cached* clock, so a 5ms timer can resolve a millisecond or more
  // before performance.now() — the clock toolSpan measures on — has advanced 5ms. The
  // delta then rounds to 4 and the floor below fails (seen in the nightly
  // provider-drift run, node/core cell). Burning the time on the measured clock makes
  // the elapsed span deterministic; overhead only ever adds to it.
  it("toolSpan ~5ms of work → duration_ms in [5, 50]", async () => {
    await toolSpan({ name: "sleeper" }, async () => {
      const spinStart = performance.now();
      while (performance.now() - spinStart < 6) {
        // busy-wait
      }
    });
    const d = logged[0][EXTRAS_ARG].call_outcome.duration_ms;
    expect(d).toBeGreaterThanOrEqual(5);
    expect(d).toBeLessThanOrEqual(50);
  });

  it("toolSpan failure path rethrows and still measures duration", () => {
    expect(() =>
      toolSpan({ name: "boom" }, () => {
        throw new Error("kaboom");
      }),
    ).toThrow("kaboom");
    expect(logged).toHaveLength(1);
    expect(logged[0][EXTRAS_ARG].call_outcome.status).toBe("failed");
    expect(logged[0][EXTRAS_ARG].call_outcome.duration_ms).toBeGreaterThanOrEqual(1);
  });

  it("toolSpan async reject rethrows and still measures duration", async () => {
    await expect(
      toolSpan({ name: "async_boom" }, async () => {
        throw new Error("async-kaboom");
      }),
    ).rejects.toThrow("async-kaboom");
    expect(logged).toHaveLength(1);
    expect(logged[0][EXTRAS_ARG].call_outcome.status).toBe("failed");
    expect(logged[0][EXTRAS_ARG].call_outcome.duration_ms).toBeGreaterThanOrEqual(1);
  });
});

describe("tool capture fail-open", () => {
  it("toolSpan never throws when the client log throws", () => {
    setClient({ log: () => { throw new Error("boom"); } } as any);
    expect(() => toolSpan({ name: "t", args: "x" }, () => "ok")).not.toThrow();
    setClient(null as any);
  });

  it("toolSpan never throws when no client is configured", () => {
    setClient(null as any);
    expect(() => toolSpan({ name: "t" }, () => "ok")).not.toThrow();
  });

  it("tool() wrapper propagates the user's original error unchanged", () => {
    setClient(null as any);
    const explode = tool({ name: "x" }, () => {
      throw new TypeError("user-error");
    });
    expect(() => explode()).toThrow(TypeError);
  });
});
