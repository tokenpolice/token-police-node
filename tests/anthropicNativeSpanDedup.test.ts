/**
 * @anthropic-ai/sdk >=0.134.0 ships its own OpenTelemetry tracer, on by
 * default: every API call, stream and tool run opens a gen_ai.* span on scope
 * "com.anthropic.sdk.typescript" (plus ".<platform>" for Bedrock/Vertex/
 * Foundry) once a tracer provider is registered. Ours always is, so before the
 * fix each anthropic call logged two rows — the instrumentor's and the native
 * one — at exactly 2x cost (provider-drift run 2026-10-15).
 *
 * These drive TokenPoliceSpanProcessor directly with synthetic spans, so they
 * pin the drop whatever @anthropic-ai/sdk version is installed. The real-SDK
 * round trip lives in realProviderSeams.anthropic.test.ts.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { setClient } from "../src/state";
import { TokenPoliceSpanProcessor } from "../src/telemetry";

const SPAN_ARG = 10; // tp.log(...) positional index of the `span` object

let logged: any[][];

beforeEach(() => {
  logged = [];
  setClient({ firewall: "off", log: (...args: any[]) => logged.push(args) } as any);
});

afterEach(() => {
  setClient(null as any);
});

const flush = async () => {
  for (let i = 0; i < 5; i++) await new Promise((r) => setImmediate(r));
};

let _seq = 0;
function makeSpan(opts: {
  scopeName: string;
  attrs?: Record<string, unknown>;
  name?: string;
  parentSpanId?: string;
}): any {
  const spanId = `abcd${(_seq++).toString(16).padStart(12, "0")}`;
  const s: any = {
    name: opts.name ?? "anthropic.chat",
    attributes: { ...(opts.attrs ?? {}) },
    instrumentationScope: { name: opts.scopeName },
    parentSpanId: opts.parentSpanId,
    startTime: [1000, 0],
    endTime: [1001, 0],
    status: { code: 0 },
    setAttribute(k: string, v: any) {
      s.attributes[k] = v;
      return s;
    },
    spanContext() {
      return { traceId: "a".repeat(32), spanId, traceFlags: 1 };
    },
  };
  return s;
}

function usageAttrs(): Record<string, unknown> {
  return {
    "gen_ai.provider.name": "anthropic",
    "gen_ai.operation.name": "chat",
    "gen_ai.request.model": "claude-sonnet-4-5",
    "gen_ai.usage.input_tokens": 11,
    "gen_ai.usage.output_tokens": 7,
  };
}

function run(span: any): void {
  const proc = new TokenPoliceSpanProcessor();
  proc.onStart(span);
  proc.onEnd(span);
}

describe("anthropic native OTel span is not logged", () => {
  it("native scope LLM span → no row", async () => {
    run(makeSpan({ scopeName: "com.anthropic.sdk.typescript", attrs: usageAttrs() }));
    await flush();
    expect(logged.length).toBe(0);
  });

  it("platform-suffixed native scope (Bedrock client) → no row", async () => {
    run(makeSpan({ scopeName: "com.anthropic.sdk.typescript.bedrock", attrs: usageAttrs() }));
    await flush();
    expect(logged.length).toBe(0);
  });

  it("native tool-run span → no tool row", async () => {
    run(
      makeSpan({
        scopeName: "com.anthropic.sdk.typescript",
        name: "execute_tool get_weather",
        attrs: { "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "get_weather" },
      }),
    );
    await flush();
    expect(logged.length).toBe(0);
  });

  it("onEnd alone (no onStart) still drops it", async () => {
    const proc = new TokenPoliceSpanProcessor();
    proc.onEnd(makeSpan({ scopeName: "com.anthropic.sdk.typescript", attrs: usageAttrs() }));
    await flush();
    expect(logged.length).toBe(0);
  });

  it("control: same span on the instrumentor's scope → exactly one row", async () => {
    run(makeSpan({ scopeName: "@traceloop/instrumentation-anthropic", attrs: usageAttrs() }));
    await flush();
    expect(logged.filter((r) => r[SPAN_ARG]?.span_kind === "llm").length).toBe(1);
  });

  it("an instrumentor span nested under a native span parents onto the native span's parent", async () => {
    const proc = new TokenPoliceSpanProcessor();
    const outer = "f".repeat(16);
    const native = makeSpan({
      scopeName: "com.anthropic.sdk.typescript",
      attrs: usageAttrs(),
      parentSpanId: outer,
    });
    proc.onStart(native);
    const inner = makeSpan({
      scopeName: "@traceloop/instrumentation-anthropic",
      attrs: usageAttrs(),
      parentSpanId: native.spanContext().spanId,
    });
    proc.onStart(inner);
    proc.onEnd(inner);
    proc.onEnd(native);
    await flush();

    const rows = logged.filter((r) => r[SPAN_ARG]?.span_kind === "llm");
    expect(rows.length).toBe(1);
    expect(rows[0][SPAN_ARG].parent_span_id).toBe(outer);
  });
});
