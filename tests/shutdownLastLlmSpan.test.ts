/**
 * Regression: last auto-instrumented LLM span dropped at tp.shutdown().
 *
 * Root cause: LLM onEnd schedules client.log on process.nextTick; close() used
 * to seal _closed first and only await _pendingPromises, so the deferred log
 * no-op'd. Fix: track deferred work; flush/close drain it before sealing.
 *
 * Design: node_sdk_shutdown_last_llm_span_design.md
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { TokenPolice } from "../src/client";
import { setClient } from "../src/state";
import { TokenPoliceSpanProcessor } from "../src/telemetry";

interface Recorded {
  url: string;
  options: RequestInit;
  body: any;
}

function stubFetch(recorded: Recorded[]) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, options: RequestInit) => {
      let body: any = null;
      try {
        body = options?.body ? JSON.parse(String(options.body)) : null;
      } catch {
        body = options?.body;
      }
      recorded.push({ url: String(url), options, body });
      return new Response(JSON.stringify({ allowed: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }),
  );
}

function makeClient(): TokenPolice {
  return new TokenPolice({
    apiKey: "tp_sk_test_shutdown_last_llm",
    baseUrl: "http://localhost:13099",
    timeout: 0.5,
    deployment: "daemon",
  });
}

/** Synthetic finished LLM span with nonzero usage (passes usage gate). */
function fakeLlmSpan(opts?: {
  input?: number;
  output?: number;
  spanId?: string;
  model?: string;
}) {
  const input = opts?.input ?? 10;
  const output = opts?.output ?? 5;
  const spanId = opts?.spanId ?? "0000000000001234";
  return {
    attributes: {
      "gen_ai.system": "openai",
      "gen_ai.request.model": opts?.model ?? "gpt-4o",
      "gen_ai.usage.input_tokens": input,
      "gen_ai.usage.output_tokens": output,
      "tp.user_id": "u1",
      "tp.paid_plan": "free",
      "tp.workflow_name": "test_wf",
      "tp.session_id": "sess1",
    },
    name: "chat gpt-4o",
    startTime: [0, 0] as [number, number],
    endTime: [1, 0] as [number, number],
    spanContext: () => ({
      traceId: "0000000000000000000000000000abcd",
      spanId,
    }),
    parentSpanId: undefined,
  } as any;
}

/** Structural agent anchor — logs immediately (not via nextTick). */
function fakeAgentSpan() {
  return {
    attributes: {
      "tp.kind": "agent",
      "tp.user_id": "u1",
      "tp.paid_plan": "free",
      "tp.workflow_name": "test_wf",
      "tp.session_id": "sess1",
    },
    name: "test_wf",
    startTime: [0, 0] as [number, number],
    endTime: [1, 0] as [number, number],
    spanContext: () => ({
      traceId: "0000000000000000000000000000abcd",
      spanId: "000000000000aaaa",
    }),
    parentSpanId: undefined,
  } as any;
}

function logRecords(recorded: Recorded[]): Recorded[] {
  return recorded.filter((r) => r.url.includes("/v1/guard/log"));
}

function llmLogRecords(recorded: Recorded[]): Recorded[] {
  return logRecords(recorded).filter((r) => {
    const kind = r.body?.span?.span_kind;
    // LLM rows have a model name; structural agent rows use empty model
    return r.body?.model?.name && kind !== "agent" && kind !== "chain" && kind !== "tool";
  });
}

beforeEach(() => {
  // Clear any prior singleton so setClient does not closeSync a previous client mid-suite
  setClient(null as any);
});

afterEach(async () => {
  setClient(null as any);
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("shutdown last LLM span (nextTick deferred drain)", () => {
  it("3.2 primary: onEnd then immediate shutdown → exactly one LLM log (no extra tick)", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    setClient(client);
    const proc = new TokenPoliceSpanProcessor();

    proc.onEnd(fakeLlmSpan());
    // CRITICAL: no await setTimeout(0) / setImmediate — race the real bug
    await client.shutdown();

    const llm = llmLogRecords(recorded);
    expect(llm.length).toBe(1);
    expect(llm[0].body.model.name).toBe("gpt-4o");
  });

  it("3.3 flush alone (no close): same immediate sequence → log; client stays open", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    setClient(client);
    const proc = new TokenPoliceSpanProcessor();

    proc.onEnd(fakeLlmSpan({ spanId: "0000000000000001" }));
    await client.flush();

    expect(llmLogRecords(recorded).length).toBe(1);

    // Still open: direct log still issues a POST
    const before = recorded.length;
    client.log("u", "free", "default", "", "gpt-4o-mini", "openai", 1, 1, 0);
    await client.flush();
    expect(logRecords(recorded).length).toBeGreaterThan(
      logRecords(recorded.slice(0, before)).length,
    );
    // Cleanup
    await client.close();
  });

  it("3.4 agent structural path logs immediately without nextTick wait", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    setClient(client);
    const proc = new TokenPoliceSpanProcessor();

    proc.onEnd(fakeAgentSpan());
    // No tick — structural log is sync in onEnd
    const logs = logRecords(recorded);
    expect(logs.length).toBe(1);
    expect(logs[0].body?.span?.span_kind).toBe("agent");

    await client.close();
  });

  it("3.5 double shutdown / close is idempotent and never throws", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    setClient(client);
    const proc = new TokenPoliceSpanProcessor();

    proc.onEnd(fakeLlmSpan());
    await expect(client.shutdown()).resolves.toBeUndefined();
    await expect(client.shutdown()).resolves.toBeUndefined();
    await expect(client.close()).resolves.toBeUndefined();
    expect(llmLogRecords(recorded).length).toBe(1);
  });

  it("3.6 after successful close, new direct log is a no-op", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    setClient(client);

    await client.close();
    const n = recorded.length;
    client.log("u", "free", "default", "", "gpt-4o", "openai", 10, 5, 0);
    // Give any accidental async a chance
    await new Promise((r) => setTimeout(r, 20));
    expect(logRecords(recorded).length).toBe(logRecords(recorded.slice(0, n)).length);
    expect(recorded.length).toBe(n);
  });

  it("3.7 two LLM onEnds in one turn → both land after one flush", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    setClient(client);
    const proc = new TokenPoliceSpanProcessor();

    proc.onEnd(fakeLlmSpan({ spanId: "0000000000000001", model: "gpt-4o" }));
    proc.onEnd(fakeLlmSpan({ spanId: "0000000000000002", model: "gpt-4o-mini" }));
    await client.flush();

    const llm = llmLogRecords(recorded);
    expect(llm.length).toBe(2);
    const names = llm.map((r) => r.body.model.name).sort();
    expect(names).toEqual(["gpt-4o", "gpt-4o-mini"]);
    await client.close();
  });

  it("C10 piggyback: two processor instances share module deferred set", async () => {
    // setupOpenTelemetry may attach a second TokenPoliceSpanProcessor on a
    // customer TracerProvider; deferred work must be module-level, not per-instance.
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    setClient(client);
    const privateProc = new TokenPoliceSpanProcessor();
    const piggybackProc = new TokenPoliceSpanProcessor();

    privateProc.onEnd(fakeLlmSpan({ spanId: "00000000000000a1", model: "gpt-4o" }));
    piggybackProc.onEnd(fakeLlmSpan({ spanId: "00000000000000a2", model: "gpt-4o-mini" }));
    await client.flush();

    const llm = llmLogRecords(recorded);
    expect(llm.length).toBe(2);
    const names = llm.map((r) => r.body.model.name).sort();
    expect(names).toEqual(["gpt-4o", "gpt-4o-mini"]);
    await client.close();
  });

  it("3.8 zero-token LLM span: deferred settles (flush does not hang); no log", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    setClient(client);
    const proc = new TokenPoliceSpanProcessor();

    proc.onEnd(fakeLlmSpan({ input: 0, output: 0, spanId: "00000000000000zz" }));

    const flushOrTimeout = Promise.race([
      client.flush().then(() => "flushed" as const),
      new Promise<"timeout">((r) => setTimeout(() => r("timeout"), 2000)),
    ]);
    await expect(flushOrTimeout).resolves.toBe("flushed");
    expect(llmLogRecords(recorded).length).toBe(0);
    await client.close();
  });
});
