/**
 * Concurrent same-session span attribution.
 *
 * The instrumented wrapper reserves each call's span order BEFORE the provider
 * call and telemetry `onStart` consumes that reservation — so two concurrent
 * calls in one `tp.session()` can never cross-attribute their prompt/response/
 * service-tier stashes (which were previously keyed off a mutable, shared span
 * counter peeked/decremented at pre- and post-call time).
 *
 * These tests drive the REAL generic async wrapper (installed via `protect`
 * with an in-memory fake client) and the REAL TokenPoliceSpanProcessor, so a
 * regression in either the reservation plumbing or the order threading fails
 * here. The span lifecycle is driven manually (processor.onStart / onEnd) so
 * the concurrency window is deterministic — no real OTel provider needed.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { protect } from "../src/enforcer";
import { setClient } from "../src/state";
import {
  session as tpSession,
  TPSession,
  _getSessionStorage,
  runWithReservedSpanOrder,
} from "../src/context";
import { TokenPoliceSpanProcessor } from "../src/telemetry";

// Positional indices into the client.log() argument list (see telemetry.ts).
const PROMPT_ARG = 11;
const RESPONSE_ARG = 12;
const EXTRAS_ARG = 13;

const flush = () => new Promise((r) => setTimeout(r, 0));

function deferred<T = void>() {
  let resolve!: (v: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

let logged: any[][];
let spanIdCtr = 0;
let moduleCtr = 0;

/** A minimal span the processor can both onStart (setAttribute) and onEnd. */
function makeSpan(): any {
  const spanId = (++spanIdCtr).toString(16).padStart(16, "0");
  const attributes: Record<string, any> = {};
  return {
    name: "openai.chat",
    attributes,
    setAttribute(k: string, v: any) {
      attributes[k] = v;
      return this;
    },
    spanContext: () => ({ traceId: "1".repeat(32), spanId }),
    parentSpanId: undefined,
    startTime: [0, 0] as [number, number],
    endTime: [1, 0] as [number, number],
    status: { code: 0 },
  };
}

/** An OpenAI-chat-shaped response carrying a distinct completion + tier. */
function openaiResponse(content: string, tier: string): any {
  return {
    choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }],
    usage: { prompt_tokens: 10, completion_tokens: 5 },
    service_tier: tier,
  };
}

beforeEach(() => {
  logged = [];
  // firewall:"off" makes the wrapper's pre-flight a no-op; log() captures rows.
  setClient({ firewall: "off", log: (...a: any[]) => logged.push(a) } as any);
});
afterEach(() => {
  setClient(null as any);
});

/** (promptLen: responseLen: tier) tuple for one logged row. */
function tupleOf(row: any[]): string {
  const prompt = row[PROMPT_ARG] as any[];
  const response = row[RESPONSE_ARG] as any[];
  const tier = row[EXTRAS_ARG]?.usage?.tier ?? "";
  const pLen = prompt?.[0]?.length ?? -1;
  const rLen = response?.[0]?.length ?? -1;
  return `${pLen}:${rLen}:${tier}`;
}

describe("concurrent same-session attribution", () => {
  it("two concurrent calls keep prompt+response+tier on their own span (no crossing, no loss)", async () => {
    const proc = new TokenPoliceSpanProcessor();
    const spans: any[] = [];
    const startGate = deferred();
    const proceedGate = deferred();
    let started = 0;

    class FakeClient {
      async create(this: any, body: any): Promise<any> {
        await startGate.promise;
        // Open the span (processor.onStart consumes this call's reservation).
        const span = makeSpan();
        spans.push(span);
        proc.onStart(span);
        span.attributes["gen_ai.system"] = "openai";
        span.attributes["gen_ai.request.model"] = body.model;
        span.attributes["gen_ai.usage.input_tokens"] = 10;
        span.attributes["gen_ai.usage.output_tokens"] = 5;
        // Hold until BOTH calls have started their span — this is the exact
        // interleaving that made the old peek/decrement logic cross-attribute.
        started += 1;
        if (started >= 2) proceedGate.resolve();
        await proceedGate.promise;
        return body.__resp;
      }
    }

    protect(`fake-openai-${moduleCtr++}`, ["prototype"], "create", true, {
      provider: "openai",
      module: FakeClient,
    });
    const client = new FakeClient();

    await tpSession({ name: "wf", userId: "u1" }, async () => {
      const pA = client.create({
        model: "gpt-a",
        messages: [{ role: "user", content: "AAAA" }], // prompt len 4
        __resp: openaiResponse("aaaaaaaaaa", "flex"), // response len 10
      });
      const pB = client.create({
        model: "gpt-b",
        messages: [{ role: "user", content: "BBBBBBB" }], // prompt len 7
        __resp: openaiResponse("bbb", "priority"), // response len 3
      });
      // Let both wrappers reach the (parked) provider call, then release.
      await flush();
      startGate.resolve();
      await Promise.all([pA, pB]);

      // End the spans (inside the session so the deferred nextTick log reads
      // this session's composition map), then drain the deferred logs.
      for (const s of spans) proc.onEnd(s);
      await flush();
    });

    expect(logged.length).toBe(2);
    const tuples = logged.map(tupleOf).sort();
    // Each row is internally coherent: A=(4,10,flex), B=(7,3,priority).
    expect(tuples).toEqual(["4:10:flex", "7:3:priority"]);
  });

  it("serial calls attribute correctly with contiguous orders (no off-by-one)", async () => {
    const proc = new TokenPoliceSpanProcessor();
    const spans: any[] = [];

    class FakeClient {
      async create(this: any, body: any): Promise<any> {
        const span = makeSpan();
        spans.push(span);
        proc.onStart(span);
        span.attributes["gen_ai.system"] = "openai";
        span.attributes["gen_ai.request.model"] = body.model;
        span.attributes["gen_ai.usage.input_tokens"] = 10;
        span.attributes["gen_ai.usage.output_tokens"] = 5;
        return body.__resp;
      }
    }

    protect(`fake-openai-${moduleCtr++}`, ["prototype"], "create", true, {
      provider: "openai",
      module: FakeClient,
    });
    const client = new FakeClient();

    await tpSession({ name: "wf", userId: "u1" }, async () => {
      // First call end-to-end (await returns AFTER post-call stash), then log.
      await client.create({
        model: "gpt-a",
        messages: [{ role: "user", content: "AAAA" }],
        __resp: openaiResponse("aaaaaaaaaa", "flex"),
      });
      proc.onEnd(spans[0]);
      await flush();

      // Second call — must land on its OWN (next) span order, no off-by-one.
      await client.create({
        model: "gpt-b",
        messages: [{ role: "user", content: "BBBBBBB" }],
        __resp: openaiResponse("bbb", "priority"),
      });
      proc.onEnd(spans[1]);
      await flush();
    });

    // Contiguous, monotonic orders assigned to the two spans.
    expect(spans.map((s) => s.attributes["tp.span_order"])).toEqual([0, 1]);
    expect(logged.length).toBe(2);
    const tuples = logged.map(tupleOf).sort();
    expect(tuples).toEqual(["4:10:flex", "7:3:priority"]);
  });

  it("a span started OUTSIDE any wrapped call allocates fresh (no stale reservation consumed)", () => {
    const proc = new TokenPoliceSpanProcessor();
    const session = new TPSession({
      userId: "u",
      paidPlan: "p",
      workflowName: "w",
      traceId: "a".repeat(32),
      rootSpanId: "b".repeat(16),
    });

    _getSessionStorage().run(session, () => {
      // A reservation scope that exits WITHOUT a span consuming it must not leak.
      runWithReservedSpanOrder({ order: 99, consumed: false }, () => {
        /* no span opened here */
      });

      // Bare spans (no active reservation) → fresh, contiguous nextSpanOrder().
      const s0 = makeSpan();
      proc.onStart(s0);
      expect(s0.attributes["tp.span_order"]).toBe(0);
      const s1 = makeSpan();
      proc.onStart(s1);
      expect(s1.attributes["tp.span_order"]).toBe(1);

      // A span opened INSIDE a reservation consumes it verbatim...
      const sr = makeSpan();
      runWithReservedSpanOrder({ order: 77, consumed: false }, () => proc.onStart(sr));
      expect(sr.attributes["tp.span_order"]).toBe(77);

      // ...and consuming a reservation does NOT advance the counter, so the next
      // bare span continues from where fresh allocation left off.
      const s2 = makeSpan();
      proc.onStart(s2);
      expect(s2.attributes["tp.span_order"]).toBe(2);
    });
  });
});
