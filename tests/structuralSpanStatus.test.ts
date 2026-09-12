/**
 * T3 — structural root (workflow/agent/chain) reports status='failed' when the
 * decorated body throws/rejects uncaught, instead of the server's default
 * 'success'. Two units, matching the two halves of the fix:
 *
 * A. context.ts `_runWithStructuralSpan` — on a body throw (sync) OR rejection
 * (async) it stamps the anchor span SpanStatusCode.ERROR *before* ending it
 * (Node OTel does NOT auto-set status, unlike Python), then re-throws the
 * customer's ORIGINAL error object unchanged. A body that returns / catches
 * internally leaves the span UNSET.
 * B. telemetry.ts `_logStructuralSpan` — maps that OTel status onto the emitted
 * structural row's call_outcome: code=ERROR → {status:'failed'}, else
 * {status:'success'}. Mirrors `_logToolSpan`. The scrubbed status message
 * rides along (redacted by default) and no raw text leaks.
 *
 * The golden rule (SDK never alters the customer's control flow) is pinned:
 * setStatus only annotates; the exact error object is re-thrown / re-rejected.
 */
import { describe, it, expect, afterEach } from "vitest";
import { SpanStatusCode, type Tracer } from "@opentelemetry/api";
import { session, agent, chain, workflow, _setAgentTracerFactory } from "../src/context";
import { TokenPoliceSpanProcessor } from "../src/telemetry";
import { setClient } from "../src/state";

// tp.log positional signature (see client.ts): span is args[10], extras args[13].
const SPAN_ARG = 10;
const EXTRAS_ARG = 13;

// ── A. Wrapper stamps ERROR on throw/reject, leaves UNSET otherwise ──────────

interface StatusCall {
  code: number;
  message?: string;
}
class FakeSpan {
  ended = false;
  statuses: StatusCall[] = [];
  constructor(
    public name: string,
    public attributes: Record<string, unknown>,
  ) {}
  spanContext() {
    return { traceId: "a".repeat(32), spanId: "b".repeat(16), traceFlags: 1 };
  }
  setStatus(s: StatusCall) {
    this.statuses.push(s);
  }
  end() {
    this.ended = true;
  }
}
class FakeTracer {
  spans: FakeSpan[] = [];
  startActiveSpan(name: string, options: any, fn: (span: any) => any) {
    const span = new FakeSpan(name, options?.attributes ?? {});
    this.spans.push(span);
    return fn(span);
  }
}
function register(tracer: unknown): void {
  _setAgentTracerFactory(() => tracer as Tracer);
}

afterEach(() => {
  _setAgentTracerFactory(() => undefined);
  setClient(null as any);
});

describe("structural wrapper stamps span status on failure", () => {
  it("sync throw → span.setStatus(ERROR) called before end; original error re-thrown", () => {
    const tracer = new FakeTracer();
    register(tracer);
    const bodyErr = new Error("kaboom");
    expect(() =>
      session({ name: "wf" }, () => {
        throw bodyErr;
      }),
    ).toThrow(bodyErr); // exact object identity
    const span = tracer.spans[0];
    expect(span.statuses).toHaveLength(1);
    expect(span.statuses[0].code).toBe(SpanStatusCode.ERROR);
    expect(span.ended).toBe(true);
  });

  it("async rejection → span.setStatus(ERROR); original rejection preserved", async () => {
    const tracer = new FakeTracer();
    register(tracer);
    const bodyErr = new Error("async boom");
    await expect(
      session({ name: "wf" }, async () => {
        throw bodyErr;
      }),
    ).rejects.toBe(bodyErr); // exact object identity
    const span = tracer.spans[0];
    expect(span.statuses).toHaveLength(1);
    expect(span.statuses[0].code).toBe(SpanStatusCode.ERROR);
    expect(span.ended).toBe(true);
  });

  it("chain() async rejection also stamps ERROR (async is the motivating case)", async () => {
    const tracer = new FakeTracer();
    register(tracer);
    await expect(
      chain({ name: "pipe" }, async () => {
        throw new Error("chain reject");
      }),
    ).rejects.toThrow("chain reject");
    expect(tracer.spans[0].statuses[0]?.code).toBe(SpanStatusCode.ERROR);
  });

  it("healthy body (returns) → span status left UNSET (no setStatus call)", () => {
    const tracer = new FakeTracer();
    register(tracer);
    const out = session({ name: "wf" }, () => "ok");
    expect(out).toBe("ok");
    expect(tracer.spans[0].statuses).toHaveLength(0);
    expect(tracer.spans[0].ended).toBe(true);
  });

  it("inner-caught error (body returns normally) → no ERROR stamped", () => {
    const tracer = new FakeTracer();
    register(tracer);
    session({ name: "wf" }, () => {
      try {
        throw new Error("handled internally");
      } catch {
        /* swallowed by the customer */
      }
      return "recovered";
    });
    expect(tracer.spans[0].statuses).toHaveLength(0);
  });

  it("agent() sync throw → ERROR stamped; error re-thrown", () => {
    const tracer = new FakeTracer();
    register(tracer);
    expect(() =>
      agent({ name: "a" }, () => {
        throw new Error("agent boom");
      }),
    ).toThrow("agent boom");
    expect(tracer.spans[0].statuses[0]?.code).toBe(SpanStatusCode.ERROR);
  });
});

// ── B. _logStructuralSpan maps OTel status → call_outcome ────────────────────

function fakeStructuralSpan(kind: string, status: { code: number; message?: string }) {
  return {
    name: kind === "chain" ? "pipeline" : "wf",
    spanContext: () => ({ traceId: "c".repeat(32), spanId: "d".repeat(16), traceFlags: 1 }),
    attributes: {
      "tp.kind": kind,
      "tp.workflow_name": kind === "chain" ? "pipeline" : "wf",
      "tp.user_id": "u-t3",
      "tp.paid_plan": "pro",
    },
    parentSpanContext: undefined,
    parentSpanId: undefined,
    startTime: [0, 0] as [number, number],
    endTime: [0, 5_000_000] as [number, number], // 5ms
    status,
    ended: true,
  };
}

async function captureOnEnd(span: any): Promise<any[]> {
  const captured: any[] = [];
  setClient({ log: (...args: any[]) => captured.push(args) } as any);
  new TokenPoliceSpanProcessor().onEnd(span as any);
  for (let i = 0; i < 5 && captured.length === 0; i++) {
    await new Promise((r) => setImmediate(r));
  }
  return captured;
}

describe("_logStructuralSpan maps span status → call_outcome", () => {
  it("chain root with status=ERROR → call_outcome.status='failed'", async () => {
    const captured = await captureOnEnd(
      fakeStructuralSpan("chain", { code: SpanStatusCode.ERROR, message: "voyage 429" }),
    );
    expect(captured).toHaveLength(1);
    const [span, extras] = [captured[0][SPAN_ARG], captured[0][EXTRAS_ARG]];
    expect(span.span_kind).toBe("chain");
    expect(extras.call_outcome.status).toBe("failed");
    expect(extras.call_outcome.duration_ms).toBe(5);
    // Privacy: raw status text must not ship (scrubbed → hash by default).
    expect(JSON.stringify(captured[0])).not.toContain("voyage 429");
  });

  it("agent root with status=UNSET → call_outcome.status='success'", async () => {
    const captured = await captureOnEnd(
      fakeStructuralSpan("agent", { code: SpanStatusCode.UNSET }),
    );
    expect(captured).toHaveLength(1);
    expect(captured[0][SPAN_ARG].span_kind).toBe("agent");
    expect(captured[0][EXTRAS_ARG].call_outcome.status).toBe("success");
  });

  it("status=OK → success (only ERROR flips to failed)", async () => {
    const captured = await captureOnEnd(
      fakeStructuralSpan("chain", { code: SpanStatusCode.OK }),
    );
    expect(captured[0][EXTRAS_ARG].call_outcome.status).toBe("success");
  });
});
