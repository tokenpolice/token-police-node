/**
 * I1 — phantom parent / missing anchor for unscoped calls.
 *
 * Unscoped getCurrentSession() mints a throwaway TPSession with a random
 * rootSpanId that is never logged as an agent/chain row. Manual, Mode A,
 * and tools must parent with "" in that case — not the throwaway root.
 * Scoped sessions set session._anchored when the structural span binds real
 * OTel ids; post-scope finalize holding that object must still parent to root.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import {
  TPSession,
  getCurrentSession,
  manualSpanIds,
  sessionParentSpanId,
  session,
  _getSessionStorage,
} from "../src/context";
import { init, getClient } from "../src/index";
import { setClient } from "../src/state";
import { TokenPoliceSpanProcessor } from "../src/telemetry";

describe("sessionParentSpanId / manualSpanIds unscoped", () => {
  it("throwaway session → parent \"\"", () => {
    const s = new TPSession();
    expect(s._anchored).toBe(false);
    expect(sessionParentSpanId(s)).toBe("");
    expect(manualSpanIds(s).parent_span_id).toBe("");
  });

  it("getCurrentSession() outside scope → parent \"\"", () => {
    const s = getCurrentSession();
    expect(s._anchored).toBe(false);
    expect(manualSpanIds(s).parent_span_id).toBe("");
  });

  it("anchored session without live OTel → parent = root", () => {
    const s = new TPSession({ rootSpanId: "c".repeat(16) });
    s._anchored = true;
    expect(sessionParentSpanId(s)).toBe("c".repeat(16));
    expect(manualSpanIds(s).parent_span_id).toBe("c".repeat(16));
  });
});

describe("post-scope held session still parents", () => {
  beforeEach(() => {
    init({ apiKey: "tp_sk_test_i1", baseUrl: "http://localhost:59999" });
  });
  afterEach(() => {
    const c = getClient();
    if (c) c.closeSync();
  });

  it("held session after context exit keeps _anchored parent", () => {
    let held: TPSession | null = null;
    let root = "";
    session({ name: "post_scope_probe" }, (s) => {
      held = s;
      root = s.rootSpanId;
      expect(s._anchored).toBe(true);
    });
    // ALS store cleared after exit.
    expect(_getSessionStorage().getStore()).toBeUndefined();
    expect(held).not.toBeNull();
    expect(held!._anchored).toBe(true);
    expect(manualSpanIds(held!).parent_span_id).toBe(root);
    expect(sessionParentSpanId(held!)).toBe(root);
  });

  it("scoped session sets _anchored and manual parents to agent", () => {
    session({ name: "anchor_probe" }, (s) => {
      expect(s._anchored).toBe(true);
      expect(s.rootSpanId).toBeTruthy();
      const ids = manualSpanIds(s);
      // Active OTel structural span is preferred parent.
      expect(ids.parent_span_id).toBe(s.rootSpanId);
    });
  });
});

describe("Mode A onStart / onEnd parent", () => {
  it("onStart outside session stamps tp.root_span_id = \"\"", () => {
    const stamped: Record<string, unknown> = {};
    const span: any = {
      attributes: {},
      name: "ChatOpenAI.chat",
      spanContext: () => ({
        traceId: "a".repeat(32),
        spanId: "b".repeat(16),
        traceFlags: 1,
      }),
      setAttribute(k: string, v: unknown) {
        stamped[k] = v;
      },
      instrumentationScope: { name: "opentelemetry.instrumentation.openai" },
    };
    const proc = new TokenPoliceSpanProcessor();
    proc.onStart(span as any, {} as any);
    expect(stamped["tp.root_span_id"]).toBe("");
  });

  it("onStart inside session stamps real root", () => {
    init({ apiKey: "tp_sk_test_i1_scoped_start", baseUrl: "http://localhost:59999" });
    try {
      session({ name: "scoped_stamp" }, (s) => {
        expect(s._anchored).toBe(true);
        const stamped: Record<string, unknown> = {};
        const span: any = {
          attributes: {},
          name: "ChatOpenAI.chat",
          spanContext: () => ({
            traceId: "a".repeat(32),
            spanId: "b".repeat(16),
            traceFlags: 1,
          }),
          setAttribute(k: string, v: unknown) {
            stamped[k] = v;
          },
          instrumentationScope: { name: "opentelemetry.instrumentation.openai" },
        };
        const proc = new TokenPoliceSpanProcessor();
        proc.onStart(span as any, {} as any);
        expect(stamped["tp.root_span_id"]).toBe(s.rootSpanId);
        expect(stamped["tp.root_span_id"]).toBeTruthy();
      });
    } finally {
      const c = getClient();
      if (c) c.closeSync();
    }
  });

  it("onEnd with empty stamped root + no OTel parent → parent_span_id \"\"", async () => {
    const captured: any[] = [];
    const fakeClient: any = {
      log: (...args: any[]) => {
        captured.push(args);
      },
    };
    setClient(fakeClient);

    const span: any = {
      name: "ChatOpenAI.chat",
      spanContext: () => ({
        traceId: "1".repeat(32),
        spanId: "2".repeat(16),
        traceFlags: 1,
      }),
      attributes: {
        "gen_ai.system": "openai",
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.usage.input_tokens": 5,
        "gen_ai.usage.output_tokens": 3,
        "tp.root_span_id": "",
        "tp.trace_id": "a".repeat(32),
        "tp.user_id": "anonymous",
        "tp.workflow_name": "default_workflow",
      },
      parentSpanContext: undefined,
      parentSpanId: undefined,
      instrumentationScope: { name: "opentelemetry.instrumentation.openai" },
      startTime: [0, 0],
      endTime: [0, 1],
      status: { code: 0 },
      ended: true,
    };

    const proc = new TokenPoliceSpanProcessor();
    proc.onEnd(span as any);

    // onEnd may defer via nextTick — wait a few ticks.
    for (let i = 0; i < 5; i++) {
      await new Promise((r) => setImmediate(r));
      if (captured.length > 0) break;
    }

    expect(captured.length).toBeGreaterThan(0);
    let parent: string | undefined;
    for (const call of captured) {
      for (const arg of call) {
        if (arg && typeof arg === "object" && "parent_span_id" in arg) {
          parent = (arg as { parent_span_id: string }).parent_span_id;
        }
      }
    }
    expect(parent).toBe("");
  });
});
