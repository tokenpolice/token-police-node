/**
 * Structural-span fail-open — opening a session must never throw customer-side.
 *
 * `session()`/`agent()`/`chain()`/`workflow()` open a structural anchor span
 * around the customer's callback. Span setup is best-effort telemetry: a
 * failure while building span attributes (e.g. hostile/exotic metadata whose
 * enumeration throws) or a tracer whose span-open call itself throws must
 * degrade to running the body untraced — the same documented behavior as
 * "telemetry not set up" — with the body invoked exactly once and its
 * return/throw forwarded unchanged. Only the body's own errors may propagate.
 *
 * These tests pin:
 * 1. hostile metadata (throwing ownKeys/getOwnPropertyDescriptor proxy
 * traps) → body runs once, value forwarded, no throw;
 * 2. a tracer whose startActiveSpan throws before invoking its callback →
 * body runs exactly once, value forwarded;
 * 3. a tracer whose startActiveSpan throws AFTER its callback ran → the
 * body is NOT re-run and its result is still forwarded;
 * 4. the healthy path is unchanged (span recorded, tp.* attributes set,
 * session ids adopted from the span, result forwarded);
 * 5. async bodies resolve and end the span on both healthy and degraded
 * paths;
 * 6. the body's own throw propagates unchanged on every path.
 */
import { describe, it, expect, afterEach } from "vitest";
import type { Tracer } from "@opentelemetry/api";
import {
  session,
  chain,
  getCurrentSession,
  _setAgentTracerFactory,
} from "../src/context";

// ── Fakes ──────────────────────────────────────────────────────────────

class FakeSpan {
  ended = false;
  constructor(
    public name: string,
    public attributes: Record<string, unknown>,
  ) {}
  spanContext() {
    return {
      traceId: "a".repeat(32),
      spanId: "b".repeat(16),
      traceFlags: 1,
    };
  }
  end() {
    this.ended = true;
  }
}

/** Well-behaved tracer: runs the callback synchronously (the OTel contract). */
class FakeTracer {
  spans: FakeSpan[] = [];
  startActiveSpan(name: string, options: any, fn: (span: any) => any) {
    const span = new FakeSpan(name, options?.attributes ?? {});
    this.spans.push(span);
    return fn(span);
  }
}

/** Tracer whose span-open throws BEFORE ever invoking the callback. */
class ThrowingTracer {
  startActiveSpan(): never {
    throw new Error("tracer down");
  }
}

/** Tracer that runs the callback, then throws on its own way out. */
class ThrowAfterCallbackTracer {
  spans: FakeSpan[] = [];
  startActiveSpan(name: string, options: any, fn: (span: any) => any) {
    const span = new FakeSpan(name, options?.attributes ?? {});
    this.spans.push(span);
    fn(span);
    throw new Error("tracer exploded after callback");
  }
}

function register(tracer: unknown): void {
  _setAgentTracerFactory(() => tracer as Tracer);
}

function hostileMetadata(): Record<string, unknown> {
  return new Proxy(
    {},
    {
      ownKeys() {
        throw new Error("hostile ownKeys trap");
      },
      getOwnPropertyDescriptor() {
        throw new Error("hostile getOwnPropertyDescriptor trap");
      },
    },
  ) as Record<string, unknown>;
}

afterEach(() => {
  // Unregister the fake tracer so other suites see the default (no tracer).
  _setAgentTracerFactory(() => undefined);
});

// ── 1. Hostile metadata degrades to running the body untraced ──────────

describe("hostile metadata (attribute build throws)", () => {
  it("session(): body runs once and its value is forwarded, no throw", () => {
    register(new FakeTracer());
    let ran = 0;
    const out = session({ name: "wf", metadata: hostileMetadata() }, () => {
      ran++;
      return "ok";
    });
    expect(out).toBe("ok");
    expect(ran).toBe(1);
  });

  it("session(): the session context is still established for the body", () => {
    register(new FakeTracer());
    session(
      { name: "wf_hostile", userId: "u1", metadata: hostileMetadata() },
      () => {
        const s = getCurrentSession();
        expect(s.workflowName).toBe("wf_hostile");
        expect(s.userId).toBe("u1");
      },
    );
  });

  it("session(): the body's own throw propagates unchanged", () => {
    register(new FakeTracer());
    const bodyErr = new Error("customer body error");
    expect(() =>
      session({ name: "wf", metadata: hostileMetadata() }, () => {
        throw bodyErr;
      }),
    ).toThrow(bodyErr);
  });
});

// ── 2. Tracer that throws before invoking its callback ─────────────────

describe("throwing tracer (startActiveSpan throws pre-callback)", () => {
  it("session(): body runs exactly once, value forwarded", () => {
    register(new ThrowingTracer());
    let ran = 0;
    const out = session({ name: "wf" }, () => {
      ran++;
      return 123;
    });
    expect(out).toBe(123);
    expect(ran).toBe(1);
  });

  it("chain(): body runs exactly once, value forwarded", () => {
    register(new ThrowingTracer());
    let ran = 0;
    const out = chain({ name: "pipeline" }, () => {
      ran++;
      return "chained";
    });
    expect(out).toBe("chained");
    expect(ran).toBe(1);
  });

  it("session(): the body's own throw propagates unchanged", () => {
    register(new ThrowingTracer());
    const bodyErr = new Error("customer body error");
    let ran = 0;
    expect(() =>
      session({ name: "wf" }, () => {
        ran++;
        throw bodyErr;
      }),
    ).toThrow(bodyErr);
    expect(ran).toBe(1);
  });
});

// ── 3. Tracer that throws AFTER running its callback ───────────────────

describe("tracer throws after its callback already ran", () => {
  it("session(): body is NOT re-run; result still forwarded; no throw", () => {
    register(new ThrowAfterCallbackTracer());
    let ran = 0;
    const out = session({ name: "wf" }, () => {
      ran++;
      return "once";
    });
    expect(out).toBe("once");
    expect(ran).toBe(1);
  });

  it("session(): a genuine body throw still propagates (not swallowed)", () => {
    register(new ThrowAfterCallbackTracer());
    const bodyErr = new Error("customer body error");
    let ran = 0;
    expect(() =>
      session({ name: "wf" }, () => {
        ran++;
        throw bodyErr;
      }),
    ).toThrow(bodyErr);
    expect(ran).toBe(1);
  });
});

// ── 4. Healthy path pinned (no regression) ─────────────────────────────

describe("healthy tracer path unchanged", () => {
  it("records the span with tp.* attributes and forwards the result", () => {
    const tracer = new FakeTracer();
    register(tracer);
    const out = session(
      { name: "healthy_wf", userId: "u9", paidPlan: "pro", metadata: { k: "v" } },
      () => "result",
    );
    expect(out).toBe("result");
    expect(tracer.spans).toHaveLength(1);
    const span = tracer.spans[0];
    expect(span.name).toBe("healthy_wf");
    expect(span.attributes["tp.kind"]).toBe("agent");
    expect(span.attributes["tp.workflow_name"]).toBe("healthy_wf");
    expect(span.attributes["tp.user_id"]).toBe("u9");
    expect(span.attributes["tp.paid_plan"]).toBe("pro");
    expect(span.attributes["tp.meta.k"]).toBe("v");
    expect(span.ended).toBe(true);
  });

  it("chain(): tags tp.kind=chain", () => {
    const tracer = new FakeTracer();
    register(tracer);
    chain({ name: "pipe" }, () => null);
    expect(tracer.spans[0].attributes["tp.kind"]).toBe("chain");
  });

  it("adopts the span's trace/span ids onto the session", () => {
    register(new FakeTracer());
    session({ name: "ids_wf" }, () => {
      const s = getCurrentSession();
      expect(s.traceId).toBe("a".repeat(32));
      expect(s.rootSpanId).toBe("b".repeat(16));
    });
  });

  it("ends the span when the body throws, and rethrows unchanged", () => {
    const tracer = new FakeTracer();
    register(tracer);
    const bodyErr = new Error("boom");
    expect(() =>
      session({ name: "wf" }, () => {
        throw bodyErr;
      }),
    ).toThrow(bodyErr);
    expect(tracer.spans[0].ended).toBe(true);
  });
});

// ── 5. Async bodies on healthy and degraded paths ───────────────────────

describe("async bodies", () => {
  it("healthy: promise resolves and the span ends after settlement", async () => {
    const tracer = new FakeTracer();
    register(tracer);
    const out = await session({ name: "async_wf" }, async () => {
      await new Promise((r) => setTimeout(r, 5));
      return 42;
    });
    expect(out).toBe(42);
    expect(tracer.spans[0].ended).toBe(true);
  });

  it("healthy: async rejection propagates and the span still ends", async () => {
    const tracer = new FakeTracer();
    register(tracer);
    await expect(
      session({ name: "async_wf" }, async () => {
        throw new Error("async body error");
      }),
    ).rejects.toThrow("async body error");
    expect(tracer.spans[0].ended).toBe(true);
  });

  it("degraded (hostile metadata): promise resolves", async () => {
    register(new FakeTracer());
    const out = await session(
      { name: "wf", metadata: hostileMetadata() },
      async () => {
        await new Promise((r) => setTimeout(r, 5));
        return "async-ok";
      },
    );
    expect(out).toBe("async-ok");
  });

  it("degraded (throwing tracer): promise resolves, body ran once", async () => {
    register(new ThrowingTracer());
    let ran = 0;
    const out = await session({ name: "wf" }, async () => {
      ran++;
      return "async-degraded";
    });
    expect(out).toBe("async-degraded");
    expect(ran).toBe(1);
  });
});
