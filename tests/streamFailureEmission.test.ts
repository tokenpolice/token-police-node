/**
 * Streamed provider failures must land exactly one `llm` failure row.
 *
 * On the traceloop-instrumented OpenAI-wire streaming path `await create()`
 * resolves BEFORE any HTTP happens — traceloop's `_streamingWrapPromise`
 * returns an async generator synchronously and the request only fires on the
 * first `next()`. So:
 *
 * - the enforcer's request-failure catch has already returned (and stamped a
 *   SUCCESS `_call_outcome`) while the stream is still undrained, and
 * - the stream tap `_tapStreamUsageForOnEnd` had NO failure emission at all.
 *
 * Result before the fix: a zero-usage streamed failure produced **no llm row**
 * (telemetry's zero-usage gate dropped the span), and a failure that arrived
 * AFTER a usage chunk produced a **mislabeled success row** (span attrs /
 * usage stash carried tokens, so the gate passed).
 *
 * Fix under test:
 * - the tap catches the `inner.next()` rejection, emits ONE failure row
 *   (`buildCallOutcome` → error_kind/http_status) inside the captured obs
 *   scope, and rethrows the ORIGINAL error by identity;
 * - it first marks `${traceId}:${order}` in `session._failedStreamCompKeys`,
 *   which telemetry's deferred onEnd log consumes (marker deleted, pending
 *   composition dropped, no span row) so the two can never both land.
 *
 * Control arms that must NOT regress: the streaming success path (no
 * emission, no marker, usage stash untouched) and early consumer break
 * (`break` / `return()` is not a provider failure).
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { setClient } from "../src/state";
import { TPSession, _getSessionStorage } from "../src/context";
import { __test__ as enforcerTest } from "../src/enforcer";
import { TokenPoliceSpanProcessor } from "../src/telemetry";

const { _tapStreamUsageForOnEnd, _stashAttemptContext } = enforcerTest as any;

let logged: any[][];

beforeEach(() => {
  logged = [];
  setClient({
    firewall: "off",
    captureStreamUsage: true,
    log: (...args: any[]) => logged.push(args),
  } as any);
});

afterEach(() => {
  setClient(null as any);
});

function newSession(): TPSession {
  return new TPSession({
    userId: "u1",
    paidPlan: "pro",
    workflowName: "wf",
    traceId: "c".repeat(32),
    rootSpanId: "d".repeat(16),
  });
}

/** `tp.log`'s trailing options arg — where `call_outcome` lives. */
function extrasOf(logArgs: any[]): any {
  return logArgs[logArgs.length - 1];
}

/**
 * A re-iterable OpenAI-wire stream: every `Symbol.asyncIterator` call mints a
 * fresh cursor (a bare async generator returns *itself*, which would hide the
 * once-guard's real job). Once the chunk list is exhausted it either rejects
 * with `opts.err` forever (provider failure) or reports `done`.
 */
function mkStream(
  chunks: any[],
  opts: { err?: unknown; onReturn?: () => void } = {},
): any {
  return {
    [Symbol.asyncIterator]() {
      let i = 0;
      return {
        async next(): Promise<IteratorResult<any>> {
          if (i < chunks.length) return { value: chunks[i++], done: false as const };
          if (opts.err !== undefined) throw opts.err;
          return { value: undefined, done: true as const };
        },
        async return(v?: any): Promise<IteratorResult<any>> {
          opts.onReturn?.();
          return { value: v, done: true as const };
        },
      };
    },
  };
}

const contentChunk = { choices: [{ delta: { content: "Hi" } }] };
const usageChunk = {
  usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
  choices: [],
};

/** OpenAI-SDK-shaped 401 (`AuthenticationError` + `status`). */
function auth401(): any {
  class AuthenticationError extends Error {}
  return Object.assign(new AuthenticationError("401 Incorrect API key provided"), {
    status: 401,
  });
}

/**
 * Installs the tap over `stream` for span order 0 with a stashed attempt
 * context, then hands back the tapped stream + its compKey.
 */
function tap(
  session: TPSession,
  stream: any,
  order = 0,
  suppressUsageChunk = false,
): { tapped: any; compKey: string } {
  _stashAttemptContext(session, "openai", [{ model: "gpt-4o-mini" }]);
  const tapped = _tapStreamUsageForOnEnd(
    "openai", // wire parse key — the module client, not the serving vendor
    stream,
    session,
    [{ model: "gpt-4o-mini", messages: [{ role: "user", content: "x" }] }],
    order,
    suppressUsageChunk,
  );
  return { tapped, compKey: `${session.traceId}:${order}` };
}

/** Drains `tapped` and returns the error the CONSUMER saw (or undefined). */
async function drainCatching(tapped: any, seen: any[] = []): Promise<unknown> {
  try {
    for await (const c of tapped) seen.push(c);
    return undefined;
  } catch (e) {
    return e;
  }
}

/** A ReadableSpan-shaped stub keyed to `session`'s trace + span order. */
function fakeSpan(
  session: TPSession,
  order: number,
  opts: { input?: number; output?: number; model?: string; system?: string } = {},
): any {
  const attrs: Record<string, unknown> = {
    "gen_ai.system": opts.system ?? "openai",
    "gen_ai.request.model": opts.model ?? "gpt-4o-mini",
    "tp.trace_id": session.traceId,
    "tp.span_order": order,
  };
  if (opts.input != null) attrs["gen_ai.usage.input_tokens"] = opts.input;
  if (opts.output != null) attrs["gen_ai.usage.output_tokens"] = opts.output;
  return {
    attributes: attrs,
    name: "chat gpt-4o-mini",
    startTime: [0, 0] as [number, number],
    endTime: [1, 0] as [number, number],
    spanContext: () => ({
      traceId: "0000000000000000000000000000ef01",
      spanId: "0000000000005678",
    }),
    parentSpanId: undefined,
    status: { code: 0 },
  };
}

/** Runs the span processor's deferred (nextTick) onEnd log to completion. */
async function runOnEnd(session: TPSession, span: any): Promise<void> {
  await _getSessionStorage().run(session, async () => {
    new TokenPoliceSpanProcessor().onEnd(span);
    await new Promise((r) => setTimeout(r, 0));
  });
}

// ── 1. Request-time failure (first next() rejects) ───────────────

describe("stream tap failure emission — first next() rejects", () => {
  it("401 on the first pull → exactly one failed llm row (auth_error / 401), original error by identity", async () => {
    const session = newSession();
    const boom = auth401();
    await _getSessionStorage().run(session, async () => {
      const { tapped } = tap(session, mkStream([], { err: boom }));
      const seen: any[] = [];
      const caught = await drainCatching(tapped, seen);
      // GOLDEN RULE: the provider's own error object, not a wrapper.
      expect(caught).toBe(boom);
      expect(seen).toEqual([]);
    });

    expect(logged).toHaveLength(1);
    const extras = extrasOf(logged[0]);
    expect(extras.call_outcome.status).toBe("failed");
    expect(extras.call_outcome.error_kind).toBe("auth_error");
    expect(extras.call_outcome.http_status).toBe(401);
    // The stashed attempt context survives into the row (not model="unknown").
    expect(logged[0][4]).toBe("gpt-4o-mini");
    expect(logged[0][5]).toBe("openai");
    // A failed call has no usage — the row stays unmeasured.
    expect(logged[0][6]).toBe(0);
    expect(logged[0][7]).toBe(0);
  });

  it("classification is real, not hardcoded: 429 RateLimitError → rate_limited / 429", async () => {
    const session = newSession();
    class RateLimitError extends Error {}
    const boom = Object.assign(new RateLimitError("429 slow down"), { status: 429 });
    await _getSessionStorage().run(session, async () => {
      const { tapped } = tap(session, mkStream([], { err: boom }));
      expect(await drainCatching(tapped)).toBe(boom);
    });
    expect(logged).toHaveLength(1);
    const outcome = extrasOf(logged[0]).call_outcome;
    expect(outcome.error_kind).toBe("rate_limited");
    expect(outcome.http_status).toBe(429);
  });

  it("marks the compKey in session._failedStreamCompKeys for telemetry to consume", async () => {
    const session = newSession();
    const boom = auth401();
    let compKey = "";
    await _getSessionStorage().run(session, async () => {
      const t = tap(session, mkStream([], { err: boom }), 3);
      compKey = t.compKey;
      expect(await drainCatching(t.tapped)).toBe(boom);
    });
    const marks = (session as any)._failedStreamCompKeys;
    expect(marks instanceof Set).toBe(true);
    expect(marks.has(compKey)).toBe(true);
    expect(compKey).toBe(`${session.traceId}:3`);
  });
});

// ── 2. Mid-stream failure after a usage chunk (double-log guard) ──

describe("stream tap failure emission — failure AFTER a usage-carrying chunk", () => {
  it("one failure row, and telemetry's deferred span log is suppressed (no second, success-labeled row)", async () => {
    const session = newSession();
    const boom = Object.assign(new Error("upstream 500"), { status: 500 });
    let compKey = "";
    await _getSessionStorage().run(session, async () => {
      const t = tap(session, mkStream([contentChunk, usageChunk], { err: boom }));
      compKey = t.compKey;
      const seen: any[] = [];
      expect(await drainCatching(t.tapped, seen)).toBe(boom);
      // The usage chunk WAS consumed before the failure — this is exactly the
      // case that used to land a mislabeled success row.
      expect(seen).toHaveLength(2);
      expect(
        (session as any)._pendingCompositions[compKey]?.usage?.input_tokens,
      ).toBe(10);
    });

    expect(logged).toHaveLength(1);
    expect(extrasOf(logged[0]).call_outcome.status).toBe("failed");
    expect(extrasOf(logged[0]).call_outcome.error_kind).toBe("server_error");

    // The instrumentor's span still ends and reaches the processor — with
    // token attrs, so the zero-usage gate would NOT have saved us.
    await runOnEnd(session, fakeSpan(session, 0, { input: 10, output: 2 }));

    expect(logged).toHaveLength(1); // still exactly one row for this call
    // Marker consumed (Set never outgrows in-flight failed streams) and the
    // pending composition dropped so nothing leaks.
    expect((session as any)._failedStreamCompKeys.has(compKey)).toBe(false);
    expect((session as any)._pendingCompositions[compKey]).toBeUndefined();
  });

  it("control: the SAME span lands a row when the marker is absent (proves the marker is what suppresses)", async () => {
    const session = newSession();
    const boom = Object.assign(new Error("upstream 500"), { status: 500 });
    await _getSessionStorage().run(session, async () => {
      const t = tap(session, mkStream([contentChunk, usageChunk], { err: boom }));
      expect(await drainCatching(t.tapped)).toBe(boom);
    });
    expect(logged).toHaveLength(1);

    // Drop the marker — pre-fix telemetry behavior.
    (session as any)._failedStreamCompKeys.clear();
    await runOnEnd(session, fakeSpan(session, 0, { input: 10, output: 2 }));
    expect(logged).toHaveLength(2); // the mislabeled second row the fix kills
  });
});

// ── 3. Streaming success path (control arm 2 — must not regress) ──

describe("stream tap — streaming success path unchanged", () => {
  it("full drain → no failure emission, marker Set never created, usage stash intact", async () => {
    const session = newSession();
    let compKey = "";
    await _getSessionStorage().run(session, async () => {
      const t = tap(session, mkStream([contentChunk, usageChunk]));
      compKey = t.compKey;
      const seen: any[] = [];
      expect(await drainCatching(t.tapped, seen)).toBeUndefined();
      expect(seen).toHaveLength(2);
    });

    expect(logged).toHaveLength(0); // the tap never logs on success
    expect((session as any)._failedStreamCompKeys).toBeUndefined();
    expect((session as any)._call_outcome).toBeUndefined();
    const cd = (session as any)._pendingCompositions[compKey];
    expect(cd.usage).toMatchObject({
      input_tokens: 10,
      output_tokens: 2,
      cached_tokens: 0,
    });
    expect(cd.usage.raw.prompt_tokens).toBe(10);
    expect(cd.response.length).toBeGreaterThan(0);
  });

  it("the span row still lands on success (telemetry's marker check is a no-op)", async () => {
    const session = newSession();
    await _getSessionStorage().run(session, async () => {
      const t = tap(session, mkStream([contentChunk, usageChunk]));
      expect(await drainCatching(t.tapped)).toBeUndefined();
    });
    // Span attrs carry no usage — the row comes from the tap's stash.
    await runOnEnd(session, fakeSpan(session, 0));
    expect(logged).toHaveLength(1);
    expect(logged[0][6]).toBe(10);
    expect(logged[0][7]).toBe(2);
    expect(extrasOf(logged[0]).call_outcome?.status).not.toBe("failed");
  });
});

// ── 4. Early consumer break is not a provider failure ────────────

describe("stream tap — early consumer break / return()", () => {
  it("break after one chunk → no failure row, no marker, inner return() still delegated", async () => {
    const session = newSession();
    let returned = 0;
    await _getSessionStorage().run(session, async () => {
      const { tapped } = tap(
        session,
        mkStream([contentChunk, contentChunk, usageChunk], {
          onReturn: () => {
            returned++;
          },
        }),
      );
      for await (const _c of tapped) break;
    });
    expect(returned).toBe(1);
    expect(logged).toHaveLength(0);
    expect((session as any)._failedStreamCompKeys).toBeUndefined();
  });

  it("explicit iterator.return() mid-iteration → no failure row", async () => {
    const session = newSession();
    await _getSessionStorage().run(session, async () => {
      const { tapped } = tap(session, mkStream([contentChunk, usageChunk]));
      const it = tapped[Symbol.asyncIterator]();
      await it.next();
      await it.return?.();
    });
    expect(logged).toHaveLength(0);
    expect((session as any)._failedStreamCompKeys).toBeUndefined();
  });
});

// ── 5. Once-guard lives in the OUTER tap closure ─────────────────

describe("stream tap — once-guard (exactly one emission per call)", () => {
  it("repeated next() after a rejection still emits once, and keeps rethrowing the same error", async () => {
    const session = newSession();
    const boom = auth401();
    await _getSessionStorage().run(session, async () => {
      const { tapped } = tap(session, mkStream([], { err: boom }));
      const it = tapped[Symbol.asyncIterator]();
      await expect(it.next()).rejects.toBe(boom);
      await expect(it.next()).rejects.toBe(boom);
      await expect(it.next()).rejects.toBe(boom);
    });
    expect(logged).toHaveLength(1);
  });

  it("a SECOND Symbol.asyncIterator invocation (fresh inner cursor) does not double-emit", async () => {
    const session = newSession();
    const boom = auth401();
    await _getSessionStorage().run(session, async () => {
      const { tapped } = tap(session, mkStream([], { err: boom }));
      // Per-iterator state would reset here; the guard is in the outer closure.
      await expect(tapped[Symbol.asyncIterator]().next()).rejects.toBe(boom);
      await expect(tapped[Symbol.asyncIterator]().next()).rejects.toBe(boom);
    });
    expect(logged).toHaveLength(1);
  });

  it("two concurrent consumers of one tapped stream still produce one row", async () => {
    const session = newSession();
    const boom = auth401();
    await _getSessionStorage().run(session, async () => {
      const { tapped } = tap(session, mkStream([], { err: boom }));
      // Handlers attached at creation so neither in-flight rejection can be
      // seen as unhandled while the other is awaited.
      const a = tapped[Symbol.asyncIterator]().next().catch((e: unknown) => e);
      const b = tapped[Symbol.asyncIterator]().next().catch((e: unknown) => e);
      expect(await a).toBe(boom);
      expect(await b).toBe(boom);
    });
    expect(logged).toHaveLength(1);
  });
});

// ── 8. Golden rule: the emission path itself must never escape ───

describe("stream tap — golden rule (emission failures never reach the app)", () => {
  it("tp.log throwing → the consumer still receives the ORIGINAL provider error", async () => {
    setClient({
      firewall: "off",
      captureStreamUsage: true,
      log: () => {
        throw new Error("log exploded");
      },
    } as any);
    const session = newSession();
    const boom = auth401();
    await _getSessionStorage().run(session, async () => {
      const { tapped } = tap(session, mkStream([], { err: boom }));
      expect(await drainCatching(tapped)).toBe(boom);
    });
  });

  it("a hostile session field (throwing getter on the marker slot) → original error, nothing else propagates", async () => {
    const session = newSession();
    // The marker write is the FIRST statement of the emit helper — a throw
    // here must degrade to telemetry loss, never to a customer-visible error.
    Object.defineProperty(session, "_failedStreamCompKeys", {
      configurable: true,
      get(): never {
        throw new Error("hostile marker slot");
      },
      set(_v: unknown) {
        /* no-op */
      },
    });
    const boom = auth401();
    await _getSessionStorage().run(session, async () => {
      const { tapped } = tap(session, mkStream([contentChunk], { err: boom }));
      const seen: any[] = [];
      expect(await drainCatching(tapped, seen)).toBe(boom);
      expect(seen).toHaveLength(1);
    });
    expect(logged).toHaveLength(0); // emission aborted, but silently
  });

  it("no client at all → no throw, no row", async () => {
    setClient(null as any);
    const session = newSession();
    const boom = auth401();
    await _getSessionStorage().run(session, async () => {
      const { tapped } = tap(session, mkStream([], { err: boom }));
      expect(await drainCatching(tapped)).toBe(boom);
    });
    expect(logged).toHaveLength(0);
  });
});
