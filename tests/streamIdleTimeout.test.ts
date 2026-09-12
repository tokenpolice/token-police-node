/**
 * Node SSE client had no idle read timeout, so the heartbeat / stale-data
 * watchdog (stream.ts:120-122) could never fire on a silent half-open socket —
 * `reader.read()` (stream.ts:124) parks forever and the inline watchdog above it
 * is never re-evaluated, leaving the local rule/blocked-set cache silently stale
 * with no reconnect.
 *
 * Fix (connectAndPump() only): after the reader is obtained (AFTER the:103 and
 * :112 early returns), arm an `idleTimer = setTimeout(() => controller?.abort(),
 * HEARTBEAT_TIMEOUT_MS)`, `.unref()` it, `.refresh()` it on ANY received bytes
 * (incl. comment keepalives — a raw read()-byte-level reset matching Python's
 * httpx read=45s, NOT a dispatch-level reset), and clear it in `finally`. The
 * abort rejects the in-flight read → connectAndPump throws → the EXISTING
 * runForever catch + backoff reconnects. Node-only; Python already correct.
 *
 * These tests inject a fake fetch transport under fake timers so the idle timer
 * is the only thing that can move the loop, making the 44_999-vs-45_000 boundary
 * deterministic. They encode rubric items 1-10.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { StreamClient } from "../src/stream";

function abortError(): Error {
  return Object.assign(new Error("aborted"), { name: "AbortError" });
}

type Chunk = { value?: Uint8Array; done: boolean };

/** A reader whose read() never resolves on its own but rejects AbortError when
 * the captured signal fires — faithfully models undici aborting an in-flight
 * read on signal abort. */
function silentReader(signal: AbortSignal) {
  return {
    read: (): Promise<Chunk> =>
      new Promise<Chunk>((_resolve, reject) => {
        if (signal.aborted) return reject(abortError());
        signal.addEventListener("abort", () => reject(abortError()), { once: true });
      }),
  };
}

/** A reader the test drives: read() pends until push() supplies bytes; rejects
 * AbortError if the signal fires while pending. */
function controllableReader(signal: AbortSignal) {
  const queue: Chunk[] = [];
  let pendingResolve: ((v: Chunk) => void) | null = null;
  let pendingReject: ((e: Error) => void) | null = null;
  signal.addEventListener(
    "abort",
    () => {
      if (pendingReject) {
        const r = pendingReject;
        pendingResolve = pendingReject = null;
        r(abortError());
      }
    },
    { once: true },
  );
  const reader = {
    read: (): Promise<Chunk> =>
      new Promise<Chunk>((resolve, reject) => {
        if (signal.aborted) return reject(abortError());
        if (queue.length) return resolve(queue.shift()!);
        pendingResolve = resolve;
        pendingReject = reject;
      }),
  };
  function push(text: string): void {
    const v: Chunk = { value: new TextEncoder().encode(text), done: false };
    if (pendingResolve) {
      const r = pendingResolve;
      pendingResolve = pendingReject = null;
      r(v);
    } else {
      queue.push(v);
    }
  }
  return { reader, push };
}

function okResponse(reader: { read: () => Promise<Chunk> }) {
  return { ok: true, body: { getReader: () => reader } };
}

function makeClient(): StreamClient {
  return new StreamClient({
    baseUrl: "http://localhost:15099",
    apiKey: "tp_sk_test",
    sdkVersion: "test",
    deployment: "daemon",
    clientId: "cid",
    firewall: "enforce",
  });
}

function fetchCalls(): number {
  return (globalThis.fetch as unknown as { mock: { calls: unknown[] } }).mock.calls.length;
}

describe("SSE idle read timeout", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  // Items 1 + 2 + 11 (value pinned to HEARTBEAT_TIMEOUT_MS by the boundary).
  it("silent socket aborts after the heartbeat timeout (and not a tick before)", async () => {
    const signals: AbortSignal[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, opts: { signal: AbortSignal }) => {
        signals.push(opts.signal);
        return okResponse(silentReader(opts.signal));
      }),
    );
    const client = makeClient();
    client.start();
    // Flush microtasks so fetch + getReader settle and the idle timer arms at t=0.
    await vi.advanceTimersByTimeAsync(0);
    expect(signals.length).toBe(1);

    await vi.advanceTimersByTimeAsync(44_999);
    expect(signals[0].aborted).toBe(false); // item 2: 44_999 does NOT abort

    await vi.advanceTimersByTimeAsync(1);
    expect(signals[0].aborted).toBe(true); // item 1: crossing to 45_000 aborts

    client.stop();
  });

  // Item 3: idle-abort drives a reconnect via the EXISTING runForever path.
  it("idle-abort drives a reconnect through the existing runForever backoff", async () => {
    const signals: AbortSignal[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, opts: { signal: AbortSignal }) => {
        signals.push(opts.signal);
        return okResponse(silentReader(opts.signal));
      }),
    );
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchCalls()).toBe(1);

    await vi.advanceTimersByTimeAsync(45_000); // idle abort
    expect(signals[0].aborted).toBe(true);

    await vi.advanceTimersByTimeAsync(2_000); // past the first backoff step (>= 1000ms)
    expect(fetchCalls()).toBeGreaterThanOrEqual(2); // reconnected — no new backoff path

    client.stop();
  });

  // Item 4: idle-abort is a swallowed rejection — never a throw into customer
  // code, never an unhandled rejection.
  it("idle-abort never throws into customer code nor surfaces an unhandled rejection", async () => {
    const unhandled: unknown[] = [];
    const onUnhandled = (e: unknown) => unhandled.push(e);
    process.on("unhandledRejection", onUnhandled);
    try {
      vi.stubGlobal(
        "fetch",
        vi.fn(async (_url: string, opts: { signal: AbortSignal }) => okResponse(silentReader(opts.signal))),
      );
      const client = makeClient();
      expect(() => client.start()).not.toThrow();
      await vi.advanceTimersByTimeAsync(0);
      await vi.advanceTimersByTimeAsync(45_000); // abort
      await vi.advanceTimersByTimeAsync(2_000); // reconnect
      await Promise.resolve();
      expect(unhandled).toEqual([]);
      client.stop();
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
  });

  // Item 5: idle timer is RESET at the raw read()-byte level on each DATA chunk
  // — continuous data never trips the timeout (idle reset, not a hard ceiling).
  it("resets on each data chunk — 90s of data flowing every 30s never aborts", async () => {
    const signals: AbortSignal[] = [];
    let push: ((t: string) => void) | null = null;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, opts: { signal: AbortSignal }) => {
        signals.push(opts.signal);
        const c = controllableReader(opts.signal);
        push = c.push;
        return okResponse(c.reader);
      }),
    );
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0);

    for (let i = 0; i < 3; i++) {
      await vi.advanceTimersByTimeAsync(30_000); // under 45s since last byte
      expect(signals[0].aborted).toBe(false);
      push!('event: delta\ndata: {"version":0}\n\n'); // valid frame, harmless (version 0 → no-op)
      await vi.advanceTimersByTimeAsync(0); // read resolves → idleTimer.refresh()
      expect(signals[0].aborted).toBe(false);
    }

    expect(signals.length).toBe(1); // single connection across 90s
    expect(fetchCalls()).toBe(1);
    client.stop();
  });

  // Item 6: idle timer is RESET by COMMENT-ONLY keepalive bytes — pins the reset
  // at read()-byte level, NOT dispatch level (a dispatch-level reset would abort
  // a comment-only stream at 45s and diverge from Python's httpx read-timeout).
  it("resets on comment-only keepalive frames — ~120s of `: keepalive` never aborts", async () => {
    const signals: AbortSignal[] = [];
    let push: ((t: string) => void) | null = null;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, opts: { signal: AbortSignal }) => {
        signals.push(opts.signal);
        const c = controllableReader(opts.signal);
        push = c.push;
        return okResponse(c.reader);
      }),
    );
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0);

    // > 45s heartbeat window, < STALE_DATA_THRESHOLD_MS (600_000) so the inline
    // :122 stale-data check never trips and muddies the signal.
    for (let t = 15_000; t <= 120_000; t += 15_000) {
      await vi.advanceTimersByTimeAsync(15_000);
      expect(signals[0].aborted).toBe(false);
      push!(": keepalive\n"); // comment only — no data: line → dispatch() never called
      await vi.advanceTimersByTimeAsync(0); // read resolves → idleTimer.refresh()
    }

    expect(signals[0].aborted).toBe(false);
    expect(signals.length).toBe(1);
    expect(fetchCalls()).toBe(1);
    client.stop();
  });

  // SSE-11: 10 min of `: keepalive` only trips the stale-data clean return
  // (`lastDataEventAt` never moves; idle 45s is refreshed). Pump returns →
  // conn1 is aborted before fetch N+1 starts.
  it("stale-data clean return aborts conn1 before the 2nd fetch starts", async () => {
    const unhandled: unknown[] = [];
    const onUnhandled = (e: unknown) => unhandled.push(e);
    process.on("unhandledRejection", onUnhandled);
    try {
      const signals: AbortSignal[] = [];
      let push: ((t: string) => void) | null = null;
      vi.stubGlobal(
        "fetch",
        vi.fn(async (_url: string, opts: { signal: AbortSignal }) => {
          signals.push(opts.signal);
          if (signals.length === 2) {
            expect(signals[0].aborted).toBe(true); // N dead before N+1 starts
            expect(signals[1]).not.toBe(signals[0]);
          }
          const c = controllableReader(opts.signal);
          if (signals.length === 1) push = c.push;
          return okResponse(c.reader);
        }),
      );
      const client = makeClient();
      client.start();
      await vi.advanceTimersByTimeAsync(0);

      // Keepalives every 15s for 10 min + one more tick past STALE_DATA_THRESHOLD_MS.
      // 600_000 is NOT > 600_000; 615_000 is.
      for (let t = 15_000; t <= 615_000; t += 15_000) {
        await vi.advanceTimersByTimeAsync(15_000);
        expect(signals[0].aborted).toBe(false); // idle 45s never fires
        expect(fetchCalls()).toBe(1);
        push!(": keepalive\n");
        await vi.advanceTimersByTimeAsync(0);
      }

      await vi.advanceTimersByTimeAsync(1_500); // ~1s floor after the clean return
      expect(fetchCalls()).toBeGreaterThanOrEqual(2);
      expect(signals[0].aborted).toBe(true);
      await Promise.resolve();
      expect(unhandled).toEqual([]); // controllableReader has no cancel()
      client.stop();
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
  });

  // Item 7: idle timer is CLEARED on a normal done:true exit — a stale timer can
  // never fire against the NEXT connection's controller.
  it("clears the idle timer on done:true so it cannot abort the next connection", async () => {
    const signals: AbortSignal[] = [];
    let call = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, opts: { signal: AbortSignal }) => {
        call += 1;
        signals.push(opts.signal);
        if (call === 1) {
          // Connection 1 closes immediately: read() → { done: true }.
          return okResponse({ read: () => Promise.resolve<Chunk>({ done: true }) });
        }
        return okResponse(silentReader(opts.signal)); // connection 2: silent
      }),
    );
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0); // conn1 reads done:true → returns → finally clears its timer
    await vi.advanceTimersByTimeAsync(1_500); // backoff (~1s) → reconnect → conn2
    expect(signals.length).toBeGreaterThanOrEqual(2);
    expect(signals[0].aborted).toBe(true); // this pump's fetch is dead before conn2 starts

    // Conn1's timer (armed ~t0) would, if leaked, fire at 45_000 and abort conn2.
    // Cleared, conn2 only aborts via its OWN window (~46_000+). Check at 45_500.
    await vi.advanceTimersByTimeAsync(45_500 - 1_500);
    expect(signals[1].aborted).toBe(false);

    client.stop();
  });

  // Item 8: mixed fixture (non-ok throw / bodyless-200 throw / !reader return)
  // NEVER arms a live idle timer. Abort happens in finally immediately, not
  // via the 45s idle timer.
  it("never arms a live idle timer on an early-return response", async () => {
    const signals: AbortSignal[] = [];
    let call = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, opts: { signal: AbortSignal }) => {
        call += 1;
        signals.push(opts.signal);
        if (call === 1) return { ok: false, body: { getReader: () => silentReader(opts.signal) } }; // no status → throw stream connect …
        if (call === 2) return { ok: true, body: null }; // bodyless 200 → throw
        if (call === 3) return { ok: true, body: {} }; // body present but no getReader → !reader return
        return { ok: true, body: null }; // keep early-returning
      }),
    );
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0);
    // After the first connect the only scheduled timer is the backoff
    // wakeTimer — no idle timer was armed. Abort is immediate via finally.
    expect(vi.getTimerCount()).toBe(1);
    expect(signals[0].aborted).toBe(true);

    await vi.advanceTimersByTimeAsync(60_000);
    expect(signals.every((s) => s.aborted)).toBe(true);
    expect(vi.getTimerCount()).toBe(1); // still only the backoff timer

    client.stop();
  });

  // Item 9: idle timer is CLEARED on stop() — clean teardown, no dangling timer.
  it("clears the idle timer on stop() and does not reconnect", async () => {
    const signals: AbortSignal[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, opts: { signal: AbortSignal }) => {
        signals.push(opts.signal);
        return okResponse(silentReader(opts.signal));
      }),
    );
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(vi.getTimerCount()).toBe(1); // the armed idle timer

    client.stop();
    await vi.advanceTimersByTimeAsync(0); // stop aborts → read rejects → loop exits → finally clears
    expect(signals[0].aborted).toBe(true); // stop()'s controller.abort()
    expect(vi.getTimerCount()).toBe(0); // idle timer cleared; no backoff (stopped)

    const before = fetchCalls();
    await vi.advanceTimersByTimeAsync(60_000);
    expect(fetchCalls()).toBe(before); // no reconnect after stop
  });

  // Item 10: the idle timer is .unref()'d so it never pins the event loop.
  it("calls .unref() on the armed idle timer", async () => {
    const unrefCalls: number[] = [];
    const fakeSetTimeout = globalThis.setTimeout;
    vi.stubGlobal("setTimeout", ((fn: (...a: unknown[]) => void, ms?: number, ...rest: unknown[]) => {
      const t = (fakeSetTimeout as unknown as (...a: unknown[]) => { unref?: () => unknown })(fn, ms, ...rest);
      if (ms === 45_000 && t && typeof t.unref === "function") {
        const orig = t.unref.bind(t);
        t.unref = () => {
          unrefCalls.push(ms);
          return orig();
        };
      }
      return t;
    }) as unknown as typeof setTimeout);
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, opts: { signal: AbortSignal }) => okResponse(silentReader(opts.signal))),
    );
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(unrefCalls).toContain(45_000); // .unref() was invoked on the 45s idle timer
    client.stop();
  });

  // Item 11: the backoff wake timer is .unref()'d too — a reconnect sleep during
  // an outage must never keep the customer's process alive.
  it("calls .unref() on the backoff wake timer", async () => {
    const unrefCalls: number[] = [];
    const fakeSetTimeout = globalThis.setTimeout;
    // jitter 0 → first backoff delay is exactly 1000ms (steps[0]*1000).
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    vi.stubGlobal("setTimeout", ((fn: (...a: unknown[]) => void, ms?: number, ...rest: unknown[]) => {
      const t = (fakeSetTimeout as unknown as (...a: unknown[]) => { unref?: () => unknown })(fn, ms, ...rest);
      if (ms === 1000 && t && typeof t.unref === "function") {
        const orig = t.unref.bind(t);
        t.unref = () => {
          unrefCalls.push(ms);
          return orig();
        };
      }
      return t;
    }) as unknown as typeof setTimeout);
    // A bodyless non-terminal connect throws → runForever catch → backoff sleep
    // (no reader obtained, so no idle timer competes for the setTimeout).
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 503, body: null })));
    const client = makeClient();
    client.start();
    // Let the failed connect + throw + catch settle and schedule the backoff sleep.
    await vi.advanceTimersByTimeAsync(0);
    expect(unrefCalls).toContain(1000); // .unref() was invoked on the backoff wake timer
    client.stop();
  });
});
