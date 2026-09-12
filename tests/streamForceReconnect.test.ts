/**
 * A `delta` frame arriving while the local cache is UNHEALTHY (no
 * snapshot landed yet, or a prior apply/TTL failure poisoned the pack) used to
 * reset the reconnect cursor (`this.lastEventId = null`, stream.ts) but NOT
 * force a reconnect. Because `Last-Event-ID` is only ever sent in the connect
 * headers (stream.ts:106), nulling the cursor is inert until an actual
 * reconnect happens — and while the server keeps streaming deltas the
 * idle timer keeps `refresh()`ing on the bytes and `lastEventAt` never ages,
 * so no watchdog ever fires and the cache is silently stale forever (inline
 * /check fallback for the whole duration).
 *
 * Fix (dispatch + connectAndPump only): a per-connection `forceReconnect` bool
 * set ONLY on the unhealthy-cache delta branch, plus a single loop-top guard
 * `if (this.forceReconnect) return;` that returns NORMALLY (no throw) so the
 * EXISTING runForever backoff reconnects WITHOUT Last-Event-ID → fresh
 * snapshot. Never sets `stopped`; lifecycle + idle timer untouched.
 *
 * These tests reuse the fake-transport harness under fake timers so the
 * only thing that can move the loop is the injected reader, making the
 * reconnect deterministic. They encode rubric assertions 5, 6, 8, 10, 11.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { StreamClient } from "../src/stream";
import * as state from "../src/state";

type Chunk = { value?: Uint8Array; done: boolean };

function abortError(): Error {
  return Object.assign(new Error("aborted"), { name: "AbortError" });
}

/** A reader the test drives: read() pends until push() supplies bytes; rejects
 * AbortError if the captured signal fires while pending. Mirrors the. */
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

/** A reader that emits exactly ONE valid `event: delta` frame on its first
 * read(), then parks (subsequent reads never resolve on their own; they reject
 * AbortError if the signal fires). Models a server pushing an unappliable
 * delta to an unhealthy cache. Bounded (one frame per connection) so that
 * WITHOUT the fix the pump simply parks (no reconnect) — a clean "fetch stays
 * 1" RED — and WITH the fix the single delta forces a reconnect. */
function oneDeltaThenParkReader(signal: AbortSignal) {
  let emitted = false;
  return {
    read: (): Promise<Chunk> =>
      new Promise<Chunk>((resolve, reject) => {
        if (signal.aborted) return reject(abortError());
        if (!emitted) {
          emitted = true;
          const frame = `event: delta\ndata: {"version":1,"ops":[]}\n\n`;
          return resolve({ value: new TextEncoder().encode(frame), done: false });
        }
        signal.addEventListener("abort", () => reject(abortError()), { once: true });
      }),
  };
}

function okResponse(reader: { read: () => Promise<Chunk> }) {
  return { ok: true, body: { getReader: () => reader } };
}

function makeClient(): StreamClient {
  return new StreamClient({
    baseUrl: "http://localhost:15098",
    apiKey: "tp_sk_test",
    sdkVersion: "test",
    deployment: "daemon",
    clientId: "cid",
    firewall: "enforce",
  });
}

function fetchMock() {
  return globalThis.fetch as unknown as {
    mock: { calls: [string, { headers: Record<string, string>; signal: AbortSignal }][] };
  };
}
function fetchCalls(): number {
  return fetchMock().mock.calls.length;
}

const SNAPSHOT = { version: 1, tenant_id: "t1", project_id: "p1", directives: [], loop_blocks: [] };
const SNAPSHOT_FRAME = `event: snapshot\nid: 1\ndata: ${JSON.stringify(SNAPSHOT)}\n\n`;

describe("Force reconnect on unhealthy-cache delta", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    state.resetPack(); // cache UNHEALTHY: _pack === null → isCacheHealthy() false
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    state.resetPack();
  });

  // Assertion 8 (RED-pre / GREEN-post): a delta while the cache is unhealthy
  // forces a reconnect. Pre-fix the pump just parks after the delta (no
  // watchdog fires) so fetch stays 1 forever (RED); post-fix the delta flags a
  // reconnect and a 2nd fetch fires after one backoff step.
  it("reconnects on an unappliable delta with an unhealthy cache", async () => {
    const unhandled: unknown[] = [];
    const onUnhandled = (e: unknown) => unhandled.push(e);
    process.on("unhandledRejection", onUnhandled);
    try {
      const signals: AbortSignal[] = [];
      vi.stubGlobal(
        "fetch",
        vi.fn(async (_url: string, opts: { signal: AbortSignal }) => {
          signals.push(opts.signal);
          if (signals.length === 2) {
            expect(signals[0].aborted).toBe(true); // N dead before N+1 starts
            expect(signals[1]).not.toBe(signals[0]);
          }
          return okResponse(oneDeltaThenParkReader(opts.signal));
        }),
      );
      const client = makeClient();
      client.start();
      await vi.advanceTimersByTimeAsync(0); // fetch + delta → forceReconnect set → loop returns
      expect(fetchCalls()).toBe(1); // pre-fix would stay 1 forever (RED)

      await vi.advanceTimersByTimeAsync(1_500); // > the 1000ms min backoff (stream.ts:83)
      expect(fetchCalls()).toBeGreaterThanOrEqual(2); // GREEN: reconnected via the existing backoff
      expect(signals[0].aborted).toBe(true);
      await Promise.resolve();
      expect(unhandled).toEqual([]); // cancel missing on default stub — optional-chain, no rejection
      client.stop();
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
  });

  // Assertion 6: the reconnect drops Last-Event-ID → server sends a fresh
  // snapshot. The 2nd fetch's headers must NOT carry Last-Event-ID.
  it("omits Last-Event-ID on the forced reconnect", async () => {
    const signals: AbortSignal[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, opts: { signal: AbortSignal }) => {
        signals.push(opts.signal);
        if (signals.length === 2) {
          expect(signals[0].aborted).toBe(true); // N dead before N+1 starts
          expect(signals[1]).not.toBe(signals[0]);
        }
        return okResponse(oneDeltaThenParkReader(opts.signal));
      }),
    );
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0);
    await vi.advanceTimersByTimeAsync(1_500);
    expect(fetchCalls()).toBeGreaterThanOrEqual(2);
    const secondHeaders = fetchMock().mock.calls[1][1].headers; // the 1st reconnect
    expect(secondHeaders["Last-Event-ID"]).toBeUndefined();
    expect(signals[0].aborted).toBe(true);
    client.stop();
  });

  // Assertion 10 (+ invariant 1 / ): the signal never sets `stopped` — the
  // reader survives and keeps reconnecting. A stopped reader would stall at 1.
  it("keeps the reader alive across the forced reconnect (never stops)", async () => {
    const signals: AbortSignal[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, opts: { signal: AbortSignal }) => {
        signals.push(opts.signal);
        if (signals.length >= 2) {
          expect(signals[signals.length - 2].aborted).toBe(true); // previous dead as each new fetch starts
          expect(signals[signals.length - 1]).not.toBe(signals[signals.length - 2]);
        }
        return okResponse(oneDeltaThenParkReader(opts.signal));
      }),
    );
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0);
    await vi.advanceTimersByTimeAsync(1_500);
    expect(fetchCalls()).toBeGreaterThanOrEqual(2);
    // Still alive → a THIRD connection follows the same forced-reconnect cycle
    // (a stopped reader would stall at 2).
    await vi.advanceTimersByTimeAsync(1_500);
    expect(fetchCalls()).toBeGreaterThanOrEqual(3);
    client.stop();
  });

  // Assertion 5 + 11: after a forced reconnect, a NEW connection that streams a
  // healthy snapshot then a healthy delta stays OPEN (flag did not leak into the
  // fresh connection) and applies the delta WITHOUT reconnecting.
  it("healthy snapshot+delta on the next connection stays open (flag reset per-connection)", async () => {
    const signals: AbortSignal[] = [];
    let call = 0;
    let push: ((t: string) => void) | null = null;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, opts: { signal: AbortSignal }) => {
        call += 1;
        signals.push(opts.signal);
        if (call === 1) return okResponse(oneDeltaThenParkReader(opts.signal)); // unhealthy → forces reconnect
        expect(signals[0].aborted).toBe(true); // N dead before N+1 starts
        expect(signals[1]).not.toBe(signals[0]);
        const c = controllableReader(opts.signal); // connection 2: test-driven
        push = c.push;
        return okResponse(c.reader);
      }),
    );
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0); // conn1 forces reconnect
    await vi.advanceTimersByTimeAsync(1_500); // reconnect → conn2 (parks, no auto delta)
    expect(fetchCalls()).toBe(2);
    expect(signals[0].aborted).toBe(true);
    expect(signals[1].aborted).toBe(false);

    push!(SNAPSHOT_FRAME); // cache goes healthy
    await vi.advanceTimersByTimeAsync(0);
    expect(state.getPackVersion()).toBe(1);

    // Healthy delta applies; no reconnect (flag never re-set on the healthy path).
    push!(`event: delta\nid: 2\ndata: {"version":2,"ops":[]}\n\n`);
    await vi.advanceTimersByTimeAsync(0);
    expect(state.getPackVersion()).toBe(2);

    await vi.advanceTimersByTimeAsync(5_000); // give any stray reconnect a chance
    expect(fetchCalls()).toBe(2); // still a single (2nd) connection — no self-kill
    expect(signals[0].aborted).toBe(true);
    expect(signals[1].aborted).toBe(false);
    client.stop();
  });

  // Assertion 11 (healthy-delta byte-identical): a delta on a HEALTHY cache from
  // the very first connection never triggers a reconnect and advances version.
  it("healthy-cache delta applies in-place without reconnecting", async () => {
    let push: ((t: string) => void) | null = null;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, opts: { signal: AbortSignal }) => {
        const c = controllableReader(opts.signal);
        push = c.push;
        return okResponse(c.reader);
      }),
    );
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0);
    push!(SNAPSHOT_FRAME);
    await vi.advanceTimersByTimeAsync(0);
    expect(state.getPackVersion()).toBe(1);

    push!(`event: delta\nid: 2\ndata: {"version":2,"ops":[]}\n\n`);
    await vi.advanceTimersByTimeAsync(0);
    expect(state.getPackVersion()).toBe(2);

    await vi.advanceTimersByTimeAsync(5_000);
    expect(fetchCalls()).toBe(1); // no reconnect on the healthy path
    client.stop();
  });

  // SSE-11: after two force-reconnects the live connection is the 3rd; stop()
  // aborts it. Connections 1–2 were already aborted in their own finally
  // (N dead before N+1) — stop() must not be what reaps leaked GETs.
  it("stop() after two reconnects aborts current; previous already aborted", async () => {
    const signals: AbortSignal[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, opts: { signal: AbortSignal }) => {
        signals.push(opts.signal);
        if (signals.length >= 2) {
          expect(signals[signals.length - 2].aborted).toBe(true);
          expect(signals[signals.length - 1]).not.toBe(signals[signals.length - 2]);
        }
        if (signals.length === 3) {
          expect(signals[0].aborted).toBe(true);
          expect(signals[1].aborted).toBe(true);
        }
        // Conn 1–2 force-reconnect immediately; conn 3 parks so stop() has a live current.
        if (signals.length <= 2) return okResponse(oneDeltaThenParkReader(opts.signal));
        const c = controllableReader(opts.signal);
        return okResponse(c.reader);
      }),
    );
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0); // conn1 → forceReconnect → abort
    await vi.advanceTimersByTimeAsync(1_500); // conn2
    await vi.advanceTimersByTimeAsync(1_500); // conn3 parks
    expect(fetchCalls()).toBe(3);
    expect(signals[0].aborted).toBe(true);
    expect(signals[1].aborted).toBe(true);
    expect(signals[2].aborted).toBe(false);

    client.stop();
    await vi.advanceTimersByTimeAsync(0);
    expect(signals[2].aborted).toBe(true); // current
    expect(signals[0].aborted).toBe(true);
    expect(signals[1].aborted).toBe(true);
  });

  // SSE-11: finally fire-and-forget `reader.cancel?.()` (never awaited) still
  // must call cancel, swallow reject, no unhandledRejection, reconnect at
  // the ~1s floor. Dedicated cancel stubs only — do not add cancel to
  // silent/controllable/oneDeltaThenPark / done:true readers.
  it("reader.cancel() reject still reconnects, does not reject, backoff stays at the floor", async () => {
    const unhandled: unknown[] = [];
    const onUnhandled = (e: unknown) => unhandled.push(e);
    process.on("unhandledRejection", onUnhandled);
    vi.spyOn(Math, "random").mockReturnValue(0.5); // jitter 0 → first two backoffs are 1000ms if reset
    try {
      const signals: AbortSignal[] = [];
      const cancelCalls: number[] = [];
      function oneDeltaThenParkReaderRejectingCancel(signal: AbortSignal) {
        let emitted = false;
        return {
          read: (): Promise<Chunk> =>
            new Promise<Chunk>((resolve, reject) => {
              if (signal.aborted) return reject(abortError());
              if (!emitted) {
                emitted = true;
                const frame = `event: delta\ndata: {"version":1,"ops":[]}\n\n`;
                return resolve({ value: new TextEncoder().encode(frame), done: false });
              }
              signal.addEventListener("abort", () => reject(abortError()), { once: true });
            }),
          cancel: () => {
            cancelCalls.push(1);
            return Promise.reject(new Error("x"));
          },
        };
      }
      vi.stubGlobal(
        "fetch",
        vi.fn(async (_url: string, opts: { signal: AbortSignal }) => {
          signals.push(opts.signal);
          if (signals.length === 2) {
            expect(signals[0].aborted).toBe(true);
            expect(signals[1]).not.toBe(signals[0]);
          }
          return okResponse(oneDeltaThenParkReaderRejectingCancel(opts.signal));
        }),
      );
      const client = makeClient();
      client.start();
      await vi.advanceTimersByTimeAsync(0); // delta → forceReconnect return → finally cancel rejects
      expect(fetchCalls()).toBe(1);

      await vi.advanceTimersByTimeAsync(1_500); // ~1s floor, not a climbed step
      expect(fetchCalls()).toBeGreaterThanOrEqual(2);
      expect(signals[0].aborted).toBe(true);
      expect(cancelCalls.length).toBeGreaterThanOrEqual(1);

      // Traffic was seen (delta dispatched) → attempt reset. A finally-throw
      // would climb 1s→2s and the 3rd fetch would NOT land in another 1.5s.
      await vi.advanceTimersByTimeAsync(1_500);
      expect(fetchCalls()).toBeGreaterThanOrEqual(3);

      await Promise.resolve();
      expect(unhandled).toEqual([]);
      client.stop();
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
  });

  // SSE-11: hanging cancel must NOT stall connectAndPump (would skip
  // markStreamDisconnected and leave _streamConnected=true). Abort stays
  // first; cancel is fire-and-forget. 2nd fetch after ~1s floor is the pin.
  it("hanging reader.cancel() still reconnects; 2nd fetch sees prior abort", async () => {
    const unhandled: unknown[] = [];
    const onUnhandled = (e: unknown) => unhandled.push(e);
    process.on("unhandledRejection", onUnhandled);
    vi.spyOn(Math, "random").mockReturnValue(0.5); // jitter 0 → first backoff 1000ms
    try {
      const signals: AbortSignal[] = [];
      const cancelCalls: number[] = [];
      function oneDeltaThenParkReaderHangingCancel(signal: AbortSignal) {
        let emitted = false;
        return {
          read: (): Promise<Chunk> =>
            new Promise<Chunk>((resolve, reject) => {
              if (signal.aborted) return reject(abortError());
              if (!emitted) {
                emitted = true;
                const frame = `event: delta\ndata: {"version":1,"ops":[]}\n\n`;
                return resolve({ value: new TextEncoder().encode(frame), done: false });
              }
              signal.addEventListener("abort", () => reject(abortError()), { once: true });
            }),
          cancel: () => {
            cancelCalls.push(1);
            return new Promise(() => {});
          },
        };
      }
      vi.stubGlobal(
        "fetch",
        vi.fn(async (_url: string, opts: { signal: AbortSignal }) => {
          signals.push(opts.signal);
          if (signals.length === 2) {
            expect(signals[0].aborted).toBe(true);
            expect(signals[1]).not.toBe(signals[0]);
          }
          return okResponse(oneDeltaThenParkReaderHangingCancel(opts.signal));
        }),
      );
      const client = makeClient();
      client.start();
      await vi.advanceTimersByTimeAsync(0); // delta → forceReconnect return → hanging cancel
      expect(fetchCalls()).toBe(1);
      expect(cancelCalls.length).toBeGreaterThanOrEqual(1);

      await vi.advanceTimersByTimeAsync(1_500); // ~1s floor; must NOT be stalled by cancel
      expect(fetchCalls()).toBeGreaterThanOrEqual(2);
      expect(signals[0].aborted).toBe(true);

      await Promise.resolve();
      expect(unhandled).toEqual([]);
      client.stop();
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
  });

  // SSE-11: resolving cancel still reconnects; cancel was invoked; no unhandledRejection.
  it("resolving reader.cancel() still reconnects", async () => {
    const unhandled: unknown[] = [];
    const onUnhandled = (e: unknown) => unhandled.push(e);
    process.on("unhandledRejection", onUnhandled);
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    try {
      const signals: AbortSignal[] = [];
      const cancelCalls: number[] = [];
      function oneDeltaThenParkReaderResolvingCancel(signal: AbortSignal) {
        let emitted = false;
        return {
          read: (): Promise<Chunk> =>
            new Promise<Chunk>((resolve, reject) => {
              if (signal.aborted) return reject(abortError());
              if (!emitted) {
                emitted = true;
                const frame = `event: delta\ndata: {"version":1,"ops":[]}\n\n`;
                return resolve({ value: new TextEncoder().encode(frame), done: false });
              }
              signal.addEventListener("abort", () => reject(abortError()), { once: true });
            }),
          cancel: () => {
            cancelCalls.push(1);
            return Promise.resolve();
          },
        };
      }
      vi.stubGlobal(
        "fetch",
        vi.fn(async (_url: string, opts: { signal: AbortSignal }) => {
          signals.push(opts.signal);
          if (signals.length === 2) {
            expect(signals[0].aborted).toBe(true);
            expect(signals[1]).not.toBe(signals[0]);
          }
          return okResponse(oneDeltaThenParkReaderResolvingCancel(opts.signal));
        }),
      );
      const client = makeClient();
      client.start();
      await vi.advanceTimersByTimeAsync(0); // delta → forceReconnect return → cancel resolves
      expect(fetchCalls()).toBe(1);
      expect(cancelCalls.length).toBeGreaterThanOrEqual(1);

      await vi.advanceTimersByTimeAsync(1_500);
      expect(fetchCalls()).toBeGreaterThanOrEqual(2);
      expect(signals[0].aborted).toBe(true);

      await Promise.resolve();
      expect(unhandled).toEqual([]);
      client.stop();
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
  });
});
