/**
 * An expired Decision Pack (TTL stamped on the snapshot by the server but
 * never enforced) kept enforcing while heartbeats arrived but no snapshot/delta
 * refreshed it. Fix (state + stream.ts pump watchdog, REUSING the flag): the
 * pump watchdog, after the heartbeat/stale checks, calls `state.isPackExpired()`
 * → `invalidatePack()`, nulls the reconnect cursor, and sets the EXISTING
 * `forceReconnect` flag. The loop-top guard returns → the EXISTING backoff
 * reconnects WITHOUT Last-Event-ID → fresh snapshot → pack un-expires. ZERO new
 * setTimeout/backoff/fetch/socket code is added.
 *
 * These tests reuse the fake-transport harness under fake timers: the only
 * thing that moves the loop is the injected reader, so the reconnect is
 * deterministic. Assertions 9, 11, 12 (Node side); no 24h sleep.
 * Sibling: token-police-python/tests/test_snapshot_ttl.py.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { readFileSync } from "fs";
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

/** A reader that parks forever (rejects AbortError if the signal fires). */
function parkReader(signal: AbortSignal) {
  return {
    read: (): Promise<Chunk> =>
      new Promise<Chunk>((_resolve, reject) => {
        if (signal.aborted) return reject(abortError());
        signal.addEventListener("abort", () => reject(abortError()), { once: true });
      }),
  };
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

function fetchMock() {
  return globalThis.fetch as unknown as {
    mock: { calls: [string, { headers: Record<string, string>; signal: AbortSignal }][] };
  };
}
function fetchCalls(): number {
  return fetchMock().mock.calls.length;
}

const snapshotFrame = (ttl?: number) =>
  `event: snapshot\nid: 1\ndata: ${JSON.stringify({
    version: 1, tenant_id: "t1", project_id: "p1", ttl_seconds: ttl,
    directives: [], loop_blocks: [],
  })}\n\n`;

describe("Snapshot TTL watchdog reconnect (Node)", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(1_700_000_000_000);
    state.resetPack();
  });
  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    state.resetPack();
  });

  // Assertions 9 + 11: an expired pack forces a reconnect that omits
  // Last-Event-ID (fresh snapshot). Heartbeats keep the heartbeat/stale
  // watchdogs quiet so ONLY the TTL branch can fire.
  it("expired pack forces a reconnect that drops Last-Event-ID (#9,#11)", async () => {
    const c1 = { push: null as null | ((t: string) => void) };
    const signals: AbortSignal[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, opts: { signal: AbortSignal }) => {
        signals.push(opts.signal);
        if (fetchCalls() === 1) {
          const c = controllableReader(opts.signal);
          c1.push = c.push;
          return okResponse(c.reader);
        }
        expect(signals[0].aborted).toBe(true); // N dead before N+1 starts
        return okResponse(parkReader(opts.signal)); // conn2 parks
      }),
    );
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0); // conn1 fetch, pump awaiting read

    // Snapshot with a 50s TTL lands and sets Last-Event-ID = 1.
    c1.push!(snapshotFrame(50));
    await vi.advanceTimersByTimeAsync(0);
    expect(state.getPackVersion()).toBe(1);
    expect(state.isCacheHealthy()).toBe(true);

    // Heartbeat at +40s keeps the heartbeat watchdog quiet (40 < 45s); pack age
    // 40s < 50s TTL, not yet expired.
    await vi.advanceTimersByTimeAsync(40_000);
    c1.push!(":ka\n");
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchCalls()).toBe(1); // still connected

    // Heartbeat at +80s: heartbeat watchdog still quiet (40s since last byte),
    // but pack age 80s > 50s TTL → the branch sets forceReconnect (it does
    // NOT return — it reuses the loop-top guard, consumed on the NEXT loop).
    await vi.advanceTimersByTimeAsync(40_000);
    c1.push!(":ka\n");
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchCalls()).toBe(1); // still parked at read() — guard not yet hit

    // Next heartbeat drives one more loop iteration → the loop-top guard
    // (`if (this.forceReconnect) return;`) fires → pump returns.
    c1.push!(":ka\n");
    await vi.advanceTimersByTimeAsync(0);

    await vi.advanceTimersByTimeAsync(1_500); // > the 1000ms min backoff
    expect(fetchCalls()).toBe(2); // reconnected via the EXISTING backoff
    const reconnectHeaders = fetchMock().mock.calls[1][1].headers;
    expect(reconnectHeaders["Last-Event-ID"]).toBeUndefined(); // fresh snapshot
    client.stop();
  });

  // Assertion 12: a no-TTL (legacy) snapshot never expires — no reconnect no
  // matter how much time passes with heartbeats.
  it("no-TTL snapshot never triggers the TTL reconnect (#12)", async () => {
    const c1 = { push: null as null | ((t: string) => void) };
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, opts: { signal: AbortSignal }) => {
        const c = controllableReader(opts.signal);
        c1.push = c.push;
        return okResponse(c.reader);
      }),
    );
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0);
    c1.push!(snapshotFrame(undefined)); // no ttl_seconds
    await vi.advanceTimersByTimeAsync(0);
    expect(state.getPackVersion()).toBe(1);

    // Pump many heartbeats over a long span; the pack never expires.
    for (let i = 0; i < 5; i++) {
      await vi.advanceTimersByTimeAsync(40_000);
      c1.push!(":ka\n");
      await vi.advanceTimersByTimeAsync(0);
    }
    expect(fetchCalls()).toBe(1); // never reconnected — byte-identical to today
    expect(state.isCacheHealthy()).toBe(true);
    client.stop();
  });

  // Assertion 9 (static pin): the ONLY thing added to stream.ts is the
  // isPackExpired branch — no new setTimeout/fetch/backoff/socket code.
  it("stream.ts adds only the expiry flag branch — no new reconnect code (#9)", () => {
    const src = readFileSync(new URL("../src/stream.ts", import.meta.url), "utf8");
    // The branch exists and reuses the flag + cursor-null only.
    expect(src).toContain("state.isPackExpired()");
    expect(src).toContain('state.invalidatePack("snapshot_ttl_expired")');
    // Extract the branch body and prove it contains no new machinery.
    const idx = src.indexOf("if (state.isPackExpired())");
    const branch = src.slice(idx, src.indexOf("}", idx) + 1);
    expect(branch).toContain("this.lastEventId = null");
    expect(branch).toContain("this.forceReconnect = true");
    expect(branch).not.toContain("setTimeout");
    expect(branch).not.toContain("fetch");
    expect(branch).not.toMatch(/\bnew Promise\b/);
    // exactly one fetch call site in the whole file (the connect) — no new one
    expect((src.match(/await fetch\(/g) || []).length).toBe(1);
    expect((src.match(/setTimeout\(/g) || []).length).toBe(2); // idleTimer + backoff wakeTimer (pre-existing)
  });
});
