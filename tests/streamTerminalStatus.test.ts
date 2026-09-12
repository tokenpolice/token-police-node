/**
 * A terminal SSE connect status (revoked/invalid key → 401/403, or a
 * misconfigured base_url/route → 404) must NOT permanently stop the rule-stream
 * reader. A single transient 404 from a load-balancer blip during a rolling
 * deploy, or a 401/403 from server-side key-cache staleness, would otherwise kill
 * the fast path for the whole process lifetime (every call degrading to inline
 * /check forever).
 *
 * Contract:
 * - A terminal status invalidates the pack (→ inline /check, fail-open), records
 * the status, and returns WITHOUT stopping the loop.
 * - The outer loop re-probes at the reconnect cap (the ladder TOP, ~one probe
 * per cap period) instead of resetting to the ~1s backoff floor — no hammering.
 * - A later non-terminal connect resumes normal streaming; a healthy 200 with a
 * fresh snapshot self-heals the pack with no manual restart. Terminal also
 * drops Last-Event-ID (SSE-12) so the cap re-probe cannot resume-at-head —
 * collector skip-when-equal would otherwise withhold the snapshot.
 * - stop() still wins instantly at every point; manual stop()→start() unchanged.
 *
 * Node emits no logging on this path (a deliberate residual — the SDK has no
 * logging mechanism); the streak-warning assertion lives only in the Python
 * sibling token-police-python/tests/test_stream_terminal.py.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { StreamClient } from "../src/stream";
import * as state from "../src/state";

const CAP_MS = 300_000; // default reconnectCapSeconds (300) in ms

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

function stoppedOf(client: StreamClient): boolean {
  return (client as unknown as { stopped: boolean }).stopped;
}

function loopPromiseOf(client: StreamClient): Promise<void> | null {
  return (client as unknown as { loopPromise: Promise<void> | null }).loopPromise;
}

function lastEventIdOf(client: StreamClient): number | null {
  return (client as unknown as { lastEventId: number | null }).lastEventId;
}

function setLastEventId(client: StreamClient, id: number | null): void {
  (client as unknown as { lastEventId: number | null }).lastEventId = id;
}

function headersOf(callIdx: number): Record<string, string> {
  const calls = (globalThis.fetch as unknown as {
    mock: { calls: [unknown, { headers: Record<string, string> }][] };
  }).mock.calls;
  return calls[callIdx][1].headers;
}

const SNAPSHOT = { version: 1, tenant_id: "t1", project_id: "p1", directives: [], loop_blocks: [] };

/** A connect-gate response with a chosen status/ok/body — mirrors the shape the
 * gate inspects (`resp.ok`, `resp.status`, `resp.body`). */
function gateResponse(ok: boolean, status: number, body: unknown = null) {
  return { ok, status, body };
}

/** A healthy 200 whose reader emits one snapshot frame then EOF. */
function snapshotResponse() {
  const frame = `event: snapshot\nid: 1\ndata: ${JSON.stringify(SNAPSHOT)}\n\n`;
  let sent = false;
  return {
    ok: true,
    status: 200,
    body: {
      getReader: () => ({
        read: () =>
          sent
            ? Promise.resolve({ done: true })
            : ((sent = true),
              Promise.resolve({ done: false, value: new TextEncoder().encode(frame) })),
      }),
    },
  };
}

/** Collector resume-at-head: 200 + keepalive, no snapshot. */
function keepaliveOnlyResponse() {
  let sent = false;
  return {
    ok: true,
    status: 200,
    body: {
      getReader: () => ({
        read: () =>
          sent
            ? Promise.resolve({ done: true })
            : ((sent = true),
              Promise.resolve({
                done: false,
                value: new TextEncoder().encode(": keepalive\n"),
              })),
      }),
    },
  };
}

describe("terminal SSE connect status → re-probe, never a permanent stop", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.spyOn(Math, "random").mockReturnValue(0.5); // zero jitter → exact cap delays
    state.resetPack();
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    state.resetPack();
  });

  // A terminal 401 does NOT stop the loop; the next probe is scheduled at the cap
  // (ladder top), so no reconnect fires before the cap elapses, then exactly one.
  it("401 → does NOT stop; re-probes at the reconnect cap, not the 1s floor", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => gateResponse(false, 401, null)));
    const setTimeoutSpy = vi.spyOn(globalThis, "setTimeout");
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0); // connect #1 → terminal
    expect(fetchCalls()).toBe(1);
    expect(stoppedOf(client)).toBe(false); // RED (old): 401 stopped the loop

    // The scheduled backoff is at the cap — several seconds must NOT reconnect.
    const backoff = setTimeoutSpy.mock.calls.map((c) => Number(c[1])).filter((d) => d !== 45_000);
    expect(backoff[0]).toBe(CAP_MS);
    await vi.advanceTimersByTimeAsync(5_000);
    expect(fetchCalls()).toBe(1);

    // Only after the cap elapses does exactly one re-probe fire.
    await vi.advanceTimersByTimeAsync(CAP_MS);
    expect(fetchCalls()).toBe(2);
    client.stop();
  });

  // SSE-11 all-exits: a terminal 401 return still aborts this connect's signal
  // (no reader obtained). Does not throw, does not stop, still re-probes at the
  // cap. Drops Last-Event-ID (SSE-12) so the re-probe cannot resume-at-head.
  it("401 terminal return still aborts the signal and does not throw", async () => {
    const unhandled: unknown[] = [];
    const onUnhandled = (e: unknown) => unhandled.push(e);
    process.on("unhandledRejection", onUnhandled);
    try {
      expect(state.applySnapshot(SNAPSHOT)).toBe(true);
      expect(state.isCacheHealthy()).toBe(true);

      const signals: AbortSignal[] = [];
      vi.stubGlobal(
        "fetch",
        vi.fn(async (_url: string, opts: { signal: AbortSignal }) => {
          signals.push(opts.signal);
          if (signals.length === 2) {
            expect(signals[0].aborted).toBe(true);
            expect(signals[1]).not.toBe(signals[0]);
          }
          return gateResponse(false, 401, null);
        }),
      );
      const client = makeClient();
      setLastEventId(client, 10);
      expect(() => client.start()).not.toThrow();
      await vi.advanceTimersByTimeAsync(0);
      expect(fetchCalls()).toBe(1);
      expect(signals[0].aborted).toBe(true);
      expect(stoppedOf(client)).toBe(false);
      expect(state.getPack()).toBeNull();
      expect(state.isCacheHealthy()).toBe(false);
      expect(lastEventIdOf(client)).toBeNull();

      await vi.advanceTimersByTimeAsync(5_000);
      expect(fetchCalls()).toBe(1); // cap re-probe, not the 1s floor

      await vi.advanceTimersByTimeAsync(CAP_MS);
      expect(fetchCalls()).toBe(2);
      expect(headersOf(1)["Last-Event-ID"]).toBeUndefined();
      expect(signals[1].aborted).toBe(true);
      expect(stoppedOf(client)).toBe(false);

      await Promise.resolve();
      expect(unhandled).toEqual([]);
      client.stop();
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
  });

  // 403 and 404 behave identically to 401 (terminal set parity).
  it.each([403, 404])("%d → does NOT stop; re-probes at the cap (terminal set parity)", async (status) => {
    vi.stubGlobal("fetch", vi.fn(async () => gateResponse(false, status, null)));
    const client = makeClient();
    setLastEventId(client, 10);
    client.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchCalls()).toBe(1);
    expect(stoppedOf(client)).toBe(false);
    expect(lastEventIdOf(client)).toBeNull();
    await vi.advanceTimersByTimeAsync(5_000);
    expect(fetchCalls()).toBe(1); // no hammering before the cap
    await vi.advanceTimersByTimeAsync(CAP_MS);
    expect(fetchCalls()).toBe(2);
    expect(headersOf(1)["Last-Event-ID"]).toBeUndefined();
    client.stop();
  });

  // A terminal status STILL invalidates a previously-healthy pack (→ inline
  // /check, fail-open), but the reader is NOT stopped.
  it("401 invalidates a healthy pack (getPack null / unhealthy) yet does NOT stop", async () => {
    expect(state.applySnapshot(SNAPSHOT)).toBe(true);
    expect(state.isCacheHealthy()).toBe(true);

    vi.stubGlobal("fetch", vi.fn(async () => gateResponse(false, 401, null)));
    const client = makeClient();
    setLastEventId(client, 10);
    client.start();
    await vi.advanceTimersByTimeAsync(0);

    expect(state.getPack()).toBeNull();
    expect(state.isCacheHealthy()).toBe(false);
    expect(lastEventIdOf(client)).toBeNull();
    expect(stoppedOf(client)).toBe(false);
    client.stop();
  });

  // Two consecutive terminals both schedule their re-probe at the cap (the streak
  // stays at the ladder top — no drift back toward the 1s floor).
  it("two consecutive terminals → both re-probe at the cap (streak stays at ladder top)", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => gateResponse(false, 401, null)));
    const setTimeoutSpy = vi.spyOn(globalThis, "setTimeout");
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0); // connect #1 terminal → backoff at cap
    await vi.advanceTimersByTimeAsync(CAP_MS); // fire re-probe → connect #2 terminal
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchCalls()).toBe(2);

    const backoff = setTimeoutSpy.mock.calls.map((c) => Number(c[1])).filter((d) => d !== 45_000);
    expect(backoff.slice(0, 2)).toEqual([CAP_MS, CAP_MS]);
    client.stop();
  });

  // The full self-healing story: terminal → re-probe at the cap → server now
  // returns 200 with a fresh snapshot → the pack heals with NO manual restart.
  // Mock mirrors collector skip-when-equal: Last-Event-ID present → keepalives
  // only (no snapshot). Pre-seed a cursor so a leftover bookmark cannot fake-heal.
  it("terminal → re-probe → 200 fresh snapshot → pack self-heals (no manual restart)", async () => {
    expect(state.applySnapshot(SNAPSHOT)).toBe(true);
    vi.stubGlobal("fetch", vi.fn(async (_url: string, opts: { headers: Record<string, string> }) => {
      if (fetchCalls() === 1) return gateResponse(false, 401, null);
      if (opts.headers["Last-Event-ID"]) return keepaliveOnlyResponse();
      return snapshotResponse();
    }));
    const client = makeClient();
    setLastEventId(client, 10);
    client.start();
    await vi.advanceTimersByTimeAsync(0); // connect #1 → terminal → invalidated
    expect(state.isCacheHealthy()).toBe(false);
    expect(lastEventIdOf(client)).toBeNull();
    expect(fetchCalls()).toBe(1);

    await vi.advanceTimersByTimeAsync(CAP_MS); // cap elapses → re-probe → 200 snapshot
    await vi.advanceTimersByTimeAsync(0); // flush the snapshot read
    expect(fetchCalls()).toBe(2);
    expect(headersOf(1)["Last-Event-ID"]).toBeUndefined();
    expect(state.isCacheHealthy()).toBe(true); // healed with no manual restart
    expect(state.getPack()).not.toBeNull();
    client.stop();
  });

  // stop() wins instantly even while a terminal status keeps the loop alive.
  it("manual stop() wins instantly (loop exits, no reconnect ever fires)", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => gateResponse(false, 401, null)));
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0); // connect #1 terminal → waiting at cap
    expect(stoppedOf(client)).toBe(false);

    client.stop(); // customer shutdown
    expect(stoppedOf(client)).toBe(true);
    await vi.advanceTimersByTimeAsync(0); // wake settles, loop exits
    expect(loopPromiseOf(client)).toBeNull();

    await vi.advanceTimersByTimeAsync(2 * CAP_MS); // no reconnect ever, even past the cap
    expect(fetchCalls()).toBe(1);
  });

  // After a manual stop the instance stays restartable.
  it("restartable after a manual stop (fresh start() relaunches, fetch resumes)", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => gateResponse(false, 401, null)));
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0); // connect #1 terminal
    expect(fetchCalls()).toBe(1);

    client.stop();
    await vi.advanceTimersByTimeAsync(0); // loop settles
    expect(loopPromiseOf(client)).toBeNull();

    client.start(); // relaunch
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchCalls()).toBe(2); // reconnected after restart
    client.stop();
  });

  // Transient statuses are NOT terminal: the existing return→retry path is
  // preserved (reconnects at the 1s floor) and the loop is NOT stopped.
  it("500 → still reconnects, does NOT give up (transient path)", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => gateResponse(false, 500, null)));
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchCalls()).toBe(1);
    await vi.advanceTimersByTimeAsync(3_000); // > the 1s backoff floor
    expect(fetchCalls()).toBeGreaterThanOrEqual(2);
    expect(stoppedOf(client)).toBe(false);
    client.stop();
  });

  it("500 does NOT invalidate a healthy pack", async () => {
    expect(state.applySnapshot(SNAPSHOT)).toBe(true);
    vi.stubGlobal("fetch", vi.fn(async () => gateResponse(false, 500, null)));
    const client = makeClient();
    setLastEventId(client, 10);
    client.start();
    await vi.advanceTimersByTimeAsync(0);
    await vi.advanceTimersByTimeAsync(3_000);
    expect(state.getPack()).not.toBeNull();
    expect(state.isCacheHealthy()).toBe(true);
    expect(lastEventIdOf(client)).toBe(10); // transient path keeps the bookmark
    client.stop();
  });

  it("network error (fetch rejects) → still reconnects, never stops", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("ECONNREFUSED"); }));
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchCalls()).toBe(1);
    await vi.advanceTimersByTimeAsync(3_000);
    expect(fetchCalls()).toBeGreaterThanOrEqual(2);
    expect(stoppedOf(client)).toBe(false);
    client.stop();
  });

  it("bodyless-200 → does NOT give up (falls through to transient retry)", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => gateResponse(true, 200, null)));
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchCalls()).toBe(1);
    await vi.advanceTimersByTimeAsync(3_000);
    expect(fetchCalls()).toBeGreaterThanOrEqual(2);
    expect(stoppedOf(client)).toBe(false);
    client.stop();
  });
});
