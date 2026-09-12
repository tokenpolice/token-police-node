/**
 * A TRANSIENT SSE connect failure (429 / 5xx / any non-terminal 4xx /
 * bodyless-200) used to bare-`return` out of connectAndPump(). A clean return is
 * treated by runForever as a *successful disconnect* → it resets `attempt = 0`,
 * collapsing the exponential backoff ladder to its ~1s floor — the SDK
 * tight-loops reconnecting to /v1/guard/stream ~once a second during a
 * sustained outage (a reconnect storm).
 *
 * Fix (connect gate): the terminal 401/403/404 branch keeps its `return`
 * (no throw), the non-terminal branch now
 * `throw new Error("stream connect " + resp.status)`
 * so the EXISTING runForever catch (stream.ts:132) preserves the climbed
 * `attempt` and the next reconnect climbs the ladder (1→2→4→8…).
 *
 * Second fix (traffic-evidence gate, RC-2): a clean 200-body EOF is ALSO not
 * automatically a healthy disconnect. Some 200 connections die before ever
 * delivering a frame or a keepalive (the reconnect-storm bug seen with
 * Open-WebUI + local collector — the pump never dispatches anything, yet the
 * old code still reset attempt=0 every time, producing a permanent ~1 Hz
 * storm). runForever now resets `attempt = 0` on a clean return ONLY when
 * `this.sawStreamTraffic` is true (stream.ts:124-131) — set at frame-dispatch
 * time or on any SSE comment other than the `: stream-open` banner
 * (stream.ts:293-297, since the banner precedes subscribe/snapshot on every
 * connection including ones about to fail, so it proves nothing). A
 * no-traffic EOF preserves `attempt` and climbs exactly like a transient
 * failure.
 *
 * These tests encode the rubric:
 * - assertion 1: static-source pin of the exact throw message + placement.
 * - assertion 2: terminal branch still contains `return` (not `throw`).
 * - assertion 5: RED→GREEN ladder-climb under a 503 connect storm.
 * - assertion 6: same climb for a 429 (not 503-specific).
 * - assertion 10: static pin that the pump loop still `return`s (no throw added).
 * - assertion 11 (RC-2, updated): a data-less 200 EOF (empty body, or only the
 * stream-open banner) is a failed connect in disguise and MUST climb; a
 * connection that delivers a keepalive comment resets to the ~1s floor.
 * - assertions 12/13: throw stays inside the daemon (no unhandled rejection) and
 * the transient path never sets `stopped`.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { StreamClient } from "../src/stream";

const BASE = {
  baseUrl: "http://localhost:15097",
  apiKey: "tp_sk_test",
  sdkVersion: "test",
  deployment: "daemon",
  clientId: "cid",
  firewall: "enforce",
} as const;

function makeClient(): StreamClient {
  return new StreamClient({ ...BASE });
}

function readSrc(): string {
  const here = dirname(fileURLToPath(import.meta.url));
  return readFileSync(join(here, "..", "src", "stream.ts"), "utf8");
}

describe("Transient SSE connect failures climb the backoff ladder", () => {
  // ---- assertion 1: non-terminal throw present, exact message, in the connect gate ----
  it("stream.ts throws `stream connect <status>` on the non-terminal connect-gate path", () => {
    const src = readSrc();
    expect(src).toContain('throw new Error("stream connect " + resp.status);');

    // The throw lives inside the connect gate `if (!resp.ok || !resp.body)`, AFTER
    // the terminal branch, and BEFORE the reader is obtained.
    const gateIdx = src.indexOf("if (!resp.ok || !resp.body) {");
    const throwIdx = src.indexOf('throw new Error("stream connect " + resp.status);');
    const readerIdx = src.indexOf("(resp.body as any).getReader");
    expect(gateIdx).toBeGreaterThan(-1);
    expect(throwIdx).toBeGreaterThan(gateIdx);
    expect(readerIdx).toBeGreaterThan(throwIdx);
  });

  // ---- assertion 2: terminal branch keeps `return`, not `throw` ----
  it("the terminal 401/403/404 branch still returns (does not throw)", () => {
    const src = readSrc();
    const termStart = src.indexOf("if (resp.status === 401");
    // The terminal if-block body: from the `if` up to the `}` that closes it
    // (right after `this.terminalStatus = resp.status; return;`) — excludes the
    // following non-terminal comment/throw. The terminal branch records the
    // status and returns WITHOUT stopping the loop (it re-probes at the cap).
    const termEnd = src.indexOf("}", src.indexOf("this.terminalStatus = resp.status;", termStart));
    const terminalRegion = src.slice(termStart, termEnd);
    expect(termStart).toBeGreaterThan(-1);
    expect(terminalRegion).toContain("this.terminalStatus = resp.status;");
    expect(terminalRegion).toContain("this.lastEventId = null");
    expect(terminalRegion).toContain("return;");
    expect(terminalRegion).not.toContain("throw");
  });

  // ---- assertion 10: the pump loop still returns (no throw introduced there) ----
  it("the pump while-loop still uses `if (this.forceReconnect) return;` (no throw)", () => {
    const src = readSrc();
    expect(src).toContain("if (this.forceReconnect) return;");
    // Scope to connectAndPump's pump `while` loop (NOT runForever's while) and its
    // `finally`; the connect-gate throw is BEFORE this loop (reader obtained after).
    const pumpMethodIdx = src.indexOf("private async connectAndPump");
    const pumpStart = src.indexOf("while (!this.stopped) {", pumpMethodIdx);
    const pumpEnd = src.indexOf("} finally {", pumpStart);
    expect(pumpStart).toBeGreaterThan(-1);
    expect(pumpEnd).toBeGreaterThan(pumpStart);
    // No throw STATEMENT in the pump loop (a comment saying "no throw" is fine).
    expect(src.slice(pumpStart, pumpEnd)).not.toContain("throw new");
  });

  describe("dynamic ladder-climb (assertions 5/6/11/12/13)", () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });
    afterEach(() => {
      vi.clearAllTimers();
      vi.useRealTimers();
      vi.unstubAllGlobals();
      vi.restoreAllMocks();
    });

    // Collect only the backoff-sleep delays. During a pre-reader connect storm the
    // ONLY setTimeout is the:104 backoff sleep (no reader obtained → no idle
    // timer). With Math.random → 0.5, jitter = base*0.2*(0.5*2-1) = 0, so each
    // delay is exactly steps[attempt]*1000.
    async function drive(
      status: number,
      body: unknown,
      iterations: number,
    ): Promise<{ delays: number[]; fetchCount: () => number }> {
      vi.spyOn(Math, "random").mockReturnValue(0.5);
      const fetchFn = vi.fn(async () => ({ ok: false, status, body }));
      vi.stubGlobal("fetch", fetchFn);
      const setTimeoutSpy = vi.spyOn(globalThis, "setTimeout");

      const client = makeClient();
      client.start();
      for (let i = 0; i < iterations; i++) {
        await Promise.resolve(); // let the rejected connect + catch settle
        await vi.advanceTimersToNextTimerAsync(); // fire exactly one backoff sleep
      }
      const delays = setTimeoutSpy.mock.calls.map((c) => Number(c[1]));
      client.stop();
      return { delays, fetchCount: () => fetchFn.mock.calls.length };
    }

    // ---- assertion 5: 503 connect storm climbs 1000→2000→4000→8000… ----
    it("a sustained 503 (bodyless) connect storm climbs the ladder (RED: all 1000)", async () => {
      const { delays } = await drive(503, null, 6);
      // First few ladder steps, strictly increasing.
      expect(delays[0]).toBe(1000);
      expect(delays[1]).toBe(2000);
      expect(delays[2]).toBe(4000);
      expect(delays[3]).toBe(8000);
      for (let i = 1; i < 5; i++) {
        expect(delays[i]).toBeGreaterThan(delays[i - 1]);
      }
    });

    // ---- assertion 6: 429 climbs identically (not 503-specific) ----
    it("a sustained 429 connect storm also climbs the ladder", async () => {
      const { delays } = await drive(429, null, 4);
      expect(delays[0]).toBe(1000);
      expect(delays[1]).toBe(2000);
      expect(delays[1]).toBeGreaterThan(delays[0]);
    });

    // ---- assertion 13: transient path never sets `stopped` — keeps reconnecting ----
    it("keeps scheduling reconnects during the outage (transient never stops the stream)", async () => {
      const { fetchCount } = await drive(503, null, 5);
      // ≥5 connect attempts landed → the loop was never permanently stopped.
      expect(fetchCount()).toBeGreaterThanOrEqual(5);
    });

    // ---- assertion 12: throw stays inside the daemon (no unhandled rejection) ----
    it("the transient throw never surfaces an unhandled rejection", async () => {
      const unhandled: unknown[] = [];
      const onUnhandled = (e: unknown) => unhandled.push(e);
      process.on("unhandledRejection", onUnhandled);
      try {
        await drive(503, null, 5);
        await Promise.resolve();
        expect(unhandled).toEqual([]);
      } finally {
        process.off("unhandledRejection", onUnhandled);
      }
    });

    // ---- assertion 11 (RC-2, inverted): a data-less clean EOF is a failed
    // connect in disguise and MUST climb, exactly like a transient failure ----
    it("a clean done:true EOF each connect (zero traffic) climbs the ladder", async () => {
      vi.spyOn(Math, "random").mockReturnValue(0.5);
      // Healthy response whose reader immediately reports done, with no frame
      // and no comment ever delivered → sawStreamTraffic stays false →
      // runForever preserves `attempt` instead of resetting it. This is the
      // reconnect-storm bug: every 200 that dies before any traffic used to
      // collapse the ladder to its ~1s floor forever.
      const fetchFn = vi.fn(async () => ({
        ok: true,
        body: { getReader: () => ({ read: () => Promise.resolve({ done: true }) }) },
      }));
      vi.stubGlobal("fetch", fetchFn);
      const setTimeoutSpy = vi.spyOn(globalThis, "setTimeout");

      const client = makeClient();
      client.start();
      for (let i = 0; i < 6; i++) {
        await Promise.resolve();
        await vi.advanceTimersToNextTimerAsync();
      }
      // The ok:true path obtains a reader → arms the idle timer
      // (setTimeout(…, 45000)). Exclude it; the backoff sleeps are the rest.
      const backoffDelays = setTimeoutSpy.mock.calls
        .map((c) => Number(c[1]))
        .filter((d) => d !== 45_000);
      client.stop();

      expect(backoffDelays.length).toBeGreaterThanOrEqual(5);
      expect(backoffDelays[0]).toBe(1000);
      expect(backoffDelays[1]).toBe(2000);
      expect(backoffDelays[2]).toBe(4000);
      expect(backoffDelays[3]).toBe(8000);
      for (let i = 1; i < backoffDelays.length; i++) {
        expect(backoffDelays[i]).toBeGreaterThan(backoffDelays[i - 1]);
      }
    });

    /** A healthy 200 whose reader emits exactly the given raw SSE bytes then
     * EOF. A FRESH closure (and read cursor) is produced on every call, so
     * each reconnect inside the same test gets its own single-shot reader —
     * mirrors streamTerminalStatus.test.ts's snapshotResponse(). */
    function trafficThenEofResponse(raw: string) {
      let sent = false;
      return {
        ok: true,
        body: {
          getReader: () => ({
            read: () =>
              sent
                ? Promise.resolve({ done: true })
                : ((sent = true),
                  Promise.resolve({ done: false, value: new TextEncoder().encode(raw) })),
          }),
        },
      };
    }

    /** Drives `iterations` reconnects of a 2-read body (one comment/frame
     * chunk, then EOF) and returns every recorded backoff-sleep delay
     * (idle-timer 45000 entries excluded). A 2-read body needs one MORE
     * microtask hop than a single-read done:true body before the backoff is
     * scheduled, so each step polls with bounded `advanceTimersByTimeAsync(0)`
     * passes until the expected NEW backoff entry actually lands, rather than
     * guessing a fixed flush count. */
    async function driveTraffic(raw: string, iterations: number): Promise<number[]> {
      vi.spyOn(Math, "random").mockReturnValue(0.5);
      const fetchFn = vi.fn(async () => trafficThenEofResponse(raw));
      vi.stubGlobal("fetch", fetchFn);
      const setTimeoutSpy = vi.spyOn(globalThis, "setTimeout");

      const backoffCount = () => setTimeoutSpy.mock.calls.filter((c) => Number(c[1]) !== 45_000).length;
      const flushUntilBackoffScheduled = async (before: number) => {
        for (let guard = 0; guard < 10 && backoffCount() <= before; guard++) {
          await vi.advanceTimersByTimeAsync(0);
        }
      };

      const client = makeClient();
      client.start();
      await flushUntilBackoffScheduled(0); // connect #1 → first backoff armed
      for (let i = 0; i < iterations; i++) {
        const before = backoffCount();
        await vi.advanceTimersToNextTimerAsync(); // fire the scheduled backoff sleep
        await flushUntilBackoffScheduled(before); // flush the resulting connect → next backoff armed
      }
      const delays = setTimeoutSpy.mock.calls.map((c) => Number(c[1])).filter((d) => d !== 45_000);
      client.stop();
      return delays;
    }

    it("stream-open-comment-only then EOF climbs the ladder (banner ≠ traffic)", async () => {
      // The `: stream-open` banner is written before subscribe/snapshot on
      // EVERY connection, including ones about to fail — it must NOT count as
      // traffic evidence. A connection that only ever emits stream-open
      // before EOF climbs exactly like the zero-traffic case above.
      const delays = await driveTraffic(": stream-open\n", 4);
      expect(delays[0]).toBe(1000);
      expect(delays[1]).toBe(2000);
      expect(delays[2]).toBe(4000);
      for (let i = 1; i < delays.length; i++) {
        expect(delays[i]).toBeGreaterThan(delays[i - 1]);
      }
    });

    it("a `: keepalive` comment then EOF resets every backoff to the 1000ms floor", async () => {
      // Any comment OTHER than stream-open IS traffic evidence — it proves the
      // server pipeline is alive even though no data frame was ever sent
      // (e.g. resume-at-head with nothing new to deliver). Every reconnect
      // after a keepalive-then-EOF disconnect stays at the ~1s floor.
      const delays = await driveTraffic(": keepalive\n", 4);
      expect(delays.length).toBeGreaterThanOrEqual(4);
      expect(delays.every((d) => d === 1000)).toBe(true);
    });
  });
});
