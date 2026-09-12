/**
 * Node SSE `reconnectCapSeconds` had no 60s floor — a reconnect-storm
 * parity gap vs the Python reference.
 *
 * Python clamps the cap at construction: `max(60, int(reconnect_cap_seconds or
 * 300))` (token_police/stream.py:46). The Node port honored a caller-supplied
 * `reconnectCapSeconds` verbatim (stream.ts:45-50), so a config < 60s let Node
 * reconnect to /v1/guard/stream faster than 60s indefinitely during a sustained
 * outage — a reconnect storm the Python SDK cannot produce.
 *
 * Fix (constructor only): one additive line
 * `this.opts.reconnectCapSeconds = Math.max(60, this.opts.reconnectCapSeconds || 300);`
 * immediately after the opts spread, before the ladder is ever read.
 *
 * These tests encode the rubric:
 * - assertion 1: static-source pin of the exact clamp line.
 * - assertion 2: exactly ONE `Math.max(60, ...)`, in the constructor body.
 * - assertion 3 / 8 / 9 / 10: the behavioral-equivalence table (Node ≡ Python).
 * - assertion 15: the dynamic backoff-timing test with teeth — drives the real
 * runForever() loop to ladder saturation and asserts the scheduled reconnect
 * delay is exactly 60_000ms (an unclamped cap=5 build would schedule 5_000ms).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { StreamClient } from "../src/stream";

/** Base client fields (no reconnectCapSeconds — added per-case). */
const BASE = {
  baseUrl: "http://localhost:15098",
  apiKey: "tp_sk_test",
  sdkVersion: "test",
  deployment: "daemon",
  clientId: "cid",
  firewall: "enforce",
} as const;

/** Read the private, now-clamped effective cap. */
function effectiveCap(client: StreamClient): number {
  return (client as unknown as { opts: { reconnectCapSeconds: number } }).opts.reconnectCapSeconds;
}

describe("SSE reconnectCapSeconds 60s floor (Node↔Python parity)", () => {
  // ---- assertion 1: exact clamp line, exact spot (static source pin) ----
  it("stream.ts contains the exact clamp line, after the opts spread", () => {
    const here = dirname(fileURLToPath(import.meta.url));
    const src = readFileSync(join(here, "..", "src", "stream.ts"), "utf8");

    // The exact clamp line is present verbatim.
    expect(src).toContain(
      "this.opts.reconnectCapSeconds = Math.max(60, this.opts.reconnectCapSeconds || 300);",
    );

    // It appears AFTER the opts object literal (the spread) and BEFORE the
    // ladder read in runForever (`const steps = [`).
    const spreadIdx = src.indexOf("reconnectCapSeconds: 300,");
    const clampIdx = src.indexOf(
      "this.opts.reconnectCapSeconds = Math.max(60, this.opts.reconnectCapSeconds || 300);",
    );
    const ladderIdx = src.indexOf("const steps = [");
    expect(spreadIdx).toBeGreaterThan(-1);
    expect(clampIdx).toBeGreaterThan(spreadIdx);
    expect(ladderIdx).toBeGreaterThan(clampIdx);
  });

  // ---- assertion 2: clamp runs at construction, exactly once, not in the loop ----
  it("has exactly one Math.max(60, ...) clamp, inside the constructor (not runForever)", () => {
    const here = dirname(fileURLToPath(import.meta.url));
    const src = readFileSync(join(here, "..", "src", "stream.ts"), "utf8");

    const clamps = src.match(/Math\.max\(60,/g) ?? [];
    expect(clamps.length).toBe(1);

    // The single clamp lives in the constructor body (before start()), not in
    // runForever's loop body.
    const ctorBody = src.slice(src.indexOf("constructor("), src.indexOf("start(): void"));
    expect(ctorBody).toContain("Math.max(60,");

    const loopBody = src.slice(
      src.indexOf("private async runForever"),
      src.indexOf("private async connectAndPump"),
    );
    expect(loopBody).not.toContain("Math.max(60,");
  });

  // ---- assertion 3 / 8 / 9 / 10: behavioral-equivalence table (Node ≡ Python) ----
  // Each row: input reconnectCapSeconds → expected effective cap after clamp.
  const TABLE: Array<{ input: number | undefined; expected: number; note: string }> = [
    { input: undefined, expected: 300, note: "omitted → default 300 (assertion 8)" },
    { input: 0, expected: 300, note: "0 falsy → 300 (assertion 8)" },
    { input: 5, expected: 60, note: "5 < 60 → floor 60 (assertion 3)" },
    { input: 59, expected: 60, note: "59 < 60 → floor 60 (assertion 3)" },
    { input: 60, expected: 60, note: "60 → unchanged (assertion 9 boundary)" },
    { input: 120, expected: 120, note: "120 ≥ 60 → unchanged (assertion 9)" },
    { input: 300, expected: 300, note: "300 → unchanged (assertion 9)" },
    { input: 900, expected: 900, note: "900 large → unchanged (assertion 9)" },
    { input: -5, expected: 60, note: "-5 truthy → max(60,-5)=60 (assertion 10)" },
    { input: NaN, expected: 300, note: "NaN falsy → 300 (assertion 10, degrades safely)" },
  ];

  for (const { input, expected, note } of TABLE) {
    it(`effective cap for ${String(input)} is ${expected} — ${note}`, () => {
      const client =
        input === undefined
          ? new StreamClient({ ...BASE })
          : new StreamClient({ ...BASE, reconnectCapSeconds: input });
      expect(effectiveCap(client)).toBe(expected);
    });
  }

  // ---- assertion 15: dynamic backoff-timing test (the one with teeth) ----
  // Drive the REAL runForever() loop to ladder saturation with a fetch that
  // throws (transient network error → backoff path), and assert the scheduled
  // reconnect delay at the terminal step is exactly 60_000ms.
  //
  // Formula (stream.ts:78/88-90): steps = [1,2,4,8,16,32,64,128,256, cap]
  // (len 10, terminal idx 9). `attempt` reads steps[min(attempt,9)] at:88 then
  // increments at:91, so steps[9]=cap is FIRST read on the 10th consecutive
  // failed connect. With Math.random stubbed → 0.5, jitter = base*0.2*(0.5*2-1)
  // = 0 (no jitter band → exact integer). Clamped cap 5 → 60:
  // sleepMs = max(1000, min(60*1000, 60*1000)) = 60_000 (exact).
  // Unclamped cap 5 would give max(1000, min(5000, 5000)) = 5_000 → RED.
  describe("dynamic backoff timing (assertion 15)", () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });
    afterEach(() => {
      vi.clearAllTimers();
      vi.useRealTimers();
      vi.unstubAllGlobals();
      vi.restoreAllMocks();
    });

    it("schedules a 60_000ms reconnect at ladder saturation with cap=5 (unclamped would be 5000)", async () => {
      // Deterministic: jitter term becomes exactly 0.
      vi.spyOn(Math, "random").mockReturnValue(0.5);

      // A THROWING fetch → connectAndPump() rejects → runForever catch → the
      // transient backoff path (:88-95). Never a terminal 401/403/404, so the
      // Stop path (:130-133) is not taken and no reader is obtained, so the
      // idle-read timer (:161) never arms — the ONLY setTimeout during this
      // connect storm is the:94 backoff sleep.
      vi.stubGlobal(
        "fetch",
        vi.fn(async () => {
          throw new Error("econnrefused");
        }),
      );

      // Capture every backoff-sleep delay.
      const setTimeoutSpy = vi.spyOn(globalThis, "setTimeout");

      const client = new StreamClient({ ...BASE, reconnectCapSeconds: 5 });
      client.start();

      // Drive ≥10 consecutive failed connects so `attempt` saturates at the
      // terminal index 9 (steps[9] = cap). advanceTimersToNextTimerAsync fires
      // EXACTLY the single pending backoff sleep (no cascade / no over-advance),
      // provoking the next failed connect + next scheduled sleep.
      for (let i = 0; i < 12; i++) {
        await Promise.resolve(); // let the rejected fetch + catch settle
        await vi.advanceTimersToNextTimerAsync(); // fire exactly one backoff sleep
      }

      // The saturated-step delay (terminal ladder value) is the last backoff
      // sleep scheduled — must be exactly 60_000. unclamped cap=5 would schedule
      // 5000ms here → RED. (setTimeout is the ONLY timer during a connect storm.)
      const delays = setTimeoutSpy.mock.calls.map((c) => Number(c[1]));
      const saturated = delays[delays.length - 1];
      expect(saturated).toBe(60_000);

      // Sanity: every recorded backoff-sleep delay is a finite ms.
      expect(delays.every((d) => Number.isFinite(d))).toBe(true);

      client.stop();
    });

    it("under the fixed 60s cap, advancing 5000ms after saturation does NOT reconnect (teeth)", async () => {
      // A second discriminator that does not read the private delay: after
      // saturation the sleep is 60_000ms, so advancing only 5_000ms must not
      // fire the next connect. An unclamped cap=5 build would have already
      // reconnected at 5_000ms.
      vi.spyOn(Math, "random").mockReturnValue(0.5);
      const fetchFn = vi.fn(async () => {
        throw new Error("econnrefused");
      });
      vi.stubGlobal("fetch", fetchFn);

      const client = new StreamClient({ ...BASE, reconnectCapSeconds: 5 });
      client.start();

      // Saturate the ladder: fire exactly one pending timer per iteration so we
      // land cleanly on a single fresh 60s backoff sleep (no cascade).
      for (let i = 0; i < 12; i++) {
        await Promise.resolve();
        await vi.advanceTimersToNextTimerAsync();
      }
      await Promise.resolve(); // ensure the next 60s sleep is scheduled
      const countAtSaturation = fetchFn.mock.calls.length;

      // Now we sit on a fresh 60s backoff sleep. Advancing 5_000ms must NOT
      // provoke a new connect (unclamped cap=5 would have already reconnected).
      await vi.advanceTimersByTimeAsync(5_000);
      expect(fetchFn.mock.calls.length).toBe(countAtSaturation);

      // Advancing the remaining time past 60_000ms DOES reconnect.
      await vi.advanceTimersByTimeAsync(55_000);
      expect(fetchFn.mock.calls.length).toBeGreaterThan(countAtSaturation);

      client.stop();
    });
  });
});
