/**
 * Per-call observation attribution (state.ts keyed queue).
 *
 * The observations queue tags every push with the pushing call's minted obs
 * key (AsyncLocalStorage scope opened by each enforcer wrapper) and a /log
 * drain claims only its own call's entries + untagged entries + stale
 * orphans. These tests pin the state-layer contract:
 *  - keyed isolation (a drain with key Y never steals key X's entries)
 *  - staleness fallback (an orphan past OBS_STALE_MS ships on any drain)
 *  - untagged fallback (no-scope pushes ship on the next keyed drain)
 *  - legacy no-arg drain-all (shutdown paths / older tests)
 *  - greedy flush mode (forceFlushTokenPoliceSpans window drains everything)
 *  - greedy windows are DEPTH-COUNTED: overlapping terminal flushes never
 *    close each other early, and the per-request SpanProcessor.forceFlush is
 *    never greedy
 *  - concurrency: overlapping scoped "calls" each drain exactly their own
 *    observation — the cross-trace-theft incident shape.
 */
import { describe, test, expect, beforeEach } from "vitest";

import {
  resetPack,
  pushObservation,
  drainObservations,
  runWithCallObsScope,
  runWithObsKey,
  getCurrentObsKey,
  newObsKey,
  setObservationsDrainAll,
  beginObservationsDrainAll,
  endObservationsDrainAll,
  OBS_STALE_MS,
  _observationsSet,
} from "../src/state";
import { forceFlushTokenPoliceSpans, TokenPoliceSpanProcessor } from "../src/telemetry";

const obsA = { rule_id: "rA", outcome: "would_block", mode: "dry_run" };
const obsB = { rule_id: "rB", outcome: "would_block", mode: "dry_run" };

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

beforeEach(() => {
  resetPack(); // clears the queue AND the greedy flag
});

describe("keyed isolation + staleness", () => {
  test("a drain with key Y does NOT claim key X's fresh entries; past the stale window it does", () => {
    // Fresh entry tagged X → a Y-keyed drain must leave it queued.
    _observationsSet([{ obs: obsA, key: "X", ts: Date.now() }]);
    expect(drainObservations("Y")).toEqual([]);

    // Age X past the stale window (inject ts via the test seam) → the same
    // Y drain now ships it: an orphan whose owning /log never fired must
    // eventually leave the process (loss is worse than late delivery).
    _observationsSet([{ obs: obsA, key: "X", ts: Date.now() - OBS_STALE_MS - 1000 }]);
    expect(drainObservations("Y")).toEqual([obsA]);
    // Queue is empty afterwards.
    expect(drainObservations()).toEqual([]);
  });

  test("a keyed drain claims exactly its own key + untagged, preserving order of the rest", () => {
    const now = Date.now();
    _observationsSet([
      { obs: obsA, key: "X", ts: now },
      { obs: { tag: "untagged" }, key: null, ts: now },
      { obs: obsB, key: "Y", ts: now },
    ]);
    expect(drainObservations("X")).toEqual([obsA, { tag: "untagged" }]);
    // Y's entry survived for its own drain.
    expect(drainObservations("Y")).toEqual([obsB]);
  });

  test("explicit null (unknown claimant) claims only untagged + stale", () => {
    const now = Date.now();
    _observationsSet([
      { obs: obsA, key: "X", ts: now },
      { obs: { tag: "untagged" }, key: null, ts: now },
      { obs: obsB, key: "Y", ts: now - OBS_STALE_MS - 1000 },
    ]);
    expect(drainObservations(null)).toEqual([{ tag: "untagged" }, obsB]);
    // X's fresh tagged entry is still queued.
    expect(drainObservations("X")).toEqual([obsA]);
  });
});

describe("untagged fallback", () => {
  test("a push outside any call scope ships on the next keyed drain", () => {
    pushObservation(obsA); // no ALS scope active → untagged
    expect(drainObservations("some-other-call")).toEqual([obsA]);
  });
});

describe("no-arg drain-all (legacy)", () => {
  test("returns everything regardless of keys", () => {
    const now = Date.now();
    _observationsSet([
      { obs: obsA, key: "X", ts: now },
      { obs: obsB, key: "Y", ts: now },
      { obs: { tag: "untagged" }, key: null, ts: now },
    ]);
    expect(drainObservations()).toEqual([obsA, obsB, { tag: "untagged" }]);
    expect(drainObservations()).toEqual([]);
  });
});

describe("greedy flush mode", () => {
  test("setObservationsDrainAll(true) makes ANY drain return everything", () => {
    const now = Date.now();
    _observationsSet([
      { obs: obsA, key: "X", ts: now },
      { obs: obsB, key: "Y", ts: now },
    ]);
    setObservationsDrainAll(true);
    try {
      expect(drainObservations("Z")).toEqual([obsA, obsB]);
    } finally {
      setObservationsDrainAll(false);
    }
  });

  test("forceFlushTokenPoliceSpans clears the greedy flag afterwards", async () => {
    await forceFlushTokenPoliceSpans();
    // After the flush window, keyed drains are selective again.
    _observationsSet([{ obs: obsA, key: "X", ts: Date.now() }]);
    expect(drainObservations("Y")).toEqual([]);
    expect(drainObservations("X")).toEqual([obsA]);
  });
});

/**
 * Regression: the greedy window used to be a global BOOLEAN cleared in a
 * finally. Two overlapping flush windows (the beforeExit fire-and-forget
 * flush() racing a user's awaited close(); two concurrent close() calls) let
 * the FIRST window's finally clear the flag while the second was still open —
 * its deferred drains went back to keyed mid-flush and stranded every other
 * call's tagged entry (silent audit loss). And SpanProcessor.forceFlush — a
 * PUBLIC per-request OTel API platform integrations call on live traffic —
 * opened a greedy window, reopening cross-call stealing on every request.
 * Fixed with a depth counter + a non-greedy forceFlush.
 *
 * A fresh X-keyed entry probed with a Y drain is the greedy detector: greedy →
 * claimed, keyed → left queued.
 */
describe("greedy window depth counting", () => {
  const probeIsGreedy = (): boolean => {
    _observationsSet([{ obs: obsA, key: "X", ts: Date.now() }]);
    const claimed = drainObservations("Y");
    _observationsSet([]); // leave the queue clean either way
    return claimed.length === 1;
  };

  test("an overlapping window survives another flush's finally", async () => {
    // Window 1: a close() already in flight when the beforeExit flush fires.
    beginObservationsDrainAll();
    // Window 2: a whole real flush, begin → await → end.
    await forceFlushTokenPoliceSpans();
    // Window 1 is STILL open, so drains must still be greedy. The boolean
    // implementation cleared the flag in window 2's finally and failed here.
    expect(probeIsGreedy()).toBe(true);
    // Only when the last window closes do drains go back to keyed.
    endObservationsDrainAll();
    expect(probeIsGreedy()).toBe(false);
  });

  test("greedy only ends when BOTH nested windows end", () => {
    beginObservationsDrainAll();
    beginObservationsDrainAll();
    endObservationsDrainAll();
    expect(probeIsGreedy()).toBe(true); // depth 1 → still open
    endObservationsDrainAll();
    expect(probeIsGreedy()).toBe(false); // depth 0 → keyed again
  });

  test("end() is clamped at 0 — orphaned ends never poison a later window", () => {
    // resetPack (or a stray end) can discard in-flight windows; an underflow
    // to a negative depth would make the NEXT begin() a no-op and silently
    // disable greedy mode for the rest of the process.
    endObservationsDrainAll();
    endObservationsDrainAll();
    endObservationsDrainAll();
    expect(probeIsGreedy()).toBe(false);
    beginObservationsDrainAll();
    expect(probeIsGreedy()).toBe(true);
    endObservationsDrainAll();
    expect(probeIsGreedy()).toBe(false);
  });

  test("resetPack zeroes an open window", () => {
    beginObservationsDrainAll();
    resetPack();
    expect(probeIsGreedy()).toBe(false);
  });

  test("SpanProcessor.forceFlush() is NOT greedy — even mid-await", async () => {
    // The window (if any) is opened synchronously, before the first await —
    // so probing between the call and the await observes the in-flight state.
    const p = new TokenPoliceSpanProcessor().forceFlush();
    expect(probeIsGreedy()).toBe(false); // keyed for the whole flush
    await p;
    expect(probeIsGreedy()).toBe(false);
  });

  test("terminal forceFlushTokenPoliceSpans() IS greedy mid-await", async () => {
    const p = forceFlushTokenPoliceSpans(); // default greedy (flush/close/shutdown)
    expect(probeIsGreedy()).toBe(true);
    await p;
    expect(probeIsGreedy()).toBe(false);
  });

  test("explicit {greedy:false} leaves drains keyed mid-await", async () => {
    const p = forceFlushTokenPoliceSpans({ greedy: false });
    expect(probeIsGreedy()).toBe(false);
    await p;
    expect(probeIsGreedy()).toBe(false);
  });
});

describe("per-call scopes", () => {
  test("pushObservation tags with the current scope key and the scope's own drain claims it", () => {
    runWithCallObsScope(() => {
      const key = getCurrentObsKey();
      expect(typeof key).toBe("string");
      pushObservation(obsA);
      // A DIFFERENT call's drain must not steal it…
      expect(drainObservations("someone-else")).toEqual([]);
      // …but this call's own drain does.
      expect(drainObservations(key ?? null)).toEqual([obsA]);
    });
  });

  test("reuse-if-exists: a nested scope shares the outer call's key (SDK-internal delegation)", () => {
    runWithCallObsScope(() => {
      const outer = getCurrentObsKey();
      runWithCallObsScope(() => {
        expect(getCurrentObsKey()).toBe(outer);
      });
    });
  });

  test("runWithObsKey re-enters a captured scope (stream drain callbacks)", () => {
    const key = newObsKey();
    expect(key).toBeTruthy();
    _observationsSet([{ obs: obsA, key: key as string, ts: Date.now() }]);
    let drained: unknown[] = [];
    // Simulates a stream-completion callback running in the consumer's
    // context: re-enter the captured scope, then drain via the ALS read.
    runWithObsKey(key, () => {
      drained = drainObservations(getCurrentObsKey() ?? null);
    });
    expect(drained).toEqual([obsA]);
  });

  test("runWithCallObsScope/runWithObsKey run fn exactly once and propagate its error", () => {
    let calls = 0;
    expect(() =>
      runWithCallObsScope(() => {
        calls += 1;
        throw new Error("boom");
      }),
    ).toThrow("boom");
    expect(calls).toBe(1);

    calls = 0;
    expect(() =>
      runWithObsKey("k", () => {
        calls += 1;
        throw new Error("boom2");
      }),
    ).toThrow("boom2");
    expect(calls).toBe(1);
  });

  test("CONCURRENCY: two overlapping scoped calls each drain exactly their own observation", async () => {
    // The incident shape: call A pushes its observations, call B's /log
    // fires first — B must NOT walk away with A's entries.
    const results: Record<string, unknown[]> = {};
    const runCall = (name: string, obs: object, delayMs: number) =>
      runWithCallObsScope(async () => {
        pushObservation(obs); // pre-flight push, tagged with this call's key
        await sleep(delayMs); // provider call in flight; the other call logs meanwhile
        results[name] = drainObservations(getCurrentObsKey() ?? null);
      });
    await Promise.all([runCall("A", obsA, 30), runCall("B", obsB, 1)]);
    expect(results.B).toEqual([obsB]); // B logged first — only its own entry
    expect(results.A).toEqual([obsA]); // A's entry waited for A's own drain
    expect(drainObservations()).toEqual([]); // nothing stranded
  });
});
