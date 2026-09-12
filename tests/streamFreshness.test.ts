/**
 * SSE stream liveness (C-14) — `markStreamConnected` / `markStreamDisconnected`
 * / `isStreamFresh`. Tracked SEPARATELY from pack health so a disconnect never
 * invalidates the Decision Pack (last-known-good blocking during an outage is
 * unchanged design intent) — these three primitives only let the enforcer
 * decide a locally ALLOWED call whose decision hinged on a streamed entity
 * list has gone too stale to trust.
 *
 * `_streamConnected` / `_streamDisconnectedAt` / `_streamGeneration` are
 * process/module-lifetime state — state.ts deliberately does NOT reset them in
 * resetPack() (see the C-14 fix plan). vitest gives every test FILE its own
 * fresh module registry, so within this file:
 *  - the "never connected" test below MUST run first (nothing earlier in this
 *    file may call markStreamConnected()) — vitest runs a file's tests
 *    top-to-bottom by default (no shuffle configured), so this is deterministic.
 *  - every later test re-establishes a clean baseline itself by calling
 *    markStreamConnected() at its own start (which always clears
 *    `_streamDisconnectedAt` and flips connected=true), so those tests never
 *    depend on ordering beyond that single first one.
 *
 * All timing uses vi.useFakeTimers()/vi.setSystemTime() — never a real sleep —
 * and grace-0 boundary checks always advance the fake clock at least 1ms past
 * the disconnect instant, avoiding the same-millisecond flake a real clock risks.
 *
 * Sibling: token-police-python/tests/test_stream_freshness.py pins the same
 * scenarios against the Python SDK.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import * as state from "../src/state";

describe("isStreamFresh — never connected (must run before any connect in this file)", () => {
  it("a stream that has never connected is never fresh, at grace 0 or a huge grace", () => {
    expect(state.isStreamFresh(0)).toBe(false);
    expect(state.isStreamFresh(3_600_000)).toBe(false);
  });
});

describe("isStreamFresh — connected stream", () => {
  it("is fresh regardless of grace, including grace 0", () => {
    state.markStreamConnected();
    expect(state.isStreamFresh(0)).toBe(true);
    expect(state.isStreamFresh(3_600_000)).toBe(true);
  });
});

describe("isStreamFresh — disconnect freshness window", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(1_000_000_000_000);
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("fresh immediately on disconnect, fresh up to the grace boundary (inclusive), stale one ms past", () => {
    const gen = state.markStreamConnected();
    state.markStreamDisconnected(gen);
    expect(state.isStreamFresh(1000)).toBe(true); // 0ms elapsed
    vi.advanceTimersByTime(999);
    expect(state.isStreamFresh(1000)).toBe(true); // 999ms < 1000ms grace
    vi.advanceTimersByTime(1);
    expect(state.isStreamFresh(1000)).toBe(true); // exactly at the boundary — inclusive
    vi.advanceTimersByTime(1);
    expect(state.isStreamFresh(1000)).toBe(false); // 1001ms > 1000ms grace
  });

  it("grace 0: fresh at the disconnect instant, stale once the clock advances at all", () => {
    const gen = state.markStreamConnected();
    state.markStreamDisconnected(gen);
    expect(state.isStreamFresh(0)).toBe(true); // 0 <= 0
    vi.advanceTimersByTime(1);
    expect(state.isStreamFresh(0)).toBe(false); // 1 > 0
  });

  it("earliest-stamp: a repeated markStreamDisconnected within the same generation does not refresh the stamp", () => {
    const gen = state.markStreamConnected();
    state.markStreamDisconnected(gen); // t=0 — the real disconnect
    vi.advanceTimersByTime(40_000);
    state.markStreamDisconnected(gen); // idempotent no-op (a failed reconnect attempt)
    vi.advanceTimersByTime(30_000); // total elapsed since the ORIGINAL disconnect = 70s
    // grace=60s measured from the ORIGINAL (t=0) disconnect => stale now.
    // If the second call had refreshed the stamp to t=40s, elapsed since it
    // would be only 30s (< 60s) and this would still read fresh.
    expect(state.isStreamFresh(60_000)).toBe(false);
  });

  it("earliest-stamp control: without the second (no-op) call the same elapsed time would still be fresh under a bigger grace", () => {
    // Sanity control proving the assertion above is non-vacuous: a single
    // disconnect with the SAME total elapsed time (70s) under a grace that
    // covers it (90s) is fresh — confirming the 60s-grace staleness above
    // comes from the EARLIEST stamp, not from some other effect.
    const gen = state.markStreamConnected();
    state.markStreamDisconnected(gen);
    vi.advanceTimersByTime(70_000);
    expect(state.isStreamFresh(90_000)).toBe(true);
  });

  it("zombie: a stale generation's disconnect cannot mark a newer connection down", () => {
    const gen1 = state.markStreamConnected();
    const gen2 = state.markStreamConnected();
    expect(gen2).not.toBe(gen1);
    state.markStreamDisconnected(gen1); // zombie reader's late exit — ignored
    expect(state.isStreamFresh(0)).toBe(true); // still connected under gen2
    // the CURRENT generation can still legitimately disconnect
    state.markStreamDisconnected(gen2);
    vi.advanceTimersByTime(1);
    expect(state.isStreamFresh(0)).toBe(false);
  });

  it("a reconnect after a long stale disconnect makes the stream fresh again immediately, even at grace 0", () => {
    const gen = state.markStreamConnected();
    state.markStreamDisconnected(gen);
    vi.advanceTimersByTime(10_000); // well beyond any sane grace
    expect(state.isStreamFresh(0)).toBe(false); // stale
    state.markStreamConnected(); // reconnect
    // Connected bypasses the clock check entirely (no residual window).
    expect(state.isStreamFresh(0)).toBe(true);
  });
});
