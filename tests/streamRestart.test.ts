/**
 * Node SSE `stop()` was not restart-safe. `start()` guards on a truthy
 * `loopPromise` (stream.ts:53) but nothing ever nulled that field, so once the
 * reader had been started even a STOPPED instance kept a settled `loopPromise`
 * reference forever — a later `start()` on the same instance short-circuited and
 * never re-invoked `runForever()`. A `stop()` → `start()` cycle on one instance
 * permanently failed to restart the SSE reader.
 *
 * Python is the reference: `start()` guards on `self._thread.is_alive()`
 * (stream.py:64), so a stopped thread is no longer alive and a later `start()`
 * spawns a fresh thread → clean restart.
 *
 * Fix (start() only): chain `.finally(() => { this.loopPromise = null; })` onto
 * the EXISTING `runForever().catch(() => {})` so `loopPromise` is nulled on the
 * loop's TRUE exit. It is then non-null iff the loop is running (the JS analog of
 * `is_alive()`): the:53 guard still no-ops while running (running path
 * byte-identical) and a synchronous stop()+start() still no-ops (loop not yet
 * exited → no double-loop race), while a stop() → await teardown → start()
 * restarts cleanly. stop() is UNCHANGED (nulling loopPromise there is the
 * rejected alternative — it would spawn a 2nd concurrent loop on a sync
 * stop()+start()).
 *
 * These tests reuse the / fake-timer + fake-fetch harness so the only
 * thing that can move the loop is the injected reader / the captured signal,
 * making the restart deterministic (no real network, no real sleep). They encode
 * Rubric assertions 3, 4, 5, 6, 7.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { StreamClient } from "../src/stream";

type Chunk = { value?: Uint8Array; done: boolean };

function abortError(): Error {
  return Object.assign(new Error("aborted"), { name: "AbortError" });
}

/** A reader whose read() never resolves on its own but rejects AbortError when
 * the captured signal fires — faithfully models undici aborting an in-flight
 * read on signal abort (as in the silentReader). */
function silentReader(signal: AbortSignal) {
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
    baseUrl: "http://localhost:15097",
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

/** Read the private lifecycle promise the way the rubric's harness does. */
function loopPromiseOf(client: StreamClient): Promise<void> | null {
  return (client as unknown as { loopPromise: Promise<void> | null }).loopPromise;
}

describe("SSE stop() → start() restart safety", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  // Assertion 6 (loopPromise non-null while running) + assertion 3 (running path
  // byte-identical: a 2nd start() WITHOUT an intervening stop() no-ops — no
  // double loop / no 2nd connection).
  it("a double start() without stopping keeps a single connection (running path unchanged)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, opts: { signal: AbortSignal }) => okResponse(silentReader(opts.signal))),
    );
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0); // connection 1 established
    expect(fetchCalls()).toBe(1);
    // Assertion 6: while the loop runs, loopPromise is a pending Promise.
    expect(loopPromiseOf(client)).not.toBeNull();

    client.start(); // guard at stream.ts:53 short-circuits — loop already running
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchCalls()).toBe(1); // still ONE connection — no 2nd concurrent loop

    client.stop();
  });

  // Assertion 4 (RED-without-fix / GREEN-with-fix) + assertion 5 (loopPromise is
  // null after stop() settles). This is the load-bearing restart test.
  it("restarts the reader on stop() → await teardown → start() (fetch 1 → 2)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, opts: { signal: AbortSignal }) => okResponse(silentReader(opts.signal))),
    );
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0); // connection 1
    expect(fetchCalls()).toBe(1);

    // Capture the lifecycle promise, stop, and await its TRUE settle: abort →
    // in-flight read() rejects AbortError → connectAndPump throws → runForever
    // swallows and hits `if (this.stopped) return` (stream.ts:80) → resolves →
    // .catch settles → .finally nulls loopPromise.
    const p = loopPromiseOf(client);
    expect(p).not.toBeNull();
    client.stop();
    await p; // teardown completes
    await vi.advanceTimersByTimeAsync(0); // flush the trailing .finally microtask

    // Assertion 5: loopPromise nulled on true exit. RED without fix: it stays the
    // settled Promise object (truthy) → the next start() no-ops.
    expect(loopPromiseOf(client)).toBeNull();

    // Assertion 4: a fresh start() re-invokes runForever → connection 2.
    client.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchCalls()).toBe(2); // RED without fix: stays 1 (no reconnection)

    client.stop();
  });

  // Assertion 7 (no-double-loop discriminator): a SYNCHRONOUS stop()+start() in
  // the same tick — while runForever is still winding down — must NOT spawn a 2nd
  // concurrent loop. Because .finally nulls loopPromise only on TRUE exit (which
  // has not happened yet), the sync start() sees a still-pending loopPromise and
  // no-ops. fetch never exceeds 1. This is the property the rejected
  // stop()-nulling alternative would FAIL (it would null synchronously → the
  // start() would spawn a 2nd runForever while the first is still tearing down).
  it("synchronous stop()+start() does not spawn a second concurrent loop (matches Python)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, opts: { signal: AbortSignal }) => okResponse(silentReader(opts.signal))),
    );
    const client = makeClient();
    client.start();
    await vi.advanceTimersByTimeAsync(0); // connection 1
    expect(fetchCalls()).toBe(1);

    // Same tick, no await between: stop() then immediately start().
    client.stop();
    client.start(); // loop still winding down → loopPromise still pending → no-op
    // Let the (single) teardown complete; the re-entrant start() must not have
    // added a connection.
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchCalls()).not.toBeGreaterThan(1); // never a 2nd overlapping connection

    client.stop();
  });

  // Assertions 1/2/5 (structural, static): `this.loopPromise = null` appears
  // exactly ONCE in stream.ts and inside the .finally callback — NEVER in stop().
  // Reading the source keeps this an executable pin, not just a prose claim.
  it("nulls loopPromise exactly once, inside start()'s .finally, never in stop()", () => {
    const here = dirname(fileURLToPath(import.meta.url));
    const src = readFileSync(join(here, "..", "src", "stream.ts"), "utf8");

    // Exactly one assignment of loopPromise = null.
    const nullAssignments = src.match(/this\.loopPromise\s*=\s*null/g) ?? [];
    expect(nullAssignments.length).toBe(1);

    // That assignment lives inside a .finally(...) chained on runForever().
    expect(src).toMatch(/\.finally\(\(\)\s*=>\s*\{\s*this\.loopPromise\s*=\s*null;\s*\}\)/);

    // The stop() body contains no loopPromise reference at all.
    const stopBody = src.slice(src.indexOf("stop(): void"), src.indexOf("private async runForever"));
    expect(stopBody).not.toContain("loopPromise");
  });
});
