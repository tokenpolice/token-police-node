/**
 * Init() silently tears down and replaces an already-installed client.
 *
 * The teardown-and-replace on a second init() is intentional (test isolation /
 * hot-reload), but doing it SILENTLY hides a double-init bug. This fix emits ONE
 * developer-facing warning when init() replaces a prior client. The detection
 * reads getClient() BEFORE construct/swap, and the read+warn is wrapped so a
 * broken logger can never throw out of init() (Golden Rule / fail-open).
 *
 * The SDK `logger` (client.ts:20-23) is a module-private const whose `warning`
 * sink is `console.warn`, so `console.warn` is the only spyable observation
 * point. Mirrors token-police-python/tests/test_init_reinit.py.
 */
import { describe, test, expect, afterEach, vi } from "vitest";
import { init, TokenPolice } from "../src/client";
import { setClient, getClient } from "../src/state";
import { uninstrument } from "../src/enforcer";

const REPLACE_MSG = "replacing the previously initialized client";

// A valid tp_sk_ key + firewall:"off" keeps the ONLY console.warn reachable in
// the init path the new replacement warning (avoids the unwrapped apiKey-prefix
// warning at client.ts:605-609 and provider/tap paths).
const OPTS = { apiKey: "tp_sk_test", firewall: "off", deployment: "serverless" } as const;

afterEach(() => {
  try { uninstrument(); } catch { /* ignore */ }
  try { setClient(undefined as any); } catch { /* ignore */ }
  vi.restoreAllMocks();
});

describe("Warn on re-init (replace)", () => {
  // Assertion 2 — no replacement warning on FIRST init (no prior client).
  test("does NOT warn on the first init", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    init({ ...OPTS } as any);
    const replaceCalls = warnSpy.mock.calls.filter((c) =>
      String(c[0]).includes(REPLACE_MSG),
    );
    expect(replaceCalls).toHaveLength(0);
  });

  // Assertion 1 — warns on the SECOND init, even with IDENTICAL options.
  test("warns on the second init with identical options", () => {
    init({ ...OPTS } as any); // first — no warning
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    init({ ...OPTS } as any); // second, same options — must warn
    const replaceCalls = warnSpy.mock.calls.filter((c) =>
      String(c[0]).includes(REPLACE_MSG),
    );
    expect(replaceCalls.length).toBeGreaterThanOrEqual(1);
  });

  // Assertion 13 — EXACTLY one warning on the replacing init (valid key +
  // firewall:"off" → replacement warning is the only console.warn in the path).
  test("emits exactly one warning on the replacing init", () => {
    init({ ...OPTS } as any); // first
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    init({ ...OPTS } as any); // replacing
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(String(warnSpy.mock.calls[0][0])).toContain(REPLACE_MSG);
  });

  // Assertion 5 — getClient() returns the SECOND client after re-init; the
  // second init() return value is that client and NOT the first.
  test("getClient() returns the second client after re-init", () => {
    const first = init({ ...OPTS } as any);
    const second = init({ ...OPTS } as any);
    expect(second).toBeInstanceOf(TokenPolice);
    expect(second).not.toBe(first);
    expect(getClient()).toBe(second);
  });

  // Assertion 7 — a throwing console.warn (broken logger) must NOT escape
  // init(); the replacing init() still returns the new client. Valid key +
  // firewall:"off" guarantees the replacement warning is the only console.warn
  // reachable, so surviving proves it is wrapped in try/catch.
  test("a throwing logger cannot escape the replacing init()", () => {
    init({ ...OPTS } as any); // first, with a normal console.warn
    vi.spyOn(console, "warn").mockImplementation(() => {
      throw new Error("boom: broken logger");
    });
    let second: any;
    expect(() => {
      second = init({ ...OPTS } as any); // replacing — warn throws, must be swallowed
    }).not.toThrow();
    expect(second).toBeInstanceOf(TokenPolice);
  });
});

describe("beforeExit cleanup: one listener, acting on the current client", () => {
  // (a) init() registers AT MOST one process-level beforeExit listener the first
  // time and NEVER accumulates more across re-inits — no MaxListenersExceeded, no
  // retained dead clients. Baseline is measured live because the runtime (vitest)
  // may already hold beforeExit listeners of its own.
  test("registers at most one beforeExit listener and stays flat across re-inits", () => {
    const baseline = process.listenerCount("beforeExit");
    init({ ...OPTS } as any);
    const afterFirst = process.listenerCount("beforeExit");
    expect(afterFirst - baseline).toBeLessThanOrEqual(1);
    // Three further re-inits must not add a single listener.
    init({ ...OPTS } as any);
    init({ ...OPTS } as any);
    init({ ...OPTS } as any);
    expect(process.listenerCount("beforeExit")).toBe(afterFirst);
  });

  // (b) the (single) listener acts on the CURRENT client, not the init that
  // happened to register it: after a re-init, beforeExit must stop/flush the NEW
  // client and leave the replaced one untouched.
  test("beforeExit acts on the current client after re-init, not a stale one", () => {
    // Both inits FIRST — the second init's setClient() legitimately closeSync()s
    // the replaced client, so attach the observation fakes only afterward to
    // isolate what the beforeExit hook itself touches.
    const first = init({ ...OPTS } as any);
    const second = init({ ...OPTS } as any);

    const firstStop = vi.fn();
    (first as any)._streamClient = { stop: firstStop };
    const firstFlush = vi.spyOn(first, "flush").mockResolvedValue(undefined);

    const secondStop = vi.fn();
    (second as any)._streamClient = { stop: secondStop };
    const secondFlush = vi.spyOn(second, "flush").mockResolvedValue(undefined);

    // Fire the process-level hook.
    (process as any).emit("beforeExit", 0);

    // Current (second) client got the cleanup; the replaced (first) did not.
    expect(secondStop).toHaveBeenCalledTimes(1);
    expect(secondFlush).toHaveBeenCalledTimes(1);
    expect(firstStop).not.toHaveBeenCalled();
    expect(firstFlush).not.toHaveBeenCalled();
  });
});
