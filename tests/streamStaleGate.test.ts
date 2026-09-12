/**
 * Stream-stale allow-path /check fallback gate (C-14).
 *
 * A locally ALLOWED call whose decision hinged on a missed entity-list arm
 * (`armableMiss`) is re-verified with the existing inline /check ONLY once the
 * SSE stream has been disconnected longer than `streamStaleGraceSeconds`.
 * Healthy stream, within-grace windows, and unguarded (non-armable-miss)
 * traffic all keep the zero-round-trip hot path untouched.
 *
 * The fallback reuses the EXACT State-B `/check` code path (`_runAsyncCheck`),
 * so bounded timeout + fail-open semantics apply automatically — see the
 * GOLDEN RULE test below, which is the most important test in this file.
 *
 * Sibling: token-police-python/tests/test_stream_stale_gate.py pins the same
 * scenarios (sync + async) against the Python SDK.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  applySnapshot,
  resetPack,
  setClient,
  markStreamConnected,
  markStreamDisconnected,
} from "../src/state";
import { TokenPolice } from "../src/client";
import { TokenPoliceBlockedError } from "../src/exceptions";
import { __test__ as enforcerTest } from "../src/enforcer";

// ENTITY_BLOCK on user_id EXISTS / group_by user_id, with an empty entities
// list. TPSession's default userId is "anonymous" — always truthy, so the
// selector always matches and "anonymous" is never armed here, producing an
// armableMiss on every locally-allowed call that reaches this directive.
const ARMABLE_MISS_SNAPSHOT = {
  schema_version: 1,
  type: "snapshot",
  version: 1,
  tenant_id: "t",
  project_id: "p",
  ttl_seconds: 600,
  loop_blocks: [],
  directives: [
    {
      id: "eb1",
      kind: "ENTITY_BLOCK",
      mode: "enforce",
      priority: 10,
      selector: { match: { field: "user_id", operator: "EXISTS" }, group_by: ["user_id"] },
      entities: [],
    },
  ],
};

// No directives at all → no armableMiss is ever possible.
const NO_MISS_SNAPSHOT = {
  schema_version: 1,
  type: "snapshot",
  version: 1,
  tenant_id: "t",
  project_id: "p",
  ttl_seconds: 600,
  loop_blocks: [],
  directives: [],
};

function makeClient(streamStaleGraceSeconds = 60) {
  // Direct construction (not init()) so no SSE stream starts — this file
  // owns stream liveness state itself via markStreamConnected/Disconnected.
  const client = new TokenPolice({
    apiKey: "tp_sk_test123",
    baseUrl: "http://localhost:99999",
    timeout: 0.1,
    firewall: "enforce",
    deployment: "daemon",
    streamStaleGraceSeconds,
  });
  setClient(client);
  return client;
}

describe("stream-stale allow-path /check gate (Node)", () => {
  beforeEach(() => {
    resetPack();
    vi.useFakeTimers();
    vi.setSystemTime(1_000_000_000_000);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("healthy stream + armableMiss → NO /check call (hot path preserved)", async () => {
    const client = makeClient();
    applySnapshot(ARMABLE_MISS_SNAPSHOT);
    markStreamConnected();
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({ status: "allowed" } as never);
    await expect(enforcerTest._runAsyncCheck({ model: "gpt-4" }, "openai")).resolves.toBeUndefined();
    expect(checkSpy).not.toHaveBeenCalled();
    checkSpy.mockRestore();
  });

  it("within grace after disconnect → NO /check", async () => {
    const client = makeClient(60);
    applySnapshot(ARMABLE_MISS_SNAPSHOT);
    const gen = markStreamConnected();
    markStreamDisconnected(gen);
    vi.advanceTimersByTime(30_000); // 30s < 60s grace
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({ status: "allowed" } as never);
    await expect(enforcerTest._runAsyncCheck({ model: "gpt-4" }, "openai")).resolves.toBeUndefined();
    expect(checkSpy).not.toHaveBeenCalled();
    checkSpy.mockRestore();
  });

  it("stale beyond grace + armableMiss → /check IS called; a blocked result throws TokenPoliceBlockedError (enforce mode)", async () => {
    const client = makeClient(60);
    applySnapshot(ARMABLE_MISS_SNAPSHOT);
    const gen = markStreamConnected();
    markStreamDisconnected(gen);
    vi.advanceTimersByTime(61_000); // past the 60s grace
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({
      status: "blocked",
      reason: "budget exceeded",
      ruleId: "eb1",
    } as never);
    await expect(enforcerTest._runAsyncCheck({ model: "gpt-4" }, "openai")).rejects.toBeInstanceOf(
      TokenPoliceBlockedError,
    );
    expect(checkSpy).toHaveBeenCalledTimes(1);
    checkSpy.mockRestore();
  });

  // ── GOLDEN RULE ──────────────────────────────────────────────────────
  // The single most important test in this file: a /check failure on the new
  // stale-stream fallback path must NEVER throw anything other than
  // TokenPoliceBlockedError into the customer's call — and here /check
  // doesn't even resolve to a decision, it outright rejects (simulating a raw
  // network failure escaping client.check(), which in production already
  // fail-opens internally — this proves the enforcer's OWN failSafeAsync
  // wrapper is a second, independent safety net on this new code path).
  it("GOLDEN RULE: stale + armableMiss + /check network error → fail-open ALLOWED, no exception escapes", async () => {
    const client = makeClient(60);
    applySnapshot(ARMABLE_MISS_SNAPSHOT);
    const gen = markStreamConnected();
    markStreamDisconnected(gen);
    vi.advanceTimersByTime(61_000);
    const checkSpy = vi.spyOn(client, "check").mockRejectedValue(new Error("ECONNREFUSED"));
    await expect(enforcerTest._runAsyncCheck({ model: "gpt-4" }, "openai")).resolves.toBeUndefined();
    expect(checkSpy).toHaveBeenCalledTimes(1);
    checkSpy.mockRestore();
  });

  it("stale beyond grace but NO armableMiss → NO /check (unguarded traffic is unaffected)", async () => {
    const client = makeClient(60);
    applySnapshot(NO_MISS_SNAPSHOT);
    const gen = markStreamConnected();
    markStreamDisconnected(gen);
    vi.advanceTimersByTime(61_000);
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({ status: "allowed" } as never);
    await expect(enforcerTest._runAsyncCheck({ model: "gpt-4" }, "openai")).resolves.toBeUndefined();
    expect(checkSpy).not.toHaveBeenCalled();
    checkSpy.mockRestore();
  });

  it("grace 0: /check fires as soon as the stream is disconnected at all", async () => {
    const client = makeClient(0);
    applySnapshot(ARMABLE_MISS_SNAPSHOT);
    const gen = markStreamConnected();
    markStreamDisconnected(gen);
    vi.advanceTimersByTime(1); // any elapsed time > 0 is already stale at grace 0
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({ status: "allowed" } as never);
    await expect(enforcerTest._runAsyncCheck({ model: "gpt-4" }, "openai")).resolves.toBeUndefined();
    expect(checkSpy).toHaveBeenCalledTimes(1);
    checkSpy.mockRestore();
  });
});
