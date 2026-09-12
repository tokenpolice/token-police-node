/**
 * Sync-wrapper branch runs NO pre-flight check (Node-only).
 *
 * `_wrapMethod`'s sync `else` branch (reachable ONLY via the public
 * `protect(name, path, method, isAsync=false, ...)` escape hatch — every
 * internal auto-instrumentation target passes `isAsync:true` and takes the
 * async branch) does composition-capture + attempt-context + call-outcome +
 * response-composition + failure-logging ONLY. It cannot run the pre-flight
 * `_runAsyncCheck` (async — a sync context can't `await`), so it never
 * enforces. This suite proves:
 * - the wrapper still WRAPS (function identity changes) (11a)
 * - it returns the raw, non-thenable synchronous value (11b)
 * - it dispatches ZERO `client.check()` even under firewall:enforce
 * + a BLOCK snapshot — the real invariant the removed no-op stub
 * only pretended to provide (11c)
 * - a thrown customer error passes straight through, unwrapped (12)
 * - success sync call ⇒ session `_call_outcome` populated (13 success / item 8)
 * - failure sync call ⇒ failure `_call_outcome` built AND the
 * failure `/log` fired BEFORE the customer error re-raises (13 failure / items 8+9)
 *
 * The async-branch enforcement that the deletion did NOT remove is anchored by
 * the existing `tests/neverFail.test.ts` → "enforcer: dry_run parity vs
 * enforce" → "(c) enforce STILL throws TokenPoliceBlockedError on a verified
 * block" (assertion 14). No live SDK import is needed: we patch a synthetic
 * in-app class via `protect`'s `options.module` (mirrors groqNativeTarget's
 * driving style) and spy `client.check` / `client.log` (mirrors neverFail).
 */
import { describe, it, expect, beforeEach, vi } from "vitest";
import { protect } from "../src/enforcer";
import { session } from "../src/context";
import { TokenPolice } from "../src/client";
import { TokenPoliceBlockedError } from "../src/exceptions";
import { applySnapshot, resetPack, setClient } from "../src/state";

// A verified-enforce BLOCK directive — identical shape to neverFail's, so if
// the sync branch DID run a pre-flight it would see a block. It must not.
const BLOCK_SNAPSHOT = {
  schema_version: 1, type: "snapshot", version: 1,
  tenant_id: "t", project_id: "p", ttl_seconds: 600, loop_blocks: [],
  directives: [
    {
      id: "r1", kind: "UNCONDITIONAL_BLOCK", mode: "enforce", priority: 10,
      selector: { match: null, group_by: [] },
    },
  ],
};

function makeClient(firewall: "enforce" | "dry_run" | "off") {
  // Direct construction (not init()) so no SSE stream starts; an unreachable
  // 5xxxx baseUrl makes any background tp.log()/tp.check() fail-open silently.
  const client = new TokenPolice({
    apiKey: "tp_sk_test123",
    baseUrl: "http://localhost:59999",
    timeout: 0.1,
    firewall,
    deployment: "daemon",
  });
  setClient(client);
  return client;
}

// Build a fresh synthetic in-app class with a SYNC `create` method and wrap it
// via the public sync escape hatch. A unique moduleName per call keeps
// `_wrapMethod`'s `_originals` de-dupe from skipping re-instrumentation across
// tests. moduleName contains "openai" so `_detectProvider` → "openai" (only
// affects best-effort composition capture). Returns the class + the original
// method reference captured BEFORE `protect`.
function makeSyncTarget(moduleName: string, impl: (...a: any[]) => any) {
  class SyncClient {
    create(...args: any[]): any {
      return impl(...args);
    }
  }
  const orig = SyncClient.prototype.create;
  // isAsync=false → the sync `else` branch under test.
  protect(moduleName, ["prototype"], "create", false, { module: SyncClient });
  return { SyncClient, orig };
}

const REQ = { model: "gpt-4", messages: [{ role: "user", content: "hi" }] };

describe("Sync wrapper — wraps but does NOT enforce", () => {
  beforeEach(() => resetPack());

  it("11a/11b/11c: wraps the method, returns a raw non-thenable value, and dispatches NO /check even under enforce + a BLOCK snapshot", () => {
    const client = makeClient("enforce");
    applySnapshot(BLOCK_SNAPSHOT);
    const checkSpy = vi
      .spyOn(client, "check")
      .mockResolvedValue({ status: "blocked", reason: "x" } as never);

    const SENTINEL = { id: "resp-1", ok: true };
    const { SyncClient, orig } = makeSyncTarget(
      "openai-syncnoenforce-a",
      () => SENTINEL,
    );

    // 11a — anti-parity-by-deletion: the method identity changed (would be
    // unchanged if the whole sync `else` branch were deleted).
    expect(SyncClient.prototype.create).not.toBe(orig);

    const result = new SyncClient().create(REQ);

    // 11b — raw synchronous value, unchanged AND non-thenable. A smuggled
    // `await _runAsyncCheck` would force the wrapper async → a Promise return.
    expect(result).toBe(SENTINEL);
    expect(typeof (result as any)?.then).not.toBe("function");

    // 11c — the load-bearing invariant: zero pre-flight dispatch, even though
    // firewall is "enforce" and a BLOCK directive is live.
    expect(checkSpy).not.toHaveBeenCalled();

    checkSpy.mockRestore();
  });

  it("12: a thrown customer error propagates unchanged (not wrapped, not a TP error)", () => {
    const client = makeClient("enforce");
    applySnapshot(BLOCK_SNAPSHOT);
    vi.spyOn(client, "log").mockImplementation((() => undefined) as never);

    const boom = Object.assign(new Error("customer upstream failure"), {
      status: 500,
    });
    const { SyncClient } = makeSyncTarget("openai-syncnoenforce-b", () => {
      throw boom;
    });

    let caught: unknown;
    try {
      new SyncClient().create(REQ);
    } catch (e) {
      caught = e;
    }
    // Exact same object surfaces — never a TokenPoliceBlockedError.
    expect(caught).toBe(boom);
    expect(caught).not.toBeInstanceOf(TokenPoliceBlockedError);
  });

  it("13 success: a successful sync call populates the session call-outcome (buildCallOutcome null-path)", () => {
    makeClient("enforce");
    applySnapshot(BLOCK_SNAPSHOT);

    const { SyncClient } = makeSyncTarget(
      "openai-syncnoenforce-c",
      () => ({ id: "ok" }),
    );

    // Run INSIDE a session scope so the wrapper's `getCurrentSession()` and the
    // one we read are the SAME instance (outside a scope each call gets a fresh
    // session). `session()` runs the sync body directly (no tracer configured).
    const outcome = session({ name: "wf_sync_success" }, (s) => {
      new SyncClient().create(REQ);
      return (s as any)._call_outcome;
    });

    expect(outcome).toBeTruthy();
    expect(outcome.status).toBe("success");
    expect(typeof outcome.duration_ms).toBe("number");
  });

  it("13 failure: a thrown sync call builds a FAILURE call-outcome AND fires the failure /log before the customer error re-raises", () => {
    const client = makeClient("enforce");
    applySnapshot(BLOCK_SNAPSHOT);
    // `_emitCallFailureLog` dispatches to `tp.log` (enforcer.ts:687) — the
    // feasible spy seam. Mock it so nothing hits the network.
    const logSpy = vi
      .spyOn(client, "log")
      .mockImplementation((() => undefined) as never);

    const boom = new Error("customer stream drained mid-flight");
    const { SyncClient } = makeSyncTarget("openai-syncnoenforce-d", () => {
      throw boom;
    });

    // The wrapper's `catch` runs `_emitCallFailureLog(...)` THEN `throw err`, so
    // a recorded log call co-occurring with the thrown customer error proves the
    // ordering (log recorded before the error surfaced).
    let caught: unknown;
    session({ name: "wf_sync_failure" }, () => {
      try {
        new SyncClient().create(REQ);
      } catch (e) {
        caught = e;
      }
    });

    expect(caught).toBe(boom);
    // Failure /log fired (item 9).
    expect(logSpy).toHaveBeenCalled();
    // The failure call-outcome was built and carried on the /log payload — the
    // last positional arg of tp.log() is the meta object holding `call_outcome`
    // (enforcer.ts:707). Read it here because `_emitCallFailureLog` nulls
    // `session._call_outcome` after emitting (enforcer.ts:711).
    const meta: any = logSpy.mock.calls[0]?.at(-1);
    expect(meta?.call_outcome?.status).toBe("failed");

    logSpy.mockRestore();
  });
});
