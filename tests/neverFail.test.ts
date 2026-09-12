/**
 * Fault-injection matrix — verifies the SDK NEVER throws into customer code
 * from any internal failure. Only TokenPoliceBlockedError may propagate,
 * and only on explicit enforce=true denial.
 *
 * Mirrors token-police-python/tests/test_never_fail.py.
 */
import { describe, test, expect, beforeEach, vi } from "vitest";
import { classifyException, buildCallOutcome } from "../src/_classify";
import { evaluate } from "../src/localEvaluator";
import {
  applySnapshot, applyDeltas, resetPack, isCacheHealthy,
  invalidatePack, pushObservation, drainObservations, getPack, getPackVersion,
  setClient,
} from "../src/state";
import { detectDeploymentMode, resolveDeployment } from "../src/runtime";
import { TokenPolice } from "../src/client";
import { TokenPoliceBlockedError } from "../src/exceptions";
import { __test__ as enforcerTest } from "../src/enforcer";

class DummySession {
  user_id = "u1";
  paid_plan = "free";
  workflow_name = "wf";
  session_id = "s1";
  metadata = {};
  trace_id = "trace_test";
}

describe("never-fail: classifier", () => {
  beforeEach(() => resetPack());

  test("returns unknown on weird input", () => {
    class Weird extends Error {
      get status_code() { throw new Error("nope"); }
    }
    const r = classifyException(new Weird());
    expect(r.error_kind).toBe("unknown");
    expect(r.http_status).toBe(0);
  });

  test("handles string input gracefully", () => {
    const r = classifyException("not an exception" as unknown);
    expect(r.error_kind).toBe("unknown");
  });

  test("buildCallOutcome success", () => {
    expect(buildCallOutcome(null, 123)).toEqual({ status: "success", duration_ms: 123 });
  });

  test("buildCallOutcome failure scrubs the raw message", async () => {
    // With no client configured the default detail mode is 'redacted',
    // so the raw string no longer ships — a SHA-256 hash of the FULL message
    // does instead (truncation now only applies in opt-in 'raw' mode, covered
    // by tests/errorMessageScrub.test.ts). The outcome must never carry raw text.
    const { createHash } = await import("crypto");
    const big = "x".repeat(5000);
    const err = new Error(big);
    const out = buildCallOutcome(err, 12);
    expect(out.status).toBe("failed");
    expect(out.error_message).toBeUndefined();
    expect(out.error_message_hash).toBe(createHash("sha256").update(big, "utf8").digest("hex"));
  });

  test("recognizes 429 by status", () => {
    const err = Object.assign(new Error("slow down"), { status: 429 });
    expect(classifyException(err).error_kind).toBe("rate_limited");
  });

  test("recognizes 401 by status", () => {
    const err = Object.assign(new Error("bad key"), { status: 401 });
    const out = classifyException(err);
    expect(out.error_kind).toBe("auth_error");
    expect(out.http_status).toBe(401);
  });

  test("recognizes 502 as server_error", () => {
    const err = Object.assign(new Error("upstream"), { status: 502 });
    expect(classifyException(err).error_kind).toBe("server_error");
  });
});

describe("never-fail: localEvaluator", () => {
  beforeEach(() => resetPack());

  test("null pack returns allowed", () => {
    const r = evaluate(null, new DummySession(), { model: "gpt-4", provider: "openai" });
    expect(r.decision.status).toBe("allowed");
  });

  test("malformed directives field doesn't throw", () => {
    const bad = { directives: "not a list", loop_blocks: null };
    const r = evaluate(bad, new DummySession(), { model: "", provider: "" });
    expect(["allowed", "blocked", "rerouted"]).toContain(r.decision.status);
  });

  test("directives missing fields doesn't throw", () => {
    const pack = {
      directives: [
        { id: "r1", kind: "UNCONDITIONAL_BLOCK" },
        { id: "r2" },
        { id: "r3", kind: "REROUTE", selector: {}, reroute: null },
      ],
      loop_blocks: [],
    };
    const r = evaluate(pack, new DummySession(), { model: "m", provider: "openai" });
    expect(r.decision).toHaveProperty("status");
  });

  test("cross-provider reroute emits reroute_rejected observation", () => {
    const pack = {
      directives: [
        {
          id: "r1", kind: "REROUTE", mode: "enforce", priority: 50,
          selector: { match: null, group_by: [] },
          reroute: { from: null, to: { provider: "anthropic", model: "claude-3-haiku" } },
        },
      ],
      loop_blocks: [],
    };
    const r = evaluate(pack, new DummySession(), { model: "gpt-4", provider: "openai" });
    expect(r.decision.status).toBe("allowed");
    expect(r.observations.some((o) => o.outcome === "reroute_rejected")).toBe(true);
  });

  test("loop_blocked trace short-circuits to blocked", () => {
    const pack = { directives: [], loop_blocks: ["trace_xyz"] };
    const r = evaluate(pack, new DummySession(), { model: "m", provider: "openai", trace_id: "trace_xyz" });
    expect(r.decision.status).toBe("blocked");
  });

  test("forceShadow (dry_run) turns an ENFORCE block directive into observe-only", () => {
    const pack = {
      directives: [
        {
          id: "r1", kind: "UNCONDITIONAL_BLOCK", mode: "enforce", priority: 10,
          selector: { match: null, group_by: [] },
        },
      ],
      loop_blocks: [],
    };
    // Without forceShadow, an enforce directive blocks.
    const live = evaluate(pack, new DummySession(), { model: "m", provider: "openai" });
    expect(live.decision.status).toBe("blocked");
    // With forceShadow (dry_run), it never blocks — only a would_block observation.
    const dry = evaluate(pack, new DummySession(), { model: "m", provider: "openai" }, true);
    expect(dry.decision.status).toBe("allowed");
    expect(dry.observations.some((o) => o.outcome === "would_block" && o.mode === "dry_run")).toBe(true);
  });
});

describe("never-fail: state pack lifecycle", () => {
  beforeEach(() => resetPack());

  test("applySnapshot accepts a minimal snapshot", () => {
    const ok = applySnapshot({
      schema_version: 1, type: "snapshot", version: 1,
      tenant_id: "t", project_id: "p", directives: [], loop_blocks: [], ttl_seconds: 10,
    });
    expect(ok).toBe(true);
    expect(isCacheHealthy()).toBe(true);
  });

  test("applyDeltas with empty pack returns false", () => {
    const ok = applyDeltas([{ op: "entity_blocked", rule_id: "r1", entity: "u1" }], 1);
    expect(ok).toBe(false);
  });

  test("applyDeltas with old version is idempotently discarded", () => {
    applySnapshot({
      schema_version: 1, type: "snapshot", version: 5,
      tenant_id: "t", project_id: "p", directives: [], loop_blocks: [], ttl_seconds: 10,
    });
    const ok = applyDeltas([], 3);
    expect(ok).toBe(true);
  });

  test("applyDeltas with version gap poisons the cache", () => {
    applySnapshot({
      schema_version: 1, type: "snapshot", version: 5,
      tenant_id: "t", project_id: "p", directives: [], loop_blocks: [], ttl_seconds: 10,
    });
    const ok = applyDeltas([], 10);
    expect(ok).toBe(false);
    expect(isCacheHealthy()).toBe(false);
  });

  test("snapshot tenant mismatch poisons the cache", () => {
    applySnapshot({
      schema_version: 1, type: "snapshot", version: 1,
      tenant_id: "A", project_id: "p", directives: [], loop_blocks: [], ttl_seconds: 10,
    });
    const ok = applySnapshot({
      schema_version: 1, type: "snapshot", version: 2,
      tenant_id: "B", project_id: "p", directives: [], loop_blocks: [], ttl_seconds: 10,
    });
    expect(ok).toBe(false);
    expect(isCacheHealthy()).toBe(false);
  });

  test("snapshot missing ids is refused and pin arms after heal", () => {
    // Failure-shape 1: a first snapshot with NO tenant_id is refused (cache
    // poisoned → inline /check), so the cross-tenant pin never arms from a
    // malformed snapshot. A later well-formed snapshot heals the cache AND arms
    // the pin, after which a different-tenant snapshot is correctly rejected.
    expect(applySnapshot({
      schema_version: 1, type: "snapshot", version: 1,
      project_id: "p", directives: [], loop_blocks: [], ttl_seconds: 10,
    })).toBe(false);
    expect(isCacheHealthy()).toBe(false);

    // A well-formed snapshot heals the cache and arms the pin from a clean state.
    expect(applySnapshot({
      schema_version: 1, type: "snapshot", version: 2,
      tenant_id: "A", project_id: "p", directives: [], loop_blocks: [], ttl_seconds: 10,
    })).toBe(true);
    expect(isCacheHealthy()).toBe(true);

    // Pin is now armed: a different-tenant snapshot is rejected.
    expect(applySnapshot({
      schema_version: 1, type: "snapshot", version: 3,
      tenant_id: "B", project_id: "p", directives: [], loop_blocks: [], ttl_seconds: 10,
    })).toBe(false);
    expect(isCacheHealthy()).toBe(false);
  });

  test("snapshot with tenant_id but no project_id does not arm a half-pin", () => {
    // Failure-shape 2: refused, and must NOT arm a half-pin (project=null). Proof:
    // a later snapshot with the same tenant AND a real project_id still applies —
    // a half-pin would reject it as a project mismatch forever.
    expect(applySnapshot({
      schema_version: 1, type: "snapshot", version: 1,
      tenant_id: "A", directives: [], loop_blocks: [], ttl_seconds: 10,
    })).toBe(false);
    expect(isCacheHealthy()).toBe(false);

    expect(applySnapshot({
      schema_version: 1, type: "snapshot", version: 2,
      tenant_id: "A", project_id: "realproj", directives: [], loop_blocks: [], ttl_seconds: 10,
    })).toBe(true);
    expect(isCacheHealthy()).toBe(true);
  });

  test("empty-string tenant_id is treated as absent (refused)", () => {
    // Failure-shape 3: an empty-string tenant_id is as good as absent — refused.
    expect(applySnapshot({
      schema_version: 1, type: "snapshot", version: 1,
      tenant_id: "", project_id: "p", directives: [], loop_blocks: [], ttl_seconds: 10,
    })).toBe(false);
    expect(isCacheHealthy()).toBe(false);
  });

  test("unknown delta op is skipped and version advances", () => {
    applySnapshot({
      schema_version: 1, type: "snapshot", version: 1,
      tenant_id: "t", project_id: "p", directives: [], loop_blocks: [], ttl_seconds: 10,
    });
    const ok = applyDeltas([{ op: "future_unknown", data: 42 } as unknown as { op: string }], 2);
    expect(ok).toBe(true);
    expect(getPackVersion()).toBe(2);
  });

  test("invalidate marks unhealthy", () => {
    applySnapshot({
      schema_version: 1, type: "snapshot", version: 1,
      tenant_id: "t", project_id: "p", directives: [], loop_blocks: [], ttl_seconds: 10,
    });
    invalidatePack();
    expect(isCacheHealthy()).toBe(false);
    expect(getPack()).toBeNull();
  });
});

describe("never-fail: observations queue", () => {
  beforeEach(() => resetPack());

  test("non-object observations are ignored", () => {
    pushObservation("not an obj" as unknown);
    pushObservation(null as unknown);
    expect(drainObservations()).toEqual([]);
  });

  test("drain returns then clears", () => {
    pushObservation({ rule_id: "r1", outcome: "would_block", mode: "dry_run" });
    expect(drainObservations()).toHaveLength(1);
    expect(drainObservations()).toEqual([]);
  });
});

describe("never-fail: runtime detection", () => {
  test("auto returns one of three modes", () => {
    expect(["daemon", "serverless", "edge"]).toContain(detectDeploymentMode());
  });

  test("invalid string falls back to auto", () => {
    expect(["daemon", "serverless", "edge"]).toContain(resolveDeployment("not_a_real_mode"));
  });
});

describe("never-fail: embedding composition under malformed input", () => {
  test("non-string array items still produce role='input' markers", async () => {
    const { buildPromptComposition } = await import("../src/composition");
    const result = buildPromptComposition(
      "openai",
      { input: [{}, 3.14, null, undefined] },
      "embedding",
    );
    expect(Array.isArray(result)).toBe(true);
    for (const entry of result) {
      expect(entry.role).toBe("input");
    }
  });

  test("empty kwargs returns []", async () => {
    const { buildPromptComposition } = await import("../src/composition");
    expect(buildPromptComposition("openai", {}, "embedding")).toEqual([]);
  });

  test("buildResponseComposition handles unknown embedding shape gracefully", async () => {
    const { buildResponseComposition } = await import("../src/composition");
    // An unknown shape that isn't in EMBEDDING_SHAPES — must not throw.
    const result = buildResponseComposition(
      "openai",
      { something: "weird" },
      "made_up_shape",
    );
    expect(Array.isArray(result)).toBe(true);
  });
});

// ── dry_run = enforce parity: same local-eval + /check, suppress the action ──
describe("enforcer: dry_run parity vs enforce", () => {
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
  const REROUTE_SNAPSHOT = {
    schema_version: 1, type: "snapshot", version: 1,
    tenant_id: "t", project_id: "p", ttl_seconds: 600, loop_blocks: [],
    directives: [
      {
        id: "r2", kind: "REROUTE", mode: "enforce", priority: 10,
        selector: { match: null, group_by: [] },
        reroute: { from: {}, to: { provider: "openai", model: "gpt-3.5-turbo" } },
      },
    ],
  };

  function makeClient(firewall: "enforce" | "dry_run" | "off") {
    // Direct construction (not init()) so no SSE stream starts; unreachable
    // baseUrl makes any background tp.log() fail-open silently.
    const client = new TokenPolice({
      apiKey: "tp_sk_test123",
      baseUrl: "http://localhost:99999",
      timeout: 0.1,
      firewall,
      deployment: "daemon",
    });
    setClient(client);
    return client;
  }

  beforeEach(() => resetPack());

  test("(a) dry_run still calls /check but NEVER throws on a block decision", async () => {
    const client = makeClient("dry_run");
    applySnapshot(BLOCK_SNAPSHOT);
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({ status: "blocked", reason: "x" } as never);
    await expect(enforcerTest._runAsyncCheck({ model: "gpt-4" }, "openai")).resolves.toBeUndefined();
    expect(checkSpy).toHaveBeenCalledTimes(1); // full parity: dry_run hits /check
    checkSpy.mockRestore();
  });

  test("(b) dry_run does NOT mutate body for a reroute decision", async () => {
    const client = makeClient("dry_run");
    applySnapshot(REROUTE_SNAPSHOT);
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({ status: "allowed" } as never);
    const body: Record<string, unknown> = { model: "gpt-4" };
    await enforcerTest._runAsyncCheck(body, "openai");
    expect(checkSpy).toHaveBeenCalledTimes(1);
    expect(body.model).toBe("gpt-4"); // suppressed — model untouched
    checkSpy.mockRestore();
  });

  test("(c) enforce STILL throws TokenPoliceBlockedError on a verified block", async () => {
    const client = makeClient("enforce");
    applySnapshot(BLOCK_SNAPSHOT);
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({ status: "blocked", reason: "x" } as never);
    await expect(enforcerTest._runAsyncCheck({ model: "gpt-4" }, "openai")).rejects.toBeInstanceOf(
      TokenPoliceBlockedError,
    );
    checkSpy.mockRestore();
  });

  // (d) C-12: the local-block /log emission (_emitLocalBlockLog) must carry a
  // real span_id/trace_id like every other /log row — without one the row is
  // invisible to the collector's per-span idempotency guard, so an infra
  // replay of the same body would double-count budgets and audit rows.
  test("(d) a verified block's /log emission carries a non-empty span_id + trace_id (C-12)", async () => {
    const client = makeClient("enforce");
    applySnapshot(BLOCK_SNAPSHOT);
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({ status: "blocked", reason: "x" } as never);
    const logSpy = vi.spyOn(client, "log").mockImplementation(() => {});
    await expect(enforcerTest._runAsyncCheck({ model: "gpt-4" }, "openai")).rejects.toBeInstanceOf(
      TokenPoliceBlockedError,
    );
    expect(logSpy).toHaveBeenCalledTimes(1);
    // client.log() positional arg 10 is `span` (see tests/parentSpanIdCompat.test.ts).
    const span = logSpy.mock.calls[0][10] as any;
    expect(span.span_id).toMatch(/^[0-9a-f]{16}$/);
    expect(span.trace_id).toBeTruthy();
    checkSpy.mockRestore();
    logSpy.mockRestore();
  });

  test("off never runs an active pre-flight (no /check)", async () => {
    const client = makeClient("off");
    applySnapshot(BLOCK_SNAPSHOT);
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({ status: "blocked" } as never);
    await enforcerTest._runAsyncCheck({ model: "gpt-4" }, "openai");
    expect(checkSpy).not.toHaveBeenCalled();
    checkSpy.mockRestore();
  });
});
