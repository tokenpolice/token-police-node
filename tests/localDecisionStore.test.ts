/**
 * B3 — per-call keyed store for the applied `local_decision` audit stash.
 *
 * Bug: `local_decision` (the SOLE provenance the collector turns into a
 * `REQUEST_REROUTED` audit event) used to live in ONE flat slot on the
 * session (`session._local_decision`). N concurrent calls overwrote each
 * other (N-1 decisions lost) and the first call to finish drained the
 * survivor: one applied-reroute event per burst, stamped on an arbitrary row
 * with an arbitrary sibling's rule/from/to — and in a heterogeneous burst, an
 * un-rerouted call's row could carry a sibling's decision (a false-positive
 * REQUEST_REROUTED).
 *
 * Fix: `session._local_decisions` — a bounded per-call keyed FIFO list — via
 * `stashLocalDecision` / `claimLocalDecision` in `src/localDecisionStore.ts`.
 * This file covers:
 *
 *   (1) store unit tests — claim precedence, at-most-one, replace, cap,
 *       sweep, no-stale-claim, hostile inputs (mirrors the Python twin's
 *       unit tests, minus the lock — Node has none, see the store's header);
 *   (2) the greedy drain-all window (state.setObservationsDrainAll /
 *       begin/endObservationsDrainAll) must NOT cause claimLocalDecision to
 *       return a foreign-keyed entry — the store deliberately never honors
 *       that window (D3's difference #2 from the obs queue);
 *   (3) a Promise.all burst at the enforcer level (mirrors
 *       tests/preflightCtxF7.test.ts's harness): N concurrent /check calls on
 *       ONE shared session, each carrying its OWN decision to its own "row".
 */
import { describe, it, expect, afterEach, vi } from "vitest";
import {
  applySnapshot,
  resetPack,
  setClient,
  getCurrentObsKey,
  runWithCallObsScope,
  beginObservationsDrainAll,
  endObservationsDrainAll,
  setObservationsDrainAll,
} from "../src/state";
import { session as tpSession, getCurrentSession } from "../src/context";
import { TokenPolice } from "../src/client";
import { __test__ as enforcerTest } from "../src/enforcer";
import {
  stashLocalDecision,
  claimLocalDecision,
  LOCAL_DECISION_STALE_MS,
  type LocalDecision,
} from "../src/localDecisionStore";

const { _runAsyncCheck } = enforcerTest;

// ══════════════════════════════════════════════════════════════════════════
// (1) Store unit tests
// ══════════════════════════════════════════════════════════════════════════
describe("localDecisionStore — unit", () => {
  it("claim prefers the own-key entry over an untagged one", () => {
    const session: Record<string, unknown> = {};
    stashLocalDecision(session, { id: "untagged" }, null);
    stashLocalDecision(session, { id: "own" }, "key-a");

    expect(claimLocalDecision(session, "key-a")).toEqual({ id: "own" });
    // The untagged entry is untouched — reachable by its own eventual claim.
    const remaining = (session._local_decisions as Array<{ ld: unknown }>).map((e) => e.ld);
    expect(remaining).toEqual([{ id: "untagged" }]);
  });

  it("claim falls back to untagged when there is no own-key entry", () => {
    const session: Record<string, unknown> = {};
    stashLocalDecision(session, { id: "untagged" }, null);

    expect(claimLocalDecision(session, "some-other-key")).toEqual({ id: "untagged" });
    expect(session._local_decisions).toEqual([]);
  });

  it("claim is at most one, even with multiple untagged entries", () => {
    const session: Record<string, unknown> = {};
    stashLocalDecision(session, { id: "a" }, null);
    stashLocalDecision(session, { id: "b" }, null);

    // Newest untagged entry wins; exactly one is consumed.
    expect(claimLocalDecision(session, "any-key")).toEqual({ id: "b" });
    const remaining = (session._local_decisions as Array<{ ld: unknown }>).map((e) => e.ld);
    expect(remaining).toEqual([{ id: "a" }]);
  });

  it("same-key stash REPLACES, never appends", () => {
    const session: Record<string, unknown> = {};
    stashLocalDecision(session, { id: "first" }, "key-a");
    stashLocalDecision(session, { id: "second" }, "key-a");

    expect((session._local_decisions as unknown[]).length).toBe(1);
    expect(claimLocalDecision(session, "key-a")).toEqual({ id: "second" });
  });

  it("a foreign-key claim returns undefined and leaves the entry reachable", () => {
    const session: Record<string, unknown> = {};
    stashLocalDecision(session, { id: "mine" }, "key-a");

    expect(claimLocalDecision(session, "key-b")).toBeUndefined();
    expect((session._local_decisions as unknown[]).length).toBe(1);
    expect(claimLocalDecision(session, "key-a")).toEqual({ id: "mine" });
  });

  it("caps at 64 entries, dropping the oldest", () => {
    const session: Record<string, unknown> = {};
    const cap = 64; // LOCAL_DECISION_CAP in src/localDecisionStore.ts (not exported)
    const total = cap + 6;
    for (let i = 0; i < total; i += 1) {
      stashLocalDecision(session, { id: i }, `key-${i}`);
    }

    const entries = session._local_decisions as Array<{ ld: { id: number } }>;
    expect(entries.length).toBe(cap);
    const ids = entries.map((e) => e.ld.id);
    // Oldest 6 (0..5) dropped; 6..(total-1) survive, oldest-first.
    expect(ids).toEqual(Array.from({ length: cap }, (_, i) => total - cap + i));
  });

  it("sweeps an expired entry on claim — unreachable even by its OWN key (no stale-claim)", () => {
    const session: Record<string, unknown> = {};
    stashLocalDecision(session, { id: "stale" }, "key-a");
    const entries = session._local_decisions as Array<{ ts: number }>;
    entries[0].ts -= LOCAL_DECISION_STALE_MS + 1;

    expect(claimLocalDecision(session, "key-a")).toBeUndefined();
    expect(session._local_decisions).toEqual([]);
  });

  it("sweeps an expired entry on stash", () => {
    const session: Record<string, unknown> = {};
    stashLocalDecision(session, { id: "stale" }, "key-a");
    const entries = session._local_decisions as Array<{ ts: number }>;
    entries[0].ts -= LOCAL_DECISION_STALE_MS + 1;

    stashLocalDecision(session, { id: "fresh" }, "key-b");

    const remaining = (session._local_decisions as Array<{ ld: { id: string } }>).map(
      (e) => e.ld.id,
    );
    expect(remaining).toEqual(["fresh"]);
  });

  it("does not hand an expired untagged entry to a stranger's claim", () => {
    const session: Record<string, unknown> = {};
    stashLocalDecision(session, { id: "stale" }, null);
    const entries = session._local_decisions as Array<{ ts: number }>;
    entries[0].ts -= LOCAL_DECISION_STALE_MS + 1;

    expect(claimLocalDecision(session, "whoever-asks")).toBeUndefined();
    expect(session._local_decisions).toEqual([]);
  });

  // ── Golden rule: every helper is fail-open — never throws, on anything ──
  it("hostile inputs never throw", () => {
    expect(() => stashLocalDecision(null, { id: 1 }, "k")).not.toThrow();
    expect(() => stashLocalDecision(undefined, { id: 1 }, "k")).not.toThrow();
    expect(claimLocalDecision(null, "k")).toBeUndefined();
    expect(claimLocalDecision(undefined, "k")).toBeUndefined();

    // Missing attribute (never stashed to).
    const s: Record<string, unknown> = {};
    expect(claimLocalDecision(s, "k")).toBeUndefined();

    // Corrupt attribute (not an array) — claim degrades to no-op; a later
    // stash self-heals by replacing it with a fresh array.
    const s2: Record<string, unknown> = { _local_decisions: "not-an-array" };
    expect(claimLocalDecision(s2, "k")).toBeUndefined();
    stashLocalDecision(s2, { id: 1 }, "k");
    expect(Array.isArray(s2._local_decisions)).toBe(true);

    // Falsy ld (null / undefined) — never stashed.
    const s3: Record<string, unknown> = {};
    stashLocalDecision(s3, null as unknown as LocalDecision, "k");
    stashLocalDecision(s3, undefined as unknown as LocalDecision, "k");
    expect(s3._local_decisions).toBeUndefined();

    // Non-string (but truthy) key must not throw, and stash/claim stay
    // consistent as long as the same value is used both times.
    const s4: Record<string, unknown> = {};
    // @ts-expect-error — deliberately hostile key type for the fail-open check
    stashLocalDecision(s4, { id: "num" }, 12345);
    // @ts-expect-error — same hostile key on claim
    expect(claimLocalDecision(s4, 12345)).toEqual({ id: "num" });
  });
});

// ══════════════════════════════════════════════════════════════════════════
// (2) The greedy drain-all window must NOT leak foreign entries
// ══════════════════════════════════════════════════════════════════════════
describe("localDecisionStore — greedy drain-all window (D3 difference #2)", () => {
  afterEach(() => {
    setObservationsDrainAll(false);
  });

  it("claimLocalDecision ignores the obs queue's greedy drain-all window entirely", () => {
    const session: Record<string, unknown> = {};
    stashLocalDecision(session, { id: "mine" }, "key-a");
    stashLocalDecision(session, { id: "sibling" }, "key-b");

    setObservationsDrainAll(true);
    try {
      // Even with the obs queue's greedy window open (a terminal flush would
      // drain EVERY observation regardless of key), the decision store must
      // behave exactly as normal: own-key first, and a foreign key must
      // never come back — a stranded decision beats a misattributed one.
      expect(claimLocalDecision(session, "key-a")).toEqual({ id: "mine" });
      const remaining = (session._local_decisions as Array<{ ld: unknown }>).map((e) => e.ld);
      expect(remaining).toEqual([{ id: "sibling" }]);

      // A key that owns nothing must still get nothing — not the sibling's
      // entry, even though a greedy observations drain would ship everything.
      expect(claimLocalDecision(session, "unrelated-key")).toBeUndefined();
      expect(
        (session._local_decisions as Array<{ ld: unknown }>).map((e) => e.ld),
      ).toEqual([{ id: "sibling" }]);
    } finally {
      setObservationsDrainAll(false);
    }
  });

  it("same property holds using the real begin/end pair (forceFlush-style)", () => {
    const session: Record<string, unknown> = {};
    stashLocalDecision(session, { id: "mine" }, "key-a");
    stashLocalDecision(session, { id: "sibling" }, "key-b");

    beginObservationsDrainAll();
    try {
      expect(claimLocalDecision(session, "unrelated-key")).toBeUndefined();
    } finally {
      endObservationsDrainAll();
    }
    // Both entries survive the greedy window untouched.
    const remaining = (session._local_decisions as Array<{ ld: { id: string } }>)
      .map((e) => e.ld.id)
      .sort();
    expect(remaining).toEqual(["mine", "sibling"]);
  });
});

// ══════════════════════════════════════════════════════════════════════════
// (3) Enforcer-level Promise.all burst (mirrors tests/preflightCtxF7.test.ts)
// ══════════════════════════════════════════════════════════════════════════
function makeClient(firewall: "enforce" | "dry_run" | "off" = "enforce"): TokenPolice {
  const client = new TokenPolice({
    apiKey: "tp_sk_test_keyed_store",
    baseUrl: "http://localhost:59999",
    timeout: 0.1,
    firewall,
    deployment: "daemon",
  } as never);
  setClient(client);
  return client;
}

function stubClient(client: TokenPolice, checkResult: unknown = { status: "allowed" }) {
  vi.spyOn(client, "check").mockResolvedValue(checkResult as never);
  vi.spyOn(client, "log").mockImplementation(() => undefined as never);
  return client;
}

const rerouteAllSnapshot = (ruleId: string) => ({
  schema_version: 1,
  type: "snapshot",
  version: 1,
  tenant_id: "t",
  project_id: "p",
  ttl_seconds: 600,
  loop_blocks: [],
  directives: [
    {
      id: ruleId,
      kind: "REROUTE",
      mode: "enforce",
      priority: 10,
      selector: { match: null, group_by: [] },
      reroute: { from: {}, to: { provider: "openai", model: "gpt-4o-mini" } },
    },
  ],
});

const rerouteForModelsSnapshot = (
  rules: Array<{ id: string; fromModel: string; toModel: string }>,
) => ({
  schema_version: 1,
  type: "snapshot",
  version: 1,
  tenant_id: "t",
  project_id: "p",
  ttl_seconds: 600,
  loop_blocks: [],
  directives: rules.map((r) => ({
    id: r.id,
    kind: "REROUTE",
    mode: "enforce",
    priority: 10,
    selector: { match: { field: "model", operator: "EQ", value: r.fromModel }, group_by: [] },
    reroute: { from: {}, to: { provider: "openai", model: r.toModel } },
  })),
});

/**
 * Run one call's /check on the SHARED (ambient) session, then claim +
 * return whatever THIS call's own key found. `runWithCallObsScope` is the
 * exact primitive `_withCallObsScope` wraps every patched provider call
 * with in production (src/enforcer.ts) — using it directly here exercises
 * the real per-call key threading without needing a fake provider SDK.
 *
 * The `setTimeout(0)` between the check and the claim stands in for the
 * real gap in production — the provider's actual LLM HTTP call — a genuine
 * suspension point where OTHER concurrent calls' checks+stashes land on
 * this SAME shared session before this one claims. Without an explicit
 * macrotask yield here, Node's microtask scheduling could let a mocked
 * check+stash+claim run close enough to atomically that it would pass even
 * against the OLD single-slot bug and prove nothing.
 */
async function checkedCall(model: string): Promise<LocalDecision | undefined> {
  return runWithCallObsScope(async () => {
    await _runAsyncCheck({ model }, "openai", null, true, false, undefined);
    const key = getCurrentObsKey() ?? null;
    await new Promise((resolve) => setTimeout(resolve, 0));
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return claimLocalDecision(getCurrentSession() as any, key);
  });
}

describe("localDecisionStore — enforcer-level Promise.all burst", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    resetPack();
  });

  it("N concurrent same-rule reroutes on ONE session each land their own decision", async () => {
    const client = makeClient();
    stubClient(client);
    applySnapshot(rerouteAllSnapshot("rr"));

    const n = 6;
    const results = await tpSession({ userId: "u1" }, async () =>
      Promise.all(Array.from({ length: n }, () => checkedCall("gpt-4o"))),
    );

    expect(results).toHaveLength(n);
    for (const ld of results) {
      expect(ld).toBeTruthy();
      expect((ld as Record<string, unknown>).outcome).toBe("rerouted");
      expect((ld as Record<string, unknown>).rule_id).toBe("rr");
    }
  });

  it("heterogeneous burst: distinct rules + one clean call, no cross-contamination", async () => {
    const client = makeClient();
    stubClient(client);
    applySnapshot(
      rerouteForModelsSnapshot([
        { id: "rr-0", fromModel: "m0", toModel: "t0" },
        { id: "rr-1", fromModel: "m1", toModel: "t1" },
        { id: "rr-2", fromModel: "m2", toModel: "t2" },
      ]),
    );

    const [ld0, ld1, ld2, ldClean] = await tpSession({ userId: "u1" }, async () =>
      Promise.all([
        checkedCall("m0"),
        checkedCall("m1"),
        checkedCall("m2"),
        checkedCall("clean-model"), // matches no rule
      ]),
    );

    expect((ld0 as Record<string, unknown>).rule_id).toBe("rr-0");
    expect((ld1 as Record<string, unknown>).rule_id).toBe("rr-1");
    expect((ld2 as Record<string, unknown>).rule_id).toBe("rr-2");
    for (const ld of [ld0, ld1, ld2]) {
      expect((ld as Record<string, unknown>).outcome).toBe("rerouted");
    }
    // No false positive on the clean row.
    expect(ldClean).toBeUndefined();
  });
});
