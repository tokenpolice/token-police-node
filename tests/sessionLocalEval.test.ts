import { describe, it, expect, vi, afterEach } from "vitest";
import { evaluate } from "../src/localEvaluator";
import { _stashLocalDecision } from "../src/enforcer";
import { TokenPolice } from "../src/client";

// Per-session sliding-budget rules ship to the SDK as ordinary ENTITY_BLOCK /
// REROUTE directives whose group_by is ["session_id"]. The local fast path must
// surface session_id (from the session object) so it can compute the per-session
// group tag and consult the directive's entities set — blocking sessions that are
// armed and skipping /check (allowed) for the rest.

const entityBlockPack = (entities: string[]) => ({
  directives: [
    {
      id: "rule_sess",
      kind: "ENTITY_BLOCK",
      mode: "enforce",
      selector: {
        match: { field: "session_id", operator: "EXISTS" },
        group_by: ["session_id"],
      },
      entities,
    },
  ],
});

describe("per-session local evaluation", () => {
  it("blocks a session whose id is in the directive's entities set", () => {
    const pack = entityBlockPack(["conv_42"]);
    const res = evaluate(pack, { sessionId: "conv_42" }, {});
    expect(res.decision.status).toBe("blocked");
    expect(res.decision.rule_id).toBe("rule_sess");
  });

  it("allows (no local block) a session not in the entities set — caller skips /check", () => {
    const pack = entityBlockPack(["conv_42"]);
    const res = evaluate(pack, { sessionId: "conv_99" }, {});
    expect(res.decision.status).toBe("allowed");
    expect(res.observations).toHaveLength(0);
  });

  it("reads session_id from the snake_case alias too", () => {
    const pack = entityBlockPack(["conv_42"]);
    const res = evaluate(pack, { session_id: "conv_42" }, {});
    expect(res.decision.status).toBe("blocked");
  });

  it("a session with no id collapses to 'unknown' and is not blocked unless armed", () => {
    const pack = entityBlockPack(["conv_42"]);
    const res = evaluate(pack, {}, {});
    expect(res.decision.status).toBe("allowed");
  });
});

describe("_stashLocalDecision null/undefined guard", () => {
  // A missing session must be a silent no-op (fail-open), never a TypeError.
  // NOTE (negative control / assertion 16): if `if (!session) return;` is removed
  // from _stashLocalDecision, the two calls below throw
  // `TypeError: Cannot set properties of null (setting '_local_decision')` and this
  // block goes RED — proving the guard is exercised (non-vacuous).
  it("null session no-ops and does not throw (direct call)", () => {
    expect(() => _stashLocalDecision(null, "blocked", "r1", true)).not.toThrow();
  });

  it("undefined session no-ops and does not throw (direct call)", () => {
    expect(() => _stashLocalDecision(undefined, "blocked", "r1", true)).not.toThrow();
  });

  it("real session: stashed decision is byte-for-byte unchanged (no reroute)", () => {
    // No obs scope is active in a plain unit-test call, so the stash lands
    // untagged (key=null) in the keyed store — same-shape entry as before,
    // just addressed via `_local_decisions` instead of the old flat slot.
    const session: Record<string, unknown> = { userId: "u1" };
    _stashLocalDecision(session, "blocked", "r1", true);
    const entries = session._local_decisions as Array<{ ld: unknown; key: unknown }>;
    expect(entries).toHaveLength(1);
    expect(entries[0].key).toBeNull();
    expect(entries[0].ld).toEqual({
      outcome: "blocked",
      rule_id: "r1",
      mode: "enforce",
      verified_by_check: true,
    });
  });

  it("real session: reroute key is present only when reroute is truthy", () => {
    const withReroute: Record<string, unknown> = {};
    _stashLocalDecision(withReroute, "rerouted", "r2", true, { to: "x" });
    const entries = withReroute._local_decisions as Array<{ ld: Record<string, unknown> }>;
    expect(entries[entries.length - 1].ld).toEqual({
      outcome: "rerouted",
      rule_id: "r2",
      mode: "enforce",
      verified_by_check: true,
      reroute: { to: "x" },
    });

    const noReroute: Record<string, unknown> = {};
    _stashLocalDecision(noReroute, "blocked", "r3", false);
    const noRerouteEntries = noReroute._local_decisions as Array<{ ld: Record<string, unknown> }>;
    expect(noRerouteEntries[noRerouteEntries.length - 1].ld).not.toHaveProperty("reroute");
  });
});

describe("check() sends session_id in the /check body", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("includes session_id when provided, omits it otherwise", async () => {
    const fetchMock = vi.spyOn(global, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ status: "allowed" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    const client = new TokenPolice({ apiKey: "tp_sk_test123", baseUrl: "http://localhost:3001" });

    await client.check("user1", "free", "default", "conv_42");
    const withSession = JSON.parse((fetchMock.mock.calls[0][1] as RequestInit).body as string);
    expect(withSession.session_id).toBe("conv_42");

    await client.check("user1", "free", "default", "");
    const noSession = JSON.parse((fetchMock.mock.calls[1][1] as RequestInit).body as string);
    expect("session_id" in noSession).toBe(false);
  });
});
