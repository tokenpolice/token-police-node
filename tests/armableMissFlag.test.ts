/**
 * `armableMiss` — local-evaluator metadata for the C-14 stale-stream gate.
 *
 * Set only when an entity-gated directive's selector matched the call but the
 * computed group tag was absent from its streamed `entities` set AND the
 * directive would otherwise ENFORCE (not a directive-level dry_run, and not
 * `forceShadow`) — i.e. a missed `entity_blocked` / `entity_rerouted` delta
 * that would have flipped the decision. Never affects the decision object
 * itself, and is present as a key ONLY when true — an allow with no armable
 * miss stays byte-identical to the pre-fix shape (no `armableMiss` key at all).
 *
 * Sibling: token-police-python/tests/test_armable_miss.py pins the same
 * scenarios against the Python SDK.
 */
import { describe, it, expect } from "vitest";
import { evaluate } from "../src/localEvaluator";

const SESSION = { userId: "u1" };
const CALL = { model: "gpt-4", provider: "openai" };

function entityBlockDirective(opts: {
  mode?: "enforce" | "dry_run";
  entities?: string[];
  matchField?: string | null;
} = {}) {
  return {
    id: "eb1",
    kind: "ENTITY_BLOCK",
    mode: opts.mode ?? "enforce",
    priority: 10,
    selector: {
      match:
        opts.matchField === null
          ? null
          : { field: opts.matchField ?? "user_id", operator: "EXISTS" },
      group_by: ["user_id"],
    },
    entities: opts.entities ?? [],
  };
}

function entityRerouteDirective(opts: {
  mode?: "enforce" | "dry_run";
  entities?: string[] | null;
} = {}) {
  return {
    id: "er1",
    kind: "REROUTE",
    mode: opts.mode ?? "enforce",
    priority: 10,
    selector: { match: { field: "user_id", operator: "EXISTS" }, group_by: ["user_id"] },
    entities: "entities" in opts ? opts.entities : [],
    reroute: { from: null, to: { provider: "openai", model: "gpt-3.5-turbo" } },
  };
}

describe("armableMiss — ENTITY_BLOCK (Node)", () => {
  it("enforce directive, selector matches, tag NOT armed → armableMiss true, decision stays allowed", () => {
    const pack = { directives: [entityBlockDirective({ entities: [] })], loop_blocks: [] };
    const res = evaluate(pack, SESSION, CALL);
    expect(res.decision).toEqual({ status: "allowed", rule_id: null, mode: "enforce", reroute: null });
    expect(res.observations).toEqual([]);
    expect(res.armableMiss).toBe(true);
  });

  it("an allow with NO armable miss has no armableMiss key at all (byte-identical to the pre-fix shape)", () => {
    const pack = { directives: [], loop_blocks: [] };
    const res = evaluate(pack, SESSION, CALL);
    expect(res).toEqual({
      decision: { status: "allowed", rule_id: null, mode: "enforce", reroute: null },
      observations: [],
    });
    expect("armableMiss" in res).toBe(false);
  });

  it("tag armed → normal block, no armableMiss (the miss branch is never taken)", () => {
    const pack = { directives: [entityBlockDirective({ entities: ["u1"] })], loop_blocks: [] };
    const res = evaluate(pack, SESSION, CALL);
    expect(res.decision.status).toBe("blocked");
    expect(res.decision.rule_id).toBe("eb1");
    expect("armableMiss" in res).toBe(false);
  });

  it("selector non-match → no armableMiss even though the tag isn't armed", () => {
    const pack = {
      directives: [entityBlockDirective({ entities: [], matchField: "does_not_exist" })],
      loop_blocks: [],
    };
    const res = evaluate(pack, SESSION, CALL);
    expect(res.decision.status).toBe("allowed");
    expect("armableMiss" in res).toBe(false);
  });

  it("DIRECTIVE-level dry_run entity miss → armableMiss NOT set (an armed delta would only ever produce a would_block observation, never a real remote block)", () => {
    const pack = { directives: [entityBlockDirective({ mode: "dry_run", entities: [] })], loop_blocks: [] };
    const res = evaluate(pack, SESSION, CALL);
    expect(res.decision.status).toBe("allowed");
    // dry_run only emits would_block on a HIT, not a miss — the miss branch
    // `continue`s before the dry_run observation is ever built.
    expect(res.observations).toEqual([]);
    expect("armableMiss" in res).toBe(false);
  });

  it("forceShadow=true (client-level shadow) entity miss → armableMiss NOT set even for an enforce-mode directive", () => {
    const pack = { directives: [entityBlockDirective({ mode: "enforce", entities: [] })], loop_blocks: [] };
    const res = evaluate(pack, SESSION, CALL, true);
    expect(res.decision.status).toBe("allowed");
    expect("armableMiss" in res).toBe(false);
  });
});

describe("armableMiss — entity-gated REROUTE (Node)", () => {
  it("enforce reroute directive, selector matches, tag NOT armed → armableMiss true, decision stays allowed", () => {
    const pack = { directives: [entityRerouteDirective({ entities: [] })], loop_blocks: [] };
    const res = evaluate(pack, SESSION, CALL);
    expect(res.decision.status).toBe("allowed");
    expect(res.armableMiss).toBe(true);
  });

  it("DIRECTIVE-level dry_run reroute miss → armableMiss NOT set", () => {
    const pack = { directives: [entityRerouteDirective({ mode: "dry_run", entities: [] })], loop_blocks: [] };
    const res = evaluate(pack, SESSION, CALL);
    expect(res.decision.status).toBe("allowed");
    expect("armableMiss" in res).toBe(false);
  });

  it("an unconditional (non-entity-gated) REROUTE directive — entities absent — never sets armableMiss", () => {
    const pack = { directives: [entityRerouteDirective({ entities: null })], loop_blocks: [] };
    const res = evaluate(pack, SESSION, CALL);
    // Unconditional: applies whenever it matches — no entities check at all.
    expect(res.decision.status).toBe("rerouted");
    expect("armableMiss" in res).toBe(false);
  });
});

describe("armableMiss — unconditional directives never set the flag (Node)", () => {
  it("UNCONDITIONAL_BLOCK match → blocked, no armableMiss key", () => {
    const pack = {
      directives: [
        { id: "u1", kind: "UNCONDITIONAL_BLOCK", mode: "enforce", priority: 10, selector: { match: null, group_by: [] } },
      ],
      loop_blocks: [],
    };
    const res = evaluate(pack, SESSION, CALL);
    expect(res.decision.status).toBe("blocked");
    expect("armableMiss" in res).toBe(false);
  });
});
