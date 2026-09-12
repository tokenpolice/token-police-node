/**
 * `rule_name` must flow from the directive pack through the local
 * evaluator and synthetic reroute into `_tp_routing`, and be omitted when
 * absent/empty (Node/Python shape parity).
 */
import { describe, it, expect, beforeEach } from "vitest";
import { __test__ } from "../src/enforcer";
import { session } from "../src/context";
import { evaluate } from "../src/localEvaluator";
import * as state from "../src/state";

const { _applyReroute } = __test__;

const reroutePack = (name?: string | null) => ({
  directives: [
    {
      id: "rule_rr",
      kind: "REROUTE",
      mode: "enforce",
      selector: { match: null, group_by: [] },
      ...(name != null && name !== "" ? { name } : name === "" ? { name: "" } : {}),
      reroute: {
        from: null,
        to: { provider: "openai", model: "gpt-4o-mini" },
      },
    },
  ],
});

describe("Rule_name on local-eval + _tp_routing", () => {
  it("local eval decision carries rule_name when pack has name", () => {
    const res = evaluate(
      reroutePack("Send free users to a cheaper model"),
      { paidPlan: "free" },
      { provider: "openai", model: "gpt-4o" },
    );
    expect(res.decision.status).toBe("rerouted");
    expect(res.decision.rule_id).toBe("rule_rr");
    expect(res.decision.rule_name).toBe("Send free users to a cheaper model");
  });

  it("local eval omits rule_name when pack has no name", () => {
    const res = evaluate(
      reroutePack(),
      {},
      { provider: "openai", model: "gpt-4o" },
    );
    expect(res.decision.status).toBe("rerouted");
    expect(res.decision.rule_name).toBeUndefined();
    expect(res.decision).not.toHaveProperty("rule_name");
  });

  it("local eval omits rule_name when pack name is empty string", () => {
    const res = evaluate(
      reroutePack(""),
      {},
      { provider: "openai", model: "gpt-4o" },
    );
    expect(res.decision.status).toBe("rerouted");
    expect(res.decision).not.toHaveProperty("rule_name");
  });

  // Observations, not just the enforce decision: a DRY_RUN rule never produces a
  // decision, so the observation was the only carrier of the name — and it
  // shipped without one, which is why the routing feed rendered rule UUIDs.
  it("would_reroute observation carries rule_name when pack has name", () => {
    const pack = reroutePack("Send free users to a cheaper model");
    pack.directives[0].mode = "dry_run";
    const res = evaluate(pack, { paidPlan: "free" }, { provider: "openai", model: "gpt-4o" });
    expect(res.decision.status).toBe("allowed");
    expect(res.observations).toHaveLength(1);
    expect(res.observations[0].outcome).toBe("would_reroute");
    expect(res.observations[0].rule_name).toBe("Send free users to a cheaper model");
  });

  it("would_reroute observation omits rule_name when pack has no name", () => {
    const pack = reroutePack();
    pack.directives[0].mode = "dry_run";
    const res = evaluate(pack, {}, { provider: "openai", model: "gpt-4o" });
    expect(res.observations).toHaveLength(1);
    expect(res.observations[0]).not.toHaveProperty("rule_name");
  });

  it("reroute_rejected observation carries rule_name (cross-provider)", () => {
    // Target provider != observed provider → rejected, not applied.
    const res = evaluate(
      reroutePack("Send free users to a cheaper model"),
      {},
      { provider: "anthropic", model: "claude-haiku-4-5" },
    );
    expect(res.observations).toHaveLength(1);
    expect(res.observations[0].outcome).toBe("reroute_rejected");
    expect(res.observations[0].rule_name).toBe("Send free users to a cheaper model");
  });

  it("would_block observation carries rule_name", () => {
    const res = evaluate(
      {
        directives: [
          {
            id: "rule_b",
            kind: "UNCONDITIONAL_BLOCK",
            mode: "dry_run",
            name: "Block everything",
            selector: { match: null, group_by: [] },
          },
        ],
      },
      {},
      { provider: "openai", model: "gpt-4o" },
    );
    expect(res.decision.status).toBe("allowed");
    expect(res.observations).toHaveLength(1);
    expect(res.observations[0].outcome).toBe("would_block");
    expect(res.observations[0].rule_name).toBe("Block everything");
  });

  it("_applyReroute stashes _tp_routing.rule_name when present", () => {
    const body: Record<string, unknown> = { model: "gpt-4o" };
    const s = session({ name: "t" }, (sess) => {
      _applyReroute(
        {
          reroute: {
            mode: "enforce",
            model: "gpt-4o-mini",
            provider: "openai",
            rule_id: "rule_rr",
            rule_name: "Send free users to a cheaper model",
            original: { provider: "openai", model: "gpt-4o" },
          },
        },
        body,
        "openai",
      );
      return sess;
    });
    expect(body.model).toBe("gpt-4o-mini");
    const routing = (s.metadata as Record<string, unknown>)._tp_routing as Record<
      string,
      unknown
    >;
    expect(routing).toBeDefined();
    expect(routing.rule_id).toBe("rule_rr");
    expect(routing.rule_name).toBe("Send free users to a cheaper model");
    expect(routing.actual_model).toBe("gpt-4o-mini");
  });

  it("_applyReroute omits rule_name key when absent", () => {
    const body: Record<string, unknown> = { model: "gpt-4o" };
    const s = session({ name: "t" }, (sess) => {
      _applyReroute(
        {
          reroute: {
            mode: "enforce",
            model: "gpt-4o-mini",
            provider: "openai",
            rule_id: "rule_rr",
            original: { provider: "openai", model: "gpt-4o" },
          },
        },
        body,
        "openai",
      );
      return sess;
    });
    const routing = (s.metadata as Record<string, unknown>)._tp_routing as Record<
      string,
      unknown
    >;
    expect(routing).toBeDefined();
    expect(routing.rule_id).toBe("rule_rr");
    expect(routing).not.toHaveProperty("rule_name");
  });

  it("_applyReroute omits rule_name key when empty string", () => {
    const body: Record<string, unknown> = { model: "gpt-4o" };
    const s = session({ name: "t" }, (sess) => {
      _applyReroute(
        {
          reroute: {
            mode: "enforce",
            model: "gpt-4o-mini",
            provider: "openai",
            rule_id: "rule_rr",
            rule_name: "",
            original: { provider: "openai", model: "gpt-4o" },
          },
        },
        body,
        "openai",
      );
      return sess;
    });
    const routing = (s.metadata as Record<string, unknown>)._tp_routing as Record<
      string,
      unknown
    >;
    expect(routing).toBeDefined();
    expect(routing).not.toHaveProperty("rule_name");
  });
});

// State B: the reroute directive comes from the /check HTTP response, so the
// local evaluator never runs and the observation is `_applyReroute`'s own.
// It shipped without `rule_name`, so the collector's customer-visible message
// fell back to the raw rule UUID.
describe("Rule_name on the State B reroute_rejected observation", () => {
  beforeEach(() => {
    // Drain any leftover observations from prior tests.
    try { state.drainObservations(); } catch { /* */ }
  });

  const rejectedObs = () =>
    state
      .drainObservations()
      .find((o: any) => o.outcome === "reroute_rejected");

  it("cross_provider_unsupported observation carries rule_name", () => {
    const body: Record<string, any> = { model: "gpt-4o" };
    const status = _applyReroute(
      {
        reroute: {
          mode: "enforce",
          model: "claude-haiku-4-5",
          provider: "anthropic", // != call provider → cross-provider reject
          rule_id: "rule_rr",
          rule_name: "Send free users to a cheaper model",
          original: { provider: "openai", model: "gpt-4o" },
        },
      },
      body,
      "openai",
    );
    expect(status).toBe("rejected");
    const obs = rejectedObs();
    expect(obs).toBeTruthy();
    expect(obs.outcome).toBe("reroute_rejected");
    expect(obs.rejection_reason).toBe("cross_provider_unsupported");
    expect(obs.rule_name).toBe("Send free users to a cheaper model");
    // The collector falls back to rule_id, so it must still be present.
    expect(obs.rule_id).toBe("rule_rr");
  });

  it("serving_unverified observation carries rule_name", () => {
    const body: Record<string, any> = { model: "gpt-4o" };
    const status = _applyReroute(
      {
        reroute: {
          mode: "enforce",
          model: "gpt-4o-mini",
          provider: "openai", // same provider — only the flag rejects
          rule_id: "rule_rr",
          rule_name: "Send free users to a cheaper model",
          original: { provider: "openai", model: "gpt-4o" },
        },
      },
      body,
      "openai",
      true, // servingUnverified
    );
    expect(status).toBe("rejected");
    const obs = rejectedObs();
    expect(obs).toBeTruthy();
    expect(obs.rejection_reason).toBe("serving_unverified");
    expect(obs.rule_name).toBe("Send free users to a cheaper model");
  });

  it("omits rule_name key when empty string", () => {
    const body: Record<string, any> = { model: "gpt-4o" };
    _applyReroute(
      {
        reroute: {
          mode: "enforce",
          model: "claude-haiku-4-5",
          provider: "anthropic",
          rule_id: "rule_rr",
          rule_name: "",
          original: { provider: "openai", model: "gpt-4o" },
        },
      },
      body,
      "openai",
    );
    const obs = rejectedObs();
    expect(obs).toBeTruthy();
    expect(obs.rule_id).toBe("rule_rr");
    expect(obs).not.toHaveProperty("rule_name");
  });

  it("omits rule_name key when absent", () => {
    const body: Record<string, any> = { model: "gpt-4o" };
    _applyReroute(
      {
        reroute: {
          mode: "enforce",
          model: "claude-haiku-4-5",
          provider: "anthropic",
          rule_id: "rule_rr",
          original: { provider: "openai", model: "gpt-4o" },
        },
      },
      body,
      "openai",
    );
    const obs = rejectedObs();
    expect(obs).toBeTruthy();
    expect(obs.rule_id).toBe("rule_rr");
    expect(obs).not.toHaveProperty("rule_name");
  });

  it("omits rule_name key when null (never emit a null key)", () => {
    const body: Record<string, any> = { model: "gpt-4o" };
    _applyReroute(
      {
        reroute: {
          mode: "enforce",
          model: "claude-haiku-4-5",
          provider: "anthropic",
          rule_id: "rule_rr",
          rule_name: null,
          original: { provider: "openai", model: "gpt-4o" },
        },
      },
      body,
      "openai",
    );
    const obs = rejectedObs();
    expect(obs).toBeTruthy();
    expect(obs.rule_id).toBe("rule_rr");
    expect(obs).not.toHaveProperty("rule_name");
  });
});
