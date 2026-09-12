import { describe, it, expect, beforeEach } from "vitest";
import { __test__ } from "../src/enforcer";
import { session } from "../src/context";
import { evaluate } from "../src/localEvaluator";
import * as state from "../src/state";
import { isNoopReroute, stripDateSuffix } from "../src/rerouteNoop";

// A REROUTE rule targeting the alias `anthropic/claude-haiku-4-5` fires
// against apps that pin the dated snapshot `claude-haiku-4-5-20251001`. The
// rewrite is a no-op cost-wise but produced phantom REQUEST_REROUTED events and
// silently unpinned the customer's snapshot. Both the apply path (State B) and
// the local evaluator (State A) must treat it as nothing-to-do.

const { _applyReroute } = __test__;

describe("StripDateSuffix", () => {
  it("strips exactly one trailing dated-snapshot suffix in all three forms", () => {
    expect(stripDateSuffix("claude-haiku-4-5-20251001")).toBe("claude-haiku-4-5");
    expect(stripDateSuffix("gpt-4o-2024-08-06")).toBe("gpt-4o");
    expect(stripDateSuffix("claude-3-5-sonnet@20240620")).toBe("claude-3-5-sonnet");
  });

  it("strips only ONE suffix", () => {
    expect(stripDateSuffix("m-20240101-20250101")).toBe("m-20240101");
  });

  it("leaves everything else alone (conservative by design)", () => {
    expect(stripDateSuffix("claude-3-5-sonnet-latest")).toBe("claude-3-5-sonnet-latest");
    expect(stripDateSuffix("gemini-1.5-pro-002")).toBe("gemini-1.5-pro-002");
    expect(stripDateSuffix("anthropic.claude-3-sonnet-v1:0")).toBe("anthropic.claude-3-sonnet-v1:0");
    expect(stripDateSuffix("gpt-4o-mini")).toBe("gpt-4o-mini");
    expect(stripDateSuffix("model-19991231")).toBe("model-19991231");
    expect(stripDateSuffix("")).toBe("");
  });
});

describe("IsNoopReroute", () => {
  it("identical models are a no-op", () => {
    expect(isNoopReroute("gpt-4o-mini", "gpt-4o-mini")).toBe(true);
  });

  it("trims + lowercases both sides", () => {
    expect(isNoopReroute("  GPT-4o-Mini ", "gpt-4o-mini")).toBe(true);
    expect(isNoopReroute(" Claude-Haiku-4-5-20251001 ", "CLAUDE-HAIKU-4-5")).toBe(true);
  });

  it("dated snapshot of the target alias is a no-op (all three date forms)", () => {
    expect(isNoopReroute("claude-haiku-4-5-20251001", "claude-haiku-4-5")).toBe(true);
    expect(isNoopReroute("gpt-4o-2024-08-06", "gpt-4o")).toBe(true);
    expect(isNoopReroute("claude-3-5-sonnet@20240620", "claude-3-5-sonnet")).toBe(true);
  });

  it("alias → dated target is NOT a no-op (an explicitly dated target is deliberate)", () => {
    expect(isNoopReroute("claude-haiku-4-5", "claude-haiku-4-5-20251001")).toBe(false);
  });

  it("dated → different dated snapshot is NOT a no-op", () => {
    expect(isNoopReroute("claude-haiku-4-5-20250101", "claude-haiku-4-5-20251001")).toBe(false);
  });

  it("non-date suffixes are never stripped, so they are NOT no-ops", () => {
    expect(isNoopReroute("claude-3-5-sonnet-latest", "claude-3-5-sonnet")).toBe(false);
    expect(isNoopReroute("gemini-1.5-pro-002", "gemini-1.5-pro")).toBe(false);
  });

  it("genuine reroutes are not no-ops", () => {
    expect(isNoopReroute("claude-sonnet-4-5", "claude-haiku-4-5")).toBe(false);
  });

  it("non-string / empty input → false (never claim a no-op on garbage)", () => {
    expect(isNoopReroute(undefined, "gpt-4o")).toBe(false);
    expect(isNoopReroute("gpt-4o", undefined)).toBe(false);
    expect(isNoopReroute(null, null)).toBe(false);
    expect(isNoopReroute(42, 42)).toBe(false);
    expect(isNoopReroute({}, "gpt-4o")).toBe(false);
    expect(isNoopReroute("", "")).toBe(false);
    expect(isNoopReroute("   ", "   ")).toBe(false);
  });
});

describe("_applyReroute no-op guard (State B)", () => {
  beforeEach(() => {
    try { state.drainObservations(); } catch { /* */ }
  });

  const directive = (model: string) => ({
    reroute: {
      mode: "enforce",
      model,
      provider: "anthropic",
      rule_id: "rule_rr",
      rule_name: "downgrade to haiku",
      original: { provider: "anthropic", model: "claude-sonnet-4-5" },
    },
  });

  it("dated-snapshot request vs alias target → noop, body untouched, no _tp_routing, no observation", () => {
    const body: Record<string, any> = { model: "claude-haiku-4-5-20251001" };
    let status: string | undefined;
    const s = session({ name: "t" }, (sess) => {
      status = _applyReroute(directive("claude-haiku-4-5"), body, "anthropic");
      return sess;
    });
    expect(status).toBe("noop");
    expect(body.model).toBe("claude-haiku-4-5-20251001");
    expect((s.metadata as Record<string, unknown> | undefined)?._tp_routing).toBeUndefined();
    expect(state.drainObservations()).toHaveLength(0);
  });

  it("exact-equal request/target → noop", () => {
    const body: Record<string, any> = { model: "claude-haiku-4-5" };
    const status = _applyReroute(directive("claude-haiku-4-5"), body, "anthropic");
    expect(status).toBe("noop");
    expect(body.model).toBe("claude-haiku-4-5");
    expect(state.drainObservations()).toHaveLength(0);
  });

  it("genuine sonnet→haiku reroute still applies and stamps _tp_routing (regression guard)", () => {
    const body: Record<string, any> = { model: "claude-sonnet-4-5" };
    let status: string | undefined;
    const s = session({ name: "t" }, (sess) => {
      status = _applyReroute(directive("claude-haiku-4-5"), body, "anthropic");
      return sess;
    });
    expect(status).toBe("applied");
    expect(body.model).toBe("claude-haiku-4-5");
    const routing = (s.metadata as Record<string, any> | undefined)?._tp_routing;
    expect(routing).toBeTruthy();
    expect(routing.original_model).toBe("claude-sonnet-4-5");
    expect(routing.actual_model).toBe("claude-haiku-4-5");
  });

  it("alias request → dated target still applies (direction-sensitive)", () => {
    const body: Record<string, any> = { model: "claude-haiku-4-5" };
    const status = _applyReroute(directive("claude-haiku-4-5-20251001"), body, "anthropic");
    expect(status).toBe("applied");
    expect(body.model).toBe("claude-haiku-4-5-20251001");
  });

  it("no-op guard never throws on a malformed body model", () => {
    const body: Record<string, any> = { model: { nested: true } };
    expect(() => _applyReroute(directive("claude-haiku-4-5"), body, "anthropic")).not.toThrow();
    expect(body.model).toBe("claude-haiku-4-5"); // non-string request model → not a no-op
  });
});

describe("Local evaluator no-op guard (State A)", () => {
  const pack = (targetModel: string) => ({
    version: 1,
    directives: [
      {
        id: "rule_rr",
        kind: "REROUTE",
        mode: "enforce",
        priority: 10,
        selector: { match: { field: "model", operator: "EXISTS" } },
        reroute: { to: { provider: "anthropic", model: targetModel } },
      },
    ],
  });

  it("dated snapshot of the target alias → no rerouted decision, no observation", () => {
    const res = evaluate(pack("claude-haiku-4-5"), {}, {
      provider: "anthropic",
      model: "claude-haiku-4-5-20251001",
    });
    expect(res.decision.status).toBe("allowed");
    expect(res.decision.reroute).toBeNull();
    expect(res.observations).toHaveLength(0);
  });

  it("exact-equal model → no rerouted decision, no observation", () => {
    const res = evaluate(pack("claude-haiku-4-5"), {}, {
      provider: "anthropic",
      model: "Claude-Haiku-4-5",
    });
    expect(res.decision.status).toBe("allowed");
    expect(res.observations).toHaveLength(0);
  });

  it("genuine sonnet→haiku directive still reroutes (regression guard)", () => {
    const res = evaluate(pack("claude-haiku-4-5"), {}, {
      provider: "anthropic",
      model: "claude-sonnet-4-5",
    });
    expect(res.decision.status).toBe("rerouted");
    expect(res.decision.reroute?.to.model).toBe("claude-haiku-4-5");
  });

  it("alias request → dated target still reroutes (direction-sensitive)", () => {
    const res = evaluate(pack("claude-haiku-4-5-20251001"), {}, {
      provider: "anthropic",
      model: "claude-haiku-4-5",
    });
    expect(res.decision.status).toBe("rerouted");
    expect(res.decision.reroute?.to.model).toBe("claude-haiku-4-5-20251001");
  });

  it("dry_run no-op directive also yields nothing (guard sits before the mode split)", () => {
    const p = pack("claude-haiku-4-5");
    p.directives[0].mode = "dry_run";
    const res = evaluate(p, {}, {
      provider: "anthropic",
      model: "claude-haiku-4-5-20251001",
    });
    expect(res.decision.status).toBe("allowed");
    expect(res.observations).toHaveLength(0);
  });
});
