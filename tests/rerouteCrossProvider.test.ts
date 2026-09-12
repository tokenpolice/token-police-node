import { describe, it, expect, beforeEach, vi } from "vitest";
import { __test__ } from "../src/enforcer";
import { session } from "../src/context";
import { evaluate, effectiveProvider } from "../src/localEvaluator";
import * as state from "../src/state";
import { setClient, resetPack } from "../src/state";

// A cross-provider REROUTE directive is rejected by the local fast path
// (State A, `cross_provider_unsupported`) but was applied unconditionally by
// `_applyReroute` (State B, /check response). The guard makes the two paths
// agree: `_applyReroute` now skips the model swap when the reroute's target
// provider is present and differs from the call provider, using the SAME
// `effectiveProvider()` semantics the evaluator uses.

const { _applyReroute } = __test__;

const rerouteResult = (provider: string | undefined, model = "target-model") => ({
  reroute: {
    mode: "enforce",
    model,
    provider,
    rule_id: "rule_rr",
    rule_name: "reroute rule",
  },
});

describe("Cross-provider reroute guard (_applyReroute)", () => {
  beforeEach(() => {
    // Drain any leftover observations from prior tests.
    try { state.drainObservations(); } catch { /* */ }
  });

  it("same-provider reroute STILL applies (assertion 1)", () => {
    const body: Record<string, any> = { model: "gpt-4o" };
    const status = _applyReroute(
      { reroute: { mode: "enforce", model: "gpt-4o-mini", provider: "openai" } },
      body,
      "openai",
    );
    expect(status).toBe("applied");
    expect(body.model).toBe("gpt-4o-mini");
  });

  it("cross-provider reroute is NOT applied and stashes no _tp_routing (assertion 3)", () => {
    const body: Record<string, any> = { model: "gpt-4o" };
    let status: string | undefined;
    const s = session({ name: "t" }, (sess) => {
      status = _applyReroute(
        {
          reroute: {
            mode: "enforce",
            model: "claude-3-haiku",
            provider: "anthropic",
          },
        },
        body,
        "openai",
      );
      return sess;
    });
    expect(status).toBe("rejected");
    expect(body.model).toBe("gpt-4o"); // unchanged
    expect((s.metadata as Record<string, unknown> | undefined)?._tp_routing).toBeUndefined();
    const obs = state.drainObservations();
    expect(obs.some((o: any) => o.outcome === "reroute_rejected" && o.rejection_reason === "cross_provider_unsupported")).toBe(true);
  });

  // State B refuse must be auditable via observation; applied returns status.
  it("ServingUnverified returns rejected + serving_unverified observation", () => {
    const body: Record<string, any> = { model: "gpt-4o" };
    const status = _applyReroute(
      { reroute: { mode: "enforce", model: "gpt-4o-mini", provider: "openai" } },
      body,
      "openai",
      true, // servingUnverified
    );
    expect(status).toBe("rejected");
    expect(body.model).toBe("gpt-4o");
    const obs = state.drainObservations();
    expect(obs.some((o: any) => o.outcome === "reroute_rejected" && o.rejection_reason === "serving_unverified")).toBe(true);
  });

  // Superseded: a live ENFORCE directive on a body-less call shape now
  // reports the refusal instead of resolving silently — "rejected" with a
  // reroute_rejected/unappliable_call_shape observation, not a silent noop.
  // See tests/rerouteUnappliableShape.test.ts for the full coverage.
  it("Missing body/model returns rejected WITH a reroute_rejected/unappliable_call_shape observation", () => {
    const status = _applyReroute(
      { reroute: { mode: "enforce", model: "gpt-4o-mini", provider: "openai" } },
      null,
      "openai",
    );
    expect(status).toBe("rejected");
    const obs = state.drainObservations();
    expect(obs).toHaveLength(1);
    expect(obs[0].outcome).toBe("reroute_rejected");
    expect(obs[0].rejection_reason).toBe("unappliable_call_shape");
  });

  // When dial dry_run suppresses ENFORCE apply, observation.mode must be
  // the **rule** executionMode ('enforce'), not the dial. Dial is only
  // sdk_firewall_mode on /log (two-dial attribution: "SDK in dry-run").
  it("Dial-suppressed State B would_reroute uses rule mode enforce", async () => {
    const { _runAsyncCheck } = __test__;
    resetPack(); // ensure no healthy daemon pack → State B (inline /check)
    const body: Record<string, any> = { model: "gpt-4o" };
    const pushed: any[] = [];
    const spy = vi.spyOn(state, "pushObservation").mockImplementation((o: any) => {
      pushed.push(o);
    });
    let checkCalled = 0;
    const fakeClient = {
      firewall: "dry_run",
      deployment: "serverless",
      check: async () => {
        checkCalled += 1;
        return {
          status: "allowed",
          reroute: {
            mode: "enforce",
            model: "gpt-4o-mini",
            provider: "openai",
            rule_id: "rule_rr",
            original: { provider: "openai", model: "gpt-4o" },
          },
        };
      },
    };
    setClient(fakeClient as any);
    try {
      await _runAsyncCheck(body, "openai");
    } finally {
      setClient(null as any);
      resetPack();
      spy.mockRestore();
    }
    expect(checkCalled).toBe(1);
    expect(body.model).toBe("gpt-4o"); // not applied
    const would = pushed.find((o) => o.outcome === "would_reroute");
    expect(would).toBeTruthy();
    expect(would.mode).toBe("enforce");
    expect(would.mode).not.toBe("dry_run");
  });

  it("State A and State B AGREE on a cross-provider directive — neither mutates the model (assertion 5)", () => {
    // State A: local evaluator rejects the cross-provider reroute.
    const pack = {
      directives: [
        {
          id: "rule_rr",
          kind: "REROUTE",
          mode: "enforce",
          selector: { match: { field: "model", operator: "EXISTS" } },
          reroute: { to: { provider: "anthropic", model: "claude-3-haiku" } },
        },
      ],
    };
    const res = evaluate(pack, {}, { provider: "openai", model: "gpt-4o" });
    const rejected = res.observations.find(
      (o: any) => o.outcome === "reroute_rejected",
    );
    expect(rejected).toBeTruthy();
    expect(rejected.rejection_reason).toBe("cross_provider_unsupported");

    // State B: apply-path leaves the body unchanged for the same inputs.
    const body: Record<string, any> = { model: "gpt-4o" };
    _applyReroute(rerouteResult("anthropic", "claude-3-haiku"), body, "openai");
    expect(body.model).toBe("gpt-4o");
  });

  it("guard uses effectiveProvider() casing — mixed-case call provider still applies (assertion 6)", () => {
    const body: Record<string, any> = { model: "gpt-4o" };
    _applyReroute(
      { reroute: { mode: "enforce", model: "gpt-4o-mini", provider: "openai" } },
      body,
      "OpenAI", // effectiveProvider("OpenAI") === "openai"
    );
    expect(body.model).toBe("gpt-4o-mini");
  });

  it("absent target provider preserves the fallback and STILL applies (assertion 7)", () => {
    const body: Record<string, any> = { model: "gpt-4o" };
    _applyReroute(
      { reroute: { mode: "enforce", model: "gpt-4o-mini" } },
      body,
      "openai",
    );
    expect(body.model).toBe("gpt-4o-mini");
  });

  it("falsy call provider + present target → guard fires, model untouched, no _tp_routing (assertion 8)", () => {
    const body: Record<string, any> = { model: "gpt-4o" };
    const s = session({ name: "t" }, (sess) => {
      _applyReroute(rerouteResult("anthropic", "claude-3-haiku"), body, "");
      return sess;
    });
    expect(body.model).toBe("gpt-4o");
    expect((s.metadata as Record<string, unknown> | undefined)?._tp_routing).toBeUndefined();
  });

  it("effectiveProvider() is now a pure trim+lowercase — OpenRouter remap removed (assertion 9)", () => {
    // Pre-fix, "openai" was the only input the dead `p === "openai" && bu.includes("openrouter")`
    // branch keyed on; with the branch and its baseUrl arg deleted, a bare "openai" can only
    // ever return "openai".
    expect(effectiveProvider("openai")).toBe("openai");
    expect(effectiveProvider(" OpenAI ")).toBe("openai");
    expect(effectiveProvider("")).toBe("");
  });

  it("golden rule — never throws, even on malformed input (assertions 11/12)", () => {
    expect(() => _applyReroute(undefined, undefined, undefined)).not.toThrow();
    expect(() => _applyReroute({ reroute: null }, null, "openai")).not.toThrow();
    expect(() =>
      _applyReroute(rerouteResult("anthropic"), { model: "gpt-4o" }, "openai"),
    ).not.toThrow();
  });
});

// ── Provider-alias canonicalization parity with the Python SDK ──
// effectiveProvider() now canonicalizes alias slugs (together_ai → together,
// etc.), so an alias directive and its non-alias observed twin are treated as
// the SAME provider on both the State A (local evaluator) and State B
// (_applyReroute) paths. Mirrors token-police-python's
// tests/test_reroute_cross_provider.py.
describe("effectiveProvider() alias canonicalization", () => {
  it("trims, lowercases, then canonicalizes alias slugs", () => {
    expect(effectiveProvider("openai")).toBe("openai");
    expect(effectiveProvider(" OpenAI ")).toBe("openai");
    expect(effectiveProvider("")).toBe("");
    expect(effectiveProvider("together_ai")).toBe("together");
    expect(effectiveProvider("together")).toBe("together");
    expect(effectiveProvider(" Together_AI ")).toBe("together");
  });

  // Responses pseudo-provider is runtime-only (not a published provider slug).
  it("maps openai_responses runtime slug to openai", () => {
    expect(effectiveProvider("openai_responses")).toBe("openai");
    expect(effectiveProvider("OpenAI_Responses")).toBe("openai");
  });
});

describe("Openai_responses entity block uses openai group tag", () => {
  it("ENTITY_BLOCK armed as openai hits when observed provider is openai_responses", () => {
    const pack = {
      version: 1,
      directives: [
        {
          id: "r-prov",
          kind: "ENTITY_BLOCK",
          mode: "enforce",
          priority: 10,
          selector: {
            match: { field: "model", operator: "EXISTS" },
            group_by: ["provider"],
          },
          entities: ["openai"],
        },
      ],
    };
    const { decision } = evaluate(
      pack,
      { user_id: "u" },
      { model: "gpt-4o-mini", provider: "openai_responses" },
    );
    expect(decision.status).toBe("blocked");
    expect(decision.rule_id).toBe("r-prov");
  });
});

describe("alias-provider reroute is NOT rejected as cross-provider", () => {
  const aliasPack = (targetProvider: string) => ({
    version: 1,
    tenant_id: "t",
    project_id: "p",
    directives: [
      {
        id: "rr",
        kind: "REROUTE",
        mode: "enforce",
        priority: 10,
        selector: { match: { field: "model", operator: "EXISTS" } },
        reroute: { to: { provider: targetProvider, model: "target-model" } },
      },
    ],
  });

  it("State A: directive 'together_ai' matches observed 'together' → rerouted", () => {
    const res = evaluate(aliasPack("together_ai"), {}, { provider: "together", model: "llama-3" });
    expect(res.decision.status).toBe("rerouted");
    expect(res.observations.find((o: any) => o.outcome === "reroute_rejected")).toBeUndefined();
  });

  it("State A: the reverse — directive 'together' matches observed 'together_ai' → rerouted", () => {
    const res = evaluate(aliasPack("together"), {}, { provider: "together_ai", model: "llama-3" });
    expect(res.decision.status).toBe("rerouted");
    expect(res.observations.find((o: any) => o.outcome === "reroute_rejected")).toBeUndefined();
  });

  it("State B: _applyReroute applies the model swap when alias slugs canonicalize equal", () => {
    const body: Record<string, any> = { model: "llama-3" };
    session({ name: "t" }, () => {
      _applyReroute(rerouteResult("together_ai", "target-model"), body, "together");
    });
    expect(body.model).toBe("target-model");
  });
});

// ── serving-provider from base_url drives the cross-provider guard ──
// The guard must compare the *serving* host (api.minimax.io → minimax), not
// the client SDK module name (anthropic/openai). Otherwise a gateway-hosted
// Anthropic-compatible app is false-accepted and billed from the wrong catalog.

const {
  _effectiveProvider,
  _resolveServingProvider,
  _resolveServingFromBaseUrl,
  _matchHostToProvider,
  _extractHost,
} = __test__;

function clientWithBaseURL(baseURL: string | undefined) {
  if (baseURL === undefined) return { _client: {} };
  return { _client: { baseURL } };
}

const anthropicToHaikuPack = {
  directives: [
    {
      id: "rule_rr",
      kind: "REROUTE",
      mode: "enforce",
      selector: { match: { field: "model", operator: "EXISTS" } },
      reroute: { to: { provider: "anthropic", model: "claude-haiku-4-5" } },
    },
  ],
};

const openaiToMiniPack = {
  directives: [
    {
      id: "rule_rr",
      kind: "REROUTE",
      mode: "enforce",
      selector: { match: { field: "model", operator: "EXISTS" } },
      reroute: { to: { provider: "openai", model: "gpt-4o-mini" } },
    },
  ],
};

describe("Serving-provider host map", () => {
  it("extracts host from full URLs and strips port/userinfo via URL parse", () => {
    expect(_extractHost("https://api.minimax.io/anthropic/v1")).toBe("api.minimax.io");
    expect(_extractHost("https://user:pass@api.anthropic.com:443/v1")).toBe("api.anthropic.com");
    expect(_extractHost("")).toBe("");
  });

  it("maps known hosts (minimax, anthropic, openrouter, azure pattern)", () => {
    expect(_matchHostToProvider("api.minimax.io")).toBe("minimax");
    expect(_matchHostToProvider("api.minimaxi.com")).toBe("minimax");
    expect(_matchHostToProvider("api.anthropic.com")).toBe("anthropic");
    expect(_matchHostToProvider("openrouter.ai")).toBe("openrouter");
    expect(_matchHostToProvider("my-resource.openai.azure.com")).toBe("azure-openai");
    expect(_matchHostToProvider("llm.corp.example")).toBeNull();
  });

  it("resolveServingFromBaseUrl three-way: absent / recognized / unrecognized", () => {
    expect(_resolveServingFromBaseUrl("")).toEqual({ kind: "absent" });
    expect(_resolveServingFromBaseUrl("https://api.minimax.io/anthropic")).toEqual({
      kind: "recognized",
      provider: "minimax",
    });
    expect(_resolveServingFromBaseUrl("https://openrouter.ai/api/v1")).toEqual({
      kind: "recognized",
      provider: "openrouter",
    });
    expect(_resolveServingFromBaseUrl("https://llm.corp.example/v1")).toEqual({
      kind: "unrecognized",
    });
  });
});

describe("_effectiveProvider from baseURL", () => {
  it("(a) Anthropic module + api.minimax.io → minimax", () => {
    expect(
      _effectiveProvider("anthropic", clientWithBaseURL("https://api.minimax.io/anthropic/v1")),
    ).toBe("minimax");
  });

  it("(b) Anthropic module + default api.anthropic.com → anthropic", () => {
    expect(
      _effectiveProvider("anthropic", clientWithBaseURL("https://api.anthropic.com")),
    ).toBe("anthropic");
  });

  it("(b2) Anthropic module + absent baseURL → anthropic (module)", () => {
    expect(_effectiveProvider("anthropic", clientWithBaseURL(undefined))).toBe("anthropic");
    expect(_effectiveProvider("anthropic", null)).toBe("anthropic");
  });

  it("(c) OpenAI module + api.minimax.io → minimax (not openai)", () => {
    expect(
      _effectiveProvider("openai", clientWithBaseURL("https://api.minimax.io/v1")),
    ).toBe("minimax");
  });

  it("(d) unknown custom baseURL → keeps module provider + servingUnverified", () => {
    // Provider field stays "anthropic" for matchConditions/groupBy (check/log
    // mirror); REROUTE refuse is via servingUnverified, not an empty sentinel.
    expect(
      _effectiveProvider("anthropic", clientWithBaseURL("https://llm.corp.example/v1")),
    ).toBe("anthropic");
    expect(
      _resolveServingProvider("anthropic", clientWithBaseURL("https://llm.corp.example/v1")),
    ).toEqual({ provider: "anthropic", servingUnverified: true });
  });

  it("(e) OpenAI + openrouter.ai → openrouter (regression)", () => {
    expect(
      _effectiveProvider("openai", clientWithBaseURL("https://openrouter.ai/api/v1")),
    ).toBe("openrouter");
    expect(
      _resolveServingProvider("openai", clientWithBaseURL("https://openrouter.ai/api/v1")),
    ).toEqual({ provider: "openrouter", servingUnverified: false });
  });

  it("never throws on malformed thisArg / baseURL", () => {
    expect(() => _effectiveProvider("openai", { _client: { baseURL: null } })).not.toThrow();
    expect(() => _effectiveProvider("openai", { get baseURL() { throw new Error("boom"); } })).not.toThrow();
    expect(_effectiveProvider("openai", { get baseURL() { throw new Error("boom"); } })).toBe("openai");
  });
});

describe("Reroute guard uses serving provider (State A + State B)", () => {
  it("(a) Anthropic→minimax host + anthropic target → rejected; model unchanged", () => {
    const serving = _effectiveProvider(
      "anthropic",
      clientWithBaseURL("https://api.minimax.io/anthropic/v1"),
    );
    expect(serving).toBe("minimax");

    const res = evaluate(anthropicToHaikuPack, {}, {
      provider: serving,
      model: "MiniMax-M2.5",
    });
    const rejected = res.observations.find((o: any) => o.outcome === "reroute_rejected");
    expect(rejected).toBeTruthy();
    expect(rejected!.rejection_reason).toBe("cross_provider_unsupported");
    expect(res.decision.status).toBe("allowed");

    const body: Record<string, any> = { model: "MiniMax-M2.5" };
    session({ name: "t" }, () => {
      _applyReroute(rerouteResult("anthropic", "claude-haiku-4-5"), body, serving);
    });
    expect(body.model).toBe("MiniMax-M2.5");
  });

  it("(b) default Anthropic + anthropic target → applied", () => {
    const serving = _effectiveProvider(
      "anthropic",
      clientWithBaseURL("https://api.anthropic.com"),
    );
    expect(serving).toBe("anthropic");

    const res = evaluate(anthropicToHaikuPack, {}, {
      provider: serving,
      model: "claude-sonnet-4",
    });
    expect(res.decision.status).toBe("rerouted");
    expect(res.decision.reroute?.to.model).toBe("claude-haiku-4-5");

    const body: Record<string, any> = { model: "claude-sonnet-4" };
    session({ name: "t" }, () => {
      _applyReroute(rerouteResult("anthropic", "claude-haiku-4-5"), body, serving);
    });
    expect(body.model).toBe("claude-haiku-4-5");
  });

  it("(c) OpenAI→minimax host + openai target → rejected (wrong-reason hole closed)", () => {
    const serving = _effectiveProvider(
      "openai",
      clientWithBaseURL("https://api.minimax.io/v1"),
    );
    expect(serving).toBe("minimax");

    const res = evaluate(openaiToMiniPack, {}, {
      provider: serving,
      model: "MiniMax-M2.5",
    });
    expect(res.observations.some((o: any) => o.outcome === "reroute_rejected")).toBe(true);
    expect(res.decision.status).toBe("allowed");

    const body: Record<string, any> = { model: "MiniMax-M2.5" };
    session({ name: "t" }, () => {
      _applyReroute(rerouteResult("openai", "gpt-4o-mini"), body, serving);
    });
    expect(body.model).toBe("MiniMax-M2.5");
  });

  it("(d) unknown custom baseURL + same-module anthropic target → rejected via serving_unverified", () => {
    const resolved = _resolveServingProvider(
      "anthropic",
      clientWithBaseURL("https://llm.corp.example/v1"),
    );
    expect(resolved).toEqual({ provider: "anthropic", servingUnverified: true });

    // Provider stays anthropic (match/groupBy mirror) but flag refuses REROUTE.
    const res = evaluate(anthropicToHaikuPack, {}, {
      provider: resolved.provider,
      model: "claude-sonnet-4",
      serving_unverified: true,
    });
    const rejected = res.observations.find((o: any) => o.outcome === "reroute_rejected");
    expect(rejected).toBeTruthy();
    expect(rejected!.rejection_reason).toBe("serving_unverified");
    expect(res.decision.status).toBe("allowed");

    // Without the flag, same-module target would apply — proves the flag is what refuses.
    const wouldApply = evaluate(anthropicToHaikuPack, {}, {
      provider: "anthropic",
      model: "claude-sonnet-4",
    });
    expect(wouldApply.decision.status).toBe("rerouted");

    const body: Record<string, any> = { model: "claude-sonnet-4" };
    session({ name: "t" }, () => {
      _applyReroute(
        rerouteResult("anthropic", "claude-haiku-4-5"),
        body,
        resolved.provider,
        true, // servingUnverified
      );
    });
    expect(body.model).toBe("claude-sonnet-4");
  });

  it("(d2) serving_unverified + genuinely cross-provider target → reason is serving_unverified (precedence)", () => {
    // Observed provider ("anthropic") and the directive's target ("openai")
    // are genuinely different modules, so cross_provider_unsupported would
    // also fire on its own — but serving_unverified must win, mirroring
    // _applyReroute's precedence (B4/B5 parity).
    const res = evaluate(openaiToMiniPack, {}, {
      provider: "anthropic",
      model: "claude-sonnet-4",
      serving_unverified: true,
    });
    const rejected = res.observations.find((o: any) => o.outcome === "reroute_rejected");
    expect(rejected).toBeTruthy();
    expect(rejected!.rejection_reason).toBe("serving_unverified");
    expect(res.decision.status).toBe("allowed");
  });

  it("(e) OpenAI→openrouter + openrouter target still applies", () => {
    const serving = _effectiveProvider(
      "openai",
      clientWithBaseURL("https://openrouter.ai/api/v1"),
    );
    expect(serving).toBe("openrouter");

    const pack = {
      directives: [
        {
          id: "rule_rr",
          kind: "REROUTE",
          mode: "enforce",
          selector: { match: { field: "model", operator: "EXISTS" } },
          reroute: { to: { provider: "openrouter", model: "openai/gpt-4o-mini" } },
        },
      ],
    };
    const res = evaluate(pack, {}, { provider: serving, model: "openai/gpt-4o" });
    expect(res.decision.status).toBe("rerouted");

    const body: Record<string, any> = { model: "openai/gpt-4o" };
    session({ name: "t" }, () => {
      _applyReroute(rerouteResult("openrouter", "openai/gpt-4o-mini"), body, serving);
    });
    expect(body.model).toBe("openai/gpt-4o-mini");
  });
});
