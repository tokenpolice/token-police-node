/**
 * A live ENFORCE REROUTE directive reaching `_applyReroute` on a call shape
 * with no appliable request body (framework/hint-only paths: LangChain/
 * LlamaIndex, framework embeddings, Bedrock Converse envelope, native
 * OpenRouter envelope) used to silently return "noop" — no observation, so
 * the audit trail had REROUTE_DIRECTIVE_ISSUED with no resolution.
 *
 * `_applyReroute` now runs an appliability check FIRST (before the
 * servingUnverified / cross-provider guards): a body that is not an
 * object-with-top-level-"model" pushes a `reroute_rejected` observation with
 * `rejection_reason: "unappliable_call_shape"` and returns "rejected". The
 * new trailing `modelHint` param supplies `reroute.from.model` on that
 * observation (never a swap target — the body is never mutated).
 *
 * Mirrors token-police-python/tests/test_reroute_unappliable_shape.py.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { __test__, uninstrument } from "../src/enforcer";
import { session } from "../src/context";
import * as state from "../src/state";
import { setClient, resetPack } from "../src/state";
import { TokenPolice } from "../src/client";

const { _applyReroute, _runAsyncCheck, _instrumentLangChainChatModels } = __test__;

const directive = (
  model: string,
  provider = "openai",
  ruleName?: string | null,
) => ({
  reroute: {
    mode: "enforce",
    model,
    provider,
    rule_id: "rule_rr",
    ...(ruleName !== undefined ? { rule_name: ruleName } : {}),
  },
});

beforeEach(() => {
  // Drain any leftover observations from prior tests.
  try {
    state.drainObservations();
  } catch {
    /* */
  }
});

describe("_applyReroute — unappliable call shape (State B)", () => {
  it("null body + modelHint → rejected, exactly one observation, from.model = hint, to = directive target", () => {
    const status = _applyReroute(
      directive("gpt-4o-mini", "openai"),
      null,
      "openai",
      false,
      "gpt-4o",
    );
    expect(status).toBe("rejected");
    const obs = state.drainObservations();
    expect(obs).toHaveLength(1);
    const o = obs[0];
    expect(o.outcome).toBe("reroute_rejected");
    expect(o.rejection_reason).toBe("unappliable_call_shape");
    expect(o.reroute.from.model).toBe("gpt-4o");
    expect(o.reroute.to.model).toBe("gpt-4o-mini");
    expect(o.reroute.to.provider).toBe("openai");
  });

  it("null body, NO modelHint → rejected, from.model = '' (never throws)", () => {
    let status: string | undefined;
    expect(() => {
      status = _applyReroute(directive("gpt-4o-mini", "openai"), null, "openai");
    }).not.toThrow();
    expect(status).toBe("rejected");
    const obs = state.drainObservations();
    expect(obs).toHaveLength(1);
    expect(obs[0].reroute.from.model).toBe("");
  });

  it("Bedrock-Converse-like body without a top-level `model` key + hint → rejected, from.model = hint, body untouched", () => {
    const body: Record<string, any> = {
      modelId: "anthropic.claude-3-haiku",
      input: { messages: [] },
    };
    const snapshot = JSON.parse(JSON.stringify(body));
    const status = _applyReroute(
      directive("gpt-4o-mini", "openai"),
      body,
      "openai",
      false,
      "anthropic.claude-3-haiku",
    );
    expect(status).toBe("rejected");
    expect(body).toEqual(snapshot); // no mutation, no invented `model` key
    const obs = state.drainObservations();
    expect(obs).toHaveLength(1);
    expect(obs[0].reroute.from.model).toBe("anthropic.claude-3-haiku");
    expect(obs[0].rejection_reason).toBe("unappliable_call_shape");
  });

  it("appliability runs BEFORE the cross-provider guard — null body + cross-provider directive → unappliable_call_shape, not cross_provider_unsupported", () => {
    const status = _applyReroute(
      directive("claude-3-haiku", "anthropic"),
      null,
      "openai",
      false,
      "gpt-4o",
    );
    expect(status).toBe("rejected");
    const obs = state.drainObservations();
    expect(obs).toHaveLength(1);
    expect(obs[0].rejection_reason).toBe("unappliable_call_shape");
  });

  it("per-rule DRY_RUN directive + null body → noop, zero observations (mode guard runs before appliability)", () => {
    const status = _applyReroute(
      { reroute: { mode: "dry_run", model: "gpt-4o-mini", provider: "openai", rule_id: "rule_rr" } },
      null,
      "openai",
      false,
      "gpt-4o",
    );
    expect(status).toBe("noop");
    expect(state.drainObservations()).toHaveLength(0);
  });
});

describe("_applyReroute — unappliable-shape rejection carries rule_name (omit-when-empty convention)", () => {
  it("carries rule_name when present", () => {
    const status = _applyReroute(
      directive("gpt-4o-mini", "openai", "Send free users to a cheaper model"),
      null,
      "openai",
      false,
      "gpt-4o",
    );
    expect(status).toBe("rejected");
    const obs = state.drainObservations();
    expect(obs[0].rule_name).toBe("Send free users to a cheaper model");
    expect(obs[0].rule_id).toBe("rule_rr");
  });

  it("omits rule_name key when absent", () => {
    _applyReroute(directive("gpt-4o-mini", "openai"), null, "openai", false, "gpt-4o");
    const obs = state.drainObservations();
    expect(obs[0]).not.toHaveProperty("rule_name");
    expect(obs[0].rule_id).toBe("rule_rr");
  });

  it("omits rule_name key when empty string", () => {
    _applyReroute(directive("gpt-4o-mini", "openai", ""), null, "openai", false, "gpt-4o");
    const obs = state.drainObservations();
    expect(obs[0]).not.toHaveProperty("rule_name");
  });

  it("omits rule_name key when null (never emit a null key)", () => {
    _applyReroute(directive("gpt-4o-mini", "openai", null), null, "openai", false, "gpt-4o");
    const obs = state.drainObservations();
    expect(obs[0]).not.toHaveProperty("rule_name");
  });
});

describe("_applyReroute — golden rule: unappliable-shape path never throws", () => {
  it("(a) body's `has` trap throws → no throw escapes, degrades to noop", () => {
    const hostileBody = new Proxy(
      {},
      {
        has() {
          throw new Error("hostile has-trap");
        },
      },
    );
    let status: string | undefined;
    expect(() => {
      status = _applyReroute(
        directive("gpt-4o-mini", "openai"),
        hostileBody as any,
        "openai",
        false,
        "gpt-4o",
      );
    }).not.toThrow();
    // The whole function's try/catch swallows the thrown `in` check.
    expect(status).toBe("noop");
    expect(state.drainObservations()).toHaveLength(0);
  });

  it("(b) a `model` getter that throws → no throw escapes; a failed observation push never propagates", () => {
    const hostileBody = {
      get model() {
        throw new Error("hostile model getter");
      },
    };
    let status: string | undefined;
    expect(() => {
      // servingUnverified=true so the FIRST branch that reads body.model
      // (inside pushRejected, building the observation) is the one that
      // throws — proving a push failure fails open rather than propagating.
      status = _applyReroute(
        directive("gpt-4o-mini", "openai"),
        hostileBody as any,
        "openai",
        true,
        "gpt-4o",
      );
    }).not.toThrow();
    expect(status).toBe("rejected");
    // The observation build threw while reading body.model, so the push
    // itself never completed — no half-built observation is emitted.
    expect(state.drainObservations()).toHaveLength(0);
  });
});

describe("State B integration — _runAsyncCheck on an unappliable-shape path", () => {
  afterEach(() => {
    setClient(null as any);
    resetPack();
  });

  it("resolves without throwing and pushes exactly one reroute_rejected observation", async () => {
    resetPack(); // no healthy daemon pack → State B (inline /check)
    const fakeClient = {
      firewall: "enforce",
      deployment: "serverless",
      check: async () => ({
        status: "allowed",
        reroute: {
          mode: "enforce",
          model: "gpt-4o-mini",
          provider: "openai",
          rule_id: "rule_rr",
        },
      }),
    };
    setClient(fakeClient as any);
    await expect(
      _runAsyncCheck(null, "openai", null, true, false, "gpt-4o"),
    ).resolves.toBeUndefined();
    const obs = state.drainObservations();
    expect(obs).toHaveLength(1);
    expect(obs[0].outcome).toBe("reroute_rejected");
    expect(obs[0].rejection_reason).toBe("unappliable_call_shape");
    expect(obs[0].reroute.from.model).toBe("gpt-4o");
  });

  it("canReroute=false — zero observations even with a live reroute directive (B7 regression fence)", async () => {
    resetPack();
    const fakeClient = {
      firewall: "enforce",
      deployment: "serverless",
      check: async () => ({
        status: "allowed",
        reroute: {
          mode: "enforce",
          model: "gpt-4o-mini",
          provider: "openai",
          rule_id: "rule_rr",
        },
      }),
    };
    setClient(fakeClient as any);
    await _runAsyncCheck(null, "openai", null, false, false, "gpt-4o");
    expect(state.drainObservations()).toHaveLength(0);
  });
});

// ── Full LangChain-wrapper end-to-end: a real BaseChatModel.generate patch
// via _instrumentLangChainChatModels, driven through the wrapper exactly as
// a customer app would call it (never _runAsyncCheck directly).
describe("State B integration — LangChain chat wrapper end-to-end", () => {
  function makeFakeBaseChatModel(model: string, namespace: string[]) {
    const generateCalls: any[] = [];
    class BaseChatModel {
      model = model;
      lc_namespace = namespace;
      async generate(...args: any[]): Promise<any> {
        generateCalls.push(args);
        return { generations: [[{ message: { content: "hi" } }]] };
      }
    }
    return { BaseChatModel, generateCalls };
  }

  function makeClient(firewall: "enforce" | "dry_run" | "off" = "enforce"): TokenPolice {
    const client = new TokenPolice({
      apiKey: "tp_sk_test123",
      baseUrl: "http://localhost:59999",
      timeout: 0.1,
      firewall,
      deployment: "serverless",
    } as never);
    setClient(client);
    return client;
  }

  function stubClient(client: TokenPolice, checkResult: any) {
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue(checkResult as never);
    vi.spyOn(client, "log").mockImplementation(() => {});
    return checkSpy;
  }

  afterEach(() => {
    __test__._setInstrumented(true);
    uninstrument();
    setClient(null as any);
    resetPack();
    vi.restoreAllMocks();
  });

  it("REROUTE directive on the hint-only LangChain path emits ONE reroute_rejected observation; generate runs unmodified; no _tp_routing", async () => {
    const { BaseChatModel, generateCalls } = makeFakeBaseChatModel(
      "gpt-4o-mini",
      ["langchain", "chat_models", "openai"],
    );
    const client = makeClient("enforce");
    stubClient(client, {
      status: "allowed",
      reroute: { mode: "enforce", model: "gpt-4o", provider: "openai", rule_id: "rule_rr" },
    });
    _instrumentLangChainChatModels({ BaseChatModel });
    const inst = new BaseChatModel();
    const msgs = [[{ content: "hi" }]];
    const s = await session({ name: "wf" }, async (sess) => {
      await inst.generate(msgs);
      return sess;
    });
    // The underlying call ran with the EXACT same args — no clone, no swap.
    expect(generateCalls).toHaveLength(1);
    expect(generateCalls[0][0]).toBe(msgs);
    expect(inst.model).toBe("gpt-4o-mini"); // instance never mutated
    expect((s.metadata as Record<string, unknown> | undefined)?._tp_routing).toBeUndefined();
    const obs = state.drainObservations();
    const rejected = obs.filter((o: any) => o.outcome === "reroute_rejected");
    expect(rejected).toHaveLength(1);
    expect(rejected[0].rejection_reason).toBe("unappliable_call_shape");
    expect(rejected[0].reroute.from.model).toBe("gpt-4o-mini"); // hint from `this.model`
    expect(rejected[0].reroute.to.model).toBe("gpt-4o");
  });
});
