/**
 * Node auto-instrumented gateway attribution parity (Node-only).
 *
 * 1:1 mirror of the Python reference `tests/test_gateway_attribution.py`
 * (`TestInstrumentedGatewayAttribution` + `test_stash_helper_is_fail_open`).
 *
 * An OpenAI-SDK call routed through an OpenRouter base URL is auto-instrumented,
 * and Traceloop-JS STRIPS the vendor prefix from `gen_ai.request.model`
 * ("openai/gpt-4.1-nano" -> "gpt-4.1-nano") before telemetry `onEnd` sees it.
 * The enforcer now stashes the verbatim slug + its vendor head on gateway
 * detection; `onEnd` must restore the full slug and forward
 * `model_extras.original_provider`. Non-gateway spans stay byte-identical.
 *
 * The `api_base` and `original_provider` keys are INDEPENDENT — an absent
 * `api_base` must NOT drop `original_provider` (assertion 16).
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { setClient } from "../src/state";
import { TPSession, _getSessionStorage } from "../src/context";
import { TokenPoliceSpanProcessor } from "../src/telemetry";
import { __test__ } from "../src/enforcer";

const { _stashGatewayRequestModel, _stashProviderOverride, _stashApiBase } =
  __test__ as any;

// Positional indices into the client.log() argument list (see telemetry.ts:1196).
const MODEL_ARG = 4;
const PROVIDER_ARG = 5;

let logged: any[];
beforeEach(() => {
  logged = [];
  setClient({ log: (...args: any[]) => logged.push(args) } as any);
});
afterEach(() => {
  setClient(null as any);
});

function fakeSpan(session: TPSession, model = "gpt-4.1-nano") {
  return {
    attributes: {
      "gen_ai.system": "openai",
      "gen_ai.request.model": model,
      "gen_ai.usage.input_tokens": 10,
      "gen_ai.usage.output_tokens": 5,
      "tp.trace_id": session.traceId,
      "tp.span_order": 0,
    },
    name: "openai.chat",
    startTime: [0, 0] as [number, number],
    endTime: [1, 0] as [number, number],
    spanContext: () => ({
      traceId: "0000000000000000000000000000abcd",
      spanId: "0000000000001234",
    }),
    parentSpanId: undefined,
    status: { code: 0 },
  } as any;
}

/** Drive onEnd inside `session`'s ambient store, return the captured log args. */
async function runOnEnd(session: TPSession, span: any): Promise<any[]> {
  await _getSessionStorage().run(session, async () => {
    new TokenPoliceSpanProcessor().onEnd(span);
    // The log is deferred to process.nextTick — let it drain.
    await new Promise((r) => setTimeout(r, 0));
  });
  return logged[0];
}

function newSession(): TPSession {
  return new TPSession({
    userId: "u1",
    paidPlan: "pro",
    workflowName: "wf",
    traceId: "a".repeat(32),
    rootSpanId: "b".repeat(16),
  });
}

// ── The exported enforcer stash helper (unit) ──────────────────────────

describe("_stashGatewayRequestModel (helper unit)", () => {
  function stashModel(model: unknown): Record<string, any> {
    const session: any = { traceId: "t", spanCounter: 0, _pendingCompositions: {} };
    _stashGatewayRequestModel(session, { model });
    return session._pendingCompositions["t:0"] ?? {};
  }

  it("stashes full slug + lower-cased vendor head (assertion 1, 2, 3)", () => {
    expect(stashModel("openai/gpt-4.1-nano")).toEqual({
      model: "openai/gpt-4.1-nano",
      original_provider: "openai",
    });
  });

  it("byte-for-byte transform table matches Python (assertion 2)", () => {
    // vendor head lower-cased
    expect(stashModel("Anthropic/claude-3.5").original_provider).toBe("anthropic");
    // leading/trailing space on the head is trimmed
    expect(stashModel("  google/gemini ").original_provider).toBe("google");
    // only the FIRST slash delimits the head
    expect(stashModel("openai/org/model").original_provider).toBe("openai");
    // unprefixed slug -> empty head
    expect(stashModel("gpt-4.1-nano").original_provider).toBe("");
    // ...but the full slug is still stashed
    expect(stashModel("gpt-4.1-nano").model).toBe("gpt-4.1-nano");
  });

  it("keys on `${traceId}:${spanCounter}` — same as sibling stashes (assertion 8)", () => {
    const session: any = { traceId: "abc", spanCounter: 3, _pendingCompositions: {} };
    _stashGatewayRequestModel(session, { model: "openai/gpt-4.1-nano" });
    _stashProviderOverride("openrouter", session);
    _stashApiBase("https://openrouter.ai/api/v1", session);
    expect(session._pendingCompositions["abc:3"]).toEqual({
      model: "openai/gpt-4.1-nano",
      original_provider: "openai",
      provider: "openrouter",
      api_base: "https://openrouter.ai/api/v1",
    });
  });

  it("is fail-open on hostile input — never throws, never stashes (assertion 12, invariant 1)", () => {
    const session: any = { traceId: "t", spanCounter: 0, _pendingCompositions: {} };
    expect(() => _stashGatewayRequestModel(session, null)).not.toThrow();
    expect(() => _stashGatewayRequestModel(session, {})).not.toThrow();
    expect(() => _stashGatewayRequestModel(session, { model: 42 })).not.toThrow();
    expect(() => _stashGatewayRequestModel(null, { model: "x/y" })).not.toThrow();
    expect(session._pendingCompositions).toEqual({});
  });
});

// ── The full onEnd drive (integration with the stash) ──────────────────

describe("Telemetry onEnd gateway recovery", () => {
  it("restores full slug + forwards original_provider + api_base (assertion 3, 4)", async () => {
    const session = newSession();
    _stashProviderOverride("openrouter", session);
    _stashGatewayRequestModel(session, { model: "openai/gpt-4.1-nano" });
    _stashApiBase("https://openrouter.ai/api/v1", session);

    const args = await runOnEnd(session, fakeSpan(session));
    const extras = args[args.length - 1].model_extras;

    // Full vendor-prefixed slug restored over the stripped attr value.
    expect(args[MODEL_ARG]).toBe("openai/gpt-4.1-nano");
    expect(args[PROVIDER_ARG]).toBe("openrouter");
    expect(extras).toBeDefined();
    expect(extras.original_provider).toBe("openai");
    expect(extras.api_base).toContain("openrouter.ai");
    // Tokens unaffected.
    expect(args[6]).toBe(10);
    expect(args[7]).toBe(5);
  });

  it("unprefixed slug omits original_provider, keeps api_base (assertion 5, 16b)", async () => {
    const session = newSession();
    _stashProviderOverride("openrouter", session);
    _stashGatewayRequestModel(session, { model: "gpt-4.1-nano" });
    _stashApiBase("https://openrouter.ai/api/v1", session);

    const args = await runOnEnd(session, fakeSpan(session));
    const extras = args[args.length - 1].model_extras;

    expect(extras).toBeDefined();
    expect("original_provider" in extras).toBe(false);
    expect(extras.api_base).toContain("openrouter.ai");
    expect(Object.keys(extras).length).toBe(1);
  });

  it("original_provider SURVIVES an absent/empty api_base (assertion 16a — the naive-nesting bug)", async () => {
    const session = newSession();
    _stashProviderOverride("openrouter", session);
    _stashGatewayRequestModel(session, { model: "openai/gpt-4.1-nano" });
    // No _stashApiBase call — api_base is empty/absent.

    const args = await runOnEnd(session, fakeSpan(session));
    const extras = args[args.length - 1].model_extras;

    expect(args[MODEL_ARG]).toBe("openai/gpt-4.1-nano");
    expect(extras).toBeDefined();
    expect(extras.original_provider).toBe("openai");
    expect("api_base" in extras).toBe(false);
    expect(Object.keys(extras).length).toBe(1);
  });

  it("both keys present, no clobber (assertion 4, 16c)", async () => {
    const session = newSession();
    _stashProviderOverride("openrouter", session);
    _stashGatewayRequestModel(session, { model: "anthropic/claude-3.5-haiku" });
    _stashApiBase("https://openrouter.ai/api/v1", session);

    const args = await runOnEnd(session, fakeSpan(session));
    const extras = args[args.length - 1].model_extras;

    expect(extras.original_provider).toBe("anthropic");
    expect(extras.api_base).toBe("https://openrouter.ai/api/v1");
    expect(Object.keys(extras).sort()).toEqual(["api_base", "original_provider"]);
  });

  it("non-gateway span is byte-identical — no model_extras, model unchanged (assertion 6, 7)", async () => {
    // No gateway stash at all (the enforcer never entered the remap gate).
    const session = newSession();
    const args = await runOnEnd(session, fakeSpan(session));

    expect(args[MODEL_ARG]).toBe("gpt-4.1-nano");
    expect(args[PROVIDER_ARG]).toBe("openai");
    expect(args[args.length - 1].model_extras).toBeUndefined();
  });

  it("never emits an empty model_extras object (assertion 16 gate)", async () => {
    const session = newSession();
    const args = await runOnEnd(session, fakeSpan(session));
    // No api_base, no original_provider -> the whole key must be absent, not {}.
    expect("model_extras" in args[args.length - 1]).toBe(false);
  });
});
