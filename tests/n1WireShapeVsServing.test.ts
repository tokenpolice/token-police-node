/**
 * N1 — usage.shape follows the WIRE surface, provider follows the SERVING host.
 *
 * Made the emitted `provider` the serving slug (Anthropic SDK pointed at
 * api.minimax.io reports "minimax"). The Mode-A shape derivation, however, kept
 * keying off that same serving slug — "minimax" has no case, so the span went
 * out as `openai_compatible_chat` even though the bytes are Anthropic-shaped.
 * The server then ran the OpenAI mapper over Anthropic fields:
 * `cache_creation_input_tokens` fell into extra_units (cache writes billed $0)
 * and the already cache-EXCLUSIVE `input_tokens` had cache reads subtracted a
 * second time.
 *
 * These assert the two axes at the emitted-payload level:
 * shape = wire surface (gen_ai.system / instrumentation scope)
 * provider = serving slug (stash override) must still hold
 *
 * The gate is anthropic-ONLY: every non-anthropic wire keeps the exact shape it
 * produced before the fix.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { setClient } from "../src/state";
import { TPSession, _getSessionStorage } from "../src/context";
import { TokenPoliceSpanProcessor } from "../src/telemetry";
import { __test__ } from "../src/enforcer";

const { _stashProviderOverride } = __test__ as any;

// Positional indices into the client.log() argument list (see telemetry.ts).
const PROVIDER_ARG = 5;
const EXTRAS_ARG = 13;

let logged: any[];
beforeEach(() => {
  logged = [];
  setClient({ log: (...args: any[]) => logged.push(args) } as any);
});
afterEach(() => {
  setClient(null as any);
});

function newSession(): TPSession {
  return new TPSession({
    userId: "u1",
    paidPlan: "pro",
    workflowName: "wf",
    traceId: "a".repeat(32),
    rootSpanId: "b".repeat(16),
  });
}

interface SpanOpts {
  system?: string;
  scope?: string;
  name?: string;
  model?: string;
  extraAttrs?: Record<string, unknown>;
}

function fakeSpan(session: TPSession, opts: SpanOpts = {}) {
  const attrs: Record<string, unknown> = {
    "gen_ai.request.model": opts.model ?? "MiniMax-M2",
    "gen_ai.usage.input_tokens": 100,
    "gen_ai.usage.output_tokens": 50,
    "tp.trace_id": session.traceId,
    "tp.span_order": 0,
    ...(opts.extraAttrs ?? {}),
  };
  if (opts.system !== undefined) attrs["gen_ai.system"] = opts.system;
  return {
    attributes: attrs,
    name: opts.name ?? "anthropic.chat",
    instrumentationScope: opts.scope ? { name: opts.scope } : undefined,
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

async function emit(opts: SpanOpts & { override?: string }): Promise<any[]> {
  const session = newSession();
  if (opts.override) _stashProviderOverride(opts.override, session, 0);
  return runOnEnd(session, fakeSpan(session, opts));
}

describe("N1 — wire shape vs serving provider", () => {
  it("1. Anthropic SDK -> minimax host: shape anthropic_messages, provider minimax", async () => {
    const args = await emit({ system: "Anthropic", override: "minimax" });
    expect(args[EXTRAS_ARG].usage.shape).toBe("anthropic_messages");
    // The emitted provider is the SERVING slug, not the wire vendor.
    expect(args[PROVIDER_ARG]).toBe("minimax");
  });

  it("2. Anthropic SDK, no host remap: shape anthropic_messages (unchanged)", async () => {
    const args = await emit({ system: "Anthropic" });
    expect(args[EXTRAS_ARG].usage.shape).toBe("anthropic_messages");
    expect(args[PROVIDER_ARG]).toBe("anthropic");
  });

  it("3. OpenAI SDK -> minimax host stays openai_compatible_chat (unchanged)", async () => {
    const args = await emit({
      system: "openai",
      name: "openai.chat",
      override: "minimax",
    });
    expect(args[EXTRAS_ARG].usage.shape).toBe("openai_compatible_chat");
    expect(args[PROVIDER_ARG]).toBe("minimax");
  });

  it("4. OpenAI SDK -> openrouter stays openrouter_routed (unchanged)", async () => {
    const args = await emit({
      system: "openai",
      name: "openai.chat",
      override: "openrouter",
    });
    expect(args[EXTRAS_ARG].usage.shape).toBe("openrouter_routed");
    expect(args[PROVIDER_ARG]).toBe("openrouter");
  });

  it("5. scope-only signal: no gen_ai.system, anthropic instrumentation scope", async () => {
    const args = await emit({
      scope: "@traceloop/instrumentation-anthropic",
      override: "minimax",
    });
    expect(args[EXTRAS_ARG].usage.shape).toBe("anthropic_messages");
    expect(args[PROVIDER_ARG]).toBe("minimax");
  });

  it("5b. the Python scope literal is also accepted (cross-SDK parity)", async () => {
    const args = await emit({
      scope: "opentelemetry.instrumentation.anthropic",
      override: "minimax",
    });
    expect(args[EXTRAS_ARG].usage.shape).toBe("anthropic_messages");
  });

  it("6. genuine OpenAI + openai scope stays openai_chat (unchanged)", async () => {
    const args = await emit({
      system: "openai",
      name: "openai.chat",
      scope: "@traceloop/instrumentation-openai",
      model: "gpt-4.1-mini",
    });
    expect(args[EXTRAS_ARG].usage.shape).toBe("openai_chat");
    expect(args[PROVIDER_ARG]).toBe("openai");
  });

  it("7. LangChain-Gemini keeps google_genai — the gate never widens", async () => {
    const args = await emit({
      system: "Google",
      name: "langchain.chat",
      scope: "@traceloop/instrumentation-langchain",
      model: "models/gemini-2.5-flash",
    });
    expect(args[EXTRAS_ARG].usage.shape).toBe("google_genai");
    expect(args[PROVIDER_ARG]).toBe("google");
  });

  it("8. a hostile instrumentationScope getter falls back, never throws", async () => {
    const session = newSession();
    const span = fakeSpan(session, { name: "cerebras.chat" });
    Object.defineProperty(span, "instrumentationScope", {
      get() {
        throw new Error("nope");
      },
    });
    const args = await runOnEnd(session, span);
    // Fail-open: the pre-existing serving-slug switch still decides.
    expect(args[EXTRAS_ARG].usage.shape).toBe("openai_compatible_chat");
  });

  it("9. anthropic wire carries cache-write tokens on the native key", async () => {
    const args = await emit({
      system: "Anthropic",
      override: "minimax",
      extraAttrs: {
        "gen_ai.usage.cache_read.input_tokens": 200,
        "gen_ai.usage.cache_creation_input_tokens": 1000,
      },
    });
    const usage = args[EXTRAS_ARG].usage;
    expect(usage.shape).toBe("anthropic_messages");
    // The server's anthropic mapper reads these two keys; under the old
    // openai_compatible_chat shape the creation tokens were unpriced extras.
    expect(usage.raw.cache_creation_input_tokens).toBe(1000);
    expect(usage.raw.cache_read_input_tokens).toBe(200);
  });
});
