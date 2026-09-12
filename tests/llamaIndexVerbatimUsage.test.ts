/**
 * LlamaIndex manual-log path must forward the provider's verbatim usage object.
 *
 * Audit F1 (under-bill): the LlamaIndex extractor produced positional counts
 * with cached tokens ALREADY subtracted out of input. Passed as positional
 * counts, client.log synthesised an `openai_compatible_chat` shape whose
 * prompt_tokens the server's mapOpenAIChat treats as cache-INCLUSIVE — so it
 * subtracted the cached amount a SECOND time, under-counting uncached input by
 * the full cached amount (worst for Gemini prompt caching).
 *
 * Fix: forward the provider's raw usage tagged with its server-side shape
 * (`google_genai` / `openai_compatible_chat` / `anthropic_messages`) so the
 * mapper applies that provider's own cache semantics exactly once. The
 * positional counts stay as the fallback used only when no verbatim block is
 * available.
 *
 * All offline against fake ChatResponse objects — no network, no LlamaIndex.
 */
import { describe, it, test, expect, beforeEach, afterEach, vi } from "vitest";
import * as contextModule from "../src/context";
import { TPSession } from "../src/context";
import { resetPack, setClient } from "../src/state";
import { TokenPolice } from "../src/client";
import { __test__ as enforcerTest } from "../src/enforcer";

const _extractLlamaIndexUsage = enforcerTest._extractLlamaIndexUsage;
const _logLlamaIndex = enforcerTest._logLlamaIndex;
const _buildLlamaIndexVerbatimUsage = enforcerTest._buildLlamaIndexVerbatimUsage;

// Provider-class stand-ins: _llamaIndexProvider() keys off constructor.name.
class Gemini {
  model = "gemini-2.0-flash";
}
class OpenAI {
  model = "gpt-4o";
}
class Anthropic {
  model = "claude-3-5-sonnet";
}
// @llamaindex/openai's OpenAIResponses is a SIBLING of OpenAI (its own
// chat/streamChat against /v1/responses) with the Responses usage shape.
class OpenAIResponses {
  model = "gpt-5";
}

function makeClient() {
  const client = new TokenPolice({
    apiKey: "tp_sk_test123",
    baseUrl: "http://localhost:59999",
    timeout: 0.1,
    firewall: "off",
    deployment: "daemon",
  });
  setClient(client);
  return client;
}

// The extras block is the 14th positional arg of tp.log (index 13).
const EXTRAS_ARG = 13;
const MODEL_ARG = 4;
const PROVIDER_ARG = 5;
const INPUT_TOKENS_ARG = 6;
const OUTPUT_TOKENS_ARG = 7;
const CACHED_TOKENS_ARG = 8;

describe("_extractLlamaIndexUsage — verbatim usage forwarding", () => {
  it("google: usageMetadata with cachedContentTokenCount>0 → verbatim google_genai, positional counts unchanged", () => {
    const um = {
      promptTokenCount: 1000,
      cachedContentTokenCount: 800,
      candidatesTokenCount: 200,
      thoughtsTokenCount: 60,
    };
    const u = _extractLlamaIndexUsage("google", { raw: { usageMetadata: um } }, new Gemini());
    // Positional fallback — EXACTLY today's computation (netted, cache-subtracted).
    expect(u.inputTokens).toBe(200); // max(0, 1000 - 800)
    expect(u.outputTokens).toBe(260); // 200 candidates + 60 thoughts
    expect(u.cachedTokens).toBe(800);
    // Verbatim block carries the cache-inclusive usageMetadata content as a
    // JSON snapshot (deep-equal, deliberately NOT the same reference — the
    // snapshot freezes it against later mutation and pre-validates stringify).
    expect(u.verbatimUsage).toEqual({ shape: "google_genai", raw: um });
    expect(u.verbatimUsage!.raw).not.toBe(um);
  });

  it("openai: prompt_tokens_details.cached_tokens>0 → verbatim openai_compatible_chat keeps the ORIGINAL cache-inclusive prompt_tokens", () => {
    const usage = {
      prompt_tokens: 500,
      completion_tokens: 120,
      prompt_tokens_details: { cached_tokens: 300 },
    };
    const u = _extractLlamaIndexUsage("openai", { raw: { usage } }, new OpenAI());
    expect(u.inputTokens).toBe(200); // positional netted: 500 - 300
    expect(u.cachedTokens).toBe(300);
    expect(u.verbatimUsage).toEqual({ shape: "openai_compatible_chat", raw: usage });
    // The forwarded prompt_tokens is the cache-INCLUSIVE 500, not the netted 200.
    expect((u.verbatimUsage!.raw as any).prompt_tokens).toBe(500);
  });

  it("anthropic: usage with positive counts → verbatim anthropic_messages (cache-exclusive input)", () => {
    const usage = {
      input_tokens: 100,
      output_tokens: 50,
      cache_read_input_tokens: 200,
    };
    const u = _extractLlamaIndexUsage("anthropic", { raw: { usage, model: "claude-x" } }, new Anthropic());
    expect(u.inputTokens).toBe(100);
    expect(u.cachedTokens).toBe(200);
    expect(u.model).toBe("claude-x");
    expect(u.verbatimUsage).toEqual({ shape: "anthropic_messages", raw: usage });
  });

  it("fallback: usageMetadata absent → verbatimUsage undefined, positional zeros (today's values)", () => {
    const u = _extractLlamaIndexUsage("google", { raw: {} }, new Gemini());
    expect(u.inputTokens).toBe(0);
    expect(u.outputTokens).toBe(0);
    expect(u.cachedTokens).toBe(0);
    expect(u.verbatimUsage).toBeUndefined();
  });

  it("fallback: empty usageMetadata (all zero) → no verbatim block", () => {
    const u = _extractLlamaIndexUsage(
      "google",
      { raw: { usageMetadata: { promptTokenCount: 0, candidatesTokenCount: 0 } } },
      new Gemini(),
    );
    expect(u.verbatimUsage).toBeUndefined();
  });

  it("fallback: a raw.usageMetadata getter that THROWS → verbatimUsage undefined, zeros, no throw (golden rule)", () => {
    const hostileRaw: any = {};
    Object.defineProperty(hostileRaw, "usageMetadata", {
      get() {
        throw new Error("boom: hostile getter");
      },
    });
    let u: any;
    expect(() => {
      u = _extractLlamaIndexUsage("google", { raw: hostileRaw }, new Gemini());
    }).not.toThrow();
    expect(u.inputTokens).toBe(0);
    expect(u.outputTokens).toBe(0);
    expect(u.verbatimUsage).toBeUndefined();
    // Model still resolves via the fallback (instance.model).
    expect(u.model).toBe("gemini-2.0-flash");
  });

  it("_buildLlamaIndexVerbatimUsage: non-object / all-zero / positive gating", () => {
    expect(_buildLlamaIndexVerbatimUsage("google_genai", null)).toBeUndefined();
    expect(_buildLlamaIndexVerbatimUsage("google_genai", 42)).toBeUndefined();
    expect(_buildLlamaIndexVerbatimUsage("google_genai", {})).toBeUndefined();
    expect(_buildLlamaIndexVerbatimUsage("google_genai", { a: 0 })).toBeUndefined();
    expect(_buildLlamaIndexVerbatimUsage("google_genai", { a: 5 })).toEqual({
      shape: "google_genai",
      raw: { a: 5 },
    });
  });

  it("fallback: usageMetadata that fails JSON snapshot (BigInt) → no verbatim block, positional counts intact", () => {
    // JSON.stringify throws on BigInt — without the snapshot this would only
    // surface in the background POST and silently drop the whole log row.
    const um = {
      promptTokenCount: 1000,
      cachedContentTokenCount: 800,
      candidatesTokenCount: 200,
      totalTokenCount: BigInt(2000),
    };
    let u: any;
    expect(() => {
      u = _extractLlamaIndexUsage("google", { raw: { usageMetadata: um } }, new Gemini());
    }).not.toThrow();
    expect(u.verbatimUsage).toBeUndefined();
    // Positional fallback still carries today's netted counts — row not lost.
    expect(u.inputTokens).toBe(200);
    expect(u.outputTokens).toBe(200);
    expect(u.cachedTokens).toBe(800);
  });

  it("fallback: self-referencing usageMetadata (cycle) → no verbatim block, positional counts intact", () => {
    const um: any = { promptTokenCount: 100, candidatesTokenCount: 40 };
    um.self = um; // JSON.stringify throws on cycles
    const u = _extractLlamaIndexUsage("google", { raw: { usageMetadata: um } }, new Gemini());
    expect(u.verbatimUsage).toBeUndefined();
    expect(u.inputTokens).toBe(100);
    expect(u.outputTokens).toBe(40);
  });
});

describe("_logLlamaIndex — forwards extras.usage to tp.log", () => {
  beforeEach(() => resetPack());

  test("google call → tp.log receives extras.usage = {shape:'google_genai', raw:<usageMetadata>}", () => {
    const client = makeClient();
    const logSpy = vi.spyOn(client, "log").mockImplementation(() => {});
    // Fresh isolated session so the read is deterministic (no ambient scope).
    vi.spyOn(contextModule, "getCurrentSession").mockReturnValue(new TPSession());

    const um = {
      promptTokenCount: 1000,
      cachedContentTokenCount: 800,
      candidatesTokenCount: 200,
    };
    _logLlamaIndex(new Gemini(), { raw: { usageMetadata: um } }, 0, "gemini_call", new Date());

    expect(logSpy).toHaveBeenCalledTimes(1);
    const args = logSpy.mock.calls[0];
    expect(args[EXTRAS_ARG]).toMatchObject({ usage: { shape: "google_genai", raw: um } });
    // Positional input arg still the netted fallback (ignored server-side when
    // usage is present, but must remain today's value).
    expect(args[INPUT_TOKENS_ARG]).toBe(200);
    vi.restoreAllMocks();
  });

  test("anthropic streaming stash path → builds anthropic_messages from the stash usage", () => {
    const client = makeClient();
    const logSpy = vi.spyOn(client, "log").mockImplementation(() => {});
    const session = new TPSession();
    // Simulate _tapLlamaIndexAnthropicUsage having stashed the merged
    // message_start/message_delta usage (incl. cache_read_input_tokens).
    (session as any)._pendingLlamaIndexUsage = {
      model: "claude-3-5-sonnet",
      usage: { input_tokens: 100, output_tokens: 50, cache_read_input_tokens: 200 },
    };
    vi.spyOn(contextModule, "getCurrentSession").mockReturnValue(session);

    // Anthropic streamed ChatResponse surfaces no usage → extractor yields 0/0
    // → the stash fills in and builds the verbatim block.
    _logLlamaIndex(new Anthropic(), { raw: {} }, 0, "claude_call", new Date());

    expect(logSpy).toHaveBeenCalledTimes(1);
    const args = logSpy.mock.calls[0];
    expect(args[EXTRAS_ARG]).toMatchObject({
      usage: {
        shape: "anthropic_messages",
        raw: { input_tokens: 100, output_tokens: 50, cache_read_input_tokens: 200 },
      },
    });
    // Stash also filled the positional counts.
    expect(args[INPUT_TOKENS_ARG]).toBe(100);
    // The stash is consumed once.
    expect((session as any)._pendingLlamaIndexUsage).toBeNull();
    vi.restoreAllMocks();
  });

  test("fallback: no usable usage → tp.log still fires with positional counts and NO extras.usage", () => {
    const client = makeClient();
    const logSpy = vi.spyOn(client, "log").mockImplementation(() => {});
    vi.spyOn(contextModule, "getCurrentSession").mockReturnValue(new TPSession());

    _logLlamaIndex(new Gemini(), { raw: {} }, 0, "gemini_call", new Date());

    expect(logSpy).toHaveBeenCalledTimes(1);
    const args = logSpy.mock.calls[0];
    // Row is never lost — log fires — but no verbatim usage is attached.
    expect((args[EXTRAS_ARG] as any).usage).toBeUndefined();
    expect(args[INPUT_TOKENS_ARG]).toBe(0);
    vi.restoreAllMocks();
  });
});

/**
 * OpenAI Responses API via @llamaindex/openai's OpenAIResponses class.
 *
 * Before the fix `OpenAIResponses` was not in the instrumented class list at
 * all, so llamaindex_node on `--api responses` emitted ZERO llm rows. Wiring
 * it up needed a matching provider mapping + usage branch: the Responses
 * `usage` block is `{input_tokens, output_tokens, input_tokens_details,
 * output_tokens_details}` — a different shape from Chat Completions'
 * `{prompt_tokens, completion_tokens, prompt_tokens_details}`.
 *
 * `input_tokens` is cache-INCLUSIVE (mirrors the direct-SDK openai_responses
 * mapping) and `output_tokens` already includes reasoning tokens, so the
 * positional counts net cached out and never re-add reasoning; the verbatim
 * block carries the untouched usage so the server mapper nets cache once.
 */
const RESPONSES_USAGE = {
  input_tokens: 100,
  output_tokens: 20,
  total_tokens: 120,
  input_tokens_details: { cached_tokens: 30 },
  output_tokens_details: { reasoning_tokens: 5 },
};

describe("_extractLlamaIndexUsage — openai_responses branch", () => {
  it("non-stream Response (usage directly on raw) → cache netted once, verbatim forwarded", () => {
    const u = _extractLlamaIndexUsage(
      "openai_responses",
      { raw: { model: "gpt-5-2025-08-07", usage: RESPONSES_USAGE } },
      new OpenAIResponses(),
    );
    expect(u.inputTokens).toBe(70); // 100 - 30 cached
    expect(u.cachedTokens).toBe(30);
    expect(u.outputTokens).toBe(20); // reasoning already folded in, not re-added
    expect(u.model).toBe("gpt-5-2025-08-07");
    expect(u.verbatimUsage).toEqual({
      shape: "openai_responses",
      raw: RESPONSES_USAGE,
    });
  });

  it("stream terminal event (usage under raw.response) → same extraction via unwrap", () => {
    // `response.completed` is the only stream event carrying usage; the
    // Response (and its model) nest under `.response`.
    const u = _extractLlamaIndexUsage(
      "openai_responses",
      {
        raw: {
          type: "response.completed",
          response: { model: "gpt-5-2025-08-07", usage: RESPONSES_USAGE },
        },
      },
      new OpenAIResponses(),
    );
    expect(u.inputTokens).toBe(70);
    expect(u.cachedTokens).toBe(30);
    expect(u.outputTokens).toBe(20);
    expect(u.model).toBe("gpt-5-2025-08-07");
    expect(u.verbatimUsage).toEqual({
      shape: "openai_responses",
      raw: RESPONSES_USAGE,
    });
  });

  it("no cache details → nothing subtracted, cachedTokens 0", () => {
    const usage = { input_tokens: 100, output_tokens: 20 };
    const u = _extractLlamaIndexUsage(
      "openai_responses",
      { raw: { model: "gpt-5", usage } },
      new OpenAIResponses(),
    );
    expect(u.inputTokens).toBe(100);
    expect(u.cachedTokens).toBe(0);
    expect(u.verbatimUsage).toEqual({ shape: "openai_responses", raw: usage });
  });

  it("cached_tokens > input_tokens (hostile) → inputTokens clamped at 0, never negative", () => {
    const u = _extractLlamaIndexUsage(
      "openai_responses",
      { raw: { usage: { input_tokens: 10, output_tokens: 5, input_tokens_details: { cached_tokens: 99 } } } },
      new OpenAIResponses(),
    );
    expect(u.inputTokens).toBe(0);
    expect(u.cachedTokens).toBe(99);
  });

  it("raw.response present but usage null (mid-stream event) → falls back to raw, zeros, no verbatim", () => {
    // Every event before `response.completed` carries `response.usage: null`;
    // the truthy test must not treat those as the terminal event.
    const u = _extractLlamaIndexUsage(
      "openai_responses",
      { raw: { type: "response.output_text.delta", response: { model: "gpt-5", usage: null } } },
      new OpenAIResponses(),
    );
    expect(u.inputTokens).toBe(0);
    expect(u.outputTokens).toBe(0);
    expect(u.cachedTokens).toBe(0);
    expect(u.verbatimUsage).toBeUndefined();
    // raw itself has no `model` → instance fallback.
    expect(u.model).toBe("gpt-5");
  });

  it("garbage raw (no usage at all) → zeros + fallbackModel from the instance", () => {
    const u = _extractLlamaIndexUsage("openai_responses", { raw: {} }, new OpenAIResponses());
    expect(u.inputTokens).toBe(0);
    expect(u.outputTokens).toBe(0);
    expect(u.verbatimUsage).toBeUndefined();
    expect(u.model).toBe("gpt-5");
  });

  it("garbage raw + empty instance → model 'unknown', no throw", () => {
    let u: any;
    expect(() => {
      u = _extractLlamaIndexUsage("openai_responses", { raw: 42 }, {});
    }).not.toThrow();
    expect(u.model).toBe("unknown");
    expect(u.inputTokens).toBe(0);
    expect(u.outputTokens).toBe(0);
    expect(u.verbatimUsage).toBeUndefined();
  });

  it("a raw.response getter that THROWS → zero fallback, no throw (golden rule)", () => {
    const hostileRaw: any = {};
    Object.defineProperty(hostileRaw, "response", {
      get() {
        throw new Error("boom: hostile getter");
      },
    });
    let u: any;
    expect(() => {
      u = _extractLlamaIndexUsage("openai_responses", { raw: hostileRaw }, new OpenAIResponses());
    }).not.toThrow();
    expect(u.inputTokens).toBe(0);
    expect(u.outputTokens).toBe(0);
    expect(u.cachedTokens).toBe(0);
    expect(u.verbatimUsage).toBeUndefined();
    expect(u.model).toBe("gpt-5");
  });

  it("usage that fails the JSON snapshot (BigInt) → no verbatim block, positional counts intact", () => {
    const usage = {
      input_tokens: 100,
      output_tokens: 20,
      input_tokens_details: { cached_tokens: 30 },
      total_tokens: BigInt(120),
    };
    let u: any;
    expect(() => {
      u = _extractLlamaIndexUsage(
        "openai_responses",
        { raw: { model: "gpt-5", usage } },
        new OpenAIResponses(),
      );
    }).not.toThrow();
    expect(u.verbatimUsage).toBeUndefined();
    expect(u.inputTokens).toBe(70);
    expect(u.outputTokens).toBe(20);
    expect(u.cachedTokens).toBe(30);
  });
});

describe("LlamaIndex provider mapping — OpenAIResponses is its own provider", () => {
  beforeEach(() => resetPack());

  /** Provider slug + counts tp.log receives for one LlamaIndex class. */
  function logOnce(instance: any, result: any): any[] {
    const client = makeClient();
    const logSpy = vi.spyOn(client, "log").mockImplementation(() => {});
    vi.spyOn(contextModule, "getCurrentSession").mockReturnValue(new TPSession());
    _logLlamaIndex(instance, result, 0, "li_call", new Date());
    expect(logSpy).toHaveBeenCalledTimes(1);
    return logSpy.mock.calls[0] as any[];
  }

  afterEach(() => vi.restoreAllMocks());

  test("OpenAIResponses → provider 'openai_responses' + Responses-shaped counts", () => {
    const args = logOnce(new OpenAIResponses(), {
      raw: { model: "gpt-5-2025-08-07", usage: RESPONSES_USAGE },
    });
    expect(args[PROVIDER_ARG]).toBe("openai_responses");
    expect(args[MODEL_ARG]).toBe("gpt-5-2025-08-07");
    expect(args[INPUT_TOKENS_ARG]).toBe(70);
    expect(args[OUTPUT_TOKENS_ARG]).toBe(20);
    expect(args[CACHED_TOKENS_ARG]).toBe(30);
    expect(args[EXTRAS_ARG]).toMatchObject({
      usage: { shape: "openai_responses", raw: RESPONSES_USAGE },
    });
  });

  test("OpenAI (sibling class) still maps to 'openai' with the Completions shape", () => {
    const usage = {
      prompt_tokens: 100,
      completion_tokens: 20,
      prompt_tokens_details: { cached_tokens: 30 },
    };
    const args = logOnce(new OpenAI(), { raw: { model: "gpt-4o", usage } });
    expect(args[PROVIDER_ARG]).toBe("openai");
    expect(args[INPUT_TOKENS_ARG]).toBe(70);
    expect(args[EXTRAS_ARG]).toMatchObject({
      usage: { shape: "openai_compatible_chat", raw: usage },
    });
  });

  test("Anthropic / Gemini mappings unchanged by the responses arm", () => {
    expect(
      logOnce(new Anthropic(), { raw: { model: "claude-x", usage: { input_tokens: 5, output_tokens: 2 } } })[
        PROVIDER_ARG
      ],
    ).toBe("anthropic");
    vi.restoreAllMocks();
    expect(
      logOnce(new Gemini(), {
        raw: { modelVersion: "gemini-2.0-flash", usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 2 } },
      })[PROVIDER_ARG],
    ).toBe("google");
  });
});
