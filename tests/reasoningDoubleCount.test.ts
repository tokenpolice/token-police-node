// Reasoning-token accounting regressions (Node parity of the shipped Python C1).
//
// OpenAI Chat Completions `usage.completion_tokens` (camelCase
// `completionTokens`) ALREADY includes
// `completion_tokens_details.reasoning_tokens` — they are a breakdown of the
// completion total, not an addition to it. Previously the Node SDK added the
// reasoning count on top of the completion count in three OpenAI-shaped usage
// extractors, inflating output tokens — and therefore cost — by up to ~2x for
// reasoning models, which could trip budget limits prematurely. These tests
// pin the corrected behavior at all three sites:
//
// - `_extractUsage` openrouter branch (camelCase)
// - `_extractUsage` OpenAI-group branch (snake_case: groq/openai/litellm/...)
// - `_extractLlamaIndexUsage` OpenAI-default branch (via __test__)
//
// The Responses-API shape (`output_tokens` already includes reasoning) and the
// Gemini shape (`thoughtsTokenCount` IS reported separately from
// `candidatesTokenCount`, so summing there is correct) are pinned unchanged.
//
// Everything runs offline against fake usage objects — no network.
import { describe, it, expect } from "vitest";
import { _extractUsage, __test__ } from "../src/enforcer";

const _extractLlamaIndexUsage = __test__._extractLlamaIndexUsage;

describe("reasoning tokens not double-counted — _extractUsage openrouter (site 1, camelCase)", () => {
  it("completionTokens already includes reasoningTokens ⇒ output=1000 not 1600", () => {
    const u = _extractUsage(
      "openrouter",
      {
        model: "m",
        usage: {
          completionTokens: 1000,
          completionTokensDetails: { reasoningTokens: 600 },
        },
      },
      [],
    );
    expect(u.outputTokens).toBe(1000);
  });

  it("non-reasoning, details ABSENT ⇒ output=50", () => {
    const u = _extractUsage(
      "openrouter",
      { usage: { completionTokens: 50 } },
      [],
    );
    expect(u.outputTokens).toBe(50);
  });

  it("non-reasoning, details PRESENT but reasoningTokens=0 ⇒ output=50", () => {
    const u = _extractUsage(
      "openrouter",
      { usage: { completionTokens: 50, completionTokensDetails: { reasoningTokens: 0 } } },
      [],
    );
    expect(u.outputTokens).toBe(50);
  });

  it("cached-prompt/input accounting unchanged (prompt includes cached)", () => {
    const u = _extractUsage(
      "openrouter",
      {
        usage: {
          promptTokens: 100,
          completionTokens: 1000,
          promptTokensDetails: { cachedTokens: 40 },
          completionTokensDetails: { reasoningTokens: 600 },
        },
      },
      [],
    );
    expect(u.inputTokens).toBe(60);
    expect(u.cachedTokens).toBe(40);
    expect(u.outputTokens).toBe(1000);
  });
});

describe("reasoning tokens not double-counted — _extractUsage OpenAI group (site 2, snake_case)", () => {
  it("groq: completion_tokens already includes reasoning_tokens ⇒ output=1000 not 1600", () => {
    const u = _extractUsage(
      "groq",
      {
        usage: {
          completion_tokens: 1000,
          completion_tokens_details: { reasoning_tokens: 600 },
        },
      },
      [],
    );
    expect(u.outputTokens).toBe(1000);
  });

  it("non-reasoning, details ABSENT ⇒ output=50", () => {
    const u = _extractUsage("openai", { usage: { completion_tokens: 50 } }, []);
    expect(u.outputTokens).toBe(50);
  });

  it("non-reasoning, details PRESENT but reasoning_tokens=0 ⇒ output=50", () => {
    const u = _extractUsage(
      "openai",
      { usage: { completion_tokens: 50, completion_tokens_details: { reasoning_tokens: 0 } } },
      [],
    );
    expect(u.outputTokens).toBe(50);
  });

  it("cached-prompt/input accounting unchanged (prompt includes cached)", () => {
    const u = _extractUsage(
      "groq",
      {
        usage: {
          prompt_tokens: 100,
          completion_tokens: 1000,
          prompt_tokens_details: { cached_tokens: 40 },
          completion_tokens_details: { reasoning_tokens: 600 },
        },
      },
      [],
    );
    expect(u.inputTokens).toBe(60);
    expect(u.cachedTokens).toBe(40);
    expect(u.outputTokens).toBe(1000);
  });
});

describe("reasoning tokens not double-counted — _extractLlamaIndexUsage OpenAI default (site 3)", () => {
  it("completion_tokens already includes reasoning_tokens ⇒ output=1000 not 1600", () => {
    const u = _extractLlamaIndexUsage(
      "openai",
      {
        raw: {
          usage: {
            completion_tokens: 1000,
            completion_tokens_details: { reasoning_tokens: 600 },
          },
        },
      },
      {},
    );
    expect(u.outputTokens).toBe(1000);
  });

  it("non-reasoning, details ABSENT ⇒ output=50", () => {
    const u = _extractLlamaIndexUsage(
      "openai",
      { raw: { usage: { completion_tokens: 50 } } },
      {},
    );
    expect(u.outputTokens).toBe(50);
  });

  it("non-reasoning, details PRESENT but reasoning_tokens=0 ⇒ output=50", () => {
    const u = _extractLlamaIndexUsage(
      "openai",
      { raw: { usage: { completion_tokens: 50, completion_tokens_details: { reasoning_tokens: 0 } } } },
      {},
    );
    expect(u.outputTokens).toBe(50);
  });
});

describe("Responses API pinned — output_tokens already includes reasoning (untouched)", () => {
  it("openai_responses: output stays 1000 with reasoning=600", () => {
    const u = _extractUsage(
      "openai_responses",
      {
        usage: {
          input_tokens: 100,
          output_tokens: 1000,
          output_tokens_details: { reasoning_tokens: 600 },
        },
      },
      [],
    );
    expect(u.outputTokens).toBe(1000);
    expect(u.inputTokens).toBe(100);
  });
});

describe("Gemini thoughts STILL summed — separate bucket, carve-out preserved", () => {
  it("_extractUsage google: candidates=100 + thoughts=60 ⇒ output=160", () => {
    const u = _extractUsage(
      "google",
      { usageMetadata: { candidatesTokenCount: 100, thoughtsTokenCount: 60 } },
      [],
    );
    expect(u.outputTokens).toBe(160);
  });

  it("_extractLlamaIndexUsage google: candidates=100 + thoughts=60 ⇒ output=160", () => {
    const u = _extractLlamaIndexUsage(
      "google",
      { raw: { usageMetadata: { candidatesTokenCount: 100, thoughtsTokenCount: 60 } } },
      {},
    );
    expect(u.outputTokens).toBe(160);
  });
});

describe("fail-safe / malformed usage ⇒ zeros, no throw (golden rule)", () => {
  it("_extractUsage openrouter: no usage ⇒ output=0", () => {
    expect(_extractUsage("openrouter", {}, []).outputTokens).toBe(0);
  });
  it("_extractUsage groq: no usage ⇒ output=0", () => {
    expect(_extractUsage("groq", {}, []).outputTokens).toBe(0);
  });
  it("_extractUsage openrouter: bare {} usage ⇒ zeros", () => {
    const u = _extractUsage("openrouter", { usage: {} }, []);
    expect(u.outputTokens).toBe(0);
    expect(u.inputTokens).toBe(0);
    expect(u.cachedTokens).toBe(0);
  });
  it("_extractUsage groq: bare {} usage ⇒ zeros", () => {
    const u = _extractUsage("groq", { usage: {} }, []);
    expect(u.outputTokens).toBe(0);
    expect(u.inputTokens).toBe(0);
  });
  it("_extractLlamaIndexUsage openai: no raw/usage ⇒ output=0", () => {
    expect(_extractLlamaIndexUsage("openai", {}, {}).outputTokens).toBe(0);
  });
  it("_extractLlamaIndexUsage openai: bare {} raw.usage ⇒ zeros", () => {
    const u = _extractLlamaIndexUsage("openai", { raw: { usage: {} } }, {});
    expect(u.outputTokens).toBe(0);
    expect(u.inputTokens).toBe(0);
    expect(u.cachedTokens).toBe(0);
  });
});
