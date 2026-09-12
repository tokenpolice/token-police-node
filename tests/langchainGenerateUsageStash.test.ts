/**
 * LangChain/LangGraph GENERATE-path usage stash (OpenAI Responses API).
 *
 * `@langchain/openai` on the Responses API reports token usage only under
 * `LLMResult.llmOutput.estimatedTokenUsage`. Traceloop's handleLLMEnd reads
 * exactly four keys — `llmOutput.usage.{input_tokens,output_tokens}` and
 * `llmOutput.tokenUsage.{promptTokens,completionTokens}` — so the LLM span
 * ended with zero usage attrs and telemetry dropped the whole row: the
 * langchain_node / langgraph_node apps emitted ZERO llm rows on Responses
 * (unmetered, unenforced spend).
 *
 * Fix: `_captureLangchainResponse` reads the endpoint-agnostic
 * `generations[][].message.usage_metadata` and stashes it (usage_source
 * "langchain_message") so the telemetry merge repairs the zeros.
 *
 * The stash is GATED on those four wire-usage keys all being non-positive.
 * Ungated it would perturb providers whose wire usage traceloop DOES read —
 * notably Anthropic, whose `llmOutput.usage.input_tokens` is cache-EXCLUSIVE
 * while LC's `usage_metadata.input_tokens` is cache-INCLUSIVE, which would
 * flip telemetry's preferStash comparison on cached non-stream calls.
 *
 * Sums across the OUTER generations list (batch generate = one provider call
 * per entry) but reads only the FIRST usage-bearing candidate of each INNER
 * list (LC stamps the same full-call usage_metadata on every n>1 candidate).
 *
 * All offline against fake LLMResult objects — no network, no LangChain.
 * Streaming-path stash lives in langchainStreamUsageStash.test.ts.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { setClient } from "../src/state";
import { session } from "../src/context";
import { __test__ as enforcerTest } from "../src/enforcer";

const { _captureLangchainResponse } = enforcerTest as any;

beforeEach(() => {
  setClient({ firewall: "off", log: () => {} } as any);
});
afterEach(() => setClient(null as any));

/** A LangChain Generation carrying an AIMessage with usage_metadata. */
function gen(usage_metadata: unknown, text = "hi"): any {
  return { text, message: { content: text, usage_metadata } };
}

/** The stashed usage block for span order 0 of `s`, or undefined. */
function stashedUsage(s: any, order = 0): any {
  return s._pendingCompositions[`${s.traceId}:${order}`]?.usage;
}

describe("LangChain generate-path usage stash — Responses shape (repair)", () => {
  it("Responses LLMResult (usage only under estimatedTokenUsage) → stash written", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponse(
        {
          // Exactly what @langchain/openai emits on /v1/responses: the four
          // keys traceloop reads are all absent.
          llmOutput: { estimatedTokenUsage: { promptTokens: 1000, completionTokens: 200 } },
          generations: [
            [
              gen({
                input_tokens: 1000,
                output_tokens: 200,
                total_tokens: 1200,
                // Raw Response.usage detail shape (not LC-standard).
                input_tokens_details: { cached_tokens: 800 },
              }),
            ],
          ],
        },
        0,
      );
      expect(stashedUsage(s)).toEqual({
        usage_source: "langchain_message",
        input_tokens: 1000,
        output_tokens: 200,
        cached_tokens: 800,
        cache_creation_tokens: 0,
      });
      // The stash shares the compKey with the response composition and must
      // not displace it.
      expect(s._pendingCompositions[`${s.traceId}:0`].response).toBeDefined();
    });
  });

  it("stash lands under the compKey of the ORDER passed, not the span counter", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponse(
        { generations: [[gen({ input_tokens: 11, output_tokens: 3 })]] },
        7,
      );
      expect(stashedUsage(s, 7)).toMatchObject({ input_tokens: 11, output_tokens: 3 });
      expect(stashedUsage(s, 0)).toBeUndefined();
    });
  });

  it("LC-standard cache detail (input_token_details.cache_read/cache_creation) forwarded", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponse(
        {
          llmOutput: {},
          generations: [
            [
              gen({
                input_tokens: 500,
                output_tokens: 40,
                input_token_details: { cache_read: 50, cache_creation: 70 },
              }),
            ],
          ],
        },
        0,
      );
      expect(stashedUsage(s)).toMatchObject({
        input_tokens: 500,
        output_tokens: 40,
        cached_tokens: 50,
        cache_creation_tokens: 70,
      });
    });
  });

  it("dual cache read: LC-standard cache_read=0 falls through to raw cached_tokens", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponse(
        {
          generations: [
            [
              gen({
                input_tokens: 300,
                output_tokens: 10,
                input_token_details: { cache_read: 0 },
                input_tokens_details: { cached_tokens: 250 },
              }),
            ],
          ],
        },
        0,
      );
      expect(stashedUsage(s)).toMatchObject({ cached_tokens: 250 });
    });
  });

  it("batch generate: usage SUMS across the outer generations lists", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponse(
        {
          generations: [
            [gen({ input_tokens: 10, output_tokens: 1 })],
            [gen({ input_tokens: 20, output_tokens: 2 })],
          ],
        },
        0,
      );
      expect(stashedUsage(s)).toMatchObject({ input_tokens: 30, output_tokens: 3 });
    });
  });

  it("n>1 candidates: identical per-candidate usage_metadata counted ONCE", () => {
    session({ name: "wf" }, (s) => {
      // LC stamps the same full-call usage_metadata on every choice; summing
      // the inner list would double-bill an n=2 call.
      const um = { input_tokens: 100, output_tokens: 20 };
      _captureLangchainResponse(
        { generations: [[gen(um, "a"), gen(um, "b")]] },
        0,
      );
      expect(stashedUsage(s)).toMatchObject({ input_tokens: 100, output_tokens: 20 });
    });
  });

  it("batch × n>1: first usage-bearing candidate per list, summed across lists", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponse(
        {
          generations: [
            // First candidate of list 0 carries no usage → the second is read.
            [gen(undefined, "a"), gen({ input_tokens: 10, output_tokens: 1 }, "b")],
            [gen({ input_tokens: 20, output_tokens: 2 }, "c"), gen({ input_tokens: 20, output_tokens: 2 }, "d")],
          ],
        },
        0,
      );
      expect(stashedUsage(s)).toMatchObject({ input_tokens: 30, output_tokens: 3 });
    });
  });

  it("non-array inner entry (bare Generation) is normalized, not dropped", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponse(
        { generations: [gen({ input_tokens: 5, output_tokens: 2 })] },
        0,
      );
      expect(stashedUsage(s)).toMatchObject({ input_tokens: 5, output_tokens: 2 });
    });
  });
});

describe("LangChain generate-path usage stash — wire-usage gate (must NOT stash)", () => {
  it("Completions shape (llmOutput.tokenUsage positive) → no stash", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponse(
        {
          llmOutput: { tokenUsage: { promptTokens: 100, completionTokens: 20, totalTokens: 120 } },
          generations: [[gen({ input_tokens: 100, output_tokens: 20 })]],
        },
        0,
      );
      expect(stashedUsage(s)).toBeUndefined();
    });
  });

  it("Completions shape with only completionTokens positive → no stash", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponse(
        {
          llmOutput: { tokenUsage: { promptTokens: 0, completionTokens: 20 } },
          generations: [[gen({ input_tokens: 100, output_tokens: 20 })]],
        },
        0,
      );
      expect(stashedUsage(s)).toBeUndefined();
    });
  });

  it("Anthropic shape (llmOutput.usage.input_tokens positive) → no stash", () => {
    session({ name: "wf" }, (s) => {
      // llmOutput.usage.input_tokens is cache-EXCLUSIVE (17) while
      // usage_metadata.input_tokens is cache-INCLUSIVE (9638). Stashing here
      // would flip telemetry's preferStash arm and over-bill uncached input.
      _captureLangchainResponse(
        {
          llmOutput: {
            usage: {
              input_tokens: 17,
              output_tokens: 10,
              cache_read_input_tokens: 9621,
            },
          },
          generations: [
            [
              gen({
                input_tokens: 9638,
                output_tokens: 10,
                input_token_details: { cache_read: 9621, cache_creation: 0 },
              }),
            ],
          ],
        },
        0,
      );
      expect(stashedUsage(s)).toBeUndefined();
    });
  });

  it("Anthropic shape with only output_tokens positive → no stash", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponse(
        {
          llmOutput: { usage: { input_tokens: 0, output_tokens: 10 } },
          generations: [[gen({ input_tokens: 9638, output_tokens: 10 })]],
        },
        0,
      );
      expect(stashedUsage(s)).toBeUndefined();
    });
  });

  it("all-zero usage_metadata → no stash (helper refuses all-zero)", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponse(
        {
          llmOutput: {},
          generations: [[gen({ input_tokens: 0, output_tokens: 0, total_tokens: 0 })]],
        },
        0,
      );
      expect(stashedUsage(s)).toBeUndefined();
    });
  });
});

describe("LangChain generate-path usage stash — hostile shapes never throw (golden rule)", () => {
  const hostile: Array<[string, unknown]> = [
    ["null result", null],
    ["undefined result", undefined],
    ["non-object result", 42],
    ["generations not an array", { generations: "nope" }],
    ["generations null", { generations: null }],
    ["empty generations", { generations: [] }],
    ["inner list of nulls", { generations: [[null, undefined]] }],
    ["generation without message", { generations: [[{ text: "hi" }]] }],
    ["message without usage_metadata", { generations: [[{ message: { content: "hi" } }]] }],
    ["usage_metadata is a string", { generations: [[{ message: { usage_metadata: "lots" } }]] }],
    [
      "usage_metadata counts are garbage strings",
      { generations: [[{ message: { usage_metadata: { input_tokens: "abc", output_tokens: "def" } } }]] },
    ],
    [
      "usage_metadata counts are NaN/objects",
      { generations: [[{ message: { usage_metadata: { input_tokens: NaN, output_tokens: {} } } }]] },
    ],
    [
      "cache detail is a garbage string",
      { generations: [[{ message: { usage_metadata: { input_tokens: 0, output_tokens: 0, input_token_details: "x" } } }]] },
    ],
  ];

  for (const [name, result] of hostile) {
    it(`${name} → no throw, no stash`, () => {
      session({ name: "wf" }, (s) => {
        expect(() => _captureLangchainResponse(result, 0)).not.toThrow();
        expect(stashedUsage(s)).toBeUndefined();
      });
    });
  }

  it("llmOutput getter that THROWS → no throw, no stash", () => {
    session({ name: "wf" }, (s) => {
      const result: any = { generations: [[gen({ input_tokens: 10, output_tokens: 2 })]] };
      Object.defineProperty(result, "llmOutput", {
        get() {
          throw new Error("boom: hostile getter");
        },
      });
      expect(() => _captureLangchainResponse(result, 0)).not.toThrow();
      expect(stashedUsage(s)).toBeUndefined();
    });
  });

  it("usage_metadata getter that THROWS mid-iteration → no throw, no stash", () => {
    session({ name: "wf" }, (s) => {
      const message: any = { content: "hi" };
      Object.defineProperty(message, "usage_metadata", {
        get() {
          throw new Error("boom: hostile getter");
        },
      });
      expect(() =>
        _captureLangchainResponse({ generations: [[{ message }]] }, 0),
      ).not.toThrow();
      expect(stashedUsage(s)).toBeUndefined();
    });
  });
});
