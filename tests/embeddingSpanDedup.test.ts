/**
 * Embedding-span de-duplication (Node↔Python parity restore).
 *
 * opentelemetry-instrumentation-openai / -cohere patch embeddings and emit an
 * OTel span tagged `llm.request.type="embedding"` (span names "openai.embeddings"
 * / "cohere.embed"). TokenPolice captures embeddings authoritatively via its own
 * manual wrapper (which creates NO OTel span), so any such span reaching the
 * SpanProcessor is the instrumentor's spurious duplicate and must be dropped —
 * otherwise it is mis-logged as a chat-shaped row (~2x spend for OpenAI).
 *
 * Part 1 (truth table) is a 1:1 port of tests/test_embedding_dedup.py, importing
 * the exported helper directly (mirrors Python's `from ... import
 * _is_instrumentor_embedding_span`). Part 2 drives a synthetic finished span
 * through the real `onEnd` via the sdkUsageParity.test.ts harness pattern.
 *
 * RED/GREEN (item 15): pre-fix, `onEnd` has no embedding guard between the tool
 * guard and `isLlmSpan`, so the embedding span (gen_ai.system + nonzero usage)
 * passes `isLlmSpan` and logs 1 row → the integration `logged.length === 0`
 * assertion FAILS (RED). The pure-predicate rows reference a helper that does
 * not exist pre-fix (import error). With the early-return added, both PASS
 * (GREEN).
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

const logged: any[] = [];
vi.mock("../src/state", () => ({
  getClient: () => ({
    log: (...args: any[]) => {
      logged.push(args);
    },
  }),
  drainObservations: () => [],
}));

import { TokenPoliceSpanProcessor, isInstrumentorEmbeddingSpan } from "../src/telemetry";

// ── Part 1: pure truth table (1:1 port of test_embedding_dedup.py) ──
describe("isInstrumentorEmbeddingSpan — truth table (Python parity oracle)", () => {
  // test_llm_request_type_embedding
  it("llm.request.type=embedding → true", () => {
    expect(isInstrumentorEmbeddingSpan({ "llm.request.type": "embedding" }, "anything")).toBe(true);
  });
  // test_llm_request_type_embedding_uppercase
  it("llm.request.type=EMBEDDING (case-insensitive) → true", () => {
    expect(isInstrumentorEmbeddingSpan({ "llm.request.type": "EMBEDDING" }, "")).toBe(true);
  });
  // test_openai_embeddings_span_name
  it('name "openai.embeddings" → true', () => {
    expect(isInstrumentorEmbeddingSpan({}, "openai.embeddings")).toBe(true);
  });
  // test_cohere_embed_span_name
  it('name "cohere.embed" → true', () => {
    expect(isInstrumentorEmbeddingSpan({}, "cohere.embed")).toBe(true);
  });
  // test_generic_dot_embeddings_name
  it('name endsWith ".embeddings" → true', () => {
    expect(isInstrumentorEmbeddingSpan({}, "something.embeddings")).toBe(true);
  });
  // test_bedrock_titan_embed_model_id
  it("gen_ai.request.model titan-embed → true", () => {
    expect(isInstrumentorEmbeddingSpan({ "gen_ai.request.model": "titan-embed-text-v2:0" }, "")).toBe(true);
  });
  // test_bedrock_cohere_embed_model_id
  it("gen_ai.request.model embed-english-v3 → true", () => {
    expect(isInstrumentorEmbeddingSpan({ "gen_ai.request.model": "embed-english-v3" }, "")).toBe(true);
  });
  // test_bedrock_voyage_embed_model_id
  it("gen_ai.request.model voyage-3 → true", () => {
    expect(isInstrumentorEmbeddingSpan({ "gen_ai.request.model": "voyage-3" }, "")).toBe(true);
  });
  // test_legacy_llm_request_model_key
  it("legacy llm.request.model titan-embed → true", () => {
    expect(isInstrumentorEmbeddingSpan({ "llm.request.model": "titan-embed-text-v2:0" }, "")).toBe(true);
  });
  // Falsy-coalescing fallthrough (teeth vs ??): empty primary key falls through
  // to legacy key. FAILS a `??` impl, PASSES the pinned `||` impl (item 4).
  it("empty gen_ai.request.model falls through to legacy llm.request.model (|| not ??) → true", () => {
    expect(
      isInstrumentorEmbeddingSpan(
        { "gen_ai.request.model": "", "llm.request.model": "titan-embed-text-v2:0" },
        "",
      ),
    ).toBe(true);
  });
  // test_bedrock_chat_models_are_not_embedding
  it("Bedrock chat model ids → false", () => {
    expect(isInstrumentorEmbeddingSpan({ "gen_ai.request.model": "nova-lite-v1:0" }, "")).toBe(false);
    expect(isInstrumentorEmbeddingSpan({ "gen_ai.request.model": "claude-3-5-sonnet" }, "")).toBe(false);
    expect(isInstrumentorEmbeddingSpan({ "gen_ai.request.model": "titan-text-express-v1" }, "")).toBe(false);
  });
  // test_chat_span_is_not_embedding
  it("llm.request.type=chat / openai.chat → false", () => {
    expect(isInstrumentorEmbeddingSpan({ "llm.request.type": "chat" }, "openai.chat")).toBe(false);
  });
  // test_plain_chat_span
  it("plain chat span → false", () => {
    expect(isInstrumentorEmbeddingSpan({ "gen_ai.system": "openai" }, "chat gpt-4o-mini")).toBe(false);
  });
  // test_empty_inputs
  it("empty inputs → false", () => {
    expect(isInstrumentorEmbeddingSpan({}, "")).toBe(false);
    expect(isInstrumentorEmbeddingSpan({}, null as any)).toBe(false);
  });
  // Hostile inputs — pure/total, never throws (item 9, Golden-Rule encode)
  it("hostile inputs return a boolean, never throw", () => {
    expect(isInstrumentorEmbeddingSpan({}, null as any)).toBe(false);
    expect(isInstrumentorEmbeddingSpan({ "llm.request.type": {} as any }, "")).toBe(false);
    expect(isInstrumentorEmbeddingSpan(null as any, "")).toBe(false);
  });
});

// ── Part 2: integration through the real onEnd (item 6) ──
function fakeSpan(attrs: Record<string, unknown>, name: string) {
  return {
    attributes: attrs,
    name,
    startTime: [0, 0] as [number, number],
    endTime: [1, 0] as [number, number],
    spanContext: () => ({
      traceId: "0000000000000000000000000000abcd",
      spanId: "0000000000001234",
    }),
    parentSpanId: undefined,
  } as any;
}

describe("onEnd drops instrumentor embedding spans before isLlmSpan", () => {
  beforeEach(() => {
    logged.length = 0;
  });

  it("embedding span → 0 /log calls (dropped), chat span → 1 /log call", async () => {
    const proc = new TokenPoliceSpanProcessor();

    // Embedding span: carries the SAME log-qualifying attrs as a chat span
    // (gen_ai.system + nonzero usage) PLUS llm.request.type=embedding. Without
    // the guard it would pass isLlmSpan and log (RED); with it, dropped (GREEN).
    proc.onEnd(
      fakeSpan(
        {
          "gen_ai.system": "openai",
          "llm.request.type": "embedding",
          "gen_ai.usage.input_tokens": 100,
          "gen_ai.usage.output_tokens": 0,
        },
        "openai.embeddings",
      ),
    );
    await new Promise((r) => setTimeout(r, 0));
    expect(logged.length).toBe(0);

    // Chat span with the same shape (minus the embedding tag) still logs.
    proc.onEnd(
      fakeSpan(
        {
          "gen_ai.system": "openai",
          "gen_ai.usage.input_tokens": 100,
          "gen_ai.usage.output_tokens": 50,
        },
        "openai.chat",
      ),
    );
    await new Promise((r) => setTimeout(r, 0));
    expect(logged.length).toBe(1);
  });
});
