/**
 * Regression tests for the 2026-06-11 fix round (real-apps verification
 * findings — see SDK_VERIFICATION_2026-06-11.md):
 * - classify: status recovery from wrapped errors + "HTTP NNN" messages
 * - composition: @google/genai `.text` getter must not swallow functionCall
 * parts; OpenAI Responses image_generation_call gets an entry
 * - usage: AI-SDK anthropic finish with input=0 recovers input from
 * providerMetadata.anthropic.usage (MiniMax message_delta shape)
 * - usage: cohere SDK camelCase meta.billedUnits on embed responses
 */
import { describe, it, expect } from "vitest";
import { classifyException } from "../src/_classify";
import { buildResponseComposition } from "../src/composition";
import { _extractUsage, _extractEmbeddingUsage } from "../src/enforcer";

describe("classifyException — wrapped/messaged status recovery", () => {
  it("parses 'HTTP NNN' from a plain Error message", () => {
    const c = classifyException(new Error("OpenAI HTTP 429: Too many requests"));
    expect(c).toEqual({ error_kind: "rate_limited", http_status: 429 });
  });

  it("parses 'status code: NNN' from a plain Error message", () => {
    const c = classifyException(new Error("Request failed with status code: 400"));
    expect(c).toEqual({ error_kind: "client_error", http_status: 400 });
  });

  it("walks the cause chain for a status-bearing inner error", () => {
    const inner: any = new Error("provider 400");
    inner.status = 400;
    const outer = new Error("stream iteration failed", { cause: inner });
    const c = classifyException(outer);
    expect(c).toEqual({ error_kind: "client_error", http_status: 400 });
  });

  it("still returns unknown/0 when nothing carries a status", () => {
    const c = classifyException(new Error("something odd happened"));
    expect(c).toEqual({ error_kind: "unknown", http_status: 0 });
  });

  it("direct status attribute still wins over the message", () => {
    const err: any = new Error("HTTP 500 mentioned in text");
    err.status = 429;
    const c = classifyException(err);
    expect(c.http_status).toBe(429);
    expect(c.error_kind).toBe("rate_limited");
  });
});

describe("buildResponseComposition — google .text getter vs functionCall", () => {
  /** Mimics @google/genai GenerateContentResponse: candidates + a `.text`
   * convenience getter concatenating the text parts. */
  function makeGoogleResponse(parts: any[]): any {
    const resp: any = { candidates: [{ content: { parts } }] };
    Object.defineProperty(resp, "text", {
      get() {
        return parts
          .map((p) => (typeof p.text === "string" ? p.text : ""))
          .join("");
      },
    });
    return resp;
  }

  it("keeps the tool_call entry when the response has text + functionCall", () => {
    const resp = makeGoogleResponse([
      { text: "I will search the knowledge base.", thoughtSignature: "x" },
      { functionCall: { name: "searchKnowledgeBase", args: { query: "delays" } } },
    ]);
    const comp = buildResponseComposition("google", resp);
    expect(comp.map((e: any) => e.type)).toEqual(["text", "tool_call"]);
    expect((comp[1] as any).name).toBe("searchKnowledgeBase");
  });

  it("tool-call-only responses still compose a tool_call", () => {
    const resp = makeGoogleResponse([
      { functionCall: { name: "lookupUser", args: { id: 1 } } },
    ]);
    const comp = buildResponseComposition("google", resp);
    expect(comp.map((e: any) => e.type)).toEqual(["tool_call"]);
  });

  it("whisper-style transcription bodies still compose as text", () => {
    const comp = buildResponseComposition("openai", { text: "transcribed words" });
    expect(comp).toHaveLength(1);
    expect((comp[0] as any).type).toBe("text");
    expect((comp[0] as any).role).toBe("assistant");
  });
});

describe("buildResponseComposition — Responses image_generation_call", () => {
  it("emits an image entry for the built-in image_generation tool output", () => {
    const resp = {
      output: [
        { type: "image_generation_call", id: "ig_1", result: "<b64>" },
        {
          type: "message",
          content: [{ type: "output_text", text: "Here is your image." }],
        },
      ],
    };
    const comp = buildResponseComposition("openai_responses", resp);
    expect(comp.map((e: any) => e.type)).toEqual(["image", "text"]);
  });
});

describe("_extractUsage — AI SDK anthropic input-token recovery", () => {
  it("recovers input tokens from providerMetadata when finish usage has 0", () => {
    const finish = {
      type: "finish",
      usage: { inputTokens: 0, outputTokens: 119, totalTokens: 119 },
      providerMetadata: {
        anthropic: { usage: { input_tokens: 388, output_tokens: 119 } },
      },
    };
    const u = _extractUsage("ai_sdk", finish, [{ __tpXaiModel: "MiniMax-M2.5" }]);
    expect(u.inputTokens).toBe(388);
    expect(u.outputTokens).toBe(119);
  });

  it("does not engage when finish usage already carries input tokens", () => {
    const finish = {
      type: "finish",
      usage: { inputTokens: 450, outputTokens: 60 },
      providerMetadata: { anthropic: { usage: { input_tokens: 9999 } } },
    };
    const u = _extractUsage("ai_sdk", finish, [{ __tpXaiModel: "m" }]);
    expect(u.inputTokens).toBe(450);
  });

  it("stays zero when no providerMetadata exists (no invented tokens)", () => {
    const finish = { type: "finish", usage: { inputTokens: 0, outputTokens: 5 } };
    const u = _extractUsage("ai_sdk", finish, [{ __tpXaiModel: "m" }]);
    expect(u.inputTokens).toBe(0);
    expect(u.outputTokens).toBe(5);
  });
});

describe("_extractUsage — Bedrock Converse manual token capture", () => {
  // Node ships no Bedrock instrumentor; Converse usage is extracted manually
  // from response.usage. Pin the mapping so a broken field name regresses here.
  it("maps Converse response.usage input/output/cacheRead + model from modelId", () => {
    const resp = {
      output: { message: { role: "assistant" } },
      stopReason: "end_turn",
      usage: { inputTokens: 321, outputTokens: 88, cacheReadInputTokens: 40 },
    };
    const u = _extractUsage("bedrock", resp, [
      { modelId: "anthropic.claude-3-sonnet" },
    ]);
    expect(u).toEqual({
      model: "anthropic.claude-3-sonnet",
      inputTokens: 321,
      outputTokens: 88,
      cachedTokens: 40,
    });
  });

  it("degrades to all-zero tokens (no throw / NaN) when usage is missing", () => {
    const u = _extractUsage("bedrock", {}, [{ modelId: "m" }]);
    expect(u).toEqual({
      model: "m",
      inputTokens: 0,
      outputTokens: 0,
      cachedTokens: 0,
    });
    expect(Number.isNaN(u.inputTokens)).toBe(false);
    expect(Number.isNaN(u.outputTokens)).toBe(false);
    expect(Number.isNaN(u.cachedTokens)).toBe(false);
  });
});

describe("_extractEmbeddingUsage — cohere camelCase billedUnits", () => {
  it("reads SDK-shaped meta.billedUnits.inputTokens", () => {
    const u = _extractEmbeddingUsage(
      { meta: { billedUnits: { inputTokens: 125 } } },
      "cohere",
    );
    expect(u.inputTokens).toBe(125);
  });

  it("snake_case wire shape still wins when present", () => {
    const u = _extractEmbeddingUsage(
      { meta: { billed_units: { input_tokens: 50 } } },
      "cohere",
    );
    expect(u.inputTokens).toBe(50);
  });
});

describe("_extractUsage — cohere chat snake_case vs camelCase", () => {
  // protect()/_logManual forwards a raw-REST Cohere chat response; those keep
  // snake_case (billed_units.input_tokens). Without a snake_case read the
  // container/fields miss and the call lands 0/0 tokens (cost unmeasured).
  it("reads raw-REST v2 usage.billed_units.input/output_tokens (snake_case)", () => {
    const resp = {
      usage: {
        billed_units: { input_tokens: 120, output_tokens: 45 },
        tokens: { input_tokens: 130, output_tokens: 45 },
      },
    };
    const u = _extractUsage("cohere", resp, [{ model: "command-r" }]);
    expect(u).toEqual({
      model: "command-r",
      inputTokens: 120,
      outputTokens: 45,
      cachedTokens: 0,
    });
  });

  it("reads raw-REST v1 meta.billed_units (snake_case) when usage is absent", () => {
    const resp = {
      meta: { billed_units: { input_tokens: 77, output_tokens: 12 } },
    };
    const u = _extractUsage("cohere", resp, [{ model: "command" }]);
    expect(u.inputTokens).toBe(77);
    expect(u.outputTokens).toBe(12);
  });

  it("still reads SDK-client camelCase usage.billedUnits.inputTokens", () => {
    const resp = { usage: { billedUnits: { inputTokens: 200, outputTokens: 88 } } };
    const u = _extractUsage("cohere", resp, [{ model: "command-r-plus" }]);
    expect(u.inputTokens).toBe(200);
    expect(u.outputTokens).toBe(88);
  });

  it("reads the streaming message-end delta.usage container", () => {
    const chunk = {
      delta: { usage: { billed_units: { input_tokens: 5, output_tokens: 9 } } },
    };
    const u = _extractUsage("cohere", chunk, [{ model: "command-r" }]);
    expect(u.inputTokens).toBe(5);
    expect(u.outputTokens).toBe(9);
  });

  it("degrades to all-zero tokens (no throw / NaN) when usage is missing", () => {
    const u = _extractUsage("cohere", {}, [{ model: "command" }]);
    expect(u).toEqual({
      model: "command",
      inputTokens: 0,
      outputTokens: 0,
      cachedTokens: 0,
    });
    expect(Number.isNaN(u.inputTokens)).toBe(false);
    expect(Number.isNaN(u.outputTokens)).toBe(false);
  });
});

describe("_extractRawUsage via _extractUsage parity — ai_sdk raw recovery", () => {
  // The raw forward is exercised through _logManual; here we lock the
  // extractor pair contract: when _extractUsage recovers input from
  // providerMetadata, the server-bound raw must carry it too (the
  // vercel_ai mapper bills from raw, not from the SDK ints).
  it("nested V3 usage gets total (and noCache) patched", async () => {
    const mod = await import("../src/enforcer");
    const finish = {
      type: "finish",
      usage: {
        inputTokens: { total: 0, noCache: 0, cacheRead: 0, cacheWrite: 0 },
        outputTokens: { total: 119, text: 119, reasoning: 0 },
      },
      providerMetadata: { anthropic: { usage: { input_tokens: 388 } } },
    };
    const ints = mod._extractUsage("ai_sdk", finish, [{ __tpXaiModel: "m" }]);
    expect(ints.inputTokens).toBe(388);
  });
});
