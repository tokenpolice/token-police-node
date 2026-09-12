/**
 * Native `groq-sdk` manual target (Node parity with Python).
 *
 * The native `groq-sdk` npm package is a separate Stainless SDK with its own
 * client — NOT the `openai` package — so no OpenLLMetry-JS instrumentor patches
 * it. Before this fix a direct `import Groq from "groq-sdk"` call was BOTH
 * un-metered (no /log) AND un-enforced (no /check). The fix adds a native groq
 * manual-tap target mirroring the shipped cerebras/together targets.
 *
 * Neither `groq-sdk` is installed in this repo, so these assertions drive the
 * exported enforcer internals directly (no live SDK import needed):
 * - _detectProvider("groq-sdk") === "groq" (#11)
 * - _resolveUsageShape("groq") === "groq_chat" (#12)
 * - _TARGET_METHODS has a manual groq-sdk entry (#9 + enforcement #16)
 * - _extractUsage("groq", nonStreamingResp) (#13)
 * - _chunkHasUsage + _extractUsage on a groq FINAL chunk (#14 — x_groq.usage)
 * - groq present in ALL THREE stream sets (#15)
 * - Vercel @ai-sdk/groq path unaffected (#19)
 * - no regression to cerebras/together/openai (#19)
 *
 * Pins verified against the published `groq-sdk` (npm v1.3.0):
 * - export shape: default export `Groq`; runtime `Groq.Chat = Chat`,
 * `Chat.Completions = Completions`, `Completions.prototype.create` — so the
 * objectPath is ["Groq","Chat","Completions","prototype"] (client.js:500-501,
 * resources/chat/chat.js:16).
 * - streaming usage: `ChatCompletionChunk` (resources/chat/completions.d.ts:290)
 * has NO top-level `usage`; usage lives ONLY at `chunk.x_groq.usage`
 * (ChatCompletionChunk.XGroq.usage, completions.d.ts:826). Non-streaming
 * `ChatCompletion.usage` is top-level OpenAI-shaped.
 */
import { describe, it, expect } from "vitest";
import { _extractUsage, __test__ } from "../src/enforcer";

const {
  _detectProvider,
  _resolveUsageShape,
  _chunkHasUsage,
  _newStreamAccumulator,
  _accumulateStreamChunk,
  _streamAccumulatorToResponse,
  _isAiSdkParse,
  _mapAiSdkProvider,
  _TARGET_METHODS,
} = __test__ as any;

describe("Native groq-sdk target — Node", () => {
  // ── #11: provider detection. RED without the fix (returns ""). ──
  it("_detectProvider maps groq-sdk (and any groq module) to 'groq'", () => {
    expect(_detectProvider("groq-sdk")).toBe("groq");
    expect(_detectProvider("GROQ-SDK")).toBe("groq");
    // must not shadow / be shadowed: openai + together still resolve correctly
    expect(_detectProvider("openai")).toBe("openai");
    expect(_detectProvider("together-ai")).toBe("together");
    expect(_detectProvider("@cerebras/cerebras_cloud_sdk")).toBe("cerebras");
  });

  // ── #12: slug. ──
  it("groq resolves to the existing 'groq_chat' usage shape", () => {
    expect(_resolveUsageShape("groq", [])).toBe("groq_chat");
  });

  // ── #9 + #16: manual target entry present with the verified objectPath. ──
  it("_TARGET_METHODS has a manual groq-sdk create entry (verified export path)", () => {
    const entry = _TARGET_METHODS.find((t: any) => t.moduleName === "groq-sdk");
    expect(entry).toBeDefined();
    expect(entry.manualTelemetry).toBe(true);
    expect(entry.method).toBe("create");
    expect(entry.isAsync).toBe(true);
    // Verified against published groq-sdk v1.3.0 runtime export shape.
    expect(entry.objectPath).toEqual(["Groq", "Chat", "Completions", "prototype"]);
  });

  // ── #16 enforcement parity: structurally identical wrapper site to
  // cerebras/together (same keys → same failSafe manual wrapper that issues the
  // single pre-flight /check; enforce BLOCK raises TokenPoliceBlockedError). No
  // separate/duplicate check path is introduced. ──
  it("groq entry is the same manual-wrapper shape as cerebras/together", () => {
    const groq = _TARGET_METHODS.find((t: any) => t.moduleName === "groq-sdk");
    const cerebras = _TARGET_METHODS.find(
      (t: any) => t.moduleName === "@cerebras/cerebras_cloud_sdk",
    );
    const together = _TARGET_METHODS.find((t: any) => t.moduleName === "together-ai");
    for (const ref of [cerebras, together]) {
      expect(Object.keys(groq).sort()).toEqual(Object.keys(ref).sort());
      expect(groq.method).toBe(ref.method);
      expect(groq.isAsync).toBe(ref.isAsync);
      expect(groq.manualTelemetry).toBe(ref.manualTelemetry);
    }
  });

  // ── #13: NON-streaming OpenAI-shaped usage extraction. ──
  it("_extractUsage extracts tokens from a non-streaming groq response", () => {
    const resp = {
      model: "llama-3.3-70b-versatile",
      usage: { prompt_tokens: 120, completion_tokens: 34, total_tokens: 154 },
    };
    const u = _extractUsage("groq", resp, [{ model: "llama-3.3-70b-versatile" }]);
    expect(u.inputTokens).toBe(120);
    expect(u.outputTokens).toBe(34);
    expect(u.cachedTokens).toBe(0);
    expect(u.model).toBe("llama-3.3-70b-versatile");
  });

  // ── #14 (FIRST-CLASS streaming metering): a realistically-shaped groq FINAL
  // stream chunk carries usage ONLY under `x_groq.usage` (top-level `.usage`
  // absent, per verified groq-sdk chunk type). Detection AND extraction must
  // work. RED without the x_groq unwrap (_chunkHasUsage false → chunk never
  // latched; _extractUsage reads {} → 0/0). ──
  it("meters a groq streaming FINAL chunk whose usage is nested under x_groq.usage", () => {
    const finalChunk = {
      id: "chatcmpl-abc",
      object: "chat.completion.chunk",
      model: "llama-3.3-70b-versatile",
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
      // No top-level `usage` — matches the groq-sdk ChatCompletionChunk type.
      x_groq: {
        id: "req_01",
        usage: { prompt_tokens: 210, completion_tokens: 88, total_tokens: 298 },
      },
    };
    expect(_chunkHasUsage("groq", finalChunk)).toBe(true);
    const u = _extractUsage("groq", finalChunk, [{ model: "llama-3.3-70b-versatile" }]);
    expect(u.inputTokens).toBe(210);
    expect(u.outputTokens).toBe(88);
    // a non-final chunk (no usage anywhere) must NOT latch
    expect(
      _chunkHasUsage("groq", {
        choices: [{ index: 0, delta: { content: "hi" } }],
      }),
    ).toBe(false);
  });

  // ── #15: groq present in ALL THREE stream-composition sets. Missing the
  // third (_streamAccumulatorToResponse) would return null → composition never
  // builds. ──
  it("groq is wired into all three stream-composition stages", () => {
    // 1) accumulator is created (not null)
    const acc = _newStreamAccumulator("groq");
    expect(acc).not.toBeNull();
    expect(acc).toHaveProperty("textParts");
    // 2) chunks fold in
    _accumulateStreamChunk("groq", acc, {
      choices: [{ index: 0, delta: { content: "Hello " } }],
    });
    _accumulateStreamChunk("groq", acc, {
      choices: [{ index: 0, delta: { content: "world" } }],
    });
    // 3) composition builds a non-null OpenAI-shaped response
    const composed = _streamAccumulatorToResponse("groq", acc);
    expect(composed).not.toBeNull();
    expect(composed.choices[0].message.content).toBe("Hello world");
  });

  // ── #19 Vercel-path guard: the native _detectProvider groq branch cannot
  // hijack the Vercel @ai-sdk/groq path. That path derives its provider from
  // `model.provider` via _mapAiSdkProvider (a disjoint code path from
  // _detectProvider) and parses under the "ai_sdk" key. Native groq ("groq")
  // is OpenAI-shaped, NOT an ai_sdk parse shape. ──
  it("Vercel @ai-sdk/groq path resolves via _mapAiSdkProvider, unchanged", () => {
    // Vercel groq model.provider heads still map to "groq" (unchanged).
    expect(_mapAiSdkProvider("groq.chat")).toBe("groq");
    expect(_mapAiSdkProvider("groq")).toBe("groq");
    // native groq is OpenAI-shaped — NOT treated as an ai_sdk parse shape.
    expect(_isAiSdkParse("groq")).toBe(false);
    // the disjoint internal parse keys are unchanged.
    expect(_isAiSdkParse("ai_sdk")).toBe(true);
    expect(_isAiSdkParse("xai")).toBe(true);
  });

  // ── #19 regression: cerebras/together/openai unchanged by the groq additions. ──
  it("no regression — cerebras/together/openai still meter identically", () => {
    const resp = {
      model: "m",
      usage: { prompt_tokens: 7, completion_tokens: 3 },
    };
    for (const p of ["cerebras", "together", "openai", "litellm"]) {
      const u = _extractUsage(p, resp, [{ model: "m" }]);
      expect([u.inputTokens, u.outputTokens]).toEqual([7, 3]);
    }
    // cerebras/together top-level usage still detected on a stream chunk
    expect(_chunkHasUsage("together", { usage: { prompt_tokens: 1 } })).toBe(true);
    // and these providers do NOT gain an x_groq path (chunk without usage = false)
    expect(_chunkHasUsage("together", { x_groq: { usage: {} } })).toBe(false);
  });
});
