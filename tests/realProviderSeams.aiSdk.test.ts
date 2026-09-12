/**
 * REAL-PACKAGE seam contract — `ai` (Vercel AI SDK) + `@ai-sdk/*` providers.
 *
 * See tests/helpers/realProviderSeam.ts for why this family exists.
 *
 * The AI SDK seam is structural, not name-based: the enforcer constructs a
 * throw-away probe model from each provider factory and patches the
 * `doGenerate` / `doStream` / `doEmbed` methods on whatever prototype that
 * probe lands on. Nothing in that chain is a documented public API, so it can
 * move at any minor release:
 *   - the exported factory name (`createOpenAI` / `openai`),
 *   - the entry points the probe tries (`provider(id)`, `.languageModel()`,
 *     `.chat()`, `.responses()`),
 *   - the LanguageModel spec method names,
 *   - the `usage` shape, which has already changed generations (V2 flat
 *     numbers → V3 nested `{ total, cacheRead, ... }` detail objects).
 *
 * The usage shape is covered BOTH ways here regardless of which generation is
 * installed: the flat and nested forms are driven through
 * `tokenPoliceAiSdkMiddleware()` — the exact same `_aiSdkRunGenerate` /
 * `_aiSdkRunStream` code the prototype patch calls — with a REAL model instance
 * supplying the model id and provider head. `Number(nestedObject)` is NaN, so a
 * branch that stops detecting the nested form does not under-report, it reports
 * zero.
 */
import {
  describe,
  it,
  expect,
  beforeAll,
  afterAll,
  beforeEach,
  afterEach,
} from "vitest";

import {
  autoInstrument,
  tokenPoliceAiSdkMiddleware,
  uninstrument,
} from "../src/enforcer";
import {
  createSeamHarness,
  declaredDevDependency,
  flushLogs,
  installNoNetworkGuard,
  installedVersion,
  jsonResponder,
  requireCjs,
  sseResponder,
} from "./helpers/realProviderSeam";

const AI_VERSION = installedVersion("ai");
const OPENAI_PROVIDER_VERSION = installedVersion("@ai-sdk/openai");
const ANTHROPIC_PROVIDER_VERSION = installedVersion("@ai-sdk/anthropic");

// Loaded at MODULE scope, not in beforeAll. These are ESM-only packages, so the
// import has to be dynamic — but doing it in a hook forced the gate to be
// `if (!available) return`, which reports a load failure as a PASSING test. The
// whole point of this family is to break that kind of silence, so: top-level
// await, a real `it.skip` when the package is absent, and an explicit failure
// when it is a declared devDependency that would not load.
const loaded = await (async () => {
  try {
    return {
      ai: await import("ai"),
      openai: await import("@ai-sdk/openai"),
      anthropic: await import("@ai-sdk/anthropic"),
      error: null as unknown,
    };
  } catch (err) {
    return { ai: null, openai: null, anthropic: null, error: err };
  }
})();
const ai: any = loaded.ai;
const aiSdkOpenAI: any = loaded.openai;
const aiSdkAnthropic: any = loaded.anthropic;

const h = createSeamHarness();
let restoreNetwork: () => void;

beforeAll(() => {
  restoreNetwork = installNoNetworkGuard();
  console.log(
    `[realProviderSeams] ai@${AI_VERSION ?? "ABSENT"} ` +
      `@ai-sdk/openai@${OPENAI_PROVIDER_VERSION ?? "ABSENT"} ` +
      `@ai-sdk/anthropic@${ANTHROPIC_PROVIDER_VERSION ?? "ABSENT"}`,
  );
});
afterAll(() => restoreNetwork());

beforeEach(() => h.install());
afterEach(async () => {
  await flushLogs(50);
  uninstrument();
  h.reset();
});

/** Skip cleanly when the optional peer deps aren't installed. */
const maybe = loaded.ai ? it : it.skip;

// A declared devDependency that will not import is a HARD failure, never a
// skip — otherwise a broken package silently empties this whole file.
const declared = ["ai", "@ai-sdk/openai", "@ai-sdk/anthropic"].filter(
  declaredDevDependency,
);
(declared.length ? it : it.skip)(
  "every declared @ai-sdk devDependency imports",
  () => {
    expect(
      loaded.error,
      `[${declared.join(", ")}] are declared devDependencies but the import ` +
        `threw: ${loaded.error instanceof Error ? loaded.error.message : String(loaded.error)}`,
    ).toBeNull();
  },
);

/** An OpenAI Responses-API body — what @ai-sdk/openai's default model speaks. */
const RESPONSES_BODY = (usage: Record<string, unknown>) => ({
  id: "resp_aisdk",
  object: "response",
  status: "completed",
  created_at: 1,
  model: "gpt-4o-mini",
  output: [
    {
      type: "message",
      id: "m1",
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text: "hi", annotations: [] }],
    },
  ],
  usage,
});

// ═══════════════════════════════════════════════════════════════════════
// Layer 1 — SHAPE
// ═══════════════════════════════════════════════════════════════════════
describe("ai / @ai-sdk — shape: the probe chain the enforcer relies on", () => {
  maybe("the provider factories the registry names are still exported", () => {
    // _AI_SDK_REGISTRY tries these names in order; losing both forms for a
    // package silently drops that provider from auto-instrumentation.
    expect(typeof aiSdkOpenAI.createOpenAI === "function" || typeof aiSdkOpenAI.openai === "function")
      .toBe(true);
    expect(
      typeof aiSdkAnthropic.createAnthropic === "function" ||
        typeof aiSdkAnthropic.anthropic === "function",
    ).toBe(true);
  });

  maybe("a probe model exposes the spec methods the patch targets", () => {
    const provider = aiSdkOpenAI.createOpenAI({ apiKey: "sk-test-not-a-real-key" });
    const model: any = provider("gpt-4o-mini");
    expect(typeof model.doGenerate).toBe("function");
    expect(typeof model.doStream).toBe("function");
    // The reported model id and provider head are read off the instance by
    // `_aiSdkCallMeta`; losing either turns every row's model into "unknown".
    expect(model.modelId).toBe("gpt-4o-mini");
    expect(String(model.provider)).toMatch(/^openai/);
    // The spec generation this run is testing, recorded for the drift report.
    expect(String(model.specificationVersion)).toMatch(/^v\d+$/);
  });

  maybe("the alternate probe entry points still return models", () => {
    const provider: any = aiSdkOpenAI.createOpenAI({ apiKey: "sk-test-not-a-real-key" });
    // `_collectAiSdkProbeModels` tries all of these because providers expose
    // DIFFERENT (and, across versions, differently-defaulted) model classes per
    // entry point. Losing one silently leaves that class unpatched.
    for (const entry of ["languageModel", "chat", "responses"]) {
      if (typeof provider[entry] !== "function") continue;
      const m = provider[entry]("gpt-4o-mini");
      expect(typeof m?.doGenerate, `${entry}() model`).toBe("function");
    }
  });

  maybe("autoInstrument() patches the real provider's model prototypes", () => {
    const provider = aiSdkOpenAI.createOpenAI({ apiKey: "sk-test-not-a-real-key" });
    const proto = Object.getPrototypeOf(provider("gpt-4o-mini"));
    const before = { gen: proto.doGenerate, stream: proto.doStream };

    autoInstrument({ aiSdkProviders: [aiSdkOpenAI] } as any);

    expect(
      proto.doGenerate,
      "doGenerate was NOT wrapped — every generateText() call through this " +
        "provider runs with no pre-flight /check and produces no row",
    ).not.toBe(before.gen);
    expect(proto.doStream).not.toBe(before.stream);
    expect((proto.doGenerate as any).__tp_aisdk_wrapped).toBe(true);
  });

  maybe("the exported middleware satisfies the LanguageModel middleware contract", () => {
    const mw = tokenPoliceAiSdkMiddleware();
    expect(typeof mw.wrapGenerate).toBe("function");
    expect(typeof mw.wrapStream).toBe("function");
    // `wrapLanguageModel` is the documented escape hatch for bundled/edge
    // providers; it must still accept our structural middleware object.
    expect(typeof ai.wrapLanguageModel).toBe("function");
    const wrapped = ai.wrapLanguageModel({
      model: aiSdkOpenAI.createOpenAI({ apiKey: "sk-test-not-a-real-key" })("gpt-4o-mini"),
      middleware: mw,
    });
    expect(typeof wrapped.doGenerate).toBe("function");
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Layer 2 — ROUND TRIP
// ═══════════════════════════════════════════════════════════════════════
describe("ai / @ai-sdk — round trip: generateText / streamText", () => {
  maybe("generateText reports the provider's exact tokens and model id", async () => {
    autoInstrument();
    const provider = aiSdkOpenAI.createOpenAI({
      apiKey: "sk-test-not-a-real-key",
      fetch: jsonResponder(
        RESPONSES_BODY({
          input_tokens: 41,
          output_tokens: 3,
          total_tokens: 44,
          input_tokens_details: { cached_tokens: 0 },
          output_tokens_details: { reasoning_tokens: 0 },
        }),
      ),
    });
    await ai.generateText({ model: provider("gpt-4o-mini"), prompt: "hello" });
    await flushLogs(40);

    const row = h.only((l) => l.extra?.usage?.shape === "vercel_ai", "ai sdk generate");
    expect(row.model).toBe("gpt-4o-mini");
    expect(row.inputTokens).toBe(41);
    expect(row.outputTokens).toBe(3);
    // The reported provider is the mapped head of `model.provider`, not the
    // literal "ai_sdk" — that is how a Gateway/Bedrock-routed call gets
    // attributed to its real vendor.
    expect(row.provider).toBe("openai");
  });

  maybe("streamText reports usage from the terminal finish part", async () => {
    autoInstrument();
    const provider = aiSdkOpenAI.createOpenAI({
      apiKey: "sk-test-not-a-real-key",
      // No `response.created` frame: the provider's zod schema for it is
      // strict and an incomplete one only produces validation noise — the
      // parts below are what actually carry text and usage.
      fetch: sseResponder([
        {
          type: "response.output_item.added",
          output_index: 0,
          item: { type: "message", id: "m1", role: "assistant", status: "in_progress", content: [] },
        },
        {
          type: "response.content_part.added",
          item_id: "m1",
          output_index: 0,
          content_index: 0,
          part: { type: "output_text", text: "", annotations: [] },
        },
        {
          type: "response.output_text.delta",
          item_id: "m1",
          output_index: 0,
          content_index: 0,
          delta: "hi",
        },
        {
          type: "response.completed",
          response: RESPONSES_BODY({
            input_tokens: 52,
            output_tokens: 8,
            total_tokens: 60,
            input_tokens_details: { cached_tokens: 0 },
            output_tokens_details: { reasoning_tokens: 0 },
          }),
        },
      ]),
    });
    const result: any = ai.streamText({ model: provider("gpt-4o-mini"), prompt: "hello" });
    for await (const _ of result.textStream) {
      /* drain */
    }
    await flushLogs(80);

    const row = h.only((l) => l.extra?.usage?.shape === "vercel_ai", "ai sdk stream");
    expect(row.model).toBe("gpt-4o-mini");
    expect(row.inputTokens).toBe(52);
    expect(row.outputTokens).toBe(8);
  });

  maybe("a second @ai-sdk provider attributes to its own vendor head", async () => {
    autoInstrument();
    const provider = aiSdkAnthropic.createAnthropic({
      apiKey: "sk-ant-test-not-a-real-key",
      fetch: jsonResponder({
        id: "msg_aisdk",
        type: "message",
        role: "assistant",
        model: "claude-sonnet-4-5",
        content: [{ type: "text", text: "hi" }],
        stop_reason: "end_turn",
        stop_sequence: null,
        usage: { input_tokens: 26, output_tokens: 5 },
      }),
    });
    await ai.generateText({ model: provider("claude-sonnet-4-5"), prompt: "hello" });
    await flushLogs(40);

    const row = h.only((l) => l.extra?.usage?.shape === "vercel_ai", "ai sdk anthropic");
    expect(row.model).toBe("claude-sonnet-4-5");
    expect(row.provider).toBe("anthropic");
    expect(row.inputTokens).toBe(26);
    expect(row.outputTokens).toBe(5);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Layer 2b — BOTH usage generations, through the real middleware entry point.
//
// Whichever spec version is installed, only ONE of these two shapes can be
// produced by a real call. Driving both through `tokenPoliceAiSdkMiddleware()`
// — the same `_aiSdkRunGenerate` the prototype patch calls, with a REAL model
// instance for the meta — keeps the other generation covered.
// ═══════════════════════════════════════════════════════════════════════
describe("ai / @ai-sdk — usage shapes: V2/V3-flat and V3-nested", () => {
  function realModel(): any {
    return aiSdkOpenAI.createOpenAI({ apiKey: "sk-test-not-a-real-key" })("gpt-4o-mini");
  }

  maybe("flat usage (ai v5 'v2' spec): inputTokens/outputTokens as numbers", async () => {
    const mw = tokenPoliceAiSdkMiddleware();
    await mw.wrapGenerate({
      model: realModel(),
      params: { prompt: "hello" },
      doGenerate: async () => ({
        content: [{ type: "text", text: "hi" }],
        finishReason: "stop",
        usage: { inputTokens: 120, outputTokens: 30, totalTokens: 150, cachedInputTokens: 20 },
        warnings: [],
      }),
    });
    await flushLogs(30);

    const row = h.only((l) => l.extra?.usage?.shape === "vercel_ai", "flat usage");
    expect(row.model).toBe("gpt-4o-mini");
    expect(row.cachedTokens).toBe(20);
    expect(row.inputTokens).toBe(100); // 120 reported − 20 cached
    expect(row.outputTokens).toBe(30);
  });

  maybe("nested usage (ai v6 'v3' spec): inputTokens/outputTokens as objects", async () => {
    const mw = tokenPoliceAiSdkMiddleware();
    await mw.wrapGenerate({
      model: realModel(),
      params: { prompt: "hello" },
      doGenerate: async () => ({
        content: [{ type: "text", text: "hi" }],
        finishReason: "stop",
        usage: {
          inputTokens: { total: 120, noCache: 100, cacheRead: 20, cacheWrite: 0 },
          outputTokens: { total: 30, text: 25, reasoning: 5 },
        },
        warnings: [],
      }),
    });
    await flushLogs(30);

    const row = h.only((l) => l.extra?.usage?.shape === "vercel_ai", "nested usage");
    // `Number({total: 120})` is NaN → a branch that stops detecting the nested
    // form reports 0, not a slightly-wrong number. These exact values are the
    // guard against that silent-zero mode.
    expect(row.inputTokens).toBe(100);
    expect(row.cachedTokens).toBe(20);
    expect(row.outputTokens).toBe(30);
  });
});
