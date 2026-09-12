/**
 * REAL-PACKAGE seam contract — `openai`.
 *
 * See tests/helpers/realProviderSeam.ts for why this family exists: every other
 * Node openai test drives a hand-built fake of the Stainless export shape, and
 * a fake cannot drift. This file loads the package a customer would install and
 * asserts, without a network or an API key, that the enforcer's instrumentation
 * still attaches and still reports the provider's own token counts.
 *
 * Layer 1 (shape) is driven from `__test__._TARGET_METHODS`, so it guards
 * exactly the rows `src/enforcer.ts` declares — no hand-copied list to go stale.
 *
 * Layer 2 (round trip) covers both telemetry modes openai uses, because they
 * break independently:
 *   - Mode A (Chat Completions): usage arrives via the Traceloop OpenAI
 *     instrumentor's span, read by `TokenPoliceSpanProcessor.onEnd`.
 *   - Mode C manual (Responses / Embeddings / Images / Audio): the enforcer
 *     wrapper reads `response.usage` itself and calls `tp.log` directly.
 *
 * NOTE ON AUTO-DISCOVERY: `@traceloop/instrumentation-openai` declares
 * `supportedVersions` for the `openai` module, and the installed openai may sit
 * OUTSIDE that range — in which case the OTel loader hook silently declines to
 * patch and only the explicit `instrumentModules` path meters Chat Completions.
 * That cap is a standing, separately-tracked signal (see
 * `warnIfInstrumentorSkippedModule` + tests/unsupportedVersionWarning.test.ts
 * and the `traceloop-instrumentors` cell in provider-drift-suite/catalog.json).
 * The Mode-A round trip below therefore uses the documented `instrumentModules`
 * form, which is version-gate-independent, so this file tests OUR seam rather
 * than re-reporting that cap.
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
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { trace, context, propagation } from "@opentelemetry/api";

import { _extractServiceTier, autoInstrument, uninstrument } from "../src/enforcer";
import { setupOpenTelemetry, unsetupOpenTelemetry } from "../src/telemetry";
import {
  createSeamHarness,
  flushLogs,
  installNoNetworkGuard,
  installedVersion,
  jsonResponder,
  methodNames,
  requireCjs,
  snapshotMethods,
  sseResponder,
  targetLabel,
  targetsFor,
  walkPath,
} from "./helpers/realProviderSeam";

/** Cross-language contract shared with the Python SDK — see sdkUsageParity.test.ts. */
const parityFixture = JSON.parse(
  readFileSync(
    join(dirname(fileURLToPath(import.meta.url)), "../../shared/sdk-usage-parity-fixture.json"),
    "utf8",
  ),
);

const PKG = "openai";
const VERSION = installedVersion(PKG);
const openaiModule = (() => {
  try {
    return requireCjs(PKG);
  } catch {
    return undefined;
  }
})();
// `openai` is a devDependency here; skip cleanly rather than fail if a future
// change drops it, exactly like the optionalDependency gates elsewhere in
// tests/.
const maybe = openaiModule ? it : it.skip;

/** What `_pickClassExport(mod, "OpenAI")` resolves — the class the walk starts from. */
const OpenAI: any = openaiModule?.OpenAI ?? openaiModule?.default;

const h = createSeamHarness();
let restoreNetwork: () => void;

beforeAll(() => {
  restoreNetwork = installNoNetworkGuard();
  // Recorded in the run log so a red cell in provider-drift-suite says which
  // release it ran against without anyone re-deriving it.
  console.log(`[realProviderSeams] openai@${VERSION ?? "ABSENT"}`);
});
afterAll(() => restoreNetwork());

beforeEach(() => h.install());
afterEach(async () => {
  // Drain first: the span processor logs on a deferred tick, so a row produced
  // by THIS test must be allowed to land here rather than leaking into the next
  // test's captures. (`unsetupOpenTelemetry` cannot be relied on for isolation
  // — the enforcer's internal teardown require is a no-op under vitest.)
  await flushLogs(150);
  try {
    unsetupOpenTelemetry();
    trace.disable();
    context.disable();
    propagation.disable();
  } catch {
    /* best effort */
  }
  uninstrument();
  h.reset();
});

// ═══════════════════════════════════════════════════════════════════════
// Layer 1 — SHAPE. Every `openai` row in the enforcer's own registry.
// ═══════════════════════════════════════════════════════════════════════
describe("openai — shape: the enforcer's patch targets exist on the real package", () => {
  const targets = targetsFor(PKG);

  it("the registry still declares openai rows (guards an accidental deletion)", () => {
    // Chat Completions, legacy Completions, Responses, Images, 3x Audio,
    // Embeddings. A shrinking registry is itself a regression worth failing on.
    expect(targets.length).toBeGreaterThanOrEqual(8);
  });

  maybe("the package root exposes the OpenAI class _pickClassExport resolves", () => {
    expect(typeof OpenAI).toBe("function");
  });

  maybe("APIPromise is exported at the package root", () => {
    // The Anthropic wrapper feature-detects `mod.APIPromise` to decide whether
    // the Traceloop streaming branch is safe; openai's own `.withResponse()` /
    // `.asResponse()` surface rides the same class. A root-level rename here is
    // a drift signal for both.
    expect(typeof openaiModule.APIPromise).toBe("function");
  });

  for (const t of targets) {
    maybe(`${targetLabel(t)} resolves and is a function`, () => {
      const { obj, brokeAt } = walkPath(openaiModule, t.objectPath);
      expect(
        obj,
        `objectPath [${t.objectPath.join(", ")}] broke at "${brokeAt}" — the ` +
          `enforcer would log "module loaded but objectPath did not resolve" ` +
          `and this seam would silently stop enforcing`,
      ).toBeTruthy();
      expect(
        typeof obj[t.method],
        `method "${t.method}" is gone from ${t.objectPath.join(".")}; the ` +
          `object still carries: ${methodNames(obj).join(", ")}`,
      ).toBe("function");
    });
  }

  maybe("autoInstrument() actually installs a wrapper on every openai target", () => {
    // Identity change is the only honest proof the patch landed: `_wrapMethod`
    // fails SILENTLY (a logErrors-gated warn) when a path or method has moved,
    // so "still a function afterwards" would pass on a total install failure.
    const before = snapshotMethods(openaiModule, targets);
    autoInstrument();
    for (const t of targets) {
      const { obj } = walkPath(openaiModule, t.objectPath);
      expect(
        obj?.[t.method],
        `${targetLabel(t)} was NOT wrapped — the enforcer left the vendor ` +
          `method in place, so this seam runs with no pre-flight /check`,
      ).not.toBe(before.get(targetLabel(t)));
    }
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Layer 2 — ROUND TRIP. Real client, canned wire bytes, known token counts.
// ═══════════════════════════════════════════════════════════════════════

/** Real client whose ONLY transport is the supplied stub. No network, no key. */
function makeClient(fetchStub: (...a: any[]) => Promise<Response>): any {
  return new OpenAI({ apiKey: "sk-test-not-a-real-key", fetch: fetchStub });
}

const CHAT_BODY = (extra: Record<string, unknown> = {}) => ({
  id: "chatcmpl-seam",
  object: "chat.completion",
  created: 1,
  model: "gpt-4o-mini",
  choices: [
    { index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" },
  ],
  usage: { prompt_tokens: 21, completion_tokens: 5, total_tokens: 26 },
  ...extra,
});

describe("openai — round trip: Chat Completions (Mode A, Traceloop span)", () => {
  maybe("non-stream reports the provider's exact prompt/completion counts", async () => {
    setupOpenTelemetry({ openAI: OpenAI } as any);
    autoInstrument({ openAI: OpenAI } as any);

    await makeClient(jsonResponder(CHAT_BODY())).chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hello" }],
    });
    await flushLogs(120);

    const row = h.only((l) => l.provider === "openai", "openai chat");
    expect(row.model).toBe("gpt-4o-mini");
    expect(row.inputTokens).toBe(21);
    expect(row.outputTokens).toBe(5);
    expect(row.extra?.usage?.shape).toBe("openai_chat");
  });

  maybe("service_tier survives the real client's response parser", async () => {
    // The tier drives batch/flex/priority pricing, so the field has to survive
    // BOTH the wire and the vendor's response model. Asserted on the parsed
    // object the enforcer actually sees, and canonicalized with the
    // cross-language table in shared/sdk-usage-parity-fixture.json so Node and
    // Python cannot disagree about what "flex" or "scale" means.
    const cases = (parityFixture as any).service_tier_cases as Record<string, string>;

    for (const [wire, canonical] of Object.entries(cases)) {
      const parsed: any = await makeClient(
        jsonResponder(CHAT_BODY({ service_tier: wire })),
      ).chat.completions.create({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "hello" }],
      });
      expect(parsed.service_tier, `wire tier "${wire}" was dropped by the parser`).toBe(
        wire,
      );
      expect(_extractServiceTier(parsed), `wire tier "${wire}"`).toBe(canonical);
    }
  });

  maybe("stream reports usage from the final usage-bearing chunk", async () => {
    setupOpenTelemetry({ openAI: OpenAI } as any);
    autoInstrument({ openAI: OpenAI } as any);

    const chunk = (o: Record<string, unknown>) => ({
      id: "chatcmpl-seam",
      object: "chat.completion.chunk",
      created: 1,
      model: "gpt-4o-mini",
      ...o,
    });
    const stream: any = await makeClient(
      sseResponder([
        chunk({ choices: [{ index: 0, delta: { role: "assistant", content: "hi" } }] }),
        chunk({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }),
        chunk({
          choices: [],
          usage: { prompt_tokens: 33, completion_tokens: 7, total_tokens: 40 },
        }),
      ]),
    ).chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hello" }],
      stream: true,
      stream_options: { include_usage: true },
    });
    for await (const _ of stream) {
      /* drain */
    }
    await flushLogs(150);

    const row = h.only((l) => l.provider === "openai", "openai chat stream");
    expect(row.model).toBe("gpt-4o-mini");
    expect(row.inputTokens).toBe(33);
    expect(row.outputTokens).toBe(7);
  });
});

describe("openai — round trip: Responses API (Mode C manual)", () => {
  const RESP = (usage: Record<string, unknown>, extra: Record<string, unknown> = {}) => ({
    id: "resp_seam",
    object: "response",
    status: "completed",
    created_at: 1,
    model: "gpt-5-mini",
    output: [
      {
        type: "message",
        id: "msg_1",
        role: "assistant",
        status: "completed",
        content: [{ type: "output_text", text: "hi", annotations: [] }],
      },
    ],
    usage,
    ...extra,
  });

  maybe("non-stream: cached tokens are subtracted from input, never double-counted", async () => {
    autoInstrument();
    await makeClient(
      jsonResponder(
        RESP({
          input_tokens: 11,
          output_tokens: 7,
          total_tokens: 18,
          input_tokens_details: { cached_tokens: 3 },
        }),
      ),
    ).responses.create({ model: "gpt-5-mini", input: "hello" });
    await flushLogs();

    const row = h.only((l) => l.provider === "openai_responses", "responses");
    expect(row.model).toBe("gpt-5-mini");
    // 11 reported, 3 of them cached → 8 fresh input + 3 cached. Their sum must
    // still be the provider's 11; a shape change that stops finding
    // `input_tokens_details` would show up as 11/0 here.
    expect(row.inputTokens).toBe(8);
    expect(row.cachedTokens).toBe(3);
    expect(row.inputTokens + row.cachedTokens).toBe(11);
    expect(row.outputTokens).toBe(7);
    expect(row.extra?.usage?.shape).toBe("openai_responses");
    expect(row.extra?.usage?.raw?.input_tokens).toBe(11);
  });

  maybe("stream: usage rides the terminal response.completed event", async () => {
    autoInstrument();
    const stream: any = await makeClient(
      sseResponder([
        { type: "response.created", response: { id: "r", model: "gpt-5-mini", status: "in_progress" } },
        { type: "response.output_text.delta", delta: "hi" },
        {
          type: "response.completed",
          response: RESP({
            input_tokens: 13,
            output_tokens: 4,
            total_tokens: 17,
            input_tokens_details: { cached_tokens: 0 },
          }),
        },
      ]),
    ).responses.create({ model: "gpt-5-mini", input: "hello", stream: true });
    for await (const _ of stream) {
      /* drain */
    }
    await flushLogs(50);

    const row = h.only((l) => l.provider === "openai_responses", "responses stream");
    expect(row.inputTokens).toBe(13);
    expect(row.outputTokens).toBe(4);
    expect(row.extra?.latency?.is_streaming).toBe(true);
  });
});

describe("openai — round trip: Embeddings + Images (Mode C manual)", () => {
  maybe("embeddings.create logs operation=embedding with the prompt count", async () => {
    autoInstrument();
    await makeClient(
      jsonResponder({
        object: "list",
        model: "text-embedding-3-small",
        data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2] }],
        usage: { prompt_tokens: 9, total_tokens: 9 },
      }),
    ).embeddings.create({ model: "text-embedding-3-small", input: "hello" });
    await flushLogs();

    const row = h.only((l) => l.extra?.operation === "embedding", "embedding");
    expect(row.model).toBe("text-embedding-3-small");
    expect(row.inputTokens).toBe(9);
    expect(row.outputTokens).toBe(0);
    expect(row.extra?.usage?.shape).toBe("openai_embeddings");
  });

  maybe("images.generate logs the image_gen modality shape with an item count", async () => {
    autoInstrument();
    await makeClient(
      jsonResponder({
        created: 1,
        data: [{ b64_json: "aGk=" }, { b64_json: "aGk=" }],
        usage: { input_tokens: 12, output_tokens: 0, total_tokens: 12 },
      }),
    ).images.generate({ model: "gpt-image-1", prompt: "a cat", size: "1024x1024", n: 2 });
    await flushLogs();

    const row = h.only((l) => l.extra?.usage?.shape === "openai_images", "image");
    expect(row.model).toBe("gpt-image-1");
    expect(row.extra?.operation).toBe("image_gen");
    expect(row.extra?.usage?.items?.images_generated).toBe(2);
    expect(row.extra?.usage?.items?.image_size).toBe("1024x1024");
  });
});
