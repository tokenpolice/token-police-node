/**
 * Node telemetry provider normalization: Mistral & Together parity.
 *
 * The provider-normalization chain in `TokenPoliceSpanProcessor.onEnd`
 * (src/telemetry.ts ~:794-802) lowercases `gen_ai.system` and special-cases
 * gemini/google, bedrock/aws, mistral and together, then passes everything else
 * through verbatim. This locks the two arms that bring Node to parity with
 * Python's reference chain (token_police/telemetry.py:515-529):
 * "MistralAI"/"mistralai"/"mistral" -> "mistral" (was "mistralai" pre-fix)
 * "TogetherAI"/"togetherai"/"together" -> "together" (was "togetherai" pre-fix)
 *
 * Mechanism (mirrors tests/sdkUsageParity.test.ts): drive a synthetic finished
 * span through onEnd, capture the args handed to client.log via vi.mock of
 * ../src/state. provider is positional arg index 5 (client.ts:344-350); the
 * usage block is the last positional arg's `.usage` field.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

// Capture the kwargs the processor hands to the client's log().
const logged: any[] = [];
vi.mock("../src/state", () => ({
  getClient: () => ({
    log: (...args: any[]) => {
      logged.push(args);
    },
  }),
  drainObservations: () => [],
}));

import { TokenPoliceSpanProcessor } from "../src/telemetry";

// Synthetic finished LLM span. Nonzero usage so the usage>0 gate
// (telemetry.ts:739-740) passes and the row is actually logged.
function fakeSpan(genAiSystem: string) {
  return {
    attributes: {
      "gen_ai.system": genAiSystem,
      "gen_ai.request.model": "some-model",
      "gen_ai.usage.input_tokens": 10,
      "gen_ai.usage.output_tokens": 5,
    },
    name: "llm.chat",
    startTime: [0, 0] as [number, number],
    endTime: [1, 0] as [number, number],
    spanContext: () => ({
      traceId: "0000000000000000000000000000abcd",
      spanId: "0000000000001234",
    }),
    parentSpanId: undefined,
  } as any;
}

// Drive one span through onEnd and return the captured log() args.
async function runProvider(genAiSystem: string): Promise<any[]> {
  const proc = new TokenPoliceSpanProcessor();
  proc.onEnd(fakeSpan(genAiSystem));
  // The log is deferred to process.nextTick — let it run.
  await new Promise((r) => setTimeout(r, 0));
  expect(logged.length).toBe(1);
  return logged[0];
}

// provider is positional arg index 5 (client.ts:344-350).
const PROVIDER_IDX = 5;

describe("Provider normalization — Mistral & Together parity", () => {
  beforeEach(() => {
    logged.length = 0;
  });

  // ── Assertion 1 + 14 (RED/GREEN guard): the real-bug input. ──
  // Pre-fix (only gemini/google + bedrock/aws arms) this span falls through to
  // `else provider = _pl` and logs "mistralai", so this assertion is RED on
  // unfixed code; the two new arms make it log "mistral" (GREEN).
  it("MistralAI normalizes to mistral", async () => {
    const args = await runProvider("MistralAI");
    expect(args[PROVIDER_IDX]).toBe("mistral"); // pre-fix value was "mistralai"
  });

  // ── Assertion 2: lowercase + bare forms → "mistral". ──
  it("mistralai and mistral normalize to mistral", async () => {
    expect((await runProvider("mistralai"))[PROVIDER_IDX]).toBe("mistral");
    logged.length = 0;
    expect((await runProvider("mistral"))[PROVIDER_IDX]).toBe("mistral");
  });

  // ── Assertion 3: mixed-case / substring Mistral → "mistral". ──
  it("mixed-case and substring Mistral normalize to mistral", async () => {
    expect((await runProvider("MISTRAL"))[PROVIDER_IDX]).toBe("mistral");
    logged.length = 0;
    expect((await runProvider("mistral-large"))[PROVIDER_IDX]).toBe("mistral");
  });

  // ── Assertion 4 + 14 (TogetherAI RED/GREEN): real-bug parity input. ──
  // Pre-fix logs "togetherai"; the new arm makes it "together".
  it("TogetherAI normalizes to together", async () => {
    const args = await runProvider("TogetherAI");
    expect(args[PROVIDER_IDX]).toBe("together"); // pre-fix value was "togetherai"
  });

  // ── Assertion 5: lowercase + bare forms → "together". ──
  it("togetherai and together normalize to together", async () => {
    expect((await runProvider("togetherai"))[PROVIDER_IDX]).toBe("together");
    logged.length = 0;
    expect((await runProvider("together"))[PROVIDER_IDX]).toBe("together");
  });

  // ── Assertion 6: produced slugs equal the canonical shared contract AND ──
  // Python's output. Python telemetry.py:524-529 yields exactly these literals
  // for the same inputs; the published provider-slug table canonical slugs are
  // "mistral" (line 33) and "together" (line 51).
  it("emits the canonical slugs Python produces (cross-SDK parity)", async () => {
    expect((await runProvider("MistralAI"))[PROVIDER_IDX]).toBe("mistral");
    logged.length = 0;
    expect((await runProvider("TogetherAI"))[PROVIDER_IDX]).toBe("together");
  });

  // ── Assertion 7: gemini/google arm still wins (precedence intact). ──
  it("gemini/google arm precedes the new arms", async () => {
    expect((await runProvider("google_genai"))[PROVIDER_IDX]).toBe("google");
    logged.length = 0;
    // crafted edge: contains both "google" and "mistral" — google wins (first arm)
    expect((await runProvider("google-mistral"))[PROVIDER_IDX]).toBe("google");
  });

  // ── Assertion 8: bedrock/aws arm still resolves before mistral/together. ──
  it("bedrock/aws arm precedes the new arms", async () => {
    expect((await runProvider("aws"))[PROVIDER_IDX]).toBe("bedrock");
    logged.length = 0;
    expect((await runProvider("AmazonBedrock"))[PROVIDER_IDX]).toBe("bedrock");
  });

  // ── Assertion 9: regression guard — every existing branch unchanged. ──
  it("leaves every existing (non-mistral/non-together) branch unchanged", async () => {
    const cases: Array<[string, string]> = [
      ["openai", "openai"],
      ["Anthropic", "anthropic"],
      ["Google", "google"],
      ["gemini", "google"],
      ["bedrock", "bedrock"],
      ["Cohere", "cohere"], // final-else lowercase passthrough
      ["cohere", "cohere"],
    ];
    for (const [input, expected] of cases) {
      logged.length = 0;
      expect((await runProvider(input))[PROVIDER_IDX]).toBe(expected);
    }
  });

  // ── Assertion 15: usage.shape UNCHANGED for the mistral/together re-key. ──
  // The usageShape switch (telemetry.ts:1028-1039) has no mistral/together case,
  // so both fall through to default → "openai_compatible_chat" — identical to
  // the pre-fix "mistralai"/"togetherai" shape. The usage block is the last
  // positional arg's `.usage` field (same accessor sdkUsageParity.test.ts uses).
  it("keeps usage.shape openai_compatible_chat for mistral and together", async () => {
    const mistralArgs = await runProvider("MistralAI");
    expect(mistralArgs[mistralArgs.length - 1].usage.shape).toBe(
      "openai_compatible_chat",
    );
    logged.length = 0;
    const togetherArgs = await runProvider("TogetherAI");
    expect(togetherArgs[togetherArgs.length - 1].usage.shape).toBe(
      "openai_compatible_chat",
    );
  });
});
