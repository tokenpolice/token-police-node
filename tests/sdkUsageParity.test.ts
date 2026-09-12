/**
 * Node↔Python SDK usage-key parity (G1/G5).
 *
 * Both SDK SpanProcessors reconstruct a `usage` block from the same `gen_ai.*`
 * semconv attributes (Mode-A). They MUST emit the same `usage.raw` keys/values
 * so the server's usage-mapper sees one shape regardless of SDK. The shared
 * fixture is the single source of truth; the Python counterpart
 * (tests/test_sdk_usage_parity.py) asserts the identical contract.
 *
 * Specifically locks G5: cache *read* and cache *creation* (write) stay DISJOINT,
 * and reasoning tokens are forwarded.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const fixture = JSON.parse(
  readFileSync(join(__dirname, "../../shared/sdk-usage-parity-fixture.json"), "utf8"),
);

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
import { _extractServiceTier, _stashRequestServiceTier } from "../src/enforcer";

function fakeSpan(attrs: Record<string, unknown>) {
  return {
    attributes: attrs,
    name: "anthropic.chat",
    startTime: [0, 0] as [number, number],
    endTime: [1, 0] as [number, number],
    spanContext: () => ({
      traceId: "0000000000000000000000000000abcd",
      spanId: "0000000000001234",
    }),
    parentSpanId: undefined,
  } as any;
}

describe("SDK usage synthesis parity (shared fixture)", () => {
  beforeEach(() => {
    logged.length = 0;
  });

  it("emits disjoint cache_read / cache_creation + reasoning matching the fixture", async () => {
    const proc = new TokenPoliceSpanProcessor();
    proc.onEnd(fakeSpan(fixture.input_attrs));
    // The log is deferred to process.nextTick — let it run.
    await new Promise((r) => setTimeout(r, 0));

    expect(logged.length).toBe(1);
    const lastArg = logged[0][logged[0].length - 1];
    const raw = lastArg.usage.raw;

    const exp = fixture.expected_usage_raw;
    expect(raw.cache_read_input_tokens).toBe(exp.cache_read_input_tokens);
    expect(raw.cache_creation_input_tokens).toBe(exp.cache_creation_input_tokens);
    expect(raw.prompt_tokens_details.cached_tokens).toBe(
      exp.prompt_tokens_details.cached_tokens,
    );
    expect(raw.completion_tokens_details.reasoning_tokens).toBe(
      exp.completion_tokens_details.reasoning_tokens,
    );
  });

  it("canonicalizes service_tier per the shared fixture (A1)", () => {
    for (const [input, expected] of Object.entries(
      fixture.service_tier_cases as Record<string, string>,
    )) {
      // OpenAI shape: response-level service_tier.
      expect(_extractServiceTier({ service_tier: input })).toBe(expected);
      // Anthropic shape: inside the usage object.
      expect(_extractServiceTier({ usage: { service_tier: input } })).toBe(expected);
    }
    expect(_extractServiceTier(null)).toBe("");
    expect(_extractServiceTier({})).toBe("");
  });
});

describe("Google GenAI request-config service tier capture (R4)", () => {
  // Gemini takes `service_tier` on the request GenerateContentConfig and never
  // echoes it in the response — so the response-side path is inert and the tier
  // must be harvested from the request config. _stashRequestServiceTier writes it
  // under the call's compKey; _logManual reads it back as usage.tier.
  const ORDER = 3;
  function makeSession() {
    return { traceId: "trace-r4", spanCounter: ORDER, _pendingCompositions: {} as any };
  }
  const stashed = (s: any) => s._pendingCompositions[`${s.traceId}:${ORDER}`]?.service_tier;

  it("captures camelCase serviceTier from request config", () => {
    const s = makeSession();
    _stashRequestServiceTier("google", [{ model: "gemini-2.5-pro", config: { serviceTier: "priority" } }], s, ORDER);
    expect(stashed(s)).toBe("priority");
  });

  it("captures snake_case service_tier from request config", () => {
    const s = makeSession();
    _stashRequestServiceTier("google", [{ config: { service_tier: "priority" } }], s, ORDER);
    expect(stashed(s)).toBe("priority");
  });

  it("no config → nothing stashed", () => {
    const s = makeSession();
    _stashRequestServiceTier("google", [{ model: "gemini-2.5-pro" }], s, ORDER);
    expect(stashed(s)).toBeUndefined();
  });

  it("'standard' tier is dropped (normalizes to '')", () => {
    const s = makeSession();
    _stashRequestServiceTier("google", [{ config: { serviceTier: "standard" } }], s, ORDER);
    expect(stashed(s)).toBeUndefined();
  });

  it("a throwing config getter is swallowed → nothing stashed", () => {
    const s = makeSession();
    const hostile = new Proxy({}, {
      get(_t, prop) {
        if (prop === "serviceTier" || prop === "service_tier") throw new Error("boom");
        return undefined;
      },
    });
    expect(() =>
      _stashRequestServiceTier("google", [{ config: hostile }], s, ORDER),
    ).not.toThrow();
    expect(stashed(s)).toBeUndefined();
  });

  it("non-google provider is ignored", () => {
    const s = makeSession();
    _stashRequestServiceTier("openai", [{ config: { serviceTier: "priority" } }], s, ORDER);
    expect(stashed(s)).toBeUndefined();
  });

  it("response echo precedence: _logManual prefers the response tier over the stashed config", () => {
    // _logManual computes `_extractServiceTier(result) || stashedServiceTier`.
    // A response-echoed tier (if a provider ever adds one) wins over the config.
    const responseTier = _extractServiceTier({ service_tier: "flex" });
    const stashedTier = "priority";
    expect(responseTier || stashedTier).toBe("flex");
    // With no response echo (the real Gemini case) the config fallback is used.
    const noEcho = _extractServiceTier({ usageMetadata: {} });
    expect(noEcho || stashedTier).toBe("priority");
  });
});
