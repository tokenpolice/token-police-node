/**
 * T-N1 + T-N2 — wire-shape × serving-slug matrix + host-map exhaustiveness
 * (sdk_test_hardening_design.md §4.1; gap classes G1+G2).
 *
 * After host remap feeds serving slugs into /check + pricing, but stream
 * parsers must keep the **module** wire key (`_wireParseKey`). These tests lock
 * every host-map slug so the next serving-provider mapping cannot silently
 * break streamed telemetry the way did.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { setClient } from "../src/state";
import { TPSession, _getSessionStorage } from "../src/context";
import { __test__ as enforcerTest, _extractUsage } from "../src/enforcer";

const {
  _wireParseKey,
  _chunkHasUsage,
  _newStreamAccumulator,
  _accumulateStreamChunk,
  _streamAccumulatorToResponse,
  _tapStreamUsageForOnEnd,
  _matchHostToProvider,
  _resolveServingFromBaseUrl,
} = enforcerTest as any;

/** OpenAI-compatible serving slugs from host maps (design T-N1 list + sentinels). */
const OPENAI_COMPAT_SERVING_SLUGS = [
  "minimax",
  "xai",
  "deepseek",
  "moonshot",
  "zhipu",
  "perplexity",
  "fireworks",
  "deepinfra",
  "novita",
  "nebius",
  "vercel-gateway",
  "azure-openai",
  "azure-ai",
  "self_hosted",
  // green sentinels (module allowlist / known good)
  "openrouter",
  "together",
  "cerebras",
  "openai",
  "groq",
] as const;

/**
 * Representative hosts for every slug the SDK host map can produce.
 * Kept in the test (not by exporting private maps) — assert via
 * `_matchHostToProvider` so drift still fails CI when a host stops resolving.
 */
const HOST_EXAMPLES: Record<string, string> = {
  openai: "api.openai.com",
  anthropic: "api.anthropic.com",
  google: "generativelanguage.googleapis.com",
  mistral: "api.mistral.ai",
  xai: "api.x.ai",
  deepseek: "api.deepseek.com",
  moonshot: "api.moonshot.ai",
  minimax: "api.minimax.io",
  perplexity: "api.perplexity.ai",
  cohere: "api.cohere.com",
  zhipu: "open.bigmodel.cn",
  "vercel-gateway": "ai-gateway.vercel.sh",
  openrouter: "openrouter.ai",
  together: "api.together.xyz",
  fireworks: "api.fireworks.ai",
  deepinfra: "api.deepinfra.com",
  novita: "api.novita.ai",
  groq: "api.groq.com",
  cerebras: "api.cerebras.ai",
  nebius: "api.studio.nebius.com",
  // pattern hosts
  "azure-openai": "my-resource.openai.azure.com",
  "azure-ai": "my.services.ai.azure.com",
  bedrock: "bedrock-runtime.us-east-1.amazonaws.com",
  "vertex-ai": "us-central1-aiplatform.googleapis.com",
  self_hosted: "localhost",
};

function openaiUsageChunk(prompt = 12, completion = 7) {
  return {
    usage: {
      prompt_tokens: prompt,
      completion_tokens: completion,
      total_tokens: prompt + completion,
    },
    choices: [],
  };
}

function openaiDeltaChunk(content: string) {
  return { choices: [{ delta: { content } }] };
}

function uniqueHostMapSlugs(): string[] {
  return Object.keys(HOST_EXAMPLES).sort();
}

function loadSharedProviderTypes(): Record<string, string> {
  const path = join(__dirname, "..", "..", "shared", "provider-slugs.json");
  const raw = JSON.parse(readFileSync(path, "utf8"));
  return raw.types as Record<string, string>;
}

// ─── T-N2 structural exhaustiveness ─────────────────────────────────────────

describe("T-N2 host-map → wire routing exhaustiveness", () => {
  it("HOST_EXAMPLES resolve via public _matchHostToProvider (map drift lock)", () => {
    for (const [slug, host] of Object.entries(HOST_EXAMPLES)) {
      expect(_matchHostToProvider(host), host).toBe(slug);
    }
  });

  it("every host-map slug is classified in the published provider-slug table types", () => {
    const types = loadSharedProviderTypes();
    const missing: string[] = [];
    for (const slug of uniqueHostMapSlugs()) {
      if (!types[slug]) missing.push(slug);
    }
    expect(missing, `host-map slugs missing from shared types: ${missing.join(",")}`).toEqual([]);
  });

  it("wire parse key is module identity for every host-map serving slug", () => {
    // Structural lock: adding a host never remaps the wire key.
    for (const slug of uniqueHostMapSlugs()) {
      expect(_wireParseKey("openai")).toBe("openai");
      expect(_wireParseKey("anthropic")).toBe("anthropic");
      // Serving slug is not a wire key transform of the module.
      expect(_wireParseKey(slug)).toBe(slug); // identity, never host-derived
    }
  });

  it("OpenAI-shaped stream under serving slug fails usable latch; under wire=openai succeeds", () => {
    const usage = openaiUsageChunk(9, 3);
    // Module/wire key always latches OpenAI usage with real tokens.
    expect(_chunkHasUsage(_wireParseKey("openai"), usage)).toBe(true);
    const u = _extractUsage("openai", usage, [{ model: "m" }]);
    expect(u.inputTokens).toBe(9);
    expect(u.outputTokens).toBe(3);

    // Regression class: gateway host remaps that selected a wrong parser.
    // Under the *serving* key alone these must not produce usable OpenAI tokens.
    // (Wire key openai always does — asserted above.)
    const b18FootgunSlugs = [
      "minimax",
      "xai",
      "deepseek",
      "moonshot",
      "zhipu",
      "perplexity",
      "fireworks",
      "deepinfra",
      "novita",
      "nebius",
      "vercel-gateway",
      "azure-openai",
      "azure-ai",
      "self_hosted",
      "vertex-ai",
      "google",
    ];
    for (const slug of b18FootgunSlugs) {
      if (!uniqueHostMapSlugs().includes(slug)) continue;
      const tokens = _extractUsage(slug, usage, [{ model: "m" }]);
      const usable =
        _chunkHasUsage(slug, usage) &&
        tokens.inputTokens + tokens.outputTokens > 0;
      expect(usable, `serving=${slug} must not usable-latch OpenAI usage`).toBe(
        false,
      );
    }
  });

  it("do not paper over by adding host-map gateway slugs to openai stream allowlist", () => {
    // Pure gateway/serving remaps that broke under no openai-shaped
    // accumulator under the *serving* slug (except AI-SDK `xai` which has its
    // own accumulator for Vercel wire shape — still wrong for OpenAI deltas).
    for (const slug of ["minimax", "deepseek", "moonshot", "zhipu", "perplexity"]) {
      expect(_newStreamAccumulator(slug), `serving ${slug}`).toBeNull();
    }
    // xai has an AI-SDK accumulator but OpenAI content deltas do not compose.
    const accX = _newStreamAccumulator("xai");
    if (accX != null) {
      _accumulateStreamChunk("xai", accX, {
        choices: [{ delta: { content: "hi" } }],
      });
      expect(_streamAccumulatorToResponse("xai", accX)).toBeNull();
    }
    expect(_newStreamAccumulator(_wireParseKey("openai"))).not.toBeNull();
  });
});

// ─── T-N1 cross-axis matrix ─────────────────────────────────────────────────

describe("T-N1 wire-shape × serving-slug matrix", () => {
  let session: TPSession;

  beforeEach(() => {
    setClient({
      log: () => {},
      captureStreamUsage: true,
    } as any);
    session = new TPSession({
      userId: "u-matrix",
      paidPlan: "pro",
      workflowName: "wire-matrix",
      traceId: "e".repeat(32),
      rootSpanId: "f".repeat(16),
    });
  });

  afterEach(() => {
    setClient(null as any);
  });

  it("host examples resolve to the expected serving slug", () => {
    for (const [slug, host] of Object.entries(HOST_EXAMPLES)) {
      expect(_matchHostToProvider(host), host).toBe(slug);
      const resolved = _resolveServingFromBaseUrl(`https://${host}/v1`);
      if (slug === "self_hosted") {
        // localhost may resolve recognized
        expect(resolved.kind === "recognized" || resolved.kind === "unrecognized").toBe(true);
        if (resolved.kind === "recognized") expect(resolved.provider).toBe("self_hosted");
      } else {
        expect(resolved).toEqual({ kind: "recognized", provider: slug });
      }
    }
  });

  for (const serving of OPENAI_COMPAT_SERVING_SLUGS) {
    it(`OpenAI-shaped stream under wire=openai latches usage (serving would be ${serving})`, async () => {
      const wire = _wireParseKey("openai");
      expect(wire).toBe("openai");

      // Host resolve (when we have an example) still yields the serving slug —
      // billing axis independent of wire.
      const host = HOST_EXAMPLES[serving];
      if (host) {
        expect(_matchHostToProvider(host)).toBe(serving);
      }

      const chunks = [
        openaiDeltaChunk("Hi "),
        openaiDeltaChunk("there"),
        openaiUsageChunk(15, 4),
      ];
      async function* gen() {
        for (const c of chunks) yield c;
      }

      await _getSessionStorage().run(session, async () => {
        const tapped = _tapStreamUsageForOnEnd(
          wire,
          gen(),
          session,
          [{ model: "gateway-model", messages: [{ role: "user", content: "x" }] }],
          0,
          true, // strip injected usage-only terminal
        );
        const seen: any[] = [];
        for await (const c of tapped) seen.push(c);

        // Usage-only terminal stripped from customer iterator
        expect(seen.length).toBe(2);
        expect(seen[0].choices[0].delta.content).toBe("Hi ");
        expect(seen[1].choices[0].delta.content).toBe("there");

        const cd = (session as any)._pendingCompositions[`${session.traceId}:0`];
        expect(cd?.usage?.input_tokens, `serving=${serving}`).toBe(15);
        expect(cd?.usage?.output_tokens).toBe(4);
        // Composition from OpenAI deltas under wire key
        expect(cd?.response?.length ?? 0).toBeGreaterThan(0);
      });
    });
  }

  it("customer-owned include_usage chunk is NOT stripped (suppress=false)", async () => {
    const chunks = [openaiDeltaChunk("x"), openaiUsageChunk(5, 1)];
    async function* gen() {
      for (const c of chunks) yield c;
    }
    await _getSessionStorage().run(session, async () => {
      const tapped = _tapStreamUsageForOnEnd(
        _wireParseKey("openai"),
        gen(),
        session,
        [{ model: "m", messages: [] }],
        0,
        false,
      );
      const seen: any[] = [];
      for await (const c of tapped) seen.push(c);
      expect(seen).toHaveLength(2);
      expect(seen[1].usage.prompt_tokens).toBe(5);
    });
  });

  it("Anthropic-shaped usage latches under wire=anthropic even when host is minimax", () => {
    // Anthropic client → api.minimax.io: serving=minimax, wire=anthropic.
    expect(_matchHostToProvider("api.minimax.io")).toBe("minimax");
    const anthropicUsage = {
      usage: { input_tokens: 20, output_tokens: 8 },
    };
    expect(_chunkHasUsage(_wireParseKey("anthropic"), anthropicUsage)).toBe(true);
    // Serving slug alone uses Google-style / wrong branch → no latch on Anthropic usage shape
    // (minimax falls through to usageMetadata).
    expect(_chunkHasUsage("minimax", anthropicUsage)).toBe(false);
  });

  it("OpenAI stream accumulator builds composition under wire key for gateway traffic", () => {
    const acc = _newStreamAccumulator(_wireParseKey("openai"));
    expect(acc).not.toBeNull();
    _accumulateStreamChunk("openai", acc, {
      choices: [
        {
          delta: {
            tool_calls: [
              {
                index: 0,
                id: "call_matrix",
                type: "function",
                function: { name: "lookup", arguments: '{"q":"x"}' },
              },
            ],
          },
        },
      ],
    });
    const resp = _streamAccumulatorToResponse("openai", acc);
    expect(resp).not.toBeNull();
    expect(resp.choices[0].message.tool_calls[0].id).toBe("call_matrix");
  });
});
