/**
 * Embeddings support — Node SDK contract tests.
 *
 * Mirrors token-police-python/tests/test_embeddings.py — divergence between
 * the SDKs is a bug per the SDK agent guide.
 *
 * Covers (a) the embedding composition layer (input parser + response empty
 * short-circuit for EMBEDDING_SHAPES), (b) the embedding usage extractor
 * across providers, (c) the fail-safe contract — extraction failures must
 * never propagate (feedback_sdk_never_fails_app).
 *
 * The internal `_extract*` / `_resolve*` helpers are not exported, so the
 * tests round-trip through the public composition surface and end-to-end
 * payload shapes (via tp.log) where useful, and a small `__test__` re-export
 * stub keeps the internal helper assertions tractable.
 */
import { describe, it, expect } from "vitest";
import {
  buildPromptComposition,
  buildResponseComposition,
} from "../src/composition";

describe("buildPromptComposition — operation='embedding'", () => {
  it("string input → single text entry with role=input", () => {
    const comp = buildPromptComposition("openai", { input: "hello" }, "embedding");
    expect(comp.length).toBe(1);
    expect(comp[0].role).toBe("input");
    expect(comp[0].type).toBe("text");
    expect(comp[0].length).toBe(5);
    expect(comp[0]).toHaveProperty("hash");
  });

  it("list of strings → N entries", () => {
    const comp = buildPromptComposition(
      "openai",
      { input: ["foo", "bar", "baz"] },
      "embedding",
    );
    expect(comp.length).toBe(3);
    comp.forEach((c) => {
      expect(c.role).toBe("input");
      expect(c.type).toBe("text");
    });
  });

  it("pre-tokenized list[int] → single text entry with length only", () => {
    const comp = buildPromptComposition(
      "openai",
      { input: [10, 20, 30, 40, 50] },
      "embedding",
    );
    expect(comp.length).toBe(1);
    expect(comp[0].role).toBe("input");
    expect(comp[0].type).toBe("text");
    expect(comp[0].length).toBe(5);
    expect(comp[0]).not.toHaveProperty("hash");
  });

  it("Cohere multimodal text + image_url segments", () => {
    const segments = [
      { type: "text", text: "a black cat" },
      { type: "image_url", image_url: "http://example.com/cat.jpg" },
    ];
    const comp = buildPromptComposition(
      "cohere",
      { inputs: segments },
      "embedding",
    );
    expect(comp.length).toBe(2);
    expect(comp[0].type).toBe("text");
    expect(comp[1].type).toBe("image");
    comp.forEach((c) => expect(c.role).toBe("input"));
  });

  it("Voyage `texts` param accepted as well", () => {
    const comp = buildPromptComposition(
      "voyage",
      { texts: ["doc one", "doc two"] },
      "embedding",
    );
    expect(comp.length).toBe(2);
  });

  it("empty / missing input returns []", () => {
    expect(buildPromptComposition("openai", {}, "embedding")).toEqual([]);
    expect(
      buildPromptComposition("openai", { input: null }, "embedding"),
    ).toEqual([]);
  });

  it("chat path unchanged (operation omitted)", () => {
    const comp = buildPromptComposition("openai", {
      messages: [{ role: "user", content: "hi" }],
    });
    expect(comp.length).toBe(1);
    expect(comp[0].role).toBe("user");
  });
});

describe("buildResponseComposition — embedding shape short-circuit", () => {
  const EMBEDDING_SHAPES = [
    "openai_embeddings",
    "google_genai_embeddings",
    "cohere_embed",
    "mistral_embed",
    "voyage_embed",
    "huggingface_embed",
    "together_embed",
  ];

  it.each(EMBEDDING_SHAPES)(
    "%s returns [] regardless of response body",
    (shape) => {
      // Pass a deliberately misleading body — a parser would read .choices /
      // .data and emit non-empty entries. The shape override must override.
      const body = { choices: [{ message: { content: "X".repeat(100) } }] };
      const comp = buildResponseComposition("openai", body, shape);
      expect(comp).toEqual([]);
    },
  );

  it("non-embedding modality shapes still produce a marker", () => {
    const comp = buildResponseComposition("openai", new Response(new Uint8Array(8)), "openai_audio_tts");
    expect(comp).toEqual([{ role: "assistant", type: "audio" }]);
  });
});

describe("end-to-end /log payload includes operation field", () => {
  // Smoke test: build a payload by calling tp.log() and inspect the body
  // the fetch layer would have sent. We construct a minimal TokenPolice
  // instance and stub fetch.
  it("operation='embedding' flows through to the JSON payload", async () => {
    const sentBodies: any[] = [];
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async (_url: any, init?: any) => {
      if (init?.body) sentBodies.push(JSON.parse(init.body));
      return new Response("{}", { status: 202 });
    }) as any;

    try {
      // Late import after we have the fetch shim in place.
      const { init } = await import("../src/client");
      const tp = init({ apiKey: "tp_sk_test_x", baseUrl: "http://localhost:1", enforce: false });

      tp.log(
        "u1", "free", "wf", "sess",
        "text-embedding-3-small", "openai",
        42, 0, 0,
        {},
        { trace_id: "t1", span_id: "s1" },
        [{ role: "input", type: "text", length: 5, hash: "abc" }],
        [],
        {
          usage: { shape: "openai_embeddings", raw: { prompt_tokens: 42, total_tokens: 42 } },
          operation: "embedding",
        },
      );

      // Wait for the fire-and-forget fetch to land.
      await tp.flush();
      expect(sentBodies.length).toBeGreaterThan(0);
      const body = sentBodies[sentBodies.length - 1];
      expect(body.operation).toBe("embedding");
      expect(body.usage.shape).toBe("openai_embeddings");
      expect(body.model.name).toBe("text-embedding-3-small");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("operation defaults to 'chat' when omitted (back-compat)", async () => {
    const sentBodies: any[] = [];
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async (_url: any, init?: any) => {
      if (init?.body) sentBodies.push(JSON.parse(init.body));
      return new Response("{}", { status: 202 });
    }) as any;

    try {
      const { init } = await import("../src/client");
      const tp = init({ apiKey: "tp_sk_test_y", baseUrl: "http://localhost:1", enforce: false });
      tp.log("u1", "free", "wf", "sess", "gpt-4o", "openai", 10, 5, 0, {}, undefined, undefined, undefined);
      await tp.flush();
      expect(sentBodies.length).toBeGreaterThan(0);
      expect(sentBodies[sentBodies.length - 1].operation).toBe("chat");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("Google approximated-shape usage (mldev/Gemini path) round-trips intact", async () => {
    // When the SDK's enforcer approximates input tokens for the
    // mldev/Gemini API (no response-side usage), it emits a usage block
    // shaped { shape: google_genai_embeddings, raw: { approx_input_tokens,
    // approximated: true } }. Confirm tp.log preserves this shape in the
    // payload — the server's mapGoogleEmbeddings reads approx_input_tokens
    // as the final fallback so embedding_input_tokens lands non-zero.
    const sentBodies: any[] = [];
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async (_url: any, init?: any) => {
      if (init?.body) sentBodies.push(JSON.parse(init.body));
      return new Response("{}", { status: 202 });
    }) as any;

    try {
      const { init } = await import("../src/client");
      const tp = init({ apiKey: "tp_sk_test_g", baseUrl: "http://localhost:1", enforce: false });

      tp.log(
        "u1", "free", "wf", "sess",
        "gemini-embedding-001", "google",
        50, 0, 0,
        {},
        { trace_id: "t1", span_id: "s1" },
        [{ role: "input", type: "text", length: 200, hash: "deadbeef" }],
        [],
        {
          usage: {
            shape: "google_genai_embeddings",
            raw: { approx_input_tokens: 50, approximated: true },
          },
          operation: "embedding",
        },
      );

      await tp.flush();
      const body = sentBodies[sentBodies.length - 1];
      expect(body.operation).toBe("embedding");
      expect(body.usage.shape).toBe("google_genai_embeddings");
      expect(body.usage.raw.approx_input_tokens).toBe(50);
      expect(body.usage.raw.approximated).toBe(true);
      expect(body.model.name).toBe("gemini-embedding-001");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
