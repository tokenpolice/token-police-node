/**
 * Tests for the response composition builder — focus on the TTS / binary
 * audio misclassification fix.
 *
 * Background: OpenAI's `audio.speech.create` returns a fetch `Response` with
 * `__binaryResponse: true`. A future SDK or vendor wrapper that exposes
 * `.text` as a string property (the equivalent of the Python
 * `HttpxBinaryResponseContent.text` property) would otherwise be misclassified
 * as `{type: "text", length: <byte_count>, hash: ...}`.
 */
import { describe, it, expect } from "vitest";
import {
  buildPromptComposition,
  buildResponseComposition,
  textEntry,
} from "../src/composition";

describe("buildResponseComposition — modality / binary detection", () => {
  it("OpenAI TTS fetch Response is classified as audio (no length, no hash)", () => {
    // Build a real Web `Response` whose body is binary audio.
    const audioBytes = new Uint8Array(110_462);
    const resp = new Response(audioBytes, {
      headers: { "content-type": "audio/mpeg" },
    });
    const comp = buildResponseComposition("openai", resp);
    expect(comp).toEqual([{ role: "assistant", type: "audio" }]);
    expect(comp[0]).not.toHaveProperty("length");
    expect(comp[0]).not.toHaveProperty("hash");
  });

  it("raw Buffer is classified as audio", () => {
    const buf = Buffer.alloc(2048);
    const comp = buildResponseComposition("openai", buf);
    expect(comp).toEqual([{ role: "assistant", type: "audio" }]);
  });

  it("ArrayBuffer is classified as audio", () => {
    const ab = new ArrayBuffer(2048);
    const comp = buildResponseComposition("openai", ab);
    expect(comp).toEqual([{ role: "assistant", type: "audio" }]);
  });

  it("Uint8Array is classified as audio", () => {
    const u8 = new Uint8Array(2048);
    const comp = buildResponseComposition("openai", u8);
    expect(comp).toEqual([{ role: "assistant", type: "audio" }]);
  });

  it("vendor wrapper exposing .text + body/arrayBuffer is treated as audio", () => {
    // Simulates a future SDK that exposes a `.text` STRING property on a
    // binary body — defensive: the body+arrayBuffer pair flags it as binary.
    const fakeWrapper = {
      text: "X".repeat(110_462),
      arrayBuffer: async () => new ArrayBuffer(110_462),
      body: { getReader: () => ({}) },
    };
    const comp = buildResponseComposition("openai", fakeWrapper);
    expect(comp[0].type).toBe("audio");
    expect(comp[0]).not.toHaveProperty("length");
  });

  it("vendor class named *BinaryResponseContent is treated as audio", () => {
    class HttpxBinaryResponseContent {
      text = "X".repeat(50_000);
    }
    const comp = buildResponseComposition("openai", new HttpxBinaryResponseContent());
    expect(comp).toEqual([{ role: "assistant", type: "audio" }]);
  });
});

describe("buildResponseComposition — text & image still work", () => {
  it("chat completion text response preserves length and hash", () => {
    const chat = {
      choices: [{ message: { content: "hello world", tool_calls: null } }],
    };
    const comp = buildResponseComposition("openai", chat);
    expect(comp).toHaveLength(1);
    expect(comp[0].role).toBe("assistant");
    expect(comp[0].type).toBe("text");
    expect(comp[0].length).toBe("hello world".length);
    expect(typeof comp[0].hash).toBe("string");
  });

  it("DALL-E / images.generate response → image entries (one per image)", () => {
    const imgResp = {
      data: [{ url: "https://...", b64_json: null }],
    };
    const comp = buildResponseComposition("openai", imgResp);
    expect(comp).toEqual([{ role: "assistant", type: "image" }]);
  });

  it("Whisper-style transcription (text property + no binary body) stays text", () => {
    const transcription = { text: "hello world from whisper" };
    const comp = buildResponseComposition("openai", transcription);
    expect(comp[0].type).toBe("text");
    expect(comp[0].length).toBe("hello world from whisper".length);
  });
});

describe("buildResponseComposition — usageShape authoritative override", () => {
  it("usageShape='openai_audio_tts' forces audio entry even on misleading body", () => {
    // Pass something that LOOKS like text (chat dict). The shape override
    // must win — this is the belt-and-braces defence against any new vendor
    // wrapper that slips past the binary heuristic.
    const misleading = {
      choices: [{ message: { content: "shouldn't be hashed" } }],
    };
    const comp = buildResponseComposition("openai", misleading, "openai_audio_tts");
    expect(comp).toEqual([{ role: "assistant", type: "audio" }]);
  });

  it("usageShape='openai_images' forces image entry", () => {
    const comp = buildResponseComposition("openai", null, "openai_images");
    expect(comp).toEqual([{ role: "assistant", type: "image" }]);
  });

  it("usageShape='google_veo' forces video entry", () => {
    const comp = buildResponseComposition("google", null, "google_veo");
    expect(comp).toEqual([{ role: "assistant", type: "video" }]);
  });

  it("unknown usageShape falls through to heuristic parsing", () => {
    const chat = {
      choices: [{ message: { content: "hello" } }],
    };
    const comp = buildResponseComposition("openai", chat, "some_new_shape");
    expect(comp[0].type).toBe("text");
    expect(comp[0].length).toBe(5);
  });
});

describe("buildResponseComposition — never throws", () => {
  it("returns empty/fallback on cursed inputs rather than throwing", () => {
    const cursed = new Proxy(
      {},
      {
        get() {
          throw new Error("accessor exploded");
        },
      },
    );
    // Must not throw.
    expect(() => buildResponseComposition("openai", cursed)).not.toThrow();
  });
});

describe("buildResponseComposition — LangChain tool_calls de-duplication", () => {
  // A tool-calling LangChain AIMessage exposes its calls in BOTH content[]
  // (as tool_use blocks) AND the normalized .tool_calls list. The parser must
  // emit each tool_call exactly ONCE (regression: it previously emitted twice).
  it("emits each tool_call once when content[] AND .tool_calls both carry it", () => {
    const aiMessage = {
      type: "ai",
      content: [
        { type: "text", text: "let me look that up" },
        { type: "tool_use", name: "search_knowledge_base", input: { q: "laptop" } },
        { type: "tool_use", name: "get_customer_info", input: { email: "a@b.com" } },
      ],
      tool_calls: [
        { name: "search_knowledge_base", args: { q: "laptop" }, id: "t1" },
        { name: "get_customer_info", args: { email: "a@b.com" }, id: "t2" },
      ],
    };
    const llmResult = { generations: [[{ message: aiMessage }]] };
    const comp = buildResponseComposition("langchain", llmResult);
    const toolCalls = comp.filter((e: any) => e.type === "tool_call");
    expect(toolCalls).toHaveLength(2);
    expect(toolCalls.map((e: any) => e.name).sort()).toEqual([
      "get_customer_info",
      "search_knowledge_base",
    ]);
    // The assistant text is present exactly once.
    expect(comp.filter((e: any) => e.type === "text")).toHaveLength(1);
  });

  it("still emits tool_calls when ONLY .tool_calls is present (no content blocks)", () => {
    const aiMessage = {
      type: "ai",
      content: "",
      tool_calls: [{ name: "only_via_normalized", args: {}, id: "t1" }],
    };
    const llmResult = { generations: [[{ message: aiMessage }]] };
    const comp = buildResponseComposition("langchain", llmResult);
    const toolCalls = comp.filter((e: any) => e.type === "tool_call");
    expect(toolCalls).toHaveLength(1);
    expect((toolCalls[0] as any).name).toBe("only_via_normalized");
  });
});

describe("textEntry — whitespace-normalized hashing", () => {
  // The same logical text must fingerprint identically across calls even when
  // a framework (e.g. CrewAI) rstrips it between turns.
  it("hashes 'hi' and 'hi\\n ' to the same hash + length", () => {
    const a = buildResponseComposition("openai", {
      choices: [{ message: { content: "hello world" } }],
    });
    const b = buildResponseComposition("openai", {
      choices: [{ message: { content: "  hello world\n" } }],
    });
    expect(a[0].hash).toBe(b[0].hash);
    expect(a[0].length).toBe(b[0].length);
    expect(a[0].length).toBe("hello world".length);
  });
});

describe("textEntry — malformed text part degrades to empty, never aborts", () => {
  // A malformed (non-string) text part must degrade to an empty-text entry,
  // never abort the whole composition. Before the fix, a null/numeric/object
  // `text` either read `(null).length` (throw) or silently produced an
  // undefined `length` then threw in `fastHash` — the throw escaped to the
  // outer guard in buildPromptComposition and the ENTIRE call collapsed to a
  // single coarse `complete_prompt` entry, silently losing per-message
  // granularity. These pin the per-message survival plus the exact
  // empty/numeric/object fingerprints.

  // sha1-first-16 of the trimmed canonical string, PINNED byte-for-byte and
  // shared with the Python suite (test_composition.py) as the cross-SDK parity
  // proof — computed independently, not derived from the code under test.
  const HASH_EMPTY = "da39a3ee5e6b4b0d"; // ""
  const HASH_FIVE = "ac3478d69a3c81fa"; // canonical "5"
  const HASH_OBJ = "9f89c740ceb46d74"; // canonical '{"a":1}'

  it("null text part keeps per-message composition (Google system_instruction)", () => {
    // The genuinely unguarded Node site: system_instruction parts push
    // textEntry("system", p.text) with no `?? ""` guard.
    // PRE-FIX (verified by stashing the helper change): this collapsed to a
    // single `complete_prompt` entry because `(null).length` threw.
    const comp = buildPromptComposition("google", {
      system_instruction: {
        parts: [{ text: "Healthy instruction." }, { text: null }],
      },
      contents: [{ role: "user", parts: [{ text: "Hello there." }] }],
    });

    expect(comp.some((e) => e.role === "complete_prompt")).toBe(false);
    // healthy system + null system + user = 3 per-message entries.
    expect(comp.length).toBe(3);
    expect(comp[0].length).toBe("Healthy instruction.".length);
    expect(comp[1].length).toBe(0);
    expect(comp[1].hash).toBe(HASH_EMPTY);
    // Sibling user part unchanged.
    expect(comp[2].length).toBe("Hello there.".length);
  });

  it("numeric text part keeps per-message composition (OpenAI messages)", () => {
    // `part.text ?? ""` passes a number straight through; PRE-FIX `(5).length`
    // was undefined then `fastHash(5)` threw, collapsing to complete_prompt.
    const comp = buildPromptComposition("openai", {
      messages: [
        { role: "system", content: "You are helpful." },
        {
          role: "user",
          content: [
            { type: "text", text: "Healthy part." },
            { type: "text", text: 5 },
          ],
        },
      ],
    });

    expect(comp.some((e) => e.role === "complete_prompt")).toBe(false);
    expect(comp.length).toBe(3);
    expect(comp[1].length).toBe("Healthy part.".length);
    expect(comp[2].length).toBe(1);
    expect(comp[2].hash).toBe(HASH_FIVE);
  });

  it("non-string text parity: numeric 5 and object {a:1}", () => {
    // VERBATIM cross-SDK parity — the same pinned length + hash are asserted in
    // the Python suite.
    const five = textEntry("user", 5);
    expect(five.length).toBe(1);
    expect(five.hash).toBe(HASH_FIVE);

    const obj = textEntry("user", { a: 1 });
    expect(obj.length).toBe(7); // '{"a":1}'
    expect(obj.hash).toBe(HASH_OBJ);
  });

  it("never raises on malformed content (helper sweep)", () => {
    const hostile = {
      toString() {
        throw new Error("nope");
      },
    };
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    const values: unknown[] = [
      null,
      undefined,
      5,
      true,
      10n,
      Symbol("s"),
      () => 1,
      [1, { x: [2, 3] }],
      hostile,
      circular,
      new Date(0),
    ];
    for (const value of values) {
      const entry = textEntry("user", value);
      expect(entry.role).toBe("user");
      expect(entry.type).toBe("text");
      expect(typeof entry.length).toBe("number");
      expect(typeof entry.hash).toBe("string");
      expect(entry.hash!.length).toBe(16);
    }
  });

  it("healthy string entry is byte-identical (hash-identity guard)", () => {
    const entry = textEntry("user", "hello world");
    expect(entry.length).toBe(11);
    expect(entry.hash).toBe("2aae6c35c94fcfb4");
    // Whitespace trimming still applies identically.
    expect(textEntry("user", "  hello world\n").hash).toBe(entry.hash);
  });
});
