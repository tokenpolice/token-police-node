/**
 * Composition robustness fixes:
 * - Message parsers skip null/undefined/non-object array elements instead of
 * throwing (which would collapse the whole composition to a coarse fallback).
 * Parity: the Python parser likewise emits no entry for such elements.
 * - Pre-tokenized int-array elements in a batch embedding input reach the
 * pre-tokenized branch (previously shadowed by the generic object branch,
 * since `typeof [] === "object"`).
 * - A non-audio fetch `Response` (e.g. application/json) is no longer
 * mislabeled as binary audio; a missing/unreadable content-type still is.
 */
import { describe, it, expect } from "vitest";
import {
  buildPromptComposition,
  buildResponseComposition,
} from "../src/composition";

describe("message parsers — null/undefined element guard (fix 2)", () => {
  it("OpenAI messages: null/undefined/non-object elements are skipped, valid ones kept", () => {
    const comp = buildPromptComposition(
      "openai",
      {
        messages: [
          null,
          { role: "user", content: "hello" },
          undefined,
          42,
          "loose-string",
          { role: "assistant", content: "world" },
        ],
      },
      undefined,
    );
    // Exactly the two valid object messages produce text entries.
    expect(comp.length).toBe(2);
    expect(comp[0]).toMatchObject({ role: "user", type: "text" });
    expect(comp[1]).toMatchObject({ role: "assistant", type: "text" });
  });

  it("OpenAI messages: leading null does not collapse to a single fallback entry", () => {
    const comp = buildPromptComposition(
      "openai",
      { messages: [null, { role: "user", content: "a" }, { role: "user", content: "b" }] },
      undefined,
    );
    // Per-message granularity preserved (2 entries), not one coarse fallback.
    expect(comp.length).toBe(2);
  });

  it("Anthropic messages: null/undefined/non-object elements are skipped", () => {
    const comp = buildPromptComposition(
      "anthropic",
      {
        messages: [
          null,
          { role: "user", content: "hi" },
          undefined,
          { role: "assistant", content: [{ type: "text", text: "yo" }] },
        ],
      },
      undefined,
    );
    expect(comp.length).toBe(2);
    expect(comp[0]).toMatchObject({ role: "user", type: "text" });
    expect(comp[1]).toMatchObject({ role: "assistant", type: "text" });
  });

  it("all-null message array yields an empty composition, never throws", () => {
    expect(() =>
      buildPromptComposition("openai", { messages: [null, undefined] }, undefined),
    ).not.toThrow();
    const comp = buildPromptComposition("openai", { messages: [null, undefined] }, undefined);
    expect(comp).toEqual([]);
  });
});

describe("batch embedding — pre-tokenized int-array branch (fix 3)", () => {
  it("batch of int-arrays produces the pre-tokenized entry shape (length only)", () => {
    const comp = buildPromptComposition(
      "openai",
      { input: [[1, 2, 3, 4], [5, 6]] },
      "embedding",
    );
    expect(comp.length).toBe(2);
    expect(comp[0]).toEqual({ role: "input", type: "text", length: 4 });
    expect(comp[1]).toEqual({ role: "input", type: "text", length: 2 });
    // Length-only: no hash for a pre-tokenized entry.
    expect(comp[0]).not.toHaveProperty("hash");
  });

  it("single flat int-array still short-circuits to one length-only entry", () => {
    const comp = buildPromptComposition("openai", { input: [1, 2, 3] }, "embedding");
    expect(comp).toEqual([{ role: "input", type: "text", length: 3 }]);
  });

  it("object-batch behavior unchanged (text / image typed entries)", () => {
    const comp = buildPromptComposition(
      "cohere",
      {
        inputs: [
          { type: "text", text: "hello" },
          { type: "image_url" },
          { type: "weird" },
        ],
      },
      "embedding",
    );
    expect(comp.length).toBe(3);
    expect(comp[0]).toMatchObject({ role: "input", type: "text" });
    expect(comp[1]).toEqual({ role: "input", type: "image" });
    expect(comp[2]).toEqual({ role: "input", type: "weird" });
  });

  it("mixed batch: string, int-array, and object each map correctly", () => {
    const comp = buildPromptComposition(
      "openai",
      { input: ["plain", [7, 8, 9], { type: "text", text: "obj" }] },
      "embedding",
    );
    expect(comp.length).toBe(3);
    expect(comp[0]).toMatchObject({ role: "input", type: "text" }); // string → hashed
    expect(comp[0]).toHaveProperty("hash");
    expect(comp[1]).toEqual({ role: "input", type: "text", length: 3 }); // int-array → length only
    expect(comp[2]).toMatchObject({ role: "input", type: "text" }); // object text → hashed
    expect(comp[2]).toHaveProperty("hash");
  });
});

describe("Response classification by content-type (fix 5)", () => {
  function makeResponse(contentType: string | null, opts: { throwOnGet?: boolean } = {}) {
    const headers = opts.throwOnGet
      ? {
          get() {
            throw new Error("headers getter exploded");
          },
        }
      : {
          get(name: string) {
            return name.toLowerCase() === "content-type" ? contentType : null;
          },
        };
    // A real fetch Response instance so `instanceof Response` matches, with our
    // headers stub swapped in.
    const r = new Response("body-bytes");
    Object.defineProperty(r, "headers", { value: headers, configurable: true });
    return r;
  }

  const AUDIO = new Set(["audio"]);
  function isAudio(resp: any): boolean {
    // Route through the public response builder; an audio Response yields a
    // single non-text audio entry.
    const comp = buildResponseComposition("openai", resp);
    return comp.length === 1 && comp[0].type === "audio";
  }

  it("audio content-type → classified as audio", () => {
    expect(isAudio(makeResponse("audio/mpeg"))).toBe(true);
  });

  it("application/octet-stream → classified as audio", () => {
    expect(isAudio(makeResponse("application/octet-stream"))).toBe(true);
  });

  it("application/json → NOT audio", () => {
    expect(AUDIO.has("audio")).toBe(true); // sanity
    expect(isAudio(makeResponse("application/json"))).toBe(false);
  });

  it("text/plain → NOT audio", () => {
    expect(isAudio(makeResponse("text/plain"))).toBe(false);
  });

  it("missing content-type header → classified as audio (preserves TTS case)", () => {
    expect(isAudio(makeResponse(null))).toBe(true);
  });

  it("throwing headers getter → classified as audio, never throws", () => {
    let result: boolean | undefined;
    expect(() => {
      result = isAudio(makeResponse(null, { throwOnGet: true }));
    }).not.toThrow();
    expect(result).toBe(true);
  });
});
