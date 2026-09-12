/**
 * Node-15 — one shared code-point counter + consolidated `hashLen`.
 *
 * `codePointLength` is the SINGLE allocation-free Unicode code-point counter now
 * routed through by (a) the primary composition `length` field (`textEntry`),
 * (b) the OTel-fallback composition path in telemetry.ts, and (c) the tool
 * arg/result `hashLen` helper (formerly duplicated in telemetry.ts /
 * frameworkTools.ts / context.ts). It must be byte-identical to `[...s].length`
 * for well-formed strings and NO-THROW on lone surrogates.
 *
 * Cross-SDK: code-point length matches Python's `len(str)`. Pinned literals here
 * are the values Python's twin suite (test_composition*.py) asserts for the same
 * inputs — computed independently (Python counts code points natively), not
 * derived from the code under test.
 */
import { describe, it, expect } from "vitest";
import { createHash } from "node:crypto";
import {
  codePointLength,
  hashLen,
  canonicalJson,
  buildResponseComposition,
} from "../src/composition";

// Independent sha1-16 oracle (NOT the code under test) — mirrors the hash the
// OLD duplicated hashLen copies produced: `createHash("sha1").update(s)` with
// utf8-default encoding, first 16 hex chars.
function sha116(s: string): string {
  return createHash("sha1").update(s).digest("hex").slice(0, 16);
}

describe("codePointLength — counts Unicode code points, allocation-free", () => {
  it("ASCII: same as .length and [...s].length", () => {
    expect(codePointLength("hello")).toBe(5);
    expect(codePointLength("hello")).toBe([..."hello"].length);
    expect(codePointLength("hello")).toBe("hello".length);
  });

  it("empty string → 0", () => {
    expect(codePointLength("")).toBe(0);
  });

  it("BMP non-ASCII (precomposed é, U+00E9): 1 code point each", () => {
    const s = "héllo"; // héllo — é escaped to U+00E9 so the test is
    // independent of this file's on-disk Unicode normalization (NFC vs NFD).
    expect(codePointLength(s)).toBe(5);
    expect(codePointLength(s)).toBe([...s].length);
    // BMP chars occupy exactly one UTF-16 unit, so .length agrees here.
    expect(s.length).toBe(5);
  });

  it("astral (emoji): 1 code point but 2 UTF-16 units", () => {
    expect(codePointLength("😀")).toBe(1);
    expect("😀".length).toBe(2); // UTF-16 units — the OLD fallback bug
    expect(codePointLength("a😀b")).toBe(3);
    expect(codePointLength("a😀b")).toBe([..."a😀b"].length);
  });

  it("multiple astral + a surrogate-pair sequence", () => {
    const s = "\u{1F44B}\u{1F30D}"; // 👋🌍
    expect(codePointLength(s)).toBe(2);
    expect(codePointLength(s)).toBe([...s].length);
    expect(s.length).toBe(4); // 4 UTF-16 units
  });

  it("lone/unpaired surrogates do NOT throw and count 1 each (matches spread)", () => {
    const loneHigh = "a\uD800b"; // unpaired high surrogate
    const loneLow = "\uDC00"; // unpaired low surrogate
    const highThenHigh = "\uD800\uD800"; // two highs, no low
    for (const s of [loneHigh, loneLow, highThenHigh]) {
      expect(() => codePointLength(s)).not.toThrow();
      // The array spread yields lone surrogates as single elements too, so the
      // counter must agree even on malformed input.
      expect(codePointLength(s)).toBe([...s].length);
    }
    expect(codePointLength(loneHigh)).toBe(3);
    expect(codePointLength(loneLow)).toBe(1);
    expect(codePointLength(highThenHigh)).toBe(2);
  });

  it("random well-formed corpus: codePointLength === [...s].length", () => {
    const corpus = [
      "",
      "plain",
      "café ☕",
      "emoji 👨‍👩‍👧‍👦 family (ZWJ)",
      "math 𝕏 and 𝟙",
      "mixed aéb😀c\u{1F30D}d",
      "surrogate pair only 𐍈",
    ];
    for (const s of corpus) {
      expect(codePointLength(s)).toBe([...s].length);
    }
  });
});

describe("codePointLength — pinned cross-SDK constant (Python len() twin)", () => {
  it('"héllo 👋🌍" is 8 code points (Python len() == 8)', () => {
    // h é l l o SPACE 👋 🌍 = 8 code points. Python's test_composition twin
    // pins len("héllo 👋🌍") == 8 (Python counts code points natively).
    const s = "héllo \u{1F44B}\u{1F30D}";
    expect(codePointLength(s)).toBe(8);
    expect(s.length).toBe(10); // 10 UTF-16 units — what the OLD path reported
  });
});

describe("fallback-vs-primary length parity (astral)", () => {
  it("primary composition length uses the SAME code-point counter as the fallback path", () => {
    // Primary path (textEntry via buildResponseComposition).
    const content = "hi 😀🌍 there";
    const comp = buildResponseComposition("openai", {
      choices: [{ message: { content } }],
    });
    expect(comp[0].type).toBe("text");
    // The telemetry.ts OTel-fallback path now computes `length:
    // codePointLength(content.trim())` — the IDENTICAL call the primary path
    // makes here — so both routes report the same code-point length for astral
    // content (previously the fallback reported UTF-16 units).
    expect(comp[0].length).toBe(codePointLength(content.trim()));
    expect(comp[0].length).toBe([...content.trim()].length);
    // Sanity: differs from the old UTF-16 count.
    expect(comp[0].length).not.toBe(content.trim().length);
  });
});

describe("hashLen — consolidated helper (hash byte-identical to the old copies)", () => {
  it("null / undefined → ['', 0]", () => {
    expect(hashLen(null)).toEqual(["", 0]);
    expect(hashLen(undefined)).toEqual(["", 0]);
  });

  it("empty string → ['', 0]", () => {
    expect(hashLen("")).toEqual(["", 0]);
  });

  it("string args: sha1-16 + code-point length (independent oracle)", () => {
    for (const s of ["hello", "héllo \u{1F44B}\u{1F30D}", "😀", "a\uD800b"]) {
      const [h, len] = hashLen(s);
      expect(h).toBe(sha116(s));
      expect(len).toBe(codePointLength(s));
    }
  });

  it("PINNED hashes match values computed BEFORE the refactor (regression lock)", () => {
    // Literals captured from the pre-refactor duplicated helpers, so the
    // consolidation is proven to leave the hash byte-identical.
    expect(hashLen("hello")).toEqual(["aaf4c61ddcc5e8a2", 5]);
    expect(hashLen("héllo \u{1F44B}\u{1F30D}")).toEqual(["5dce3daa51cf520b", 8]);
    expect(hashLen("😀")).toEqual(["9c533688a979a858", 1]);
    expect(hashLen("a\uD800b")).toEqual(["c3693aea616c886c", 3]);
    expect(hashLen("\uDC00")).toEqual(["9bdb77276c1852e1", 1]);
  });

  it("non-string args route through canonicalJson (byte-identical hash + length)", () => {
    const [h, len] = hashLen({ a: 1 });
    const canon = canonicalJson({ a: 1 });
    expect(canon).toBe('{"a":1}');
    expect(h).toBe(sha116(canon));
    expect(len).toBe(codePointLength(canon));
    expect(len).toBe(7);
    expect(h).toBe("9f89c740ceb46d74");
  });
});
