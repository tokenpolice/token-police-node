/**
 * Canonical tool-arg serializer parity (Node side).
 *
 * Locks that `canonicalJson` produces a byte-identical canonical string to the
 * Python SDK's `_canonical_json` for JSON-native values, so the composition
 * `hash` (sha1-16 over the canonical string) and `length` (code points) of a
 * tool-call entry match across SDKs. Reads the SHARED fixture
 * `shared/composition-canonical-fixture.json` (also read by
 * test_composition_canonical.py) so both suites assert the identical inputs.
 *
 * Also absorbs the textEntry `length` is now counted in Unicode code
 * points (not UTF-16 code units).
 */
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { canonicalJson, buildResponseComposition } from "../src/composition";

const fixture = JSON.parse(
  readFileSync(join(__dirname, "../../shared/composition-canonical-fixture.json"), "utf8"),
) as { cases: Array<{ _name: string; input: unknown; canonical: string; length: number }> };

// PINNED cross-SDK constant — present byte-for-byte in test_composition_canonical.py.
// canonicalJson({"b":2,"a":1}) === '{"a":1,"b":2}'; sha1-first-16 of that string.
const CANON_HASH_AB = "4acc71e0547112eb";

function sha116(s: string): string {
  return createHash("sha1").update(s, "utf8").digest("hex").slice(0, 16);
}

describe("canonicalJson — shared fixture (cross-SDK byte-identical, assertions 1,3,5,21,22)", () => {
  for (const c of fixture.cases) {
    it(`canonical string matches pinned: ${c._name}`, () => {
      const got = canonicalJson(c.input);
      expect(got).toBe(c.canonical); // assertion 3 / 21 / 22
      // assertion 5: length = code points of the (already-trimmed) canonical string
      expect([...got].length).toBe(c.length);
    });
  }
});

describe("canonicalJson — pinned hash + integer keys (assertions 4, 8)", () => {
  it("identical hash across SDKs via CANON_HASH_AB (assertion 4)", () => {
    const canon = canonicalJson({ b: 2, a: 1 });
    expect(canon).toBe('{"a":1,"b":2}');
    expect(sha116(canon)).toBe(CANON_HASH_AB);
  });

  it("integer-like keys sort lexicographically, NOT numerically (assertion 8)", () => {
    // JSON.stringify alone would reorder these to 1,2,10 (numeric); canonicalJson must not.
    expect(canonicalJson({ "10": 0, "2": 0, "1": 0 })).toBe('{"1":0,"10":0,"2":0}');
  });
});

describe("canonicalJson — numbers (assertions 6, 7, 21)", () => {
  it("NaN / +-Infinity → null (assertion 6)", () => {
    // Cannot live in a .json fixture — JSON has no NaN literal.
    expect(canonicalJson({ x: NaN, y: Infinity, z: -Infinity })).toBe('{"x":null,"y":null,"z":null}');
  });

  it("integer-valued float → integer literal; 0.5 stays 0.5 (assertion 7)", () => {
    expect(canonicalJson({ n: 1.0 })).toBe('{"n":1}');
    expect(canonicalJson({ n: 0.5 })).toBe('{"n":0.5}');
  });

  it("negative zero → 0 (Decision rule 5iii)", () => {
    expect(canonicalJson({ z: -0 })).toBe('{"z":0}');
  });

  it("float notation normalized to plain decimal (assertion 21)", () => {
    // JS would natively render 1e-7 in exponential; the expander forces plain.
    expect(canonicalJson({ lr: 1e-5 })).toBe('{"lr":0.00001}');
    expect(canonicalJson({ x: 1e-7 })).toBe('{"x":0.0000001}');
    expect(canonicalJson({ big: 1e30 })).toBe('{"big":1000000000000000000000000000000}');
  });
});

describe("canonicalJson — string escaping (assertion 22)", () => {
  it("U+0001 → lowercase 4-digit \\u0001 (in-code; not a JSON literal)", () => {
    const u1 = String.fromCharCode(1);
    expect(canonicalJson({ c: u1 })).toBe('{"c":"\\u0001"}');
  });

  it("does NOT escape '/', U+2028, U+2029 (emitted literally)", () => {
    expect(canonicalJson({ a: "/" })).toBe('{"a":"/"}');
    expect(canonicalJson({ a: "  " })).toBe('{"a":"  "}');
  });
});

describe("canonicalJson — total / no-throw (assertion 11)", () => {
  it("never throws on a hostile getter, a Date, or a circular reference", () => {
    const hostile: any = {};
    Object.defineProperty(hostile, "boom", {
      enumerable: true,
      get() {
        throw new Error("getter explodes");
      },
    });
    expect(() => canonicalJson(hostile)).not.toThrow();

    expect(() => canonicalJson({ d: new Date(0) })).not.toThrow();

    const circular: any = { a: 1 };
    circular.self = circular;
    expect(() => canonicalJson(circular)).not.toThrow();

    // Within-SDK determinism for the datetime case (NOT cross-SDK; Decision rule 8).
    const a = canonicalJson({ d: new Date(0) });
    const b = canonicalJson({ d: new Date(0) });
    expect(a).toBe(b);
    expect(typeof a).toBe("string");
  });
});

describe("big-integer residual — named, within-SDK only (assertion 23)", () => {
  it("handles >2^53 ints without throwing + deterministically (NOT asserted cross-SDK)", () => {
    // Decision rule 5b: JS lost precision at JSON.parse before the serializer ran,
    // so this is DELIBERATELY excluded from the byte-identical fixture and only
    // asserted within-SDK. (Python keeps 1234567890123456789 exact; JS does not.)
    const big = 1234567890123456789;
    let out1 = "";
    let out2 = "";
    expect(() => {
      out1 = canonicalJson({ big });
      out2 = canonicalJson({ big });
    }).not.toThrow();
    expect(out1).toBe(out2);
    expect(out1.length).toBeGreaterThan(0);
  });
});

describe("textEntry length unit = code points", () => {
  it("astral char counts 1 code point, not 2 UTF-16 units", () => {
    const comp = buildResponseComposition("openai", {
      choices: [{ message: { content: "a😀b" } }],
    });
    expect(comp[0].type).toBe("text");
    expect(comp[0].length).toBe(3); // was 4 (UTF-16) before
  });

  it("BMP-only content length unchanged (no re-baseline)", () => {
    const comp = buildResponseComposition("openai", {
      choices: [{ message: { content: "hello world" } }],
    });
    expect(comp[0].length).toBe(11);
  });
});

describe("anchor tool_use routed through canonicalJson (assertions 10, 14, 16)", () => {
  it("Anthropic tool_use input hashes via canonicalJson (anchor:716/response)", () => {
    const comp = buildResponseComposition("anthropic", {
      content: [{ type: "tool_use", name: "calc", input: { b: 2, a: 1 } }],
    });
    const tc = comp.find((e: any) => e.type === "tool_call") as any;
    expect(tc).toBeTruthy();
    // canonical {"a":1,"b":2} → same string + hash + length the Python anchor produces.
    expect(tc.hash).toBe(CANON_HASH_AB);
    expect(tc.length).toBe(13);
  });

  it("empty / absent tool input → {} with length 2 (assertion 16)", () => {
    const comp = buildResponseComposition("anthropic", {
      content: [{ type: "tool_use", name: "noargs", input: {} }],
    });
    const tc = comp.find((e: any) => e.type === "tool_call") as any;
    expect(tc.length).toBe(2);
    expect(tc.hash).toBe("bf21a9e8fbc5a384"); // sha1-16 of "{}"
  });

  it("plain text hash unchanged — serializer NOT invoked for string content (assertion 14)", () => {
    const comp = buildResponseComposition("openai", {
      choices: [{ message: { content: "hello" } }],
    });
    expect(comp[0].hash).toBe("aaf4c61ddcc5e8a2"); // sha1-16 of "hello"
  });
});
