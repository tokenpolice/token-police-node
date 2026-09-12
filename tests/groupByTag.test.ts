import { describe, it, expect } from "vitest";
import { generateGroupByTag } from "../src/localEvaluator";

// The SDK local fast-path group tag must BYTE-MATCH the server's tag
// so `entities.has(tag)` lookups against a server-armed blocked/rerouted set
// hit. The server's canonical coercion (the server-side rule evaluator) is:
//
// groupByArr.map(field => resolveField(payload, field) || 'unknown').join('_')
//
// i.e. JS-falsy -> 'unknown' sentinel; ELSE the String()-coercion (via .join) of
// the kept value. We reimplement that exact 2-line coercion below as an
// INDEPENDENT ORACLE (assertion 15, option b) and assert the SDK output equals it
// across the FULL falsy+truthy fixture, then pin the resulting literals.
//
// RESOLVER-SEAM (assertion 5): every fixture value is delivered THROUGH a real
// payload — a present top-level field OR a metadata.* field — so the real
// resolver (`payloadField`) resolves it and feeds the coercion. We never call a
// coercion helper directly with a bare value.

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

// The server oracle — verbatim 2-line coercion from the server-side rule evaluator.
const collectorCoerce = (resolved: Any[]): string =>
  resolved.map((val) => val || "unknown").join("_");

interface Fixture {
  name: string;
  payload: Any;
  groupBy: string[];
  resolved: Any[]; // raw value(s) the resolver returns — fed to the oracle
  literal: string; // pinned expected tag (identical literal the Python test pins)
}

// SCALAR fixtures — Node + Python pin the IDENTICAL `literal`. Routed via a mix
// of top-level fields and metadata.* fields to exercise both resolver paths.
const scalarFixtures: Fixture[] = [
  // ── Half 1: JS-falsy -> "unknown" ──────────────────────────────────
  { name: "int 0 (top-level)", payload: { score: 0 }, groupBy: ["score"], resolved: [0], literal: "unknown" },
  { name: "float 0.0 (metadata)", payload: { metadata: { ratio: 0.0 } }, groupBy: ["ratio"], resolved: [0.0], literal: "unknown" },
  { name: "empty string (top-level)", payload: { name: "" }, groupBy: ["name"], resolved: [""], literal: "unknown" },
  { name: "false (top-level)", payload: { flag: false }, groupBy: ["flag"], resolved: [false], literal: "unknown" },
  { name: "false (metadata)", payload: { metadata: { is_active: false } }, groupBy: ["is_active"], resolved: [false], literal: "unknown" },
  { name: "NaN (top-level)", payload: { val: NaN }, groupBy: ["val"], resolved: [NaN], literal: "unknown" },
  { name: "negative zero (top-level)", payload: { z: -0 }, groupBy: ["z"], resolved: [-0], literal: "unknown" },
  { name: "null (top-level)", payload: { x: null }, groupBy: ["x"], resolved: [null], literal: "unknown" },
  { name: "absent key", payload: {}, groupBy: ["missing"], resolved: [null], literal: "unknown" },

  // ── Half 1: TRUTHY string edge values KEPT verbatim ────────────────
  { name: 'string "0" (top-level)', payload: { code: "0" }, groupBy: ["code"], resolved: ["0"], literal: "0" },
  { name: 'string "false" (metadata)', payload: { metadata: { label: "false" } }, groupBy: ["label"], resolved: ["false"], literal: "false" },
  { name: 'space " " (top-level)', payload: { s: " " }, groupBy: ["s"], resolved: [" "], literal: " " },

  // ── Half 2: JS-faithful stringification of the survivor ────────────
  { name: "bool true (top-level)", payload: { is_premium: true }, groupBy: ["is_premium"], resolved: [true], literal: "true" },
  { name: "bool true (metadata)", payload: { metadata: { is_premium: true } }, groupBy: ["is_premium"], resolved: [true], literal: "true" },
  { name: "float 1.0 (top-level)", payload: { m: 1.0 }, groupBy: ["m"], resolved: [1.0], literal: "1" },
  { name: "float 2.0 (metadata)", payload: { metadata: { n: 2.0 } }, groupBy: ["n"], resolved: [2.0], literal: "2" },
  { name: "float 10.0 (top-level)", payload: { m: 10.0 }, groupBy: ["m"], resolved: [10.0], literal: "10" },
  { name: "float 2.5 (top-level)", payload: { m: 2.5 }, groupBy: ["m"], resolved: [2.5], literal: "2.5" },
  { name: "plain string alice (top-level)", payload: { user: "alice" }, groupBy: ["user"], resolved: ["alice"], literal: "alice" },
  { name: "plain int 5 (top-level)", payload: { n: 5 }, groupBy: ["n"], resolved: [5], literal: "5" },

  // ── Multi-field joins (pinned literals) ────────────────────────────
  { name: "join (0, '') -> all unknown", payload: { a: 0, b: "" }, groupBy: ["a", "b"], resolved: [0, ""], literal: "unknown_unknown" },
  { name: "join ('alice', false) -> mixed", payload: { user: "alice", flag: false }, groupBy: ["user", "flag"], resolved: ["alice", false], literal: "alice_unknown" },
];

describe("GenerateGroupByTag — server oracle parity (scalar)", () => {
  for (const fx of scalarFixtures) {
    it(`${fx.name}: SDK == server oracle == pinned literal`, () => {
      const sdk = generateGroupByTag(fx.payload, fx.groupBy);
      const oracle = collectorCoerce(fx.resolved);
      // SDK output matches the independent server oracle...
      expect(sdk).toBe(oracle);
      // ...and the oracle equals the pinned literal (Python pins the same).
      expect(oracle).toBe(fx.literal);
    });
  }

  // empty groupBy -> "global" sentinel (unchanged behavior).
  it("empty groupBy -> 'global'", () => {
    expect(generateGroupByTag({ a: 1 }, [])).toBe("global");
  });
});

// COLLECTION fixtures — assertions 8 & 9. JS []/{} are TRUTHY so the server
// KEEPS them (not "unknown"). Node's String() yields ""/"[object Object]"; the
// exact byte-stringification of collections is a NAMED residual on the Python
// side (scope ii), so the Python test only asserts "not unknown" for these.
const collectionFixtures: Fixture[] = [
  { name: "empty array (metadata)", payload: { metadata: { tags: [] } }, groupBy: ["tags"], resolved: [[]], literal: "" },
  { name: "empty object (metadata)", payload: { metadata: { obj: {} } }, groupBy: ["obj"], resolved: [{}], literal: "[object Object]" },
];

describe("GenerateGroupByTag — empty collections KEPT (not 'unknown')", () => {
  for (const fx of collectionFixtures) {
    it(`${fx.name}: kept, matches server oracle`, () => {
      const sdk = generateGroupByTag(fx.payload, fx.groupBy);
      const oracle = collectorCoerce(fx.resolved);
      expect(sdk).not.toBe("unknown");
      expect(sdk).toBe(oracle);
      expect(oracle).toBe(fx.literal);
    });
  }
});

// Non-regression (rubric assertion f): after the flat-key resolver
// returns the PRESENT `null` (no longer skips to the metadata shadow), so a
// present-null groupBy field now collapses to the server-aligned "unknown"
// sentinel (resolver→null→ `val || 'unknown'`), NOT the old "pro". This is a
// CONVERGENCE with the server (resolveField→null, then `val || 'unknown'`),
// not a regression. See the flat-key resolver regression notes + the group-by tag regression notes.
describe("Present-null groupBy field collapses to 'unknown' (server-aligned)", () => {
  it("present-null + metadata shadow -> 'unknown' (not the shadowed 'pro')", () => {
    const payload = { paid_plan: null, metadata: { paid_plan: "pro" } };
    const sdk = generateGroupByTag(payload, ["paid_plan"]);
    // After the fix the resolver yields `null`; the server oracle over that
    // resolved value is `null || 'unknown'` → 'unknown'.
    const oracle = collectorCoerce([null]);
    expect(sdk).toBe(oracle);
    expect(sdk).toBe("unknown");
    expect(sdk).not.toBe("pro");
  });
});
