/**
 * matchesCondition AND/OR + full-operator tests. Must stay byte-compatible with
 * the server-side rule evaluator matcher and the Python local_evaluator — see
 * match_condition_boolean_composition_design.md.
 */
import { describe, it, expect } from "vitest";
import { matchesCondition, evaluate } from "../src/localEvaluator";

const payload = { paid_plan: "free", model: "gpt-4", provider: "openai", metadata: { team: "x" } };

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

describe("matchesCondition — shapes", () => {
  it("empty / null match everything", () => {
    expect(matchesCondition(payload, null)).toBe(true);
    expect(matchesCondition(payload, {})).toBe(true);
  });

  it("bare leaf EQ", () => {
    expect(matchesCondition(payload, { field: "paid_plan", operator: "EQ", value: "free" })).toBe(true);
    expect(matchesCondition(payload, { field: "paid_plan", operator: "EQ", value: "pro" })).toBe(false);
  });
});

describe("matchesCondition — full operator set", () => {
  it("NEQ (absent field => true)", () => {
    expect(matchesCondition(payload, { field: "paid_plan", operator: "NEQ", value: "pro" })).toBe(true);
    expect(matchesCondition(payload, { field: "paid_plan", operator: "NEQ", value: "free" })).toBe(false);
    expect(matchesCondition(payload, { field: "absent", operator: "NEQ", value: "pro" })).toBe(true);
  });
  it("CONTAINS", () => {
    expect(matchesCondition(payload, { field: "provider", operator: "CONTAINS", value: "open" })).toBe(true);
    expect(matchesCondition(payload, { field: "provider", operator: "CONTAINS", value: "zzz" })).toBe(false);
  });
  it("IN (non-array never matches)", () => {
    expect(matchesCondition(payload, { field: "model", operator: "IN", value: ["gpt-4", "gpt-4o"] })).toBe(true);
    expect(matchesCondition(payload, { field: "model", operator: "IN", value: ["claude"] })).toBe(false);
    expect(matchesCondition(payload, { field: "model", operator: "IN", value: "gpt-4" })).toBe(false);
  });
  it("EXISTS (empty string counts as absent, matching SQL)", () => {
    expect(matchesCondition(payload, { field: "paid_plan", operator: "EXISTS" })).toBe(true);
    expect(matchesCondition(payload, { field: "session_id", operator: "EXISTS" })).toBe(false);
    expect(matchesCondition({ ...payload, session_id: "" }, { field: "session_id", operator: "EXISTS" })).toBe(false);
  });
  it("unknown operator fails closed", () => {
    expect(matchesCondition(payload, { field: "model", operator: "LIKE", value: "gpt" })).toBe(false);
  });
});

describe("matchesCondition — composite AND/OR", () => {
  it("AND requires all", () => {
    expect(matchesCondition(payload, { combinator: "AND", conditions: [
      { field: "paid_plan", operator: "EQ", value: "free" },
      { field: "model", operator: "IN", value: ["gpt-4"] },
    ] })).toBe(true);
    expect(matchesCondition(payload, { combinator: "AND", conditions: [
      { field: "paid_plan", operator: "EQ", value: "free" },
      { field: "model", operator: "EQ", value: "claude" },
    ] })).toBe(false);
  });
  it("OR requires one (metadata field resolves)", () => {
    expect(matchesCondition(payload, { combinator: "OR", conditions: [
      { field: "paid_plan", operator: "EQ", value: "pro" },
      { field: "metadata.team", operator: "EQ", value: "x" },
    ] })).toBe(true);
    expect(matchesCondition(payload, { combinator: "OR", conditions: [
      { field: "paid_plan", operator: "EQ", value: "pro" },
      { field: "metadata.team", operator: "EQ", value: "y" },
    ] })).toBe(false);
  });
  it("defaults to AND; empty AND true, empty OR false", () => {
    expect(matchesCondition(payload, { conditions: [{ field: "paid_plan", operator: "EQ", value: "free" }] })).toBe(true);
    expect(matchesCondition(payload, { combinator: "AND", conditions: [] })).toBe(true);
    expect(matchesCondition(payload, { combinator: "OR", conditions: [] })).toBe(false);
  });
});

// ── flat-key resolver must NOT treat a present-`null` as absent ──────
// The authoritative server flat-key resolver (the server-side rule evaluator) is:
// payload[field] !== undefined ? payload[field]: (payload.metadata && payload.metadata[field])
// i.e. a PRESENT key (even value `null`) WINS; only a truly-ABSENT key falls to
// the metadata bag. Before the SDK additionally treated a present-`null` as
// absent and skipped to metadata, so a present-`null` shadowed in metadata
// resolved to the metadata value. This block reimplements the server flat-key
// oracle INLINE and drives every oracle-equality assertion through the PUBLIC
// `matchesCondition` entry point — `payloadField` is intentionally NOT exported,
// so we infer the resolved value with EQ/EXISTS/NEQ leaf probes.
//
// Node↔Python asymmetry (stated verbatim in tests/test_local_matcher.py too):
// Python's `_payload_field` IS module-importable, so the Python twin asserts the
// resolved value DIRECTLY; Node infers it via public probes. The parity assertion
// (rubric 8) compares the RESOLVED VALUE, not the invocation mechanism.

const MISSING = Symbol("missing");

// Inline server flat-key resolver (the server-side rule evaluator). Present (incl.
// `null`) wins; only truly-absent falls to the metadata bag.
const resolveFlatOracle = (p: Any, field: string): Any => {
  if (p && field in p) return p[field];
  const md = p && p.metadata;
  if (md && field in md) return md[field];
  return MISSING;
};

// Server matchesLeaf semantics over an already-resolved value
// (the server-side rule evaluator). The MISSING sentinel is treated as absent
// (`undefined`) — indistinguishable from the SDK's null-for-absent under EQ
// (non-null) / NEQ / EXISTS, the only operators probed on absent cases.
const leafOracle = (resolved: Any, op: string, value: Any): boolean => {
  const v = resolved === MISSING ? undefined : resolved;
  if (op === "EQ") return v === value;
  if (op === "NEQ") return v !== value;
  if (op === "EXISTS") return v !== undefined && v !== null && v !== "";
  return false;
};

interface Probe { op: string; value?: Any; }
interface F27Case {
  name: string;
  payload: Any;
  field: string;
  // Pinned RESOLVED VALUE (for cross-SDK parity, rubric 8). `MISSING` == the
  // server's `undefined`; the SDK returns `null` for that absent case, which
  // is equivalent under the probed operators.
  resolvedValue: Any;
  probes: Probe[];
}

const f27Cases: F27Case[] = [
  {
    // (a) + (a-NEQ, rubric 13): present-`null` + metadata shadow → `null`, NOT "pro".
    name: "(a) present-null + metadata shadow -> null (not the metadata 'pro')",
    payload: { paid_plan: null, metadata: { paid_plan: "pro" } },
    field: "paid_plan",
    resolvedValue: null,
    probes: [
      { op: "EQ", value: "pro" }, // false — did NOT resolve to metadata "pro"
      { op: "EXISTS" },           // false — resolved null is absent to matchesLeaf
      { op: "EQ", value: null },  // true — resolver returned null
      { op: "NEQ", value: "pro" },// true — resolved null flows into NEQ (13)
    ],
  },
  {
    // (b) present-non-null → unchanged; metadata never consulted.
    name: "(b) present-non-null + shadow -> 'free' (unchanged)",
    payload: { paid_plan: "free", metadata: { paid_plan: "pro" } },
    field: "paid_plan",
    resolvedValue: "free",
    probes: [
      { op: "EQ", value: "free" },
      { op: "EQ", value: "pro" },
      { op: "NEQ", value: "pro" },
      { op: "EXISTS" },
    ],
  },
  {
    // (c) truly-absent + metadata present → still falls back to metadata.
    name: "(c) absent + metadata present -> 'cli' (metadata fallback intact)",
    payload: { metadata: { source: "cli" } },
    field: "source",
    resolvedValue: "cli",
    probes: [
      { op: "EQ", value: "cli" },
      { op: "EXISTS" },
      { op: "NEQ", value: "cli" },
    ],
  },
  {
    // (d) truly-absent + no metadata → absent (SDK null / server undefined).
    name: "(d) absent + no metadata -> absent",
    payload: {},
    field: "paid_plan",
    resolvedValue: MISSING,
    probes: [
      { op: "EXISTS" },
      { op: "EQ", value: "x" },
      { op: "NEQ", value: "x" },
    ],
  },
  {
    // (e) present-`null` + NO metadata shadow → null, identical before & after.
    name: "(e) present-null + no shadow -> null (regression guard)",
    payload: { paid_plan: null },
    field: "paid_plan",
    resolvedValue: null,
    probes: [
      { op: "EXISTS" },
      { op: "EQ", value: null },
      { op: "NEQ", value: "pro" },
    ],
  },
];

describe("Flat-key resolver — server oracle parity via public matchesCondition", () => {
  for (const c of f27Cases) {
    it(`${c.name}: every probe == inline server oracle`, () => {
      const resolved = resolveFlatOracle(c.payload, c.field);
      for (const pr of c.probes) {
        const leaf = { field: c.field, operator: pr.op, value: pr.value };
        const expected = leafOracle(resolved, pr.op, pr.value);
        expect(matchesCondition(c.payload, leaf)).toBe(expected);
      }
    });
  }

  // Headline (a): explicit pinned expectations (rubric 1 + 13), independent of
  // the oracle loop so a regression is unambiguous.
  it("(a) present-null+shadow: EQ 'pro'→false, EXISTS→false, EQ null→true, NEQ 'pro'→true", () => {
    const p = { paid_plan: null, metadata: { paid_plan: "pro" } };
    expect(matchesCondition(p, { field: "paid_plan", operator: "EQ", value: "pro" })).toBe(false);
    expect(matchesCondition(p, { field: "paid_plan", operator: "EXISTS" })).toBe(false);
    expect(matchesCondition(p, { field: "paid_plan", operator: "EQ", value: null })).toBe(true);
    expect(matchesCondition(p, { field: "paid_plan", operator: "NEQ", value: "pro" })).toBe(true);
  });

  // (rubric 8) Cross-SDK parity on the RESOLVED VALUE, inferred through public
  // probes. `MISSING` → EXISTS false; a concrete value → EQ that-value true.
  it("(g) resolved value matches the pinned parity table (inferred via public probes)", () => {
    for (const c of f27Cases) {
      if (c.resolvedValue === MISSING) {
        expect(matchesCondition(c.payload, { field: c.field, operator: "EXISTS" })).toBe(false);
      } else {
        expect(matchesCondition(c.payload, { field: c.field, operator: "EQ", value: c.resolvedValue })).toBe(true);
      }
    }
  });

  // (rubric 11) Dotted-leaf branch is untouched — resolves exactly as before.
  it("dotted-leaf resolution unchanged (out of scope for )", () => {
    expect(matchesCondition({ intent: { kind: "code" } }, { field: "intent.kind", operator: "EQ", value: "code" })).toBe(true);
    expect(matchesCondition({ intent: { kind: "code" } }, { field: "intent.kind", operator: "EQ", value: "chat" })).toBe(false);
  });
});

// ── metadata-vs-canonical precedence in the payload BUILDER ─────────
// buildPayload must mirror the server's evaluationPayload spread/field order
// (the remote /check evaluation payload): `end_user_id` + `metadata` sit
// BEFORE the `...md` spread (metadata CAN override them); every other canonical
// field sits AFTER the spread (canonical WINS over a metadata bag key of the same
// name). Pre-fix the SDK spread `...md` LAST, so metadata overrode EVERY canonical
// field — the opposite direction from the server (State-A ≠ State-B).
//
// Node↔Python ASYMMETRY (stated verbatim in tests/test_local_matcher.py too):
// `buildPayload` is NOT exported (mirror the precedent — no new public
// export). So Node probes the RESOLVED value through the PUBLIC `evaluate` entry
// point with discriminating twin directives: an enforce-mode UNCONDITIONAL_BLOCK
// whose match is `{field, EQ, value:C}` BLOCKS iff the builder resolved <field> to
// C; its twin `value:M` BLOCKS iff it resolved to M. The Python twin imports
// `_build_payload` and reads the top-level key directly. The parity assertion
// compares the RESOLVED VALUE, not the invocation mechanism.
describe("Metadata-vs-canonical precedence (builder mirrors server spread order)", () => {
  // Enforce-mode UNCONDITIONAL_BLOCK on one leaf: blocks iff the builder
  // resolved <field> to <value>.
  const blockPack = (field: string, value: Any) => ({
    directives: [
      {
        id: "d1",
        kind: "UNCONDITIONAL_BLOCK",
        mode: "enforce",
        priority: 10,
        selector: { match: { field, operator: "EQ", value } },
      },
    ],
  });
  const blocks = (pack: Any, session: Any, observed: Any): boolean =>
    evaluate(pack, session, observed).decision.status === "blocked";

  // Helper: assert <field> resolved to `canonical` (blocks) and NOT to `meta`
  // (twin allows) — the canonical-wins truth-table row.
  const expectCanonicalWins = (
    session: Any,
    observed: Any,
    field: string,
    canonical: Any,
    meta: Any,
  ) => {
    expect(blocks(blockPack(field, canonical), session, observed)).toBe(true);
    expect(blocks(blockPack(field, meta), session, observed)).toBe(false);
  };

  it("(5) provider both-present -> canonical wins", () => {
    const session = { user_id: "u", metadata: { provider: "meta_prov" } };
    const observed = { provider: "canon_prov" };
    expectCanonicalWins(session, observed, "provider", "canon_prov", "meta_prov");
  });

  it("(6) model both-present -> canonical wins", () => {
    const session = { user_id: "u", metadata: { model: "meta_model" } };
    const observed = { model: "canon_model" };
    expectCanonicalWins(session, observed, "model", "canon_model", "meta_model");
  });

  it("(7) user_id both-present -> canonical wins", () => {
    const session = { user_id: "u_canon", metadata: { user_id: "meta_user" } };
    const observed = {};
    expectCanonicalWins(session, observed, "user_id", "u_canon", "meta_user");
  });

  it("(8) trace_id both-present -> canonical wins", () => {
    const session = { user_id: "u", metadata: { trace_id: "meta_trace" } };
    const observed = { trace_id: "canon_trace" };
    expectCanonicalWins(session, observed, "trace_id", "canon_trace", "meta_trace");
  });

  it("(9) session_id both-present -> canonical wins", () => {
    const session = { user_id: "u", session_id: "sess_canon", metadata: { session_id: "meta_sess" } };
    const observed = {};
    expectCanonicalWins(session, observed, "session_id", "sess_canon", "meta_sess");
  });

  it("(10) end_user_id asymmetry — metadata STILL wins while user_id is canonical (same bag)", () => {
    // One payload: metadata carries BOTH end_user_id (before the spread -> metadata
    // wins) and user_id (after the spread -> canonical wins), pinning the
    // before/after-spread asymmetry in a single scenario (server:113 vs:117).
    const session = { user_id: "u_canon", metadata: { end_user_id: "meta_eu", user_id: "meta_user" } };
    const observed = {};
    // end_user_id -> metadata "meta_eu" (BEFORE spread): the "meta_eu" twin blocks,
    // the canonical "u_canon" twin allows.
    expect(blocks(blockPack("end_user_id", "meta_eu"), session, observed)).toBe(true);
    expect(blocks(blockPack("end_user_id", "u_canon"), session, observed)).toBe(false);
    // user_id -> canonical "u_canon" (AFTER spread).
    expectCanonicalWins(session, observed, "user_id", "u_canon", "meta_user");
  });

  it("(11) paid_plan / modality both-present -> net canonical at top level", () => {
    const session = { user_id: "u", paid_plan: "free", metadata: { paid_plan: "meta_plan", modality: "meta_mode" } };
    const observed = { modality: "embedding" };
    expectCanonicalWins(session, observed, "paid_plan", "free", "meta_plan");
    expectCanonicalWins(session, observed, "modality", "embedding", "meta_mode");
  });

  it("(12) meta-only non-reserved key still surfaces via the spread", () => {
    const session = { user_id: "u", metadata: { source: "cli" } };
    const observed = {};
    expect(blocks(blockPack("source", "cli"), session, observed)).toBe(true);
  });

  it("(13) canonical-only key unchanged (no metadata shadow)", () => {
    const session = { user_id: "u", metadata: {} };
    const observed = { provider: "canon_prov" };
    expect(blocks(blockPack("provider", "canon_prov"), session, observed)).toBe(true);
  });
});

// The SDK builder now emits a canonical `operation` field (= the resolved
// `modality`) after the `...md` spread, between `modality` and `intent`, mirroring
// the server's evaluationPayload (the remote /check evaluation payload).
// Pre-fix the builder emitted NO `operation` key, so a rule `operation EQ "chat"`
// resolved to null on the warm-local path (State A) while the inline /check path
// (State B) resolved it to a canonical string — the one divergence scoped out.
//
// Node<->Python ASYMMETRY: `buildPayload` is NOT exported (/ precedent —
// no new public export), so Node probes the RESOLVED `operation` through the PUBLIC
// `evaluate` entry with an enforce-mode UNCONDITIONAL_BLOCK whose match is
// `{field:"operation", EQ, value:X}` — it BLOCKS iff the builder resolved
// `operation` to X. Python reads `_build_payload(...)["operation"]` directly. The
// parity assertion compares the RESOLVED VALUE, not the invocation mechanism.
describe("Canonical operation field (builder mirrors server remote /check evaluation)", () => {
  const blockPack = (field: string, value: Any) => ({
    directives: [
      {
        id: "d1",
        kind: "UNCONDITIONAL_BLOCK",
        mode: "enforce",
        priority: 10,
        selector: { match: { field, operator: "EQ", value } },
      },
    ],
  });
  const blocks = (pack: Any, session: Any, observed: Any): boolean =>
    evaluate(pack, session, observed).decision.status === "blocked";

  it("(F60-1) default operation is 'chat' and matches operation EQ chat", () => {
    // No modality and no intent.kind -> operation defaults to "chat" (server
    // remote /check evaluation:110 default). RED pre-fix (no key -> null), GREEN post-fix.
    const session = { user_id: "u", metadata: {} };
    const observed = {};
    expect(blocks(blockPack("operation", "chat"), session, observed)).toBe(true);
    expect(blocks(blockPack("operation", "embedding"), session, observed)).toBe(false);
  });

  it("(F60-2) observed.modality drives operation (operation === modality)", () => {
    const session = { user_id: "u", metadata: {} };
    const observed = { modality: "embedding" };
    expect(blocks(blockPack("operation", "embedding"), session, observed)).toBe(true);
    expect(blocks(blockPack("operation", "chat"), session, observed)).toBe(false);
    // twin: the `modality` field resolves to the SAME value (SDK operation===modality)
    expect(blocks(blockPack("modality", "embedding"), session, observed)).toBe(true);
  });

  it("(F60-3) intent.kind fallback drives operation when modality is unset", () => {
    const session = { user_id: "u", metadata: {} };
    const observed = { intent: { kind: "embedding" } };
    expect(blocks(blockPack("operation", "embedding"), session, observed)).toBe(true);
    expect(blocks(blockPack("operation", "chat"), session, observed)).toBe(false);
  });

  it("(F60-4) metadata cannot override the canonical operation (after the spread)", () => {
    // metadata bag carries operation:"embedding" but modality is unset -> canonical
    // operation resolves to "chat" (default) and sits AFTER the ...md spread, so the
    // "chat" twin BLOCKS and the "embedding" twin ALLOWS. Mirrors server:123>:115.
    const session = { user_id: "u", metadata: { operation: "embedding" } };
    const observed = {};
    expect(blocks(blockPack("operation", "chat"), session, observed)).toBe(true);
    expect(blocks(blockPack("operation", "embedding"), session, observed)).toBe(false);
  });
});

// ── JS coercion parity — the reference the Python twin is matched against ──
// This SDK's matchesLeaf already uses JS ===/String() (byte-compatible with the
// server-side evaluator), so no src change is needed. This block is the PARITY
// PROOF: the SAME divergence matrix asserted in tests/test_local_matcher.py's
// TestJsCoercionParity, pinning the SERVER's answer so the two SDKs stay twins.
// The divergences (True vs 1, str(True)="True" vs String(true)="true",
// str(100.0)="100.0" vs String(100.0)="100") are where the loose-Python SDK could
// locally ALLOW a call the server BLOCKS — the enforcement hole this closes.
//
// Node<->Python ASYMMETRY (stated verbatim in tests/test_local_matcher.py too):
// the String()/=== analogs are module-importable in Python (spot-checked there
// directly) but NOT exported here, so Node drives the identical matrix through the
// public `matchesCondition`. The parity assertion compares the RESOLVED bool.
describe("JS coercion parity — matchesLeaf ===/String() (Python twin: TestJsCoercionParity)", () => {
  // Bare leaf on field "f"; `payloadVal` sits as the payload-side value.
  const leaf = (op: string, value: Any, payloadVal: Any): boolean =>
    matchesCondition({ f: payloadVal }, { field: "f", operator: op, value });

  it("EQ — JS strict semantics", () => {
    expect(leaf("EQ", 1, true)).toBe(false);        // true !== 1
    expect(leaf("EQ", true, true)).toBe(true);
    expect(leaf("EQ", 1, 1.0)).toBe(true);          // one JS number type
    expect(leaf("EQ", 1, "1")).toBe(false);         // string vs number
    expect(leaf("EQ", "true", true)).toBe(false);   // bool vs its string form
    // absent field resolves to null; EQ null => null === null.
    expect(matchesCondition({}, { field: "f", operator: "EQ", value: null })).toBe(true);
  });

  it("NEQ — mirrors EQ", () => {
    expect(leaf("NEQ", 1, true)).toBe(true);
    expect(leaf("NEQ", true, true)).toBe(false);
    expect(leaf("NEQ", 1, 1.0)).toBe(false);
    expect(leaf("NEQ", 1, "1")).toBe(true);
    expect(leaf("NEQ", "true", true)).toBe(true);
  });

  it("IN — JS strict membership", () => {
    expect(leaf("IN", [1], true)).toBe(false);
    expect(leaf("IN", [true], true)).toBe(true);
    expect(leaf("IN", [1.0], 1)).toBe(true);
    expect(leaf("IN", ["a"], "a")).toBe(true);
    expect(leaf("IN", "a", "a")).toBe(false);       // non-array never matches
  });

  it("CONTAINS — JS String() coercion", () => {
    expect(leaf("CONTAINS", "true", true)).toBe(true);   // enforcement-hole case
    expect(leaf("CONTAINS", "True", true)).toBe(false);  // String(true) is "true"
    expect(leaf("CONTAINS", ".0", 100.0)).toBe(false);   // String(100.0) is "100"
    expect(leaf("CONTAINS", "100", 100.0)).toBe(true);
    expect(leaf("CONTAINS", "0.00001", 1e-5)).toBe(true); // never exponential
    expect(leaf("CONTAINS", "1e-05", 1e-5)).toBe(false);
    expect(leaf("CONTAINS", "Infinity", Infinity)).toBe(true);
    expect(leaf("CONTAINS", "NaN", NaN)).toBe(true);
  });

  it("EXISTS — behavior unchanged", () => {
    expect(matchesCondition({ f: "" }, { field: "f", operator: "EXISTS" })).toBe(false);
    expect(matchesCondition({ f: 0 }, { field: "f", operator: "EXISTS" })).toBe(true);
    expect(matchesCondition({ f: false }, { field: "f", operator: "EXISTS" })).toBe(true);
  });

  it("no-throw for exotic payload values (returns a bool, never throws)", () => {
    const exotic: Any[] = [{ k: "v" }, ["a", "b"], () => 0];
    const ops: Array<[string, Any]> = [
      ["EQ", "x"], ["EQ", 1], ["EQ", null],
      ["NEQ", "x"], ["EXISTS", undefined],
      ["CONTAINS", "x"], ["CONTAINS", {}],
      ["IN", ["x"]], ["IN", "x"],
    ];
    for (const val of exotic) {
      for (const [op, value] of ops) {
        expect(typeof matchesCondition({ f: val }, { field: "f", operator: op, value })).toBe("boolean");
      }
    }
  });
});
