import { describe, it, expect } from "vitest";
import { evaluate } from "../src/localEvaluator";

// Canonical total order for directives — priority ascending
// (missing/non-numeric → 100), then id ascending by JS UTF-16 code-UNIT order
// (== Python code-POINT order over the ASCII/BMP lowercase-alnum CUID id space).
// These tests pin the SDK side; the Python sibling test_directive_ordering.py
// pins the SAME literal fixture so Node == Python on a real locale≠code-point case.

// The evaluator sorts pack.directives internally. We probe the resulting order by
// making EVERY directive an UNCONDITIONAL_BLOCK in dry_run mode that matches the
// (empty) payload — each emits one would_block observation in evaluation order,
// so the observation rule_id sequence reveals the canonical sort.
const blockPackFromIds = (ids: string[], priority = 10) => ({
  directives: ids.map((id) => ({
    id,
    kind: "UNCONDITIONAL_BLOCK",
    mode: "dry_run",
    priority,
    selector: { match: null, group_by: [] },
  })),
});

const orderedIds = (ids: string[], priority = 10): string[] => {
  const res = evaluate(blockPackFromIds(ids, priority), {}, {});
  return res.observations.map((o) => o.rule_id);
};

describe("Directive canonical ordering (Node)", () => {
  it("LITERAL parity fixture sorts by code-point, not locale", () => {
    // locale collation would put 'a' first and be case-insensitive; code-point
    // puts digits < uppercase < lowercase: '1'(0x31) '9'(0x39) 'B'(0x42) 'Z'(0x5A) 'a'(0x61).
    expect(orderedIds(["Z", "a", "10", "9", "Bb"])).toEqual(["10", "9", "Bb", "Z", "a"]);
  });

  it("input array order does not affect the canonical result (shuffled == sorted)", () => {
    expect(orderedIds(["a", "9", "Z", "Bb", "10"])).toEqual(["10", "9", "Bb", "Z", "a"]);
  });

  it("priority ascending dominates the id tie-break", () => {
    const pack = {
      directives: [
        { id: "aaa", kind: "UNCONDITIONAL_BLOCK", mode: "dry_run", priority: 50, selector: { match: null, group_by: [] } },
        { id: "zzz", kind: "UNCONDITIONAL_BLOCK", mode: "dry_run", priority: 10, selector: { match: null, group_by: [] } },
      ],
    };
    const res = evaluate(pack, {}, {});
    expect(res.observations.map((o) => o.rule_id)).toEqual(["zzz", "aaa"]);
  });

  it("missing/non-numeric priority is treated as 100", () => {
    const pack = {
      directives: [
        { id: "aaa", kind: "UNCONDITIONAL_BLOCK", mode: "dry_run", selector: { match: null, group_by: [] } }, // → 100
        { id: "bbb", kind: "UNCONDITIONAL_BLOCK", mode: "dry_run", priority: 5, selector: { match: null, group_by: [] } },
      ],
    };
    const res = evaluate(pack, {}, {});
    expect(res.observations.map((o) => o.rule_id)).toEqual(["bbb", "aaa"]);
  });

  it("equal-priority UNCONDITIONAL_BLOCK tie — enforce picks canonical winner aaa regardless of array order", () => {
    const mk = (ids: string[]) => ({
      directives: ids.map((id) => ({
        id,
        kind: "UNCONDITIONAL_BLOCK",
        mode: "enforce",
        priority: 10,
        selector: { match: null, group_by: [] },
      })),
    });
    for (const order of [["aaa", "bbb"], ["bbb", "aaa"]]) {
      const res = evaluate(mk(order), {}, {});
      expect(res.decision.status).toBe("blocked");
      expect(res.decision.rule_id).toBe("aaa");
    }
  });

  it("equal-priority REROUTE tie — enforce picks canonical winner aaa's model regardless of array order", () => {
    const mk = (ids: string[]) => ({
      directives: ids.map((id) => ({
        id,
        kind: "REROUTE",
        mode: "enforce",
        priority: 10,
        selector: { match: null, group_by: [] },
        reroute: { from: null, to: { provider: "openai", model: id === "aaa" ? "gpt-aaa" : "gpt-bbb" } },
      })),
    });
    for (const order of [["aaa", "bbb"], ["bbb", "aaa"]]) {
      const res = evaluate(mk(order), {}, { provider: "openai", model: "gpt-4" });
      expect(res.decision.status).toBe("rerouted");
      expect(res.decision.rule_id).toBe("aaa");
      expect(res.decision.reroute?.to.model).toBe("gpt-aaa");
    }
  });
});
