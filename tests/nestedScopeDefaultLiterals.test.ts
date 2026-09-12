/**
 * M7 regression — a nested session()/agent()/chain()/workflow() scope must be
 * able to EXPLICITLY set the literal default strings "default_workflow" /
 * "anonymous" / "free" and have them OVERRIDE the parent.
 *
 * Defect (pre-fix, src/context.ts `_sessionImpl` nested branch): inheritance
 * was decided by comparing the provided value against the literal default
 * (`options.userId && options.userId !== "anonymous" ? ...: existing.userId`),
 * so a nested scope could NEVER set exactly those three values — the parent's
 * value silently won for those literals.
 *
 * Fix: the nested branch decides inheritance on provided-ness (truthiness)
 * only — `options.userId || existing.userId`. Empty-string still behaves as
 * "unset" (unchanged); the ONLY behavior change is that the three default
 * literals become settable in a nested scope.
 */
import { describe, it, expect } from "vitest";
import { session, agent, chain, workflow, getCurrentSession } from "../src/context";

describe("M7 — nested scope can set the literal default strings", () => {
  // ── Case 1: nested explicitly setting the default literals OVERRIDES ──────
  it("nested session setting the default literals overrides the parent", () => {
    session({ name: "outer", userId: "user_1", paidPlan: "pro" }, () => {
      session(
        { name: "default_workflow", userId: "anonymous", paidPlan: "free" },
        (inner) => {
          expect(inner.userId).toBe("anonymous");
          expect(inner.paidPlan).toBe("free");
          expect(inner.workflowName).toBe("default_workflow");
          expect(getCurrentSession().userId).toBe("anonymous");
        },
      );
    });
  });

  it("override works across agent()/chain() entry points too", () => {
    agent({ name: "outer", userId: "user_1", paidPlan: "pro" }, () => {
      chain(
        { name: "default_workflow", userId: "anonymous", paidPlan: "free" },
        (inner) => {
          expect(inner.userId).toBe("anonymous");
          expect(inner.paidPlan).toBe("free");
          expect(inner.workflowName).toBe("default_workflow");
        },
      );
    });
  });

  // ── Case 2: nested OMITTING them INHERITS the parent (unchanged) ──────────
  it("nested session omitting fields inherits the parent", () => {
    session({ name: "outer", userId: "user_1", paidPlan: "pro" }, () => {
      session({ name: "inner" }, (inner) => {
        expect(inner.workflowName).toBe("inner");
        expect(inner.userId).toBe("user_1");
        expect(inner.paidPlan).toBe("pro");
      });
    });
  });

  it("empty-string still behaves as unset (inherits parent) — unchanged", () => {
    session({ name: "outer", userId: "user_1", paidPlan: "pro" }, () => {
      session({ name: "", userId: "", paidPlan: "" }, (inner) => {
        expect(inner.workflowName).toBe("outer");
        expect(inner.userId).toBe("user_1");
        expect(inner.paidPlan).toBe("pro");
      });
    });
  });

  // ── Case 3: root-scope defaults unchanged when nothing is provided ────────
  it("root-scope defaults are unchanged", () => {
    session({}, (s) => {
      expect(s.userId).toBe("anonymous");
      expect(s.paidPlan).toBe("free");
      expect(s.workflowName).toBe("default_workflow");
    });
  });

  it("root-scope explicit values are unchanged", () => {
    session({ name: "wf", userId: "u1", paidPlan: "pro" }, (s) => {
      expect(s.userId).toBe("u1");
      expect(s.paidPlan).toBe("pro");
      expect(s.workflowName).toBe("wf");
    });
  });

  // ── workflow() parity: nested workflow can set the literals ───────────────
  it("nested workflow() can set the default literals", () => {
    const inner = workflow(
      { name: "default_workflow", userId: "anonymous", paidPlan: "free" },
      () => {
        const s = getCurrentSession();
        return [s.userId, s.paidPlan, s.workflowName];
      },
    );
    session({ name: "outer", userId: "user_1", paidPlan: "pro" }, () => {
      expect(inner()).toEqual(["anonymous", "free", "default_workflow"]);
    });
  });

  it("nested workflow() omitting fields inherits the parent", () => {
    const inner = workflow({ name: "inner_wf" }, () => {
      const s = getCurrentSession();
      return [s.userId, s.paidPlan];
    });
    session({ name: "outer", userId: "user_1", paidPlan: "pro" }, () => {
      expect(inner()).toEqual(["user_1", "pro"]);
    });
  });
});
