/**
 * Harmonize the two SDKs' exception-classifier chain traversal to their
 * union. Node gains a BFS over the full wrapper-attr set (cause/inner/original/
 * originalError/original_error/original_exception), depth-capped (4) and
 * cycle-safe, plus message-parse over the chained links (top-first). These
 * tests exercise the PUBLIC `classifyException` surface only.
 *
 * `_classify` is pure telemetry (drives call_outcome.{error_kind,http_status}
 * on /log). It must stay TOTAL / NO-THROW — never into customer code.
 */
import { describe, it, expect } from "vitest";
import { classifyException } from "../src/_classify";

describe("ClassifyException — union chain traversal", () => {
  // A1 — status carried ONLY on a non-`.cause` wrapper attr is now recovered.
  it("recovers status from `err.inner` (not `.cause`)", () => {
    const inner: any = new Error("provider rate limited");
    inner.status_code = 429;
    const top: any = new Error("stream iteration failed");
    top.inner = inner;
    expect(classifyException(top)).toEqual({ error_kind: "rate_limited", http_status: 429 });
  });

  it("recovers status from `err.original`", () => {
    const orig: any = new Error("provider bad request");
    orig.status_code = 400;
    const top: any = new Error("wrapper");
    top.original = orig;
    expect(classifyException(top)).toEqual({ error_kind: "client_error", http_status: 400 });
  });

  // A9 — snake_case `http_status` direct attr (Python-shaped) now detected on
  // Node, closing the last direct-attr asymmetry. RED before adding
  // "http_status" to _directStatusOf's attr array (was unknown/0).
  it("recovers status from a top-level `http_status` attr (snake_case)", () => {
    const e: any = new Error("provider rate limited");
    e.http_status = 429;
    expect(classifyException(e)).toEqual({ error_kind: "rate_limited", http_status: 429 });
  });

  it("recovers status from `err.originalError` (JS camel variant)", () => {
    const orig: any = new Error("provider unauthorized");
    orig.status = 401;
    const top: any = new Error("wrapper");
    top.originalError = orig;
    expect(classifyException(top)).toEqual({ error_kind: "auth_error", http_status: 401 });
  });

  // A2 — a 429 encoded in the MESSAGE of a wrapper link is recovered top-first.
  it("parses a message-encoded status on a wrapper link", () => {
    const top: any = new Error("stream failed");
    top.inner = new Error("HTTP 429 rate limited");
    expect(classifyException(top)).toEqual({ error_kind: "rate_limited", http_status: 429 });
  });

  // A3 — cyclic wrapper input is TOTAL: the test completes (no hang) and the
  // result is a defined unknown/0. (Completeness/no-throw case; the Set-guard
  // itself is a code-read per rubric A13.)
  it("terminates on a cyclic wrapper (A.cause=B, B.inner=A)", () => {
    const a: any = new Error("a");
    const b: any = new Error("b");
    a.cause = b;
    b.inner = a;
    expect(classifyException(a)).toEqual({ error_kind: "unknown", http_status: 0 });
  });

  // A4 — depth cap (maxDepth=4): a status carried only on the 5th link is NOT
  // reached, so the classifier returns unknown/0 (bounded traversal).
  it("honors the depth cap: a status on the 5th link is excluded", () => {
    const l5: any = new Error("deep");
    l5.status_code = 429;
    const l4: any = new Error("l4", { cause: l5 });
    const l3: any = new Error("l3", { cause: l4 });
    const l2: any = new Error("l2", { cause: l3 });
    const l1: any = new Error("l1", { cause: l2 });
    const top: any = new Error("top", { cause: l1 });
    expect(classifyException(top)).toEqual({ error_kind: "unknown", http_status: 0 });
  });

  // A14 — per-attr guarding: a throwing wrapper getter on one attr must NOT
  // abort scanning sibling attrs on the same node.
  it("per-attr guard: throwing `inner` getter still lets `original` resolve", () => {
    const original: any = new Error("provider rate limited");
    original.status_code = 429;
    const top: any = new Error("wrapper");
    Object.defineProperty(top, "inner", {
      get() {
        throw new Error("hostile getter");
      },
      enumerable: true,
      configurable: true,
    });
    top.original = original;
    expect(classifyException(top)).toEqual({ error_kind: "rate_limited", http_status: 429 });
  });

  // A8 — no-throw on hostile inputs.
  it("never throws on hostile inputs", () => {
    const inputs: any[] = [null, undefined, "a plain string", 12345, {}, []];
    for (const bad of inputs) {
      expect(() => classifyException(bad)).not.toThrow();
      const c = classifyException(bad);
      expect(c).toHaveProperty("error_kind");
      expect(c).toHaveProperty("http_status");
    }
    // getter-that-throws on a wrapper attr
    const g: any = {};
    Object.defineProperty(g, "cause", {
      get() {
        throw new Error("no");
      },
    });
    expect(() => classifyException(g)).not.toThrow();
    // self-referential
    const o: any = {};
    o.cause = o;
    expect(() => classifyException(o)).not.toThrow();
    expect(classifyException(o)).toEqual({ error_kind: "unknown", http_status: 0 });
  });

  // A11 — unknown/unclassifiable still → unknown/0.
  it("returns unknown/0 for an exc with no status hint anywhere", () => {
    expect(classifyException(new Error("something odd happened"))).toEqual({
      error_kind: "unknown",
      http_status: 0,
    });
  });

  // A7 — no regression on existing top-level detection.
  it("top-level status_code still wins (no regression)", () => {
    const e: any = new Error("boom");
    e.status_code = 429;
    expect(classifyException(e)).toEqual({ error_kind: "rate_limited", http_status: 429 });
  });

  it("top-level message 'HTTP 400' still classifies (no regression)", () => {
    expect(classifyException(new Error("Request failed HTTP 400"))).toEqual({
      error_kind: "client_error",
      http_status: 400,
    });
  });
});
