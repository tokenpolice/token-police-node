/**
 * C-04 plan provenance regression suite.
 *
 * TPSession gains `planSource` ("app" | "default"): "app" when the resolved
 * `paidPlan` was ultimately supplied by application code (this scope, or an
 * ancestor scope), "default" when the SDK synthesized the "free" fallback
 * with no app input anywhere in the chain. The client emits `user.plan_source`
 * on /check and /log payloads only when the value is exactly "app"/"default".
 *
 * Part 1 mirrors tests/nestedScopeDefaultLiterals.test.ts's session()/
 * agent()/chain()/workflow() conventions (provided-ness, not value, decides
 * inheritance — see that file's header for the `||` rationale).
 * Part 2 mirrors tests/workflowNameForwarding.test.ts's stubFetch capture
 * seam for the client payload builders.
 */
import { describe, it, expect, vi, afterEach } from "vitest";
import {
  TPSession,
  session,
  agent,
  chain,
  workflow,
  getCurrentSession,
} from "../src/context";
import { TokenPolice } from "../src/client";

// ═══════════════════════════════════════════════════════════════════
// Part 1: TPSession / session() planSource resolution
// ═══════════════════════════════════════════════════════════════════

describe("TPSession planSource resolution", () => {
  it("bare constructor default (no paidPlan) → planSource 'default', paidPlan 'free'", () => {
    const s = new TPSession();
    expect(s.paidPlan).toBe("free");
    expect(s.planSource).toBe("default");
  });

  it("root session() explicit paidPlan 'pro' → planSource 'app'", () => {
    session({ name: "wf", paidPlan: "pro" }, (s) => {
      expect(s.paidPlan).toBe("pro");
      expect(s.planSource).toBe("app");
    });
  });

  it("root session() explicit paidPlan 'free' (app-supplied, same literal as the default) → planSource 'app'", () => {
    session({ name: "wf", paidPlan: "free" }, (s) => {
      expect(s.paidPlan).toBe("free");
      expect(s.planSource).toBe("app");
    });
  });

  it("root session() with no paidPlan at all → planSource 'default'", () => {
    session({ name: "wf" }, (s) => {
      expect(s.paidPlan).toBe("free");
      expect(s.planSource).toBe("default");
    });
  });

  it("nested session omitting paidPlan inherits BOTH the parent's value and its source ('app' parent)", () => {
    session({ name: "outer", paidPlan: "pro" }, () => {
      session({ name: "inner" }, (inner) => {
        expect(inner.paidPlan).toBe("pro");
        expect(inner.planSource).toBe("app");
      });
    });
  });

  it("nested session omitting paidPlan inherits BOTH the parent's value and its source ('default' parent)", () => {
    session({ name: "outer" }, () => {
      session({ name: "inner" }, (inner) => {
        expect(inner.paidPlan).toBe("free");
        expect(inner.planSource).toBe("default");
      });
    });
  });

  it("PINNED QUIRK: nested paidPlan:'' inherits the parent's value AND source (never becomes 'app')", () => {
    session({ name: "outer", paidPlan: "pro" }, () => {
      session({ name: "inner", paidPlan: "" }, (inner) => {
        expect(inner.paidPlan).toBe("pro");
        expect(inner.planSource).toBe("app");
      });
    });
  });

  it("PINNED QUIRK holds under a 'default' parent too: nested '' still inherits 'default'", () => {
    session({ name: "outer" }, () => {
      session({ name: "inner", paidPlan: "" }, (inner) => {
        expect(inner.paidPlan).toBe("free");
        expect(inner.planSource).toBe("default");
      });
    });
  });

  it("nested explicit override → planSource 'app' regardless of the parent's source", () => {
    session({ name: "outer" }, () => {
      // parent's own source is 'default' (no paidPlan supplied)
      session({ name: "inner", paidPlan: "enterprise" }, (inner) => {
        expect(inner.paidPlan).toBe("enterprise");
        expect(inner.planSource).toBe("app");
      });
    });
  });

  it("three-deep chain: an 'app' source set at the root survives two inheriting levels", () => {
    session({ name: "l1", paidPlan: "pro" }, () => {
      session({ name: "l2" }, () => {
        session({ name: "l3" }, (l3) => {
          expect(l3.paidPlan).toBe("pro");
          expect(l3.planSource).toBe("app");
        });
      });
    });
  });

  it("agent()/chain() nesting resolves the same way as session()", () => {
    agent({ name: "outer", paidPlan: "pro" }, () => {
      chain({ name: "inner" }, (inner) => {
        expect(inner.paidPlan).toBe("pro");
        expect(inner.planSource).toBe("app");
      });
    });
  });

  it("getCurrentSession() outside any scope → planSource 'default', paidPlan 'free'", () => {
    const s = getCurrentSession();
    expect(s.planSource).toBe("default");
    expect(s.paidPlan).toBe("free");
  });

  it("workflow() static paidPlan option → planSource 'app'", () => {
    const run = workflow({ name: "wf", paidPlan: "enterprise" }, () =>
      getCurrentSession(),
    );
    const s = run();
    expect(s.paidPlan).toBe("enterprise");
    expect(s.planSource).toBe("app");
  });

  it("workflow() dynamic arg-binding paid_plan (snake_case first-arg key) → planSource 'app'", () => {
    const run = workflow({ name: "wf" }, () => getCurrentSession());
    const s = run({ paid_plan: "pro" } as any);
    expect(s.paidPlan).toBe("pro");
    expect(s.planSource).toBe("app");
  });

  it("workflow() dynamic arg-binding paidPlan (camelCase first-arg key) → planSource 'app'", () => {
    const run = workflow({ name: "wf" }, () => getCurrentSession());
    const s = run({ paidPlan: "pro" } as any);
    expect(s.paidPlan).toBe("pro");
    expect(s.planSource).toBe("app");
  });

  it("workflow() with no static and no dynamic paidPlan → planSource 'default'", () => {
    const run = workflow({ name: "wf" }, () => getCurrentSession());
    const s = run();
    expect(s.paidPlan).toBe("free");
    expect(s.planSource).toBe("default");
  });
});

// ═══════════════════════════════════════════════════════════════════
// Part 2: client wire payload — user.plan_source emission
// ═══════════════════════════════════════════════════════════════════

interface Recorded {
  url: string;
  options: RequestInit;
}

function stubFetch(recorded: Recorded[]) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, options: RequestInit) => {
      recorded.push({ url, options });
      return new Response(JSON.stringify({ status: "allowed" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }),
  );
}

function makeClient(): TokenPolice {
  return new TokenPolice({
    apiKey: "tp_sk_test123",
    baseUrl: "http://localhost:13098",
    timeout: 0.5,
    deployment: "daemon",
  });
}

function bodyFor(recorded: Recorded[], path: string): Record<string, any> {
  const rec = recorded.find((r) => r.url.endsWith(path));
  if (!rec) throw new Error(`no request captured for ${path}`);
  return JSON.parse(rec.options.body as string);
}

describe("client payload: user.plan_source emission", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  // ── check() ──────────────────────────────────────────────────────────────

  it("check(): planSource 'app' → user.plan_source 'app', paid_plan byte-identical", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    await client.check(
      "u", "pro", "wf", "", {}, undefined, undefined, undefined, undefined, "app",
    );
    const user = bodyFor(recorded, "/v1/guard/check").user;
    expect(user.paid_plan).toBe("pro");
    expect(user.plan_source).toBe("app");
  });

  it("check(): planSource 'default' → user.plan_source 'default'", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    await client.check(
      "u", "free", "wf", "", {}, undefined, undefined, undefined, undefined, "default",
    );
    const user = bodyFor(recorded, "/v1/guard/check").user;
    expect(user.paid_plan).toBe("free");
    expect(user.plan_source).toBe("default");
  });

  it("check(): planSource omitted (undefined, default arg) → no plan_source key, paid_plan unaffected", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    await client.check("u", "pro", "wf", "", {});
    const user = bodyFor(recorded, "/v1/guard/check").user;
    expect(user.paid_plan).toBe("pro");
    expect("plan_source" in user).toBe(false);
  });

  it("check(): an invalid planSource value is omitted, never forwarded verbatim", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    await client.check(
      "u", "pro", "wf", "", {}, undefined, undefined, undefined, undefined, "banana",
    );
    const user = bodyFor(recorded, "/v1/guard/check").user;
    expect(user.paid_plan).toBe("pro");
    expect("plan_source" in user).toBe(false);
  });

  // ── log() ────────────────────────────────────────────────────────────────

  // log() carries provenance in the `extras` bag (its 14th and LAST positional
  // argument) rather than a 15th parameter: call spies across the suite read
  // the last positional arg as `extras`, so appending a parameter would break
  // them. `extras.planSource` is read explicitly in client.log — `extras` is
  // never spread — so it can only surface as `user.plan_source`.
  it("log(): extras.planSource 'app' → user.plan_source 'app', paid_plan byte-identical", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    client.log(
      "u", "pro", "wf", "", "gpt-4o", "openai", 100, 50, 0, {},
      undefined, undefined, undefined, { planSource: "app" },
    );
    await client.flush();
    const body = bodyFor(recorded, "/v1/guard/log");
    expect(body.user.paid_plan).toBe("pro");
    expect(body.user.plan_source).toBe("app");
    // Leak guard: the extras key must not surface anywhere else on the wire —
    // including body.user itself, the one object the code actually writes
    // into (it emits the snake_case `plan_source` there; a future refactor
    // leaking the camelCase key would otherwise go unnoticed).
    expect("planSource" in body.user).toBe(false);
    expect("planSource" in body).toBe(false);
    expect("plan_source" in body).toBe(false);
    expect("planSource" in body.metadata).toBe(false);
    expect("planSource" in body.model).toBe(false);
  });

  it("log(): extras.planSource 'default' → user.plan_source 'default'", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    client.log(
      "u", "free", "wf", "", "gpt-4o", "openai", 100, 50, 0, {},
      undefined, undefined, undefined, { planSource: "default" },
    );
    await client.flush();
    const user = bodyFor(recorded, "/v1/guard/log").user;
    expect(user.paid_plan).toBe("free");
    expect(user.plan_source).toBe("default");
    expect("planSource" in user).toBe(false);
  });

  it("log(): no extras at all → no plan_source key, paid_plan unaffected", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    client.log("u", "pro", "wf", "", "gpt-4o", "openai", 100, 50, 0, {});
    await client.flush();
    const user = bodyFor(recorded, "/v1/guard/log").user;
    expect(user.paid_plan).toBe("pro");
    expect("plan_source" in user).toBe(false);
    expect("planSource" in user).toBe(false);
  });

  it("log(): extras present but carrying no planSource → no plan_source key", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    client.log(
      "u", "pro", "wf", "", "gpt-4o", "openai", 100, 50, 0, {},
      undefined, undefined, undefined, { operation: "chat" },
    );
    await client.flush();
    const user = bodyFor(recorded, "/v1/guard/log").user;
    expect(user.paid_plan).toBe("pro");
    expect("plan_source" in user).toBe(false);
    expect("planSource" in user).toBe(false);
  });

  it("log(): an invalid extras.planSource value is omitted, never forwarded verbatim", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    client.log(
      "u", "pro", "wf", "", "gpt-4o", "openai", 100, 50, 0, {},
      undefined, undefined, undefined, { planSource: "garbage" },
    );
    await client.flush();
    const user = bodyFor(recorded, "/v1/guard/log").user;
    expect(user.paid_plan).toBe("pro");
    expect("plan_source" in user).toBe(false);
    expect("planSource" in user).toBe(false);
  });
});
