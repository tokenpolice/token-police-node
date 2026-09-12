/**
 * Regression suite — the manual API forwards the positional workflowName.
 *
 * `check`/`log` document a positional `workflowName` but the payload builders
 * historically never read it, so a manual caller
 * (`tp.log(uid, plan, "checkout_agent", ...)`) silently lost workflow
 * attribution: workflow-scoped rules/budgets never matched and dashboard
 * grouping fell back to "default". These tests pin the fix: the positional name
 * is surfaced into `payload.metadata.workflow_name` when (a) it is non-default
 * and (b) the caller has not already placed the key in metadata — on a shallow
 * copy so the caller's object is never mutated, and guarded so hostile metadata
 * is a no-op (Golden Rule).
 *
 * Capture seam: stub the global `fetch` and read `JSON.parse(options.body)`
 * (the idiom in tests/flushSyncKeepalive.test.ts:24-30).
 */
import { describe, it, expect, vi, afterEach } from "vitest";
import { TokenPolice } from "../src/client";

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
    baseUrl: "http://localhost:13099",
    timeout: 0.5,
    deployment: "daemon",
  });
}

function bodyFor(recorded: Recorded[], path: string): Record<string, any> {
  const rec = recorded.find((r) => r.url.endsWith(path));
  if (!rec) throw new Error(`no request captured for ${path}`);
  return JSON.parse(rec.options.body as string);
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("manual workflowName forwarding into metadata.workflow_name", () => {
  // ── A19: check surfaces the positional workflowName ──────────────────────
  it("check(): forwards a non-default workflowName", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    await client.check("u", "free", "checkout");
    expect(bodyFor(recorded, "/v1/guard/check").metadata.workflow_name).toBe("checkout");
  });

  // ── A20: log surfaces the positional workflowName ────────────────────────
  it("log(): forwards a non-default workflowName", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    client.log("u", "free", "checkout", "", "gpt-4o", "openai", 100, 50, 0);
    await client.flush();
    expect(bodyFor(recorded, "/v1/guard/log").metadata.workflow_name).toBe("checkout");
  });

  // ── A21: default name + {} metadata ⇒ no key (byte-identical path) ────────
  it("check(): default workflowName leaves metadata without the key", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    await client.check("u", "free", "default", "", {});
    const md = bodyFor(recorded, "/v1/guard/check").metadata;
    expect("workflow_name" in md).toBe(false);
    expect(md).toEqual({});
  });

  it("log(): default workflowName leaves metadata without the key", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    client.log("u", "free", "default", "", "gpt-4o", "openai", 100, 50, 0, {});
    await client.flush();
    const md = bodyFor(recorded, "/v1/guard/log").metadata;
    expect("workflow_name" in md).toBe(false);
    expect(md).toEqual({});
  });

  // ── A22: explicit metadata.workflow_name wins over the positional name ────
  it("check(): explicit metadata.workflow_name is not overwritten", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    await client.check("u", "free", "other", "", { workflow_name: "explicit" });
    expect(bodyFor(recorded, "/v1/guard/check").metadata.workflow_name).toBe("explicit");
  });

  it("log(): explicit metadata.workflow_name is not overwritten", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    client.log("u", "free", "other", "", "gpt-4o", "openai", 100, 50, 0, {
      workflow_name: "explicit",
    });
    await client.flush();
    expect(bodyFor(recorded, "/v1/guard/log").metadata.workflow_name).toBe("explicit");
  });

  // ── A9-parity: internal auto-path (positional == pre-seeded key) unchanged ─
  it("check(): pre-seeded key equal to the positional stays byte-identical", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    // Session default value "default_workflow" (!= "default") pre-seeded.
    await client.check("u", "free", "default_workflow", "", {
      workflow_name: "default_workflow",
    });
    const md = bodyFor(recorded, "/v1/guard/check").metadata;
    expect(md).toEqual({ workflow_name: "default_workflow" });
  });

  // ── A23: non-mutation — the caller's object is never mutated (shallow copy)─
  it("check(): does not mutate the caller's metadata object", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    const md: Record<string, unknown> = { foo: "bar" };
    await client.check("u", "free", "checkout", "", md);
    // caller object untouched...
    expect("workflow_name" in md).toBe(false);
    expect(md.foo).toBe("bar");
    // ...but the wire payload carries the name.
    const sent = bodyFor(recorded, "/v1/guard/check").metadata;
    expect(sent.workflow_name).toBe("checkout");
    expect(sent.foo).toBe("bar");
  });

  it("log(): does not mutate the caller's metadata object", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    const md: Record<string, unknown> = { foo: "bar" };
    client.log("u", "free", "checkout", "", "gpt-4o", "openai", 100, 50, 0, md);
    await client.flush();
    expect("workflow_name" in md).toBe(false);
    expect(md.foo).toBe("bar");
    const sent = bodyFor(recorded, "/v1/guard/log").metadata;
    expect(sent.workflow_name).toBe("checkout");
    expect(sent.foo).toBe("bar");
  });

  // ── A24: golden rule — hostile metadata (throwing `has` trap) never throws ─
  it("check(): hostile metadata proxy does not throw and still resolves allowed", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    const hostile = new Proxy(
      {},
      {
        has() {
          throw new Error("hostile has trap");
        },
      },
    );
    const result = await client.check("u", "free", "x", "", hostile);
    expect(result.status).toBe("allowed");
  });

  it("log(): hostile metadata proxy does not throw", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    const hostile = new Proxy(
      {},
      {
        has() {
          throw new Error("hostile has trap");
        },
      },
    );
    expect(() =>
      client.log("u", "free", "x", "", "gpt-4o", "openai", 100, 50, 0, hostile),
    ).not.toThrow();
    await client.flush();
  });
});
