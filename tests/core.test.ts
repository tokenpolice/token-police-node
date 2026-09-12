import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { TokenPolice, init, flush } from "../src/client";
import { TokenPoliceBlockedError } from "../src/exceptions";
import { getClient, setClient } from "../src/state";
import { failSafeSync, failSafeAsync } from "../src/safe";
import { TPSession, getCurrentSession, session, workflow, serverless, manualSpanIds } from "../src/context";
import { trace } from "@opentelemetry/api";

// ── TokenPolice Client Tests ────────────────────────────────────

describe("TokenPolice Client", () => {
  afterEach(() => {
    // Reset global state
    const client = getClient();
    if (client) client.closeSync();
  });

  it("should throw on empty apiKey", () => {
    expect(() => new TokenPolice({ apiKey: "" })).toThrow(
      "apiKey is required",
    );
  });

  it("should warn on invalid apiKey format", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    new TokenPolice({ apiKey: "bad_key_123" });
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining("apiKey format invalid"),
    );
    warnSpy.mockRestore();
  });

  it("should accept valid apiKey without warning", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    new TokenPolice({ apiKey: "tp_sk_test123" });
    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it("should default baseUrl to https://collect.tokenpolice.ai", () => {
    const client = new TokenPolice({ apiKey: "tp_sk_test123" });
    expect(client.baseUrl).toBe("https://collect.tokenpolice.ai");
  });

  it("should strip trailing slash from baseUrl", () => {
    const client = new TokenPolice({
      apiKey: "tp_sk_test123",
      baseUrl: "http://localhost:3001/",
    });
    expect(client.baseUrl).toBe("http://localhost:3001");
  });

  it("should default firewall to 'dry_run'", () => {
    const client = new TokenPolice({ apiKey: "tp_sk_test123" });
    expect(client.firewall).toBe("dry_run");
  });

  it("maps the deprecated enforce boolean to a firewall mode", () => {
    expect(new TokenPolice({ apiKey: "tp_sk_test123", enforce: true }).firewall).toBe("enforce");
    expect(new TokenPolice({ apiKey: "tp_sk_test123", enforce: false }).firewall).toBe("off");
    // explicit firewall wins over the legacy alias
    expect(
      new TokenPolice({ apiKey: "tp_sk_test123", firewall: "dry_run", enforce: true }).firewall,
    ).toBe("dry_run");
  });

  it("check() should fail-open on network error", async () => {
    const client = new TokenPolice({
      apiKey: "tp_sk_test123",
      baseUrl: "http://localhost:99999", // unreachable
      timeout: 0.1,
    });

    const result = await client.check("test_user");
    expect(result.status).toBe("allowed");
    expect(result.fail_open).toBe(true);
  });

  it("log() should not throw on network error", () => {
    const client = new TokenPolice({
      apiKey: "tp_sk_test123",
      baseUrl: "http://localhost:99999",
      timeout: 0.1,
    });

    // Should not throw
    expect(() => {
      client.log("test_user", "free", "default", "", "gpt-4o", 100, 50);
    }).not.toThrow();
  });
});

// ── init() Tests ────────────────────────────────────────────────

describe("init()", () => {
  afterEach(() => {
    const client = getClient();
    if (client) client.closeSync();
  });

  it("should throw on missing apiKey", () => {
    expect(() => init({ apiKey: "" })).toThrow("apiKey is required");
  });

  it("should set the global client", () => {
    const client = init({
      apiKey: "tp_sk_test456",
      baseUrl: "http://localhost:3001",
    });
    expect(getClient()).toBe(client);
  });

  it("should return a TokenPolice instance", () => {
    const client = init({
      apiKey: "tp_sk_test789",
      baseUrl: "http://localhost:3001",
    });
    expect(client).toBeInstanceOf(TokenPolice);
  });
});

// ── Fail-Safe Tests ─────────────────────────────────────────────

describe("failSafe", () => {
  it("failSafeSync should swallow generic errors", () => {
    const fn = failSafeSync(() => {
      throw new Error("boom");
    });
    expect(fn()).toBeUndefined();
  });

  it("failSafeSync should NOT swallow TokenPoliceBlockedError", () => {
    const fn = failSafeSync(() => {
      throw new TokenPoliceBlockedError("blocked!");
    });
    expect(() => fn()).toThrow(TokenPoliceBlockedError);
  });

  it("failSafeAsync should swallow generic errors", async () => {
    const fn = failSafeAsync(async () => {
      throw new Error("boom");
    });
    const result = await fn();
    expect(result).toBeUndefined();
  });

  it("failSafeAsync should NOT swallow TokenPoliceBlockedError", async () => {
    const fn = failSafeAsync(async () => {
      throw new TokenPoliceBlockedError("blocked!");
    });
    await expect(fn()).rejects.toThrow(TokenPoliceBlockedError);
  });

  // ── Handler totality: the catch block is the SDK's outermost fail-open
  // boundary, so its own best-effort reporting (message extraction, client
  // lookup, console.error) must never throw into the caller's LLM call.

  const hostile = () => {
    const err = new Error("boom");
    Object.defineProperty(err, "message", {
      get() {
        throw new Error("hostile message getter");
      },
    });
    return err;
  };

  it("failSafeSync swallows a thrown value with a hostile message getter", () => {
    const prev = getClient();
    try {
      setClient({ logErrors: true } as any);
      const fn = failSafeSync(() => {
        throw hostile();
      });
      expect(fn()).toBeUndefined();
    } finally {
      setClient(prev as any);
    }
  });

  it("failSafeAsync swallows a thrown value with a hostile message getter", async () => {
    const prev = getClient();
    try {
      setClient({ logErrors: true } as any);
      const fn = failSafeAsync(async () => {
        throw hostile();
      });
      await expect(fn()).resolves.toBeUndefined();
    } finally {
      setClient(prev as any);
    }
  });

  it("failSafe swallows when the client's logErrors accessor throws", async () => {
    const prev = getClient();
    try {
      const raising = {} as any;
      Object.defineProperty(raising, "logErrors", {
        get() {
          throw new Error("logErrors accessor exploded");
        },
      });
      setClient(raising);
      expect(failSafeSync(() => {
        throw new Error("boom");
      })()).toBeUndefined();
      await expect(failSafeAsync(async () => {
        throw new Error("boom");
      })()).resolves.toBeUndefined();
    } finally {
      setClient(prev as any);
    }
  });

  it("failSafe swallows when console.error itself throws", async () => {
    const prev = getClient();
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {
      throw new Error("customer console replacement exploded");
    });
    try {
      setClient({ logErrors: true } as any);
      expect(failSafeSync(() => {
        throw new Error("boom");
      })()).toBeUndefined();
      await expect(failSafeAsync(async () => {
        throw new Error("boom");
      })()).resolves.toBeUndefined();
    } finally {
      consoleSpy.mockRestore();
      setClient(prev as any);
    }
  });

  it("failSafe still rethrows TokenPoliceBlockedError under a raising reporter", () => {
    const prev = getClient();
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {
      throw new Error("customer console replacement exploded");
    });
    try {
      setClient({ logErrors: true } as any);
      const fn = failSafeSync(() => {
        throw new TokenPoliceBlockedError("blocked!");
      });
      expect(() => fn()).toThrow(TokenPoliceBlockedError);
    } finally {
      consoleSpy.mockRestore();
      setClient(prev as any);
    }
  });
});

// ── Context / Session Tests ─────────────────────────────────────

describe("TPSession", () => {
  it("should have sensible defaults", () => {
    const s = new TPSession();
    expect(s.userId).toBe("anonymous");
    expect(s.paidPlan).toBe("free");
    expect(s.workflowName).toBe("default_workflow");
    expect(s.sessionId).toBeTruthy();
    expect(s.metadata).toEqual({});
  });

  it("should accept custom values", () => {
    const s = new TPSession({
      userId: "user_42",
      paidPlan: "pro",
      workflowName: "search",
      metadata: { source: "web" },
    });
    expect(s.userId).toBe("user_42");
    expect(s.paidPlan).toBe("pro");
    expect(s.workflowName).toBe("search");
    expect(s.metadata).toEqual({ source: "web" });
  });
});

describe("getCurrentSession()", () => {
  it("should return default session outside of any context", () => {
    const s = getCurrentSession();
    expect(s.userId).toBe("anonymous");
    expect(s.paidPlan).toBe("free");
  });
});

describe("session()", () => {
  it("should propagate session context to the callback", () => {
    session({ name: "test_workflow", userId: "user_99" }, (s) => {
      expect(s.userId).toBe("user_99");
      expect(s.workflowName).toBe("test_workflow");

      // getCurrentSession should also return the active session
      const current = getCurrentSession();
      expect(current.userId).toBe("user_99");
    });
  });

  it("should support nested sessions with metadata merging", () => {
    session({ name: "outer", userId: "user_1", metadata: { a: 1 } }, (outer) => {
      session({ name: "inner", metadata: { b: 2 } }, (inner) => {
        // Inner should inherit user_id from outer
        expect(inner.userId).toBe("user_1");
        // Inner should have its own workflow_name
        expect(inner.workflowName).toBe("inner");
        // Metadata should be merged
        expect(inner.metadata).toEqual({ a: 1, b: 2 });
        // Session IDs should be inherited (grouped)
        expect(inner.sessionId).toBe(outer.sessionId);
      });
    });
  });

  it("should restore context after callback", () => {
    session({ userId: "user_inside" }, () => {
      expect(getCurrentSession().userId).toBe("user_inside");
    });

    // Outside the session — should be default
    expect(getCurrentSession().userId).toBe("anonymous");
  });

  it("should work with async callbacks", async () => {
    await session({ userId: "async_user" }, async (s) => {
      expect(s.userId).toBe("async_user");
      // Simulate async work
      await new Promise((resolve) => setTimeout(resolve, 10));
      // Context should still be active
      expect(getCurrentSession().userId).toBe("async_user");
    });
  });
});

describe("W3C span hierarchy", () => {
  const hex32 = (s: string) => /^[0-9a-f]{32}$/.test(s);
  const hex16 = (s: string) => /^[0-9a-f]{16}$/.test(s);

  beforeEach(() => {
    init({ apiKey: "tp_sk_test_w3c", baseUrl: "http://localhost:59999" });
  });
  afterEach(() => {
    const c = getClient();
    if (c) c.closeSync();
  });

  it("opens a real agent span with W3C-format ids", () => {
    workflow({ name: "orchestrator" }, () => {
      const s = getCurrentSession();
      expect(hex32(s.traceId)).toBe(true);
      expect(hex16(s.rootSpanId)).toBe(true);
      const active = trace.getActiveSpan();
      expect(active).toBeDefined();
      expect(active!.spanContext().traceId).toBe(s.traceId);
      expect(active!.spanContext().spanId).toBe(s.rootSpanId);
    })();
  });

  it("nests a sub-agent under its parent (same trace, real child span)", () => {
    workflow({ name: "orchestrator" }, () => {
      const top = getCurrentSession();
      session({ name: "researcher" }, () => {
        const sub = getCurrentSession();
        expect(sub.traceId).toBe(top.traceId); // same run
        expect(sub.rootSpanId).not.toBe(top.rootSpanId); // real child node
        // A manual (Mode C/D) leaf parents onto the active sub-agent span.
        const ids = manualSpanIds(sub);
        expect(ids.parent_span_id).toBe(sub.rootSpanId);
        expect(hex16(ids.span_id)).toBe(true);
        expect(ids.trace_id).toBe(top.traceId);
      });
    })();
  });
});

describe("workflow()", () => {
  it("should wrap a function with session context", () => {
    const fn = workflow(
      { name: "my_workflow", userId: "wf_user" },
      () => {
        const s = getCurrentSession();
        return s.userId;
      },
    );

    expect(fn()).toBe("wf_user");
  });

  it("should wrap an async function", async () => {
    const fn = workflow(
      { name: "async_workflow", userId: "async_wf_user" },
      async () => {
        await new Promise((resolve) => setTimeout(resolve, 5));
        return getCurrentSession().userId;
      },
    );

    expect(await fn()).toBe("async_wf_user");
  });

  it("should support dynamic binding from object first argument", () => {
    const fn = workflow(
      { name: "dynamic_wf" },
      (opts: { userId: string; query: string }) => {
        return getCurrentSession().userId;
      },
    );

    expect(fn({ userId: "dynamic_user", query: "test" })).toBe("dynamic_user");
  });
});

describe("serverless()", () => {
  it("should wrap and return the same value", () => {
    const handler = serverless(() => 42);
    expect(handler()).toBe(42);
  });

  it("should wrap async functions", async () => {
    const handler = serverless(async () => {
      await new Promise((resolve) => setTimeout(resolve, 5));
      return "done";
    });
    expect(await handler()).toBe("done");
  });
});

// ── TokenPoliceBlockedError Tests ───────────────────────────────

describe("TokenPoliceBlockedError", () => {
  it("should be an instance of Error", () => {
    const err = new TokenPoliceBlockedError("test");
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(TokenPoliceBlockedError);
  });

  it("should have correct name", () => {
    const err = new TokenPoliceBlockedError("test");
    expect(err.name).toBe("TokenPoliceBlockedError");
  });

  it("should preserve message", () => {
    const err = new TokenPoliceBlockedError("budget exceeded");
    expect(err.message).toBe("budget exceeded");
  });
});
