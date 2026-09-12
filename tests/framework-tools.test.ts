/**
 * Tests for framework tool-span capture.
 *
 * OpenAI Agents JS and LlamaIndex JS run tools off the OTel path, so TokenPolice
 * emits tool rows via `emitToolRow` (session-correlated), driven by an Agents
 * TracingProcessor (`handleAgentsSpanEnd`) and LlamaIndex callbackManager events.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { createHash } from "crypto";
import { setClient } from "../src/state";
import { TPSession, _getSessionStorage } from "../src/context";
import {
  emitToolRow,
  handleAgentsSpanEnd,
  registerAgentsOn,
  wrapLlamaIndexTools,
  pickEsmEntry,
  maybeRegisterOpenAIAgentsTracing,
} from "../src/frameworkTools";
import { __test__ as enforcerTest } from "../src/enforcer";

function makeFakeClient() {
  const calls: any[] = [];
  const client: any = {
    log: (...args: any[]) => {
      calls.push(args);
    },
  };
  return { client, calls };
}

const SESSION = new TPSession({
  userId: "u1",
  paidPlan: "pro",
  workflowName: "wf",
  traceId: "a".repeat(32),
  rootSpanId: "b".repeat(16),
});

function withSession<T>(fn: () => T): T {
  return _getSessionStorage().run(SESSION, fn);
}

describe("emitToolRow", () => {
  let calls: any[];
  beforeEach(() => {
    const f = makeFakeClient();
    calls = f.calls;
    setClient(f.client);
    SESSION.setPendingToolCalls([]); // isolate I7 pending-pop tests
  });

  it("logs a tool row correlated to the session, with hashed args/result", () => {
    withSession(() =>
      emitToolRow({
        name: "getCustomerInfo",
        callId: "call_1",
        input: { email: "a@b.com" },
        output: "Premium customer",
      }),
    );
    expect(calls.length).toBe(1);
    const args = calls[0];
    const spanObj = args[10];
    const extras = args[13];
    expect(spanObj.span_kind).toBe("tool");
    expect(spanObj.span_name).toBe("getCustomerInfo");
    expect(spanObj.trace_id).toBe("a".repeat(32));
    expect(extras.tool.name).toBe("getCustomerInfo");
    expect(extras.tool.call_id).toBe("call_1");
    expect(extras.tool.param_hash).toBeTruthy();
    expect(extras.tool.result_hash).toBeTruthy();
    expect(extras.call_outcome.status).toBe("success");
    // raw args/result never stored
    const flat = JSON.stringify(args);
    expect(flat).not.toContain("a@b.com");
    expect(flat).not.toContain("Premium customer");
  });

  it("marks failed outcome and scrubs the raw error", () => {
    // By default the raw error string no longer ships; a SHA-256 hash
    // does instead. The fake client has no errorDetail → resolves to redacted.
    withSession(() =>
      emitToolRow({ name: "escalate", input: {}, failed: true, errorMessage: "boom" }),
    );
    expect(calls[0][13].call_outcome.status).toBe("failed");
    expect(calls[0][13].call_outcome.error_message).toBeUndefined();
    expect(calls[0][13].call_outcome.error_message_hash).toBe(
      createHash("sha256").update("boom", "utf8").digest("hex"),
    );
  });

  it("restores the raw error_message under errorDetail:'raw'", () => {
    setClient({ log: (...a: any[]) => calls.push(a), errorDetail: "raw" } as any);
    withSession(() =>
      emitToolRow({ name: "escalate", input: {}, failed: true, errorMessage: "boom" }),
    );
    expect(calls[0][13].call_outcome.error_message).toBe("boom");
    expect(calls[0][13].call_outcome.error_message_hash).toBeUndefined();
  });

  it("is a no-op (no throw) when no client is set", () => {
    setClient(undefined as any);
    expect(() => withSession(() => emitToolRow({ name: "x" }))).not.toThrow();
  });

  // Residual: explicit fractional durationMs preferred over same-ms wall stamps.
  it("prefers durationMs over wall start/end (sub-ms → 1)", () => {
    const wall = Date.now();
    withSession(() =>
      emitToolRow({
        name: "fast",
        startedAtMs: wall,
        endedAtMs: wall, // same ms → wall delta 0
        durationMs: 0.3, // mono residual
      }),
    );
    expect(calls[0][13].call_outcome.duration_ms).toBe(1);
  });

  // N6: two ms-truncated stamps colliding on the same ms mean the tool ran in
  // <1ms ("ran but fast", convention) — not "no duration recorded".
  it("N6: same-ms explicit start+end stamps without durationMs → 1", () => {
    const wall = Date.now();
    withSession(() =>
      emitToolRow({ name: "agents_iso", startedAtMs: wall, endedAtMs: wall }),
    );
    expect(calls[0][13].call_outcome.duration_ms).toBe(1);
  });

  it("N6: endedAtMs only (no start stamp) still reports 0 — no duration recorded", () => {
    withSession(() => emitToolRow({ name: "endonly", endedAtMs: Date.now() }));
    expect(calls[0][13].call_outcome.duration_ms).toBe(0);
  });

  it("N6: negative stamp delta stays 0", () => {
    const wall = Date.now();
    withSession(() =>
      emitToolRow({ name: "skewed", startedAtMs: wall, endedAtMs: wall - 5 }),
    );
    expect(calls[0][13].call_outcome.duration_ms).toBe(0);
  });

  it("N6: positive stamp delta is unchanged", () => {
    const wall = Date.now();
    withSession(() =>
      emitToolRow({ name: "slow", startedAtMs: wall, endedAtMs: wall + 5 }),
    );
    expect(calls[0][13].call_outcome.duration_ms).toBe(5);
  });

  it("N6: explicit durationMs 0 wins over same-ms stamps (gate must not engage)", () => {
    const wall = Date.now();
    withSession(() =>
      emitToolRow({ name: "explicit_zero", startedAtMs: wall, endedAtMs: wall, durationMs: 0 }),
    );
    expect(calls[0][13].call_outcome.duration_ms).toBe(0);
  });

  it("Intact: explicit fractional durationMs 0.4 → 1", () => {
    withSession(() => emitToolRow({ name: "frac", durationMs: 0.4 }));
    expect(calls[0][13].call_outcome.duration_ms).toBe(1);
  });
});

describe("handleAgentsSpanEnd", () => {
  let calls: any[];
  beforeEach(() => {
    const f = makeFakeClient();
    calls = f.calls;
    setClient(f.client);
    SESSION.setPendingToolCalls([]);
  });

  it("emits for function spans", () => {
    withSession(() =>
      handleAgentsSpanEnd({
        spanData: { type: "function", name: "getCustomerInfo", input: "{}", output: "ok" },
        startedAt: "2026-06-03T10:00:00Z",
        endedAt: "2026-06-03T10:00:01Z",
        error: null,
      }),
    );
    expect(calls.length).toBe(1);
    expect(calls[0][13].tool.name).toBe("getCustomerInfo");
    // No pending stash → call_id empty (never invent).
    expect(calls[0][13].tool.call_id).toBe("");
  });

  it("I7: pops pending call_id by tool name when FunctionSpanData has no id", () => {
    withSession(() => {
      SESSION.setPendingToolCalls([
        { id: "call_agents_42", name: "getCustomerInfo" },
        { id: "call_other", name: "other" },
      ]);
      handleAgentsSpanEnd({
        spanData: { type: "function", name: "getCustomerInfo", input: "{}", output: "ok" },
      });
      expect(calls[0][13].tool.call_id).toBe("call_agents_42");
      expect(SESSION.popPendingToolCallId("other")).toBe("call_other");
    });
  });

  it("I7: explicit callId on span data wins over pending stash", () => {
    withSession(() => {
      SESSION.setPendingToolCalls([{ id: "stashed", name: "getCustomerInfo" }]);
      handleAgentsSpanEnd({
        spanData: {
          type: "function",
          name: "getCustomerInfo",
          callId: "from_span",
          input: "{}",
          output: "ok",
        },
      });
      expect(calls[0][13].tool.call_id).toBe("from_span");
      // Stash not drained when explicit id present.
      expect(SESSION.popPendingToolCallId("getCustomerInfo")).toBe("stashed");
    });
  });

  // N6 end-to-end: the vendor's `timeIso()` is ms-precision, so a sub-ms sync
  // tool callback yields identical startedAt/endedAt ISO strings.
  it("N6: identical ms-truncated ISO stamps → duration_ms 1, not 0", () => {
    const iso = "2026-07-27T04:00:00.123Z";
    withSession(() =>
      handleAgentsSpanEnd({
        spanData: { type: "function", name: "fastTool", input: "{}", output: "ok" },
        startedAt: iso,
        endedAt: iso,
        error: null,
      }),
    );
    expect(calls.length).toBe(1);
    expect(calls[0][13].call_outcome.duration_ms).toBe(1);
  });

  it("ignores non-function spans (response/generation)", () => {
    withSession(() => handleAgentsSpanEnd({ spanData: { type: "response" } }));
    withSession(() => handleAgentsSpanEnd({ spanData: { type: "generation" } }));
    withSession(() => handleAgentsSpanEnd({}));
    expect(calls.length).toBe(0);
  });
});

describe("registerAgentsOn — de-duped by trace-PROVIDER instance (G1-24-1)", () => {
  // `@openai/agents-core` >= 0.4.0 stores its TraceProvider on
  // `globalThis[Symbol.for("openai.agents.core.traceProvider")]`, so the CJS and
  // ESM builds expose two `addTraceProcessor` identities that feed ONE provider.
  // De-duping on function identity registered twice → every function span
  // produced two tool rows (the exact 2× in verification runs #24/#25).
  it("two module handles (CJS + ESM) sharing one provider register ONE processor", () => {
    const registered: any[] = [];
    const provider = { registerProcessor: (p: any) => registered.push(p) };
    const cjs = {
      addTraceProcessor: (p: any) => provider.registerProcessor(p),
      getGlobalTraceProvider: () => provider,
    };
    const esm = {
      addTraceProcessor: (p: any) => provider.registerProcessor(p),
      getGlobalTraceProvider: () => provider,
    };
    expect(cjs.addTraceProcessor).not.toBe(esm.addTraceProcessor); // two identities…
    expect(registerAgentsOn(cjs)).toBe(true);
    expect(registerAgentsOn(esm)).toBe(false); // …one provider → one processor
    expect(registerAgentsOn(cjs)).toBe(false);
    expect(registered.length).toBe(1);
  });

  it("two handles with DISTINCT providers (pre-0.4 per-build singletons) each register", () => {
    const a: any[] = [];
    const b: any[] = [];
    const provA = { registerProcessor: (p: any) => a.push(p) };
    const provB = { registerProcessor: (p: any) => b.push(p) };
    expect(
      registerAgentsOn({
        addTraceProcessor: (p: any) => provA.registerProcessor(p),
        getGlobalTraceProvider: () => provA,
      }),
    ).toBe(true);
    expect(
      registerAgentsOn({
        addTraceProcessor: (p: any) => provB.registerProcessor(p),
        getGlobalTraceProvider: () => provB,
      }),
    ).toBe(true);
    expect(a.length).toBe(1);
    expect(b.length).toBe(1);
  });

  it("falls back to addTraceProcessor identity when a handle exposes no getGlobalTraceProvider", () => {
    const registered: any[] = [];
    const fn = (p: any) => registered.push(p);
    expect(registerAgentsOn({ addTraceProcessor: fn })).toBe(true);
    expect(registerAgentsOn({ addTraceProcessor: fn })).toBe(false); // same fn → skip
    expect(registered.length).toBe(1);
    const registered2: any[] = [];
    expect(registerAgentsOn({ addTraceProcessor: (p: any) => registered2.push(p) })).toBe(true);
    expect(registered2.length).toBe(1);
  });

  it("ignores a module without addTraceProcessor", () => {
    expect(registerAgentsOn(undefined)).toBe(false);
    expect(registerAgentsOn({})).toBe(false);
  });

  it("a getGlobalTraceProvider that throws degrades to the identity fallback, never throws", () => {
    const registered: any[] = [];
    const mod = {
      addTraceProcessor: (p: any) => registered.push(p),
      getGlobalTraceProvider: () => {
        throw new Error("hostile");
      },
    };
    expect(() => registerAgentsOn(mod)).not.toThrow();
    expect(registered.length).toBe(1);
  });

  it("the registered processor routes function spans to a tool row", () => {
    const f = makeFakeClient();
    setClient(f.client);
    let processor: any;
    registerAgentsOn({ addTraceProcessor: (p: any) => (processor = p) });
    withSession(() =>
      processor.onSpanEnd({ spanData: { type: "function", name: "t", input: "{}", output: "ok" } }),
    );
    expect(f.calls.length).toBe(1);
    expect(f.calls[0][13].tool.name).toBe("t");
  });
});

const AGENTS_PROVIDER_SYMBOL = Symbol.for("openai.agents.core.traceProvider");

describe("maybeRegisterOpenAIAgentsTracing — shared globalThis provider (agents-core >= 0.4)", () => {
  // The provider the running agent emits through is reachable WITHOUT loading
  // any agents module. Registering there (a) cannot double up across builds and
  // (b) never `require()`s the umbrella `@openai/agents` CJS build into an ESM
  // app — whose module init calls `setDefaultOpenAITracingExporter()` and
  // REPLACES every processor on the shared provider (the customer's included).
  let saved: any;
  beforeEach(() => {
    saved = (globalThis as any)[AGENTS_PROVIDER_SYMBOL];
  });
  afterEach(() => {
    if (saved === undefined) delete (globalThis as any)[AGENTS_PROVIDER_SYMBOL];
    else (globalThis as any)[AGENTS_PROVIDER_SYMBOL] = saved;
  });

  it("registers exactly once on the global provider across repeated per-call checks", () => {
    const registered: any[] = [];
    (globalThis as any)[AGENTS_PROVIDER_SYMBOL] = {
      registerProcessor: (p: any) => registered.push(p),
    };
    maybeRegisterOpenAIAgentsTracing();
    maybeRegisterOpenAIAgentsTracing();
    maybeRegisterOpenAIAgentsTracing();
    expect(registered.length).toBe(1);
  });

  it("re-registers a FRESH processor after the app's setTraceProcessors() evicts ours", async () => {
    // MultiTracingProcessor.setProcessors() calls shutdown() on every processor
    // it drops — the only signal that the app (or the umbrella's module init)
    // replaced the list after we registered.
    const registered: any[] = [];
    (globalThis as any)[AGENTS_PROVIDER_SYMBOL] = {
      registerProcessor: (p: any) => registered.push(p),
    };
    maybeRegisterOpenAIAgentsTracing();
    expect(registered.length).toBe(1);
    await registered[0].shutdown();
    maybeRegisterOpenAIAgentsTracing();
    expect(registered.length).toBe(2);
    expect(registered[1]).not.toBe(registered[0]);
    maybeRegisterOpenAIAgentsTracing();
    expect(registered.length).toBe(2);
  });

  it("the processor registered on the global provider routes function spans to a tool row", () => {
    const f = makeFakeClient();
    setClient(f.client);
    const registered: any[] = [];
    (globalThis as any)[AGENTS_PROVIDER_SYMBOL] = {
      registerProcessor: (p: any) => registered.push(p),
    };
    maybeRegisterOpenAIAgentsTracing();
    withSession(() =>
      registered[0].onSpanEnd({ spanData: { type: "function", name: "g", input: "{}", output: "ok" } }),
    );
    expect(f.calls.length).toBe(1);
    expect(f.calls[0][13].tool.name).toBe("g");
  });

  it("a global provider whose registerProcessor throws never throws into the call path", () => {
    (globalThis as any)[AGENTS_PROVIDER_SYMBOL] = {
      registerProcessor() {
        throw new Error("upstream refused the processor");
      },
    };
    expect(() => maybeRegisterOpenAIAgentsTracing()).not.toThrow();
  });
});

describe("wrapLlamaIndexTools — wraps each tool's own .call to emit a tool row", () => {
  let calls: any[];
  beforeEach(() => {
    const f = makeFakeClient();
    calls = f.calls;
    setClient(f.client);
    SESSION.setPendingToolCalls([]);
  });

  it("emits one tool row per execution (manual-loop tool.call), hashed args/result", async () => {
    const tools = [
      { metadata: { name: "getCustomerInfo" }, call: async (i: any) => `cust:${i.q}` },
    ];
    wrapLlamaIndexTools(tools);
    const out = await withSession(() => tools[0].call({ q: 7 }));
    expect(out).toBe("cust:7"); // original behavior preserved
    expect(calls.length).toBe(1);
    expect(calls[0][13].tool.name).toBe("getCustomerInfo");
    expect(calls[0][13].call_outcome.status).toBe("success");
    const flat = JSON.stringify(calls[0]);
    expect(flat).not.toContain("cust:7");
  });

  // Residual: mono duration on LlamaIndex wrap so sub-ms tools report >=1.
  it("sync no-op tool.call → duration_ms >= 1", () => {
    const tools = [{ metadata: { name: "noop" }, call: () => "ok" }];
    wrapLlamaIndexTools(tools);
    withSession(() => tools[0].call({}));
    expect(calls.length).toBe(1);
    expect(calls[0][13].call_outcome.duration_ms).toBeGreaterThanOrEqual(1);
  });

  it("I7 PR-D: pops pending call_id by tool name on tool.call", async () => {
    const tools = [
      { metadata: { name: "getCustomerInfo" }, call: async () => "ok" },
    ];
    wrapLlamaIndexTools(tools);
    await withSession(async () => {
      SESSION.setPendingToolCalls([{ id: "call_li_9", name: "getCustomerInfo" }]);
      await tools[0].call({});
      expect(calls[0][13].tool.call_id).toBe("call_li_9");
    });
  });

  it("is idempotent — wrapping twice still emits one row per call", async () => {
    const tools = [{ metadata: { name: "t" }, call: async () => "ok" }];
    wrapLlamaIndexTools(tools);
    wrapLlamaIndexTools(tools);
    await withSession(() => tools[0].call({}));
    expect(calls.length).toBe(1);
  });

  it("re-raises a tool error AND emits a failed row", async () => {
    const tools = [
      {
        metadata: { name: "boom" },
        call: async () => {
          throw new Error("kaboom");
        },
      },
    ];
    wrapLlamaIndexTools(tools);
    await expect(withSession(() => tools[0].call({}))).rejects.toThrow("kaboom");
    expect(calls.length).toBe(1);
    expect(calls[0][13].call_outcome.status).toBe("failed");
  });

  it("no-ops on non-array / tools without .call", () => {
    expect(() => wrapLlamaIndexTools(undefined)).not.toThrow();
    expect(() => wrapLlamaIndexTools([{ metadata: { name: "x" } }])).not.toThrow();
    expect(calls.length).toBe(0);
  });

  it("preserves SYNC synchronicity — sync tool.call returns the VALUE (not a Promise), row still emitted", () => {
    const tools = [{ metadata: { name: "syncTool" }, call: (i: any) => `v:${i.q}` }];
    wrapLlamaIndexTools(tools);
    const out = withSession(() => tools[0].call({ q: 3 }));
    // Must be the raw value, NOT a Promise — the whole point of the fix.
    expect(out).toBe("v:3");
    expect(typeof (out as any)?.then).not.toBe("function");
    expect(calls.length).toBe(1);
    expect(calls[0][13].tool.name).toBe("syncTool");
    expect(calls[0][13].call_outcome.status).toBe("success");
  });

  it("sync tool that throws — original error propagates synchronously, failure row emitted", () => {
    const tools = [
      {
        metadata: { name: "syncBoom" },
        call: () => {
          throw new Error("sync-kaboom");
        },
      },
    ];
    wrapLlamaIndexTools(tools);
    expect(() => withSession(() => tools[0].call({}))).toThrow("sync-kaboom");
    expect(calls.length).toBe(1);
    expect(calls[0][13].call_outcome.status).toBe("failed");
  });

  it("async tool rejection — rejection propagates AND a failed row is emitted", async () => {
    const tools = [
      {
        metadata: { name: "asyncBoom" },
        call: async () => {
          throw new Error("async-kaboom");
        },
      },
    ];
    wrapLlamaIndexTools(tools);
    await expect(withSession(() => tools[0].call({}))).rejects.toThrow("async-kaboom");
    expect(calls.length).toBe(1);
    expect(calls[0][13].call_outcome.status).toBe("failed");
  });

  it("per-tool isolation — a frozen tool first in the array does not abort wrapping the rest", async () => {
    const frozen = Object.freeze({
      metadata: { name: "frozen" },
      call: (i: any) => `frozen:${i.q}`, // sync original, unchanged
    });
    const later = { metadata: { name: "later" }, call: async (i: any) => `later:${i.q}` };
    // Assigning frozen.call throws in strict mode → must be caught per-tool.
    expect(() => wrapLlamaIndexTools([frozen, later])).not.toThrow();

    // The frozen tool is left with its ORIGINAL .call (proves it was NOT poisoned
    // into _wrappedTools — a stale mark would block a future re-wrap attempt).
    expect(frozen.call({ q: 1 })).toBe("frozen:1");
    expect(calls.length).toBe(0); // frozen original emits no row

    // The later tool DID get wrapped and emits a row.
    const out = await withSession(() => later.call({ q: 2 }));
    expect(out).toBe("later:2");
    expect(calls.length).toBe(1);
    expect(calls[0][13].tool.name).toBe("later");
  });
});

describe("pickEsmEntry — selects the ESM target from a package.json exports['.'] node", () => {
  it("picks top-level `default` (.mjs) when there is no `import` condition (@openai/agents shape)", () => {
    const node = {
      require: { types: "./dist/index.d.ts", default: "./dist/index.js" },
      types: "./dist/index.d.ts",
      default: "./dist/index.mjs",
    };
    expect(pickEsmEntry(node)).toBe("./dist/index.mjs");
  });

  it("prefers the `import` condition (llamaindex shape)", () => {
    const node = {
      import: { types: "./dist/type/index.d.ts", default: "./dist/index.js" },
      require: { default: "./dist/cjs/index.cjs" },
    };
    expect(pickEsmEntry(node)).toBe("./dist/index.js");
  });

  it("returns a string node as-is, and undefined for missing/CJS-only", () => {
    expect(pickEsmEntry("./dist/index.js")).toBe("./dist/index.js");
    expect(pickEsmEntry(undefined)).toBeUndefined();
    expect(pickEsmEntry({ require: { default: "./x.cjs" } })).toBeUndefined();
  });
});

describe("registration is safe when frameworks are absent", () => {
  it("does not throw when @openai/agents is not installed", () => {
    // The package is not in token-police-node's node_modules.
    expect(() => maybeRegisterOpenAIAgentsTracing()).not.toThrow();
  });
});

/**
 * End-to-end: the producer that was missing, not the consumer.
 *
 * `wrapLlamaIndexTools` already popped by name — but nothing filled the stash
 * on the LlamaIndex paths, so 8/8 tool rows in the 2026-07-26 run shipped an
 * empty tool_call_id. Drive the real capture function (not a hand-seeded
 * stash) so a refactor that drops the producer call fails here.
 */
describe("LlamaIndex capture → tool row carries the provider id", () => {
  let calls: any[];
  beforeEach(() => {
    const f = makeFakeClient();
    calls = f.calls;
    setClient(f.client);
    SESSION.setPendingToolCalls([]);
  });

  it("non-stream response id reaches the emitted tool row", async () => {
    const { _captureLlamaIndexResponseAt } = enforcerTest as any;
    const tools = [
      { metadata: { name: "getCustomerInfo" }, call: async () => "ok" },
    ];
    wrapLlamaIndexTools(tools);
    await withSession(async () => {
      _captureLlamaIndexResponseAt(
        {
          message: {
            role: "assistant",
            content: "looking that up",
            options: { toolCall: [{ id: "call_li_row", name: "getCustomerInfo" }] },
          },
        },
        0,
      );
      await tools[0].call({});
      expect(calls[0][13].tool.call_id).toBe("call_li_row");
      expect(calls[0][13].tool.name).toBe("getCustomerInfo");
    });
  });

  it("streamed tool-only turn (no text) also reaches the row", async () => {
    const { _captureLlamaIndexResponseAt } = enforcerTest as any;
    const tools = [{ metadata: { name: "escalate" }, call: async () => "ok" }];
    wrapLlamaIndexTools(tools);
    await withSession(async () => {
      _captureLlamaIndexResponseAt(
        {
          raw: {},
          message: {
            role: "assistant",
            content: "",
            options: { toolCall: [{ id: "call_li_stream_row", name: "escalate" }] },
          },
        },
        0,
      );
      await tools[0].call({});
      expect(calls[0][13].tool.call_id).toBe("call_li_stream_row");
    });
  });

  it("a following no-tool turn leaves no stale id on the next row", async () => {
    const { _captureLlamaIndexResponseAt } = enforcerTest as any;
    const tools = [{ metadata: { name: "lookup" }, call: async () => "ok" }];
    wrapLlamaIndexTools(tools);
    await withSession(async () => {
      _captureLlamaIndexResponseAt(
        {
          message: {
            options: { toolCall: [{ id: "turn1", name: "lookup" }] },
          },
        },
        0,
      );
      _captureLlamaIndexResponseAt(
        { message: { role: "assistant", content: "all done" } },
        1,
      );
      await tools[0].call({});
      expect(calls[0][13].tool.call_id).toBe("");
    });
  });
});
