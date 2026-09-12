/**
 * OpenAI stream guard — instance-level wrap of the Traceloop OpenAI
 * instrumentor's `_streamingWrapPromise`.
 *
 * Two confirmed customer bugs in @traceloop/instrumentation-openai
 * (>=0.26.0 <0.28.0), hit by an app using the OpenAI SDK against Gemini's
 * OpenAI-compatible endpoint with stream + tools:
 *
 * - Bug 1: the streamed-chat accumulator NEVER copies `chunk.usage` into its
 *   result, so streamed spans end with no gen_ai.usage.* attrs and the span
 *   processor drops them — metering lost wherever the enforcer wrapper isn't
 *   above the patched copy (loader-hook nested/duplicate copies).
 * - Bug 2: the chat accumulator has no try/catch (the text-completion branch
 *   does — upstream oversight) and crashes INTO CUSTOMER CODE on
 *   Gemini-shaped tool-call deltas (no `index` → `tool_calls[undefined].id`
 *   TypeError), chunks with no `choices` key, or choices with no `delta`;
 *   the span leaks (never ends).
 *
 * Fix under test: installOpenAIStreamGuard(instrumentor) tees the provider
 * stream — Traceloop accumulates sanitized shallow clones, the customer app
 * receives the byte-identical ORIGINAL chunk objects — captures each chunk's
 * usage onto the span with the exact attr names the span processor reads,
 * and direct-drives the remaining chunks if the accumulator still crashes on
 * an unknown shape. Provider errors keep their identity; only they may reach
 * the app (golden rule).
 */
import { describe, it, expect, vi, afterEach } from "vitest";
import { createRequire } from "module";
import { BasicTracerProvider } from "@opentelemetry/sdk-trace-base";

import {
  installOpenAIStreamGuard,
  setupOpenTelemetry,
  unsetupOpenTelemetry,
  TP_STREAM_RAW_USAGE_ATTR,
} from "../src/telemetry";

const requireCjs = createRequire(import.meta.url);

const tracer = new BasicTracerProvider().getTracer("tp-openai-stream-guard-test");
const mkSpan = (): any => tracer.startSpan("chat gemini-2.5-flash");

/** Simple async-iterable stream over a fixed chunk list. */
function mkStream(chunks: any[]) {
  return {
    async *[Symbol.asyncIterator]() {
      for (const c of chunks) yield c;
    },
  };
}

/** Stream whose iterator has a spyable return() (early-break assertions). */
function mkSpyStream(chunks: any[]) {
  const returnSpy = vi.fn(async (v?: any) => ({ value: v, done: true as const }));
  let i = 0;
  const stream = {
    [Symbol.asyncIterator]() {
      return {
        next: async () =>
          i < chunks.length
            ? { value: chunks[i++], done: false as const }
            : { value: undefined, done: true as const },
        return: returnSpy,
      };
    },
  };
  return { stream, returnSpy };
}

/**
 * Gemini-OpenAI-compat streamed tool call: NO `index` on any tool-call delta
 * (the confirmed customer crash shape), id + name on the first delta only,
 * arguments split across deltas, then a usage-only terminal frame.
 */
function mkGeminiToolCallChunks() {
  return [
    {
      id: "c1", created: 1, model: "gemini-2.5-flash",
      choices: [{
        index: 0,
        delta: {
          role: "assistant",
          content: null,
          tool_calls: [{ id: "call_1", type: "function", function: { name: "get_weather", arguments: '{"city":' } }],
        },
        finish_reason: null,
      }],
    },
    {
      id: "c1", created: 1, model: "gemini-2.5-flash",
      choices: [{
        index: 0,
        delta: { tool_calls: [{ function: { arguments: '"SF"}' } }] },
        finish_reason: null,
      }],
    },
    {
      id: "c1", created: 1, model: "gemini-2.5-flash",
      choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
    },
    {
      id: "c1", created: 1, model: "gemini-2.5-flash",
      choices: [],
      usage: { prompt_tokens: 42, completion_tokens: 7, total_tokens: 49 },
    },
  ];
}

/** Drives a guarded/unguarded generator to completion, collecting chunks. */
async function drain(gen: AsyncIterable<any>): Promise<any[]> {
  const out: any[] = [];
  for await (const c of gen) out.push(c);
  return out;
}

/** Fake instrumentor that records exactly what Traceloop would iterate. */
function mkRecordingInstrumentor() {
  const received: any[] = [];
  const inst: any = {
    _streamingWrapPromise({ span, promise }: any) {
      return (async function* () {
        const stream = await promise;
        for await (const c of stream as any) {
          received.push(c);
          yield c;
        }
        span?.end?.();
      })();
    },
  };
  return { inst, received };
}

afterEach(() => {
  vi.restoreAllMocks();
});

// ── Install-level behavior ───────────────────────────────────────

describe("installOpenAIStreamGuard — install", () => {
  it("wraps _streamingWrapPromise with shimmer-compatible markers", () => {
    const original = function _streamingWrapPromise() {};
    const inst: any = { _streamingWrapPromise: original };
    installOpenAIStreamGuard(inst);
    const g: any = inst._streamingWrapPromise;
    expect(g).not.toBe(original);
    expect(g.name).toBe("tpOpenAIStreamGuard");
    expect(g.__wrapped).toBe(true);
    expect(g.__original).toBe(original);
    expect(typeof g.__unwrap).toBe("function");
  });

  it("is idempotent — a second install is a no-op", () => {
    const inst: any = { _streamingWrapPromise: function _streamingWrapPromise() {} };
    installOpenAIStreamGuard(inst);
    const first = inst._streamingWrapPromise;
    installOpenAIStreamGuard(inst);
    expect(inst._streamingWrapPromise).toBe(first);
  });

  it("__unwrap restores the original method", () => {
    const original = function _streamingWrapPromise() {};
    const inst: any = { _streamingWrapPromise: original };
    installOpenAIStreamGuard(inst);
    inst._streamingWrapPromise.__unwrap();
    expect(inst._streamingWrapPromise).toBe(original);
  });

  it("no-ops when _streamingWrapPromise is missing or the input is garbage", () => {
    const inst: any = {};
    expect(() => installOpenAIStreamGuard(inst)).not.toThrow();
    expect(inst._streamingWrapPromise).toBeUndefined();
    expect(() => installOpenAIStreamGuard(null)).not.toThrow();
    expect(() => installOpenAIStreamGuard(undefined)).not.toThrow();
    expect(() => installOpenAIStreamGuard(42)).not.toThrow();
    expect(() => installOpenAIStreamGuard("str")).not.toThrow();
    // Hostile getter — treated as nothing to guard.
    const hostile: any = {};
    Object.defineProperty(hostile, "_streamingWrapPromise", {
      get() { throw new Error("hostile"); },
    });
    expect(() => installOpenAIStreamGuard(hostile)).not.toThrow();
  });

  it("non-chat type passes through untouched (identity-equal arg, verbatim result)", () => {
    const calls: any[] = [];
    const inst: any = {
      _streamingWrapPromise(arg: any) { calls.push(arg); return "sentinel"; },
    };
    installOpenAIStreamGuard(inst);
    const arg = { span: {}, type: "text_completion", params: {}, promise: Promise.resolve(1) };
    expect(inst._streamingWrapPromise(arg)).toBe("sentinel");
    expect(calls).toHaveLength(1);
    expect(calls[0]).toBe(arg); // untouched — no clone, no promise swap
  });

  it("malformed args delegate verbatim (no arg / non-object arg)", () => {
    const calls: any[] = [];
    const inst: any = {
      _streamingWrapPromise(...args: any[]) { calls.push(args); return "sentinel"; },
    };
    installOpenAIStreamGuard(inst);
    expect(inst._streamingWrapPromise()).toBe("sentinel");
    expect(inst._streamingWrapPromise(null)).toBe("sentinel");
    expect(inst._streamingWrapPromise("chat")).toBe("sentinel");
    expect(calls).toEqual([[], [null], ["chat"]]);
  });

  it("a chat call whose inner result is not a generator is returned verbatim", () => {
    const inst: any = { _streamingWrapPromise(_arg: any) { return "not-a-generator"; } };
    installOpenAIStreamGuard(inst);
    const out = inst._streamingWrapPromise({
      span: mkSpan(), type: "chat", params: {}, promise: Promise.resolve(mkStream([])),
    });
    expect(out).toBe("not-a-generator");
  });
});

// ── Sanitized view (what Traceloop sees) vs originals (what the app sees) ──

describe("installOpenAIStreamGuard — sanitization + identity", () => {
  const drive = async (chunks: any[], span = mkSpan()) => {
    const { inst, received } = mkRecordingInstrumentor();
    installOpenAIStreamGuard(inst);
    const gen = inst._streamingWrapPromise({
      span, type: "chat", params: { model: "gemini-2.5-flash", messages: [] },
      promise: Promise.resolve(mkStream(chunks)),
    });
    const delivered = await drain(gen);
    return { received, delivered, span };
  };

  it("the app receives the byte-identical ORIGINAL chunk objects, 1:1 and in order", async () => {
    const chunks = mkGeminiToolCallChunks();
    const { delivered } = await drive(chunks);
    expect(delivered).toHaveLength(chunks.length);
    for (let i = 0; i < chunks.length; i++) expect(delivered[i]).toBe(chunks[i]);
  });

  it("index-less tool-call deltas get synthetic indexes on CLONES; originals are never mutated", async () => {
    const chunks = mkGeminiToolCallChunks();
    const { received } = await drive(chunks);
    // Traceloop saw clones with synthetic indexes…
    expect(received[0]).not.toBe(chunks[0]);
    expect(received[0].choices[0].delta.tool_calls[0].index).toBe(0);
    expect(received[1].choices[0].delta.tool_calls[0].index).toBe(0); // id-less continuation → same slot
    // …while the originals still have NO index anywhere.
    expect("index" in chunks[0].choices[0].delta.tool_calls[0]).toBe(false);
    expect("index" in chunks[1].choices[0].delta.tool_calls[0]).toBe(false);
    // Untouched fields ride by reference on the clone (shallow along the path).
    expect(received[0].choices[0].delta.tool_calls[0].function)
      .toBe(chunks[0].choices[0].delta.tool_calls[0].function);
    // Well-formed chunks (usage-only frame, delta:{} frame) pass through as-is.
    expect(received[2]).toBe(chunks[2]);
    expect(received[3]).toBe(chunks[3]);
  });

  it("two sequential complete tool calls (own ids, no index) get distinct indexes 0 and 1", async () => {
    const chunks = [
      { id: "c", created: 1, model: "m", choices: [{ index: 0, delta: { tool_calls: [{ id: "call_a", type: "function", function: { name: "f1", arguments: "{}" } }] }, finish_reason: null }] },
      { id: "c", created: 1, model: "m", choices: [{ index: 0, delta: { tool_calls: [{ id: "call_b", type: "function", function: { name: "f2", arguments: '{"x":1}' } }] }, finish_reason: null }] },
    ];
    const { received } = await drive(chunks);
    expect(received[0].choices[0].delta.tool_calls[0].index).toBe(0);
    expect(received[1].choices[0].delta.tool_calls[0].index).toBe(1);
  });

  it("provider-supplied numeric indexes are kept and re-sync the synthetic counters", async () => {
    const chunks = [
      // Provider index 2 → keep; nextIndex re-syncs to 3.
      { id: "c", created: 1, model: "m", choices: [{ index: 0, delta: { tool_calls: [{ index: 2, id: "call_a", type: "function", function: { name: "f1", arguments: "" } }] }, finish_reason: null }] },
      // Index-less continuation (no id) → last assigned index (2).
      { id: "c", created: 1, model: "m", choices: [{ index: 0, delta: { tool_calls: [{ function: { arguments: "{}" } }] }, finish_reason: null }] },
      // Index-less NEW call (has id) → next synthetic index (3).
      { id: "c", created: 1, model: "m", choices: [{ index: 0, delta: { tool_calls: [{ id: "call_b", type: "function", function: { name: "f2", arguments: "{}" } }] }, finish_reason: null }] },
    ];
    const { received } = await drive(chunks);
    expect(received[0]).toBe(chunks[0]); // numeric index — no clone needed
    expect(received[1].choices[0].delta.tool_calls[0].index).toBe(2);
    expect(received[2].choices[0].delta.tool_calls[0].index).toBe(3);
  });

  it("chunk with no `choices` key → Traceloop sees choices: []; app sees the original", async () => {
    const bare = { id: "c", created: 1, model: "m", usage: { prompt_tokens: 3, completion_tokens: 1 } };
    const { received, delivered } = await drive([bare]);
    expect(received[0]).not.toBe(bare);
    expect(received[0].choices).toEqual([]);
    expect(received[0].usage).toBe(bare.usage); // by reference
    expect(delivered[0]).toBe(bare);
    expect("choices" in bare).toBe(false); // original untouched
  });

  it("choice with no `delta` → Traceloop sees delta: {}; app sees the original", async () => {
    const deltaless = { id: "c", created: 1, model: "m", choices: [{ index: 0, finish_reason: "stop" }] };
    const { received, delivered } = await drive([deltaless]);
    expect(received[0].choices[0].delta).toEqual({});
    expect(delivered[0]).toBe(deltaless);
    expect("delta" in deltaless.choices[0]).toBe(false);
  });
});

// ── Usage capture (Bug 1) ────────────────────────────────────────

describe("installOpenAIStreamGuard — streamed usage capture", () => {
  const drive = async (chunks: any[]) => {
    const span = mkSpan();
    const { inst } = mkRecordingInstrumentor();
    installOpenAIStreamGuard(inst);
    await drain(inst._streamingWrapPromise({
      span, type: "chat", params: { model: "m", messages: [] },
      promise: Promise.resolve(mkStream(chunks)),
    }));
    return span;
  };

  it("final usage-only chunk (openai shape) → gen_ai.usage.* attrs with the exact names onEnd reads", async () => {
    const span = await drive([
      { id: "c", created: 1, model: "m", choices: [{ index: 0, delta: { content: "hi" }, finish_reason: null }] },
      { id: "c", created: 1, model: "m", choices: [], usage: { prompt_tokens: 42, completion_tokens: 7, total_tokens: 49 } },
    ]);
    expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(42);
    expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(7);
    expect(span.attributes["gen_ai.usage.cache_read_tokens"]).toBeUndefined();
    // Verbatim raw usage rides along for the row's raw block.
    expect(JSON.parse(String(span.attributes[TP_STREAM_RAW_USAGE_ATTR]))).toEqual(
      { prompt_tokens: 42, completion_tokens: 7, total_tokens: 49 },
    );
  });

  it("cached tokens (prompt_tokens_details.cached_tokens) → gen_ai.usage.cache_read_tokens", async () => {
    const span = await drive([
      { id: "c", created: 1, model: "m", choices: [], usage: { prompt_tokens: 100, completion_tokens: 5, total_tokens: 105, prompt_tokens_details: { cached_tokens: 60 } } },
    ]);
    expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(100);
    expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(5);
    expect(span.attributes["gen_ai.usage.cache_read_tokens"]).toBe(60);
  });

  it("usage on every chunk (cumulative-compat endpoints) → last one wins", async () => {
    const span = await drive([
      { id: "c", created: 1, model: "m", choices: [{ index: 0, delta: { content: "a" }, finish_reason: null }], usage: { prompt_tokens: 10, completion_tokens: 1 } },
      { id: "c", created: 1, model: "m", choices: [{ index: 0, delta: { content: "b" }, finish_reason: null }], usage: { prompt_tokens: 10, completion_tokens: 5 } },
      { id: "c", created: 1, model: "m", choices: [], usage: { prompt_tokens: 10, completion_tokens: 9, total_tokens: 19 } },
    ]);
    expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(10);
    expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(9);
    expect(JSON.parse(String(span.attributes[TP_STREAM_RAW_USAGE_ATTR])).completion_tokens).toBe(9);
  });

  it("hostile usage getters / absurd shapes never throw and never break delivery", async () => {
    const hostile: any = { id: "c", created: 1, model: "m", choices: [] };
    Object.defineProperty(hostile, "usage", { get() { throw new Error("hostile usage"); } });
    const span = await drive([
      hostile,
      { id: "c", created: 1, model: "m", choices: [], usage: { prompt_tokens: "NaN-ish", completion_tokens: { weird: true } } },
    ]);
    expect(span.attributes["gen_ai.usage.input_tokens"]).toBeUndefined();
    expect(span.attributes["gen_ai.usage.output_tokens"]).toBeUndefined();
  });
});

// ── Error semantics (golden rule) ────────────────────────────────

describe("installOpenAIStreamGuard — provider errors, early break, fallback", () => {
  it("mid-stream provider error is rethrown with the SAME identity; span is ended", async () => {
    const span = mkSpan();
    const boom = Object.assign(new Error("upstream 500"), { status: 500 });
    const first = { id: "c", created: 1, model: "m", choices: [{ index: 0, delta: { content: "x" }, finish_reason: null }] };
    const stream = {
      [Symbol.asyncIterator]() {
        let i = 0;
        return {
          next: async () => {
            if (i === 0) { i++; return { value: first, done: false as const }; }
            throw boom;
          },
        };
      },
    };
    const { inst } = mkRecordingInstrumentor();
    installOpenAIStreamGuard(inst);
    const gen = inst._streamingWrapPromise({
      span, type: "chat", params: { model: "m", messages: [] }, promise: Promise.resolve(stream),
    });
    const seen: any[] = [];
    let caught: any;
    try {
      for await (const c of gen) seen.push(c);
    } catch (e) {
      caught = e;
    }
    expect(seen).toEqual([first]);
    expect(caught).toBe(boom); // identity-preserved provider error
    expect((span as any)._ended).toBe(true); // no span leak
  });

  it("pre-flight rejection (request failed before any chunk) keeps identity too", async () => {
    const span = mkSpan();
    const boom = Object.assign(new Error("401 bad key"), { status: 401 });
    const { inst } = mkRecordingInstrumentor();
    installOpenAIStreamGuard(inst);
    const gen = inst._streamingWrapPromise({
      span, type: "chat", params: { model: "m", messages: [] }, promise: Promise.reject(boom),
    });
    let caught: any;
    try {
      await drain(gen);
    } catch (e) {
      caught = e;
    }
    expect(caught).toBe(boom);
    expect((span as any)._ended).toBe(true);
  });

  it("early consumer break → real iterator return() called, span ended, no unhandled rejections", async () => {
    const rejections: any[] = [];
    const onRej = (r: any) => rejections.push(r);
    process.on("unhandledRejection", onRej);
    try {
      const span = mkSpan();
      const chunks = [
        { id: "c", created: 1, model: "m", choices: [{ index: 0, delta: { content: "a" }, finish_reason: null }] },
        { id: "c", created: 1, model: "m", choices: [{ index: 0, delta: { content: "b" }, finish_reason: null }] },
        { id: "c", created: 1, model: "m", choices: [{ index: 0, delta: { content: "c" }, finish_reason: null }] },
      ];
      const { stream, returnSpy } = mkSpyStream(chunks);
      const { inst } = mkRecordingInstrumentor();
      installOpenAIStreamGuard(inst);
      const gen = inst._streamingWrapPromise({
        span, type: "chat", params: { model: "m", messages: [] }, promise: Promise.resolve(stream),
      });
      const seen: any[] = [];
      for await (const c of gen) {
        seen.push(c);
        if (seen.length === 2) break;
      }
      expect(seen).toEqual([chunks[0], chunks[1]]);
      expect(returnSpy).toHaveBeenCalled();
      expect((span as any)._ended).toBe(true);
      await new Promise((r) => setTimeout(r, 10));
      expect(rejections).toEqual([]);
    } finally {
      process.off("unhandledRejection", onRej);
    }
  });
});

// ── Real OpenAIInstrumentation (customer scenario) ───────────────

describe("real OpenAIInstrumentation — Gemini streamed tool calls", () => {
  // Gated: the instrumentor is an optionalDependency and may be absent.
  let OpenAIInstrumentation: any;
  try {
    OpenAIInstrumentation = requireCjs(
      "@traceloop/instrumentation-openai",
    ).OpenAIInstrumentation;
  } catch {
    OpenAIInstrumentation = undefined;
  }
  const maybe = OpenAIInstrumentation ? it : it.skip;

  const driveReal = (inst: any, chunks: any[], span = mkSpan()) => ({
    span,
    gen: inst._streamingWrapPromise({
      span,
      type: "chat",
      params: { model: "gemini-2.5-flash", messages: [{ role: "user", content: "weather?" }] },
      promise: Promise.resolve(mkStream(chunks)),
    }),
  });

  maybe("UNGUARDED: index-less tool-call deltas crash the accumulator INTO the consumer and leak the span (proves the bug)", async () => {
    const inst = new OpenAIInstrumentation({ enabled: false });
    const { span, gen } = driveReal(inst, mkGeminiToolCallChunks());
    let caught: any;
    try {
      await drain(gen);
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(TypeError);
    expect((span as any)._ended).toBe(false); // span leak — never ended
  });

  maybe("GUARDED: same stream → all original chunks delivered, no throw, span ended, tool call accumulated", async () => {
    const inst = new OpenAIInstrumentation({ enabled: false });
    installOpenAIStreamGuard(inst);
    const chunks = mkGeminiToolCallChunks();
    const { span, gen } = driveReal(inst, chunks);
    const delivered = await drain(gen);
    expect(delivered).toHaveLength(chunks.length);
    for (let i = 0; i < chunks.length; i++) expect(delivered[i]).toBe(chunks[i]);
    expect((span as any)._ended).toBe(true);
    // Bug 1 fixed: usage attrs present with the exact names onEnd reads.
    expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(42);
    expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(7);
    // Accumulation worked via the synthetic index: one merged tool call with
    // the args concatenated across deltas.
    const out = JSON.parse(String(span.attributes["gen_ai.output.messages"]));
    const toolParts = out[0].parts.filter((p: any) => p.type === "tool_call");
    expect(toolParts).toHaveLength(1);
    expect(toolParts[0].id).toBe("call_1");
    expect(toolParts[0].name).toBe("get_weather");
    expect(toolParts[0].arguments).toEqual({ city: "SF" });
  });

  maybe("GUARDED: two sequential complete tool calls land in DISTINCT accumulator slots", async () => {
    const inst = new OpenAIInstrumentation({ enabled: false });
    installOpenAIStreamGuard(inst);
    const chunks = [
      { id: "c", created: 1, model: "m", choices: [{ index: 0, delta: { tool_calls: [{ id: "call_a", type: "function", function: { name: "f1", arguments: "{}" } }] }, finish_reason: null }] },
      { id: "c", created: 1, model: "m", choices: [{ index: 0, delta: { tool_calls: [{ id: "call_b", type: "function", function: { name: "f2", arguments: '{"x":1}' } }] }, finish_reason: null }] },
      { id: "c", created: 1, model: "m", choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] },
    ];
    const { span, gen } = driveReal(inst, chunks);
    await drain(gen);
    const out = JSON.parse(String(span.attributes["gen_ai.output.messages"]));
    const toolParts = out[0].parts.filter((p: any) => p.type === "tool_call");
    expect(toolParts).toHaveLength(2); // same index would have merged them into one garbled slot
    expect(toolParts.map((p: any) => p.name)).toEqual(["f1", "f2"]);
    expect(toolParts.map((p: any) => p.id)).toEqual(["call_a", "call_b"]);
  });

  maybe("GUARDED: continuation deltas (id on first delta only) stay a single call with concatenated args", async () => {
    const inst = new OpenAIInstrumentation({ enabled: false });
    installOpenAIStreamGuard(inst);
    const chunks = [
      { id: "c", created: 1, model: "m", choices: [{ index: 0, delta: { tool_calls: [{ id: "call_a", type: "function", function: { name: "lookup", arguments: '{"q":' } }] }, finish_reason: null }] },
      { id: "c", created: 1, model: "m", choices: [{ index: 0, delta: { tool_calls: [{ function: { arguments: '"a' } }] }, finish_reason: null }] },
      { id: "c", created: 1, model: "m", choices: [{ index: 0, delta: { tool_calls: [{ function: { arguments: 'bc"}' } }] }, finish_reason: null }] },
    ];
    const { span, gen } = driveReal(inst, chunks);
    await drain(gen);
    const out = JSON.parse(String(span.attributes["gen_ai.output.messages"]));
    const toolParts = out[0].parts.filter((p: any) => p.type === "tool_call");
    expect(toolParts).toHaveLength(1);
    expect(toolParts[0].arguments).toEqual({ q: "abc" });
  });

  maybe("GUARDED: chunk with no `choices` key and choice with no `delta` complete cleanly (no throw, span ends)", async () => {
    const inst = new OpenAIInstrumentation({ enabled: false });
    installOpenAIStreamGuard(inst);
    const chunks = [
      { id: "c", created: 1, model: "m" }, // no choices key at all
      { id: "c", created: 1, model: "m", choices: [{ index: 0, finish_reason: "stop" }] }, // no delta
      { id: "c", created: 1, model: "m", choices: [], usage: { prompt_tokens: 5, completion_tokens: 2 } },
    ];
    const { span, gen } = driveReal(inst, chunks);
    const delivered = await drain(gen);
    expect(delivered).toHaveLength(3);
    for (let i = 0; i < chunks.length; i++) expect(delivered[i]).toBe(chunks[i]);
    expect((span as any)._ended).toBe(true);
    expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(5);
    expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(2);
  });

  maybe("FALLBACK: an unknown shape still crashes the accumulator → direct-drive delivers every remaining chunk, one warning, span ended with usage", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const inst = new OpenAIInstrumentation({ enabled: false });
    installOpenAIStreamGuard(inst);
    const chunks = [
      { id: "c", created: 1, model: "m", choices: [{ index: 0, delta: { content: "a" }, finish_reason: null }] },
      // Poison: tool_calls is not an array → the sanitizer passes it through
      // (unknown shape) and the accumulator's for..of throws.
      { id: "c", created: 1, model: "m", choices: [{ index: 0, delta: { tool_calls: 42 }, finish_reason: null }] },
      { id: "c", created: 1, model: "m", choices: [{ index: 0, delta: { content: "b" }, finish_reason: null }] },
      { id: "c", created: 1, model: "m", choices: [], usage: { prompt_tokens: 11, completion_tokens: 3, total_tokens: 14 } },
    ];
    const { span, gen } = driveReal(inst, chunks);
    const delivered = await drain(gen);
    // The app saw EVERY chunk, identity-preserved, despite the crash.
    expect(delivered).toHaveLength(chunks.length);
    for (let i = 0; i < chunks.length; i++) expect(delivered[i]).toBe(chunks[i]);
    // Warned exactly once (per process).
    const tpWarnings = warnSpy.mock.calls.filter((c) =>
      String(c[0]).includes("stream instrumentor crashed"),
    );
    expect(tpWarnings).toHaveLength(1);
    // Span defensively ended, with the usage captured in fallback mode.
    expect((span as any)._ended).toBe(true);
    expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(11);
    expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(3);
  });
});

// ── Wiring: both setup modes guard the constructed instance ──────

describe("setupOpenTelemetry wiring — the openai instrumentor instance is guarded", () => {
  let traceloopMod: any;
  try {
    traceloopMod = requireCjs("@traceloop/instrumentation-openai");
  } catch {
    traceloopMod = undefined;
  }
  const maybe = traceloopMod?.OpenAIInstrumentation ? it : it.skip;

  /**
   * Fake `openai` module in the DEFAULT-EXPORT class shape that
   * manuallyInstrument() expects (`module.Chat.Completions.prototype` /
   * `module.Completions.prototype`).
   */
  function mkFakeOpenAIModule() {
    const mkSurface = () => {
      function Surface(this: any) {}
      (Surface as any).prototype.create = function create() { return Promise.resolve({}); };
      return Surface;
    };
    function OpenAI(this: any) {}
    (OpenAI as any).Chat = { Completions: mkSurface() };
    (OpenAI as any).Completions = mkSurface();
    return OpenAI;
  }

  /** Captures every OpenAIInstrumentation constructed during fn(). */
  function withCapturedInstances(fn: () => void): any[] {
    const instances: any[] = [];
    const Orig = traceloopMod.OpenAIInstrumentation;
    traceloopMod.OpenAIInstrumentation = class OpenAIInstrumentation extends Orig {
      constructor(...args: any[]) {
        super(...args);
        instances.push(this);
      }
    };
    try {
      fn();
    } finally {
      traceloopMod.OpenAIInstrumentation = Orig;
    }
    return instances;
  }

  const expectGuarded = (inst: any) => {
    expect(Object.prototype.hasOwnProperty.call(inst, "_streamingWrapPromise")).toBe(true);
    expect(inst._streamingWrapPromise.name).toBe("tpOpenAIStreamGuard");
    expect(inst._streamingWrapPromise.__wrapped).toBe(true);
    expect(typeof inst._streamingWrapPromise.__original).toBe("function");
  };

  maybe("manual mode (instrumentModules.openAI)", () => {
    const instances = withCapturedInstances(() => {
      setupOpenTelemetry({ openAI: mkFakeOpenAIModule() } as any);
    });
    try {
      expect(instances.length).toBeGreaterThanOrEqual(1);
      for (const inst of instances) expectGuarded(inst);
    } finally {
      unsetupOpenTelemetry();
    }
  });

  maybe("auto-discovery mode (no instrumentModules)", () => {
    const instances = withCapturedInstances(() => {
      setupOpenTelemetry();
    });
    try {
      expect(instances.length).toBeGreaterThanOrEqual(1);
      for (const inst of instances) expectGuarded(inst);
    } finally {
      unsetupOpenTelemetry();
    }
  });
});
