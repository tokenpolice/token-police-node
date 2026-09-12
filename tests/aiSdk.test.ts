/**
 * Generic Vercel AI SDK instrumentation — registry-less fake providers
 * exercising the LanguageModel V2 (flat usage) and V3 (nested usage) specs.
 *
 * Covers:
 * - instrumentModules.aiSdkProviders (model instance / provider fn entries)
 * - provider identity mapping ("minimax.messages" → minimax,
 * "gateway" → vercel-gateway) + api_base + usage shape `vercel_ai`
 * - V3 nested usage extraction (the legacy flat read produced NaN here)
 * - doStream wrapping (accumulate parts, log the terminal finish usage)
 * - GOLDEN RULE: internal failures never break the customer call; only
 * TokenPoliceBlockedError propagates (enforce=true denial)
 * - tokenPoliceAiSdkMiddleware double-log guard
 * - uninstrument() restores patched prototypes
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

const h = vi.hoisted(() => {
  const logged: any[] = [];
  const clientBox: { client: any } = { client: null };
  return { logged, clientBox };
});

// Spread the real module so every export the enforcer reads (including the
// per-call observation-scope helpers) stays real — a hand-rolled factory makes
// vitest's mock proxy throw on any export it omits.
vi.mock("../src/state", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/state")>();
  return {
    ...actual,
    getClient: () => h.clientBox.client,
    pushObservation: () => {},
    drainObservations: () => [],
    getPack: () => null,
    isCacheHealthy: () => false,
  };
});

import { autoInstrument, uninstrument, tokenPoliceAiSdkMiddleware } from "../src/enforcer";
import { TokenPoliceBlockedError } from "../src/exceptions";

function makeClient(overrides: Record<string, any> = {}) {
  const base: Record<string, any> = {
    enforce: false,
    deployment: undefined,
    logErrors: false,
    check: vi.fn(async () => ({ status: "allowed" })),
    log: (...args: any[]) => {
      h.logged.push(args);
    },
    ...overrides,
  };
  // Mirror the real client's enforce→firewall aliasing so the enforcer's
  // firewall gate sees a value (mocks skip the constructor that resolves it).
  if (base.firewall === undefined) {
    base.firewall =
      base.enforce === true ? "enforce" : base.enforce === false ? "off" : "dry_run";
  }
  return base;
}

/** LanguageModel V2 (ai v5) — flat camelCase usage numbers. */
class FakeModelV2 {
  readonly specificationVersion = "v2";
  readonly provider = "minimax.messages";
  readonly modelId: string;
  config = { baseURL: "https://api.minimax.io/anthropic/v1" };
  constructor(modelId: string) {
    this.modelId = modelId;
  }
  async doGenerate(_options: any): Promise<any> {
    return {
      content: [{ type: "text", text: "hello there" }],
      finishReason: "stop",
      usage: { inputTokens: 100, outputTokens: 20, totalTokens: 120, cachedInputTokens: 30 },
    };
  }
  async doStream(_options: any): Promise<any> {
    const parts = [
      { type: "text-delta", id: "1", delta: "he" },
      { type: "text-delta", id: "1", delta: "llo" },
      { type: "finish", finishReason: "stop", usage: { inputTokens: 10, outputTokens: 5 } },
    ];
    return {
      stream: new ReadableStream({
        start(controller) {
          for (const p of parts) controller.enqueue(p);
          controller.close();
        },
      }),
    };
  }
}

/** LanguageModel V3 (ai v6) — NESTED usage detail objects. */
class FakeModelV3 {
  readonly specificationVersion = "v3";
  readonly provider = "gateway";
  readonly modelId: string;
  config = { baseURL: "https://ai-gateway.vercel.sh/v3/ai" };
  constructor(modelId: string) {
    this.modelId = modelId;
  }
  async doGenerate(_options: any): Promise<any> {
    return {
      content: [{ type: "text", text: "gateway says hi" }],
      finishReason: "stop",
      usage: {
        inputTokens: { total: 100, noCache: 60, cacheRead: 30, cacheWrite: 10 },
        outputTokens: { total: 25, text: 20, reasoning: 5 },
        totalTokens: 125,
      },
    };
  }
  // Spec fidelity: every LanguageModel has doStream — kind detection relies
  // on it (doGenerate-only instances are modality models).
  async doStream(_options: any): Promise<any> {
    return { stream: new ReadableStream({ start: (c) => c.close() }) };
  }
}

async function flushLogs(): Promise<void> {
  await new Promise((r) => setTimeout(r, 0));
}

describe("Vercel AI SDK generic instrumentation", () => {
  beforeEach(() => {
    h.logged.length = 0;
    h.clientBox.client = makeClient();
  });

  afterEach(() => {
    uninstrument();
  });

  it("instruments a model instance passed via aiSdkProviders (V2 flat usage)", async () => {
    autoInstrument({ aiSdkProviders: [new FakeModelV2("probe")] });
    const model = new FakeModelV2("MiniMax-M2");

    const res = await model.doGenerate({ prompt: [{ role: "user", content: [{ type: "text", text: "hi" }] }] });
    await flushLogs();

    expect(res.finishReason).toBe("stop"); // customer result untouched
    expect(h.logged.length).toBe(1);
    const args = h.logged[0];
    expect(args[4]).toBe("MiniMax-M2"); // model
    expect(args[5]).toBe("minimax"); // provider head of "minimax.messages"
    expect(args[6]).toBe(70); // input minus cached
    expect(args[7]).toBe(20);
    expect(args[8]).toBe(30); // cached
    const extra = args[args.length - 1];
    expect(extra.usage.shape).toBe("vercel_ai");
    expect(extra.usage.raw).toEqual({ inputTokens: 100, outputTokens: 20, totalTokens: 120, cachedInputTokens: 30 });
    expect(extra.model_extras.api_base).toBe("https://api.minimax.io/anthropic/v1");
  });

  it("extracts V3 nested usage (would be NaN under the flat read) and maps gateway identity", async () => {
    autoInstrument({ aiSdkProviders: [new FakeModelV3("probe")] });
    const model = new FakeModelV3("openai/gpt-4o");

    await model.doGenerate({ prompt: [] });
    await flushLogs();

    expect(h.logged.length).toBe(1);
    const args = h.logged[0];
    expect(args[4]).toBe("openai/gpt-4o"); // slug forwarded verbatim
    expect(args[5]).toBe("vercel-gateway");
    expect(args[6]).toBe(70); // total 100 − cacheRead 30
    expect(args[7]).toBe(25);
    expect(args[8]).toBe(30);
    expect(Number.isNaN(args[6])).toBe(false);
    const extra = args[args.length - 1];
    expect(extra.usage.shape).toBe("vercel_ai");
    expect(extra.usage.raw.inputTokens).toEqual({ total: 100, noCache: 60, cacheRead: 30, cacheWrite: 10 });
    expect(extra.model_extras.api_base).toBe("https://ai-gateway.vercel.sh/v3/ai");
  });

  it("accepts a provider factory entry and wraps doStream (logs finish usage + latency)", async () => {
    const providerFn = (modelId: string) => new FakeModelV2(modelId);
    autoInstrument({ aiSdkProviders: [providerFn] });
    const model = providerFn("MiniMax-M2");

    const res = await model.doStream({ prompt: [] });
    const seen: any[] = [];
    for await (const chunk of res.stream as any) seen.push(chunk);
    await flushLogs();

    // Customer sees every part untouched, in order.
    expect(seen.map((c) => c.type)).toEqual(["text-delta", "text-delta", "finish"]);
    expect(h.logged.length).toBe(1);
    const args = h.logged[0];
    expect(args[6]).toBe(10);
    expect(args[7]).toBe(5);
    const extra = args[args.length - 1];
    expect(extra.usage.shape).toBe("vercel_ai");
    expect(extra.latency.is_streaming).toBe(true);
    expect(extra.latency.ttft_ms).toBeGreaterThanOrEqual(0);
  });

  it("propagates TokenPoliceBlockedError on enforce=true denial without calling the provider", async () => {
    h.clientBox.client = makeClient({
      enforce: true,
      check: vi.fn(async () => ({ status: "blocked", reason: "Budget exceeded" })),
    });
    const originalSpy = vi.spyOn(FakeModelV2.prototype, "doGenerate");
    autoInstrument({ aiSdkProviders: [new FakeModelV2("probe")] });
    const model = new FakeModelV2("MiniMax-M2");

    await expect(model.doGenerate({ prompt: [] })).rejects.toBeInstanceOf(TokenPoliceBlockedError);
    // The spy observes the WRAPPER call; the stored original must not have
    // produced a result — assert via the absence of a generation log row
    // carrying usage (only the blocked row is emitted).
    const generationRows = h.logged.filter((a) => a[6] > 0 || a[7] > 0);
    expect(generationRows.length).toBe(0);
    originalSpy.mockRestore();
  });

  it("GOLDEN RULE: a throwing log() / check() never breaks the customer call", async () => {
    h.clientBox.client = makeClient({
      enforce: true,
      check: vi.fn(async () => {
        throw new Error("collector unreachable");
      }),
      log: () => {
        throw new Error("log pipe broken");
      },
    });
    autoInstrument({ aiSdkProviders: [new FakeModelV2("probe")] });
    const model = new FakeModelV2("MiniMax-M2");

    const res = await model.doGenerate({ prompt: [] });
    expect(res.finishReason).toBe("stop");
  });

  it("GOLDEN RULE: a model with hostile meta accessors still completes", async () => {
    class HostileModel {
      readonly specificationVersion = "v2";
      get provider(): string {
        throw new Error("nope");
      }
      get modelId(): string {
        throw new Error("nope");
      }
      get config(): any {
        throw new Error("nope");
      }
      async doGenerate(_o: any): Promise<any> {
        return { finishReason: "stop", usage: { inputTokens: 1, outputTokens: 1 } };
      }
    }
    autoInstrument({ aiSdkProviders: [new HostileModel() as any] });
    const model = new HostileModel();
    const res = await model.doGenerate({});
    expect(res.finishReason).toBe("stop");
  });

  it("provider errors propagate to the customer unchanged", async () => {
    class FailingModel extends FakeModelV2 {
      async doGenerate(_o: any): Promise<any> {
        throw Object.assign(new Error("invalid api key"), { status: 401 });
      }
    }
    autoInstrument({ aiSdkProviders: [new FailingModel("probe")] });
    const model = new FailingModel("MiniMax-M2");
    await expect(model.doGenerate({})).rejects.toThrow("invalid api key");
  });

  it("middleware logs exactly once when the model prototype is ALSO patched", async () => {
    autoInstrument({ aiSdkProviders: [new FakeModelV2("probe")] });
    const model = new FakeModelV2("MiniMax-M2");
    const mw = tokenPoliceAiSdkMiddleware();

    const res = await mw.wrapGenerate({
      doGenerate: () => model.doGenerate({ prompt: [] }),
      params: { prompt: [] },
      model,
    });
    await flushLogs();

    expect(res.finishReason).toBe("stop");
    expect(h.logged.length).toBe(1);
    expect(h.logged[0][5]).toBe("minimax");
  });

  it("middleware works standalone (no prototype patch) for out-of-reach models", async () => {
    const model = new FakeModelV3("openai/gpt-4o"); // NOT instrumented
    const mw = tokenPoliceAiSdkMiddleware();

    await mw.wrapGenerate({
      doGenerate: () => model.doGenerate({ prompt: [] }),
      params: { prompt: [] },
      model,
    });
    await flushLogs();

    expect(h.logged.length).toBe(1);
    expect(h.logged[0][5]).toBe("vercel-gateway");
  });

  it("uninstrument() restores the original prototype methods", async () => {
    const before = FakeModelV2.prototype.doGenerate;
    autoInstrument({ aiSdkProviders: [new FakeModelV2("probe")] });
    expect(FakeModelV2.prototype.doGenerate).not.toBe(before);
    uninstrument();
    expect(FakeModelV2.prototype.doGenerate).toBe(before);

    // And a call after uninstrument logs nothing.
    await new FakeModelV2("m").doGenerate({});
    await flushLogs();
    expect(h.logged.length).toBe(0);
  });

  it("re-instruments after uninstrument (WeakSet reset)", async () => {
    autoInstrument({ aiSdkProviders: [new FakeModelV2("probe")] });
    uninstrument();
    autoInstrument({ aiSdkProviders: [new FakeModelV2("probe")] });
    await new FakeModelV2("m").doGenerate({});
    await flushLogs();
    expect(h.logged.length).toBe(1);
  });
});

// ── AI SDK stream wrapper backpressure + terminal-path once-guard ─────
//
// The wrapped `stream` must read ONE provider chunk per pull() so a slow /
// paused / abandoning consumer never forces the whole response to be buffered,
// and the usage/failure log must fire exactly once across close / error /
// cancel. These tests drive the real `tokenPoliceAiSdkMiddleware().wrapStream`
// path; the provider "stream" is a probe async-iterable whose next()/return()
// count production and record release, so backpressure is directly observable.
describe("Vercel AI SDK streaming backpressure", () => {
  beforeEach(() => {
    h.logged.length = 0;
    h.clientBox.client = makeClient();
  });
  afterEach(() => {
    uninstrument();
  });

  const model = {
    modelId: "MiniMax-M2",
    provider: "minimax.messages",
    specificationVersion: "v2",
  };

  const textDelta = (i: number) => ({ type: "text-delta", id: "1", delta: `d${i}` });
  const finishChunk = () => ({
    type: "finish",
    finishReason: "stop",
    usage: { inputTokens: 10, outputTokens: 5 },
  });

  /** Probe provider stream: one `produced++` per delivered chunk; `return()`
   * records release (and optionally throws to model a hostile source). */
  function makeProbeStream(opts: { chunks: any[]; throwAt?: number; returnThrows?: boolean }) {
    const state = { produced: 0, released: false };
    const { chunks, throwAt, returnThrows } = opts;
    let i = 0;
    const iterator: AsyncIterator<any> = {
      async next() {
        if (throwAt !== undefined && i === throwAt) {
          i++;
          throw Object.assign(new Error("provider stream exploded"), { status: 500 });
        }
        if (i >= chunks.length) return { value: undefined, done: true };
        state.produced++;
        const value = chunks[i];
        i++;
        return { value, done: false };
      },
      async return() {
        state.released = true;
        if (returnThrows) throw new Error("source return exploded");
        return { value: undefined, done: true };
      },
    };
    const stream: any = { [Symbol.asyncIterator]: () => iterator };
    return { stream, state };
  }

  async function wrapProbe(probe: { stream: any }): Promise<any> {
    const mw = tokenPoliceAiSdkMiddleware();
    return mw.wrapStream({
      doStream: async () => ({ stream: probe.stream }),
      params: { prompt: [] },
      model,
    });
  }

  const settle = (): Promise<void> => new Promise((r) => setTimeout(r, 20));

  it("A1: a paused consumer does NOT drain the whole provider stream (backpressure)", async () => {
    const N = 20;
    const chunks = Array.from({ length: N }, (_, i) => textDelta(i));
    const probe = makeProbeStream({ chunks });
    const wrapped = await wrapProbe(probe);
    const reader = (wrapped.stream as ReadableStream).getReader();

    await reader.read(); // read exactly one chunk...
    const consumed = 1;
    await settle(); // ...then pause and let any pull() settle.
    expect(probe.state.produced).toBeLessThan(N); // not drained up front
    expect(probe.state.produced).toBeLessThanOrEqual(consumed + 2); // ≤ one in-flight (HWM 1)

    // Drain the rest so nothing dangles; full consumption pulls all N.
    for (;;) {
      const { done } = await reader.read();
      if (done) break;
    }
    expect(probe.state.produced).toBe(N);
  });

  it("A2/A3/A5: full drain delivers every chunk in order and logs usage exactly once", async () => {
    const chunks = [textDelta(0), textDelta(1), textDelta(2), finishChunk()]; // ≥3 + usage
    const probe = makeProbeStream({ chunks });
    const wrapped = await wrapProbe(probe);
    const seen: any[] = [];
    for await (const c of wrapped.stream as any) seen.push(c);
    await flushLogs();
    expect(seen.map((c) => c.type)).toEqual(["text-delta", "text-delta", "text-delta", "finish"]);
    expect(h.logged.length).toBe(1); // once, never per-pull
    expect(h.logged[0][6]).toBe(10);
    expect(h.logged[0][7]).toBe(5);
    const extra = h.logged[0][h.logged[0].length - 1];
    expect(extra.usage.shape).toBe("vercel_ai");
    expect(extra.latency.is_streaming).toBe(true);
  });

  it("A4: wrapped stream survives real .pipeThrough and keeps backpressure through it", async () => {
    const N = 30;
    const chunks = Array.from({ length: N }, (_, i) => textDelta(i));
    const probe = makeProbeStream({ chunks });
    const wrapped = await wrapProbe(probe);
    expect(wrapped.stream instanceof ReadableStream).toBe(true);

    const transformed = (wrapped.stream as ReadableStream).pipeThrough(
      new TransformStream({
        transform(chunk, controller) {
          controller.enqueue(chunk);
        },
      }),
    );
    const reader = transformed.getReader();
    const first = await reader.read();
    expect((first.value as any).type).toBe("text-delta");
    await settle();
    // Backpressure survives the transform: not drained to N up front.
    expect(probe.state.produced).toBeLessThan(N);
    expect(probe.state.produced).toBeLessThanOrEqual(6); // a few HWM=1 queue slots, far below N

    // Order preserved end-to-end through the transform.
    const seen: any[] = [(first.value as any).delta];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      seen.push((value as any).delta);
    }
    expect(seen).toEqual(chunks.map((c) => c.delta));
  });

  it("A6: early abandonment stops upstream production, releases source, logs nothing", async () => {
    const N = 20;
    const chunks = Array.from({ length: N }, (_, i) => textDelta(i)); // NO usage chunk
    const probe = makeProbeStream({ chunks });
    const wrapped = await wrapProbe(probe);
    const reader = (wrapped.stream as ReadableStream).getReader();

    await reader.read();
    await settle();
    const producedAtCancel = probe.state.produced;
    expect(producedAtCancel).toBeLessThan(N); // abandonment saved provider work

    await reader.cancel();
    await settle();
    expect(probe.state.released).toBe(true); // source released via cancel()->return()
    expect(probe.state.produced).toBe(producedAtCancel); // frozen: no pulls after cancel
    await flushLogs();
    expect(h.logged.length).toBe(0); // no usage chunk seen → exactly 0 logs
  });

  it("A7: mid-stream provider error → controller.error, one failure row, no usage row, no throw-out", async () => {
    const chunks = [textDelta(0), finishChunk()]; // a usage chunk is seen BEFORE the error
    const probe = makeProbeStream({ chunks, throwAt: 2 });
    const wrapped = await wrapProbe(probe);
    const reader = (wrapped.stream as ReadableStream).getReader();

    let caught: any;
    try {
      for (;;) {
        const { done } = await reader.read();
        if (done) break;
      }
    } catch (e) {
      caught = e;
    }
    await flushLogs();

    // Surfaced to the consumer via controller.error (read() rejects), not a throw.
    expect(caught).toBeDefined();
    expect(String(caught?.message)).toContain("provider stream exploded");
    // Exactly one row — the failure row (0 tokens). The `failed` flag suppressed
    // the usage log even though a usage chunk was already seen.
    expect(h.logged.length).toBe(1);
    expect(h.logged[0][6]).toBe(0);
    expect(h.logged[0][7]).toBe(0);
    expect(h.logged.filter((a) => a[6] > 0 || a[7] > 0).length).toBe(0);
  });

  it("A8a: error-then-cancel emits exactly one log (once-guard across paths)", async () => {
    const chunks = [textDelta(0)]; // no usage chunk
    const probe = makeProbeStream({ chunks, throwAt: 1 });
    const wrapped = await wrapProbe(probe);
    const reader = (wrapped.stream as ReadableStream).getReader();

    try {
      for (;;) {
        const { done } = await reader.read();
        if (done) break;
      }
    } catch {
      /* expected provider error */
    }
    try {
      await reader.cancel(); // second terminal path — must not add a log
    } catch {
      /* cancel() on an errored stream may reject; irrelevant to the log count */
    }
    await flushLogs();
    expect(h.logged.length).toBe(1); // the failure row only
    expect(h.logged[0][6]).toBe(0);
  });

  it("A8b: close-then-cancel emits exactly one usage log (no double-fire)", async () => {
    const chunks = [textDelta(0), textDelta(1), finishChunk()];
    const probe = makeProbeStream({ chunks });
    const wrapped = await wrapProbe(probe);
    const reader = (wrapped.stream as ReadableStream).getReader();

    for (;;) {
      const { done } = await reader.read();
      if (done) break;
    }
    await reader.cancel(); // after close — must not add a second log
    await flushLogs();
    expect(h.logged.length).toBe(1);
    expect(h.logged[0][6]).toBe(10);
  });

  it("A9: a source whose return() throws does not reject reader.cancel() (fail-open)", async () => {
    const N = 5;
    const chunks = Array.from({ length: N }, (_, i) => textDelta(i));
    const probe = makeProbeStream({ chunks, returnThrows: true });
    const wrapped = await wrapProbe(probe);
    const reader = (wrapped.stream as ReadableStream).getReader();

    await reader.read();
    // The wrapper's cancel() swallows the source's throwing return() so the
    // customer's cancel() promise still resolves.
    await expect(reader.cancel()).resolves.toBeUndefined();
    expect(probe.state.released).toBe(true); // return() WAS attempted, then swallowed
  });

  it("A13: dry_run / off firewall modes never block or throw on a wrapped stream", async () => {
    for (const firewall of ["dry_run", "off"]) {
      h.logged.length = 0;
      h.clientBox.client = makeClient({ enforce: undefined, firewall });
      const chunks = [textDelta(0), finishChunk()];
      const probe = makeProbeStream({ chunks });
      const wrapped = await wrapProbe(probe);
      const seen: any[] = [];
      for await (const c of wrapped.stream as any) seen.push(c);
      expect(seen.map((c) => c.type)).toEqual(["text-delta", "finish"]); // all chunks, no block
    }
  });
});

// ── Modality models (embed / generateImage / generateSpeech / transcribe /
// generateVideo) ──────────────────────────────────────────────────────────

class FakeEmbeddingModel {
  readonly specificationVersion = "v3";
  readonly provider = "openai.embedding";
  readonly modelId: string;
  config = { baseURL: "https://api.openai.com/v1" };
  constructor(modelId: string) {
    this.modelId = modelId;
  }
  async doEmbed(_options: any): Promise<any> {
    return { embeddings: [[0.1, 0.2], [0.3, 0.4]], usage: { tokens: 50 } };
  }
}

class FakeImageModel {
  readonly specificationVersion = "v3";
  readonly provider = "openai.image";
  readonly modelId: string;
  readonly maxImagesPerCall = 10;
  config = { baseURL: "https://api.openai.com/v1" };
  constructor(modelId: string) {
    this.modelId = modelId;
  }
  async doGenerate(_options: any): Promise<any> {
    return { images: ["aGk=", "aGk="], warnings: [], usage: { inputTokens: undefined, outputTokens: undefined } };
  }
}

/** Speech and transcription models share an identical surface — kind "auto". */
class FakeSpeechModel {
  readonly specificationVersion = "v3";
  readonly provider = "openai.speech";
  readonly modelId: string;
  constructor(modelId: string) {
    this.modelId = modelId;
  }
  async doGenerate(_options: any): Promise<any> {
    return { audio: new Uint8Array(8), warnings: [] };
  }
}

class FakeTranscriptionModel {
  readonly specificationVersion = "v3";
  readonly provider = "openai.transcription";
  readonly modelId: string;
  constructor(modelId: string) {
    this.modelId = modelId;
  }
  async doGenerate(_options: any): Promise<any> {
    return {
      text: "hello",
      segments: [{ text: "hello", startSecond: 0, endSecond: 12.5 }],
      durationInSeconds: 12.5,
      warnings: [],
    };
  }
}

class FakeVideoModel {
  readonly specificationVersion = "v3";
  readonly provider = "gateway";
  readonly modelId: string;
  readonly maxVideosPerCall = 4;
  config = { baseURL: "https://ai-gateway.vercel.sh/v3/ai" };
  constructor(modelId: string) {
    this.modelId = modelId;
  }
  async doGenerate(_options: any): Promise<any> {
    return { videos: [{}, {}], warnings: [] };
  }
}

describe("Vercel AI SDK modality models", () => {
  beforeEach(() => {
    h.logged.length = 0;
    h.clientBox.client = makeClient();
  });

  afterEach(() => {
    uninstrument();
  });

  it("embeddings: doEmbed logs operation=embedding with usage.tokens", async () => {
    autoInstrument({ aiSdkProviders: [new FakeEmbeddingModel("probe")] });
    const model = new FakeEmbeddingModel("text-embedding-3-small");

    const res = await model.doEmbed({ values: ["a", "b"] });
    await flushLogs();

    expect(res.embeddings.length).toBe(2); // customer result untouched
    expect(h.logged.length).toBe(1);
    const args = h.logged[0];
    expect(args[4]).toBe("text-embedding-3-small");
    expect(args[5]).toBe("openai");
    expect(args[6]).toBe(50); // embedding input tokens
    const extra = args[args.length - 1];
    expect(extra.operation).toBe("embedding");
    expect(extra.usage.shape).toBe("vercel_ai_embed");
    expect(extra.usage.raw).toEqual({ tokens: 50 });
  });

  it("image: doGenerate logs operation=image_gen with images_generated", async () => {
    autoInstrument({ aiSdkProviders: [new FakeImageModel("probe")] });
    const model = new FakeImageModel("gpt-image-1");

    const res = await model.doGenerate({ prompt: "a cat", n: 2, size: "1024x1024" });
    await flushLogs();

    expect(res.images.length).toBe(2);
    expect(h.logged.length).toBe(1);
    const args = h.logged[0];
    expect(args[4]).toBe("gpt-image-1");
    expect(args[5]).toBe("openai");
    const extra = args[args.length - 1];
    expect(extra.operation).toBe("image_gen");
    expect(extra.usage.shape).toBe("vercel_ai_image");
    expect(extra.usage.items.images_generated).toBe(2);
    expect(extra.usage.items.image_size).toBe("1024x1024");
    expect(extra.model_extras.api_base).toBe("https://api.openai.com/v1");
  });

  it("speech: auto-kind resolves audio_tts from options.text and bills characters", async () => {
    autoInstrument({ aiSdkProviders: [new FakeSpeechModel("probe")] });
    const model = new FakeSpeechModel("tts-1");

    await model.doGenerate({ text: "hello world", voice: "alloy" });
    await flushLogs();

    expect(h.logged.length).toBe(1);
    const extra = h.logged[0][h.logged[0].length - 1];
    expect(extra.operation).toBe("audio_tts");
    expect(extra.usage.shape).toBe("vercel_ai_speech");
    expect(extra.usage.items.tts_characters).toBe(11);
    expect(h.logged[0][4]).toBe("tts-1");
  });

  it("speech S5: bills code points, not UTF-16 units, for non-BMP text", async () => {
    autoInstrument({ aiSdkProviders: [new FakeSpeechModel("probe")] });
    const model = new FakeSpeechModel("tts-1");

    // "a😀b" = 3 code points; the emoji is 2 UTF-16 units so .length would be 4.
    await model.doGenerate({ text: "a😀b", voice: "alloy" });
    await flushLogs();

    expect(h.logged.length).toBe(1);
    const extra = h.logged[0][h.logged[0].length - 1];
    expect(extra.operation).toBe("audio_tts");
    expect(extra.usage.items.tts_characters).toBe(3);
  });

  it("transcription: auto-kind resolves audio_stt from options.audio with durationInSeconds", async () => {
    autoInstrument({ aiSdkProviders: [new FakeTranscriptionModel("probe")] });
    const model = new FakeTranscriptionModel("whisper-1");

    await model.doGenerate({ audio: new Uint8Array(4), mediaType: "audio/mpeg" });
    await flushLogs();

    expect(h.logged.length).toBe(1);
    const extra = h.logged[0][h.logged[0].length - 1];
    expect(extra.operation).toBe("audio_stt");
    expect(extra.usage.shape).toBe("vercel_ai_transcribe");
    expect(extra.usage.duration.audio_seconds).toBe(12.5);
  });

  it("video: maxVideosPerCall kind, duration × count billed via gateway identity", async () => {
    autoInstrument({ aiSdkProviders: [new FakeVideoModel("probe")] });
    const model = new FakeVideoModel("google/veo-3");

    await model.doGenerate({ prompt: "a dog", n: 2, duration: 5 });
    await flushLogs();

    expect(h.logged.length).toBe(1);
    const args = h.logged[0];
    expect(args[4]).toBe("google/veo-3");
    expect(args[5]).toBe("vercel-gateway");
    const extra = args[args.length - 1];
    expect(extra.operation).toBe("video_gen");
    expect(extra.usage.shape).toBe("vercel_ai_video");
    expect(extra.usage.duration.video_seconds).toBe(10); // 5s × 2 videos
    expect(extra.usage.items.videos_generated).toBe(2);
  });

  it("probes modality accessors on a provider object (imageModel + embeddingModel)", async () => {
    const provider = {
      languageModel: (id: string) => new FakeModelV2(id),
      imageModel: (id: string) => new FakeImageModel(id),
      embeddingModel: (id: string) => new FakeEmbeddingModel(id),
    };
    autoInstrument({ aiSdkProviders: [provider] });

    await provider.imageModel("gpt-image-1").doGenerate({ prompt: "x", n: 1 });
    await provider.embeddingModel("text-embedding-3-small").doEmbed({ values: ["a"] });
    await flushLogs();

    expect(h.logged.length).toBe(2);
    const ops = h.logged.map((a) => a[a.length - 1].operation).sort();
    expect(ops).toEqual(["embedding", "image_gen"]);
  });

  it("modality calls are budget-enforced (image blocked pre-flight)", async () => {
    h.clientBox.client = makeClient({
      enforce: true,
      check: vi.fn(async () => ({ status: "blocked", reason: "Budget exceeded" })),
    });
    autoInstrument({ aiSdkProviders: [new FakeImageModel("probe")] });
    const model = new FakeImageModel("gpt-image-1");

    await expect(model.doGenerate({ prompt: "x", n: 1 })).rejects.toBeInstanceOf(
      TokenPoliceBlockedError,
    );
  });

  it("GOLDEN RULE: modality logging failure never breaks the customer call", async () => {
    h.clientBox.client = makeClient({
      log: () => {
        throw new Error("log pipe broken");
      },
    });
    autoInstrument({ aiSdkProviders: [new FakeEmbeddingModel("probe"), new FakeImageModel("probe")] });

    const e = await new FakeEmbeddingModel("m").doEmbed({ values: ["a"] });
    expect(e.embeddings.length).toBe(2);
    const i = await new FakeImageModel("m").doGenerate({ prompt: "x" });
    expect(i.images.length).toBe(2);
  });
});
