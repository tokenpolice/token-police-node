/**
 * LangChain JS streaming usage under-billing + cache double-count fix.
 *
 * Traceloop stamps span usage from last-chunk llmOutput; Gemini/Anthropic emit
 * delta usage_metadata so last chunk is often input=0. The streamIterator
 * wrapper stashes concat'd AIMessage.usage_metadata (usage_source:
 * "langchain_message"); telemetry prefers that stash on under-bill without
 * OpenAI N1 add-cached-back.
 *
 * Cache fields (cache_read / cache_creation) are aggregated with max-across-
 * chunks (not concat sum) because Anthropic re-states absolute values on
 * multiple stream chunks — see langchain_stream_cache_doublecount_design.md.
 *
 * Cases track langchain_stream_usage_underbill_design.md §7.1 and
 * langchain_stream_cache_doublecount_design.md §0.3 Phase 2.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { __test__ } from "../src/enforcer";
import { setClient } from "../src/state";
import { TPSession, _getSessionStorage } from "../src/context";
import { TokenPoliceSpanProcessor } from "../src/telemetry";

let logged: any[];
beforeEach(() => {
  logged = [];
  setClient({ log: (...args: any[]) => logged.push(args) } as any);
});
afterEach(() => {
  setClient(null as any);
  for (const thunk of __test__._restoreThunks.splice(0)) {
    try {
      thunk();
    } catch {
      /* ignore */
    }
  }
});

function newSession(): TPSession {
  return new TPSession({
    userId: "u1",
    paidPlan: "pro",
    workflowName: "wf",
    traceId: "c".repeat(32),
    rootSpanId: "d".repeat(16),
  });
}

/** Minimal AIMessageChunk with concat that sums usage_metadata like LC. */
function makeUsageChunk(
  text: string,
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    cache_read?: number;
    cache_creation?: number;
  },
): any {
  const um =
    usage == null
      ? undefined
      : {
          input_tokens: usage.input_tokens ?? 0,
          output_tokens: usage.output_tokens ?? 0,
          total_tokens:
            (usage.input_tokens ?? 0) + (usage.output_tokens ?? 0),
          input_token_details:
            usage.cache_read || usage.cache_creation
              ? {
                  cache_read: usage.cache_read ?? 0,
                  cache_creation: usage.cache_creation ?? 0,
                }
              : undefined,
        };
  return {
    content: text,
    tool_calls: [],
    usage_metadata: um,
    _getType() {
      return "ai";
    },
    concat(other: any) {
      const a = this.usage_metadata;
      const b = other?.usage_metadata;
      let mergedUsage: any = undefined;
      if (a || b) {
        mergedUsage = {
          input_tokens: (a?.input_tokens ?? 0) + (b?.input_tokens ?? 0),
          output_tokens: (a?.output_tokens ?? 0) + (b?.output_tokens ?? 0),
          total_tokens: 0,
          input_token_details: {
            cache_read:
              (a?.input_token_details?.cache_read ?? 0) +
              (b?.input_token_details?.cache_read ?? 0),
            cache_creation:
              (a?.input_token_details?.cache_creation ?? 0) +
              (b?.input_token_details?.cache_creation ?? 0),
          },
        };
        mergedUsage.total_tokens =
          mergedUsage.input_tokens + mergedUsage.output_tokens;
      }
      const out = makeUsageChunk(this.content + (other?.content ?? ""), undefined);
      out.usage_metadata = mergedUsage;
      return out;
    },
  };
}

function makeFakeBaseChatModel(chunks: any[]): any {
  class BaseChatModel {
    model = "fake-chat-model";
    async *_streamIterator(_input: any, _options?: any): AsyncGenerator<any> {
      for (const c of chunks) yield c;
    }
  }
  return BaseChatModel;
}

function instrument(BaseChatModel: any): void {
  __test__._instrumentLangChainChatModels({ BaseChatModel });
}

function stash(
  session: TPSession,
  order: number,
  usage: Record<string, unknown>,
): void {
  (session as any)._pendingCompositions[`${session.traceId}:${order}`] = {
    usage,
  };
}

function fakeSpan(
  session: TPSession,
  order: number,
  opts: {
    system?: string;
    model?: string;
    input?: number;
    output?: number;
    cached?: number;
  } = {},
): any {
  const attrs: Record<string, unknown> = {
    "gen_ai.system": opts.system ?? "google",
    "gen_ai.request.model": opts.model ?? "gemini-2.5-flash",
    "tp.trace_id": session.traceId,
    "tp.span_order": order,
  };
  if (opts.input != null) attrs["gen_ai.usage.input_tokens"] = opts.input;
  if (opts.output != null) attrs["gen_ai.usage.output_tokens"] = opts.output;
  if (opts.cached != null) attrs["gen_ai.usage.cached_tokens"] = opts.cached;
  return {
    attributes: attrs,
    name: "langchain.chat",
    startTime: [0, 0] as [number, number],
    endTime: [1, 0] as [number, number],
    spanContext: () => ({
      traceId: "0000000000000000000000000000ef01",
      spanId: "0000000000005678",
    }),
    parentSpanId: undefined,
    status: { code: 0 },
  } as any;
}

async function runOnEnd(session: TPSession, span: any): Promise<any[]> {
  await _getSessionStorage().run(session, async () => {
    new TokenPoliceSpanProcessor().onEnd(span);
    await new Promise((r) => setTimeout(r, 0));
  });
  return logged[0];
}

/** Drain a stream under session ALS. */
async function drainStream(
  session: TPSession,
  model: any,
): Promise<void> {
  await _getSessionStorage().run(session, async () => {
    for await (const _ of model._streamIterator("hi")) {
      /* drain */
    }
  });
}

describe("LangChain stream usage stash (enforcer)", () => {
  it("case 1: Gemini-like multi-chunk deltas → stashed concat totals", async () => {
    const chunks = [
      makeUsageChunk("a", { input_tokens: 130, output_tokens: 1 }),
      makeUsageChunk("b", { input_tokens: 0, output_tokens: 7 }),
    ];
    const BCM = makeFakeBaseChatModel(chunks);
    instrument(BCM);
    const session = newSession();
    await drainStream(session, new BCM());

    const usage = session._pendingCompositions[`${session.traceId}:0`]?.usage;
    expect(usage).toMatchObject({
      usage_source: "langchain_message",
      input_tokens: 130,
      output_tokens: 8,
    });
  });

  it("usage stash is live after last content chunk (before EOS finally / nextTick race)", async () => {
    // Simulates LC handleLLMEnd racing process.nextTick before our finally:
    // after the last yielded chunk, stash must already hold concat totals.
    const chunks = [
      makeUsageChunk("a", { input_tokens: 130, output_tokens: 1 }),
      makeUsageChunk("b", { input_tokens: 0, output_tokens: 7 }),
    ];
    const BCM = makeFakeBaseChatModel(chunks);
    instrument(BCM);
    const session = newSession();
    let midStreamUsage: any;
    await _getSessionStorage().run(session, async () => {
      let n = 0;
      for await (const _ of new BCM()._streamIterator("hi")) {
        n++;
        if (n === chunks.length) {
          midStreamUsage =
            session._pendingCompositions[`${session.traceId}:0`]?.usage;
        }
      }
    });
    expect(midStreamUsage).toMatchObject({
      usage_source: "langchain_message",
      input_tokens: 130,
      output_tokens: 8,
    });
  });

  it("case 2: Anthropic-like split → stashed full in+out", async () => {
    const chunks = [
      makeUsageChunk("x", { input_tokens: 100, output_tokens: 0 }),
      makeUsageChunk("y", { input_tokens: 0, output_tokens: 50 }),
    ];
    const BCM = makeFakeBaseChatModel(chunks);
    instrument(BCM);
    const session = newSession();
    await drainStream(session, new BCM());

    const usage = session._pendingCompositions[`${session.traceId}:0`]?.usage;
    expect(usage).toMatchObject({
      usage_source: "langchain_message",
      input_tokens: 100,
      output_tokens: 50,
    });
  });

  it("case 3: OpenAI-like final-only usage → stash equals final chunk", async () => {
    const chunks = [
      makeUsageChunk("a"),
      makeUsageChunk("b"),
      makeUsageChunk("c", { input_tokens: 199, output_tokens: 17 }),
    ];
    const BCM = makeFakeBaseChatModel(chunks);
    instrument(BCM);
    const session = newSession();
    await drainStream(session, new BCM());

    const usage = session._pendingCompositions[`${session.traceId}:0`]?.usage;
    expect(usage).toMatchObject({
      usage_source: "langchain_message",
      input_tokens: 199,
      output_tokens: 17,
    });
  });

  it("stashes cache_creation + cache_read from usage_metadata", async () => {
    const chunks = [
      makeUsageChunk("a", {
        input_tokens: 50,
        output_tokens: 10,
        cache_read: 20,
        cache_creation: 80,
      }),
    ];
    const BCM = makeFakeBaseChatModel(chunks);
    instrument(BCM);
    const session = newSession();
    await drainStream(session, new BCM());

    const usage = session._pendingCompositions[`${session.traceId}:0`]?.usage;
    expect(usage).toMatchObject({
      usage_source: "langchain_message",
      input_tokens: 50,
      output_tokens: 10,
      cached_tokens: 20,
      cache_creation_tokens: 80,
    });
  });

  // --- Cache double-count fix (langchain_stream_cache_doublecount_design.md) ---
  // Tests 2.1–2.3 MUST full-drain the real instrumented wrapper so finally
  // re-stashes with cacheAgg max (not concat sum). Helper-only calls can green
  // while production full-drain still doubles cache.

  it("2.1: absolute cache_creation repeated → max not sum (real wrapper full drain post-finally)", async () => {
    // Smoke shape: chunk0 creation=9621, chunk1 creation=9621; concat would sum
    // to 19242. After full drain (finally ran), stash must be max=9621.
    const chunks = [
      makeUsageChunk("a", {
        input_tokens: 17,
        output_tokens: 1,
        cache_creation: 9621,
        cache_read: 0,
      }),
      makeUsageChunk("b", {
        input_tokens: 0,
        output_tokens: 10,
        cache_creation: 9621,
        cache_read: 0,
      }),
    ];
    // Sanity: concat double still models LC sum (proves the bug class).
    expect(chunks[0].concat(chunks[1]).usage_metadata.input_token_details.cache_creation).toBe(
      19242,
    );

    const BCM = makeFakeBaseChatModel(chunks);
    instrument(BCM);
    const session = newSession();
    await drainStream(session, new BCM());

    const usage = session._pendingCompositions[`${session.traceId}:0`]?.usage;
    expect(usage).toMatchObject({
      usage_source: "langchain_message",
      input_tokens: 17,
      output_tokens: 11,
      cached_tokens: 0,
      cache_creation_tokens: 9621, // max, NOT 19242
    });
    expect(usage?.cache_creation_tokens).not.toBe(19242);
  });

  it("2.2: absolute cache_read repeated → max not sum (real wrapper full drain)", async () => {
    const chunks = [
      makeUsageChunk("a", {
        input_tokens: 14,
        output_tokens: 1,
        cache_read: 9621,
      }),
      makeUsageChunk("b", {
        input_tokens: 0,
        output_tokens: 39,
        cache_read: 9621,
      }),
    ];
    expect(chunks[0].concat(chunks[1]).usage_metadata.input_token_details.cache_read).toBe(
      19242,
    );

    const BCM = makeFakeBaseChatModel(chunks);
    instrument(BCM);
    const session = newSession();
    await drainStream(session, new BCM());

    const usage = session._pendingCompositions[`${session.traceId}:0`]?.usage;
    expect(usage).toMatchObject({
      usage_source: "langchain_message",
      input_tokens: 14,
      output_tokens: 40,
      cached_tokens: 9621, // max, NOT 19242
      cache_creation_tokens: 0,
    });
    expect(usage?.cached_tokens).not.toBe(19242);
  });

  it("2.3: cache only on first chunk (second 0/undefined) → max equals first", async () => {
    const chunks = [
      makeUsageChunk("a", {
        input_tokens: 17,
        output_tokens: 1,
        cache_creation: 9621,
        cache_read: 100,
      }),
      makeUsageChunk("b", {
        input_tokens: 0,
        output_tokens: 10,
        // no cache fields
      }),
    ];
    const BCM = makeFakeBaseChatModel(chunks);
    instrument(BCM);
    const session = newSession();
    await drainStream(session, new BCM());

    const usage = session._pendingCompositions[`${session.traceId}:0`]?.usage;
    expect(usage).toMatchObject({
      usage_source: "langchain_message",
      input_tokens: 17,
      output_tokens: 11,
      cached_tokens: 100,
      cache_creation_tokens: 9621,
    });
  });

  it("2.4: Gemini-like in/out deltas still sum (#109 regression); cache 0", async () => {
    const chunks = [
      makeUsageChunk("a", { input_tokens: 130, output_tokens: 1 }),
      makeUsageChunk("b", { input_tokens: 0, output_tokens: 7 }),
    ];
    const BCM = makeFakeBaseChatModel(chunks);
    instrument(BCM);
    const session = newSession();
    await drainStream(session, new BCM());

    const usage = session._pendingCompositions[`${session.traceId}:0`]?.usage;
    expect(usage).toMatchObject({
      usage_source: "langchain_message",
      input_tokens: 130,
      output_tokens: 8,
      cached_tokens: 0,
      cache_creation_tokens: 0,
    });
  });

  it("case 6: composition rebuild still once at EOS (multi-chunk)", async () => {
    // Smoke: multi-chunk drain does not throw and leaves response + usage.
    const chunks = ["a", "b", "c"].map((t, i) =>
      makeUsageChunk(t, i === 0 ? { input_tokens: 10, output_tokens: 1 } : { input_tokens: 0, output_tokens: 1 }),
    );
    const BCM = makeFakeBaseChatModel(chunks);
    instrument(BCM);
    const session = newSession();
    await drainStream(session, new BCM());
    const cd = session._pendingCompositions[`${session.traceId}:0`];
    expect(cd?.response).toBeDefined();
    expect(cd?.usage?.input_tokens).toBe(10);
    expect(cd?.usage?.output_tokens).toBe(3);
  });

  it("case 7: two streams get distinct reserved orders / no cross-key", async () => {
    const BCM = makeFakeBaseChatModel([
      makeUsageChunk("s1", { input_tokens: 11, output_tokens: 2 }),
    ]);
    instrument(BCM);
    const session = newSession();

    await _getSessionStorage().run(session, async () => {
      const m1 = new BCM();
      const m2 = new BCM();
      // Sequential drains still allocate distinct orders via nextSpanOrder.
      for await (const _ of m1._streamIterator("a")) {
        /* */
      }
      for await (const _ of m2._streamIterator("b")) {
        /* */
      }
    });

    const u0 = session._pendingCompositions[`${session.traceId}:0`]?.usage;
    const u1 = session._pendingCompositions[`${session.traceId}:1`]?.usage;
    expect(u0?.input_tokens).toBe(11);
    expect(u1?.input_tokens).toBe(11);
    expect(u0).toBeDefined();
    expect(u1).toBeDefined();
  });
});

describe("LangChain stream usage merge (telemetry)", () => {
  it("case 1+10: Gemini under-bill span + stash → log has concat totals", async () => {
    const session = newSession();
    stash(session, 0, {
      usage_source: "langchain_message",
      input_tokens: 130,
      output_tokens: 8,
      cached_tokens: 0,
      cache_creation_tokens: 0,
    });
    // Last-chunk wrong: 0 in / 7 out
    const args = await runOnEnd(
      session,
      fakeSpan(session, 0, { input: 0, output: 7, system: "google" }),
    );
    expect(args).toBeDefined();
    const opts = args.at(-1);
    expect(opts.usage.shape).toBe("google_genai");
    expect(opts.usage.raw.input_tokens).toBe(130);
    expect(opts.usage.raw.prompt_tokens).toBe(130);
    expect(opts.usage.raw.output_tokens).toBe(8);
    expect(opts.usage.raw.completion_tokens).toBe(8);
  });

  it("case 2: Anthropic split — span last-only overridden by stash", async () => {
    const session = newSession();
    stash(session, 0, {
      usage_source: "langchain_message",
      input_tokens: 100,
      output_tokens: 50,
    });
    const opts = (
      await runOnEnd(
        session,
        fakeSpan(session, 0, {
          input: 0,
          output: 50,
          system: "anthropic",
          model: "claude-haiku",
        }),
      )
    ).at(-1);
    expect(opts.usage.shape).toBe("anthropic_messages");
    expect(opts.usage.raw.input_tokens).toBe(100);
    expect(opts.usage.raw.output_tokens).toBe(50);
  });

  it("case 3: OpenAI-like span == stash → no change / no N1 mutation", async () => {
    const session = newSession();
    stash(session, 0, {
      usage_source: "langchain_message",
      input_tokens: 199,
      output_tokens: 17,
    });
    const opts = (
      await runOnEnd(
        session,
        fakeSpan(session, 0, {
          input: 199,
          output: 17,
          system: "openai",
          model: "gpt-4o",
        }),
      )
    ).at(-1);
    expect(opts.usage.shape).toBe("openai_chat");
    expect(opts.usage.raw.prompt_tokens).toBe(199);
    expect(opts.usage.raw.completion_tokens).toBe(17);
  });

  it("case 4: both span and stash empty → no log", async () => {
    const session = newSession();
    // No usage attrs, no stash
    await runOnEnd(session, fakeSpan(session, 0, { system: "google" }));
    expect(logged).toHaveLength(0);
  });

  it("case 5: span equal stash → prefer not applied in a mutating way", async () => {
    const session = newSession();
    stash(session, 0, {
      usage_source: "langchain_message",
      input_tokens: 200,
      output_tokens: 20,
    });
    const opts = (
      await runOnEnd(
        session,
        fakeSpan(session, 0, { input: 200, output: 20, system: "google" }),
      )
    ).at(-1);
    expect(opts.usage.raw.input_tokens).toBe(200);
    expect(opts.usage.raw.output_tokens).toBe(20);
  });

  it("case 8 Amendment 1: LC stash cached>0 does NOT add-cached-back", async () => {
    const session = newSession();
    stash(session, 0, {
      usage_source: "langchain_message",
      input_tokens: 1000,
      output_tokens: 10,
      cached_tokens: 400,
      cache_creation_tokens: 0,
    });
    // Under-bill span so prefer fires
    const opts = (
      await runOnEnd(
        session,
        fakeSpan(session, 0, { input: 0, output: 10, system: "google" }),
      )
    ).at(-1);
    expect(opts.usage.shape).toBe("google_genai");
    // Must be 1000, NOT 1400
    expect(opts.usage.raw.prompt_tokens).toBe(1000);
    expect(opts.usage.raw.input_tokens).toBe(1000);
    expect(opts.usage.raw.prompt_tokens_details?.cached_tokens).toBe(400);
    expect(opts.usage.raw.cache_read_input_tokens).toBe(400);
  });

  it("case 9 Amendment 4: cache_creation_tokens forwarded on Anthropic shape", async () => {
    const session = newSession();
    stash(session, 0, {
      usage_source: "langchain_message",
      input_tokens: 100,
      output_tokens: 20,
      cached_tokens: 0,
      cache_creation_tokens: 80,
    });
    const opts = (
      await runOnEnd(
        session,
        fakeSpan(session, 0, {
          input: 0,
          output: 20,
          system: "anthropic",
          model: "claude",
        }),
      )
    ).at(-1);
    expect(opts.usage.raw.cache_creation_input_tokens).toBe(80);
    expect(opts.usage.raw.input_tokens).toBe(100);
  });

  it("2.5: log-apply cache_creation max 9621 not 19242; exclusive input unchanged", async () => {
    const session = newSession();
    stash(session, 0, {
      usage_source: "langchain_message",
      input_tokens: 17,
      output_tokens: 11,
      cached_tokens: 0,
      cache_creation_tokens: 9621,
    });
    const opts = (
      await runOnEnd(
        session,
        fakeSpan(session, 0, {
          input: 0,
          output: 10,
          system: "anthropic",
          model: "claude",
        }),
      )
    ).at(-1);
    expect(opts.usage.raw.cache_creation_input_tokens).toBe(9621);
    expect(opts.usage.raw.cache_creation_input_tokens).not.toBe(19242);
    // Exclusive input — not input+cache
    expect(opts.usage.raw.input_tokens).toBe(17);
    expect(opts.usage.raw.output_tokens).toBe(11);
  });

  it("2.5b: end-to-end enforcer max cache_creation + telemetry log", async () => {
    const chunks = [
      makeUsageChunk("a", {
        input_tokens: 17,
        output_tokens: 1,
        cache_creation: 9621,
      }),
      makeUsageChunk("b", {
        input_tokens: 0,
        output_tokens: 10,
        cache_creation: 9621,
      }),
    ];
    const BCM = makeFakeBaseChatModel(chunks);
    instrument(BCM);
    const session = newSession();

    await _getSessionStorage().run(session, async () => {
      const model = new BCM();
      for await (const _ of model._streamIterator("hi")) {
        /* full drain so finally runs */
      }
      new TokenPoliceSpanProcessor().onEnd(
        fakeSpan(session, 0, {
          input: 0,
          output: 10,
          system: "anthropic",
          model: "claude",
        }),
      );
      await new Promise((r) => setTimeout(r, 0));
    });

    expect(logged.length).toBe(1);
    const opts = logged[0].at(-1);
    expect(opts.usage.raw.input_tokens).toBe(17);
    expect(opts.usage.raw.output_tokens).toBe(11);
    expect(opts.usage.raw.cache_creation_input_tokens).toBe(9621);
  });

  it("non-LC OpenAI N1 path still add-cached-back when both-zero span", async () => {
    const session = newSession();
    stash(session, 0, {
      // no usage_source → OpenAI stream tap path
      input_tokens: 60,
      output_tokens: 20,
      cached_tokens: 40,
    });
    const opts = (
      await runOnEnd(
        session,
        fakeSpan(session, 0, { system: "openai", model: "gpt-4o" }),
      )
    ).at(-1);
    // N1: exclusive 60 + cached 40 → inclusive 100
    expect(opts.usage.raw.prompt_tokens).toBe(100);
    expect(opts.usage.raw.input_tokens).toBe(100);
  });

  it("span better than LC stash (larger totals) keeps span", async () => {
    const session = newSession();
    stash(session, 0, {
      usage_source: "langchain_message",
      input_tokens: 10,
      output_tokens: 1,
    });
    const opts = (
      await runOnEnd(
        session,
        fakeSpan(session, 0, { input: 500, output: 50, system: "google" }),
      )
    ).at(-1);
    expect(opts.usage.raw.input_tokens).toBe(500);
    expect(opts.usage.raw.output_tokens).toBe(50);
  });

  it("case 10: end-to-end enforcer stash + telemetry log (wrong span, right log)", async () => {
    // Drive real streamIterator to write stash, then onEnd with wrong span attrs
    // under the reserved order 0 (wrapper nextSpanOrder → 0).
    const chunks = [
      makeUsageChunk("a", { input_tokens: 130, output_tokens: 1 }),
      makeUsageChunk("b", { input_tokens: 0, output_tokens: 7 }),
    ];
    const BCM = makeFakeBaseChatModel(chunks);
    instrument(BCM);
    const session = newSession();

    await _getSessionStorage().run(session, async () => {
      const model = new BCM();
      for await (const _ of model._streamIterator("hi")) {
        /* */
      }
      // Span attrs wrong last-chunk; order matches stash key 0.
      new TokenPoliceSpanProcessor().onEnd(
        fakeSpan(session, 0, { input: 0, output: 7, system: "google" }),
      );
      await new Promise((r) => setTimeout(r, 0));
    });

    expect(logged.length).toBe(1);
    const opts = logged[0].at(-1);
    expect(opts.usage.raw.input_tokens).toBe(130);
    expect(opts.usage.raw.output_tokens).toBe(8);
  });

  it("Amendment 2: first pull under reservation → onStart gets reserved order; log applies stash", async () => {
    // If per-pull ALS is broken, onStart allocates a NEW order and stash at
    // order 0 never applies → this test must fail.
    const proc = new TokenPoliceSpanProcessor();
    let startedSpan: any = null;

    class BaseChatModel {
      model = "gemini-fake";
      async *_streamIterator(_input: any): AsyncGenerator<any> {
        // Runs inside runWithReservedSpanOrder (wrapper wraps it.next()).
        const span = {
          attributes: {} as Record<string, unknown>,
          name: "langchain.chat",
          setAttribute(k: string, v: any) {
            this.attributes[k] = v;
            return this;
          },
          spanContext: () => ({
            traceId: "e".repeat(32),
            spanId: "f".repeat(16),
          }),
          parentSpanId: undefined,
          startTime: [0, 0] as [number, number],
          endTime: [1, 0] as [number, number],
          status: { code: 0 },
        };
        proc.onStart(span as any);
        startedSpan = span;
        // Wrong last-chunk-style attrs (Gemini delta final chunk).
        span.attributes["gen_ai.system"] = "google";
        span.attributes["gen_ai.request.model"] = "gemini-2.5-flash";
        span.attributes["gen_ai.usage.input_tokens"] = 0;
        span.attributes["gen_ai.usage.output_tokens"] = 7;
        yield makeUsageChunk("a", { input_tokens: 130, output_tokens: 1 });
        yield makeUsageChunk("b", { input_tokens: 0, output_tokens: 7 });
      }
    }
    instrument(BaseChatModel);
    const session = newSession();

    await _getSessionStorage().run(session, async () => {
      for await (const _ of new BaseChatModel()._streamIterator("hi")) {
        /* */
      }
      expect(startedSpan).not.toBeNull();
      // Reservation order was 0 (first nextSpanOrder in fresh session).
      expect(startedSpan.attributes["tp.span_order"]).toBe(0);
      // Stash must be under the same key as the span.
      expect(
        session._pendingCompositions[`${session.traceId}:0`]?.usage?.input_tokens,
      ).toBe(130);
      proc.onEnd(startedSpan);
      await new Promise((r) => setTimeout(r, 0));
    });

    expect(logged.length).toBe(1);
    const opts = logged[0].at(-1);
    expect(opts.usage.raw.input_tokens).toBe(130);
    expect(opts.usage.raw.output_tokens).toBe(8);
  });
});

describe("LangChain stream fail-open", () => {
  it("usage_metadata missing → no usage stash, no throw", async () => {
    const chunks = [makeUsageChunk("only text")];
    // strip usage
    chunks[0].usage_metadata = undefined;
    const BCM = makeFakeBaseChatModel(chunks);
    instrument(BCM);
    const session = newSession();
    await expect(drainStream(session, new BCM())).resolves.toBeUndefined();
    expect(
      session._pendingCompositions[`${session.traceId}:0`]?.usage,
    ).toBeUndefined();
  });

  it("concat throw mid-stream still yields and exits cleanly", async () => {
    const bad = makeUsageChunk("x", { input_tokens: 1, output_tokens: 1 });
    bad.concat = () => {
      throw new Error("concat boom");
    };
    const chunks = [
      makeUsageChunk("a", { input_tokens: 5, output_tokens: 1 }),
      bad,
    ];
    const BCM = makeFakeBaseChatModel(chunks);
    instrument(BCM);
    const session = newSession();
    const received: any[] = [];
    await _getSessionStorage().run(session, async () => {
      for await (const c of new BCM()._streamIterator("hi")) received.push(c);
    });
    expect(received).toHaveLength(2);
  });

  it("early break calls underlying iterator.return() once (fail-open if return throws)", async () => {
    let returnCalls = 0;
    class BaseChatModel {
      model = "fake-chat-model";
      async *_streamIterator(_input: any): AsyncGenerator<any> {
        try {
          yield makeUsageChunk("a", { input_tokens: 10, output_tokens: 1 });
          yield makeUsageChunk("b", { input_tokens: 0, output_tokens: 2 });
          yield makeUsageChunk("c", { input_tokens: 0, output_tokens: 3 });
        } finally {
          // Generator finally runs when consumer breaks and outer wrapper
          // propagates return() — count as close signal.
          returnCalls++;
        }
      }
    }
    instrument(BaseChatModel);
    const session = newSession();
    await _getSessionStorage().run(session, async () => {
      let n = 0;
      for await (const _ of new BaseChatModel()._streamIterator("hi")) {
        if (++n === 1) break;
      }
    });
    expect(returnCalls).toBe(1);
    // Partial usage still stashed under reserved order 0.
    const usage = session._pendingCompositions[`${session.traceId}:0`]?.usage;
    expect(usage?.input_tokens).toBe(10);
    expect(usage?.output_tokens).toBe(1);
  });

  it("underlying iterator.return() throw does not break customer early-break", async () => {
    class BaseChatModel {
      model = "fake";
      _streamIterator(_input: any): AsyncIterable<any> {
        const chunks = [
          makeUsageChunk("a", { input_tokens: 1, output_tokens: 1 }),
          makeUsageChunk("b", { input_tokens: 0, output_tokens: 1 }),
        ];
        let i = 0;
        return {
          [Symbol.asyncIterator]() {
            return {
              async next() {
                if (i >= chunks.length) return { done: true, value: undefined };
                return { done: false, value: chunks[i++] };
              },
              async return() {
                throw new Error("return boom");
              },
            };
          },
        };
      }
    }
    instrument(BaseChatModel);
    const session = newSession();
    await expect(
      _getSessionStorage().run(session, async () => {
        let n = 0;
        for await (const _ of new BaseChatModel()._streamIterator("hi")) {
          if (++n === 1) break;
        }
      }),
    ).resolves.toBeUndefined();
  });
});
