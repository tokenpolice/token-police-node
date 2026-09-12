/**
 * N1 — Mode-A stream-fallback double-net (under-bills cached streaming calls).
 *
 * The enforcer's OpenAI-SDK Traceloop stream tap stashes token usage in
 * `_pendingCompositions[compKey].usage`; telemetry's Mode-A onEnd falls back to
 * it when the instrumentor span carried no `gen_ai.usage.*` attrs (streaming).
 * The stashed positional `input_tokens` is cache-EXCLUSIVE (netted by
 * `_extractUsage`), but the OpenAI-family shape mappers (openai_chat /
 * openrouter_routed / openai_compatible_chat) treat `raw.prompt_tokens` as
 * cache-INCLUSIVE and subtract cached themselves — so the netted value was
 * double-subtracted, under-billing by exactly `cached` at the input rate.
 *
 * The fix: the tap also stashes the provider's verbatim chunk usage
 * (`cd.usage.raw`, cache-inclusive prompt_tokens + prompt_tokens_details), and
 * onEnd emits it verbatim as `raw`. When no verbatim clone is present (legacy /
 * clone failed) it adds cached back to make prompt_tokens inclusive — but ONLY
 * when the numbers came from the stash. The attr path stays bit-identical.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
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
});

function newSession(): TPSession {
  return new TPSession({
    userId: "u1",
    paidPlan: "pro",
    workflowName: "wf",
    traceId: "a".repeat(32),
    rootSpanId: "b".repeat(16),
  });
}

/** OpenAI chat span. When `withUsageAttrs` is false the gen_ai.usage.* attrs are
 * absent (the streaming case) so onEnd falls back to the stash. */
function fakeSpan(
  session: TPSession,
  withUsageAttrs: boolean,
  usageAttrs: Record<string, number> = {},
) {
  const attrs: Record<string, unknown> = {
    "gen_ai.system": "openai",
    "gen_ai.request.model": "gpt-4o",
    "tp.trace_id": session.traceId,
    "tp.span_order": 0,
  };
  if (withUsageAttrs) Object.assign(attrs, usageAttrs);
  return {
    attributes: attrs,
    name: "openai.chat",
    startTime: [0, 0] as [number, number],
    endTime: [1, 0] as [number, number],
    spanContext: () => ({
      traceId: "0000000000000000000000000000abcd",
      spanId: "0000000000001234",
    }),
    parentSpanId: undefined,
    status: { code: 0 },
  } as any;
}

/** Seed the stream tap's stash at the span's compKey (`${traceId}:0`). */
function stash(session: TPSession, usage: any) {
  (session as any)._pendingCompositions[`${session.traceId}:0`] = { usage };
}

async function runOnEnd(session: TPSession, span: any): Promise<any[]> {
  await _getSessionStorage().run(session, async () => {
    new TokenPoliceSpanProcessor().onEnd(span);
    // The log is deferred to process.nextTick — let it drain.
    await new Promise((r) => setTimeout(r, 0));
  });
  return logged[0];
}

describe("N1 stream-stash verbatim usage (no double-net)", () => {
  it("cached stash + verbatim raw + empty attrs → raw.prompt_tokens inclusive (100, not 60)", async () => {
    const session = newSession();
    stash(session, {
      // positional counts are cache-EXCLUSIVE (netted): 60 = 100 - 40
      input_tokens: 60,
      output_tokens: 20,
      cached_tokens: 40,
      // provider verbatim chunk usage — cache-INCLUSIVE prompt_tokens
      raw: {
        prompt_tokens: 100,
        completion_tokens: 20,
        prompt_tokens_details: { cached_tokens: 40 },
      },
    });

    const opts = (await runOnEnd(session, fakeSpan(session, false))).at(-1);
    expect(opts.usage.shape).toBe("openai_chat");
    // Verbatim provider usage forwarded — inclusive prompt_tokens, NOT the netted 60.
    expect(opts.usage.raw.prompt_tokens).toBe(100);
    expect(opts.usage.raw.prompt_tokens_details.cached_tokens).toBe(40);
    expect(opts.usage.raw.completion_tokens).toBe(20);
  });

  it("cached stash, NO verbatim raw (legacy) → synthesized inclusive prompt_tokens (60+40=100)", async () => {
    const session = newSession();
    stash(session, {
      input_tokens: 60,
      output_tokens: 20,
      cached_tokens: 40,
      // no `raw` — the clone failed / older tap
    });

    const opts = (await runOnEnd(session, fakeSpan(session, false))).at(-1);
    expect(opts.usage.shape).toBe("openai_chat");
    // cached added back so the mapper's single subtraction lands on 60.
    expect(opts.usage.raw.prompt_tokens).toBe(100);
    expect(opts.usage.raw.input_tokens).toBe(100);
    expect(opts.usage.raw.prompt_tokens_details.cached_tokens).toBe(40);
    expect(opts.usage.raw.output_tokens).toBe(20);
  });

  it("no-cache stash (cached 0) with verbatim raw → prompt_tokens == input (unchanged)", async () => {
    const session = newSession();
    stash(session, {
      input_tokens: 100,
      output_tokens: 20,
      cached_tokens: 0,
      raw: { prompt_tokens: 100, completion_tokens: 20 },
    });

    const opts = (await runOnEnd(session, fakeSpan(session, false))).at(-1);
    expect(opts.usage.raw.prompt_tokens).toBe(100);
    expect(opts.usage.raw.completion_tokens).toBe(20);
    // no cached ⇒ no details block from the verbatim object
    expect(opts.usage.raw.prompt_tokens_details).toBeUndefined();
  });

  it("attr path wins over stash raw — stash verbatim NOT used", async () => {
    const session = newSession();
    // Attrs carry usage → the stash fallback must NOT fire.
    stash(session, {
      input_tokens: 60,
      output_tokens: 20,
      cached_tokens: 40,
      raw: { prompt_tokens: 999, completion_tokens: 888 },
    });

    const opts = (
      await runOnEnd(
        session,
        fakeSpan(session, true, {
          "gen_ai.usage.input_tokens": 80,
          "gen_ai.usage.output_tokens": 15,
        }),
      )
    ).at(-1);
    // Attr-derived usage wins; the stash raw (999/888) is never read.
    expect(opts.usage.raw.prompt_tokens).toBe(80);
    expect(opts.usage.raw.input_tokens).toBe(80);
    expect(opts.usage.raw.completion_tokens).toBe(15);
    expect(opts.usage.raw.prompt_tokens).not.toBe(999);
  });
});
