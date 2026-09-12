/**
 * G3-14-2 — NON-streaming openai-wire calls lost their token detail objects.
 *
 * `@traceloop/instrumentation-openai` sets only the bare
 * `gen_ai.usage.{input,output}_tokens` attrs on a non-streaming span, silently
 * dropping the response's `prompt_tokens_details` / `completion_tokens_details`
 * (cached_tokens, reasoning_tokens). Telemetry then synthesized a bare 4-key
 * `usage.raw`, so reasoning/cached tokens never reached the collector for any
 * serving provider reached through the `openai` npm package (OpenAI, xAI,
 * Together via baseURL). Streamed calls were already covered by two verbatim
 * escape hatches, both stream-gated.
 *
 * The fix, in two halves:
 * 1. enforcer `_stashNonStreamVerbatimUsage` — at the non-streaming post-call
 *    seam, deep-clones the response's verbatim `usage` into
 *    `_pendingCompositions[`${traceId}:${order}`].nonStreamVerbatimRawUsage`.
 *    Gated to wireKey "openai" (Anthropic shares the seam), to a plain
 *    non-array usage object, to DETAIL-BEARING usage only, and to a positive
 *    finite prompt/completion count. Fail-open — never throws, no stash.
 * 2. telemetry onEnd — a new `usageRaw` precedence arm placed AFTER the stream
 *    stash and stream-guard arms and BEFORE the constructed synth. It fires
 *    only when the clone is detail-bearing AND its prompt/completion counts
 *    match the attr-derived counts EXACTLY (a mismatch means the stash belongs
 *    to a different call). When it fires, the provider's usage ships verbatim;
 *    when it doesn't, the constructed raw stays byte-identical to before.
 *
 * Everything here is offline: fake spans + planted composition entries, no
 * OpenAI SDK, no instrumentor, no network.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { setClient } from "../src/state";
import { TPSession, _getSessionStorage } from "../src/context";
import { TokenPoliceSpanProcessor } from "../src/telemetry";
import { __test__ as enforcerTest } from "../src/enforcer";

const _stashNonStreamVerbatimUsage = enforcerTest._stashNonStreamVerbatimUsage;
const _hasUsageDetailObjects = enforcerTest._hasUsageDetailObjects;

// ─────────────────────────── telemetry harness ───────────────────────────
// Mirrors tests/streamStashVerbatimUsage.test.ts exactly — same session, same
// fake span, same nextTick drain — so the two suites stay comparable.

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

/** OpenAI chat span. `usageAttrs` empty ⇒ the streaming case (no token attrs). */
function fakeSpan(session: TPSession, usageAttrs: Record<string, number> | null) {
  const attrs: Record<string, unknown> = {
    "gen_ai.system": "openai",
    "gen_ai.request.model": "grok-4",
    "tp.trace_id": session.traceId,
    "tp.span_order": 0,
  };
  if (usageAttrs) Object.assign(attrs, usageAttrs);
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

/** Plant a pending-composition entry at the span's compKey (`${traceId}:0`). */
function plant(session: TPSession, entry: Record<string, unknown>) {
  (session as any)._pendingCompositions[`${session.traceId}:0`] = entry;
}

/**
 * Drives onEnd and returns the tp.log args THIS invocation produced — indexed
 * off the pre-run length, not `logged[0]`, so a test that runs two spans reads
 * its own second row. Returns undefined when the span logged nothing (the
 * usage-less early-return case below relies on that).
 */
async function runOnEnd(session: TPSession, span: any): Promise<any[]> {
  const before = logged.length;
  await _getSessionStorage().run(session, async () => {
    new TokenPoliceSpanProcessor().onEnd(span);
    // The log is deferred to process.nextTick — let it drain.
    await new Promise((r) => setTimeout(r, 0));
  });
  return logged[before];
}

/** tp.log positional args: …, inputTokens(6), outputTokens(7), cachedTokens(8). */
const INPUT_TOKENS_ARG = 6;
const OUTPUT_TOKENS_ARG = 7;
const CACHED_TOKENS_ARG = 8;

/** The eight keys the constructed (non-verbatim) raw always carries. */
const SYNTH_RAW_KEYS = [
  "cache_creation_input_tokens",
  "cache_read_input_tokens",
  "completion_tokens",
  "completion_tokens_details",
  "input_tokens",
  "output_tokens",
  "prompt_tokens",
  "prompt_tokens_details",
];

/** Asserts `raw` is exactly today's constructed synth for the given counts. */
function expectConstructedSynth(raw: any, prompt: number, completion: number) {
  expect(Object.keys(raw).sort()).toEqual(SYNTH_RAW_KEYS);
  expect(raw.prompt_tokens).toBe(prompt);
  expect(raw.input_tokens).toBe(prompt);
  expect(raw.completion_tokens).toBe(completion);
  expect(raw.output_tokens).toBe(completion);
  expect(raw.prompt_tokens_details).toBeUndefined();
  expect(raw.completion_tokens_details).toBeUndefined();
  expect(raw.cache_read_input_tokens).toBeUndefined();
  expect(raw.cache_creation_input_tokens).toBeUndefined();
  // The provider's total_tokens is NOT a key of the constructed shape.
  expect("total_tokens" in raw).toBe(false);
}

/** Realistic xAI (openai-wire) non-streaming usage for a reasoning call. */
function xaiUsage() {
  return {
    prompt_tokens: 577,
    completion_tokens: 200,
    total_tokens: 1200,
    prompt_tokens_details: {
      text_tokens: 449,
      audio_tokens: 0,
      image_tokens: 0,
      cached_tokens: 128,
    },
    completion_tokens_details: {
      reasoning_tokens: 423,
      audio_tokens: 0,
      accepted_prediction_tokens: 0,
      rejected_prediction_tokens: 0,
    },
    num_sources_used: 0,
  };
}

const ATTRS_577_200 = {
  "gen_ai.usage.input_tokens": 577,
  "gen_ai.usage.output_tokens": 200,
};

// ───────────────────── telemetry precedence arm ─────────────────────

describe("G3-14-2 telemetry: non-stream verbatim usage.raw arm", () => {
  it("happy path: detail-bearing stash matching the attrs → usage.raw is the verbatim provider object", async () => {
    const session = newSession();
    const verbatim = xaiUsage();
    plant(session, { nonStreamVerbatimRawUsage: verbatim });

    const args = await runOnEnd(session, fakeSpan(session, ATTRS_577_200));
    const opts = args.at(-1);

    // Whole provider usage forwarded — details AND total_tokens survive.
    // Compared against a FRESH copy, not the planted reference.
    expect(opts.usage.raw).toEqual(xaiUsage());
    expect(opts.usage.raw).toBe(verbatim);
    expect(opts.usage.raw.total_tokens).toBe(1200);
    expect(opts.usage.raw.prompt_tokens_details.cached_tokens).toBe(128);
    expect(opts.usage.raw.completion_tokens_details.reasoning_tokens).toBe(423);
    expect(opts.usage.raw.num_sources_used).toBe(0);
    expect(opts.usage.shape).toBe("openai_chat");

    // Positional counts stay attr-derived (unchanged by this fix).
    expect(args[INPUT_TOKENS_ARG]).toBe(577);
    expect(args[OUTPUT_TOKENS_ARG]).toBe(200);
    expect(args[CACHED_TOKENS_ARG]).toBe(0);
  });

  it("attr-only path (no stash) → usage.raw is byte-identical to the constructed synth", async () => {
    const session = newSession();
    // No pending-composition entry at all: the pre-fix world.
    const args = await runOnEnd(session, fakeSpan(session, ATTRS_577_200));
    const opts = args.at(-1);

    expectConstructedSynth(opts.usage.raw, 577, 200);
    expect(args[INPUT_TOKENS_ARG]).toBe(577);
    expect(args[OUTPUT_TOKENS_ARG]).toBe(200);
  });

  it("entry present but WITHOUT the verbatim field → constructed synth (arm inert)", async () => {
    const session = newSession();
    plant(session, { service_tier: "flex" });

    const opts = (await runOnEnd(session, fakeSpan(session, ATTRS_577_200))).at(-1);
    expectConstructedSynth(opts.usage.raw, 577, 200);
    // Unrelated stash fields still work.
    expect(opts.usage.tier).toBe("flex");
  });

  it("consistency rejection: stash prompt_tokens mismatches the attrs → constructed synth", async () => {
    const session = newSession();
    plant(session, {
      nonStreamVerbatimRawUsage: { ...xaiUsage(), prompt_tokens: 999 },
    });

    const opts = (await runOnEnd(session, fakeSpan(session, ATTRS_577_200))).at(-1);
    // Attr-derived counts win and NO detail objects leak from the rejected stash.
    expectConstructedSynth(opts.usage.raw, 577, 200);
    expect(opts.usage.raw.prompt_tokens).not.toBe(999);
  });

  it("consistency rejection: stash completion_tokens mismatches the attrs → constructed synth", async () => {
    const session = newSession();
    plant(session, {
      nonStreamVerbatimRawUsage: { ...xaiUsage(), completion_tokens: 888 },
    });

    const opts = (await runOnEnd(session, fakeSpan(session, ATTRS_577_200))).at(-1);
    expectConstructedSynth(opts.usage.raw, 577, 200);
    expect(opts.usage.raw.completion_tokens).not.toBe(888);
  });

  it("detail-bearing rejection: counts match but no detail objects → constructed synth", async () => {
    const session = newSession();
    plant(session, {
      nonStreamVerbatimRawUsage: {
        prompt_tokens: 577,
        completion_tokens: 200,
        total_tokens: 777,
      },
    });

    const opts = (await runOnEnd(session, fakeSpan(session, ATTRS_577_200))).at(-1);
    // total_tokens absent proves the stash was not forwarded.
    expectConstructedSynth(opts.usage.raw, 577, 200);
  });

  it("detail-bearing rejection: detail keys present but not objects → constructed synth", async () => {
    const session = newSession();
    plant(session, {
      nonStreamVerbatimRawUsage: {
        prompt_tokens: 577,
        completion_tokens: 200,
        total_tokens: 777,
        prompt_tokens_details: null,
        completion_tokens_details: 0,
      },
    });

    const opts = (await runOnEnd(session, fakeSpan(session, ATTRS_577_200))).at(-1);
    expectConstructedSynth(opts.usage.raw, 577, 200);
  });

  it("camelCase detail objects also qualify (promptTokensDetails)", async () => {
    const session = newSession();
    const verbatim = {
      prompt_tokens: 577,
      completion_tokens: 200,
      total_tokens: 777,
      promptTokensDetails: { cachedTokens: 128 },
    };
    plant(session, { nonStreamVerbatimRawUsage: verbatim });

    const opts = (await runOnEnd(session, fakeSpan(session, ATTRS_577_200))).at(-1);
    expect(opts.usage.raw).toEqual(verbatim);
  });

  it("PRIORITY: stream stash raw still wins when both stashes are present", async () => {
    const session = newSession();
    plant(session, {
      // Stream tap: positional counts are cache-EXCLUSIVE, raw is inclusive.
      usage: {
        input_tokens: 60,
        output_tokens: 20,
        cached_tokens: 40,
        raw: {
          prompt_tokens: 100,
          completion_tokens: 20,
          prompt_tokens_details: { cached_tokens: 40 },
          _from: "stream-tap",
        },
      },
      // The non-stream clone ALSO satisfies the consistency gate here
      // (100 === 60 + 40, 20 === 20) — so this asserts arm ORDER, not luck.
      nonStreamVerbatimRawUsage: {
        prompt_tokens: 100,
        completion_tokens: 20,
        prompt_tokens_details: { cached_tokens: 40 },
        _from: "non-stream-stash",
      },
    });

    // No token attrs ⇒ the stream-stash fallback supplies the numbers.
    const opts = (await runOnEnd(session, fakeSpan(session, null))).at(-1);
    expect(opts.usage.raw._from).toBe("stream-tap");
  });

  it("usedStashNumbers arithmetic: gate compares against the cache-INCLUSIVE count", async () => {
    const session = newSession();
    // Stream stash supplies the numbers but carries no verbatim raw (legacy /
    // clone failed), so the non-stream arm is reached with usedStashNumbers
    // true — its gate must use inputTokens + cachedTokens (60 + 40 = 100).
    plant(session, {
      usage: { input_tokens: 60, output_tokens: 20, cached_tokens: 40 },
      nonStreamVerbatimRawUsage: {
        prompt_tokens: 100,
        completion_tokens: 20,
        prompt_tokens_details: { cached_tokens: 40 },
        _from: "non-stream-stash",
      },
    });

    const opts = (await runOnEnd(session, fakeSpan(session, null))).at(-1);
    expect(opts.usage.raw._from).toBe("non-stream-stash");
    expect(opts.usage.raw.prompt_tokens).toBe(100);

    // …and the cache-EXCLUSIVE 60 does NOT satisfy it.
    const s2 = newSession();
    plant(s2, {
      usage: { input_tokens: 60, output_tokens: 20, cached_tokens: 40 },
      nonStreamVerbatimRawUsage: {
        prompt_tokens: 60,
        completion_tokens: 20,
        prompt_tokens_details: { cached_tokens: 40 },
        _from: "non-stream-stash",
      },
    });
    const opts2 = (await runOnEnd(s2, fakeSpan(s2, null))).at(-1);
    expect(opts2.usage.raw._from).toBeUndefined();
    // Falls through to the constructed synth, which adds cached back.
    expect(opts2.usage.raw.prompt_tokens).toBe(100);
  });

  it("attrs beat the stream stash yet the non-stream verbatim arm still applies", async () => {
    const session = newSession();
    // Attrs carry usage ⇒ usedStashNumbers stays false and the stream raw
    // (999/888) is never read; the non-stream clone is checked against attrs.
    plant(session, {
      usage: {
        input_tokens: 60,
        output_tokens: 20,
        cached_tokens: 40,
        raw: { prompt_tokens: 999, completion_tokens: 888 },
      },
      nonStreamVerbatimRawUsage: xaiUsage(),
    });

    const opts = (await runOnEnd(session, fakeSpan(session, ATTRS_577_200))).at(-1);
    expect(opts.usage.raw.prompt_tokens).toBe(577);
    expect(opts.usage.raw.prompt_tokens).not.toBe(999);
    expect(opts.usage.raw.completion_tokens_details.reasoning_tokens).toBe(423);
  });

  it("zero-attr span with ONLY the non-stream stash still early-returns (no row)", async () => {
    const session = newSession();
    // No token attrs and no stream stash ⇒ inputTokens/outputTokens both 0.
    // The verbatim stash must NOT resurrect a usage-less span into a row.
    plant(session, { nonStreamVerbatimRawUsage: xaiUsage() });

    await runOnEnd(session, fakeSpan(session, null));
    expect(logged.length).toBe(0);
  });
});

// ───────────────────── enforcer stash unit tests ─────────────────────
// Reached through the module's existing `__test__` re-export (the codebase's
// established convention for driving non-exported enforcer internals).

/** Minimal stand-in for the TPSession fields the stash touches. */
function fakeSession(spanCounter = 5) {
  return {
    traceId: "t".repeat(32),
    spanCounter,
    _pendingCompositions: {} as Record<string, any>,
  } as any;
}

function detailUsage() {
  return {
    prompt_tokens: 577,
    completion_tokens: 200,
    total_tokens: 1200,
    prompt_tokens_details: { cached_tokens: 128 },
    completion_tokens_details: { reasoning_tokens: 423 },
  };
}

describe("G3-14-2 enforcer: _hasUsageDetailObjects", () => {
  it("true for snake_case prompt/completion detail objects", () => {
    expect(_hasUsageDetailObjects({ prompt_tokens_details: {} })).toBe(true);
    expect(_hasUsageDetailObjects({ completion_tokens_details: { a: 1 } })).toBe(true);
  });

  it("true for camelCase detail objects", () => {
    expect(_hasUsageDetailObjects({ promptTokensDetails: {} })).toBe(true);
    expect(_hasUsageDetailObjects({ completionTokensDetails: {} })).toBe(true);
  });

  it("false for bare token usage and for non-object detail values", () => {
    expect(_hasUsageDetailObjects({ prompt_tokens: 5, completion_tokens: 6 })).toBe(false);
    expect(_hasUsageDetailObjects({ prompt_tokens_details: null })).toBe(false);
    expect(_hasUsageDetailObjects({ prompt_tokens_details: 0 })).toBe(false);
    expect(_hasUsageDetailObjects({ completion_tokens_details: "x" })).toBe(false);
  });

  it("never throws on hostile input", () => {
    expect(_hasUsageDetailObjects(null)).toBe(false);
    expect(_hasUsageDetailObjects(undefined)).toBe(false);
    expect(_hasUsageDetailObjects(42)).toBe(false);
    expect(
      _hasUsageDetailObjects({
        get prompt_tokens_details(): any {
          throw new Error("boom");
        },
      }),
    ).toBe(false);
  });
});

describe("G3-14-2 enforcer: _stashNonStreamVerbatimUsage", () => {
  it("openai wire + detail-bearing usage → stashed under `${traceId}:${order}`", () => {
    const session = fakeSession();
    _stashNonStreamVerbatimUsage("openai", { usage: detailUsage() }, session, 3);

    const entry = session._pendingCompositions[`${session.traceId}:3`];
    expect(entry).toBeDefined();
    expect(entry.nonStreamVerbatimRawUsage).toEqual(detailUsage());
  });

  it("stashes a DEEP CLONE — later mutation of the response cannot change it", () => {
    const session = fakeSession();
    const usage = detailUsage();
    _stashNonStreamVerbatimUsage("openai", { usage }, session, 0);

    const stashed = session._pendingCompositions[`${session.traceId}:0`]
      .nonStreamVerbatimRawUsage;
    expect(stashed).not.toBe(usage);
    expect(stashed.prompt_tokens_details).not.toBe(usage.prompt_tokens_details);

    // Mutate the original, top level and nested.
    usage.prompt_tokens = 1;
    usage.prompt_tokens_details.cached_tokens = 1;
    (usage as any).completion_tokens_details.reasoning_tokens = 1;

    expect(stashed.prompt_tokens).toBe(577);
    expect(stashed.prompt_tokens_details.cached_tokens).toBe(128);
    expect(stashed.completion_tokens_details.reasoning_tokens).toBe(423);
  });

  it("merges into an existing composition entry instead of clobbering it", () => {
    const session = fakeSession();
    session._pendingCompositions[`${session.traceId}:2`] = {
      service_tier: "flex",
      prompt: [{ role: "user" }],
    };
    _stashNonStreamVerbatimUsage("openai", { usage: detailUsage() }, session, 2);

    const entry = session._pendingCompositions[`${session.traceId}:2`];
    expect(entry.service_tier).toBe("flex");
    expect(entry.prompt).toEqual([{ role: "user" }]);
    expect(entry.nonStreamVerbatimRawUsage.total_tokens).toBe(1200);
  });

  it("omitted order falls back to spanCounter - 1 (floored at 0)", () => {
    const s1 = fakeSession(5);
    _stashNonStreamVerbatimUsage("openai", { usage: detailUsage() }, s1);
    expect(s1._pendingCompositions[`${s1.traceId}:4`]).toBeDefined();

    const s2 = fakeSession(0);
    _stashNonStreamVerbatimUsage("openai", { usage: detailUsage() }, s2);
    expect(s2._pendingCompositions[`${s2.traceId}:0`]).toBeDefined();

    // order 0 is honoured, not treated as absent.
    const s3 = fakeSession(9);
    _stashNonStreamVerbatimUsage("openai", { usage: detailUsage() }, s3, 0);
    expect(s3._pendingCompositions[`${s3.traceId}:0`]).toBeDefined();
    expect(s3._pendingCompositions[`${s3.traceId}:8`]).toBeUndefined();
  });

  it("non-openai wire keys never stash (anthropic shares this seam)", () => {
    for (const wire of ["anthropic", "google", "bedrock", "OpenAI", "", undefined as any]) {
      const session = fakeSession();
      _stashNonStreamVerbatimUsage(wire, { usage: detailUsage() }, session, 0);
      expect(Object.keys(session._pendingCompositions)).toEqual([]);
    }
  });

  it("usage without detail objects never stashes", () => {
    const session = fakeSession();
    _stashNonStreamVerbatimUsage(
      "openai",
      { usage: { prompt_tokens: 577, completion_tokens: 200, total_tokens: 777 } },
      session,
      0,
    );
    expect(Object.keys(session._pendingCompositions)).toEqual([]);
  });

  it("null / non-object / array usage never stashes and never throws", () => {
    for (const usage of [null, undefined, 0, 1, "usage", true, [detailUsage()]]) {
      const session = fakeSession();
      expect(() =>
        _stashNonStreamVerbatimUsage("openai", { usage }, session, 0),
      ).not.toThrow();
      expect(Object.keys(session._pendingCompositions)).toEqual([]);
    }
    // result itself missing / not an object.
    for (const result of [null, undefined, 5, "x"]) {
      const session = fakeSession();
      expect(() =>
        _stashNonStreamVerbatimUsage("openai", result, session, 0),
      ).not.toThrow();
      expect(Object.keys(session._pendingCompositions)).toEqual([]);
    }
  });

  it("zero / absent / non-finite token counts never stash", () => {
    const bad = [
      { prompt_tokens: 0, completion_tokens: 0, prompt_tokens_details: {} },
      { prompt_tokens_details: {} },
      { prompt_tokens: NaN, completion_tokens: NaN, prompt_tokens_details: {} },
      { prompt_tokens: "abc", completion_tokens: null, prompt_tokens_details: {} },
      { prompt_tokens: Infinity, completion_tokens: -5, prompt_tokens_details: {} },
    ];
    for (const usage of bad) {
      const session = fakeSession();
      _stashNonStreamVerbatimUsage("openai", { usage }, session, 0);
      expect(Object.keys(session._pendingCompositions)).toEqual([]);
    }
  });

  it("a single positive count is enough (prompt 0 + completion > 0)", () => {
    const session = fakeSession();
    _stashNonStreamVerbatimUsage(
      "openai",
      {
        usage: {
          prompt_tokens: 0,
          completion_tokens: 200,
          completion_tokens_details: { reasoning_tokens: 199 },
        },
      },
      session,
      0,
    );
    expect(
      session._pendingCompositions[`${session.traceId}:0`].nonStreamVerbatimRawUsage
        .completion_tokens,
    ).toBe(200);
  });

  it("circular usage object → fails open (no throw, no stash)", () => {
    const usage: any = detailUsage();
    usage.self = usage;
    const session = fakeSession();
    expect(() =>
      _stashNonStreamVerbatimUsage("openai", { usage }, session, 0),
    ).not.toThrow();
    expect(Object.keys(session._pendingCompositions)).toEqual([]);
  });

  it("golden rule: never throws for a battery of hostile inputs", () => {
    const hostile: Array<[string, any, any]> = [
      // session missing entirely.
      ["null session", { usage: detailUsage() }, null],
      ["undefined session", { usage: detailUsage() }, undefined],
      // session without the composition map (property access on undefined).
      ["session w/o _pendingCompositions", { usage: detailUsage() }, { traceId: "t", spanCounter: 1 }],
      // frozen composition map — assignment throws under module strict mode.
      [
        "frozen _pendingCompositions",
        { usage: detailUsage() },
        { traceId: "t", spanCounter: 1, _pendingCompositions: Object.freeze({}) },
      ],
      // getters that blow up on the paths the stash walks.
      [
        "throwing result.usage getter",
        {
          get usage(): any {
            throw new Error("boom");
          },
        },
        fakeSession(),
      ],
      [
        "throwing prompt_tokens valueOf",
        {
          usage: {
            prompt_tokens: {
              valueOf() {
                throw new Error("boom");
              },
            },
            prompt_tokens_details: {},
          },
        },
        fakeSession(),
      ],
      [
        "throwing toJSON during clone",
        {
          usage: {
            prompt_tokens: 5,
            completion_tokens: 5,
            prompt_tokens_details: {
              toJSON() {
                throw new Error("boom");
              },
            },
          },
        },
        fakeSession(),
      ],
      [
        "BigInt in usage (JSON.stringify throws)",
        {
          usage: {
            prompt_tokens: 5,
            completion_tokens: 5,
            prompt_tokens_details: { cached_tokens: BigInt(1) },
          },
        },
        fakeSession(),
      ],
      [
        "throwing traceId getter",
        { usage: detailUsage() },
        {
          get traceId(): string {
            throw new Error("boom");
          },
          spanCounter: 1,
          _pendingCompositions: {},
        },
      ],
    ];

    for (const [label, result, session] of hostile) {
      expect(
        () => _stashNonStreamVerbatimUsage("openai", result, session, 0),
        label,
      ).not.toThrow();
    }
  });
});
