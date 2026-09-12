/**
 * G3-O1 (Fix C) — LangChain / LangGraph reasoning tokens (Node).
 *
 * Traceloop's LangChain instrumentor emits only input/output/total token attrs,
 * so `gen_ai.usage.reasoning_tokens` is ALWAYS 0 on an LC span — and that attr
 * was telemetry's only reasoning source. Every o-series / gpt-5 and Gemini
 * thinking call routed through LangChain or LangGraph therefore landed with
 * `reasoning_output_tokens = 0`.
 *
 * The data IS available: `@langchain/openai` (and the Gemini binding) populate
 * `AIMessage.usage_metadata.output_token_details.reasoning`.
 *
 * Fix under test:
 *   C1 `_stashLangchainUsageFromMessage` (stream path) additionally stashes
 *      `reasoning_tokens` on the pending-composition SLOT — MAX across the
 *      per-chunk refreshes, never a sum (concat() already sums it, and a
 *      provider restating absolutes would double-count). Slot-level, because
 *      the `.usage` sub-object is REPLACED on every refresh.
 *   C2 `_captureLangchainResponse` (generate path) does the same, deliberately
 *      OUTSIDE the `hasWireUsage` guard — ChatOpenAI always carries
 *      `llmOutput.tokenUsage`, so that guarded block never runs — and WITHOUT
 *      writing a `.usage` stash, which would flip telemetry's preferStash /
 *      stream-guard decisions and perturb the billed in/out/cached numbers.
 *   C3 telemetry falls back from the zero attr to the stash, clamped to
 *      `outputTokens`, and emits it under the shape's native key:
 *        * openai_chat → `completion_tokens_details.reasoning_tokens`
 *          (SUBSET: mapper does text = completion − reasoning);
 *        * google_genai → `thoughts_token_count` + a netted
 *          `candidates_token_count = output − reasoning` (ADDITIVE: the google
 *          mapper reads candidates as EXCLUSIVE of thoughts, so emitting
 *          thoughts alone would INFLATE total output).
 *
 * Load-bearing invariants pinned here:
 *   * The ATTR path (every non-LC span) stays BIT-IDENTICAL — attr > 0 wins and
 *     never triggers the google split.
 *   * TOTAL OUTPUT never moves on either shape.
 *   * in/out/cached stash fields are untouched by the new code.
 *   * Fail-open at every seam (golden rule).
 *
 * All offline against fake LLMResult / AIMessageChunk objects — no network,
 * no LangChain.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { setClient } from "../src/state";
import { session, TPSession, _getSessionStorage } from "../src/context";
import { __test__ as enforcerTest } from "../src/enforcer";
import { TokenPoliceSpanProcessor } from "../src/telemetry";

const { _captureLangchainResponse, _captureLangchainResponseFromMessage } =
  enforcerTest as any;

let logged: any[];
beforeEach(() => {
  logged = [];
  setClient({
    firewall: "off",
    log: (...args: any[]) => logged.push(args),
  } as any);
});
afterEach(() => {
  setClient(null as any);
  for (const thunk of enforcerTest._restoreThunks.splice(0)) {
    try {
      thunk();
    } catch {
      /* ignore */
    }
  }
});

// ── fakes ────────────────────────────────────────────────────────────────

/** A LangChain-standard usage_metadata object. */
function um(
  input_tokens = 100,
  output_tokens = 400,
  reasoning?: number,
): Record<string, unknown> {
  const o: Record<string, unknown> = {
    input_tokens,
    output_tokens,
    total_tokens: input_tokens + output_tokens,
  };
  if (reasoning !== undefined) {
    o.output_token_details = { reasoning };
  }
  return o;
}

/** A LangChain Generation carrying an AIMessage with usage_metadata. */
function gen(usage_metadata: unknown, text = "hi"): any {
  return { text, message: { content: text, usage_metadata } };
}

/** A folded AIMessageChunk as the streamIterator wrapper hands it over. */
function msg(usage_metadata: unknown, text = "hi"): any {
  return { content: text, tool_calls: [], usage_metadata, _getType: () => "ai" };
}

function slot(s: any, order = 0): any {
  return s._pendingCompositions[`${s.traceId}:${order}`];
}

function newSession(): TPSession {
  return new TPSession({
    userId: "u1",
    paidPlan: "pro",
    workflowName: "wf",
    traceId: "c".repeat(32),
    rootSpanId: "d".repeat(16),
  });
}

function fakeSpan(
  s: TPSession,
  order: number,
  opts: {
    system?: string;
    model?: string;
    input?: number;
    output?: number;
    cached?: number;
    reasoning?: number;
  } = {},
): any {
  const attrs: Record<string, unknown> = {
    "gen_ai.system": opts.system ?? "openai",
    "gen_ai.request.model": opts.model ?? "gpt-5",
    "tp.trace_id": s.traceId,
    "tp.span_order": order,
  };
  if (opts.input != null) attrs["gen_ai.usage.input_tokens"] = opts.input;
  if (opts.output != null) attrs["gen_ai.usage.output_tokens"] = opts.output;
  if (opts.cached != null) attrs["gen_ai.usage.cached_tokens"] = opts.cached;
  if (opts.reasoning != null) {
    attrs["gen_ai.usage.reasoning_tokens"] = opts.reasoning;
  }
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

/** Run onEnd under the session ALS and return the log() call args. */
async function runOnEnd(s: TPSession, span: any): Promise<any[]> {
  await _getSessionStorage().run(s, async () => {
    new TokenPoliceSpanProcessor().onEnd(span);
    await new Promise((r) => setTimeout(r, 0));
  });
  return logged[0];
}

/** Seed a pending-composition slot with arbitrary fields. */
function seedSlot(s: TPSession, order: number, data: Record<string, any>): void {
  (s as any)._pendingCompositions[`${s.traceId}:${order}`] = data;
}

// ═══════════════════════════════════════════════════════════════════════
// C1 — _stashLangchainUsageFromMessage (driven via the exported
//      _captureLangchainResponseFromMessage, its only public caller)
// ═══════════════════════════════════════════════════════════════════════
describe("C1 stream-path reasoning stash", () => {
  it("reasoning stashed from output_token_details.reasoning", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponseFromMessage(msg(um(100, 400, 357)), 0, {
        cached_tokens: 0,
        cache_creation_tokens: 0,
      });
      expect(slot(s, 0).reasoning_tokens).toBe(357);
    });
  });

  it("stash is slot-level, NOT on .usage (which is replaced per refresh)", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponseFromMessage(msg(um(100, 400, 357)), 0);
      expect(slot(s, 0).reasoning_tokens).toBe(357);
      expect(slot(s, 0).usage.reasoning_tokens).toBeUndefined();
    });
  });

  it("in/out/cached stash fields are unchanged by the reasoning write", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponseFromMessage(msg(um(100, 400, 357)), 0, {
        cached_tokens: 40,
        cache_creation_tokens: 80,
      });
      expect(slot(s, 0).usage).toEqual({
        usage_source: "langchain_message",
        input_tokens: 100,
        output_tokens: 400,
        cached_tokens: 40,
        cache_creation_tokens: 80,
      });
    });
  });

  it("MAX across refreshes: a lower later value does NOT clobber", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponseFromMessage(msg(um(100, 400, 357)), 0);
      _captureLangchainResponseFromMessage(msg(um(100, 400, 12)), 0);
      expect(slot(s, 0).reasoning_tokens).toBe(357);
    });
  });

  it("MAX across refreshes: a details-less later refresh does NOT clobber", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponseFromMessage(msg(um(100, 400, 357)), 0);
      _captureLangchainResponseFromMessage(msg(um(100, 420)), 0);
      expect(slot(s, 0).reasoning_tokens).toBe(357);
      // ...while the usage numbers DO track the latest refresh.
      expect(slot(s, 0).usage.output_tokens).toBe(420);
    });
  });

  it("MAX across refreshes: a higher later value wins (concat growth)", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponseFromMessage(msg(um(100, 200, 100)), 0);
      _captureLangchainResponseFromMessage(msg(um(100, 400, 357)), 0);
      expect(slot(s, 0).reasoning_tokens).toBe(357);
    });
  });

  it("never SUMS across refreshes (concat already summed)", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponseFromMessage(msg(um(100, 400, 357)), 0);
      _captureLangchainResponseFromMessage(msg(um(100, 400, 357)), 0);
      expect(slot(s, 0).reasoning_tokens).toBe(357);
      expect(slot(s, 0).reasoning_tokens).not.toBe(714);
    });
  });

  it("no reasoning detail → no reasoning_tokens key", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponseFromMessage(msg(um(100, 400)), 0);
      expect(slot(s, 0).reasoning_tokens).toBeUndefined();
    });
  });

  it("zero reasoning → no reasoning_tokens key", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponseFromMessage(msg(um(100, 400, 0)), 0);
      expect(slot(s, 0).reasoning_tokens).toBeUndefined();
    });
  });

  it("stash lands under the ORDER passed, not the span counter", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponseFromMessage(msg(um(100, 400, 357)), 7);
      expect(slot(s, 7).reasoning_tokens).toBe(357);
      expect(slot(s, 0)).toBeUndefined();
    });
  });

  const hostileMessages: Array<[string, unknown]> = [
    ["usage_metadata is a string", "lots"],
    ["output_token_details is a string", { input_tokens: 1, output_tokens: 1, output_token_details: "x" }],
    ["reasoning is NaN", { input_tokens: 1, output_tokens: 1, output_token_details: { reasoning: NaN } }],
    ["reasoning is an object", { input_tokens: 1, output_tokens: 1, output_token_details: { reasoning: {} } }],
    ["reasoning is a garbage string", { input_tokens: 1, output_tokens: 1, output_token_details: { reasoning: "abc" } }],
  ];
  for (const [name, meta] of hostileMessages) {
    it(`${name} → no throw, no reasoning stash`, () => {
      session({ name: "wf" }, (s) => {
        expect(() =>
          _captureLangchainResponseFromMessage(msg(meta), 0),
        ).not.toThrow();
        expect(slot(s, 0)?.reasoning_tokens).toBeUndefined();
      });
    });
  }

  it("output_token_details getter that THROWS → no throw, usage still stashed", () => {
    session({ name: "wf" }, (s) => {
      const meta: any = { input_tokens: 100, output_tokens: 400 };
      Object.defineProperty(meta, "output_token_details", {
        get() {
          throw new Error("boom: hostile getter");
        },
      });
      expect(() =>
        _captureLangchainResponseFromMessage(msg(meta), 0),
      ).not.toThrow();
      expect(slot(s, 0).reasoning_tokens).toBeUndefined();
      // The reasoning read is in its own try/catch — it must not take the
      // in/out usage stash down with it.
      expect(slot(s, 0).usage.input_tokens).toBe(100);
    });
  });
});

// ═══════════════════════════════════════════════════════════════════════
// C2 — _captureLangchainResponse generate path
// ═══════════════════════════════════════════════════════════════════════
describe("C2 generate-path reasoning stash", () => {
  it("reasoning stashed even when llmOutput.tokenUsage is present (hasWireUsage)", () => {
    session({ name: "wf" }, (s) => {
      // ChatOpenAI ALWAYS reports llmOutput.tokenUsage, so the pre-existing
      // usage-repair block never runs — the reasoning write must not be gated
      // behind it, or this whole fix is dead code on the primary path.
      _captureLangchainResponse(
        {
          llmOutput: {
            tokenUsage: {
              promptTokens: 100,
              completionTokens: 400,
              totalTokens: 500,
            },
          },
          generations: [[gen(um(100, 400, 357))]],
        },
        0,
      );
      expect(slot(s, 0).reasoning_tokens).toBe(357);
    });
  });

  it("the reasoning-only write creates NO .usage stash", () => {
    session({ name: "wf" }, (s) => {
      // A `.usage` stash here would flip telemetry's preferStash arm and the
      // stream-guard raw decision — in/out/cached must stay attr-derived.
      _captureLangchainResponse(
        {
          llmOutput: {
            tokenUsage: { promptTokens: 100, completionTokens: 400 },
          },
          generations: [[gen(um(100, 400, 357))]],
        },
        0,
      );
      expect(slot(s, 0).usage).toBeUndefined();
      expect(slot(s, 0).reasoning_tokens).toBe(357);
    });
  });

  it("Responses-shape repair path still stashes usage AND reasoning", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponse(
        {
          llmOutput: {
            estimatedTokenUsage: { promptTokens: 100, completionTokens: 400 },
          },
          generations: [[gen(um(100, 400, 357))]],
        },
        0,
      );
      expect(slot(s, 0).usage).toMatchObject({
        usage_source: "langchain_message",
        input_tokens: 100,
        output_tokens: 400,
      });
      expect(slot(s, 0).reasoning_tokens).toBe(357);
    });
  });

  it("n>1 candidates: identical per-candidate reasoning counted ONCE", () => {
    session({ name: "wf" }, (s) => {
      const meta = um(100, 400, 357);
      _captureLangchainResponse(
        { generations: [[gen(meta, "a"), gen(meta, "b")]] },
        0,
      );
      expect(slot(s, 0).reasoning_tokens).toBe(357);
      expect(slot(s, 0).reasoning_tokens).not.toBe(714);
    });
  });

  it("batch generate: reasoning SUMS across the outer generations lists", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponse(
        {
          generations: [
            [gen(um(10, 100, 10))],
            [gen(um(20, 200, 20))],
          ],
        },
        0,
      );
      expect(slot(s, 0).reasoning_tokens).toBe(30);
    });
  });

  it("first usage-bearing candidate per outer list is the one read", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponse(
        {
          generations: [
            [gen(undefined, "a"), gen(um(10, 100, 10), "b")],
            [gen(um(20, 200, 20), "c"), gen(um(20, 200, 20), "d")],
          ],
        },
        0,
      );
      expect(slot(s, 0).reasoning_tokens).toBe(30);
    });
  });

  it("non-array inner entry (bare Generation) is normalized, not dropped", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponse({ generations: [gen(um(10, 100, 5))] }, 0);
      expect(slot(s, 0).reasoning_tokens).toBe(5);
    });
  });

  it("MAX semantics: a later smaller capture does not clobber", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponse({ generations: [[gen(um(10, 100, 357))]] }, 0);
      _captureLangchainResponse({ generations: [[gen(um(10, 100, 4))]] }, 0);
      expect(slot(s, 0).reasoning_tokens).toBe(357);
    });
  });

  it("no reasoning → no reasoning_tokens key", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponse(
        {
          llmOutput: { tokenUsage: { promptTokens: 100, completionTokens: 400 } },
          generations: [[gen(um(100, 400))]],
        },
        0,
      );
      expect(slot(s, 0)?.reasoning_tokens).toBeUndefined();
    });
  });

  it("stash lands under the ORDER passed", () => {
    session({ name: "wf" }, (s) => {
      _captureLangchainResponse({ generations: [[gen(um(10, 100, 357))]] }, 7);
      expect(slot(s, 7).reasoning_tokens).toBe(357);
      expect(slot(s, 0)).toBeUndefined();
    });
  });

  const hostile: Array<[string, unknown]> = [
    ["null result", null],
    ["non-object result", 42],
    ["generations not an array", { generations: "nope" }],
    ["inner list of nulls", { generations: [[null, undefined]] }],
    ["generation without message", { generations: [[{ text: "hi" }]] }],
    [
      "usage_metadata is a string",
      { generations: [[{ message: { usage_metadata: "lots" } }]] },
    ],
    [
      "output_token_details is a string",
      {
        generations: [
          [{ message: { usage_metadata: { output_token_details: "x" } } }],
        ],
      },
    ],
    [
      "reasoning is NaN",
      {
        generations: [
          [
            {
              message: {
                usage_metadata: { output_token_details: { reasoning: NaN } },
              },
            },
          ],
        ],
      },
    ],
  ];
  for (const [name, result] of hostile) {
    it(`${name} → no throw, no reasoning stash`, () => {
      session({ name: "wf" }, (s) => {
        expect(() => _captureLangchainResponse(result, 0)).not.toThrow();
        expect(slot(s, 0)?.reasoning_tokens).toBeUndefined();
      });
    });
  }

  it("usage_metadata getter that THROWS mid-iteration → no throw, no stash", () => {
    session({ name: "wf" }, (s) => {
      const message: any = { content: "hi" };
      Object.defineProperty(message, "usage_metadata", {
        get() {
          throw new Error("boom: hostile getter");
        },
      });
      expect(() =>
        _captureLangchainResponse({ generations: [[{ message }]] }, 0),
      ).not.toThrow();
      expect(slot(s, 0)?.reasoning_tokens).toBeUndefined();
    });
  });
});

// ═══════════════════════════════════════════════════════════════════════
// C3 — telemetry emission
// ═══════════════════════════════════════════════════════════════════════
describe("C3 telemetry emission — openai_chat (subset semantics)", () => {
  it("zero attr + stash → completion_tokens_details.reasoning_tokens", async () => {
    const s = newSession();
    seedSlot(s, 0, { reasoning_tokens: 357 });
    const opts = (
      await runOnEnd(
        s,
        fakeSpan(s, 0, {
          system: "openai",
          model: "gpt-5",
          input: 100,
          output: 400,
          reasoning: 0,
        }),
      )
    ).at(-1);

    expect(opts.usage.shape).toBe("openai_chat");
    expect(opts.usage.raw.completion_tokens_details.reasoning_tokens).toBe(357);
    // SUBSET: total output must not move (mapper does completion − reasoning).
    expect(opts.usage.raw.completion_tokens).toBe(400);
    expect(opts.usage.raw.prompt_tokens).toBe(100);
    // openai shape must NOT get the google split keys.
    expect(opts.usage.raw.thoughts_token_count).toBeUndefined();
    expect(opts.usage.raw.candidates_token_count).toBeUndefined();
  });

  it("missing reasoning attr entirely + stash → still emitted", async () => {
    const s = newSession();
    seedSlot(s, 0, { reasoning_tokens: 357 });
    const opts = (
      await runOnEnd(s, fakeSpan(s, 0, { input: 100, output: 400 }))
    ).at(-1);
    expect(opts.usage.raw.completion_tokens_details.reasoning_tokens).toBe(357);
  });

  it("stash clamped to outputTokens", async () => {
    const s = newSession();
    seedSlot(s, 0, { reasoning_tokens: 5000 });
    const opts = (
      await runOnEnd(s, fakeSpan(s, 0, { input: 100, output: 400 }))
    ).at(-1);
    expect(opts.usage.raw.completion_tokens_details.reasoning_tokens).toBe(400);
    expect(
      opts.usage.raw.completion_tokens -
        opts.usage.raw.completion_tokens_details.reasoning_tokens,
    ).toBeGreaterThanOrEqual(0);
  });

  it("no stash → byte-identical to today (no details key)", async () => {
    const s = newSession();
    const opts = (
      await runOnEnd(s, fakeSpan(s, 0, { input: 100, output: 400 }))
    ).at(-1);
    expect(opts.usage.raw.completion_tokens_details).toBeUndefined();
  });

  it("zero stash → no details key", async () => {
    const s = newSession();
    seedSlot(s, 0, { reasoning_tokens: 0 });
    const opts = (
      await runOnEnd(s, fakeSpan(s, 0, { input: 100, output: 400 }))
    ).at(-1);
    expect(opts.usage.raw.completion_tokens_details).toBeUndefined();
  });

  it("negative/garbage stash never emits a negative count", async () => {
    for (const bad of [-5, "abc", null, {}]) {
      logged = [];
      const s = newSession();
      seedSlot(s, 0, { reasoning_tokens: bad as any });
      const opts = (
        await runOnEnd(s, fakeSpan(s, 0, { input: 100, output: 400 }))
      ).at(-1);
      expect(opts.usage.raw.completion_tokens_details).toBeUndefined();
    }
  });

  it("attr-sourced reasoning still wins over the stash (attr path unchanged)", async () => {
    const s = newSession();
    seedSlot(s, 0, { reasoning_tokens: 357 });
    const opts = (
      await runOnEnd(
        s,
        fakeSpan(s, 0, { input: 100, output: 400, reasoning: 42 }),
      )
    ).at(-1);
    expect(opts.usage.raw.completion_tokens_details.reasoning_tokens).toBe(42);
  });
});

describe("C3 telemetry emission — google_genai (additive semantics)", () => {
  it("stash → thoughts + netted candidates, total output unchanged", async () => {
    const s = newSession();
    seedSlot(s, 0, { reasoning_tokens: 120 });
    const opts = (
      await runOnEnd(
        s,
        fakeSpan(s, 0, {
          system: "google",
          model: "gemini-2.5-pro",
          input: 1103,
          output: 500,
        }),
      )
    ).at(-1);

    expect(opts.usage.shape).toBe("google_genai");
    expect(opts.usage.raw.thoughts_token_count).toBe(120);
    // LC's output_tokens is thoughts-INCLUSIVE; the google mapper's candidates
    // is EXCLUSIVE — so candidates must be netted or total output inflates.
    expect(opts.usage.raw.candidates_token_count).toBe(380);
    expect(
      opts.usage.raw.candidates_token_count + opts.usage.raw.thoughts_token_count,
    ).toBe(500);
    // Positional counts stay as-is.
    expect(opts.usage.raw.completion_tokens).toBe(500);
    expect(opts.usage.raw.prompt_tokens).toBe(1103);
  });

  it("clamped stash → candidates floors at 0, never negative", async () => {
    const s = newSession();
    seedSlot(s, 0, { reasoning_tokens: 9000 });
    const opts = (
      await runOnEnd(
        s,
        fakeSpan(s, 0, { system: "google", input: 10, output: 100 }),
      )
    ).at(-1);
    expect(opts.usage.raw.thoughts_token_count).toBe(100);
    expect(opts.usage.raw.candidates_token_count).toBe(0);
  });

  it("attr-sourced reasoning does NOT trigger the split (bit-identical)", async () => {
    const s = newSession();
    const opts = (
      await runOnEnd(
        s,
        fakeSpan(s, 0, {
          system: "google",
          input: 1103,
          output: 500,
          reasoning: 120,
        }),
      )
    ).at(-1);
    expect(opts.usage.raw.thoughts_token_count).toBeUndefined();
    expect(opts.usage.raw.candidates_token_count).toBeUndefined();
    expect(opts.usage.raw.completion_tokens_details.reasoning_tokens).toBe(120);
  });

  it("no stash → no google split keys", async () => {
    const s = newSession();
    const opts = (
      await runOnEnd(
        s,
        fakeSpan(s, 0, { system: "google", input: 1103, output: 500 }),
      )
    ).at(-1);
    expect(opts.usage.raw.thoughts_token_count).toBeUndefined();
    expect(opts.usage.raw.candidates_token_count).toBeUndefined();
  });
});

describe("C3 telemetry emission — other shapes", () => {
  it("anthropic_messages gets no google split and no new keys", async () => {
    const s = newSession();
    seedSlot(s, 0, { reasoning_tokens: 120 });
    const opts = (
      await runOnEnd(
        s,
        fakeSpan(s, 0, {
          system: "anthropic",
          model: "claude-sonnet-4",
          input: 100,
          output: 400,
        }),
      )
    ).at(-1);
    expect(opts.usage.shape).toBe("anthropic_messages");
    expect(opts.usage.raw.thoughts_token_count).toBeUndefined();
    expect(opts.usage.raw.candidates_token_count).toBeUndefined();
    // The generic details key is shape-agnostic in the constructed raw; the
    // anthropic mapper simply ignores it. Total output must be untouched.
    expect(opts.usage.raw.output_tokens).toBe(400);
  });

  it("the LC usage stash still drives in/out (reasoning write is additive only)", async () => {
    const s = newSession();
    seedSlot(s, 0, {
      usage: {
        usage_source: "langchain_message",
        input_tokens: 130,
        output_tokens: 8,
        cached_tokens: 0,
        cache_creation_tokens: 0,
      },
      reasoning_tokens: 3,
    });
    const opts = (
      await runOnEnd(
        s,
        fakeSpan(s, 0, { system: "google", input: 0, output: 7 }),
      )
    ).at(-1);
    // Pre-existing preferStash behavior is unchanged...
    expect(opts.usage.raw.input_tokens).toBe(130);
    expect(opts.usage.raw.output_tokens).toBe(8);
    // ...and the reasoning rides on top without moving the totals.
    expect(opts.usage.raw.thoughts_token_count).toBe(3);
    expect(opts.usage.raw.candidates_token_count).toBe(5);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// End-to-end: enforcer stash → telemetry log
// ═══════════════════════════════════════════════════════════════════════
describe("end-to-end LC reasoning (generate path → log)", () => {
  it("openai: stashed reasoning reaches the logged raw", async () => {
    const s = newSession();
    await _getSessionStorage().run(s, async () => {
      _captureLangchainResponse(
        {
          llmOutput: {
            tokenUsage: { promptTokens: 100, completionTokens: 400 },
          },
          generations: [[gen(um(100, 400, 357))]],
        },
        0,
      );
      new TokenPoliceSpanProcessor().onEnd(
        fakeSpan(s, 0, { system: "openai", input: 100, output: 400 }),
      );
      await new Promise((r) => setTimeout(r, 0));
    });

    expect(logged.length).toBe(1);
    const opts = logged[0].at(-1);
    expect(opts.usage.raw.completion_tokens_details.reasoning_tokens).toBe(357);
    expect(opts.usage.raw.completion_tokens).toBe(400);
  });

  it("gemini: stashed reasoning splits thoughts/candidates in the logged raw", async () => {
    const s = newSession();
    await _getSessionStorage().run(s, async () => {
      _captureLangchainResponseFromMessage(msg(um(1103, 500, 120)), 0, {
        cached_tokens: 0,
        cache_creation_tokens: 0,
      });
      new TokenPoliceSpanProcessor().onEnd(
        fakeSpan(s, 0, {
          system: "google",
          model: "gemini-2.5-pro",
          input: 1103,
          output: 500,
        }),
      );
      await new Promise((r) => setTimeout(r, 0));
    });

    expect(logged.length).toBe(1);
    const opts = logged[0].at(-1);
    expect(opts.usage.raw.thoughts_token_count).toBe(120);
    expect(opts.usage.raw.candidates_token_count).toBe(380);
  });
});
