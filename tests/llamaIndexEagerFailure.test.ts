/**
 * L-2 / F-G6b-2 (Node LlamaIndex): an EAGER provider rejection must land ONE
 * failure row — pre-fix it landed ZERO.
 *
 * `_setLlamaIndexWrapper` awaits `original.apply(this, args)` itself. Two real
 * adapter shapes reject AT that await rather than at a later pull:
 *
 *   - `@llamaindex/anthropic` — `chat()` (non-stream) issues the HTTP request
 *     inside the awaited call;
 *   - `@llamaindex/google@0.4.x` — its streaming path is a plain `async`
 *     method that awaits `sendMessageStream` BEFORE returning an iterator, so
 *     a 401/400/429 at connect never reaches an iterator body.
 *
 * Every raw provider wrapper is deliberately inert while `inLlamaIndex` is set
 * (`if (session.inLlamaIndex) return await original.apply(...)`), so this
 * wrapper is the SOLE emitter on the LlamaIndex path. Its catch used to be
 * just `session.exitLlamaIndex(); throw e;` — the customer's failed call
 * produced no `llm` row at all: invisible spend attempts, invisible auth/quota
 * breakage, and a trace with a hole where the model call should be.
 *
 * Fix under test: that catch now runs an inner
 * `try { … } catch {} finally { session.exitLlamaIndex(); }` before
 * `throw e` — stamping `buildCallOutcome(e, elapsed)`, running the
 * conditional-restore protocol over the `_attempted_*` slots around
 * `_emitCallFailureLog`, and rethrowing the ORIGINAL error by identity. The
 * lazy path (`_guardedLlamaIndexStream`'s catch, covered by
 * llamaIndexStreamFailure.test.ts) is unchanged and mutually exclusive with
 * this one: if the eager catch fired, no iterable was ever returned, so the two
 * can never both emit.
 *
 * All offline against fake LlamaIndex provider classes — no network, no
 * LlamaIndex. Harness mirrors tests/llamaIndexStreamFailure.test.ts (same
 * class+prototype patch, same pinned session, same captured `tp.log` args).
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { setClient } from "../src/state";
import { TPSession, _getSessionStorage } from "../src/context";
import { TokenPoliceBlockedError } from "../src/exceptions";
import { __test__ as enforcerTest } from "../src/enforcer";

const MODEL_ARG = 4;
const PROVIDER_ARG = 5;
const INPUT_TOKENS_ARG = 6;
const OUTPUT_TOKENS_ARG = 7;
const CACHED_TOKENS_ARG = 8;

let logged: any[][];

beforeEach(() => {
  logged = [];
  // firewall:"off" short-circuits the async pre-flight; log captures the row.
  setClient({ firewall: "off", log: (...args: any[]) => logged.push(args) } as any);
});

afterEach(() => {
  setClient(null as any);
  // Undo every prototype patch this file installed.
  for (const thunk of enforcerTest._restoreThunks.splice(0)) {
    try {
      thunk();
    } catch {
      /* ignore */
    }
  }
});

/** `tp.log`'s trailing options arg — where `call_outcome` lives. */
function extrasOf(logArgs: any[]): any {
  return logArgs[logArgs.length - 1];
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

/**
 * Build a LlamaIndex-provider-shaped class. `_llamaIndexProvider` keys off
 * `constructor.name` and the instrumentor patches `prototype.chat`, so both
 * must be real.
 */
function defineProviderClass(
  name: string,
  modelName: string,
  chat: (this: any, params: any) => any,
): any {
  const Cls = {
    [name]: class {
      model = modelName;
    },
  }[name];
  (Cls.prototype as any).chat = chat;
  return Cls;
}

/** Install the chat patch on a one-off class and return it. */
function instrument(name: string, model: string, chat: (this: any, p: any) => any): any {
  const mod: any = { [name]: defineProviderClass(name, model, chat) };
  enforcerTest._instrumentLlamaIndexProvider(mod, [name]);
  return mod[name];
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * EAGER adapter body: awaits (so `duration_ms` is real) and then rejects —
 * the rejection surfaces at the WRAPPER's own `await original.apply()`, which
 * is the seam under test. Used for both the non-streaming (@llamaindex/
 * anthropic) and the "plain async streamChat" (@llamaindex/google 0.4.x)
 * shapes; the only difference between them is the caller's `stream` flag,
 * since neither ever hands back an iterator.
 */
function eagerReject(err: unknown, onCall?: () => void): () => Promise<never> {
  return async () => {
    await sleep(2);
    onCall?.();
    throw err;
  };
}

/**
 * LAZY adapter body (the PR #382 shape, kept here only as the mutual-exclusion
 * control): `original.apply()` resolves to an async generator, so the wrapper's
 * eager catch never sees the rejection — `_guardedLlamaIndexStream`'s catch does.
 */
function lazyReject(err: unknown): () => Promise<AsyncIterable<any>> {
  return async () =>
    (async function* () {
      await sleep(2);
      throw err;
    })();
}

/** OpenAI-SDK-shaped 401 (`AuthenticationError` + `status`). */
function auth401(): any {
  class AuthenticationError extends Error {}
  return Object.assign(new AuthenticationError("401 Incorrect API key provided"), {
    status: 401,
  });
}

/** Drive one chat inside a pinned session; capture the error the customer sees. */
async function driveChat(
  session: TPSession,
  Cls: any,
  params: any = { messages: [] },
): Promise<{ caught: unknown; seen: any[] }> {
  const seen: any[] = [];
  let caught: unknown;
  await _getSessionStorage().run(session, async () => {
    try {
      const result = await new Cls().chat(params);
      // An eager rejection never gets here; a lazy one fails during the drain.
      if (result != null && typeof result[Symbol.asyncIterator] === "function") {
        for await (const c of result) seen.push(c);
      }
    } catch (e) {
      caught = e;
    }
  });
  return { caught, seen };
}

const failedRows = (): any[][] =>
  logged.filter((r) => extrasOf(r)?.call_outcome?.status === "failed");

const ANTHROPIC_OK = {
  raw: {
    model: "claude-sonnet-4-5",
    usage: { input_tokens: 100, output_tokens: 20, cache_read_input_tokens: 30 },
  },
  message: { role: "assistant", content: "hi" },
};
const RESP_USAGE = {
  input_tokens: 100,
  output_tokens: 20,
  total_tokens: 120,
  input_tokens_details: { cached_tokens: 30 },
};

// ── 1. Eager NON-STREAM rejection (@llamaindex/anthropic shape) ───

describe("eager LlamaIndex failure — non-streaming chat() rejects at the wrapper's await", () => {
  it("emits exactly one row, classified failed, carrying the stashed model/provider", async () => {
    const boom = auth401();
    const Cls = instrument("Anthropic", "claude-sonnet-4-5", eagerReject(boom));
    const session = newSession();

    const { caught } = await driveChat(session, Cls);

    expect(caught).toBe(boom);
    // Pre-fix this was 0 — the catch only released the guard and rethrew.
    expect(logged).toHaveLength(1);

    const row = logged[0];
    const outcome = extrasOf(row).call_outcome;
    expect(outcome.status).toBe("failed");
    expect(outcome.error_kind).toBe("auth_error");
    expect(outcome.http_status).toBe(401);
    expect(typeof outcome.duration_ms).toBe("number");

    // Attribution comes from the wrapper's _stashAttemptContext, not "unknown".
    expect(row[MODEL_ARG]).toBe("claude-sonnet-4-5");
    expect(row[PROVIDER_ARG]).toBe("anthropic");
    // A failed call has no usage — the row must stay unmeasured.
    expect(row[INPUT_TOKENS_ARG]).toBe(0);
    expect(row[OUTPUT_TOKENS_ARG]).toBe(0);
    expect(row[CACHED_TOKENS_ARG]).toBe(0);
    // Same usage_shape a SUCCESSFUL @llamaindex/anthropic row stamps.
    expect(extrasOf(row).usage).toEqual({
      shape: "anthropic_messages",
      raw: { prompt_tokens: 0, total_tokens: 0 },
    });
  });

  it("no success row is emitted alongside it (the success emit is skipped by the throw)", async () => {
    const boom = auth401();
    const Cls = instrument("Anthropic", "claude-sonnet-4-5", eagerReject(boom));
    const session = newSession();

    await driveChat(session, Cls);

    expect(logged).toHaveLength(1);
    expect(
      logged.filter((r) => extrasOf(r).call_outcome?.status === "success"),
    ).toHaveLength(0);
  });

  it("a 500 classifies as server_error (classification is not hard-coded)", async () => {
    const boom = Object.assign(new Error("500 upstream exploded"), { status: 500 });
    const Cls = instrument("Anthropic", "claude-sonnet-4-5", eagerReject(boom));
    const session = newSession();

    const { caught } = await driveChat(session, Cls);

    expect(caught).toBe(boom);
    expect(logged).toHaveLength(1);
    expect(extrasOf(logged[0]).call_outcome.error_kind).toBe("server_error");
    expect(extrasOf(logged[0]).call_outcome.http_status).toBe(500);
  });
});

// ── 2. Eager STREAM rejection (@llamaindex/google 0.4.x shape) ────

describe("eager LlamaIndex failure — a plain async streaming method rejects before any iterator", () => {
  it("emits exactly one failed row even though the caller asked for a stream", async () => {
    const boom = Object.assign(new Error("429 quota exceeded"), { status: 429 });
    const Cls = instrument("Gemini", "gemini-2.0-flash", eagerReject(boom));
    const session = newSession();

    const { caught, seen } = await driveChat(session, Cls, { messages: [], stream: true });

    expect(caught).toBe(boom);
    expect(seen).toEqual([]); // no iterator ever reached the customer
    // Pre-fix: 0 rows. The stream catch cannot cover this shape — nothing
    // iterable was ever produced for it to wrap.
    expect(logged).toHaveLength(1);
    const row = logged[0];
    expect(extrasOf(row).call_outcome.status).toBe("failed");
    expect(extrasOf(row).call_outcome.error_kind).toBe("rate_limited");
    expect(extrasOf(row).call_outcome.http_status).toBe(429);
    expect(row[PROVIDER_ARG]).toBe("google");
    expect(row[MODEL_ARG]).toBe("gemini-2.0-flash");
    expect(row[INPUT_TOKENS_ARG]).toBe(0);
    expect(row[OUTPUT_TOKENS_ARG]).toBe(0);
    expect(extrasOf(row).usage.shape).toBe("google_genai");
  });

  it("the eager and lazy catches are mutually exclusive — each shape lands exactly ONE row", async () => {
    const eagerBoom = auth401();
    const lazyBoom = auth401();
    const Eager = instrument("Gemini", "gemini-2.0-flash", eagerReject(eagerBoom));
    // The @llamaindex/openai shape: streamChat IS an async generator, so the
    // rejection first surfaces at the first pull (PR #382's catch).
    const Lazy = instrument("OpenAI", "gpt-4o", lazyReject(lazyBoom));

    const eagerSession = newSession();
    const { caught: eagerCaught } = await driveChat(eagerSession, Eager, {
      messages: [],
      stream: true,
    });
    expect(eagerCaught).toBe(eagerBoom);
    expect(logged).toHaveLength(1); // eager catch only — the stream catch never ran

    const lazySession = newSession();
    const { caught: lazyCaught } = await driveChat(lazySession, Lazy, {
      messages: [],
      stream: true,
    });
    expect(lazyCaught).toBe(lazyBoom);
    // One more, from the stream catch — never two for one call.
    expect(logged).toHaveLength(2);
    expect(failedRows()).toHaveLength(2);
    expect(logged[1][PROVIDER_ARG]).toBe("openai");
  });
});

// ── 3. Error identity (golden rule) ──────────────────────────────

describe("golden rule — the SDK never substitutes the customer's error", () => {
  it("a non-streaming eager rejection reaches the caller by identity", async () => {
    const boom = auth401();
    const Cls = instrument("Anthropic", "claude-sonnet-4-5", eagerReject(boom));
    const session = newSession();

    const { caught } = await driveChat(session, Cls);

    expect(caught).toBe(boom); // identity, not just instanceof
    expect(caught).not.toBeInstanceOf(TokenPoliceBlockedError);
    expect((caught as any).status).toBe(401);
    expect((caught as Error).message).toBe("401 Incorrect API key provided");
  });

  it("a streaming eager rejection reaches the caller by identity", async () => {
    const boom = auth401();
    const Cls = instrument("Gemini", "gemini-2.0-flash", eagerReject(boom));
    const session = newSession();

    const { caught } = await driveChat(session, Cls, { messages: [], stream: true });

    expect(caught).toBe(boom);
    expect(caught).not.toBeInstanceOf(TokenPoliceBlockedError);
  });

  it("a non-Error thrown value also propagates by identity and still logs", async () => {
    const boom = { code: "ECONNRESET" } as any;
    const Cls = instrument("Anthropic", "claude-sonnet-4-5", eagerReject(boom));
    const session = newSession();

    const { caught } = await driveChat(session, Cls);

    expect(caught).toBe(boom);
    expect(logged).toHaveLength(1);
    expect(extrasOf(logged[0]).call_outcome.status).toBe("failed");
  });
});

// ── 4. Guard hygiene after an eager failure ──────────────────────

describe("guard hygiene — an eager failure leaves the session clean", () => {
  it("inLlamaIndex is released and the attempt slots / outcome were consumed", async () => {
    const boom = auth401();
    const Cls = instrument("Anthropic", "claude-sonnet-4-5", eagerReject(boom));
    const session = newSession();

    await driveChat(session, Cls);

    expect(session.inLlamaIndex).toBe(false);
    expect((session as any)._call_outcome).toBeFalsy();
    expect((session as any)._attempted_model).toBeUndefined();
    expect((session as any)._attempted_provider).toBeUndefined();
    expect((session as any)._attempted_operation).toBeUndefined();
    expect((session as any)._attempted_shape).toBeUndefined();
    expect((session as any)._attempted_wire_key).toBeUndefined();
  });

  it("a subsequent successful call on the SAME session logs normally", async () => {
    const boom = auth401();
    const Failing = instrument("Anthropic", "claude-sonnet-4-5", eagerReject(boom));
    const session = newSession();

    await driveChat(session, Failing);
    expect(logged).toHaveLength(1);
    expect(extrasOf(logged[0]).call_outcome.status).toBe("failed");

    // Second call, same session — must not be short-circuited by a leaked
    // inLlamaIndex guard (which would make the wrapper a pass-through and
    // silently drop the row) and must not inherit the failed outcome.
    const Ok = instrument("Gemini", "gemini-2.0-flash", async () => {
      await sleep(2);
      return {
        raw: {
          modelVersion: "gemini-2.0-flash",
          usageMetadata: { promptTokenCount: 100, candidatesTokenCount: 20 },
        },
        message: { role: "assistant", content: "hi" },
      };
    });
    await _getSessionStorage().run(session, async () => {
      await new Ok().chat({ messages: [] });
    });

    expect(logged).toHaveLength(2);
    const row = logged[1];
    expect(row[PROVIDER_ARG]).toBe("google");
    expect(row[MODEL_ARG]).toBe("gemini-2.0-flash");
    expect(row[INPUT_TOKENS_ARG]).toBe(100);
    expect(extrasOf(row).call_outcome?.status).not.toBe("failed");
    expect(session.inLlamaIndex).toBe(false);
  });

  it("the inner provider wrapper is inert during the call — proving this wrapper is the only emitter", async () => {
    // The nested-call short-circuit: while inLlamaIndex is set the wrapper
    // delegates straight through, so a raw provider wrapper firing inside the
    // adapter contributes NO row. This is exactly why a missing eager emit
    // meant zero rows, not a degraded one.
    const boom = auth401();
    const session = newSession();
    let depthDuringCall: boolean | undefined;
    const Cls = instrument(
      "Anthropic",
      "claude-sonnet-4-5",
      eagerReject(boom, () => {
        depthDuringCall = session.inLlamaIndex;
      }),
    );

    await driveChat(session, Cls);

    expect(depthDuringCall).toBe(true); // guard held while the provider ran
    expect(session.inLlamaIndex).toBe(false); // released by the catch's finally
    expect(logged).toHaveLength(1);
  });
});

// ── 5. Success path — behavior UNCHANGED by the fix (control) ────

describe("eager path success — unchanged by the fix (control)", () => {
  it("a successful non-streaming chat still emits exactly one success row", async () => {
    const Cls = instrument("Anthropic", "claude-sonnet-4-5", async () => {
      await sleep(2);
      return ANTHROPIC_OK;
    });
    const session = newSession();

    const { caught } = await driveChat(session, Cls);

    expect(caught).toBeUndefined();
    expect(logged).toHaveLength(1);
    const row = logged[0];
    expect(row[MODEL_ARG]).toBe("claude-sonnet-4-5");
    expect(row[PROVIDER_ARG]).toBe("anthropic");
    expect(row[INPUT_TOKENS_ARG]).toBe(100); // anthropic input_tokens is cache-EXCLUSIVE
    expect(row[OUTPUT_TOKENS_ARG]).toBe(20);
    expect(row[CACHED_TOKENS_ARG]).toBe(30);
    // No failure residue: no error classification anywhere on the row.
    expect(extrasOf(row).call_outcome?.status).not.toBe("failed");
    expect(extrasOf(row).call_outcome?.error_kind).toBeUndefined();
    expect(failedRows()).toHaveLength(0);
    // …and none left on the session either.
    expect((session as any)._call_outcome).toBeFalsy();
    expect(session.inLlamaIndex).toBe(false);
  });
});

// ── 6. Conditional-restore protocol ──────────────────────────────

describe("conditional restore — a concurrent same-session call keeps its attempt context", () => {
  it("the row carries THIS call's stash while the concurrent call's slots are handed back", async () => {
    const boom = auth401();
    const session = newSession();
    // Mid-flight, a genuinely concurrent same-session call stashes ITS attempt
    // context over the session-global _attempted_* slots. The snapshot taken
    // before `await original.apply()` is what must reach the emitted row.
    const Cls = instrument(
      "Anthropic",
      "claude-sonnet-4-5",
      eagerReject(boom, () => {
        (session as any)._attempted_model = "gpt-4o-concurrent";
        (session as any)._attempted_provider = "openai";
        (session as any)._attempted_operation = "embedding";
        (session as any)._attempted_shape = "openai_embeddings";
        (session as any)._attempted_wire_key = "openai";
      }),
    );

    const { caught } = await driveChat(session, Cls);

    expect(caught).toBe(boom);
    expect(logged).toHaveLength(1);
    // The failed row is THIS call's, not the concurrent one's.
    expect(logged[0][MODEL_ARG]).toBe("claude-sonnet-4-5");
    expect(logged[0][PROVIDER_ARG]).toBe("anthropic");
    expect(extrasOf(logged[0]).operation).toBe("chat");
    expect(extrasOf(logged[0]).usage.shape).toBe("anthropic_messages");

    // …and the concurrent call's context survives the emit (which otherwise
    // clears every slot), so ITS eventual failure row is not degraded.
    expect((session as any)._attempted_model).toBe("gpt-4o-concurrent");
    expect((session as any)._attempted_provider).toBe("openai");
    expect((session as any)._attempted_operation).toBe("embedding");
    expect((session as any)._attempted_shape).toBe("openai_embeddings");
    expect((session as any)._attempted_wire_key).toBe("openai");
  });

  it("with no concurrent writer the slots are simply left consumed (control)", async () => {
    const boom = auth401();
    const Cls = instrument("Anthropic", "claude-sonnet-4-5", eagerReject(boom));
    const session = newSession();

    await driveChat(session, Cls);

    expect(logged).toHaveLength(1);
    expect((session as any)._attempted_model).toBeUndefined();
    expect((session as any)._attempted_provider).toBeUndefined();
  });
});

// ── 7. OpenAIResponses pseudo-provider parity ────────────────────

describe("OpenAIResponses — an eager failure carries the same identity as its successful siblings", () => {
  it("failed non-streamed Responses call → provider + usage_shape 'openai_responses'", async () => {
    const boom = auth401();
    const Cls = instrument("OpenAIResponses", "gpt-5", eagerReject(boom));
    const session = newSession();

    const { caught } = await driveChat(session, Cls);

    expect(caught).toBe(boom);
    expect(logged).toHaveLength(1);
    const row = logged[0];
    expect(row[PROVIDER_ARG]).toBe("openai_responses");
    expect(row[MODEL_ARG]).toBe("gpt-5");
    expect(extrasOf(row).call_outcome.status).toBe("failed");
    expect(extrasOf(row).usage.shape).toBe("openai_responses");
  });

  it("the successful sibling stamps that same shape (parity control)", async () => {
    const Cls = instrument("OpenAIResponses", "gpt-5", async () => {
      await sleep(2);
      return {
        raw: { model: "gpt-5-2025-08-07", usage: RESP_USAGE },
        message: { role: "assistant", content: "hi" },
      };
    });
    const session = newSession();

    await driveChat(session, Cls);

    expect(logged).toHaveLength(1);
    expect(logged[0][PROVIDER_ARG]).toBe("openai_responses");
    expect(extrasOf(logged[0]).usage.shape).toBe("openai_responses");
    expect(extrasOf(logged[0]).call_outcome?.status).not.toBe("failed");
  });
});

// ── 8. Enforce block — NOT a provider failure ────────────────────

describe("enforce block — a denial is not a failed provider call", () => {
  it("TokenPoliceBlockedError propagates, no failure row, and the guard was never entered", async () => {
    // No `deployment` → _localEvaluate returns null, so the State-B inline
    // /check verdict drives the block (same setup as blockedErrorFields.test.ts).
    setClient({
      firewall: "enforce",
      check: async () => ({
        status: "blocked",
        reason: "budget exceeded",
        ruleId: "rule_budget_1",
        traceId: "tr_budget",
      }),
      log: (...args: any[]) => logged.push(args),
    } as any);

    let providerCalled = false;
    const Cls = instrument("Anthropic", "claude-sonnet-4-5", async () => {
      providerCalled = true;
      return ANTHROPIC_OK;
    });
    const session = newSession();

    const { caught } = await driveChat(session, Cls);

    expect(caught).toBeInstanceOf(TokenPoliceBlockedError);
    expect((caught as TokenPoliceBlockedError).ruleId).toBe("rule_budget_1");
    // The pre-flight blocks BEFORE the provider runs and before enterLlamaIndex.
    expect(providerCalled).toBe(false);
    expect(session.inLlamaIndex).toBe(false);
    // The block-decision row is the only dispatch; a block is not a call failure.
    expect(failedRows()).toHaveLength(0);
    expect(logged).toHaveLength(1);
    expect(logged[0][MODEL_ARG]).toBe("blocked");
    expect(extrasOf(logged[0]).local_decision?.outcome).toBe("blocked");
    expect(extrasOf(logged[0]).call_outcome).toBeUndefined();
  });
});

// ── 9. Fail-open hygiene inside the new emit block ───────────────

describe("fail-open — the added emit can never affect the customer's call", () => {
  it("tp.log exploding inside the failure emit still surfaces only the provider error", async () => {
    setClient({
      firewall: "off",
      log: () => {
        throw new Error("log exploded");
      },
    } as any);
    const boom = auth401();
    const Cls = instrument("Anthropic", "claude-sonnet-4-5", eagerReject(boom));
    const session = newSession();

    const { caught } = await driveChat(session, Cls);

    expect(caught).toBe(boom);
    expect(session.inLlamaIndex).toBe(false); // guard still released
  });

  it("a hostile _call_outcome setter aborts the emit but never the rethrow or the guard release", async () => {
    const boom = auth401();
    const Cls = instrument("Anthropic", "claude-sonnet-4-5", eagerReject(boom));
    const session = newSession();
    // Throws on the FIRST statement of the new emit block, so the inner catch
    // and its finally are what keep the call well-behaved.
    Object.defineProperty(session, "_call_outcome", {
      configurable: true,
      get: () => null,
      set: () => {
        throw new Error("hostile setter");
      },
    });

    const { caught } = await driveChat(session, Cls);

    expect(caught).toBe(boom);
    expect(session.inLlamaIndex).toBe(false); // finally still ran
    expect(logged).toHaveLength(0); // emit aborted, but silently
  });
});
