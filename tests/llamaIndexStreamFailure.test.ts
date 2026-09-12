/**
 * L-1 (Node LlamaIndex): a streamed provider rejection must land ONE failure
 * row — never a synthetic success row.
 *
 * `_guardedLlamaIndexStream` had a `finally` and NO `catch`. `streamChat` is an
 * async generator, so `await original.apply()` resolves BEFORE any HTTP is
 * issued — the wrapper's eager catch never sees a provider rejection. When the
 * rejection surfaced at (or after) the first pull, the finally still ran the
 * synthetic-success emit and the call landed as:
 *
 *   - zero-pull 401  → one `llm` row, status='success', no error classification;
 *   - mid-stream 429 after a usage-bearing chunk (Gemini/Responses carry usage
 *     on the terminal event) → a **priced** success row for a call that failed.
 *
 * Fix under test: the wrapper stashes the attempt context
 * (`_stashAttemptContext`), snapshots the `_attempted_*` slots and captures its
 * obs key, and hands both to the generator; the generator gained a `catch` that
 * stamps `buildCallOutcome(err, elapsed)`, restores the snapshot around
 * `_emitCallFailureLog` (re-entering the captured obs scope, since the pull
 * resumes in the CONSUMER's async context), and rethrows the ORIGINAL error by
 * identity. The finally's success emit is now gated on `!failed`; a consumer
 * early `break` skips the catch entirely and keeps today's partial success row.
 *
 * All offline against fake LlamaIndex provider classes — no network, no
 * LlamaIndex. Harness mirrors llamaIndexResponsesWiring.test.ts (class +
 * prototype patch) and langchainFailureEmission.test.ts (pinned session,
 * captured `tp.log` args).
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { setClient } from "../src/state";
import { TPSession, _getSessionStorage } from "../src/context";
import { __test__ as enforcerTest } from "../src/enforcer";

const MODEL_ARG = 4;
const PROVIDER_ARG = 5;
const INPUT_TOKENS_ARG = 6;
const OUTPUT_TOKENS_ARG = 7;
const CACHED_TOKENS_ARG = 8;
const EXTRAS_ARG = 13;

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

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Async-iterable stand-in for a streamChat result that yields `chunks` (with a
 * real 2ms gap so latency/duration are non-zero and the success-row assertions
 * are deterministic) and then throws `err`. `err === null` → clean end.
 */
function streamThatThrows(chunks: any[], err: unknown): AsyncIterable<any> {
  return (async function* () {
    for (const c of chunks) {
      await sleep(2);
      yield c;
    }
    if (err !== null) throw err;
  })();
}

/** OpenAI-SDK-shaped 401 (`AuthenticationError` + `status`). */
function auth401(): any {
  class AuthenticationError extends Error {}
  return Object.assign(new AuthenticationError("401 Incorrect API key provided"), {
    status: 401,
  });
}

/** Install the chat patch on a one-off class and return it. */
function instrument(name: string, model: string, chat: (this: any, p: any) => any): any {
  const mod: any = { [name]: defineProviderClass(name, model, chat) };
  enforcerTest._instrumentLlamaIndexProvider(mod, [name]);
  return mod[name];
}

/** Drive one streamed chat inside a pinned session; capture chunks + error. */
async function driveStream(
  session: TPSession,
  Cls: any,
  params: any = { messages: [], stream: true },
): Promise<{ caught: unknown; seen: any[] }> {
  const seen: any[] = [];
  let caught: unknown;
  await _getSessionStorage().run(session, async () => {
    try {
      const result = await new Cls().chat(params);
      for await (const c of result) seen.push(c);
    } catch (e) {
      caught = e;
    }
  });
  return { caught, seen };
}

const OPENAI_USAGE = {
  prompt_tokens: 100,
  completion_tokens: 20,
  prompt_tokens_details: { cached_tokens: 30 },
};
const RESP_USAGE = {
  input_tokens: 100,
  output_tokens: 20,
  total_tokens: 120,
  input_tokens_details: { cached_tokens: 30 },
};

// ── 1. Rejection at the FIRST pull ───────────────────────────────

describe("guarded LlamaIndex stream — rejection at the first pull", () => {
  it("exactly one row, classified failed, carrying the stashed model/provider", async () => {
    const boom = auth401();
    const Cls = instrument("OpenAI", "gpt-4o", async () => streamThatThrows([], boom));
    const session = newSession();

    const { caught, seen } = await driveStream(session, Cls);

    expect(caught).toBe(boom);
    expect(seen).toEqual([]);
    expect(logged).toHaveLength(1);

    const row = logged[0];
    // Pre-fix this row existed too — but as a success row with model from the
    // instance and no outcome. The classification is the whole fix.
    const outcome = extrasOf(row).call_outcome;
    expect(outcome.status).toBe("failed");
    expect(outcome.error_kind).toBe("auth_error");
    expect(outcome.http_status).toBe(401);
    expect(typeof outcome.duration_ms).toBe("number");

    // Attribution comes from the wrapper's _stashAttemptContext, not "unknown".
    expect(row[MODEL_ARG]).toBe("gpt-4o");
    expect(row[PROVIDER_ARG]).toBe("openai");
    // A failed call has no usage — the row must stay unmeasured.
    expect(row[INPUT_TOKENS_ARG]).toBe(0);
    expect(row[OUTPUT_TOKENS_ARG]).toBe(0);
    expect(row[CACHED_TOKENS_ARG]).toBe(0);
    expect(extrasOf(row).usage).toEqual({
      shape: "openai_chat",
      raw: { prompt_tokens: 0, total_tokens: 0 },
    });
  });

  it("NO success row is emitted alongside it (the finally is gated on !failed)", async () => {
    const boom = auth401();
    const Cls = instrument("OpenAI", "gpt-4o", async () => streamThatThrows([], boom));
    const session = newSession();

    await driveStream(session, Cls);

    expect(logged).toHaveLength(1);
    expect(
      logged.filter((r) => extrasOf(r).call_outcome?.status === "success"),
    ).toHaveLength(0);
  });

  it("a 429 classifies as rate_limited (classification is not hard-coded)", async () => {
    const boom = Object.assign(new Error("429 slow down"), { status: 429 });
    const Cls = instrument("OpenAI", "gpt-4o", async () => streamThatThrows([], boom));
    const session = newSession();

    const { caught } = await driveStream(session, Cls);

    expect(caught).toBe(boom);
    expect(logged).toHaveLength(1);
    expect(extrasOf(logged[0]).call_outcome.error_kind).toBe("rate_limited");
    expect(extrasOf(logged[0]).call_outcome.http_status).toBe(429);
  });
});

// ── 2. Mid-stream failure AFTER a usage-bearing chunk ────────────

describe("guarded LlamaIndex stream — mid-stream failure after a usage chunk", () => {
  it("failure row only — no PRICED success row from the accumulated usage", async () => {
    const boom = Object.assign(new Error("upstream 500"), { status: 500 });
    const Cls = instrument("OpenAI", "gpt-4o", async () =>
      streamThatThrows(
        [
          { delta: "Hel", raw: { model: "gpt-4o" } },
          // The escalation: this chunk's usage is exactly what the pre-fix
          // finally turned into a fully-priced success row for a failed call.
          { delta: "lo", raw: { model: "gpt-4o", usage: OPENAI_USAGE } },
        ],
        boom,
      ),
    );
    const session = newSession();

    const { caught, seen } = await driveStream(session, Cls);

    expect(caught).toBe(boom);
    expect(seen).toHaveLength(2); // customer got every chunk before the failure

    expect(logged).toHaveLength(1);
    const row = logged[0];
    expect(extrasOf(row).call_outcome.status).toBe("failed");
    expect(extrasOf(row).call_outcome.error_kind).toBe("server_error");
    // Not priced: the observed usage never reaches the row.
    expect(row[INPUT_TOKENS_ARG]).toBe(0);
    expect(row[OUTPUT_TOKENS_ARG]).toBe(0);
    expect(row[CACHED_TOKENS_ARG]).toBe(0);
    expect(extrasOf(row).usage?.raw).toEqual({ prompt_tokens: 0, total_tokens: 0 });
    // No row anywhere carries the provider's verbatim usage block.
    expect(
      logged.filter((r) => extrasOf(r).usage?.raw?.prompt_tokens === 100),
    ).toHaveLength(0);
  });

  it("Gemini-shaped usage on every chunk still yields exactly one failed row", async () => {
    // google chunks carry usageMetadata on EVERY chunk, so pre-fix even a
    // first-chunk failure landed priced.
    const boom = Object.assign(new Error("504 gateway timeout"), { status: 504 });
    const Cls = instrument("Gemini", "gemini-2.0-flash", async () =>
      streamThatThrows(
        [
          {
            delta: "a",
            raw: {
              modelVersion: "gemini-2.0-flash",
              usageMetadata: { promptTokenCount: 100, candidatesTokenCount: 20 },
            },
          },
        ],
        boom,
      ),
    );
    const session = newSession();

    const { caught } = await driveStream(session, Cls);

    expect(caught).toBe(boom);
    expect(logged).toHaveLength(1);
    expect(extrasOf(logged[0]).call_outcome.status).toBe("failed");
    expect(logged[0][PROVIDER_ARG]).toBe("google");
    expect(logged[0][MODEL_ARG]).toBe("gemini-2.0-flash");
    expect(logged[0][INPUT_TOKENS_ARG]).toBe(0);
  });
});

// ── 3. Consumer early break — behavior UNCHANGED by the fix ──────

describe("guarded LlamaIndex stream — consumer early break", () => {
  it("partial success row exactly as before the fix, and no failure row", async () => {
    const boom = auth401();
    const Cls = instrument("OpenAI", "gpt-4o", async () =>
      streamThatThrows(
        [
          { delta: "a", raw: { model: "gpt-4o", usage: OPENAI_USAGE } },
          { delta: "b", raw: { model: "gpt-4o" } },
        ],
        boom, // never reached: the consumer breaks first
      ),
    );
    const session = newSession();

    await _getSessionStorage().run(session, async () => {
      const result = await new Cls().chat({ messages: [], stream: true });
      for await (const _c of result) break; // .return() → finally only, no catch
    });

    expect(logged).toHaveLength(1);
    const outcome = extrasOf(logged[0]).call_outcome;
    expect(outcome.status).toBe("success");
    expect(outcome.error_kind).toBeUndefined();
    // The partial row is still built from what was actually observed.
    expect(logged[0][INPUT_TOKENS_ARG]).toBe(70); // 100 - 30 cached
    expect(logged[0][OUTPUT_TOKENS_ARG]).toBe(20);
    expect(session.inLlamaIndex).toBe(false);
  });

  it("a fully drained healthy stream is unchanged (control)", async () => {
    const Cls = instrument("OpenAI", "gpt-4o", async () =>
      streamThatThrows([{ delta: "a", raw: { model: "gpt-4o", usage: OPENAI_USAGE } }], null),
    );
    const session = newSession();

    const { caught, seen } = await driveStream(session, Cls);

    expect(caught).toBeUndefined();
    expect(seen).toHaveLength(1);
    expect(logged).toHaveLength(1);
    expect(extrasOf(logged[0]).call_outcome.status).toBe("success");
    expect(logged[0][INPUT_TOKENS_ARG]).toBe(70);
  });
});

// ── 4. Error identity (golden rule) ──────────────────────────────

describe("golden rule — the SDK never substitutes the customer's error", () => {
  it("the consumer catches the EXACT Error object the stream threw", async () => {
    const boom = auth401();
    const Cls = instrument("OpenAI", "gpt-4o", async () =>
      streamThatThrows([{ delta: "a", raw: {} }], boom),
    );
    const session = newSession();

    let caught: unknown;
    await _getSessionStorage().run(session, async () => {
      try {
        for await (const _c of await new Cls().chat({ messages: [], stream: true })) {
          /* drain until it throws */
        }
      } catch (e) {
        caught = e;
      }
    });

    expect(caught).toBe(boom); // identity, not just instanceof
    expect((caught as any).status).toBe(401);
    expect((caught as Error).message).toBe("401 Incorrect API key provided");
  });

  it("a non-Error thrown value also propagates by identity", async () => {
    const boom = { code: "ECONNRESET" } as any;
    const Cls = instrument("OpenAI", "gpt-4o", async () => streamThatThrows([], boom));
    const session = newSession();

    const { caught } = await driveStream(session, Cls);
    expect(caught).toBe(boom);
    expect(logged).toHaveLength(1);
    expect(extrasOf(logged[0]).call_outcome.status).toBe("failed");
  });

  it("tp.log exploding inside the failure emit still surfaces only the provider error", async () => {
    setClient({
      firewall: "off",
      log: () => {
        throw new Error("log exploded");
      },
    } as any);
    const boom = auth401();
    const Cls = instrument("OpenAI", "gpt-4o", async () => streamThatThrows([], boom));
    const session = newSession();

    const { caught } = await driveStream(session, Cls);

    expect(caught).toBe(boom);
    expect(session.inLlamaIndex).toBe(false); // guard still released
  });
});

// ── 5. Guard hygiene after a failed stream ───────────────────────

describe("guard hygiene — a failed stream leaves the session clean", () => {
  it("exitLlamaIndex ran exactly once and the attempt slots/outcome were consumed", async () => {
    const boom = auth401();
    const Cls = instrument("OpenAI", "gpt-4o", async () => streamThatThrows([], boom));
    const session = newSession();

    await driveStream(session, Cls);

    // Exactly once: the catch must NOT call exitLlamaIndex (the finally does).
    // A double-exit would clamp at 0 and go unnoticed here, but a MISSING exit
    // makes every later call short-circuit — asserted by the next test.
    expect(session.inLlamaIndex).toBe(false);
    expect((session as any)._call_outcome).toBeFalsy();
    expect((session as any)._attempted_model).toBeUndefined();
    expect((session as any)._attempted_provider).toBeUndefined();
    expect((session as any)._attempted_shape).toBeUndefined();
  });

  it("a subsequent successful call on the SAME session logs normally", async () => {
    const boom = auth401();
    const Failing = instrument("OpenAI", "gpt-4o", async () => streamThatThrows([], boom));
    const session = newSession();

    await driveStream(session, Failing);
    expect(logged).toHaveLength(1);
    expect(extrasOf(logged[0]).call_outcome.status).toBe("failed");

    // Second call, non-streaming, same session — must not be short-circuited by
    // a leaked inLlamaIndex guard and must not inherit the failed outcome.
    const Ok = instrument("Gemini", "gemini-2.0-flash", async () => ({
      raw: {
        modelVersion: "gemini-2.0-flash",
        usageMetadata: { promptTokenCount: 100, candidatesTokenCount: 20 },
      },
      message: { role: "assistant", content: "hi" },
    }));
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
});

// ── 6. OpenAIResponses pseudo-provider parity ────────────────────

describe("OpenAIResponses — the failure row carries the same identity as its successful siblings", () => {
  it("failed streamed Responses call → provider + usage_shape 'openai_responses'", async () => {
    const boom = auth401();
    const Cls = instrument("OpenAIResponses", "gpt-5", async () =>
      streamThatThrows(
        [
          {
            delta: "Hi",
            raw: { type: "response.output_text.delta", response: { model: "gpt-5", usage: null } },
          },
        ],
        boom,
      ),
    );
    const session = newSession();

    const { caught } = await driveStream(session, Cls);

    expect(caught).toBe(boom);
    expect(logged).toHaveLength(1);
    const row = logged[0];
    expect(row[PROVIDER_ARG]).toBe("openai_responses");
    expect(row[MODEL_ARG]).toBe("gpt-5");
    expect(extrasOf(row).call_outcome.status).toBe("failed");
    // Same usage_shape a SUCCESSFUL OpenAIResponses row stamps — asserted
    // against the success path below rather than hard-coded twice.
    expect(extrasOf(row).usage.shape).toBe("openai_responses");
  });

  it("the successful sibling stamps that same shape (parity control)", async () => {
    const Cls = instrument("OpenAIResponses", "gpt-5", async () =>
      streamThatThrows(
        [
          {
            delta: "Hi",
            raw: {
              type: "response.completed",
              response: { model: "gpt-5-2025-08-07", usage: RESP_USAGE },
            },
          },
        ],
        null,
      ),
    );
    const session = newSession();

    await driveStream(session, Cls);

    expect(logged).toHaveLength(1);
    expect(logged[0][PROVIDER_ARG]).toBe("openai_responses");
    expect(extrasOf(logged[0]).usage.shape).toBe("openai_responses");
    expect(extrasOf(logged[0]).call_outcome.status).toBe("success");
  });
});
