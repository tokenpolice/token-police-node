/**
 * Framework SUCCESS emitters must ship the `reroute_rejected` observation
 * `_applyReroute` mints for an unappliable (body-less) call shape — not just
 * mint it into the module-global queue and leave it there.
 *
 * Bug: on a live ENFORCE REROUTE directive, `_applyReroute(directive, null,
 * …)` correctly refuses (a hint-only LlamaIndex call has no request body to
 * swap a model into) and pushes a `reroute_rejected` observation keyed to the
 * call's obs key. Only the FAILURE emitter drained that queue — `_logLlamaIndex`
 * (the SUCCESS emitter, both the non-streaming tail and the streaming
 * generator's `finally`) shipped no `observations` at all, so on every PAID
 * (successful) call the observation sat in the queue for OBS_STALE_MS and was
 * later swept onto an arbitrary unrelated row, or died with the process.
 *
 * Fix under test: `_logLlamaIndex` now normalizes its obs key
 * (`obsKey !== undefined ? obsKey : _currentObsKey()`, never letting
 * `undefined` reach `drainObservations`'s drain-ALL sentinel), drains that key
 * in its own try/catch, and spreads `observations` into the `/log` extras when
 * non-empty. The streaming wrapper threads the SAME key it captured at
 * wrapper entry into the generator's `finally` — required because ALS does
 * not survive `yield`, so a live read there would be null or a sibling's key.
 *
 * The suite-wide blind spot this bug exposed: every PRE-EXISTING rejection
 * test asserts against `state.drainObservations()` directly with `client.log`
 * stubbed to a no-op — minted, never verified SHIPPED. Every assertion below
 * instead reads `observations` off the CAPTURED `/log` payload (the
 * `extrasOf(...)` style from tests/llamaIndexEagerFailure.test.ts:577).
 *
 * Harness mirrors tests/llamaIndexEagerFailure.test.ts and
 * tests/llamaIndexStreamFailure.test.ts (fake LlamaIndex provider classes via
 * `_instrumentLlamaIndexProvider`, pinned session, captured `tp.log` args) and
 * tests/rerouteUnappliableShape.test.ts's "State B integration" /
 * "LangChain wrapper end-to-end" sections (real `TokenPolice` client,
 * `deployment: "serverless"` + `resetPack()` to force State-B inline /check,
 * `client.check` stubbed to resolve a live reroute directive).
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import * as state from "../src/state";
import { setClient, resetPack } from "../src/state";
import { TPSession, _getSessionStorage } from "../src/context";
import { __test__ as enforcerTest } from "../src/enforcer";
import { TokenPolice } from "../src/client";
import { stashLocalDecision, claimLocalDecision } from "../src/localDecisionStore";

const MODEL_ARG = 4;
const PROVIDER_ARG = 5;

/** `tp.log`'s trailing options arg — where `observations`/`call_outcome` live. */
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

/** OpenAI-SDK-shaped 401 (`AuthenticationError` + `status`). */
function auth401(): any {
  class AuthenticationError extends Error {}
  return Object.assign(new AuthenticationError("401 Incorrect API key provided"), {
    status: 401,
  });
}

function eagerReject(err: unknown): () => Promise<never> {
  return async () => {
    await sleep(2);
    throw err;
  };
}

/** Async-iterable stand-in for a streamChat result that yields then ends cleanly. */
function streamThatEnds(chunks: any[]): AsyncIterable<any> {
  return (async function* () {
    for (const c of chunks) {
      await sleep(2);
      yield c;
    }
  })();
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
      if (result != null && typeof result[Symbol.asyncIterator] === "function") {
        for await (const c of result) seen.push(c);
      }
    } catch (e) {
      caught = e;
    }
  });
  return { caught, seen };
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

const ANTHROPIC_OK = {
  raw: {
    model: "claude-sonnet-4-5",
    usage: { input_tokens: 100, output_tokens: 20, cache_read_input_tokens: 30 },
  },
  message: { role: "assistant", content: "hi" },
};
const OPENAI_USAGE = {
  prompt_tokens: 100,
  completion_tokens: 20,
  prompt_tokens_details: { cached_tokens: 30 },
};

function makeClient(firewall: "enforce" | "dry_run" | "off" = "enforce"): TokenPolice {
  const client = new TokenPolice({
    apiKey: "tp_sk_test_rr_success_emit",
    baseUrl: "http://localhost:59999",
    timeout: 0.1,
    firewall,
    deployment: "serverless",
  } as never);
  setClient(client);
  return client;
}

function stubCheck(client: TokenPolice, checkResult: unknown) {
  return vi.spyOn(client, "check").mockResolvedValue(checkResult as never);
}

function captureLog(client: TokenPolice): any[][] {
  const logged: any[][] = [];
  vi.spyOn(client, "log").mockImplementation((...args: any[]) => {
    logged.push(args);
  });
  return logged;
}

/** A live ENFORCE REROUTE directive, unappliable against every body-less call. */
const rerouteDirective = (
  model = "gpt-4o-mini",
  provider = "openai",
  ruleId = "rule_rr",
) => ({
  status: "allowed",
  reroute: { mode: "enforce", model, provider, rule_id: ruleId },
});

beforeEach(() => {
  resetPack(); // no healthy daemon pack → every /check goes through State B
  try {
    state.drainObservations();
  } catch {
    /* drain any leftover observations from a prior test */
  }
});

afterEach(() => {
  setClient(null as any);
  vi.restoreAllMocks();
  resetPack();
  // Undo every prototype patch this file installed.
  for (const thunk of enforcerTest._restoreThunks.splice(0)) {
    try {
      thunk();
    } catch {
      /* ignore */
    }
  }
});

// ── N1. Non-stream success ships the rejection ──────────────────

describe("N1 — LlamaIndex non-stream success under an unappliable REROUTE rule", () => {
  it("captured /log payload extras.observations contains exactly one reroute_rejected observation", async () => {
    const client = makeClient("enforce");
    stubCheck(client, rerouteDirective());
    const logged = captureLog(client);
    const Cls = instrument("Anthropic", "claude-sonnet-4-5", async () => {
      await sleep(2);
      return ANTHROPIC_OK;
    });
    const session = newSession();

    const { caught } = await driveChat(session, Cls);

    expect(caught).toBeUndefined();
    expect(logged).toHaveLength(1);
    const extras = extrasOf(logged[0]);
    expect(extras.observations).toHaveLength(1);
    expect(extras.observations[0].outcome).toBe("reroute_rejected");
    expect(extras.observations[0].rejection_reason).toBe("unappliable_call_shape");
    expect(extras.observations[0].rule_id).toBe("rule_rr");
    // The call itself ran unmodified — enforcement never differs from the app's request.
    expect(logged[0][MODEL_ARG]).toBe("claude-sonnet-4-5");
    expect(logged[0][PROVIDER_ARG]).toBe("anthropic");
  });
});

// ── N2. Stream success ships the rejection (generator finally) ──

describe("N2 — LlamaIndex stream success under an unappliable REROUTE rule", () => {
  it("captured /log payload extras.observations contains exactly one reroute_rejected observation", async () => {
    const client = makeClient("enforce");
    stubCheck(client, rerouteDirective());
    const logged = captureLog(client);
    const Cls = instrument("OpenAI", "gpt-4o", async () =>
      streamThatEnds([
        { delta: "Hel", raw: { model: "gpt-4o" } },
        { delta: "lo", raw: { model: "gpt-4o", usage: OPENAI_USAGE } },
      ]),
    );
    const session = newSession();

    const { caught, seen } = await driveStream(session, Cls);

    expect(caught).toBeUndefined();
    expect(seen).toHaveLength(2); // the customer's stream is untouched
    expect(logged).toHaveLength(1);
    const extras = extrasOf(logged[0]);
    expect(extras.observations).toHaveLength(1);
    expect(extras.observations[0].outcome).toBe("reroute_rejected");
    expect(extras.observations[0].rejection_reason).toBe("unappliable_call_shape");
    // Proves the priced success row is intact alongside the shipped observation.
    expect(logged[0][MODEL_ARG]).toBe("gpt-4o");
  });
});

// ── N3. Sibling isolation — no cross-call theft ──────────────────

describe("N3 — sibling isolation: an in-flight stream and a call that completes mid-flight", () => {
  it("each row ships only its own observation; neither steals the other's", async () => {
    const client = makeClient("enforce");
    stubCheck(client, rerouteDirective());
    const logged = captureLog(client);

    let releaseGate: () => void = () => {};
    const gate = new Promise<void>((resolve) => {
      releaseGate = resolve;
    });

    // Call A: a stream that pauses mid-flight (after its first chunk) until
    // the test explicitly releases it — stands in for the real gap between
    // the pre-flight and the provider's own HTTP response.
    const ClsA = instrument("OpenAI", "gpt-4o", async () =>
      (async function* () {
        await sleep(2);
        yield { delta: "a1", raw: { model: "gpt-4o" } };
        await gate;
        yield { delta: "a2", raw: { model: "gpt-4o", usage: OPENAI_USAGE } };
      })(),
    );
    const ClsB = instrument("Anthropic", "claude-3-haiku", async () => {
      await sleep(2);
      return {
        raw: {
          model: "claude-3-haiku",
          usage: { input_tokens: 10, output_tokens: 5 },
        },
        message: { role: "assistant", content: "hi" },
      };
    });

    // Distinct sessions: `session.inLlamaIndex` serializes concurrent calls
    // WITHIN one session by design (a second call on the same session while
    // one is in flight is a deliberate pass-through, covered by
    // llamaIndexEagerFailure.test.ts's "inner provider wrapper is inert"
    // test) — genuine LlamaIndex concurrency happens across sessions, e.g.
    // two different customer requests in flight at once.
    const sessionA = newSession();
    const sessionB = newSession();

    // Start stream A but do not await it to completion yet.
    let caughtA: unknown;
    const seenA: any[] = [];
    const promiseA = _getSessionStorage().run(sessionA, async () => {
      try {
        const result = await new ClsA().chat({ messages: [], stream: true });
        for await (const c of result) seenA.push(c);
      } catch (e) {
        caughtA = e;
      }
    });

    // Let stream A reach its paused point (past the first yield).
    await sleep(10);

    // Run call B to full completion on its own session while A is paused.
    await _getSessionStorage().run(sessionB, async () => {
      await new ClsB().chat({ messages: [] });
    });

    // Now resume and finish stream A.
    releaseGate();
    await promiseA;

    expect(caughtA).toBeUndefined();
    expect(seenA).toHaveLength(2);
    expect(logged).toHaveLength(2);

    const rowA = logged.find((r) => r[MODEL_ARG] === "gpt-4o")!;
    const rowB = logged.find((r) => r[MODEL_ARG] === "claude-3-haiku")!;
    expect(rowA).toBeTruthy();
    expect(rowB).toBeTruthy();

    const obsA = extrasOf(rowA).observations;
    const obsB = extrasOf(rowB).observations;
    expect(obsA).toHaveLength(1);
    expect(obsB).toHaveLength(1);
    // Each observation's `from.model` (the wrapper's own model hint) proves
    // it belongs to its own call, not a stolen sibling's.
    expect(obsA[0].reroute.from.model).toBe("gpt-4o");
    expect(obsB[0].reroute.from.model).toBe("claude-3-haiku");
  });
});

// ── N4. Failure path unchanged — no double emission ──────────────

describe("N4 — failure path: exactly one row, exactly one copy of the observation", () => {
  it("the failure emitter ships the rejection; the success emitter never also fires", async () => {
    const client = makeClient("enforce");
    stubCheck(client, rerouteDirective());
    const logged = captureLog(client);
    const boom = auth401();
    const Cls = instrument("Anthropic", "claude-sonnet-4-5", eagerReject(boom));
    const session = newSession();

    const { caught } = await driveChat(session, Cls);

    expect(caught).toBe(boom);
    expect(logged).toHaveLength(1);
    const extras = extrasOf(logged[0]);
    expect(extras.call_outcome.status).toBe("failed");
    expect(extras.observations).toHaveLength(1);
    expect(extras.observations[0].outcome).toBe("reroute_rejected");
    // No success row also emitted alongside the failure row.
    expect(
      logged.filter((r) => extrasOf(r).call_outcome?.status === "success"),
    ).toHaveLength(0);
  });
});

// ── N6. local_decision fence — observations-only, never a claim ──

describe("N6 — local_decision fence: the success emitter never claims a local_decision", () => {
  it("an untagged local_decision seeded before the call ships nowhere and survives afterward", async () => {
    const client = makeClient("enforce");
    stubCheck(client, rerouteDirective());
    const logged = captureLog(client);
    const Cls = instrument("Anthropic", "claude-sonnet-4-5", async () => {
      await sleep(2);
      return ANTHROPIC_OK;
    });
    const session = newSession();
    const untagged = {
      outcome: "rerouted",
      rule_id: "someone_elses_rule",
      reroute: { from: { model: "x" }, to: { model: "y" } },
    };
    // Simulates a genuinely concurrent degraded-path sibling's stash that
    // never got tagged to its own key (mirrors localDecisionStore.test.ts's
    // untagged-fallback tests).
    stashLocalDecision(session as any, untagged, null);

    const { caught } = await driveChat(session, Cls);

    expect(caught).toBeUndefined();
    expect(logged).toHaveLength(1);
    const extras = extrasOf(logged[0]);
    // Observations-only: the mint still ships…
    expect(extras.observations).toHaveLength(1);
    // …but no local_decision key at all — this emitter never claims one.
    expect(extras.local_decision).toBeUndefined();

    // The untagged decision was never touched — still claimable by whoever
    // actually owns it (not stolen by this call's row).
    const claimed = claimLocalDecision(session as any, "any-later-key");
    expect(claimed).toEqual(untagged);
  });
});

// ── N7. No-op reroute — nothing minted, nothing shipped ──────────

describe("N7 — no-op reroute arm: zero observations shipped, zero minted", () => {
  it("a per-rule DRY_RUN directive never reaches appliability — no observations on the row", async () => {
    const client = makeClient("enforce");
    stubCheck(client, {
      status: "allowed",
      reroute: { mode: "dry_run", model: "gpt-4o-mini", provider: "openai", rule_id: "rule_rr" },
    });
    const logged = captureLog(client);
    const Cls = instrument("Anthropic", "claude-sonnet-4-5", async () => {
      await sleep(2);
      return ANTHROPIC_OK;
    });
    const session = newSession();

    const { caught } = await driveChat(session, Cls);

    expect(caught).toBeUndefined();
    expect(logged).toHaveLength(1);
    expect(extrasOf(logged[0]).observations).toBeUndefined();
    // Nothing left behind in the queue either.
    expect(state.drainObservations()).toHaveLength(0);
  });

  it("no reroute directive at all → same (the ordinary allowed-with-nothing-to-report case)", async () => {
    const client = makeClient("enforce");
    stubCheck(client, { status: "allowed" });
    const logged = captureLog(client);
    const Cls = instrument("Anthropic", "claude-sonnet-4-5", async () => {
      await sleep(2);
      return ANTHROPIC_OK;
    });
    const session = newSession();

    await driveChat(session, Cls);

    expect(logged).toHaveLength(1);
    expect(extrasOf(logged[0]).observations).toBeUndefined();
    expect(state.drainObservations()).toHaveLength(0);
  });
});
