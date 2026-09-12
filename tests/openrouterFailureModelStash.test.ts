/**
 * Native-OpenRouter failure rows must carry the inner `chatRequest.model`, and a
 * mid-stream failure must not have its model stolen by a later same-session call.
 *
 * The native OpenRouter SDK (`@openrouter/sdk`) sends a Speakeasy envelope —
 * `{ chatRequest: { model, messages } }` — with NO top-level `model`. Two
 * mechanisms keep that model on the failure row, and this suite locks both for
 * the openrouter shape (the bedrock `modelId` shape is covered by the twin
 * `tests/bedrockFailureModelStash.test.ts`):
 *
 * 1. `_stashAttemptContext`'s trailing `modelHint` fallback, forwarded by the
 *    generic manual wrapper from the SAME hint it derives for the pre-flight
 *    check (`reqBody.chatRequest.model`). Without it a rejected `chat.send()`
 *    emits `model='unknown'`.
 * 2. `_wrapManualStream` now snapshots the session-global `_attempted_*` slots
 *    at wrap time and restores them before emitting a mid-stream failure
 *    (mirroring `_wrapBedrockConverseStream`), so a LATER same-session call that
 *    overwrote them cannot steal the stream's model — and the newer call's slots
 *    are handed back afterwards, since `_emitCallFailureLog` clears whatever it
 *    emitted.
 *
 * Losing the model on a gateway provider costs more than the model column: the
 * collector derives `original_provider` from the `vendor/` prefix of the model
 * slug, so a `model='unknown'` openrouter row blanks the routed-vendor
 * attribution too.
 *
 * Drives the REAL instrumented `Chat.prototype.send` through a fake
 * `@openrouter/sdk` namespace (the `instrumentModules` pattern from the bedrock
 * twin; the envelope fixture is the one from
 * `tests/preflightCtxF7.test.ts` §3), because the stash happens inside the
 * module-internal wrapper — a direct `__test__` call would not exercise it.
 *
 * `tp.log` positional args (see `_logManual` / `_emitCallFailureLog`):
 * [4]=model, [5]=provider, [6]=inputTokens, [7]=outputTokens; the trailing
 * options arg carries `call_outcome`.
 *
 * GOLDEN RULE arm: in every failure case the customer receives the provider's
 * OWN error object by identity, including when the /log dispatch itself throws.
 */
import { describe, test, expect, afterEach, vi } from "vitest";
import { init } from "../src/client";
import { uninstrument, __test__ as enforcerTest } from "../src/enforcer";
import { setClient, resetPack } from "../src/state";
import { session, getCurrentSession } from "../src/context";

const { _stashAttemptContext } = enforcerTest as any;

// A realistic OpenRouter slug — `vendor/model`, the value that used to be lost
// as "unknown" (taking the collector's original_provider with it).
const MODEL = "openai/gpt-4o-mini";

/** The Speakeasy envelope: the model lives ONLY on `chatRequest`. */
function envelope(extra: Record<string, any> = {}): any {
  return {
    chatRequest: {
      model: MODEL,
      messages: [{ role: "user", content: "hi" }],
      ...extra,
    },
  };
}

/** Speakeasy `SDKError` stand-in: the status lives on `statusCode`. */
function unauthorized(): any {
  const err: any = new Error("API error occurred: Status 401 - No auth credentials found");
  err.name = "SDKError";
  err.statusCode = 401;
  return err;
}

/** An OpenAI-wire content delta — what OpenRouter streams back. */
function deltaChunk(text: string): any {
  return { id: "gen-1", choices: [{ index: 0, delta: { content: text } }] };
}

/**
 * Fake `@openrouter/sdk` namespace. `Chat` is NOT root-exported (matching the
 * real SDK), so `_instrumentOpenRouter` has to probe an `OpenRouter` instance to
 * reach `Chat.prototype.send` — the exact path under test.
 */
function makeFakeOpenRouter(sendImpl: (req: any) => Promise<any>) {
  const sendCalls: any[] = [];
  class Chat {
    async send(req: any): Promise<any> {
      sendCalls.push(req);
      return await sendImpl(req);
    }
  }
  class OpenRouter {
    chat = new Chat();
    constructor(_opts: any) {}
  }
  return { ns: { OpenRouter }, sendCalls };
}

/** Instrument the fake module and capture every `tp.log` call. */
function arrange(
  sendImpl: (req: any) => Promise<any>,
  opts: { logThrows?: boolean } = {},
) {
  const { ns, sendCalls } = makeFakeOpenRouter(sendImpl);
  const client = init({
    apiKey: "tp_sk_test",
    deployment: "serverless",
    firewall: "dry_run",
    instrumentModules: { openrouter: ns },
  } as any);
  vi.spyOn(client, "check").mockResolvedValue({
    status: "allowed",
    fail_open: false,
  } as any);
  const logs: any[][] = [];
  vi.spyOn(client, "log").mockImplementation((...args: any[]) => {
    logs.push(args);
    if (opts.logThrows) throw new Error("log exploded");
  });
  return { ns, sendCalls, client, logs };
}

/** `tp.log`'s trailing options arg — where `call_outcome` lives. */
function extrasOf(logArgs: any[]): any {
  return logArgs[logArgs.length - 1];
}

/**
 * Only the rows emitted by the failure funnel. The status check matters: the
 * enclosing `session()` also lands a structural row carrying a SUCCESS
 * `call_outcome` (model/provider empty), which is not this call's llm row.
 */
function failureRows(logs: any[][]): any[][] {
  return logs.filter((c) => {
    const extras = extrasOf(c);
    return (
      extras &&
      typeof extras === "object" &&
      extras.call_outcome?.status === "failed"
    );
  });
}

afterEach(() => {
  try {
    uninstrument();
  } catch {
    /* ignore */
  }
  try {
    setClient(undefined as any);
  } catch {
    /* ignore */
  }
  try {
    resetPack();
  } catch {
    /* ignore */
  }
  vi.restoreAllMocks();
});

// ═══════════════════════════════════════════════════════════════════
// 1 — chat.send() rejected (no stream ever established)
// ═══════════════════════════════════════════════════════════════════
describe("native OpenRouter chatRequest envelope failure carries the inner model", () => {
  test("401 at chat.send → failed row model=chatRequest.model, provider=openrouter; error rethrown by identity", async () => {
    const boom = unauthorized();
    const { ns, logs } = arrange(async () => {
      throw boom;
    });

    const or: any = new ns.OpenRouter({ apiKey: "k" });
    let caught: unknown;
    await session({ name: "wf" }, async () => {
      try {
        await or.chat.send(envelope());
      } catch (e) {
        caught = e;
      }
    });
    // GOLDEN RULE: the provider's own error object, not a wrapper.
    expect(caught).toBe(boom);

    const rows = failureRows(logs);
    expect(rows).toHaveLength(1);
    expect(rows[0][4]).toBe(MODEL); // was "unknown" before the modelHint fix
    expect(rows[0][5]).toBe("openrouter");
    // A failed call has no usage — the row stays unmeasured.
    expect(rows[0][6]).toBe(0);
    expect(rows[0][7]).toBe(0);
    const extras = extrasOf(rows[0]);
    expect(extras.call_outcome.status).toBe("failed");
    expect(extras.call_outcome.error_kind).toBe("auth_error");
    expect(extras.call_outcome.http_status).toBe(401);
    // The failed row carries the SAME usage_shape its successful siblings log.
    expect(extras.usage.shape).toBe("openrouter_routed");
  });

  test("GOLDEN RULE: tp.log itself throwing still surfaces only the provider error", async () => {
    const boom = unauthorized();
    const { ns, logs } = arrange(
      async () => {
        throw boom;
      },
      { logThrows: true },
    );

    const or: any = new ns.OpenRouter({ apiKey: "k" });
    let caught: unknown;
    await session({ name: "wf" }, async () => {
      try {
        await or.chat.send(envelope());
      } catch (e) {
        caught = e;
      }
    });
    expect(caught).toBe(boom);
    // The dispatch was attempted (and its throw swallowed) with the right model.
    const rows = failureRows(logs);
    expect(rows).toHaveLength(1);
    expect(rows[0][4]).toBe(MODEL);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 2 — mid-stream failure through _wrapManualStream (stream established,
//     iterator throws partway). OpenRouter's single `send` serves both
//     modes, so the runtime async-iterable check routes it here.
// ═══════════════════════════════════════════════════════════════════
describe("_wrapManualStream mid-stream failure carries the inner model", () => {
  test("iterator throws after one chunk → failed row model=chatRequest.model; chunks up to the failure still delivered", async () => {
    const boom = Object.assign(new Error("upstream 500"), { statusCode: 500 });
    const { ns, logs } = arrange(async () => {
      async function* gen() {
        yield deltaChunk("a");
        throw boom;
      }
      return gen();
    });

    const or: any = new ns.OpenRouter({ apiKey: "k" });
    const seen: any[] = [];
    let caught: unknown;
    await session({ name: "wf" }, async () => {
      const stream: any = await or.chat.send(envelope({ stream: true }));
      try {
        for await (const chunk of stream) seen.push(chunk);
      } catch (e) {
        caught = e;
      }
    });
    expect(caught).toBe(boom);
    expect(seen).toHaveLength(1); // chunks delivered up to the failure

    const rows = failureRows(logs);
    expect(rows).toHaveLength(1);
    expect(rows[0][4]).toBe(MODEL);
    expect(rows[0][5]).toBe("openrouter");
    const outcome = extrasOf(rows[0]).call_outcome;
    expect(outcome.status).toBe("failed");
    expect(outcome.error_kind).toBe("server_error");
    expect(outcome.http_status).toBe(500);
    // No SUCCESS usage row — the stream never reached a usage chunk.
    expect(
      logs.filter((r) => (r[6] as number) > 0 || (r[7] as number) > 0),
    ).toHaveLength(0);
  });

  test("GOLDEN RULE: tp.log throwing on the mid-stream failure path never reaches the consumer", async () => {
    const boom = Object.assign(new Error("upstream 500"), { statusCode: 500 });
    const { ns, logs } = arrange(
      async () => {
        async function* gen() {
          yield deltaChunk("a");
          throw boom;
        }
        return gen();
      },
      { logThrows: true },
    );

    const or: any = new ns.OpenRouter({ apiKey: "k" });
    let caught: unknown;
    await session({ name: "wf" }, async () => {
      const stream: any = await or.chat.send(envelope({ stream: true }));
      try {
        for await (const _chunk of stream) void _chunk;
      } catch (e) {
        caught = e;
      }
    });
    expect(caught).toBe(boom);
    const rows = failureRows(logs);
    expect(rows).toHaveLength(1);
    expect(rows[0][4]).toBe(MODEL);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 3 — snapshot/restore: a newer same-session call must not steal the model
// ═══════════════════════════════════════════════════════════════════
describe("_wrapManualStream attempt-context snapshot/restore", () => {
  test("a later call overwrote the slots → the failure row keeps the stream's model, and the newer call's slots are handed back", async () => {
    const boom = Object.assign(new Error("upstream 500"), { statusCode: 500 });
    const { ns, logs } = arrange(async () => {
      async function* gen() {
        yield deltaChunk("a");
        throw boom;
      }
      return gen();
    });

    const or: any = new ns.OpenRouter({ apiKey: "k" });
    let sess: any;
    let caught: unknown;
    await session({ name: "wf" }, async () => {
      sess = getCurrentSession();
      // Stream established → this call's context is stashed (via the envelope
      // hint) and the wrapper snapshotted it.
      const stream: any = await or.chat.send(envelope({ stream: true }));
      expect(sess._attempted_model).toBe(MODEL);

      // A SECOND in-flight call on the same session overwrites the
      // session-global slots (exactly what the real wrapper does for it)
      // BEFORE the openrouter stream fails.
      _stashAttemptContext(sess, "openai", [{ model: "gpt-4o-mini" }], "chat");
      expect(sess._attempted_model).toBe("gpt-4o-mini");

      try {
        for await (const _chunk of stream) void _chunk;
      } catch (e) {
        caught = e;
      }
    });
    // GOLDEN RULE: the stream error reaches the consumer unchanged.
    expect(caught).toBe(boom);

    // The stream's OWN model, not the newer call's — and, because the slug
    // keeps its `openai/` prefix, the collector can still derive
    // original_provider from this row.
    const rows = failureRows(logs);
    expect(rows).toHaveLength(1);
    expect(rows[0][4]).toBe(MODEL);
    expect(rows[0][5]).toBe("openrouter");

    // …and the newer call's context is back in place afterwards (the emit
    // clears the slots, so without the hand-back its own failure row would
    // degrade to model="unknown").
    expect(sess._attempted_model).toBe("gpt-4o-mini");
    expect(sess._attempted_provider).toBe("openai");
    expect(sess._attempted_operation).toBe("chat");
  });

  test("no overwrite → the slots are left cleared by the emit (no phantom restore)", async () => {
    const boom = Object.assign(new Error("upstream 500"), { statusCode: 500 });
    const { ns, logs } = arrange(async () => {
      async function* gen() {
        yield deltaChunk("a");
        throw boom;
      }
      return gen();
    });

    const or: any = new ns.OpenRouter({ apiKey: "k" });
    let sess: any;
    await session({ name: "wf" }, async () => {
      sess = getCurrentSession();
      const stream: any = await or.chat.send(envelope({ stream: true }));
      try {
        for await (const _chunk of stream) void _chunk;
      } catch {
        /* expected */
      }
    });
    expect(failureRows(logs)[0][4]).toBe(MODEL);
    // Nothing newer to hand back — the post-emit state stays clean.
    expect(sess._attempted_model).toBeUndefined();
    expect(sess._attempted_provider).toBeUndefined();
  });
});
