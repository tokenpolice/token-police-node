/**
 * Bedrock failure rows must carry the real `modelId` (Node analogue of F-16-3).
 *
 * `_stashAttemptContext` read ONLY `body.model`, but a Bedrock
 * `ConverseCommand` / `ConverseStreamCommand` input carries `modelId` — so a
 * `client.send()` rejection (403 `AccessDeniedException`, throttling, …)
 * emitted a failed row with `model='unknown'`, hiding which model the customer
 * was denied on. Two fixes are under test here:
 *
 * 1. `_stashAttemptContext` gained a trailing `modelHint` fallback, forwarded
 *    by the generic manual wrapper from the SAME hint it already derives for
 *    the pre-flight check (`reqBody.modelId` / the native-OpenRouter
 *    `chatRequest.model` envelope). `body.model`, when present, still wins.
 * 2. `_wrapBedrockConverseStream` now snapshots the session-global
 *    `_attempted_*` slots at wrap time and restores them before emitting a
 *    mid-stream failure (mirroring `_tapStreamUsageForOnEnd`), so a LATER
 *    same-session call that overwrote them cannot steal the stream's model —
 *    and the newer call's slots are handed back afterwards, since
 *    `_emitCallFailureLog` clears whatever it emitted.
 *
 * Drives the REAL instrumented `send` path through a fake
 * `@aws-sdk/client-bedrock-runtime` namespace (the `instrumentModules` pattern
 * from `tests/bedrockConverseStream.test.ts`), because the stash happens inside
 * the module-internal wrapper — a direct `__test__` call would not exercise it.
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

// A realistic Bedrock model id — the value that used to be lost as "unknown".
const MODEL_ID = "amazon.nova-lite-v1:0";
const INPUT = {
  modelId: MODEL_ID,
  messages: [{ role: "user", content: [{ text: "hi" }] }],
};

/** AWS SDK v3 ServiceException stand-in: status lives only on `$metadata`. */
function accessDenied(): any {
  const err: any = new Error(
    "User is not authorized to perform bedrock:InvokeModelWithResponseStream",
  );
  err.name = "AccessDeniedException";
  err.$metadata = { httpStatusCode: 403, requestId: "req-1", attempts: 1 };
  return err;
}

function deltaEvent(text: string): any {
  return { contentBlockDelta: { delta: { text }, contentBlockIndex: 0 } };
}

/**
 * Fake `@aws-sdk/client-bedrock-runtime` namespace. `send` is the ORIGINAL the
 * enforcer wraps; the Command classes carry the request `input`.
 */
function makeFakeBedrock(sendImpl: (command: any) => Promise<any>) {
  const sendCalls: any[] = [];
  class BedrockRuntimeClient {
    async send(command: any): Promise<any> {
      sendCalls.push(command);
      return await sendImpl(command);
    }
  }
  class ConverseCommand {
    input: any;
    constructor(input: any) {
      this.input = input;
    }
  }
  class ConverseStreamCommand {
    input: any;
    constructor(input: any) {
      this.input = input;
    }
  }
  return {
    ns: { BedrockRuntimeClient, ConverseCommand, ConverseStreamCommand },
    sendCalls,
  };
}

/** Instrument the fake module and capture every `tp.log` call. */
function arrange(
  sendImpl: (command: any) => Promise<any>,
  opts: { logThrows?: boolean } = {},
) {
  const { ns, sendCalls } = makeFakeBedrock(sendImpl);
  const client = init({
    apiKey: "tp_sk_test",
    deployment: "serverless",
    firewall: "dry_run",
    instrumentModules: { bedrock: ns },
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
// 1 — non-streaming ConverseCommand rejected at send()
// ═══════════════════════════════════════════════════════════════════
describe("bedrock ConverseCommand failure carries modelId", () => {
  test("403 AccessDeniedException → failed row model=modelId, provider=bedrock; error rethrown by identity", async () => {
    const boom = accessDenied();
    const { ns, logs } = arrange(async () => {
      throw boom;
    });

    const c = new ns.BedrockRuntimeClient();
    let caught: unknown;
    await session({ name: "wf" }, async () => {
      try {
        await (c as any).send(new ns.ConverseCommand(INPUT));
      } catch (e) {
        caught = e;
      }
    });
    // GOLDEN RULE: the provider's own error object, not a wrapper.
    expect(caught).toBe(boom);

    const rows = failureRows(logs);
    expect(rows).toHaveLength(1);
    expect(rows[0][4]).toBe(MODEL_ID); // was "unknown" before the fix
    expect(rows[0][5]).toBe("bedrock");
    // A failed call has no usage — the row stays unmeasured.
    expect(rows[0][6]).toBe(0);
    expect(rows[0][7]).toBe(0);
    const outcome = extrasOf(rows[0]).call_outcome;
    expect(outcome.status).toBe("failed");
    expect(outcome.error_kind).toBe("auth_error");
    expect(outcome.http_status).toBe(403);
  });

  test("GOLDEN RULE: tp.log itself throwing still surfaces only the provider error", async () => {
    const boom = accessDenied();
    const { ns, logs } = arrange(
      async () => {
        throw boom;
      },
      { logThrows: true },
    );

    const c = new ns.BedrockRuntimeClient();
    let caught: unknown;
    await session({ name: "wf" }, async () => {
      try {
        await (c as any).send(new ns.ConverseCommand(INPUT));
      } catch (e) {
        caught = e;
      }
    });
    expect(caught).toBe(boom);
    // The dispatch was attempted (and its throw swallowed) with the right model.
    const rows = failureRows(logs);
    expect(rows).toHaveLength(1);
    expect(rows[0][4]).toBe(MODEL_ID);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 2 — ConverseStreamCommand rejected at send() (stream never established)
// ═══════════════════════════════════════════════════════════════════
describe("bedrock ConverseStreamCommand establishment failure carries modelId", () => {
  test("403 at send() → failed row model=modelId; error rethrown by identity", async () => {
    const boom = accessDenied();
    const { ns, logs } = arrange(async () => {
      throw boom;
    });

    const c = new ns.BedrockRuntimeClient();
    let caught: unknown;
    await session({ name: "wf" }, async () => {
      try {
        await (c as any).send(new ns.ConverseStreamCommand(INPUT));
      } catch (e) {
        caught = e;
      }
    });
    expect(caught).toBe(boom);

    const rows = failureRows(logs);
    expect(rows).toHaveLength(1);
    expect(rows[0][4]).toBe(MODEL_ID);
    expect(rows[0][5]).toBe("bedrock");
    expect(extrasOf(rows[0]).call_outcome.http_status).toBe(403);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 3 — mid-stream failure (stream established, iterator throws partway)
// ═══════════════════════════════════════════════════════════════════
describe("bedrock ConverseStream mid-stream failure carries modelId", () => {
  test("iterator throws after one chunk → failed row model=modelId; error rethrown by identity", async () => {
    const boom = Object.assign(new Error("upstream 500"), {
      $metadata: { httpStatusCode: 500 },
    });
    const { ns, logs } = arrange(async () => {
      async function* gen() {
        yield deltaEvent("a");
        throw boom;
      }
      return { stream: gen(), $metadata: { httpStatusCode: 200 } };
    });

    const c = new ns.BedrockRuntimeClient();
    const seen: any[] = [];
    let caught: unknown;
    await session({ name: "wf" }, async () => {
      const out: any = await (c as any).send(new ns.ConverseStreamCommand(INPUT));
      try {
        for await (const chunk of out.stream) seen.push(chunk);
      } catch (e) {
        caught = e;
      }
    });
    expect(caught).toBe(boom);
    expect(seen).toHaveLength(1); // chunks delivered up to the failure

    const rows = failureRows(logs);
    expect(rows).toHaveLength(1);
    expect(rows[0][4]).toBe(MODEL_ID);
    expect(rows[0][5]).toBe("bedrock");
    const outcome = extrasOf(rows[0]).call_outcome;
    expect(outcome.status).toBe("failed");
    expect(outcome.error_kind).toBe("server_error");
    expect(outcome.http_status).toBe(500);
    // No SUCCESS usage row — no terminal metadata event was ever seen.
    expect(logs.filter((r) => (r[6] as number) > 0 || (r[7] as number) > 0)).toHaveLength(0);
  });

  test("GOLDEN RULE: tp.log throwing on the mid-stream failure path never reaches the consumer", async () => {
    const boom = Object.assign(new Error("upstream 500"), {
      $metadata: { httpStatusCode: 500 },
    });
    const { ns, logs } = arrange(
      async () => {
        async function* gen() {
          yield deltaEvent("a");
          throw boom;
        }
        return { stream: gen(), $metadata: { httpStatusCode: 200 } };
      },
      { logThrows: true },
    );

    const c = new ns.BedrockRuntimeClient();
    let caught: unknown;
    await session({ name: "wf" }, async () => {
      const out: any = await (c as any).send(new ns.ConverseStreamCommand(INPUT));
      try {
        for await (const _chunk of out.stream) void _chunk;
      } catch (e) {
        caught = e;
      }
    });
    expect(caught).toBe(boom);
    const rows = failureRows(logs);
    expect(rows).toHaveLength(1);
    expect(rows[0][4]).toBe(MODEL_ID);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 4 — snapshot/restore: a newer same-session call must not steal the model
// ═══════════════════════════════════════════════════════════════════
describe("bedrock ConverseStream attempt-context snapshot/restore", () => {
  test("a later call overwrote the slots → failure row keeps the bedrock modelId, and the newer call's slots are handed back", async () => {
    const boom = Object.assign(new Error("upstream 500"), {
      $metadata: { httpStatusCode: 500 },
    });
    const { ns, logs } = arrange(async () => {
      async function* gen() {
        yield deltaEvent("a");
        throw boom;
      }
      return { stream: gen(), $metadata: { httpStatusCode: 200 } };
    });

    const c = new ns.BedrockRuntimeClient();
    let sess: any;
    let caught: unknown;
    await session({ name: "wf" }, async () => {
      sess = getCurrentSession();
      // Bedrock stream established → this call's context is stashed and the
      // wrapper snapshotted it.
      const out: any = await (c as any).send(new ns.ConverseStreamCommand(INPUT));
      expect(sess._attempted_model).toBe(MODEL_ID);

      // A SECOND in-flight call on the same session overwrites the
      // session-global slots (exactly what the real wrapper does for it)
      // BEFORE the bedrock stream fails.
      _stashAttemptContext(sess, "openai", [{ model: "gpt-4o-mini" }], "chat");
      expect(sess._attempted_model).toBe("gpt-4o-mini");

      try {
        for await (const _chunk of out.stream) void _chunk;
      } catch (e) {
        caught = e;
      }
    });
    expect(caught).toBe(boom);

    // The stream's OWN model, not the newer call's.
    const rows = failureRows(logs);
    expect(rows).toHaveLength(1);
    expect(rows[0][4]).toBe(MODEL_ID);
    expect(rows[0][5]).toBe("bedrock");

    // …and the newer call's context is back in place afterwards (the emit
    // clears the slots, so without the hand-back its own failure row would
    // degrade to model="unknown").
    expect(sess._attempted_model).toBe("gpt-4o-mini");
    expect(sess._attempted_provider).toBe("openai");
    expect(sess._attempted_operation).toBe("chat");
  });

  test("no overwrite → the slots are left cleared by the emit (no phantom restore)", async () => {
    const boom = Object.assign(new Error("upstream 500"), {
      $metadata: { httpStatusCode: 500 },
    });
    const { ns, logs } = arrange(async () => {
      async function* gen() {
        yield deltaEvent("a");
        throw boom;
      }
      return { stream: gen(), $metadata: { httpStatusCode: 200 } };
    });

    const c = new ns.BedrockRuntimeClient();
    let sess: any;
    await session({ name: "wf" }, async () => {
      sess = getCurrentSession();
      const out: any = await (c as any).send(new ns.ConverseStreamCommand(INPUT));
      try {
        for await (const _chunk of out.stream) void _chunk;
      } catch {
        /* expected */
      }
    });
    expect(failureRows(logs)[0][4]).toBe(MODEL_ID);
    // Nothing newer to hand back — the post-emit state stays clean.
    expect(sess._attempted_model).toBeUndefined();
    expect(sess._attempted_provider).toBeUndefined();
  });
});

// ═══════════════════════════════════════════════════════════════════
// 5 — _stashAttemptContext modelHint precedence (unit)
// ═══════════════════════════════════════════════════════════════════
describe("_stashAttemptContext modelHint precedence", () => {
  function stash(body: any, hint?: string): any {
    const s: any = {};
    _stashAttemptContext(s, "bedrock", [body], "chat", undefined, undefined, hint);
    return s;
  }

  test("a real top-level body.model WINS over the hint", () => {
    expect(stash({ model: "gpt-4o-mini", modelId: MODEL_ID }, MODEL_ID)._attempted_model)
      .toBe("gpt-4o-mini");
  });

  test("no body.model → the hint is used", () => {
    expect(stash({ modelId: MODEL_ID }, MODEL_ID)._attempted_model).toBe(MODEL_ID);
  });

  test("an EMPTY body.model falls through to the hint (was the '' → undefined hole)", () => {
    expect(stash({ model: "" }, MODEL_ID)._attempted_model).toBe(MODEL_ID);
  });

  test("no model and no hint → undefined (unchanged: _emitCallFailureLog degrades to 'unknown')", () => {
    expect(stash({ messages: [] })._attempted_model).toBeUndefined();
  });

  test("an empty-string hint is ignored", () => {
    expect(stash({ messages: [] }, "")._attempted_model).toBeUndefined();
  });

  test("the helper never re-derives modelId itself — a bedrock body with NO hint is the pre-fix result", () => {
    // Pins the contract: the hint is caller-derived (the wrapper's own
    // fail-open read), so this call site's `undefined` is exactly the
    // model="unknown" row the fix removes at the wrapper, not in here.
    const s: any = {};
    _stashAttemptContext(s, "bedrock", [{ modelId: MODEL_ID }], "chat");
    expect(s._attempted_model).toBeUndefined();
  });

  test("REGRESSION: a non-bedrock body with a model is unaffected when no hint is passed", () => {
    const s: any = {};
    _stashAttemptContext(s, "openai", [{ model: "gpt-4o-mini" }], "chat");
    expect(s._attempted_model).toBe("gpt-4o-mini");
    expect(s._attempted_provider).toBe("openai");
  });
});
