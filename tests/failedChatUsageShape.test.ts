/**
 * A failed CHAT call must not ship an OpenAI usage shape.
 *
 * N4 fixed this for embeddings only: every other operation left `usage`
 * undefined on `_emitCallFailureLog`'s extras, so `client.log` synthesized
 * `{ shape: "openai_compatible_chat", ... }` and EVERY failed chat row landed
 * with `usage_shape='openai_compatible_chat'` — cohere rows whose successful
 * siblings read `cohere_chat`, google rows that should read `google_genai`,
 * huggingface `huggingface_chat`, cerebras `cerebras_chat` (verification run
 * #7's top finding).
 *
 * Asserts:
 * - per-provider chat resolution (cohere / huggingface / cerebras /
 * openai_responses) with zero counts
 * - an unmapped/empty provider still lands on `openai_compatible_chat`
 * - the wire-key gate: a host-remapped Anthropic client (serving
 * "minimax") logs `anthropic_messages`, and "minimax" without a wire key
 * keeps the default
 * - a wrapper-stashed shape (modality/registry override) wins over the
 * operation-based resolution
 * - a shape-resolution failure degrades to the client synth and can never
 * mask the customer's exception (GOLDEN RULE)
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import * as contextModule from "../src/context";
import { resetPack, setClient } from "../src/state";
import { TokenPolice } from "../src/client";
import { protect, uninstrument, __test__ as enforcerTest } from "../src/enforcer";

const { _stashAttemptContext, _emitCallFailureLog } = enforcerTest as any;

function makeClient() {
  // Direct construction (not init()) so no SSE stream starts; an unreachable
  // 5xxxx baseUrl makes any background tp.log()/tp.check() fail-open silently.
  const client = new TokenPolice({
    apiKey: "tp_sk_test123",
    baseUrl: "http://localhost:59999",
    timeout: 0.1,
    firewall: "off",
    deployment: "daemon",
  });
  setClient(client);
  return client;
}

function makeSession() {
  return {
    userId: "u",
    paidPlan: "free",
    workflowName: "f1",
    sessionId: "s",
    metadata: {},
    inLangchain: false,
    inLlamaIndex: false,
    enterLangchain: vi.fn(),
    exitLangchain: vi.fn(),
    enterLlamaIndex: vi.fn(),
    exitLlamaIndex: vi.fn(),
    nextSpanOrder: () => 0,
    traceId: "trace-f1",
    rootSpanId: "root-f1",
    _pendingCompositions: {},
  } as any;
}

/** Install the spies + a mocked session; returns the captured `tp.log` args. */
function arrange(session?: any): { client: any; logs: any[][]; session: any } {
  const client = makeClient();
  const logs: any[][] = [];
  vi.spyOn(client, "log").mockImplementation((...args: any[]) => {
    logs.push(args);
  });
  vi.spyOn(client, "check").mockResolvedValue({ status: "allowed" } as never);
  const s = session ?? makeSession();
  vi.spyOn(contextModule, "getCurrentSession").mockReturnValue(s);
  return { client, logs, session: s };
}

/** `tp.log`'s trailing options arg — where `usage` lives. */
function extrasOf(logArgs: any[]): any {
  return logArgs[logArgs.length - 1];
}

/**
 * Drives a failed SYNC chat call through the real `protect()` wrapper for
 * `provider`, and returns the extras of the one emitted row. The sync escape
 * hatch is the smallest real path that stashes attempt context and funnels
 * through `_emitCallFailureLog`.
 */
function failChatVia(provider: string, tag: string): any {
  const { logs } = arrange();
  const boom = Object.assign(new Error("upstream 500"), { status: 500 });
  class SyncChatClient {
    create(..._args: any[]): any {
      throw boom;
    }
  }
  protect(`f1-${tag}`, ["prototype"], "create", false, {
    module: SyncChatClient,
    provider,
  });

  expect(() => new SyncChatClient().create({ model: "m-1", messages: [] })).toThrow(
    boom,
  );
  expect(logs.length).toBe(1);
  return extrasOf(logs[0]);
}

/** Stash an arbitrary attempt context and fire the failure funnel directly. */
function emitWith(
  provider: string,
  operation: string,
  shape?: string | null,
  wireKey?: string | null,
): any {
  const { client, logs, session } = arrange();
  _stashAttemptContext(session, provider, [{ model: "m-1" }], operation, shape, wireKey);
  session._call_outcome = { status: "failed", error_type: "APIError" };
  _emitCallFailureLog(client, session);
  expect(logs.length).toBe(1);
  return extrasOf(logs[0]);
}

describe("Failed chat usage shape", () => {
  beforeEach(() => {
    resetPack();
    vi.restoreAllMocks();
  });

  afterEach(() => {
    // _wrapMethod dedupes on its own `_originals` key — restore so a later
    // test can re-instrument the same fake module cleanly.
    try {
      (enforcerTest as any)._setInstrumented(true);
      uninstrument();
    } catch {
      /* ignore */
    }
  });

  it("cohere chat failure → cohere_chat with zero counts", () => {
    const extras = failChatVia("cohere", "cohere");
    expect(extras?.operation).toBe("chat");
    expect(extras?.call_outcome?.status).toBe("failed");
    expect(extras?.usage?.shape).toBe("cohere_chat");
    // A failed call has no usage — zero units keep the row unmeasured.
    expect(extras?.usage?.raw).toEqual({ prompt_tokens: 0, total_tokens: 0 });
  });

  it("huggingface chat failure → huggingface_chat", () => {
    expect(failChatVia("huggingface", "hf")?.usage?.shape).toBe("huggingface_chat");
  });

  it("cerebras chat failure → cerebras_chat", () => {
    expect(failChatVia("cerebras", "cerebras")?.usage?.shape).toBe("cerebras_chat");
  });

  it("openai_responses chat failure → openai_responses", () => {
    expect(failChatVia("openai_responses", "responses")?.usage?.shape).toBe(
      "openai_responses",
    );
  });

  it("empty provider → openai_compatible_chat (unchanged default)", () => {
    expect(failChatVia("", "unknown")?.usage?.shape).toBe("openai_compatible_chat");
  });

  it("host-remapped client: serving minimax + anthropic wire key → anthropic_messages", () => {
    // An Anthropic SDK pointed at api.minimax.io stashes serving provider
    // "minimax" but its successful siblings log `anthropic_messages`.
    const extras = emitWith("minimax", "chat", undefined, "anthropic");
    expect(extras?.usage?.shape).toBe("anthropic_messages");
    expect(extras?.usage?.raw).toEqual({ prompt_tokens: 0, total_tokens: 0 });
  });

  it("serving minimax with NO wire key keeps openai_compatible_chat", () => {
    expect(emitWith("minimax", "chat")?.usage?.shape).toBe("openai_compatible_chat");
  });

  it("an in-table provider is never overridden by the wire key", () => {
    // The gate only engages when the serving slug fell through to the default.
    expect(emitWith("cohere", "chat", undefined, "anthropic")?.usage?.shape).toBe(
      "cohere_chat",
    );
  });

  it("a stashed wrapper shape wins over the operation-based resolution", () => {
    const extras = emitWith("openai", "image_gen", "openai_images");
    expect(extras?.operation).toBe("image_gen");
    expect(extras?.usage?.shape).toBe("openai_images");
  });

  it("a shape-resolution failure degrades to the client synth, error still propagates", () => {
    // Force the resolver block to throw from INSIDE its own try: a session
    // whose `_attempted_shape` read raises (the stash write stays a no-op so
    // the rest of the attempt context still lands).
    const session = makeSession();
    Object.defineProperty(session, "_attempted_shape", {
      configurable: true,
      get(): string {
        throw new Error("shape probe exploded");
      },
      set(_v: unknown) {
        /* no-op */
      },
    });
    const { logs } = arrange(session);

    const boom = Object.assign(new Error("upstream 500"), { status: 500 });
    class SyncChatClient {
      create(..._args: any[]): any {
        throw boom;
      }
    }
    protect("f1-failopen", ["prototype"], "create", false, {
      module: SyncChatClient,
      provider: "cohere",
    });

    // GOLDEN RULE: the customer's own error, unchanged.
    expect(() => new SyncChatClient().create({ model: "m-1" })).toThrow(boom);

    expect(logs.length).toBe(1);
    const extras = extrasOf(logs[0]);
    expect(extras?.usage).toBeUndefined();
  });
});
