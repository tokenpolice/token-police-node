/**
 * N4 — a failed embedding call must not ship a CHAT usage shape.
 *
 * `_emitCallFailureLog` put no `usage` on its `extras`, so `client.log`
 * synthesized `{ shape: "openai_compatible_chat", ... }` unconditionally and
 * every failed embedding row landed with `usage_shape='openai_compatible_chat'`
 * on an embedding span (5/5 failed embedding rows in the 2026-07-27 real-apps
 * run), while successful siblings carried `voyage_embed` / `openai_embeddings`.
 *
 * Asserts:
 * - native embedding failure (voyage) → usage.shape === "voyage_embed"
 * - framework embedding failure → usage.shape === "openai_embeddings"
 * - counts stay zero (a failed call has no usage; the row stays unmeasured)
 * - a failed CHAT call now resolves its provider's chat shape too
 * - the original exception still propagates (GOLDEN RULE)
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import * as contextModule from "../src/context";
import { resetPack, setClient } from "../src/state";
import { TokenPolice } from "../src/client";
import { protect, uninstrument, __test__ as enforcerTest } from "../src/enforcer";

const { _instrumentVoyage, _makeFrameworkEmbeddingWrapper, _setInstrumented } =
  enforcerTest as any;

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
    workflowName: "n4",
    sessionId: "s",
    metadata: {},
    inLangchain: false,
    inLlamaIndex: false,
    enterLangchain: vi.fn(),
    exitLangchain: vi.fn(),
    enterLlamaIndex: vi.fn(),
    exitLlamaIndex: vi.fn(),
    nextSpanOrder: () => 0,
    traceId: "trace-n4",
    rootSpanId: "root-n4",
    _pendingCompositions: {},
  };
}

/** Install the spies + a mocked session; returns the captured `tp.log` args. */
function arrange(): any[][] {
  const client = makeClient();
  const logs: any[][] = [];
  vi.spyOn(client, "log").mockImplementation((...args: any[]) => {
    logs.push(args);
  });
  vi.spyOn(client, "check").mockResolvedValue({ status: "allowed" } as never);
  vi.spyOn(contextModule, "getCurrentSession").mockReturnValue(makeSession() as any);
  return logs;
}

/** `tp.log`'s trailing options arg — where `usage` lives. */
function extrasOf(logArgs: any[]): any {
  return logArgs[logArgs.length - 1];
}

describe("N4 failed embedding usage shape", () => {
  beforeEach(() => {
    resetPack();
    vi.restoreAllMocks();
  });

  afterEach(() => {
    // _wrapMethod dedupes on its own `_originals` key — restore so a later
    // test can re-instrument the same fake module cleanly.
    try {
      _setInstrumented(true);
      uninstrument();
    } catch {
      /* ignore */
    }
  });

  it("native voyage embedding failure → voyage_embed shape with zero counts", async () => {
    const logs = arrange();

    const providerErr = Object.assign(new Error("rate limit"), { status: 429 });
    class VoyageAIClient {
      async embed(_req: any): Promise<any> {
        throw providerErr;
      }
      async multimodalEmbed(_req: any): Promise<any> {
        return {};
      }
    }
    _instrumentVoyage({ VoyageAIClient });

    await expect(new VoyageAIClient().embed({ model: "voyage-3" })).rejects.toBe(
      providerErr,
    );

    expect(logs.length).toBe(1);
    const extras = extrasOf(logs[0]);
    expect(extras?.operation).toBe("embedding");
    expect(extras?.call_outcome?.status).toBe("failed");
    // The fix: an embed shape, not the client's openai_compatible_chat synth.
    expect(extras?.usage?.shape).toBe("voyage_embed");
    // A failed call has no usage — zero units keep the row unmeasured.
    expect(extras?.usage?.raw).toEqual({ prompt_tokens: 0, total_tokens: 0 });
    expect(logs[0][6]).toBe(0);
    expect(logs[0][7]).toBe(0);
  });

  it("framework embedding failure → openai_embeddings shape (unmapped provider)", async () => {
    const logs = arrange();

    const providerErr = Object.assign(new Error("OpenAI 500"), { status: 500 });
    const original = vi.fn().mockRejectedValue(providerErr);
    const wrapped = _makeFrameworkEmbeddingWrapper(
      original,
      "langchain",
      "embedDocuments",
      "openai",
    ) as (this: any, ...args: any[]) => Promise<any>;

    await expect(
      wrapped.call({ model: "text-embedding-3-small" }, ["hello"]),
    ).rejects.toBe(providerErr);

    expect(logs.length).toBe(1);
    const extras = extrasOf(logs[0]);
    expect(extras?.operation).toBe("embedding");
    // `_stashAttemptContext` stashes the framework as the provider; it is not
    // in the embed table, so it resolves to the same default its successful
    // sibling logs.
    expect(extras?.usage?.shape).toBe("openai_embeddings");
    expect(extras?.usage?.raw).toEqual({ prompt_tokens: 0, total_tokens: 0 });
  });

  it("failed CHAT call now resolves its provider's chat shape too", () => {
    const logs = arrange();

    const boom = Object.assign(new Error("upstream 500"), { status: 500 });
    class SyncChatClient {
      create(..._args: any[]): any {
        throw boom;
      }
    }
    // Sync escape hatch: no registry `operation`, so the failure row keeps the
    // "chat" default. resolves a shape for chat too — the module name
    // detects as "openai", so the row now carries the same `openai_chat` its
    // successful sibling would, not the client's openai_compatible_chat synth.
    protect("openai-n4-chat", ["prototype"], "create", false, {
      module: SyncChatClient,
    });

    expect(() =>
      new SyncChatClient().create({ model: "gpt-4.1-mini", messages: [] }),
    ).toThrow(boom);

    expect(logs.length).toBe(1);
    const extras = extrasOf(logs[0]);
    expect(extras?.operation).toBe("chat");
    expect(extras?.usage?.shape).toBe("openai_chat");
    expect(extras?.usage?.raw).toEqual({ prompt_tokens: 0, total_tokens: 0 });
  });

  it("a usage-shape failure cannot mask the customer exception", async () => {
    const client = makeClient();
    vi.spyOn(client, "log").mockImplementation(() => {
      throw new Error("telemetry noise");
    });
    vi.spyOn(contextModule, "getCurrentSession").mockReturnValue(makeSession() as any);

    const providerErr = new Error("AccessDenied");
    const wrapped = _makeFrameworkEmbeddingWrapper(
      vi.fn().mockRejectedValue(providerErr),
      "llamaindex",
      "getTextEmbedding",
    ) as (this: any, ...args: any[]) => Promise<any>;

    await expect(wrapped.call({ model: "m" }, "x")).rejects.toBe(providerErr);
  });
});
