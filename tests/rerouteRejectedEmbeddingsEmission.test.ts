/**
 * N5 — `_makeFrameworkEmbeddingWrapper`'s SUCCESS path (LangChain +
 * LlamaIndex embeddings) must ship the `reroute_rejected` observation
 * `_applyReroute` mints when a live ENFORCE REROUTE directive hits this
 * body-less seam. Same class of gap as the LlamaIndex chat emitter covered
 * in tests/rerouteRejectedSuccessEmission.test.ts — see that file's header
 * for the full bug narrative — applied to the embeddings wrapper, whose
 * inline `tp.log` call shipped no `observations` key at all pre-fix.
 *
 * This also closes the orphan `REROUTE_DIRECTIVE_ISSUED` §8.2 flagged on
 * `embeddings_llama_index_python`'s Node twin: the rejection IS minted (as
 * `unappliable_call_shape`) and now ships on the success row, so every
 * directive resolves to a REJECTED instead of dangling unresolved.
 *
 * Harness mirrors tests/frameworkEmbeddingFailureLog.test.ts (direct
 * `_makeFrameworkEmbeddingWrapper` invocation, `getCurrentSession` mocked to
 * a plain fake session object — the obs-key scope `_withCallObsScope` manages
 * internally is a SEPARATE AsyncLocalStorage from the session one, so mocking
 * the session is orthogonal to a real keyed drain happening) combined with
 * tests/rerouteUnappliableShape.test.ts's real-client / stubbed-`check`
 * State-B setup (`deployment: "serverless"` + `resetPack()`).
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import * as contextModule from "../src/context";
import * as state from "../src/state";
import { resetPack, setClient } from "../src/state";
import { TokenPolice } from "../src/client";
import { __test__ as enforcerTest } from "../src/enforcer";
import { stashLocalDecision, claimLocalDecision } from "../src/localDecisionStore";

function makeClient(firewall: "enforce" | "dry_run" | "off" = "enforce"): TokenPolice {
  const client = new TokenPolice({
    apiKey: "tp_sk_test_rr_embed_success",
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

function extrasOf(logArgs: any[]): any {
  return logArgs[logArgs.length - 1];
}

function makeSession() {
  return {
    userId: "u",
    paidPlan: "free",
    workflowName: "n5",
    sessionId: "s",
    metadata: {},
    inLangchain: false,
    inLlamaIndex: false,
    enterLangchain: vi.fn(),
    exitLangchain: vi.fn(),
    enterLlamaIndex: vi.fn(),
    exitLlamaIndex: vi.fn(),
    nextSpanOrder: () => 0,
    traceId: "trace-n5",
    rootSpanId: "root-n5",
    _pendingCompositions: {},
  };
}

const rerouteDirective = (
  model = "text-embedding-3-large",
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
});

describe("N5 — framework embeddings success ships the reroute_rejected observation", () => {
  it("langchain embeddings arm", async () => {
    const client = makeClient("enforce");
    stubCheck(client, rerouteDirective());
    const logged = captureLog(client);

    const session = makeSession();
    vi.spyOn(contextModule, "getCurrentSession").mockReturnValue(session as any);

    const original = vi.fn().mockResolvedValue([[0.1, 0.2]]);
    const wrapped = enforcerTest._makeFrameworkEmbeddingWrapper(
      original,
      "langchain",
      "embedDocuments",
      "openai",
    ) as (this: any, ...args: any[]) => Promise<any>;

    const self = { model: "text-embedding-3-small" };
    const result = await wrapped.call(self, ["hello"]);

    expect(result).toEqual([[0.1, 0.2]]);
    expect(original).toHaveBeenCalledTimes(1);
    expect(logged.length).toBe(1);
    const extras = extrasOf(logged[0]);
    expect(extras.operation).toBe("embedding");
    expect(extras.call_outcome?.status).not.toBe("failed");
    expect(extras.observations).toHaveLength(1);
    expect(extras.observations[0].outcome).toBe("reroute_rejected");
    expect(extras.observations[0].rejection_reason).toBe("unappliable_call_shape");
    expect(extras.observations[0].rule_id).toBe("rule_rr");
    expect(logged[0][5]).toBe("langchain"); // provider is positional arg 5
  });

  it("llamaindex embeddings arm", async () => {
    const client = makeClient("enforce");
    stubCheck(client, rerouteDirective("text-embedding-ada-002"));
    const logged = captureLog(client);

    const session = makeSession();
    vi.spyOn(contextModule, "getCurrentSession").mockReturnValue(session as any);

    const original = vi.fn().mockResolvedValue([0.1, 0.2]);
    const wrapped = enforcerTest._makeFrameworkEmbeddingWrapper(
      original,
      "llamaindex",
      "getTextEmbedding",
    ) as (this: any, ...args: any[]) => Promise<any>;

    const self = { modelName: "text-embedding-ada-002" };
    const result = await wrapped.call(self, "hello");

    expect(result).toEqual([0.1, 0.2]);
    expect(logged.length).toBe(1);
    const extras = extrasOf(logged[0]);
    expect(extras.operation).toBe("embedding");
    expect(extras.observations).toHaveLength(1);
    expect(extras.observations[0].outcome).toBe("reroute_rejected");
    expect(extras.observations[0].rejection_reason).toBe("unappliable_call_shape");
    expect(logged[0][5]).toBe("llamaindex");
  });

  it("failure path unchanged: a provider raise still emits exactly one row, one copy of the observation", async () => {
    const client = makeClient("enforce");
    stubCheck(client, rerouteDirective());
    const logged = captureLog(client);

    const session = makeSession();
    vi.spyOn(contextModule, "getCurrentSession").mockReturnValue(session as any);

    const providerErr = Object.assign(new Error("OpenAI 500"), { status: 500 });
    const original = vi.fn().mockRejectedValue(providerErr);
    const wrapped = enforcerTest._makeFrameworkEmbeddingWrapper(
      original,
      "langchain",
      "embedDocuments",
      "openai",
    ) as (this: any, ...args: any[]) => Promise<any>;

    await expect(wrapped.call({ model: "text-embedding-3-small" }, ["hello"])).rejects.toBe(
      providerErr,
    );

    expect(logged.length).toBe(1);
    const extras = extrasOf(logged[0]);
    expect(extras.call_outcome?.status).toBe("failed");
    expect(extras.observations).toHaveLength(1);
    expect(extras.observations[0].outcome).toBe("reroute_rejected");
  });

  it("local_decision fence: an untagged decision ships nowhere and survives the call", async () => {
    const client = makeClient("enforce");
    stubCheck(client, rerouteDirective());
    const logged = captureLog(client);

    const session = makeSession();
    vi.spyOn(contextModule, "getCurrentSession").mockReturnValue(session as any);
    const untagged = {
      outcome: "rerouted",
      rule_id: "someone_elses_rule",
      reroute: { from: { model: "x" }, to: { model: "y" } },
    };
    stashLocalDecision(session as any, untagged, null);

    const original = vi.fn().mockResolvedValue([[0.1, 0.2]]);
    const wrapped = enforcerTest._makeFrameworkEmbeddingWrapper(
      original,
      "langchain",
      "embedDocuments",
      "openai",
    ) as (this: any, ...args: any[]) => Promise<any>;

    await wrapped.call({ model: "text-embedding-3-small" }, ["hello"]);

    expect(logged.length).toBe(1);
    const extras = extrasOf(logged[0]);
    expect(extras.observations).toHaveLength(1);
    expect(extras.local_decision).toBeUndefined();

    const claimed = claimLocalDecision(session as any, "any-later-key");
    expect(claimed).toEqual(untagged);
  });

  it("no-op reroute (dry_run mode): zero observations shipped, zero minted", async () => {
    const client = makeClient("enforce");
    stubCheck(client, {
      status: "allowed",
      reroute: { mode: "dry_run", model: "text-embedding-3-large", provider: "openai", rule_id: "rule_rr" },
    });
    const logged = captureLog(client);

    const session = makeSession();
    vi.spyOn(contextModule, "getCurrentSession").mockReturnValue(session as any);

    const original = vi.fn().mockResolvedValue([[0.1, 0.2]]);
    const wrapped = enforcerTest._makeFrameworkEmbeddingWrapper(
      original,
      "langchain",
      "embedDocuments",
      "openai",
    ) as (this: any, ...args: any[]) => Promise<any>;

    await wrapped.call({ model: "text-embedding-3-small" }, ["hello"]);

    expect(logged.length).toBe(1);
    expect(extrasOf(logged[0]).observations).toBeUndefined();
    expect(state.drainObservations()).toHaveLength(0);
  });
});
