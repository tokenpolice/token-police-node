/**
 * Failed framework embedding must emit a failed embedding row.
 *
 * `_makeFrameworkEmbeddingWrapper` (LangChain + LlamaIndex) previously only
 * logged on success. A provider raise re-threw (golden rule OK for the call)
 * but shipped zero embedding rows. Assert: one failed /log with
 * operation=embedding + classified call_outcome, exception identity preserved,
 * success path unchanged, logging failure cannot mask the customer error.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import * as contextModule from "../src/context";
import { resetPack, setClient } from "../src/state";
import { TokenPolice } from "../src/client";
import { __test__ as enforcerTest } from "../src/enforcer";

function makeClient() {
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
    workflowName: "b21",
    sessionId: "s",
    metadata: {},
    inLangchain: false,
    inLlamaIndex: false,
    enterLangchain: vi.fn(),
    exitLangchain: vi.fn(),
    enterLlamaIndex: vi.fn(),
    exitLlamaIndex: vi.fn(),
    nextSpanOrder: () => 0,
    traceId: "trace-b21",
    rootSpanId: "root-b21",
    _pendingCompositions: {},
  };
}

describe("Framework embedding failure log", () => {
  beforeEach(() => {
    resetPack();
    vi.restoreAllMocks();
  });

  it("provider raise → one failed embedding log + rethrow same error (langchain)", async () => {
    const client = makeClient();
    const logs: any[] = [];
    vi.spyOn(client, "log").mockImplementation((...args: any[]) => {
      logs.push(args);
    });

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

    const self = { model: "text-embedding-3-small" };
    await expect(wrapped.call(self, ["hello"])).rejects.toBe(providerErr);

    expect(original).toHaveBeenCalledTimes(1);
    expect(logs.length).toBe(1);
    const extra = logs[0][logs[0].length - 1];
    expect(extra?.operation).toBe("embedding");
    expect(extra?.call_outcome?.status).toBe("failed");
    expect(extra?.call_outcome?.http_status).toBe(500);
    expect(extra?.call_outcome?.error_kind).toBe("server_error");
    // model is positional arg 4 in tp.log signature
    expect(logs[0][4]).toBe("text-embedding-3-small");
    // provider is positional arg 5
    expect(logs[0][5]).toBe("langchain");
    // zero tokens on failure path
    expect(logs[0][6]).toBe(0);
    expect(logs[0][7]).toBe(0);
    // guard was entered and exited even on failure
    expect(session.enterLangchain).toHaveBeenCalled();
    expect(session.exitLangchain).toHaveBeenCalled();
  });

  it("provider raise → failed embedding log (llamaindex)", async () => {
    const client = makeClient();
    const logs: any[] = [];
    vi.spyOn(client, "log").mockImplementation((...args: any[]) => {
      logs.push(args);
    });

    const session = makeSession();
    vi.spyOn(contextModule, "getCurrentSession").mockReturnValue(session as any);

    const providerErr = Object.assign(new Error("rate limit"), { status: 429 });
    const original = vi.fn().mockRejectedValue(providerErr);
    const wrapped = enforcerTest._makeFrameworkEmbeddingWrapper(
      original,
      "llamaindex",
      "getTextEmbedding",
    ) as (this: any, ...args: any[]) => Promise<any>;

    const self = { modelName: "text-embedding-ada-002" };
    await expect(wrapped.call(self, "hello")).rejects.toBe(providerErr);

    expect(logs.length).toBe(1);
    const extra = logs[0][logs[0].length - 1];
    expect(extra?.operation).toBe("embedding");
    expect(extra?.call_outcome?.status).toBe("failed");
    expect(extra?.call_outcome?.http_status).toBe(429);
    expect(extra?.call_outcome?.error_kind).toBe("rate_limited");
    expect(logs[0][5]).toBe("llamaindex");
    expect(session.enterLlamaIndex).toHaveBeenCalled();
    expect(session.exitLlamaIndex).toHaveBeenCalled();
  });

  it("success path still logs approx tokens without failed outcome", async () => {
    const client = makeClient();
    const logs: any[] = [];
    vi.spyOn(client, "log").mockImplementation((...args: any[]) => {
      logs.push(args);
    });

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
    const result = await wrapped.call(self, ["hello world"]);
    expect(result).toEqual([[0.1, 0.2]]);
    expect(logs.length).toBe(1);
    const extra = logs[0][logs[0].length - 1];
    expect(extra?.operation).toBe("embedding");
    expect(extra?.call_outcome?.status).not.toBe("failed");
    // success path: non-zero approx input tokens (positional arg 6)
    expect(logs[0][6]).toBeGreaterThan(0);
  });

  it("logging failure cannot mask the customer exception", async () => {
    const client = makeClient();
    vi.spyOn(client, "log").mockImplementation(() => {
      throw new Error("telemetry noise");
    });

    const session = makeSession();
    vi.spyOn(contextModule, "getCurrentSession").mockReturnValue(session as any);

    const providerErr = new Error("AccessDenied");
    const original = vi.fn().mockRejectedValue(providerErr);
    const wrapped = enforcerTest._makeFrameworkEmbeddingWrapper(
      original,
      "langchain",
      "embedQuery",
    ) as (this: any, ...args: any[]) => Promise<any>;

    await expect(wrapped.call({ model: "m" }, "x")).rejects.toBe(providerErr);
  });
});

/**
 * LlamaIndex embedding rows must attribute the underlying vendor.
 *
 * `resolvedOriginal` was hardcoded `undefined` for framework === "llamaindex",
 * so every LlamaIndex embedding row shipped model_extras={framework} with no
 * original_provider and the server could not resolve the deployer's price
 * table. Each LlamaIndex embedding class lives in its own @llamaindex/*
 * subpackage, so the slug is known at patch time and threaded via the closure.
 */
describe("LlamaIndex embedding original_provider", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    resetPack();
  });

  async function runEmbedding(
    framework: "langchain" | "llamaindex",
    originalProvider?: string,
    self: any = { model: "text-embedding-3-small" },
  ) {
    const client = makeClient();
    const logs: any[] = [];
    vi.spyOn(client, "log").mockImplementation((...args: any[]) => {
      logs.push(args);
    });
    const session = makeSession();
    vi.spyOn(contextModule, "getCurrentSession").mockReturnValue(session as any);

    const original = vi.fn().mockResolvedValue([0.1, 0.2]);
    const wrapped = enforcerTest._makeFrameworkEmbeddingWrapper(
      original,
      framework,
      "getTextEmbedding",
      originalProvider,
    ) as (this: any, ...args: any[]) => Promise<any>;

    await wrapped.call(self, "hello");
    expect(logs.length).toBe(1);
    return logs[0][logs[0].length - 1]?.model_extras;
  }

  it("threads the patch-time slug into model_extras", async () => {
    const extras = await runEmbedding("llamaindex", "openai");
    expect(extras).toEqual({ framework: "llamaindex", original_provider: "openai" });
  });

  it("every mapped subpackage slug is carried verbatim", async () => {
    for (const slug of ["openai", "cohere", "mistral", "gemini", "huggingface"]) {
      const extras = await runEmbedding("llamaindex", slug);
      expect(extras?.original_provider).toBe(slug);
    }
  });

  it("no slug (unknown subpackage) → exactly today's extras", async () => {
    const extras = await runEmbedding("llamaindex", undefined);
    expect(extras).toEqual({ framework: "llamaindex" });
  });

  it("LangChain behavior is unchanged: closure slug still wins", async () => {
    const extras = await runEmbedding("langchain", "openai");
    expect(extras).toEqual({ framework: "langchain", original_provider: "openai" });
  });

  it("LangChain base-class path still derives from lc_namespace at call time", async () => {
    const extras = await runEmbedding("langchain", undefined, {
      model: "text-embedding-3-small",
      lc_namespace: ["langchain", "embeddings", "openai"],
    });
    expect(extras).toEqual({ framework: "langchain", original_provider: "openai" });
  });
});

/**
 * Wiring — the registration path itself, not just the wrapper.
 *
 * Both new tests above hand the slug straight to _makeFrameworkEmbeddingWrapper,
 * so dropping the 3rd argument at either registration site would stay green.
 * These drive _instrumentLlamaIndexEmbeddings end to end.
 */
describe("LlamaIndex embedding registration threads the slug", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    resetPack();
  });

  it("patching a fake subpackage class carries its slug onto the row", async () => {
    const client = makeClient();
    const logs: any[] = [];
    vi.spyOn(client, "log").mockImplementation((...args: any[]) => {
      logs.push(args);
    });
    const session = makeSession();
    vi.spyOn(contextModule, "getCurrentSession").mockReturnValue(session as any);

    class FakeEmbeddingCls {
      model = "text-embedding-3-small";
      async getTextEmbedding(_t: string) {
        return [0.1, 0.2];
      }
    }
    const fakeMod = { FakeEmbeddingCls };

    enforcerTest._instrumentLlamaIndexEmbeddings(fakeMod, ["FakeEmbeddingCls"], "openai");

    await new FakeEmbeddingCls().getTextEmbedding("hello");

    expect(logs.length).toBe(1);
    const extras = logs[0][logs[0].length - 1]?.model_extras;
    expect(extras).toEqual({ framework: "llamaindex", original_provider: "openai" });
  });

  it("every registered package declares a slug", () => {
    const table = enforcerTest._LI_EMBEDDING_PACKAGES as Array<[string, string[], string]>;
    expect(table.length).toBeGreaterThan(0);
    for (const [pkg, classes, slug] of table) {
      expect(pkg.startsWith("@llamaindex/")).toBe(true);
      expect(classes.length).toBeGreaterThan(0);
      expect(typeof slug).toBe("string");
      expect(slug.length).toBeGreaterThan(0);
    }
  });
});
