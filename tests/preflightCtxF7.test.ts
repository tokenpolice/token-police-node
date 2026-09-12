/**
 * Several pre-flight `/check` call sites fed the enforcer a bare or
 * wrong-shaped context: the target model was read ONLY from `body.model`, so on
 * Bedrock (`modelId`), the native OpenRouter SDK (`{ chatRequest: { model } }`),
 * HuggingFace chat/featureExtraction, LangChain / LlamaIndex chat and the
 * framework embedding wrappers the pre-flight ran with an EMPTY model (and, on
 * several of them, no provider at all). Consequences: model/provider-scoped
 * BLOCK rules never matched, budget group-bys bucketed to "unknown", REROUTE was
 * dead and `/check` audit events carried empty from_model/from_provider.
 *
 * The fix threads an optional `modelHint` (7th arg of `_runAsyncCheck`) used for
 * the local-evaluator ctx + the `/check` payload and NEVER merged into `body`.
 * That distinction is the load-bearing invariant these tests pin: on hint-only
 * paths a REROUTE must still end noop/rejected with NO `_tp_routing` stash (the
 * wire call would never see the swap → phantom "applied" reroute). HuggingFace
 * is the sanctioned exception: the REAL request body is passed there (it carries
 * a literal `model` key that the original call re-reads), so a same-provider
 * reroute genuinely mutates the outgoing request — asserted below.
 *
 * `tp.check` positional args: [0]=userId [1]=paidPlan [2]=workflowName
 * [3]=sessionId [4]=metadata [5]=traceId [6]=model [7]=provider [8]=intent.
 */
import { describe, test, expect, beforeEach, afterEach, vi } from "vitest";
import { createRequire } from "module";
import { applySnapshot, resetPack, setClient } from "../src/state";
import { session } from "../src/context";
import { TokenPolice, init } from "../src/client";
import { TokenPoliceBlockedError } from "../src/exceptions";
import { uninstrument, __test__ as enforcerTest } from "../src/enforcer";

const requireCjs = createRequire(import.meta.url);

// ── shared harness ──────────────────────────────────────────────────
function makeClient(
  firewall: "enforce" | "dry_run" | "off" = "dry_run",
  deployment = "serverless",
): TokenPolice {
  const client = new TokenPolice({
    apiKey: "tp_sk_test123",
    baseUrl: "http://localhost:59999",
    timeout: 0.1,
    firewall,
    deployment,
  } as never);
  setClient(client);
  return client;
}

/** Spy on check + log so nothing leaves the process. */
function stubClient(client: TokenPolice, checkResult: any = { status: "allowed" }) {
  const checkSpy = vi
    .spyOn(client, "check")
    .mockResolvedValue(checkResult as never);
  vi.spyOn(client, "log").mockImplementation(() => {});
  return checkSpy;
}

/** A model-scoped enforce BLOCK — only fires if the ctx model resolved. */
const modelBlockSnapshot = (model: string) => ({
  schema_version: 1,
  type: "snapshot",
  version: 1,
  tenant_id: "t",
  project_id: "p",
  ttl_seconds: 600,
  loop_blocks: [],
  directives: [
    {
      id: "b1",
      kind: "UNCONDITIONAL_BLOCK",
      mode: "enforce",
      priority: 10,
      selector: { match: { field: "model", operator: "EQ", value: model } },
    },
  ],
});

/** A State-B `/check` REROUTE directive for `provider`. */
const rerouteResult = (provider: string, model: string) => ({
  status: "allowed",
  reroute: { mode: "enforce", model, provider, rule_id: "rr" },
});

afterEach(() => {
  vi.restoreAllMocks();
  resetPack();
});

// ═══════════════════════════════════════════════════════════════════
// 1. modelHint plumbing (drives _runAsyncCheck directly)
// ═══════════════════════════════════════════════════════════════════
describe("(1) modelHint plumbing", () => {
  test("body without a model → hint reaches the /check payload", async () => {
    const client = makeClient();
    const checkSpy = stubClient(client);
    await session({ name: "wf" }, async () => {
      await enforcerTest._runAsyncCheck(
        { messages: [] }, "bedrock", null, true, false, "anthropic.claude-3",
      );
    });
    expect(checkSpy.mock.calls[0][6]).toBe("anthropic.claude-3");
    expect(checkSpy.mock.calls[0][7]).toBe("bedrock");
  });

  test("null body → hint still reaches the /check payload", async () => {
    const client = makeClient();
    const checkSpy = stubClient(client);
    await session({ name: "wf" }, async () => {
      await enforcerTest._runAsyncCheck(null, "openai", null, true, false, "gpt-4o");
    });
    expect(checkSpy.mock.calls[0][6]).toBe("gpt-4o");
  });

  test("body.model WINS over the hint", async () => {
    const client = makeClient();
    const checkSpy = stubClient(client);
    await session({ name: "wf" }, async () => {
      await enforcerTest._runAsyncCheck(
        { model: "real-model" }, "openai", null, true, false, "hint-model",
      );
    });
    expect(checkSpy.mock.calls[0][6]).toBe("real-model");
  });

  test("no hint + no body.model → undefined model (unchanged behavior)", async () => {
    const client = makeClient();
    const checkSpy = stubClient(client);
    await session({ name: "wf" }, async () => {
      await enforcerTest._runAsyncCheck(null, "openai");
    });
    expect(checkSpy.mock.calls[0][6]).toBeUndefined();
  });

  test("empty-string hint is ignored (no empty model on the payload)", async () => {
    const client = makeClient();
    const checkSpy = stubClient(client);
    await session({ name: "wf" }, async () => {
      await enforcerTest._runAsyncCheck(null, "openai", null, true, false, "");
    });
    expect(checkSpy.mock.calls[0][6]).toBeUndefined();
  });

  test("the local evaluator sees the hint: model-scoped BLOCK now matches (enforce → throws)", async () => {
    const client = makeClient("enforce", "daemon");
    applySnapshot(modelBlockSnapshot("anthropic.claude-3"));
    stubClient(client, { status: "blocked", reason: "x" });
    await expect(
      session({ name: "wf" }, async () => {
        await enforcerTest._runAsyncCheck(
          { messages: [] }, "bedrock", null, true, false, "anthropic.claude-3",
        );
      }),
    ).rejects.toBeInstanceOf(TokenPoliceBlockedError);
  });

  test("the local evaluator's model ctx is the hint, not a wildcard (non-matching model allows)", async () => {
    const client = makeClient("enforce", "daemon");
    applySnapshot(modelBlockSnapshot("some-other-model"));
    stubClient(client, { status: "allowed" });
    await session({ name: "wf" }, async () => {
      await enforcerTest._runAsyncCheck(
        { messages: [] }, "bedrock", null, true, false, "anthropic.claude-3",
      );
    });
    // no throw — the hint resolved to a model the rule does not name
  });

  test("a hint NEVER reaches the body → REROUTE stays a no-op, no _tp_routing", async () => {
    const client = makeClient("enforce");
    stubClient(client, rerouteResult("bedrock", "anthropic.claude-haiku"));
    const body: Record<string, any> = { messages: [] };
    const s = await session({ name: "wf" }, async (sess) => {
      await enforcerTest._runAsyncCheck(
        body, "bedrock", null, true, false, "anthropic.claude-3",
      );
      return sess;
    });
    expect("model" in body).toBe(false);
    expect((s.metadata as Record<string, unknown> | undefined)?._tp_routing).toBeUndefined();
  });
});

// ═══════════════════════════════════════════════════════════════════
// 2. Bedrock ConverseCommand — `.input.modelId`
// ═══════════════════════════════════════════════════════════════════
function makeFakeBedrock() {
  const sendCalls: any[] = [];
  class BedrockRuntimeClient {
    async send(command: any): Promise<any> {
      sendCalls.push(command);
      return {
        output: { message: { role: "assistant", content: [{ text: "ok" }] } },
        usage: { inputTokens: 1, outputTokens: 1 },
        $metadata: { httpStatusCode: 200 },
      };
    }
  }
  class ConverseCommand {
    input: any;
    constructor(input: any) {
      this.input = input;
    }
  }
  return { ns: { BedrockRuntimeClient, ConverseCommand }, sendCalls };
}

describe("(2) Bedrock ConverseCommand modelId", () => {
  afterEach(() => uninstrument());

  test("check sees `modelId` as the model (was empty)", async () => {
    const { ns } = makeFakeBedrock();
    const client = init({
      apiKey: "tp_sk_test",
      deployment: "serverless",
      firewall: "dry_run",
      instrumentModules: { bedrock: ns },
    } as never);
    const checkSpy = stubClient(client);
    const c = new ns.BedrockRuntimeClient();
    await session({ name: "wf" }, async () => {
      await (c as any).send(
        new ns.ConverseCommand({
          modelId: "anthropic.claude-3-5-sonnet-20240620-v1:0",
          messages: [{ role: "user", content: [{ text: "hi" }] }],
        }),
      );
    });
    expect(checkSpy).toHaveBeenCalledTimes(1);
    expect(checkSpy.mock.calls[0][6]).toBe("anthropic.claude-3-5-sonnet-20240620-v1:0");
    expect(checkSpy.mock.calls[0][7]).toBe("bedrock");
  });

  test("a same-provider REROUTE still NO-OPs: input untouched, no _tp_routing", async () => {
    const { ns, sendCalls } = makeFakeBedrock();
    const client = init({
      apiKey: "tp_sk_test",
      deployment: "serverless",
      firewall: "enforce",
      instrumentModules: { bedrock: ns },
    } as never);
    stubClient(client, rerouteResult("bedrock", "anthropic.claude-3-haiku-20240307-v1:0"));
    const c = new ns.BedrockRuntimeClient();
    const s = await session({ name: "wf" }, async (sess) => {
      await (c as any).send(
        new ns.ConverseCommand({
          modelId: "anthropic.claude-3-5-sonnet-20240620-v1:0",
          messages: [{ role: "user", content: [{ text: "hi" }] }],
        }),
      );
      return sess;
    });
    // The hint was NOT written into the command input, so nothing to swap.
    expect(sendCalls[0].input.modelId).toBe("anthropic.claude-3-5-sonnet-20240620-v1:0");
    expect(sendCalls[0].input.model).toBeUndefined();
    expect((s.metadata as Record<string, unknown> | undefined)?._tp_routing).toBeUndefined();
  });
});

// ═══════════════════════════════════════════════════════════════════
// 3. Native OpenRouter SDK — `{ chatRequest: { model } }`
// ═══════════════════════════════════════════════════════════════════
function makeFakeOpenRouter() {
  const sendCalls: any[] = [];
  class Chat {
    async send(req: any): Promise<any> {
      sendCalls.push(req);
      return {
        choices: [{ message: { role: "assistant", content: "ok" } }],
        usage: { prompt_tokens: 3, completion_tokens: 2 },
      };
    }
  }
  class OpenRouter {
    chat = new Chat();
    constructor(_opts: any) {}
  }
  return { ns: { OpenRouter }, sendCalls };
}

describe("(3) native OpenRouter chatRequest envelope", () => {
  afterEach(() => {
    enforcerTest._setInstrumented(true);
    uninstrument();
  });

  test("check sees the INNER chatRequest.model (was empty)", async () => {
    const { ns } = makeFakeOpenRouter();
    const client = makeClient("dry_run");
    const checkSpy = stubClient(client);
    enforcerTest._instrumentOpenRouter(ns);
    const or: any = new ns.OpenRouter({ apiKey: "k" });
    await session({ name: "wf" }, async () => {
      await or.chat.send({
        chatRequest: { model: "openai/gpt-4o-mini", messages: [{ role: "user", content: "hi" }] },
      });
    });
    expect(checkSpy).toHaveBeenCalledTimes(1);
    expect(checkSpy.mock.calls[0][6]).toBe("openai/gpt-4o-mini");
    expect(checkSpy.mock.calls[0][7]).toBe("openrouter");
  });

  test("a same-provider REROUTE still NO-OPs: envelope untouched, no _tp_routing", async () => {
    const { ns, sendCalls } = makeFakeOpenRouter();
    const client = makeClient("enforce");
    stubClient(client, rerouteResult("openrouter", "openai/gpt-4o"));
    enforcerTest._instrumentOpenRouter(ns);
    const or: any = new ns.OpenRouter({ apiKey: "k" });
    const s = await session({ name: "wf" }, async (sess) => {
      await or.chat.send({
        chatRequest: { model: "openai/gpt-4o-mini", messages: [{ role: "user", content: "hi" }] },
      });
      return sess;
    });
    expect(sendCalls[0].chatRequest.model).toBe("openai/gpt-4o-mini");
    // No top-level `model` was invented on the envelope either.
    expect(sendCalls[0].model).toBeUndefined();
    expect((s.metadata as Record<string, unknown> | undefined)?._tp_routing).toBeUndefined();
  });
});

// ═══════════════════════════════════════════════════════════════════
// 4. HuggingFace chat + featureExtraction — REAL body passed
// ═══════════════════════════════════════════════════════════════════
function hfSubmod(suffix: string): any {
  requireCjs("@huggingface/inference");
  const cache = requireCjs.cache as any;
  const key = Object.keys(cache).find(
    (k) => k.includes("@huggingface/inference") && k.replace(/\\/g, "/").endsWith(suffix),
  );
  return key ? cache[key].exports : null;
}

const HF_CHAT = "/tasks/nlp/chatCompletion.js";
const HF_EMBED = "/tasks/nlp/featureExtraction.js";

describe("(4) HuggingFace chat + featureExtraction", () => {
  const realOriginals: Record<string, { sub: any; orig: any }> = {};

  beforeEach(() => {
    for (const suffix of [HF_CHAT, HF_EMBED]) {
      const sub = hfSubmod(suffix);
      const method = suffix.includes("chatCompletion") ? "chatCompletion" : "featureExtraction";
      realOriginals[method] = { sub, orig: sub[method] };
    }
  });

  afterEach(() => {
    uninstrument();
    for (const { sub, orig } of Object.values(realOriginals)) {
      const method = orig?.name || "";
      if (method && sub) sub[method] = orig;
    }
    // Belt and braces: restore by explicit key too.
    if (realOriginals.chatCompletion) {
      realOriginals.chatCompletion.sub.chatCompletion = realOriginals.chatCompletion.orig;
    }
    if (realOriginals.featureExtraction) {
      realOriginals.featureExtraction.sub.featureExtraction = realOriginals.featureExtraction.orig;
    }
  });

  test("chat: check sees the real body model + huggingface provider, no intent", async () => {
    const sub = hfSubmod(HF_CHAT);
    sub.chatCompletion = async (_a: any) => ({
      choices: [{ message: { role: "assistant", content: "hi" } }],
      usage: { prompt_tokens: 3, completion_tokens: 2 },
    });
    const client = init({
      apiKey: "tp_sk_test",
      deployment: "serverless",
      firewall: "dry_run",
      instrumentModules: { huggingface: requireCjs("@huggingface/inference") },
    } as never);
    const checkSpy = stubClient(client);
    await session({ name: "wf" }, async () => {
      await sub.chatCompletion({
        model: "meta-llama/Llama-3.1-8B-Instruct",
        messages: [{ role: "user", content: "hi" }],
      });
    });
    expect(checkSpy.mock.calls[0][6]).toBe("meta-llama/Llama-3.1-8B-Instruct");
    expect(checkSpy.mock.calls[0][7]).toBe("huggingface");
    expect(checkSpy.mock.calls[0][8]).toBeUndefined(); // chat carries no intent
  });

  test("featureExtraction: check gets an embedding intent (mirrors operation=embedding)", async () => {
    const sub = hfSubmod(HF_EMBED);
    sub.featureExtraction = async (_a: any) => [[0.1, 0.2]];
    const client = init({
      apiKey: "tp_sk_test",
      deployment: "serverless",
      firewall: "dry_run",
      instrumentModules: { huggingface: requireCjs("@huggingface/inference") },
    } as never);
    const checkSpy = stubClient(client);
    await session({ name: "wf" }, async () => {
      await sub.featureExtraction({
        model: "sentence-transformers/all-MiniLM-L6-v2",
        inputs: "hello",
      });
    });
    expect(checkSpy.mock.calls[0][6]).toBe("sentence-transformers/all-MiniLM-L6-v2");
    expect(checkSpy.mock.calls[0][7]).toBe("huggingface");
    expect(checkSpy.mock.calls[0][8]).toEqual({ kind: "embedding" });
  });

  test("SANCTIONED: a same-provider REROUTE genuinely reaches the original call's args", async () => {
    const sub = hfSubmod(HF_CHAT);
    const seen: any[] = [];
    sub.chatCompletion = async (a: any) => {
      seen.push(a);
      return {
        choices: [{ message: { role: "assistant", content: "hi" } }],
        usage: { prompt_tokens: 3, completion_tokens: 2 },
      };
    };
    const client = init({
      apiKey: "tp_sk_test",
      deployment: "serverless",
      firewall: "enforce",
      instrumentModules: { huggingface: requireCjs("@huggingface/inference") },
    } as never);
    stubClient(client, rerouteResult("huggingface", "meta-llama/Llama-3.2-1B-Instruct"));
    const body: Record<string, any> = {
      model: "meta-llama/Llama-3.1-8B-Instruct",
      messages: [{ role: "user", content: "hi" }],
    };
    const s = await session({ name: "wf" }, async (sess) => {
      await sub.chatCompletion(body);
      return sess;
    });
    // The wire call really saw the rerouted model — so the applied stash is real.
    expect(seen[0].model).toBe("meta-llama/Llama-3.2-1B-Instruct");
    expect(body.model).toBe("meta-llama/Llama-3.2-1B-Instruct");
    expect((s.metadata as Record<string, unknown> | undefined)?._tp_routing).toBeTruthy();
  });
});

// ═══════════════════════════════════════════════════════════════════
// 5. LangChain / LlamaIndex chat + framework embeddings
// ═══════════════════════════════════════════════════════════════════
function makeFakeBaseChatModel(opts: { model?: any; namespace?: string[] } = {}): any {
  class BaseChatModel {
    model = opts.model;
    lc_namespace = opts.namespace;
    async generate(_msgs: any, _o?: any): Promise<any> {
      return { generations: [[{ message: { content: "hi" } }]] };
    }
    async *_streamIterator(_i: any, _o?: any): AsyncGenerator<any> {
      yield { content: "hi", tool_calls: [], _getType: () => "ai", concat: (x: any) => x };
    }
  }
  return BaseChatModel;
}

describe("(5a) LangChain chat", () => {
  afterEach(() => {
    enforcerTest._setInstrumented(true);
    uninstrument();
  });

  test("generate: hint from `this.model` + provider derived from lc_namespace", async () => {
    const BCM = makeFakeBaseChatModel({
      model: "gpt-4o-mini",
      namespace: ["langchain", "chat_models", "openai"],
    });
    const client = makeClient("dry_run");
    const checkSpy = stubClient(client);
    enforcerTest._instrumentLangChainChatModels({ BaseChatModel: BCM });
    await session({ name: "wf" }, async () => {
      await new BCM().generate([[{ content: "hi" }]]);
    });
    expect(checkSpy.mock.calls[0][6]).toBe("gpt-4o-mini");
    expect(checkSpy.mock.calls[0][7]).toBe("openai");
  });

  test("_streamIterator: same hint + provider", async () => {
    const BCM = makeFakeBaseChatModel({
      model: "claude-3-5-sonnet",
      namespace: ["langchain", "chat_models", "openai"],
    });
    const client = makeClient("dry_run");
    const checkSpy = stubClient(client);
    enforcerTest._instrumentLangChainChatModels({ BaseChatModel: BCM });
    await session({ name: "wf" }, async () => {
      for await (const _c of new BCM()._streamIterator([])) void _c;
    });
    expect(checkSpy.mock.calls[0][6]).toBe("claude-3-5-sonnet");
    expect(checkSpy.mock.calls[0][7]).toBe("openai");
  });

  test("underivable provider → provider OMITTED (never the literal 'langchain')", async () => {
    const BCM = makeFakeBaseChatModel({ model: "mystery-1", namespace: ["langchain", "chat_models"] });
    const client = makeClient("dry_run");
    const checkSpy = stubClient(client);
    enforcerTest._instrumentLangChainChatModels({ BaseChatModel: BCM });
    await session({ name: "wf" }, async () => {
      await new BCM().generate([[{ content: "hi" }]]);
    });
    expect(checkSpy.mock.calls[0][6]).toBe("mystery-1");
    expect(checkSpy.mock.calls[0][7]).toBeUndefined();
  });

  test("REROUTE on the hint-only LangChain path stays a no-op (no _tp_routing)", async () => {
    const BCM = makeFakeBaseChatModel({
      model: "gpt-4o-mini",
      namespace: ["langchain", "chat_models", "openai"],
    });
    const client = makeClient("enforce");
    stubClient(client, rerouteResult("openai", "gpt-4o"));
    enforcerTest._instrumentLangChainChatModels({ BaseChatModel: BCM });
    const inst = new BCM();
    const s = await session({ name: "wf" }, async (sess) => {
      await inst.generate([[{ content: "hi" }]]);
      return sess;
    });
    expect(inst.model).toBe("gpt-4o-mini"); // instance never mutated
    expect((s.metadata as Record<string, unknown> | undefined)?._tp_routing).toBeUndefined();
  });

  test("non-string model (hostile shape) → no hint, still a valid bare check", async () => {
    const BCM = makeFakeBaseChatModel({ model: { nested: 1 } as any, namespace: ["langchain"] });
    const client = makeClient("dry_run");
    const checkSpy = stubClient(client);
    enforcerTest._instrumentLangChainChatModels({ BaseChatModel: BCM });
    await session({ name: "wf" }, async () => {
      await new BCM().generate([[{ content: "hi" }]]);
    });
    expect(checkSpy.mock.calls[0][6]).toBeUndefined();
  });
});

describe("(5b) LlamaIndex chat", () => {
  afterEach(() => {
    enforcerTest._setInstrumented(true);
    uninstrument();
  });

  function makeFakeLI(model: any, name = "OpenAI") {
    const Cls = {
      [name]: class {
        model = model;
        async chat(_params: any): Promise<any> {
          return { raw: { usage: { prompt_tokens: 1, completion_tokens: 1 } }, message: { content: "hi" } };
        }
      },
    }[name];
    return { [name]: Cls } as any;
  }

  test("hint from the LLM instance + provider mirroring the /log slug", async () => {
    const mod = makeFakeLI("gpt-4o-mini");
    const client = makeClient("dry_run");
    const checkSpy = stubClient(client);
    enforcerTest._instrumentLlamaIndexProvider(mod, ["OpenAI"]);
    await session({ name: "wf" }, async () => {
      await new mod.OpenAI().chat({ messages: [{ role: "user", content: "hi" }] });
    });
    expect(checkSpy.mock.calls[0][6]).toBe("gpt-4o-mini");
    expect(checkSpy.mock.calls[0][7]).toBe("openai");
  });

  test("OpenAIResponses class → openai_responses provider (pre-flight restored by wiring it up)", async () => {
    // @llamaindex/openai's OpenAIResponses was missing from the instrumented
    // class list, so /v1/responses calls ran with NO pre-flight check at all.
    const mod = makeFakeLI("gpt-5", "OpenAIResponses");
    const client = makeClient("dry_run");
    const checkSpy = stubClient(client);
    enforcerTest._instrumentLlamaIndexProvider(mod, ["OpenAI", "OpenAIResponses"]);
    await session({ name: "wf" }, async () => {
      await new mod.OpenAIResponses().chat({ messages: [{ role: "user", content: "hi" }] });
    });
    expect(checkSpy).toHaveBeenCalledTimes(1);
    expect(checkSpy.mock.calls[0][6]).toBe("gpt-5");
    expect(checkSpy.mock.calls[0][7]).toBe("openai_responses");
  });

  test("Anthropic class → anthropic provider (same derivation _logLlamaIndex uses)", async () => {
    const mod = makeFakeLI("claude-3-5-sonnet-latest", "Anthropic");
    const client = makeClient("dry_run");
    const checkSpy = stubClient(client);
    enforcerTest._instrumentLlamaIndexProvider(mod, ["Anthropic"]);
    await session({ name: "wf" }, async () => {
      await new mod.Anthropic().chat({ messages: [{ role: "user", content: "hi" }] });
    });
    expect(checkSpy.mock.calls[0][6]).toBe("claude-3-5-sonnet-latest");
    expect(checkSpy.mock.calls[0][7]).toBe("anthropic");
  });

  test("REROUTE on the hint-only LlamaIndex path stays a no-op (no _tp_routing)", async () => {
    const mod = makeFakeLI("gpt-4o-mini");
    const client = makeClient("enforce");
    stubClient(client, rerouteResult("openai", "gpt-4o"));
    enforcerTest._instrumentLlamaIndexProvider(mod, ["OpenAI"]);
    const inst = new mod.OpenAI();
    const s = await session({ name: "wf" }, async (sess) => {
      await inst.chat({ messages: [{ role: "user", content: "hi" }] });
      return sess;
    });
    expect(inst.model).toBe("gpt-4o-mini");
    expect((s.metadata as Record<string, unknown> | undefined)?._tp_routing).toBeUndefined();
  });

  test("non-string model → no hint (bare check), provider still derived", async () => {
    const mod = makeFakeLI(undefined);
    const client = makeClient("dry_run");
    const checkSpy = stubClient(client);
    enforcerTest._instrumentLlamaIndexProvider(mod, ["OpenAI"]);
    await session({ name: "wf" }, async () => {
      await new mod.OpenAI().chat({ messages: [] });
    });
    expect(checkSpy.mock.calls[0][6]).toBeUndefined();
    expect(checkSpy.mock.calls[0][7]).toBe("openai");
  });
});

describe("(5c) framework embedding wrappers", () => {
  const origEmbed = async (_t: any) => [[0.1, 0.2]];

  test("langchain: hint from this.model + provider derived from lc_namespace", async () => {
    const client = makeClient("dry_run");
    const checkSpy = stubClient(client);
    const wrapped = enforcerTest._makeFrameworkEmbeddingWrapper(
      origEmbed, "langchain", "embedDocuments",
    );
    const inst: any = {
      model: "text-embedding-3-small",
      lc_namespace: ["langchain", "embeddings", "openai"],
      embedDocuments: wrapped,
    };
    await session({ name: "wf" }, async () => {
      await inst.embedDocuments(["hi"]);
    });
    expect(checkSpy.mock.calls[0][6]).toBe("text-embedding-3-small");
    expect(checkSpy.mock.calls[0][7]).toBe("openai");
    expect(checkSpy.mock.calls[0][8]).toEqual({ kind: "embedding" });
  });

  test("closure originalProvider wins (llamaindex concrete-class patch path)", async () => {
    const client = makeClient("dry_run");
    const checkSpy = stubClient(client);
    const wrapped = enforcerTest._makeFrameworkEmbeddingWrapper(
      origEmbed, "llamaindex", "getTextEmbedding", "cohere",
    );
    const inst: any = { model: "embed-english-v3.0", getTextEmbedding: wrapped };
    await session({ name: "wf" }, async () => {
      await inst.getTextEmbedding("hi");
    });
    expect(checkSpy.mock.calls[0][6]).toBe("embed-english-v3.0");
    expect(checkSpy.mock.calls[0][7]).toBe("cohere");
  });

  test("underivable provider → falls back to the framework slug (status quo)", async () => {
    const client = makeClient("dry_run");
    const checkSpy = stubClient(client);
    const wrapped = enforcerTest._makeFrameworkEmbeddingWrapper(
      origEmbed, "langchain", "embedQuery",
    );
    const inst: any = { model: "mystery-embed", embedQuery: wrapped };
    await session({ name: "wf" }, async () => {
      await inst.embedQuery("hi");
    });
    expect(checkSpy.mock.calls[0][6]).toBe("mystery-embed");
    expect(checkSpy.mock.calls[0][7]).toBe("langchain");
  });

  test("methodName is NEVER used as the check model hint (log side still uses it)", async () => {
    const client = makeClient("dry_run");
    const checkSpy = stubClient(client);
    const logSpy = vi.fn();
    (client as any).log = logSpy;
    const wrapped = enforcerTest._makeFrameworkEmbeddingWrapper(
      origEmbed, "langchain", "embedQuery",
    );
    const inst: any = { embedQuery: wrapped }; // no model / modelName at all
    await session({ name: "wf" }, async () => {
      await inst.embedQuery("hi");
    });
    expect(checkSpy.mock.calls[0][6]).toBeUndefined();
    // the LOG row keeps the historical methodName fallback (arg[4] = model)
    expect(logSpy.mock.calls[0][4]).toBe("embedQuery");
  });

  test("REROUTE on the embedding path stays a no-op (no _tp_routing)", async () => {
    const client = makeClient("enforce");
    stubClient(client, rerouteResult("openai", "text-embedding-3-large"));
    const wrapped = enforcerTest._makeFrameworkEmbeddingWrapper(
      origEmbed, "langchain", "embedQuery",
    );
    const inst: any = {
      model: "text-embedding-3-small",
      lc_namespace: ["langchain", "embeddings", "openai"],
      embedQuery: wrapped,
    };
    const s = await session({ name: "wf" }, async (sess) => {
      await inst.embedQuery("hi");
      return sess;
    });
    expect(inst.model).toBe("text-embedding-3-small");
    expect((s.metadata as Record<string, unknown> | undefined)?._tp_routing).toBeUndefined();
  });
});

// ═══════════════════════════════════════════════════════════════════
// 6. GOLDEN RULE — hostile instances degrade to a bare check, never throw
// ═══════════════════════════════════════════════════════════════════
describe("(6) hostile instances degrade to a bare check", () => {
  afterEach(() => {
    enforcerTest._setInstrumented(true);
    uninstrument();
  });

  /** An object whose `model` / `modelName` / `lc_namespace` getters all throw. */
  function hostileProps(obj: any): any {
    for (const p of ["model", "modelName", "lc_namespace"]) {
      Object.defineProperty(obj, p, {
        get() {
          throw new Error("hostile getter");
        },
        configurable: true,
      });
    }
    return obj;
  }

  test("LangChain chat: throwing getters → call succeeds, check ran bare", async () => {
    class BaseChatModel {
      constructor() {
        hostileProps(this);
      }
      async generate(_m: any): Promise<any> {
        return { generations: [[{ message: { content: "hi" } }]] };
      }
    }
    const client = makeClient("dry_run");
    const checkSpy = stubClient(client);
    enforcerTest._instrumentLangChainChatModels({ BaseChatModel });
    const out = await session({ name: "wf" }, async () => {
      return await new BaseChatModel().generate([[{ content: "hi" }]]);
    });
    expect(out).toBeTruthy();
    expect(checkSpy).toHaveBeenCalledTimes(1);
    expect(checkSpy.mock.calls[0][6]).toBeUndefined();
    expect(checkSpy.mock.calls[0][7]).toBeUndefined();
  });

  test("LlamaIndex chat: throwing getters → call succeeds, check ran", async () => {
    class OpenAI {
      constructor() {
        hostileProps(this);
      }
      async chat(_p: any): Promise<any> {
        return { raw: {}, message: { content: "hi" } };
      }
    }
    const client = makeClient("dry_run");
    const checkSpy = stubClient(client);
    enforcerTest._instrumentLlamaIndexProvider({ OpenAI }, ["OpenAI"]);
    const out = await session({ name: "wf" }, async () => {
      return await new OpenAI().chat({ messages: [] });
    });
    expect(out).toBeTruthy();
    expect(checkSpy).toHaveBeenCalledTimes(1);
    expect(checkSpy.mock.calls[0][6]).toBeUndefined();
  });

  test("framework embeddings: throwing getters → check still runs with the framework slug", async () => {
    const client = makeClient("dry_run");
    const checkSpy = stubClient(client);
    const wrapped = enforcerTest._makeFrameworkEmbeddingWrapper(
      async (_t: any) => [[0.1]], "langchain", "embedQuery",
    );
    const inst: any = hostileProps({});
    inst.embedQuery = wrapped;
    // The pre-existing log-side `modelHint` read is outside our try, so the
    // hostile getter surfaces there — the assertion that matters is that the
    // CHECK ran first, bare, with no new throw introduced by.
    await session({ name: "wf" }, async () => {
      await inst.embedQuery("hi").catch(() => undefined);
    });
    expect(checkSpy).toHaveBeenCalledTimes(1);
    expect(checkSpy.mock.calls[0][6]).toBeUndefined();
    expect(checkSpy.mock.calls[0][7]).toBe("langchain");
  });

  test("Bedrock: hostile `modelId` getter → check still runs, call proceeds", async () => {
    const sendCalls: any[] = [];
    class BedrockRuntimeClient {
      async send(command: any): Promise<any> {
        sendCalls.push(command);
        return { output: { message: { content: [{ text: "ok" }] } }, $metadata: {} };
      }
    }
    class ConverseCommand {
      input: any;
      constructor(input: any) {
        this.input = input;
      }
    }
    const ns = { BedrockRuntimeClient, ConverseCommand };
    const client = init({
      apiKey: "tp_sk_test",
      deployment: "serverless",
      firewall: "dry_run",
      instrumentModules: { bedrock: ns },
    } as never);
    const checkSpy = stubClient(client);
    const input: any = { messages: [] };
    Object.defineProperty(input, "modelId", {
      get() {
        throw new Error("hostile getter");
      },
    });
    const c = new BedrockRuntimeClient();
    await session({ name: "wf" }, async () => {
      await (c as any).send(new ConverseCommand(input));
    });
    expect(sendCalls.length).toBe(1);
    expect(checkSpy).toHaveBeenCalledTimes(1);
    expect(checkSpy.mock.calls[0][6]).toBeUndefined();
  });
});
