/**
 * LangChain wrapper failure emission — both branches.
 *
 * `_setLangchainWrapper` previously had NO failure handling at all: no catch,
 * no `_stashAttemptContext`, no `buildCallOutcome`. The traceloop-langchain
 * instrumentor DOES end its span with ERROR status (handleLLMError), but:
 *
 * - with zero usage attrs telemetry's zero-usage gate drops it → **no llm row**
 *   for a failed LangChain call; and
 * - when streamed chunks carried `usage_metadata` before the failure, the
 *   `langchain_message` stash passes that gate and a **mislabeled success row**
 *   lands instead.
 *
 * Fix under test (both branches):
 * - stash the attempt context off `_langchainChatCheckCtx` so the failure row
 *   carries a real model/provider instead of model="unknown";
 * - catch the provider rejection, emit exactly ONE failure row
 *   (`buildCallOutcome` → error_kind/http_status), rethrow the ORIGINAL error
 *   by identity;
 * - mark `${traceId}:${order}` in `session._failedStreamCompKeys` so the
 *   deferred span log is suppressed (no second row);
 * - `exitLangchain()` still runs on every exit path (guard never leaks).
 *
 * Early consumer `break` is NOT a failure: finally only, no row.
 *
 * All offline against a fake BaseChatModel — no network, no LangChain.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { setClient } from "../src/state";
import { TPSession, _getSessionStorage } from "../src/context";
import { __test__ as enforcerTest } from "../src/enforcer";
import { TokenPoliceSpanProcessor } from "../src/telemetry";

let logged: any[][];

beforeEach(() => {
  logged = [];
  setClient({ firewall: "off", log: (...args: any[]) => logged.push(args) } as any);
});

afterEach(() => {
  setClient(null as any);
  for (const thunk of enforcerTest._restoreThunks.splice(0)) {
    try {
      thunk();
    } catch {
      /* ignore */
    }
  }
});

function newSession(): TPSession {
  return new TPSession({
    userId: "u1",
    paidPlan: "pro",
    workflowName: "wf",
    traceId: "c".repeat(32),
    rootSpanId: "d".repeat(16),
  });
}

/** `tp.log`'s trailing options arg — where `call_outcome` lives. */
function extrasOf(logArgs: any[]): any {
  return logArgs[logArgs.length - 1];
}

/** Minimal AIMessageChunk with a concat that sums usage_metadata like LC. */
function makeUsageChunk(
  text: string,
  usage?: { input_tokens?: number; output_tokens?: number },
): any {
  return {
    content: text,
    tool_calls: [],
    usage_metadata:
      usage == null
        ? undefined
        : {
            input_tokens: usage.input_tokens ?? 0,
            output_tokens: usage.output_tokens ?? 0,
            total_tokens: (usage.input_tokens ?? 0) + (usage.output_tokens ?? 0),
          },
    _getType() {
      return "ai";
    },
    concat(other: any): any {
      const a = this.usage_metadata;
      const b = other?.usage_metadata;
      const out = makeUsageChunk(this.content + (other?.content ?? ""));
      if (a || b) {
        out.usage_metadata = {
          input_tokens: (a?.input_tokens ?? 0) + (b?.input_tokens ?? 0),
          output_tokens: (a?.output_tokens ?? 0) + (b?.output_tokens ?? 0),
          total_tokens: 0,
        };
      }
      return out;
    },
  };
}

/** OpenAI-SDK-shaped 401 (`AuthenticationError` + `status`). */
function auth401(): any {
  class AuthenticationError extends Error {}
  return Object.assign(new AuthenticationError("401 Incorrect API key provided"), {
    status: 401,
  });
}

function instrument(BaseChatModel: any): void {
  enforcerTest._instrumentLangChainChatModels({ BaseChatModel });
}

/** ReadableSpan stub for the LangChain LLM span at `order`. */
function fakeSpan(
  session: TPSession,
  order: number,
  opts: { input?: number; output?: number; system?: string; model?: string } = {},
): any {
  const attrs: Record<string, unknown> = {
    "gen_ai.system": opts.system ?? "cohere",
    "gen_ai.request.model": opts.model ?? "command-r7b-12-2024",
    "tp.trace_id": session.traceId,
    "tp.span_order": order,
  };
  if (opts.input != null) attrs["gen_ai.usage.input_tokens"] = opts.input;
  if (opts.output != null) attrs["gen_ai.usage.output_tokens"] = opts.output;
  return {
    attributes: attrs,
    name: "langchain.chat",
    startTime: [0, 0] as [number, number],
    endTime: [1, 0] as [number, number],
    spanContext: () => ({
      traceId: "0000000000000000000000000000ef01",
      spanId: "0000000000005678",
    }),
    parentSpanId: undefined,
    status: { code: 0 },
  };
}

/** Runs the span processor's deferred (nextTick) onEnd log to completion. */
async function runOnEnd(session: TPSession, span: any): Promise<void> {
  await _getSessionStorage().run(session, async () => {
    new TokenPoliceSpanProcessor().onEnd(span);
    await new Promise((r) => setTimeout(r, 0));
  });
}

// ── 6. generate (async) branch ───────────────────────────────────

describe("LangChain generate branch — provider rejection", () => {
  /** ChatOpenAI-shaped fake whose `generate` rejects. */
  function mkFailingModel(err: unknown): any {
    class BaseChatModel {
      model = "gpt-4o-mini";
      lc_namespace = ["langchain", "chat_models", "openai"];
      async generate(_messages: any, _options?: any): Promise<any> {
        throw err;
      }
    }
    return BaseChatModel;
  }

  it("exactly one failed llm row carrying the stashed model/provider (not model='unknown')", async () => {
    const boom = auth401();
    const BCM = mkFailingModel(boom);
    instrument(BCM);
    const session = newSession();

    await _getSessionStorage().run(session, async () => {
      await expect(new BCM().generate([[]])).rejects.toBe(boom); // identity
    });

    expect(logged).toHaveLength(1);
    expect(logged[0][4]).toBe("gpt-4o-mini");
    expect(logged[0][5]).toBe("openai"); // derived from lc_namespace
    const outcome = extrasOf(logged[0]).call_outcome;
    expect(outcome.status).toBe("failed");
    expect(outcome.error_kind).toBe("auth_error");
    expect(outcome.http_status).toBe(401);
  });

  it("marks the compKey and releases the inLangchain guard", async () => {
    const boom = Object.assign(new Error("upstream 500"), { status: 500 });
    const BCM = mkFailingModel(boom);
    instrument(BCM);
    const session = newSession();

    await _getSessionStorage().run(session, async () => {
      await expect(new BCM().generate([[]])).rejects.toBe(boom);
    });

    expect((session as any)._failedStreamCompKeys.has(`${session.traceId}:0`)).toBe(
      true,
    );
    expect(session.inLangchain).toBe(false);
    expect(extrasOf(logged[0]).call_outcome.error_kind).toBe("server_error");
  });

  it("a successful generate still logs nothing here and creates no marker", async () => {
    class BaseChatModel {
      model = "gpt-4o-mini";
      lc_namespace = ["langchain", "chat_models", "openai"];
      async generate(_messages: any): Promise<any> {
        return { generations: [[{ text: "hi", message: { content: "hi" } }]] };
      }
    }
    instrument(BaseChatModel);
    const session = newSession();

    await _getSessionStorage().run(session, async () => {
      await new BaseChatModel().generate([[]]);
    });

    expect(logged).toHaveLength(0); // the row comes from the span, not here
    expect((session as any)._failedStreamCompKeys).toBeUndefined();
    expect(session.inLangchain).toBe(false);
  });

  it("golden rule: tp.log throwing still surfaces only the provider error", async () => {
    setClient({
      firewall: "off",
      log: () => {
        throw new Error("log exploded");
      },
    } as any);
    const boom = auth401();
    const BCM = mkFailingModel(boom);
    instrument(BCM);
    const session = newSession();

    await _getSessionStorage().run(session, async () => {
      await expect(new BCM().generate([[]])).rejects.toBe(boom);
    });
    expect(session.inLangchain).toBe(false);
  });
});

// ── 7. streamIterator branch ─────────────────────────────────────

describe("LangChain streamIterator branch — rejection mid-iteration", () => {
  /**
   * ChatCohere-shaped fake: yields `n` usage-carrying chunks, then throws.
   * The pre-failure chunks are what used to produce the mislabeled success
   * row (their concat'd usage_metadata passes telemetry's zero-usage gate).
   */
  function mkFailingStreamModel(err: unknown, chunks: any[]): any {
    class BaseChatModel {
      model = "command-r7b-12-2024";
      lc_namespace = ["langchain", "chat_models", "cohere"];
      async *_streamIterator(_input: any, _options?: any): AsyncGenerator<any> {
        for (const c of chunks) yield c;
        throw err;
      }
    }
    return BaseChatModel;
  }

  const preFailureChunks = (): any[] => [
    makeUsageChunk("a", { input_tokens: 130, output_tokens: 1 }),
    makeUsageChunk("b", { input_tokens: 0, output_tokens: 4 }),
  ];

  async function driveFailing(
    session: TPSession,
    BCM: any,
  ): Promise<{ caught: unknown; seen: any[] }> {
    const seen: any[] = [];
    let caught: unknown;
    await _getSessionStorage().run(session, async () => {
      try {
        for await (const c of new BCM()._streamIterator("hi")) seen.push(c);
      } catch (e) {
        caught = e;
      }
    });
    return { caught, seen };
  }

  it("exactly one failed llm row + the ORIGINAL error by identity, guard released", async () => {
    const boom = auth401();
    const BCM = mkFailingStreamModel(boom, preFailureChunks());
    instrument(BCM);
    const session = newSession();

    const { caught, seen } = await driveFailing(session, BCM);
    expect(caught).toBe(boom);
    expect(seen).toHaveLength(2); // usage-carrying chunks were consumed first

    expect(logged).toHaveLength(1);
    expect(logged[0][4]).toBe("command-r7b-12-2024");
    expect(logged[0][5]).toBe("cohere");
    const outcome = extrasOf(logged[0]).call_outcome;
    expect(outcome.status).toBe("failed");
    expect(outcome.error_kind).toBe("auth_error");
    expect(outcome.http_status).toBe(401);
    expect(session.inLangchain).toBe(false);
  });

  it("marks ${traceId}:${order} and telemetry's deferred span log is suppressed (no mislabeled success row)", async () => {
    const boom = auth401();
    const BCM = mkFailingStreamModel(boom, preFailureChunks());
    instrument(BCM);
    const session = newSession();

    const { caught } = await driveFailing(session, BCM);
    expect(caught).toBe(boom);
    const compKey = `${session.traceId}:0`;
    expect((session as any)._failedStreamCompKeys.has(compKey)).toBe(true);
    // The finally re-stashed the concat'd usage — exactly the stash that would
    // otherwise carry the span row past the zero-usage gate.
    expect(
      (session as any)._pendingCompositions[compKey]?.usage?.input_tokens,
    ).toBe(130);

    await runOnEnd(session, fakeSpan(session, 0, { input: 0, output: 4 }));

    expect(logged).toHaveLength(1); // still just the failure row
    expect((session as any)._failedStreamCompKeys.has(compKey)).toBe(false);
    expect((session as any)._pendingCompositions[compKey]).toBeUndefined();
  });

  it("control: without the marker the same span DOES land a second row (proves the suppression)", async () => {
    const boom = auth401();
    const BCM = mkFailingStreamModel(boom, preFailureChunks());
    instrument(BCM);
    const session = newSession();

    await driveFailing(session, BCM);
    expect(logged).toHaveLength(1);

    (session as any)._failedStreamCompKeys.clear();
    await runOnEnd(session, fakeSpan(session, 0, { input: 0, output: 4 }));
    expect(logged).toHaveLength(2); // the pre-fix mislabeled row
  });

  it("zero-usage failure (no chunk arrived) still emits the row", async () => {
    const boom = Object.assign(new Error("429 slow down"), { status: 429 });
    const BCM = mkFailingStreamModel(boom, []);
    instrument(BCM);
    const session = newSession();

    const { caught, seen } = await driveFailing(session, BCM);
    expect(caught).toBe(boom);
    expect(seen).toEqual([]);
    expect(logged).toHaveLength(1);
    expect(extrasOf(logged[0]).call_outcome.error_kind).toBe("rate_limited");
    // The span would carry zero usage and be gate-dropped — the row here is
    // the only record of the failed call.
    await runOnEnd(session, fakeSpan(session, 0));
    expect(logged).toHaveLength(1);
  });

  it("early consumer break → no failure row, no marker, guard still released", async () => {
    const boom = auth401();
    const BCM = mkFailingStreamModel(boom, [
      makeUsageChunk("a", { input_tokens: 10, output_tokens: 1 }),
      makeUsageChunk("b", { input_tokens: 0, output_tokens: 1 }),
    ]);
    instrument(BCM);
    const session = newSession();

    await _getSessionStorage().run(session, async () => {
      let n = 0;
      for await (const _c of new BCM()._streamIterator("hi")) {
        if (++n === 1) break;
      }
    });

    expect(logged).toHaveLength(0);
    expect((session as any)._failedStreamCompKeys).toBeUndefined();
    expect(session.inLangchain).toBe(false);
  });

  it("full drain of a healthy stream → no failure row, no marker, usage stash unchanged", async () => {
    class BaseChatModel {
      model = "command-r7b-12-2024";
      lc_namespace = ["langchain", "chat_models", "cohere"];
      async *_streamIterator(_input: any): AsyncGenerator<any> {
        for (const c of preFailureChunks()) yield c;
      }
    }
    instrument(BaseChatModel);
    const session = newSession();

    await _getSessionStorage().run(session, async () => {
      for await (const _c of new BaseChatModel()._streamIterator("hi")) {
        /* drain */
      }
    });

    expect(logged).toHaveLength(0);
    expect((session as any)._failedStreamCompKeys).toBeUndefined();
    expect(session.inLangchain).toBe(false);
    expect(
      (session as any)._pendingCompositions[`${session.traceId}:0`]?.usage,
    ).toMatchObject({
      usage_source: "langchain_message",
      input_tokens: 130,
      output_tokens: 5,
    });
  });

  it("golden rule: tp.log throwing still surfaces only the provider error, guard released", async () => {
    setClient({
      firewall: "off",
      log: () => {
        throw new Error("log exploded");
      },
    } as any);
    const boom = auth401();
    const BCM = mkFailingStreamModel(boom, preFailureChunks());
    instrument(BCM);
    const session = newSession();

    const { caught } = await driveFailing(session, BCM);
    expect(caught).toBe(boom);
    expect(session.inLangchain).toBe(false);
  });
});

// ── 8. provider derivation on the failure row (F-G6b-3) ──────────
//
// Both branches above stash `ctx.provider ?? ""` from `_langchainChatCheckCtx`
// → `_deriveLangChainProvider`, which resolves `lc_namespace` against
// `_LC_NAMESPACE_TO_PROVIDER`. That table listed only the packages LangChain
// ships EMBEDDINGS for, so `ChatAnthropic`
// (lc_namespace ["langchain","chat_models","anthropic"]) missed the lookup:
// its failure rows landed with provider='' / original_provider='' and — since
// `_emitCallFailureLog` resolves the shape from that same empty provider —
// usage_shape='openai_compatible_chat', while anthropic SUCCESS rows and the
// openai/gemini failure twins were all correct.
//
// The constructor-name fallback could not save it either: it does a
// `startsWith` on the lowered ctor name, and "chatanthropic" does not start
// with "anthropic". So these fixtures pin the namespace table specifically.
//
// Every case above hardcoded the openai namespace, which is exactly what let
// the bug ship — these cover the other vendors on both branches.

describe("LangChain failure rows — provider derived from lc_namespace", () => {
  /** Same drive-to-failure loop §7 uses, re-declared for this section. */
  async function driveStream(
    session: TPSession,
    BCM: any,
  ): Promise<{ caught: unknown; seen: any[] }> {
    const seen: any[] = [];
    let caught: unknown;
    await _getSessionStorage().run(session, async () => {
      try {
        for await (const c of new BCM()._streamIterator("hi")) seen.push(c);
      } catch (e) {
        caught = e;
      }
    });
    return { caught, seen };
  }

  /** ChatAnthropic-shaped fake whose `generate` rejects. */
  function mkFailingAnthropicModel(err: unknown): any {
    class ChatAnthropic {
      model = "claude-haiku-4-5-20251001";
      lc_namespace = ["langchain", "chat_models", "anthropic"];
      async generate(_messages: any, _options?: any): Promise<any> {
        throw err;
      }
    }
    return ChatAnthropic;
  }

  /** ChatAnthropic-shaped fake whose `_streamIterator` rejects mid-iteration. */
  function mkFailingAnthropicStreamModel(err: unknown, chunks: any[]): any {
    class ChatAnthropic {
      model = "claude-haiku-4-5-20251001";
      lc_namespace = ["langchain", "chat_models", "anthropic"];
      async *_streamIterator(_input: any, _options?: any): AsyncGenerator<any> {
        for (const c of chunks) yield c;
        throw err;
      }
    }
    return ChatAnthropic;
  }

  it("generate: ChatAnthropic 401 lands provider='anthropic' (was '')", async () => {
    const boom = auth401();
    const BCM = mkFailingAnthropicModel(boom);
    instrument(BCM);
    const session = newSession();

    await _getSessionStorage().run(session, async () => {
      await expect(new BCM().generate([[]])).rejects.toBe(boom); // identity
    });

    expect(logged).toHaveLength(1);
    expect(logged[0][4]).toBe("claude-haiku-4-5-20251001");
    expect(logged[0][5]).toBe("anthropic"); // derived from lc_namespace
    const outcome = extrasOf(logged[0]).call_outcome;
    expect(outcome.status).toBe("failed");
    expect(outcome.error_kind).toBe("auth_error");
    expect(outcome.http_status).toBe(401);
    expect(session.inLangchain).toBe(false);
  });

  it("generate: ChatAnthropic failure row carries anthropic_messages, not openai_compatible_chat", async () => {
    const boom = auth401();
    const BCM = mkFailingAnthropicModel(boom);
    instrument(BCM);
    const session = newSession();

    await _getSessionStorage().run(session, async () => {
      await expect(new BCM().generate([[]])).rejects.toBe(boom);
    });

    // `_emitCallFailureLog` resolves the shape off the stashed provider, so an
    // empty provider silently stamped the OpenAI CHAT shape onto every failed
    // ChatAnthropic row while its successful siblings read anthropic_messages.
    const usage = extrasOf(logged[0]).usage;
    expect(usage.shape).toBe("anthropic_messages");
    expect(usage.shape).not.toBe("openai_compatible_chat");
    // A failed call has no usage — zero counts keep the row unmeasured.
    expect(usage.raw).toEqual({ prompt_tokens: 0, total_tokens: 0 });
  });

  it("streamIterator: ChatAnthropic rejection mid-stream lands provider='anthropic' + anthropic_messages", async () => {
    const boom = auth401();
    const BCM = mkFailingAnthropicStreamModel(boom, [
      makeUsageChunk("a", { input_tokens: 130, output_tokens: 1 }),
      makeUsageChunk("b", { input_tokens: 0, output_tokens: 4 }),
    ]);
    instrument(BCM);
    const session = newSession();

    const { caught, seen } = await driveStream(session, BCM);
    expect(caught).toBe(boom); // identity
    expect(seen).toHaveLength(2);

    expect(logged).toHaveLength(1);
    expect(logged[0][4]).toBe("claude-haiku-4-5-20251001");
    expect(logged[0][5]).toBe("anthropic");
    expect(extrasOf(logged[0]).usage.shape).toBe("anthropic_messages");
    expect(extrasOf(logged[0]).call_outcome.error_kind).toBe("auth_error");
    expect((session as any)._failedStreamCompKeys.has(`${session.traceId}:0`)).toBe(
      true,
    );
    expect(session.inLangchain).toBe(false);
  });

  it("streamIterator: zero-usage ChatAnthropic failure still carries the provider", async () => {
    const boom = Object.assign(new Error("429 slow down"), { status: 429 });
    const BCM = mkFailingAnthropicStreamModel(boom, []);
    instrument(BCM);
    const session = newSession();

    const { caught } = await driveStream(session, BCM);
    expect(caught).toBe(boom);
    expect(logged).toHaveLength(1);
    expect(logged[0][5]).toBe("anthropic");
    expect(extrasOf(logged[0]).usage.shape).toBe("anthropic_messages");
    expect(extrasOf(logged[0]).call_outcome.error_kind).toBe("rate_limited");
  });

  it("GOLDEN RULE: an unknown vendor namespace degrades (provider='') and never throws", async () => {
    const boom = auth401();
    class ChatSomeFutureProvider {
      model = "future-model-1";
      lc_namespace = ["langchain", "chat_models", "somefutureprovider"];
      async generate(_messages: any): Promise<any> {
        throw boom;
      }
    }
    instrument(ChatSomeFutureProvider);
    const session = newSession();

    await _getSessionStorage().run(session, async () => {
      // ONLY the provider's own error surfaces — an unmapped vendor must
      // degrade, never raise out of the derivation/stash path.
      await expect(new ChatSomeFutureProvider().generate([[]])).rejects.toBe(boom);
    });

    expect(logged).toHaveLength(1);
    expect(logged[0][4]).toBe("future-model-1"); // model still real
    expect(logged[0][5]).toBe(""); // unresolved provider stays empty
    expect(extrasOf(logged[0]).usage.shape).toBe("openai_compatible_chat"); // default
    expect(extrasOf(logged[0]).call_outcome.status).toBe("failed");
    expect(session.inLangchain).toBe(false);
  });

  it("GOLDEN RULE: an unknown vendor streaming failure degrades the same way", async () => {
    const boom = auth401();
    class ChatSomeFutureProvider {
      model = "future-model-1";
      lc_namespace = ["langchain", "chat_models", "somefutureprovider"];
      async *_streamIterator(_input: any): AsyncGenerator<any> {
        yield makeUsageChunk("a", { input_tokens: 5, output_tokens: 1 });
        throw boom;
      }
    }
    instrument(ChatSomeFutureProvider);
    const session = newSession();

    const { caught } = await driveStream(session, ChatSomeFutureProvider);
    expect(caught).toBe(boom);
    expect(logged).toHaveLength(1);
    expect(logged[0][5]).toBe("");
    expect(extrasOf(logged[0]).usage.shape).toBe("openai_compatible_chat");
    expect(session.inLangchain).toBe(false);
  });

  it("gemini twin: ChatGoogleGenerativeAI still resolves to 'gemini' (google_genai segment)", async () => {
    const boom = auth401();
    class ChatGoogleGenerativeAI {
      model = "gemini-2.5-flash";
      lc_namespace = ["langchain", "chat_models", "google_genai"];
      async generate(_messages: any): Promise<any> {
        throw boom;
      }
    }
    instrument(ChatGoogleGenerativeAI);
    const session = newSession();

    await _getSessionStorage().run(session, async () => {
      await expect(new ChatGoogleGenerativeAI().generate([[]])).rejects.toBe(boom);
    });

    expect(logged[0][5]).toBe("gemini");
    expect(extrasOf(logged[0]).usage.shape).toBe("google_genai");
  });
});

// ── 9. the remaining chat-only vendors (F-G6b-3 follow-up) ───────
//
// §8 fixed anthropic, but the same hole stayed open for every other vendor
// LangChain ships a chat class — but no embeddings — for. Three more verified
// namespaces joined `_LC_NAMESPACE_TO_PROVIDER`:
//
//   groq     ChatGroq      ["langchain","chat_models","groq"]
//   xai      ChatXAI       ["langchain","chat_models","xai"]
//   deepseek ChatDeepSeek  ["langchain","chat_models","deepseek"]
//
// ChatXAI and ChatDeepSeek both extend ChatOpenAI but OVERRIDE its
// lc_namespace, so they did not even inherit the working `openai` entry —
// their failure rows landed blank exactly like anthropic's in §8.
// And as in §8 the ctor-name `startsWith` fallback
// cannot cover any of them ("chatgroq"/"chatxai"/"chatdeepseek" all fail it),
// so these fixtures pin the namespace table specifically.

describe("LangChain failure rows — groq/xai/deepseek lc_namespace", () => {
  /** Same drive-to-failure loop §7/§8 use, re-declared for this section. */
  async function driveStream(
    session: TPSession,
    BCM: any,
  ): Promise<{ caught: unknown; seen: any[] }> {
    const seen: any[] = [];
    let caught: unknown;
    await _getSessionStorage().run(session, async () => {
      try {
        for await (const c of new BCM()._streamIterator("hi")) seen.push(c);
      } catch (e) {
        caught = e;
      }
    });
    return { caught, seen };
  }

  it("generate: ChatGroq 401 lands provider='groq' + groq_chat", async () => {
    const boom = auth401();
    class ChatGroq {
      model = "llama-3.3-70b-versatile";
      lc_namespace = ["langchain", "chat_models", "groq"];
      async generate(_messages: any): Promise<any> {
        throw boom;
      }
    }
    instrument(ChatGroq);
    const session = newSession();

    await _getSessionStorage().run(session, async () => {
      await expect(new ChatGroq().generate([[]])).rejects.toBe(boom); // identity
    });

    expect(logged).toHaveLength(1);
    expect(logged[0][4]).toBe("llama-3.3-70b-versatile");
    expect(logged[0][5]).toBe("groq"); // derived from lc_namespace
    expect(extrasOf(logged[0]).usage.shape).toBe("groq_chat");
    const outcome = extrasOf(logged[0]).call_outcome;
    expect(outcome.status).toBe("failed");
    expect(outcome.error_kind).toBe("auth_error");
    expect(outcome.http_status).toBe(401);
    expect(session.inLangchain).toBe(false); // guard released
  });

  it("generate: ChatXAI 401 lands provider='xai' + xai_chat, not its ChatOpenAI parent's", async () => {
    const boom = auth401();
    class ChatXAI {
      model = "grok-4";
      // ChatXAI extends ChatOpenAI but replaces its lc_namespace — the
      // namespace path, not the ctor fallback, is what must resolve this.
      lc_namespace = ["langchain", "chat_models", "xai"];
      async generate(_messages: any): Promise<any> {
        throw boom;
      }
    }
    instrument(ChatXAI);
    const session = newSession();

    await _getSessionStorage().run(session, async () => {
      await expect(new ChatXAI().generate([[]])).rejects.toBe(boom);
    });

    expect(logged).toHaveLength(1);
    expect(logged[0][4]).toBe("grok-4");
    expect(logged[0][5]).toBe("xai");
    expect(logged[0][5]).not.toBe("openai"); // parent must not leak through
    expect(extrasOf(logged[0]).usage.shape).toBe("xai_chat");
    expect(extrasOf(logged[0]).call_outcome.error_kind).toBe("auth_error");
    expect(session.inLangchain).toBe(false);
  });

  it("generate: ChatDeepSeek 401 lands provider='deepseek' with the openai_compatible_chat default", async () => {
    const boom = auth401();
    class ChatDeepSeek {
      model = "deepseek-chat";
      lc_namespace = ["langchain", "chat_models", "deepseek"];
      async generate(_messages: any): Promise<any> {
        throw boom;
      }
    }
    instrument(ChatDeepSeek);
    const session = newSession();

    await _getSessionStorage().run(session, async () => {
      await expect(new ChatDeepSeek().generate([[]])).rejects.toBe(boom);
    });

    expect(logged).toHaveLength(1);
    expect(logged[0][4]).toBe("deepseek-chat");
    // DeepSeek's wire usage IS the OpenAI chat shape, so `_resolveUsageShape`
    // deliberately has no `deepseek` case and the DEFAULT is the correct
    // answer here — unlike §8's anthropic row, where the default was the bug.
    // What the fix owes us is the PROVIDER: it must resolve, not stay blank,
    // so the row attributes cost/spend to deepseek rather than to nobody.
    expect(logged[0][5]).toBe("deepseek");
    expect(logged[0][5]).not.toBe("");
    expect(extrasOf(logged[0]).usage.shape).toBe("openai_compatible_chat");
    expect(extrasOf(logged[0]).call_outcome.error_kind).toBe("auth_error");
    expect(session.inLangchain).toBe(false);
  });

  it("streamIterator: ChatGroq rejection mid-stream lands provider='groq' + groq_chat", async () => {
    const boom = auth401();
    class ChatGroq {
      model = "llama-3.3-70b-versatile";
      lc_namespace = ["langchain", "chat_models", "groq"];
      async *_streamIterator(_input: any, _options?: any): AsyncGenerator<any> {
        yield makeUsageChunk("a", { input_tokens: 130, output_tokens: 1 });
        yield makeUsageChunk("b", { input_tokens: 0, output_tokens: 4 });
        throw boom;
      }
    }
    instrument(ChatGroq);
    const session = newSession();

    const { caught, seen } = await driveStream(session, ChatGroq);
    expect(caught).toBe(boom); // identity
    expect(seen).toHaveLength(2);

    expect(logged).toHaveLength(1);
    expect(logged[0][4]).toBe("llama-3.3-70b-versatile");
    expect(logged[0][5]).toBe("groq");
    expect(extrasOf(logged[0]).usage.shape).toBe("groq_chat");
    expect(extrasOf(logged[0]).call_outcome.error_kind).toBe("auth_error");
    // deferred span log suppressed — no mislabeled second row from the
    // pre-failure usage chunks
    expect((session as any)._failedStreamCompKeys.has(`${session.traceId}:0`)).toBe(
      true,
    );
    expect(session.inLangchain).toBe(false); // guard released
  });
});
