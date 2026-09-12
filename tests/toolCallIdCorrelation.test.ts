/**
 * Auto-correlation of LLM tool-call ids to manual tool-execution spans.
 *
 * When an app uses tp.tool / tp.toolSpan without passing a callId, the SDK
 * stashes the (id, name) pairs from the preceding LLM response (in the
 * enforcer's _captureResponseComposition / _captureCompositionAt →
 * session.setPendingToolCalls) and the tool-span path pops the first name-match
 * to attach the id — so tool_call_id is populated with no app code changes.
 *
 * Mirrors token-police-python/tests/test_tool_call_id_correlation.py.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { setClient } from "../src/state";
import { tool, toolSpan, session } from "../src/context";
import { extractPendingToolCalls } from "../src/composition";
import { __test__ as enforcerTest } from "../src/enforcer";
import { TokenPoliceSpanProcessor } from "../src/telemetry";

const EXTRAS_ARG = 13; // extras object in tp.log positional args (see toolSpans.test.ts)
const { _captureCompositionAt } = enforcerTest as any;

describe("extractPendingToolCalls — per-provider shapes", () => {
  it("OpenAI chat-completions tool_calls (id + function.name, ordered)", () => {
    const resp = { choices: [{ message: { content: null, tool_calls: [
      { id: "call_1", type: "function", function: { name: "get_customer_info", arguments: "{}" } },
      { id: "call_2", type: "function", function: { name: "search_kb", arguments: "{}" } },
    ] } }] };
    expect(extractPendingToolCalls("openai", resp)).toEqual([
      { id: "call_1", name: "get_customer_info" },
      { id: "call_2", name: "search_kb" },
    ]);
  });

  it("Anthropic content[] tool_use blocks", () => {
    const resp = { content: [
      { type: "text", text: "hi" },
      { type: "tool_use", id: "toolu_9", name: "escalate", input: {} },
    ] };
    expect(extractPendingToolCalls("anthropic", resp)).toEqual([
      { id: "toolu_9", name: "escalate" },
    ]);
  });

  it("Cohere v2 message.toolCalls", () => {
    const resp = { message: { toolCalls: [
      { id: "cohere_x", type: "function", function: { name: "lookup", arguments: "{}" } },
    ] } };
    expect(extractPendingToolCalls("cohere", resp)).toEqual([
      { id: "cohere_x", name: "lookup" },
    ]);
  });

  it("OpenAI Responses API function_call items (call_id)", () => {
    const resp = { output: [
      { type: "reasoning" },
      { type: "function_call", call_id: "fc_7", name: "do_thing", arguments: "{}" },
    ] };
    expect(extractPendingToolCalls("openai_responses", resp)).toEqual([
      { id: "fc_7", name: "do_thing" },
    ]);
  });

  it("Gemini FunctionCall.id omitted → empty (never invent)", () => {
    const resp = { candidates: [{ content: { parts: [
      { functionCall: { name: "foo", args: {} } },
    ] } }] };
    expect(extractPendingToolCalls("google", resp)).toEqual([]);
  });

  it("Gemini FunctionCall.id present (snake + camel) → stashed", () => {
    const resp = { candidates: [{ content: { parts: [
      { function_call: { id: "fc_gem_1", name: "lookup", args: { q: "x" } } },
      { functionCall: { id: "fc_gem_2", name: "search", args: {} } },
    ] } }] };
    expect(extractPendingToolCalls("google", resp)).toEqual([
      { id: "fc_gem_1", name: "lookup" },
      { id: "fc_gem_2", name: "search" },
    ]);
  });

  it("LangChain AIMessage tool_calls (id + top-level name)", () => {
    const resp = {
      tool_calls: [
        { id: "call_lc_1", name: "get_customer_info", args: { id: 1 } },
        { id: "call_lc_2", name: "search_kb", args: {} },
      ],
    };
    expect(extractPendingToolCalls("langchain", resp)).toEqual([
      { id: "call_lc_1", name: "get_customer_info" },
      { id: "call_lc_2", name: "search_kb" },
    ]);
  });

  it("LangChain LLMResult generations[][].message.tool_calls", () => {
    const resp = {
      generations: [[{
        message: {
          tool_calls: [{ id: "call_gen", name: "lookup", args: {} }],
        },
      }]],
    };
    expect(extractPendingToolCalls("langchain", resp)).toEqual([
      { id: "call_gen", name: "lookup" },
    ]);
  });

  it("OpenAI chat tool_calls also accept top-level name (LangChain-shaped)", () => {
    const resp = {
      choices: [{
        message: {
          tool_calls: [{ id: "call_tl", name: "lookup", args: {} }],
        },
      }],
    };
    expect(extractPendingToolCalls("openai", resp)).toEqual([
      { id: "call_tl", name: "lookup" },
    ]);
  });

  it("AI SDK content[] tool-call parts (toolCallId + toolName)", () => {
    // @ai-sdk/xai and other AI SDK providers; extract existed but stash was dead until I4.
    const resp = {
      content: [
        { type: "text", text: "hi" },
        {
          type: "tool-call",
          toolCallId: "call_ai_1",
          toolName: "get_customer_info",
          input: {},
        },
        {
          type: "tool-call",
          toolCallId: "call_ai_2",
          toolName: "search_kb",
          input: {},
        },
      ],
    };
    expect(extractPendingToolCalls("ai_sdk", resp)).toEqual([
      { id: "call_ai_1", name: "get_customer_info" },
      { id: "call_ai_2", name: "search_kb" },
    ]);
  });

  it("malformed / text-only responses → []", () => {
    expect(extractPendingToolCalls("x", null)).toEqual([]);
    expect(extractPendingToolCalls("x", {})).toEqual([]);
    expect(extractPendingToolCalls("x", { choices: "nonsense" })).toEqual([]);
    expect(extractPendingToolCalls("openai", { choices: [{ message: { content: "hi" } }] })).toEqual([]);
  });
});

describe("TPSession pending tool-call stash — FIFO match", () => {
  it("distinct names match exactly and drain", () => {
    session({ name: "wf" }, (s) => {
      s.setPendingToolCalls([{ id: "a", name: "f1" }, { id: "b", name: "f2" }]);
      expect(s.popPendingToolCallId("f2")).toBe("b");
      expect(s.popPendingToolCallId("f1")).toBe("a");
      expect(s.popPendingToolCallId("f1")).toBe("");
    });
  });

  it("same name pops in FIFO order", () => {
    session({ name: "wf" }, (s) => {
      s.setPendingToolCalls([{ id: "x1", name: "f" }, { id: "x2", name: "f" }]);
      expect(s.popPendingToolCallId("f")).toBe("x1");
      expect(s.popPendingToolCallId("f")).toBe("x2");
    });
  });

  it("replace clears stale ids; junk input is safe", () => {
    session({ name: "wf" }, (s) => {
      s.setPendingToolCalls([{ id: "old", name: "f" }]);
      s.setPendingToolCalls([]); // no-tool response clears
      expect(s.popPendingToolCallId("f")).toBe("");
      s.setPendingToolCalls(undefined as any);
      expect(s.popPendingToolCallId("f")).toBe("");
    });
  });
});

describe("decorator auto-correlation (no app code change)", () => {
  let logged: any[];
  beforeEach(() => {
    logged = [];
    setClient({ log: (...args: any[]) => logged.push(args) } as any);
  });
  afterEach(() => setClient(null as any));

  it("attaches the model's tool-call id to a tool() execution by name", () => {
    session({ name: "wf" }, (s) => {
      s.setPendingToolCalls(extractPendingToolCalls("openai", { choices: [{ message: { tool_calls: [
        { id: "call_42", function: { name: "get_customer_info", arguments: "{}" } },
      ] } }] }));
      const fn = tool({ name: "get_customer_info" }, () => "ok");
      fn();
      expect(logged[0][EXTRAS_ARG].tool.call_id).toBe("call_42");
      expect(logged[0][EXTRAS_ARG].tool.name).toBe("get_customer_info");
    });
  });

  it("an explicit callId wins over the stash", () => {
    session({ name: "wf" }, (s) => {
      s.setPendingToolCalls([{ id: "stashed", name: "web_search" }]);
      toolSpan({ name: "web_search", callId: "explicit", args: "q" }, () => "r");
      expect(logged[0][EXTRAS_ARG].tool.call_id).toBe("explicit");
    });
  });

  it("no name match leaves call_id empty", () => {
    session({ name: "wf" }, (s) => {
      s.setPendingToolCalls([{ id: "call_x", name: "other_tool" }]);
      toolSpan({ name: "unmatched_tool", args: "q" }, () => "r");
      expect(logged[0][EXTRAS_ARG].tool.call_id).toBe("");
    });
  });
});

describe("I7 OTel _logToolSpan pending fallback", () => {
  let logged: any[];
  beforeEach(() => {
    logged = [];
    setClient({ log: (...args: any[]) => logged.push(args) } as any);
  });
  afterEach(() => setClient(null as any));

  function fakeToolSpan(attrs: Record<string, unknown>) {
    return {
      name: "web_search.tool",
      attributes: attrs,
      startTime: [1, 0] as [number, number],
      endTime: [2, 0] as [number, number],
      spanContext: () => ({
        traceId: "a".repeat(32),
        spanId: "b".repeat(16),
        traceFlags: 1,
      }),
      parentSpanContext: undefined,
      status: { code: 0 },
    } as any;
  }

  it("pops pending when gen_ai.tool.call.id empty", () => {
    session({ name: "wf" }, (s) => {
      s.setPendingToolCalls([
        { id: "call_otel_1", name: "web_search" },
        { id: "call_other", name: "other" },
      ]);
      const proc = new TokenPoliceSpanProcessor() as any;
      proc._logToolSpan(
        fakeToolSpan({ "gen_ai.tool.name": "web_search" }),
        { "gen_ai.tool.name": "web_search" },
        "web_search.tool",
      );
      expect(logged[0][EXTRAS_ARG].tool.call_id).toBe("call_otel_1");
      expect(s.popPendingToolCallId("other")).toBe("call_other");
    });
  });

  it("prefers OTel attr over pending (does not drain stash)", () => {
    session({ name: "wf" }, (s) => {
      s.setPendingToolCalls([{ id: "stashed", name: "web_search" }]);
      const proc = new TokenPoliceSpanProcessor() as any;
      proc._logToolSpan(
        fakeToolSpan({
          "gen_ai.tool.name": "web_search",
          "gen_ai.tool.call.id": "from_otel",
        }),
        {
          "gen_ai.tool.name": "web_search",
          "gen_ai.tool.call.id": "from_otel",
        },
        "web_search.tool",
      );
      expect(logged[0][EXTRAS_ARG].tool.call_id).toBe("from_otel");
      expect(s.popPendingToolCallId("web_search")).toBe("stashed");
    });
  });
});

describe("I7 LangChain capture stashes pending tool ids", () => {
  const {
    _captureLangchainResponse,
    _captureLangchainResponseFromMessage,
  } = enforcerTest as any;

  it("LLMResult generate path stashes AIMessage tool_calls", () => {
    session({ name: "wf" }, (s) => {
      const result = {
        generations: [[{
          message: {
            tool_calls: [{ id: "call_lc_gen", name: "lookup", args: {} }],
          },
        }]],
      };
      _captureLangchainResponse(result, 0);
      expect(s.popPendingToolCallId("lookup")).toBe("call_lc_gen");
    });
  });

  it("stream AIMessage path stashes top-level tool_calls", () => {
    session({ name: "wf" }, (s) => {
      const msg = {
        tool_calls: [{ id: "call_lc_stream", name: "search_kb", args: {} }],
      };
      _captureLangchainResponseFromMessage(msg, 0);
      expect(s.popPendingToolCallId("search_kb")).toBe("call_lc_stream");
    });
  });

  it("no-tool LangChain response clears stale stash", () => {
    session({ name: "wf" }, (s) => {
      s.setPendingToolCalls([{ id: "stale", name: "lookup" }]);
      _captureLangchainResponse({ generations: [[{ message: { content: "hi" } }]] }, 0);
      expect(s.popPendingToolCallId("lookup")).toBe("");
    });
  });
});

describe("T6 ai_sdk streaming stashes tool-call ids mid-stream (before execute())", () => {
  // The Vercel AI SDK executes each tool the moment it reads the consolidated
  // `tool-call` stream part — which is BEFORE the stream's `finish`/close where
  // the end-of-stream finalize used to be the only stash. These tests drive the
  // real _aiSdkRunStream wrapper and pop the stash exactly when the AI SDK would
  // (right after reading each tool-call chunk), asserting correlation resolves.
  const { _aiSdkRunStream } = enforcerTest as any;
  let logged: any[];
  beforeEach(() => {
    logged = [];
    // firewall:"off" short-circuits the async pre-flight; log captures finalize.
    setClient({ firewall: "off", log: (...args: any[]) => logged.push(args) } as any);
  });
  afterEach(() => setClient(null as any));

  async function* aiSdkStream(chunks: any[]) {
    for (const c of chunks) yield c;
  }

  function runStream(chunks: any[]): Promise<any> {
    const modelLike = { modelId: "grok-2", provider: "xai.chat", specificationVersion: "v2" };
    const callOriginal = async () => ({ stream: aiSdkStream(chunks) });
    return _aiSdkRunStream(callOriginal, modelLike, { prompt: [] });
  }

  const FINISH = { type: "finish", finishReason: "tool-calls", usage: { inputTokens: 10, outputTokens: 4 } };

  // Consume the wrapped stream; whenever a `tool-call` chunk surfaces, pop the
  // stash by name exactly like the AI SDK's execute() would at that instant.
  async function drainAndCorrelate(s: any, wrapped: any): Promise<{ chunks: any[]; ids: string[] }> {
    const reader = wrapped.stream.getReader();
    const chunks: any[] = [];
    const ids: string[] = [];
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      chunks.push(value);
      if (value?.type === "tool-call") {
        ids.push(s.popPendingToolCallId(value.toolName));
      }
    }
    return { chunks, ids };
  }

  it("(a) single tool executed mid-stream resolves the id; enqueue order preserved", async () => {
    await session({ name: "wf" }, async (s) => {
      const tc = { type: "tool-call", toolCallId: "call_s1", toolName: "get_customer_info", input: {} };
      const wrapped = await runStream([tc, FINISH]);
      const { chunks, ids } = await drainAndCorrelate(s, wrapped);
      expect(ids).toEqual(["call_s1"]);
      expect(chunks).toEqual([tc, FINISH]); // customer stream unchanged
    });
  });

  it("(b) two distinct-named calls in one step each correlate in order", async () => {
    await session({ name: "wf" }, async (s) => {
      const wrapped = await runStream([
        { type: "tool-call", toolCallId: "call_a", toolName: "get_customer_info", input: {} },
        { type: "tool-call", toolCallId: "call_b", toolName: "search_kb", input: {} },
        FINISH,
      ]);
      const { ids } = await drainAndCorrelate(s, wrapped);
      expect(ids).toEqual(["call_a", "call_b"]);
    });
  });

  it("(c) two SAME-named parallel calls keep FIFO order (append, not replace)", async () => {
    await session({ name: "wf" }, async (s) => {
      const wrapped = await runStream([
        { type: "tool-call", toolCallId: "call_1", toolName: "lookup", input: {} },
        { type: "tool-call", toolCallId: "call_2", toolName: "lookup", input: {} },
        FINISH,
      ]);
      const { ids } = await drainAndCorrelate(s, wrapped);
      expect(ids).toEqual(["call_1", "call_2"]); // NOT ["call_1","call_1"]
    });
  });

  it("(d) end-of-stream finalize does not resurrect ids the tools already consumed", async () => {
    await session({ name: "wf" }, async (s) => {
      const wrapped = await runStream([
        { type: "tool-call", toolCallId: "call_x", toolName: "lookup", input: {} },
        FINISH,
      ]);
      const { ids } = await drainAndCorrelate(s, wrapped);
      expect(ids).toEqual(["call_x"]);
      // Stream fully closed → finalizeUsage ran. The consumed id must stay gone.
      expect(s.popPendingToolCallId("lookup")).toBe("");
    });
  });
});

describe("I4 _captureCompositionAt stashes pending tool ids", () => {
  it("AI SDK tool-call response populates pending stash", () => {
    session({ name: "wf" }, (s) => {
      const result = {
        content: [
          {
            type: "tool-call",
            toolCallId: "call_ai_42",
            toolName: "get_customer_info",
            input: {},
          },
        ],
      };
      _captureCompositionAt("ai_sdk", [{ messages: [] }], result, 0);
      expect(s.popPendingToolCallId("get_customer_info")).toBe("call_ai_42");
    });
  });

  it("no-tool / embedding response replaces stash with empty", () => {
    // Side effect of replace-on-every-response (matches Python + Mode A).
    session({ name: "wf" }, (s) => {
      s.setPendingToolCalls([{ id: "stale", name: "get_customer_info" }]);
      _captureCompositionAt("ai_sdk", [{ input: "hi" }], { embeddings: [[0.1]] }, 0, "vercel_ai_embed", "embedding");
      expect(s.popPendingToolCallId("get_customer_info")).toBe("");
    });
  });

  it("prompt-only capture (result undefined) does not clear stash", () => {
    session({ name: "wf" }, (s) => {
      s.setPendingToolCalls([{ id: "keep_me", name: "lookup" }]);
      _captureCompositionAt("ai_sdk", [{ messages: [{ role: "user", content: "hi" }] }], undefined, 0);
      expect(s.popPendingToolCallId("lookup")).toBe("keep_me");
    });
  });
});

// ── LlamaIndex ──────────────────────────────────────────────────
//
// LlamaIndex TS hangs assistant tool calls off `message.options.toolCall`.
// The composition parser already reads that shape but drops the id, and the
// inner provider wrappers are short-circuited by `session.inLlamaIndex` — so
// nothing ever filled the stash and 8/8 LlamaIndex tool rows in the
// 2026-07-26 run had an empty tool_call_id.

describe("ExtractPendingToolCalls — LlamaIndex message.options", () => {
  it("options.toolCall (singular key, array value)", () => {
    const resp = {
      message: {
        role: "assistant",
        content: "",
        options: { toolCall: [{ id: "call_li_1", name: "getCustomerInfo", input: {} }] },
      },
    };
    expect(extractPendingToolCalls("llamaindex", resp)).toEqual([
      { id: "call_li_1", name: "getCustomerInfo" },
    ]);
  });

  it("options.toolCalls (plural key)", () => {
    const resp = {
      message: { options: { toolCalls: [{ id: "call_li_2", name: "searchKb" }] } },
    };
    expect(extractPendingToolCalls("llamaindex", resp)).toEqual([
      { id: "call_li_2", name: "searchKb" },
    ]);
  });

  it("toolCallId / toolName spellings", () => {
    const resp = {
      message: { options: { toolCall: [{ toolCallId: "call_li_3", toolName: "escalate" }] } },
    };
    expect(extractPendingToolCalls("llamaindex", resp)).toEqual([
      { id: "call_li_3", name: "escalate" },
    ]);
  });

  it("multiple calls keep response order", () => {
    const resp = {
      message: {
        options: {
          toolCall: [
            { id: "call_a", name: "lookup" },
            { id: "call_b", name: "lookup" },
          ],
        },
      },
    };
    expect(extractPendingToolCalls("llamaindex", resp)).toEqual([
      { id: "call_a", name: "lookup" },
      { id: "call_b", name: "lookup" },
    ]);
  });

  it("id-less tool call contributes nothing (never invent)", () => {
    const resp = { message: { options: { toolCall: [{ name: "lookup" }] } } };
    expect(extractPendingToolCalls("llamaindex", resp)).toEqual([]);
  });

  it("text-only LlamaIndex response → []", () => {
    const resp = { message: { role: "assistant", content: "hi", options: {} } };
    expect(extractPendingToolCalls("llamaindex", resp)).toEqual([]);
  });

  it("does not shadow the Cohere branch", () => {
    const resp = {
      message: {
        toolCalls: [{ id: "tc_cohere", function: { name: "search" } }],
        options: { toolCall: [{ id: "should_not_win", name: "search" }] },
      },
    };
    expect(extractPendingToolCalls("cohere", resp)).toEqual([
      { id: "tc_cohere", name: "search" },
    ]);
  });
});

describe("LlamaIndex capture stashes pending tool ids", () => {
  const { _captureLlamaIndexResponseAt } = enforcerTest as any;

  it("non-stream ChatResponse stashes options.toolCall ids", () => {
    session({ name: "wf" }, (s) => {
      _captureLlamaIndexResponseAt(
        {
          message: {
            role: "assistant",
            content: "on it",
            options: { toolCall: [{ id: "call_li_ns", name: "getCustomerInfo" }] },
          },
        },
        0,
      );
      expect(s.popPendingToolCallId("getCustomerInfo")).toBe("call_li_ns");
    });
  });

  it("stream synthetic (empty text, options only) still stashes", () => {
    // A pure tool-call stream accumulates no text; the old early-return on an
    // empty composition would have skipped the stash entirely.
    session({ name: "wf" }, (s) => {
      _captureLlamaIndexResponseAt(
        {
          raw: {},
          message: {
            role: "assistant",
            content: "",
            options: { toolCall: [{ id: "call_li_stream", name: "escalate" }] },
          },
        },
        0,
      );
      expect(s.popPendingToolCallId("escalate")).toBe("call_li_stream");
    });
  });

  it("no-tool LlamaIndex response clears stale stash", () => {
    session({ name: "wf" }, (s) => {
      s.setPendingToolCalls([{ id: "stale", name: "getCustomerInfo" }]);
      _captureLlamaIndexResponseAt(
        { message: { role: "assistant", content: "just text" } },
        0,
      );
      expect(s.popPendingToolCallId("getCustomerInfo")).toBe("");
    });
  });

  it("never throws on a hostile response (golden rule)", () => {
    session({ name: "wf" }, () => {
      const hostile = {
        get message() {
          throw new Error("boom");
        },
      };
      expect(() => _captureLlamaIndexResponseAt(hostile, 0)).not.toThrow();
      expect(() => _captureLlamaIndexResponseAt(null, 0)).not.toThrow();
      expect(() => _captureLlamaIndexResponseAt(undefined, 0)).not.toThrow();
    });
  });
});

describe("LlamaIndex additionalKwargs fallback (parity with the Python branch)", () => {
  it("additionalKwargs.tool_calls (OpenAI shape) when no options.toolCall", () => {
    const resp = {
      message: {
        role: "assistant",
        additionalKwargs: {
          tool_calls: [
            { id: "call_ak_1", type: "function", function: { name: "lookup" } },
          ],
        },
      },
    };
    expect(extractPendingToolCalls("llamaindex", resp)).toEqual([
      { id: "call_ak_1", name: "lookup" },
    ]);
  });

  it("snake_case additional_kwargs spelling also works", () => {
    const resp = {
      message: {
        additional_kwargs: { tool_calls: [{ id: "call_ak_2", name: "escalate" }] },
      },
    };
    expect(extractPendingToolCalls("llamaindex", resp)).toEqual([
      { id: "call_ak_2", name: "escalate" },
    ]);
  });

  it("options wins and the mirror cannot double-count", () => {
    const resp = {
      message: {
        options: { toolCall: [{ id: "call_dup", name: "lookup" }] },
        additionalKwargs: {
          tool_calls: [{ id: "call_dup", function: { name: "lookup" } }],
        },
      },
    };
    expect(extractPendingToolCalls("llamaindex", resp)).toEqual([
      { id: "call_dup", name: "lookup" },
    ]);
  });

  it("id-less entries in the mirror contribute nothing", () => {
    const resp = {
      message: { additionalKwargs: { tool_calls: [{ function: { name: "lookup" } }] } },
    };
    expect(extractPendingToolCalls("llamaindex", resp)).toEqual([]);
  });

  it("still does not shadow Cohere when both message shapes exist", () => {
    const resp = {
      message: {
        toolCalls: [{ id: "tc_cohere", function: { name: "search" } }],
        additionalKwargs: { tool_calls: [{ id: "should_not_win", name: "search" }] },
      },
    };
    expect(extractPendingToolCalls("cohere", resp)).toEqual([
      { id: "tc_cohere", name: "search" },
    ]);
  });
});

describe("Capture is resilient when composition yields nothing", () => {
  const { _captureLlamaIndexResponseAt } = enforcerTest as any;

  it("empty/degenerate response still clears the stash, never throws", () => {
    session({ name: "wf" }, (s) => {
      s.setPendingToolCalls([{ id: "stale", name: "lookup" }]);
      expect(() => _captureLlamaIndexResponseAt({}, 0)).not.toThrow();
      expect(s.popPendingToolCallId("lookup")).toBe("");
    });
  });

  it("a response whose stringification throws does not break the stash path", () => {
    session({ name: "wf" }, (s) => {
      const hostile = {
        message: { options: { toolCall: [{ id: "call_ok", name: "lookup" }] } },
        toString() {
          throw new Error("nope");
        },
      };
      expect(() => _captureLlamaIndexResponseAt(hostile, 0)).not.toThrow();
      expect(s.popPendingToolCallId("lookup")).toBe("call_ok");
    });
  });
});
