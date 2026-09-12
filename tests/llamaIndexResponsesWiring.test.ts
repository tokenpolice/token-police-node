/**
 * @llamaindex/openai OpenAIResponses wiring + Responses stream usage detection.
 *
 * `OpenAIResponses` is a SIBLING class of `OpenAI` in @llamaindex/openai (its
 * own chat/streamChat against /v1/responses), and it was missing from the
 * instrumented class list at all three wiring sites (user-supplied ESM module,
 * CJS resolve, ESM dynamic import). Consequence on `llamaindex_node --api
 * responses`: no prototype patch → no pre-flight `/check`, no tool-row
 * wrapping, and ZERO llm rows — unmetered, unenforced spend that exits 0.
 *
 * The stream side needed its own repair: the guarded stream picked the usage
 * chunk by `raw.usage` (OpenAI) / `raw.usageMetadata` (Gemini), but Responses
 * chunks carry the raw stream EVENT — usage arrives once, on the terminal
 * `response.completed` event at `raw.response.usage`, and every earlier event
 * carries `response.usage: null`.
 *
 * All offline against fake provider classes — no network, no LlamaIndex.
 * Pure extraction/mapping cases live in llamaIndexVerbatimUsage.test.ts.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { setClient } from "../src/state";
import { session } from "../src/context";
import { __test__ as enforcerTest } from "../src/enforcer";

const MODEL_ARG = 4;
const PROVIDER_ARG = 5;
const INPUT_TOKENS_ARG = 6;
const OUTPUT_TOKENS_ARG = 7;
const CACHED_TOKENS_ARG = 8;
const EXTRAS_ARG = 13;

let logged: any[];

beforeEach(() => {
  logged = [];
  // firewall:"off" short-circuits the async pre-flight; log captures the row.
  setClient({ firewall: "off", log: (...args: any[]) => logged.push(args) } as any);
});

afterEach(() => {
  setClient(null as any);
  // Undo every prototype patch this file installed.
  for (const thunk of enforcerTest._restoreThunks.splice(0)) {
    try {
      thunk();
    } catch {
      /* ignore */
    }
  }
});

/**
 * Build a LlamaIndex-provider-shaped class. `_llamaIndexProvider` keys off
 * `constructor.name` and `_instrumentLlamaIndexProvider` patches
 * `prototype.chat`, so both must be real.
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

/** Async-iterable stand-in for a LlamaIndex streamChat result. */
function streamOf(chunks: any[]): AsyncIterable<any> {
  return (async function* () {
    for (const c of chunks) yield c;
  })();
}

const RESP_USAGE = {
  input_tokens: 100,
  output_tokens: 20,
  total_tokens: 120,
  input_tokens_details: { cached_tokens: 30 },
  output_tokens_details: { reasoning_tokens: 5 },
};

/** Drive one wrapped chat call inside a real session and drain if streaming. */
async function callAndDrain(instance: any, params: any = { messages: [] }): Promise<any> {
  return await session({ name: "wf" }, async () => {
    const result = await instance.chat(params);
    if (result != null && typeof result[Symbol.asyncIterator] === "function") {
      const seen: any[] = [];
      for await (const c of result) seen.push(c);
      return seen;
    }
    return result;
  });
}

describe("_instrumentLlamaIndexProvider — OpenAIResponses is wired alongside OpenAI", () => {
  it("module exporting both classes → BOTH prototypes' chat are wrapped", () => {
    const mod = {
      OpenAI: defineProviderClass("OpenAI", "gpt-4o", async () => ({ raw: {} })),
      OpenAIResponses: defineProviderClass("OpenAIResponses", "gpt-5", async () => ({ raw: {} })),
    };
    const beforeChat = mod.OpenAI.prototype.chat;
    const beforeResponsesChat = mod.OpenAIResponses.prototype.chat;

    enforcerTest._instrumentLlamaIndexProvider(mod, ["OpenAI", "OpenAIResponses"]);

    expect(mod.OpenAI.prototype.chat).not.toBe(beforeChat);
    expect(mod.OpenAIResponses.prototype.chat).not.toBe(beforeResponsesChat);
    expect((mod.OpenAIResponses.prototype.chat as any).__tp_li_wrapped).toBe(true);
  });

  it("old-shape module (only OpenAI exported) → no throw, OpenAI still wrapped", () => {
    // Older @llamaindex/openai builds have no OpenAIResponses export; the
    // extra class name must be skipped silently.
    const mod: any = {
      OpenAI: defineProviderClass("OpenAI", "gpt-4o", async () => ({ raw: {} })),
    };
    const beforeChat = mod.OpenAI.prototype.chat;
    expect(() =>
      enforcerTest._instrumentLlamaIndexProvider(mod, ["OpenAI", "OpenAIResponses"]),
    ).not.toThrow();
    expect(mod.OpenAI.prototype.chat).not.toBe(beforeChat);
    expect(mod.OpenAIResponses).toBeUndefined();
  });

  it("all three wiring sites hit the same module → wrapped exactly once", () => {
    // init() calls this for the user-supplied ESM module, the CJS resolve and
    // the async import — the same prototype can be visited three times.
    const mod = {
      OpenAIResponses: defineProviderClass("OpenAIResponses", "gpt-5", async () => ({ raw: {} })),
    };
    enforcerTest._instrumentLlamaIndexProvider(mod, ["OpenAI", "OpenAIResponses"]);
    const firstWrap = mod.OpenAIResponses.prototype.chat;
    enforcerTest._instrumentLlamaIndexProvider(mod, ["OpenAI", "OpenAIResponses"]);
    enforcerTest._instrumentLlamaIndexProvider(mod, ["OpenAI", "OpenAIResponses"]);
    expect(mod.OpenAIResponses.prototype.chat).toBe(firstWrap);
  });

  it("non-stream OpenAIResponses call → one llm row with openai_responses counts", async () => {
    const mod = {
      OpenAIResponses: defineProviderClass("OpenAIResponses", "gpt-5", async () => ({
        raw: { model: "gpt-5-2025-08-07", usage: RESP_USAGE },
        message: { role: "assistant", content: "hi" },
      })),
    };
    enforcerTest._instrumentLlamaIndexProvider(mod, ["OpenAI", "OpenAIResponses"]);

    await callAndDrain(new mod.OpenAIResponses());

    expect(logged.length).toBe(1);
    expect(logged[0][PROVIDER_ARG]).toBe("openai_responses");
    expect(logged[0][MODEL_ARG]).toBe("gpt-5-2025-08-07");
    expect(logged[0][INPUT_TOKENS_ARG]).toBe(70);
    expect(logged[0][OUTPUT_TOKENS_ARG]).toBe(20);
    expect(logged[0][CACHED_TOKENS_ARG]).toBe(30);
  });

  it("wrapping OpenAIResponses.chat also restores tool-row wrapping", async () => {
    // wrapLlamaIndexTools is the first statement of the chat wrapper — with
    // the class unwrapped, LlamaIndex tool executions produced no tool rows.
    const tool = { metadata: { name: "getCustomerInfo" }, call: async (i: any) => `cust:${i.q}` };
    const mod = {
      OpenAIResponses: defineProviderClass("OpenAIResponses", "gpt-5", async () => ({
        raw: { model: "gpt-5", usage: RESP_USAGE },
        message: { role: "assistant", content: "hi" },
      })),
    };
    enforcerTest._instrumentLlamaIndexProvider(mod, ["OpenAI", "OpenAIResponses"]);

    await session({ name: "wf" }, async () => {
      await new mod.OpenAIResponses().chat({ messages: [], tools: [tool] });
      const out = await tool.call({ q: 7 });
      expect(out).toBe("cust:7"); // original tool behavior preserved
    });

    const toolRows = logged.filter((r) => (r[EXTRAS_ARG] as any)?.tool);
    expect(toolRows.length).toBe(1);
    expect((toolRows[0][EXTRAS_ARG] as any).tool.name).toBe("getCustomerInfo");
  });
});

describe("guarded LlamaIndex stream — Responses usage-chunk detection", () => {
  function instrumentResponsesStream(chunks: any[]): any {
    const mod = {
      OpenAIResponses: defineProviderClass("OpenAIResponses", "gpt-5", async () =>
        streamOf(chunks),
      ),
    };
    enforcerTest._instrumentLlamaIndexProvider(mod, ["OpenAI", "OpenAIResponses"]);
    return mod;
  }

  const completedEvent = {
    delta: "",
    raw: {
      type: "response.completed",
      response: { model: "gpt-5-2025-08-07", usage: RESP_USAGE },
    },
  };
  const midStreamEvent = (text: string) => ({
    delta: text,
    // Every pre-terminal event carries a null usage — the truthy test must
    // skip them, otherwise the row is built from a usage-less event.
    raw: { type: "response.output_text.delta", response: { model: "gpt-5", usage: null } },
  });

  it("terminal response.completed event supplies the usage (raw.response.usage)", async () => {
    const mod = instrumentResponsesStream([
      midStreamEvent("Hel"),
      midStreamEvent("lo"),
      completedEvent,
    ]);

    const seen = await callAndDrain(new mod.OpenAIResponses(), { messages: [], stream: true });

    expect(seen.length).toBe(3); // customer stream untouched
    expect(logged.length).toBe(1);
    expect(logged[0][PROVIDER_ARG]).toBe("openai_responses");
    expect(logged[0][MODEL_ARG]).toBe("gpt-5-2025-08-07");
    expect(logged[0][INPUT_TOKENS_ARG]).toBe(70);
    expect(logged[0][OUTPUT_TOKENS_ARG]).toBe(20);
    expect(logged[0][CACHED_TOKENS_ARG]).toBe(30);
    expect(logged[0][EXTRAS_ARG]).toMatchObject({
      usage: { shape: "openai_responses", raw: RESP_USAGE },
    });
  });

  it("a usage-less event AFTER response.completed does not displace the usage chunk", async () => {
    // Pins the selection on `lastChunkWithUsage` rather than the last-chunk
    // fallback — with detection broken the row would be built from the
    // trailing null-usage event and log zeros.
    const mod = instrumentResponsesStream([
      midStreamEvent("Hi"),
      completedEvent,
      midStreamEvent(""),
    ]);

    await callAndDrain(new mod.OpenAIResponses(), { messages: [], stream: true });

    expect(logged.length).toBe(1);
    expect(logged[0][INPUT_TOKENS_ARG]).toBe(70);
    expect(logged[0][OUTPUT_TOKENS_ARG]).toBe(20);
  });

  it("stream with no usage anywhere → row still logged (fail-open), zero counts", async () => {
    const mod = instrumentResponsesStream([midStreamEvent("a"), midStreamEvent("b")]);

    await callAndDrain(new mod.OpenAIResponses(), { messages: [], stream: true });

    expect(logged.length).toBe(1);
    expect(logged[0][PROVIDER_ARG]).toBe("openai_responses");
    expect(logged[0][INPUT_TOKENS_ARG]).toBe(0);
    expect(logged[0][MODEL_ARG]).toBe("gpt-5"); // instance fallback
  });

  it("regression: Chat Completions chunks (raw.usage) still select the usage chunk", async () => {
    const usage = {
      prompt_tokens: 100,
      completion_tokens: 20,
      prompt_tokens_details: { cached_tokens: 30 },
    };
    const mod = {
      OpenAI: defineProviderClass("OpenAI", "gpt-4o", async () =>
        streamOf([
          { delta: "a", raw: { model: "gpt-4o" } },
          { delta: "b", raw: { model: "gpt-4o", usage } },
          { delta: "", raw: { model: "gpt-4o" } },
        ]),
      ),
    };
    enforcerTest._instrumentLlamaIndexProvider(mod, ["OpenAI", "OpenAIResponses"]);

    await callAndDrain(new mod.OpenAI(), { messages: [], stream: true });

    expect(logged.length).toBe(1);
    expect(logged[0][PROVIDER_ARG]).toBe("openai");
    expect(logged[0][INPUT_TOKENS_ARG]).toBe(70);
    expect(logged[0][OUTPUT_TOKENS_ARG]).toBe(20);
    expect(logged[0][EXTRAS_ARG]).toMatchObject({
      usage: { shape: "openai_compatible_chat", raw: usage },
    });
  });

  it("regression: Gemini chunks (raw.usageMetadata) still select the usage chunk", async () => {
    const um = { promptTokenCount: 100, candidatesTokenCount: 20 };
    const mod = {
      Gemini: defineProviderClass("Gemini", "gemini-2.0-flash", async () =>
        streamOf([
          { delta: "a", raw: {} },
          { delta: "b", raw: { modelVersion: "gemini-2.0-flash", usageMetadata: um } },
          { delta: "", raw: {} },
        ]),
      ),
    };
    enforcerTest._instrumentLlamaIndexProvider(mod, ["Gemini", "GoogleGenAI"]);

    await callAndDrain(new mod.Gemini(), { messages: [], stream: true });

    expect(logged.length).toBe(1);
    expect(logged[0][PROVIDER_ARG]).toBe("google");
    expect(logged[0][INPUT_TOKENS_ARG]).toBe(100);
    expect(logged[0][OUTPUT_TOKENS_ARG]).toBe(20);
  });
});

describe("golden rule — the OpenAIResponses wrapper never breaks the customer call", () => {
  it("non-stream: a poisoned `raw` getter → call resolves normally, row logged with zeros", async () => {
    const poisoned: any = { message: { role: "assistant", content: "hi" } };
    Object.defineProperty(poisoned, "raw", {
      get() {
        throw new Error("boom: hostile getter");
      },
      enumerable: true,
    });
    const mod = {
      OpenAIResponses: defineProviderClass("OpenAIResponses", "gpt-5", async () => poisoned),
    };
    enforcerTest._instrumentLlamaIndexProvider(mod, ["OpenAI", "OpenAIResponses"]);

    let out: any;
    await expect(
      session({ name: "wf" }, async () => {
        out = await new mod.OpenAIResponses().chat({ messages: [] });
      }),
    ).resolves.toBeUndefined();
    expect(out).toBe(poisoned); // customer gets the untouched response object
    expect(logged.length).toBe(1);
    expect(logged[0][INPUT_TOKENS_ARG]).toBe(0);
    expect(logged[0][MODEL_ARG]).toBe("gpt-5");
  });

  it("stream: garbage chunk shapes → stream drains untouched, row still logged", async () => {
    // The three-way usage probe (raw.usage / raw.usageMetadata /
    // raw.response.usage) walks every chunk — non-object and null `raw`s must
    // traverse it without disturbing the customer's stream.
    const chunks = [
      { delta: "a", raw: null },
      { delta: "b", raw: 42 },
      { delta: "c" },
      { raw: { response: 7 } },
      { delta: "d", raw: { type: "response.completed", response: { model: "gpt-5", usage: RESP_USAGE } } },
    ];
    const mod = {
      OpenAIResponses: defineProviderClass("OpenAIResponses", "gpt-5", async () => streamOf(chunks)),
    };
    enforcerTest._instrumentLlamaIndexProvider(mod, ["OpenAI", "OpenAIResponses"]);

    const seen = await callAndDrain(new mod.OpenAIResponses(), { messages: [], stream: true });

    expect(seen).toEqual(chunks); // customer stream unchanged
    expect(logged.length).toBe(1);
    expect(logged[0][INPUT_TOKENS_ARG]).toBe(70);
  });

  it("a provider error propagates unchanged (not swallowed, not re-wrapped)", async () => {
    const boom = new Error("upstream 500");
    const mod = {
      OpenAIResponses: defineProviderClass("OpenAIResponses", "gpt-5", async () => {
        throw boom;
      }),
    };
    enforcerTest._instrumentLlamaIndexProvider(mod, ["OpenAI", "OpenAIResponses"]);

    await expect(
      session({ name: "wf" }, async () => {
        await new mod.OpenAIResponses().chat({ messages: [] });
      }),
    ).rejects.toBe(boom);
  });
});
