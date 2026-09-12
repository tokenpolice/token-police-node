/**
 * G5-1 — Together direct-SDK streaming silently lost 30–50% of llm rows.
 *
 * Root cause: whether a Together chat stream carries a usage payload AT ALL is
 * replica-dependent when the request does not ask for it via
 * `stream_options.include_usage` (some serving backends attach usage to the
 * final chunk unasked, others omit it entirely). `_wrapManualStream` logged the
 * row only `if (lastUsageChunk)` — a drained stream with no usage chunk
 * produced NOTHING: no row, no warning, app exit 0.
 *
 * Fix under test (two independent layers):
 *  1. `_injectStreamUsageOption` now covers `provider === "together"` — the
 *     manual wrapper routes the call through `_callWithInjectedStreamUsage`,
 *     making the usage chunk deterministic; the synthetic usage-only terminal
 *     chunk is stripped from the customer-visible iteration via
 *     `_wrapManualStream`'s new `suppressUsageChunk` flag.
 *  2. Fallback: a together stream that drains cleanly with NO usage chunk now
 *     logs an APPROXIMATED row (chars/4 of request messages + accumulated
 *     response, `raw.approximated: true`) instead of nothing.
 *
 * `tp.log` positional args (see `_logManual`): [4]=model, [5]=provider,
 * [6]=inputTokens, [7]=outputTokens, [13]=extras ({usage:{shape,raw}, ...}).
 */
import { describe, test, expect, afterEach, vi } from "vitest";
import { __test__ as enforcerTest, uninstrument } from "../src/enforcer";
import { TokenPolice } from "../src/client";
import { setClient, resetPack } from "../src/state";
import { session } from "../src/context";

const { _wrapManualStream, _injectStreamUsageOption } = enforcerTest as any;

function contentChunk(text: string): any {
  return { choices: [{ index: 0, delta: { content: text } }] };
}
// Terminal chunk as sent by the usage-LESS Together replicas: finish_reason
// set, NO usage key anywhere on the stream.
function finishChunkNoUsage(): any {
  return {
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
  };
}
// Usage-only terminal chunk (what include_usage appends): usage set, choices [].
function usageOnlyChunk(input: number, output: number): any {
  return {
    choices: [],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}
// Usage riding a CONTENT chunk (what some replicas send unasked).
function usageOnContentChunk(text: string, input: number, output: number): any {
  return {
    choices: [{ index: 0, delta: { content: text }, finish_reason: "stop" }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

function fakeStream(chunks: any[]): any {
  const obj: any = {};
  Object.defineProperty(obj, Symbol.asyncIterator, {
    configurable: true,
    writable: true,
    value: async function* () {
      for (const c of chunks) yield c;
    },
  });
  return obj;
}

function makeClient(): { client: TokenPolice; logSpy: any } {
  const client = new TokenPolice({
    apiKey: "tp_sk_test123",
    baseUrl: "http://localhost:59999",
    timeout: 0.1,
    firewall: "dry_run",
  } as any);
  setClient(client);
  const logSpy = vi.spyOn(client, "log").mockImplementation(() => {});
  return { client, logSpy };
}

function rowsFor(logSpy: any, provider: string): any[][] {
  return logSpy.mock.calls.filter((c: any[]) => c[5] === provider);
}

async function drain(iterable: any): Promise<any[]> {
  const out: any[] = [];
  for await (const c of iterable) out.push(c);
  return out;
}

const ARGS = [
  {
    model: "MiniMaxAI/MiniMax-M3",
    messages: [{ role: "user", content: "My laptop from order #12345 isn't turning on." }],
  },
];

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
// 1. The incident shape: together stream with NO usage anywhere.
//    RED pre-fix: zero rows. GREEN: exactly one APPROXIMATED row.
// ═══════════════════════════════════════════════════════════════════
describe("G5-1 fallback: usage-less together stream still logs a row", () => {
  test("drained stream with no usage chunk → one approximated together row", async () => {
    const { logSpy } = makeClient();
    const fake = fakeStream([contentChunk("Try a "), contentChunk("hard reset."), finishChunkNoUsage()]);
    await session({ name: "wf" }, async () => {
      const ret = await _wrapManualStream(fake, "together", ARGS, 0, "agent_step_1", new Date(), performance.now());
      const received = await drain(ret);
      expect(received.length).toBe(3); // customer iteration untouched
    });
    const rows = rowsFor(logSpy, "together");
    expect(rows.length).toBe(1);
    // Tokens approximated (> 0 on both sides — messages and streamed content exist).
    expect(rows[0][6]).toBeGreaterThan(0);
    expect(rows[0][7]).toBeGreaterThan(0);
    // Honestly flagged: raw.approximated → collector prices as 'approximated',
    // never a fake exact 'measured'.
    const extras = rows[0][13];
    expect(extras?.usage?.raw?.approximated).toBe(true);
    // Streaming latency still carried.
    expect(extras?.latency?.is_streaming).toBe(true);
  });

  test("usage present (replica sends it unasked) → real row, NOT approximated (no regression)", async () => {
    const { logSpy } = makeClient();
    const fake = fakeStream([contentChunk("hi"), usageOnContentChunk("!", 120, 34)]);
    await session({ name: "wf" }, async () => {
      const ret = await _wrapManualStream(fake, "together", ARGS, 0, null, new Date());
      await drain(ret);
    });
    const rows = rowsFor(logSpy, "together");
    expect(rows.length).toBe(1);
    expect(rows[0][6]).toBe(120);
    expect(rows[0][7]).toBe(34);
    expect(rows[0][13]?.usage?.raw?.approximated).toBeUndefined();
  });

  test("fallback is together-gated: a usage-less huggingface stream stays row-less (unchanged)", async () => {
    const { logSpy } = makeClient();
    const fake = fakeStream([contentChunk("a"), finishChunkNoUsage()]);
    await session({ name: "wf" }, async () => {
      const ret = await _wrapManualStream(fake, "huggingface", ARGS, 0, null, new Date());
      await drain(ret);
    });
    expect(rowsFor(logSpy, "huggingface").length).toBe(0);
  });

  test("failed stream never triggers the fallback (failure log owns that path)", async () => {
    const { logSpy } = makeClient();
    const err = new Error("provider fail");
    const obj: any = {};
    Object.defineProperty(obj, Symbol.asyncIterator, {
      configurable: true,
      writable: true,
      value: async function* () {
        yield contentChunk("a");
        throw err;
      },
    });
    await expect(
      session({ name: "wf" }, async () => {
        const ret = await _wrapManualStream(obj, "together", ARGS, 0, null, new Date());
        for await (const _c of ret) void _c;
      }),
    ).rejects.toBe(err);
    // No approximated success row — only the failure log (call_outcome).
    const approxRows = rowsFor(logSpy, "together").filter(
      (c) => c[13]?.usage?.raw?.approximated === true,
    );
    expect(approxRows.length).toBe(0);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 2. suppressUsageChunk: the SDK-injected usage-only terminal chunk is
//    tapped for metering but stripped from the customer's iteration.
// ═══════════════════════════════════════════════════════════════════
describe("suppressUsageChunk strips only the synthetic usage-only chunk", () => {
  test("suppress=true: usage-only chunk hidden, row logged with REAL usage", async () => {
    const { logSpy } = makeClient();
    const chunks = [contentChunk("a"), contentChunk("b"), usageOnlyChunk(50, 7)];
    const fake = fakeStream(chunks);
    let received: any[] = [];
    await session({ name: "wf" }, async () => {
      const ret = await _wrapManualStream(
        fake, "together", ARGS, 0, null, new Date(), performance.now(), true,
      );
      received = await drain(ret);
    });
    expect(received).toEqual(chunks.slice(0, 2)); // synthetic chunk invisible
    const rows = rowsFor(logSpy, "together");
    expect(rows.length).toBe(1);
    expect(rows[0][6]).toBe(50);
    expect(rows[0][7]).toBe(7);
    expect(rows[0][13]?.usage?.raw?.approximated).toBeUndefined();
  });

  test("suppress=false (customer asked): usage-only chunk yielded through", async () => {
    makeClient();
    const chunks = [contentChunk("a"), usageOnlyChunk(5, 2)];
    const fake = fakeStream(chunks);
    let received: any[] = [];
    await session({ name: "wf" }, async () => {
      const ret = await _wrapManualStream(fake, "together", ARGS, 0, null, new Date());
      received = await drain(ret);
    });
    expect(received).toEqual(chunks);
  });

  test("suppress=true never strips usage riding a CONTENT chunk", async () => {
    makeClient();
    const chunks = [contentChunk("a"), usageOnContentChunk("tail", 9, 3)];
    const fake = fakeStream(chunks);
    let received: any[] = [];
    await session({ name: "wf" }, async () => {
      const ret = await _wrapManualStream(
        fake, "together", ARGS, 0, null, new Date(), performance.now(), true,
      );
      received = await drain(ret);
    });
    expect(received).toEqual(chunks);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 3. _injectStreamUsageOption now covers together (and still refuses
//    everything else + every non-qualifying body).
// ═══════════════════════════════════════════════════════════════════
describe("_injectStreamUsageOption together gate", () => {
  test("together + stream chat body → injects; restore reverts the body", () => {
    makeClient();
    const body: any = { model: "m", messages: [{ role: "user", content: "x" }], stream: true };
    const handle = _injectStreamUsageOption("together", [body]);
    expect(handle).toBeTruthy();
    expect(body.stream_options).toEqual({ include_usage: true });
    handle.restore();
    expect("stream_options" in body).toBe(false);
  });

  test("customer already asked → null handle, body untouched", () => {
    makeClient();
    const body: any = {
      model: "m",
      messages: [],
      stream: true,
      stream_options: { include_usage: true },
    };
    expect(_injectStreamUsageOption("together", [body])).toBeNull();
  });

  test("non-stream / non-chat / other manual providers → null handle", () => {
    makeClient();
    expect(_injectStreamUsageOption("together", [{ model: "m", messages: [] }])).toBeNull();
    expect(_injectStreamUsageOption("together", [{ model: "m", stream: true }])).toBeNull();
    expect(
      _injectStreamUsageOption("groq", [{ model: "m", messages: [], stream: true }]),
    ).toBeNull();
    expect(
      _injectStreamUsageOption("cerebras", [{ model: "m", messages: [], stream: true }]),
    ).toBeNull();
  });

  test("captureStreamUsage=false disables the together injection", () => {
    const client = new TokenPolice({
      apiKey: "tp_sk_test123",
      baseUrl: "http://localhost:59999",
      timeout: 0.1,
      firewall: "dry_run",
      captureStreamUsage: false,
    } as any);
    setClient(client);
    vi.spyOn(client, "log").mockImplementation(() => {});
    expect(
      _injectStreamUsageOption("together", [{ model: "m", messages: [], stream: true }]),
    ).toBeNull();
  });
});
