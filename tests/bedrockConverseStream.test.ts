/**
 * Node Bedrock `ConverseStreamCommand` runs the pre-flight check but
 * DROPS token logging. Before this fix the wrapper early-returned on
 * ConverseStream (no body-carrying check, no `_logManual`), so streamed Bedrock
 * spend was invisible. The fix removes the early-return (ConverseStream now
 * falls through the shared Bedrock manual path) and adds a dedicated stream tap
 * (`_wrapBedrockConverseStream`) that harvests the terminal
 * `{ metadata: { usage } }` event and logs ONCE on drain — fail-open throughout.
 *
 * Most assertions drive the exported internal `__test__._wrapBedrockConverseStream`
 * directly with a FAKE `{ stream, $metadata }` output (no real
 * `@aws-sdk/client-bedrock-runtime` dependency — mirrors the fake-module pattern
 * in `tests/anthropicStream.test.ts`). The check-parity / enforce-block
 * assertions (13) drive the REAL instrumented `send` path via a fake Bedrock
 * module namespace, because `_runAsyncCheck` is a module-internal closure that a
 * spy on the `__test__` re-export cannot intercept.
 *
 * `tp.log` positional args (see `_logManual`): [4]=model, [5]=provider,
 * [6]=inputTokens, [7]=outputTokens, [8]=cachedTokens.
 */
import { describe, test, expect, afterEach, vi } from "vitest";
import {
  autoInstrument,
  uninstrument,
  _extractUsage,
  __test__ as enforcerTest,
} from "../src/enforcer";
import { init } from "../src/client";
import { TokenPolice } from "../src/client";
import { setClient, resetPack, applySnapshot } from "../src/state";
import { assertStreamedLogPresent } from "./helpers/assertStreamPresence";
import { session } from "../src/context";
import { TokenPoliceBlockedError } from "../src/exceptions";

// ── helpers ─────────────────────────────────────────────────────────
const KWARGS = [
  {
    modelId: "anthropic.claude-3-5-sonnet-20240620-v1:0",
    messages: [{ role: "user", content: [{ text: "hi" }] }],
  },
];

// A terminal Bedrock ConverseStream metadata event carrying usage.
function metaEvent(usage: Record<string, number>): any {
  return { metadata: { usage } };
}
function deltaEvent(text: string): any {
  return { contentBlockDelta: { delta: { text }, contentBlockIndex: 0 } };
}

// Build a fake ConverseStream output `{ stream, $metadata }` from an event list.
function fakeOutput(events: any[], meta?: any): any {
  async function* gen() {
    for (const e of events) yield e;
  }
  return {
    stream: gen(),
    $metadata: meta ?? { httpStatusCode: 200, requestId: "r" },
  };
}

function makeClient(firewall: "enforce" | "dry_run" | "off", deployment = "serverless") {
  const client = new TokenPolice({
    apiKey: "tp_sk_test123",
    baseUrl: "http://localhost:59999",
    timeout: 0.1,
    firewall,
    deployment,
  } as any);
  setClient(client);
  const logSpy = vi.spyOn(client, "log").mockImplementation(() => {});
  return { client, logSpy };
}

// Only the manual bedrock /log rows (arg[5] === provider).
function bedrockRows(logSpy: any): any[][] {
  return logSpy.mock.calls.filter((c: any[]) => c[5] === "bedrock");
}

// Drive the wrapper + fully drain it inside a session (so _logManual's
// getCurrentSession/getClient resolve). Returns the received chunks.
async function wrapAndDrain(fakeOut: any, kwargs = KWARGS): Promise<any[]> {
  return await session({ name: "wf" }, async () => {
    const wrapped = enforcerTest._wrapBedrockConverseStream(
      fakeOut,
      "bedrock",
      kwargs,
      0,
      null,
      new Date(),
    );
    const received: any[] = [];
    for await (const chunk of wrapped.stream) received.push(chunk);
    return received;
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
// Assertion 1 — early-return removed (source-text guard).
// ═══════════════════════════════════════════════════════════════════
describe("A1: early-return removed, non-Converse guard intact", () => {
  test("no ConverseStreamCommand early-return; non-Converse guard byte-unchanged", async () => {
    const fs = await import("node:fs/promises");
    const src = await fs.readFile(new URL("../src/enforcer.ts", import.meta.url), "utf8");
    expect(src).not.toMatch(/if \(cmdName === "ConverseStreamCommand"\)/);
    expect(src).toContain(
      'if (cmdName !== "ConverseCommand" && cmdName !== "ConverseStreamCommand") {',
    );
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 3/4/5 — FinOps fix: log once with correct tokens/model/cache.
// ═══════════════════════════════════════════════════════════════════
describe("A3/4/5: FinOps log-once with correct tokens", () => {
  test("3: exactly one bedrock row with input=42, output=17", async () => {
    const { logSpy } = makeClient("dry_run");
    await wrapAndDrain(fakeOutput([deltaEvent("hello"), metaEvent({ inputTokens: 42, outputTokens: 17 })]));
    // T-N3 / G4 presence: usage-bearing mock MUST yield exactly one row with tokens > 0.
    const rows = assertStreamedLogPresent(logSpy, "bedrock");
    expect(rows[0][6]).toBe(42);
    expect(rows[0][7]).toBe(17);
  });

  test("4: model read from kwargs.modelId", async () => {
    const { logSpy } = makeClient("dry_run");
    await wrapAndDrain(fakeOutput([metaEvent({ inputTokens: 1, outputTokens: 1 })]));
    expect(bedrockRows(logSpy)[0][4]).toBe(
      "anthropic.claude-3-5-sonnet-20240620-v1:0",
    );
  });

  test("5: cacheReadInputTokens → cachedTokens; cacheWriteInputTokens ignored", async () => {
    const { logSpy } = makeClient("dry_run");
    await wrapAndDrain(
      fakeOutput([
        metaEvent({
          inputTokens: 100,
          outputTokens: 20,
          cacheReadInputTokens: 9,
          cacheWriteInputTokens: 7, // must NOT be added anywhere
        }),
      ]),
    );
    const row = bedrockRows(logSpy)[0];
    expect(row[8]).toBe(9); // cachedTokens
    expect(row[6]).toBe(100); // inputTokens unaffected by cacheWrite
    // cacheWrite (7) appears in no numeric token slot
    expect([row[6], row[7], row[8]]).not.toContain(7);
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 6 — exactly-once / zero-row on no usage.
// ═══════════════════════════════════════════════════════════════════
describe("A6: exactly-once / zero-row-on-no-usage", () => {
  test("6(a): single drain → exactly one row", async () => {
    const { logSpy } = makeClient("dry_run");
    await wrapAndDrain(fakeOutput([deltaEvent("a"), deltaEvent("b"), metaEvent({ inputTokens: 5, outputTokens: 6 })]));
    expect(bedrockRows(logSpy).length).toBe(1);
  });

  test("6(c): no terminal usage event → ZERO rows (no phantom zero-usage row)", async () => {
    const { logSpy } = makeClient("dry_run");
    const received = await wrapAndDrain(fakeOutput([deltaEvent("a"), deltaEvent("b")]));
    expect(received.length).toBe(2); // chunks still delivered
    expect(bedrockRows(logSpy).length).toBe(0);
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 7 — stream integrity: same N objects, order, identity.
// ═══════════════════════════════════════════════════════════════════
describe("A7: stream integrity (order + reference identity)", () => {
  test("customer receives the SAME N objects in order, terminal event NOT stripped", async () => {
    const { logSpy } = makeClient("dry_run");
    const source = [
      { messageStart: { role: "assistant" } },
      deltaEvent("x"),
      deltaEvent("y"),
      metaEvent({ inputTokens: 3, outputTokens: 4 }),
    ];
    const received = await wrapAndDrain(fakeOutputFrom(source));
    expect(received.length).toBe(source.length);
    for (let i = 0; i < source.length; i++) {
      expect(Object.is(received[i], source[i])).toBe(true);
    }
    // terminal metadata event yielded unchanged (last received === terminal)
    expect(received[received.length - 1]).toBe(source[source.length - 1]);
    expect(bedrockRows(logSpy).length).toBe(1);
  });
});

// A variant of fakeOutput that yields the EXACT same object references passed in.
function fakeOutputFrom(events: any[]): any {
  async function* gen() {
    for (const e of events) yield e;
  }
  return { stream: gen(), $metadata: { httpStatusCode: 200 } };
}

// ═══════════════════════════════════════════════════════════════════
// Assertion 8 — fail-open on tap/log throw.
// ═══════════════════════════════════════════════════════════════════
describe("A8: fail-open on tap/log throw", () => {
  test("client.log throws → customer for-await completes with ALL chunks, no escape", async () => {
    const { client } = makeClient("dry_run");
    vi.spyOn(client, "log").mockImplementation(() => {
      throw new Error("boom");
    });
    const source = [deltaEvent("a"), deltaEvent("b"), metaEvent({ inputTokens: 1, outputTokens: 1 })];
    let received: any[] = [];
    await expect(
      (async () => {
        received = await wrapAndDrain(fakeOutputFrom(source));
      })(),
    ).resolves.toBeUndefined();
    expect(received.length).toBe(3);
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 9 — provider error re-thrown + failure log, no success row.
// ═══════════════════════════════════════════════════════════════════
describe("A9: provider error mid-stream re-thrown with failure log", () => {
  test("re-throws the SAME error; a failure log emitted; no usage row", async () => {
    const { client, logSpy } = makeClient("dry_run");
    const providerErr = new Error("provider fail");
    async function* boom() {
      yield deltaEvent("a");
      throw providerErr;
    }
    const fakeOut = { stream: boom(), $metadata: { httpStatusCode: 500 } };
    await expect(
      session({ name: "wf" }, async () => {
        const wrapped = enforcerTest._wrapBedrockConverseStream(
          fakeOut,
          "bedrock",
          KWARGS,
          0,
          null,
          new Date(),
        );
        for await (const _c of wrapped.stream) {
          void _c;
        }
      }),
    ).rejects.toThrow("provider fail");
    // No SUCCESS usage row (no metadata.usage was ever seen).
    const success = bedrockRows(logSpy).filter(
      (c) => (c[6] as number) > 0 || (c[7] as number) > 0,
    );
    expect(success.length).toBe(0);
    // A failure log WAS emitted via _emitCallFailureLog (call_outcome carried).
    const failing = logSpy.mock.calls.filter((c: any[]) => {
      const extra = c[c.length - 1];
      return extra && typeof extra === "object" && (extra as any).call_outcome;
    });
    expect(failing.length).toBeGreaterThanOrEqual(1);
    void client;
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 10 — object identity + $metadata preserved.
// ═══════════════════════════════════════════════════════════════════
describe("A10: output object identity preserved", () => {
  test("returns SAME object; $metadata unchanged; .stream still async-iterable", async () => {
    makeClient("dry_run");
    const fakeOut = fakeOutput([metaEvent({ inputTokens: 1, outputTokens: 1 })]);
    const meta = fakeOut.$metadata;
    await session({ name: "wf" }, async () => {
      const wrapped = enforcerTest._wrapBedrockConverseStream(
        fakeOut,
        "bedrock",
        KWARGS,
        0,
        null,
        new Date(),
      );
      expect(Object.is(wrapped, fakeOut)).toBe(true);
      expect(Object.is(wrapped.$metadata, meta)).toBe(true);
      expect(typeof wrapped.stream[Symbol.asyncIterator]).toBe("function");
      // drain to flush
      for await (const _c of wrapped.stream) void _c;
    });
  });

  test("non-stream output (no .stream) passed through untouched", () => {
    makeClient("dry_run");
    const bare = { $metadata: { httpStatusCode: 200 } };
    const out = enforcerTest._wrapBedrockConverseStream(bare, "bedrock", KWARGS, 0, null, new Date());
    expect(Object.is(out, bare)).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 11 — _chunkHasUsage bedrock unit.
// ═══════════════════════════════════════════════════════════════════
describe("A11: _chunkHasUsage bedrock branch", () => {
  test("metadata.usage true; delta false; empty false", () => {
    expect(enforcerTest._chunkHasUsage("bedrock", { metadata: { usage: { inputTokens: 1 } } })).toBe(true);
    expect(enforcerTest._chunkHasUsage("bedrock", { contentBlockDelta: { delta: { text: "x" } } })).toBe(false);
    expect(enforcerTest._chunkHasUsage("bedrock", {})).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 12 — _extractUsage bedrock-from-metadata + non-stream regression.
// ═══════════════════════════════════════════════════════════════════
describe("A12: _extractUsage bedrock reads metadata.usage ?? usage", () => {
  test("stream shape: metadata.usage", () => {
    const u = _extractUsage(
      "bedrock",
      { metadata: { usage: { inputTokens: 12, outputTokens: 34, cacheReadInputTokens: 5 } } },
      [{ modelId: "m" }],
    );
    expect(u).toEqual({ model: "m", inputTokens: 12, outputTokens: 34, cachedTokens: 5 });
  });

  test("non-stream shape: top-level usage still extracts identically (regression)", () => {
    const u = _extractUsage(
      "bedrock",
      { usage: { inputTokens: 12, outputTokens: 34, cacheReadInputTokens: 5 } },
      [{ modelId: "m" }],
    );
    expect(u).toEqual({ model: "m", inputTokens: 12, outputTokens: 34, cachedTokens: 5 });
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 17 — lazy passthrough / no buffering (interleave).
// ═══════════════════════════════════════════════════════════════════
describe("A17: lazy passthrough (no buffering)", () => {
  test("pull_i → recv_i strictly interleaved (source advanced exactly one ahead)", async () => {
    makeClient("dry_run");
    const events = [deltaEvent("a"), deltaEvent("b"), metaEvent({ inputTokens: 1, outputTokens: 1 })];
    const pulls: number[] = [];
    const recvs: number[] = [];
    async function* source() {
      for (let i = 0; i < events.length; i++) {
        pulls.push(i); // record when the wrapper asks for chunk i
        yield events[i];
      }
    }
    const fakeOut = { stream: source(), $metadata: {} };
    await session({ name: "wf" }, async () => {
      const wrapped = enforcerTest._wrapBedrockConverseStream(fakeOut, "bedrock", KWARGS, 0, null, new Date());
      let idx = 0;
      for await (const _c of wrapped.stream) {
        // At the moment recv_i is about to be recorded the source has been
        // advanced exactly one chunk ahead: pulls.length === recvs.length + 1.
        expect(pulls.length).toBe(recvs.length + 1);
        recvs.push(idx++);
        void _c;
      }
    });
    expect(pulls).toEqual([0, 1, 2]);
    expect(recvs).toEqual([0, 1, 2]);
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 13/14 — REAL instrumented send path: check parity + enforce block
// + no regression to non-bedrock streaming + non-Converse pass-through.
// ═══════════════════════════════════════════════════════════════════
// Build a fake @aws-sdk/client-bedrock-runtime namespace. `send` is the ORIGINAL
// the wrapper calls; the Command classes carry the request `input`.
function makeFakeBedrock(streamEvents: any[]) {
  const sendCalls: Array<{ thisArg: any; command: any }> = [];
  class BedrockRuntimeClient {
    async send(command: any): Promise<any> {
      sendCalls.push({ thisArg: this, command });
      async function* gen() {
        for (const e of streamEvents) yield e;
      }
      return { stream: gen(), $metadata: { httpStatusCode: 200, requestId: "r" } };
    }
  }
  class ConverseStreamCommand {
    input: any;
    constructor(input: any) {
      this.input = input;
    }
  }
  class ConverseCommand {
    input: any;
    constructor(input: any) {
      this.input = input;
    }
  }
  return { ns: { BedrockRuntimeClient, ConverseStreamCommand, ConverseCommand }, sendCalls };
}

const BLOCK_SNAPSHOT = {
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
      selector: { match: null, group_by: [] },
    },
  ],
};

describe("A13/14: real instrumented send path", () => {
  test("13(b): ConverseStream runs the body-carrying check with provider 'bedrock' + logs once", async () => {
    const { ns } = makeFakeBedrock([deltaEvent("hi"), metaEvent({ inputTokens: 42, outputTokens: 17 })]);
    const client = init({
      apiKey: "tp_sk_test",
      deployment: "serverless",
      firewall: "dry_run",
      instrumentModules: { bedrock: ns },
    } as any);
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({ status: "allowed", fail_open: false } as any);
    const logSpy = vi.spyOn(client, "log").mockImplementation(() => {});

    const c = new ns.BedrockRuntimeClient();
    await session({ name: "wf" }, async () => {
      const out: any = await (c as any).send(new ns.ConverseStreamCommand(KWARGS[0]));
      for await (const _c of out.stream) void _c;
    });

    // Pre-flight check ran with provider "bedrock" (arg index 7). Under the OLD
    // early-return the check ran as `_runAsyncCheck()` with provider undefined.
    expect(checkSpy).toHaveBeenCalledTimes(1);
    expect(checkSpy.mock.calls[0][7]).toBe("bedrock");
    // FinOps row logged once via the stream wrapper.
    const rows = logSpy.mock.calls.filter((cc: any[]) => cc[5] === "bedrock" && cc[6] === 42);
    expect(rows.length).toBe(1);
    expect(rows[0][7]).toBe(17);
  });

  test("13(a): enforce-mode verified BLOCK throws TokenPoliceBlockedError on the ConverseStream path", async () => {
    const { ns, sendCalls } = makeFakeBedrock([metaEvent({ inputTokens: 1, outputTokens: 1 })]);
    const client = init({
      apiKey: "tp_sk_test",
      deployment: "daemon",
      firewall: "enforce",
      instrumentModules: { bedrock: ns },
    } as any);
    applySnapshot(BLOCK_SNAPSHOT);
    vi.spyOn(client, "check").mockResolvedValue({ status: "blocked", reason: "x" } as any);
    vi.spyOn(client, "log").mockImplementation(() => {});

    const c = new ns.BedrockRuntimeClient();
    await expect(
      session({ name: "wf" }, async () => {
        await (c as any).send(new ns.ConverseStreamCommand(KWARGS[0]));
      }),
    ).rejects.toBeInstanceOf(TokenPoliceBlockedError);
    // Blocked BEFORE the provider call — original send never invoked.
    expect(sendCalls.length).toBe(0);
  });

  test("13/6(b): non-streaming ConverseCommand still logs exactly once", async () => {
    // ConverseCommand output is a plain non-stream response with top-level usage.
    const sendCalls: any[] = [];
    class BedrockRuntimeClient {
      async send(command: any): Promise<any> {
        sendCalls.push(command);
        return {
          output: { message: { role: "assistant", content: [{ text: "ok" }] } },
          stopReason: "end_turn",
          usage: { inputTokens: 11, outputTokens: 22 },
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
    const ns = { BedrockRuntimeClient, ConverseCommand };
    const client = init({
      apiKey: "tp_sk_test",
      deployment: "serverless",
      firewall: "dry_run",
      instrumentModules: { bedrock: ns },
    } as any);
    vi.spyOn(client, "check").mockResolvedValue({ status: "allowed", fail_open: false } as any);
    const logSpy = vi.spyOn(client, "log").mockImplementation(() => {});
    const c = new BedrockRuntimeClient();
    await session({ name: "wf" }, async () => {
      await (c as any).send(new ConverseCommand(KWARGS[0]));
    });
    const rows = logSpy.mock.calls.filter((cc: any[]) => cc[5] === "bedrock");
    expect(rows.length).toBe(1);
    expect(rows[0][6]).toBe(11);
    expect(rows[0][7]).toBe(22);
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 14 — non-bedrock streaming does NOT enter the bedrock branch.
// ═══════════════════════════════════════════════════════════════════
describe("A14: non-bedrock streaming unaffected", () => {
  test("an OpenAI-shaped async-iterable (top-level, no .stream) is not touched by the bedrock wrapper", async () => {
    makeClient("dry_run");
    // OpenAI-shaped streaming result: the RESULT itself is async-iterable, and
    // has no `.stream`. The bedrock branch gate is `provider==="bedrock" &&
    // result?.stream` — assert the guard would be false for this shape.
    async function* openaiStream() {
      yield { choices: [{ delta: { content: "hi" } }] };
    }
    const result: any = openaiStream();
    // No `.stream` property → bedrock branch guard false.
    expect(result?.stream).toBeUndefined();
    // And driving the bedrock wrapper on a bare async-iterable (no .stream)
    // returns it untouched (pass-through), never wrapping it.
    const out = enforcerTest._wrapBedrockConverseStream(result, "bedrock", KWARGS, 0, null, new Date());
    expect(Object.is(out, result)).toBe(true);
  });
});
