/**
 * `firewall: "off"` = telemetry only must actually deliver full telemetry.
 *
 * Before this fix, `init({ firewall:"off" })` skipped `autoInstrument`, so every
 * manual-tap-only provider (Vercel AI SDK, Bedrock, Cohere, @google/genai,
 * streaming-usage taps) went dark in `off` — no `/log` at all — contradicting the
 * documented "telemetry only" promise. The fix opens the install gate so taps are
 * installed in `off` too and run LOG-ONLY: the enforcer choke point
 * (`_runAsyncCheck`) short-circuits on `firewall === "off"` BEFORE any `/check`,
 * block, or reroute, so a tapped call in `off` emits exactly one `/log`, makes
 * zero `/check`, never reroutes, and never throws — even with a would-block rule.
 *
 * Harness: real `../src/state` (no module mock) so the runnable SSE-off check can
 * drive the real `init`. Behavioral tests inject a fake client via `setClient`
 * with spied `check`/`log`; with no cached pack the enforcer takes the inline
 * `/check` path in every mode, so the spied `check` drives the verdict directly
 * (mirrors tests/aiSdk.test.ts).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { autoInstrument, uninstrument } from "../src/enforcer";
import { init } from "../src/client";
import { setClient } from "../src/state";
import { TokenPoliceBlockedError } from "../src/exceptions";

const logged: any[] = [];

function makeClient(overrides: Record<string, any> = {}) {
  const base: Record<string, any> = {
    enforce: false,
    deployment: undefined,
    logErrors: false,
    check: vi.fn(async () => ({ status: "allowed" })),
    log: (...args: any[]) => {
      logged.push(args);
    },
    ...overrides,
  };
  if (base.firewall === undefined) {
    base.firewall =
      base.enforce === true ? "enforce" : base.enforce === false ? "off" : "dry_run";
  }
  return base;
}

/** LanguageModel V2 (ai v5) — flat camelCase usage numbers (mirror aiSdk.test.ts). */
class FakeModelV2 {
  readonly specificationVersion = "v2";
  readonly provider = "minimax.messages";
  readonly modelId: string;
  config = { baseURL: "https://api.minimax.io/anthropic/v1" };
  constructor(modelId: string) {
    this.modelId = modelId;
  }
  async doGenerate(_options: any): Promise<any> {
    return {
      content: [{ type: "text", text: "hello there" }],
      finishReason: "stop",
      usage: { inputTokens: 100, outputTokens: 20, totalTokens: 120, cachedInputTokens: 30 },
    };
  }
  async doStream(_options: any): Promise<any> {
    return { stream: new ReadableStream({ start: (c) => c.close() }) };
  }
}

async function flushLogs(): Promise<void> {
  await new Promise((r) => setTimeout(r, 0));
}

describe("Firewall 'off' installs taps and logs (log-only)", () => {
  beforeEach(() => {
    logged.length = 0;
  });

  afterEach(() => {
    uninstrument();
  });

  // Assertions 1, 4, 6, 8, 12: off installs the tap; a tapped call emits exactly
  // ONE /log with correct usage, makes ZERO /check, never reroutes, never throws
  // — even though the client's check would return a verified block.
  it("off: manual-tap call logs exactly once, zero /check, no throw, no reroute (would-block present)", async () => {
    const client = makeClient({
      firewall: "off",
      check: vi.fn(async () => ({ status: "blocked", reason: "Budget exceeded" })),
    });
    setClient(client as any);
    autoInstrument({ aiSdkProviders: [new FakeModelV2("probe")] });
    const model = new FakeModelV2("MiniMax-M2");

    const res = await model.doGenerate({
      prompt: [{ role: "user", content: [{ type: "text", text: "hi" }] }],
    });
    await flushLogs();

    // Golden Rule: customer result untouched, no throw.
    expect(res.finishReason).toBe("stop");
    // Assertion 6: zero /check in off (never call the server pre-flight).
    expect((client.check as any).mock.calls.length).toBe(0);
    // Assertions 4 & 12: exactly one /log with correct usage (no double-log).
    expect(logged.length).toBe(1);
    const args = logged[0];
    // Assertion 8: model unchanged (no reroute could occur — /check never ran).
    expect(args[4]).toBe("MiniMax-M2");
    expect(args[5]).toBe("minimax");
    expect(args[6]).toBe(70); // input minus cached
    expect(args[7]).toBe(20);
    expect(args[8]).toBe(30); // cached
    const extra = args[args.length - 1];
    expect(extra.usage.shape).toBe("vercel_ai");
  });

  // Assertion 1 (install gate opens in off): real init({ firewall:"off" }) with
  // aiSdkProviders MUST patch the provider prototype (RED on the old gate that
  // skipped autoInstrument in off). Routes through init — the actual gate — not
  // a direct autoInstrument call.
  it("off: real init installs the tap (patches provider prototype via the gate)", () => {
    const before = FakeModelV2.prototype.doGenerate;
    init({
      apiKey: "tp_sk_test_off",
      firewall: "off",
      instrumentModules: { aiSdkProviders: [new FakeModelV2("probe")] } as any,
    });
    expect(FakeModelV2.prototype.doGenerate).not.toBe(before);
  });

  // Assertion 3 (RUNNABLE): after real init({ firewall:"off" }) no SSE stream is
  // opened — _streamClient stays null (its declared default) even though taps
  // now install in off.
  it("off: real init opens NO SSE stream (_streamClient === null)", () => {
    const client = init({ apiKey: "tp_sk_test_off", firewall: "off", deployment: "daemon" });
    expect(client._streamClient).toBe(null);
  });

  // Assertion 9 (regression): enforce STILL throws on a verified block AND never
  // reaches the provider. Spy the prototype BEFORE autoInstrument so it is the
  // captured original; on a block the wrapper throws before calling it.
  it("enforce: verified block still throws TokenPoliceBlockedError and provider called 0×", async () => {
    const client = makeClient({
      firewall: "enforce",
      check: vi.fn(async () => ({ status: "blocked", reason: "Budget exceeded" })),
    });
    setClient(client as any);
    const originalSpy = vi.spyOn(FakeModelV2.prototype, "doGenerate");
    autoInstrument({ aiSdkProviders: [new FakeModelV2("probe")] });
    const model = new FakeModelV2("MiniMax-M2");

    await expect(model.doGenerate({ prompt: [] })).rejects.toBeInstanceOf(
      TokenPoliceBlockedError,
    );
    expect(originalSpy).toHaveBeenCalledTimes(0);
    originalSpy.mockRestore();
  });

  // Assertion 10 (regression): dry_run still runs /check and suppresses the block
  // — check IS called, no throw, model unchanged.
  it("dry_run: still runs /check on a block but suppresses the action", async () => {
    const client = makeClient({
      firewall: "dry_run",
      check: vi.fn(async () => ({ status: "blocked", reason: "Budget exceeded" })),
    });
    setClient(client as any);
    autoInstrument({ aiSdkProviders: [new FakeModelV2("probe")] });
    const model = new FakeModelV2("MiniMax-M2");

    const res = await model.doGenerate({ prompt: [] });
    await flushLogs();

    expect(res.finishReason).toBe("stop"); // no throw
    expect((client.check as any).mock.calls.length).toBeGreaterThanOrEqual(1);
    expect(logged.length).toBe(1);
    expect(logged[0][4]).toBe("MiniMax-M2"); // model unchanged (block suppressed)
  });
});
