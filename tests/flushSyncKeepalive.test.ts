/**
 * Node `flushSync()` is a no-op → telemetry loss on serverless freeze.
 *
 * Fix: the fire-and-forget `/v1/guard/log` POST issued in `_fetch()` attaches
 * `keepalive: true` ONLY on freeze-prone deployments (`serverless`/`edge`) so
 * the runtime can complete the request after the handler returns. Narrowly
 * gated: `/check` (always awaited) and daemon `/log` (long-lived hot path)
 * stay byte-for-byte unchanged. No blocking primitive — Golden Rule preserved.
 *
 * These tests stub global `fetch`, capture every `(url, options)` pair, and
 * assert the gate behaves exactly per the rubric (items 1-9, 12).
 */
import { describe, it, expect, vi, afterEach } from "vitest";
import { TokenPolice } from "../src/client";

type Deployment = "daemon" | "serverless" | "edge";

interface Recorded {
  url: string;
  options: RequestInit;
}

/**
 * Install a stubbed global `fetch` that records every call and resolves with a
 * minimal OK Response. Returns the recording array.
 */
function stubFetch(recorded: Recorded[], reject = false) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, options: RequestInit) => {
      recorded.push({ url, options });
      if (reject) throw new Error("network down");
      return new Response(JSON.stringify({ allowed: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }),
  );
}

function makeClient(deployment: Deployment): TokenPolice {
  return new TokenPolice({
    apiKey: "tp_sk_test123",
    baseUrl: "http://localhost:13099",
    timeout: 0.5,
    deployment,
  });
}

/** Fire a /log POST and wait for the fire-and-forget promise to settle. */
async function logOnce(client: TokenPolice): Promise<void> {
  // C-12: pass an explicit, fixed span_id. An omitted span_id is now
  // synthesized per-call (randomHex16()), which would otherwise make item 6's
  // daemon-vs-serverless payload-parity bodies differ by construction.
  client.log(
    "u", "free", "default", "", "gpt-4o", "openai", 100, 50, 0, {},
    { span_id: "span-keepalive-parity" },
  );
  await client.flush();
}

function logRecord(recorded: Recorded[]): Recorded | undefined {
  return recorded.find((r) => r.url.endsWith("/v1/guard/log"));
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("Keepalive on serverless/edge /log", () => {
  it("item 1: /log uses keepalive under serverless", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient("serverless");
    await logOnce(client);

    const log = logRecord(recorded);
    expect(log).toBeDefined();
    expect((log!.options as RequestInit & { keepalive?: boolean }).keepalive).toBe(true);
  });

  it("item 2: /log uses keepalive under edge", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient("edge");
    await logOnce(client);

    const log = logRecord(recorded);
    expect(log).toBeDefined();
    expect((log!.options as RequestInit & { keepalive?: boolean }).keepalive).toBe(true);
  });

  it("item 3: /log does NOT use keepalive under daemon", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient("daemon");
    await logOnce(client);

    const log = logRecord(recorded);
    expect(log).toBeDefined();
    // keepalive key absent (or falsy) — never true.
    expect((log!.options as RequestInit & { keepalive?: boolean }).keepalive).not.toBe(true);
    expect("keepalive" in (log!.options as object)).toBe(false);
  });

  it("item 4: /check NEVER uses keepalive even under serverless", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient("serverless");
    await client.check("u", "free", "default");

    const check = recorded.find((r) => r.url.endsWith("/v1/guard/check"));
    expect(check).toBeDefined();
    expect((check!.options as RequestInit & { keepalive?: boolean }).keepalive).not.toBe(true);
    expect("keepalive" in (check!.options as object)).toBe(false);
  });

  it("item 5: daemon /log options carry no keepalive key (byte-for-byte preserved)", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient("daemon");
    await logOnce(client);

    const log = logRecord(recorded)!;
    expect(Object.keys(log.options).sort()).toEqual(
      ["body", "headers", "method", "signal"].sort(),
    );
  });

  it("item 6: serverless /log = daemon /log + ONLY keepalive:true (payload integrity)", async () => {
    const daemonRec: Recorded[] = [];
    stubFetch(daemonRec);
    const daemonClient = makeClient("daemon");
    await logOnce(daemonClient);
    const daemonOpts = { ...(logRecord(daemonRec)!.options as Record<string, unknown>) };

    vi.unstubAllGlobals();

    const slsRec: Recorded[] = [];
    stubFetch(slsRec);
    const slsClient = makeClient("serverless");
    await logOnce(slsClient);
    const slsOpts = { ...(logRecord(slsRec)!.options as Record<string, unknown>) };

    // keepalive: true is the SOLE added field.
    expect((slsOpts as { keepalive?: boolean }).keepalive).toBe(true);
    expect("keepalive" in daemonOpts).toBe(false);
    delete (slsOpts as { keepalive?: boolean }).keepalive;

    // The only *behavioral* header difference is X-TP-Deployment-Mode (daemon
    // vs serverless). X-TP-Client-Id is a per-instance UUID (unrelated to the
    // fix) — normalize both before comparing.
    const daemonHeaders = { ...(daemonOpts.headers as Record<string, string>) };
    const slsHeaders = { ...(slsOpts.headers as Record<string, string>) };
    expect(daemonHeaders["X-TP-Deployment-Mode"]).toBe("daemon");
    expect(slsHeaders["X-TP-Deployment-Mode"]).toBe("serverless");
    daemonHeaders["X-TP-Deployment-Mode"] = "<mode>";
    slsHeaders["X-TP-Deployment-Mode"] = "<mode>";
    daemonHeaders["X-TP-Client-Id"] = "<client-id>";
    slsHeaders["X-TP-Client-Id"] = "<client-id>";
    expect(slsHeaders).toEqual(daemonHeaders);

    // method + body identical; signal present in both (AbortSignal instance).
    expect(slsOpts.method).toBe(daemonOpts.method);
    expect(slsOpts.body).toBe(daemonOpts.body);
    expect(slsOpts.signal).toBeInstanceOf(AbortSignal);
    expect(daemonOpts.signal).toBeInstanceOf(AbortSignal);

    // Aside from headers (compared above) and signal (per-call instance), the
    // remaining keys match exactly.
    const stripVolatile = (o: Record<string, unknown>) => {
      const c = { ...o };
      delete c.headers;
      delete c.signal;
      return c;
    };
    expect(stripVolatile(slsOpts)).toEqual(stripVolatile(daemonOpts));
  });

  it("item 9: flushSync() returns undefined and does not throw with pending logs", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient("serverless");
    client.log("u", "free", "default", "", "gpt-4o", "openai", 100, 50, 0);
    // pending log present; flushSync must be a safe no-op (diagnostic only).
    let result: unknown = "sentinel";
    expect(() => {
      result = client.flushSync();
    }).not.toThrow();
    expect(result).toBeUndefined();
    await client.flush();
  });

  it("item 12: /log stays fire-and-forget + swallowed when fetch rejects (serverless)", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded, /* reject */ true);
    const client = makeClient("serverless");

    // log() must not throw even though the underlying fetch rejects.
    expect(() => {
      client.log("u", "free", "default", "", "gpt-4o", "openai", 100, 50, 0);
    }).not.toThrow();

    // flush() over a rejected (swallowed) pending promise resolves cleanly.
    await expect(client.flush()).resolves.toBeUndefined();
    // The rejecting fetch was still issued with keepalive (serverless gate).
    const log = logRecord(recorded);
    expect(log).toBeDefined();
    expect((log!.options as RequestInit & { keepalive?: boolean }).keepalive).toBe(true);
  });
});
