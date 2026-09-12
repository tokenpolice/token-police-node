import { describe, it, expect, vi, afterEach } from "vitest";
import { TokenPolice, init } from "../src/client";
import { getClient } from "../src/state";
import { serverless } from "../src/context";
import { protect, uninstrument } from "../src/enforcer";

// Batch of public-surface fixes: CheckResult.failOpen (#13), log()-after-close
// no-op (#10), timeout validation (#9), serverless thenable duck-typing (#11),
// and manual-log routed original_provider derivation (#7).

const API_KEY = "tp_sk_surface_test";

afterEach(() => {
  const c = getClient();
  if (c) c.closeSync();
  vi.restoreAllMocks();
});

// ── #13: CheckResult carries a camelCase `failOpen` mirroring `fail_open` ──
describe("#13 CheckResult.failOpen", () => {
  it("fail-open path: failOpen === fail_open === true", async () => {
    const client = new TokenPolice({
      apiKey: API_KEY,
      baseUrl: "http://localhost:1", // unreachable → forces the catch/fail-open path
      timeout: 0.05,
    });
    const result = await client.check("u");
    expect(result.status).toBe("allowed");
    expect(result.failOpen).toBe(true);
    expect(result.fail_open).toBe(true);
    expect(result.failOpen).toBe(result.fail_open);
  });

  it("blocked path: failOpen === fail_open (both absent for a real server decision)", async () => {
    const client = new TokenPolice({ apiKey: API_KEY, baseUrl: "http://localhost:2" });
    // Return a 429 blocked body with no fail-open marker (a genuine decision).
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ status: "blocked", reason: "budget" }), {
        status: 429,
        headers: { "Content-Type": "application/json" },
      }),
    );
    const result = await client.check("u");
    expect(result.status).toBe("blocked");
    // Neither field is set by a genuine server decision, so they stay equal.
    expect(result.failOpen).toBe(result.fail_open);
  });
});

// ── #10: log() after close() is a silent no-op (fail-open) ──
describe("#10 log() after close()", () => {
  it("issues no POST and does not throw once the client is closed", async () => {
    const client = new TokenPolice({ apiKey: API_KEY, baseUrl: "http://localhost:3" });
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("{}"));

    await client.close();

    expect(() => {
      client.log("u", "free", "wf", "", "gpt-4o", "openai", 10, 5);
    }).not.toThrow();

    // No background /log POST should have been issued after close().
    expect(fetchSpy).not.toHaveBeenCalled();
    // And a subsequent flush resolves cleanly (nothing pending).
    await expect(client.flush()).resolves.toBeUndefined();
  });

  it("log() BEFORE close() still issues a POST (guard is close-scoped)", () => {
    const client = new TokenPolice({ apiKey: API_KEY, baseUrl: "http://localhost:4" });
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("{}"));
    client.log("u", "free", "wf", "", "gpt-4o", "openai", 10, 5);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    client.closeSync();
  });
});

// ── #9: timeout is in SECONDS; invalid values warn + fall back to 2.0 ──
describe("#9 timeout validation", () => {
  it("timeout: 0 warns and falls back to the 2.0s default", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const client = new TokenPolice({ apiKey: API_KEY, timeout: 0 });
    expect(client.timeout).toBe(2.0);
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(String(warnSpy.mock.calls[0][0])).toContain("timeout");
    client.closeSync();
  });

  it("negative / non-finite timeouts warn and fall back to 2.0", () => {
    for (const bad of [-1, Number.NaN, Number.POSITIVE_INFINITY]) {
      const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
      const client = new TokenPolice({ apiKey: API_KEY, timeout: bad });
      expect(client.timeout).toBe(2.0);
      expect(warnSpy).toHaveBeenCalledTimes(1);
      client.closeSync();
      warnSpy.mockRestore();
    }
  });

  // Regression pin: 0.5s was a LAN/sidecar number. Public-internet RTT to the
  // Cloudflare-fronted collector is ~0.44s warm / 0.9-1.5s cold from a distant
  // region, and a `/check` timeout fails open — the old default silently
  // disabled enforcement for distant customers.
  it("the default timeout is 2.0 seconds", () => {
    const client = new TokenPolice({ apiKey: API_KEY });
    expect(client.timeout).toBe(2.0);
    client.closeSync();
  });

  it("timeout: 5 is preserved unchanged with no warning", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const client = new TokenPolice({ apiKey: API_KEY, timeout: 5 });
    expect(client.timeout).toBe(5);
    expect(warnSpy).not.toHaveBeenCalled();
    client.closeSync();
  });

  it("omitted timeout uses the default silently", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const client = new TokenPolice({ apiKey: API_KEY });
    expect(client.timeout).toBe(2.0);
    expect(warnSpy).not.toHaveBeenCalled();
    client.closeSync();
  });
});

// ── #11: serverless() duck-types on `.then`, so foreign thenables get the
// async flush path (not just `instanceof Promise`). ──
describe("#11 serverless thenable duck-typing", () => {
  it("a foreign thenable WITH .finally still awaits flush", async () => {
    const client = init({ apiKey: API_KEY, baseUrl: "http://localhost:5", firewall: "off" });
    const flushSpy = vi.spyOn(client, "flush").mockResolvedValue(undefined);

    // A Bluebird-like thenable: has then + finally but is NOT `instanceof Promise`.
    const foreign = (value: unknown) => {
      const real = Promise.resolve(value);
      return {
        then: (a: any, b: any) => real.then(a, b),
        finally: (cb: any) => real.finally(cb),
      };
    };
    expect(foreign("x") instanceof Promise).toBe(false);

    const handler = serverless(() => foreign("done") as any);
    const out = await handler();
    expect(out).toBe("done");
    expect(flushSpy).toHaveBeenCalledTimes(1);
  });

  it("a bare thenable (only .then) also awaits flush via the fallback", async () => {
    const client = init({ apiKey: API_KEY, baseUrl: "http://localhost:6", firewall: "off" });
    const flushSpy = vi.spyOn(client, "flush").mockResolvedValue(undefined);

    const bare = (value: unknown) => ({
      then: (onF: any) => Promise.resolve(value).then(onF),
    });
    expect(bare("y") instanceof Promise).toBe(false);

    const handler = serverless(() => bare("ok") as any);
    const out = await handler();
    expect(out).toBe("ok");
    expect(flushSpy).toHaveBeenCalledTimes(1);
  });

  it("a plain sync return still takes the sync path (no flush await)", () => {
    const client = init({ apiKey: API_KEY, baseUrl: "http://localhost:7", firewall: "off" });
    const flushSpy = vi.spyOn(client, "flush");
    const handler = serverless(() => 42);
    expect(handler()).toBe(42);
    expect(flushSpy).not.toHaveBeenCalled();
  });
});

// ── #7: manual-log routed original_provider is the model-slug vendor head ──
class FakeOpenRouterClient {
  async create(params: any): Promise<any> {
    return { model: params.model, usage: { prompt_tokens: 10, completion_tokens: 5 } };
  }
}

describe("#7 manual log routed original_provider", () => {
  afterEach(() => {
    try {
      uninstrument();
    } catch {
      /* ignore */
    }
  });

  async function logExtrasFor(model: string): Promise<Record<string, any> | undefined> {
    const client = init({ apiKey: API_KEY, baseUrl: "http://localhost:8", firewall: "off" });
    const logSpy = vi.spyOn(client, "log").mockImplementation(() => {});
    protect("openrouter-fake", ["prototype"], "create", true, {
      manual: true,
      provider: "openai",
      module: FakeOpenRouterClient,
    });
    const instance = new FakeOpenRouterClient();
    // baseURL carries "openrouter" so the OpenRouter-via-OpenAI-SDK branch fires.
    await instance.create({
      model,
      baseURL: "https://openrouter.ai/api/v1",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(logSpy).toHaveBeenCalledTimes(1);
    const extras = logSpy.mock.calls[0][13] as any;
    return extras?.model_extras;
  }

  it("routed 'vendor/model' slug → original_provider is the vendor head", async () => {
    const modelExtras = await logExtrasFor("anthropic/claude-3-haiku");
    expect(modelExtras).toBeDefined();
    expect(modelExtras!.original_provider).toBe("anthropic");
    expect(modelExtras!.deployment).toBe("routed");
  });

  it("plain (non-routed) model → NO original_provider, deployment still routed", async () => {
    const modelExtras = await logExtrasFor("gpt-4o-mini");
    expect(modelExtras).toBeDefined();
    expect("original_provider" in modelExtras!).toBe(false);
    expect(modelExtras!.deployment).toBe("routed");
  });
});
