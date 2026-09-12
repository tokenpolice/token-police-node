/**
 * Bedrock embedding InvokeModel handler — fail-open telemetry contract.
 *
 * The post-check telemetry setup (session resolution, span ordering, attempt
 * stash) and the manual log must never throw into customer code: any
 * SDK-internal failure degrades to returning the provider's response un-logged.
 * Only a TokenPoliceBlockedError from the enforce-mode pre-flight may propagate.
 *
 * Regression guard for the handler's un-guarded `getCurrentSession()` + no-op
 * catch-rethrow that previously let a telemetry error surface to the caller.
 */
import { describe, test, expect, beforeEach, vi } from "vitest";
import * as contextModule from "../src/context";
import { resetPack, setClient } from "../src/state";
import { TokenPolice } from "../src/client";
import { TokenPoliceBlockedError } from "../src/exceptions";
import { __test__ as enforcerTest } from "../src/enforcer";

function makeClient(firewall: "enforce" | "dry_run" | "off") {
  const client = new TokenPolice({
    apiKey: "tp_sk_test123",
    baseUrl: "http://localhost:59999",
    timeout: 0.1,
    firewall,
    deployment: "daemon",
  });
  setClient(client);
  return client;
}

// A fake Bedrock InvokeModel response body (Titan embeddings shape).
function fakeResponse() {
  const payload = JSON.stringify({ inputTextTokenCount: 7, embedding: [0.1, 0.2] });
  return { body: new TextEncoder().encode(payload) };
}

const ARGS = [{ input: { modelId: "amazon.titan-embed-text-v1", body: "{}" } }];

describe("Bedrock embedding handler — telemetry fail-open", () => {
  beforeEach(() => resetPack());

  test("session resolution throwing → still returns the provider response, no throw", async () => {
    const client = makeClient("enforce");
    vi.spyOn(client, "check").mockResolvedValue({ status: "allowed" } as never);
    // Poison the telemetry path: session resolution throws.
    const spy = vi
      .spyOn(contextModule, "getCurrentSession")
      .mockImplementation(() => {
        throw new Error("boom: session store exploded");
      });

    const original = vi.fn().mockResolvedValue(fakeResponse());
    const resp = await enforcerTest._handleBedrockEmbeddingInvoke({}, original, ARGS);

    expect(spy).toHaveBeenCalled(); // the poisoned seam was actually exercised
    expect(original).toHaveBeenCalledTimes(1); // customer's call still made
    expect(resp).toBeDefined();
    expect((resp as any).body).toBeInstanceOf(Uint8Array); // provider response returned verbatim
    spy.mockRestore();
  });

  test("logging throwing → still returns the provider response, no throw", async () => {
    const client = makeClient("enforce");
    vi.spyOn(client, "check").mockResolvedValue({ status: "allowed" } as never);
    // Real session resolves fine, but the manual log throws.
    const logSpy = vi.spyOn(client, "log").mockImplementation(() => {
      throw new Error("boom: log exploded");
    });

    const original = vi.fn().mockResolvedValue(fakeResponse());
    const resp = await enforcerTest._handleBedrockEmbeddingInvoke({}, original, ARGS);

    expect(original).toHaveBeenCalledTimes(1);
    expect(resp).toBeDefined();
    expect((resp as any).body).toBeInstanceOf(Uint8Array);
    logSpy.mockRestore();
  });

  test("enforce-mode BLOCK still propagates as TokenPoliceBlockedError", async () => {
    const client = makeClient("enforce");
    vi.spyOn(client, "check").mockResolvedValue({
      status: "blocked",
      reason: "budget exceeded",
    } as never);

    const original = vi.fn().mockResolvedValue(fakeResponse());
    await expect(
      enforcerTest._handleBedrockEmbeddingInvoke({}, original, ARGS),
    ).rejects.toBeInstanceOf(TokenPoliceBlockedError);
    // The provider call must NOT happen once blocked.
    expect(original).not.toHaveBeenCalled();
  });

  test("provider call error propagates verbatim, not masked by telemetry", async () => {
    const client = makeClient("enforce");
    vi.spyOn(client, "check").mockResolvedValue({ status: "allowed" } as never);
    // Failure-log path must not swap the customer's error.
    vi.spyOn(client, "log").mockImplementation(() => {
      throw new Error("telemetry noise");
    });

    const providerErr = new Error("AccessDeniedException");
    const original = vi.fn().mockRejectedValue(providerErr);
    await expect(
      enforcerTest._handleBedrockEmbeddingInvoke({}, original, ARGS),
    ).rejects.toBe(providerErr);
  });
});
