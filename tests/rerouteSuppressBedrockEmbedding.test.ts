/**
 * On the Bedrock-embedding InvokeModel path the pre-flight `/check` is fed
 * a THROWAWAY `{ model }` body; the real InvokeModel re-reads `args` unchanged, so
 * a reroute swap could never reach the provider. The handler now passes
 * `canReroute=false` (the 4th arg of `_runAsyncCheck`) so reroute is suppressed
 * (no body mutation, no phantom `_tp_routing` stash) while BLOCK/entity-block on
 * the same path still enforces. This test drives `_runAsyncCheck` directly with the
 * exact calling convention the Bedrock handler now uses.
 *
 * Mirrors token-police-python/tests/test_reroute_suppress_bedrock_embedding.py.
 */
import { describe, test, expect, beforeEach, vi } from "vitest";
import { applySnapshot, resetPack, setClient } from "../src/state";
import { session } from "../src/context";
import { TokenPolice } from "../src/client";
import { TokenPoliceBlockedError } from "../src/exceptions";
import { __test__ as enforcerTest } from "../src/enforcer";

// Same-provider (bedrock) reroute so the cross-provider guard does NOT fire —
// this isolates the canReroute gate as the sole reason a swap is suppressed.
const BEDROCK_REROUTE_SNAPSHOT = {
  schema_version: 1, type: "snapshot", version: 1,
  tenant_id: "t", project_id: "p", ttl_seconds: 600, loop_blocks: [],
  directives: [
    {
      id: "rr", kind: "REROUTE", mode: "enforce", priority: 10,
      selector: { match: null, group_by: [] },
      reroute: { from: {}, to: { provider: "bedrock", model: "amazon.titan-embed-text-v2:0" } },
    },
  ],
};

const BLOCK_SNAPSHOT = {
  schema_version: 1, type: "snapshot", version: 1,
  tenant_id: "t", project_id: "p", ttl_seconds: 600, loop_blocks: [],
  directives: [
    {
      id: "b1", kind: "UNCONDITIONAL_BLOCK", mode: "enforce", priority: 10,
      selector: { match: null, group_by: [] },
    },
  ],
};

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

describe("Suppress reroute on Bedrock embedding throwaway-body path", () => {
  beforeEach(() => resetPack());

  test("A1: Bedrock embedding + matching REROUTE, canReroute=false → body model unchanged, no _tp_routing (State A)", async () => {
    const client = makeClient("enforce");
    applySnapshot(BEDROCK_REROUTE_SNAPSHOT);
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({ status: "allowed" } as never);
    const body: Record<string, any> = { model: "amazon.titan-embed-text-v1" };
    const s = await session({ name: "t" }, async (sess) => {
      await enforcerTest._runAsyncCheck(body, "bedrock", { kind: "embedding" }, false);
      return sess;
    });
    expect(body.model).toBe("amazon.titan-embed-text-v1"); // suppressed — unchanged
    expect((s.metadata as Record<string, unknown> | undefined)?._tp_routing).toBeUndefined();
    checkSpy.mockRestore();
  });

  test("A1 (State B): reroute in /check result, canReroute=false → body unchanged, no _tp_routing", async () => {
    const client = makeClient("enforce");
    // no pack → State B fallback
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({
      status: "allowed",
      reroute: { mode: "enforce", model: "amazon.titan-embed-text-v2:0", provider: "bedrock", rule_id: "rr" },
    } as never);
    const body: Record<string, any> = { model: "amazon.titan-embed-text-v1" };
    const s = await session({ name: "t" }, async (sess) => {
      await enforcerTest._runAsyncCheck(body, "bedrock", { kind: "embedding" }, false);
      return sess;
    });
    expect(body.model).toBe("amazon.titan-embed-text-v1");
    expect((s.metadata as Record<string, unknown> | undefined)?._tp_routing).toBeUndefined();
    checkSpy.mockRestore();
  });

  test("A3: BLOCK on the same path STILL throws with canReroute=false (State A verified block)", async () => {
    const client = makeClient("enforce");
    applySnapshot(BLOCK_SNAPSHOT);
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({ status: "blocked", reason: "x" } as never);
    await expect(
      enforcerTest._runAsyncCheck({ model: "amazon.titan-embed-text-v1" }, "bedrock", { kind: "embedding" }, false),
    ).rejects.toBeInstanceOf(TokenPoliceBlockedError);
    checkSpy.mockRestore();
  });

  test("A3: BLOCK on the same path STILL throws with canReroute=false (State B block)", async () => {
    const client = makeClient("enforce");
    // no pack → State B fallback
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({ status: "blocked", reason: "x" } as never);
    await expect(
      enforcerTest._runAsyncCheck({ model: "amazon.titan-embed-text-v1" }, "bedrock", { kind: "embedding" }, false),
    ).rejects.toBeInstanceOf(TokenPoliceBlockedError);
    checkSpy.mockRestore();
  });

  test("A3b: a REROUTE local decision but /check says blocked STILL throws even with canReroute=false", async () => {
    const client = makeClient("enforce");
    applySnapshot(BEDROCK_REROUTE_SNAPSHOT);
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({ status: "blocked", reason: "x" } as never);
    await expect(
      enforcerTest._runAsyncCheck({ model: "amazon.titan-embed-text-v1" }, "bedrock", { kind: "embedding" }, false),
    ).rejects.toBeInstanceOf(TokenPoliceBlockedError);
    checkSpy.mockRestore();
  });

  test("A5: genuine path with DEFAULT canReroute (3 args) STILL reroutes + stashes _tp_routing (anti-over-suppression, State A)", async () => {
    const client = makeClient("enforce");
    applySnapshot(BEDROCK_REROUTE_SNAPSHOT);
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({ status: "allowed" } as never);
    const body: Record<string, any> = { model: "amazon.titan-embed-text-v1" };
    const s = await session({ name: "t" }, async (sess) => {
      await enforcerTest._runAsyncCheck(body, "bedrock"); // default canReroute=true
      return sess;
    });
    expect(body.model).toBe("amazon.titan-embed-text-v2:0"); // swapped
    expect((s.metadata as Record<string, unknown> | undefined)?._tp_routing).toBeDefined();
    checkSpy.mockRestore();
  });

  test("A5 (State B): default canReroute + reroute in /check result STILL swaps model", async () => {
    const client = makeClient("enforce");
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({
      status: "allowed",
      reroute: { mode: "enforce", model: "amazon.titan-embed-text-v2:0", provider: "bedrock", rule_id: "rr" },
    } as never);
    const body: Record<string, any> = { model: "amazon.titan-embed-text-v1" };
    const s = await session({ name: "t" }, async (sess) => {
      await enforcerTest._runAsyncCheck(body, "bedrock"); // default canReroute=true
      return sess;
    });
    expect(body.model).toBe("amazon.titan-embed-text-v2:0");
    expect((s.metadata as Record<string, unknown> | undefined)?._tp_routing).toBeDefined();
    checkSpy.mockRestore();
  });

  test("A10 golden rule: canReroute=false never throws on an allowed decision", async () => {
    const client = makeClient("enforce");
    applySnapshot(BEDROCK_REROUTE_SNAPSHOT);
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({ status: "allowed" } as never);
    await expect(
      enforcerTest._runAsyncCheck({ model: "amazon.titan-embed-text-v1" }, "bedrock", { kind: "embedding" }, false),
    ).resolves.toBeUndefined();
    checkSpy.mockRestore();
  });
});
