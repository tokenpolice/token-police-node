/**
 * A snapshot that applies successfully but arrives WITHOUT a numeric `id:` line
 * must NOT be discarded. The dispatch snapshot branch mirrors the delta branch:
 * advance the reconnect cursor only when the apply succeeds AND the frame carries
 * a numeric id; when the apply fails, invalidate the cache and drop the cursor so
 * the next reconnect pulls a fresh snapshot; when the apply succeeds but there is
 * no numeric id, keep the pack healthy and leave the cursor untouched.
 *
 * Sibling: token-police-python/tests/test_stream_snapshot_id.py pins the same
 * cases.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { StreamClient } from "../src/stream";
import * as state from "../src/state";

const SNAPSHOT = { version: 1, tenant_id: "t1", project_id: "p1", directives: [], loop_blocks: [] };

function makeClient(): StreamClient {
  return new StreamClient({
    baseUrl: "http://localhost:15098",
    apiKey: "tp_sk_test",
    sdkVersion: "test",
    deployment: "daemon",
    clientId: "cid",
    firewall: "enforce",
  });
}

// dispatch + lastEventId are private; reach them the same way the sibling
// terminal-status suite reaches internals.
function dispatch(client: StreamClient, eventType: string, eventId: string | null, data: string): void {
  (client as unknown as { dispatch: (t: string, i: string | null, d: string) => void }).dispatch(
    eventType, eventId, data,
  );
}
function cursorOf(client: StreamClient): number | null {
  return (client as unknown as { lastEventId: number | null }).lastEventId;
}
function setCursor(client: StreamClient, v: number | null): void {
  (client as unknown as { lastEventId: number | null }).lastEventId = v;
}

describe("snapshot dispatch: a missing id must not discard a good snapshot", () => {
  beforeEach(() => state.resetPack());
  afterEach(() => state.resetPack());

  it("ok + numeric id → applies and advances the cursor (happy path)", () => {
    const client = makeClient();
    dispatch(client, "snapshot", "7", JSON.stringify(SNAPSHOT));
    expect(state.isCacheHealthy()).toBe(true);
    expect(state.getPack()).not.toBeNull();
    expect(cursorOf(client)).toBe(7);
  });

  it("ok + NO id → pack stays healthy, cursor not advanced, no invalidate", () => {
    const client = makeClient();
    dispatch(client, "snapshot", null, JSON.stringify(SNAPSHOT));
    expect(state.isCacheHealthy()).toBe(true); // applied pack kept
    expect(state.getPack()).not.toBeNull();
    expect(cursorOf(client)).toBeNull(); // not advanced (no numeric id)
  });

  it("ok + non-numeric id → same: healthy, cursor untouched, no invalidate", () => {
    const client = makeClient();
    dispatch(client, "snapshot", "abc", JSON.stringify(SNAPSHOT));
    expect(state.isCacheHealthy()).toBe(true);
    expect(state.getPack()).not.toBeNull();
    expect(cursorOf(client)).toBeNull();
  });

  it("ok + NO id LEAVES an existing cursor untouched (a missing id is not a failure)", () => {
    const client = makeClient();
    setCursor(client, 3);
    dispatch(client, "snapshot", null, JSON.stringify(SNAPSHOT));
    expect(state.isCacheHealthy()).toBe(true);
    expect(cursorOf(client)).toBe(3);
  });

  it("apply FAILS → invalidate + drop cursor (fresh snapshot next reconnect)", () => {
    const client = makeClient();
    setCursor(client, 5);
    // A snapshot missing tenant/project identity makes applySnapshot return false
    // and poison the cache — the failure path, even with a numeric id present.
    const bad = { version: 1, directives: [], loop_blocks: [] };
    dispatch(client, "snapshot", "9", JSON.stringify(bad));
    expect(state.isCacheHealthy()).toBe(false);
    expect(state.getPack()).toBeNull();
    expect(cursorOf(client)).toBeNull(); // cursor dropped on a real apply failure
  });
});
