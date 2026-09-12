/**
 * SSE-11 lock-in: abort in connectAndPump's finally must destroy the real
 * undici socket, not just flip `signal.aborted` on a stubbed fetch.
 *
 * A local `http.createServer` counts live GETs. Unhealthy-cache deltas
 * (`state.resetPack()` + `event: delta version:1`) trip the existing
 * forceReconnect path → reconnects at the ~1s floor. If abort is a no-op,
 * peak live climbs with opens. If abort works, peak stays ≤2 and stop()
 * drives live to 0. Real fetch, real timers — no stubbed fetch, no fake timers.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import http from "node:http";
import type { AddressInfo } from "node:net";
import { StreamClient } from "../src/stream";
import * as state from "../src/state";

const POLL_MS = 25;
const SETTLE_MS = 8_000;

type Counts = { live: number; opens: number; closes: number; peak: number };

function snapshot(c: Counts): string {
  return `live=${c.live} peak=${c.peak} opens=${c.opens} closes=${c.closes}`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

async function waitUntil(pred: () => boolean, timeoutMs: number, label: string, c: Counts): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (pred()) return;
    await sleep(POLL_MS);
  }
  throw new Error(`${label} timed out after ${timeoutMs}ms: ${snapshot(c)}`);
}

function closeServer(server: http.Server): Promise<void> {
  return new Promise((resolve) => {
    server.closeAllConnections?.();
    server.close(() => resolve());
    // close() can hang if a leaked socket ignored closeAllConnections —
    // unref so the test process can still exit.
    server.unref();
  });
}

describe("SSE-11 real socket abort (undici)", () => {
  let server: http.Server;
  let port: number;
  let client: StreamClient | null;
  const c: Counts = { live: 0, opens: 0, closes: 0, peak: 0 };

  beforeEach(async () => {
    state.resetPack(); // cache UNHEALTHY → delta forces reconnect
    c.live = 0;
    c.opens = 0;
    c.closes = 0;
    c.peak = 0;
    client = null;

    server = http.createServer((req, res) => {
      if (!req.url || !req.url.includes("/v1/guard/stream")) {
        res.statusCode = 404;
        res.end();
        return;
      }
      c.live += 1;
      c.opens += 1;
      if (c.live > c.peak) c.peak = c.live;
      // ONE close source. IncomingMessage 'close' fires when the socket
      // actually dies (confirmed: not on GET-complete), which is the leak signal.
      let closed = false;
      req.on("close", () => {
        if (closed) return;
        closed = true;
        c.live -= 1;
        c.closes += 1;
      });
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      });
      res.write(": stream-open\n\n");
      // Unhealthy cache + this delta → forceReconnect (existing SDK path).
      res.write('event: delta\ndata: {"version":1,"ops":[]}\n\n');
      // Hold the socket like a real SSE server — never res.end().
    });

    port = await new Promise<number>((resolve, reject) => {
      server.once("error", reject);
      server.listen(0, "127.0.0.1", () => {
        server.unref();
        resolve((server.address() as AddressInfo).port);
      });
    });
  });

  afterEach(async () => {
    try {
      client?.stop();
      try {
        await waitUntil(() => c.live === 0, 2_000, "afterEach live===0", c);
      } catch {
        // still tear the server down so a leak doesn't pin the worker
      }
    } finally {
      await closeServer(server);
      state.resetPack();
      client = null;
    }
  });

  it("abort closes the real HTTP socket across force-reconnects", async () => {
    client = new StreamClient({
      baseUrl: `http://127.0.0.1:${port}`,
      apiKey: "tp_sk_test",
      sdkVersion: "test",
      deployment: "daemon",
      clientId: "cid",
      firewall: "enforce",
    });
    client.start();

    await waitUntil(() => c.opens >= 1, SETTLE_MS, "initial connect", c);
    expect(c.opens).toBeGreaterThanOrEqual(1);
    expect(c.peak).toBeGreaterThanOrEqual(1);

    // Initial + 3 reconnects. Each delta aborts within a tick, so we often
    // sample live=0 during the ~1s backoff — that is not a leak. The leak
    // signal is peak climbing with opens (old sockets still held).
    await waitUntil(() => c.opens >= 4, SETTLE_MS, "opens>=4 (3 reconnects)", c);
    expect(c.opens).toBeGreaterThanOrEqual(4);
    expect(c.live).toBeLessThanOrEqual(1);
    expect(c.peak).toBeLessThanOrEqual(2); // 1 brief overlap at handoff, never 4 leaked
    expect(c.closes).toBeGreaterThanOrEqual(c.opens - 1);

    client.stop();
    await waitUntil(() => c.live === 0 && c.closes === c.opens, SETTLE_MS, "stop() live===0", c);
    expect(c.live).toBe(0);
    expect(c.closes).toBe(c.opens);
  }, 15_000);
});
