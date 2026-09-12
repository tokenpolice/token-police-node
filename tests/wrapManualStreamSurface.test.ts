/**
 * `_wrapManualStream` must preserve the native Stainless `Stream` surface
 * (`.tee()`/`.controller`/`.toReadableStream()`) instead of returning a bare
 * async generator that drops every member but `[Symbol.asyncIterator]`.
 *
 * The fix is an IN-PLACE `Symbol.asyncIterator` override (NOT a Proxy — Stainless
 * `Stream` uses class private `#` fields a Proxy would break) that returns the
 * SAME object. It captures the ORIGINAL iterator BEFORE installing the override
 * and iterates `getOrig()` (never the object — that would self-recurse). If the
 * override cannot be installed (frozen/non-configurable), it fails open and
 * returns the original native stream un-tapped.
 *
 * These assertions drive the exported `__test__._wrapManualStream` directly with
 * hand-rolled synthetic streams (no real `openai` Stream construction — Decision
 * 3), observing the tap via the `client.log` seam the manual-tap tests use.
 *
 * `tp.log` positional args (see `_logManual`): [4]=model, [5]=provider,
 * [6]=inputTokens, [7]=outputTokens, [8]=cachedTokens, [13]=extra ({latency,
 * call_outcome, ...}).
 */
import { describe, test, expect, afterEach, vi } from "vitest";
import { __test__ as enforcerTest, uninstrument } from "../src/enforcer";
import { TokenPolice } from "../src/client";
import { setClient, resetPack } from "../src/state";
import { session } from "../src/context";

const { _wrapManualStream } = enforcerTest as any;

// A single OpenAI-shaped content chunk.
function contentChunk(text: string): any {
  return { choices: [{ index: 0, delta: { content: text } }] };
}
// A terminal OpenAI-shaped usage-carrying chunk.
function usageChunk(input: number, output: number): any {
  return {
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/**
 * A synthetic Stainless-like `Stream`: async-iterable + `.tee()` + `.controller`
 * (a real AbortController) + `.toReadableStream()`. Models the surface the fix
 * must preserve WITHOUT constructing a real `openai` Stream (Decision 3).
 */
function fakeStainlessStream(chunks: any[], opts: { throwAt?: number; rejectWith?: any } = {}): any {
  const controller = new AbortController();
  const obj: any = {
    controller,
    tee() {
      async function* branch() {
        for (const c of chunks) yield c;
      }
      return [branch(), branch()];
    },
    toReadableStream() {
      return { __fakeReadable: true };
    },
  };
  Object.defineProperty(obj, Symbol.asyncIterator, {
    configurable: true,
    writable: true,
    value: async function* () {
      let i = 0;
      for (const c of chunks) {
        if (opts.throwAt === i) throw opts.rejectWith;
        i++;
        yield c;
      }
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

// Only the manual openai /log rows (arg[5] === provider).
function openaiRows(logSpy: any): any[][] {
  return logSpy.mock.calls.filter((c: any[]) => c[5] === "openai");
}

async function drain(iterable: any): Promise<any[]> {
  const out: any[] = [];
  for await (const c of iterable) out.push(c);
  return out;
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
// Assertion 2 — native surface preserved: identity.
// ═══════════════════════════════════════════════════════════════════
describe("A2: identity — returns the SAME object", () => {
  test("ret === fakeStream (not a copy/generator/proxy)", async () => {
    makeClient();
    const fake = fakeStainlessStream([contentChunk("hi"), usageChunk(10, 3)]);
    const ret = await _wrapManualStream(fake, "openai", [{ model: "m" }], 0, null, new Date());
    expect(Object.is(ret, fake)).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 3 — native Stainless members callable + controller identity.
// ═══════════════════════════════════════════════════════════════════
describe("A3: Stainless members preserved (RED pre-fix)", () => {
  test("tee/controller/toReadableStream present; controller is the ORIGINAL", async () => {
    makeClient();
    const fake = fakeStainlessStream([contentChunk("hi"), usageChunk(10, 3)]);
    const origController = fake.controller;
    const ret = await _wrapManualStream(fake, "openai", [{ model: "m" }], 0, null, new Date());
    expect(typeof ret.tee).toBe("function");
    expect(typeof ret.toReadableStream).toBe("function");
    expect(Object.is(ret.controller, origController)).toBe(true);
    // sanity: tee() still returns two async-iterables
    const [a, b] = ret.tee();
    expect(typeof a[Symbol.asyncIterator]).toBe("function");
    expect(typeof b[Symbol.asyncIterator]).toBe("function");
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 4 — iteration still yields ALL chunks in order.
// ═══════════════════════════════════════════════════════════════════
describe("A4: iteration yields all chunks in order", () => {
  test("for-await over the wrapped stream yields the full sequence", async () => {
    makeClient();
    const chunks = [contentChunk("a"), contentChunk("b"), usageChunk(7, 4)];
    const fake = fakeStainlessStream(chunks);
    await session({ name: "wf" }, async () => {
      const ret = await _wrapManualStream(fake, "openai", [{ model: "m" }], 0, null, new Date());
      const received = await drain(ret);
      expect(received).toEqual(chunks);
    });
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 5 — fail-open fallback: frozen stream returns original, no throw.
// ═══════════════════════════════════════════════════════════════════
describe("A5: fail-open — override-install failure returns the ORIGINAL stream", () => {
  test("frozen stream → no throw, identity, surface intact, iteration works", async () => {
    makeClient();
    const chunks = [contentChunk("a"), usageChunk(5, 2)];
    const fake = fakeStainlessStream(chunks);
    const origController = fake.controller;
    Object.freeze(fake); // non-extensible → Object.defineProperty(asyncIterator) throws

    let ret: any;
    await expect(
      (async () => {
        ret = await _wrapManualStream(fake, "openai", [{ model: "m" }], 0, null, new Date());
      })(),
    ).resolves.toBeUndefined();

    expect(Object.is(ret, fake)).toBe(true);
    expect(typeof ret.tee).toBe("function");
    expect(typeof ret.toReadableStream).toBe("function");
    expect(Object.is(ret.controller, origController)).toBe(true);
    // iteration over the UN-TAPPED original still yields every chunk
    const received = await drain(ret);
    expect(received).toEqual(chunks);
  });

  test("non-configurable Symbol.asyncIterator → same fail-open (no throw)", async () => {
    makeClient();
    const chunks = [contentChunk("x"), usageChunk(1, 1)];
    const obj: any = { controller: new AbortController() };
    Object.defineProperty(obj, Symbol.asyncIterator, {
      configurable: false,
      writable: false,
      value: async function* () {
        for (const c of chunks) yield c;
      },
    });
    const ret = await _wrapManualStream(obj, "openai", [{ model: "m" }], 0, null, new Date());
    expect(Object.is(ret, obj)).toBe(true);
    expect(await drain(ret)).toEqual(chunks);
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 6 — bare-generator input (HF/Gemini shape) tolerated.
// ═══════════════════════════════════════════════════════════════════
describe("A6: bare async-generator input still iterates (override shadows prototype)", () => {
  test("no .tee/.controller; iteration completes with all chunks", async () => {
    makeClient();
    const chunks = [contentChunk("g1"), contentChunk("g2"), usageChunk(3, 9)];
    async function* bare() {
      for (const c of chunks) yield c;
    }
    const input = bare();
    await session({ name: "wf" }, async () => {
      const ret = await _wrapManualStream(input, "openai", [{ model: "m" }], 0, null, new Date());
      const received = await drain(ret);
      expect(received).toEqual(chunks);
    });
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 7 — usage logged EXACTLY ONCE after drain; malformed chunk safe.
// ═══════════════════════════════════════════════════════════════════
describe("A7: usage logged exactly once (no regression)", () => {
  test("terminal usage chunk → exactly one _logManual row with accumulated usage", async () => {
    const { logSpy } = makeClient();
    const fake = fakeStainlessStream([contentChunk("hello"), usageChunk(120, 34)]);
    await session({ name: "wf" }, async () => {
      const ret = await _wrapManualStream(fake, "openai", [{ model: "m" }], 0, null, new Date());
      await drain(ret);
    });
    const rows = openaiRows(logSpy);
    expect(rows.length).toBe(1);
    expect(rows[0][6]).toBe(120);
    expect(rows[0][7]).toBe(34);
  });

  test("a malformed (throwing-tap) chunk does not break iteration nor change the count", async () => {
    const { logSpy } = makeClient();
    // A chunk that blows up _accumulateStreamChunk (choices not an array) — the
    // inner per-chunk try/catch must swallow it.
    const bad = { choices: 42 };
    const fake = fakeStainlessStream([contentChunk("a"), bad, usageChunk(11, 5)]);
    let received: any[] = [];
    await session({ name: "wf" }, async () => {
      const ret = await _wrapManualStream(fake, "openai", [{ model: "m" }], 0, null, new Date());
      received = await drain(ret);
    });
    expect(received.length).toBe(3); // bad chunk still yielded to the customer
    expect(openaiRows(logSpy).length).toBe(1);
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 8 — TTFT captured + _logManual failure swallowed.
// ═══════════════════════════════════════════════════════════════════
describe("A8: TTFT captured; log-seam failure swallowed", () => {
  test("latency payload present (TTFT marked) on the logged row", async () => {
    const { logSpy } = makeClient();
    const fake = fakeStainlessStream([contentChunk("hi"), usageChunk(9, 2)]);
    await session({ name: "wf" }, async () => {
      const ret = await _wrapManualStream(fake, "openai", [{ model: "m" }], 0, null, new Date(), performance.now());
      await drain(ret);
    });
    const extra = openaiRows(logSpy)[0][13];
    expect(extra).toBeTruthy();
    expect(extra.latency).toBeTruthy();
    expect(typeof extra.latency.ttft_ms === "number" || extra.latency.ttft_ms === null).toBe(true);
  });

  test("client.log throws → for-await still completes with all chunks (no escape)", async () => {
    const { client } = makeClient();
    vi.spyOn(client, "log").mockImplementation(() => {
      throw new Error("boom");
    });
    const chunks = [contentChunk("a"), contentChunk("b"), usageChunk(1, 1)];
    const fake = fakeStainlessStream(chunks);
    let received: any[] = [];
    await expect(
      session({ name: "wf" }, async () => {
        const ret = await _wrapManualStream(fake, "openai", [{ model: "m" }], 0, null, new Date());
        received = await drain(ret);
      }),
    ).resolves.toBeUndefined();
    expect(received.length).toBe(3);
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 9 — provider error re-propagated UNSWALLOWED + failure log.
// ═══════════════════════════════════════════════════════════════════
describe("A9: provider error mid-stream re-thrown with failure log", () => {
  test("SAME sentinel thrown to caller; failure log emitted; no success usage row", async () => {
    const { logSpy } = makeClient();
    const providerErr = new Error("provider fail");
    // Reject on the 2nd pull (after yielding the first content chunk).
    const fake = fakeStainlessStream([contentChunk("a"), usageChunk(1, 1)], {
      throwAt: 1,
      rejectWith: providerErr,
    });
    await expect(
      session({ name: "wf" }, async () => {
        const ret = await _wrapManualStream(fake, "openai", [{ model: "m" }], 0, null, new Date());
        for await (const _c of ret) void _c;
      }),
    ).rejects.toBe(providerErr);

    // No SUCCESS usage row (streamFailed guard).
    const success = openaiRows(logSpy).filter((c) => (c[6] as number) > 0 || (c[7] as number) > 0);
    expect(success.length).toBe(0);
    // A failure log WAS emitted via _emitCallFailureLog (call_outcome carried).
    const failing = logSpy.mock.calls.filter((c: any[]) => {
      const extra = c[c.length - 1];
      return extra && typeof extra === "object" && (extra as any).call_outcome;
    });
    expect(failing.length).toBeGreaterThanOrEqual(1);
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 1 (structural) — source-read checks on the post-fix wrapper body.
// ═══════════════════════════════════════════════════════════════════
describe("A1: structural source-read of _wrapManualStream", () => {
  test("in-place override, getOrig bound before defineProperty, no Proxy, no self-recursion", async () => {
    const fs = await import("node:fs/promises");
    const src = await fs.readFile(new URL("../src/enforcer.ts", import.meta.url), "utf8");
    // Isolate the _wrapManualStream body (up to the next top-level function).
    const start = src.indexOf("async function _wrapManualStream(");
    expect(start).toBeGreaterThan(-1);
    const end = src.indexOf("\nfunction _wrapBedrockConverseStream(", start);
    expect(end).toBeGreaterThan(start);
    const body = src.slice(start, end);

    // (a) getOrig bound BEFORE the Object.defineProperty install.
    const bindIdx = body.indexOf("stream[Symbol.asyncIterator].bind(stream)");
    const defineIdx = body.indexOf("Object.defineProperty(stream, Symbol.asyncIterator");
    expect(bindIdx).toBeGreaterThan(-1);
    expect(defineIdx).toBeGreaterThan(-1);
    expect(bindIdx).toBeLessThan(defineIdx);

    // (b) override drives the captured getOrig() iterator directly via .next()
    // (mirror _tapStreamUsageForOnEnd) — getOrig() called, then inner.next().
    expect(body).toContain("getOrig()");
    expect(body).toContain("await inner.next()");

    // (c) NO `for await ... of stream` (or of getOrig()) self-recursion inside
    // the override — the manual .next() loop is the ONLY chunk source.
    expect(body).not.toMatch(/for await \(const \w+ of stream\)/);
    expect(body).not.toMatch(/for await \(const \w+ of getOrig\(\)\)/);
    // no re-read of stream[Symbol.asyncIterator] as a CALL inside the body
    expect(body).not.toContain("stream[Symbol.asyncIterator]()");

    // NOT a bare generator, NOT a Proxy; returns the same object.
    expect(body).not.toContain("return (async function* () {");
    expect(body).not.toContain("new Proxy(");
    expect(body).toContain("return stream;");
  });
});
