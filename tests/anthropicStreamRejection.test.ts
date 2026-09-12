/**
 * F-17-A — a REJECTED Anthropic `.stream()` must land EXACTLY ONE failure row.
 *
 * `client.messages.stream()` is TokenPolice-patched, and the vendor
 * MessageStream internally delegates to the ALSO-patched
 * `messages.create({stream:true})`. A request-time rejection (e.g. a 401 at
 * connect) was therefore observed TWICE:
 *
 *  1. the create wrapper's catch — a GOOD row: `_stashAttemptContext` had just
 *     run, so the row carries the real model + provider; then
 *  2. the vendor MessageStream catches that same rejection internally and
 *     RE-DELIVERS it to the consumer's `for await`, where the `.stream()`
 *     iterator wrapper's catch emitted a SECOND, DEGRADED row
 *     (model='unknown' / provider='' — the first emit cleared the
 *     `_attempted_*` stash).
 *
 * Mid-stream failures double-emitted the same way via the create-path stream
 * taps.
 *
 * Fix under test: the per-call `StreamLatch` (already deduping the SUCCESS row
 * across the two layers, shared via AsyncLocalStorage and stamped on the
 * manager as `__tpLogLatch`) gained an INDEPENDENT `failureLogged` flag.
 * `_emitCallFailureLog` now reports whether it really dispatched; SOURCE layers
 * (create catch / bypass-tap drain catch / `_tapStreamUsageForOnEnd`) mark the
 * latch ONLY after a real dispatch, and the CONSUMER layer (the `.stream()`
 * iterator catch) checks first and bare-rethrows when marked. Dedupe therefore
 * can never cost the only row.
 *
 * GOLDEN RULE pinned throughout: the provider's own error object reaches the
 * customer BY IDENTITY; only `TokenPoliceBlockedError` is ever substituted, and
 * only on a verified enforce denial.
 *
 * HARNESS NOTES (mirrors tests/anthropicStream.test.ts):
 * • Fake anthropic module injected via `init({ instrumentModules })` — no real
 *   @anthropic-ai/sdk. No `APIPromise` export ⇒ the create-streaming BYPASS
 *   path engages, exactly as on the SDK versions this bug was found on.
 * • `delegate: true` models the real vendor delegation: `.stream()` fires
 *   `create({stream:true})` INSIDE its factory window (so the create tap
 *   inherits the `.stream()` latch through async context) and the
 *   MessageStream's own surfaces (`Symbol.asyncIterator`, `done()` and hence
 *   `finalMessage()`) RE-DELIVER whatever that promise does — which is what
 *   produces the second observation of one rejection.
 * • Row counts assert on ACTUAL dispatches (`logSpy.mock.calls`), never on a
 *   provider-filtered view: the degraded duplicate this fix kills carries
 *   provider='' and would be invisible to a `c[5] === "anthropic"` filter.
 * • `autoInstrument` early-returns once instrumented, so each test does
 *   EXACTLY ONE init; afterEach uninstruments and resets the client.
 */
import { describe, test, expect, afterEach, vi } from "vitest";
import { uninstrument } from "../src/enforcer";
import { init } from "../src/client";
import { setClient } from "../src/state";
import { session as tpSession } from "../src/context";
import { TokenPoliceBlockedError } from "../src/exceptions";

const flush = () => new Promise((r) => setTimeout(r, 0));

/** Anthropic-SDK-shaped 401 (`AuthenticationError` + `status`). */
function auth401(): any {
  class AuthenticationError extends Error {}
  return Object.assign(
    new AuthenticationError("401 invalid x-api-key"),
    { status: 401 },
  );
}

function defaultEvents(): any[] {
  return [
    {
      type: "message_start",
      message: { model: "claude-3", usage: { input_tokens: 10, output_tokens: 0 } },
    },
    { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Hello" } },
    {
      type: "message_delta",
      usage: { output_tokens: 5 },
      delta: { stop_reason: "end_turn" },
    },
  ];
}
function defaultFinal(): any {
  return {
    model: "claude-3",
    stop_reason: "end_turn",
    usage: { input_tokens: 10, output_tokens: 5 },
    content: [{ type: "text", text: "Hello world" }],
  };
}

interface FakeOpts {
  // Model the real SDK: `.stream()` internally fires `create({stream:true})`
  // within its factory window, and the MessageStream re-delivers that promise's
  // outcome (events OR rejection) through its own surfaces.
  delegate?: boolean;
  // `create()` rejects with this at request time (before any stream exists).
  createError?: any;
  // Events the delegated raw stream yields, and an optional mid-drain throw.
  rawEvents?: any[];
  rawErrorAt?: number;
  rawError?: any;
  // Non-delegated arm: the MessageStream's OWN iterator throws (customer-side
  // stream error with no create layer in play).
  iterError?: any;
  iterErrorIndex?: number;
}

/** The raw anthropic Stream `create({stream:true})` resolves to (tapped by
 * `_tapAnthropicStreamBypass`). Re-iterable; throws at `rawErrorAt` if armed. */
class FakeRawStream {
  events: any[];
  errAt: number;
  err: any;
  controller = { abort() {} };
  constructor(events: any[], errAt = -1, err: any = null) {
    this.events = events;
    this.errAt = errAt;
    this.err = err;
  }
  tee() {
    return [this, this];
  }
  [Symbol.asyncIterator]() {
    let i = 0;
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const self = this;
    return {
      async next(): Promise<IteratorResult<any>> {
        if (self.err && self.errAt === i) throw self.err;
        if (i >= self.events.length) return { done: true, value: undefined };
        return { done: false, value: self.events[i++] };
      },
    };
  }
}

/**
 * A fake Anthropic MessageStream. Async-iterable + EventEmitter-ish + the
 * context-manager surface the wrapper gates (finalMessage / finalText / done /
 * emitted / controller / abort / tee).
 *
 * The load-bearing fidelity detail for F-17-A: when the vendor delegated to
 * `create({stream:true})`, EVERY consumer surface resolves THROUGH that
 * promise — so a request-time rejection is re-delivered to whichever surface
 * the customer used, which is precisely the second observation of one failure.
 */
class FakeMessageStream {
  _opts: FakeOpts;
  _events: any[];
  _final: any;
  /** The patched create's promise (delegation), null for an independent stream. */
  _delegate: Promise<any> | null = null;
  _rawIter: AsyncIterator<any> | null = null;
  _ready: Promise<void>;
  receivedMessages: any[] = [];
  _listeners = new Map<string, Array<{ cb: (...a: any[]) => void; once: boolean }>>();
  innerNextCount = 0;
  doneCalls = 0;
  aborted = false;
  controllerAborted = false;
  controller = {
    abort: () => {
      this.controllerAborted = true;
      this.aborted = true;
    },
  };

  constructor(opts: FakeOpts = {}) {
    this._opts = opts;
    this._events = defaultEvents();
    this._final = defaultFinal();
    this._ready = new Promise<void>((res) => {
      queueMicrotask(() => {
        this.receivedMessages.push(this._final);
        res();
      });
    });
  }

  on(evt: string, cb: (...a: any[]) => void) {
    let l = this._listeners.get(evt);
    if (!l) this._listeners.set(evt, (l = []));
    l.push({ cb, once: false });
    return this;
  }
  once(evt: string, cb: (...a: any[]) => void) {
    let l = this._listeners.get(evt);
    if (!l) this._listeners.set(evt, (l = []));
    l.push({ cb, once: true });
    return this;
  }
  off(evt: string, cb: (...a: any[]) => void) {
    const l = this._listeners.get(evt);
    if (!l) return this;
    const i = l.findIndex((e) => e.cb === cb);
    if (i >= 0) l.splice(i, 1);
    return this;
  }
  emit(evt: string, ...args: any[]) {
    const l = this._listeners.get(evt);
    if (!l) return;
    for (const entry of [...l]) {
      if (entry.once) {
        const i = l.indexOf(entry);
        if (i !== -1) l.splice(i, 1);
      }
      entry.cb(...args);
    }
  }
  abort() {
    this.aborted = true;
  }
  tee() {
    return [this, this];
  }
  emitted(event: string): Promise<any> {
    return new Promise((res, rej) => {
      if (event !== "error") (this as any).once("error", rej);
      (this as any).once(event, res);
    });
  }
  /** Completion routes through the delegated request — a rejection surfaces here. */
  async done(): Promise<void> {
    this.doneCalls++;
    if (this._delegate) await this._delegate;
    await this._ready;
  }
  async finalMessage(): Promise<any> {
    // Route through `this.done()` (the GATED done, as the real SDK does).
    await (this as any).done();
    return this.receivedMessages[this.receivedMessages.length - 1];
  }
  async finalText(): Promise<string> {
    await (this as any).done();
    const msg = this.receivedMessages[this.receivedMessages.length - 1];
    const blocks = Array.isArray(msg?.content) ? msg.content : [];
    return blocks
      .filter((b: any) => b && b.type === "text")
      .map((b: any) => b.text)
      .join("");
  }
  [Symbol.asyncIterator]() {
    let i = 0;
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const self = this;
    return {
      async next(): Promise<IteratorResult<any>> {
        self.innerNextCount++;
        if (self._delegate) {
          // Request-time rejection → re-delivered here (the F-17-A case).
          const raw: any = await self._delegate;
          if (!self._rawIter) self._rawIter = raw[Symbol.asyncIterator]();
          // Mid-stream rejection → re-delivered here.
          return self._rawIter.next();
        }
        if (self._opts.iterError && self._opts.iterErrorIndex === i) {
          throw self._opts.iterError;
        }
        if (i >= self._events.length) return { done: true, value: undefined };
        return { done: false, value: self._events[i++] };
      },
    };
  }
}

// Fresh prototypes per call → no cross-test idempotency-marker contamination.
// NO APIPromise export → the create-streaming bypass path engages.
function makeFakeAnthropic(opts: FakeOpts = {}) {
  const streams: FakeMessageStream[] = [];
  const created: any[] = [];
  class Messages {
    stream(body: any) {
      const s = new FakeMessageStream(opts);
      streams.push(s);
      if (opts.delegate) {
        // `this.create` is the PATCHED create; firing it here — inside the
        // `.stream()` factory window — is what lets the create layer inherit
        // the `.stream()` latch through async context, exactly as a real
        // delegating SDK does.
        const p: Promise<any> = (this as any).create({ ...body, stream: true });
        s._delegate = p;
        // The vendor holds this promise internally and only re-delivers it via
        // the MessageStream surfaces; swallow here so a consumer that never
        // touches those surfaces can't trip an unhandled rejection.
        void p.catch(() => {});
      }
      return s;
    }
    create(body: any) {
      created.push(body);
      if (opts.createError) return Promise.reject(opts.createError);
      if (body?.stream) {
        return Promise.resolve(
          new FakeRawStream(
            opts.rawEvents ?? defaultEvents(),
            opts.rawErrorAt ?? -1,
            opts.rawError ?? null,
          ),
        );
      }
      return Promise.resolve({});
    }
  }
  class AsyncMessages {
    stream(_body: any) {
      const s = new FakeMessageStream(opts);
      streams.push(s);
      return s;
    }
  }
  function Anthropic(this: any) {
    this.messages = new Messages();
  }
  (Anthropic as any).Messages = Messages;
  (Anthropic as any).AsyncMessages = AsyncMessages;
  return { Anthropic, Messages, AsyncMessages, streams, created };
}

const BODY = { model: "claude-3", messages: [{ role: "user", content: "hi" }] };

function initClient(firewall: "enforce" | "dry_run" | "off", fake: any) {
  const client = init({
    apiKey: "tp_sk_test",
    deployment: "serverless",
    firewall,
    instrumentModules: { anthropic: fake.Anthropic },
  } as any);
  const logSpy = vi.spyOn(client, "log").mockImplementation(() => {});
  return { client, logSpy };
}

// Install the wrapper (dry_run) with the pre-flight stubbed ALLOWED.
function initAllowed(fake: any) {
  const { client, logSpy } = initClient("dry_run", fake);
  vi.spyOn(client, "check").mockResolvedValue({ status: "allowed", fail_open: false } as any);
  return { client, logSpy };
}

/** `tp.log`'s trailing options arg — where `call_outcome` lives. */
function extrasOf(call: any[]): any {
  return call[call.length - 1];
}
/** Rows the FAILURE emitter dispatched (whatever provider they landed with). */
function failureRows(logSpy: any): any[][] {
  return logSpy.mock.calls.filter(
    (c: any[]) => extrasOf(c)?.call_outcome?.status === "failed",
  );
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
  vi.restoreAllMocks();
});

// ═══════════════════════════════════════════════════════════════════
// 1 — request-time rejection, drained with `for await`.
// ═══════════════════════════════════════════════════════════════════
describe("F-17-A/1: delegated create rejects at request time (for-await drain)", () => {
  test("exactly ONE failure dispatch, carrying the REAL model + provider; original error by identity", async () => {
    const boom = auth401();
    const fake = makeFakeAnthropic({ delegate: true, createError: boom });
    const { logSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      let caught: any;
      try {
        for await (const _ev of stream) void _ev;
      } catch (e) {
        caught = e;
      }
      await flush();

      // GOLDEN RULE — the provider's own object, not a wrapper, not a block.
      expect(caught).toBe(boom);
      expect(caught).not.toBeInstanceOf(TokenPoliceBlockedError);

      // Pre-fix: 2 dispatches (the good one + the degraded consumer-layer dup).
      expect(logSpy.mock.calls.length).toBe(1);
      const rows = failureRows(logSpy);
      expect(rows.length).toBe(1);
      // The SURVIVOR is the good row — not model='unknown' / provider=''.
      expect(rows[0][4]).toBe("claude-3");
      expect(rows[0][5]).toBe("anthropic");
      const outcome = extrasOf(rows[0]).call_outcome;
      expect(outcome.status).toBe("failed");
      expect(outcome.error_kind).toBe("auth_error");
      expect(outcome.http_status).toBe(401);
      // A failed call has no usage — the row stays unmeasured.
      expect(rows[0][6]).toBe(0);
      expect(rows[0][7]).toBe(0);

      // The failure one-shot is INDEPENDENT of the success one-shot.
      expect(stream.__tpLogLatch.failureLogged).toBe(true);
      expect(stream.__tpLogLatch.logged).toBe(false);
    });
  });

  test("the consumer-layer skip leaves NO stale failed outcome on the session", async () => {
    const boom = auth401();
    const fake = makeFakeAnthropic({ delegate: true, createError: boom });
    initAllowed(fake);
    await tpSession({}, async (s: any) => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      await expect(
        (async () => {
          for await (const _ev of stream) void _ev;
        })(),
      ).rejects.toBe(boom);
      await flush();
      // The dedupe arm rethrows WITHOUT stamping `_call_outcome`; the emitting
      // layer cleared it. A leftover would mis-tag the NEXT call's row.
      expect(s._call_outcome).toBeFalsy();
    });
  });
});

// ═══════════════════════════════════════════════════════════════════
// 2 — the same rejection seen only through `finalMessage()` (never zero).
// ═══════════════════════════════════════════════════════════════════
describe("F-17-A/2: request-time rejection surfaced via finalMessage() only", () => {
  test("consumer never iterates → STILL exactly one dispatch (the create layer's)", async () => {
    const boom = auth401();
    const fake = makeFakeAnthropic({ delegate: true, createError: boom });
    const { logSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      // The gate passes (allowed), the native finalMessage rejects → identity.
      await expect(stream.finalMessage()).rejects.toBe(boom);
      await flush();
      // Guards the fix against regressing to ZERO rows for a non-iterating
      // consumer (the layer that would have emitted here never runs).
      expect(logSpy.mock.calls.length).toBe(1);
      expect(failureRows(logSpy).length).toBe(1);
      expect(failureRows(logSpy)[0][4]).toBe("claude-3");
    });
  });
});

// ═══════════════════════════════════════════════════════════════════
// 3 — mid-stream failure (create RESOLVED, the stream dies during drain).
// ═══════════════════════════════════════════════════════════════════
describe("F-17-A/3: mid-stream failure on a delegated .stream()", () => {
  test("failure after the first event → exactly ONE dispatch, original error by identity", async () => {
    const boom = Object.assign(new Error("upstream 500"), { status: 500 });
    const fake = makeFakeAnthropic({
      delegate: true,
      rawEvents: defaultEvents(),
      rawErrorAt: 1, // message_start delivered, then the connection dies
      rawError: boom,
    });
    const { logSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      const seen: any[] = [];
      let caught: any;
      try {
        for await (const ev of stream) seen.push(ev);
      } catch (e) {
        caught = e;
      }
      await flush();
      expect(caught).toBe(boom);
      expect(seen).toHaveLength(1); // the tokens that DID arrive still flowed
      // Pre-fix: the create-path tap emitted, then the re-delivered rejection
      // made the consumer layer emit a degraded second row.
      expect(logSpy.mock.calls.length).toBe(1);
      const rows = failureRows(logSpy);
      expect(rows.length).toBe(1);
      expect(rows[0][4]).toBe("claude-3");
      expect(rows[0][5]).toBe("anthropic");
      expect(extrasOf(rows[0]).call_outcome.error_kind).toBe("server_error");
      expect(extrasOf(rows[0]).call_outcome.http_status).toBe(500);
    });
  });
});

// ═══════════════════════════════════════════════════════════════════
// 4 — re-pulling a rejected iterator must not re-emit.
// ═══════════════════════════════════════════════════════════════════
describe("F-17-A/4: repeated next() after a rejection", () => {
  test("delegated (consumer layer SKIPS): three next() calls → one dispatch, same error each time", async () => {
    const boom = auth401();
    const fake = makeFakeAnthropic({ delegate: true, createError: boom });
    const { logSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      const it = stream[Symbol.asyncIterator]();
      await expect(it.next()).rejects.toBe(boom);
      await expect(it.next()).rejects.toBe(boom);
      await expect(it.next()).rejects.toBe(boom);
      await flush();
      expect(logSpy.mock.calls.length).toBe(1);
    });
  });

  test("independent .stream() (consumer layer EMITS): its own mark stops the re-emit", async () => {
    // No delegation → no source layer emitted, so the consumer layer is the
    // one that dispatches; it must mark the latch for its OWN dispatch too.
    const boom = Object.assign(new Error("stream died"), { status: 500 });
    const fake = makeFakeAnthropic({ iterError: boom, iterErrorIndex: 0 });
    const { logSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      const it = stream[Symbol.asyncIterator]();
      await expect(it.next()).rejects.toBe(boom);
      await expect(it.next()).rejects.toBe(boom);
      await expect(it.next()).rejects.toBe(boom);
      await flush();
      expect(failureRows(logSpy).length).toBe(1);
      expect(logSpy.mock.calls.length).toBe(1);
      expect(stream.__tpLogLatch.failureLogged).toBe(true);
    });
  });
});

// ═══════════════════════════════════════════════════════════════════
// 5 — an INDEPENDENT create({stream:true}) is untouched by the latch.
// ═══════════════════════════════════════════════════════════════════
describe("F-17-A/5: independent create({stream:true}) rejection", () => {
  test("no .stream() in scope → exactly one dispatch (mark is a no-op, behavior unchanged)", async () => {
    const boom = auth401();
    const fake = makeFakeAnthropic({ createError: boom }); // delegate:false
    const { logSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const messages: any = new fake.Anthropic().messages;
      await expect(messages.create({ ...BODY, stream: true })).rejects.toBe(boom);
      await flush();
      expect(logSpy.mock.calls.length).toBe(1);
      const rows = failureRows(logSpy);
      expect(rows.length).toBe(1);
      expect(rows[0][4]).toBe("claude-3");
      expect(rows[0][5]).toBe("anthropic");
      expect(extrasOf(rows[0]).call_outcome.http_status).toBe(401);
    });
  });
});

// ═══════════════════════════════════════════════════════════════════
// 6 — success metering untouched (latch flags are independent).
// ═══════════════════════════════════════════════════════════════════
describe("F-17-A/6: successful delegated .stream()", () => {
  test("full drain → exactly one SUCCESS row, zero failure dispatches, only `logged` taken", async () => {
    const fake = makeFakeAnthropic({ delegate: true });
    const { logSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      const seen: any[] = [];
      for await (const ev of stream) seen.push(ev);
      await flush();
      expect(seen).toHaveLength(3);
      expect(failureRows(logSpy).length).toBe(0);
      expect(logSpy.mock.calls.length).toBe(1); // the two layers still dedupe
      expect(logSpy.mock.calls[0][6]).toBe(10); // input_tokens — really metered
      expect(logSpy.mock.calls[0][7]).toBe(5); // output_tokens
      // The success one-shot is taken; the failure one-shot was never touched.
      expect(stream.__tpLogLatch.logged).toBe(true);
      expect(stream.__tpLogLatch.failureLogged).toBeFalsy();
    });
  });
});

// ═══════════════════════════════════════════════════════════════════
// 7 — enforce block still wins (a denial is never a provider failure).
// ═══════════════════════════════════════════════════════════════════
describe("F-17-A/7: enforce block on .stream()", () => {
  test("TokenPoliceBlockedError propagates, inner never pulled, and NO failure row is emitted", async () => {
    const fake = makeFakeAnthropic();
    const { client, logSpy } = initClient("enforce", fake);
    vi.spyOn(client, "check").mockResolvedValue({
      status: "blocked",
      fail_open: false,
      reason: "x",
    } as any);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    await expect(
      (async () => {
        for await (const _ev of stream) void _ev;
      })(),
    ).rejects.toBeInstanceOf(TokenPoliceBlockedError);
    await flush();
    expect(stream.innerNextCount).toBe(0); // provider stream never surfaced
    expect(stream.aborted).toBe(true);
    // A denial must never be reclassified as a provider-failure row.
    expect(failureRows(logSpy).length).toBe(0);
  });
});
