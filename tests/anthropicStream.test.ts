/**
 * Anthropic `Messages.stream()` context-manager instrumentation (Node).
 *
 * Ports Python's `_instrument_anthropic_stream` (enforcer.py:6227-6290): the
 * first-party streaming helper `client.messages.stream()` (the `.on('text')` /
 * `for await` / `await .finalMessage()` API — DISTINCT from
 * `create({stream:true})`) now gets a pre-flight `/check` gated at EVERY
 * token-delivery surface plus one `/log` from `finalMessage()`.
 *
 * All tests use a FAKE anthropic module + FAKE MessageStream injected via
 * autoInstrument/instrumentModules (mirrors tests/initNeverThrows.test.ts) — no
 * real @anthropic-ai/sdk dependency. The fake MessageStream models Anthropic's
 * EAGER internal emitter (an internal post-construction consumption loop, NOT
 * driven by the customer's for-await) whose firing order vs the async check is
 * controllable so the buffer→flush→live race is drivable both ways.
 *
 * HARNESS NOTES:
 * • `autoInstrument` installs the taps in EVERY firewall mode (including
 * `off`, where they run log-only) and early-returns once `_isInstrumented`
 * — so each test does EXACTLY ONE init and uses `firewall:"dry_run"` (with
 * the check stubbed allowed) whenever the wrapper must be installed AND its
 * pre-flight /check must run without blocking. afterEach uninstruments +
 * resets the client for clean per-test isolation.
 */
import { describe, test, expect, afterEach, beforeEach, vi } from "vitest";
import { autoInstrument, uninstrument } from "../src/enforcer";
import { init, TokenPolice } from "../src/client";
import { setClient, applySnapshot, resetPack, drainObservations } from "../src/state";
import { assertStreamedLogPresent } from "./helpers/assertStreamPresence";
import {
  session as tpSession,
  runWithAnthropicStreamOtelSuppress,
} from "../src/context";
import { TokenPoliceSpanProcessor } from "../src/telemetry";
import { TokenPoliceBlockedError } from "../src/exceptions";
import { textEntry } from "../src/composition";

// ── helpers ────────────────────────────────────────────────────────
function deferred<T = unknown>() {
  let resolve!: (v: T | PromiseLike<T>) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

// Race a promise against a short HANG sentinel so a regression that never settles
// (e.g. the `emitted()` held-map-drop hang) fails FAST with a distinct `HANG`
// error instead of stalling the whole suite until the runner's own timeout.
function raceHang<T>(p: Promise<T>, ms = 200): Promise<T> {
  return Promise.race([
    p,
    new Promise<T>((_res, rej) =>
      setTimeout(() => rej(new Error("HANG")), ms),
    ),
  ]);
}

function defaultEvents(): any[] {
  return [
    { type: "message_start", message: { model: "claude-3", usage: { input_tokens: 10, output_tokens: 0 } } },
    { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Hello" } },
    { type: "message_delta", usage: { output_tokens: 5 }, delta: { stop_reason: "end_turn" } },
  ];
}
function defaultFinal(): any {
  return {
    model: "claude-3",
    stop_reason: "end_turn",
    usage: {
      input_tokens: 10,
      output_tokens: 5,
      cache_read_input_tokens: 3,
      cache_creation_input_tokens: 2,
    },
    content: [{ type: "text", text: "Hello world" }],
  };
}

interface FakeOpts {
  events?: any[];
  final?: any;
  tokens?: string[];
  finalMessageThrows?: boolean;
  iterError?: any;
  iterErrorIndex?: number;
  useMidGate?: boolean;
  nonConfigurableOn?: boolean;
  blockLatchWrite?: boolean;
  // Extra token/control events the eager emitter fires after the text tokens and
  // before the terminal `message` (drives inputJson/thinking/citation/signature).
  extraEmits?: Array<{ event: string; value: any }>;
  // Model a real (non-extensible) MessageStream instance: the wrapper's
  // `Object.defineProperty` of new own members throws → ALL overrides degrade to
  // native together (T-4). Existing OWN props remain writable.
  nonExtensible?: boolean;
  // Model real @anthropic-ai/sdk delegation: `.stream()` internally fires
  // `create({stream:true})` WITHIN the stream factory window (so the create tap
  // inherits the `.stream()` log latch via async context). The delegated raw
  // stream is exposed on the MessageStream as `_delegateRaw` for the test to
  // drain, driving BOTH log points.
  delegateToCreate?: boolean;
  // Model the INNER OTel anthropic instrumentor starting its duplicate span:
  // the real instrumentor is wrapped inner of our `.stream()` patch, so its
  // `tracer.startSpan` (→ SpanProcessor.onStart) fires synchronously inside the
  // ORIGINAL `.stream()` body — i.e. inside our per-call suppression window.
  // The fake original stream method invokes this hook to reproduce that timing,
  // letting a test tag its duplicate span exactly as production does.
  onNativeStreamStart?: () => void;
  // F-23-2: by default `receivedMessages` populates on an UNGATED microtask
  // (fires within ~1 tick of construction, independent of both the emitter
  // gate and the consumer's own iteration progress) — realistic for the
  // finalMessage()/done()-only tests that never drive the async iterator, but
  // it means an early-break test can never observe `receivedMessages` still
  // empty (the population microtask always wins the race). When true, that
  // population instead awaits `releaseFinal()` so a test can break out of
  // `for await` BEFORE the message is "received" — modeling a customer who
  // walks away mid-message, exactly the F-23-2 drain-end/return() regression.
  gateFinalMessage?: boolean;
  // F-23-2 hostile input: makes the `receivedMessages` GETTER throw on every
  // access (models a hostile/adversarial MessageStream). `_finalFromReceived`
  // in enforcer.ts must swallow this (GOLDEN RULE: never throw into the
  // consumer) and degrade to a usage-only, empty-composition row.
  hostileReceivedMessages?: boolean;
}

// A fake Anthropic MessageStream. Async-iterable + EventEmitter-ish + the
// context-manager surface (finalMessage/controller/abort/tee).
class FakeMessageStream {
  _events: any[];
  _final: any;
  _tokens: string[];
  _listeners = new Map<string, Array<{ cb: (...a: any[]) => void; once: boolean }>>();
  aborted = false;
  controllerAborted = false;
  controller = {
    abort: () => {
      this.controllerAborted = true;
      this.aborted = true;
    },
  };
  innerNextCount = 0;
  finalMessageThrows: boolean;
  finalMessageCalled = false;
  _iterError: any;
  _iterErrorIndex: number;
  _useMidGate: boolean;
  _emitGate = deferred<void>();
  _midGate = deferred<void>();
  emitDone = deferred<void>();
  // ── AM-1 fidelity: model the real SDK's end-promise / receivedMessages /
  // finalMessage→done / finalText→done coupling and `emitted()`→`once()` routing.
  receivedMessages: any[] = [];
  _endDeferred = deferred<void>();
  // F-23-2: gate on receivedMessages population (see FakeOpts.gateFinalMessage).
  _finalGate: ReturnType<typeof deferred> | null = null;
  _doneMeta = { calls: 0 }; // freeze-safe done() spy (A21)
  _extraEmits: Array<{ event: string; value: any }>;
  // The raw stream from an internally-delegated create({stream:true}) (delegation
  // fidelity) — populated by Messages.stream when `delegateToCreate` is set.
  _delegateRaw?: Promise<any>;
  // The SDK's `_emit('abort')` fires an APIUserAbortError; a distinctive identity
  // so A22's unhandled-rejection capture can attribute it.
  _abortError: Error = (() => {
    const e = new Error("Request was aborted.");
    (e as any).name = "APIUserAbortError";
    return e;
  })();

  constructor(opts: FakeOpts = {}) {
    this._events = opts.events ?? defaultEvents();
    this._final = opts.final ?? defaultFinal();
    this._tokens = opts.tokens ?? ["Hello", " world"];
    this.finalMessageThrows = !!opts.finalMessageThrows;
    this._iterError = opts.iterError ?? null;
    this._iterErrorIndex = opts.iterErrorIndex ?? -1;
    this._useMidGate = !!opts.useMidGate;
    this._extraEmits = opts.extraEmits ?? [];
    this._finalGate = opts.gateFinalMessage ? deferred<void>() : null;
    // Model the SDK's end-promise + receivedMessages population that `done()`
    // awaits. Resolved in an ungated microtask (independent of the token-emitter
    // gate) so `done()`/`finalMessage()`/`finalText()` resolve without requiring
    // `releaseEmitter()` — exactly as the real reader drives completion. When
    // `gateFinalMessage` is set, population instead awaits `releaseFinal()` so
    // an early-break test can observe `receivedMessages` still empty.
    queueMicrotask(async () => {
      if (this._finalGate) await this._finalGate.promise;
      try {
        // Guarded: `hostileReceivedMessages` makes the GETTER throw, which
        // would otherwise surface as an uncaught exception in this bare
        // microtask (unrelated to the consumer-side try/catch in enforcer.ts).
        this.receivedMessages.push(this._final);
      } catch {
        /* hostileReceivedMessages: population itself may throw */
      }
      this._endDeferred.resolve();
    });
    if (opts.hostileReceivedMessages) {
      Object.defineProperty(this, "receivedMessages", {
        configurable: true,
        enumerable: true,
        get(): any[] {
          throw new Error("receivedMessages boom (hostile)");
        },
      });
    }
    if (opts.nonConfigurableOn) {
      Object.defineProperty(this, "on", {
        value: this.on.bind(this),
        configurable: false,
        writable: false,
        enumerable: false,
      });
    }
    if (opts.blockLatchWrite) {
      // Make the wrapper's Object.defineProperty(__tpLogLatch) throw (redefine of
      // a non-configurable prop) WITHOUT breaking the rest of the object.
      Object.defineProperty(this, "__tpLogLatch", {
        value: undefined,
        configurable: false,
        writable: false,
        enumerable: false,
      });
    }
    // Internal post-construction consumption loop (eager; NOT customer-driven).
    // Gated on releaseEmitter() so tests can order the race vs the check.
    queueMicrotask(async () => {
      try {
        await this._emitGate.promise;
        for (let k = 0; k < this._tokens.length; k++) {
          this.emit("text", this._tokens[k]);
          if (k === 0 && this._useMidGate) await this._midGate.promise;
        }
        for (const ex of this._extraEmits) this.emit(ex.event, ex.value);
        this.emit("message", this._final);
        this.emitDone.resolve();
      } catch {
        this.emitDone.resolve();
      }
    });
    if (opts.nonExtensible) {
      // A real MessageStream degradation: adding a new OWN member throws, so the
      // wrapper's `Object.defineProperty(mgr, "on"/"finalText"/"emitted"/…)` all
      // fail and every override degrades to native together (T-4). Existing OWN
      // props stay writable (innerNextCount++, etc. keep working).
      Object.preventExtensions(this);
    }
  }
  releaseEmitter() {
    this._emitGate.resolve();
  }
  releaseMid() {
    this._midGate.resolve();
  }
  // F-23-2: releases a `gateFinalMessage`-deferred receivedMessages population.
  releaseFinal() {
    this._finalGate?.resolve();
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
  // Model @anthropic-ai/sdk `MessageStream.off`: removes AT MOST ONE matching
  // listener from the event's list and returns `this` for chaining (the SDK has
  // NO removeListener/removeAllListeners — off is the sole unsubscribe surface).
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
    // Model @anthropic-ai/sdk `MessageStream._emit('abort', APIUserAbortError)`:
    // fire any 'abort' listeners, then — if NO 'abort'/'error' listener exists —
    // schedule a deliberate `Promise.reject` (which Node surfaces as an unhandled
    // rejection). AM-3's just-in-time swallow listeners must neutralize this.
    const abortListeners = this._listeners.get("abort");
    if (abortListeners) for (const e of [...abortListeners]) e.cb(this._abortError);
    const listenerCount =
      (this._listeners.get("abort")?.length ?? 0) +
      (this._listeners.get("error")?.length ?? 0);
    if (listenerCount === 0) void Promise.reject(this._abortError);
  }
  tee() {
    return [this, this];
  }
  // done() awaits the internal end-promise (AM-1b). Spied via _doneMeta so A21 can
  // prove finalMessage()/finalText() route THROUGH it.
  async done() {
    this._doneMeta.calls++;
    await this._endDeferred.promise;
  }
  async finalMessage() {
    this.finalMessageCalled = true;
    // Route through this.done() (the overridden, gated done) then read
    // receivedMessages.at(-1) — the real finalMessage→done coupling (AM-1c).
    await (this as any).done();
    if (this.finalMessageThrows) throw new Error("finalMessage boom");
    return this.receivedMessages[this.receivedMessages.length - 1];
  }
  async finalText() {
    // Route through this.done() then concatenate text blocks (AM-1c).
    await (this as any).done();
    const msg = this.receivedMessages[this.receivedMessages.length - 1];
    const blocks = Array.isArray(msg?.content) ? msg.content : [];
    return blocks
      .filter((b: any) => b && b.type === "text")
      .map((b: any) => b.text)
      .join("");
  }
  // emitted(event) resolves the first time `event` fires — routed through once()
  // exactly as the SDK does, so a gated token event's held res is genuinely
  // subject to the block-path held.clear() drop (AM-1a → A8 hang reproducible).
  emitted(event: string): Promise<any> {
    return new Promise((res, rej) => {
      if (event !== "error") (this as any).once("error", rej);
      (this as any).once(event, res);
    });
  }
  [Symbol.asyncIterator]() {
    let i = 0;
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const self = this;
    return {
      async next(): Promise<IteratorResult<any>> {
        self.innerNextCount++;
        if (self._iterErrorIndex === i && self._iterError) throw self._iterError;
        if (i >= self._events.length) return { done: true, value: undefined };
        return { done: false, value: self._events[i++] };
      },
      // F-23-2: a real (async-generator-backed) MessageStream iterator
      // supports early return (the customer `break`s out of `for await`);
      // without this the wrapper's `return: inner.return ? … : undefined`
      // guard leaves `return` undefined and the return()-path fix is
      // unreachable from a plain `break`.
      async return(v?: any): Promise<IteratorResult<any>> {
        i = self._events.length; // mark exhausted — no further next() delivers events
        return { done: true, value: v };
      },
    };
  }
}

// A fake raw anthropic Stream (what create({stream:true}) returns on the
// bypass path) — async-iterable of events, tapped by _tapAnthropicStreamBypass.
class FakeRawStream {
  _events: any[];
  controller = { abort() {} };
  constructor(events: any[]) {
    this._events = events;
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
        if (i >= self._events.length) return { done: true, value: undefined };
        return { done: false, value: self._events[i++] };
      },
    };
  }
}

interface MakeOpts {
  streamOpts?: FakeOpts;
  createEvents?: any[];
}
// Build a fresh fake Anthropic namespace (fresh prototypes each call → no
// cross-test idempotency-marker contamination). NO APIPromise export → the
// create-streaming bypass path engages.
function makeFakeAnthropic(opts: MakeOpts = {}) {
  const lastStreams: FakeMessageStream[] = [];
  const lastRaw: FakeRawStream[] = [];
  class Messages {
    stream(body: any) {
      // Inner OTel instrumentor span-start fidelity: fires synchronously inside
      // the original `.stream()` body (= inside our per-call suppression window).
      opts.streamOpts?.onNativeStreamStart?.();
      const s = new FakeMessageStream(opts.streamOpts);
      lastStreams.push(s);
      if (opts.streamOpts?.delegateToCreate) {
        // `this.create` is the patched create; invoking it here — inside the
        // `.stream()` factory window — lets its bypass tap inherit the `.stream()`
        // log latch via async context, exactly as a real SDK's internal delegation.
        s._delegateRaw = (this as any).create({ ...body, stream: true });
      }
      return s;
    }
    create(body: any) {
      if (body?.stream) {
        const rs = new FakeRawStream(opts.createEvents ?? defaultEvents());
        lastRaw.push(rs);
        return Promise.resolve(rs);
      }
      return Promise.resolve({});
    }
  }
  class AsyncMessages {
    stream(_body: any) {
      opts.streamOpts?.onNativeStreamStart?.();
      const s = new FakeMessageStream(opts.streamOpts);
      lastStreams.push(s);
      return s;
    }
  }
  function Anthropic(this: any) {
    this.messages = new Messages();
  }
  (Anthropic as any).Messages = Messages;
  (Anthropic as any).AsyncMessages = AsyncMessages;
  return { Anthropic, Messages, AsyncMessages, lastStreams, lastRaw };
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

// Install the wrapper (dry_run) with the pre-flight stubbed ALLOWED — the common
// setup for allowed-path / logging / latch tests.
function initAllowed(fake: any) {
  const { client, logSpy } = initClient("dry_run", fake);
  vi.spyOn(client, "check").mockResolvedValue({ status: "allowed", fail_open: false } as any);
  return { client, logSpy };
}

// Count only the manual anthropic /log rows (arg[5] === provider).
function anthropicRows(logSpy: any): any[][] {
  return logSpy.mock.calls.filter((c: any[]) => c[5] === "anthropic");
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
// Assertion 1 — both targets patched + idempotent.
// ═══════════════════════════════════════════════════════════════════
describe("A1: targets patched + idempotent", () => {
  test("Messages.stream + AsyncMessages.stream replaced; re-install is a no-op", () => {
    const fake = makeFakeAnthropic();
    const origSync = fake.Messages.prototype.stream;
    const origAsync = fake.AsyncMessages.prototype.stream;
    initClient("dry_run", fake);
    expect(fake.Messages.prototype.stream).not.toBe(origSync);
    expect(fake.AsyncMessages.prototype.stream).not.toBe(origAsync);
    const wrapped = fake.Messages.prototype.stream;
    autoInstrument({ anthropic: fake.Anthropic } as any); // re-install (guarded)
    expect(fake.Messages.prototype.stream).toBe(wrapped); // no double-wrap
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 2 — ALLOWED path proceeds on iterator AND emitter surfaces.
// ═══════════════════════════════════════════════════════════════════
describe("A2: allowed path (iterator + emitter completeness/ordering)", () => {
  test("2(a) for-await iterates to completion, inner pulled, no throw", async () => {
    const fake = makeFakeAnthropic();
    initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const seen: any[] = [];
    for await (const ev of stream) seen.push(ev);
    expect(seen.length).toBe(3);
    expect(stream.innerNextCount).toBeGreaterThan(0);
  });

  test("2(b-i) emitter-first: buffered tokens flushed in order, exactly once", async () => {
    const fake = makeFakeAnthropic();
    const { client } = initClient("dry_run", fake);
    const chk = deferred<any>();
    vi.spyOn(client, "check").mockReturnValue(chk.promise as any);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const got: string[] = [];
    const msgs: any[] = [];
    stream.on("text", (t: string) => got.push(t));
    stream.on("message", (m: any) => msgs.push(m));
    stream.releaseEmitter();
    await stream.emitDone.promise; // all events emitted (buffered)
    expect(got).toEqual([]); // nothing delivered before check
    chk.resolve({ status: "allowed", fail_open: false });
    await flush();
    expect(got).toEqual(["Hello", " world"]); // flushed in order, exactly once
    expect(msgs.length).toBe(1);
  });

  test("2(b-ii) check-first: live tokens delivered in order, exactly once", async () => {
    const fake = makeFakeAnthropic();
    initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const got: string[] = [];
    stream.on("text", (t: string) => got.push(t));
    await flush(); // let the check resolve (released)
    stream.releaseEmitter();
    await stream.emitDone.promise;
    await flush();
    expect(got).toEqual(["Hello", " world"]);
  });

  test("2(b-iii) handoff: buffered token + live token, no drop/duplicate at boundary", async () => {
    const fake = makeFakeAnthropic({ streamOpts: { useMidGate: true } });
    const { client } = initClient("dry_run", fake);
    const chk = deferred<any>();
    vi.spyOn(client, "check").mockReturnValue(chk.promise as any);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const got: string[] = [];
    stream.on("text", (t: string) => got.push(t));
    stream.releaseEmitter();
    await flush(); // first token emitted (buffered), loop paused at midGate
    expect(got).toEqual([]);
    chk.resolve({ status: "allowed", fail_open: false });
    await flush(); // flush buffered "Hello"
    expect(got).toEqual(["Hello"]);
    stream.releaseMid();
    await stream.emitDone.promise;
    await flush();
    expect(got).toEqual(["Hello", " world"]); // live " world", no dup/drop
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 3 — BLOCK throws + zero tokens on ANY surface.
// ═══════════════════════════════════════════════════════════════════
describe("A3: enforce block", () => {
  test("3(a-iter) iterator throws TokenPoliceBlockedError; inner never surfaced; abort called", async () => {
    const fake = makeFakeAnthropic();
    const { client } = initClient("enforce", fake);
    vi.spyOn(client, "check").mockResolvedValue({ status: "blocked", fail_open: false, reason: "x" } as any);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    await expect(
      (async () => {
        for await (const _ of stream) void _;
      })(),
    ).rejects.toBeInstanceOf(TokenPoliceBlockedError);
    expect(stream.innerNextCount).toBe(0); // inner next() never surfaced
    expect(stream.aborted).toBe(true);
    expect(stream.controllerAborted).toBe(true);
  });

  test("3(a-final) finalMessage throws TokenPoliceBlockedError; orig never called", async () => {
    const fake = makeFakeAnthropic();
    const { client } = initClient("enforce", fake);
    vi.spyOn(client, "check").mockResolvedValue({ status: "blocked", fail_open: false, reason: "x" } as any);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    await expect(stream.finalMessage()).rejects.toBeInstanceOf(TokenPoliceBlockedError);
    expect(stream.finalMessageCalled).toBe(false);
  });

  test("3(b-i) emitter-first: buffered tokens DROPPED on block; control events pass through", async () => {
    const fake = makeFakeAnthropic();
    const { client } = initClient("enforce", fake);
    const chk = deferred<any>();
    vi.spyOn(client, "check").mockReturnValue(chk.promise as any);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const got: string[] = [];
    const errs: any[] = [];
    stream.on("text", (t: string) => got.push(t));
    stream.on("error", (e: any) => errs.push(e));
    stream.releaseEmitter();
    await stream.emitDone.promise; // buffered
    chk.resolve({ status: "blocked", fail_open: false, reason: "x" });
    await flush();
    expect(got).toEqual([]); // dropped
    stream.emit("error", new Error("boom")); // control event still passes through
    expect(errs.length).toBe(1);
  });

  test("3(b-ii) check-first: live tokens SWALLOWED on block", async () => {
    const fake = makeFakeAnthropic();
    const { client } = initClient("enforce", fake);
    vi.spyOn(client, "check").mockResolvedValue({ status: "blocked", fail_open: false, reason: "x" } as any);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const got: string[] = [];
    stream.on("text", (t: string) => got.push(t));
    await flush(); // block settles, dropped=true
    stream.releaseEmitter();
    await stream.emitDone.promise;
    await flush();
    expect(got).toEqual([]);
  });
});

// ═══════════════════════════════════════════════════════════════════
// A3-OFF — gated unsubscribe: `off()` clears both the wrapper-held buffer
// AND the native emitter (the Anthropic MessageStream exposes ONLY `off` —
// no removeListener / removeAllListeners), so a customer that unsubscribes a
// token listener stops receiving events on both the buffered and live paths.
// ═══════════════════════════════════════════════════════════════════
describe("A3-OFF: gated off() unsubscribe", () => {
  test("off before release (buffered path) → cb NEVER fires on allow", async () => {
    const fake = makeFakeAnthropic();
    const { client } = initClient("dry_run", fake);
    const chk = deferred<any>();
    vi.spyOn(client, "check").mockReturnValue(chk.promise as any);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const got: string[] = [];
    const cb = (t: string) => got.push(t);
    stream.on("text", cb);
    expect(stream.off("text", cb)).toBe(stream); // chaining preserved (returns mgr)
    stream.releaseEmitter();
    await stream.emitDone.promise; // tokens emitted → buffered
    chk.resolve({ status: "allowed", fail_open: false });
    await flush();
    expect(got).toEqual([]); // held entry removed before release → never delivered
  });

  test("off after release (live path) → cb stops firing mid-stream", async () => {
    const fake = makeFakeAnthropic({ streamOpts: { useMidGate: true } });
    const { client } = initClient("dry_run", fake);
    vi.spyOn(client, "check").mockResolvedValue({ status: "allowed", fail_open: false } as any);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const got: string[] = [];
    const cb = (t: string) => got.push(t);
    stream.on("text", cb);
    await flush(); // check resolves → released
    stream.releaseEmitter();
    await flush(); // first token "Hello" delivered live; loop paused at midGate
    expect(got).toEqual(["Hello"]);
    stream.off("text", cb); // unsubscribe AFTER release (live-forward path)
    stream.releaseMid();
    await stream.emitDone.promise;
    await flush();
    expect(got).toEqual(["Hello"]); // " world" not delivered after off
  });

  test("once() removed before release never fires", async () => {
    const fake = makeFakeAnthropic();
    const { client } = initClient("dry_run", fake);
    const chk = deferred<any>();
    vi.spyOn(client, "check").mockReturnValue(chk.promise as any);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const got: string[] = [];
    const cb = (t: string) => got.push(t);
    stream.once("text", cb);
    stream.off("text", cb);
    stream.releaseEmitter();
    await stream.emitDone.promise;
    chk.resolve({ status: "allowed", fail_open: false });
    await flush();
    expect(got).toEqual([]);
  });

  test("unsubscribing a never-subscribed cb / unknown event is a harmless no-op", async () => {
    const fake = makeFakeAnthropic();
    initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const cb = () => {};
    expect(() => stream.off("text", cb)).not.toThrow();
    expect(() => stream.off("nonexistent-event", cb)).not.toThrow();
    expect(stream.off("text", cb)).toBe(stream); // still returns mgr for chaining
    // drain so no dangling gate/log work leaks into other tests
    stream.releaseEmitter();
    await stream.emitDone.promise;
    await flush();
  });

  test("off delegates to native for control (error) listeners", async () => {
    const fake = makeFakeAnthropic();
    initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const errs: any[] = [];
    const cb = (e: any) => errs.push(e);
    stream.on("error", cb); // control event → attached to native emitter
    stream.off("error", cb); // must clear it via native delegation
    stream.releaseEmitter();
    await stream.emitDone.promise;
    await flush();
    stream.emit("error", new Error("boom"));
    expect(errs).toEqual([]); // native listener removed → not fired
  });

  test("block still leaks zero tokens to a REMAINING subscriber after one is off'd", async () => {
    const fake = makeFakeAnthropic();
    const { client } = initClient("enforce", fake);
    const chk = deferred<any>();
    vi.spyOn(client, "check").mockReturnValue(chk.promise as any);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const gotA: string[] = [];
    const gotB: string[] = [];
    const cbA = (t: string) => gotA.push(t);
    const cbB = (t: string) => gotB.push(t);
    stream.on("text", cbA);
    stream.on("text", cbB);
    stream.off("text", cbA); // remove one; cbB remains
    stream.releaseEmitter();
    await stream.emitDone.promise; // buffered
    chk.resolve({ status: "blocked", fail_open: false, reason: "x" });
    await flush();
    expect(gotA).toEqual([]); // unsubscribed
    expect(gotB).toEqual([]); // remaining subscriber still gets zero on block
  });

  test("gate never invents removeListener / removeAllListeners (surface parity)", async () => {
    const fake = makeFakeAnthropic();
    initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    // The real MessageStream exposes on/once/off/emitted only — we mirror off and
    // must NOT fabricate EventEmitter methods the class never had.
    expect(stream.removeListener).toBeUndefined();
    expect(stream.removeAllListeners).toBeUndefined();
    expect(typeof stream.off).toBe("function");
    stream.releaseEmitter();
    await stream.emitDone.promise;
    await flush();
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 4 — usage /log'd from finalMessage with correct fields.
// ═══════════════════════════════════════════════════════════════════
describe("A4: usage logged from finalMessage", () => {
  test("one anthropic row with model + input/output/cache tokens", async () => {
    const fake = makeFakeAnthropic();
    const { logSpy } = initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    await stream.finalMessage();
    await flush();
    // T-N3 / G4: usage-bearing mock stream MUST dispatch exactly one row with tokens > 0.
    const rows = assertStreamedLogPresent(logSpy, "anthropic");
    const r = rows[0];
    expect(r[4]).toBe("claude-3"); // model
    expect(r[6]).toBe(10); // input_tokens
    expect(r[7]).toBe(5); // output_tokens
    expect(r[8]).toBe(3); // cache_read → cachedTokens
  });

  // The default fixture requests and echoes the SAME string, so it cannot
  // see the split. A reroute target is an alias (`claude-haiku-4-5`) that the
  // provider echoes as its dated snapshot — the row must stay on the alias, the
  // same string the auto span path reports, or cost-by-model shows two models.
  test("dated-snapshot echo of the requested alias logs the alias", async () => {
    const fake = makeFakeAnthropic({
      streamOpts: {
        events: [
          {
            type: "message_start",
            message: {
              model: "claude-haiku-4-5-20251001",
              usage: { input_tokens: 10, output_tokens: 0 },
            },
          },
          { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Hello" } },
          { type: "message_delta", usage: { output_tokens: 5 }, delta: { stop_reason: "end_turn" } },
        ],
        final: {
          model: "claude-haiku-4-5-20251001",
          stop_reason: "end_turn",
          usage: { input_tokens: 10, output_tokens: 5 },
          content: [{ type: "text", text: "Hello world" }],
        },
      },
    });
    const { logSpy } = initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream({
      model: "claude-haiku-4-5",
      messages: [{ role: "user", content: "hi" }],
    });
    await stream.finalMessage();
    await flush();
    const rows = assertStreamedLogPresent(logSpy, "anthropic");
    expect(rows[0][4]).toBe("claude-haiku-4-5");
  });

  // NEW-1: on Bedrock the model id IS the price — a cross-region inference
  // profile (`us.anthropic.claude-haiku-4-5-20251001-v1:0`) bills at a
  // different rate than its bare echo (`claude-haiku-4-5-20251001`). This is
  // the exact seam the production bug shipped from: `_wrapAnthropicMessageStream`
  // → `logFinal` → `_logManual` (enforcer.ts ~8113), driven here through the
  // REAL `client.messages.stream()` wrapper and a fake MessageStream whose
  // eager emitter echoes the bare model id — not the `preferRequestedModel`
  // helper called directly.
  test("Bedrock CRIS profile requested, bare-model echo → the profile id is logged", async () => {
    const fake = makeFakeAnthropic({
      streamOpts: {
        events: [
          {
            type: "message_start",
            message: {
              model: "claude-haiku-4-5-20251001",
              usage: { input_tokens: 10, output_tokens: 0 },
            },
          },
          { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Hello" } },
          { type: "message_delta", usage: { output_tokens: 5 }, delta: { stop_reason: "end_turn" } },
        ],
        final: {
          model: "claude-haiku-4-5-20251001",
          stop_reason: "end_turn",
          usage: { input_tokens: 10, output_tokens: 5 },
          content: [{ type: "text", text: "Hello world" }],
        },
      },
    });
    const { logSpy } = initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream({
      model: "us.anthropic.claude-haiku-4-5-20251001-v1:0",
      messages: [{ role: "user", content: "hi" }],
    });
    await stream.finalMessage();
    await flush();
    const rows = assertStreamedLogPresent(logSpy, "anthropic");
    expect(rows[0][4]).toBe("us.anthropic.claude-haiku-4-5-20251001-v1:0");
  });

  // Negative twin: a genuinely different served model must keep its echo —
  // the family gate must not over-widen at the wire level either.
  test("Bedrock request, genuinely different served model → the echo is logged", async () => {
    const fake = makeFakeAnthropic({
      streamOpts: {
        events: [
          {
            type: "message_start",
            message: {
              model: "claude-sonnet-4-6",
              usage: { input_tokens: 10, output_tokens: 0 },
            },
          },
          { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Hello" } },
          { type: "message_delta", usage: { output_tokens: 5 }, delta: { stop_reason: "end_turn" } },
        ],
        final: {
          model: "claude-sonnet-4-6",
          stop_reason: "end_turn",
          usage: { input_tokens: 10, output_tokens: 5 },
          content: [{ type: "text", text: "Hello world" }],
        },
      },
    });
    const { logSpy } = initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream({
      model: "us.anthropic.claude-haiku-4-5-20251001-v1:0",
      messages: [{ role: "user", content: "hi" }],
    });
    await stream.finalMessage();
    await flush();
    const rows = assertStreamedLogPresent(logSpy, "anthropic");
    expect(rows[0][4]).toBe("claude-sonnet-4-6");
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 5 — fail-open on EVERY new failure point.
// ═══════════════════════════════════════════════════════════════════
describe("A5: fail-open", () => {
  test("5(a) check throws non-block error → iteration completes, no throw/block", async () => {
    const fake = makeFakeAnthropic();
    const { client } = initClient("enforce", fake);
    vi.spyOn(client, "check").mockRejectedValue(new Error("check boom"));
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const seen: any[] = [];
    await expect(
      (async () => {
        for await (const ev of stream) seen.push(ev);
      })(),
    ).resolves.toBeUndefined();
    expect(seen.length).toBe(3);
  });

  test("5(b) manual log emit throws → iteration completes (telemetry loss only)", async () => {
    const fake = makeFakeAnthropic();
    const { logSpy } = initAllowed(fake);
    logSpy.mockImplementation(() => {
      throw new Error("log boom");
    });
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const seen: any[] = [];
    await expect(
      (async () => {
        for await (const ev of stream) seen.push(ev);
      })(),
    ).resolves.toBeUndefined();
    expect(seen.length).toBe(3);
  });

  test("5(c) finalMessage throws → for-await iteration still completes, no throw", async () => {
    const fake = makeFakeAnthropic({ streamOpts: { finalMessageThrows: true } });
    initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const seen: any[] = [];
    await expect(
      (async () => {
        for await (const ev of stream) seen.push(ev);
      })(),
    ).resolves.toBeUndefined();
    expect(seen.length).toBe(3);
  });

  test("5(d) latch-marker defineProperty throws → completes, no throw", async () => {
    const fake = makeFakeAnthropic({ streamOpts: { blockLatchWrite: true } });
    initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const seen: any[] = [];
    await expect(
      (async () => {
        for await (const ev of stream) seen.push(ev);
      })(),
    ).resolves.toBeUndefined();
    expect(seen.length).toBe(3);
  });

  test("5(e) stream wrapper sets no session-level suppress flag (per-call ALS)", async () => {
    // Suppression moved off the session onto a per-call async-context window, so
    // the wrapper must leave NO `_suppressAnthropicOtelStream` on the session.
    const fake = makeFakeAnthropic();
    initAllowed(fake);
    await tpSession({}, async (s: any) => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      const seen: any[] = [];
      await expect(
        (async () => {
          for await (const ev of stream) seen.push(ev);
        })(),
      ).resolves.toBeUndefined();
      expect(seen.length).toBe(3);
      expect(s._suppressAnthropicOtelStream).toBeUndefined();
    });
  });

  test("5(e2) onStart tagging is fail-open on a frozen instrumentor span", () => {
    // Inside the suppression window, tagging a non-extensible span must never
    // throw out of onStart (defense-in-depth for exotic span objects).
    const proc = new TokenPoliceSpanProcessor();
    const span = Object.freeze({
      name: "anthropic.chat",
      attributes: { "gen_ai.system": "anthropic" },
      instrumentationScope: { name: "@traceloop/instrumentation-anthropic" },
      setAttribute() {
        return this;
      },
    });
    expect(() =>
      runWithAnthropicStreamOtelSuppress(() => proc.onStart(span as any)),
    ).not.toThrow();
  });

  test("5(f) .on interception setup throws (non-configurable on) → pass-through, completes", async () => {
    const fake = makeFakeAnthropic({ streamOpts: { nonConfigurableOn: true } });
    initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const got: string[] = [];
    // native .on still registers (degraded pass-through) — customer gets tokens
    expect(() => stream.on("text", (t: string) => got.push(t))).not.toThrow();
    stream.releaseEmitter();
    await stream.emitDone.promise;
    await flush();
    expect(got).toEqual(["Hello", " world"]); // pass-through delivered
    const seen: any[] = [];
    await expect(
      (async () => {
        for await (const ev of stream) seen.push(ev);
      })(),
    ).resolves.toBeUndefined();
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 6 — customer's OWN stream error re-propagates unswallowed.
// ═══════════════════════════════════════════════════════════════════
describe("A6: customer stream error propagates", () => {
  test("inner next() rejection surfaces the SAME error", async () => {
    const sentinel = new Error("provider exploded");
    const fake = makeFakeAnthropic({ streamOpts: { iterError: sentinel, iterErrorIndex: 1 } });
    initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    let caught: any;
    try {
      for await (const _ of stream) void _;
    } catch (e) {
      caught = e;
    }
    expect(caught).toBe(sentinel);
    expect(caught).not.toBeInstanceOf(TokenPoliceBlockedError);
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 7 — EXACTLY ONE /log row (never 0, never 2).
// ═══════════════════════════════════════════════════════════════════
describe("A7: exactly-one-row latch", () => {
  test("7(a) independent stream: finalMessage + drain both reach log → 1", async () => {
    const fake = makeFakeAnthropic();
    const { logSpy } = initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    await stream.finalMessage(); // log point 1
    const seen: any[] = [];
    for await (const ev of stream) seen.push(ev); // drain → log point 2
    await flush();
    expect(anthropicRows(logSpy).length).toBe(1); // latch dedups
  });

  // 7(b)/(b-reverse) model REAL SDK delegation: `.stream()` internally fires
  // `create({stream:true})` WITHIN its factory window (delegateToCreate), so the
  // create tap inherits the `.stream()` latch via async context. `_delegateRaw`
  // is that internally-delegated raw stream — draining it drives the create-tap
  // log point, exactly the second observer a real delegating SDK produces.
  test("7(b) delegation: .stream finalMessage AND delegated create tap → 1 (not 0/2)", async () => {
    const fake = makeFakeAnthropic({ streamOpts: { delegateToCreate: true } });
    const { logSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const messages: any = new fake.Anthropic().messages;
      const stream: any = messages.stream(BODY); // delegates to create within factory
      await stream.finalMessage(); // .stream wrapper log point
      const raw: any = await stream._delegateRaw; // the internally-delegated raw stream
      for await (const _ of raw) void _; // drain → create-tap finalize log point
      await flush();
      expect(anthropicRows(logSpy).length).toBe(1);
    });
  });

  test("7(b-reverse) delegated create tap drains FIRST, then .stream finalMessage → still 1", async () => {
    const fake = makeFakeAnthropic({ streamOpts: { delegateToCreate: true } });
    const { logSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const messages: any = new fake.Anthropic().messages;
      const stream: any = messages.stream(BODY); // delegates to create within factory
      const raw: any = await stream._delegateRaw;
      for await (const _ of raw) void _; // create-tap logs first
      await stream.finalMessage(); // .stream wrapper skips (latched)
      await flush();
      expect(anthropicRows(logSpy).length).toBe(1);
    });
  });

  // THE DEFECT REPRO: an independent `.stream()` fully consumed, THEN a separate,
  // independent `create({stream:true})` in the SAME session scope. The old
  // session-scoped latch was already "taken" by the .stream() call, so the later
  // create silently skipped its only log point → total metering loss (1 row). With
  // per-call async-context scoping the create sees no inherited latch → 2 rows.
  test("7(d) later independent create({stream:true}) after a .stream() in one session → 2", async () => {
    const fake = makeFakeAnthropic();
    const { logSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const messages: any = new fake.Anthropic().messages;
      const stream: any = messages.stream(BODY);
      await stream.finalMessage();
      const seen: any[] = [];
      for await (const ev of stream) seen.push(ev); // fully drain the .stream() call
      // Separate, independent bypass call — NOT delegated from any .stream().
      const raw: any = await messages.create({ ...BODY, stream: true });
      for await (const _ of raw) void _;
      await flush();
      expect(anthropicRows(logSpy).length).toBe(2);
    });
  });

  test("7(e) two concurrent independent create({stream:true}) in one session → 2", async () => {
    const fake = makeFakeAnthropic();
    const { logSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const messages: any = new fake.Anthropic().messages;
      const [r1, r2]: any[] = await Promise.all([
        messages.create({ ...BODY, stream: true }),
        messages.create({ ...BODY, stream: true }),
      ]);
      await Promise.all([
        (async () => { for await (const _ of r1) void _; })(),
        (async () => { for await (const _ of r2) void _; })(),
      ]);
      await flush();
      expect(anthropicRows(logSpy).length).toBe(2);
    });
  });

  test("7(c) Traceloop anthropic OTel duplicate span suppressed (per-call) → still 1", async () => {
    const proc = new TokenPoliceSpanProcessor();
    // The instrumentor's duplicate span starts (onStart) synchronously inside the
    // manual `.stream()` construction window — reproduced via onNativeStreamStart.
    const dupSpan = makeAnthropicSpan();
    const fake = makeFakeAnthropic({
      streamOpts: { onNativeStreamStart: () => proc.onStart(dupSpan as any) },
    });
    const { logSpy } = initAllowed(fake);
    await tpSession({}, async (s: any) => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      await stream.finalMessage();
      await flush();
      expect(s._suppressAnthropicOtelStream).toBeUndefined(); // no session flag
      expect(anthropicRows(logSpy).length).toBe(1); // one manual row

      // The tagged duplicate span ends → dropped, no new row.
      const before = logSpy.mock.calls.length;
      proc.onEnd(dupSpan as any);
      await flush();
      expect(logSpy.mock.calls.length).toBe(before);

      // A later INDEPENDENT anthropic span (started outside any window) is NOT
      // tagged → survives (no stale suppression leaks across calls).
      const indep = makeAnthropicSpan();
      proc.onStart(indep as any); // no suppression window active
      proc.onEnd(indep as any);
      await flush();
      expect(logSpy.mock.calls.length).toBe(before + 1); // survived → logged
      expect(anthropicRows(logSpy).length).toBe(2); // manual + surviving indep
    });
  });

  test("7(f) two concurrent same-session .stream() → both duplicates dropped, ZERO manual rows lost", async () => {
    const proc = new TokenPoliceSpanProcessor();
    // A distinct duplicate span per `.stream()` call, tagged in ITS OWN window.
    const dupSpans = [makeAnthropicSpan(), makeAnthropicSpan()];
    let i = 0;
    const fake = makeFakeAnthropic({
      streamOpts: { onNativeStreamStart: () => proc.onStart(dupSpans[i++] as any) },
    });
    const { logSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const s1: any = new fake.Anthropic().messages.stream(BODY);
      const s2: any = new fake.Anthropic().messages.stream(BODY);
      await Promise.all([s1.finalMessage(), s2.finalMessage()]);
      await flush();
      // 4 rows would exist if the duplicates weren't suppressed (2 manual + 2
      // instrumentor). The manual rows are the two we keep.
      expect(anthropicRows(logSpy).length).toBe(2);
      const before = logSpy.mock.calls.length;
      // Both instrumentor duplicate spans end (order-independent) → both dropped.
      proc.onEnd(dupSpans[0] as any);
      proc.onEnd(dupSpans[1] as any);
      await flush();
      expect(logSpy.mock.calls.length).toBe(before); // no duplicate rows added
      expect(anthropicRows(logSpy).length).toBe(2); // exactly the 2 manual rows
    });
  });

  test("7(g) concurrent manual stream + non-stream anthropic → wrong-span-eaten regression", async () => {
    const proc = new TokenPoliceSpanProcessor();
    const streamDup = makeAnthropicSpan(); // the stream's duplicate → must DROP
    // A genuine non-stream anthropic instrumentor span, distinguishable by model.
    const nonStreamSpan = makeAnthropicSpan();
    nonStreamSpan.attributes["gen_ai.request.model"] = "claude-nonstream";
    const fake = makeFakeAnthropic({
      streamOpts: { onNativeStreamStart: () => proc.onStart(streamDup as any) },
    });
    const { logSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const s1: any = new fake.Anthropic().messages.stream(BODY);
      // Concurrent non-stream call: its span starts OUTSIDE any manual-stream
      // window → must NOT be tagged.
      proc.onStart(nonStreamSpan as any);
      await s1.finalMessage();
      await flush();
      const before = logSpy.mock.calls.length; // includes the 1 manual stream row
      // The non-stream span ends FIRST — the exact case the old session boolean
      // mis-ate (it consumed the flag and lost this real row). It must SURVIVE;
      // the stream duplicate ends after → dropped.
      proc.onEnd(nonStreamSpan as any);
      proc.onEnd(streamDup as any);
      await flush();
      // Exactly one new row, and it is the NON-STREAM survivor (model pins it).
      expect(logSpy.mock.calls.length).toBe(before + 1);
      const newCall = logSpy.mock.calls[logSpy.mock.calls.length - 1];
      expect(newCall[4]).toBe("claude-nonstream"); // survivor is the real call
    });
  });
});

let _anthropicSpanSeq = 0;
// A minimally OTel-Span-shaped duplicate span: async-suppression tags it in
// onStart; if NOT suppressed it is fully loggable (spanContext + setAttribute
// present) so survivor-vs-dropped assertions are observable.
function makeAnthropicSpan(): any {
  const spanId = `deadbeef${(_anthropicSpanSeq++).toString(16).padStart(8, "0")}`;
  const s: any = {
    name: "anthropic.chat",
    attributes: {
      "gen_ai.system": "anthropic",
      "gen_ai.request.model": "claude-3",
      "gen_ai.usage.input_tokens": 7,
      "gen_ai.usage.output_tokens": 2,
      "tp.trace_id": "t1",
      "tp.span_order": 99,
    },
    instrumentationScope: { name: "@traceloop/instrumentation-anthropic" },
    startTime: [1000, 0],
    endTime: [1001, 0],
    setAttribute(k: string, v: any) {
      s.attributes[k] = v;
      return s;
    },
    spanContext() {
      return { traceId: "a".repeat(32), spanId };
    },
  };
  return s;
}

// ═══════════════════════════════════════════════════════════════════
// Assertion 8 — framework scope → pass-through.
// ═══════════════════════════════════════════════════════════════════
describe("A8: framework scope pass-through", () => {
  test("inLangchain → raw manager returned, no side effects", async () => {
    const fake = makeFakeAnthropic();
    const { logSpy } = initAllowed(fake);
    await tpSession({}, async (s: any) => {
      s.enterLangchain();
      try {
        const stream: any = new fake.Anthropic().messages.stream(BODY);
        // raw fake stream: no gate override → identity + native surface preserved
        expect(stream instanceof FakeMessageStream).toBe(true);
        expect((stream as any).__tpLogLatch).toBeUndefined();
        stream.releaseEmitter();
        await stream.emitDone.promise;
        await flush();
        expect(anthropicRows(logSpy).length).toBe(0); // no manual log
        expect(s._suppressAnthropicOtelStream).toBeFalsy(); // no suppress flag
      } finally {
        s.exitLangchain();
      }
    });
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 9 — AsyncMessages.stream covered (allowed / block / single-log).
// ═══════════════════════════════════════════════════════════════════
describe("A9: AsyncMessages.stream parity", () => {
  test("allowed proceeds + single log", async () => {
    const fake = makeFakeAnthropic();
    const { logSpy } = initAllowed(fake);
    const stream: any = new fake.AsyncMessages().stream(BODY);
    const seen: any[] = [];
    for await (const ev of stream) seen.push(ev);
    await flush();
    expect(seen.length).toBe(3);
    expect(anthropicRows(logSpy).length).toBe(1);
  });

  test("block throws TokenPoliceBlockedError", async () => {
    const fake = makeFakeAnthropic();
    const { client } = initClient("enforce", fake);
    vi.spyOn(client, "check").mockResolvedValue({ status: "blocked", fail_open: false, reason: "x" } as any);
    const stream: any = new fake.AsyncMessages().stream(BODY);
    await expect(stream.finalMessage()).rejects.toBeInstanceOf(TokenPoliceBlockedError);
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 10 — init isolation: broken anthropic module never throws.
// ═══════════════════════════════════════════════════════════════════
describe("A10: init isolation", () => {
  function throwingProxy(): any {
    return new Proxy(
      {},
      {
        get() {
          throw new Error("boom: throwing property access");
        },
      },
    );
  }
  function makeValidOpenAI(): any {
    function create(this: any) {
      return Promise.resolve({});
    }
    function Completions(this: any) {}
    Completions.prototype.create = create;
    const Chat = { Completions };
    function OpenAI(this: any) {}
    (OpenAI as any).Chat = Chat;
    return OpenAI;
  }
  test("throwing anthropic module → init does not throw; a valid target still instruments", () => {
    const OpenAI = makeValidOpenAI();
    const orig = OpenAI.Chat.Completions.prototype.create;
    expect(() =>
      autoInstrument({ anthropic: throwingProxy(), openai: OpenAI } as any),
    ).not.toThrow();
    expect(OpenAI.Chat.Completions.prototype.create).not.toBe(orig);
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 11 — full MessageStream surface preserved (same object).
// ═══════════════════════════════════════════════════════════════════
describe("A11: surface preservation", () => {
  test("returned === fake stream; controller/abort/tee identity-preserved; gated members callable", async () => {
    const fake = makeFakeAnthropic();
    // capture the concrete stream the SDK returned, beneath the enforcer patch.
    const realStream = fake.Messages.prototype.stream;
    let captured: FakeMessageStream | null = null;
    (fake.Messages.prototype as any).stream = function (this: any, b: any) {
      const s = realStream.call(this, b);
      captured = s;
      return s;
    };
    initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    expect(stream).toBe(captured); // SAME object reference
    expect(stream.controller).toBe(captured!.controller); // untouched member
    expect(stream.abort).toBe(captured!.abort);
    expect(stream.tee).toBe(captured!.tee);
    expect(typeof stream.on).toBe("function"); // gated overrides callable
    expect(typeof stream.once).toBe("function");
    expect(typeof stream.finalMessage).toBe("function");
    expect(typeof stream[Symbol.asyncIterator]).toBe("function");
  });
});

// ═══════════════════════════════════════════════════════════════════
// Assertion 15 — dry_run + off: would-block never throws.
// ═══════════════════════════════════════════════════════════════════
describe("A15: dry_run / off never throw on would-block", () => {
  test("dry_run + would-block → iteration completes, no throw, tokens surfaced", async () => {
    const fake = makeFakeAnthropic();
    const { client } = initClient("dry_run", fake);
    vi.spyOn(client, "check").mockResolvedValue({ status: "blocked", fail_open: false, reason: "x" } as any);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const seen: any[] = [];
    await expect(
      (async () => {
        for await (const ev of stream) seen.push(ev);
      })(),
    ).resolves.toBeUndefined();
    expect(seen.length).toBe(3); // tokens surfaced
  });

  test("off (runtime) → pre-flight no-op, iteration completes", async () => {
    const fake = makeFakeAnthropic();
    // install the wrapper (dry_run installs), then flip runtime firewall to off.
    const { client } = initClient("dry_run", fake);
    const checkSpy = vi.spyOn(client, "check");
    (client as any).firewall = "off";
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const seen: any[] = [];
    for await (const ev of stream) seen.push(ev);
    expect(seen.length).toBe(3);
    expect(checkSpy).not.toHaveBeenCalled(); // pre-flight was a no-op
  });
});

// ═══════════════════════════════════════════════════════════════════
// ND-01 — finalText() / done() / emitted() gates + inputJson/thinking/
// citation/signature token events. (Contract assertions A6–A17, A21, A22.)
// ═══════════════════════════════════════════════════════════════════

// ── Block path (enforce + check → blocked) ─────────────────────────
describe("ND-01 block: finalText/done/emitted gated", () => {
  test("A6: finalText() rejects with TokenPoliceBlockedError; no native text path; abort called", async () => {
    const fake = makeFakeAnthropic();
    const { client } = initClient("enforce", fake);
    vi.spyOn(client, "check").mockResolvedValue({ status: "blocked", fail_open: false, reason: "x" } as any);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    await expect(stream.finalText()).rejects.toBeInstanceOf(TokenPoliceBlockedError);
    expect(stream._doneMeta.calls).toBe(0); // native text path never entered
    expect(stream.aborted).toBe(true);
  });

  test("A7: done() rejects with TokenPoliceBlockedError; native completion never awaited", async () => {
    const fake = makeFakeAnthropic();
    const { client } = initClient("enforce", fake);
    vi.spyOn(client, "check").mockResolvedValue({ status: "blocked", fail_open: false, reason: "x" } as any);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    await expect(stream.done()).rejects.toBeInstanceOf(TokenPoliceBlockedError);
    expect(stream._doneMeta.calls).toBe(0); // native done() never reached
    expect(stream.aborted).toBe(true);
  });

  test("A8: emitted('inputJson') rejects with block AND does not hang", async () => {
    const fake = makeFakeAnthropic({
      streamOpts: { extraEmits: [{ event: "inputJson", value: { partial: "{" } }] },
    });
    const { client } = initClient("enforce", fake);
    vi.spyOn(client, "check").mockResolvedValue({ status: "blocked", fail_open: false, reason: "x" } as any);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const p = stream.emitted("inputJson"); // routes through the gated once()
    stream.releaseEmitter();
    // A regression to the held-map hang makes raceHang reject with HANG (≠ block).
    await expect(raceHang(p)).rejects.toBeInstanceOf(TokenPoliceBlockedError);
  });

  test("A9: on('inputJson') buffered events DROPPED on block; error control passes through", async () => {
    const fake = makeFakeAnthropic({
      streamOpts: { extraEmits: [{ event: "inputJson", value: { partial: "{" } }] },
    });
    const { client } = initClient("enforce", fake);
    const chk = deferred<any>();
    vi.spyOn(client, "check").mockReturnValue(chk.promise as any);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const got: any[] = [];
    const errs: any[] = [];
    stream.on("inputJson", (v: any) => got.push(v));
    stream.on("error", (e: any) => errs.push(e));
    stream.releaseEmitter();
    await stream.emitDone.promise; // buffered
    chk.resolve({ status: "blocked", fail_open: false, reason: "x" });
    await flush();
    expect(got).toEqual([]); // dropped on block
    stream.emit("error", new Error("boom")); // control event still delivered
    expect(errs.length).toBe(1);
  });
});

// ── Allow path (dry_run + check → allowed) ─────────────────────────
describe("ND-01 allow: finalText/done/emitted deliver + meter", () => {
  test("A10: finalText() returns full text + exactly ONE anthropic row", async () => {
    const fake = makeFakeAnthropic();
    const { logSpy } = initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const text = await stream.finalText();
    await flush();
    expect(text).toBe("Hello world");
    expect(anthropicRows(logSpy).length).toBe(1);
  });

  test("A11: done() resolves (void) + exactly ONE anthropic row", async () => {
    const fake = makeFakeAnthropic();
    const { logSpy } = initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const r = await stream.done();
    await flush();
    expect(r).toBeUndefined();
    expect(anthropicRows(logSpy).length).toBe(1);
  });

  test("A12: emitted('inputJson') resolves with the emitted value (no hang/reject)", async () => {
    const fake = makeFakeAnthropic({
      streamOpts: { extraEmits: [{ event: "inputJson", value: { tok: "x" } }] },
    });
    initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const p = stream.emitted("inputJson");
    stream.releaseEmitter();
    const v = await raceHang(p); // guard: allow-path must not hang either (T-1)
    expect(v).toEqual({ tok: "x" });
  });

  test("A13: on('inputJson') receives every payload in order, exactly once", async () => {
    const fake = makeFakeAnthropic({
      streamOpts: { extraEmits: [{ event: "inputJson", value: 1 }, { event: "inputJson", value: 2 }] },
    });
    initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const got: any[] = [];
    stream.on("inputJson", (v: any) => got.push(v));
    stream.releaseEmitter();
    await stream.emitDone.promise;
    await flush();
    expect(got).toEqual([1, 2]);
  });
});

// ── Metering parity (AM-2) ─────────────────────────────────────────
describe("ND-01 metering parity", () => {
  test("A14: finalText-only, done-only, finalMessage-only rows are field-identical", async () => {
    const fake = makeFakeAnthropic();
    const { logSpy } = initAllowed(fake);
    const stFT: any = new fake.Anthropic().messages.stream(BODY);
    await stFT.finalText();
    const stDone: any = new fake.Anthropic().messages.stream(BODY);
    await stDone.done();
    const stFM: any = new fake.Anthropic().messages.stream(BODY);
    await stFM.finalMessage();
    await flush();
    const rows = anthropicRows(logSpy);
    expect(rows.length).toBe(3); // three independent streams, one row each
    const [rFT, rDone, rFM] = rows;
    for (const r of [rFT, rDone, rFM]) {
      expect(r[4]).toBe("claude-3"); // model
      expect(r[6]).toBe(10); // input_tokens
      expect(r[7]).toBe(5); // output_tokens
      expect(r[8]).toBe(3); // cache_read → cachedTokens
    }
    // full raw usage identity (incl. cache_creation_input_tokens) from the shared
    // receivedMessages.at(-1) source — not two divergent rows that merely dedupe.
    const rawFT = rFT[13]?.usage?.raw;
    expect(rawFT?.cache_creation_input_tokens).toBe(2);
    expect(rawFT?.cache_read_input_tokens).toBe(3);
    expect(rDone[13]?.usage?.raw).toEqual(rawFT);
    expect(rFM[13]?.usage?.raw).toEqual(rawFT);
  });

  test("A14b: mixed consumer (finalText + done + iterate + finalMessage) → ONE row, no deadlock", async () => {
    const fake = makeFakeAnthropic();
    const { logSpy } = initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    // Concurrent, closing adversarial concern (a): finalText() + iterate together.
    await raceHang(
      Promise.all([
        stream.finalText(),
        stream.done(),
        stream.finalMessage(),
        (async () => {
          const seen: any[] = [];
          for await (const ev of stream) seen.push(ev);
          return seen;
        })(),
      ]),
      1000,
    );
    await flush();
    expect(anthropicRows(logSpy).length).toBe(1); // latch dedups every path
  });
});

// ── Forward-compat gating: thinking / citation / signature (T-3) ───
describe("ND-01 forward-compat token events", () => {
  test("A15fc-block: thinking/citation/signature each DROPPED on block (proves membership)", async () => {
    const fake = makeFakeAnthropic({
      streamOpts: {
        extraEmits: [
          { event: "thinking", value: "th" },
          { event: "citation", value: "ci" },
          { event: "signature", value: "si" },
        ],
      },
    });
    const { client } = initClient("enforce", fake);
    const chk = deferred<any>();
    vi.spyOn(client, "check").mockReturnValue(chk.promise as any);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const got: Record<string, any[]> = { thinking: [], citation: [], signature: [] };
    stream.on("thinking", (v: any) => got.thinking.push(v));
    stream.on("citation", (v: any) => got.citation.push(v));
    stream.on("signature", (v: any) => got.signature.push(v));
    stream.releaseEmitter();
    await stream.emitDone.promise; // buffered
    chk.resolve({ status: "blocked", fail_open: false, reason: "x" });
    await flush();
    // A non-gated event would pass through native .on and be delivered here — the
    // empty arrays prove each name IS in the gated token-event set.
    expect(got.thinking).toEqual([]);
    expect(got.citation).toEqual([]);
    expect(got.signature).toEqual([]);
  });

  test("A15fc-allow: thinking/citation/signature each delivered on allow", async () => {
    const fake = makeFakeAnthropic({
      streamOpts: {
        extraEmits: [
          { event: "thinking", value: "th" },
          { event: "citation", value: "ci" },
          { event: "signature", value: "si" },
        ],
      },
    });
    initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const got: Record<string, any[]> = { thinking: [], citation: [], signature: [] };
    stream.on("thinking", (v: any) => got.thinking.push(v));
    stream.on("citation", (v: any) => got.citation.push(v));
    stream.on("signature", (v: any) => got.signature.push(v));
    stream.releaseEmitter();
    await stream.emitDone.promise;
    await flush();
    expect(got.thinking).toEqual(["th"]);
    expect(got.citation).toEqual(["ci"]);
    expect(got.signature).toEqual(["si"]);
  });

  test("A15fc-noop: never-emitted forward-compat event → listener never fires, no error", async () => {
    const fake = makeFakeAnthropic(); // no extraEmits → thinking never emitted (0.30.x)
    initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    let fired = false;
    stream.on("thinking", () => {
      fired = true;
    });
    stream.releaseEmitter();
    await stream.emitDone.promise;
    await flush();
    expect(fired).toBe(false);
  });
});

// ── Fail-open + setup-throw degradation ────────────────────────────
describe("ND-01 fail-open + degradation", () => {
  test("A16: check throws non-block → finalText/done/emitted behave natively (no throw/block)", async () => {
    const fake = makeFakeAnthropic({
      streamOpts: { extraEmits: [{ event: "inputJson", value: { ok: 1 } }] },
    });
    const { client } = initClient("enforce", fake);
    vi.spyOn(client, "check").mockRejectedValue(new Error("check boom"));
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const p = stream.emitted("inputJson");
    stream.releaseEmitter();
    const text = await stream.finalText();
    const d = await stream.done();
    const v = await raceHang(p);
    expect(text).toBe("Hello world"); // native text surfaced (fail-open)
    expect(d).toBeUndefined();
    expect(v).toEqual({ ok: 1 });
  });

  test("A17: non-extensible mgr → all overrides degrade to native; drains, no hang, no throw", async () => {
    const fake = makeFakeAnthropic({
      streamOpts: { nonExtensible: true, extraEmits: [{ event: "inputJson", value: { z: 1 } }] },
    });
    initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const p = stream.emitted("inputJson"); // native emitted (override degraded)
    stream.releaseEmitter();
    const seen: any[] = [];
    await expect(
      (async () => {
        for await (const ev of stream) seen.push(ev);
      })(),
    ).resolves.toBeUndefined();
    expect(seen.length).toBe(3); // native iterator drains
    const text = await stream.finalText(); // native finalText still callable
    expect(text).toBe("Hello world");
    await expect(stream.done()).resolves.toBeUndefined(); // native done callable
    const v = await raceHang(p);
    expect(v).toEqual({ z: 1 }); // native emitted resolves (no hang)
  });
});

// ── Fake-stream fidelity + block-abort unhandled-rejection hygiene ──
describe("ND-01 fidelity + abort hygiene", () => {
  test("A21: finalMessage() and finalText() route THROUGH the gated done()", async () => {
    const fake = makeFakeAnthropic();
    initAllowed(fake);
    const sFM: any = new fake.Anthropic().messages.stream(BODY);
    await sFM.finalMessage();
    await flush();
    expect(sFM._doneMeta.calls).toBeGreaterThan(0); // coupling present, not assumed
    const sFT: any = new fake.Anthropic().messages.stream(BODY);
    await sFT.finalText();
    await flush();
    expect(sFT._doneMeta.calls).toBeGreaterThan(0);
  });

  test("A22: block-triggered abort via done()/finalText() → ZERO unhandled rejections", async () => {
    const captured: any[] = [];
    const handler = (e: any) => captured.push(e);
    process.on("unhandledRejection", handler);
    try {
      const fake = makeFakeAnthropic();
      const { client } = initClient("enforce", fake);
      vi.spyOn(client, "check").mockResolvedValue({ status: "blocked", fail_open: false, reason: "x" } as any);
      const s1: any = new fake.Anthropic().messages.stream(BODY);
      await expect(s1.done()).rejects.toBeInstanceOf(TokenPoliceBlockedError);
      const s2: any = new fake.Anthropic().messages.stream(BODY);
      await expect(s2.finalText()).rejects.toBeInstanceOf(TokenPoliceBlockedError);
      // Let the (would-be) unhandled-rejection detector run.
      await flush();
      await flush();
      const abortRejections = captured.filter(
        (e: any) => e && e.name === "APIUserAbortError",
      );
      expect(abortRejections).toEqual([]); // AM-3 swallow listeners neutralized it
    } finally {
      process.off("unhandledRejection", handler);
    }
  });
});

// ── one observation per (rule, outcome) per streamed call ──────────
// An Anthropic `.stream()` call used to run TWO pre-flights for ONE request:
// this file's `.stream()` wrapper, and the `create({stream:true})` the SDK
// internally delegates to (modeled by `delegateToCreate`). They evaluate the
// same rules, so every observation was pushed twice and REROUTE_REJECTED audit
// counts came out at ~2x the number of streamed calls. TWO fixes stack here and
// this suite must keep passing under both:
//   1. observation dedup — the per-call stream latch (already deduping the /log
//      row) also carries an observation dedup set, keyed (rule_id, outcome);
//   2. single-flight (G2-3) — the delegated create now AWAITS the wrapper's
//      in-flight check (`latch.checkPromise`) instead of issuing its own, so
//      the normal delegated path pushes each observation exactly once at the
//      source. See tests/anthropicStreamSingleFlight.test.ts.
// The dedup set is retained deliberately: the degraded paths (delegation fired
// outside the construction window, a delegated body without `stream:true`, a
// kickoff that never published) still run two layer-local checks — degradation
// is "two checks", never "zero checks" — and those two must still yield one
// observation. Per-RULE multiplicity is preserved: the key is (rule_id, outcome).
describe("Streamed pre-flight observation dedup", () => {
  const snapshot = (directives: any[]) => ({
    schema_version: 1,
    type: "snapshot",
    version: 1,
    tenant_id: "t",
    project_id: "p",
    ttl_seconds: 600,
    loop_blocks: [],
    directives,
  });

  // Cross-provider (openai target on an anthropic call) → the evaluator refuses
  // the swap and pushes `reroute_rejected` (localEvaluator.ts) on EVERY
  // pre-flight. This is the exact production shape from
  // support_agent_minimax_anthropic_node.
  const crossProviderRule = (id: string, model: string) => ({
    id,
    kind: "REROUTE",
    mode: "enforce",
    priority: 10,
    selector: { match: null, group_by: [] },
    reroute: { from: {}, to: { provider: "openai", model } },
  });

  // Same-provider ENFORCE reroute + dial dry_run → the `would_reroute`
  // observation pushed by the State A dry-run branch in `_runAsyncCheck`.
  const sameProviderRule = (id: string, mode: string) => ({
    id,
    kind: "REROUTE",
    mode,
    priority: 10,
    selector: { match: null, group_by: [] },
    reroute: { from: {}, to: { provider: "anthropic", model: "claude-haiku-4-5" } },
  });

  // Install the taps via init() (serverless — no SSE), then swap in a daemon
  // client so the local fast path (State A) evaluates the seeded pack. The pack
  // must be applied AFTER setClient (setClient resets it).
  function initDaemonWithPack(
    fake: any,
    directives: any[],
    firewall: "enforce" | "dry_run" = "dry_run",
  ) {
    initClient(firewall, fake);
    const daemon = new TokenPolice({
      apiKey: "tp_sk_test123",
      baseUrl: "http://127.0.0.1:59999",
      timeout: 0.1,
      firewall,
      deployment: "daemon",
    });
    setClient(daemon);
    applySnapshot(snapshot(directives));
    const logSpy = vi.spyOn(daemon, "log").mockImplementation(() => {});
    vi.spyOn(daemon, "check").mockResolvedValue({
      status: "allowed",
      fail_open: false,
    } as any);
    return { client: daemon, logSpy };
  }

  // Observations ride the LAST /log argument (the extras bag).
  function loggedObservations(logSpy: any): any[] {
    const rows = anthropicRows(logSpy);
    return rows.flatMap((c: any[]) => {
      const extras = c[c.length - 1];
      return Array.isArray(extras?.observations) ? extras.observations : [];
    });
  }

  const outcomes = (obs: any[], outcome: string) =>
    obs.filter((o: any) => o && o.outcome === outcome);

  beforeEach(() => {
    try { resetPack(); } catch { /* ignore */ }
    try { drainObservations(); } catch { /* ignore */ }
  });

  test("One .stream() call → exactly ONE reroute_rejected observation", async () => {
    const fake = makeFakeAnthropic({ streamOpts: { delegateToCreate: true } });
    const { logSpy } = initDaemonWithPack(fake, [crossProviderRule("rr", "gpt-4o-mini")]);
    await tpSession({}, async () => {
      const messages: any = new fake.Anthropic().messages;
      const stream: any = messages.stream(BODY); // delegates to create within factory
      await stream.finalMessage();
      await flush();
    });
    await flush();
    const obs = loggedObservations(logSpy);
    expect(outcomes(obs, "reroute_rejected")).toHaveLength(1);
  });

  test("Non-streamed create() under the same pack still yields exactly 1", async () => {
    const fake = makeFakeAnthropic();
    initDaemonWithPack(fake, [crossProviderRule("rr", "gpt-4o-mini")]);
    await tpSession({}, async () => {
      const messages: any = new fake.Anthropic().messages;
      await messages.create({ ...BODY });
      await flush();
    });
    await flush();
    // No usage on the fake non-streamed response → no /log row; the queued
    // observations are the direct measure.
    const obs = drainObservations();
    expect(outcomes(obs, "reroute_rejected")).toHaveLength(1);
  });

  test("Two sequential .stream() calls → 1 each (the dedup set is per-call)", async () => {
    const fake = makeFakeAnthropic({ streamOpts: { delegateToCreate: true } });
    const { logSpy } = initDaemonWithPack(fake, [crossProviderRule("rr", "gpt-4o-mini")]);
    await tpSession({}, async () => {
      const messages: any = new fake.Anthropic().messages;
      const s1: any = messages.stream(BODY);
      await s1.finalMessage();
      await flush();
      const s2: any = messages.stream(BODY);
      await s2.finalMessage();
      await flush();
    });
    await flush();
    const rows = anthropicRows(logSpy);
    expect(rows.length).toBe(2);
    for (const c of rows) {
      const extras = c[c.length - 1];
      const obs = Array.isArray(extras?.observations) ? extras.observations : [];
      expect(outcomes(obs, "reroute_rejected")).toHaveLength(1);
    }
  });

  test("Two CONCURRENT .stream() calls → each /log row carries exactly its own observation", async () => {
    // Cross-call attribution: before per-call obs keys, whichever call's
    // /log fired first drained the WHOLE queue — one row got 2 observations,
    // the other 0. With keyed drains each call's row must carry exactly the
    // one observation its own pre-flight pushed (and the latch dedup still
    // holds it to one per streamed call).
    const fake = makeFakeAnthropic({ streamOpts: { delegateToCreate: true } });
    const { logSpy } = initDaemonWithPack(fake, [crossProviderRule("rr", "gpt-4o-mini")]);
    await tpSession({}, async () => {
      const messages: any = new fake.Anthropic().messages;
      // Start both BEFORE awaiting either — the pre-flights and log drains
      // interleave across the two in-flight calls.
      const s1: any = messages.stream(BODY);
      const s2: any = messages.stream(BODY);
      await Promise.all([s1.finalMessage(), s2.finalMessage()]);
      await flush();
    });
    await flush();
    const rows = anthropicRows(logSpy);
    expect(rows.length).toBe(2);
    for (const c of rows) {
      const extras = c[c.length - 1];
      const obs = Array.isArray(extras?.observations) ? extras.observations : [];
      expect(outcomes(obs, "reroute_rejected")).toHaveLength(1);
    }
    // Nothing stranded in the queue after both calls logged.
    expect(drainObservations()).toHaveLength(0);
  });

  test("Dry-run dial + ENFORCE same-provider rule → exactly ONE would_reroute", async () => {
    const fake = makeFakeAnthropic({ streamOpts: { delegateToCreate: true } });
    const { logSpy } = initDaemonWithPack(
      fake,
      [sameProviderRule("rr", "enforce")],
      "dry_run",
    );
    await tpSession({}, async () => {
      const messages: any = new fake.Anthropic().messages;
      const stream: any = messages.stream(BODY);
      await stream.finalMessage();
      await flush();
    });
    await flush();
    const obs = loggedObservations(logSpy);
    expect(outcomes(obs, "would_reroute")).toHaveLength(1);
  });

  test("DRY_RUN-mode rule under an enforce dial → exactly ONE would_reroute", async () => {
    const fake = makeFakeAnthropic({ streamOpts: { delegateToCreate: true } });
    const { logSpy } = initDaemonWithPack(
      fake,
      [sameProviderRule("rr", "dry_run")],
      "enforce",
    );
    await tpSession({}, async () => {
      const messages: any = new fake.Anthropic().messages;
      const stream: any = messages.stream(BODY);
      await stream.finalMessage();
      await flush();
    });
    await flush();
    const obs = loggedObservations(logSpy);
    expect(outcomes(obs, "would_reroute")).toHaveLength(1);
  });

  test("Two DIFFERENT matching rules on one .stream() → 2 (per-rule multiplicity kept)", async () => {
    const fake = makeFakeAnthropic({ streamOpts: { delegateToCreate: true } });
    const { logSpy } = initDaemonWithPack(fake, [
      crossProviderRule("rr1", "gpt-4o-mini"),
      crossProviderRule("rr2", "gpt-4.1-mini"),
    ]);
    await tpSession({}, async () => {
      const messages: any = new fake.Anthropic().messages;
      const stream: any = messages.stream(BODY);
      await stream.finalMessage();
      await flush();
    });
    await flush();
    const obs = loggedObservations(logSpy);
    const rejected = outcomes(obs, "reroute_rejected");
    expect(rejected).toHaveLength(2);
    expect(new Set(rejected.map((o: any) => o.rule_id))).toEqual(new Set(["rr1", "rr2"]));
  });
});

// ═══════════════════════════════════════════════════════════════════
// F-23-2 — drain-end/return() logFinal now reads `receivedMessages.at(-1)`
// instead of `null`, so a for-await consumer that never calls
// `finalMessage()` still gets a RICH `response_composition` (previously
// EMPTY, and the empty row's latch blocked a later `finalMessage()` from
// ever re-logging with the real composition — total loss, not just a
// degraded row). See `_wrapAnthropicMessageStream` (enforcer.ts ~7992),
// `logFinal`/`_finalFromReceived` (~8119/~8340) and the `Symbol.asyncIterator`
// `next()`/`return` handlers (~8556/~8569).
// ═══════════════════════════════════════════════════════════════════
describe("F-23-2: drain-path rich response composition", () => {
  // Both tests below run inside `tpSession()`: composition round-trips through
  // `session._pendingCompositions[traceId:order]`, keyed by whatever
  // `getCurrentSession()` returns. Outside an explicit session scope,
  // `getCurrentSession()` mints a FRESH `TPSession()` (fresh traceId) on every
  // call (context.ts `_sessionStorage.getStore() ?? new TPSession()`) — the
  // wrapper-setup capture (`order = session.nextSpanOrder()`) and the drain-time
  // `_captureCompositionAt` call (fired from the consumer's own iteration
  // context, not the wrapper's) would then key onto two DIFFERENT sessions and
  // never rejoin. `tpSession()` pins one stable session for the whole callback
  // (setup AND later drain) via AsyncLocalStorage — the same requirement the
  // file's existing delegation/observation-dedup tests already wrap for.
  test("for-await drain to completion → rich composition, correct usage, exactly ONE row", async () => {
    const fake = makeFakeAnthropic();
    const { logSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      const seen: any[] = [];
      for await (const ev of stream) seen.push(ev);
      await flush();
      const rows = assertStreamedLogPresent(logSpy, "anthropic");
      expect(rows.length).toBe(1);
      const r = rows[0];
      expect(r[4]).toBe("claude-3"); // model
      expect(r[6]).toBe(10); // input_tokens
      expect(r[7]).toBe(5); // output_tokens
      expect(r[8]).toBe(3); // cache_read → cachedTokens
      // The regression: pre-fix this was `[]` (logFinal(null) never captured
      // composition). Post-fix it must match the final message's real content —
      // same shape `_captureCompositionAt`/`buildResponseComposition` produce
      // for a non-stream Anthropic response
      // (`textEntry("assistant", "Hello world")`).
      expect(r[12]).toEqual([textEntry("assistant", "Hello world")]);
    });
  });

  test("for-await drain then finalMessage() afterwards → still ONE row, rich composition (no dup, no empty-comp row)", async () => {
    const fake = makeFakeAnthropic();
    const { logSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      const seen: any[] = [];
      for await (const ev of stream) seen.push(ev); // drain-end logs the rich row now
      await flush();
      const fm = await stream.finalMessage(); // latched no-op — must NOT add/replace a row
      await flush();
      expect(fm).toBeDefined();
      const rows = anthropicRows(logSpy);
      expect(rows.length).toBe(1); // never 2
      expect(rows[0][12]).toEqual([textEntry("assistant", "Hello world")]); // rich, not empty
    });
  });

  test("early break (return() path) mid-message → no throw; receivedMessages genuinely empty; no dup row", async () => {
    // gateFinalMessage + never releasing it models a customer who walks away
    // before the vendor ever finishes receiving the message — the exact
    // pre-fix condition `logFinal(null)` degraded on.
    const fake = makeFakeAnthropic({ streamOpts: { gateFinalMessage: true } });
    const { logSpy } = initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const seen: any[] = [];
    await expect(
      (async () => {
        for await (const ev of stream) {
          seen.push(ev);
          if (seen.length === 1) break; // leave mid-message — triggers the iterator's return()
        }
      })(),
    ).resolves.toBeUndefined(); // no throw (GOLDEN RULE)
    await flush();
    expect(stream.receivedMessages).toEqual([]); // message never completed — pre-fix condition
    const rows = anthropicRows(logSpy);
    expect(rows.length).toBeLessThanOrEqual(1); // never 2 (latch still holds)
    if (rows.length === 1) {
      // Usage guard: message_start (event 0, already consumed before break)
      // accumulated input_tokens onto `acc` independently of receivedMessages.
      expect(rows[0][6]).toBeGreaterThan(0); // input_tokens
      expect(rows[0][12]).toEqual([]); // acceptable: empty composition, not a throw/dup
    }
  });

  test("hostile receivedMessages getter (throws on access) → drain does not throw into consumer; usage-only row, empty composition", async () => {
    const fake = makeFakeAnthropic({ streamOpts: { hostileReceivedMessages: true } });
    const { logSpy } = initAllowed(fake);
    const stream: any = new fake.Anthropic().messages.stream(BODY);
    const seen: any[] = [];
    await expect(
      (async () => {
        for await (const ev of stream) seen.push(ev);
      })(),
    ).resolves.toBeUndefined(); // GOLDEN RULE: never throws into the customer's iteration
    expect(seen.length).toBe(3); // events still delivered
    await flush();
    const rows = assertStreamedLogPresent(logSpy, "anthropic"); // usage-only row still dispatched
    expect(rows[0][6]).toBe(10); // input_tokens (from acc, not receivedMessages)
    expect(rows[0][7]).toBe(5); // output_tokens
    expect(rows[0][12]).toEqual([]); // composition degrades to empty, not a throw
  });

  test("pre-taken latch (delegated create-tap logs first) makes the .stream() drain-path logFinal a no-op", async () => {
    // Exercises the FIX's own drain-end call (`for await` to completion on the
    // `.stream()` wrapper itself, not `finalMessage()`) against a latch already
    // taken by the internally-delegated create tap — mirrors 7(b-reverse) but
    // for the drain path this change touches.
    const fake = makeFakeAnthropic({ streamOpts: { delegateToCreate: true } });
    const { logSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const messages: any = new fake.Anthropic().messages;
      const stream: any = messages.stream(BODY); // delegates to create within factory window
      const raw: any = await stream._delegateRaw;
      for await (const _ of raw) void _; // create-tap drains + logs FIRST → takes the latch
      const seen: any[] = [];
      for await (const ev of stream) seen.push(ev); // .stream() drain-end sees latch taken → no-op
      await flush();
      expect(anthropicRows(logSpy).length).toBe(1); // still exactly 1, not 2
    });
  });
});
