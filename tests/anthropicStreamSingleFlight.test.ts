/**
 * G2-3 — an Anthropic `messages.stream()` call must run EXACTLY ONE pre-flight.
 *
 * `client.messages.stream()` is TokenPolice-patched, and the vendor
 * MessageStream internally delegates to the ALSO-patched
 * `messages.create({stream:true})`. Both layers used to issue their own
 * `POST /v1/guard/check` for the SAME logical request, so every streamed call
 * cost:
 *
 *  • 2× `/check` — doubled hot-path latency, and 2× REROUTE_DIRECTIVE_ISSUED
 *    audit rows on the customer's own audit surface (one logical call read as
 *    two directives);
 *  • 2× the synthetic "blocked" `/log` row on an enforce denial
 *    (`_emitLocalBlockLog` fires inside `_runAsyncCheck`, once per check).
 *
 * The SUCCESS `/log` row was already deduped by the per-call `StreamLatch`, so
 * metering never doubled — this is a pre-flight/audit defect, not a cost one.
 *
 * Fix under test (single-flight): `StreamLatch` gained `checkPromise` +
 * `checkedBody`. The `.stream()` wrapper publishes its in-flight check on the
 * per-call latch (inside the kickoff try, only after a successful kickoff); the
 * delegated create — gated on `provider === "anthropic"`, a real body,
 * `args[0].stream === true` and an INHERITED latch that carries a
 * `checkPromise` — awaits that shared promise instead of issuing its own, then
 * syncs `checkedBody.model → reqBody.model`. That sync is load-bearing: the
 * shared check applies a REROUTE by mutating the `.stream()` wrapper's body IN
 * PLACE, and the vendor already shallow-copied its params before the check
 * resolved, so without the sync an applied reroute could never reach the wire.
 *
 * Every degraded path (no delegation, delegation outside the construction
 * window, a delegated body without `stream:true`, a kickoff that never
 * published) falls back to the layer-local check: degradation is "two checks",
 * NEVER "zero checks" — enforcement can never silently disappear.
 *
 * GOLDEN RULE pinned throughout: the only error TokenPolice ever puts into
 * customer code is `TokenPoliceBlockedError`, and only on a verified enforce
 * denial. The shared-promise await re-throws that type and swallows everything
 * else (fail-open).
 *
 * HARNESS NOTES (mirrors tests/anthropicStreamRejection.test.ts):
 * • Fake anthropic module injected via `init({ instrumentModules })` — no real
 *   @anthropic-ai/sdk. No `APIPromise` export ⇒ the create-streaming BYPASS
 *   path engages.
 * • `delegate: true` models the real vendor delegation: `.stream()` fires
 *   `create({stream:true})` INSIDE its factory window (so the create layer
 *   inherits the `.stream()` latch through async context), and the
 *   MessageStream's surfaces resolve THROUGH that promise.
 * • The pre-flight count is read straight off the `client.check` spy — that
 *   spy IS the `POST /v1/guard/check` seam.
 * • `created` / `createdModels` record what the REAL (unpatched) create
 *   received: the request as it reaches the wire. `copiedModels` records the
 *   model on the vendor's params copy at the instant it was taken — pinning
 *   that the copy predates the check, which is why the reroute sync exists.
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
  /** Vendor delegation INSIDE the `.stream()` factory window (the normal case). */
  delegate?: boolean;
  /**
   * Delegation deferred until the test fires `stream._fire()` — i.e. OUTSIDE the
   * `.stream()` construction window, so the create layer inherits NO latch.
   * A degraded shape: it must still run its own check (never zero).
   */
  lazyDelegate?: boolean;
  /**
   * Delegation whose params carry no `stream: true`. The single-flight gate is
   * deliberately narrow and must not engage → layer-local check (never zero).
   */
  delegateWithoutStreamFlag?: boolean;
  /** Bound-client base URL — drives serving-provider remap (e.g. minimax). */
  baseURL?: string;
}

/** The raw anthropic Stream `create({stream:true})` resolves to. */
class FakeRawStream {
  events: any[];
  controller = { abort() {} };
  constructor(events: any[]) {
    this.events = events;
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
        if (i >= self.events.length) return { done: true, value: undefined };
        return { done: false, value: self.events[i++] };
      },
    };
  }
}

/**
 * A fake Anthropic MessageStream: async-iterable + EventEmitter-ish + the
 * context-manager surface the wrapper gates (finalMessage / finalText / done /
 * emitted / controller / abort / tee).
 *
 * When the vendor delegated, EVERY consumer surface resolves THROUGH the
 * delegated create promise — the real coupling that makes a block or a
 * request-time rejection reach whichever surface the customer used.
 */
class FakeMessageStream {
  _events: any[] = defaultEvents();
  _final: any = defaultFinal();
  /** The patched create's promise (delegation), null for an independent stream. */
  _delegate: Promise<any> | null = null;
  /** Deferred delegation trigger (lazyDelegate arm). */
  _fire: (() => Promise<any>) | null = null;
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

  constructor() {
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
          const raw: any = await self._delegate;
          if (!self._rawIter) self._rawIter = raw[Symbol.asyncIterator]();
          return self._rawIter.next();
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
  /** Bodies the REAL (unpatched) create received — what reaches the wire. */
  const created: any[] = [];
  /** `model` on each of those bodies, read at dispatch time. */
  const createdModels: Array<string | undefined> = [];
  /** `model` on the vendor's params copy at the instant the copy was taken. */
  const copiedModels: Array<string | undefined> = [];

  class Messages {
    _client?: Record<string, unknown>;
    constructor() {
      if (opts.baseURL) this._client = { baseURL: opts.baseURL };
    }
    stream(body: any) {
      const s = new FakeMessageStream();
      streams.push(s);
      const delegating =
        !!opts.delegate || !!opts.lazyDelegate || !!opts.delegateWithoutStreamFlag;
      if (delegating) {
        // The vendor shallow-copies params SYNCHRONOUSLY here — before any
        // pre-flight can resolve. That copy is exactly why an in-place body
        // mutation by the shared check cannot reach the wire on its own.
        const params: any = opts.delegateWithoutStreamFlag
          ? { ...body }
          : { ...body, stream: true };
        copiedModels.push(params.model);
        const fire = (): Promise<any> => {
          // `this.create` is the PATCHED create; firing it inside the factory
          // window is what lets the create layer inherit the `.stream()` latch
          // through async context, exactly as a real delegating SDK does.
          const p: Promise<any> = (this as any).create(params);
          // The vendor holds this promise internally and only re-delivers it
          // via the MessageStream surfaces; swallow so a consumer that never
          // touches those surfaces can't trip an unhandled rejection.
          void p.catch(() => {});
          return p;
        };
        if (opts.lazyDelegate) {
          s._fire = fire; // fired by the test, OUTSIDE the construction window
        } else {
          const p = fire();
          if (!opts.delegateWithoutStreamFlag) s._delegate = p;
        }
      }
      return s;
    }
    create(body: any) {
      created.push(body);
      createdModels.push(body?.model);
      if (body?.stream) return Promise.resolve(new FakeRawStream(defaultEvents()));
      return Promise.resolve({});
    }
  }
  class AsyncMessages {
    stream(_body: any) {
      const s = new FakeMessageStream();
      streams.push(s);
      return s;
    }
  }
  function Anthropic(this: any) {
    this.messages = new Messages();
  }
  (Anthropic as any).Messages = Messages;
  (Anthropic as any).AsyncMessages = AsyncMessages;
  return {
    Anthropic,
    Messages,
    AsyncMessages,
    streams,
    created,
    createdModels,
    copiedModels,
  };
}

const BODY = { model: "claude-3", messages: [{ role: "user", content: "hi" }] };
const freshBody = () => ({ model: "claude-3", messages: [{ role: "user", content: "hi" }] });

/** A State-B ENFORCE reroute directive: same provider, cheaper model. */
function rerouteAllowed(): any {
  return {
    status: "allowed",
    fail_open: false,
    reroute: {
      mode: "enforce",
      provider: "anthropic",
      model: "claude-haiku-4-5",
      rule_id: "rr-1",
      rule_name: "cheap-haiku",
      original: { provider: "anthropic", model: "claude-3" },
    },
  };
}

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

/** Install the wrapper (dry_run) with the pre-flight stubbed ALLOWED. */
function initAllowed(fake: any) {
  const { client, logSpy } = initClient("dry_run", fake);
  const checkSpy = vi
    .spyOn(client, "check")
    .mockResolvedValue({ status: "allowed", fail_open: false } as any);
  return { client, logSpy, checkSpy };
}

/** Rows the synthetic block-decision emitter dispatched (`_emitLocalBlockLog`). */
function blockedRows(logSpy: any): any[][] {
  return logSpy.mock.calls.filter((c: any[]) => c[4] === "blocked" && c[5] === "");
}
/** Metered anthropic rows (arg[5] === provider). */
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
// 1 — single-flight happy path: ONE /check per streamed call.
// ═══════════════════════════════════════════════════════════════════
describe("G2-3/1: one delegated .stream() → exactly ONE pre-flight", () => {
  test("for-await drain: 1 /check (was 2) and 1 metered /log row", async () => {
    const fake = makeFakeAnthropic({ delegate: true });
    const { logSpy, checkSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      const seen: any[] = [];
      for await (const ev of stream) seen.push(ev);
      await flush();
      expect(seen).toHaveLength(3); // customer's tokens all delivered
      // THE regression: pre-fix the `.stream()` wrapper AND the delegated
      // create each issued their own /check for this one request.
      expect(checkSpy).toHaveBeenCalledTimes(1);
      // The provider request still went out exactly once, unchanged.
      expect(fake.createdModels).toEqual(["claude-3"]);
      // Latch behavior unchanged: exactly one metered row, really metered.
      expect(logSpy.mock.calls).toHaveLength(1);
      expect(anthropicRows(logSpy)).toHaveLength(1);
      expect(logSpy.mock.calls[0][6]).toBe(10); // input_tokens
      expect(logSpy.mock.calls[0][7]).toBe(5); // output_tokens
    });
  });

  test("finalMessage() consumer (never iterates): still exactly 1 /check + 1 row", async () => {
    const fake = makeFakeAnthropic({ delegate: true });
    const { logSpy, checkSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      const msg = await stream.finalMessage();
      await flush();
      expect(msg.model).toBe("claude-3");
      expect(checkSpy).toHaveBeenCalledTimes(1);
      expect(anthropicRows(logSpy)).toHaveLength(1);
    });
  });

  test("the shared pre-flight is the SERVING-aware one (provider forwarded once)", async () => {
    const fake = makeFakeAnthropic({ delegate: true });
    const { checkSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      await stream.finalMessage();
      await flush();
      expect(checkSpy).toHaveBeenCalledTimes(1);
      expect(checkSpy.mock.calls[0][6]).toBe("claude-3"); // target model
      expect(checkSpy.mock.calls[0][7]).toBe("anthropic"); // serving provider
    });
  });

  test("a NON-delegating .stream() still runs its own single pre-flight", async () => {
    const fake = makeFakeAnthropic(); // vendor never delegates
    const { logSpy, checkSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      const seen: any[] = [];
      for await (const ev of stream) seen.push(ev);
      await flush();
      expect(seen).toHaveLength(3);
      expect(checkSpy).toHaveBeenCalledTimes(1);
      expect(anthropicRows(logSpy)).toHaveLength(1);
    });
  });
});

// ═══════════════════════════════════════════════════════════════════
// 2 — the single-flight channel is PER CALL (never shared across calls).
// ═══════════════════════════════════════════════════════════════════
describe("G2-3/2: per-call scoping — one check per call, never one for many", () => {
  test("two SEQUENTIAL delegated .stream() calls → 2 checks (1 each)", async () => {
    const fake = makeFakeAnthropic({ delegate: true });
    const { logSpy, checkSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const s1: any = new fake.Anthropic().messages.stream(BODY);
      await s1.finalMessage();
      await flush();
      const s2: any = new fake.Anthropic().messages.stream(BODY);
      await s2.finalMessage();
      await flush();
      expect(checkSpy).toHaveBeenCalledTimes(2);
      expect(anthropicRows(logSpy)).toHaveLength(2);
    });
  });

  test("two CONCURRENT delegated .stream() calls → 2 checks (no cross-call reuse)", async () => {
    const fake = makeFakeAnthropic({ delegate: true });
    const { logSpy, checkSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const messages: any = new fake.Anthropic().messages;
      const s1: any = messages.stream(BODY);
      const s2: any = messages.stream(BODY);
      await Promise.all([s1.finalMessage(), s2.finalMessage()]);
      await flush();
      expect(checkSpy).toHaveBeenCalledTimes(2);
      expect(anthropicRows(logSpy)).toHaveLength(2);
    });
  });

  test("a later INDEPENDENT create({stream:true}) after a .stream() gets its OWN check", async () => {
    // The channel rides the per-call ALS latch, so it must not leak onto a
    // separate call later in the same session (that would be zero checks for
    // a request nobody pre-flighted).
    const fake = makeFakeAnthropic({ delegate: true });
    const { checkSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const messages: any = new fake.Anthropic().messages;
      const stream: any = messages.stream(BODY);
      await stream.finalMessage();
      await flush();
      expect(checkSpy).toHaveBeenCalledTimes(1);
      const raw: any = await messages.create({ ...BODY, stream: true });
      for await (const _ev of raw) void _ev;
      await flush();
      expect(checkSpy).toHaveBeenCalledTimes(2); // 1 for the stream, 1 for the create
    });
  });
});

// ═══════════════════════════════════════════════════════════════════
// 3 — an applied ENFORCE reroute still reaches the WIRE.
// ═══════════════════════════════════════════════════════════════════
describe("G2-3/3: applied REROUTE reaches the dispatched request", () => {
  test("one check, and the body the delegated create dispatches carries the target model", async () => {
    const fake = makeFakeAnthropic({ delegate: true });
    const { client, logSpy } = initClient("enforce", fake);
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue(rerouteAllowed() as any);
    const body = freshBody();
    await tpSession({}, async () => {
      const stream: any = new fake.Anthropic().messages.stream(body);
      const seen: any[] = [];
      for await (const ev of stream) seen.push(ev);
      await flush();
      expect(seen).toHaveLength(3); // reroute is not a block — the call proceeds

      expect(checkSpy).toHaveBeenCalledTimes(1);

      // The vendor's params copy was taken BEFORE the check resolved — it still
      // carried the ORIGINAL model. So mutating the `.stream()` wrapper's body
      // in place (all `_applyReroute` does) could never reach the wire on its
      // own: the reroute sync in the delegated create is what lands it.
      expect(fake.copiedModels).toEqual(["claude-3"]);
      // What the provider actually received.
      expect(fake.createdModels).toEqual(["claude-haiku-4-5"]);
      // ...and it is genuinely a DIFFERENT object from the wrapper's body.
      expect(fake.created[0]).not.toBe(body);
      // The wrapper's own body was mutated too (unchanged `_applyReroute`).
      expect(body.model).toBe("claude-haiku-4-5");

      expect(anthropicRows(logSpy)).toHaveLength(1);
    });
  });

  test("no-op when the directive names the model already requested (no phantom swap)", async () => {
    const fake = makeFakeAnthropic({ delegate: true });
    const { client } = initClient("enforce", fake);
    const noop = rerouteAllowed();
    noop.reroute.model = "claude-3"; // same model → `_applyReroute` no-ops
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue(noop as any);
    await tpSession({}, async () => {
      const stream: any = new fake.Anthropic().messages.stream(freshBody());
      await stream.finalMessage();
      await flush();
      expect(checkSpy).toHaveBeenCalledTimes(1);
      expect(fake.createdModels).toEqual(["claude-3"]); // wire untouched
    });
  });
});

// ═══════════════════════════════════════════════════════════════════
// 4 — enforce BLOCK: typed error, zero dispatch, ONE blocked row.
// ═══════════════════════════════════════════════════════════════════
describe("G2-3/4: enforce block on a delegated .stream()", () => {
  test("for-await rejects TokenPoliceBlockedError; provider never dispatched; ONE blocked row", async () => {
    const fake = makeFakeAnthropic({ delegate: true });
    const { client, logSpy } = initClient("enforce", fake);
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({
      status: "blocked",
      fail_open: false,
      reason: "x",
    } as any);
    await tpSession({}, async () => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      const err = await (async () => {
        try {
          for await (const _ev of stream) void _ev;
          return null;
        } catch (e) {
          return e;
        }
      })();
      await flush();
      // GOLDEN RULE: the ONLY error TokenPolice puts into customer code.
      expect(err).toBeInstanceOf(TokenPoliceBlockedError);
      expect(checkSpy).toHaveBeenCalledTimes(1);
      // The delegated create re-threw the typed block BEFORE calling the real
      // method, so nothing ever reached the provider.
      expect(fake.created).toHaveLength(0);
      expect(stream.aborted).toBe(true);
      // THE second half of the regression: `_emitLocalBlockLog` fires once per
      // check, so two pre-flights silently doubled the synthetic blocked row.
      expect(blockedRows(logSpy)).toHaveLength(1);
      // A denial is never re-reported as a metered provider call.
      expect(anthropicRows(logSpy)).toHaveLength(0);
    });
  });

  test("finalMessage() consumer sees the same typed block, still ONE blocked row", async () => {
    const fake = makeFakeAnthropic({ delegate: true });
    const { client, logSpy } = initClient("enforce", fake);
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({
      status: "blocked",
      fail_open: false,
      reason: "x",
    } as any);
    await tpSession({}, async () => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      await expect(stream.finalMessage()).rejects.toBeInstanceOf(TokenPoliceBlockedError);
      await flush();
      expect(checkSpy).toHaveBeenCalledTimes(1);
      expect(fake.created).toHaveLength(0);
      expect(blockedRows(logSpy)).toHaveLength(1);
    });
  });

  test("dry_run dial + would-block: no throw, call proceeds, still ONE check", async () => {
    const fake = makeFakeAnthropic({ delegate: true });
    const { client, logSpy } = initClient("dry_run", fake);
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({
      status: "blocked",
      fail_open: false,
      reason: "x",
    } as any);
    await tpSession({}, async () => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      const seen: any[] = [];
      await expect(
        (async () => {
          for await (const ev of stream) seen.push(ev);
        })(),
      ).resolves.toBeUndefined();
      await flush();
      expect(seen).toHaveLength(3); // dry_run never blocks
      expect(checkSpy).toHaveBeenCalledTimes(1);
      expect(fake.createdModels).toEqual(["claude-3"]);
      expect(blockedRows(logSpy)).toHaveLength(0);
    });
  });
});

// ═══════════════════════════════════════════════════════════════════
// 5 — degradation is "two checks", NEVER "zero checks".
// ═══════════════════════════════════════════════════════════════════
describe("G2-3/5: never-zero degradation", () => {
  test("independent create({stream:true}) with no .stream() in scope → its own check", async () => {
    const fake = makeFakeAnthropic(); // no delegation at all
    const { logSpy, checkSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const messages: any = new fake.Anthropic().messages;
      const raw: any = await messages.create({ ...BODY, stream: true });
      for await (const _ev of raw) void _ev;
      await flush();
      expect(checkSpy).toHaveBeenCalledTimes(1); // 1, never 0
      expect(fake.createdModels).toEqual(["claude-3"]);
      expect(anthropicRows(logSpy)).toHaveLength(1);
    });
  });

  test("delegation fired OUTSIDE the construction window → the create layer checks itself", async () => {
    // No inherited latch (the ALS window closed when `.stream()` returned), so
    // the shared channel is invisible and the layer-local check must run.
    const fake = makeFakeAnthropic({ lazyDelegate: true });
    const { checkSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      await flush();
      expect(checkSpy).toHaveBeenCalledTimes(1); // the wrapper's own
      const raw: any = await stream._fire!(); // vendor delegates late
      for await (const _ev of raw) void _ev;
      await flush();
      // Two checks for one request — the documented worst case. Never zero.
      expect(checkSpy).toHaveBeenCalledTimes(2);
      expect(fake.createdModels).toEqual(["claude-3"]);
    });
  });

  test("delegated params without `stream: true` → the narrow gate declines to share", async () => {
    const fake = makeFakeAnthropic({ delegateWithoutStreamFlag: true });
    const { checkSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      await stream.finalMessage();
      await flush();
      // Wrapper check + the delegated layer's own check. Never zero.
      expect(checkSpy).toHaveBeenCalledTimes(2);
      expect(fake.created).toHaveLength(1);
    });
  });

  test("enforce block still lands on a degraded (late-delegation) path — dispatch is refused", async () => {
    // The point of "never zero": a degraded shape must still be blocked.
    const fake = makeFakeAnthropic({ lazyDelegate: true });
    const { client } = initClient("enforce", fake);
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({
      status: "blocked",
      fail_open: false,
      reason: "x",
    } as any);
    await tpSession({}, async () => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      await flush();
      await expect(stream._fire!()).rejects.toBeInstanceOf(TokenPoliceBlockedError);
      await flush();
      expect(checkSpy).toHaveBeenCalledTimes(2); // two layer-local checks
      expect(fake.created).toHaveLength(0); // nothing reached the provider
    });
  });
});

// ═══════════════════════════════════════════════════════════════════
// 6 — fail-open: a broken pre-flight never breaks the customer's call.
// ═══════════════════════════════════════════════════════════════════
describe("G2-3/6: fail-open on a failing pre-flight", () => {
  // NOTE ON COVERAGE: the delegated layer's `catch` re-throws only
  // `TokenPoliceBlockedError` and swallows everything else. That non-block arm
  // is defense-in-depth and cannot be exercised directly from the public
  // surface: `_runAsyncCheck` is `failSafeAsync`-wrapped (src/safe.ts), so the
  // promise published on the latch can ONLY ever reject with
  // `TokenPoliceBlockedError` — every other error is swallowed one layer
  // lower. Injecting a raw non-block rejection would mean patching src/, so
  // what is asserted here is the end-to-end fail-open OUTCOME instead: a
  // pre-flight that dies mid-flight must leave the customer's stream intact.
  test("check() rejects (non-block) → no throw, tokens delivered, still ONE check", async () => {
    const fake = makeFakeAnthropic({ delegate: true });
    const { client, logSpy } = initClient("enforce", fake);
    const checkSpy = vi.spyOn(client, "check").mockRejectedValue(new Error("check boom"));
    await tpSession({}, async () => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      const seen: any[] = [];
      await expect(
        (async () => {
          for await (const ev of stream) seen.push(ev);
        })(),
      ).resolves.toBeUndefined();
      await flush();
      expect(seen).toHaveLength(3); // fail-open: the call proceeded
      expect(checkSpy).toHaveBeenCalledTimes(1); // shared, not re-issued
      expect(fake.createdModels).toEqual(["claude-3"]); // dispatched to the wire
      expect(blockedRows(logSpy)).toHaveLength(0); // a failure is never a block
    });
  });

  test("firewall off → no pre-flight at all on either layer", async () => {
    const fake = makeFakeAnthropic({ delegate: true });
    // dry_run installs the taps; flip the runtime dial to off afterwards.
    const { client } = initClient("dry_run", fake);
    const checkSpy = vi
      .spyOn(client, "check")
      .mockResolvedValue({ status: "allowed", fail_open: false } as any);
    (client as any).firewall = "off";
    await tpSession({}, async () => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      const seen: any[] = [];
      for await (const ev of stream) seen.push(ev);
      await flush();
      expect(seen).toHaveLength(3);
      expect(checkSpy).not.toHaveBeenCalled();
    });
  });
});

// ═══════════════════════════════════════════════════════════════════
// 7 — MiniMax over the Anthropic wire (serving remap) — still one check.
// ═══════════════════════════════════════════════════════════════════
describe("G2-3/7: serving-provider remap (Anthropic SDK → api.minimax.io)", () => {
  test("module stays anthropic, serving is minimax, and the pre-flight fires ONCE", async () => {
    const fake = makeFakeAnthropic({
      delegate: true,
      baseURL: "https://api.minimax.io/anthropic/v1",
    });
    const { logSpy, checkSpy } = initAllowed(fake);
    await tpSession({}, async () => {
      const stream: any = new fake.Anthropic().messages.stream(BODY);
      const seen: any[] = [];
      for await (const ev of stream) seen.push(ev);
      await flush();
      expect(seen).toHaveLength(3);
      // The single-flight gate keys on the MODULE provider (anthropic), so a
      // host remap must not reopen the double pre-flight.
      expect(checkSpy).toHaveBeenCalledTimes(1);
      // ...and the one check that runs is the serving-aware one.
      expect(checkSpy.mock.calls[0][7]).toBe("minimax");
      expect(logSpy.mock.calls).toHaveLength(1);
    });
  });
});
