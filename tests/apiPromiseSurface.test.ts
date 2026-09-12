/**
 * PR #502 follow-ups — Anthropic `APIPromise` surface preservation
 * (`_preserveApiPromiseSurface`, src/enforcer.ts) + Beta.Messages enforcement
 * coverage (the new `_TARGET_METHODS` row for `Anthropic.Beta.Messages.prototype.create`
 * + the `__tpStreamPatched` own-property hardening in `_patchAnthropicStreamMethod`).
 *
 * ── Change 1 — the P0 fix ──────────────────────────────────────────────────
 * TokenPolice replaces `Anthropic.Messages.prototype.create` with a wrapper that
 * (pre-fix) returned a plain `Promise`, destroying the vendor's `APIPromise`
 * surface (`.withResponse()` / `.asResponse()` / `.parse()`). From
 * `@anthropic-ai/sdk` 0.35.0 onward the vendor's OWN `MessageStream` internally
 * does `await messages.create({...,stream:true}, opts).withResponse()`, so
 * `client.messages.stream()` threw `TypeError: ... .withResponse is not a
 * function` INTO CUSTOMER CODE — a GOLDEN RULE violation. `_preserveApiPromiseSurface`
 * re-attaches the three surfaces onto whatever the wrapped method returns,
 * gated on `typeof p?.then === "function" && typeof p.withResponse !== "function"`,
 * each assignment in its own try/catch (withResponse FIRST — load-bearing),
 * with a lazy per-promise MEMOIZED synthetic `Response`.
 *
 * ── Change 2 — Beta.Messages enforcement ────────────────────────────────────
 * `Anthropic.Beta.Messages` is a SIBLING class (extends `APIResource`, NOT
 * `Messages`), so `client.beta.messages.*` previously got ZERO pre-flight
 * `/check`. A new `_TARGET_METHODS` row wraps it, and the `__tpStreamPatched`
 * idempotency guard became an OWN-property check so a hypothetical future
 * `Beta.Messages extends Messages` could not silently skip patching it.
 *
 * ── Harness notes ────────────────────────────────────────────────────────
 * `@anthropic-ai/sdk` is NOT a devDependency here — every fixture below is a
 * hand-built fake reproducing only the vendor CONTRACT this code depends on
 * (mirrors tests/anthropicStreamSingleFlight.test.ts / tests/syncWrapperNoEnforce.test.ts).
 * `_preserveApiPromiseSurface` itself is NOT exported, so it is driven two ways:
 *   - Groups A/B (realistic shapes) go through the REAL async `create`
 *     `_TARGET_METHODS` row via `init({ instrumentModules })` — this is the
 *     actual customer-facing path the P0 bug lived on.
 *   - Group C (hostile shapes) uses the public `protect(..., isAsync:false, ...)`
 *     escape hatch (see tests/syncWrapperNoEnforce.test.ts): the sync branch
 *     returns the wrapped `create`'s result completely unwrapped by a Promise,
 *     so `_preserveApiPromiseSurface` sees EXACTLY whatever the fake `create`
 *     returned — the only way to hand it a frozen/sealed/hostile-getter value
 *     directly (the real async `_TARGET_METHODS` row always coerces its return
 *     into a genuine native Promise, which can never itself be hostile).
 * No real network, no fake timers, no sleeps — the one `setTimeout(r, 0)` flush
 * (mirroring the existing anthropicStreamSingleFlight suite) only yields to the
 * macrotask queue after a fully-drained async iterator, never waits on a
 * condition.
 */
import { describe, it, expect, afterEach, vi } from "vitest";
import { init, TokenPolice } from "../src/client";
import { protect, uninstrument } from "../src/enforcer";
import { setClient } from "../src/state";

const flush = () => new Promise((r) => setTimeout(r, 0));

const BODY = { model: "claude-3", messages: [{ role: "user", content: "hi" }] };

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

// ═══════════════════════════════════════════════════════════════════════
// Shared fixtures
// ═══════════════════════════════════════════════════════════════════════

/** The raw anthropic Stream a streaming `create({stream:true})` resolves to. */
class FakeRawStream {
  events: any[];
  constructor(events: any[]) {
    this.events = events;
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
function defaultStreamEvents(): any[] {
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

const defaultCreate = async (body: any): Promise<any> => {
  if (body?.stream) return new FakeRawStream(defaultStreamEvents());
  return {
    id: "msg_1",
    model: body?.model,
    role: "assistant",
    content: [{ type: "text", text: "hello" }],
    stop_reason: "end_turn",
    usage: { input_tokens: 10, output_tokens: 5 },
  };
};

/**
 * A fake Anthropic root exposing `Messages` and a SIBLING `Beta.Messages`
 * (extends nothing — matches the REAL vendor shape verified in both 0.30.1
 * and 0.90.0). Fresh classes per call so no cross-test `__tpStreamPatched` /
 * `_originals` dedupe contamination. No `APIPromise` static export ⇒ the
 * anthropic create-streaming bypass path engages for `stream:true` calls,
 * exactly as it does against real pre-0.35.0 SDKs.
 */
function makeFakeAnthropic(createImpl: (body: any) => Promise<any> = defaultCreate) {
  class Messages {
    async create(body: any): Promise<any> {
      return createImpl(body);
    }
    async countTokens(_body: any): Promise<any> {
      return { input_tokens: 1 };
    }
  }
  class BetaMessages {
    async create(body: any): Promise<any> {
      return createImpl(body);
    }
    async countTokens(_body: any): Promise<any> {
      return { input_tokens: 1 };
    }
  }
  function Anthropic(this: any) {
    this.messages = new Messages();
    this.beta = { messages: new BetaMessages() };
  }
  (Anthropic as any).Messages = Messages;
  (Anthropic as any).Beta = { Messages: BetaMessages };
  return { Anthropic, Messages, BetaMessages };
}

/** Install the wrapper with the pre-flight stubbed ALLOWED (mirrors anthropicStreamSingleFlight.test.ts). */
function initAllowed(fake: ReturnType<typeof makeFakeAnthropic>) {
  const client = init({
    apiKey: "tp_sk_test",
    deployment: "serverless",
    firewall: "dry_run",
    instrumentModules: { anthropic: fake.Anthropic },
  } as any);
  const logSpy = vi.spyOn(client, "log").mockImplementation(() => {});
  const checkSpy = vi
    .spyOn(client, "check")
    .mockResolvedValue({ status: "allowed", fail_open: false } as any);
  return { client, logSpy, checkSpy };
}

function anthropicRows(logSpy: any): any[][] {
  return logSpy.mock.calls.filter((c: any[]) => c[5] === "anthropic");
}

// ═══════════════════════════════════════════════════════════════════════
// Group A — the surface is restored (real async `create` row)
// ═══════════════════════════════════════════════════════════════════════
describe("Group A: patched anthropic create() re-attaches the APIPromise surface", () => {
  it("1: the returned thenable HAS withResponse, asResponse, and parse (pre-fix: none of the three existed)", () => {
    const fake = makeFakeAnthropic();
    initAllowed(fake);
    const p: any = new fake.Anthropic().messages.create(BODY);
    // Decoration is synchronous — happens the instant `create()` returns, no await needed.
    expect(typeof p.then).toBe("function");
    expect(typeof p.withResponse).toBe("function");
    expect(typeof p.asResponse).toBe("function");
    expect(typeof p.parse).toBe("function");
  });

  it("2: withResponse() resolves an object with EXACTLY the three keys data/response/request_id", async () => {
    const fake = makeFakeAnthropic();
    initAllowed(fake);
    const p: any = new fake.Anthropic().messages.create(BODY);
    const wr = await p.withResponse();
    // Exact key set — a 4th key would be a contract break the vendor doesn't expect.
    expect(Object.keys(wr).sort()).toEqual(["data", "request_id", "response"]);
  });

  it("3: data is the SAME value the underlying promise resolves to", async () => {
    const fake = makeFakeAnthropic();
    initAllowed(fake);
    const p: any = new fake.Anthropic().messages.create(BODY);
    const viaAwait = await p;
    const { data } = await p.withResponse();
    expect(data).toBe(viaAwait);
  });

  it("4: parse() returns the promise itself (not a copy, not a re-wrap)", () => {
    const fake = makeFakeAnthropic();
    initAllowed(fake);
    const p: any = new fake.Anthropic().messages.create(BODY);
    expect(p.parse()).toBe(p);
  });

  it("5: request_id is always null (accepted degradation — telemetry loss beats a throw)", async () => {
    const fake = makeFakeAnthropic();
    initAllowed(fake);
    const p: any = new fake.Anthropic().messages.create(BODY);
    const { request_id } = await p.withResponse();
    expect(request_id).toBeNull();
  });

  it("6: response is a synthetic Response with status 200 and no request-id header", async () => {
    const fake = makeFakeAnthropic();
    initAllowed(fake);
    const p: any = new fake.Anthropic().messages.create(BODY);
    const { response } = await p.withResponse();
    expect(response).toBeInstanceOf(Response);
    expect(response.status).toBe(200);
    expect(response.headers.get("request-id")).toBeNull();
    // asResponse() must hand back the identical instance (Group B pins this further).
    const viaAsResponse = await p.asResponse();
    expect(viaAsResponse).toBe(response);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Group B — memoization: one Response per PROMISE, built lazily, shared by
// withResponse() and asResponse(), never leaking across calls.
// ═══════════════════════════════════════════════════════════════════════
describe("Group B: synthetic Response memoization", () => {
  it("7: withResponse()x2 + asResponse()x2 on ONE promise ⇒ Response constructed EXACTLY ONCE, all four references identical", async () => {
    const fake = makeFakeAnthropic();
    initAllowed(fake);
    const OriginalResponse = (globalThis as any).Response;
    let constructCount = 0;
    class CountingResponse extends OriginalResponse {
      constructor(body?: any, init?: any) {
        super(body, init);
        constructCount++;
      }
    }
    (globalThis as any).Response = CountingResponse;
    try {
      const p: any = new fake.Anthropic().messages.create(BODY);
      const r1 = await p.withResponse();
      const r2 = await p.withResponse();
      const a1 = await p.asResponse();
      const a2 = await p.asResponse();
      expect(constructCount).toBe(1);
      expect(r1.response).toBeInstanceOf(CountingResponse);
      expect(r1.response).toBe(r2.response);
      expect(r1.response).toBe(a1);
      expect(a1).toBe(a2);
    } finally {
      (globalThis as any).Response = OriginalResponse;
    }
  });

  it("8: two DIFFERENT calls get DIFFERENT Response instances (per-promise, never module-level shared)", async () => {
    const fake = makeFakeAnthropic();
    initAllowed(fake);
    const OriginalResponse = (globalThis as any).Response;
    let constructCount = 0;
    class CountingResponse extends OriginalResponse {
      constructor(body?: any, init?: any) {
        super(body, init);
        constructCount++;
      }
    }
    (globalThis as any).Response = CountingResponse;
    try {
      const anthropic = new fake.Anthropic();
      const p1: any = anthropic.messages.create(BODY);
      const p2: any = anthropic.messages.create(BODY);
      const r1 = await p1.asResponse();
      const r2 = await p2.asResponse();
      expect(constructCount).toBe(2);
      expect(r1).not.toBe(r2);
    } finally {
      (globalThis as any).Response = OriginalResponse;
    }
  });

  it("9: laziness — awaiting the promise WITHOUT touching either surface never constructs a Response", async () => {
    const fake = makeFakeAnthropic();
    initAllowed(fake);
    const OriginalResponse = (globalThis as any).Response;
    let constructCount = 0;
    class CountingResponse extends OriginalResponse {
      constructor(body?: any, init?: any) {
        super(body, init);
        constructCount++;
      }
    }
    (globalThis as any).Response = CountingResponse;
    try {
      const p: any = new fake.Anthropic().messages.create(BODY);
      await p;
      expect(constructCount).toBe(0);
    } finally {
      (globalThis as any).Response = OriginalResponse;
    }
  });

  it("10: negative memo — an always-throwing Response constructor is attempted EXACTLY ONCE (built-flag set before the try); every surface call keeps resolving response:null, never throws", async () => {
    const fake = makeFakeAnthropic();
    initAllowed(fake);
    const OriginalResponse = (globalThis as any).Response;
    let throwCount = 0;
    class ThrowingResponse {
      constructor() {
        throwCount++;
        throw new Error("Response ctor boom");
      }
    }
    (globalThis as any).Response = ThrowingResponse;
    try {
      const p: any = new fake.Anthropic().messages.create(BODY);
      const wr1 = await p.withResponse();
      const ar1 = await p.asResponse();
      const wr2 = await p.withResponse();
      expect(throwCount).toBe(1); // attempted once, never retried
      expect(wr1.response).toBeNull();
      expect(ar1).toBeNull();
      expect(wr2.response).toBeNull();
    } finally {
      (globalThis as any).Response = OriginalResponse;
    }
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Group C — the helper can never throw (golden rule). Driven via the sync
// `protect(..., isAsync:false, ...)` escape hatch so the fake `create`'s
// return value reaches `_preserveApiPromiseSurface` completely unwrapped.
// ═══════════════════════════════════════════════════════════════════════

function makeClient(): TokenPolice {
  const client = new TokenPolice({
    apiKey: "tp_sk_test123",
    baseUrl: "http://localhost:59998",
    timeout: 0.1,
    firewall: "off",
    deployment: "daemon",
  });
  setClient(client);
  return client;
}

/**
 * Patches a fresh synthetic class's sync `create` via the public escape hatch
 * (`protect(..., isAsync:false, { provider: "anthropic" })`), returning an
 * instance whose `.create(...)` calls `impl(...)` and hands the raw return
 * value straight through `_preserveApiPromiseSurface` — no Promise coercion.
 * A UNIQUE moduleName per call keeps `_wrapMethod`'s `_originals` dedupe from
 * skipping re-instrumentation across tests (mirrors tests/syncWrapperNoEnforce.test.ts).
 */
function makeSyncAnthropicCreate(moduleName: string, impl: (...a: any[]) => any) {
  class SyncMessages {
    create(...args: any[]): any {
      return impl(...args);
    }
  }
  protect(moduleName, ["prototype"], "create", false, {
    provider: "anthropic",
    module: SyncMessages,
  });
  return new SyncMessages();
}

describe("Group C: hostile shapes — the helper never throws into customer code", () => {
  // ── 11: globalThis.Response deleted ──────────────────────────────────
  it("11: globalThis.Response deleted → withResponse()/asResponse() resolve response:null, no throw", async () => {
    makeClient();
    const OriginalResponse = (globalThis as any).Response;
    delete (globalThis as any).Response;
    try {
      const instance = makeSyncAnthropicCreate("tp-hostile-11", () => Promise.resolve({ id: "s11" }));
      const p: any = instance.create({});
      const wr = await p.withResponse();
      expect(wr.response).toBeNull();
      const ar = await p.asResponse();
      expect(ar).toBeNull();
    } finally {
      (globalThis as any).Response = OriginalResponse;
    }
  });

  // ── 12: hostile Response shapes ──────────────────────────────────────
  it("12a: globalThis.Response present but not a function → response:null, no throw", async () => {
    makeClient();
    const OriginalResponse = (globalThis as any).Response;
    (globalThis as any).Response = 123;
    try {
      const instance = makeSyncAnthropicCreate("tp-hostile-12a", () => Promise.resolve({}));
      const { response } = await instance.create({}).withResponse();
      expect(response).toBeNull();
    } finally {
      (globalThis as any).Response = OriginalResponse;
    }
  });

  it("12b: Response an arrow function (passes typeof but is NOT newable) → response:null, no throw", async () => {
    // The `typeof === "function"` gate alone is insufficient — arrow functions
    // pass it but throw "X is not a constructor" on `new`. This is exactly
    // why the inner try/catch (not just the typeof check) matters.
    makeClient();
    const OriginalResponse = (globalThis as any).Response;
    (globalThis as any).Response = (() => {}) as any;
    try {
      const instance = makeSyncAnthropicCreate("tp-hostile-12b", () => Promise.resolve({}));
      const { response } = await instance.create({}).withResponse();
      expect(response).toBeNull();
    } finally {
      (globalThis as any).Response = OriginalResponse;
    }
  });

  it("12c: a Response constructor that throws → response:null, no throw", async () => {
    makeClient();
    const OriginalResponse = (globalThis as any).Response;
    class ThrowingCtor {
      constructor() {
        throw new Error("ctor boom");
      }
    }
    (globalThis as any).Response = ThrowingCtor as any;
    try {
      const instance = makeSyncAnthropicCreate("tp-hostile-12c", () => Promise.resolve({}));
      const { response } = await instance.create({}).withResponse();
      expect(response).toBeNull();
    } finally {
      (globalThis as any).Response = OriginalResponse;
    }
  });

  it("12d: a hostile getter on globalThis.Response that throws on read → response:null, no throw", async () => {
    makeClient();
    const desc = Object.getOwnPropertyDescriptor(globalThis, "Response");
    Object.defineProperty(globalThis, "Response", {
      configurable: true,
      get(): any {
        throw new Error("Response getter boom");
      },
    });
    try {
      const instance = makeSyncAnthropicCreate("tp-hostile-12d", () => Promise.resolve({}));
      let threw = false;
      let response: any;
      try {
        ({ response } = await instance.create({}).withResponse());
      } catch {
        threw = true;
      }
      expect(threw).toBe(false);
      expect(response).toBeNull();
    } finally {
      if (desc) Object.defineProperty(globalThis, "Response", desc);
    }
  });

  // ── 13: hostile VALUE shapes returned by the wrapped impl ────────────
  it("13a: a frozen thenable → no throw, same reference back, decoration silently declines", () => {
    makeClient();
    const frozen: any = Object.freeze({ then: (res: any) => res("frozen-value") });
    const instance = makeSyncAnthropicCreate("tp-hostile-13a", () => frozen);
    let result: any;
    expect(() => {
      result = instance.create({});
    }).not.toThrow();
    expect(result).toBe(frozen);
    expect(typeof result.withResponse).not.toBe("function");
  });

  it("13b: a sealed thenable → no throw, same reference back, decoration silently declines", () => {
    makeClient();
    // Sealed disallows adding NEW properties (withResponse doesn't exist yet) —
    // the assignment throws in strict mode, caught by the per-surface try/catch.
    const sealed: any = Object.seal({ then: (res: any) => res("sealed-value") });
    const instance = makeSyncAnthropicCreate("tp-hostile-13b", () => sealed);
    let result: any;
    expect(() => {
      result = instance.create({});
    }).not.toThrow();
    expect(result).toBe(sealed);
    expect(typeof result.withResponse).not.toBe("function");
  });

  it("13c: null passes through untouched, no throw", () => {
    makeClient();
    const instance = makeSyncAnthropicCreate("tp-hostile-13c", () => null);
    let result: any;
    expect(() => {
      result = instance.create({});
    }).not.toThrow();
    expect(result).toBeNull();
  });

  it("13d: undefined passes through untouched, no throw", () => {
    makeClient();
    const instance = makeSyncAnthropicCreate("tp-hostile-13d", () => undefined);
    let result: any;
    expect(() => {
      result = instance.create({});
    }).not.toThrow();
    expect(result).toBeUndefined();
  });

  it("13e: a plain string return value passes through untouched, no throw", () => {
    makeClient();
    const instance = makeSyncAnthropicCreate("tp-hostile-13e", () => "just a string");
    let result: any;
    expect(() => {
      result = instance.create({});
    }).not.toThrow();
    expect(result).toBe("just a string");
  });

  it("13f: a plain non-thenable object passes through untouched, not decorated", () => {
    makeClient();
    const obj = { id: "plain" };
    const instance = makeSyncAnthropicCreate("tp-hostile-13f", () => obj);
    const result: any = instance.create({});
    expect(result).toBe(obj);
    expect(typeof (result as any).withResponse).not.toBe("function");
  });

  it("13g: a thenable-shaped object whose `then` is a truthy NON-function → gate declines, not decorated", () => {
    makeClient();
    const obj: any = { then: 42 };
    const instance = makeSyncAnthropicCreate("tp-hostile-13g", () => obj);
    const result: any = instance.create({});
    expect(result).toBe(obj);
    expect(typeof result.withResponse).not.toBe("function");
  });

  it("13h: a throwing getter on `then` → no throw, same reference back, undecorated", () => {
    makeClient();
    const obj: any = {};
    Object.defineProperty(obj, "then", {
      configurable: true,
      get(): any {
        throw new Error("then getter boom");
      },
    });
    const instance = makeSyncAnthropicCreate("tp-hostile-13h", () => obj);
    let result: any;
    expect(() => {
      result = instance.create({});
    }).not.toThrow();
    expect(result).toBe(obj);
    expect(Object.prototype.hasOwnProperty.call(result, "withResponse")).toBe(false);
  });

  it("13i: a throwing getter on `withResponse` (on an otherwise real thenable) → no throw, same reference back, decoration skipped entirely", () => {
    makeClient();
    const obj: any = { then: (res: any) => res("x") };
    Object.defineProperty(obj, "withResponse", {
      configurable: true,
      get(): any {
        throw new Error("withResponse getter boom");
      },
    });
    const instance = makeSyncAnthropicCreate("tp-hostile-13i", () => obj);
    let result: any;
    expect(() => {
      result = instance.create({});
    }).not.toThrow();
    expect(result).toBe(obj);
    // The read of `withResponse` itself threw inside the outer feature-detect
    // try — asResponse/parse are never reached either.
    expect(Object.prototype.hasOwnProperty.call(result, "asResponse")).toBe(false);
  });

  it("13j: a non-writable, non-function `withResponse` property → that ONE assignment is caught, but asResponse/parse (independent try/catch blocks) still land", () => {
    makeClient();
    const obj: any = { then: (res: any) => res("y") };
    Object.defineProperty(obj, "withResponse", {
      value: "not-a-function",
      writable: false,
      configurable: false,
      enumerable: true,
    });
    const instance = makeSyncAnthropicCreate("tp-hostile-13j", () => obj);
    let result: any;
    expect(() => {
      result = instance.create({});
    }).not.toThrow();
    expect(result).toBe(obj);
    // The load-bearing withResponse assignment silently failed (non-writable) —
    // the original value survives untouched, exactly proving per-assignment
    // isolation (a partial failure must not abort the other two surfaces).
    expect(result.withResponse).toBe("not-a-function");
    expect(typeof result.asResponse).toBe("function");
    expect(typeof result.parse).toBe("function");
    expect(result.parse()).toBe(result);
  });

  // ── 14: already-APIPromise-like value returned untouched ─────────────
  it("14: an already-APIPromise-like value (a real withResponse function already present) is returned COMPLETELY UNTOUCHED", () => {
    makeClient();
    const origWithResponse = async () => ({ data: "already", response: null, request_id: "rid-1" });
    const origAsResponse = async () => null;
    const origParse = () => "parsed";
    const obj: any = {
      then: (res: any) => res("z"),
      withResponse: origWithResponse,
      asResponse: origAsResponse,
      parse: origParse,
    };
    const instance = makeSyncAnthropicCreate("tp-hostile-14", () => obj);
    const result: any = instance.create({});
    expect(result).toBe(obj);
    // The existing methods are NOT overwritten — identical references survive.
    expect(result.withResponse).toBe(origWithResponse);
    expect(result.asResponse).toBe(origAsResponse);
    expect(result.parse).toBe(origParse);
  });

  // ── 15: a rejecting promise still rejects through all three surfaces ─
  it("15: a rejecting promise still REJECTS through await p, withResponse(), and asResponse() — none silently swallows it", async () => {
    makeClient();
    const err = new Error("provider boom");
    const instance = makeSyncAnthropicCreate("tp-hostile-15", () => Promise.reject(err));
    const p: any = instance.create({});
    await Promise.all([
      expect(p).rejects.toBe(err),
      expect(p.withResponse()).rejects.toBe(err),
      expect(p.asResponse()).rejects.toBe(err),
    ]);
  });

  // ── 16: a synchronous throw propagates unchanged (Golden Rule anchor) ─
  it("16: a SYNCHRONOUS throw from the underlying impl propagates unchanged, same object by identity (how TokenPoliceBlockedError reaches the customer)", () => {
    makeClient();
    const boom = new Error("sync boom");
    const instance = makeSyncAnthropicCreate("tp-hostile-16", () => {
      throw boom;
    });
    let caught: unknown;
    try {
      instance.create({});
    } catch (e) {
      caught = e;
    }
    expect(caught).toBe(boom);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Group D — beta seam enforcement (Change 2)
// ═══════════════════════════════════════════════════════════════════════
describe("Group D: Anthropic Beta.Messages enforcement", () => {
  it("17: init() patches BOTH Messages.create and Beta.Messages.create as DISTINCT function objects (pre-fix: Beta.Messages.create was left completely unpatched)", () => {
    const fake = makeFakeAnthropic();
    const origMessagesCreate = fake.Messages.prototype.create;
    const origBetaCreate = fake.BetaMessages.prototype.create;
    initAllowed(fake);
    expect(fake.Messages.prototype.create).not.toBe(origMessagesCreate);
    expect(fake.BetaMessages.prototype.create).not.toBe(origBetaCreate);
    expect(fake.Messages.prototype.create).not.toBe(fake.BetaMessages.prototype.create);
  });

  it("18: client.beta.messages.create() issues exactly ONE pre-flight /check (pre-fix: ZERO)", async () => {
    const fake = makeFakeAnthropic();
    const { checkSpy } = initAllowed(fake);
    const result = await new fake.Anthropic().beta.messages.create(BODY);
    expect(result).toBeTruthy();
    expect(checkSpy).toHaveBeenCalledTimes(1);
  });

  it("no cross-seam double counting: one call to EACH seam ⇒ exactly 2 checks total (1 each, never merged, never doubled)", async () => {
    const fake = makeFakeAnthropic();
    const { checkSpy } = initAllowed(fake);
    await new fake.Anthropic().messages.create(BODY);
    await new fake.Anthropic().beta.messages.create(BODY);
    expect(checkSpy).toHaveBeenCalledTimes(2);
  });

  it("19: a streamed beta.messages.create({stream:true}), fully drained, still gets exactly ONE /check and exactly ONE metered /log row — neither doubled by the new row", async () => {
    const fake = makeFakeAnthropic();
    const { logSpy, checkSpy } = initAllowed(fake);
    const raw: any = await new fake.Anthropic().beta.messages.create({ ...BODY, stream: true });
    for await (const _ev of raw) void _ev;
    await flush();
    expect(checkSpy).toHaveBeenCalledTimes(1);
    expect(anthropicRows(logSpy)).toHaveLength(1);
  });

  it("20: countTokens on BOTH seams issues NO /check (deliberately excluded — a free metadata endpoint must never be blockable)", async () => {
    const fake = makeFakeAnthropic();
    const { checkSpy } = initAllowed(fake);
    await new fake.Anthropic().messages.countTokens(BODY);
    await new fake.Anthropic().beta.messages.countTokens(BODY);
    expect(checkSpy).not.toHaveBeenCalled();
  });

  it("21: uninstrument() restores BOTH prototypes pristine; re-init after uninstrument re-wraps, but a second init() WHILE STILL INSTRUMENTED dedupes rather than double-wrapping", () => {
    const fake = makeFakeAnthropic();
    const origMessagesCreate = fake.Messages.prototype.create;
    const origBetaCreate = fake.BetaMessages.prototype.create;

    initAllowed(fake);
    const wrapped1 = fake.Messages.prototype.create;
    const wrappedBeta1 = fake.BetaMessages.prototype.create;
    expect(wrapped1).not.toBe(origMessagesCreate);
    expect(wrappedBeta1).not.toBe(origBetaCreate);

    // Second init() while still instrumented (autoInstrument's `_isInstrumented`
    // guard) → dedupe: the prototype methods are untouched, not double-wrapped.
    init({
      apiKey: "tp_sk_test2",
      deployment: "serverless",
      firewall: "dry_run",
      instrumentModules: { anthropic: fake.Anthropic },
    } as any);
    expect(fake.Messages.prototype.create).toBe(wrapped1);
    expect(fake.BetaMessages.prototype.create).toBe(wrappedBeta1);

    uninstrument();
    expect(fake.Messages.prototype.create).toBe(origMessagesCreate);
    expect(fake.BetaMessages.prototype.create).toBe(origBetaCreate);

    // Re-init after a real uninstrument() re-wraps fresh.
    init({
      apiKey: "tp_sk_test3",
      deployment: "serverless",
      firewall: "dry_run",
      instrumentModules: { anthropic: fake.Anthropic },
    } as any);
    expect(fake.Messages.prototype.create).not.toBe(origMessagesCreate);
    expect(fake.BetaMessages.prototype.create).not.toBe(origBetaCreate);
  });

  it("22: __tpStreamPatched own-property guard — a Beta.Messages that INHERITS a truthy __tpStreamPatched from a parent Messages still gets its OWN stream patch (the hypothetical Beta.Messages extends Messages hardening case)", () => {
    // Real vendor shape today is a SIBLING class (asserted structurally by
    // test 17 above via `Messages.prototype.create !== BetaMessages.prototype.create`
    // on non-extending classes). This test targets the *guard itself*: with the
    // OLD plain-truthy read (`if (proto.__tpStreamPatched) return;`), a
    // subclass that inherits the parent's already-patched `.stream` would read
    // `__tpStreamPatched === true` through the prototype chain and silently
    // skip patching — losing enforcement/telemetry on the subclass. The FIX
    // uses `Object.prototype.hasOwnProperty.call(proto, "__tpStreamPatched")`,
    // which correctly ignores the inherited marker.
    class Messages {
      stream(_body: any): any {
        return { tag: "base-stream" };
      }
    }
    class BetaMessages extends Messages {
      // Deliberately does NOT override `.stream()` — it must be inherited,
      // exactly the scenario the own-property guard was hardened for.
    }
    function Anthropic(this: any) {}
    (Anthropic as any).Messages = Messages;
    (Anthropic as any).Beta = { Messages: BetaMessages };

    init({
      apiKey: "tp_sk_test",
      deployment: "serverless",
      firewall: "off",
      instrumentModules: { anthropic: Anthropic },
    } as any);

    // Both prototypes carry their OWN marker — Beta's is NOT merely inherited.
    expect(Object.prototype.hasOwnProperty.call(Messages.prototype, "__tpStreamPatched")).toBe(true);
    expect(Object.prototype.hasOwnProperty.call(BetaMessages.prototype, "__tpStreamPatched")).toBe(true);
    // Beta got its OWN `.stream` wrapper as an own property, not the inherited one.
    expect(Object.prototype.hasOwnProperty.call(BetaMessages.prototype, "stream")).toBe(true);
    expect(BetaMessages.prototype.stream).not.toBe(Messages.prototype.stream);
  });
});
