/**
 * NEW-5 — manual `instrumentModules.anthropic` shape normalization
 * (`_pickOpenAIClass` / `_pickAnthropicNamespace`, src/telemetry.ts) +
 * the companion `APIPromise` carry-through fix in `autoInstrument`
 * (src/enforcer.ts).
 *
 * `_applyManualInstrumentations()` used to hand the user's raw
 * `instrumentModules[key]` value STRAIGHT to `instrumentor.manuallyInstrument()`.
 * The two Traceloop instrumentors read OPPOSITE shapes off that argument:
 *   - @traceloop/instrumentation-openai (0.26) reads
 *     `openaiModule.Chat.Completions.prototype` /
 *     `openaiModule.Completions.prototype` — it wants the `OpenAI` CLASS.
 *   - @traceloop/instrumentation-anthropic (0.27) reads
 *     `module.Anthropic.Completions.prototype`,
 *     `module.Anthropic.Messages.prototype`,
 *     `module.Anthropic.Beta.Messages.prototype` — it wants the module
 *     NAMESPACE (an object carrying `.Anthropic`).
 * Passing a `@anthropic-ai/sdk` CLASS (`import Anthropic from
 * "@anthropic-ai/sdk"`, the shape the docs used to recommend) threw
 * "Cannot read properties of undefined (reading 'prototype')" inside
 * manuallyInstrument() and silently dropped ALL Anthropic telemetry, because
 * `Anthropic.Anthropic` is the inherited `BaseAnthropic` static self-ref,
 * which does not carry `.Messages`.
 *
 * The fix adds a per-registry-entry `normalizeManualTarget` (backed by
 * `_pickOpenAIClass` / `_pickAnthropicNamespace`) that coerces the value to
 * whichever shape THAT instrumentor needs before calling manuallyInstrument().
 *
 * ═══════════════════════════════════════════════════════════════════════
 * ⚠️  FIXTURE SHAPE IS THE SINGLE MOST IMPORTANT THING IN THIS FILE.
 * ═══════════════════════════════════════════════════════════════════════
 * This bug is VERSION-DEPENDENT:
 *   - On @anthropic-ai/sdk 0.30.1: `Anthropic.Anthropic === Anthropic` (a
 *     static SELF-reference — the class points at itself). So
 *     `module.Anthropic.Messages` already resolves fine off the bare class,
 *     with ZERO normalization. `tests/manualInstrumentDedupe.test.ts:136`'s
 *     `mkFakeAnthropicModule()` builds exactly this shape
 *     (`mod.default = mod.Anthropic`, i.e. the "namespace" IS the class
 *     tree) — that's correct for ITS purpose (loader-hook dedupe by module
 *     identity, orthogonal to shape normalization), but reusing it here
 *     would make every "fails pre-fix" assertion in this file pass BOTH
 *     pre-fix and post-fix, proving nothing about this bug.
 *   - On 0.58.0 / 0.65.0 / 0.120.0: `Anthropic.Anthropic` is `BaseAnthropic`
 *     — a DIFFERENT object — which does NOT carry `.Messages`.
 *     `module.Anthropic.Messages.prototype` throws
 *     "Cannot read properties of undefined (reading 'prototype')". THIS is
 *     the bug.
 *
 * Every "modern-shape" class fixture below is therefore built as:
 *   cls = function Anthropic() {}
 *   cls.Completions   = class { }   // carries .prototype
 *   cls.Messages      = class { }
 *   cls.Beta          = { Messages: class { } }
 *   cls.Anthropic     = BaseAnthropicLike   // a DIFFERENT object, no .Messages
 * — never the 0.30.1 self-ref shape.
 *
 * ── Harness notes ──────────────────────────────────────────────────────
 * `_pickOpenAIClass` / `_pickAnthropicNamespace` are module-private (not
 * exported) and deliberately NOT exported for this file — every case here
 * drives them through the real public seam instead:
 *   - `setupOpenTelemetry({ instrumentModules })` (src/telemetry.ts), which
 *     is what actually calls `normalizeManualTarget` before
 *     `manuallyInstrument()`.
 *   - The REAL `AnthropicInstrumentation` / `OpenAIInstrumentation` classes
 *     are required out of node_modules (mirrors
 *     tests/manualInstrumentDedupe.test.ts:119) and we `vi.spyOn` their
 *     shared `.prototype.manuallyInstrument` BEFORE calling
 *     `setupOpenTelemetry`. Because `_applyManualInstrumentations`
 *     constructs `new InstrumentorClass()` internally (we never get a
 *     handle to that instance), spying on the PROTOTYPE method is the only
 *     way to observe exactly what argument reached the real instrumentor —
 *     and critically, `spy.mock.calls[0][0]` gives us TRUE reference
 *     identity (`toBe` / `Object.is`) of the normalized value, without
 *     needing to export the private normalizer itself. This is the
 *     technique used to prove IDENTITY preservation in case (2) below —
 *     inspecting only the wrapped prototypes (as e.g.
 *     tests/anthropicNestedCopyStreamGuard.test.ts does) can't distinguish
 *     "returned the same namespace object" from "synthesized a new one
 *     that happens to share the same inner class", because both mutate the
 *     same shared `Messages.prototype.create`.
 *   - `@anthropic-ai/sdk` / `openai` are NOT devDependencies — every
 *     fixture is a hand-built fake reproducing only the vendor CONTRACT
 *     the instrumentors depend on (mirrors
 *     tests/instrumentModulesClassNormalization.test.ts).
 *
 * Case (5) targets a DIFFERENT function — `autoInstrument` in
 * src/enforcer.ts, which independently carries the root `APIPromise` export
 * across its own `_pickClassExport` normalization so the
 * `anthropicStreamBypass` probe in `_wrapMethod` still sees it. That fix and
 * this file's telemetry.ts fix are two separate code paths (init() calls
 * `setupOpenTelemetry` AND `autoInstrument` back to back — see
 * src/client.ts) — case (5) drives `autoInstrument` DIRECTLY, isolated from
 * the telemetry.ts normalizer, exactly like
 * tests/instrumentModulesClassNormalization.test.ts's tests (9)/(10) do.
 */
import { describe, it, expect, afterEach, vi } from "vitest";
import { createRequire } from "module";
import { trace, context, propagation } from "@opentelemetry/api";

import { setupOpenTelemetry, unsetupOpenTelemetry } from "../src/telemetry";
import { autoInstrument, uninstrument } from "../src/enforcer";

const requireCjs = createRequire(import.meta.url);

// ── Real instrumentor classes (optionalDependencies — may be absent) ──────
let AnthropicInstrumentation: any;
try {
  AnthropicInstrumentation = requireCjs(
    "@traceloop/instrumentation-anthropic",
  ).AnthropicInstrumentation;
} catch {
  AnthropicInstrumentation = undefined;
}
let OpenAIInstrumentation: any;
try {
  OpenAIInstrumentation = requireCjs(
    "@traceloop/instrumentation-openai",
  ).OpenAIInstrumentation;
} catch {
  OpenAIInstrumentation = undefined;
}
const maybeAnthropic = AnthropicInstrumentation ? it : it.skip;
const maybeOpenAI = OpenAIInstrumentation ? it : it.skip;

function resetTelemetryGlobals(): void {
  try {
    unsetupOpenTelemetry();
  } catch {
    /* ignore */
  }
  try {
    trace.disable();
  } catch {
    /* ignore */
  }
  try {
    context.disable();
  } catch {
    /* ignore */
  }
  try {
    propagation.disable();
  } catch {
    /* ignore */
  }
}

afterEach(() => {
  resetTelemetryGlobals();
  vi.restoreAllMocks();
});

// ═══════════════════════════════════════════════════════════════════════
// Shared fixture: the modern-shape (≥0.58) Anthropic class.
// ═══════════════════════════════════════════════════════════════════════
/**
 * See the file-level "FIXTURE SHAPE" comment above for why this is built
 * this way and NOT as the 0.30.1 self-ref shape.
 */
function makeModernAnthropicClass() {
  const realCreate = async function create(): Promise<string> {
    return "real-result";
  };
  class Completions {
    create(...args: any[]) {
      return realCreate.apply(this, args as any);
    }
  }
  class Messages {
    create(...args: any[]) {
      return realCreate.apply(this, args as any);
    }
  }
  class BetaMessages {
    create(...args: any[]) {
      return realCreate.apply(this, args as any);
    }
  }
  // Stand-in for the real `BaseAnthropic` — a DIFFERENT object from `cls`,
  // and (like the real BaseAnthropic) it carries none of the resource
  // statics. This is the load-bearing part of the fixture: it's what makes
  // `cls.Anthropic.Completions.prototype` throw pre-fix.
  function BaseAnthropicLike(this: any) {}

  function cls(this: any) {}
  (cls as any).Completions = Completions;
  (cls as any).Messages = Messages;
  (cls as any).Beta = { Messages: BetaMessages };
  (cls as any).Anthropic = BaseAnthropicLike;

  return { cls, Completions, Messages, BetaMessages, realCreate };
}

// ═══════════════════════════════════════════════════════════════════════
// (1) Class-form anthropic normalizes.
// ═══════════════════════════════════════════════════════════════════════
describe("(1) class-form anthropic normalizes (modern-shape ≥0.58 fixture)", () => {
  maybeAnthropic(
    "sanity: the RAW (unnormalized) class genuinely throws through the real instrumentor",
    () => {
      // Proves the fixture actually models the bug, independent of the fix
      // under test: this calls manuallyInstrument() DIRECTLY on the class,
      // reproducing the PRE-FIX code path
      // (`instrumentor.manuallyInstrument(userModule)` with no
      // normalization at all — see the src/telemetry.ts diff). `cls.Anthropic`
      // is `BaseAnthropicLike`, which has no `.Completions`, so
      // `.Completions.prototype` throws "Cannot read properties of
      // undefined (reading 'prototype')" before any other statement runs.
      const { cls } = makeModernAnthropicClass();
      const inst = new AnthropicInstrumentation({ enabled: false });
      expect(() => inst.manuallyInstrument(cls)).toThrow(/prototype/);
    },
  );

  maybeAnthropic(
    "fix: setupOpenTelemetry normalizes the class first — no throw, all three prototypes wrap",
    () => {
      const { cls, Completions, Messages, BetaMessages } =
        makeModernAnthropicClass();
      const spy = vi.spyOn(
        AnthropicInstrumentation.prototype,
        "manuallyInstrument",
      );

      // WHY THIS GENUINELY FAILS PRE-FIX: pre-fix,
      // `_applyManualInstrumentations` calls
      // `instrumentor.manuallyInstrument(userModule)` with the class
      // untouched — the sanity test above proves that throws. That throw
      // is caught by the per-entry try/catch in `_applyManualInstrumentations`
      // (a warning is logged, "calls to this provider will NOT be metered"),
      // so `setupOpenTelemetry` itself would still not throw — but
      // `manuallyInstrument` would NEVER be invoked with a working target,
      // and none of `Completions`/`Messages`/`BetaMessages.prototype.create`
      // would ever become `__wrapped`. Every assertion below — the class
      // NOT being what was passed to manuallyInstrument, and all three
      // `__wrapped` checks — fails pre-fix.
      expect(() => setupOpenTelemetry({ anthropic: cls } as any)).not.toThrow();

      expect(spy).toHaveBeenCalledTimes(1);
      const target = spy.mock.calls[0][0];
      // Normalization actually happened: NOT the bare class (which throws).
      expect(target).not.toBe(cls);
      // Synthesized namespace shadows `.Anthropic` with the real class, per
      // `_pickAnthropicNamespace`'s documented `ns.Anthropic = mod`.
      expect(target.Anthropic).toBe(cls);

      expect((Completions.prototype as any).create.__wrapped).toBe(true);
      expect((Messages.prototype as any).create.__wrapped).toBe(true);
      expect((BetaMessages.prototype as any).create.__wrapped).toBe(true);
    },
  );
});

// ═══════════════════════════════════════════════════════════════════════
// (2) Namespace-form anthropic preserved BY IDENTITY.
// ═══════════════════════════════════════════════════════════════════════
describe("(2) namespace-form anthropic preserved BY IDENTITY", () => {
  maybeAnthropic(
    "already-namespace-shaped input comes back === the exact object passed in",
    () => {
      const { cls, Completions, Messages, BetaMessages } =
        makeModernAnthropicClass();
      // Exactly the shape `import * as Anthropic from "@anthropic-ai/sdk"`
      // produces: an object literal carrying `.Anthropic` = the class, with
      // all three resource statics reachable through it.
      const ns: any = { Anthropic: cls, default: cls };
      const spy = vi.spyOn(
        AnthropicInstrumentation.prototype,
        "manuallyInstrument",
      );

      expect(() => setupOpenTelemetry({ anthropic: ns } as any)).not.toThrow();

      expect(spy).toHaveBeenCalledTimes(1);
      // THE GUARD ASSERTION: a future refactor that always synthesizes a
      // fresh namespace view (even for input that's already correctly
      // shaped) would still leave `target.Anthropic === cls`
      // (functionally indistinguishable from the wrap/telemetry side — the
      // prototypes still get patched) but would break `===` against the
      // ORIGINAL `ns` object the app imported. That's exactly the
      // regression that would silently break every one of the 8 fleet apps
      // that pass `import * as AnthropicModule` — nothing about their
      // Traceloop metering would look wrong, but any code the app itself
      // still holds a reference to `ns` for would now be talking to a
      // different object than what got patched. `toBe` uses `Object.is`,
      // so this is checking REFERENCE identity, not "does it still work".
      expect(spy.mock.calls[0][0]).toBe(ns);

      expect((Completions.prototype as any).create.__wrapped).toBe(true);
      expect((Messages.prototype as any).create.__wrapped).toBe(true);
      expect((BetaMessages.prototype as any).create.__wrapped).toBe(true);
    },
  );
});

// ═══════════════════════════════════════════════════════════════════════
// (7) OpenAI direction — _pickOpenAIClass.
// ═══════════════════════════════════════════════════════════════════════
describe("(7) OpenAI direction — _pickOpenAIClass", () => {
  function makeOpenAIClassFixture() {
    class Completions {
      create() {
        return "real-completions";
      }
    }
    class ChatCompletions {
      create() {
        return "real-chat";
      }
    }
    function OpenAI(this: any) {}
    (OpenAI as any).Chat = { Completions: ChatCompletions };
    (OpenAI as any).Completions = Completions;
    return { OpenAI, Completions, ChatCompletions };
  }

  maybeOpenAI(
    "class-form input (`import OpenAI from \"openai\"`) returned BY IDENTITY",
    () => {
      const { OpenAI, Completions, ChatCompletions } = makeOpenAIClassFixture();
      const spy = vi.spyOn(OpenAIInstrumentation.prototype, "manuallyInstrument");

      expect(() => setupOpenTelemetry({ openAI: OpenAI } as any)).not.toThrow();

      expect(spy).toHaveBeenCalledTimes(1);
      // First candidate in `_pickOpenAIClass` (`mod` itself) matches
      // immediately — verbatim, by reference.
      expect(spy.mock.calls[0][0]).toBe(OpenAI);

      expect((ChatCompletions.prototype as any).create.__wrapped).toBe(true);
      expect((Completions.prototype as any).create.__wrapped).toBe(true);
    },
  );

  maybeOpenAI(
    "namespace-form input (`import * as OpenAI from \"openai\"`): .OpenAI is extracted",
    () => {
      const { OpenAI, Completions, ChatCompletions } = makeOpenAIClassFixture();
      const ns: any = { OpenAI };
      const spy = vi.spyOn(OpenAIInstrumentation.prototype, "manuallyInstrument");

      expect(() => setupOpenTelemetry({ openAI: ns } as any)).not.toThrow();

      expect(spy).toHaveBeenCalledTimes(1);
      // Extracted OFF the namespace — the class, not the namespace wrapper.
      expect(spy.mock.calls[0][0]).toBe(OpenAI);
      expect(spy.mock.calls[0][0]).not.toBe(ns);

      expect((ChatCompletions.prototype as any).create.__wrapped).toBe(true);
      expect((Completions.prototype as any).create.__wrapped).toBe(true);
    },
  );
});

// ═══════════════════════════════════════════════════════════════════════
// (5) THE APIPromise bypass verdict — enforcer.ts `autoInstrument`.
// ═══════════════════════════════════════════════════════════════════════
describe("(5) APIPromise bypass verdict (autoInstrument, src/enforcer.ts)", () => {
  afterEach(() => {
    try {
      uninstrument();
    } catch {
      /* ignore */
    }
  });

  /**
   * Anthropic class fixture whose `Messages` / `Beta.Messages` `.create` is
   * PRE-WRAPPED with shimmer markers (`__wrapped`/`__original`) AND records
   * every invocation. This simulates the Traceloop auto-discovery patch
   * that — in production — already ran on this prototype BEFORE
   * `autoInstrument` wraps it (`init()` calls `setupOpenTelemetry()` then
   * `autoInstrument()`, in that order — src/client.ts). Isolating this from
   * the REAL Traceloop instrumentor (unlike cases 1-4 above) mirrors
   * `tests/instrumentModulesClassNormalization.test.ts`'s own harness note:
   * "every fixture is a hand-built fake reproducing only the vendor
   * CONTRACT autoInstrument depends on".
   *
   * `traceloopCalls` is the observable seam for this case: `moduleMap`
   * itself is private to `enforcer.ts` (not reachable from outside), and so
   * is the `anthropicStreamBypass` boolean it feeds. But `anthropicStreamBypass`
   * has exactly ONE externally-visible effect (`_wrapMethod`'s stream
   * branch): when true, a streamed call's `callee` is rebound to
   * `original.__original` (the RAW method, skipping the pre-wrapped
   * "Traceloop" layer above); when false, `callee` stays `original` (that
   * layer runs). So "did the pre-wrapped layer get entered for a
   * `stream:true` call" is a faithful, honest proxy for
   * `moduleMap["@anthropic-ai/sdk"].APIPromise` being a function — it's the
   * exact code path that field exists to steer.
   */
  function makeTraceloopPatchedAnthropic() {
    const traceloopCalls: string[] = [];
    const rawCreate = async function create(body: any): Promise<any> {
      return {
        id: "msg_1",
        model: body?.model,
        role: "assistant",
        content: [{ type: "text", text: "hi" }],
        stop_reason: "end_turn",
        usage: { input_tokens: 3, output_tokens: 2 },
      };
    };
    const mkTraceloopLayer = () => {
      const wrapper: any = function (this: any, ...args: any[]) {
        traceloopCalls.push((args?.[0] as any)?.stream ? "stream" : "non-stream");
        return rawCreate.apply(this, args);
      };
      wrapper.__wrapped = true;
      wrapper.__original = rawCreate;
      return wrapper;
    };
    class Messages {}
    (Messages.prototype as any).create = mkTraceloopLayer();
    class BetaMessages {}
    (BetaMessages.prototype as any).create = mkTraceloopLayer();

    function Anthropic(this: any) {
      this.messages = new Messages();
      this.beta = { messages: new BetaMessages() };
    }
    (Anthropic as any).Messages = Messages;
    (Anthropic as any).Beta = { Messages: BetaMessages };

    return { Anthropic, Messages, BetaMessages, traceloopCalls };
  }

  const STREAM_BODY = {
    model: "claude-3",
    messages: [{ role: "user", content: "hi" }],
    stream: true,
  };

  it("namespace WITH root APIPromise → the pre-wrapped layer stays engaged for stream:true (bypass OFF)", async () => {
    const { Anthropic, traceloopCalls } = makeTraceloopPatchedAnthropic();
    const ns: any = {
      Anthropic,
      default: Anthropic,
      APIPromise: function APIPromise() {},
    };

    autoInstrument({ anthropic: ns } as any);
    const client = new (Anthropic as any)();
    await client.messages.create(STREAM_BODY);

    // WHY THIS GENUINELY FAILS PRE-FIX: pre-fix, `autoInstrument` always
    // built `moduleMap["@anthropic-ai/sdk"] = { Anthropic: cls, default:
    // cls }` — NO `.APIPromise` field, regardless of what `ns` carried (see
    // the src/enforcer.ts diff: the whole `entry`/try-catch block that reads
    // `mod?.APIPromise ?? mod?.default?.APIPromise ?? mod?.Anthropic?.APIPromise`
    // and conditionally sets `entry.APIPromise` did not exist). So
    // `anthropicStreamBypass` evaluated to TRUE even for THIS namespace
    // input (which genuinely has a root APIPromise), and the streamed call's
    // `callee` was rebound to `original.__original` (rawCreate) — the
    // pre-wrapped layer was NEVER entered, so `traceloopCalls` would stay
    // EMPTY pre-fix. Post-fix, `entry.APIPromise` carries the export
    // through, `anthropicStreamBypass` is false, and `callee` stays
    // `original` — the pre-wrapped layer runs and records "stream".
    expect(traceloopCalls).toEqual(["stream"]);
  });

  it("class-form input (no APIPromise reachable anywhere) → bypass correctly stays ON", async () => {
    const { Anthropic, traceloopCalls } = makeTraceloopPatchedAnthropic();
    // Pass the CLASS directly — `mod?.APIPromise`, `mod?.default?.APIPromise`,
    // and `mod?.Anthropic?.APIPromise` are all undefined off it (the fixture
    // never sets any of them), so there is nothing to recover. This is a
    // BASELINE / parity assertion, not a "fails pre-fix" one — class-form
    // input behaves identically before and after the fix (the source
    // comment explicitly documents "for class-form input there is nothing
    // to recover and the bypass correctly stays on"). It guards against the
    // fix accidentally flipping the bypass off for this shape too.
    autoInstrument({ anthropic: Anthropic } as any);
    const client = new (Anthropic as any)();
    await client.messages.create(STREAM_BODY);

    expect(traceloopCalls).toEqual([]); // pre-wrapped layer never entered
  });
});
