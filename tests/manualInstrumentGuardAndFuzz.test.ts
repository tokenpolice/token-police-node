/**
 * NEW-5 — companion to tests/manualInstrumentShapeNormalization.test.ts.
 * See that file's header for the full bug writeup and the load-bearing
 * "modern-shape (≥0.58) fixture" rationale — it applies identically here:
 * every fixture below is built as
 *   cls.Anthropic = <a DIFFERENT object that lacks .Messages>
 * never the @anthropic-ai/sdk 0.30.1 self-ref shape
 * (`tests/manualInstrumentDedupe.test.ts:136`), which would make these
 * tests pass identically pre-fix and post-fix.
 *
 * This file covers:
 *   (3) the anthropic in-band stream guard (`installAnthropicInBandStreamGuard`,
 *       src/telemetry.ts) NO-OPS when the target already carries a root
 *       `APIPromise` — the modern namespace path stays byte-for-byte
 *       unchanged.
 *   (4) the guard INSTALLS on a synthesized (class-form-normalized,
 *       APIPromise-less) namespace, and correctly routes `stream:true`
 *       calls around the Traceloop layer while leaving non-streaming calls
 *       routed through it.
 *   (6) golden-rule fuzz: neither `_pickOpenAIClass` nor
 *       `_pickAnthropicNamespace` may ever throw, for any input.
 *
 * `installAnthropicInBandStreamGuard` and `markManuallyInstrumented` are
 * exported and used directly (mirrors
 * tests/anthropicNestedCopyStreamGuard.test.ts); `_pickAnthropicNamespace`
 * / `_pickOpenAIClass` are module-private, so (3)/(4) reproduce ONLY the
 * DOCUMENTED shape those functions produce (never calling the private
 * functions themselves) to drive the exported guard directly, and (6)
 * fuzzes through the real public seam (`setupOpenTelemetry`) exactly as
 * the sibling file does — see that file's harness note on why spying on
 * the real instrumentor's `.prototype.manuallyInstrument` is the honest way
 * to observe a private normalizer's output from outside.
 */
import { describe, it, expect, afterEach, vi } from "vitest";
import { createRequire } from "module";
import { trace, context, propagation } from "@opentelemetry/api";

import {
  setupOpenTelemetry,
  unsetupOpenTelemetry,
  hardenInstrumentorPatches,
  installAnthropicInBandStreamGuard,
} from "../src/telemetry";

const requireCjs = createRequire(import.meta.url);

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
// Shared fixture — see tests/manualInstrumentShapeNormalization.test.ts's
// file-level comment for why this exact shape is load-bearing.
// ═══════════════════════════════════════════════════════════════════════
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
  function BaseAnthropicLike(this: any) {}

  function cls(this: any) {}
  (cls as any).Completions = Completions;
  (cls as any).Messages = Messages;
  (cls as any).Beta = { Messages: BetaMessages };
  (cls as any).Anthropic = BaseAnthropicLike;

  return { cls, Completions, Messages, BetaMessages, realCreate };
}

// ═══════════════════════════════════════════════════════════════════════
// (3) Guard no-ops when the namespace already carries root APIPromise.
// ═══════════════════════════════════════════════════════════════════════
describe("(3) installAnthropicInBandStreamGuard no-ops on a safe (APIPromise-carrying) namespace", () => {
  maybeAnthropic(
    "proto.create is left BYTE-FOR-BYTE unchanged by identity",
    () => {
      const { cls, Completions, Messages, BetaMessages } =
        makeModernAnthropicClass();
      // Already namespace-shaped AND carrying a root APIPromise — the
      // "modern namespace import" case that must stay completely untouched.
      const ns: any = {
        Anthropic: cls,
        default: cls,
        APIPromise: function APIPromise() {},
      };

      const inst = new AnthropicInstrumentation({ enabled: false });
      hardenInstrumentorPatches(inst, "anthropic");
      inst.manuallyInstrument(ns);

      const beforeCompletions = (Completions.prototype as any).create;
      const beforeMessages = (Messages.prototype as any).create;
      const beforeBeta = (BetaMessages.prototype as any).create;
      // Sanity: the real instrumentor did patch these (shimmer markers).
      expect(beforeMessages.__wrapped).toBe(true);

      installAnthropicInBandStreamGuard(ns);

      expect((Completions.prototype as any).create).toBe(beforeCompletions);
      expect((Messages.prototype as any).create).toBe(beforeMessages);
      expect((BetaMessages.prototype as any).create).toBe(beforeBeta);
    },
  );
});

// ═══════════════════════════════════════════════════════════════════════
// (4) Guard installs on a synthesized (in-band) namespace.
// ═══════════════════════════════════════════════════════════════════════
describe("(4) installAnthropicInBandStreamGuard installs on a synthesized in-band namespace", () => {
  maybeAnthropic(
    "guard present on all three surfaces; stream:true reaches RAW, non-stream still routes through the instrumentor",
    async () => {
      const { cls, Completions, Messages, BetaMessages } =
        makeModernAnthropicClass();
      // Reproduces ONLY the DOCUMENTED output shape of
      // `_pickAnthropicNamespace`'s class-form branch
      // (`Object.create(mod); ns.Anthropic = mod;`) — NOT calling the
      // private function itself (module-private, not exported). No root
      // `APIPromise` anywhere reachable off `ns` — the in-band condition.
      const ns: any = Object.create(cls);
      ns.Anthropic = cls;

      const inst = new AnthropicInstrumentation({ enabled: false });
      hardenInstrumentorPatches(inst, "anthropic");
      inst.manuallyInstrument(ns);
      installAnthropicInBandStreamGuard(ns);

      for (const proto of [
        Completions.prototype,
        Messages.prototype,
        BetaMessages.prototype,
      ]) {
        expect((proto as any).create.name).toBe("tpAnthropicInBandStreamGuard");
      }

      // `startSpan` is unconditionally the FIRST thing the real Traceloop
      // `method` wrapper does (patchAnthropic, both stream and non-stream) —
      // so "was startSpan called" is a faithful signal for "did the call
      // enter the Traceloop layer at all", independent of the rest of its
      // (network-free but still real) span/tracer machinery, which we stub
      // out here purely to avoid depending on a configured OTel provider
      // for what is otherwise still the REAL routing logic under test
      // (`_wrap`, `patchAnthropic`, `_wrapPromise`/`_streamingWrapPromise`).
      const fakeSpan = {
        setAttribute: () => {},
        setStatus: () => {},
        recordException: () => {},
        end: () => {},
      };
      const spanSpy = vi
        .spyOn(AnthropicInstrumentation.prototype, "startSpan")
        .mockReturnValue(fakeSpan as any);

      const streamResult = await (Messages.prototype as any).create.call(
        { _client: {} },
        { stream: true, model: "claude-3-5-sonnet-latest", messages: [] },
      );
      // Guard routes straight to the RAW method's return value — untouched
      // by the Traceloop layer (which, unguarded, would instead throw
      // "APIPromise is not a constructor" trying to wrap this same call).
      expect(streamResult).toBe("real-result");
      expect(spanSpy).not.toHaveBeenCalled();

      spanSpy.mockClear();
      const nonStreamResult = await (Messages.prototype as any).create.call(
        { _client: {} },
        { stream: false, model: "claude-3-5-sonnet-latest", messages: [] },
      );
      expect(nonStreamResult).toBe("real-result");
      expect(spanSpy).toHaveBeenCalledTimes(1); // Traceloop layer entered
    },
  );
});

// ═══════════════════════════════════════════════════════════════════════
// (6) Golden-rule fuzz — neither normalizer may throw for ANY input.
// ═══════════════════════════════════════════════════════════════════════
//
// `_pickOpenAIClass` / `_pickAnthropicNamespace` are module-private, and
// EVERY call site inside `_applyManualInstrumentations` — including
// `entry.normalizeManualTarget(userModule)` itself — sits inside the SAME
// per-entry try/catch that also wraps `instrumentor.manuallyInstrument()`.
// That means "setupOpenTelemetry never throws" is guaranteed structurally
// by that outer catch REGARDLESS of whether the normalizer's own internal
// try/catch does its job — it cannot, on its own, distinguish "the
// normalizer's own catch fired" from "the normalizer let an exception
// escape and the OUTER catch caught it instead". Both are externally silent.
//
// The stronger, honest seam used below: spy on the REAL instrumentor's
// `.prototype.manuallyInstrument` (same technique as the sibling file).
// `entry.normalizeManualTarget(userModule)` runs BEFORE
// `instrumentor.manuallyInstrument(target)` in the SAME statement sequence
// — if the normalizer's own try/catch didn't fire and an exception escaped
// it, `manuallyInstrument` would NEVER be reached at all (the outer catch
// intercepts before that line runs) and the spy would show zero calls. So
// "the spy WAS called, with the input back verbatim" is a genuine,
// per-fuzz-value proof that the normalizer completed on its own terms
// (matched, or caught-and-passed-through) rather than merely having its
// mess cleaned up by a catch two frames up.
describe("(6) golden-rule fuzz", () => {
  const maybeBoth = AnthropicInstrumentation && OpenAIInstrumentation ? it : it.skip;

  // ── Values that DO reach the normalizer (truthy under `if (!userModule)`) ──
  const TRUTHY_FUZZ: Array<{ name: string; make: () => any }> = [
    { name: "number 42", make: () => 42 },
    { name: "non-empty string", make: () => "str" },
    { name: "boolean true", make: () => true },
    { name: "BigInt(10)", make: () => BigInt(10) },
    { name: "Symbol('tp-fuzz')", make: () => Symbol("tp-fuzz") },
    { name: "Object.create(null)", make: () => Object.create(null) },
    {
      name: "frozen class",
      make: () => Object.freeze(class FrozenAnthropic {}),
    },
    {
      name: "sealed function",
      make: () => Object.seal(function sealedFn() {}),
    },
    {
      name: "revoked Proxy (object target)",
      make: () => {
        const { proxy, revoke } = Proxy.revocable({}, {});
        revoke();
        return proxy;
      },
    },
    {
      name: "revoked Proxy (function target)",
      make: () => {
        const { proxy, revoke } = Proxy.revocable(function revocableFn() {}, {});
        revoke();
        return proxy;
      },
    },
    {
      name: "Proxy with throwing get trap (object target)",
      make: () =>
        new Proxy(
          {},
          {
            get() {
              throw new Error("tp-fuzz: hostile get trap (object)");
            },
          },
        ),
    },
    {
      name: "Proxy with throwing get trap (function target)",
      make: () =>
        new Proxy(function proxyTargetFn() {}, {
          get() {
            throw new Error("tp-fuzz: hostile get trap (function)");
          },
        }),
    },
    {
      name: "circular .default",
      make: () => {
        const c: any = {};
        c.default = c;
        return c;
      },
    },
  ];

  for (const { name, make } of TRUTHY_FUZZ) {
    maybeBoth(`${name}: no throw, both normalizers pass it through verbatim`, () => {
      const value = make();
      const anthropicSpy = vi.spyOn(
        AnthropicInstrumentation.prototype,
        "manuallyInstrument",
      );
      const openaiSpy = vi.spyOn(
        OpenAIInstrumentation.prototype,
        "manuallyInstrument",
      );

      expect(() =>
        setupOpenTelemetry({ openAI: value, anthropic: value } as any),
      ).not.toThrow();

      expect(anthropicSpy).toHaveBeenCalledTimes(1);
      expect(anthropicSpy.mock.calls[0][0]).toBe(value);
      expect(openaiSpy).toHaveBeenCalledTimes(1);
      expect(openaiSpy.mock.calls[0][0]).toBe(value);

      expect(() => unsetupOpenTelemetry()).not.toThrow();
    });
  }

  // ── Hostile getters on the SPECIFIC properties each normalizer reads ──
  const HOSTILE_ANTHROPIC: Array<{ name: string; make: () => any }> = [
    {
      name: "hostile getter on .Anthropic",
      make: () => {
        const o: any = {};
        Object.defineProperty(o, "Anthropic", {
          get() {
            throw new Error("tp-fuzz: hostile Anthropic getter");
          },
        });
        return o;
      },
    },
    {
      name: "hostile getter on .default",
      make: () => {
        const o: any = {};
        Object.defineProperty(o, "default", {
          get() {
            throw new Error("tp-fuzz: hostile default getter");
          },
        });
        return o;
      },
    },
  ];
  for (const { name, make } of HOSTILE_ANTHROPIC) {
    maybeAnthropic(`anthropic — ${name}: no throw, verbatim passthrough`, () => {
      const value = make();
      const spy = vi.spyOn(
        AnthropicInstrumentation.prototype,
        "manuallyInstrument",
      );
      expect(() =>
        setupOpenTelemetry({ anthropic: value } as any),
      ).not.toThrow();
      expect(spy).toHaveBeenCalledTimes(1);
      expect(spy.mock.calls[0][0]).toBe(value);
    });
  }

  const maybeOpenAI = OpenAIInstrumentation ? it : it.skip;
  const HOSTILE_OPENAI: Array<{ name: string; make: () => any }> = [
    {
      name: "hostile getter on .default",
      make: () => {
        const o: any = {};
        Object.defineProperty(o, "default", {
          get() {
            throw new Error("tp-fuzz: hostile default getter");
          },
        });
        return o;
      },
    },
    {
      name: "hostile getter on .OpenAI",
      make: () => {
        const o: any = {};
        Object.defineProperty(o, "OpenAI", {
          get() {
            throw new Error("tp-fuzz: hostile OpenAI getter");
          },
        });
        return o;
      },
    },
    {
      name: "hostile getter on .Chat (function-typed value)",
      make: () => {
        const fn: any = function HostileFn() {};
        Object.defineProperty(fn, "Chat", {
          get() {
            throw new Error("tp-fuzz: hostile Chat getter");
          },
        });
        return fn;
      },
    },
  ];
  for (const { name, make } of HOSTILE_OPENAI) {
    maybeOpenAI(`openAI — ${name}: no throw, verbatim passthrough`, () => {
      const value = make();
      const spy = vi.spyOn(OpenAIInstrumentation.prototype, "manuallyInstrument");
      expect(() => setupOpenTelemetry({ openAI: value } as any)).not.toThrow();
      expect(spy).toHaveBeenCalledTimes(1);
      expect(spy.mock.calls[0][0]).toBe(value);
    });
  }

  // ── Falsy values: filtered by `if (!userModule) continue` BEFORE either
  //    normalizer runs at all. See the comment inline below for why this is
  //    a distinct (weaker, and separately justified) claim from the truthy
  //    cases above. ──
  it("null / undefined / 0 / '' / false / NaN: never throw (filtered before reaching either normalizer)", () => {
    // `_applyManualInstrumentations` does `if (!userModule) continue`
    // BEFORE calling `normalizeManualTarget` — so these six values never
    // exercise either normalizer's OWN try/catch at all; this only proves
    // that early-exit gate doesn't throw. Separately: neither normalizer
    // COULD throw on null/undefined even without that gate — every access
    // in both (`mod?.Anthropic`, `mod?.OpenAI`, `mod?.default`, …) uses `?.`
    // optional chaining, and `null?.x` / `undefined?.x` is a language-level
    // no-throw guarantee, not something a try/catch does any work to
    // prevent. That's why null/undefined aren't asserted against
    // `manuallyInstrument` call counts the way the truthy cases above are —
    // for null/undefined specifically, `userModule` is never even READ into
    // a `manuallyInstrument` call, verbatim or otherwise.
    for (const value of [null, undefined, 0, "", false, NaN]) {
      expect(() =>
        setupOpenTelemetry({ openAI: value, anthropic: value } as any),
      ).not.toThrow();
      expect(() => unsetupOpenTelemetry()).not.toThrow();
    }
  });

  // ── The one internal-implementation-specific edge case named in the
  //    task: a class whose shadow-assignment (`ns.Anthropic = mod`) throws
  //    because it inherits a NON-WRITABLE `.Anthropic` static. ──
  maybeAnthropic(
    "class with a non-writable inherited .Anthropic: shadow-assignment throws (strict mode) and must be caught",
    () => {
      class InnerCompletions {}
      class InnerMessages {}
      class InnerBetaMessages {}
      class WithNonWritableAnthropic {
        static Completions = InnerCompletions;
        static Messages = InnerMessages;
        static Beta = { Messages: InnerBetaMessages };
      }
      // Own, non-writable static `.Anthropic` — a DIFFERENT object (models
      // BaseAnthropic), same as every other fixture in this suite. When
      // `_pickAnthropicNamespace`'s class-form branch does
      // `Object.create(mod)` then `ns.Anthropic = mod`, that assignment
      // walks the new object's prototype chain (= `mod` itself), finds this
      // non-writable OWN property, and — because this file (like all
      // TS/ESM output) runs in strict mode — throws a TypeError instead of
      // silently no-op'ing. The normalizer's own try/catch must swallow
      // that and fall back to `return mod` (verbatim).
      Object.defineProperty(WithNonWritableAnthropic, "Anthropic", {
        value: function DecoyBaseAnthropic() {},
        writable: false,
        configurable: true,
        enumerable: false,
      });

      const spy = vi.spyOn(
        AnthropicInstrumentation.prototype,
        "manuallyInstrument",
      );

      expect(() =>
        setupOpenTelemetry({ anthropic: WithNonWritableAnthropic } as any),
      ).not.toThrow();

      // Reached manuallyInstrument (the shadow-assignment's throw was
      // caught INSIDE normalizeManualTarget, not by the outer per-entry
      // catch), with the class handed back verbatim — the shadow
      // assignment never landed, so `.Anthropic` is still whatever the
      // class itself carries (DecoyBaseAnthropic), not re-pointed at the
      // class.
      expect(spy).toHaveBeenCalledTimes(1);
      expect(spy.mock.calls[0][0]).toBe(WithNonWritableAnthropic);
    },
  );
});
