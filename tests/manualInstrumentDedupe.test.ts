/**
 * Manual-instrumentation loader-hook dedupe (patch idempotency by module identity).
 *
 * Customer-reported crash (AnythingLLM, CJS, @anthropic-ai/sdk@0.39.0):
 * tp.init({ instrumentModules: { anthropic } }) manually patches the module at
 * init, but the Traceloop instrumentor's require-in-the-middle hook stays
 * armed — and its per-file cache has never seen the module (it was loaded
 * BEFORE the hook registered). The app's later lazy require() of the same
 * module fires the hook and applies a SECOND Traceloop patch layer on top of
 * the enforcer wrapper. The enforcer's anthropicStreamBypass now sits below
 * the broken layer, so every streamed messages.create({stream:true}) throws
 * "TypeError: moduleExports.APIPromise is not a constructor" INTO CUSTOMER
 * CODE — a golden-rule violation. For providers whose patch doesn't throw
 * (openai), the duplicate layer silently double-counts spans/cost instead.
 *
 * Fix: markManuallyInstrumented() records the module's identity on the
 * instrumentor after a successful manuallyInstrument(); the hardened loader
 * hook (hardenInstrumentorPatches) returns the exports untouched when the
 * incoming module identity-matches. Identity miss → patch runs as before.
 */
import { describe, it, expect, vi } from "vitest";
import { createRequire } from "module";

import {
  hardenInstrumentorPatches,
  markManuallyInstrumented,
} from "../src/telemetry";

const requireCjs = createRequire(import.meta.url);

describe("markManuallyInstrumented + hardened hook dedupe — unit", () => {
  const mkInst = () => {
    const patchSpy = vi.fn((exp: any) => exp);
    const unpatchSpy = vi.fn((exp: any) => exp);
    const def: any = { name: "@anthropic-ai/sdk", patch: patchSpy, unpatch: unpatchSpy };
    const inst: any = { _modules: [def] };
    hardenInstrumentorPatches(inst, "anthropic");
    return { inst, def, patchSpy, unpatchSpy };
  };

  it("skips re-patch of the exact module already manually instrumented", () => {
    const { inst, def, patchSpy } = mkInst();
    const mod: any = { Anthropic: {} };

    markManuallyInstrumented(inst, mod);

    // Loader hook fires later with the SAME exports object (CJS require cache).
    expect(def.patch(mod, "0.39.0")).toBe(mod);
    expect(patchSpy).not.toHaveBeenCalled();
  });

  it("still patches a DIFFERENT module (nested copy) of the same package", () => {
    const { inst, def, patchSpy } = mkInst();
    markManuallyInstrumented(inst, { Anthropic: {} });

    const nestedCopy: any = { Anthropic: {} }; // distinct identity
    def.patch(nestedCopy, "0.9.1");
    expect(patchSpy).toHaveBeenCalledTimes(1);
    expect(patchSpy).toHaveBeenCalledWith(nestedCopy, "0.9.1");
  });

  it("matches ESM namespace whose .default is the manually instrumented class", () => {
    const { inst, def, patchSpy } = mkInst();
    class FakeAnthropic {}
    markManuallyInstrumented(inst, FakeAnthropic);

    // import-in-the-middle hands the hook a namespace object, not the class.
    const namespace: any = { default: FakeAnthropic };
    expect(def.patch(namespace, "0.39.0")).toBe(namespace);
    expect(patchSpy).not.toHaveBeenCalled();
  });

  it("matches CJS exports when the user passed the namespace's .default", () => {
    const { inst, def, patchSpy } = mkInst();
    class FakeAnthropic {}
    const userNamespace: any = { default: FakeAnthropic };
    markManuallyInstrumented(inst, userNamespace);

    // Hook sees the raw class (module.exports === the class).
    expect(def.patch(FakeAnthropic, "0.39.0")).toBe(FakeAnthropic);
    expect(patchSpy).not.toHaveBeenCalled();
  });

  it("never dedupes unpatch", () => {
    const { inst, def, unpatchSpy } = mkInst();
    const mod: any = { Anthropic: {} };
    markManuallyInstrumented(inst, mod);

    def.unpatch(mod, "0.39.0");
    expect(unpatchSpy).toHaveBeenCalledTimes(1);
  });

  it("does not skip when nothing was marked (auto-discovery unchanged)", () => {
    const { def, patchSpy } = mkInst();
    const mod: any = { Anthropic: {} };
    def.patch(mod, "0.39.0");
    expect(patchSpy).toHaveBeenCalledTimes(1);
  });

  it("survives garbage input without throwing", () => {
    const { inst, def } = mkInst();
    expect(() => markManuallyInstrumented(null, {})).not.toThrow();
    expect(() => markManuallyInstrumented(inst, null)).not.toThrow();
    expect(() => markManuallyInstrumented(inst, 42)).not.toThrow();
    expect(() => markManuallyInstrumented(inst, "str")).not.toThrow();
    // Hostile `.default` getter on the module — both at mark and at hook time.
    const hostile: any = {};
    Object.defineProperty(hostile, "default", {
      get() {
        throw new Error("hostile getter");
      },
    });
    expect(() => markManuallyInstrumented(inst, hostile)).not.toThrow();
    expect(() => def.patch(hostile, "1.0.0")).not.toThrow();
    expect(def.patch(hostile, "1.0.0")).toBe(hostile); // patch spy returns exports
  });
});

describe("real AnthropicInstrumentation — double-patch layering (customer scenario)", () => {
  // Gated: the instrumentor is an optionalDependency and may be absent.
  let AnthropicInstrumentation: any;
  try {
    AnthropicInstrumentation = requireCjs(
      "@traceloop/instrumentation-anthropic",
    ).AnthropicInstrumentation;
  } catch {
    AnthropicInstrumentation = undefined;
  }
  const maybe = AnthropicInstrumentation ? it : it.skip;

  /** Fake @anthropic-ai/sdk exports shape traceloop's patch/manuallyInstrument walk. */
  const mkFakeAnthropicModule = () => {
    const realCreate = function create() {
      return "real-result";
    };
    const mk = () => ({ prototype: { create: realCreate } as any });
    const mod: any = {
      Anthropic: { Completions: mk(), Messages: mk(), Beta: { Messages: mk() } },
    };
    mod.default = mod.Anthropic; // mirror the SDK's default self-reference
    return { mod, realCreate };
  };

  maybe(
    "UNMARKED: loader hook re-patches a manually instrumented module (proves the bug)",
    () => {
      const inst = new AnthropicInstrumentation({ enabled: false });
      hardenInstrumentorPatches(inst, "anthropic");
      const { mod } = mkFakeAnthropicModule();

      inst.manuallyInstrument(mod);
      const afterManual = mod.Anthropic.Messages.prototype.create;
      expect((afterManual as any).__wrapped).toBe(true);

      // Simulate the enforcer wrapper installed on top (no __wrapped marker).
      const enforcerWrapper = function (this: any, ...args: any[]) {
        return afterManual.apply(this, args);
      };
      mod.Anthropic.Messages.prototype.create = enforcerWrapper;

      // No markManuallyInstrumented → the hook stacks a layer ABOVE the enforcer.
      (inst as any)._modules[0].patch(mod, "0.39.0");
      const top = mod.Anthropic.Messages.prototype.create;
      expect(top).not.toBe(enforcerWrapper);
      expect((top as any).__wrapped).toBe(true);
      expect((top as any).__original).toBe(enforcerWrapper);
    },
  );

  maybe(
    "MARKED: loader hook leaves the enforcer wrapper on top (fix)",
    () => {
      const inst = new AnthropicInstrumentation({ enabled: false });
      hardenInstrumentorPatches(inst, "anthropic");
      const { mod } = mkFakeAnthropicModule();

      inst.manuallyInstrument(mod);
      markManuallyInstrumented(inst, mod);
      const afterManual = mod.Anthropic.Messages.prototype.create;

      const enforcerWrapper = function (this: any, ...args: any[]) {
        return afterManual.apply(this, args);
      };
      mod.Anthropic.Messages.prototype.create = enforcerWrapper;

      // The app's post-init require() fires the hook with the SAME exports.
      const returned = (inst as any)._modules[0].patch(mod, "0.39.0");
      expect(returned).toBe(mod);
      // Enforcer wrapper is still the topmost layer → anthropicStreamBypass
      // keeps working for streamed calls.
      expect(mod.Anthropic.Messages.prototype.create).toBe(enforcerWrapper);
    },
  );
});
