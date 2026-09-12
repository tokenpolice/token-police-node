/**
 * B1 — loader-hook fail-open.
 *
 * @traceloop/instrumentation-openai / -anthropic are InstrumentationBase
 * subclasses whose CONSTRUCTOR registers require-in-the-middle /
 * import-in-the-middle hooks. Those hooks fire LATER, inside the customer's own
 * `require('@anthropic-ai/sdk')`, and the whole chain is bare:
 *
 * customer require() → require-in-the-middle onrequire → RequireInTheMiddleSingleton
 * → InstrumentationBase._onRequire → `return module.patch(exports, version)`
 * → traceloop patch(), which unconditionally dereferences
 * `moduleExports.Anthropic.Completions.prototype` et al.
 *
 * If a future provider-SDK version changes shape, that TypeError propagates out
 * of the customer's require and crashes their app at boot — with nothing of ours
 * on the stack (tp.init() returned long ago).
 *
 * hardenInstrumentorPatches() wraps the live `_modules[].patch/.unpatch` (and
 * nested `files[]` entries) so a throw is swallowed and the ORIGINAL exports are
 * returned — the module is simply not instrumented, and the app lives.
 *
 * PRE-FIX EVIDENCE: the last test constructs an UNHARDENED AnthropicInstrumentation
 * and asserts its raw `_modules[0].patch({})` throws — the exact crash — while the
 * hardened twin returns the empty exports object.
 */
import { describe, it, expect, vi } from "vitest";
import { createRequire } from "module";

import { hardenInstrumentorPatches } from "../src/telemetry";

const requireCjs = createRequire(import.meta.url);

describe("hardenInstrumentorPatches — unit", () => {
  it("swallows a throwing patch and returns the exports unchanged", () => {
    const boom = () => {
      throw new TypeError("Cannot read properties of undefined (reading 'prototype')");
    };
    const inst: any = {
      _modules: [
        { name: "x", patch: boom, unpatch: boom, files: [{ patch: boom, unpatch: boom }] },
      ],
    };

    hardenInstrumentorPatches(inst, "test");

    const exportsObj = { some: "module" };
    expect(() => inst._modules[0].patch(exportsObj, "1.0.0")).not.toThrow();
    expect(inst._modules[0].patch(exportsObj, "1.0.0")).toBe(exportsObj);

    // Nested InstrumentationNodeModuleFile entries are on the same hook path.
    expect(() => inst._modules[0].files[0].patch(exportsObj, "1.0.0")).not.toThrow();
    expect(inst._modules[0].files[0].patch(exportsObj, "1.0.0")).toBe(exportsObj);
  });

  it("swallows a throwing unpatch (module + nested file)", () => {
    const boom = () => {
      throw new Error("unpatch exploded");
    };
    const inst: any = {
      _modules: [{ name: "x", patch: boom, unpatch: boom, files: [{ unpatch: boom }] }],
    };

    hardenInstrumentorPatches(inst, "test");

    const exportsObj = {};
    expect(() => inst._modules[0].unpatch(exportsObj, "1.0.0")).not.toThrow();
    expect(() => inst._modules[0].files[0].unpatch(exportsObj, "1.0.0")).not.toThrow();
  });

  it("passes a non-throwing patch's return value, args and `this` through untouched", () => {
    const patched = { patched: true };
    const spy = vi.fn(function (this: any) {
      return patched;
    });
    const def: any = { name: "x", patch: spy };
    const inst: any = { _modules: [def] };

    hardenInstrumentorPatches(inst, "test");

    const exportsObj = { raw: true };
    expect(def.patch(exportsObj, "1.2.3")).toBe(patched);
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy).toHaveBeenCalledWith(exportsObj, "1.2.3");
    // Upstream's `_onRequire` invokes patch as a method on the module def, so
    // `this` must still be the def object.
    expect(spy.mock.instances[0]).toBe(def);
  });

  it("is idempotent — hardening twice does not double-wrap", () => {
    const spy = vi.fn((exp: any) => exp);
    const def: any = { name: "x", patch: spy };
    const inst: any = { _modules: [def] };

    hardenInstrumentorPatches(inst, "test");
    const firstWrapper = def.patch;
    hardenInstrumentorPatches(inst, "test");

    expect(def.patch).toBe(firstWrapper);
    def.patch({}, "1.0.0");
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it("survives garbage input without throwing", () => {
    expect(() => hardenInstrumentorPatches(null, "test")).not.toThrow();
    expect(() => hardenInstrumentorPatches(undefined, "test")).not.toThrow();
    expect(() => hardenInstrumentorPatches({}, "test")).not.toThrow();
    expect(() => hardenInstrumentorPatches({ _modules: "nope" }, "test")).not.toThrow();
    expect(() => hardenInstrumentorPatches({ _modules: [null, 42, {}] }, "test")).not.toThrow();
    expect(() =>
      hardenInstrumentorPatches({ _modules: [{ patch: 1, files: "nope" }] }, "test"),
    ).not.toThrow();
  });
});

describe("hardenInstrumentorPatches — real AnthropicInstrumentation (B1 scenario)", () => {
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

  // `enabled: false` keeps the constructor from registering real global loader
  // hooks from inside the test process.
  maybe("unhardened patch() throws on a shape-drifted module (proves the bug)", () => {
    const raw = new AnthropicInstrumentation({ enabled: false });
    // Empty exports = exactly what a provider SDK looks like after a shape
    // change: traceloop dereferences moduleExports.Anthropic.Completions.prototype.
    expect(() => (raw as any)._modules[0].patch({})).toThrow();
  });

  maybe("hardened patch() returns the exports instead of throwing", () => {
    const inst = new AnthropicInstrumentation({ enabled: false });
    hardenInstrumentorPatches(inst, "anthropic");

    const exportsObj: any = {};
    let result: any;
    expect(() => {
      result = (inst as any)._modules[0].patch(exportsObj);
    }).not.toThrow();
    expect(result).toBe(exportsObj);
  });
});
