/**
 * Anthropic in-band stream guard for NESTED/duplicate SDK copies.
 *
 * Residual of the manual-instrumentation dedupe fix (see
 * manualInstrumentDedupe.test.ts): that fix dedupes re-patch of the SAME
 * module identity only — deliberately, because DIFFERENT copies must still be
 * patched. But the armed Traceloop loader hook also patches nested/duplicate
 * copies of @anthropic-ai/sdk (e.g. under @langchain/anthropic when version
 * pins conflict and npm can't hoist). The enforcer only wraps the app's main
 * copy, so a nested IN-BAND copy (~0.28 → 0.41: patch succeeds, but the
 * streaming branch runs `new moduleExports.APIPromise(...)` against a root
 * export that doesn't exist) carries the Traceloop layer with no enforcer and
 * no anthropicStreamBypass above it — every streamed
 * messages.create({stream:true}) throws "APIPromise is not a constructor"
 * INTO CUSTOMER CODE.
 *
 * Fix: after the hardened loader hook lets the Traceloop patch succeed on an
 * in-band exports, installAnthropicInBandStreamGuard() wraps each patched
 * `create` with a shimmer-compatible guard that routes stream:true straight to
 * the REAL method (around the Traceloop layer) and everything else through the
 * Traceloop layer unchanged.
 */
import { describe, it, expect, vi } from "vitest";
import { createRequire } from "module";

import {
  hardenInstrumentorPatches,
  installAnthropicInBandStreamGuard,
  markManuallyInstrumented,
} from "../src/telemetry";

const requireCjs = createRequire(import.meta.url);

/** Minimal shimmer-alike: what OTel's wrap() leaves on a patched method. */
function shimmerWrap(proto: any, name: string, mkWrapper: (orig: any) => any) {
  const original = proto[name];
  const wrapped = mkWrapper(original);
  wrapped.__original = original;
  wrapped.__wrapped = true;
  wrapped.__unwrap = () => {
    if (proto[name] === wrapped) proto[name] = original;
  };
  proto[name] = wrapped;
  return wrapped;
}

/**
 * Fake IN-BAND @anthropic-ai/sdk exports (no root APIPromise) with a
 * traceloop-shaped patch already applied to the three `create` surfaces.
 * The fake Traceloop layer mimics the real one's failure mode: stream:true →
 * `new exports.APIPromise(...)` (TypeError — undefined), non-stream → tagged
 * pass-through so tests can tell which layer ran.
 */
function mkInBandPatchedModule() {
  const realCreate = async function create(_params: any) {
    return "real-result";
  };
  const mkSurface = () => ({ prototype: { create: realCreate } as any });
  const mod: any = {
    Anthropic: {
      Completions: mkSurface(),
      Messages: mkSurface(),
      Beta: { Messages: mkSurface() },
    },
    // NO root APIPromise — the in-band condition.
  };
  mod.default = mod.Anthropic;
  const traceloopCalls: string[] = [];
  const applyTraceloopLikePatch = (proto: any) =>
    shimmerWrap(proto, "create", (orig) => function method(this: any, ...args: any[]) {
      traceloopCalls.push(args?.[0]?.stream ? "stream" : "non-stream");
      if (args?.[0]?.stream) {
        // The real instrumentor's streaming branch, verbatim failure mode.
        return new (mod as any).APIPromise(this, orig.apply(this, args));
      }
      return orig.apply(this, args);
    });
  applyTraceloopLikePatch(mod.Anthropic.Completions.prototype);
  applyTraceloopLikePatch(mod.Anthropic.Messages.prototype);
  applyTraceloopLikePatch(mod.Anthropic.Beta.Messages.prototype);
  return { mod, realCreate, traceloopCalls };
}

describe("installAnthropicInBandStreamGuard — unit", () => {
  it("guards all three patched create surfaces on an in-band module", () => {
    const { mod, realCreate } = mkInBandPatchedModule();
    installAnthropicInBandStreamGuard(mod);
    for (const proto of [
      mod.Anthropic.Completions.prototype,
      mod.Anthropic.Messages.prototype,
      mod.Anthropic.Beta.Messages.prototype,
    ]) {
      expect(proto.create.name).toBe("tpAnthropicInBandStreamGuard");
      // shimmer-compatible: __original is the RAW method, one hop.
      expect(proto.create.__wrapped).toBe(true);
      expect(proto.create.__original).toBe(realCreate);
      expect(typeof proto.create.__unwrap).toBe("function");
    }
  });

  it("routes stream:true around the Traceloop layer to the real method", async () => {
    const { mod, traceloopCalls } = mkInBandPatchedModule();
    installAnthropicInBandStreamGuard(mod);
    const proto = mod.Anthropic.Messages.prototype;
    // Without the guard this is the customer-visible TypeError.
    await expect(proto.create({ stream: true })).resolves.toBe("real-result");
    expect(traceloopCalls).toEqual([]); // Traceloop layer never entered
  });

  it("keeps non-streaming calls flowing through the Traceloop layer", async () => {
    const { mod, traceloopCalls } = mkInBandPatchedModule();
    installAnthropicInBandStreamGuard(mod);
    const proto = mod.Anthropic.Messages.prototype;
    await expect(proto.create({ stream: false })).resolves.toBe("real-result");
    await expect(proto.create({})).resolves.toBe("real-result");
    expect(traceloopCalls).toEqual(["non-stream", "non-stream"]);
  });

  it("does NOT guard a safe module (root APIPromise export present)", () => {
    const { mod } = mkInBandPatchedModule();
    (mod as any).APIPromise = class APIPromise {};
    const before = mod.Anthropic.Messages.prototype.create;
    installAnthropicInBandStreamGuard(mod);
    expect(mod.Anthropic.Messages.prototype.create).toBe(before);
  });

  it("does NOT guard an unpatched (non-wrapped) create", () => {
    const raw = async function create() { return "raw"; };
    const mod: any = { Anthropic: { Messages: { prototype: { create: raw } } } };
    installAnthropicInBandStreamGuard(mod);
    expect(mod.Anthropic.Messages.prototype.create).toBe(raw);
  });

  it("is idempotent — a second install is a no-op", () => {
    const { mod } = mkInBandPatchedModule();
    installAnthropicInBandStreamGuard(mod);
    const first = mod.Anthropic.Messages.prototype.create;
    installAnthropicInBandStreamGuard(mod);
    expect(mod.Anthropic.Messages.prototype.create).toBe(first);
  });

  it("__unwrap restores the RAW method (full de-instrumentation in one hop)", () => {
    const { mod, realCreate } = mkInBandPatchedModule();
    installAnthropicInBandStreamGuard(mod);
    const proto = mod.Anthropic.Messages.prototype;
    proto.create.__unwrap();
    expect(proto.create).toBe(realCreate);
  });

  it("__unwrap is conditional — never clobbers a layer installed on top", () => {
    const { mod } = mkInBandPatchedModule();
    installAnthropicInBandStreamGuard(mod);
    const proto = mod.Anthropic.Messages.prototype;
    const guard = proto.create;
    const enforcerLike = function (this: any, ...args: any[]) {
      return guard.apply(this, args);
    };
    proto.create = enforcerLike;
    guard.__unwrap(); // stale unwrap of a displaced layer
    expect(proto.create).toBe(enforcerLike);
  });

  it("enforcer bypass semantics: one-level __original unwrap lands on the real method", () => {
    // The enforcer's anthropicStreamBypass does exactly this walk.
    const { mod, realCreate } = mkInBandPatchedModule();
    installAnthropicInBandStreamGuard(mod);
    const o: any = mod.Anthropic.Messages.prototype.create;
    const callee = o.__wrapped === true && typeof o.__original === "function" ? o.__original : o;
    expect(callee).toBe(realCreate);
  });

  it("survives garbage / hostile / frozen input without throwing", () => {
    expect(() => installAnthropicInBandStreamGuard(null)).not.toThrow();
    expect(() => installAnthropicInBandStreamGuard(undefined)).not.toThrow();
    expect(() => installAnthropicInBandStreamGuard(42)).not.toThrow();
    expect(() => installAnthropicInBandStreamGuard("str")).not.toThrow();
    expect(() => installAnthropicInBandStreamGuard({})).not.toThrow();

    // Hostile APIPromise getter → treated as not in-band, no throw.
    const hostileApi: any = { Anthropic: {} };
    Object.defineProperty(hostileApi, "APIPromise", {
      get() { throw new Error("hostile"); },
    });
    expect(() => installAnthropicInBandStreamGuard(hostileApi)).not.toThrow();

    // Hostile Anthropic getter → no throw.
    const hostileRoot: any = {};
    Object.defineProperty(hostileRoot, "Anthropic", {
      get() { throw new Error("hostile"); },
    });
    expect(() => installAnthropicInBandStreamGuard(hostileRoot)).not.toThrow();

    // Frozen prototype → assignment impossible, must not throw.
    const { mod } = mkInBandPatchedModule();
    Object.freeze(mod.Anthropic.Messages.prototype);
    expect(() => installAnthropicInBandStreamGuard(mod)).not.toThrow();
  });

  it("a sync throw from the real method propagates once — never re-dispatched via the patched layer", () => {
    const { mod, traceloopCalls } = mkInBandPatchedModule();
    const proto = mod.Anthropic.Messages.prototype;
    // Rebuild the surface with a real method that sync-throws (e.g. the SDK's
    // own validation), then re-apply the traceloop-shaped patch + guard.
    const syncThrower = vi.fn(function create() {
      throw new Error("sdk sync validation error");
    });
    proto.create = syncThrower;
    shimmerWrap(proto, "create", (orig) => function method(this: any, ...args: any[]) {
      traceloopCalls.push(args?.[0]?.stream ? "stream" : "non-stream");
      return orig.apply(this, args);
    });
    installAnthropicInBandStreamGuard(mod);
    expect(proto.create.name).toBe("tpAnthropicInBandStreamGuard");
    // The guard must let the SDK's own error propagate untouched, exactly
    // once, without falling back into the Traceloop layer (double invocation
    // would also resurrect the APIPromise crash on the retry).
    expect(() => proto.create({ stream: true })).toThrow("sdk sync validation error");
    expect(syncThrower).toHaveBeenCalledTimes(1);
    expect(traceloopCalls).toEqual([]);
  });

  it("hostile .stream getter on the request falls through to the patched layer", () => {
    const { mod, traceloopCalls } = mkInBandPatchedModule();
    installAnthropicInBandStreamGuard(mod);
    const proto = mod.Anthropic.Messages.prototype;
    const hostileReq: any = {};
    Object.defineProperty(hostileReq, "stream", {
      get() { throw new Error("hostile stream getter"); },
    });
    // Status quo: the Traceloop layer (and the raw SDK) read .stream too, so
    // the call still throws the getter's own error — from the PATCHED layer,
    // not from the guard (which must neither swallow nor transform it).
    expect(() => proto.create(hostileReq)).toThrow("hostile stream getter");
    expect(traceloopCalls).toEqual([]); // threw while entering the patched layer
  });
});

describe("hardened loader hook — guard install wiring", () => {
  const mkInst = (label = "anthropic", defName = "@anthropic-ai/sdk") => {
    const { mod, realCreate, traceloopCalls } = mkInBandPatchedModule();
    // patch spy re-applies nothing (module arrives pre-patched, mimicking the
    // real patch having run) — the wiring under test is the post-patch install.
    const patchSpy = vi.fn((exp: any) => exp);
    const def: any = { name: defName, patch: patchSpy, unpatch: vi.fn((e: any) => e) };
    const inst: any = { _modules: [def] };
    hardenInstrumentorPatches(inst, label);
    return { inst, def, patchSpy, mod, realCreate, traceloopCalls };
  };

  it("installs the guard after a successful anthropic loader-hook patch", () => {
    const { def, mod } = mkInst();
    def.patch(mod, "0.39.0");
    expect(mod.Anthropic.Messages.prototype.create.name).toBe(
      "tpAnthropicInBandStreamGuard",
    );
  });

  it("does not install for other instrumentor labels or module defs", () => {
    {
      const { def, mod } = mkInst("openAI", "@anthropic-ai/sdk");
      const before = mod.Anthropic.Messages.prototype.create;
      def.patch(mod, "0.39.0");
      expect(mod.Anthropic.Messages.prototype.create).toBe(before);
    }
    {
      const { def, mod } = mkInst("anthropic", "openai");
      const before = mod.Anthropic.Messages.prototype.create;
      def.patch(mod, "4.0.0");
      expect(mod.Anthropic.Messages.prototype.create).toBe(before);
    }
  });

  it("does not install when the patch itself throws (≤0.27-style failure)", () => {
    const { mod } = mkInBandPatchedModule();
    const def: any = {
      name: "@anthropic-ai/sdk",
      patch: () => { throw new Error("patch-time failure"); },
    };
    const inst: any = { _modules: [def] };
    hardenInstrumentorPatches(inst, "anthropic");
    const before = mod.Anthropic.Messages.prototype.create;
    expect(def.patch(mod, "0.27.0")).toBe(mod); // fail-open returns exports
    expect(mod.Anthropic.Messages.prototype.create).toBe(before);
  });

  it("dedupe short-circuit (manual mode main copy) skips the guard entirely", () => {
    const { inst, def, patchSpy, mod } = mkInst();
    markManuallyInstrumented(inst, mod);
    const before = mod.Anthropic.Messages.prototype.create;
    expect(def.patch(mod, "0.39.0")).toBe(mod);
    expect(patchSpy).not.toHaveBeenCalled();
    expect(mod.Anthropic.Messages.prototype.create).toBe(before);
  });
});

describe("real AnthropicInstrumentation — nested in-band copy (customer scenario)", () => {
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

  /** Fake in-band @anthropic-ai/sdk exports (NO root APIPromise). */
  const mkFakeInBandSdk = () => {
    const realCreate = async function create(_params: any) {
      return "real-result";
    };
    const mk = () => ({ prototype: { create: realCreate } as any });
    const mod: any = {
      Anthropic: { Completions: mk(), Messages: mk(), Beta: { Messages: mk() } },
    };
    mod.default = mod.Anthropic;
    return { mod, realCreate };
  };

  maybe("UNHARDENED: streamed create on a hook-patched in-band copy throws (proves the bug)", () => {
    const inst = new AnthropicInstrumentation({ enabled: false });
    const { mod } = mkFakeInBandSdk();
    // Raw loader-hook path, as it would fire for a nested copy.
    (inst as any)._modules[0].patch(mod, "0.39.0");
    expect(() =>
      mod.Anthropic.Messages.prototype.create.call(
        { _client: {} },
        { stream: true, model: "claude-3-5-sonnet-latest", messages: [] },
      ),
    ).toThrow(TypeError);
  });

  maybe("HARDENED: streamed create on the same topology resolves via the real method", async () => {
    const inst = new AnthropicInstrumentation({ enabled: false });
    hardenInstrumentorPatches(inst, "anthropic");
    const { mod } = mkFakeInBandSdk();
    (inst as any)._modules[0].patch(mod, "0.39.0");
    expect(mod.Anthropic.Messages.prototype.create.name).toBe(
      "tpAnthropicInBandStreamGuard",
    );
    await expect(
      mod.Anthropic.Messages.prototype.create.call(
        { _client: {} },
        { stream: true, model: "claude-3-5-sonnet-latest", messages: [] },
      ),
    ).resolves.toBe("real-result");
  });

  maybe("HARDENED: a SAFE copy (root APIPromise) is left exactly as Traceloop patched it", () => {
    const inst = new AnthropicInstrumentation({ enabled: false });
    hardenInstrumentorPatches(inst, "anthropic");
    const { mod } = mkFakeInBandSdk();
    (mod as any).APIPromise = class APIPromise {};
    (inst as any)._modules[0].patch(mod, "0.50.1");
    const top: any = mod.Anthropic.Messages.prototype.create;
    expect(top.name).not.toBe("tpAnthropicInBandStreamGuard");
    expect(top.__wrapped).toBe(true); // Traceloop layer on top, untouched
  });

  maybe("auto-mode main copy: eager manuallyInstrument displaces the guard (layering unchanged)", () => {
    // _applyAutoDiscoveredInstrumentations sequence: hook patch → guard →
    // manuallyInstrument. The node-platform _wrap is unwrap-first, so the
    // guard's shimmer-compatible __unwrap pops it and Traceloop re-wraps the
    // RAW method — today's exact main-copy layering, which the enforcer's
    // anthropicStreamBypass then covers.
    const inst = new AnthropicInstrumentation({ enabled: false });
    hardenInstrumentorPatches(inst, "anthropic");
    const { mod, realCreate } = mkFakeInBandSdk();
    (inst as any)._modules[0].patch(mod, "0.39.0");
    expect(mod.Anthropic.Messages.prototype.create.name).toBe(
      "tpAnthropicInBandStreamGuard",
    );
    inst.manuallyInstrument(mod);
    const top: any = mod.Anthropic.Messages.prototype.create;
    expect(top.name).not.toBe("tpAnthropicInBandStreamGuard");
    expect(top.__wrapped).toBe(true);
    expect(top.__original).toBe(realCreate); // single Traceloop layer over RAW
  });
});
