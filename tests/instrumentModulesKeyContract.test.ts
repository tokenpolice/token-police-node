/**
 * instrumentModules key-contract parity between the two instrumentation seams
 * (enforcer.ts pre-flight /check vs telemetry.ts Traceloop metering).
 *
 * `enforcer.ts` `autoInstrument()`'s prep loop lowercases every
 * `instrumentModules` key (`key.toLowerCase()`) — case-INsensitive, always has
 * been. `telemetry.ts` `_applyManualInstrumentations()` used to read
 * `modules[entry.moduleKey]` with EXACT case only, and the registry key for
 * OpenAI is `"openAI"` (camelCase). So `tp.init({ instrumentModules: { openai:
 * OpenAI } })` (all-lowercase — an easy, natural typo) made the enforcer patch
 * the prototypes (pre-flight `/check` fired, which also silenced client.ts's
 * zero-wrap audit, so nothing warned) while the Traceloop OpenAI instrumentor
 * was NEVER constructed. Auto-discovery could not cover for it either, since
 * `setupOpenTelemetry` only falls back to auto-discovery when
 * `instrumentModules` is EMPTY — and it was not. `OpenAI.Chat.Completions
 * .prototype.create` carries no `manualTelemetry` fallback, so ALL of OpenAI
 * metering hung off that missing span: an enforcing firewall producing ZERO
 * generation rows, silently.
 *
 * The fix, all in this diff:
 *  1. telemetry.ts: a new module-private `_readModuleEntry(modules, key)` —
 *     exact match first, then the first READABLE key whose `toLowerCase()`
 *     matches, in `Object.keys()` order; fully guarded, never throws. Used at
 *     `_applyManualInstrumentations` and `_applyLangChainInstrumentation`'s
 *     `userLangChain` read (which had the identical exact-case bug).
 *  2. client.ts: the zero-wrap warning's example now prints the CORRECT
 *     canonical `{ openAI: OpenAI }` spelling (it used to print the broken
 *     lowercase one — the exact typo that caused the bug).
 *  3. enforcer.ts `_warnInstallFailure()`: the module-name→key ternary (which
 *     had no branch at all for `together-ai` / `groq-sdk` / `@openrouter/sdk`
 *     / `@cerebras/cerebras_cloud_sdk`, falling through to the raw npm package
 *     name — itself NOT a valid `instrumentModules` key) became a
 *     module-scope `_INSTRUMENT_MODULE_KEYS` lookup, total over all 12 target
 *     module names, plus a drift guard: an unmapped module name drops the
 *     `pass instrumentModules: { … }` sentence instead of fabricating an
 *     invalid key.
 *  4. enforcer.ts `autoInstrument()`: a new `voyageai`/`voyage` prep-loop
 *     branch, with `_instrumentVoyage(userModule)` called BEFORE the two
 *     auto-discovery attempts — load-bearing, because `_wrapMethod` dedups on
 *     `voyageai:prototype:${method}` (no module identity), so
 *     whichever record is wrapped FIRST wins.
 *
 * HARNESS NOTES (mirrors tests/manualInstrumentShapeNormalization.test.ts,
 * tests/instrumentModulesClassNormalization.test.ts, tests/langchainManualOptIn
 * .test.ts, tests/enforcerResolution.test.ts, tests/enforcerZeroWrapWarning
 * .test.ts):
 *  - `@traceloop/instrumentation-openai` / `-anthropic` are REAL
 *    optionalDependencies, present in node_modules — Groups A/B/C drive them
 *    for real via `setupOpenTelemetry({ instrumentModules })`, spying on the
 *    shared `.prototype.manuallyInstrument` (constructing an instance directly
 *    ourselves would prove nothing about what THIS code path passes).
 *  - Group D (LangChain) and Group F (`_warnInstallFailure`) don't have a real
 *    package to spy on / a resolvable module to attach to, so they use the
 *    established synthetic-fixture techniques instead: a fake instrumentor
 *    package materialized on disk with `process.chdir()` into a temp app
 *    (Group D — copies `langchainManualOptIn.test.ts`'s `writeFakeInstrumentor`
 *    pattern, extended to record `manuallyInstrument()` call arguments), and
 *    the public `protect()` escape hatch to reach `_warnInstallFailure`
 *    (Group F — copies `syncWrapperNoEnforce.test.ts` / `apiSurfaceFixes
 *    .test.ts`'s usage).
 *  - Group E (voyage ordering) seeds the exported `__test__._resolvedModules`
 *    cache (`enforcerResolution.test.ts`'s technique) with a second, DISTINCT
 *    "auto-discovered" VoyageAIClient record instead of writing a fake
 *    `voyageai` package to disk — deterministic and synchronous,
 *    and it lets the test prove the actual claim (user copy wins, NOT the
 *    auto-discovered copy) rather than merely "the user copy is wrapped".
 *  - Every group resets the module-level once-guards between cases the same
 *    way the prior-art files do: `unsetupOpenTelemetry()` +
 *    `trace/context/propagation.disable()` for telemetry.ts state,
 *    `uninstrument()` + `setClient(undefined)` for enforcer.ts state.
 */
import {
  describe,
  it,
  test,
  expect,
  beforeAll,
  afterAll,
  beforeEach,
  afterEach,
  vi,
} from "vitest";
import { createRequire } from "module";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { trace, context, propagation } from "@opentelemetry/api";

import { setupOpenTelemetry, unsetupOpenTelemetry } from "../src/telemetry";
import { autoInstrument, uninstrument, protect, __test__ } from "../src/enforcer";
import { init } from "../src/client";
import { setClient } from "../src/state";

const { _resolvedModules } = __test__ as any;

const requireCjs = createRequire(import.meta.url);

// ── Real instrumentor classes (optionalDependencies — present in this repo's
//    node_modules; skip gracefully if a future change drops them) ──────────
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

function resetEnforcerGlobals(): void {
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
}

afterEach(() => {
  resetTelemetryGlobals();
  resetEnforcerGlobals();
  try {
    _resolvedModules.delete("voyageai");
  } catch {
    /* ignore */
  }
  vi.restoreAllMocks();
});

// ═══════════════════════════════════════════════════════════════════════
// Shared fixtures
// ═══════════════════════════════════════════════════════════════════════

/** OpenAI class-shaped fixture — statics carry the resource prototypes
 * @traceloop/instrumentation-openai wraps directly. Class-form input is
 * returned BY IDENTITY from `_pickOpenAIClass` (see
 * manualInstrumentShapeNormalization.test.ts case (7)), so `spy.mock
 * .calls[0][0]` can be compared `toBe` the exact fixture passed in. */
function makeOpenAIClassFixture(): { OpenAI: any; Completions: any; ChatCompletions: any } {
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

/** Anthropic NAMESPACE-shaped fixture (`{ Anthropic: cls, default: cls }`) —
 * already the shape `_pickAnthropicNamespace` wants, so it is returned BY
 * IDENTITY (case (2) in manualInstrumentShapeNormalization.test.ts) and still
 * genuinely goes THROUGH the normalizer on every call. Simpler than the
 * class-form fixture (no `BaseAnthropicLike` self-ref needed) because this
 * file isn't pinning shape normalization itself — that's PR #528's file. */
function makeAnthropicNamespaceFixture(): {
  ns: any;
  Anthropic: any;
  Completions: any;
  Messages: any;
  BetaMessages: any;
} {
  class Completions {
    create() {
      return "real-completions";
    }
  }
  class Messages {
    create() {
      return "real-messages";
    }
  }
  class BetaMessages {
    create() {
      return "real-beta-messages";
    }
  }
  function Anthropic(this: any) {}
  (Anthropic as any).Completions = Completions;
  (Anthropic as any).Messages = Messages;
  (Anthropic as any).Beta = { Messages: BetaMessages };
  const ns: any = { Anthropic, default: Anthropic };
  return { ns, Anthropic, Completions, Messages, BetaMessages };
}

// ═══════════════════════════════════════════════════════════════════════
// Group A — the bug itself.
// ═══════════════════════════════════════════════════════════════════════
describe("Group A — recognition parity (the bug itself)", () => {
  maybeOpenAI(
    "(1) THE BUG: instrumentModules: { openai: Fake } (lowercase) DOES construct/apply the OpenAI instrumentor",
    () => {
      const { OpenAI, Completions, ChatCompletions } = makeOpenAIClassFixture();
      const spy = vi.spyOn(OpenAIInstrumentation.prototype, "manuallyInstrument");

      // WHY THIS GENUINELY FAILS PRE-FIX: pre-fix, `_applyManualInstrumentations`
      // reads `modules[entry.moduleKey]` = `modules["openAI"]` with EXACT case.
      // This fixture only carries a lowercase "openai" own key — "openAI" is
      // absent — so `userModule` is `undefined` and the loop `continue`s before
      // ever resolving an instrumentor package, constructing an instance, or
      // calling `manuallyInstrument`. `spy` would never fire and neither
      // resource prototype would ever be wrapped.
      expect(() => setupOpenTelemetry({ openai: OpenAI } as any)).not.toThrow();

      expect(spy).toHaveBeenCalledTimes(1);
      expect(spy.mock.calls[0][0]).toBe(OpenAI);
      expect((ChatCompletions.prototype as any).create.__wrapped).toBe(true);
      expect((Completions.prototype as any).create.__wrapped).toBe(true);
    },
  );

  maybeOpenAI(
    "(2) control: instrumentModules: { openAI: Fake } (canonical) still works — proves the harness isn't vacuous",
    () => {
      const { OpenAI, Completions, ChatCompletions } = makeOpenAIClassFixture();
      const spy = vi.spyOn(OpenAIInstrumentation.prototype, "manuallyInstrument");

      // This is a POSITIVE control, not a "fails pre-fix" case — the exact-case
      // key already worked before this diff. Its job is to prove test (1)'s
      // machinery (spy + wrap assertions) genuinely detects success when the
      // key IS right, so (1)'s failure really is about the lowercase spelling
      // and not a broken harness.
      expect(() => setupOpenTelemetry({ openAI: OpenAI } as any)).not.toThrow();

      expect(spy).toHaveBeenCalledTimes(1);
      expect(spy.mock.calls[0][0]).toBe(OpenAI);
      expect((ChatCompletions.prototype as any).create.__wrapped).toBe(true);
      expect((Completions.prototype as any).create.__wrapped).toBe(true);
    },
  );

  maybeAnthropic(
    "(3) instrumentModules: { Anthropic: Fake } (capitalized; registry key is lowercase `anthropic`) reaches the instrumentor via _pickAnthropicNamespace",
    () => {
      const { ns, Completions, Messages, BetaMessages } =
        makeAnthropicNamespaceFixture();
      const spy = vi.spyOn(AnthropicInstrumentation.prototype, "manuallyInstrument");

      // WHY THIS GENUINELY FAILS PRE-FIX: the registry's anthropic entry key is
      // the all-lowercase "anthropic". Pre-fix, `modules["anthropic"]` on an
      // object whose only own key is "Anthropic" (capital A) is `undefined` —
      // exact case, no fallback — so this entry is skipped exactly like (1).
      expect(() => setupOpenTelemetry({ Anthropic: ns } as any)).not.toThrow();

      expect(spy).toHaveBeenCalledTimes(1);
      // `ns` is already namespace-shaped, so `_pickAnthropicNamespace` returns
      // it BY IDENTITY (see the fixture comment above) — this also confirms
      // the value genuinely passed THROUGH the normalizer, not around it.
      expect(spy.mock.calls[0][0]).toBe(ns);
      expect((Completions.prototype as any).create.__wrapped).toBe(true);
      expect((Messages.prototype as any).create.__wrapped).toBe(true);
      expect((BetaMessages.prototype as any).create.__wrapped).toBe(true);
    },
  );
});

// ═══════════════════════════════════════════════════════════════════════
// Group B — precedence (documented in `_readModuleEntry`'s docstring, not
// otherwise enforced by any type/lint check).
// ═══════════════════════════════════════════════════════════════════════
describe("Group B — precedence", () => {
  maybeOpenAI(
    "(4) { openAI: A, openai: B } → EXACTLY ONE instrumentation, and it uses A (exact wins, never double-instruments)",
    () => {
      const a = makeOpenAIClassFixture();
      const b = makeOpenAIClassFixture();
      const spy = vi.spyOn(OpenAIInstrumentation.prototype, "manuallyInstrument");

      // NOTE ON MUTATION-TESTING THIS CASE: unlike every other test in this
      // file, this one does NOT fail under a straight revert of this diff.
      // Pre-fix, `_applyManualInstrumentations` reads ONLY the exact key
      // "openAI" (there is no second registry entry for "openai" — it's the
      // SAME provider, ONE registry row, ONE read) — so pre-fix already
      // resolves to `a` and calls manuallyInstrument exactly once, for the
      // same reason post-fix does (exact match, first thing `_readModuleEntry`
      // tries). What this test protects is the NEW `_readModuleEntry`'s
      // documented contract — "exact match first, and no scanning past it" —
      // from a plausible-but-wrong FUTURE implementation (e.g. one that
      // iterates every case-insensitively-matching key and calls
      // manuallyInstrument for each), which the file-level docstring alone
      // does nothing to prevent today. Kept because Group B is explicitly
      // asked for as "precedence, which only a comment currently protects" —
      // see this file's report for the full mutation-testing caveat.
      expect(() => setupOpenTelemetry({ openAI: a.OpenAI, openai: b.OpenAI } as any)).not.toThrow();

      expect(spy).toHaveBeenCalledTimes(1);
      expect(spy.mock.calls[0][0]).toBe(a.OpenAI);
      expect((a.ChatCompletions.prototype as any).create.__wrapped).toBe(true);
      // b's prototypes are untouched — proves it was never even attempted.
      expect((b.ChatCompletions.prototype as any).create.__wrapped).toBeUndefined();
    },
  );

  maybeOpenAI(
    "(5) { openAI: undefined, openai: B } → resolves to B (present-but-undefined deliberately falls through)",
    () => {
      const b = makeOpenAIClassFixture();
      const spy = vi.spyOn(OpenAIInstrumentation.prototype, "manuallyInstrument");

      // WHY THIS GENUINELY FAILS PRE-FIX: pre-fix, `modules["openAI"]` is
      // explicitly `undefined` (the key IS present, just with an undefined
      // value) → `if (!userModule) continue;` skips the entry immediately —
      // pre-fix has no concept of falling through to a DIFFERENT key at all,
      // so "openai" (with a real value) is never even looked at. `spy` would
      // never fire and both prototypes stay unwrapped.
      expect(() =>
        setupOpenTelemetry({ openAI: undefined, openai: b.OpenAI } as any),
      ).not.toThrow();

      expect(spy).toHaveBeenCalledTimes(1);
      expect(spy.mock.calls[0][0]).toBe(b.OpenAI);
      expect((b.ChatCompletions.prototype as any).create.__wrapped).toBe(true);
    },
  );
});

// ═══════════════════════════════════════════════════════════════════════
// Group C — golden rule / hostile input. `_readModuleEntry` must never throw
// into the customer's init(), and (where the guarded shape is otherwise
// readable) must not lose a resolvable value to a hostile sibling key.
// ═══════════════════════════════════════════════════════════════════════
describe("Group C — golden rule / hostile input", () => {
  maybeOpenAI(
    "(6) getter throws on a case-variant, but a DIFFERENT readable case-variant is present → still resolves it (pins revision 2)",
    () => {
      const fake = makeOpenAIClassFixture();
      const hostile: any = {
        get OpenAI(): any {
          throw new Error("boom: hostile OpenAI getter");
        },
        openai: fake.OpenAI,
      };
      const spy = vi.spyOn(OpenAIInstrumentation.prototype, "manuallyInstrument");

      // WHY THIS GENUINELY FAILS PRE-FIX: pre-fix reads `modules["openAI"]"`
      // directly — that exact key is absent here (only "OpenAI" and "openai"
      // exist), so the read is simply `undefined`, no throw, no instrumentor.
      // This ALSO pins a specific intermediate revision of THIS fix: an
      // earlier draft's scan loop `return`ed as soon as it hit a throwing
      // getter instead of `continue`ing past it — that draft would throw here
      // (or silently stop, per its own early-return) and never reach the
      // perfectly-readable "openai" key sitting right next to it. The shipped
      // fix's scan treats a hostile key as merely absent and keeps scanning.
      //
      // IMPORTANT: pass `hostile` DIRECTLY, never `{ ...hostile }` — an object
      // spread eagerly reads every own enumerable property (invoking the
      // `OpenAI` getter immediately, at spread time) which would throw before
      // `setupOpenTelemetry` is even called and defeat the whole point of this
      // fixture. `_readModuleEntry`'s lazy, per-key guarded access is exactly
      // what's under test.
      expect(() => setupOpenTelemetry(hostile as any)).not.toThrow();

      expect(spy).toHaveBeenCalledTimes(1);
      expect(spy.mock.calls[0][0]).toBe(fake.OpenAI);
      expect((fake.ChatCompletions.prototype as any).create.__wrapped).toBe(true);
    },
  );

  it("(7) getter throws on the canonical key with NO alternative → degrades to not-instrumented, no throw", () => {
    const hostile: any = {
      get openAI(): any {
        throw new Error("boom: hostile canonical getter");
      },
    };

    // WHY THIS GENUINELY FAILS PRE-FIX: pre-fix, `_applyManualInstrumentations`
    // reads `modules[entry.moduleKey]` OUTSIDE its own per-entry try/catch (the
    // try starts only after `if (!userModule) continue;`), so a throwing
    // getter on the exact key escapes the whole function uncaught — and
    // `setupOpenTelemetry` calls `_applyManualInstrumentations` with no
    // try/catch of its own either. This assertion would fail with the getter's
    // own Error, not resolve gracefully.
    expect(() => setupOpenTelemetry(hostile)).not.toThrow();
    // No package could have been resolved for a value that was never read
    // successfully — nothing to assert wrapped; "did not throw" IS the
    // assertion for this shape.
  });

  maybeOpenAI(
    "(8a) frozen instrumentModules object, lowercase key → still resolves via the scan (Object.keys works fine on frozen objects)",
    () => {
      const fake = makeOpenAIClassFixture();
      const modules = Object.freeze({ openai: fake.OpenAI });
      const spy = vi.spyOn(OpenAIInstrumentation.prototype, "manuallyInstrument");

      // WHY THIS GENUINELY FAILS PRE-FIX: same root cause as test (1) — the
      // exact key "openAI" is absent (only frozen "openai" exists) — freezing
      // is orthogonal; pre-fix skips this because of the CASE, not the freeze.
      expect(() => setupOpenTelemetry(modules as any)).not.toThrow();
      expect(spy).toHaveBeenCalledTimes(1);
      expect(spy.mock.calls[0][0]).toBe(fake.OpenAI);
    },
  );

  maybeOpenAI(
    "(8b) null-prototype instrumentModules object (Object.create(null)), lowercase key → still resolves via the scan",
    () => {
      const fake = makeOpenAIClassFixture();
      const modules: any = Object.create(null);
      modules.openai = fake.OpenAI;
      const spy = vi.spyOn(OpenAIInstrumentation.prototype, "manuallyInstrument");

      // WHY THIS GENUINELY FAILS PRE-FIX: same root cause as test (1) — the
      // exact key "openAI" is absent. `_readModuleEntry`'s guard
      // (`typeof modules !== "object" && typeof modules !== "function"`)
      // exists so a null-prototype object (no inherited `hasOwnProperty` /
      // `toString` / etc.) is still accepted as a plain object rather than
      // rejected outright — this is what proves that guard doesn't over-fire.
      expect(() => setupOpenTelemetry(modules)).not.toThrow();
      expect(spy).toHaveBeenCalledTimes(1);
      expect(spy.mock.calls[0][0]).toBe(fake.OpenAI);
    },
  );

  maybeOpenAI(
    "(8c) instrumentModules as an array with an attached lowercase key → still resolves via the scan",
    () => {
      const fake = makeOpenAIClassFixture();
      // Arrays are objects and accept arbitrary own string keys alongside
      // their indices — `Object.keys` walks both. An exotic shape, but
      // `typeof [] === "object"`, so `_readModuleEntry`'s type guard accepts
      // it same as any other object.
      const modules: any = [];
      modules.openai = fake.OpenAI;
      const spy = vi.spyOn(OpenAIInstrumentation.prototype, "manuallyInstrument");

      // WHY THIS GENUINELY FAILS PRE-FIX: same root cause as test (1) — the
      // exact key "openAI" is absent on this array.
      expect(() => setupOpenTelemetry(modules)).not.toThrow();
      expect(spy).toHaveBeenCalledTimes(1);
      expect(spy.mock.calls[0][0]).toBe(fake.OpenAI);
    },
  );

  describe("(8d)/(8e) — pure golden-rule coverage: NOT fix-specific, does not discriminate pre/post-fix", () => {
    // These two do NOT fail under a revert of this diff, and are included only
    // because Group C's spec explicitly asked for revoked-Proxy / primitive /
    // array coverage via init(). Full reasoning in this file's report:
    // `init()` wraps BOTH `setupOpenTelemetry(...)` and `autoInstrument(...)`
    // each in their OWN try/catch (src/client.ts, unrelated to this diff,
    // pre-existing) — so ANY throw from anywhere inside either, for ANY
    // reason, was already silently swallowed before this fix existed. Worse,
    // for a revoked Proxy specifically, `setupOpenTelemetry`'s OWN dispatch
    // check (`Object.keys(instrumentModules).length > 0`, also pre-existing
    // and untouched by this diff) throws before `_applyManualInstrumentations`
    // (this fix's code) is ever reached — so this shape can't even be routed
    // through this fix's code to prove anything about it. Kept as honest
    // defense-in-depth coverage, not a regression pin.
    it("(8d) revoked Proxy as instrumentModules → init() never throws", () => {
      const { proxy, revoke } = Proxy.revocable({}, {});
      revoke();
      expect(() =>
        init({
          apiKey: "tp_sk_test",
          deployment: "serverless",
          firewall: "dry_run",
          instrumentModules: proxy,
        } as any),
      ).not.toThrow();
    });

    it("(8e) a primitive (number) as instrumentModules → init() never throws, and nothing is instrumented", () => {
      expect(() =>
        init({
          apiKey: "tp_sk_test",
          deployment: "serverless",
          firewall: "dry_run",
          instrumentModules: 42,
        } as any),
      ).not.toThrow();
    });
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Group D — the LangChain `userLangChain` read had the SAME exact-case bug,
// one level down from the (already case-insensitive) opt-in gate.
// ═══════════════════════════════════════════════════════════════════════
describe("Group D — LangChain userLangChain read (case parity with the opt-in gate)", () => {
  const FLAG = "__tpKeyContractLcFlag";
  const CALLS_KEY = "__tpKeyContractLcCalls";

  /** Fake @traceloop/instrumentation-langchain that records every
   * manuallyInstrument() argument by reference, so the test can prove WHICH
   * module object actually reached it (identity, not just "something did"). */
  function writeRecordingFakeInstrumentor(appDir: string): void {
    const dir = join(appDir, "node_modules", "@traceloop", "instrumentation-langchain");
    mkdirSync(dir, { recursive: true });
    writeFileSync(
      join(dir, "package.json"),
      JSON.stringify({
        name: "@traceloop/instrumentation-langchain",
        version: "0.0.0",
        main: "index.js",
      }),
    );
    writeFileSync(
      join(dir, "index.js"),
      `const FLAG = ${JSON.stringify(FLAG)};
const CALLS_KEY = ${JSON.stringify(CALLS_KEY)};
class LangChainInstrumentation {
  constructor() {
    globalThis[FLAG] = (globalThis[FLAG] || 0) + 1;
  }
  setTracerProvider() {}
  enable() {}
  disable() {}
  manuallyInstrument(arg) {
    (globalThis[CALLS_KEY] = globalThis[CALLS_KEY] || []).push(arg);
  }
}
module.exports = { LangChainInstrumentation };
`,
    );
  }

  let tmpRoot: string;
  let appDir: string;
  let originalCwd: string;

  beforeAll(() => {
    originalCwd = process.cwd();
    tmpRoot = mkdtempSync(join(tmpdir(), "tp-lc-key-contract-"));
    appDir = join(tmpRoot, "app");
    mkdirSync(appDir, { recursive: true });
    writeRecordingFakeInstrumentor(appDir);
  });

  afterAll(() => {
    try {
      process.chdir(originalCwd);
    } catch {
      /* ignore */
    }
    try {
      rmSync(tmpRoot, { recursive: true, force: true });
    } catch {
      /* ignore */
    }
  });

  beforeEach(() => {
    delete (globalThis as any)[FLAG];
    delete (globalThis as any)[CALLS_KEY];
  });

  afterEach(() => {
    try {
      process.chdir(originalCwd);
    } catch {
      /* ignore */
    }
  });

  it("(9a) THE BUG: lowercase `langchain` key reaches the direct user-module patch path (not just the opt-in gate)", () => {
    process.chdir(appDir);
    const sentinel: any = { CallbackManager: class {} };

    // WHY THIS GENUINELY FAILS PRE-FIX: the opt-in GATE above this read
    // ALREADY matched case-insensitively before this diff (that's covered by
    // tests/langchainManualOptIn.test.ts — a lowercase `langchain` key was
    // never blocked from constructing the raw instrumentor). What was broken
    // is ONE LEVEL DEEPER: pre-fix, `userLangChain = (instrumentModules as
    // any)?.langChain` read the EXACT-case "langChain" key — absent here (only
    // "langchain" exists) — so `userLangChain` was `undefined` and the whole
    // `if (userLangChain) {...}` direct-patch block never ran. The app passed
    // the gate and then silently fell through to the require()/import()
    // fallbacks — both of which fail in this fixture app (no `@langchain/core`
    // installed), so pre-fix `CALLS_KEY` stays completely empty: `sentinel` is
    // NEVER passed to `manuallyInstrument` by ANY path.
    setupOpenTelemetry({ langchain: { callbackManagerModule: sentinel } } as any);

    const calls: any[] = (globalThis as any)[CALLS_KEY] || [];
    expect(calls.some((c) => c?.callbackManagerModule === sentinel)).toBe(true);
  });

  it("(9b) control: canonical `langChain` key still reaches the direct user-module patch path", () => {
    process.chdir(appDir);
    const sentinel: any = { CallbackManager: class {} };

    // Positive control (not a "fails pre-fix" case — canonical case already
    // worked): proves the recording fixture and assertion genuinely detect
    // success, so (9a)'s failure really is about the lowercase spelling.
    setupOpenTelemetry({ langChain: { callbackManagerModule: sentinel } } as any);

    const calls: any[] = (globalThis as any)[CALLS_KEY] || [];
    expect(calls.some((c) => c?.callbackManagerModule === sentinel)).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Group E — voyage ordering: the user-supplied copy must win over an
// auto-discovered copy (dedup key carries no module identity).
// ═══════════════════════════════════════════════════════════════════════
describe("Group E — voyage ordering (autoInstrument, src/enforcer.ts)", () => {
  function makeVoyageModule(): { mod: any; VoyageAIClient: any } {
    function VoyageAIClient(this: any) {}
    (VoyageAIClient as any).prototype.embed = async function embed() {
      return { data: [] };
    };
    (VoyageAIClient as any).prototype.multimodalEmbed = async function multimodalEmbed() {
      return { data: [] };
    };
    return { mod: { VoyageAIClient }, VoyageAIClient };
  }

  it("(10) autoInstrument({ voyageai: userCopy }) wraps the USER's copy, not a distinct auto-discovered copy — user copy goes first", () => {
    const user = makeVoyageModule();
    const autoDiscovered = makeVoyageModule();
    const origUserEmbed = user.VoyageAIClient.prototype.embed;
    const origUserMultimodal = user.VoyageAIClient.prototype.multimodalEmbed;
    const origAutoEmbed = autoDiscovered.VoyageAIClient.prototype.embed;
    const origAutoMultimodal = autoDiscovered.VoyageAIClient.prototype.multimodalEmbed;

    // Seeds resolveProviderModule("voyageai") to return a SECOND,
    // DISTINCT module record — simulating the real production hazard the
    // source comment names: "the app's own imported copy is the one its calls
    // actually run through, and ... the resolveProviderModule() / import()
    // copies below can be a different record entirely." (mirrors
    // tests/enforcerResolution.test.ts's `_resolvedModules.set` technique;
    // cleared in this file's top-level afterEach).
    _resolvedModules.set("voyageai", autoDiscovered.mod);

    // WHY THIS GENUINELY FAILS PRE-FIX: pre-fix, `autoInstrument`'s prep loop
    // had NO branch for "voyageai"/"voyage" at all — `instrumentModules
    // .voyageai` was silently ignored entirely. Only the auto-discovery
    // attempts ran, and `resolveProviderModule("voyageai")` returns
    // OUR SEEDED `autoDiscovered.mod` (not `user.mod`, which pre-fix is never
    // even looked at) — so pre-fix wraps `autoDiscovered`'s prototype methods
    // and leaves `user`'s completely untouched. Post-fix, `_instrumentVoyage
    // (voyageModule)` for the user copy runs FIRST, and `_wrapMethod`'s
    // identity-agnostic dedup key (`voyageai:prototype:${method}`)
    // means whichever copy is wrapped first wins — the auto-discovered copy is
    // silently skipped.
    autoInstrument({ voyageai: user.mod } as any);

    expect(user.VoyageAIClient.prototype.embed).not.toBe(origUserEmbed);
    expect(user.VoyageAIClient.prototype.multimodalEmbed).not.toBe(origUserMultimodal);
    // The auto-discovered copy must be untouched — dedup skipped it.
    expect(autoDiscovered.VoyageAIClient.prototype.embed).toBe(origAutoEmbed);
    expect(autoDiscovered.VoyageAIClient.prototype.multimodalEmbed).toBe(origAutoMultimodal);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Group F — `_warnInstallFailure`'s module-name→key map, reached via the
// public `protect()` escape hatch (mirrors tests/syncWrapperNoEnforce.test.ts
// / tests/apiSurfaceFixes.test.ts's usage of `protect()`).
// ═══════════════════════════════════════════════════════════════════════
describe("Group F — _warnInstallFailure key map", () => {
  function warnMessages(warnSpy: any): string[] {
    return warnSpy.mock.calls.map((c: any[]) => String(c[0]));
  }

  function initWithLogErrors(): void {
    init({
      apiKey: "tp_sk_test",
      deployment: "serverless",
      firewall: "dry_run",
      logErrors: true,
    } as any);
  }

  test.each([
    ["together-ai", "together"],
    ["groq-sdk", "groq"],
    ["@openrouter/sdk", "openRouter"],
    ["@cerebras/cerebras_cloud_sdk", "cerebras"],
  ])(
    "(11) %s → the warning names the real InstrumentModules key `%s`, not the raw package name",
    (moduleName, expectedKey) => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      initWithLogErrors();

      // `{}` has no "create" method and `_resolvePath({}, ["prototype"])` is
      // `undefined` — both trigger `_warnInstallFailure` via `_wrapMethod`'s
      // install-failure branches (see enforcer.ts ~1826-1836). This is the
      // FIRST and only wrap attempt for this exact moduleName+path+method
      // combo, so there is no dedup contention.
      protect(moduleName, ["prototype"], "create", true, { module: {} });

      // Filter on the FULL target (module + objectPath + method), not just the
      // module name: `initWithLogErrors()` above ran a real autoInstrument()
      // pass, and some of these providers ARE installed as devDependencies now
      // (the realProviderSeams.* suites load them for real). Auto-discovery can
      // legitimately warn about a DIFFERENT target on the same module — e.g.
      // `together-ai Together.Images.prototype.create` — and a module-name-only
      // filter would count those as ours.
      const relevant = warnMessages(warn).filter((m) =>
        m.includes(`Could not install wrapper for ${moduleName} prototype.create `),
      );
      expect(relevant).toHaveLength(1);

      // WHY THIS GENUINELY FAILS PRE-FIX: pre-fix, the moduleKey ternary had
      // NO branch for any of these four npm names — it fell through to
      // `: target.moduleName`, i.e. the RAW PACKAGE NAME itself, which is NOT
      // a valid `InstrumentModules` field. Pre-fix this message reads
      // `instrumentModules: { ${moduleName}: ... }` — literally
      // `instrumentModules: { together-ai: ... }` — a key that TypeScript
      // rejects and the enforcer's own prep loop would silently ignore if
      // pasted verbatim, landing the customer right back in the
      // enforced-but-unmetered state this diagnostic exists to prevent.
      expect(relevant[0]).toContain(`instrumentModules: { ${expectedKey}: `);
      expect(relevant[0]).not.toContain(`instrumentModules: { ${moduleName}: `);
    },
  );

  it("(12) drift guard: an UNMAPPED module name does not fabricate an instrumentModules key", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    initWithLogErrors();

    const moduleName = "some-future-provider-sdk-not-yet-wired-up";
    protect(moduleName, ["prototype"], "create", true, { module: {} });

    const relevant = warnMessages(warn).filter((m) =>
      m.includes(`Could not install wrapper for ${moduleName} `),
    );
    expect(relevant).toHaveLength(1);

    // WHY THIS GENUINELY FAILS PRE-FIX: pre-fix, `target.moduleName === "openai"
    // ? "openAI" : ... : target.moduleName` — the FINAL fallback of that
    // ternary chain is `target.moduleName` itself, UNCONDITIONALLY, for ANY
    // string `protect()` was ever called with (the public API forwards an
    // arbitrary `moduleName`; the doc example even uses `'my-custom-llm'`).
    // So pre-fix this message reads `instrumentModules: { some-future-
    // provider-sdk-not-yet-wired-up: ... }` — a fabricated, always-invalid key
    // for ANY unmapped name. Post-fix, an unmapped name yields `moduleKey =
    // undefined` and the generic remedy sentence (no "instrumentModules: {"
    // substring at all) — less specific, never wrong.
    expect(relevant[0]).not.toContain("instrumentModules: {");
    expect(relevant[0]).toContain("pass the imported module to tp.init() via instrumentModules");
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Group G — _INSTRUMENT_MODULE_KEYS round-trip contract (totality +
// consumption). Tests 11/12 above pin what `_warnInstallFailure` PRINTS —
// but the half of the contract that actually broke in the field is the
// OTHER half: a suggested key the `autoInstrument()` prep loop then
// silently ignores. That is precisely how the lowercase `openai` form
// reached a fleet app — the printed text looked plausible and nothing
// consumed it. A warning-text-only assertion can't catch that drift.
//
// This closes both halves for EVERY reachable provider at once:
//   (i)  TOTALITY   — does `_INSTRUMENT_MODULE_KEYS` have a row for this
//        `_TARGET_METHODS` / internal-`_wrapMethod` moduleName at all?
//        Observed via the warning text (never by reading source).
//   (ii) CONSUMPTION — feed the EXACT discovered key back in as a real
//        `instrumentModules` key and confirm `autoInstrument()`'s prep loop
//        genuinely reads it (a prototype actually gets wrapped).
// Because (ii) uses the key DISCOVERED in (i) rather than a hardcoded
// expectation, this fails loudly in both directions a future change could
// drift: a new `_TARGET_METHODS`/`_wrapMethod` moduleName with no map row
// (match is null in (i)), or a map row whose value no `normalizedKey`
// branch in the prep loop consumes (the fixture is never wrapped in (ii)).
//
// The 11-name reachable set below was enumerated by grepping every
// `moduleName:` literal reaching `_TARGET_METHODS` or an internal
// `_wrapMethod(...)` call site in src/enforcer.ts (`_instrumentOpenRouter`,
// `_instrumentMistral`, `_instrumentCohere`, `_instrumentVoyage`) — the same
// universe `_INSTRUMENT_MODULE_KEYS`'s own header comment claims to be
// "Total over". `@huggingface/inference` is DELIBERATELY excluded: the map
// carries a 12th row for it, but `_instrumentHuggingFace` patches
// require-cache submodule exports directly and never calls `_wrapMethod` —
// so `_warnInstallFailure` (and thus this round-trip) can never observe it.
// Testing totality in the "map covers every reachable target" direction
// only (not the reverse) is deliberate — asserting the reverse would fail
// on this exact, intentional row.
//
// NOTE ON MUTATION-TESTING THIS GROUP (same honesty as Group B's test (4)):
// only 5 of these 11 rows fail under a straight revert of THIS diff —
// together-ai / groq-sdk / @openrouter/sdk / @cerebras/cerebras_cloud_sdk
// via part (i) (the pre-fix ternary had no branch at all for these four,
// exactly the set test (11) above already pins), and voyageai via
// part (ii) (the pre-fix prep loop had no "voyageai"/"voyage" branch, so a
// user-supplied module was silently ignored even though part (i) already
// found the right key). The other 6 rows (openai, @anthropic-ai/sdk,
// @google/genai, @aws-sdk/client-bedrock-runtime, cohere-ai,
// @mistralai/mistralai) were already fully correct on BOTH sides — pre-fix
// ternary branch AND prep-loop branch — before this diff, so those 6 do not
// discriminate pre/post-fix. They are included anyway because the ask here
// is a general-purpose drift detector (the reviewer's own framing: "fails
// loudly when a FUTURE provider is added..."), and a totality/consumption
// check that only covered the 5 currently-broken rows would silently stop
// protecting the other 6 the moment someone edits their entries.
describe("Group G — _INSTRUMENT_MODULE_KEYS round-trip contract (totality + consumption)", () => {
  function makeOpenAIRoundTripFixture() {
    const { OpenAI, ChatCompletions } = makeOpenAIClassFixture();
    return { mod: OpenAI, proto: ChatCompletions.prototype, method: "create" as const };
  }

  function makeAnthropicRoundTripFixture() {
    const { ns, Messages } = makeAnthropicNamespaceFixture();
    return { mod: ns, proto: Messages.prototype, method: "create" as const };
  }

  /** cohere-ai — CohereClientV2.clientV2 (instance-bound) exposes the
   * prototype _instrumentCohere actually patches; see _instrumentCohere. */
  function makeCohereRoundTripFixture() {
    class V2Client {
      chat() {
        return Promise.resolve({});
      }
      chatStream() {
        return {};
      }
    }
    function CohereClientV2(this: any, _opts?: any) {
      this.clientV2 = new V2Client();
    }
    return { mod: { CohereClientV2 }, proto: V2Client.prototype, method: "chat" as const };
  }

  /** @mistralai/mistralai — Chat is reached via a throw-away Mistral probe
   * instance's `.chat` property; see _instrumentMistral. */
  function makeMistralRoundTripFixture() {
    class Chat {
      complete() {
        return Promise.resolve({});
      }
      stream() {
        return {};
      }
    }
    function Mistral(this: any, _opts?: any) {
      this.chat = new Chat();
    }
    return { mod: { Mistral }, proto: Chat.prototype, method: "complete" as const };
  }

  /** voyageai — mirrors Group E's fixture shape. */
  function makeVoyageRoundTripFixture() {
    function VoyageAIClient(this: any) {}
    (VoyageAIClient as any).prototype.embed = async function embed() {
      return { data: [] };
    };
    (VoyageAIClient as any).prototype.multimodalEmbed = async function multimodalEmbed() {
      return { data: [] };
    };
    return {
      mod: { VoyageAIClient },
      proto: (VoyageAIClient as any).prototype,
      method: "embed" as const,
    };
  }

  /** together-ai / groq-sdk / @cerebras/cerebras_cloud_sdk — all three are
   * OpenAI-compatible, class-keyed entries (_pickClassExport accepts the
   * class passed directly — see makeOpenAIClassFixture's doc comment above
   * for why class-form is returned by identity). One shape covers all three. */
  function makeOpenAICompatibleRoundTripFixture() {
    class Completions {
      create() {
        return Promise.resolve({});
      }
    }
    function Cls(this: any) {
      this.chat = { completions: new Completions() };
    }
    (Cls as any).Chat = { Completions };
    return { mod: Cls, proto: Completions.prototype, method: "create" as const };
  }

  /** @openrouter/sdk — Chat class is reached via a throw-away OpenRouter
   * probe instance's `.chat` property; see _instrumentOpenRouter. */
  function makeOpenRouterRoundTripFixture() {
    class Chat {
      send() {
        return Promise.resolve({});
      }
    }
    function OpenRouter(this: any, _opts?: any) {
      this.chat = new Chat();
    }
    return { mod: { OpenRouter }, proto: Chat.prototype, method: "send" as const };
  }

  /** @google/genai — Models is passed through verbatim (no _pickClassExport). */
  function makeGoogleGenAIRoundTripFixture() {
    class Models {
      generateContentInternal() {
        return Promise.resolve({});
      }
      generateContentStreamInternal() {
        return {};
      }
    }
    return {
      mod: { Models },
      proto: Models.prototype,
      method: "generateContentInternal" as const,
    };
  }

  /** @aws-sdk/client-bedrock-runtime — BedrockRuntimeClient passed through
   * verbatim (no _pickClassExport). */
  function makeBedrockRoundTripFixture() {
    class BedrockRuntimeClient {
      send() {
        return Promise.resolve({});
      }
    }
    return {
      mod: { BedrockRuntimeClient },
      proto: BedrockRuntimeClient.prototype,
      method: "send" as const,
    };
  }

  const ROUND_TRIP_PROVIDERS: Array<{
    targetModuleName: string;
    makeFixture: () => { mod: any; proto: any; method: string };
  }> = [
    { targetModuleName: "openai", makeFixture: makeOpenAIRoundTripFixture },
    { targetModuleName: "@anthropic-ai/sdk", makeFixture: makeAnthropicRoundTripFixture },
    { targetModuleName: "cohere-ai", makeFixture: makeCohereRoundTripFixture },
    { targetModuleName: "@mistralai/mistralai", makeFixture: makeMistralRoundTripFixture },
    { targetModuleName: "voyageai", makeFixture: makeVoyageRoundTripFixture },
    { targetModuleName: "together-ai", makeFixture: makeOpenAICompatibleRoundTripFixture },
    { targetModuleName: "groq-sdk", makeFixture: makeOpenAICompatibleRoundTripFixture },
    {
      targetModuleName: "@cerebras/cerebras_cloud_sdk",
      makeFixture: makeOpenAICompatibleRoundTripFixture,
    },
    { targetModuleName: "@openrouter/sdk", makeFixture: makeOpenRouterRoundTripFixture },
    { targetModuleName: "@google/genai", makeFixture: makeGoogleGenAIRoundTripFixture },
    {
      targetModuleName: "@aws-sdk/client-bedrock-runtime",
      makeFixture: makeBedrockRoundTripFixture,
    },
  ];

  test.each(ROUND_TRIP_PROVIDERS)(
    "(13) $targetModuleName → the suggested key exists (totality) AND autoInstrument() actually consumes it (consumption)",
    ({ targetModuleName, makeFixture }) => {
      // ── Part (i): TOTALITY ────────────────────────────────────────────
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      init({
        apiKey: "tp_sk_test",
        deployment: "serverless",
        firewall: "dry_run",
        logErrors: true,
      } as any);
      // `{module: {}}` bypasses real package resolution entirely (same
      // technique as tests 11/12) — this is the FIRST wrap attempt for this
      // exact moduleName+objectPath+method combo, so no dedup contention.
      protect(targetModuleName, ["prototype"], "__tpRoundTripProbe", true, { module: {} });

      // Filter on the FULL target, not just the module name — see the note in
      // test (11): the `init()` above runs a real autoInstrument() pass, and
      // several of these packages are now installed devDependencies, so
      // auto-discovery can warn about other targets on the same module.
      const relevant = warn.mock.calls
        .map((c) => String(c[0]))
        .filter((m) =>
          m.includes(
            `Could not install wrapper for ${targetModuleName} prototype.__tpRoundTripProbe `,
          ),
        );
      expect(relevant).toHaveLength(1);

      const match = relevant[0].match(/instrumentModules: \{ ([A-Za-z0-9_]+):/);
      // FAILS LOUDLY for a future _TARGET_METHODS/_wrapMethod entry added
      // with no _INSTRUMENT_MODULE_KEYS row: the drift-guard's generic
      // remedy has no "instrumentModules: { key:" substring at all (see
      // test (12)), so `match` is null and totality is broken for this name.
      if (!match) {
        throw new Error(
          `totality gap: no instrumentModules key suggested for "${targetModuleName}" — ` +
            `add a row to _INSTRUMENT_MODULE_KEYS in src/enforcer.ts (message was: ` +
            `${JSON.stringify(relevant[0])})`,
        );
      }
      const suggestedKey = match[1];

      // Reset the once-guard: `init()` above already ran autoInstrument()
      // once (with no instrumentModules — real auto-discovery). Part (ii)
      // needs a FRESH pass that actually processes our fixture below —
      // without this, autoInstrument's `if (_isInstrumented) return;` would
      // make part (ii) a silent no-op, and the failure this test would then
      // report ("never ran") would obscure the real question ("was the key
      // consumed"). Resetting keeps the failure mode honest.
      uninstrument();

      // ── Part (ii): CONSUMPTION — the load-bearing half ─────────────────
      // Feed the EXACT string the warning suggested back in as a real
      // instrumentModules key, verbatim case included (autoInstrument's own
      // `key.toLowerCase()` handles case) — proving the SDK's own suggested
      // remedy actually works, not just that it reads plausibly.
      const { mod, proto, method } = makeFixture();
      const original = proto[method];
      autoInstrument({ [suggestedKey]: mod } as any);

      // FAILS LOUDLY for a map row whose value no `normalizedKey` branch in
      // the prep loop consumes (e.g. a typo'd key, or a provider renamed on
      // one side and not the other): the fixture's prototype method is never
      // wrapped, so this stays `=== original`. THIS is the assertion a
      // warning-text-only test structurally cannot make — the text can look
      // perfect while this line still fails.
      expect(proto[method]).not.toBe(original);
    },
  );
});
