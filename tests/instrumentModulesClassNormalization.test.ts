/**
 * instrumentModules class-entry normalization (_pickClassExport, src/enforcer.ts).
 *
 * autoInstrument's five class-keyed instrumentModules entries (openai,
 * anthropic, cerebras, together, groq) used to assume the caller passed the
 * provider CLASS directly and built `{ ClassName: mod, default: mod }` from
 * whatever value it was handed. When the value was actually a MODULE
 * NAMESPACE (`import * as A from "@anthropic-ai/sdk"`, or the package's CJS
 * root under tsx/ts-node), `_resolvePath`'s walk (e.g.
 * ["Anthropic","Messages","prototype"]) needed root-level `Messages`/`Beta`
 * re-exports that only ever existed on OLD SDK roots (anthropic ≤0.41.0,
 * openai <5, groq-sdk <1, together-ai <0.30) — the Stainless-generated roots
 * for every one of those packages stopped re-exporting resource statics at
 * the versions above, and a true ESM namespace NEVER carried them, any
 * version. The practical effect: a namespace-style `instrumentModules` entry
 * on a current SDK version got ZERO pre-flight `/check` calls — enforcement
 * silently dead while `/log` telemetry kept flowing, so dashboards looked
 * healthy.
 *
 * The fix (`_pickClassExport`) feature-detects the class off the value
 * (itself, `.ClassName`, `.default`, or `.default.ClassName`) before building
 * the moduleMap entry, never version-sniffs, and degrades to the pre-fix
 * verbatim-passthrough on any failure (hostile getter, `{}`, `undefined`) —
 * autoInstrument must never throw.
 *
 * HARNESS NOTES — mirrors tests/apiPromiseSurface.test.ts (Group A) and
 * tests/preflightCtxF7.test.ts: `init({ instrumentModules })` installs the
 * wrapper, `client.check`/`client.log` are spied so nothing hits the network,
 * and firing the pre-flight is asserted via the `client.check` spy — NOT via
 * method-identity change alone, since OTel/traceloop also re-wraps these
 * methods and an identity check can't tell the two apart. `@anthropic-ai/sdk`
 * / `openai` / `groq-sdk` / `together-ai` / `@cerebras/cerebras_cloud_sdk`
 * are NOT devDependencies here — every fixture is a hand-built fake
 * reproducing only the vendor CONTRACT autoInstrument depends on.
 */
import { describe, test, expect, afterEach, vi } from "vitest";
import {
  autoInstrument,
  uninstrument,
  getUnwrappedResolvableProviders,
} from "../src/enforcer";
import { init } from "../src/client";
import { setClient } from "../src/state";

const BODY = { model: "claude-3", messages: [{ role: "user", content: "hi" }] };

// ── Anthropic fakes ──────────────────────────────────────────────────────
// Fresh classes per call so no cross-test `__tpStreamPatched` / `_originals`
// dedupe contamination. No `APIPromise` static export anywhere reachable off
// the extracted class ⇒ the anthropic create-streaming bypass probe (see
// enforcer.ts _wrapMethod) evaluates identically to the class-form input
// today — not exercised by these (non-streaming) bodies either way.
function makeAnthropicClass(): { Anthropic: any; Messages: any; BetaMessages: any } {
  class Messages {
    async create(body: any): Promise<any> {
      return {
        id: "msg_1",
        model: body?.model,
        role: "assistant",
        content: [{ type: "text", text: "hi" }],
        stop_reason: "end_turn",
        usage: { input_tokens: 3, output_tokens: 2 },
      };
    }
    stream(_body: any): any {
      return {};
    }
  }
  class BetaMessages {
    async create(body: any): Promise<any> {
      return {
        id: "msg_beta_1",
        model: body?.model,
        role: "assistant",
        content: [{ type: "text", text: "hi" }],
        stop_reason: "end_turn",
        usage: { input_tokens: 3, output_tokens: 2 },
      };
    }
    stream(_body: any): any {
      return {};
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

/**
 * The tsx/CJS root shape (anthropic ≤0.41.0): root-level `Messages`/`Beta`
 * re-exports AND `.Anthropic`/`.default` self-refs — all pointing at the
 * SAME class, carrying the SAME statics. Pre-fix, `moduleMap.Anthropic` was
 * this whole namespace object (via the `.Anthropic` self-ref path
 * `_resolvePath` never needed to take, since root `Messages`/`Beta` resolved
 * directly) — this shape worked before the fix and must keep working after.
 */
function makeOldShapeAnthropicNamespace() {
  const { Anthropic, Messages, BetaMessages } = makeAnthropicClass();
  const ns: any = {
    Anthropic,
    default: Anthropic,
    Messages: (Anthropic as any).Messages,
    Beta: (Anthropic as any).Beta,
  };
  return { ns, Anthropic, Messages, BetaMessages };
}

/**
 * The regression shape: a CJS root ≥0.50.1 (0.42–0.50.0 don't exist) or a
 * true ESM namespace, either way carrying ONLY the class under `Anthropic`/
 * `default` — no root `Messages`/`Beta` re-exports. `APIPromise` is a decoy
 * present on real roots of this shape; it must not confuse `_pickClassExport`
 * (which only ever looks at `className`/`default`/`default.className`).
 */
function makeNewShapeAnthropicNamespace() {
  const { Anthropic, Messages, BetaMessages } = makeAnthropicClass();
  const ns: any = {
    Anthropic,
    default: Anthropic,
    APIPromise: function APIPromise() {},
  };
  return { ns, Anthropic, Messages, BetaMessages };
}

// ── OpenAI-compatible fakes (openai / groq / together / cerebras) ──────────
function makeOpenAICompatibleClass(): { Cls: any; Completions: any } {
  class Completions {
    async create(_body: any): Promise<any> {
      return {
        id: "cmpl_1",
        model: "m",
        choices: [
          { index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
      };
    }
  }
  function Cls(this: any) {
    this.chat = { completions: new Completions() };
  }
  (Cls as any).Chat = { Completions };
  return { Cls, Completions };
}

/** Install the wrapper with the pre-flight stubbed ALLOWED, spying `check` + `log`. */
function initAllowed(instrumentModules: Record<string, any>) {
  const client = init({
    apiKey: "tp_sk_test",
    deployment: "serverless",
    firewall: "dry_run",
    instrumentModules,
  } as any);
  vi.spyOn(client, "log").mockImplementation(() => {});
  const checkSpy = vi
    .spyOn(client, "check")
    .mockResolvedValue({ status: "allowed", fail_open: false } as any);
  return { client, checkSpy };
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
// 1 — CLASS form: parity baseline (byte-identical to pre-fix behaviour).
// ═══════════════════════════════════════════════════════════════════
describe("(1) anthropic CLASS form — parity baseline", () => {
  test("statics on the class directly → wraps, one pre-flight /check", async () => {
    const { Anthropic } = makeAnthropicClass();
    const { checkSpy } = initAllowed({ anthropic: Anthropic });
    await new Anthropic().messages.create(BODY);
    expect(checkSpy).toHaveBeenCalledTimes(1);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 2 — OLD-shape namespace (tsx/CJS ≤0.41.0): must keep working post-fix.
// ═══════════════════════════════════════════════════════════════════
describe("(2) anthropic OLD-shape namespace (root statics + self-refs)", () => {
  test("wraps the SAME prototype the root re-export and the self-ref both carry", async () => {
    const { ns, Anthropic, Messages } = makeOldShapeAnthropicNamespace();
    // Sanity: pre-fix, both paths already led to this one object.
    expect(ns.Messages).toBe(Anthropic.Messages);
    const origCreate = Messages.prototype.create;

    const { checkSpy } = initAllowed({ anthropic: ns });

    expect(Messages.prototype.create).not.toBe(origCreate);
    await new Anthropic().messages.create(BODY);
    expect(checkSpy).toHaveBeenCalledTimes(1);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 3 — NEW-shape namespace: THE regression case. 0/6 pre-fix, 6/6 post-fix.
// ═══════════════════════════════════════════════════════════════════
describe("(3) anthropic NEW-shape namespace (≥0.50.1 root / true ESM)", () => {
  test("no root Messages/Beta re-export → still wraps, one pre-flight /check", async () => {
    const { ns, Anthropic } = makeNewShapeAnthropicNamespace();
    // Sanity: this really is the regression shape (no root re-exports at all).
    expect(ns.Messages).toBeUndefined();
    expect(ns.Beta).toBeUndefined();

    const { checkSpy } = initAllowed({ anthropic: ns });
    await new Anthropic().messages.create(BODY);
    expect(checkSpy).toHaveBeenCalledTimes(1);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 4 — ESM-minimal namespace: only a `default` export.
// ═══════════════════════════════════════════════════════════════════
describe("(4) ESM-minimal namespace ({ default: cls } only)", () => {
  test("wraps via the default export, one pre-flight /check", async () => {
    const { Anthropic } = makeAnthropicClass();
    const ns = { default: Anthropic };
    const { checkSpy } = initAllowed({ anthropic: ns });
    await new Anthropic().messages.create(BODY);
    expect(checkSpy).toHaveBeenCalledTimes(1);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 5 — Interop double-default: `{ default: { Anthropic: cls } }`.
// ═══════════════════════════════════════════════════════════════════
describe("(5) interop double-default namespace ({ default: { Anthropic: cls } })", () => {
  test("wraps via the nested default.Anthropic lookup, one pre-flight /check", async () => {
    const { Anthropic } = makeAnthropicClass();
    const ns = { default: { Anthropic } };
    const { checkSpy } = initAllowed({ anthropic: ns });
    await new Anthropic().messages.create(BODY);
    expect(checkSpy).toHaveBeenCalledTimes(1);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 6 — Beta.Messages (SIBLING class) via the NEW-shape namespace.
// ═══════════════════════════════════════════════════════════════════
describe("(6) Beta.Messages path via the NEW-shape namespace", () => {
  test("client.beta.messages.create also wraps and checks, not just Messages", async () => {
    const { ns, Anthropic, Messages, BetaMessages } = makeNewShapeAnthropicNamespace();
    const origMessagesCreate = Messages.prototype.create;
    const origBetaCreate = BetaMessages.prototype.create;

    const { checkSpy } = initAllowed({ anthropic: ns });

    // Both classes wrapped independently — Beta.Messages is a sibling class,
    // not a subclass, so patching Messages does not reach it.
    expect(Messages.prototype.create).not.toBe(origMessagesCreate);
    expect(BetaMessages.prototype.create).not.toBe(origBetaCreate);

    const client = new Anthropic();
    await client.beta.messages.create(BODY);
    expect(checkSpy).toHaveBeenCalledTimes(1);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 7 — Sibling keys: openai / groq / together / cerebras NEW-shape
//     namespaces (class under a named export only — no default at all).
// ═══════════════════════════════════════════════════════════════════
describe("(7) sibling class-keyed providers — NEW-shape namespace (named export only)", () => {
  test("openai: { OpenAI: cls } → wraps, one pre-flight /check", async () => {
    const { Cls: OpenAI } = makeOpenAICompatibleClass();
    const ns: any = { OpenAI };
    const { checkSpy } = initAllowed({ openai: ns });
    await new OpenAI().chat.completions.create(BODY);
    expect(checkSpy).toHaveBeenCalledTimes(1);
  });

  test("groq: { Groq: cls } → wraps, one pre-flight /check", async () => {
    const { Cls: Groq } = makeOpenAICompatibleClass();
    const ns: any = { Groq };
    const { checkSpy } = initAllowed({ groq: ns });
    await new Groq().chat.completions.create(BODY);
    expect(checkSpy).toHaveBeenCalledTimes(1);
  });

  test("together: { Together: cls } → wraps, one pre-flight /check", async () => {
    const { Cls: Together } = makeOpenAICompatibleClass();
    const ns: any = { Together };
    const { checkSpy } = initAllowed({ together: ns });
    await new Together().chat.completions.create(BODY);
    expect(checkSpy).toHaveBeenCalledTimes(1);
  });

  test("cerebras: { Cerebras: cls } → wraps, one pre-flight /check", async () => {
    const { Cls: Cerebras } = makeOpenAICompatibleClass();
    const ns: any = { Cerebras };
    const { checkSpy } = initAllowed({ cerebras: ns });
    await new Cerebras().chat.completions.create(BODY);
    expect(checkSpy).toHaveBeenCalledTimes(1);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 8 — GOLDEN RULE: malformed instrumentModules never throw; degrade to
//     zero-wrap (still surfaced by the zero-wrap audit when resolvable).
// ═══════════════════════════════════════════════════════════════════
describe("(8) golden rule: malformed instrumentModules never throw", () => {
  test("hostile getter on the class-name key → no throw, zero-wrap reported", () => {
    const hostile: any = {
      get Anthropic(): any {
        throw new Error("boom: hostile Anthropic getter");
      },
    };
    expect(() => autoInstrument({ anthropic: hostile } as any)).not.toThrow();
    expect(getUnwrappedResolvableProviders(() => true)).toContain("@anthropic-ai/sdk");
  });

  test("hostile default getter (no Anthropic key at all) → no throw, zero-wrap reported", () => {
    const hostile: any = {
      get default(): any {
        throw new Error("boom: hostile default getter");
      },
    };
    expect(() => autoInstrument({ anthropic: hostile } as any)).not.toThrow();
    expect(getUnwrappedResolvableProviders(() => true)).toContain("@anthropic-ai/sdk");
  });

  test("empty object → no throw, degrades to zero-wrap", () => {
    expect(() => autoInstrument({ anthropic: {} } as any)).not.toThrow();
    expect(getUnwrappedResolvableProviders(() => true)).toContain("@anthropic-ai/sdk");
  });

  test("undefined value under the key → no throw, degrades to zero-wrap", () => {
    expect(() => autoInstrument({ anthropic: undefined } as any)).not.toThrow();
    expect(getUnwrappedResolvableProviders(() => true)).toContain("@anthropic-ai/sdk");
  });
});

// ═══════════════════════════════════════════════════════════════════
// 9 — Stream seam: the NEW-shape namespace also feeds
//     _instrumentAnthropicStream (Messages.prototype.stream AND
//     Beta.Messages.prototype.stream), and uninstrument() restores both.
// ═══════════════════════════════════════════════════════════════════
describe("(9) stream seam wired from the NEW-shape namespace", () => {
  test("Messages.prototype.stream and Beta.Messages.prototype.stream are patched, then restored", () => {
    const { ns, Messages, BetaMessages } = makeNewShapeAnthropicNamespace();
    const origMessagesStream = Messages.prototype.stream;
    const origBetaStream = BetaMessages.prototype.stream;

    autoInstrument({ anthropic: ns } as any);
    expect(Messages.prototype.stream).not.toBe(origMessagesStream);
    expect(BetaMessages.prototype.stream).not.toBe(origBetaStream);

    uninstrument();
    expect(Messages.prototype.stream).toBe(origMessagesStream);
    expect(BetaMessages.prototype.stream).toBe(origBetaStream);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 10 — uninstrument() restores the create wrappers for the NEW-shape path.
// ═══════════════════════════════════════════════════════════════════
describe("(10) uninstrument() restores originals for the NEW-shape path", () => {
  test("Messages.prototype.create and Beta.Messages.prototype.create are restored", () => {
    const { ns, Messages, BetaMessages } = makeNewShapeAnthropicNamespace();
    const origCreate = Messages.prototype.create;
    const origBetaCreate = BetaMessages.prototype.create;

    autoInstrument({ anthropic: ns } as any);
    expect(Messages.prototype.create).not.toBe(origCreate);
    expect(BetaMessages.prototype.create).not.toBe(origBetaCreate);

    uninstrument();
    expect(Messages.prototype.create).toBe(origCreate);
    expect(BetaMessages.prototype.create).toBe(origBetaCreate);
  });
});
