/**
 * F-23-1 (run #23, P0): LlamaIndex double-billing when a Traceloop OTel
 * provider instrumentor (@traceloop/instrumentation-anthropic / -openai) ALSO
 * wraps the underlying provider SDK that a `@llamaindex/*` class calls.
 *
 * The LlamaIndex wrapper (`_setLlamaIndexWrapper` / `_logLlamaIndex` /
 * `_guardedLlamaIndexStream` in src/enforcer.ts) is the SOLE billable emitter
 * for calls made through `@llamaindex/*` provider classes. Pre-fix, when a
 * provider instrumentor also patched the underlying SDK, its span was logged
 * a SECOND time by `TokenPoliceSpanProcessor.onEnd` (src/telemetry.ts) —
 * exact 2x billing for the same physical call.
 *
 * Fix under test:
 *   - src/context.ts: `runWithLlamaIndexOtelSuppress(fn)` opens a per-call ALS
 *     window; `llamaIndexOtelSuppressActive()` reads it.
 *   - src/enforcer.ts: `_setLlamaIndexWrapper` runs `original.apply` inside the
 *     window (covers non-streaming/eager calls); `_guardedLlamaIndexStream`
 *     pulls chunks through a per-pull suppression shim (covers LAZY async-
 *     generator adapters whose inner HTTP call fires at first pull, in the
 *     CONSUMER's async context, after the wrapper's own window has unwound).
 *   - src/telemetry.ts: `onStart` tags a span with a private Symbol when the
 *     window is active AND the span looks like a provider-instrumentor genai
 *     span (gen_ai.system/gen_ai.provider.name non-empty, OR
 *     instrumentationScope name includes "anthropic"/"openai"); `onEnd` drops
 *     tagged spans — AFTER structural-anchor/tool-span handling, so a
 *     mis-tagged non-llm span still emits its own row.
 *
 * RED/GREEN — this check was run as a PARTIAL revert: `git stash` on exactly
 * `src/context.ts` `src/enforcer.ts` `src/telemetry.ts` (this test file left
 * in place), rerun, then `git stash pop`. Under that partial revert, 8/13
 * tests fail (`runWithLlamaIndexOtelSuppress`/`llamaIndexOtelSuppressActive`
 * resolve as `undefined` at the call site → `TypeError: ... is not a
 * function`, caught per-test by vitest rather than aborting the whole file).
 * A FULL revert (deleting the added exports too, not just their bodies) would
 * instead fail the top-level `import { ... } from "../src/context"` and RED
 * every test in the file — a blunter but equally valid signal.
 * Independent of that blunt signal, the discriminators that would fail on a
 * *correct-looking but wrong* revert (e.g. suppression restored but scoped to
 * the session instead of per-call, or the onEnd drop misplaced ahead of the
 * tool-span check) are:
 *   - describe("1: drop case") — the core suppression behavior.
 *   - describe("2: negative guard") — the "concurrent direct span" test pins
 *     PER-CALL (not session-wide) suppression scope: a direct span created
 *     while an unrelated call's window is open concurrently must survive. (Its
 *     "no window anywhere" sibling test is a basic sanity check, not a scoping
 *     discriminator — no LlamaIndex call runs in it, so a session-scoped flag
 *     would never even be set and that test would pass under either impl.)
 *   - describe("3: non-llm mis-tag protection") — pins drop placement AFTER
 *     tool-span promotion.
 *   - describe("6: end-to-end dedupe") — the actual regression repro: exactly
 *     ONE row for a call metered by both the manual LlamaIndex row and a
 *     simulated instrumentor span for the SAME physical call.
 *
 * Harness mirrors three in-repo precedents:
 *   - tests/anthropicStream.test.ts (A7 "7(c)"/"7(f)"/"7(g)" region): drives
 *     the real `TokenPoliceSpanProcessor` directly via
 *     `runWith*OtelSuppress(() => proc.onStart(span))` / `proc.onEnd(span)`.
 *   - tests/embeddingSpanDedup.test.ts (Part 2): captures `getClient().log(...)`
 *     rows via a session-independent processor drive, synthetic finished spans.
 *   - tests/llamaIndexStreamFailure.test.ts: drives the real
 *     `_instrumentLlamaIndexProvider` with fake `@llamaindex/*`-shaped classes
 *     (class name + prototype.chat patch), a pinned `TPSession`, and a
 *     captured `tp.log` args array.
 *
 * Note on the Golden-Rule "shim construction failing" case the reviewer
 * flagged: `_llamaIndexSuppressedIter` is NOT exported from `__test__` in
 * src/enforcer.ts (checked directly), so per the brief that sub-test is
 * skipped rather than exporting it just for this file.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { setClient } from "../src/state";
import {
  TPSession,
  _getSessionStorage,
  runWithLlamaIndexOtelSuppress,
  llamaIndexOtelSuppressActive,
} from "../src/context";
import { __test__ as enforcerTest } from "../src/enforcer";
import { TokenPoliceSpanProcessor } from "../src/telemetry";

// ── shared harness (mirrors llamaIndexStreamFailure.test.ts) ────────
const SPAN_ARG = 10; // tp.log(...) positional index of the `span` object
const EXTRAS_ARG = 13; // tp.log(...) positional index of the trailing `extras`

let logged: any[][];

beforeEach(() => {
  logged = [];
  // firewall:"off" short-circuits the async pre-flight; log captures the row.
  // Shared by BOTH the manual LlamaIndex emitter (enforcer.ts, via
  // src/state's getClient()) and TokenPoliceSpanProcessor.onEnd (telemetry.ts,
  // same state module) — one array sees every row from either emitter, which
  // is exactly what the dedupe assertions need.
  setClient({ firewall: "off", log: (...args: any[]) => logged.push(args) } as any);
});

afterEach(() => {
  setClient(null as any);
  for (const thunk of enforcerTest._restoreThunks.splice(0)) {
    try {
      thunk();
    } catch {
      /* ignore */
    }
  }
});

const flush = () => new Promise((r) => setTimeout(r, 0));
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function llmRows(): any[][] {
  return logged.filter((r) => r[SPAN_ARG]?.span_kind === "llm");
}

function newSession(): TPSession {
  return new TPSession({
    userId: "u1",
    paidPlan: "pro",
    workflowName: "wf",
    traceId: "c".repeat(32),
    rootSpanId: "d".repeat(16),
  });
}

/**
 * A minimally OTel-ReadableSpan-shaped synthetic provider-instrumentor span:
 * async-suppression tags it in onStart (mutates `[TP_DROP...]` on the object,
 * never on `attributes`); if untagged, onEnd fully logs it (spanContext +
 * setAttribute present) so survivor-vs-dropped is observable.
 */
let _spanSeq = 0;
function makeGenaiSpan(opts: {
  attrs?: Record<string, unknown>;
  scopeName?: string;
  name?: string;
} = {}): any {
  const spanId = `deadbeef${(_spanSeq++).toString(16).padStart(8, "0")}`;
  const s: any = {
    name: opts.name ?? "anthropic.chat",
    attributes: { ...(opts.attrs ?? {}) },
    instrumentationScope: { name: opts.scopeName ?? "" },
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

/** Baseline attrs that satisfy `isLlmSpan` once untagged/undropped. */
function anthropicUsageAttrs(): Record<string, unknown> {
  return {
    "gen_ai.system": "anthropic",
    "gen_ai.request.model": "claude-3",
    "gen_ai.usage.input_tokens": 7,
    "gen_ai.usage.output_tokens": 2,
  };
}

// ═══════════════════════════════════════════════════════════════════
// 1. Drop case: onStart runs inside the suppression window → dropped.
// ═══════════════════════════════════════════════════════════════════
describe("1: drop case — provider-instrumentor span started inside runWithLlamaIndexOtelSuppress", () => {
  it("(a) gen_ai attrs present at onStart, neutral (non-anthropic/openai) scope → tagged and dropped", async () => {
    const proc = new TokenPoliceSpanProcessor();
    const span = makeGenaiSpan({ attrs: anthropicUsageAttrs(), scopeName: "some-neutral-scope" });

    runWithLlamaIndexOtelSuppress(() => proc.onStart(span));
    const before = logged.length;
    proc.onEnd(span);
    await flush();

    expect(logged.length).toBe(before); // dropped — no new row
  });

  it("(b) empty attrs at onStart but scope '@traceloop/instrumentation-anthropic' → tagged; usage added before onEnd still dropped", async () => {
    const proc = new TokenPoliceSpanProcessor();
    // At onStart, the instrumentor has not yet stamped gen_ai.* — its own
    // post-hook only runs after the underlying HTTP call resolves.
    const span = makeGenaiSpan({ attrs: {}, scopeName: "@traceloop/instrumentation-anthropic" });

    runWithLlamaIndexOtelSuppress(() => proc.onStart(span));
    // Usage lands only now (mirrors the instrumentor's own post-hook).
    Object.assign(span.attributes, {
      "gen_ai.system": "anthropic",
      "gen_ai.usage.input_tokens": 7,
      "gen_ai.usage.output_tokens": 2,
    });
    const before = logged.length;
    proc.onEnd(span);
    await flush();

    expect(logged.length).toBe(before); // dropped — tag was set at onStart, unaffected by later attrs
  });

  // Control for (b): identical span/attrs/timing, but onStart runs with NO
  // window active — proves the usage added post-onStart really would have
  // produced a loggable row (i.e. (b) is not vacuously passing because the
  // span never qualified as an LLM span in the first place).
  it("(b-control) same span, same late-added usage, but onStart OUTSIDE any window → WOULD have logged 1 row", async () => {
    const proc = new TokenPoliceSpanProcessor();
    const span = makeGenaiSpan({ attrs: {}, scopeName: "@traceloop/instrumentation-anthropic" });

    proc.onStart(span); // no window — untagged
    Object.assign(span.attributes, {
      "gen_ai.system": "anthropic",
      "gen_ai.usage.input_tokens": 7,
      "gen_ai.usage.output_tokens": 2,
    });
    const before = logged.length;
    proc.onEnd(span);
    await flush();

    expect(logged.length).toBe(before + 1); // confirms (b)'s drop is meaningful, not vacuous
  });

  it("(c) empty attrs at onStart but scope '@traceloop/instrumentation-openai' → tagged; usage added before onEnd still dropped", async () => {
    const proc = new TokenPoliceSpanProcessor();
    const span = makeGenaiSpan({ attrs: {}, scopeName: "@traceloop/instrumentation-openai", name: "openai.chat" });

    runWithLlamaIndexOtelSuppress(() => proc.onStart(span));
    Object.assign(span.attributes, {
      "gen_ai.system": "openai",
      "gen_ai.usage.input_tokens": 50,
      "gen_ai.usage.output_tokens": 10,
    });
    const before = logged.length;
    proc.onEnd(span);
    await flush();

    expect(logged.length).toBe(before);
  });

  // Control for (c): same idea as (b-control) — proves the late-added usage
  // on an untagged span really would have produced a loggable row.
  it("(c-control) same span, same late-added usage, but onStart OUTSIDE any window → WOULD have logged 1 row", async () => {
    const proc = new TokenPoliceSpanProcessor();
    const span = makeGenaiSpan({ attrs: {}, scopeName: "@traceloop/instrumentation-openai", name: "openai.chat" });

    proc.onStart(span); // no window — untagged
    Object.assign(span.attributes, {
      "gen_ai.system": "openai",
      "gen_ai.usage.input_tokens": 50,
      "gen_ai.usage.output_tokens": 10,
    });
    const before = logged.length;
    proc.onEnd(span);
    await flush();

    expect(logged.length).toBe(before + 1); // confirms (c)'s drop is meaningful, not vacuous
  });
});

// ═══════════════════════════════════════════════════════════════════
// 2. Negative / under-billing guard (most valuable): direct calls are
//    NEVER dropped. Pins per-call (not session-wide) suppression scope.
// ═══════════════════════════════════════════════════════════════════
describe("2: negative guard — span started OUTSIDE any suppression window is never dropped", () => {
  it("identical genai span, onStart run with NO window active anywhere → onEnd logs exactly 1 row", async () => {
    const proc = new TokenPoliceSpanProcessor();
    const span = makeGenaiSpan({ attrs: anthropicUsageAttrs(), scopeName: "@traceloop/instrumentation-anthropic" });

    proc.onStart(span); // no suppression window active — a direct provider call
    const before = logged.length;
    proc.onEnd(span);
    await flush();

    expect(logged.length).toBe(before + 1); // survives — never eaten
    expect(llmRows().length).toBe(1);
  });

  // THE actual per-call-scoping discriminator: unlike the test above (where
  // no window exists ANYWHERE, so even a broken session-wide flag would
  // trivially pass), this opens ANOTHER call's suppression window and keeps
  // it open concurrently while a DIRECT span — created OUTSIDE that window's
  // `run()` callback — starts and ends. Per-call ALS scoping must leave the
  // direct span untagged; a session-scoped (or global) suppression flag would
  // be "on" for the whole process during the concurrent window and would
  // wrongly eat it → under-billing, the failure mode this fix must never
  // introduce (see runWithLlamaIndexOtelSuppress's doc comment in context.ts).
  it("a DIRECT span started while ANOTHER call's window is open concurrently is NOT dropped", async () => {
    const proc = new TokenPoliceSpanProcessor();
    const direct = makeGenaiSpan({ attrs: anthropicUsageAttrs(), scopeName: "@traceloop/instrumentation-anthropic" });

    await Promise.all([
      runWithLlamaIndexOtelSuppress(async () => {
        await sleep(5); // window open the whole time the direct span runs
      }),
      (async () => {
        await sleep(1);
        proc.onStart(direct); // created OUTSIDE run() — no window of its own
        proc.onEnd(direct);
      })(),
    ]);
    await flush();

    expect(llmRows().length).toBe(1); // fails under a session-scoped/global flag
  });
});

// ═══════════════════════════════════════════════════════════════════
// 3. Non-llm mis-tag protection: the drop check sits AFTER tool-span
//    promotion in onEnd, so a tool span tagged in onStart still emits.
// ═══════════════════════════════════════════════════════════════════
describe("3: non-llm mis-tag protection — a tool span started inside the window still emits its tool row", () => {
  it("tool-span discriminator (gen_ai.operation.name=execute_tool) wins over the drop", async () => {
    const proc = new TokenPoliceSpanProcessor();
    const span = makeGenaiSpan({
      attrs: {
        // Non-empty gen_ai.system forces the llamaindex-inner tagging gate to
        // fire in onStart (defense-in-depth signal), same as a real span
        // would carry — but this is a TOOL span, per the discriminator below.
        "gen_ai.system": "anthropic",
        "gen_ai.operation.name": "execute_tool",
      },
      scopeName: "@traceloop/instrumentation-anthropic",
      name: "execute_tool my_tool",
    });

    runWithLlamaIndexOtelSuppress(() => proc.onStart(span));
    const before = logged.length;
    proc.onEnd(span);
    await flush();

    expect(logged.length).toBe(before + 1); // NOT dropped — tool promotion wins
    const row = logged[logged.length - 1];
    expect(row[SPAN_ARG].span_kind).toBe("tool");
  });
});

// ═══════════════════════════════════════════════════════════════════
// 4. Window scoping: the tag is decided at onStart and travels ON the
//    span object — onEnd never re-checks the live active() state.
// ═══════════════════════════════════════════════════════════════════
describe("4: window scoping — the tag travels on the span, onEnd never re-checks live state", () => {
  it("(a) started INSIDE the window, onEnd runs OUTSIDE any window → still dropped", async () => {
    const proc = new TokenPoliceSpanProcessor();
    const span = makeGenaiSpan({ attrs: anthropicUsageAttrs(), scopeName: "@traceloop/instrumentation-anthropic" });

    runWithLlamaIndexOtelSuppress(() => proc.onStart(span));
    const before = logged.length;
    proc.onEnd(span); // no window active here
    await flush();

    expect(logged.length).toBe(before);
  });

  it("(b) started OUTSIDE any window, onEnd runs INSIDE a (later, unrelated) window → NOT dropped", async () => {
    const proc = new TokenPoliceSpanProcessor();
    const span = makeGenaiSpan({ attrs: anthropicUsageAttrs(), scopeName: "@traceloop/instrumentation-anthropic" });

    proc.onStart(span); // untagged — no window active at start time
    const before = logged.length;
    runWithLlamaIndexOtelSuppress(() => proc.onEnd(span)); // window active now, tag decision already made
    await flush();

    expect(logged.length).toBe(before + 1); // survives — onEnd only reads the (absent) tag
  });
});

// ═══════════════════════════════════════════════════════════════════
// 5. Shim semantics through _guardedLlamaIndexStream (integration via a
//    fake @llamaindex/*-shaped provider class).
// ═══════════════════════════════════════════════════════════════════
function defineProviderClass(
  name: string,
  modelName: string,
  chat: (this: any, params: any) => any,
): any {
  const Cls = {
    [name]: class {
      model = modelName;
    },
  }[name];
  (Cls.prototype as any).chat = chat;
  return Cls;
}

function instrument(name: string, model: string, chat: (this: any, p: any) => any): any {
  const mod: any = { [name]: defineProviderClass(name, model, chat) };
  enforcerTest._instrumentLlamaIndexProvider(mod, [name]);
  return mod[name];
}

/** A clean (non-throwing) lazy async-generator stream of `chunks`. */
function cleanStream(chunks: any[]): AsyncIterable<any> {
  return (async function* () {
    for (const c of chunks) {
      await sleep(1); // real gap so this is genuinely lazy per-pull work
      yield c;
    }
  })();
}

describe("5: shim semantics — chunk pass-through, error identity, lazy first-pull suppression", () => {
  it("(a) chunk order and count seen by the consumer are unchanged", async () => {
    const chunks = [
      { delta: "a", raw: { model: "gpt-4o" } },
      { delta: "b", raw: { model: "gpt-4o" } },
      { delta: "c", raw: { model: "gpt-4o", usage: { prompt_tokens: 10, completion_tokens: 3 } } },
    ];
    const Cls = instrument("OpenAI", "gpt-4o", async () => cleanStream(chunks));
    const session = newSession();
    const seen: any[] = [];

    await _getSessionStorage().run(session, async () => {
      const result = await new Cls().chat({ messages: [], stream: true });
      for await (const c of result) seen.push(c);
    });

    expect(seen).toEqual(chunks);
    expect(seen.length).toBe(3);
  });

  it("(b) a consumer early break still emits the partial-success manual row", async () => {
    const chunks = [
      {
        delta: "a",
        raw: {
          model: "gpt-4o",
          usage: { prompt_tokens: 100, completion_tokens: 20, prompt_tokens_details: { cached_tokens: 30 } },
        },
      },
      { delta: "b", raw: { model: "gpt-4o" } },
    ];
    const Cls = instrument("OpenAI", "gpt-4o", async () => cleanStream(chunks));
    const session = newSession();

    await _getSessionStorage().run(session, async () => {
      const result = await new Cls().chat({ messages: [], stream: true });
      for await (const _c of result) break; // .return() path, not exhaustion
    });

    expect(logged).toHaveLength(1);
    const extras = logged[0][EXTRAS_ARG];
    expect(extras.call_outcome.status).toBe("success");
    expect(session.inLlamaIndex).toBe(false); // guard released
  });

  it("(c) a rejecting stream propagates the ORIGINAL error by identity", async () => {
    const boom = Object.assign(new Error("upstream 500"), { status: 500 });
    const Cls = instrument("OpenAI", "gpt-4o", async () =>
      (async function* () {
        await sleep(1);
        throw boom;
      })(),
    );
    const session = newSession();
    let caught: unknown;

    await _getSessionStorage().run(session, async () => {
      try {
        const result = await new Cls().chat({ messages: [], stream: true });
        for await (const _c of result) void _c;
      } catch (e) {
        caught = e;
      }
    });

    expect(caught).toBe(boom); // identity, not just instanceof/message match
  });

  it("(d) a lazy async-generator source's first-pull work runs with llamaIndexOtelSuppressActive() === true", async () => {
    let activeAtFirstPull: boolean | undefined;
    const Cls = instrument("OpenAI", "gpt-4o", async () =>
      (async function* () {
        // This body only executes at the first .next() PULL — in the
        // CONSUMER's async context, after the wrapper's own
        // `await original.apply()` window has already unwound. The per-pull
        // shim (_llamaIndexSuppressedIter) must still make the window active
        // here — that's exactly what covers @llamaindex/anthropic's lazy
        // streamChat HTTP call (F-23-1's original failure mode for streams).
        activeAtFirstPull = llamaIndexOtelSuppressActive();
        yield { delta: "a", raw: { model: "gpt-4o", usage: { prompt_tokens: 1, completion_tokens: 1 } } };
      })(),
    );
    const session = newSession();

    await _getSessionStorage().run(session, async () => {
      const result = await new Cls().chat({ messages: [], stream: true });
      for await (const _c of result) void _c;
    });

    expect(activeAtFirstPull).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 6. End-to-end dedupe — the actual F-23-1 regression repro.
// ═══════════════════════════════════════════════════════════════════
describe("6: end-to-end dedupe — the actual regression", () => {
  it("fake LlamaIndex provider whose chat() body starts+ends a real instrumentor-shaped span for the SAME call → exactly ONE llm row (not two)", async () => {
    const proc = new TokenPoliceSpanProcessor();

    const Cls = instrument("Anthropic", "claude-3", async function (this: any) {
      // Simulate the underlying provider SDK call: a real Traceloop
      // instrumentor would start+end its OWN span for this physical HTTP
      // call, WHILE the wrapper's runWithLlamaIndexOtelSuppress window (open
      // around the whole `original.apply`, ALS-propagated through awaits
      // initiated inside it) is active.
      await sleep(1);
      const innerSpan = makeGenaiSpan({
        attrs: {
          "gen_ai.system": "anthropic",
          "gen_ai.request.model": "claude-3",
          "gen_ai.usage.input_tokens": 100,
          "gen_ai.usage.output_tokens": 20,
        },
        scopeName: "@traceloop/instrumentation-anthropic",
      });
      proc.onStart(innerSpan);
      await sleep(1);
      proc.onEnd(innerSpan);

      return {
        raw: { model: "claude-3", usage: { input_tokens: 100, output_tokens: 20 } },
        message: { role: "assistant", content: "hi" },
      };
    });
    const session = newSession();

    await _getSessionStorage().run(session, async () => {
      await new Cls().chat({ messages: [] });
    });
    await flush();

    // Pre-fix: 2 rows (the manual _logLlamaIndex row + the instrumentor's
    // duplicate span surviving onEnd) — exact 2x billing (F-23-1). Post-fix:
    // the inner span is tagged in onStart and dropped in onEnd → 1 row.
    expect(llmRows().length).toBe(1);
  });

  it("the SAME inner-span emission OUTSIDE any LlamaIndex call logs its own 1 row (control — direct calls stay billable)", async () => {
    const proc = new TokenPoliceSpanProcessor();
    const span = makeGenaiSpan({
      attrs: {
        "gen_ai.system": "anthropic",
        "gen_ai.request.model": "claude-3",
        "gen_ai.usage.input_tokens": 100,
        "gen_ai.usage.output_tokens": 20,
      },
      scopeName: "@traceloop/instrumentation-anthropic",
    });

    proc.onStart(span); // no llamaIndex wrapper anywhere in this call
    proc.onEnd(span);
    await flush();

    expect(llmRows().length).toBe(1); // its own row — never eaten by an absent window
  });
});
