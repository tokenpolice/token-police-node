/**
 * B7 regression fence — the Vercel AI SDK seam.
 *
 * `_aiSdkRunGenerate` / `_aiSdkRunStream` / `_aiSdkRunEmbed` /
 * `_aiSdkRunModality` (src/enforcer.ts) used to call
 * `_runAsyncCheck({ model: meta.modelId }, meta.provider, intent, false)` —
 * a throwaway body object PLUS canReroute=false. Because canReroute was
 * false, `_runAsyncCheck`'s State-B fallback skipped its
 * `if (canReroute) { _applyReroute(...) }` guard entirely: a live
 * same-provider ENFORCE reroute directive from `/check` produced NO
 * `_applyReroute` call at all — no REROUTE_REJECTED, no REQUEST_REROUTED, no
 * `_tp_routing`, the call served un-swapped and unrecorded (observed live:
 * 16 directives / 0 resolutions).
 *
 * The fix converts all four call sites to the body-less framework convention
 * already used by LangChain / LlamaIndex / Bedrock Converse: null body,
 * canReroute=true, and the new trailing `modelHint=meta.modelId`. A null
 * body always fails `_applyReroute`'s appliability check
 * (`"model" in body`), so the directive now resolves as `reroute_rejected`
 * (`unappliable_call_shape`) — visible, auditable, matches every other
 * hint-only framework path. See tests/rerouteUnappliableShape.test.ts (the
 * PR #482 prior art for this exact rejection shape on the LangChain seam).
 *
 * IMPORTANT: tests/aiSdk.test.ts mocks `pushObservation`/`drainObservations`
 * as no-ops (see its header comment), so it could never have caught this bug
 * — a live reroute directive there just vanishes into the stub. This file
 * spreads the REAL `../src/state` module and overrides ONLY `getClient`, so
 * the real observation queue and localDecisionStore are exercised
 * end-to-end.
 *
 * NOTE on assertion style: unlike rerouteUnappliableShape.test.ts's raw
 * `_runAsyncCheck`/`_applyReroute` unit tests (which read the observation
 * straight off `state.drainObservations()`), the AI SDK seam's SUCCESS path
 * runs through `_logManual`/`_logModality`, which itself claims this call's
 * keyed observations and attaches them to the `tp.log(...)` row's trailing
 * `extra.observations` array (enforcer.ts ~5529-5545) before the wrapped
 * call returns — so by the time a driving test can call
 * `state.drainObservations()` afterward, `_logManual` has already claimed
 * and removed them from the queue. These tests therefore assert against the
 * logged row's `extra.observations` / `extra.local_decision`, which is also
 * the more precise check: it proves the rejection reaches the actual `/log`
 * payload the collector sees, not just the transient in-process queue.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

const h = vi.hoisted(() => {
  const logged: any[][] = [];
  const clientBox: { client: any } = { client: null };
  return { logged, clientBox };
});

// Spread the real module — only getClient is swapped out. Every observation/
// state export (pushObservation, drainObservations, getPack, isCacheHealthy,
// getCurrentObsKey, ...) stays the real implementation.
vi.mock("../src/state", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/state")>();
  return {
    ...actual,
    getClient: () => h.clientBox.client,
  };
});

import { autoInstrument, uninstrument } from "../src/enforcer";
import { TokenPoliceBlockedError } from "../src/exceptions";
import { session } from "../src/context";
import * as state from "../src/state";

function makeClient(overrides: Record<string, any> = {}) {
  return {
    firewall: "enforce",
    // Never "daemon" — State A's local evaluator bails (tp.deployment !==
    // "daemon"), which forces every call through the State-B inline /check
    // fallback this bug lived in, exactly like a real serverless deployment
    // with no daemon pack.
    deployment: "serverless",
    logErrors: false,
    log: (...args: any[]) => {
      h.logged.push(args);
    },
    ...overrides,
  };
}

/** A same-provider ENFORCE reroute directive, as /check would return it. */
function enforceReroute(model: string, provider = "openai", ruleName?: string) {
  return {
    status: "allowed",
    reroute: {
      mode: "enforce",
      model,
      provider,
      rule_id: "rule_b7",
      ...(ruleName ? { rule_name: ruleName } : {}),
    },
  };
}

/** Trailing positional arg on every tp.log(...) call — the row's extras. */
function extraOf(row: any[]): any {
  return row[row.length - 1];
}

/** LanguageModel V2 — flat usage, doGenerate + doStream. */
class FakeModel {
  readonly specificationVersion = "v2";
  readonly provider: string;
  readonly modelId: string;
  config = { baseURL: "https://api.openai.com/v1" };
  doGenerateCalls = 0;
  constructor(modelId: string, provider = "openai") {
    this.modelId = modelId;
    this.provider = provider;
  }
  async doGenerate(_options: any): Promise<any> {
    this.doGenerateCalls += 1;
    return {
      content: [{ type: "text", text: "hi there" }],
      finishReason: "stop",
      usage: { inputTokens: 10, outputTokens: 5 },
    };
  }
  async doStream(_options: any): Promise<any> {
    const parts = [
      { type: "text-delta", id: "1", delta: "hi" },
      { type: "finish", finishReason: "stop", usage: { inputTokens: 10, outputTokens: 5 } },
    ];
    return {
      stream: new ReadableStream({
        start(controller) {
          for (const p of parts) controller.enqueue(p);
          controller.close();
        },
      }),
    };
  }
}

class FakeEmbeddingModel {
  readonly specificationVersion = "v2";
  readonly provider = "openai.embedding";
  readonly modelId: string;
  config = { baseURL: "https://api.openai.com/v1" };
  constructor(modelId: string) {
    this.modelId = modelId;
  }
  async doEmbed(_options: any): Promise<any> {
    return { embeddings: [[0.1, 0.2]], usage: { tokens: 12 } };
  }
}

/** Modality (image) model — same shape aiSdk.test.ts's FakeImageModel uses;
 * `maxImagesPerCall` is what `_detectAiSdkModelKind` keys the "image" kind on. */
class FakeImageModel {
  readonly specificationVersion = "v3";
  readonly provider = "openai.image";
  readonly modelId: string;
  readonly maxImagesPerCall = 10;
  config = { baseURL: "https://api.openai.com/v1" };
  constructor(modelId: string) {
    this.modelId = modelId;
  }
  async doGenerate(_options: any): Promise<any> {
    return { images: ["aGk="], warnings: [], usage: {} };
  }
}

async function flushLogs(): Promise<void> {
  await new Promise((r) => setTimeout(r, 0));
}

beforeEach(() => {
  h.logged.length = 0;
  // Drain any leftover observations from a prior test / suite.
  try {
    state.drainObservations();
  } catch {
    /* */
  }
});

afterEach(() => {
  uninstrument();
  vi.restoreAllMocks();
});

describe("Vercel AI SDK seam — B7: same-provider ENFORCE reroute now resolves", () => {
  it("doGenerate: exactly ONE reroute_rejected observation on the logged row; instance unmutated; no local_decision/_tp_routing; result reaches the customer", async () => {
    const checkSpy = vi.fn(async () => enforceReroute("gpt-4o-mini"));
    h.clientBox.client = makeClient({ check: checkSpy });
    autoInstrument({ aiSdkProviders: [new FakeModel("probe")] });
    const model = new FakeModel("gpt-4o", "openai");

    let capturedSession: any;
    const res = await session({ name: "wf" }, async (sess) => {
      capturedSession = sess;
      return model.doGenerate({ prompt: [] });
    });
    await flushLogs();

    // Golden rule: nothing thrown, the provider result reaches the customer.
    expect(res.finishReason).toBe("stop");
    expect(model.modelId).toBe("gpt-4o"); // instance field never mutated — no swap target to mutate
    expect((capturedSession.metadata as Record<string, unknown> | undefined)?._tp_routing).toBeUndefined();

    expect(h.logged).toHaveLength(1);
    const extra = extraOf(h.logged[0]);
    expect(extra.local_decision).toBeUndefined(); // rejected, never applied — no REQUEST_REROUTED provenance
    expect(extra.observations).toHaveLength(1);
    const rejected = extra.observations[0];
    expect(rejected.outcome).toBe("reroute_rejected");
    expect(rejected.rejection_reason).toBe("unappliable_call_shape");
    expect(rejected.reroute.from.model).toBe("gpt-4o"); // modelHint, never a swap target
    expect(rejected.reroute.to.model).toBe("gpt-4o-mini");
    expect(rejected.reroute.to.provider).toBe("openai");
    expect(rejected.rule_id).toBe("rule_b7");
  });

  it("doStream: same-provider ENFORCE directive → one reroute_rejected observation on the logged row; every chunk still reaches the customer in order", async () => {
    const checkSpy = vi.fn(async () => enforceReroute("gpt-4o-mini"));
    h.clientBox.client = makeClient({ check: checkSpy });
    autoInstrument({ aiSdkProviders: [new FakeModel("probe")] });
    const model = new FakeModel("gpt-4o", "openai");

    const res = await session({ name: "wf" }, () => model.doStream({ prompt: [] }));
    const seen: any[] = [];
    for await (const chunk of res.stream as any) seen.push(chunk);
    await flushLogs();

    expect(seen.map((c) => c.type)).toEqual(["text-delta", "finish"]); // stream untouched
    expect(model.modelId).toBe("gpt-4o");

    expect(h.logged).toHaveLength(1);
    const extra = extraOf(h.logged[0]);
    expect(extra.local_decision).toBeUndefined();
    expect(extra.observations).toHaveLength(1);
    expect(extra.observations[0].outcome).toBe("reroute_rejected");
    expect(extra.observations[0].rejection_reason).toBe("unappliable_call_shape");
    expect(extra.observations[0].reroute.to.model).toBe("gpt-4o-mini");
  });

  it("doStream: per-call attribution — two concurrent streams in one session, only ONE rerouted, never cross-attributed onto the sibling row", async () => {
    // Call A's directive targets its own model; call B's /check returns a
    // plain allow. The keyed obs-queue (per-call ALS scope, see
    // _withCallObsScope / state.runWithObsKey — the same mechanism
    // rerouteMarkerRowScope.test.ts / observationAttribution.test.ts pin)
    // must attach A's rejection to A's own logged row only — never onto B's,
    // never dropped, never duplicated onto both.
    const checkSpy = vi.fn(async (...args: any[]) => {
      const targetModel = args[6];
      if (targetModel === "gpt-4o") return enforceReroute("gpt-4o-mini");
      return { status: "allowed" };
    });
    h.clientBox.client = makeClient({ check: checkSpy });
    autoInstrument({ aiSdkProviders: [new FakeModel("probe")] });
    const rerouted = new FakeModel("gpt-4o", "openai");
    const plain = new FakeModel("gpt-4o-mini-source", "openai");

    await session({ name: "wf" }, async () => {
      const [resA, resB] = await Promise.all([
        rerouted.doStream({ prompt: [] }),
        plain.doStream({ prompt: [] }),
      ]);
      for await (const _c of resA.stream as any) void _c;
      for await (const _c of resB.stream as any) void _c;
    });
    await flushLogs();

    expect(h.logged).toHaveLength(2);
    // args[4] is the logged `model` field (see _logManual's tp.log(...) call).
    const rerouteRow = h.logged.find((row) => row[4] === "gpt-4o");
    const plainRow = h.logged.find((row) => row[4] === "gpt-4o-mini-source");
    expect(rerouteRow).toBeDefined();
    expect(plainRow).toBeDefined();

    const rerouteExtra = extraOf(rerouteRow!);
    expect(rerouteExtra.observations).toHaveLength(1);
    expect(rerouteExtra.observations[0].outcome).toBe("reroute_rejected");
    expect(rerouteExtra.observations[0].reroute.to.model).toBe("gpt-4o-mini");

    const plainExtra = extraOf(plainRow!);
    expect(plainExtra.observations).toBeUndefined(); // never leaked onto the sibling call
  });

  it("doEmbed: same-provider ENFORCE directive → one reroute_rejected observation on the logged row; embeddings still returned", async () => {
    const checkSpy = vi.fn(async () => enforceReroute("text-embedding-3-large"));
    h.clientBox.client = makeClient({ check: checkSpy });
    autoInstrument({ aiSdkProviders: [new FakeEmbeddingModel("probe")] });
    const model = new FakeEmbeddingModel("text-embedding-3-small");

    const res = await session({ name: "wf" }, () => model.doEmbed({ values: ["a"] }));
    await flushLogs();

    expect(res.embeddings.length).toBe(1);
    expect(model.modelId).toBe("text-embedding-3-small");

    expect(h.logged).toHaveLength(1);
    const extra = extraOf(h.logged[0]);
    expect(extra.local_decision).toBeUndefined();
    expect(extra.observations).toHaveLength(1);
    expect(extra.observations[0].outcome).toBe("reroute_rejected");
    expect(extra.observations[0].rejection_reason).toBe("unappliable_call_shape");
    expect(extra.observations[0].reroute.from.model).toBe("text-embedding-3-small");
    expect(extra.observations[0].reroute.to.model).toBe("text-embedding-3-large");
  });

  // The AI SDK State-B fallback only ever emits `would_reroute` on the
  // combination of the SDK-level dry_run dial + an ENFORCE-mode rule (see
  // `_runAsyncCheck`'s "State B fallback" branch) — the try-before-enforce
  // funnel. State A's local-evaluator dry_run branch is unreachable from this
  // seam in these tests (it requires `deployment: "daemon"` + a healthy
  // pack), so this is the one path this seam can actually exercise.
  it("SDK dry_run dial + ENFORCE-mode rule directive → would_reroute observation, NO rejection, NO mutation", async () => {
    const checkSpy = vi.fn(async () => enforceReroute("gpt-4o-mini"));
    h.clientBox.client = makeClient({ firewall: "dry_run", check: checkSpy });
    autoInstrument({ aiSdkProviders: [new FakeModel("probe")] });
    const model = new FakeModel("gpt-4o", "openai");

    const res = await session({ name: "wf" }, () => model.doGenerate({ prompt: [] }));
    await flushLogs();

    expect(res.finishReason).toBe("stop");
    expect(model.modelId).toBe("gpt-4o"); // no mutation

    expect(h.logged).toHaveLength(1);
    const extra = extraOf(h.logged[0]);
    expect(extra.local_decision).toBeUndefined(); // dry_run never applies
    expect(extra.observations).toHaveLength(1);
    expect(extra.observations[0].outcome).toBe("would_reroute");
    expect(extra.observations[0].reroute.to.model).toBe("gpt-4o-mini");
    expect(extra.observations[0].rule_id).toBe("rule_b7");
  });

  it("BLOCK: a blocked /check decision still throws TokenPoliceBlockedError and never calls the provider (unchanged behavior)", async () => {
    const checkSpy = vi.fn(async () => ({ status: "blocked", reason: "Budget exceeded" }));
    h.clientBox.client = makeClient({ check: checkSpy });
    autoInstrument({ aiSdkProviders: [new FakeModel("probe")] });
    const model = new FakeModel("gpt-4o", "openai");

    await expect(
      session({ name: "wf" }, () => model.doGenerate({ prompt: [] })),
    ).rejects.toBeInstanceOf(TokenPoliceBlockedError);
    expect(model.doGenerateCalls).toBe(0); // provider never reached
  });

  it("/check payload unchanged: targetModel/provider forwarded via the modelHint fallback exactly as the old throwaway body would have", async () => {
    const checkSpy = vi.fn(async () => ({ status: "allowed" }));
    h.clientBox.client = makeClient({ check: checkSpy });
    autoInstrument({ aiSdkProviders: [new FakeModel("probe")] });
    const model = new FakeModel("claude-3-5-sonnet", "anthropic.messages");

    await session({ name: "wf" }, () => model.doGenerate({ prompt: [] }));
    await flushLogs();

    expect(checkSpy).toHaveBeenCalledTimes(1);
    const args = checkSpy.mock.calls[0];
    // args: (userId, paidPlan, workflowName, sessionId, metadata, traceId,
    //        targetModel, provider, intent, planSource)
    expect(args[6]).toBe("claude-3-5-sonnet"); // targetModel === modelHint fallback (body is null)
    expect(args[7]).toBe("anthropic"); // provider === meta.provider ("anthropic.messages" head)
  });

  it("modality (doGenerate image): same-provider ENFORCE directive → one reroute_rejected observation on the logged row; image still generated", async () => {
    const checkSpy = vi.fn(async () => enforceReroute("dall-e-2"));
    h.clientBox.client = makeClient({ check: checkSpy });
    autoInstrument({ aiSdkProviders: [new FakeImageModel("probe")] });
    const model = new FakeImageModel("dall-e-3");

    const res = await session({ name: "wf" }, () => model.doGenerate({ prompt: "a cat", n: 1 }));
    await flushLogs();

    expect(res.images.length).toBe(1);
    expect(model.modelId).toBe("dall-e-3");

    expect(h.logged).toHaveLength(1);
    const extra = extraOf(h.logged[0]);
    expect(extra.local_decision).toBeUndefined();
    expect(extra.observations).toHaveLength(1);
    expect(extra.observations[0].outcome).toBe("reroute_rejected");
    expect(extra.observations[0].rejection_reason).toBe("unappliable_call_shape");
    expect(extra.observations[0].reroute.from.model).toBe("dall-e-3");
    expect(extra.observations[0].reroute.to.model).toBe("dall-e-2");
  });
});
