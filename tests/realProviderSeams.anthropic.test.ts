/**
 * REAL-PACKAGE seam contract — `@anthropic-ai/sdk`.
 *
 * See tests/helpers/realProviderSeam.ts for why this family exists. Before
 * this file every Node anthropic test drove a hand-built fake of the Stainless
 * export shape, so `@anthropic-ai/sdk` was marked `fake-only` in
 * provider-drift-suite/catalog.json and skipped entirely.
 *
 * Three seams are covered, because they fail for different reasons:
 *   - `Anthropic.Messages.prototype.create` — Mode A: pre-flight /check plus
 *     Traceloop-span telemetry.
 *   - `Anthropic.Beta.Messages.prototype.create` — a SIBLING class (extends
 *     APIResource, NOT Messages), so patching Messages does not reach it. If
 *     the vendor ever makes Beta.Messages inherit from Messages this row would
 *     start double-wrapping; if it moves, `client.beta.messages.*` silently
 *     loses ALL pre-flight enforcement.
 *   - `messages.stream()` / the `APIPromise` surface — from
 *     @anthropic-ai/sdk 0.35 the vendor's own MessageStream awaits
 *     `messages.create(...).withResponse()`. Our wrapper is an async function,
 *     which collapses APIPromise into a plain Promise, so
 *     `_preserveApiPromiseSurface` re-attaches that surface. GOLDEN RULE: if
 *     that shim stops matching the vendor, `client.messages.stream()` throws
 *     INTO CUSTOMER CODE on an allowed call. That is the single highest-stakes
 *     seam in the SDK and it was only ever exercised against a fake.
 */
import {
  describe,
  it,
  expect,
  beforeAll,
  afterAll,
  beforeEach,
  afterEach,
} from "vitest";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { trace, context, propagation } from "@opentelemetry/api";

import { autoInstrument, uninstrument } from "../src/enforcer";
import { setupOpenTelemetry, unsetupOpenTelemetry } from "../src/telemetry";
import {
  createSeamHarness,
  flushLogs,
  installNoNetworkGuard,
  installedVersion,
  jsonResponder,
  methodNames,
  requireCjs,
  snapshotMethods,
  sseResponder,
  targetLabel,
  targetsFor,
  walkPath,
} from "./helpers/realProviderSeam";

/** Cross-language contract shared with the Python SDK — see sdkUsageParity.test.ts. */
const parityFixture = JSON.parse(
  readFileSync(
    join(dirname(fileURLToPath(import.meta.url)), "../../shared/sdk-usage-parity-fixture.json"),
    "utf8",
  ),
);

const PKG = "@anthropic-ai/sdk";
const VERSION = installedVersion(PKG);
const anthropicModule = (() => {
  try {
    return requireCjs(PKG);
  } catch {
    return undefined;
  }
})();
const maybe = anthropicModule ? it : it.skip;
const Anthropic: any = anthropicModule?.Anthropic ?? anthropicModule?.default;

const h = createSeamHarness();
let restoreNetwork: () => void;

beforeAll(() => {
  restoreNetwork = installNoNetworkGuard();
  console.log(`[realProviderSeams] @anthropic-ai/sdk@${VERSION ?? "ABSENT"}`);
});
afterAll(() => restoreNetwork());

beforeEach(() => h.install());
afterEach(async () => {
  await flushLogs(150);
  try {
    unsetupOpenTelemetry();
    trace.disable();
    context.disable();
    propagation.disable();
  } catch {
    /* best effort */
  }
  uninstrument();
  h.reset();
});

/** Real client, stubbed transport, placeholder credential. */
function makeClient(fetchStub: (...a: any[]) => Promise<Response>): any {
  return new Anthropic({ apiKey: "sk-ant-test-not-a-real-key", fetch: fetchStub });
}

const MESSAGE = (usage: Record<string, unknown>, model = "claude-sonnet-4-5") => ({
  id: "msg_seam",
  type: "message",
  role: "assistant",
  model,
  content: [{ type: "text", text: "hi" }],
  stop_reason: "end_turn",
  stop_sequence: null,
  usage,
});

// ═══════════════════════════════════════════════════════════════════════
// Layer 1 — SHAPE
// ═══════════════════════════════════════════════════════════════════════
describe("@anthropic-ai/sdk — shape: the enforcer's patch targets exist", () => {
  const targets = targetsFor(PKG);

  it("the registry still declares the Messages and Beta.Messages rows", () => {
    expect(targets.map(targetLabel).sort()).toEqual([
      "@anthropic-ai/sdk Anthropic.Beta.Messages.prototype.create",
      "@anthropic-ai/sdk Anthropic.Messages.prototype.create",
    ]);
  });

  maybe("the package root exposes the Anthropic class _pickClassExport resolves", () => {
    expect(typeof Anthropic).toBe("function");
  });

  maybe("APIPromise is exported at the root (gates the streaming bypass)", () => {
    // `_wrapMethod` reads `mod.APIPromise ?? mod.default.APIPromise ??
    // mod.Anthropic.APIPromise`; when ALL THREE are missing it flips to the
    // manual `anthropicStreamBypass` path. That fallback is correct but loses
    // the Traceloop span, so knowing which branch a released version takes
    // matters — pin the export here rather than discovering it in production.
    const found =
      anthropicModule.APIPromise ??
      anthropicModule.default?.APIPromise ??
      anthropicModule.Anthropic?.APIPromise;
    expect(typeof found).toBe("function");
  });

  maybe("Beta.Messages is a SIBLING of Messages, not a subclass", () => {
    // The comment on the Beta row in _TARGET_METHODS asserts this (verified in
    // 0.30.1 and 0.90.0) and the row only exists BECAUSE of it. If the vendor
    // ever makes Beta.Messages extend Messages, patching both prototypes would
    // double-wrap `client.beta.messages.create` — two /check calls and two
    // rows per call. Fail here instead of in a customer's bill.
    expect(Anthropic.Beta.Messages.prototype).not.toBe(Anthropic.Messages.prototype);
    expect(Anthropic.Messages.prototype.isPrototypeOf(Anthropic.Beta.Messages.prototype)).toBe(
      false,
    );
  });

  for (const t of targets) {
    maybe(`${targetLabel(t)} resolves and is a function`, () => {
      const { obj, brokeAt } = walkPath(anthropicModule, t.objectPath);
      expect(
        obj,
        `objectPath [${t.objectPath.join(", ")}] broke at "${brokeAt}"`,
      ).toBeTruthy();
      expect(
        typeof obj[t.method],
        `method "${t.method}" is gone; object carries: ${methodNames(obj).join(", ")}`,
      ).toBe("function");
    });
  }

  maybe("Messages.prototype.stream still exists (the MessageStream helper seam)", () => {
    // `_instrumentAnthropicStream` patches this; a rename would silently drop
    // pre-flight enforcement for every `client.messages.stream()` caller.
    expect(typeof Anthropic.Messages.prototype.stream).toBe("function");
  });

  maybe("autoInstrument() replaces every declared anthropic target", () => {
    const before = snapshotMethods(anthropicModule, targets);
    autoInstrument();
    for (const t of targets) {
      const { obj } = walkPath(anthropicModule, t.objectPath);
      expect(
        obj?.[t.method],
        `${targetLabel(t)} was NOT wrapped — this seam runs with no pre-flight /check`,
      ).not.toBe(before.get(targetLabel(t)));
    }
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Layer 2 — ROUND TRIP
// ═══════════════════════════════════════════════════════════════════════
describe("@anthropic-ai/sdk — round trip: messages.create (Mode A)", () => {
  maybe("non-stream reports the wire's exact tokens and the parity cache keys", async () => {
    // The wire numbers come from shared/sdk-usage-parity-fixture.json so a real
    // anthropic + real Traceloop pipeline is checked against the SAME contract
    // the Python SDK is checked against (see sdkUsageParity.test.ts).
    const attrs = parityFixture.input_attrs as Record<string, number>;
    const expected = parityFixture.expected_usage_raw as Record<string, any>;

    setupOpenTelemetry({ anthropic: anthropicModule } as any);
    autoInstrument({ anthropic: anthropicModule } as any);

    await makeClient(
      jsonResponder(
        MESSAGE({
          input_tokens: attrs["gen_ai.usage.input_tokens"],
          output_tokens: attrs["gen_ai.usage.output_tokens"],
          cache_read_input_tokens: attrs["gen_ai.usage.cache_read.input_tokens"],
          cache_creation_input_tokens: attrs["gen_ai.usage.cache_creation.input_tokens"],
        }),
      ),
    ).messages.create({
      model: "claude-sonnet-4-5",
      max_tokens: 16,
      messages: [{ role: "user", content: "hello" }],
    });
    await flushLogs(150);

    const row = h.only((l) => l.provider === "anthropic", "anthropic chat");
    expect(row.model).toBe("claude-sonnet-4-5");
    expect(row.outputTokens).toBe(attrs["gen_ai.usage.output_tokens"]);
    expect(row.cachedTokens).toBe(attrs["gen_ai.usage.cache_read.input_tokens"]);
    // G5 in the fixture: cache READ and cache CREATION must stay disjoint —
    // a version that folds one into the other would double-bill cache writes.
    const raw = row.extra?.usage?.raw ?? {};
    expect(raw.cache_read_input_tokens).toBe(expected.cache_read_input_tokens);
    expect(raw.cache_creation_input_tokens).toBe(expected.cache_creation_input_tokens);
    expect(raw.cache_read_input_tokens).not.toBe(raw.cache_creation_input_tokens);
  });

  maybe("stream reports usage merged from message_start and message_delta", async () => {
    setupOpenTelemetry({ anthropic: anthropicModule } as any);
    autoInstrument({ anthropic: anthropicModule } as any);

    const stream: any = await makeClient(
      sseResponder(
        [
          {
            type: "message_start",
            message: MESSAGE({ input_tokens: 44, output_tokens: 1 }),
          },
          { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
          { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "hi" } },
          { type: "content_block_stop", index: 0 },
          {
            type: "message_delta",
            delta: { stop_reason: "end_turn", stop_sequence: null },
            usage: { output_tokens: 9 },
          },
          { type: "message_stop" },
        ],
        { eventNames: true },
      ),
    ).messages.create({
      model: "claude-sonnet-4-5",
      max_tokens: 16,
      messages: [{ role: "user", content: "hello" }],
      stream: true,
    });
    for await (const _ of stream) {
      /* drain */
    }
    await flushLogs(200);

    const row = h.only((l) => l.provider === "anthropic", "anthropic chat stream");
    // Anthropic splits usage across two events: input on message_start, the
    // real output count on message_delta. A version that stops emitting either
    // half shows up here as a wrong number, not as a missing row.
    expect(row.inputTokens).toBe(44);
    expect(row.outputTokens).toBe(9);
  });
});

describe("@anthropic-ai/sdk — round trip: beta.messages.create", () => {
  maybe("the sibling Beta seam still runs the pre-flight /check", async () => {
    autoInstrument();
    await makeClient(
      jsonResponder(MESSAGE({ input_tokens: 12, output_tokens: 3 })),
    ).beta.messages.create({
      model: "claude-sonnet-4-5",
      max_tokens: 16,
      messages: [{ role: "user", content: "hello" }],
    });
    await flushLogs(50);

    // This row is enforcement-only by design (telemetry already comes from the
    // Traceloop beta instrumentation), so the observable contract is the
    // pre-flight, not a log.
    expect(
      h.checks.length,
      "beta.messages.create ran with NO pre-flight /check — no block, no " +
        "reroute, no budget gate for every client.beta.messages caller",
    ).toBeGreaterThan(0);
  });
});

describe("@anthropic-ai/sdk — GOLDEN RULE: messages.stream() must not throw", () => {
  maybe("the vendor MessageStream still resolves through our APIPromise shim", async () => {
    autoInstrument();
    const client = makeClient(
      sseResponder(
        [
          { type: "message_start", message: MESSAGE({ input_tokens: 7, output_tokens: 1 }) },
          { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
          { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "hi" } },
          { type: "content_block_stop", index: 0 },
          {
            type: "message_delta",
            delta: { stop_reason: "end_turn", stop_sequence: null },
            usage: { output_tokens: 2 },
          },
          { type: "message_stop" },
        ],
        { eventNames: true },
      ),
    );

    // The vendor's MessageStream internally does
    // `messages.create({...}, opts).withResponse()`. With the wrapper installed
    // and the shim broken this line throws
    // "messages.create(...).withResponse is not a function" — a TokenPolice
    // failure surfacing as a customer-visible error on an ALLOWED call.
    const stream = client.messages.stream({
      model: "claude-sonnet-4-5",
      max_tokens: 16,
      messages: [{ role: "user", content: "hello" }],
    });
    let text = "";
    for await (const ev of stream) {
      if (ev?.type === "content_block_delta" && ev.delta?.type === "text_delta") {
        text += ev.delta.text;
      }
    }
    const final = await stream.finalMessage();

    expect(text).toBe("hi");
    expect(final?.id).toBe("msg_seam");

    // `.withResponse()` must ALSO be callable on the stream itself; the shim
    // hands back a synthetic Response rather than null precisely because the
    // vendor hard-throws "Could not resolve a 'Response' object" on a falsy one.
    await expect(stream.withResponse()).resolves.toBeTruthy();
  });
});
