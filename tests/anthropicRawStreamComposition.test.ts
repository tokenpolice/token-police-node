/**
 * Anthropic raw `create({stream:true})` on the Traceloop-instrumented path
 * (@anthropic-ai/sdk ≥0.35 — root `APIPromise` export present, so
 * `anthropicStreamBypass` is off and `_tapStreamUsageForOnEnd` is the only
 * SDK-side observer of the event stream).
 *
 * Pre-fix, `_newStreamAccumulator("anthropic")` returned null, so the tap
 * stashed usage but no composition; telemetry.ts then fell back to the
 * instrumentor's `gen_ai.output.messages`, whose Anthropic mapper carries
 * tool_use blocks as `tool_call` parts that the fallback reader ignores
 * (text parts only). Observed on the 2026-09-02 pure-sdk run against 0.123.0
 * (trace 4b2d3286acaa1fb987d524c0b608093a): a streamed text+tool_use turn
 * logged a text-only response_composition, and a tool-only turn logged `[]`.
 *
 * Events below are verbatim shapes captured from @anthropic-ai/sdk 0.123.0
 * and 0.30.1 (identical on both).
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { setClient } from "../src/state";
import { TPSession, _getSessionStorage } from "../src/context";
import { __test__ as enforcerTest } from "../src/enforcer";
import { buildResponseComposition } from "../src/composition";

const {
  _newStreamAccumulator,
  _accumulateStreamChunk,
  _streamAccumulatorToResponse,
  _streamAccHasContent,
  _tapStreamUsageForOnEnd,
} = enforcerTest as any;

const TOOL_INPUT = { email: "pro-user@example.com" };

function textAndToolEvents(): any[] {
  return [
    { type: "message_start", message: { model: "claude-haiku-4-5-20251001", usage: { input_tokens: 591, output_tokens: 22 } } },
    { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
    { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Let me look " } },
    { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "that up." } },
    { type: "content_block_stop", index: 0 },
    { type: "content_block_start", index: 1, content_block: { type: "tool_use", id: "toolu_01T1kaGtMLS5XcqQfxCQat4o", name: "getCustomerInfo", input: {}, caller: { type: "direct" } } },
    { type: "content_block_delta", index: 1, delta: { type: "input_json_delta", partial_json: "" } },
    { type: "content_block_delta", index: 1, delta: { type: "input_json_delta", partial_json: "{\"emai" } },
    { type: "content_block_delta", index: 1, delta: { type: "input_json_delta", partial_json: "l\": " } },
    { type: "content_block_delta", index: 1, delta: { type: "input_json_delta", partial_json: "\"pro-user@" } },
    { type: "content_block_delta", index: 1, delta: { type: "input_json_delta", partial_json: "example.com\"}" } },
    { type: "content_block_stop", index: 1 },
    { type: "message_delta", delta: { stop_reason: "tool_use", stop_sequence: null }, usage: { input_tokens: 591, output_tokens: 61 } },
    { type: "message_stop" },
  ];
}

function toolOnlyEvents(): any[] {
  return textAndToolEvents().filter((e) => !(e.index === 0));
}

/** The non-streamed Message the same turn would have returned. */
function equivalentMessage(withText: boolean): any {
  return {
    model: "claude-haiku-4-5-20251001",
    stop_reason: "tool_use",
    usage: { input_tokens: 591, output_tokens: 61 },
    content: [
      ...(withText ? [{ type: "text", text: "Let me look that up." }] : []),
      { type: "tool_use", id: "toolu_01T1kaGtMLS5XcqQfxCQat4o", name: "getCustomerInfo", input: TOOL_INPUT },
    ],
  };
}

describe("anthropic raw-stream accumulator (Traceloop path)", () => {
  it("non-anthropic providers are untouched (null accumulator as before)", () => {
    expect(_newStreamAccumulator("google")).toBeNull();
    expect(_newStreamAccumulator("bedrock")).toBeNull();
    expect(_newStreamAccumulator("voyage")).toBeNull();
  });

  it("rebuilds text + tool_use blocks with the fully-joined input JSON", () => {
    const acc = _newStreamAccumulator("anthropic");
    expect(acc).not.toBeNull();
    for (const ev of textAndToolEvents()) _accumulateStreamChunk("anthropic", acc, ev);
    const resp = _streamAccumulatorToResponse("anthropic", acc);
    expect(resp).toEqual({
      content: [
        { type: "text", text: "Let me look that up." },
        { type: "tool_use", id: "toolu_01T1kaGtMLS5XcqQfxCQat4o", name: "getCustomerInfo", input: TOOL_INPUT },
      ],
    });
  });

  it("composition of the streamed turn fingerprints identically to the non-streamed Message", () => {
    for (const withText of [true, false]) {
      const acc = _newStreamAccumulator("anthropic");
      for (const ev of withText ? textAndToolEvents() : toolOnlyEvents()) {
        _accumulateStreamChunk("anthropic", acc, ev);
      }
      const streamed = buildResponseComposition("anthropic", _streamAccumulatorToResponse("anthropic", acc));
      const direct = buildResponseComposition("anthropic", equivalentMessage(withText));
      expect(streamed).toEqual(direct);
      expect(streamed.filter((e: any) => e.type === "tool_call")).toHaveLength(1);
      expect((streamed.find((e: any) => e.type === "tool_call") as any).name).toBe("getCustomerInfo");
    }
  });

  it("TTFT gate: no content until the first text/tool_use block; unknown block types are dropped", () => {
    const acc = _newStreamAccumulator("anthropic");
    _accumulateStreamChunk("anthropic", acc, textAndToolEvents()[0]);
    expect(_streamAccHasContent("anthropic", acc)).toBe(false);
    _accumulateStreamChunk("anthropic", acc, { type: "content_block_start", index: 0, content_block: { type: "thinking", thinking: "" } });
    _accumulateStreamChunk("anthropic", acc, { type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "hmm" } });
    expect(_streamAccHasContent("anthropic", acc)).toBe(false);
    _accumulateStreamChunk("anthropic", acc, { type: "content_block_start", index: 1, content_block: { type: "tool_use", id: "t1", name: "f", input: {} } });
    expect(_streamAccHasContent("anthropic", acc)).toBe(true);
    expect(_streamAccumulatorToResponse("anthropic", acc)).toEqual({ content: [{ type: "tool_use", id: "t1", name: "f", input: {} }] });
  });

  it("malformed events never throw (fail-open)", () => {
    const acc = _newStreamAccumulator("anthropic");
    for (const ev of [null, undefined, {}, { type: "content_block_delta" }, { type: "content_block_delta", index: "x", delta: null }, { type: "content_block_start", index: 3 }]) {
      expect(() => _accumulateStreamChunk("anthropic", acc, ev)).not.toThrow();
    }
    // delta-without-start opens a lazy tool_use slot; unparsable JSON keeps the raw string
    _accumulateStreamChunk("anthropic", acc, { type: "content_block_delta", index: 7, delta: { type: "input_json_delta", partial_json: "{\"a\":" } });
    expect(_streamAccumulatorToResponse("anthropic", acc)).toEqual({
      content: [{ type: "tool_use", id: "", name: "", input: "{\"a\":" }],
    });
  });
});

describe("_tapStreamUsageForOnEnd stashes the streamed anthropic composition", () => {
  let session: TPSession;
  beforeEach(() => {
    setClient({ log: () => {}, captureStreamUsage: true } as any);
    session = new TPSession({
      userId: "u1", paidPlan: "pro", workflowName: "wf",
      traceId: "a".repeat(32), rootSpanId: "b".repeat(16),
    });
  });
  afterEach(() => setClient(null as any));

  for (const [label, events, textEntries] of [
    ["text + tool_use", textAndToolEvents(), 1],
    ["tool_use only (was an EMPTY composition pre-fix)", toolOnlyEvents(), 0],
  ] as const) {
    it(label, async () => {
      async function* gen() { for (const e of events) yield e; }
      await _getSessionStorage().run(session, async () => {
        const tapped = _tapStreamUsageForOnEnd(
          "anthropic", gen(), session,
          [{ model: "claude-haiku-4-5", messages: [{ role: "user", content: "x" }], stream: true }],
          0, false,
        );
        const seen: any[] = [];
        for await (const c of tapped) seen.push(c);
        // customer sees every event untouched
        expect(seen).toEqual(events);

        const cd = (session as any)._pendingCompositions[`${session.traceId}:0`];
        expect(cd?.usage?.input_tokens).toBe(591);
        expect(cd?.usage?.output_tokens).toBe(61);
        const resp = cd?.response ?? [];
        expect(resp.filter((e: any) => e.type === "text")).toHaveLength(textEntries);
        const tc = resp.filter((e: any) => e.type === "tool_call");
        expect(tc).toHaveLength(1);
        expect(tc[0].name).toBe("getCustomerInfo");
        expect(cd?.latency?.ttft_ms ?? cd?.latency?.ttftMs ?? 0).toBeGreaterThanOrEqual(0);
      });
    });
  }
});
