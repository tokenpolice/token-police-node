/**
 * OTel SDK 1.x/2.x parent-id compatibility (Node-only).
 *
 * The span processor reads the ended span's parent id to link spans (orphan
 * rewrite, tree hierarchy). OTel JS SDK 1.x exposed it as
 * `ReadableSpan.parentSpanId`; SDK 2.x REMOVED that field and moved the id to
 * `parentSpanContext.spanId`. Both are 16-char hex strings. These tests drive
 * the real `onEnd` path and assert a 1.x-shaped span and a 2.x-shaped span
 * resolve the SAME `parent_span_id`, and that absence yields "" (never throws).
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { setClient } from "../src/state";
import { TPSession, _getSessionStorage } from "../src/context";
import { TokenPoliceSpanProcessor } from "../src/telemetry";

const PARENT_HEX = "1234567890abcdef"; // 16-char hex, not the zero-id sentinel

let logged: any[];
beforeEach(() => {
  logged = [];
  setClient({ log: (...args: any[]) => logged.push(args) } as any);
});
afterEach(() => {
  setClient(null as any);
});

/** An LLM span whose parent id is carried in `extra` (1.x vs 2.x shape). */
function llmSpan(session: TPSession, extra: Record<string, unknown>) {
  return {
    attributes: {
      "gen_ai.system": "openai",
      "gen_ai.request.model": "gpt-4.1-nano",
      "gen_ai.usage.input_tokens": 10,
      "gen_ai.usage.output_tokens": 5,
      "tp.trace_id": session.traceId,
      "tp.span_order": 0,
    },
    name: "openai.chat",
    startTime: [0, 0] as [number, number],
    endTime: [1, 0] as [number, number],
    spanContext: () => ({
      traceId: "0000000000000000000000000000abcd",
      spanId: "0000000000001234",
    }),
    status: { code: 0 },
    ...extra,
  } as any;
}

async function runOnEnd(session: TPSession, span: any): Promise<any[]> {
  await _getSessionStorage().run(session, async () => {
    new TokenPoliceSpanProcessor().onEnd(span);
    await new Promise((r) => setTimeout(r, 0));
  });
  return logged[0];
}

function newSession(): TPSession {
  return new TPSession({
    userId: "u1",
    paidPlan: "pro",
    workflowName: "wf",
    traceId: "a".repeat(32),
    rootSpanId: "b".repeat(16),
  });
}

// spanObj is positional arg index 10 of client.log() (see telemetry.ts tp.log call).
const SPAN_OBJ_ARG = 10;

describe("OTel 1.x/2.x parent-id compat", () => {
  it("1.x span (parentSpanId) resolves the parent id", async () => {
    const session = newSession();
    const args = await runOnEnd(session, llmSpan(session, { parentSpanId: PARENT_HEX }));
    expect(args[SPAN_OBJ_ARG].parent_span_id).toBe(PARENT_HEX);
  });

  it("2.x span (parentSpanContext.spanId) resolves the SAME parent id", async () => {
    const session = newSession();
    const args = await runOnEnd(
      session,
      llmSpan(session, { parentSpanContext: { traceId: "c".repeat(32), spanId: PARENT_HEX } }),
    );
    expect(args[SPAN_OBJ_ARG].parent_span_id).toBe(PARENT_HEX);
  });

  it("1.x and 2.x shapes are byte-identical", async () => {
    const oneX = await runOnEnd(newSession(), llmSpan(newSession(), { parentSpanId: PARENT_HEX }));
    const twoX = await runOnEnd(
      newSession(),
      llmSpan(newSession(), { parentSpanContext: { spanId: PARENT_HEX } }),
    );
    expect(oneX[SPAN_OBJ_ARG].parent_span_id).toBe(twoX[SPAN_OBJ_ARG].parent_span_id);
  });

  it("neither field present yields '' and never throws", async () => {
    const session = newSession();
    const args = await runOnEnd(session, llmSpan(session, {}));
    expect(args[SPAN_OBJ_ARG].parent_span_id).toBe("");
  });
});
