/**
 * Tests for patchLangChainToolNaming.
 *
 * @traceloop/instrumentation-langchain's handleToolStart names tool spans from
 * `tool.id[last]` (the LangChain serialization CLASS PATH), so JS tools created
 * as DynamicStructuredTool/StructuredTool are mislabeled with the class name
 * (e.g. "DynamicStructuredTool") instead of their real name (e.g.
 * "getCustomerInfo"). The real name is passed as the `runName` arg (which
 * @langchain/core defaults to the tool's name) but the instrumentor discards it.
 *
 * The patch wraps handleToolStart and, when a better name is available,
 * corrects the span name + traceloop.entity.name / gen_ai.tool.name. Guarded:
 * if no better name exists, upstream behavior is unchanged.
 */
import { describe, it, expect } from "vitest";
import { patchLangChainToolNaming } from "../src/telemetry";

/** A fake OTel span recording updateName / setAttribute calls. */
function makeFakeSpan(name: string) {
  return {
    name,
    attrs: {} as Record<string, unknown>,
    updateName(n: string) {
      this.name = n;
    },
    setAttribute(k: string, v: unknown) {
      this.attrs[k] = v;
    },
  };
}

/** Mirrors the upstream TraceloopCallbackHandler.handleToolStart behavior. */
class FakeHandler {
  spans = new Map<string, { span: ReturnType<typeof makeFakeSpan> }>();
  tracer = {
    startSpan: (name: string) => makeFakeSpan(name),
  };
  async handleToolStart(
    tool: any,
    _input: any,
    runId: string,
    _parentRunId?: any,
    _tags?: any,
    _metadata?: any,
    _runName?: any,
  ) {
    const toolName = tool?.id?.[tool.id.length - 1] || "unknown";
    const span = this.tracer.startSpan(`execute_tool ${toolName}`);
    span.setAttribute("traceloop.span.kind", "task");
    span.setAttribute("traceloop.entity.name", toolName);
    this.spans.set(runId, { span });
  }
}

const DST_TOOL = { id: ["langchain", "tools", "DynamicStructuredTool"] };

describe("patchLangChainToolNaming", () => {
  it("overrides the class name with the real name from runName", async () => {
    patchLangChainToolNaming(FakeHandler);
    const h = new FakeHandler();
    await h.handleToolStart(DST_TOOL, { customerId: 1 }, "run-1", undefined, undefined, undefined, "getCustomerInfo");
    const span = h.spans.get("run-1")!.span;
    expect(span.name).toBe("execute_tool getCustomerInfo");
    expect(span.attrs["traceloop.entity.name"]).toBe("getCustomerInfo");
    expect(span.attrs["gen_ai.tool.name"]).toBe("getCustomerInfo");
  });

  it("leaves the span unchanged when no better name is available", async () => {
    patchLangChainToolNaming(FakeHandler);
    const h = new FakeHandler();
    // No runName, and tool is a non-serializable class path → nothing better.
    await h.handleToolStart(DST_TOOL, {}, "run-2");
    const span = h.spans.get("run-2")!.span;
    expect(span.name).toBe("execute_tool DynamicStructuredTool");
    expect(span.attrs["traceloop.entity.name"]).toBe("DynamicStructuredTool");
    expect(span.attrs["gen_ai.tool.name"]).toBeUndefined();
  });

  it("does not override when runName equals the class name (no real signal)", async () => {
    patchLangChainToolNaming(FakeHandler);
    const h = new FakeHandler();
    await h.handleToolStart(DST_TOOL, {}, "run-3", undefined, undefined, undefined, "DynamicStructuredTool");
    const span = h.spans.get("run-3")!.span;
    expect(span.name).toBe("execute_tool DynamicStructuredTool");
  });

  it("is idempotent (double patch) and reports success", () => {
    expect(patchLangChainToolNaming(FakeHandler)).toBe(true);
    expect(patchLangChainToolNaming(FakeHandler)).toBe(true);
  });

  it("is fail-safe on bad input — never throws", () => {
    expect(patchLangChainToolNaming(undefined)).toBe(false);
    expect(patchLangChainToolNaming({})).toBe(false);
  });
});
