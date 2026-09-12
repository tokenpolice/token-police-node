/**
 * B4 — row scope of the applied-reroute provenance marker (`_tp_routing`).
 *
 * Bug: `_applyReroute` writes `_tp_routing` onto `session.metadata` (a
 * SESSION-wide field, unchanged by this fix — it's the /check-payload input
 * rules may match on). Every row-emission site used to copy session metadata
 * WHOLESALE into its `/log` payload, so ONE rerouted call's marker rode
 * EVERY later row of that session: tool rows (no model call at all),
 * agent/chain structural anchors, and unrelated sibling calls' rows.
 *
 * Fix under test: `src/routingMarkerStore.ts`'s per-call keyed store, wired
 * through `copySessionMetadata` (strip on every row-metadata copy) /
 * `stampRoutingMarker` (re-add ONLY on the owning model-call row, by exact
 * obs-key match) / `rowMetadataFromSession` (the local-block + failure log
 * sites that used to hand `session.metadata` to `tp.log` by reference).
 *
 * These tests drive the REAL functions (`_applyReroute`, `_logLlamaIndex`,
 * `_emitCallFailureLog`, `toolSpan`, `emitToolRow`,
 * `TokenPoliceSpanProcessor`) through the real session ALS
 * (`session()`/`tpSession`) and the real per-call obs-key scope
 * (`runWithCallObsScope`) — no network, fully offline, deterministic.
 *
 * Coverage (numbering matches the B4 test plan):
 *   (8)  rerouted call's own llm row CONTAINS _tp_routing (correct
 *        original/actual model) AND session.metadata._tp_routing remains set
 *   (9)  tool row after reroute: none — via BOTH tool-row call sites
 *        (context.ts `toolSpan`, frameworkTools.ts `emitToolRow`)
 *        [(9) FAILS IF THE FIX IS REVERTED — see the comment on that test]
 *   (10) agent/chain structural row: none
 *   (11) subsequent NON-rerouted call's row: none
 *        [(11) FAILS IF THE FIX IS REVERTED — see the comment on that test]
 *   (12) two concurrent calls, one rerouted: rerouted row carries its OWN
 *        marker, sibling carries none
 *   (13, LOW) rerouted call's FAILURE row keeps the marker
 *   (14) customer metadata keys survive on every row (folded into 8/9/10/11)
 */
import { describe, it, expect, vi, afterEach } from "vitest";
import { session as tpSession, toolSpan } from "../src/context";
import { emitToolRow } from "../src/frameworkTools";
import {
  runWithCallObsScope,
  getCurrentObsKey,
  setClient,
  resetPack,
} from "../src/state";
import { TokenPolice } from "../src/client";
import { __test__ as enforcerTest } from "../src/enforcer";
import { TokenPoliceSpanProcessor } from "../src/telemetry";
import { TP_ROUTING_ATTR } from "../src/routingMarkerStore";

const { _applyReroute, _logLlamaIndex, _emitCallFailureLog } = enforcerTest;

// `tp.log` positional signature (client.ts):
// (userId, paidPlan, workflowName, sessionId, model, provider,
//  inputTokens, outputTokens, cachedTokens, metadata, span,
//  promptComposition, responseComposition, extras)
const METADATA_ARG = 9;
const SPAN_ARG = 10;

// Provider-class stand-in: `_llamaIndexProvider()` keys off constructor.name;
// the exact resolved provider/model don't matter for these assertions.
class OpenAIStandIn {
  model = "gpt-4o-mini";
}

function makeClient(): TokenPolice {
  const client = new TokenPolice({
    apiKey: "tp_sk_test_b4_row_scope",
    baseUrl: "http://localhost:59999",
    timeout: 0.1,
    firewall: "off",
    deployment: "daemon",
  } as never);
  setClient(client);
  return client;
}

const rerouteDirective = {
  reroute: {
    mode: "enforce",
    model: "gpt-4o-mini",
    provider: "openai",
    rule_id: "rule_rr",
    original: { provider: "openai", model: "gpt-4o" },
  },
};

afterEach(() => {
  vi.restoreAllMocks();
  setClient(null as unknown as TokenPolice);
  resetPack();
});

// ══════════════════════════════════════════════════════════════════════════
// (8) rerouted call's own llm row — CONTAINS the marker
// ══════════════════════════════════════════════════════════════════════════
describe("(8) rerouted call's own llm row", () => {
  it("carries _tp_routing with the correct original/actual model+provider; session.metadata._tp_routing remains set", async () => {
    const client = makeClient();
    const logged: unknown[][] = [];
    vi.spyOn(client, "log").mockImplementation((...args: unknown[]) => {
      logged.push(args);
    });

    let capturedSessionMeta: Record<string, unknown> | undefined;
    await tpSession(
      { userId: "u1", metadata: { customer_key: "keep-me" } },
      async (sess) => {
        await runWithCallObsScope(async () => {
          const key = getCurrentObsKey() ?? null;
          const body: Record<string, unknown> = { model: "gpt-4o" };
          const status = _applyReroute(rerouteDirective, body, "openai");
          expect(status).toBe("applied");
          expect(body.model).toBe("gpt-4o-mini");
          _logLlamaIndex(new OpenAIStandIn(), { raw: {} }, 0, "call1", new Date(), key);
        });
        capturedSessionMeta = sess.metadata as Record<string, unknown>;
      },
    );

    expect(logged).toHaveLength(1);
    const metadata = logged[0][METADATA_ARG] as Record<string, unknown>;
    const routing = metadata._tp_routing as Record<string, unknown>;
    expect(routing).toBeTruthy();
    expect(routing.rule_id).toBe("rule_rr");
    expect(routing.original_model).toBe("gpt-4o");
    expect(routing.actual_model).toBe("gpt-4o-mini");
    expect(routing.original_provider).toBe("openai");
    expect(routing.actual_provider).toBe("openai");
    // (14) customer metadata key survives alongside the marker.
    expect(metadata.customer_key).toBe("keep-me");

    // The session write is UNCHANGED — still the back-compat /check input.
    expect(capturedSessionMeta).toBeTruthy();
    expect((capturedSessionMeta as Record<string, unknown>)._tp_routing).toBeTruthy();
  });
});

// ══════════════════════════════════════════════════════════════════════════
// (9) tool row after reroute — NONE (both tool-row call sites)
// ══════════════════════════════════════════════════════════════════════════
describe("(9) tool row after a reroute in the same session", () => {
  it("context.ts toolSpan(): no _tp_routing on the tool row [FAILS IF FIX REVERTED — pre-fix _emitToolRow copied session.metadata wholesale, including _tp_routing]", async () => {
    const client = makeClient();
    const logged: unknown[][] = [];
    vi.spyOn(client, "log").mockImplementation((...args: unknown[]) => {
      logged.push(args);
    });

    await tpSession({ userId: "u1", metadata: { customer_key: "keep-me" } }, async () => {
      await runWithCallObsScope(async () => {
        const body: Record<string, unknown> = { model: "gpt-4o" };
        const status = _applyReroute(rerouteDirective, body, "openai");
        expect(status).toBe("applied");
      });
      // Tool row emitted AFTER the reroute, in the SAME session — the session
      // metadata carries _tp_routing at this point (asserted above).
      toolSpan({ name: "web_search" }, () => "result");
    });

    expect(logged).toHaveLength(1);
    const metadata = logged[0][METADATA_ARG] as Record<string, unknown>;
    expect(metadata._tp_routing).toBeUndefined();
    expect(metadata.customer_key).toBe("keep-me"); // (14)
  });

  it("frameworkTools.ts emitToolRow(): no _tp_routing on the tool row", async () => {
    const client = makeClient();
    const logged: unknown[][] = [];
    vi.spyOn(client, "log").mockImplementation((...args: unknown[]) => {
      logged.push(args);
    });

    await tpSession({ userId: "u1", metadata: { customer_key: "keep-me" } }, async () => {
      await runWithCallObsScope(async () => {
        const body: Record<string, unknown> = { model: "gpt-4o" };
        _applyReroute(rerouteDirective, body, "openai");
      });
      emitToolRow({ name: "framework_tool" });
    });

    expect(logged).toHaveLength(1);
    const metadata = logged[0][METADATA_ARG] as Record<string, unknown>;
    expect(metadata._tp_routing).toBeUndefined();
    expect(metadata.customer_key).toBe("keep-me");
  });
});

// ══════════════════════════════════════════════════════════════════════════
// (10) agent/chain structural row — NONE
// ══════════════════════════════════════════════════════════════════════════
describe("(10) agent/chain structural anchor row", () => {
  /** Minimal ReadableSpan stand-in for TokenPoliceSpanProcessor.onEnd(). */
  function fakeStructuralSpan(attrs: Record<string, unknown>): unknown {
    return {
      attributes: attrs,
      name: "structural",
      startTime: [0, 0] as [number, number],
      endTime: [1, 0] as [number, number],
      spanContext: () => ({ traceId: "a".repeat(32), spanId: "b".repeat(16) }),
      parentSpanId: undefined,
      status: { code: 0 },
    };
  }

  it("never carries _tp_routing, even when the span's own tp.meta attrs include it", () => {
    const client = makeClient();
    const logged: unknown[][] = [];
    vi.spyOn(client, "log").mockImplementation((...args: unknown[]) => {
      logged.push(args);
    });

    // Simulates what `_runWithStructuralSpan` (context.ts) actually stamps at
    // span-open time when session.metadata carries `_tp_routing` (it
    // JSON-stringifies EVERY metadata key, including this one, into
    // `tp.meta.*` attrs by design — the strip is telemetry.ts's job at
    // row-build time, not the attribute build's).
    const attrs: Record<string, unknown> = {
      "tp.kind": "agent",
      "tp.user_id": "u1",
      "tp.paid_plan": "free",
      "tp.workflow_name": "wf",
      "tp.session_id": "s1",
      [TP_ROUTING_ATTR]: JSON.stringify({ rule_id: "rule_rr", actual_model: "gpt-4o-mini" }),
      "tp.meta.customer_key": "keep-me",
    };

    new TokenPoliceSpanProcessor().onEnd(fakeStructuralSpan(attrs) as never);

    expect(logged).toHaveLength(1);
    const metadata = logged[0][METADATA_ARG] as Record<string, unknown>;
    expect(metadata._tp_routing).toBeUndefined();
    expect(metadata.customer_key).toBe("keep-me"); // (14)
  });

  it("same strip applies to a `chain` kind span", () => {
    const client = makeClient();
    const logged: unknown[][] = [];
    vi.spyOn(client, "log").mockImplementation((...args: unknown[]) => {
      logged.push(args);
    });

    const attrs: Record<string, unknown> = {
      "tp.kind": "chain",
      "tp.user_id": "u1",
      "tp.paid_plan": "free",
      "tp.workflow_name": "wf",
      "tp.session_id": "s1",
      [TP_ROUTING_ATTR]: JSON.stringify({ rule_id: "rule_rr" }),
    };

    new TokenPoliceSpanProcessor().onEnd(fakeStructuralSpan(attrs) as never);

    expect(logged).toHaveLength(1);
    const metadata = logged[0][METADATA_ARG] as Record<string, unknown>;
    expect(metadata._tp_routing).toBeUndefined();
  });
});

// ══════════════════════════════════════════════════════════════════════════
// (11) subsequent NON-rerouted call's row — NONE
// ══════════════════════════════════════════════════════════════════════════
describe("(11) a subsequent call that was NOT itself rerouted", () => {
  it("carries no _tp_routing even though an earlier call in the same session was rerouted [FAILS IF FIX REVERTED — pre-fix every row copied session.metadata wholesale]", async () => {
    const client = makeClient();
    const logged: unknown[][] = [];
    vi.spyOn(client, "log").mockImplementation((...args: unknown[]) => {
      logged.push(args);
    });

    await tpSession({ userId: "u1" }, async () => {
      // Call 1: rerouted. Stashes under call 1's own key; no row emitted for it.
      await runWithCallObsScope(async () => {
        const body: Record<string, unknown> = { model: "gpt-4o" };
        const status = _applyReroute(rerouteDirective, body, "openai");
        expect(status).toBe("applied");
      });

      // Call 2: its OWN obs scope, no reroute happened under its key.
      await runWithCallObsScope(async () => {
        const key2 = getCurrentObsKey() ?? null;
        _logLlamaIndex(new OpenAIStandIn(), { raw: {} }, 1, "call2", new Date(), key2);
      });
    });

    expect(logged).toHaveLength(1); // only call 2's row
    const metadata = logged[0][METADATA_ARG] as Record<string, unknown>;
    expect(metadata._tp_routing).toBeUndefined();
  });
});

// ══════════════════════════════════════════════════════════════════════════
// (12) two concurrent calls, one rerouted — isolation holds at the ROW level
// ══════════════════════════════════════════════════════════════════════════
describe("(12) concurrent calls on one shared session", () => {
  it("the rerouted call's row carries its OWN marker; the sibling's row carries none", async () => {
    const client = makeClient();
    const logged: unknown[][] = [];
    vi.spyOn(client, "log").mockImplementation((...args: unknown[]) => {
      logged.push(args);
    });

    await tpSession({ userId: "u1" }, async () => {
      await Promise.all([
        runWithCallObsScope(async () => {
          const key = getCurrentObsKey() ?? null;
          const body: Record<string, unknown> = { model: "gpt-4o" };
          _applyReroute(rerouteDirective, body, "openai");
          // Yield a macrotask — stands in for the real gap between check and
          // the provider's own LLM HTTP call, during which the sibling below
          // interleaves on the SAME shared session (mirrors
          // localDecisionStore.test.ts's `checkedCall` burst harness).
          await new Promise((resolve) => setTimeout(resolve, 0));
          _logLlamaIndex(new OpenAIStandIn(), { raw: {} }, 0, "rerouted_call", new Date(), key);
        }),
        runWithCallObsScope(async () => {
          const key2 = getCurrentObsKey() ?? null;
          await new Promise((resolve) => setTimeout(resolve, 0));
          _logLlamaIndex(new OpenAIStandIn(), { raw: {} }, 1, "sibling_call", new Date(), key2);
        }),
      ]);
    });

    expect(logged).toHaveLength(2);
    const rerouted = logged.find(
      (a) => (a[SPAN_ARG] as Record<string, unknown>)?.span_name === "rerouted_call",
    );
    const sibling = logged.find(
      (a) => (a[SPAN_ARG] as Record<string, unknown>)?.span_name === "sibling_call",
    );
    expect(rerouted).toBeTruthy();
    expect(sibling).toBeTruthy();
    expect((rerouted![METADATA_ARG] as Record<string, unknown>)._tp_routing).toBeTruthy();
    expect((sibling![METADATA_ARG] as Record<string, unknown>)._tp_routing).toBeUndefined();
  });
});

// ══════════════════════════════════════════════════════════════════════════
// (13, LOW) rerouted call's FAILURE row keeps the marker
// ══════════════════════════════════════════════════════════════════════════
describe("(13, LOW) rerouted call's own failure row", () => {
  it("keeps _tp_routing (peek-many: the failure row is still THIS call's row)", async () => {
    const client = makeClient();
    const logged: unknown[][] = [];
    vi.spyOn(client, "log").mockImplementation((...args: unknown[]) => {
      logged.push(args);
    });

    await tpSession({ userId: "u1" }, async (sess) => {
      await runWithCallObsScope(async () => {
        const body: Record<string, unknown> = { model: "gpt-4o" };
        const status = _applyReroute(rerouteDirective, body, "openai");
        expect(status).toBe("applied");

        // Simulate the provider call then failing.
        (sess as unknown as Record<string, unknown>)._call_outcome = {
          status: "failed",
          error_type: "APIError",
        };
        _emitCallFailureLog(client, sess);
      });
    });

    expect(logged).toHaveLength(1);
    const metadata = logged[0][METADATA_ARG] as Record<string, unknown>;
    const routing = metadata._tp_routing as Record<string, unknown> | undefined;
    expect(routing).toBeTruthy();
    expect(routing?.rule_id).toBe("rule_rr");
  });

  it("a SIBLING's failure row (own key, no reroute) carries no marker", async () => {
    const client = makeClient();
    const logged: unknown[][] = [];
    vi.spyOn(client, "log").mockImplementation((...args: unknown[]) => {
      logged.push(args);
    });

    await tpSession({ userId: "u1" }, async (sess) => {
      await runWithCallObsScope(async () => {
        const body: Record<string, unknown> = { model: "gpt-4o" };
        _applyReroute(rerouteDirective, body, "openai");
      });

      await runWithCallObsScope(async () => {
        (sess as unknown as Record<string, unknown>)._call_outcome = {
          status: "failed",
          error_type: "APIError",
        };
        _emitCallFailureLog(client, sess);
      });
    });

    expect(logged).toHaveLength(1);
    const metadata = logged[0][METADATA_ARG] as Record<string, unknown>;
    expect(metadata._tp_routing).toBeUndefined();
  });
});
