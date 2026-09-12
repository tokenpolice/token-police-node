/**
 * TokenPoliceBlockedError carries structured fields so callers can branch on a
 * block (budget vs loop-detection, which rule) without parsing the message.
 *
 * Twin of token-police-python/tests/test_blocked_error_fields.py — the two
 * files pin the cross-SDK parity contract on the SDK's canonical field names
 * (camelCase here, snake_case in Python).
 */
import { describe, test, expect, beforeEach, vi } from "vitest";
import { resetPack, setClient } from "../src/state";
import { TokenPolice } from "../src/client";
import { TokenPoliceBlockedError } from "../src/exceptions";
import { __test__ as enforcerTest } from "../src/enforcer";

// ── unit: the exception itself ────────────────────────────────────
describe("TokenPoliceBlockedError structured fields", () => {
  test("message-only construction stays back-compatible", () => {
    const err = new TokenPoliceBlockedError("boom");
    expect(err.message).toBe("boom");
    expect(err.name).toBe("TokenPoliceBlockedError");
    expect(err.reason).toBeUndefined();
    expect(err.ruleId).toBeUndefined();
    expect(err.kind).toBeUndefined();
    expect(err.traceId).toBeUndefined();
  });

  test("details are exposed as fields", () => {
    const err = new TokenPoliceBlockedError("msg", {
      reason: "over budget",
      ruleId: "rule_1",
      kind: "budget",
      traceId: "tr_9",
    });
    expect(err.message).toBe("msg");
    expect(err.reason).toBe("over budget");
    expect(err.ruleId).toBe("rule_1");
    expect(err.kind).toBe("budget");
    expect(err.traceId).toBe("tr_9");
  });
});

// ── integration: caught error from an enforce-mode block ──────────
describe("TokenPoliceBlockedError from an enforce block", () => {
  function makeEnforceClient() {
    const client = new TokenPolice({
      apiKey: "tp_sk_test123",
      baseUrl: "http://localhost:99999",
      timeout: 0.1,
      firewall: "enforce",
      deployment: "daemon",
    });
    setClient(client);
    return client;
  }

  beforeEach(() => resetPack());

  test("server budget block populates reason/ruleId/traceId and kind='budget'", async () => {
    const client = makeEnforceClient();
    // No pack applied → State-B inline /check drives the verdict.
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({
      status: "blocked",
      reason: "budget exceeded",
      ruleId: "rule_budget_1",
      traceId: "tr_budget",
    } as never);
    const err = await enforcerTest
      ._runAsyncCheck({ model: "gpt-4" }, "openai")
      .then(() => null)
      .catch((e: unknown) => e);
    expect(err).toBeInstanceOf(TokenPoliceBlockedError);
    const be = err as TokenPoliceBlockedError;
    expect(be.message).toBe("TokenPolice: Budget exceeded — budget exceeded");
    expect(be.reason).toBe("budget exceeded");
    expect(be.ruleId).toBe("rule_budget_1");
    expect(be.traceId).toBe("tr_budget");
    expect(be.kind).toBe("budget");
    checkSpy.mockRestore();
  });

  test("loop block surfaces the detector kind via `kind`", async () => {
    const client = makeEnforceClient();
    const checkSpy = vi.spyOn(client, "check").mockResolvedValue({
      status: "blocked",
      reason: "loop detected",
      ruleId: "rule_loop_1",
      detail: "HASH_CYCLE",
      traceId: "tr_loop",
    } as never);
    const err = await enforcerTest
      ._runAsyncCheck({ model: "gpt-4" }, "openai")
      .then(() => null)
      .catch((e: unknown) => e);
    expect(err).toBeInstanceOf(TokenPoliceBlockedError);
    const be = err as TokenPoliceBlockedError;
    expect(be.reason).toBe("loop detected");
    expect(be.ruleId).toBe("rule_loop_1");
    expect(be.kind).toBe("HASH_CYCLE");
    expect(be.traceId).toBe("tr_loop");
    checkSpy.mockRestore();
  });
});
