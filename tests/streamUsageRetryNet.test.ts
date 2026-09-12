/**
 * Stream-usage injection retry-net predicate + customer-body restore.
 *
 * The SDK injects `stream_options.include_usage` into openai-wire chat streams
 * so streamed spend isn't silently lost. If the provider then errors, the SDK
 * strips the injection and retries ONCE. The retry must fire ONLY when the
 * rejection is plausibly caused by the injected option — a strict-compat server
 * answering 400/422 or naming the param. A rate-limit (429) or auth failure
 * (401/403) is never caused by the injection, so retrying there would issue a
 * guaranteed SECOND provider request exactly while the customer is rate-limited
 * / unauthorized (duplicate spend).
 *
 * Also: the customer's own request body must never keep the injected
 * `stream_options` after the call — the option is restored on every exit
 * (success, retry, non-retryable rethrow), while the returned `injected` flag
 * (which the downstream usage-chunk stripping keys off, NOT the body) stays true
 * on a successful injected call.
 *
 * Drives the exported `__test__._callWithInjectedStreamUsage` /
 * `_shouldRetryWithoutInjection` directly with a scripted `original`.
 *
 * Golden rule: an SDK-internal failure must never propagate into customer code,
 * and the SDK must never cause duplicate provider spend.
 */
import { describe, test, expect, afterEach, beforeEach } from "vitest";
import { __test__ as enforcerTest, uninstrument } from "../src/enforcer";
import { TokenPolice } from "../src/client";
import { setClient } from "../src/state";

const {
  _callWithInjectedStreamUsage,
  _shouldRetryWithoutInjection,
  _injectStreamUsageOption,
} = enforcerTest as any;

function makeClient(): void {
  const client = new TokenPolice({
    apiKey: "tp_sk_test123",
    baseUrl: "http://localhost:59999",
    timeout: 0.1,
    firewall: "dry_run",
  } as any);
  setClient(client);
}

beforeEach(() => makeClient());
afterEach(() => {
  try {
    uninstrument();
  } catch {
    // ignore
  }
});

// An error carrying a numeric status, either directly or on `.response`.
function statusErr(status: number, message = "err", onResponse = false): any {
  const e: any = new Error(message);
  if (onResponse) e.response = { status };
  else e.status = status;
  return e;
}

// A hostile error whose every accessed field throws.
function hostileErr(): any {
  return {
    get status(): number {
      throw new Error("boom-status");
    },
    get response(): any {
      throw new Error("boom-response");
    },
    get message(): string {
      throw new Error("boom-message");
    },
  };
}

function freshBody(): any {
  return { stream: true, messages: [{ role: "user", content: "hi" }] };
}

/**
 * Build a scripted `original`. `script[i]` is invoked for call i and either
 * returns a value or throws. Records the call count and, per call, whether the
 * (mutable) body still carried the injected stream_options.
 */
function scriptedOriginal(script: Array<() => any>) {
  const state = { count: 0, soPresent: [] as boolean[] };
  const fn = function (this: any, body: any): any {
    state.soPresent.push(
      !!(body && body.stream_options && body.stream_options.include_usage === true),
    );
    const i = state.count;
    state.count += 1;
    return script[i]();
  };
  return { fn, state };
}

// ── predicate in isolation ────────────────────────────────────────────────────

describe("_shouldRetryWithoutInjection", () => {
  test.each([429, 401, 403, 404, 500, 502])(
    "no retry on unrelated status %i",
    (status) => {
      expect(_shouldRetryWithoutInjection(statusErr(status))).toBe(false);
      expect(_shouldRetryWithoutInjection(statusErr(status, "err", true))).toBe(false);
    },
  );

  test.each([400, 422])("retry on strict-compat status %i", (status) => {
    expect(_shouldRetryWithoutInjection(statusErr(status))).toBe(true);
    expect(_shouldRetryWithoutInjection(statusErr(status, "err", true))).toBe(true);
  });

  test("message escape hatch (case-insensitive), even on a non-400/422 status", () => {
    expect(
      _shouldRetryWithoutInjection(statusErr(404, "unknown field stream_options")),
    ).toBe(true);
    expect(
      _shouldRetryWithoutInjection(statusErr(0, "Rejected: INCLUDE_USAGE")),
    ).toBe(true);
  });

  test("hostile error object: no throw, no retry", () => {
    expect(() => _shouldRetryWithoutInjection(hostileErr())).not.toThrow();
    expect(_shouldRetryWithoutInjection(hostileErr())).toBe(false);
  });
});

// ── injection is actually active in this harness ──────────────────────────────

test("sanity: injection engages for an openai stream body", () => {
  const body = freshBody();
  const handle = _injectStreamUsageOption("openai", [body]);
  expect(handle).not.toBeNull();
  expect(body.stream_options.include_usage).toBe(true);
  handle.restore();
  expect(body.stream_options).toBeUndefined();
});

// ── end-to-end via _callWithInjectedStreamUsage ───────────────────────────────

describe("_callWithInjectedStreamUsage retry net", () => {
  test.each([429, 401, 403])(
    "status %i → original called exactly once, error propagates, body restored",
    async (status) => {
      const body = freshBody();
      const original = scriptedOriginal([
        () => {
          throw statusErr(status, "denied");
        },
      ]);
      await expect(
        _callWithInjectedStreamUsage(original.fn, null, [body], "openai"),
      ).rejects.toThrow("denied");
      expect(original.state.count).toBe(1); // NO duplicate provider request
      expect(body.stream_options).toBeUndefined(); // customer body restored
    },
  );

  test.each([400, 422])(
    "status %i → strip-and-retry once, second call sees no injection, result returned",
    async (status) => {
      const body = freshBody();
      const sentinel = { id: "resp-2", ok: true };
      const original = scriptedOriginal([
        () => {
          throw statusErr(status, "unknown param");
        },
        () => sentinel,
      ]);
      const { result, injected } = await _callWithInjectedStreamUsage(
        original.fn,
        null,
        [body],
        "openai",
      );
      expect(result).toBe(sentinel);
      expect(injected).toBe(false); // retried without injection
      expect(original.state.count).toBe(2);
      expect(original.state.soPresent).toEqual([true, false]); // 1st injected, 2nd stripped
      expect(body.stream_options).toBeUndefined();
    },
  );

  test("404 whose message names the param → retry (message escape hatch)", async () => {
    const body = freshBody();
    const sentinel = { ok: true };
    const original = scriptedOriginal([
      () => {
        throw statusErr(404, "unknown field: stream_options");
      },
      () => sentinel,
    ]);
    const { result } = await _callWithInjectedStreamUsage(
      original.fn,
      null,
      [body],
      "openai",
    );
    expect(result).toBe(sentinel);
    expect(original.state.count).toBe(2);
  });

  test("500 → no retry, error propagates", async () => {
    const body = freshBody();
    const original = scriptedOriginal([
      () => {
        throw statusErr(500, "oops");
      },
    ]);
    await expect(
      _callWithInjectedStreamUsage(original.fn, null, [body], "openai"),
    ).rejects.toThrow("oops");
    expect(original.state.count).toBe(1);
  });

  test("successful injected call → body restored (deep-equal original), injected flag stays true", async () => {
    const body = freshBody();
    const originalBody = structuredClone(body);
    const sentinel = { ok: true };
    const original = scriptedOriginal([() => sentinel]);
    const { result, injected } = await _callWithInjectedStreamUsage(
      original.fn,
      null,
      [body],
      "openai",
    );
    expect(result).toBe(sentinel);
    // The downstream usage-chunk stripping keys off this flag, not the body —
    // so it must stay true even though the body has been reverted.
    expect(injected).toBe(true);
    expect(original.state.soPresent).toEqual([true]); // provider DID see the injection
    expect(body).toEqual(originalBody); // no leaked stream_options in the customer's object
    expect(body.stream_options).toBeUndefined();
  });

  test("hostile error object → no retry, no SDK throw beyond the original error, body restored", async () => {
    const body = freshBody();
    const hostile = hostileErr();
    const original = scriptedOriginal([
      () => {
        throw hostile;
      },
    ]);
    await expect(
      _callWithInjectedStreamUsage(original.fn, null, [body], "openai"),
    ).rejects.toBe(hostile); // the customer's own error, unchanged
    expect(original.state.count).toBe(1);
    expect(body.stream_options).toBeUndefined();
  });
});
