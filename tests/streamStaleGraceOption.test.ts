/**
 * `streamStaleGraceSeconds` clamp — TokenPolice constructor (C-14).
 *
 * Clamped to [0, 3600]; a non-numeric / NaN / Infinity / boolean value is a
 * misconfiguration and falls back to the 60s default (pure local arithmetic —
 * cannot throw). Direct construction (not init()) starts no SSE stream, so
 * these tests need no network mocking.
 *
 * Sibling: token-police-python/tests/test_stream_stale_grace_option.py pins
 * the same matrix against the Python SDK.
 */
import { describe, it, expect } from "vitest";
import { TokenPolice } from "../src/client";

const API_KEY = "tp_sk_test123";

describe("streamStaleGraceSeconds clamp [0, 3600], default 60 (Node)", () => {
  const cases: Array<[string, unknown, number]> = [
    ["0 stays 0", 0, 0],
    ["-5 clamps up to 0", -5, 0],
    ["99999 clamps down to 3600", 99999, 3600],
    ["non-numeric string falls back to 60", "abc", 60],
    ["null falls back to 60", null, 60],
    ["Infinity falls back to 60", Number.POSITIVE_INFINITY, 60],
    ["-Infinity falls back to 60", Number.NEGATIVE_INFINITY, 60],
    ["NaN falls back to 60", Number.NaN, 60],
    ["boolean true falls back to 60 (never coerced to 1)", true, 60],
    ["boolean false falls back to 60 (never coerced to 0)", false, 60],
  ];

  for (const [label, input, expected] of cases) {
    it(label, () => {
      const client = new TokenPolice({
        apiKey: API_KEY,
        streamStaleGraceSeconds: input as never,
      });
      expect(client.streamStaleGraceSeconds).toBe(expected);
      client.closeSync();
    });
  }

  it("omitted (undefined) falls back to the 60s default", () => {
    const client = new TokenPolice({ apiKey: API_KEY });
    expect(client.streamStaleGraceSeconds).toBe(60);
    client.closeSync();
  });

  it("an in-range value (e.g. 120) is preserved unchanged", () => {
    const client = new TokenPolice({ apiKey: API_KEY, streamStaleGraceSeconds: 120 });
    expect(client.streamStaleGraceSeconds).toBe(120);
    client.closeSync();
  });

  it("boundary: 3600 is preserved unchanged, 3601 clamps down to 3600", () => {
    const at = new TokenPolice({ apiKey: API_KEY, streamStaleGraceSeconds: 3600 });
    expect(at.streamStaleGraceSeconds).toBe(3600);
    at.closeSync();

    const over = new TokenPolice({ apiKey: API_KEY, streamStaleGraceSeconds: 3601 });
    expect(over.streamStaleGraceSeconds).toBe(3600);
    over.closeSync();
  });
});
