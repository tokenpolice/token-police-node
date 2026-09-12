/**
 * C-12 — /log span_id synthesis (manual `client.log()` wire payload).
 *
 * The collector keys BOTH its per-span idempotency guard and the persisted
 * generations row's span_id off `span.span_id`. A manual caller that never
 * builds a `span` block — or builds one without a `span_id` — used to send
 * NO `span` key at all (or a span with no id), so the row landed with an
 * empty span_id: it collapsed the trace tree and starved ClickHouse's
 * insert_deduplication_token of the entropy that keeps replayed flush
 * batches from silently deduping against each other.
 *
 * The fix: `client.log()` always emits a `span` block carrying a non-empty
 * span_id — synthesized (16 lowercase hex chars) only when the caller didn't
 * supply one. A caller-supplied span_id (including the Node SDK's own
 * deliberately deterministic Anthropic-batch ids, which *want* server-side
 * dedup) is forwarded byte-identical. The caller's object is never mutated
 * (spread copy), and the whole path is wrapped so a hostile `span` can never
 * throw into the caller (Golden Rule).
 *
 * Capture seam + conventions: stubFetch/bodyFor/makeClient idiom from
 * tests/planSource.test.ts and tests/workflowNameForwarding.test.ts.
 */
import { describe, it, expect, vi, afterEach } from "vitest";
import { TokenPolice } from "../src/client";

interface Recorded {
  url: string;
  options: RequestInit;
}

function stubFetch(recorded: Recorded[]) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, options: RequestInit) => {
      recorded.push({ url, options });
      return new Response(JSON.stringify({ status: "allowed" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }),
  );
}

function makeClient(): TokenPolice {
  return new TokenPolice({
    apiKey: "tp_sk_test123",
    baseUrl: "http://localhost:13097",
    timeout: 0.5,
    deployment: "daemon",
  });
}

function bodyFor(recorded: Recorded[], path: string): Record<string, any> {
  const rec = recorded.find((r) => r.url.endsWith(path));
  if (!rec) throw new Error(`no request captured for ${path}`);
  return JSON.parse(rec.options.body as string);
}

const HEX16 = /^[0-9a-f]{16}$/;

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("client.log(): span_id synthesis (C-12)", () => {
  it("no span argument at all → wire body carries a synthesized 16-hex span_id + span_kind", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    client.log("u", "free", "wf", "", "gpt-4o", "openai", 100, 50, 0);
    await client.flush();
    const span = bodyFor(recorded, "/v1/guard/log").span;
    expect(span).toBeTruthy();
    expect(span.span_id).toMatch(HEX16);
    expect(typeof span.span_kind).toBe("string");
    expect(span.span_kind.length).toBeGreaterThan(0);
  });

  it("two successive no-span calls produce DIFFERENT synthesized span_ids", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    client.log("u", "free", "wf", "", "gpt-4o", "openai", 100, 50, 0);
    client.log("u", "free", "wf", "", "gpt-4o", "openai", 100, 50, 0);
    await client.flush();
    const logCalls = recorded.filter((r) => r.url.endsWith("/v1/guard/log"));
    expect(logCalls).toHaveLength(2);
    const id1 = (JSON.parse(logCalls[0].options.body as string) as any).span.span_id;
    const id2 = (JSON.parse(logCalls[1].options.body as string) as any).span.span_id;
    expect(id1).toMatch(HEX16);
    expect(id2).toMatch(HEX16);
    expect(id1).not.toBe(id2);
  });

  it("span carries trace_id + span_name but no span_id → those keys survive, span_id is synthesized", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    client.log(
      "u", "free", "wf", "", "gpt-4o", "openai", 100, 50, 0, {},
      { trace_id: "t-abc", span_name: "n-abc" },
    );
    await client.flush();
    const span = bodyFor(recorded, "/v1/guard/log").span;
    expect(span.trace_id).toBe("t-abc");
    expect(span.span_name).toBe("n-abc");
    expect(span.span_id).toMatch(HEX16);
  });

  it("does not mutate the caller's span object (no span_id/span_kind added)", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    const callerSpan: Record<string, unknown> = { trace_id: "t-xyz" };
    client.log(
      "u", "free", "wf", "", "gpt-4o", "openai", 100, 50, 0, {}, callerSpan,
    );
    await client.flush();
    // Caller's object is byte-identical to what it was before the call.
    expect(callerSpan).toEqual({ trace_id: "t-xyz" });
    expect("span_id" in callerSpan).toBe(false);
    expect("span_kind" in callerSpan).toBe(false);
    // ...while the wire payload carries the synthesized id.
    const wireSpan = bodyFor(recorded, "/v1/guard/log").span;
    expect(wireSpan.trace_id).toBe("t-xyz");
    expect(wireSpan.span_id).toMatch(HEX16);
  });

  it("caller-supplied span_id passes through byte-identical (never substituted)", async () => {
    const recorded: Recorded[] = [];
    stubFetch(recorded);
    const client = makeClient();
    client.log(
      "u", "free", "wf", "", "gpt-4o", "openai", 100, 50, 0, {},
      { span_id: "my-custom-id" },
    );
    await client.flush();
    const span = bodyFor(recorded, "/v1/guard/log").span;
    expect(span.span_id).toBe("my-custom-id");
  });

  it("GOLDEN RULE: a hostile span (throwing span_id getter) never throws into the caller", () => {
    const client = makeClient();
    // Not stubbing fetch here — the point is log() itself must not throw
    // synchronously regardless of what happens downstream in _fetch.
    const hostileSpan: Record<string, unknown> = {};
    Object.defineProperty(hostileSpan, "span_id", {
      enumerable: true,
      get() {
        throw new Error("hostile getter boom");
      },
    });
    expect(() =>
      client.log(
        "u", "free", "wf", "", "gpt-4o", "openai", 100, 50, 0, {}, hostileSpan,
      ),
    ).not.toThrow();
  });
});
