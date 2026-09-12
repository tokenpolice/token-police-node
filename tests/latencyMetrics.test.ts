/**
 * Latency capture (TTFT + streaming throughput) for the manual-mode stream
 * wrapper. Mirrors token-police-python/tests/test_latency_metrics.py — SDK
 * divergence is a bug per the SDK agent guide.
 *
 * Covers the metric math (_buildStreamLatency, _streamAccHasContent) and the
 * fail-safety risk register for the streaming tap (A1-A8 in
 * latency_metrics_design.md): a probe failure, an early consumer break, a
 * mid-stream error, and a clock anomaly must never break the customer's stream,
 * alter the chunk sequence, or fabricate a metric.
 */
import { describe, it, expect, vi } from "vitest";
import { __test__ } from "../src/enforcer";
import { setClient } from "../src/state";

const {
  _buildStreamLatency,
  _streamAccHasContent,
  _newStreamAccumulator,
  _accumulateStreamChunk,
  _wrapManualStream,
} = __test__;

const contentChunk = (text: string) => ({ choices: [{ delta: { content: text } }] });
const usageChunk = (inp: number, out: number) => ({
  choices: [{ delta: {} }],
  usage: { prompt_tokens: inp, completion_tokens: out },
});

async function* gen(chunks: any[]) {
  for (const c of chunks) yield c;
}

async function drain(it: AsyncIterable<any>): Promise<any[]> {
  const out: any[] = [];
  for await (const c of it) out.push(c);
  return out;
}

describe("latency math", () => {
  it("_buildStreamLatency basic (performance.now ms deltas)", () => {
    const lat = _buildStreamLatency(1000, 1400, 2800)!;
    expect(lat.is_streaming).toBe(true);
    expect(lat.ttft_ms).toBe(400);
    expect(lat.total_ms).toBe(1800);
    expect(lat.generation_ms).toBe(1400);
    expect(lat.clock).toBe("monotonic");
    expect(lat.output_tokens).toBeNull(); // server supplies the count
  });

  it("null ttft when no content chunk ever seen", () => {
    const lat = _buildStreamLatency(1000, null, 2000)!;
    expect(lat.ttft_ms).toBeNull();
    expect(lat.generation_ms).toBeNull();
    expect(lat.total_ms).toBe(1000);
  });

  it("A6: clamps a backwards clock to 0", () => {
    const lat = _buildStreamLatency(1000, 990, 980)!;
    expect(lat.ttft_ms).toBe(0);
    expect(lat.total_ms).toBe(0);
    expect(lat.generation_ms).toBe(0);
  });

  it("missing anchor → null (no latency emitted)", () => {
    expect(_buildStreamLatency(undefined, 1, 2)).toBeNull();
  });

  it("_streamAccHasContent flips on first text", () => {
    const acc = _newStreamAccumulator("openai");
    expect(_streamAccHasContent("openai", acc)).toBe(false);
    _accumulateStreamChunk("openai", acc, contentChunk("hi"));
    expect(_streamAccHasContent("openai", acc)).toBe(true);
  });

  it("_streamAccHasContent best-effort true when provider not accumulated", () => {
    expect(_streamAccHasContent("unaccumulated", null)).toBe(true);
  });
});

describe("stream wrapper fail-safety + latency emission", () => {
  it("happy path: emits ttft + total into tp.log extras", async () => {
    const captured: any = {};
    setClient({
      enforce: false,
      log: (...a: any[]) => {
        captured.extras = a[13];
      },
    } as any);

    const wrapped = await _wrapManualStream(
      gen([contentChunk("Hello"), contentChunk(" world"), usageChunk(10, 5)]),
      "openai", [{}], 0, "span", new Date(), 12345 /* reqStartMono */,
    );
    const out = await drain(wrapped);
    expect(out.length).toBe(3);                       // A2: all chunks present
    expect(captured.extras?.latency).toBeTruthy();
    expect(captured.extras.latency.is_streaming).toBe(true);
    expect(typeof captured.extras.latency.ttft_ms).toBe("number");
    expect(captured.extras.latency.total_ms).toBeGreaterThanOrEqual(
      captured.extras.latency.ttft_ms,
    );
  });

  it("A1: a tap that throws (poison chunk) never breaks the stream", async () => {
    setClient({ enforce: false, log: () => {} } as any);
    // Throw on exactly the fields the tap reads (choices/usage/data) — NOT on
    // `then`/Symbol.asyncIterator, which would break the await protocol itself
    // rather than exercise the tap's own try/catch.
    const poison = {
      get choices() { throw new Error("boom"); },
      get usage() { throw new Error("boom"); },
      get data() { throw new Error("boom"); },
    };
    const wrapped = await _wrapManualStream(
      gen([poison, usageChunk(1, 1)]),
      "openai", [{}], 0, "span", new Date(), 100,
    );
    const out = await drain(wrapped);                 // must not throw
    expect(out[0]).toBe(poison);                        // chunk passed through
    expect(out.length).toBe(2);
  });

  it("A2: chunk sequence is identical and in order", async () => {
    setClient({ enforce: false, log: () => {} } as any);
    const chunks = [contentChunk("a"), contentChunk("b"), usageChunk(2, 2)];
    const wrapped = await _wrapManualStream(
      gen(chunks), "openai", [{}], 0, "span", new Date(), 100,
    );
    const out = await drain(wrapped);
    expect(out).toEqual(chunks);
  });

  it("A3: early consumer break runs finally without throwing", async () => {
    const log = vi.fn();
    setClient({ enforce: false, log } as any);
    const wrapped = await _wrapManualStream(
      gen([contentChunk("x"), usageChunk(1, 1)]),
      "openai", [{}], 0, "span", new Date(), 100,
    );
    // Pull one chunk then break — triggers generator .return() → finally.
    for await (const c of wrapped) {
      expect(c).toBeTruthy();
      break;
    }
    // Usage chunk never reached → lastUsageChunk unset → no log, no throw.
    expect(log).not.toHaveBeenCalled();
  });

  it("A8: a mid-stream error propagates verbatim and logs no latency", async () => {
    const log = vi.fn();
    setClient({ enforce: false, log } as any);
    async function* boom() {
      yield contentChunk("partial");
      throw new Error("network drop");
    }
    const wrapped = await _wrapManualStream(
      boom(), "openai", [{}], 0, "span", new Date(), 100,
    );
    await expect(drain(wrapped)).rejects.toThrow("network drop");
    // A failure log may fire (call_outcome), but it must NEVER carry latency —
    // partial TTFT is misleading, so the success-path latency emit is skipped.
    for (const call of log.mock.calls) {
      expect((call[13] as any)?.latency).toBeUndefined();
    }
  });
});
